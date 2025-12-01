#!/usr/bin/env python3
"""
Plan diversified actor assignments for render_label_paths without modifying the renderer.

The tool discovers multiple animated actor (PLY) sequences, estimates a per-actor foot
offset by inspecting their Gaussian bounds, randomly assigns each scene label-path to
an actor, and writes a manifest that records the chosen seed plus every pairing.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from render_label_paths import (  # type: ignore
    ACTOR_AXIS_ALIGNMENT_MATRIX,
    ActorOptions,
    DEFAULT_ACTOR_PATTERN,
    DEFAULT_ACTOR_SPEED,
    DEFAULT_VIDEO_FPS,
    EPS,
    list_actor_frame_paths,
    load_gaussian_ply,
    resolve_label_directory,
)
from utils import gaussian_ply_utils as ply_utils  # type: ignore

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCENES_DIR = BASE_DIR / "data" / "scenes"
DEFAULT_TASK_DIR = BASE_DIR / "data" / "selected_33w"
DEFAULT_ASSIGNMENT_PATH = BASE_DIR / "data" / "actor_assignment_plan.json"
FOLLOW_DISTANCE_DEFAULT = 1.5
BUFFER_DISTANCE_DEFAULT = 0.5

ALIGNMENT_TRANSFORM = np.eye(4, dtype=np.float64)
ALIGNMENT_TRANSFORM[:3, :3] = ACTOR_AXIS_ALIGNMENT_MATRIX


@dataclass(frozen=True)
class LabelEntry:
    scene_id: str
    label_id: str
    path: Path


@dataclass
class ActorInfo:
    actor_id: str
    directory: Path
    pattern: str
    height: float
    fps: float
    speed: float
    follow_distance: float
    follow_buffer: float
    loop: bool
    animation_cycle_mod: int
    frame_count: int
    foot_offset: float
    foot_percentile: float
    stats: dict[str, float]


@dataclass(frozen=True)
class Assignment:
    scene_id: str
    label_id: str
    actor_id: str
    round_index: int
    order_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly assign actor sequences to label paths and record the plan."
    )
    parser.add_argument(
        "--actor-dir",
        action="append",
        type=Path,
        default=[],
        help="Path to a directory that stores one actor sequence worth of PLY frames. "
        "Repeat to register more actors.",
    )
    parser.add_argument(
        "--actor-root",
        type=Path,
        default=None,
        help="Optional directory whose immediate child directories will all be treated as actors.",
    )
    parser.add_argument(
        "--actor-pattern",
        type=str,
        default=DEFAULT_ACTOR_PATTERN,
        help=f"Glob used to discover actor frames (default: {DEFAULT_ACTOR_PATTERN}).",
    )
    parser.add_argument(
        "--actor-height",
        type=float,
        default=1.7,
        help="Target actor height (meters) for normalisation and stats (default: 1.7).",
    )
    parser.add_argument(
        "--foot-percentile",
        type=float,
        default=2.0,
        help="Percentile used to determine the effective foot plane (default: 2.0).",
    )
    parser.add_argument(
        "--foot-sample-frames",
        type=int,
        default=12,
        help="How many uniformly-sampled frames per actor to inspect when computing stats "
        "(0 means all frames).",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Optional scene filter. Repeat to restrict planning to specific scenes.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=DEFAULT_SCENES_DIR,
        help=f"Scene reconstruction root (default: {DEFAULT_SCENES_DIR}).",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASK_DIR,
        help=f"Label-path root (default: {DEFAULT_TASK_DIR}).",
    )
    parser.add_argument(
        "--assignments-out",
        type=Path,
        default=DEFAULT_ASSIGNMENT_PATH,
        help="Destination for the JSON assignment manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the RNG. If omitted, a new one is generated and recorded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the plan but skip writing the manifest to disk.",
    )
    parser.add_argument(
        "--ban-list",
        type=Path,
        default=None,
        help="Optional text file listing actor UIDs (one per line) to exclude.",
    )
    return parser.parse_args()


def load_ban_list(path: Path | None) -> set[str]:
    banned: set[str] = set()
    if path is None:
        return banned
    if not path.is_file():
        raise FileNotFoundError(f"Ban list file not found: {path}")
    for line in path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        banned.add(entry)
    return banned


def collect_actor_dirs(explicit: Sequence[Path], root: Path | None, banned: set[str]) -> list[Path]:
    seen: dict[str, Path] = {}
    for entry in explicit:
        resolved = entry.resolve()
        if resolved.name in banned:
            continue
        seen[str(resolved)] = resolved
    if root is not None:
        for child in sorted(root.iterdir()):
            if child.is_dir():
                resolved = child.resolve()
                if resolved.name in banned:
                    continue
                seen.setdefault(str(resolved), resolved)
    return list(seen.values())


def list_label_entries(tasks_root: Path, scene_filter: set[str] | None) -> list[LabelEntry]:
    entries: list[LabelEntry] = []
    for scene_dir in sorted(tasks_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        if scene_filter and scene_id not in scene_filter:
            continue
        label_dir = resolve_label_directory(scene_dir)
        if label_dir is None:
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            entries.append(LabelEntry(scene_id=scene_id, label_id=json_path.stem, path=json_path))
    return entries


def uniform_sample_paths(paths: Sequence[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or max_count >= len(paths):
        return list(paths)
    step = max(len(paths) // max_count, 1)
    sampled: list[Path] = []
    for idx in range(0, len(paths), step):
        sampled.append(paths[idx])
        if len(sampled) >= max_count:
            break
    return sampled


def estimate_actor_info(
    directory: Path,
    *,
    pattern: str,
    target_height: float,
    percentile: float,
    sample_frames: int,
) -> ActorInfo:
    options = ActorOptions(
        sequence_dir=directory,
        pattern=pattern,
        height=target_height,
        follow_distance=FOLLOW_DISTANCE_DEFAULT,
        buffer_distance=BUFFER_DISTANCE_DEFAULT,
        speed=DEFAULT_ACTOR_SPEED,
        fps=DEFAULT_VIDEO_FPS,
        loop=True,
        foot_offset=0.0,
        animation_cycle_mod=3,
    )
    frame_paths = list_actor_frame_paths(options)
    if not frame_paths:
        raise FileNotFoundError(f"No PLY frames found under {directory}")

    sampled_paths = uniform_sample_paths(frame_paths, sample_frames)
    z_values: list[np.ndarray] = []
    for ply_path in sampled_paths:
        ply = load_gaussian_ply(ply_path)
        ply_utils.apply_transform_inplace(
            ply,
            ALIGNMENT_TRANSFORM,
            rotate_normals=True,
            rotate_sh=True,
        )
        z_values.append(ply.data["z"].astype(np.float64))

    merged = np.concatenate(z_values)
    raw_min = float(np.min(merged))
    raw_max = float(np.max(merged))
    raw_height = max(raw_max - raw_min, EPS)
    target = target_height if target_height > 0.0 else raw_height
    scale_factor = target / raw_height

    shifted = (merged - raw_min) * scale_factor
    percentile = float(np.clip(percentile, 0.0, 100.0))
    foot_height = float(np.percentile(shifted, percentile))
    adjusted_height = float(np.max(shifted))
    actor_id = directory.name

    stats = {
        "raw_min_z": raw_min,
        "raw_max_z": raw_max,
        "raw_height": raw_height,
        "scale_factor": scale_factor,
        "adjusted_height": adjusted_height,
        "foot_percentile": percentile,
        "sampled_frames": float(len(sampled_paths)),
        "sampled_points": float(merged.shape[0]),
    }

    return ActorInfo(
        actor_id=actor_id,
        directory=directory,
        pattern=pattern,
        height=target,
        fps=DEFAULT_VIDEO_FPS,
        speed=DEFAULT_ACTOR_SPEED,
        follow_distance=FOLLOW_DISTANCE_DEFAULT,
        follow_buffer=BUFFER_DISTANCE_DEFAULT,
        loop=True,
        animation_cycle_mod=3,
        frame_count=len(frame_paths),
        foot_offset=-foot_height,
        foot_percentile=percentile,
        stats=stats,
    )


def assign_actors_to_labels(
    labels: Sequence[LabelEntry],
    actors: Sequence[ActorInfo],
    rng: random.Random,
) -> list[Assignment]:
    if not actors:
        raise ValueError("At least one actor sequence is required.")
    order = list(range(len(actors)))
    rng.shuffle(order)
    order_idx = 0
    round_idx = 0
    assignments: list[Assignment] = []
    actor_ids = [actor.actor_id for actor in actors]

    for label in labels:
        if order_idx >= len(order):
            rng.shuffle(order)
            order_idx = 0
            round_idx += 1
        actor_index = order[order_idx]
        actor_id = actor_ids[actor_index]
        assignments.append(
            Assignment(
                scene_id=label.scene_id,
                label_id=label.label_id,
                actor_id=actor_id,
                round_index=round_idx,
                order_index=order_idx,
            )
        )
        order_idx += 1
    return assignments


def build_manifest(
    *,
    assignments: Sequence[Assignment],
    actors: Sequence[ActorInfo],
    labels: Sequence[LabelEntry],
    scenes_root: Path,
    tasks_root: Path,
    seed: int,
    scene_filter: set[str] | None,
) -> dict:
    actor_map = {actor.actor_id: actor for actor in actors}
    per_actor: dict[str, int] = {actor.actor_id: 0 for actor in actors}
    per_scene: dict[str, int] = {}

    assignment_rows: list[dict[str, object]] = []
    for entry in assignments:
        actor = actor_map[entry.actor_id]
        per_actor[actor.actor_id] += 1
        per_scene[entry.scene_id] = per_scene.get(entry.scene_id, 0) + 1
        assignment_rows.append(
            {
                "scene": entry.scene_id,
                "label": entry.label_id,
                "actor_id": actor.actor_id,
                "actor_dir": str(actor.directory),
                "actor_height": actor.height,
                "actor_foot_offset": actor.foot_offset,
                "round": entry.round_index,
                "order_index": entry.order_index,
            }
        )

    actors_payload = [
        {
            "id": actor.actor_id,
            "directory": str(actor.directory),
            "pattern": actor.pattern,
            "frame_count": actor.frame_count,
            "height": actor.height,
            "foot_offset": actor.foot_offset,
            "foot_percentile": actor.foot_percentile,
            "fps": actor.fps,
            "speed": actor.speed,
            "follow_distance": actor.follow_distance,
            "follow_buffer": actor.follow_buffer,
            "loop": actor.loop,
            "animation_cycle_mod": actor.animation_cycle_mod,
            "stats": actor.stats,
        }
        for actor in actors
    ]

    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": timestamp,
        "seed": seed,
        "scenes_root": str(scenes_root),
        "tasks_root": str(tasks_root),
        "scene_filter": sorted(scene_filter) if scene_filter else None,
        "label_count": len(labels),
        "scene_count": len({label.scene_id for label in labels}),
        "actors": actors_payload,
        "assignments": assignment_rows,
        "per_actor_counts": per_actor,
        "per_scene_counts": per_scene,
    }


def main() -> None:
    args = parse_args()
    banned = load_ban_list(args.ban_list)
    if banned:
        print(f"• Loaded ban list with {len(banned)} entries.", flush=True)
    actor_dirs = collect_actor_dirs(args.actor_dir, args.actor_root, banned)
    if not actor_dirs:
        raise SystemExit("No actor directories provided. Use --actor-dir or --actor-root.")

    print(f"• Discovering actors from {len(actor_dirs)} directories …", flush=True)
    actors: list[ActorInfo] = []
    for directory in actor_dirs:
        info = estimate_actor_info(
            directory,
            pattern=args.actor_pattern,
            target_height=float(args.actor_height),
            percentile=float(args.foot_percentile),
            sample_frames=int(args.foot_sample_frames),
        )
        actors.append(info)
        print(
            f"  - {info.actor_id}: {info.frame_count} frames, "
            f"foot_offset={info.foot_offset:.4f} m (percentile {info.foot_percentile})",
            flush=True,
        )

    scene_filter = set(args.scene) if args.scene else None
    labels = list_label_entries(args.tasks_dir, scene_filter)
    if not labels:
        raise SystemExit("No label-path JSON files matched the current filters.")

    rng_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    rng = random.Random(rng_seed)
    assignments = assign_actors_to_labels(labels, actors, rng)

    print(
        f"• Planned {len(assignments)} label assignments across "
        f"{len({label.scene_id for label in labels})} scenes.",
        flush=True,
    )
    per_actor_counts: dict[str, int] = {}
    for assign in assignments:
        per_actor_counts[assign.actor_id] = per_actor_counts.get(assign.actor_id, 0) + 1
    for actor in actors:
        count = per_actor_counts.get(actor.actor_id, 0)
        print(f"  - {actor.actor_id}: {count} labels", flush=True)

    manifest = build_manifest(
        assignments=assignments,
        actors=actors,
        labels=labels,
        scenes_root=args.scenes_dir.resolve(),
        tasks_root=args.tasks_dir.resolve(),
        seed=rng_seed,
        scene_filter=scene_filter,
    )
    if args.dry_run:
        print("• Dry run enabled; skipping manifest write.")
        return
    output_path = args.assignments_out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"• Assignment manifest saved to {output_path}")


if __name__ == "__main__":
    main()
