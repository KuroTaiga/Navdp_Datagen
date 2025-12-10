#!/usr/bin/env python3
"""
Summarize Gaussian actor sequences to understand bounding boxes, offsets, and scaling needs.

Given a root directory where each subdirectory contains per-frame .ply files for one actor,
the script reports per-UID statistics such as:
  • average/min/max bounding-box corners, sizes, and centers
  • per-frame point-count range
  • percentile-based foot height plus the recommended foot offset
  • scale factor required to reach a target height
  • aggregate “global shift” helpers for re-centering or grounding the actor
"""

from __future__ import annotations

import argparse
import json
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
    HIP_HEIGHT_RATIO,
    list_actor_frame_paths,
    load_gaussian_ply,
)

ALIGNMENT_TRANSFORM = np.eye(4, dtype=np.float64)
ALIGNMENT_TRANSFORM[:3, :3] = ACTOR_AXIS_ALIGNMENT_MATRIX
from utils import gaussian_ply_utils as ply_utils  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect every actor UID under a root folder and emit size/offset statistics."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory whose immediate children are per-actor folders containing PLY frames.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_ACTOR_PATTERN,
        help=f"Glob used to locate frames inside each actor folder (default: {DEFAULT_ACTOR_PATTERN}).",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=1.7,
        help="Reference height in meters for scale-factor comparisons (default: 1.7).",
    )
    parser.add_argument(
        "--foot-percentile",
        type=float,
        default=2.0,
        help="Percentile of the vertical distribution that defines the ground contact height (default: 2%%).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Sample at most this many frames per actor when gathering stats (default: 40, 0 = all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("actor_sequence_stats.json"),
        help="Path to the JSON file that will store the aggregated statistics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed table but skip writing the JSON file.",
    )
    return parser.parse_args()


def summarise_vector(samples: np.ndarray) -> dict:
    return {
        "min": samples.min(axis=0).tolist(),
        "max": samples.max(axis=0).tolist(),
        "avg": samples.mean(axis=0).tolist(),
    }


def summarise_scalar(samples: np.ndarray) -> dict:
    return {
        "min": float(samples.min()),
        "max": float(samples.max()),
        "avg": float(samples.mean()),
    }


def uniform_sample(paths: Sequence[Path], limit: int) -> list[Path]:
    if limit <= 0 or limit >= len(paths):
        return list(paths)
    step = max(len(paths) // limit, 1)
    sampled: list[Path] = []
    for idx in range(0, len(paths), step):
        sampled.append(paths[idx])
        if len(sampled) >= limit:
            break
    return sampled


def analyze_actor(
    uid: str,
    directory: Path,
    *,
    pattern: str,
    target_height: float,
    foot_percentile: float,
    max_frames: int,
) -> dict:
    options = ActorOptions(
        sequence_dir=directory,
        pattern=pattern,
        height=target_height,
        follow_distance=1.5,
        buffer_distance=0.5,
        speed=DEFAULT_ACTOR_SPEED,
        fps=DEFAULT_VIDEO_FPS,
        loop=True,
        foot_offset=0.0,
        animation_cycle_mod=3,
    )
    frame_paths = list_actor_frame_paths(options)
    if not frame_paths:
        raise FileNotFoundError(f"No PLY frames found under {directory}")

    sampled_paths = uniform_sample(frame_paths, max_frames)
    bbox_mins: list[np.ndarray] = []
    bbox_maxs: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    dims: list[np.ndarray] = []
    counts: list[int] = []
    z_all: list[np.ndarray] = []

    for ply_path in sampled_paths:
        ply = load_gaussian_ply(ply_path)
        ply_utils.apply_transform_inplace(
            ply,
            ALIGNMENT_TRANSFORM,
            rotate_normals=True,
            rotate_sh=True,
        )
        xyz = np.stack(
            [ply.data["x"].astype(np.float64), ply.data["y"].astype(np.float64), ply.data["z"].astype(np.float64)],
            axis=1,
        )
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        bbox_size = bbox_max - bbox_min
        bbox_mins.append(bbox_min)
        bbox_maxs.append(bbox_max)
        centers.append(bbox_center)
        dims.append(bbox_size)
        counts.append(xyz.shape[0])
        z_all.append(xyz[:, 2])

    bbox_min_arr = np.stack(bbox_mins)
    bbox_max_arr = np.stack(bbox_maxs)
    bbox_center_arr = np.stack(centers)
    dims_arr = np.stack(dims)
    counts_arr = np.asarray(counts, dtype=np.int64)

    merged_z = np.concatenate(z_all)
    raw_min_z = float(merged_z.min())
    raw_max_z = float(merged_z.max())
    raw_height = max(raw_max_z - raw_min_z, EPS)
    height_target = target_height if target_height > 0.0 else raw_height
    scale_factor = height_target / raw_height

    shifted = (merged_z - raw_min_z) * scale_factor
    clamped_percentile = float(np.clip(foot_percentile, 0.0, 100.0))
    foot_height = float(np.percentile(shifted, clamped_percentile))
    adjusted_height = float(shifted.max())
    hip_height = adjusted_height * HIP_HEIGHT_RATIO
    foot_offset = -foot_height

    stats = {
        "actor_id": uid,
        "directory": str(directory.resolve()),
        "total_frames": len(frame_paths),
        "sampled_frames": len(sampled_paths),
        "points_per_frame": {
            "min": int(counts_arr.min()),
            "max": int(counts_arr.max()),
            "avg": float(counts_arr.mean()),
        },
        "bbox_min": summarise_vector(bbox_min_arr),
        "bbox_max": summarise_vector(bbox_max_arr),
        "bbox_center": summarise_vector(bbox_center_arr),
        "bbox_dimensions": summarise_vector(dims_arr),
        "location_offset_ground": {
            "avg": (-bbox_min_arr.mean(axis=0)).tolist(),
            "min": (-bbox_min_arr.max(axis=0)).tolist(),
            "max": (-bbox_min_arr.min(axis=0)).tolist(),
        },
        "centering_shift": {
            "avg": (-bbox_center_arr.mean(axis=0)).tolist(),
        },
        "raw_height": raw_height,
        "adjusted_height": adjusted_height,
        "target_height": height_target,
        "scale_factor": scale_factor,
        "foot_percentile": clamped_percentile,
        "foot_height": foot_height,
        "foot_offset": foot_offset,
        "hip_height": hip_height,
        "global_shift": {
            "to_ground": [0.0, 0.0, foot_offset],
            "to_center": (-bbox_center_arr.mean(axis=0)).tolist(),
        },
        "global_scale": scale_factor,
    }
    return stats


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise SystemExit(f"Source root {args.source_root} does not exist or is not a directory.")

    actor_dirs = sorted([path for path in args.source_root.iterdir() if path.is_dir()])
    if not actor_dirs:
        raise SystemExit(f"No actor subdirectories found under {args.source_root}.")

    all_stats: list[dict] = []
    for directory in actor_dirs:
        uid = directory.name
        try:
            stats = analyze_actor(
                uid,
                directory,
                pattern=args.pattern,
                target_height=float(args.target_height),
                foot_percentile=float(args.foot_percentile),
                max_frames=int(args.max_frames),
            )
            all_stats.append(stats)
            print(
                f"[OK] {uid}: frames={stats['total_frames']} "
                f"points(avg)={stats['points_per_frame']['avg']:.0f} "
                f"height(raw)={stats['raw_height']:.3f}m scale={stats['scale_factor']:.3f} "
                f"foot_offset={stats['foot_offset']:.4f}",
                flush=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skipping {uid}: {exc}", flush=True)

    if not all_stats:
        raise SystemExit("No actor statistics were collected.")

    payload = {
        "source_root": str(args.source_root.resolve()),
        "pattern": args.pattern,
        "target_height": float(args.target_height),
        "foot_percentile": float(args.foot_percentile),
        "generated_frames": sum(entry["sampled_frames"] for entry in all_stats),
        "actors": all_stats,
    }
    if args.dry_run:
        print("[DRY-RUN] Skipping JSON write.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[DONE] Actor stats saved to {args.output}")


if __name__ == "__main__":
    main()
