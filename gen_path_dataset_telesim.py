#!/usr/bin/env python3
"""
Generate per-frame camera extrinsics + action/reverse-action JSONs for planned label paths.

JSON-only precompute: does NOT load Gaussian splats or initialize TeleSim renderers.

Outputs are written into TWO separate dataset roots (preserving <dataset>/<scene>/...):
  1) camera_root/<scene>/<label>/frame_0000_camera.json ...
  2) actions_root/<scene>/{label}_actions.json and {label}_reverse.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.render_utils import load_occupancy_metadata  # type: ignore
from utils.telesim_path_json_outputs import (
    action_payload_from_camera_payloads,
    build_camera_frame_payloads_for_path,
    prepare_path_data,
    reverse_action_payload_from_actions,
    utc_now_iso,
    write_camera_metadata_frame,
    write_tarball,
)

DETAILED_SUFFIX = "_detailed.json"


def _resolve_scene_dir(root: Path, scene_id: str) -> Path:
    exact = root / scene_id
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"{scene_id}*"))
    if not matches:
        raise FileNotFoundError(f"No scene matching '{scene_id}' under {root}")
    return matches[0]


def _resolve_label_directory(scene_task_dir: Path) -> Path | None:
    label_paths_dir = scene_task_dir / "label_paths"
    if label_paths_dir.is_dir():
        return label_paths_dir
    if scene_task_dir.is_dir() and any(scene_task_dir.glob("*.json")):
        return scene_task_dir
    return None


def _discover_scenes(tasks_dir: Path) -> list[str]:
    if not tasks_dir.is_dir():
        return []
    return [p.name for p in sorted(tasks_dir.iterdir()) if p.is_dir()]


def _iter_label_jsons(
    *,
    label_dir: Path,
    label_ids: Sequence[str] | None,
    max_labels: int | None,
    exclude_detailed: bool,
) -> list[Path]:
    if label_ids:
        resolved: list[Path] = []
        for label_id in label_ids:
            p = Path(str(label_id))
            if p.suffix != ".json":
                p = p.with_suffix(".json")
            if not p.is_file():
                candidate = label_dir / p.name
                if candidate.is_file():
                    p = candidate
            resolved.append(p)
        return resolved

    candidates = []
    for p in sorted(label_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        if exclude_detailed and p.name.endswith(DETAILED_SUFFIX):
            continue
        candidates.append(p)
    if max_labels is not None and max_labels > 0:
        candidates = candidates[: max_labels]
    return candidates


def _load_assignment_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _labels_from_manifest(
    manifest: dict,
    *,
    scene_filter: set[str] | None,
    label_filter: set[str] | None,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float]]:
    actors = {str(a.get("id")): a for a in (manifest.get("actors") or [])}
    per_scene: dict[str, list[tuple[int, str, str]]] = {}
    follow_by_label: dict[tuple[str, str], float] = {}

    for row in (manifest.get("assignments") or []):
        scene = str(row.get("scene", ""))
        label = str(row.get("label", ""))
        actor_id = str(row.get("actor_id", ""))
        if not scene or not label:
            continue
        if scene_filter and scene not in scene_filter:
            continue
        if label_filter and label not in label_filter:
            continue
        order_index = int(row.get("order_index") or 0)
        per_scene.setdefault(scene, []).append((order_index, label, actor_id))

        actor = actors.get(actor_id) or {}
        follow_distance = actor.get("follow_distance")
        if follow_distance is not None:
            follow_by_label[(scene, label)] = float(follow_distance)

    labels: dict[str, list[str]] = {}
    for scene, entries in per_scene.items():
        entries.sort(key=lambda t: (t[0], t[1]))
        labels[scene] = [label for _, label, _ in entries]
    return labels, follow_by_label


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate camera extrinsics + action/reverse-action JSONs (TeleSim planned paths).")
    ap.add_argument("--scenes-dir", type=Path, required=True)
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--camera-root", type=Path, required=True)
    ap.add_argument("--actions-root", type=Path, required=True)
    ap.add_argument("--scene", action="append", default=None)
    ap.add_argument("--label-id", action="append", default=None)
    ap.add_argument(
        "--assignment-manifest",
        type=Path,
        default=None,
        help="Optional. If set, uses the actor assignment manifest to (1) select which labels to process "
        "and (2) optionally override follow distance per label. Not needed for FPV runs.",
    )
    ap.add_argument("--max-labels", type=int, default=None)
    ap.add_argument("--exclude-detailed-labels", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--resample-step", type=float, default=0.0)
    ap.add_argument("--path-handedness", choices=["left", "right", "auto"], default="left")
    ap.add_argument("--swap-xy", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--negate-xy", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mirror-translation", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--follow-distance", type=float, default=0.0)
    ap.add_argument("--height-offset", type=float, default=0.3)
    ap.add_argument("--look-ahead", type=float, default=2.0)
    ap.add_argument("--look-down", type=float, default=0.1)
    ap.add_argument("--stabilize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--minimal-frames", type=int, default=0)

    ap.add_argument("--resolution", type=int, nargs=2, default=(960, 720), metavar=("W", "H"))
    ap.add_argument("--fov-deg", type=float, default=70.0)
    ap.add_argument("--znear", type=float, default=0.001)
    ap.add_argument("--zfar", type=float, default=30.0)

    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-reverse", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--max-next", type=int, default=8)
    ap.add_argument("--action-yaw-window", type=int, default=5)
    ap.add_argument("--move-threshold-deg", type=float, default=10.0)
    ap.add_argument("--turn-threshold-deg", type=float, default=15.0)
    ap.add_argument("--turn-threshold-scale", type=float, default=0.5)
    ap.add_argument("--ahead-dot-eps", type=float, default=1e-6)

    ap.add_argument("--scene-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--reverse-window", type=int, default=8)
    ap.add_argument("--reverse-padding-action", type=int, default=4)
    ap.add_argument("--reverse-padding-frame", type=int, default=0)
    ap.add_argument("--reverse-step-distance", type=float, default=0.05)
    ap.add_argument("--reverse-turn-threshold-deg", type=float, default=15.0)
    ap.add_argument("--tar-out", type=Path, default=None, help="If set, tar both dataset roots after generation.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    scene_filter = set(args.scene) if args.scene else None
    label_filter = set(args.label_id) if args.label_id else None

    if args.path_handedness == "auto":
        args.path_handedness = "left"
    stride = max(1, int(args.stride))
    minimal_frames = int(args.minimal_frames or 0)

    args.camera_root.mkdir(parents=True, exist_ok=True)
    args.actions_root.mkdir(parents=True, exist_ok=True)

    labels_by_scene: dict[str, list[str]] | None = None
    follow_by_label: dict[tuple[str, str], float] = {}
    if args.assignment_manifest is not None:
        if not args.assignment_manifest.is_file():
            raise FileNotFoundError(f"Assignment manifest not found: {args.assignment_manifest}")
        manifest = _load_assignment_manifest(args.assignment_manifest)
        labels_by_scene, follow_by_label = _labels_from_manifest(
            manifest,
            scene_filter=scene_filter,
            label_filter=label_filter,
        )

    scenes = sorted(labels_by_scene.keys()) if labels_by_scene is not None else (sorted(scene_filter) if scene_filter else _discover_scenes(args.tasks_dir))
    if not scenes:
        raise SystemExit(f"[ERROR] No scenes found under tasks_dir={args.tasks_dir}")

    width, height = int(args.resolution[0]), int(args.resolution[1])
    if width <= 0 or height <= 0:
        raise SystemExit("[ERROR] --resolution must be positive.")
    fov_y_rad = math.radians(float(args.fov_deg))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_scene(scene_id: str) -> tuple[str, int, int, list[str]]:
        errors: list[str] = []
        written = 0
        skipped = 0
        scene_dir = _resolve_scene_dir(args.scenes_dir, scene_id)
        task_scene_dir = _resolve_scene_dir(args.tasks_dir, scene_id)
        label_dir = _resolve_label_directory(task_scene_dir)
        if label_dir is None:
            return scene_id, written, skipped, [f"no label jsons under {task_scene_dir}"]

        meta = load_occupancy_metadata(scene_dir)

        if labels_by_scene is not None:
            label_ids = labels_by_scene.get(scene_id, [])
            if args.max_labels is not None and args.max_labels > 0:
                label_ids = label_ids[: int(args.max_labels)]
            label_jsons = [label_dir / f"{label}.json" for label in label_ids]
        else:
            label_jsons = _iter_label_jsons(
                label_dir=label_dir,
                label_ids=args.label_id,
                max_labels=args.max_labels,
                exclude_detailed=bool(args.exclude_detailed_labels),
            )

        for json_path in label_jsons:
            if not json_path.is_file():
                errors.append(f"missing label json: {json_path}")
                continue
            label_id = json_path.stem
            follow_distance = float(follow_by_label.get((scene_id, label_id), float(args.follow_distance)))

            try:
                prepared = prepare_path_data(
                    json_path,
                    meta,
                    stride=stride,
                    resample_step=float(args.resample_step),
                    mirror_translation=bool(args.mirror_translation),
                    swap_xy=bool(args.swap_xy),
                    handedness=str(args.path_handedness),
                    negate_xy=bool(args.negate_xy),
                )
                camera_payloads = build_camera_frame_payloads_for_path(
                    prepared=prepared,
                    meta=meta,
                    follow_distance=follow_distance,
                    height_offset=float(args.height_offset),
                    look_ahead=float(args.look_ahead),
                    look_down=float(args.look_down),
                    stabilize=bool(args.stabilize),
                    minimal_frames=minimal_frames,
                    resolution=(width, height),
                    fov_y_rad=fov_y_rad,
                    znear=float(args.znear),
                    zfar=float(args.zfar),
                )

                frames_dir = args.camera_root / scene_id / label_id
                frames_dir.mkdir(parents=True, exist_ok=True)
                for idx, payload in enumerate(camera_payloads):
                    cam_path = frames_dir / f"frame_{idx:04d}_camera.json"
                    if cam_path.exists() and not bool(args.overwrite):
                        skipped += 1
                        continue
                    write_camera_metadata_frame(
                        frames_dir=frames_dir,
                        frame_prefix="frame",
                        frame_idx=idx,
                        payload=payload,
                        overwrite=bool(args.overwrite),
                    )
                    written += 1

                action_payload = action_payload_from_camera_payloads(
                    camera_root=args.camera_root,
                    scene_id=scene_id,
                    label_id=label_id,
                    meta=meta,
                    camera_payloads=camera_payloads,
                    max_next=int(args.max_next),
                    action_yaw_window=int(args.action_yaw_window),
                    move_threshold_deg=float(args.move_threshold_deg),
                    turn_threshold_deg=float(args.turn_threshold_deg),
                    turn_threshold_scale=float(args.turn_threshold_scale),
                    ahead_dot_eps=float(args.ahead_dot_eps),
                )
                action_path = args.actions_root / scene_id / f"{label_id}_actions.json"
                if action_path.exists() and not bool(args.overwrite):
                    skipped += 1
                else:
                    _write_json(action_path, action_payload, overwrite=bool(args.overwrite))
                    written += 1

                if not bool(args.skip_reverse):
                    reverse_payload = reverse_action_payload_from_actions(
                        action_payload=action_payload,
                        window=int(args.reverse_window),
                        padding_action=int(args.reverse_padding_action),
                        padding_frame=int(args.reverse_padding_frame),
                        turn_threshold_deg=float(args.reverse_turn_threshold_deg),
                    )
                    reverse_path = args.actions_root / scene_id / f"{label_id}_reverse.json"
                    if reverse_path.exists() and not bool(args.overwrite):
                        skipped += 1
                    else:
                        _write_json(reverse_path, reverse_payload, overwrite=bool(args.overwrite))
                        written += 1
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"{scene_id}/{label_id}: {exc}")
                continue

        return scene_id, written, skipped, errors

    max_workers = max(1, int(args.scene_workers))
    results: list[tuple[str, int, int, list[str]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_scene, scene_id): scene_id for scene_id in scenes}
        for fut in as_completed(futs):
            results.append(fut.result())

    total_written = sum(r[1] for r in results)
    total_skipped = sum(r[2] for r in results)
    all_errors = [e for r in results for e in r[3]]
    print(f"[DONE] scenes={len(results)} written={total_written} skipped={total_skipped} errors={len(all_errors)}", flush=True)
    if all_errors:
        for msg in all_errors[:50]:
            print(f"[ERROR] {msg}", file=sys.stderr)
        if len(all_errors) > 50:
            print(f"[ERROR] ... ({len(all_errors) - 50} more)", file=sys.stderr)

    if args.tar_out is not None:
        write_tarball(Path(args.tar_out), [args.camera_root, args.actions_root])
        print(f"[TAR] wrote {args.tar_out}", flush=True)

    return 0 if not all_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
