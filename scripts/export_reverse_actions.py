#!/usr/bin/env python3
"""Generate per-frame reverse actions from a *_actions.json input."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_xy(vec: tuple[float, float]) -> tuple[float, float]:
    x, y = vec
    norm = (x * x + y * y) ** 0.5
    if norm <= 1e-9:
        return 0.0, 0.0
    return x / norm, y / norm


def _signed_angle_delta(a: float, b: float) -> float:
    delta = b - a
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _load_camera_centers(camera_dir: Path) -> dict[int, tuple[float, float]]:
    """
    Load per-frame camera center XY from:
      <camera_dir>/frame_*_camera.json
    """
    centers: dict[int, tuple[float, float]] = {}
    if not camera_dir.is_dir():
        return centers

    # No deep recursion; each file corresponds to a frame.
    for cam_path in camera_dir.glob("frame_*_camera.json"):
        stem = cam_path.stem  # frame_000123_camera
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            frame_id = int(parts[1])
        except ValueError:
            continue

        payload = _load_json(cam_path)
        center = payload.get("camera_center_world")
        if isinstance(center, list) and len(center) >= 2:
            centers[frame_id] = (float(center[0]), float(center[1]))

    return centers


def _compute_dirs(
    frames: list[dict],
    camera_centers: dict[int, tuple[float, float]] | None,
) -> list[tuple[float, float] | None]:
    dirs: list[tuple[float, float] | None] = [None] * len(frames)
    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]

        dx = None
        dy = None
        if camera_centers is not None:
            prev_id = int(prev.get("frame", i - 1))
            curr_id = int(curr.get("frame", i))
            p = camera_centers.get(prev_id)
            c = camera_centers.get(curr_id)
            if p is not None and c is not None:
                dx = c[0] - p[0]
                dy = c[1] - p[1]

        if dx is None or dy is None:
            dx = float(curr["world"][0]) - float(prev["world"][0])
            dy = float(curr["world"][1]) - float(prev["world"][1])

        dirs[i] = _normalize_xy((dx, dy))
    return dirs


def _compute_prev_actions(
    frames: list[dict],
    dirs: list[tuple[float, float] | None],
    index: int,
    window: int,
    padding_action: int,
    padding_frame: int,
    step_distance: float,
    move_threshold: float,
    turn_threshold_deg: float,
    forward_action: int,
    left_action: int,
    right_action: int,
) -> tuple[list[int], list[int]]:
    prev_actions: list[int] = []
    prev_frames: list[int] = []

    if index <= 0:
        return (
            [int(padding_action)] * int(window),
            [int(padding_frame)] * int(window),
        )

    # Hoist casts/lookups for speed in tight loops
    step_dist = float(step_distance)
    move_thresh = float(move_threshold)
    turn_thresh = float(turn_threshold_deg)
    atan2 = math.atan2
    degrees = math.degrees
    signed_delta = _signed_angle_delta

    distance_accum = 0.0
    angle_accum = 0.0
    angle_signed = 0.0

    curr_dir = dirs[index] or dirs[index - 1]
    if curr_dir is None:
        return (
            [int(padding_action)] * int(window),
            [int(padding_frame)] * int(window),
        )
    curr_yaw = atan2(curr_dir[1], curr_dir[0])

    for step in range(index - 1, -1, -1):
        if len(prev_actions) >= window:
            break

        prev_dir = dirs[step]
        if prev_dir is None:
            continue

        prev_yaw = atan2(prev_dir[1], prev_dir[0])
        delta = signed_delta(prev_yaw, curr_yaw)
        delta_deg = abs(degrees(delta))

        distance_accum += step_dist
        angle_accum += delta_deg
        angle_signed += delta_deg * (1.0 if delta >= 0 else -1.0)

        if distance_accum >= move_thresh:
            prev_actions.append(int(forward_action))
            prev_frames.append(int(frames[step]["frame"]))
            distance_accum = 0.0
            if len(prev_actions) >= window:
                break

        if angle_accum >= turn_thresh:
            turns = int(angle_accum // turn_thresh)
            direction = left_action if angle_signed >= 0 else right_action
            for _ in range(turns):
                if len(prev_actions) >= window:
                    break
                prev_actions.append(int(direction))
                prev_frames.append(int(frames[step]["frame"]))

            angle_accum = angle_accum % turn_thresh
            angle_signed = (1.0 if angle_signed >= 0 else -1.0) * angle_accum
            if len(prev_actions) >= window:
                break

        curr_yaw = prev_yaw

    while len(prev_actions) < window:
        prev_actions.append(int(padding_action))
        prev_frames.append(int(padding_frame))

    return prev_actions, prev_frames


def _iter_frames(frames: Iterable[dict]) -> list[dict]:
    ordered = list(frames)
    ordered.sort(key=lambda f: int(f.get("frame", 0)))
    return ordered


def _default_output_path(input_path: Path, label: str | None) -> Path:
    if label:
        return input_path.parent / f"{label}_reverse.json"
    stem = input_path.stem
    if stem.endswith("_actions"):
        stem = stem[: -len("_actions")]
    return input_path.parent / f"{stem}_reverse.json"


def _iter_inputs(root: Path) -> list[Path]:
    """
    Fast bounded discovery (NO rglob):
    - If root is a file: use it
    - If root contains *_actions.json directly: use those
    - Else: search exactly ONE level down: root/<child>/*_actions.json
    """
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []

    direct = list(root.glob("*_actions.json"))
    if direct:
        return sorted(direct)

    hits: list[Path] = []
    for child in root.iterdir():
        if child.is_dir():
            hits.extend(child.glob("*_actions.json"))
    return sorted(hits)


def _group_by_scene(inputs: Sequence[Path]) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for path in inputs:
        scene_dir = path.parent
        groups.setdefault(scene_dir, []).append(path)
    return groups


def _process_input(
    input_path: Path,
    window: int,
    padding_action: int,
    padding_frame: int,
    step_distance: float,
    move_threshold: float,
    turn_threshold_deg: float,
    output_override: Path | None,
    overwrite: bool,
    camera_root: Path | None,
) -> Path:
    # Need label to decide output filename. Load once.
    data = _load_json(input_path)
    label = data.get("label")

    output_path = output_override or _default_output_path(input_path, label)
    if output_path.exists() and not overwrite:
        return output_path

    frames = _iter_frames(data.get("frames") or [])
    scene = data.get("scene")

    camera_centers = None
    effective_camera_root = camera_root
    if effective_camera_root is None:
        dataset_root = data.get("dataset_root")
        if dataset_root:
            effective_camera_root = Path(dataset_root)

    if effective_camera_root is not None and scene and label:
        camera_dir = effective_camera_root / str(scene) / str(label)
        camera_centers = _load_camera_centers(camera_dir)

    dirs = _compute_dirs(frames, camera_centers)

    reverse_frames: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        frame_id = int(frame.get("frame", idx))
        prev_actions, prev_frames = _compute_prev_actions(
            frames=frames,
            dirs=dirs,
            index=idx,
            window=window,
            padding_action=padding_action,
            padding_frame=padding_frame,
            step_distance=step_distance,
            move_threshold=move_threshold,
            turn_threshold_deg=turn_threshold_deg,
            forward_action=1,
            left_action=2,
            right_action=3,
        )
        reverse_frames.append(
            {
                "frame": frame_id,
                "prev_actions": prev_actions,
                "prev_frames": prev_frames,
            }
        )

    payload = {
        "dataset_root": data.get("dataset_root"),
        "scene": data.get("scene"),
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_actions": str(input_path),
        "window": int(window),
        "padding_action": int(padding_action),
        "padding_frame": int(padding_frame),
        "step_distance": float(step_distance),
        "move_threshold": float(move_threshold),
        "turn_threshold_deg": float(turn_threshold_deg),
        "frames": reverse_frames,
    }

    _write_json(output_path, payload)
    return output_path


def _process_scene(
    scene_dir: Path,
    inputs: Sequence[Path],
    window: int,
    padding_action: int,
    padding_frame: int,
    step_distance: float,
    move_threshold: float,
    turn_threshold_deg: float,
    overwrite: bool,
    camera_root: Path | None,
) -> list[Path]:
    outputs: list[Path] = []
    for input_path in sorted(inputs):
        outputs.append(
            _process_input(
                input_path=input_path,
                window=window,
                padding_action=padding_action,
                padding_frame=padding_frame,
                step_distance=step_distance,
                move_threshold=move_threshold,
                turn_threshold_deg=turn_threshold_deg,
                output_override=None,
                overwrite=overwrite,
                camera_root=camera_root,
            )
        )
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description="Export per-frame reverse actions.")
    ap.add_argument(
        "input_actions",
        type=Path,
        help="Path to <label>_actions.json or a directory containing *_actions.json.",
    )
    ap.add_argument("--output", type=Path, default=None, help="Output path override (file input only).")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of scene workers to use when input is a directory (default: 1).",
    )
    ap.add_argument(
        "--camera-root",
        type=Path,
        default=None,
        help="Root directory containing <scene>/<label>/frame_*_camera.json (default: dataset_root in actions).",
    )

    # ✅ Default overwrite is False; pass --overwrite to enable.
    ap.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing reverse action files (default: False).",
    )

    ap.add_argument("--window", type=int, default=8, help="Number of previous actions (default: 8).")
    ap.add_argument(
        "--padding-action",
        type=int,
        default=4,
        help="Padding action value for missing history (default: 4).",
    )
    ap.add_argument(
        "--padding-frame",
        type=int,
        default=0,
        help="Padding frame id for missing history (default: 0).",
    )
    ap.add_argument(
        "--step-distance",
        type=float,
        default=0.05,
        help="Per-frame distance increment in meters (default: 0.05).",
    )
    ap.add_argument(
        "--move-threshold",
        type=float,
        default=0.25,
        help="Forward action distance threshold in meters (default: 0.25).",
    )
    ap.add_argument(
        "--turn-threshold-deg",
        type=float,
        default=15.0,
        help="Turn action threshold in degrees (default: 15).",
    )
    args = ap.parse_args()

    inputs = _iter_inputs(args.input_actions)
    if not inputs:
        raise SystemExit(f"[ERROR] No *_actions.json found at {args.input_actions}")
    if len(inputs) > 1 and args.output is not None:
        raise SystemExit("[ERROR] --output can only be used with a single input file.")

    if args.input_actions.is_file():
        output_path = _process_input(
            input_path=inputs[0],
            window=args.window,
            padding_action=args.padding_action,
            padding_frame=args.padding_frame,
            step_distance=args.step_distance,
            move_threshold=args.move_threshold,
            turn_threshold_deg=args.turn_threshold_deg,
            output_override=args.output,
            overwrite=args.overwrite,
            camera_root=args.camera_root,
        )
        print(f"Wrote reverse actions: {output_path}" if output_path.exists() else f"Skipped: {output_path}")
        return 0

    groups = _group_by_scene(inputs)
    print(f"[INFO] Found {len(inputs)} action files across {len(groups)} scene folders.")
    total_files = len(inputs)
    completed = 0
    worker_count = max(1, int(args.workers))

    if worker_count == 1 or len(groups) <= 1:
        for scene_dir, scene_inputs in sorted(groups.items()):
            outputs = _process_scene(
                scene_dir=scene_dir,
                inputs=scene_inputs,
                window=args.window,
                padding_action=args.padding_action,
                padding_frame=args.padding_frame,
                step_distance=args.step_distance,
                move_threshold=args.move_threshold,
                turn_threshold_deg=args.turn_threshold_deg,
                overwrite=args.overwrite,
                camera_root=args.camera_root,
            )
            for output_path in outputs:
                completed += 1
                print(f"[{completed}/{total_files}] {output_path}")
        return 0

    futures = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for scene_dir, scene_inputs in groups.items():
            futures.append(
                executor.submit(
                    _process_scene,
                    scene_dir,
                    scene_inputs,
                    args.window,
                    args.padding_action,
                    args.padding_frame,
                    args.step_distance,
                    args.move_threshold,
                    args.turn_threshold_deg,
                    args.overwrite,
                    args.camera_root,
                )
            )
        for fut in as_completed(futures):
            for output_path in fut.result():
                completed += 1
                print(f"[{completed}/{total_files}] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
