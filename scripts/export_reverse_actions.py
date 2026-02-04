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
eps = 1e-6

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


def _smooth_dirs(
    dirs: list[tuple[float, float] | None],
    window: int = 5,
) -> list[tuple[float, float] | None]:
    """
    Smooth heading using a centered window of yaw angles.
    - Uses circular mean (sin/cos) over the window.
    - Pads with first/last available yaw at the boundaries.
    """
    n = len(dirs)
    if n == 0 or window <= 1:
        return dirs
    half = window // 2

    # Extract yaw angles; fill gaps with nearest known yaw.
    yaws: list[float | None] = [None] * n
    last = None
    for i, d in enumerate(dirs):
        if d is not None:
            last = math.atan2(d[1], d[0])
            yaws[i] = last
        else:
            yaws[i] = None

    # Forward fill from first non-None.
    first_yaw = next((y for y in yaws if y is not None), None)
    if first_yaw is None:
        return dirs  # nothing to smooth
    for i in range(n):
        if yaws[i] is None:
            yaws[i] = last if last is not None else first_yaw
        else:
            last = yaws[i]

    # Backward fill trailing None (if any).
    last = yaws[-1]
    for i in range(n - 1, -1, -1):
        if yaws[i] is None:
            yaws[i] = last
        else:
            last = yaws[i]

    smoothed: list[tuple[float, float] | None] = [None] * n
    sin = math.sin
    cos = math.cos
    atan2 = math.atan2

    for i in range(n):
        start = max(0, i - half)
        end = min(n - 1, i + half)
        # pad by clamping indices to [0, n-1]
        angles: list[float] = []
        for k in range(i - half, i + half + 1):
            kk = min(max(0, k), n - 1)
            angles.append(yaws[kk])
        sum_sin = sum(sin(a) for a in angles)
        sum_cos = sum(cos(a) for a in angles)
        mean_angle = atan2(sum_sin, sum_cos)
        smoothed[i] = (math.cos(mean_angle), math.sin(mean_angle))

    return smoothed


def _compute_prev_actions(
    frames: list[dict],
    dirs: list[tuple[float, float] | None],
    index: int,
    window: int,
    padding_action: int,
    padding_frame: int,
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

    turn_thresh = float(turn_threshold_deg)
    signed_delta = _signed_angle_delta
    atan2 = math.atan2
    degrees = math.degrees

    # Precompute yaws for all dirs (None if dir missing).
    yaws: list[float | None] = []
    for d in dirs:
        if d is None:
            yaws.append(None)
        else:
            yaws.append(atan2(d[1], d[0]))

    curr = index
    angle_carry = 0.0  # radians; carries residual yaw between chunks
    while len(prev_actions) < window and curr > 0:
        block_start = max(0, curr - 5)
        angle_accum = angle_carry  # start with carry from previous chunk
        block_events: list[tuple[int, int]] = []  # (action, frame_id)

        # Trace back within the 5-frame chunk.
        for step in range(curr - 1, block_start - 1, -1):
            yaw_prev = yaws[step]
            yaw_curr = yaws[step + 1] if step + 1 < len(yaws) else yaw_prev
            if yaw_prev is None or yaw_curr is None:
                continue

            delta = signed_delta(yaw_prev, yaw_curr)
            angle_accum += delta

            while abs(degrees(angle_accum)) >= turn_thresh:
                direction = left_action if angle_accum > 0 else right_action
                if (
                    block_events
                    and block_events[-1][0] in (left_action, right_action)
                    and block_events[-1][0] != direction
                ):
                    block_events.append((int(forward_action), int(frames[step]["frame"])))
                block_events.append((int(direction), int(frames[step]["frame"])))
                angle_accum -= math.copysign(math.radians(turn_thresh), angle_accum)

        # If no turns inside the chunk, add a move and any net turn from chunk start to current.
        has_turn = any(a in (left_action, right_action) for a, _ in block_events)
        if not has_turn:
            block_events.append((int(forward_action), int(frames[block_start]["frame"])))
            yaw_start = yaws[block_start]
            yaw_end = yaws[curr] if curr < len(yaws) else yaw_start
            if yaw_start is not None and yaw_end is not None:
                delta_total = signed_delta(yaw_start, yaw_end) + angle_carry
                total_deg = abs(degrees(delta_total))
                direction = left_action if delta_total > 0 else right_action
                turns_needed = int(total_deg // turn_thresh)
                for _ in range(turns_needed):
                    block_events.append((int(direction), int(frames[block_start]["frame"])))

        # Push block events into history, inserting a move between opposite turns across chunks.
        for action, frame_id in block_events:
            if len(prev_actions) >= window:
                break
            if (
                prev_actions
                and action in (left_action, right_action)
                and prev_actions[-1] in (left_action, right_action)
                and prev_actions[-1] != action
                and len(prev_actions) < window
            ):
                prev_actions.append(int(forward_action))
                prev_frames.append(int(frame_id))
                if len(prev_actions) >= window:
                    break
            prev_actions.append(int(action))
            prev_frames.append(int(frame_id))

        # Carry residual yaw into next chunk.
        angle_carry = angle_accum
        curr = block_start

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
    if output_path.exists() and overwrite:
        print(f"[OVERWRITE] Removing existing reverse file: {output_path}")
        try:
            output_path.unlink()
        except OSError as exc:
            print(f"[OVERWRITE][WARN] Failed to remove {output_path}: {exc}")

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
    # Smooth heading using a 5-frame centered window (endpoint padded).
    dirs = _smooth_dirs(dirs, window=5)

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
        "--turn-threshold-deg",
        type=float,
        default=15.0,
        help="Turn action threshold in degrees (default: 15).",
    )
    args = ap.parse_args()

    print(f"[DISCOVER] Scanning input root: {args.input_actions}")
    inputs = _iter_inputs(args.input_actions)
    if not inputs:
        raise SystemExit(f"[ERROR] No *_actions.json found at {args.input_actions}")
    if len(inputs) > 1 and args.output is not None:
        raise SystemExit("[ERROR] --output can only be used with a single input file.")
    print(f"[DISCOVER] Found {len(inputs)} action file(s):")
    for i, path in enumerate(inputs, 1):
        print(f"  [{i}/{len(inputs)}] {path}")

    if args.input_actions.is_file():
        output_path = _process_input(
                input_path=inputs[0],
                window=args.window,
                padding_action=args.padding_action,
                padding_frame=args.padding_frame,
                step_distance=args.step_distance,
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
            print(f"[SCENE] {scene_dir} ({len(scene_inputs)} file(s))")
            outputs = _process_scene(
                scene_dir=scene_dir,
                inputs=scene_inputs,
                window=args.window,
                padding_action=args.padding_action,
                padding_frame=args.padding_frame,
                step_distance=args.step_distance,
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
            print(f"[SCENE][QUEUE] {scene_dir} ({len(scene_inputs)} file(s))")
            futures.append(
                executor.submit(
                    _process_scene,
                    scene_dir,
                    scene_inputs,
                    args.window,
                    args.padding_action,
                    args.padding_frame,
                    args.step_distance,
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
