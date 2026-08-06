#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a VLNPE-style dataset from FPV outputs (mp4 + *_actions.json).

Inputs (default layout):
  - FPV root:   <fpv_root>/<scene>/<label>.mp4
                <fpv_root>/<scene>/<label>_actions.json
  - Tasks dir:  <tasks_dir>/<scene>/<label>.json (instructions + goal)
  - Scenes dir: <scenes_dir>/<scene>/occupancy.png (+ occupancy.json for scale)

Outputs:
  - dataset.json / dataset.jsonl
  - images under <out_dir>/<images_dir>:
      * per-frame RGB
      * per-frame history mosaic (4x4)
      * per-frame OCC map (with start/current dots)
      * per-scene BEV image
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        return arr[..., :3]
    return arr


def iter_video_frames(path: Path) -> Iterable[Tuple[int, np.ndarray]]:
    try:
        import imageio
        reader = imageio.get_reader(str(path))
        try:
            for idx, frame in enumerate(reader):
                yield idx, ensure_rgb(np.asarray(frame))
        finally:
            reader.close()
        return
    except Exception:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx, ensure_rgb(np.asarray(frame))
            idx += 1
        cap.release()
        return
    except Exception as exc:
        raise RuntimeError(f"Failed to read video frames from {path}") from exc


def infer_task_type(text: str) -> str:
    if not text:
        return "planning"
    low = text.lower()
    tracking_terms = (
        "follow",
        "following",
        "track",
        "tracking",
        "behind",
        "trail",
        "tail",
        "keep up",
        "keep following",
        "stay behind",
        "continue along",
    )
    if any(term in low for term in tracking_terms):
        return "tracking"
    return "planning"


def get_xy(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(round(value[0])), int(round(value[1]))
    if isinstance(value, dict):
        if "pixel" in value:
            return get_xy(value["pixel"])
        if "x" in value and "y" in value:
            return int(round(value["x"])), int(round(value["y"]))
    return None


def load_scale(scene_dir: Path) -> float:
    occ_json = scene_dir / "occupancy.json"
    if not occ_json.is_file():
        return 0.05
    data = load_json(occ_json)
    return float(data.get("scale", 0.05))


def build_occ_base(
    occ_path: Path,
    *,
    free_threshold: int,
    free_is_white: bool,
    occ_mode: str,
) -> Image.Image:
    if occ_mode == "raw":
        return Image.open(occ_path).convert("RGB")

    img = Image.open(occ_path).convert("L")
    arr = np.array(img)
    if free_is_white:
        free_mask = arr >= free_threshold
    else:
        free_mask = arr <= free_threshold

    if occ_mode == "white_free":
        free_color = np.array([255, 255, 255], dtype=np.uint8)
        blocked_color = np.array([0, 0, 0], dtype=np.uint8)
    elif occ_mode == "gray_free":
        free_color = np.array([128, 128, 128], dtype=np.uint8)
        blocked_color = np.array([255, 255, 255], dtype=np.uint8)
    else:
        raise ValueError(f"Unknown occ_mode: {occ_mode}")

    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[free_mask] = free_color
    out[~free_mask] = blocked_color
    return Image.fromarray(out, mode="RGB")


def draw_dot(img: Image.Image, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(img)
    cx, cy = center
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=color, outline=color)


def build_history_mosaic(
    history: List[np.ndarray],
    size: Tuple[int, int],
    *,
    grid: int = 4,
) -> Image.Image:
    width, height = size
    tile_w = max(1, width // grid)
    tile_h = max(1, height // grid)
    mosaic = Image.new("RGB", (tile_w * grid, tile_h * grid), (0, 0, 0))

    for idx in range(grid * grid):
        if idx >= len(history):
            break
        frame = history[-1 - idx]
        tile = Image.fromarray(frame).resize((tile_w, tile_h), resample=Image.BICUBIC)
        x = (idx % grid) * tile_w
        y = (idx // grid) * tile_h
        mosaic.paste(tile, (x, y))

    if mosaic.size != (width, height):
        padded = Image.new("RGB", (width, height), (0, 0, 0))
        padded.paste(mosaic, (0, 0))
        return padded
    return mosaic


def build_waypoints_future(
    frames: List[Dict[str, Any]],
    start_pos: int,
    *,
    future_steps: int,
) -> List[List[int]]:
    pts: List[List[int]] = []
    for idx in range(start_pos, min(len(frames), start_pos + future_steps + 1)):
        pixel = frames[idx].get("pixel")
        if not (isinstance(pixel, list) and len(pixel) == 2):
            continue
        pt = [int(pixel[0]), int(pixel[1])]
        if pts and pts[-1] == pt:
            continue
        pts.append(pt)
    return pts


def build_waypoints_horizon(
    frames: List[Dict[str, Any]],
    start_pos: int,
    *,
    horizon_px: int,
    goal_px: Optional[Tuple[int, int]],
    goal_tol: float = 1.0,
) -> List[List[int]]:
    pts: List[List[int]] = []
    total = 0.0
    last = None
    for idx in range(start_pos, len(frames)):
        pixel = frames[idx].get("pixel")
        if not (isinstance(pixel, list) and len(pixel) == 2):
            continue
        pt = [int(pixel[0]), int(pixel[1])]
        if last is not None and pt == last:
            continue
        if last is not None:
            total += math.hypot(pt[0] - last[0], pt[1] - last[1])
        pts.append(pt)
        last = pt
        if goal_px is not None:
            if math.hypot(pt[0] - goal_px[0], pt[1] - goal_px[1]) <= goal_tol:
                break
        if total >= horizon_px:
            break
    return pts


def build_prompt(
    *,
    instruction: str,
    cur_pixel: Tuple[int, int],
    scale: float,
    task_type: str,
    horizon_px: int,
    occ_free_label: str,
    future_steps: int,
    waypoint_mode: str,
) -> str:
    if waypoint_mode == "future":
        waypoint_line = f"Return exactly {future_steps + 1} waypoints."
        action_line = f"Return exactly {future_steps + 1} actions (current + next {future_steps})."
        horizon_block = (
            "Map scale:\n"
            f"- Image 4 scale: 1 pixel = {scale:.4f} m.\n"
            f"- Required rollout: current step plus next {future_steps} steps (total {future_steps + 1}).\n"
        )
        action_rules = (
            f"  - Output exactly {future_steps + 1} actions.\n"
            "  - Keep STOP (0) if it appears in the future actions.\n"
        )
    else:
        waypoint_line = "Waypoints should form a feasible path consistent with the horizon distance."
        action_line = "Output an action id sequence corresponding to the same intended motion horizon as (A)."
        horizon_block = (
            "Map scale and horizon:\n"
            f"- Image 4 scale: 1 pixel = {scale:.4f} m.\n"
            "- Required rollout distance:\n"
            f"  - PLANNING: ~3.0 m (≈{horizon_px} px) unless the goal is reached earlier.\n"
            f"  - TRACKING: ~1.5 m (≈{int(round(1.5 / scale))} px) unless termination happens earlier.\n"
            "You may output shorter sequences if the goal/termination is reached before the target distance.\n"
        )
        action_rules = (
            "  - No fixed action count is required.\n"
            "  - If you reach the goal/termination within the produced horizon, include a final STOP (0).\n"
            "  - If you do NOT reach the goal/termination, do NOT append STOP.\n"
        )

    return (
        "<image><image><image><image>\n"
        "You are given FOUR images from the same navigation episode:\n\n"
        "Image 1: CURRENT first-person RGB observation.\n"
        "Image 2: HISTORY mosaic (4x4). Tiles are ordered by recency in row-major order:\n"
        "  1,2,3,4\n"
        "  5,6,7,8\n"
        "  9,10,11,12\n"
        "  13,14,15,16\n"
        "Smaller index = closer to the current time. If the task is TRACKING, Image 2 is all black (no history). "
        "If the task is PLANNING and fewer than 16 past frames exist, missing tiles are black placeholders.\n"
        "Image 3: Scene-level BEV map (top-down context).\n"
        "Image 4: OCC / traversability map. ONLY traversable pixels are allowed. It contains:\n"
        "  - GREEN dot: START position (t=0)\n"
        "  - RED dot: CURRENT position\n"
        f"In Image 4, {occ_free_label} pixels are traversable.\n\n"
        f'Instruction: "{instruction}"\n'
        "Infer TASK_TYPE from the instruction (infer task type from instruction):\n"
        "- PLANNING: navigate to a goal / destination described by the instruction.\n"
        "- TRACKING: follow a given path/leader/trajectory or continue along a known route; no history is provided (Image 2 black).\n\n"
        "Coordinate system (STRICT):\n"
        "- Any pixel coordinates MUST be on Image 4 (OCC).\n"
        "- Integer pixel coordinates (x,y), where x increases rightward and y increases downward.\n"
        f"- The FIRST waypoint MUST be exactly the RED dot location: ({cur_pixel[0]},{cur_pixel[1]}).\n\n"
        f"{horizon_block}\n"
        "Task:\n"
        "Output TWO synchronized action branches:\n"
        "(A) Pixel-space plan as waypoints on Image 4.\n"
        "(B) VLN-CE discrete actions aligned with (A).\n\n"
        "Branch (A) Pixel-space waypoints:\n"
        "- Output a JSON array of integer pixel coordinates [[x0,y0],[x1,y1],...].\n"
        "- Constraints:\n"
        "  1) x0,y0 MUST equal the current position.\n"
        "  2) All waypoints MUST lie on traversable pixels in Image 4.\n"
        "  3) No duplicate waypoints.\n"
        f"  4) {waypoint_line}\n"
        "  5) If the goal/termination is reached earlier, output only the waypoints needed to reach it.\n\n"
        "Branch (B) VLN-CE discrete actions:\n"
        f"- {action_line}\n"
        "- Action mapping:\n"
        "  0 = STOP\n"
        "  1 = MOVE_FORWARD\n"
        "  2 = TURN_LEFT\n"
        "  3 = TURN_RIGHT\n"
        "- Length rules:\n"
        f"{action_rules}\n"
        "Output format (STRICT):\n"
        "Return EXACTLY ONE JSON object with EXACTLY TWO keys: \"pixel\" and \"vlnce\".\n"
        "- \"pixel\": the waypoint array from (A).\n"
        "- \"vlnce\": an XML string: \"<action>a1,a2,...</action>\"\n"
        "No extra keys. No extra text.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VLNPE-style dataset from FPV outputs.")
    parser.add_argument("--fpv-root", required=True, help="FPV output root (e.g., data2/0500_fpv).")
    parser.add_argument("--tasks-dir", required=True, help="Instruction JSON root (e.g., data/interiorGS_0500_42).")
    parser.add_argument("--scenes-dir", required=True, help="Scenes root (e.g., data/scenes).")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--images-dir", default="navllm_vlnce_images", help="Images folder under out-dir.")
    parser.add_argument("--dataset-json", default="dataset.json", help="Dataset JSON filename.")
    parser.add_argument("--dataset-jsonl", default="dataset.jsonl", help="Dataset JSONL filename.")
    parser.add_argument("--bev-root", default="", help="Optional BEV root with <scene>.png files.")
    parser.add_argument("--bev-rotate-ccw", type=int, default=0, help="Rotate BEV CCW by degrees (default: 0).")
    parser.add_argument(
        "--occ-mode",
        choices=["raw", "white_free", "gray_free"],
        default="raw",
        help="OCC output style. raw=keep occupancy.png colors; white_free=white traversable; gray_free=gray traversable.",
    )
    parser.add_argument("--occ-free-threshold", type=int, default=250,
                        help="Pixel threshold for free space in occupancy.png.")
    parser.add_argument(
        "--occ-free-is-white",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat white as free in occupancy.png (default: true).",
    )
    parser.add_argument("--history-frames", type=int, default=16, help="History frames for mosaic (default: 16).")
    parser.add_argument("--future-actions", type=int, default=4, help="Future actions to include (default: 4).")
    parser.add_argument("--waypoint-mode", choices=["future", "horizon"], default="future",
                        help="Waypoints: future uses current+N future frames; horizon uses ~3m/1.5m distance.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride over frames (default: 1).")
    parser.add_argument("--scene", default="", help="Optional single scene id.")
    parser.add_argument("--label", default="", help="Optional single label id.")
    parser.add_argument("--max-samples", type=int, default=0, help="Stop after N samples (0 = no limit).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    fpv_root = Path(args.fpv_root)
    tasks_dir = Path(args.tasks_dir)
    scenes_dir = Path(args.scenes_dir)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / args.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset: List[Dict[str, Any]] = []
    total_samples = 0

    scenes = (
        [fpv_root / args.scene]
        if args.scene
        else sorted([p for p in fpv_root.iterdir() if p.is_dir()])
    )

    for scene_dir in scenes:
        scene_id = scene_dir.name
        if args.scene and scene_id != args.scene:
            continue

        occ_path = scenes_dir / scene_id / "occupancy.png"
        if not occ_path.is_file():
            if args.verbose:
                print(f"[skip] missing occupancy: {occ_path}")
            continue
        scale = load_scale(scenes_dir / scene_id)
        horizon_px = int(round(3.0 / scale))
        occ_base = build_occ_base(
            occ_path,
            free_threshold=args.occ_free_threshold,
            free_is_white=bool(args.occ_free_is_white),
            occ_mode=args.occ_mode,
        )

        bev_path = Path(args.bev_root) / f"{scene_id}.png" if args.bev_root else occ_path
        if not bev_path.is_file():
            bev_path = occ_path
        bev_out_path = images_dir / f"{scene_id}_bev.png"
        if not bev_out_path.is_file():
            bev_img = Image.open(bev_path).convert("RGB")
            if args.bev_rotate_ccw % 360 != 0:
                bev_img = bev_img.rotate(args.bev_rotate_ccw, expand=True, resample=Image.BICUBIC)
            bev_img.save(bev_out_path)

        action_files = sorted(scene_dir.glob("*_actions.json"))
        for action_path in action_files:
            label = action_path.stem.replace("_actions", "")
            if args.label and label != args.label:
                continue

            mp4_path = scene_dir / f"{label}.mp4"
            if not mp4_path.is_file():
                if args.verbose:
                    print(f"[skip] missing mp4: {mp4_path}")
                continue

            label_json = tasks_dir / scene_id / f"{label}.json"
            if not label_json.is_file():
                if args.verbose:
                    print(f"[skip] missing label json: {label_json}")
                continue

            task_data = load_json(label_json)
            instruction = task_data.get("instruction", "") or task_data.get("moving_instruction", "")
            if not isinstance(instruction, str):
                instruction = str(instruction)
            task_type = infer_task_type(instruction)
            goal_px = None
            goal_px = get_xy(task_data.get("goal", {})) or get_xy(task_data.get("label", {}).get("goal_pixel"))

            action_data = load_json(action_path)
            frames = action_data.get("frames", [])
            if not isinstance(frames, list) or not frames:
                if args.verbose:
                    print(f"[skip] no frames in {action_path}")
                continue

            idx_to_pos = {int(f.get("frame", i)): i for i, f in enumerate(frames)}
            history: Deque[np.ndarray] = deque(maxlen=max(1, args.history_frames))
            label_dir = images_dir / f"{scene_id}_{label}"
            label_dir.mkdir(parents=True, exist_ok=True)

            for vid_idx, frame in iter_video_frames(mp4_path):
                if args.frame_stride > 1 and (vid_idx % args.frame_stride) != 0:
                    history.append(frame)
                    continue
                if vid_idx not in idx_to_pos:
                    history.append(frame)
                    continue
                pos = idx_to_pos[vid_idx]
                frame_info = frames[pos]

                pixel = frame_info.get("pixel")
                if not (isinstance(pixel, list) and len(pixel) == 2):
                    history.append(frame)
                    continue
                cur_pixel = (int(pixel[0]), int(pixel[1]))

                start_pixel = frames[0].get("pixel")
                if not (isinstance(start_pixel, list) and len(start_pixel) == 2):
                    start_pixel = pixel
                start_pixel = (int(start_pixel[0]), int(start_pixel[1]))

                if task_type == "tracking":
                    hist_img = Image.new("RGB", (frame.shape[1], frame.shape[0]), (0, 0, 0))
                else:
                    hist_img = build_history_mosaic(list(history), (frame.shape[1], frame.shape[0]), grid=4)

                rgb_path = label_dir / f"t{vid_idx:05d}_rgb.png"
                hist_path = label_dir / f"t{vid_idx:05d}_hist.png"
                occ_path_out = label_dir / f"t{vid_idx:05d}_occ.png"
                Image.fromarray(frame).save(rgb_path)
                hist_img.save(hist_path)

                occ_img = occ_base.copy()
                draw_dot(occ_img, start_pixel, radius=3, color=(0, 255, 0))
                draw_dot(occ_img, cur_pixel, radius=3, color=(255, 0, 0))
                occ_img.save(occ_path_out)

                if args.waypoint_mode == "future":
                    waypoints = build_waypoints_future(frames, pos, future_steps=args.future_actions)
                else:
                    waypoints = build_waypoints_horizon(frames, pos, horizon_px=horizon_px, goal_px=goal_px)

                curr_action = frame_info.get("curr_action")
                next_actions = frame_info.get("next_actions", [])
                if not isinstance(next_actions, list):
                    next_actions = []
                actions = [int(curr_action)] if curr_action is not None else []
                actions.extend([int(a) for a in next_actions[: args.future_actions]])

                occ_free_label = "gray" if args.occ_mode == "gray_free" else "white"
                prompt = build_prompt(
                    instruction=instruction,
                    cur_pixel=cur_pixel,
                    scale=scale,
                    task_type=task_type,
                    horizon_px=horizon_px,
                    occ_free_label=occ_free_label,
                    future_steps=args.future_actions,
                    waypoint_mode=args.waypoint_mode,
                )
                assistant_obj = {
                    "pixel": waypoints,
                    "vlnce": f"<action>{','.join(str(a) for a in actions)}</action>",
                }

                sample = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
                    ],
                    "images": [
                        os.path.join(args.images_dir, f"{scene_id}_{label}", rgb_path.name).replace("\\", "/"),
                        os.path.join(args.images_dir, f"{scene_id}_{label}", hist_path.name).replace("\\", "/"),
                        os.path.join(args.images_dir, bev_out_path.name).replace("\\", "/"),
                        os.path.join(args.images_dir, f"{scene_id}_{label}", occ_path_out.name).replace("\\", "/"),
                    ],
                    "meta": {
                        "task_type": task_type,
                        "scene": scene_id,
                        "label": label,
                        "source_json": str(label_json),
                        "source_mp4": str(mp4_path),
                        "prompt_type": "instruction",
                        "action_index": int(vid_idx),
                        "cur_pixel": [cur_pixel[0], cur_pixel[1]],
                        "start_pixel": [start_pixel[0], start_pixel[1]],
                        "goal_pixel": [goal_px[0], goal_px[1]] if goal_px else None,
                        "scale_m_per_px": scale,
                        "horizon_px": horizon_px,
                        "future_actions": args.future_actions,
                        "waypoint_mode": args.waypoint_mode,
                    },
                }
                dataset.append(sample)
                total_samples += 1
                if args.max_samples and total_samples >= args.max_samples:
                    break

                history.append(frame)

            if args.max_samples and total_samples >= args.max_samples:
                break

        if args.max_samples and total_samples >= args.max_samples:
            break

    if not dataset:
        raise SystemExit("No samples generated.")

    out_json = out_dir / args.dataset_json
    out_jsonl = out_dir / args.dataset_jsonl
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for sample in dataset:
            handle.write(json.dumps(sample, ensure_ascii=False))
            handle.write("\n")

    print(f"[OK] Wrote {len(dataset)} samples.")
    print(f"[OK] JSON:  {out_json}")
    print(f"[OK] JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()
