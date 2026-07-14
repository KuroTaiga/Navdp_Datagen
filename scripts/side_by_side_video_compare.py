#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _pad_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(
        frame,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _stack_frames(left: np.ndarray, right: np.ndarray, *, divider: int) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1] + divider
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, : left.shape[1], :] = left
    canvas[:, left.shape[1] + divider :, :] = right
    return canvas


def _pad_to_multiple(frame: np.ndarray, multiple: int) -> np.ndarray:
    if multiple <= 1:
        return frame
    height, width = frame.shape[:2]
    pad_h = (multiple - (height % multiple)) % multiple
    pad_w = (multiple - (width % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(
        frame,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _video_fps(reader) -> float | None:
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps")
        return float(fps) if fps else None
    except Exception:
        return None


def _iter_frames(reader, max_frames: int | None):
    count = 0
    for frame in reader:
        yield frame
        count += 1
        if max_frames is not None and count >= max_frames:
            break


def _compare_pair(
    *,
    golden_path: Path,
    candidate_path: Path,
    out_path: Path,
    frames_dir: Path,
    max_frames: int | None,
    divider: int,
    fps: float | None,
    lossless: bool,
    macro_block_size: int,
    png_only: bool,
) -> dict:
    payload = {
        "golden": str(golden_path),
        "candidate": str(candidate_path),
        "output": None if png_only else str(out_path),
        "frames_dir": str(frames_dir) if png_only else None,
        "frames_written": 0,
        "error": None,
        "mode": "png" if png_only else "video",
    }
    if not golden_path.is_file():
        payload["error"] = "golden_missing"
        return payload
    if not candidate_path.is_file():
        payload["error"] = "candidate_missing"
        return payload

    try:
        with imageio.get_reader(str(golden_path), "ffmpeg") as golden_reader, imageio.get_reader(
            str(candidate_path), "ffmpeg"
        ) as cand_reader:
            fps_value = fps or _video_fps(cand_reader) or _video_fps(golden_reader) or 10.0
            if png_only:
                frames_dir.mkdir(parents=True, exist_ok=True)
                golden_iter = _iter_frames(golden_reader, max_frames)
                cand_iter = _iter_frames(cand_reader, max_frames)
                for idx, (g_frame, c_frame) in enumerate(zip(golden_iter, cand_iter)):
                    g_frame = _normalize_frame(np.asarray(g_frame))
                    c_frame = _normalize_frame(np.asarray(c_frame))
                    target_h = max(g_frame.shape[0], c_frame.shape[0])
                    target_w = max(g_frame.shape[1], c_frame.shape[1])
                    g_frame = _pad_frame(g_frame, target_h, target_w)
                    c_frame = _pad_frame(c_frame, target_h, target_w)
                    stacked = _stack_frames(g_frame, c_frame, divider=divider)
                    frame_path = frames_dir / f"frame_{idx:06d}.png"
                    imageio.imwrite(frame_path, stacked)
                    payload["frames_written"] += 1
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if lossless:
                    writer = imageio.get_writer(
                        str(out_path),
                        fps=fps_value,
                        codec="libx264rgb",
                        pixelformat="rgb24",
                        output_params=["-crf", "0", "-preset", "ultrafast"],
                        macro_block_size=macro_block_size,
                    )
                else:
                    writer = imageio.get_writer(
                        str(out_path),
                        fps=fps_value,
                        codec="libx264",
                        pixelformat="yuv420p",
                        macro_block_size=macro_block_size,
                    )
                with writer:
                    golden_iter = _iter_frames(golden_reader, max_frames)
                    cand_iter = _iter_frames(cand_reader, max_frames)
                    for g_frame, c_frame in zip(golden_iter, cand_iter):
                        g_frame = _normalize_frame(np.asarray(g_frame))
                        c_frame = _normalize_frame(np.asarray(c_frame))
                        target_h = max(g_frame.shape[0], c_frame.shape[0])
                        target_w = max(g_frame.shape[1], c_frame.shape[1])
                        g_frame = _pad_frame(g_frame, target_h, target_w)
                        c_frame = _pad_frame(c_frame, target_h, target_w)
                        stacked = _stack_frames(g_frame, c_frame, divider=divider)
                        stacked = _pad_to_multiple(stacked, macro_block_size)
                        writer.append_data(stacked)
                        payload["frames_written"] += 1
    except Exception as exc:  # pylint: disable=broad-except
        payload["error"] = str(exc)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create side-by-side MP4s comparing golden vs candidate videos."
    )
    parser.add_argument("--golden-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--label-id",
        action="append",
        dest="label_ids",
        help="Label ID to compare; repeat for multiple.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("./analysis/quick_gpu_side_by_side"),
        help="Output root for side-by-side videos.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--divider", type=int, default=6, help="Pixel divider between frames.")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--lossless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use lossless RGB encoding (default: off).",
    )
    parser.add_argument(
        "--png-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write side-by-side PNGs instead of an MP4 (default: off).",
    )
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=16,
        help="Pad output to a multiple of this size to avoid resizing (default: 16).",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    label_ids = args.label_ids or []
    if not label_ids:
        print("[ERROR] Provide at least one --label-id to compare.", file=sys.stderr)
        return 1

    report = {
        "golden_root": str(args.golden_root),
        "candidate_root": str(args.candidate_root),
        "scene": args.scene,
        "mode": "png" if args.png_only else "video",
        "results": {},
    }
    for label_id in label_ids:
        golden_path = args.golden_root / args.scene / f"{label_id}.mp4"
        candidate_path = args.candidate_root / args.scene / f"{label_id}.mp4"
        frames_dir = args.out_root / args.scene / label_id
        out_path = args.out_root / args.scene / f"{label_id}_side_by_side.mp4"
        result = _compare_pair(
            golden_path=golden_path,
            candidate_path=candidate_path,
            out_path=out_path,
            frames_dir=frames_dir,
            max_frames=args.max_frames,
            divider=args.divider,
            fps=args.fps,
            lossless=bool(args.lossless),
            macro_block_size=int(args.macro_block_size),
            png_only=bool(args.png_only),
        )
        report["results"][label_id] = result

    report_path = args.report_json or (args.out_root / args.scene / "side_by_side_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[DONE] Wrote side-by-side report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
