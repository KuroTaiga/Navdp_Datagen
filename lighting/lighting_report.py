#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from lighting.lighting_utils import LumaStats  # noqa: E402


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob(pattern))


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Report base lighting level statistics.")
    parser.add_argument(
        "input",
        type=Path,
        help="Input file or directory containing frames/videos.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern when input is a directory (default: *.mp4).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame for videos (default: 1).",
    )
    parser.add_argument(
        "--pixel-step",
        type=int,
        default=4,
        help="Sample every Nth pixel in each dimension (default: 4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames total (default: no limit).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop after this many files (default: no limit).",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Include per-file lighting stats in the output.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    frame_step = max(1, int(args.frame_step))
    pixel_step = max(1, int(args.pixel_step))
    max_frames = int(args.max_frames) if args.max_frames is not None else None
    max_files = int(args.max_files) if args.max_files is not None else None

    files = _collect_files(input_path, args.pattern)
    if not files:
        raise SystemExit(f"No files matched {args.pattern} under {input_path}")

    total_stats = LumaStats()
    per_file_stats: list[dict] = []
    frames_processed = 0
    files_processed = 0
    start = time.perf_counter()
    stop = False

    for path in files:
        if max_files is not None and files_processed >= max_files:
            break
        file_stats = LumaStats()
        file_frames = 0
        if _is_video(path):
            reader = imageio.get_reader(path)
            try:
                for idx, frame in enumerate(reader):
                    if idx % frame_step != 0:
                        continue
                    total_stats.update_from_frame(frame, pixel_step=pixel_step)
                    file_stats.update_from_frame(frame, pixel_step=pixel_step)
                    frames_processed += 1
                    file_frames += 1
                    if max_frames is not None and frames_processed >= max_frames:
                        stop = True
                        break
            finally:
                reader.close()
        else:
            frame = imageio.imread(path)
            total_stats.update_from_frame(frame, pixel_step=pixel_step)
            file_stats.update_from_frame(frame, pixel_step=pixel_step)
            frames_processed += 1
            file_frames += 1
            if max_frames is not None and frames_processed >= max_frames:
                stop = True
        files_processed += 1
        if args.per_file:
            payload = file_stats.finalize()
            payload.update(
                {
                    "file": str(path),
                    "frames_processed": file_frames,
                }
            )
            per_file_stats.append(payload)
        if stop:
            break

    elapsed = time.perf_counter() - start
    report = {
        "input": str(input_path),
        "pattern": args.pattern,
        "frame_step": frame_step,
        "pixel_step": pixel_step,
        "max_frames": max_frames,
        "max_files": max_files,
        "files_processed": files_processed,
        "frames_processed": frames_processed,
        "elapsed_sec": elapsed,
    }
    report.update(total_stats.finalize())
    if args.per_file:
        report["per_file"] = per_file_stats

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
