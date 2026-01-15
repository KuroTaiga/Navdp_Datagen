#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from lighting.lighting_utils import (  # noqa: E402
    LightFilterConfig,
    LumaStats,
    apply_light_filter,
    stable_hash_seed,
)


def _collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob(pattern))


def _output_path(
    input_root: Path,
    src_path: Path,
    output_dir: Path | None,
    suffix: str,
) -> Path:
    if output_dir is None:
        return src_path.with_name(f"{src_path.stem}{suffix}{src_path.suffix}")
    rel = src_path.relative_to(input_root)
    out_path = (output_dir / rel).with_name(f"{src_path.stem}{suffix}{src_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame * 255.0, 0.0, 255.0)
    return frame.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply lighting filter to MP4s and report stats.")
    parser.add_argument("input", type=Path, help="Input MP4 file or directory.")
    parser.add_argument("--pattern", type=str, default="*.mp4", help="Glob for input videos.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory root.")
    parser.add_argument("--suffix", type=str, default="_light", help="Suffix appended to output filenames.")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most this many files.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most this many frames per file.")
    parser.add_argument("--pixel-step", type=int, default=4, help="Sample every Nth pixel for stats.")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats without writing output videos.")

    parser.add_argument("--light-mode", choices=("disk", "cl", "global"), default="disk")
    parser.add_argument("--light-strength", type=float, default=0.25)
    parser.add_argument("--light-radius", type=float, default=0.45)
    parser.add_argument("--light-center", type=float, nargs=2, default=(0.5, 0.5), metavar=("X", "Y"))
    parser.add_argument("--light-jitter", type=float, default=0.0)
    parser.add_argument("--light-temp-k", type=float, default=0.0)
    parser.add_argument("--light-vignette", type=float, default=0.0)
    parser.add_argument("--light-seed", type=int, default=0)

    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    files = _collect_files(input_path, args.pattern)
    if not files:
        raise SystemExit(f"No files matched {args.pattern} under {input_path}")

    cfg = LightFilterConfig(
        mode=str(args.light_mode),
        strength=float(args.light_strength),
        radius_frac=float(args.light_radius),
        center_xy=(float(args.light_center[0]), float(args.light_center[1])),
        center_jitter=float(args.light_jitter),
        temp_k=float(args.light_temp_k),
        vignette=float(args.light_vignette),
        seed=int(args.light_seed),
    )

    max_files = int(args.max_files) if args.max_files is not None else None
    max_frames = int(args.max_frames) if args.max_frames is not None else None
    pixel_step = max(1, int(args.pixel_step))
    input_root = input_path if input_path.is_dir() else input_path.parent

    total_base = LumaStats()
    total_filtered = LumaStats()
    file_reports: list[dict] = []
    total_frames = 0
    total_filter_sec = 0.0
    total_elapsed = 0.0

    for idx, src_path in enumerate(files):
        if max_files is not None and idx >= max_files:
            break
        out_path = _output_path(input_root, src_path, args.output_dir, args.suffix)
        reader = imageio.get_reader(src_path)
        meta = reader.get_meta_data()
        fps = meta.get("fps", 10)
        writer = None
        if not args.dry_run:
            writer = imageio.get_writer(out_path, fps=fps)
        base_stats = LumaStats()
        filtered_stats = LumaStats()
        frames = 0
        filter_sec = 0.0
        start = time.perf_counter()
        seed_offset = stable_hash_seed(str(src_path))
        try:
            for frame_index, frame in enumerate(reader):
                if max_frames is not None and frames >= max_frames:
                    break
                base_stats.update_from_frame(frame, pixel_step=pixel_step)
                total_base.update_from_frame(frame, pixel_step=pixel_step)
                t0 = time.perf_counter()
                filtered = apply_light_filter(
                    frame,
                    cfg,
                    frame_index=frame_index,
                    seed_offset=seed_offset,
                )
                filter_sec += time.perf_counter() - t0
                filtered_stats.update_from_frame(filtered, pixel_step=pixel_step)
                total_filtered.update_from_frame(filtered, pixel_step=pixel_step)
                if writer is not None:
                    writer.append_data(_frame_to_uint8(filtered))
                frames += 1
        finally:
            reader.close()
            if writer is not None:
                writer.close()
        elapsed = time.perf_counter() - start
        total_frames += frames
        total_filter_sec += filter_sec
        total_elapsed += elapsed
        report = {
            "file": str(src_path),
            "output": None if args.dry_run else str(out_path),
            "frames": frames,
            "elapsed_sec": elapsed,
            "filter_sec": filter_sec,
            "fps_total": (frames / elapsed) if frames > 0 and elapsed > 0 else None,
            "fps_filter": (frames / filter_sec) if frames > 0 and filter_sec > 0 else None,
        }
        report.update({"base": base_stats.finalize(), "filtered": filtered_stats.finalize()})
        file_reports.append(report)

    summary = {
        "input": str(input_path),
        "pattern": args.pattern,
        "output_dir": None if args.dry_run else str(args.output_dir) if args.output_dir else None,
        "suffix": args.suffix,
        "light_config": cfg.__dict__,
        "files_processed": len(file_reports),
        "frames_processed": total_frames,
        "elapsed_sec": total_elapsed,
        "filter_sec": total_filter_sec,
        "fps_total": (total_frames / total_elapsed) if total_frames > 0 and total_elapsed > 0 else None,
        "fps_filter": (total_frames / total_filter_sec) if total_frames > 0 and total_filter_sec > 0 else None,
        "base": total_base.finalize(),
        "filtered": total_filtered.finalize(),
        "per_file": file_reports,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
