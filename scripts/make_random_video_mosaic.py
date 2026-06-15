#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = BASE_DIR / "data2" / "0500_fpv"
DEFAULT_OUTPUT = BASE_DIR / "analysis" / "video_mosaics" / "random_grid_16x10.mp4"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    duration: float


@dataclass(frozen=True)
class ClipChoice:
    path: Path
    duration: float
    start_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a random NxM video mosaic from a dataset folder. "
            "Videos are center-cropped to square (based on shorter side), then tiled."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root folder to recursively search for videos (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output MP4 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--cols", type=int, default=16, help="Number of columns (default: 16)")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows (default: 10)")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Tile size in pixels (each tile is tile-size x tile-size, default: 128)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum source video duration in seconds to be eligible (default: 5.0)",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=45.0,
        help="Output mosaic duration in seconds (default: 45.0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output FPS for every tile/output video (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=20260324, help="Random seed (default: 20260324)")
    parser.add_argument(
        "--allow-duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow reusing videos when eligible count < rows*cols (default: true)",
    )
    parser.add_argument(
        "--loop-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop each input if needed so output can exceed source duration (default: true)",
    )
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=16,
        help="Parallel workers for ffprobe metadata scan (default: 16)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="ffmpeg video codec for output (default: libx264)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="ffmpeg preset for codec (default: veryfast)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="ffmpeg CRF quality (default: 20)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON path to save selected clips metadata (default: <output>.manifest.json)",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg binary name/path (default: ffmpeg)",
    )
    parser.add_argument(
        "--ffprobe-bin",
        type=str,
        default="ffprobe",
        help="ffprobe binary name/path (default: ffprobe)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned command and selections, but do not render output video",
    )
    return parser.parse_args()


def resolve_binary(name_or_path: str, label: str) -> str:
    resolved = shutil.which(name_or_path)
    if resolved:
        return resolved
    path = Path(name_or_path)
    if path.is_file() and path.exists():
        return str(path)
    raise RuntimeError(f"Could not find {label} binary: {name_or_path}")


def discover_videos(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Input root is not a directory: {root}")
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(paths)


def probe_duration(path: Path, ffprobe_bin: str) -> VideoMeta | None:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    try:
        duration = float(text)
    except ValueError:
        return None
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return VideoMeta(path=path, duration=duration)


def probe_videos(video_paths: list[Path], ffprobe_bin: str, workers: int) -> list[VideoMeta]:
    if not video_paths:
        return []
    workers = max(1, int(workers))
    if workers == 1:
        out = [probe_duration(path, ffprobe_bin) for path in video_paths]
        return [meta for meta in out if meta is not None]

    metas: list[VideoMeta] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for meta in pool.map(lambda p: probe_duration(p, ffprobe_bin), video_paths):
            if meta is not None:
                metas.append(meta)
    return metas


def choose_clips(
    *,
    eligible: list[VideoMeta],
    total_cells: int,
    clip_duration: float,
    rng: random.Random,
    allow_duplicates: bool,
) -> list[ClipChoice]:
    if not eligible:
        raise RuntimeError("No eligible videos after filtering.")
    if len(eligible) < total_cells and not allow_duplicates:
        raise RuntimeError(
            f"Need {total_cells} videos but only found {len(eligible)} eligible and duplicates are disabled."
        )

    if len(eligible) >= total_cells:
        picked = rng.sample(eligible, total_cells)
    else:
        picked = [rng.choice(eligible) for _ in range(total_cells)]

    choices: list[ClipChoice] = []
    for meta in picked:
        # Start anywhere in the source clip; if output duration exceeds source,
        # ffmpeg input looping can fill the remaining duration.
        max_start = max(0.0, meta.duration - 1e-3)
        start_time = rng.uniform(0.0, max_start) if max_start > 0.0 else 0.0
        choices.append(ClipChoice(path=meta.path, duration=meta.duration, start_time=start_time))
    return choices


def build_filter_complex(rows: int, cols: int, tile_size: int, fps: float) -> str:
    total = rows * cols
    # Build each branch explicitly. Crop to center square:
    # - landscape: keep full height and center-crop width
    # - portrait:  center-crop height
    per_input = []
    for idx in range(total):
        per_input.append(
            ""
            f"[{idx}:v]"
            f"fps={fps},"
            "crop='if(gte(iw,ih),ih,iw)':'if(gte(iw,ih),ih,iw)':"
            "'if(gte(iw,ih),(iw-ih)/2,0)':'if(gte(iw,ih),0,(ih-iw)/2)',"
            f"scale={tile_size}:{tile_size}:flags=lanczos,"
            "setsar=1"
            f"[v{idx}]"
        )

    layout_parts = []
    for idx in range(total):
        r = idx // cols
        c = idx % cols
        layout_parts.append(f"{c * tile_size}_{r * tile_size}")

    stack_inputs = "".join(f"[v{idx}]" for idx in range(total))
    xstack = (
        f"{stack_inputs}"
        f"xstack=inputs={total}:layout={'|'.join(layout_parts)}:fill=black"
        "[outv]"
    )
    return ";".join(per_input + [xstack])


def run_ffmpeg(
    *,
    ffmpeg_bin: str,
    clips: list[ClipChoice],
    clip_duration: float,
    fps: float,
    rows: int,
    cols: int,
    tile_size: int,
    output: Path,
    codec: str,
    preset: str,
    crf: int,
    loop_inputs: bool,
    dry_run: bool,
) -> None:
    total = rows * cols
    if len(clips) != total:
        raise RuntimeError(f"Expected {total} clips, got {len(clips)}")

    filter_complex = build_filter_complex(rows=rows, cols=cols, tile_size=tile_size, fps=fps)

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [ffmpeg_bin, "-y"]
    for clip in clips:
        if loop_inputs:
            cmd.extend(["-stream_loop", "-1"])
        cmd.extend(["-ss", f"{clip.start_time:.3f}", "-t", f"{clip_duration:.3f}", "-i", str(clip.path)])

    if dry_run:
        print("[DRY-RUN] FFmpeg command preview:")
        print(" ".join(shlex.quote(c) for c in cmd[:12]))
        print(f"[DRY-RUN] ... plus {max(0, len(cmd) - 12)} more args")
        print(f"[DRY-RUN] Output path: {output}")
        print(f"[DRY-RUN] Grid: {cols}x{rows}, tile: {tile_size}, fps: {fps}, clip duration: {clip_duration}")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffmpeg_filter.txt", delete=False) as tmp:
        tmp.write(filter_complex)
        filter_path = Path(tmp.name)

    ffmpeg_cmd = cmd + [
        "-filter_complex_script",
        str(filter_path),
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]

    try:
        proc = subprocess.run(ffmpeg_cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with return code {proc.returncode}")
    finally:
        try:
            filter_path.unlink(missing_ok=True)
        except OSError:
            pass


def write_manifest(
    *,
    manifest_path: Path,
    input_root: Path,
    output: Path,
    rows: int,
    cols: int,
    tile_size: int,
    min_duration: float,
    clip_duration: float,
    fps: float,
    seed: int,
    eligible_count: int,
    clips: list[ClipChoice],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_root": str(input_root),
        "output": str(output),
        "grid": {"rows": rows, "cols": cols, "cells": rows * cols},
        "tile_size": tile_size,
        "min_duration": min_duration,
        "clip_duration": clip_duration,
        "fps": fps,
        "seed": seed,
        "eligible_video_count": eligible_count,
        "selected": [
            {
                "path": str(c.path),
                "duration": round(c.duration, 6),
                "start_time": round(c.start_time, 6),
                "clip_duration": round(clip_duration, 6),
            }
            for c in clips
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.rows <= 0 or args.cols <= 0:
        print("[ERROR] --rows and --cols must be positive.", file=sys.stderr)
        return 1
    if args.tile_size <= 0:
        print("[ERROR] --tile-size must be positive.", file=sys.stderr)
        return 1
    if args.min_duration <= 0 or args.clip_duration <= 0:
        print("[ERROR] --min-duration and --clip-duration must be positive.", file=sys.stderr)
        return 1
    if args.fps <= 0:
        print("[ERROR] --fps must be positive.", file=sys.stderr)
        return 1

    try:
        ffmpeg_bin = resolve_binary(args.ffmpeg_bin, label="ffmpeg")
        ffprobe_bin = resolve_binary(args.ffprobe_bin, label="ffprobe")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    total_cells = args.rows * args.cols
    rng = random.Random(args.seed)

    print(f"[INFO] Discovering videos under: {args.input_root}")
    try:
        video_paths = discover_videos(args.input_root)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] Found {len(video_paths)} candidate video file(s).")
    if not video_paths:
        print("[ERROR] No video files found.", file=sys.stderr)
        return 1

    print(f"[INFO] Probing durations with {args.probe_workers} worker(s)...")
    metas = probe_videos(video_paths, ffprobe_bin=ffprobe_bin, workers=args.probe_workers)
    threshold = float(args.min_duration)
    eligible = [m for m in metas if m.duration >= threshold]

    print(
        "[INFO] Eligible videos: "
        f"{len(eligible)} / {len(metas)} (duration >= {threshold:.2f}s)"
    )
    if not eligible:
        print("[ERROR] No eligible videos found after duration filter.", file=sys.stderr)
        return 1

    clips = choose_clips(
        eligible=eligible,
        total_cells=total_cells,
        clip_duration=float(args.clip_duration),
        rng=rng,
        allow_duplicates=bool(args.allow_duplicates),
    )

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_manifest(
        manifest_path=manifest_path,
        input_root=args.input_root,
        output=args.output,
        rows=int(args.rows),
        cols=int(args.cols),
        tile_size=int(args.tile_size),
        min_duration=float(args.min_duration),
        clip_duration=float(args.clip_duration),
        fps=float(args.fps),
        seed=int(args.seed),
        eligible_count=len(eligible),
        clips=clips,
    )
    print(f"[INFO] Wrote manifest: {manifest_path}")

    try:
        run_ffmpeg(
            ffmpeg_bin=ffmpeg_bin,
            clips=clips,
            clip_duration=float(args.clip_duration),
            fps=float(args.fps),
            rows=int(args.rows),
            cols=int(args.cols),
            tile_size=int(args.tile_size),
            output=args.output,
            codec=str(args.codec),
            preset=str(args.preset),
            crf=int(args.crf),
            loop_inputs=bool(args.loop_inputs),
            dry_run=bool(args.dry_run),
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("[DONE] Dry-run completed; no output video rendered.")
    else:
        print(f"[DONE] Mosaic video saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
