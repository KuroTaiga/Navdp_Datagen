#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from lighting.lighting_utils import (  # noqa: E402
    LightFilterConfig,
    LumaStats,
    apply_light_filter,
)


def _collect_mp4s(
    root: Path,
    pattern: str,
    *,
    progress_every: int = 0,
    logger: Callable[[str], None] | None = None,
) -> list[Path]:
    matches: list[Path] = []
    start = time.perf_counter()
    for idx, path in enumerate(root.rglob(pattern), start=1):
        matches.append(path)
        if progress_every > 0 and idx % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = idx / elapsed if elapsed > 0 else 0.0
            message = f"[SCAN] matched {idx} files ({rate:.1f}/s)..."
            if logger is None:
                print(message, flush=True)
            else:
                logger(message)
    return sorted(matches)


def _format_luma(value: float) -> str:
    return f"{value:.3f}"


def _format_scale(value: float) -> str:
    text = f"{value:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_ev(scale: float) -> str:
    ev = math.log2(float(scale))
    sign = "p" if ev >= 0 else "m"
    return f"EV{sign}{abs(ev):.2f}"


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "skip":
        return
    if mode == "link":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def _ensure_writable_dir(path: Path, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"{label} must be a directory: {path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_path = path / f".write_test_{os.getpid()}"
        with test_path.open("w", encoding="utf-8") as handle:
            handle.write("")
        test_path.unlink()
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"{label} is not writable: {path} ({exc})") from exc


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame * 255.0, 0.0, 255.0)
    return frame.astype(np.uint8)


def _load_mp4_list(list_path: Path, root: Path) -> list[Path]:
    if not list_path.is_file():
        raise SystemExit(f"MP4 list not found: {list_path}")
    paths: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = root / candidate
        paths.append(candidate)
    return paths

def _process_mp4_task(
    src_path_str: str,
    rel_path_str: str,
    scale_items: list[tuple[float, str]],
    *,
    overwrite: bool,
    worker_log: bool,
    task_index: int | None = None,
    task_total: int | None = None,
) -> dict:
    src_path = Path(src_path_str)
    rel_path = Path(rel_path_str)
    parts = rel_path.parts
    scene = parts[0] if len(parts) >= 1 else "-"
    path_id = parts[1] if len(parts) >= 2 else "-"
    progress = (
        f"{task_index}/{task_total}"
        if task_index is not None and task_total is not None
        else "-"
    )
    pid = os.getpid()
    if worker_log:
        print(
            f"[WORKER {pid}] start {progress} scene={scene} path={path_id} rel={rel_path}",
            flush=True,
        )
    start = time.perf_counter()
    reader = imageio.get_reader(src_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 10)
    writers: dict[float, imageio.core.format.Writer] = {}
    configs: dict[float, LightFilterConfig] = {}
    frames = 0
    try:
        for scale, out_dir_str in scale_items:
            out_dir = Path(out_dir_str)
            out_path = out_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                continue
            writers[scale] = imageio.get_writer(out_path, fps=fps)
            configs[scale] = LightFilterConfig(
                mode="global",
                strength=float(scale) - 1.0,
                radius_frac=0.0,
                center_xy=(0.5, 0.5),
                center_jitter=0.0,
                temp_k=0.0,
                vignette=0.0,
                seed=0,
            )

        for frame_index, frame in enumerate(reader):
            if not writers:
                break
            for scale, writer in writers.items():
                filtered = apply_light_filter(frame, configs[scale], frame_index=frame_index)
                writer.append_data(_frame_to_uint8(filtered))
            frames += 1
    finally:
        reader.close()
        for writer in writers.values():
            writer.close()
    if worker_log:
        elapsed = time.perf_counter() - start
        fps_rate = frames / elapsed if elapsed > 0 else 0.0
        eta_sec = 0.0
        if task_index is not None and task_total is not None:
            remaining = max(0, task_total - task_index)
            eta_sec = remaining * elapsed
        print(
            f"[WORKER {pid}] done {progress} scene={scene} path={path_id} "
            f"rel={rel_path} frames={frames} fps={fps_rate:.1f} "
            f"elapsed={elapsed:.1f}s eta~{_format_eta(eta_sec)}",
            flush=True,
        )
    return {
        "src": src_path_str,
        "rel": rel_path_str,
        "scene": scene,
        "path": path_id,
        "pid": pid,
        "frames": frames,
        "outputs": len(writers),
    }


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:02d}m{sec:02d}s"


def _write_progress_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def _group_by_scene(files: list[Path], root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        rel = path.relative_to(root)
        parts = rel.parts
        scene = parts[0] if len(parts) > 1 else "_root"
        groups.setdefault(scene, []).append(path)
    return groups


def _sample_files_per_scene(
    files: list[Path],
    *,
    root: Path,
    per_scene: int,
    seed: int,
    max_scenes: int | None = None,
) -> list[Path]:
    if per_scene <= 0:
        return files
    rng = np.random.default_rng(int(seed))
    grouped = _group_by_scene(files, root)
    scene_keys = list(grouped.keys())
    if max_scenes is not None and max_scenes > 0 and len(scene_keys) > max_scenes:
        scene_keys = list(rng.choice(scene_keys, size=max_scenes, replace=False))
    sampled: list[Path] = []
    for scene in scene_keys:
        paths = grouped[scene]
        if len(paths) <= per_scene:
            sampled.extend(paths)
            continue
        indices = rng.choice(len(paths), size=per_scene, replace=False)
        sampled.extend(paths[idx] for idx in indices)
    return sorted(sampled)


def _compute_base_luma(
    files: list[Path],
    *,
    frame_step: int,
    pixel_step: int,
    max_frames: int | None,
) -> dict:
    stats = LumaStats()
    frames_processed = 0
    start = time.perf_counter()
    for path in files:
        reader = imageio.get_reader(path)
        try:
            for idx, frame in enumerate(reader):
                if idx % frame_step != 0:
                    continue
                stats.update_from_frame(frame, pixel_step=pixel_step)
                frames_processed += 1
                if max_frames is not None and frames_processed >= max_frames:
                    break
        finally:
            reader.close()
        if max_frames is not None and frames_processed >= max_frames:
            break
    elapsed = time.perf_counter() - start
    report = stats.finalize()
    report.update({"frames_processed": frames_processed, "elapsed_sec": elapsed})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build lighting-variant datasets from existing MP4 outputs."
    )
    parser.add_argument("input", type=Path, help="Input dataset directory.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern for MP4s under input (default: *.mp4).",
    )
    parser.add_argument(
        "--mp4-list",
        type=Path,
        default=None,
        help="Optional text file with one MP4 path per line (relative to input or absolute).",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=[1.5, 0.5, 0.2],
        help="Brightness scales relative to base luma (default: 1.5 0.5 0.2).",
    )
    parser.add_argument(
        "--suffix-mode",
        choices=("scale", "luma", "ev"),
        default="scale",
        help="Output folder naming: scale (e.g. _1.5L), luma (_0.300), or ev (_EVm1.00).",
    )
    parser.add_argument(
        "--base-luma",
        type=float,
        default=None,
        help="Override base luma mean (skips computing base luma).",
    )
    parser.add_argument(
        "--compute-base-luma",
        action="store_true",
        help="Compute base luma before processing (default: off unless needed).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame when computing base luma (default: 1).",
    )
    parser.add_argument(
        "--pixel-step",
        type=int,
        default=4,
        help="Sample every Nth pixel when computing base luma (default: 4).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames when computing base luma (default: no limit).",
    )
    parser.add_argument(
        "--base-sample-per-scene",
        type=int,
        default=None,
        help="If set, sample this many MP4s per scene for base luma estimation.",
    )
    parser.add_argument(
        "--base-sample-seed",
        type=int,
        default=12345,
        help="Seed used when sampling MP4s per scene for base luma.",
    )
    parser.add_argument(
        "--base-max-scenes",
        type=int,
        default=None,
        help="If set, randomly pick at most this many scenes for base luma estimation.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only compute and report base luma; skip dataset generation.",
    )
    parser.add_argument(
        "--other-mode",
        choices=("copy", "skip"),
        default="skip",
        help="How to handle non-MP4 files after MP4 processing (default: skip).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for MP4 processing (default: 1).",
    )
    parser.add_argument(
        "--worker-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log worker start/end lines for each MP4 (default: on).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N MP4s (default: 10).",
    )
    parser.add_argument(
        "--scan-progress-every",
        type=int,
        default=0,
        help="Print scan progress every N matched files while enumerating MP4s (default: 0).",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional JSON path for progress updates.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file to append live output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output MP4s (default: false).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    def log_line(message: str) -> None:
        print(message, flush=True)
        if args.log_file is not None:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            with args.log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{message}\n")

    input_dir = args.input
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not os.access(input_dir, os.R_OK | os.X_OK):
        raise SystemExit(f"Input directory is not readable: {input_dir}")

    if args.log_file is not None:
        _ensure_writable_dir(args.log_file.parent, "Log directory")
        try:
            with args.log_file.open("a", encoding="utf-8") as handle:
                handle.write("")
        except Exception as exc:  # pylint: disable=broad-except
            raise SystemExit(f"Log file is not writable: {args.log_file} ({exc})") from exc
    if args.progress_json is not None:
        _ensure_writable_dir(args.progress_json.parent, "Progress JSON directory")
    if args.output_json is not None:
        _ensure_writable_dir(args.output_json.parent, "Output report directory")

    log_line(
        f"[START] input={input_dir} pattern={args.pattern} workers={args.workers} scales={args.scales}"
    )

    if args.mp4_list is not None:
        mp4s = _load_mp4_list(args.mp4_list, input_dir)
    else:
        log_line("[SCAN] collecting mp4s...")
        mp4s = _collect_mp4s(
            input_dir,
            args.pattern,
            progress_every=int(args.scan_progress_every),
            logger=log_line,
        )
    if not mp4s:
        raise SystemExit(f"No MP4s matched {args.pattern} under {input_dir}")
    log_line(f"[SCAN] total mp4 files: {len(mp4s)}")

    frame_step = max(1, int(args.frame_step))
    pixel_step = max(1, int(args.pixel_step))
    max_frames = int(args.max_frames) if args.max_frames is not None else None

    base_report = None
    base_files = mp4s
    base_luma = args.base_luma
    should_compute_base = args.base_only or args.compute_base_luma or args.suffix_mode == "luma"
    if base_luma is None and should_compute_base:
        if args.base_sample_per_scene is not None and int(args.base_sample_per_scene) > 0:
            base_files = _sample_files_per_scene(
                mp4s,
                root=input_dir,
                per_scene=int(args.base_sample_per_scene),
                seed=int(args.base_sample_seed),
                max_scenes=(int(args.base_max_scenes) if args.base_max_scenes is not None else None),
            )
        base_report = _compute_base_luma(
            base_files,
            frame_step=frame_step,
            pixel_step=pixel_step,
            max_frames=max_frames,
        )
        base_luma = base_report.get("luma_mean")
        if base_luma is None:
            raise SystemExit("Unable to compute base luma.")
    else:
        log_line("[BASE] skipping base luma computation.")
    if args.base_only:
        report = {
            "input": str(input_dir),
            "pattern": args.pattern,
            "base": base_report,
            "base_sample_per_scene": args.base_sample_per_scene,
            "base_sample_seed": args.base_sample_seed,
            "base_max_scenes": args.base_max_scenes,
            "base_files": len(base_files),
            "total_files": len(mp4s),
        }
        log_line(json.dumps(report, indent=2))
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(report, indent=2))
        return

    output_dirs: list[Path] = []
    scale_map: dict[float, Path] = {}
    suffix_map: dict[float, str] = {}
    for scale in args.scales:
        if args.suffix_mode == "luma":
            if base_luma is None:
                raise SystemExit("--suffix-mode luma requires --base-luma or --compute-base-luma.")
            target_luma = float(base_luma) * float(scale)
            suffix = _format_luma(target_luma)
        elif args.suffix_mode == "ev":
            suffix = _format_ev(scale)
        else:
            suffix = f"{_format_scale(scale)}L"
        out_dir = input_dir.with_name(f"{input_dir.name}_{suffix}")
        output_dirs.append(out_dir)
        scale_map[float(scale)] = out_dir
        suffix_map[float(scale)] = suffix
        _ensure_writable_dir(out_dir, "Output directory")

    # Process MP4s once and write to each output (parallel if requested).
    scale_items = [(scale, str(out_dir)) for scale, out_dir in scale_map.items()]
    total = len(mp4s)
    start = time.perf_counter()
    completed = 0
    progress_every = max(1, int(args.progress_every))
    workers = max(1, int(args.workers))

    total_frames = 0
    total_outputs = 0
    failures = 0

    def emit_progress(
        last_scene: str | None = None,
        last_path: str | None = None,
        last_rel: str | None = None,
        last_pid: int | None = None,
    ) -> None:
        elapsed = time.perf_counter() - start
        file_rate = completed / elapsed if elapsed > 0 else 0.0
        frame_rate = total_frames / elapsed if elapsed > 0 else 0.0
        avg_frames = total_frames / completed if completed > 0 else 0.0
        avg_outputs = total_outputs / completed if completed > 0 else 0.0
        eta = (total - completed) / file_rate if file_rate > 0 else 0.0
        suffix = ""
        if last_scene or last_path or last_rel or last_pid is not None:
            suffix = (
                f" | last_scene={last_scene or '-'} last_path={last_path or '-'} "
                f"last_rel={last_rel or '-'} worker={last_pid or '-'}"
            )
        log_line(
            f"[MP4] {completed}/{total} files | {total_frames} frames "
            f"({avg_frames:.1f}/file) | outputs {total_outputs} "
            f"({avg_outputs:.2f}/file) | {file_rate:.2f} files/s | "
            f"{frame_rate:.1f} fps | ETA {_format_eta(eta)} | failures {failures}{suffix}"
        )
        if args.progress_json is not None:
            _write_progress_json(
                args.progress_json,
                {
                    "completed": completed,
                    "total": total,
                    "frames": total_frames,
                    "outputs": total_outputs,
                    "failures": failures,
                    "elapsed_sec": elapsed,
                    "files_per_sec": file_rate,
                    "frames_per_sec": frame_rate,
                    "avg_frames_per_file": avg_frames,
                    "avg_outputs_per_file": avg_outputs,
                    "eta_sec": eta,
                    "last_scene": last_scene,
                    "last_path": last_path,
                    "last_rel": last_rel,
                    "last_worker_pid": last_pid,
                },
            )

    if workers == 1:
        for idx, src in enumerate(mp4s, start=1):
            rel = src.relative_to(input_dir)
            result = _process_mp4_task(
                str(src),
                str(rel),
                scale_items,
                overwrite=bool(args.overwrite),
                worker_log=bool(args.worker_log),
                task_index=idx,
                task_total=total,
            )
            completed = idx
            total_frames += int(result.get("frames", 0))
            total_outputs += int(result.get("outputs", 0))
            if completed % progress_every == 0 or completed == total:
                emit_progress(
                    result.get("scene"),
                    result.get("path"),
                    result.get("rel"),
                    result.get("pid"),
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx, src in enumerate(mp4s, start=1):
                rel = src.relative_to(input_dir)
                futures.append(
                    executor.submit(
                        _process_mp4_task,
                        str(src),
                        str(rel),
                        scale_items,
                        overwrite=bool(args.overwrite),
                        worker_log=bool(args.worker_log),
                        task_index=idx,
                        task_total=total,
                    )
                )
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    failures += 1
                    result = None
                    log_line(f"[WARN] MP4 worker failed: {exc}")
                completed += 1
                if result is not None:
                    total_frames += int(result.get("frames", 0))
                    total_outputs += int(result.get("outputs", 0))
                if completed % progress_every == 0 or completed == total:
                    emit_progress(
                        result.get("scene") if result else None,
                        result.get("path") if result else None,
                        result.get("rel") if result else None,
                        result.get("pid") if result else None,
                    )

    if args.other_mode == "copy":
        log_line("[COPY] copying non-MP4 files...")
        for src in input_dir.rglob("*"):
            if src.is_dir():
                continue
            if src.suffix.lower() == ".mp4":
                continue
            rel = src.relative_to(input_dir)
            for out_dir in output_dirs:
                _copy_or_link(src, out_dir / rel, args.other_mode)
        log_line("[COPY] done.")

    report = {
        "input": str(input_dir),
        "pattern": args.pattern,
        "scales": args.scales,
        "suffix_mode": args.suffix_mode,
        "base": base_report,
        "base_luma": base_luma,
        "base_sample_per_scene": args.base_sample_per_scene,
        "base_sample_seed": args.base_sample_seed,
        "base_max_scenes": args.base_max_scenes,
        "base_files": len(base_files),
        "total_files": len(mp4s),
        "outputs": {
            str(scale): {"dir": str(path), "suffix": suffix_map[scale]}
            for scale, path in scale_map.items()
        },
    }
    log_line(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
