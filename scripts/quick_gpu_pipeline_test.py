#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil


NPC_SWEEP = (
    (5, 0.25),
    (10, 0.5),
    (20, 0.75),
    (30, 0.8),
)


def _format_bytes(value: float | int | None) -> str:
    if value is None:
        return "-"
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f}{units[idx]}"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "-"
    seconds_int = int(round(seconds))
    days, rem = divmod(seconds_int, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d{hours:02d}h{minutes:02d}m{secs:02d}s"
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def _discover_scene(tasks_dir: Path) -> str | None:
    if not tasks_dir.exists():
        return None
    candidates = [p.name for p in sorted(tasks_dir.iterdir()) if p.is_dir()]
    return candidates[0] if candidates else None


def _sum_process_metrics(root: psutil.Process) -> tuple[int, int, int]:
    rss_total = 0
    read_total = 0
    write_total = 0
    try:
        processes = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, 0, 0
    for proc in processes:
        try:
            rss_total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        try:
            io = proc.io_counters()
            read_total += int(io.read_bytes)
            write_total += int(io.write_bytes)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass
    return rss_total, read_total, write_total


def _monitor_process(proc: subprocess.Popen, interval: float) -> dict:
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return {
            "max_rss_bytes": 0,
            "avg_rss_bytes": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "max_read_bytes_per_sec": 0.0,
            "max_write_bytes_per_sec": 0.0,
        }
    max_rss = 0
    sum_rss = 0
    samples = 0
    max_read_rate = 0.0
    max_write_rate = 0.0
    last_io = None
    last_ts = time.monotonic()
    while proc.poll() is None:
        try:
            rss_total, read_total, write_total = _sum_process_metrics(ps_proc)
        except psutil.NoSuchProcess:
            break
        max_rss = max(max_rss, rss_total)
        sum_rss += rss_total
        samples += 1
        now = time.monotonic()
        if last_io is not None:
            dt = max(now - last_ts, 1e-6)
            read_rate = (read_total - last_io[0]) / dt
            write_rate = (write_total - last_io[1]) / dt
            max_read_rate = max(max_read_rate, read_rate)
            max_write_rate = max(max_write_rate, write_rate)
        last_io = (read_total, write_total)
        last_ts = now
        time.sleep(interval)
    rss_total, read_total, write_total = _sum_process_metrics(ps_proc)
    avg_rss = (sum_rss / samples) if samples > 0 else 0
    return {
        "max_rss_bytes": max_rss,
        "avg_rss_bytes": avg_rss,
        "read_bytes": read_total,
        "write_bytes": write_total,
        "max_read_bytes_per_sec": max_read_rate,
        "max_write_bytes_per_sec": max_write_rate,
    }


def _parse_metrics(metrics_path: Path) -> dict:
    if not metrics_path.is_file():
        return {}
    payload = json.loads(metrics_path.read_text())
    paths = payload.get("paths") or []
    frames_total = 0
    duration_total = 0.0
    vram_peak_max = 0.0
    encode_total = 0.0
    mux_total = 0.0
    slot_samples: dict[str, list[float]] = {}
    for entry in paths:
        frames = entry.get("frames") or 0
        duration = entry.get("duration_sec") or 0.0
        frames_total += int(frames)
        duration_total += float(duration)
        vram_peak = entry.get("vram_peak_bytes") or 0.0
        if vram_peak:
            vram_peak_max = max(vram_peak_max, float(vram_peak))
        stage_seconds = entry.get("stage_seconds") or {}
        encode_total += float(stage_seconds.get("h264_encode_sec") or 0.0)
        mux_total += float(stage_seconds.get("h264_mux_sec") or 0.0)
        vram_avg = entry.get("vram_avg_bytes") or 0.0
        slot = entry.get("job_slot")
        slot_key = str(slot) if slot is not None else "unknown"
        if vram_avg:
            slot_samples.setdefault(slot_key, []).append(float(vram_avg))
    time_per_frame = (duration_total / frames_total) if frames_total > 0 else None
    encode_per_frame = (encode_total / frames_total) if frames_total > 0 else None
    encode_fps = (frames_total / encode_total) if encode_total > 0 else None
    mux_per_frame = (mux_total / frames_total) if frames_total > 0 else None
    mux_per_path = (mux_total / len(paths)) if paths else None
    slot_avg = [
        sum(values) / len(values) for values in slot_samples.values() if values
    ]
    max_worker_vram = max(slot_avg) if slot_avg else 0.0
    return {
        "paths_total": len(paths),
        "frames_total": frames_total,
        "duration_total_sec": duration_total,
        "time_per_frame_sec": time_per_frame,
        "h264_encode_total_sec": encode_total,
        "h264_encode_sec_per_frame": encode_per_frame,
        "h264_encode_fps": encode_fps,
        "h264_mux_total_sec": mux_total,
        "h264_mux_sec_per_frame": mux_per_frame,
        "h264_mux_sec_per_path": mux_per_path,
        "vram_peak_max_bytes": vram_peak_max,
        "vram_avg_max_worker_bytes": max_worker_vram,
    }


def _compute_total_length(
    *,
    metrics_path: Path,
    scenes_dir: Path,
    tasks_dir: Path,
    stride: int,
    swap_xy: bool,
    mirror_translation: bool,
) -> float | None:
    if not metrics_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text())
    paths = payload.get("paths") or []
    if not paths:
        return 0.0
    try:
        from render_label_paths import (  # type: ignore
            PathSampler,
            load_occupancy_metadata,
            prepare_path_data,
            resolve_label_directory,
        )
    except Exception:
        return None
    meta_cache: dict[str, dict] = {}
    label_dir_cache: dict[str, Path | None] = {}
    total_length = 0.0
    for entry in paths:
        scene_id = str(entry.get("scene_id") or "")
        label_id = str(entry.get("label_id") or "")
        if not scene_id or not label_id:
            continue
        if scene_id not in meta_cache:
            scene_dir = scenes_dir / scene_id
            if not scene_dir.is_dir():
                meta_cache[scene_id] = {}
            else:
                try:
                    meta_cache[scene_id] = load_occupancy_metadata(scene_dir)
                except Exception:
                    meta_cache[scene_id] = {}
        if scene_id not in label_dir_cache:
            label_dir_cache[scene_id] = resolve_label_directory(tasks_dir / scene_id)
        label_dir = label_dir_cache[scene_id]
        if not label_dir or not label_dir.is_dir():
            continue
        json_path = label_dir / f"{label_id}.json"
        if not json_path.is_file():
            json_path = label_dir / label_id
        if not json_path.is_file():
            continue
        meta = meta_cache.get(scene_id)
        if not meta:
            continue
        try:
            prepared = prepare_path_data(
                json_path=json_path,
                meta=meta,
                stride=stride,
                mirror_translation=mirror_translation,
                swap_xy=swap_xy,
                resample_step=0.0,
            )
            sampler = PathSampler(prepared.path_xy)
            total_length += float(sampler.total_length)
        except Exception:
            continue
    return total_length


def _estimate_eta(
    *,
    time_per_frame: float | None,
    total_paths: int,
    frames_per_path: int,
    max_workers: int,
) -> float | None:
    if time_per_frame is None or time_per_frame <= 0 or max_workers <= 0:
        return None
    total_frames = total_paths * frames_per_path
    return (total_frames * time_per_frame) / max_workers


def build_render_command(args: argparse.Namespace, *, output_dir: Path, metrics_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.render_script),
        "--scenes-dir",
        str(args.scenes_dir),
        "--tasks-dir",
        str(args.tasks_dir),
        "--scene",
        args.scene_id,
        "--output-dir",
        str(output_dir),
        "--metrics-json",
        str(metrics_path),
        "--overwrite",
        "--video",
        "--video-backend",
        "gpu",
        "--gpu-only",
        "--verbose",
        "--no-show-BEV",
        "--no-save-depth-maps",
        "--no-save-camera-metadata",
        "--no-save-follow-metadata",
        "--no-rgb-frames",
    ]
    if args.max_labels:
        cmd.extend(["--max-labels", str(args.max_labels)])
    if args.minimal_frames is not None:
        cmd.extend(["--minimal-frames", str(args.minimal_frames)])
    if args.stride:
        cmd.extend(["--stride", str(args.stride)])
    if args.height_offset is not None:
        cmd.extend(["--height-offset", str(args.height_offset)])
    if args.view_mode:
        cmd.extend(["--view-mode", args.view_mode])
    if args.video_nvenc_preset:
        cmd.extend(["--video-nvenc-preset", args.video_nvenc_preset])
    if args.video_nvenc_bitrate:
        cmd.extend(["--video-nvenc-bitrate", args.video_nvenc_bitrate])
    cmd.extend(
        [
            "--npc-render",
            "--npc-count",
            str(args.npc_count),
            "--npc-max-count",
            str(args.npc_count),
            "--npc-density-coverage",
            str(args.npc_coverage),
            "--npc-priority",
            "coverage",
            "--npc-density-mode",
            args.npc_density_mode,
            "--npc-zone-ratio",
            args.npc_zone_ratio,
            "--npc-max-range",
            str(args.npc_max_range),
            "--npc-free-threshold",
            str(args.npc_free_threshold),
        ]
    )
    if args.npc_free_white:
        cmd.append("--npc-free-white")
    else:
        cmd.append("--no-npc-free-white")
    if args.npc_rotate_mask_180:
        cmd.append("--npc-rotate-mask-180")
    cmd.extend(["--npc-actor-root", str(args.npc_actor_root)])
    cmd.extend(["--npc-frame-pool-size", str(args.npc_frame_pool_size)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick GPU pipeline test with NPC density sweep and resource reporting."
    )
    root_dir = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--render-script",
        type=Path,
        default=root_dir / "render_label_paths.py",
        help="Path to render_label_paths.py (default: repo root).",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path(os.getenv("SCENES_DIR", "./data/scenes")),
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path(os.getenv("TASKS_DIR", "./data/interiorGS_0500_42")),
    )
    parser.add_argument(
        "--scene-id",
        default=os.getenv("SCENE_ID", "0001_839920"),
        help="Scene ID to test (default: 0001_839920).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.getenv("OUTPUT_ROOT", "./data/quick_gpu_pipeline_test")),
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("CLEAN_OUTPUT", "true").lower() in ("1", "true", "yes", "on"),
        help="Remove output-root before running (default: on).",
    )
    parser.add_argument("--max-labels", type=int, default=int(os.getenv("MAX_LABELS", "10")))
    parser.add_argument(
        "--minimal-frames",
        type=int,
        default=int(os.getenv("MINIMAL_FRAMES", "0")),
    )
    parser.add_argument("--stride", type=int, default=int(os.getenv("STRIDE", "1")))
    parser.add_argument(
        "--swap-xy",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("SWAP_XY", "false").lower() in ("1", "true", "yes", "on"),
    )
    parser.add_argument(
        "--mirror-translation",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MIRROR_TRANSLATION", "true").lower() in ("1", "true", "yes", "on"),
    )
    parser.add_argument("--height-offset", type=float, default=float(os.getenv("HEIGHT_OFFSET", "0.3")))
    parser.add_argument("--view-mode", default=os.getenv("VIEW_MODE", "forward"))
    parser.add_argument("--video-nvenc-preset", default=os.getenv("VIDEO_NVENC_PRESET"))
    parser.add_argument("--video-nvenc-bitrate", default=os.getenv("VIDEO_NVENC_BITRATE"))
    parser.add_argument(
        "--npc-actor-root",
        type=Path,
        default=Path(os.getenv("NPC_ACTOR_ROOT", "./data/SHHQ_gs/walking")),
    )
    parser.add_argument("--npc-frame-pool-size", type=int, default=int(os.getenv("NPC_FRAME_POOL_SIZE", "30")))
    parser.add_argument("--npc-density-mode", default=os.getenv("NPC_DENSITY_MODE", "angular"))
    parser.add_argument("--npc-zone-ratio", default=os.getenv("NPC_ZONE_RATIO", "1:2:1"))
    parser.add_argument("--npc-max-range", type=float, default=float(os.getenv("NPC_MAX_RANGE", "15")))
    parser.add_argument("--npc-free-threshold", type=int, default=int(os.getenv("NPC_FREE_THRESHOLD", "250")))
    parser.add_argument(
        "--npc-free-white",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("NPC_FREE_WHITE", "true").lower() in ("1", "true", "yes", "on"),
    )
    parser.add_argument(
        "--npc-rotate-mask-180",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("NPC_ROTATE_MASK_180", "true").lower() in ("1", "true", "yes", "on"),
    )
    parser.add_argument("--sample-interval", type=float, default=float(os.getenv("SAMPLE_INTERVAL", "1.0")))
    parser.add_argument("--total-paths", type=int, default=int(os.getenv("TOTAL_PATHS", "178628")))
    parser.add_argument("--frames-per-path", type=int, default=int(os.getenv("FRAMES_PER_PATH", "150")))
    parser.add_argument("--gpu-vram-gb", type=float, default=float(os.getenv("GPU_VRAM_GB", "45")))
    parser.add_argument(
        "--report-length",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("REPORT_LENGTH", "true").lower() in ("1", "true", "yes", "on"),
        help="Compute total path length in meters from label JSONs (default: on).",
    )
    args = parser.parse_args()

    if not args.scene_id:
        args.scene_id = _discover_scene(args.tasks_dir)
    else:
        candidate = args.tasks_dir / args.scene_id
        if not candidate.is_dir():
            fallback = _discover_scene(args.tasks_dir)
            if fallback:
                print(
                    f"[WARN] Scene {args.scene_id} not found under {args.tasks_dir}; "
                    f"falling back to {fallback}.",
                    flush=True,
                )
                args.scene_id = fallback
    if not args.scene_id:
        print("[ERROR] Could not determine a scene to run.", file=sys.stderr)
        return 1

    output_root = args.output_root.resolve()
    if args.clean_output and output_root.exists():
        print(f"[CLEAN] Removing previous outputs under {output_root}", flush=True)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = (root_dir / "analysis").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scene_id": args.scene_id,
        "tasks_dir": str(args.tasks_dir),
        "scenes_dir": str(args.scenes_dir),
        "total_paths": args.total_paths,
        "frames_per_path": args.frames_per_path,
        "gpu_vram_gb": args.gpu_vram_gb,
        "runs": [],
    }

    for npc_count, npc_cov in NPC_SWEEP:
        args.npc_count = npc_count
        args.npc_coverage = npc_cov
        run_name = f"npc{npc_count}_cov{npc_cov:.2f}".replace(".", "p")
        run_dir = output_root / run_name
        metrics_path = analysis_dir / f"quick_gpu_metrics_{run_name}.json"
        log_path = analysis_dir / f"quick_gpu_run_{run_name}.log"

        cmd = build_render_command(args, output_dir=run_dir, metrics_path=metrics_path)
        print(f"[RUN] {run_name} -> {' '.join(cmd)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            start_wall = time.monotonic()
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            monitor = _monitor_process(proc, args.sample_interval)
            returncode = proc.wait()
            wall_time = time.monotonic() - start_wall

        metrics = _parse_metrics(metrics_path)
        total_length_m = (
            _compute_total_length(
                metrics_path=metrics_path,
                scenes_dir=args.scenes_dir,
                tasks_dir=args.tasks_dir,
                stride=args.stride,
                swap_xy=args.swap_xy,
                mirror_translation=args.mirror_translation,
            )
            if args.report_length
            else None
        )
        vram_worker_bytes = metrics.get("vram_avg_max_worker_bytes") or 0.0
        max_workers = (
            max(1, int((args.gpu_vram_gb * (1024**3)) / vram_worker_bytes))
            if vram_worker_bytes > 0
            else 1
        )
        eta_seconds = _estimate_eta(
            time_per_frame=metrics.get("time_per_frame_sec"),
            total_paths=args.total_paths,
            frames_per_path=args.frames_per_path,
            max_workers=max_workers,
        )
        run_report = {
            "name": run_name,
            "npc_count": npc_count,
            "npc_coverage": npc_cov,
            "output_dir": str(run_dir),
            "metrics_json": str(metrics_path),
            "log_path": str(log_path),
            "returncode": returncode,
            "metrics": metrics,
            "total_length_m": total_length_m,
            "wall_time_sec": wall_time,
            "monitor": monitor,
            "max_workers_estimate": max_workers,
            "eta_seconds": eta_seconds,
        }
        report["runs"].append(run_report)

        length_str = _format_bytes(None)
        if total_length_m is not None:
            length_str = f"{total_length_m:.2f}m"
        encode_fps = metrics.get("h264_encode_fps")
        mux_per_path = metrics.get("h264_mux_sec_per_path")
        encode_str = f"{encode_fps:.2f}fps" if encode_fps else "-"
        mux_str = f"{mux_per_path:.2f}s/path" if mux_per_path else "-"
        print(
            "[REPORT] "
            f"{run_name} | paths={metrics.get('paths_total')} "
            f"frames={metrics.get('frames_total')} "
            f"len={length_str} | "
            f"wall={wall_time:.2f}s t/frame={metrics.get('time_per_frame_sec')}s | "
            f"h264={encode_str} mux={mux_str} | "
            f"vram(avg/max worker)={_format_bytes(vram_worker_bytes)} | "
            f"cpu_rss(max)={_format_bytes(monitor.get('max_rss_bytes'))} | "
            f"io(read/write)={_format_bytes(monitor.get('read_bytes'))}/"
            f"{_format_bytes(monitor.get('write_bytes'))} | "
            f"eta={_format_eta(eta_seconds)} | workers~{max_workers}",
            flush=True,
        )

    report_path = analysis_dir / "quick_gpu_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[DONE] Wrote report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
