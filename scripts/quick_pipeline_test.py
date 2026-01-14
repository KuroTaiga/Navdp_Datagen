#!/usr/bin/env python3
"""Quick pipeline smoke test with basic system/resource monitoring."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import shutil


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick test run for render_label_paths.py")
    parser.add_argument("--scenes-dir", type=Path, default=Path("data/scenes"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("data/interiorGS_0500_42"))
    parser.add_argument("--output-dir", type=Path, default=Path("data1/quick_pipeline_test"))
    parser.add_argument("--scene", type=str, default=None, help="Scene id to render.")
    parser.add_argument("--label-count", type=int, default=30, help="Number of label paths to render.")
    parser.add_argument("--label-id", action="append", dest="label_ids", default=None)
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds.")
    parser.add_argument("--ply-transform-backend", type=str, default="gpu")
    parser.add_argument("--video-backend", type=str, default="nvenc")
    parser.add_argument("--video-nvenc-preset", type=str, default=None)
    parser.add_argument("--video-nvenc-bitrate", type=str, default=None)
    parser.add_argument("--enable-depth", action="store_true", help="Keep depth outputs.")
    parser.add_argument(
        "--npc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NPC placement/rendering (default: enabled).",
    )
    parser.add_argument("--npc-actor-root", type=Path, default=Path("data/SHHQ_gs/walking"))
    parser.add_argument("--npc-frame-pool-size", type=int, default=30)
    parser.add_argument("--npc-density-coverage", type=float, default=0.5)
    parser.add_argument("--npc-count", type=int, default=10)
    parser.add_argument("--npc-max-count", type=int, default=10)
    parser.add_argument("--npc-priority", type=str, default="coverage")
    parser.add_argument("--npc-density-mode", type=str, default="angular")
    parser.add_argument("--npc-zone-ratio", type=str, default="1:2:1")
    parser.add_argument("--npc-max-range", type=float, default=15.0)
    parser.add_argument("--npc-free-threshold", type=int, default=250)
    parser.add_argument("--npc-placement-backend", type=str, default="gpu")
    parser.add_argument(
        "--npc-free-white",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--npc-rotate-mask-180",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--npc-auto-clearance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="--gpu-only",
        help="Extra args forwarded to render_label_paths.py (default: --gpu-only).",
    )
    return parser.parse_args()


def _find_scene_and_labels(tasks_dir: Path, scene: str | None, label_ids: list[str] | None, limit: int) -> tuple[str, list[str]]:
    if label_ids:
        if scene is None:
            raise SystemExit("--label-id requires --scene to be set.")
        return scene, label_ids

    if scene is None:
        candidates = sorted([p for p in tasks_dir.iterdir() if p.is_dir()])
        if not candidates:
            raise SystemExit(f"No scene folders under {tasks_dir}")
        scene_dir = candidates[0]
        scene = scene_dir.name
    else:
        scene_dir = tasks_dir / scene
        if not scene_dir.is_dir():
            raise SystemExit(f"Scene directory not found: {scene_dir}")

    json_paths = sorted(
        [p for p in scene_dir.glob("*.json") if not p.name.endswith("_detailed.json")]
    )
    if not json_paths:
        raise SystemExit(f"No label JSONs found under {scene_dir}")
    label_ids = [p.stem for p in json_paths[: max(1, limit)]]
    return scene, label_ids


def _read_meminfo() -> tuple[int | None, int | None]:
    mem_total = None
    mem_available = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1])
    except Exception:
        return None, None
    return mem_total, mem_available


def _read_proc_io(pid: int) -> tuple[int | None, int | None]:
    try:
        with open(f"/proc/{pid}/io", "r", encoding="utf-8") as handle:
            values = {}
            for line in handle:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                try:
                    values[key] = int(parts[1].strip())
                except ValueError:
                    continue
        return values.get("read_bytes"), values.get("write_bytes")
    except Exception:
        return None, None


def _read_ps_stats(pid: int) -> tuple[float | None, float | None, int | None]:
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "%cpu=,%mem=,rss="],
            text=True,
        ).strip()
        if not output:
            return None, None, None
        parts = output.split()
        if len(parts) < 3:
            return None, None, None
        return float(parts[0]), float(parts[1]), int(parts[2])
    except Exception:
        return None, None, None


def _read_gpu_stats() -> tuple[float | None, float | None, float | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None, None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if not output:
            return None, None, None
        util, mem_used, mem_total = output.split(",")
        return float(util), float(mem_used), float(mem_total)
    except Exception:
        return None, None, None


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}

    def avg(key: str) -> float | None:
        vals = [s[key] for s in samples if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    total_read = None
    total_write = None
    first_read = samples[0].get("read_bytes")
    last_read = samples[-1].get("read_bytes")
    if first_read is not None and last_read is not None:
        total_read = last_read - first_read
    first_write = samples[0].get("write_bytes")
    last_write = samples[-1].get("write_bytes")
    if first_write is not None and last_write is not None:
        total_write = last_write - first_write
    return {
        "avg_cpu_percent": avg("cpu_percent"),
        "avg_mem_percent": avg("mem_percent"),
        "avg_rss_kb": avg("rss_kb"),
        "avg_gpu_util": avg("gpu_util"),
        "avg_gpu_mem_used_mb": avg("gpu_mem_used_mb"),
        "mem_total_kb": samples[-1].get("mem_total_kb"),
        "mem_available_kb": samples[-1].get("mem_available_kb"),
        "total_read_bytes": total_read,
        "total_write_bytes": total_write,
    }


def main() -> None:
    args = _parse_args()
    scene_id, label_ids = _find_scene_and_labels(
        args.tasks_dir, args.scene, args.label_ids, args.label_count
    )

    repo_root = Path(__file__).resolve().parents[1]
    render_script = repo_root / "render_label_paths.py"
    metrics_path = args.output_dir / "quick_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(render_script),
        "--scenes-dir",
        str(args.scenes_dir),
        "--tasks-dir",
        str(args.tasks_dir),
        "--scene",
        scene_id,
        "--output-dir",
        str(args.output_dir),
        "--overwrite",
        "--video",
        "--save-camera-metadata",
        "--ply-transform-backend",
        args.ply_transform_backend,
        "--video-backend",
        args.video_backend,
        "--metrics-json",
        str(metrics_path),
    ]
    if args.video_nvenc_preset:
        cmd.extend(["--video-nvenc-preset", args.video_nvenc_preset])
    if args.video_nvenc_bitrate:
        cmd.extend(["--video-nvenc-bitrate", args.video_nvenc_bitrate])
    if not args.enable_depth:
        cmd.append("--no-save-depth-maps")
    for label_id in label_ids:
        cmd.extend(["--label-id", label_id])
    if args.npc:
        cmd.extend(
            [
                "--npc-render",
                "--npc-actor-root",
                str(args.npc_actor_root),
                "--npc-frame-pool-size",
                str(args.npc_frame_pool_size),
                "--npc-density-coverage",
                str(args.npc_density_coverage),
                "--npc-count",
                str(args.npc_count),
                "--npc-max-count",
                str(args.npc_max_count),
                "--npc-priority",
                str(args.npc_priority),
                "--npc-density-mode",
                str(args.npc_density_mode),
                "--npc-zone-ratio",
                str(args.npc_zone_ratio),
                "--npc-max-range",
                str(args.npc_max_range),
                "--npc-free-threshold",
                str(args.npc_free_threshold),
                "--npc-placement-backend",
                str(args.npc_placement_backend),
            ]
        )
        if args.npc_free_white:
            cmd.append("--npc-free-white")
        else:
            cmd.append("--no-npc-free-white")
        if args.npc_rotate_mask_180:
            cmd.append("--npc-rotate-mask-180")
        if args.npc_auto_clearance:
            cmd.append("--npc-auto-clearance")
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    print("[TEST] Running:", " ".join(shlex.quote(part) for part in cmd), flush=True)

    process = subprocess.Popen(cmd)
    start_time = time.time()

    samples: list[dict[str, Any]] = []
    read_bytes_prev = None
    write_bytes_prev = None

    while process.poll() is None:
        cpu_percent, mem_percent, rss_kb = _read_ps_stats(process.pid)
        mem_total, mem_available = _read_meminfo()
        gpu_util, gpu_mem_used, gpu_mem_total = _read_gpu_stats()
        read_bytes, write_bytes = _read_proc_io(process.pid)

        sample = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "mem_percent": mem_percent,
            "rss_kb": rss_kb,
            "mem_total_kb": mem_total,
            "mem_available_kb": mem_available,
            "gpu_util": gpu_util,
            "gpu_mem_used_mb": gpu_mem_used,
            "gpu_mem_total_mb": gpu_mem_total,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "read_bytes_delta": None,
            "write_bytes_delta": None,
        }

        if read_bytes_prev is not None and read_bytes is not None:
            sample["read_bytes_delta"] = read_bytes - read_bytes_prev
        if write_bytes_prev is not None and write_bytes is not None:
            sample["write_bytes_delta"] = write_bytes - write_bytes_prev

        samples.append(sample)
        read_bytes_prev = read_bytes
        write_bytes_prev = write_bytes

        time.sleep(max(0.1, args.interval))

    return_code = process.wait()
    elapsed = time.time() - start_time

    summary = _summarize_samples(samples)
    metrics_payload = {}
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text())
        except Exception:
            metrics_payload = {}

    fps_values = [
        p.get("frames_per_sec")
        for p in metrics_payload.get("paths", [])
        if p.get("frames_per_sec") is not None
    ]
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else None
    frames_total = sum(int(p.get("frames") or 0) for p in metrics_payload.get("paths", []))

    print("\n[TEST] Summary", flush=True)
    print(f"  return_code: {return_code}", flush=True)
    print(f"  elapsed_sec: {elapsed:.2f}", flush=True)
    if avg_fps is not None:
        print(f"  avg_fps: {avg_fps:.2f}", flush=True)
    if frames_total:
        print(f"  frames_total: {frames_total}", flush=True)
    if summary:
        print(f"  avg_cpu_percent: {summary.get('avg_cpu_percent')}", flush=True)
        print(f"  avg_mem_percent: {summary.get('avg_mem_percent')}", flush=True)
        print(f"  avg_rss_kb: {summary.get('avg_rss_kb')}", flush=True)
        print(f"  avg_gpu_util: {summary.get('avg_gpu_util')}", flush=True)
        print(f"  avg_gpu_mem_used_mb: {summary.get('avg_gpu_mem_used_mb')}", flush=True)
        print(f"  total_read_bytes: {summary.get('total_read_bytes')}", flush=True)
        print(f"  total_write_bytes: {summary.get('total_write_bytes')}", flush=True)

    samples_path = args.output_dir / "quick_resource_samples.json"
    samples_path.write_text(json.dumps(samples, indent=2))
    print(f"[TEST] Samples written to {samples_path}", flush=True)

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
