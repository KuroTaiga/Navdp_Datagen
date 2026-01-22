#!/usr/bin/env python3
"""Quick pipeline smoke test with basic system/resource monitoring."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import shutil


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick test run for render_label_paths.py")
    parser.add_argument("--scenes-dir", type=Path, default=Path("./data/scenes"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("./data/interiorGS_0500_42"))
    parser.add_argument("--output-dir", type=Path, default=Path("data1/cl_test"))
    parser.add_argument("--scene", type=str, default=None, help="Scene id to render.")
    parser.add_argument("--label-count", type=int, default=30, help="Number of label paths to render.")
    parser.add_argument("--label-id", action="append", dest="label_ids", default=None)
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="Randomly sample this many (scene, path) pairs instead of the first scene.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --random.")
    parser.add_argument(
        "--path-handedness",
        choices=("left", "right", "auto"),
        default="auto",
        help="Handedness for raster_world path data (default: auto).",
    )
    parser.add_argument(
        "--negate-raster-world-xy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Negate raster_world x/y before swaps/handedness (default: auto for Waymo).",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds.")
    parser.add_argument("--ply-transform-backend", type=str, default="gpu")
    parser.add_argument("--video-backend", type=str, default="cpu")
    parser.add_argument("--video-nvenc-preset", type=str, default=None)
    parser.add_argument("--video-nvenc-bitrate", type=str, default=None)
    parser.add_argument(
        "--show-bev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save BEV debug image with path/scene overlay (default: on).",
    )
    parser.add_argument("--enable-depth", action="store_true", help="Keep depth outputs.")
    parser.add_argument(
        "--npc",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument(
        "--cl-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable camera-light shading (default: off).",
    )
    parser.add_argument(
        "--cl-light-mode",
        choices=("headlight", "bulb"),
        default="headlight",
        help="Light mode for camera light (default: headlight).",
    )
    parser.add_argument(
        "--cl-shading-model",
        choices=("classic", "lambert"),
        default="classic",
        help="Shading model for camera light (default: classic).",
    )
    parser.add_argument(
        "--cl-strength",
        type=float,
        default=1.0,
        help="Camera light strength multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--cl-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(1.0, 1.0, 1.0),
        help="Camera light color as RGB in 0..1 (default: 1 1 1).",
    )
    parser.add_argument(
        "--cl-ambient",
        type=float,
        default=0.2,
        help="Ambient term applied before camera light (default: 0.2).",
    )
    parser.add_argument(
        "--cl-base-scale",
        type=float,
        default=1.0,
        help="Scale the base image before lighting (default: 1.0).",
    )
    parser.add_argument(
        "--cl-diffuse",
        type=float,
        default=1.0,
        help="Diffuse term multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--cl-specular",
        type=float,
        default=0.2,
        help="Specular term multiplier (default: 0.2).",
    )
    parser.add_argument(
        "--cl-shininess",
        type=float,
        default=16.0,
        help="Specular shininess exponent (default: 16).",
    )
    parser.add_argument(
        "--cl-range",
        type=float,
        default=0.0,
        help="Light falloff range in meters (0 disables attenuation).",
    )
    parser.add_argument(
        "--cl-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Camera-light offset in camera coordinates (meters).",
    )
    parser.add_argument(
        "--cl-normal-smooth",
        type=int,
        default=0,
        help="Box blur radius (pixels) for depth before normal recovery (default: 0).",
    )
    parser.add_argument(
        "--cl-normal-filter",
        choices=("none", "box", "bilateral"),
        default="box",
        help="Depth filter before normal recovery (default: box).",
    )
    parser.add_argument(
        "--cl-normal-kernel",
        type=int,
        default=2,
        help="Bilateral kernel radius in pixels (default: 2).",
    )
    parser.add_argument(
        "--cl-normal-sigma-range",
        type=float,
        default=0.1,
        help="Bilateral range sigma in depth units (default: 0.1).",
    )
    parser.add_argument(
        "--cl-normal-sigma-domain",
        type=float,
        default=1.0,
        help="Bilateral domain sigma in pixels (default: 1.0).",
    )
    parser.add_argument(
        "--cl-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable shadow mapping from the camera light (default: off).",
    )
    parser.add_argument(
        "--cl-shadow-bias",
        type=float,
        default=0.02,
        help="Depth bias for shadow mapping (default: 0.02).",
    )
    parser.add_argument(
        "--cl-shadow-strength",
        type=float,
        default=0.2,
        help="Shadow strength multiplier (0=black, 1=no shadow; default: 0.2).",
    )
    parser.add_argument(
        "--cl-shadow-pcf",
        type=int,
        default=0,
        help="Shadow PCF radius in pixels for soft shadows (default: 0).",
    )
    parser.add_argument(
        "--cl-shadow-compare",
        choices=("auto", "z", "radial"),
        default="auto",
        help="Shadow depth compare mode (default: auto).",
    )
    return parser.parse_args()


def _resolve_label_directory(scene_task_dir: Path) -> Path | None:
    label_paths_dir = scene_task_dir / "label_paths"
    if label_paths_dir.is_dir() and any(label_paths_dir.glob("*.json")):
        return label_paths_dir
    if scene_task_dir.is_dir() and any(scene_task_dir.glob("*.json")):
        return scene_task_dir
    return None


def _find_scene_and_labels(tasks_dir: Path, scene: str | None, label_ids: list[str] | None, limit: int) -> tuple[str, list[str]]:
    if label_ids:
        if scene is None:
            raise SystemExit("--label-id requires --scene to be set.")
        return scene, label_ids

    if scene is None:
        candidates = sorted(
            [p for p in tasks_dir.iterdir() if p.is_dir() and _resolve_label_directory(p) is not None]
        )
        if not candidates:
            raise SystemExit(f"No scene folders with labels under {tasks_dir}")
        scene_dir = candidates[0]
        scene = scene_dir.name
    else:
        scene_dir = tasks_dir / scene
        if not scene_dir.is_dir():
            raise SystemExit(f"Scene directory not found: {scene_dir}")

    label_dir = _resolve_label_directory(scene_dir)
    if label_dir is None:
        raise SystemExit(f"No label JSONs found under {scene_dir}")

    json_paths = sorted(
        [
            p
            for p in label_dir.glob("*.json")
            if not p.name.endswith("_detailed.json") and p.name != "summary.json"
        ]
    )
    if not json_paths:
        raise SystemExit(f"No label JSONs found under {label_dir}")
    label_ids = [p.stem for p in json_paths[: max(1, limit)]]
    return scene, label_ids


def _collect_scene_label_pairs(tasks_dir: Path, scene: str | None) -> list[tuple[str, str]]:
    if scene is not None:
        scene_dir = tasks_dir / scene
        if not scene_dir.is_dir():
            raise SystemExit(f"Scene directory not found: {scene_dir}")
        if _resolve_label_directory(scene_dir) is None:
            raise SystemExit(f"No label JSONs found under {scene_dir}")
        candidates = [scene_dir]
    else:
        if not tasks_dir.is_dir():
            raise SystemExit(f"Task output directory not found: {tasks_dir}")
        candidates = sorted(
            [p for p in tasks_dir.iterdir() if p.is_dir() and _resolve_label_directory(p) is not None]
        )
        if not candidates:
            raise SystemExit(f"No scene folders with labels under {tasks_dir}")

    pairs: list[tuple[str, str]] = []
    for scene_dir in candidates:
        label_dir = _resolve_label_directory(scene_dir)
        if label_dir is None:
            continue
        json_paths = sorted(
            [
                p
                for p in label_dir.glob("*.json")
                if not p.name.endswith("_detailed.json") and p.name != "summary.json"
            ]
        )
        for json_path in json_paths:
            pairs.append((scene_dir.name, json_path.stem))
    return pairs


def _sample_random_scene_labels(
    tasks_dir: Path,
    scene: str | None,
    sample_count: int,
    seed: int,
) -> dict[str, list[str]]:
    if sample_count <= 0:
        raise SystemExit("--random must be a positive integer.")
    pairs = _collect_scene_label_pairs(tasks_dir, scene)
    if not pairs:
        raise SystemExit(f"No label JSONs found under {tasks_dir}")
    rng = random.Random(seed)
    sample_size = min(sample_count, len(pairs))
    sampled_pairs = rng.sample(pairs, sample_size)
    runs: dict[str, list[str]] = {}
    for scene_id, label_id in sampled_pairs:
        runs.setdefault(scene_id, []).append(label_id)
    return runs


def _resolve_path_handedness(handedness: str, scenes_dir: Path, tasks_dir: Path) -> str:
    if handedness != "auto":
        return handedness
    for candidate in (scenes_dir, tasks_dir):
        if "waymo" in candidate.name.lower():
            return "right"
    return "left"


def _resolve_negate_raster_world_xy(
    value: bool | None,
    scenes_dir: Path,
    tasks_dir: Path,
) -> bool:
    if value is not None:
        return value
    for candidate in (scenes_dir, tasks_dir):
        if "waymo" in candidate.name.lower():
            return True
    return False


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

    read_deltas = [s["read_bytes_delta"] for s in samples if s.get("read_bytes_delta") is not None]
    write_deltas = [s["write_bytes_delta"] for s in samples if s.get("write_bytes_delta") is not None]
    total_read = sum(read_deltas) if read_deltas else None
    total_write = sum(write_deltas) if write_deltas else None
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


def _run_render_command(cmd: list[str], interval: float) -> tuple[int, float, list[dict[str, Any]]]:
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

        time.sleep(max(0.1, interval))

    return_code = process.wait()
    elapsed = time.time() - start_time
    return return_code, elapsed, samples


def main() -> None:
    args = _parse_args()
    if args.random is not None and args.label_ids:
        raise SystemExit("--random is incompatible with --label-id.")
    if args.random is None:
        scene_id, label_ids = _find_scene_and_labels(
            args.tasks_dir, args.scene, args.label_ids, args.label_count
        )
        runs: dict[str, list[str]] = {scene_id: label_ids}
    else:
        runs = _sample_random_scene_labels(
            args.tasks_dir,
            args.scene,
            args.random,
            args.seed,
        )

    repo_root = Path(__file__).resolve().parents[1]
    render_script = repo_root / "render_label_paths.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_selected = sum(len(ids) for ids in runs.values())
    if args.random is not None:
        print(
            f"[TEST] Random selection: {total_selected} paths across {len(runs)} scenes (seed={args.seed}).",
            flush=True,
        )
    path_handedness = _resolve_path_handedness(
        args.path_handedness,
        args.scenes_dir,
        args.tasks_dir,
    )
    negate_raster_world_xy = _resolve_negate_raster_world_xy(
        args.negate_raster_world_xy,
        args.scenes_dir,
        args.tasks_dir,
    )
    print(f"[TEST] Path handedness: {path_handedness}", flush=True)
    print(f"[TEST] Negate raster_world xy: {negate_raster_world_xy}", flush=True)

    samples: list[dict[str, Any]] = []
    metrics_payloads: list[dict[str, Any]] = []
    elapsed = 0.0
    return_code = 0

    for scene_id, label_ids in runs.items():
        metrics_name = "quick_metrics.json" if len(runs) == 1 else f"quick_metrics_{scene_id}.json"
        metrics_path = args.output_dir / metrics_name
        cmd = [
            sys.executable,
            str(render_script),
            "--scenes-dir",
            str(args.scenes_dir),
            "--tasks-dir",
            str(args.tasks_dir),
            "--path-handedness",
            path_handedness,
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
        if negate_raster_world_xy:
            cmd.append("--negate-raster-world-xy")
        if args.show_bev:
            cmd.append("--show-BEV")
        else:
            cmd.append("--no-show-BEV")
        if args.video_nvenc_preset:
            cmd.extend(["--video-nvenc-preset", args.video_nvenc_preset])
        if args.video_nvenc_bitrate:
            cmd.extend(["--video-nvenc-bitrate", args.video_nvenc_bitrate])
        if not args.enable_depth:
            cmd.append("--no-save-depth-maps")
        for label_id in label_ids:
            cmd.extend(["--label-id", label_id])
        if args.cl_enable:
            cmd.append("--cl-enable")
            cmd.extend(["--cl-light-mode", args.cl_light_mode])
            cmd.extend(["--cl-shading-model", args.cl_shading_model])
            cmd.extend(["--cl-strength", str(args.cl_strength)])
            cmd.extend(
                [
                    "--cl-color",
                    str(args.cl_color[0]),
                    str(args.cl_color[1]),
                    str(args.cl_color[2]),
                ]
            )
            cmd.extend(["--cl-ambient", str(args.cl_ambient)])
            cmd.extend(["--cl-base-scale", str(args.cl_base_scale)])
            cmd.extend(["--cl-diffuse", str(args.cl_diffuse)])
            cmd.extend(["--cl-specular", str(args.cl_specular)])
            cmd.extend(["--cl-shininess", str(args.cl_shininess)])
            cmd.extend(["--cl-range", str(args.cl_range)])
            cmd.extend(
                [
                    "--cl-offset",
                    str(args.cl_offset[0]),
                    str(args.cl_offset[1]),
                    str(args.cl_offset[2]),
                ]
            )
            cmd.extend(["--cl-normal-smooth", str(args.cl_normal_smooth)])
            cmd.extend(["--cl-normal-filter", args.cl_normal_filter])
            cmd.extend(["--cl-normal-kernel", str(args.cl_normal_kernel)])
            cmd.extend(["--cl-normal-sigma-range", str(args.cl_normal_sigma_range)])
            cmd.extend(["--cl-normal-sigma-domain", str(args.cl_normal_sigma_domain)])
            if args.cl_shadow:
                cmd.append("--cl-shadow")
            else:
                cmd.append("--no-cl-shadow")
            cmd.extend(["--cl-shadow-bias", str(args.cl_shadow_bias)])
            cmd.extend(["--cl-shadow-strength", str(args.cl_shadow_strength)])
            cmd.extend(["--cl-shadow-pcf", str(args.cl_shadow_pcf)])
            cmd.extend(["--cl-shadow-compare", args.cl_shadow_compare])
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

        run_code, run_elapsed, run_samples = _run_render_command(cmd, args.interval)
        elapsed += run_elapsed
        samples.extend(run_samples)

        if metrics_path.exists():
            try:
                metrics_payloads.append(json.loads(metrics_path.read_text()))
            except Exception:
                metrics_payloads.append({})

        if run_code != 0:
            return_code = run_code
            break

    summary = _summarize_samples(samples)
    fps_values: list[float] = []
    frames_total = 0
    for metrics_payload in metrics_payloads:
        for path_payload in metrics_payload.get("paths", []):
            if path_payload.get("frames_per_sec") is not None:
                fps_values.append(path_payload["frames_per_sec"])
            frames_total += int(path_payload.get("frames") or 0)
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else None

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
