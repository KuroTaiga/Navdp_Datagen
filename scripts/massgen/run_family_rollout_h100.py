#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
H100_DEFAULT_JOBS_PER_GPU = 4
H100_DEFAULT_TARGET_GPU_UTIL = 80.0


@dataclass(frozen=True)
class WorkerSlot:
    slot_id: int
    gpu_id: str
    cpu_cores: tuple[int, ...]
    cpu_threads: int

    @property
    def core_list(self) -> str:
        return ",".join(str(item) for item in self.cpu_cores)


class NvidiaSmiMonitor:
    def __init__(
        self,
        *,
        gpu_devices: list[str],
        interval_sec: float,
        output_jsonl: Path,
    ) -> None:
        self.gpu_devices = list(gpu_devices)
        self.interval_sec = max(0.5, float(interval_sec))
        self.output_jsonl = output_jsonl
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_sec * 2.0))
        with self._lock:
            return list(self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            for sample in _sample_nvidia_smi(self.gpu_devices):
                with self._lock:
                    self.samples.append(sample)
                with self.output_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
            self._stop.wait(self.interval_sec)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MassGen family package renders on H100-class hosts with CPU H.264 "
            "encoding, persistent scene/resource scheduling, CPU affinity, and "
            "GPU utilization logging."
        )
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--pipeline-mode",
        choices=["persistent", "legacy"],
        default="persistent",
        help=(
            "persistent uses the scene/resource-aware H100 schedule runner. "
            "legacy keeps the older family-level multi-process fanout."
        ),
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--render-script", type=Path, default=REPO_ROOT / "render_label_paths_telesim.py")
    parser.add_argument("--video-backend", default="cpu", choices=["cpu", "nvenc", "gpu"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimal-frames", type=int, default=None)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Legacy mode only: maximum manifest jobs to render per family.",
    )
    parser.add_argument(
        "--max-renders",
        type=int,
        default=0,
        help="Persistent mode: global selected render-manifest cap. 0 means no cap.",
    )
    parser.add_argument(
        "--renders-per-family-source-scene",
        type=int,
        default=0,
        help="Persistent mode: cap selected manifests per (family, source, scene). 0 means no cap.",
    )
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument(
        "--gpu-devices",
        default=None,
        help="Comma-separated physical GPU ids. Defaults to CUDA_VISIBLE_DEVICES, then nvidia-smi indices.",
    )
    parser.add_argument("--jobs-per-gpu", type=int, default=H100_DEFAULT_JOBS_PER_GPU)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cpu-cores", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--cpu-cores-per-worker",
        type=int,
        default=None,
        help="CPU cores assigned to each render process. Defaults to floor(cpu_cores / workers).",
    )
    parser.add_argument(
        "--cpu-affinity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use taskset CPU affinity when available on Linux.",
    )
    parser.add_argument(
        "--actor-gpu-resident",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep human actor sequences resident on GPU. Default is on for H100 hosts.",
    )
    parser.add_argument(
        "--actor-runtime-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable renderer-side actor runtime cache for persistent schedule jobs.",
    )
    parser.add_argument(
        "--target-gpu-util",
        type=float,
        default=H100_DEFAULT_TARGET_GPU_UTIL,
        help="Soft target for average GPU utilization percent, used for reporting.",
    )
    parser.add_argument("--util-sample-interval-sec", type=float, default=0.1)
    parser.add_argument("--util-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ffmpeg-bin", type=Path, default=None)
    parser.add_argument("--materialized-root", type=Path, default=None)
    parser.add_argument("--max-items-per-chunk", type=int, default=0)
    parser.add_argument(
        "--scene-affinity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all same-scene persistent chunks on one logical lane.",
    )
    parser.add_argument("--group-max-labels-per-command", type=int, default=0)
    parser.add_argument("--command-attempts", type=int, default=3)
    parser.add_argument("--run-render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-window-min", type=float, default=4.0)
    parser.add_argument("--scene-vram-gib", type=float, default=8.0)
    parser.add_argument("--human-avatar-vram-mib", type=float, default=512.0)
    parser.add_argument("--robot-asset-vram-mib", type=float, default=512.0)
    parser.add_argument("--actor-plan-ram-mib", type=float, default=64.0)
    parser.add_argument("--simulate-cache-gib", type=float, default=None)
    parser.add_argument(
        "--baseline-summary",
        action="append",
        default=None,
        help="Optional benchmark_summary.json to include in persistent report comparisons.",
    )
    return parser.parse_args(argv)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"command: {' '.join(command)}\n")
        handle.write(f"started_at: {started}\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**os.environ, **dict(env or {})},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
        elapsed = time.perf_counter() - t0
        handle.write(f"\nended_at: {datetime.now(timezone.utc).isoformat()}\n")
        handle.write(f"wall_time_sec: {elapsed:.6f}\n")
        handle.write(f"returncode: {returncode}\n")
    return int(returncode), float(elapsed)


def _capture_to_file(command: list[str], path: Path, *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    _write_text(
        path,
        "".join(
            [
                f"command: {' '.join(command)}\n",
                f"returncode: {completed.returncode}\n",
                "\n--- stdout ---\n",
                completed.stdout or "",
                "\n--- stderr ---\n",
                completed.stderr or "",
            ]
        ),
    )


def _parse_gpu_devices(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _detect_gpu_devices(env: Mapping[str, str] | None = None) -> list[str]:
    env = env or os.environ
    visible = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible and visible.lower() not in {"all", "none", "void", "-1"}:
        devices = _parse_gpu_devices(visible)
        if devices:
            return devices
    if shutil.which("nvidia-smi") is None:
        return ["0"]
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return ["0"]
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return devices or ["0"]


def _build_worker_slots(
    gpu_devices: list[str],
    *,
    jobs_per_gpu: int,
    cpu_cores: int,
    cpu_cores_per_worker: int | None = None,
    max_workers: int | None = None,
) -> list[WorkerSlot]:
    if not gpu_devices:
        raise ValueError("at least one GPU device is required")
    jobs_per_gpu = max(1, int(jobs_per_gpu))
    cpu_cores = max(1, int(cpu_cores))
    gpu_schedule = [
        gpu_id
        for _lane in range(jobs_per_gpu)
        for gpu_id in gpu_devices
    ]
    if max_workers is not None and int(max_workers) > 0:
        gpu_schedule = gpu_schedule[: int(max_workers)]
    if not gpu_schedule:
        raise ValueError("worker schedule is empty")
    threads = (
        int(cpu_cores_per_worker)
        if cpu_cores_per_worker is not None and int(cpu_cores_per_worker) > 0
        else max(1, cpu_cores // len(gpu_schedule))
    )
    slots: list[WorkerSlot] = []
    for slot_id, gpu_id in enumerate(gpu_schedule):
        start = (slot_id * threads) % cpu_cores
        cores = tuple(range(start, min(start + threads, cpu_cores)))
        if not cores:
            cores = (slot_id % cpu_cores,)
        slots.append(
            WorkerSlot(
                slot_id=slot_id,
                gpu_id=str(gpu_id),
                cpu_cores=cores,
                cpu_threads=max(1, min(threads, len(cores))),
            )
        )
    return slots


def _slot_gpu_ids(slots: list[WorkerSlot]) -> list[str]:
    return [str(slot.gpu_id) for slot in slots]


def _assignment_cpu_core_map(schedule_json: Path, slots: list[WorkerSlot]) -> dict[str, list[int]]:
    payload = _load_json(schedule_json)
    assignments = [
        item
        for item in payload.get("assignments", [])
        if isinstance(item, Mapping)
    ]
    out: dict[str, list[int]] = {}
    for assignment, slot in zip(assignments, slots):
        assignment_id = str(assignment.get("assignment_id") or assignment.get("gpu_id") or slot.slot_id)
        out[assignment_id] = list(slot.cpu_cores)
    return out


def _cpu_threads_per_persistent_worker(slots: list[WorkerSlot]) -> int:
    if not slots:
        return 0
    return max(1, min(int(slot.cpu_threads) for slot in slots))


def _render_runtime_env(base_env: Mapping[str, str], python_bin: Path | None) -> dict[str, str]:
    env: dict[str, str] = {"PYOPENGL_PLATFORM": str(base_env.get("PYOPENGL_PLATFORM", "egl"))}
    if python_bin is None:
        return env

    env_root = Path(python_bin).expanduser().parent.parent
    env_bin = env_root / "bin"
    existing_path = str(base_env.get("PATH", ""))
    if env_bin.is_dir():
        env["PATH"] = ":".join(
            [str(env_bin)] + [part for part in existing_path.split(":") if part and part != str(env_bin)]
        )
    env["TORCH_EXTENSIONS_DIR"] = str(
        base_env.get("TORCH_EXTENSIONS_DIR")
        or "/team/telenav/code/torch_extensions/navdp_cuda121"
    )
    env["TORCH_CUDA_ARCH_LIST"] = str(base_env.get("TORCH_CUDA_ARCH_LIST") or "9.0")
    lib_dirs = [env_root / "lib"]
    lib_dirs.extend(sorted((env_root / "lib").glob("python*/site-packages/torch/lib")))
    cuda_home = Path(str(base_env.get("CUDA_HOME") or "/usr/local/cuda-12.9")).expanduser()
    env["CUDA_HOME"] = str(cuda_home)
    lib_dirs.append(cuda_home / "lib64")
    lib_dirs.append(Path("/usr/local/cuda-12.9/lib64"))

    existing = str(base_env.get("LD_LIBRARY_PATH", ""))
    parts: list[str] = []
    for lib_dir in lib_dirs:
        if lib_dir.is_dir():
            lib_dir_str = str(lib_dir)
            if lib_dir_str not in parts:
                parts.append(lib_dir_str)
    parts.extend(part for part in existing.split(":") if part and part not in parts)
    if parts:
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def _worker_env(
    *,
    base_env: Mapping[str, str],
    slot: WorkerSlot,
    ffmpeg_bin: Path | None = None,
    python_bin: Path | None = None,
) -> dict[str, str]:
    env = {str(k): str(v) for k, v in base_env.items()}
    thread_count = str(max(1, int(slot.cpu_threads)))
    env.update(_render_runtime_env(env, python_bin))
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(slot.gpu_id),
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
            "VECLIB_MAXIMUM_THREADS": thread_count,
            "TORCH_NUM_THREADS": thread_count,
            "MALLOC_ARENA_MAX": "2",
            "GAUSSIAN_RENDER_BACKEND": "gsplat",
        }
    )
    if ffmpeg_bin is not None:
        env["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_bin)
        env["FFMPEG_BIN"] = str(ffmpeg_bin)
    return env


def _taskset_command(command: list[str], *, slot: WorkerSlot, enabled: bool) -> list[str]:
    if not enabled or platform.system().lower() != "linux" or not slot.cpu_cores:
        return command
    taskset = shutil.which("taskset")
    if taskset is None:
        return command
    return [taskset, "-c", slot.core_list, *command]


def _run_capture(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str],
    slot: WorkerSlot,
    cpu_affinity: bool,
) -> tuple[subprocess.CompletedProcess[str], float]:
    effective_cmd = _taskset_command(cmd, slot=slot, enabled=cpu_affinity)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    completed = subprocess.run(
        effective_cmd,
        cwd=cwd,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    logged_env = {
        key: env.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TORCH_NUM_THREADS",
            "IMAGEIO_FFMPEG_EXE",
        )
        if env.get(key) is not None
    }
    lines = [
        f"command: {' '.join(effective_cmd)}\n",
        f"started_at: {started}\n",
        f"wall_time_sec: {elapsed:.6f}\n",
        f"returncode: {completed.returncode}\n",
        f"worker_slot: {slot.slot_id}\n",
        f"gpu_id: {slot.gpu_id}\n",
        f"cpu_cores: {slot.core_list}\n",
        f"env: {json.dumps(logged_env, sort_keys=True)}\n",
        "\n--- stdout ---\n",
        completed.stdout or "",
        "\n--- stderr ---\n",
        completed.stderr or "",
    ]
    _write_text(log_path, "".join(lines))
    return completed, elapsed


def _family_names(package_root: Path, requested: list[str] | None) -> list[str]:
    names = [
        path.name
        for path in sorted(package_root.iterdir())
        if path.is_dir() and (path / "render_manifest.json").is_file()
    ]
    if requested:
        allowed = set(requested)
        names = [name for name in names if name in allowed]
    return names


def _copy_package_metadata(package_root: Path, results_root: Path) -> None:
    for src in sorted(package_root.glob("*.json")):
        if src.is_file():
            shutil.copy2(src, results_root / src.name)


def _actor_count(actor_jsons: list[Path]) -> int | None:
    if not actor_jsons:
        return None
    try:
        payload = _load_json(actor_jsons[0])
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload.get("actor_ids"), list):
        return len(payload["actor_ids"])
    if isinstance(payload.get("actors"), list):
        return len(payload["actors"])
    frames = payload.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, Mapping) and isinstance(first.get("actors"), list):
            return len(first["actors"])
    return None


def _first_metric_values(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        "frames_total": None,
        "paths_ok": None,
        "duration_total_sec": None,
        "time_per_frame_sec": None,
    }
    if not metrics:
        return values
    first = metrics[0]
    values.update(
        {
            "frames_total": first.get("frames_total"),
            "paths_ok": first.get("paths_ok"),
            "duration_total_sec": first.get("duration_total_sec"),
            "time_per_frame_sec": first.get("time_per_frame_sec"),
        }
    )
    return values


def _collect_family(
    family_dir: Path,
    plan_payload: Mapping[str, Any],
    render_rc: int,
    wall_time_sec: float,
    retry_count: int,
    slot: WorkerSlot,
) -> dict[str, Any]:
    metric_payloads: list[dict[str, Any]] = []
    for metrics_path in sorted((family_dir / "metrics").glob("*.json")):
        try:
            payload = _load_json(metrics_path)
        except (OSError, json.JSONDecodeError) as exc:
            payload = {"error": str(exc)}
        payload["_path"] = str(metrics_path)
        metric_payloads.append(payload)

    videos = sorted((family_dir / "renders").glob("**/*.mp4"))
    actor_jsons = sorted((family_dir / "renders").glob("**/*_actors.json"))
    camera_jsons = sorted((family_dir / "renders").glob("**/*_camera.json"))
    label_paths = sorted((family_dir / "render_inputs").glob("**/label_paths/*.json"))
    actor_plans = sorted((family_dir / "render_inputs").glob("**/actor_plans/*.json"))
    first_metrics = _first_metric_values(metric_payloads)
    return {
        "family": family_dir.name,
        "status": "success" if render_rc == 0 and videos else "failed",
        "render_returncode": int(render_rc),
        "retry_count": int(retry_count),
        "wall_time_sec": float(wall_time_sec),
        "plan_status": plan_payload.get("status"),
        "job_count": plan_payload.get("job_count"),
        "frames_total": first_metrics["frames_total"],
        "paths_ok": first_metrics["paths_ok"],
        "duration_total_sec": first_metrics["duration_total_sec"],
        "time_per_frame_sec": first_metrics["time_per_frame_sec"],
        "actor_count": _actor_count(actor_jsons),
        "videos": [str(path) for path in videos],
        "actor_jsons": [str(path) for path in actor_jsons],
        "camera_jsons": [str(path) for path in camera_jsons],
        "label_paths": [str(path) for path in label_paths],
        "actor_plans": [str(path) for path in actor_plans],
        "metrics": metric_payloads,
        "worker_slot": slot.slot_id,
        "gpu_id": slot.gpu_id,
        "cpu_cores": list(slot.cpu_cores),
        "cpu_threads": slot.cpu_threads,
    }


def _render_family(
    args: argparse.Namespace,
    family: str,
    *,
    logs_root: Path,
    families_root: Path,
    slot: WorkerSlot,
) -> dict[str, Any]:
    src_family = args.package_root / family
    family_dir = families_root / family
    if family_dir.exists():
        shutil.rmtree(family_dir)
    shutil.copytree(src_family, family_dir)

    env = _worker_env(base_env=os.environ, slot=slot, ffmpeg_bin=args.ffmpeg_bin, python_bin=args.python_bin)
    manifest_json = family_dir / "render_manifest.json"
    plan_json = family_dir / "render_plan.json"
    base_cmd = [
        str(args.python_bin),
        "scripts/massgen/render_manifest_jobs.py",
        "--manifest-json",
        str(manifest_json),
        "--output-root",
        str(family_dir),
        "--write-inputs",
        "--video-backend",
        str(args.video_backend),
        "--device",
        str(args.device),
        "--json",
    ]
    if bool(args.actor_gpu_resident):
        base_cmd.append("--actor-gpu-resident")
    if args.minimal_frames is not None and int(args.minimal_frames) > 0:
        base_cmd.extend(["--minimal-frames", str(int(args.minimal_frames))])
    if args.limit is not None and int(args.limit) > 0:
        base_cmd.extend(["--limit", str(int(args.limit))])

    plan_completed, plan_elapsed = _run_capture(
        base_cmd,
        cwd=args.repo_root,
        log_path=logs_root / f"{family}_plan.log",
        env=env,
        slot=slot,
        cpu_affinity=bool(args.cpu_affinity),
    )
    _write_text(plan_json, plan_completed.stdout or "{}")
    try:
        plan_payload = json.loads(plan_completed.stdout)
    except json.JSONDecodeError as exc:
        plan_payload = {"status": "invalid", "job_count": 0, "error": str(exc)}

    render_rc = int(plan_completed.returncode or 2)
    render_elapsed = 0.0
    retry_count = 0
    if plan_completed.returncode == 0 and plan_payload.get("status") == "ready":
        render_cmd = [*base_cmd, "--execute"]
        render_completed, render_elapsed = _run_capture(
            render_cmd,
            cwd=args.repo_root,
            log_path=logs_root / f"{family}_render.log",
            env=env,
            slot=slot,
            cpu_affinity=bool(args.cpu_affinity),
        )
        render_rc = int(render_completed.returncode)
        while render_rc != 0 and retry_count < int(args.retry):
            retry_count += 1
            retry_completed, render_elapsed = _run_capture(
                render_cmd,
                cwd=args.repo_root,
                log_path=logs_root / f"{family}_render_retry{retry_count}.log",
                env=env,
                slot=slot,
                cpu_affinity=bool(args.cpu_affinity),
            )
            render_rc = int(retry_completed.returncode)
            _write_text(
                family_dir / f"time_retry{retry_count}.txt",
                f"wall_time_sec={render_elapsed:.6f}\nreturncode={render_rc}\n",
            )

    _write_text(
        family_dir / "time.txt",
        (
            f"plan_wall_time_sec={plan_elapsed:.6f}\n"
            f"render_wall_time_sec={render_elapsed:.6f}\n"
            f"returncode={render_rc}\n"
            f"retry_count={retry_count}\n"
            f"worker_slot={slot.slot_id}\n"
            f"gpu_id={slot.gpu_id}\n"
            f"cpu_cores={slot.core_list}\n"
            f"cpu_threads={slot.cpu_threads}\n"
        ),
    )
    return _collect_family(family_dir, plan_payload, render_rc, render_elapsed, retry_count, slot)


def _sample_nvidia_smi(gpu_devices: list[str]) -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    if gpu_devices:
        cmd.insert(1, f"--id={','.join(gpu_devices)}")
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return []
    samples: list[dict[str, Any]] = []
    reader = csv.reader(completed.stdout.splitlines())
    polled_at = datetime.now(timezone.utc).isoformat()
    for row in reader:
        if len(row) < 5:
            continue
        try:
            samples.append(
                {
                    "polled_at": polled_at,
                    "nvidia_timestamp": row[0].strip(),
                    "gpu_index": row[1].strip(),
                    "gpu_util_pct": float(row[2].strip()),
                    "memory_used_mb": float(row[3].strip()),
                    "memory_total_mb": float(row[4].strip()),
                }
            )
        except ValueError:
            continue
    return samples


def _summarize_gpu_utilization(
    samples: list[Mapping[str, Any]],
    *,
    target_gpu_util: float,
) -> dict[str, Any]:
    by_gpu: dict[str, list[float]] = {}
    for sample in samples:
        gpu_index = str(sample.get("gpu_index", "unknown"))
        value = sample.get("gpu_util_pct")
        if value is None:
            continue
        by_gpu.setdefault(gpu_index, []).append(float(value))
    per_gpu = {}
    for gpu_index, values in sorted(by_gpu.items()):
        per_gpu[gpu_index] = {
            "sample_count": len(values),
            "avg_gpu_util_pct": sum(values) / len(values),
            "max_gpu_util_pct": max(values),
            "target_met": (sum(values) / len(values)) >= float(target_gpu_util),
        }
    avg_values = [item["avg_gpu_util_pct"] for item in per_gpu.values()]
    return {
        "target_gpu_util_pct": float(target_gpu_util),
        "sample_count": len(samples),
        "per_gpu": per_gpu,
        "avg_gpu_util_pct": sum(avg_values) / len(avg_values) if avg_values else None,
        "target_met": all(item["target_met"] for item in per_gpu.values()) if per_gpu else None,
    }


def _persistent_plan_command(args: argparse.Namespace, *, slots: list[WorkerSlot]) -> list[str]:
    materialized_root = args.materialized_root or (args.results_root / "materialized")
    command = [
        str(args.python_bin),
        "scripts/massgen/plan_persistent_h100_schedule.py",
        "--package-root",
        str(args.package_root),
        "--materialized-root",
        str(materialized_root),
        "--render-plan-output-json",
        str(args.results_root / "aggregate_render_plan.json"),
        "--output-json",
        str(args.results_root / "persistent_schedule.json"),
        "--python-bin",
        str(args.python_bin),
        "--render-script",
        str(args.render_script),
        "--video-backend",
        str(args.video_backend),
        "--device",
        str(args.device),
        "--workers-per-gpu",
        "1",
        "--max-items-per-chunk",
        str(int(args.max_items_per_chunk or 0)),
        "--scene-vram-gib",
        str(float(args.scene_vram_gib)),
        "--human-avatar-vram-mib",
        str(float(args.human_avatar_vram_mib)),
        "--robot-asset-vram-mib",
        str(float(args.robot_asset_vram_mib)),
        "--actor-plan-ram-mib",
        str(float(args.actor_plan_ram_mib)),
        "--include-execution",
    ]
    if bool(args.scene_affinity):
        command.append("--scene-affinity")
    else:
        command.append("--no-scene-affinity")
    if args.minimal_frames is not None and int(args.minimal_frames) > 0:
        command.extend(["--minimal-frames", str(int(args.minimal_frames))])
    for family in args.family or []:
        command.extend(["--family", str(family)])
    for source in args.source or []:
        command.extend(["--source", str(source)])
    for scene in args.scene or []:
        command.extend(["--scene", str(scene)])
    if int(args.max_renders or 0) > 0:
        command.extend(["--max-renders", str(int(args.max_renders))])
    if int(args.renders_per_family_source_scene or 0) > 0:
        command.extend(
            [
                "--renders-per-family-source-scene",
                str(int(args.renders_per_family_source_scene)),
            ]
        )
    if bool(args.actor_gpu_resident):
        command.append("--actor-gpu-resident")
    else:
        command.append("--no-actor-gpu-resident")
    if bool(args.actor_runtime_cache):
        command.append("--actor-runtime-cache")
    else:
        command.append("--no-actor-runtime-cache")
    for gpu_id in _slot_gpu_ids(slots):
        command.extend(["--gpu-id", gpu_id])
    if args.simulate_cache_gib is not None:
        command.extend(["--simulate-cache-gib", str(float(args.simulate_cache_gib))])
    return command


def _persistent_run_command(
    args: argparse.Namespace,
    *,
    slots: list[WorkerSlot],
    cpu_map_json: Path,
) -> list[str]:
    command = [
        str(args.python_bin),
        "scripts/massgen/run_persistent_h100_schedule.py",
        "--schedule-json",
        str(args.results_root / "persistent_schedule.json"),
        "--results-root",
        str(args.results_root / "run_persistent"),
        "--repo-root",
        str(args.repo_root),
        "--workers",
        str(len(slots)),
        "--group-max-labels-per-command",
        str(int(args.group_max_labels_per_command or 0)),
        "--command-attempts",
        str(max(1, int(args.command_attempts or 1))),
        "--gpu-sample-interval-sec",
        str(float(args.util_sample_interval_sec)),
        "--cpu-threads-per-worker",
        str(_cpu_threads_per_persistent_worker(slots)),
        "--assignment-cpu-cores-json",
        str(cpu_map_json),
        "--preemptible-output",
    ]
    if bool(args.cpu_affinity):
        command.append("--cpu-affinity")
    else:
        command.append("--no-cpu-affinity")
    if bool(args.resume):
        command.append("--resume")
    else:
        command.append("--no-resume")
    if bool(args.clean):
        command.append("--clean")
    if bool(args.dry_run):
        command.append("--dry-run")
    return command


def _persistent_report_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python_bin),
        "scripts/massgen/report_persistent_h100_schedule_run.py",
        "--run-root",
        str(args.results_root / "run_persistent"),
        "--output-root",
        str(args.results_root / "report_persistent"),
        "--title",
        "MassGen Persistent H100 Pipeline",
        "--stage-window-min",
        str(float(args.stage_window_min)),
    ]
    natural_json = args.results_root / "natural_length_projection.json"
    if natural_json.is_file():
        command.extend(["--natural-length-json", str(natural_json)])
    for baseline in args.baseline_summary or []:
        command.extend(["--baseline-summary", str(Path(baseline).expanduser())])
    return command


def _persistent_runner_env(args: argparse.Namespace) -> dict[str, str]:
    env = _render_runtime_env(os.environ, args.python_bin)
    env["GAUSSIAN_RENDER_BACKEND"] = "gsplat"
    if args.ffmpeg_bin is not None:
        env["IMAGEIO_FFMPEG_EXE"] = str(args.ffmpeg_bin)
        env["FFMPEG_BIN"] = str(args.ffmpeg_bin)
    return env


def _run_persistent_pipeline(args: argparse.Namespace) -> int:
    if args.video_backend != "cpu":
        print(
            "[WARN] H100 persistent profile is intended for --video-backend cpu because H100 has no NVENC/RT blocks.",
            file=sys.stderr,
            flush=True,
        )

    gpu_devices = _parse_gpu_devices(args.gpu_devices) or _detect_gpu_devices()
    slots = _build_worker_slots(
        gpu_devices,
        jobs_per_gpu=int(args.jobs_per_gpu),
        cpu_cores=int(args.cpu_cores),
        cpu_cores_per_worker=args.cpu_cores_per_worker,
        max_workers=args.max_workers,
    )

    if bool(args.clean) and args.results_root.exists():
        shutil.rmtree(args.results_root)
    logs_root = args.results_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    _copy_package_metadata(args.package_root, args.results_root)

    run_config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(args.repo_root),
        "package_root": str(args.package_root),
        "results_root": str(args.results_root),
        "python_bin": str(args.python_bin),
        "render_script": str(args.render_script),
        "ffmpeg_bin": str(args.ffmpeg_bin) if args.ffmpeg_bin is not None else None,
        "pipeline_profile": "h100_cpu_encode_persistent_schedule",
        "pipeline_mode": "persistent",
        "video_backend": str(args.video_backend),
        "device": str(args.device),
        "minimal_frames": args.minimal_frames,
        "families": list(args.family or []),
        "sources": list(args.source or []),
        "scenes": list(args.scene or []),
        "max_renders": int(args.max_renders or 0),
        "renders_per_family_source_scene": int(args.renders_per_family_source_scene or 0),
        "max_items_per_chunk": int(args.max_items_per_chunk or 0),
        "scene_affinity": bool(args.scene_affinity),
        "group_max_labels_per_command": int(args.group_max_labels_per_command or 0),
        "command_attempts": int(args.command_attempts or 1),
        "run_render": bool(args.run_render),
        "run_analysis": bool(args.run_analysis),
        "resume": bool(args.resume),
        "clean": bool(args.clean),
        "dry_run": bool(args.dry_run),
        "h100_profile": {
            "gpu_devices": gpu_devices,
            "jobs_per_gpu": int(args.jobs_per_gpu),
            "worker_count": len(slots),
            "cpu_cores": int(args.cpu_cores),
            "cpu_affinity": bool(args.cpu_affinity),
            "cpu_threads_per_worker": _cpu_threads_per_persistent_worker(slots),
            "actor_gpu_resident": bool(args.actor_gpu_resident),
            "actor_runtime_cache": bool(args.actor_runtime_cache),
            "target_gpu_util_pct": float(args.target_gpu_util),
            "util_sample_interval_sec": float(args.util_sample_interval_sec),
            "slots": [
                {
                    "slot_id": slot.slot_id,
                    "gpu_id": slot.gpu_id,
                    "cpu_cores": list(slot.cpu_cores),
                    "cpu_threads": slot.cpu_threads,
                }
                for slot in slots
            ],
        },
    }
    _write_json(args.results_root / "h100_persistent_run_config.json", run_config)
    _write_text(
        args.results_root / "run_config.env",
        "\n".join(
            [
                f"started_at={datetime.now().isoformat(timespec='seconds')}",
                f"repo={args.repo_root}",
                f"package_root={args.package_root}",
                f"results_root={args.results_root}",
                f"pipeline_mode=persistent",
                f"gpu_devices={','.join(gpu_devices)}",
                f"jobs_per_gpu={int(args.jobs_per_gpu)}",
                f"worker_count={len(slots)}",
                f"minimal_frames={args.minimal_frames}",
                "",
            ]
        ),
    )
    _capture_to_file(["df", "-h"], args.results_root / "df_before.txt", cwd=args.repo_root)
    _capture_to_file(["git", "status", "--short", "--branch"], args.results_root / "git_status_before.txt", cwd=args.repo_root)
    if shutil.which("nvidia-smi") is not None:
        _capture_to_file(["nvidia-smi"], args.results_root / "nvidia_smi_before.txt", cwd=args.repo_root)

    plan_rc, _ = _run_logged(
        _persistent_plan_command(args, slots=slots),
        cwd=args.repo_root,
        log_path=logs_root / "plan_persistent.log",
    )
    if plan_rc != 0:
        return plan_rc

    schedule_json = args.results_root / "persistent_schedule.json"
    cpu_map_json = args.results_root / "assignment_cpu_cores.json"
    _write_json(cpu_map_json, _assignment_cpu_core_map(schedule_json, slots))

    length_rc, _ = _run_logged(
        [
            str(args.python_bin),
            "scripts/massgen/summarize_persistent_schedule_lengths.py",
            "--aggregate-render-plan-json",
            str(args.results_root / "aggregate_render_plan.json"),
            "--schedule-json",
            str(schedule_json),
            "--output-json",
            str(args.results_root / "natural_length_projection.json"),
        ],
        cwd=args.repo_root,
        log_path=logs_root / "natural_lengths.log",
    )
    if length_rc != 0:
        return length_rc

    run_rc = 0
    if bool(args.run_render):
        run_rc, _ = _run_logged(
            _persistent_run_command(args, slots=slots, cpu_map_json=cpu_map_json),
            cwd=args.repo_root,
            log_path=logs_root / "run_persistent.log",
            env=_persistent_runner_env(args),
        )

    report_rc = 0
    if bool(args.run_analysis) and (args.results_root / "run_persistent" / "benchmark_summary.json").is_file():
        report_env = {
            "MPLCONFIGDIR": str(args.results_root / "mplconfig"),
            "XDG_CACHE_HOME": str(args.results_root / "xdg_cache"),
            "FC_CACHEDIR": str(args.results_root / "fontconfig"),
        }
        for value in report_env.values():
            Path(value).mkdir(parents=True, exist_ok=True)
        report_rc, _ = _run_logged(
            _persistent_report_command(args),
            cwd=args.repo_root,
            log_path=logs_root / "report_persistent.log",
            env=report_env,
        )

    _capture_to_file(["df", "-h"], args.results_root / "df_after.txt", cwd=args.repo_root)
    if shutil.which("nvidia-smi") is not None:
        _capture_to_file(["nvidia-smi"], args.results_root / "nvidia_smi_after.txt", cwd=args.repo_root)
    mp4_files = sorted((args.results_root / "run_persistent").glob("**/*.mp4"))
    _write_text(args.results_root / "mp4_files.txt", "\n".join(str(path) for path in mp4_files) + ("\n" if mp4_files else ""))
    _write_text(args.results_root / "mp4_count.txt", f"{len(mp4_files)}\n")
    with (args.results_root / "run_config.env").open("a", encoding="utf-8") as handle:
        handle.write(f"finished_at={datetime.now().isoformat(timespec='seconds')}\n")

    if run_rc != 0:
        return run_rc
    return report_rc


def main() -> int:
    args = _parse_args()
    args.package_root = args.package_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    args.render_script = args.render_script.expanduser().resolve()
    args.ffmpeg_bin = args.ffmpeg_bin.expanduser().resolve() if args.ffmpeg_bin is not None else None
    args.materialized_root = args.materialized_root.expanduser().resolve() if args.materialized_root is not None else None
    args.baseline_summary = [
        str(Path(item).expanduser().resolve())
        for item in (args.baseline_summary or [])
    ]

    if args.pipeline_mode == "persistent":
        return _run_persistent_pipeline(args)

    if args.video_backend != "cpu":
        print(
            "[WARN] H100 profile is intended for --video-backend cpu because H100 has no NVENC/RT blocks.",
            file=sys.stderr,
            flush=True,
        )

    gpu_devices = _parse_gpu_devices(args.gpu_devices) or _detect_gpu_devices()
    slots = _build_worker_slots(
        gpu_devices,
        jobs_per_gpu=int(args.jobs_per_gpu),
        cpu_cores=int(args.cpu_cores),
        cpu_cores_per_worker=args.cpu_cores_per_worker,
        max_workers=args.max_workers,
    )

    if args.results_root.exists():
        shutil.rmtree(args.results_root)
    logs_root = args.results_root / "logs"
    families_root = args.results_root / "families"
    logs_root.mkdir(parents=True, exist_ok=True)
    families_root.mkdir(parents=True, exist_ok=True)
    _copy_package_metadata(args.package_root, args.results_root)

    families = _family_names(args.package_root, args.family)
    slot_queue: queue.Queue[WorkerSlot] = queue.Queue()
    for slot in slots:
        slot_queue.put(slot)

    monitor = NvidiaSmiMonitor(
        gpu_devices=gpu_devices,
        interval_sec=float(args.util_sample_interval_sec),
        output_jsonl=args.results_root / "gpu_utilization.jsonl",
    )
    if bool(args.util_monitor):
        monitor.start()

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(args.repo_root),
        "package_root": str(args.package_root),
        "results_root": str(args.results_root),
        "python_bin": str(args.python_bin),
        "pipeline_profile": "h100_cpu_encode_multi_process",
        "video_backend": str(args.video_backend),
        "device": str(args.device),
        "minimal_frames": args.minimal_frames,
        "limit": args.limit,
        "h100_profile": {
            "gpu_devices": gpu_devices,
            "jobs_per_gpu": int(args.jobs_per_gpu),
            "worker_count": len(slots),
            "cpu_cores": int(args.cpu_cores),
            "cpu_affinity": bool(args.cpu_affinity),
            "actor_gpu_resident": bool(args.actor_gpu_resident),
            "target_gpu_util_pct": float(args.target_gpu_util),
            "slots": [
                {
                    "slot_id": slot.slot_id,
                    "gpu_id": slot.gpu_id,
                    "cpu_cores": list(slot.cpu_cores),
                    "cpu_threads": slot.cpu_threads,
                }
                for slot in slots
            ],
        },
        "warnings": [],
        "families": [],
    }
    if len(families) < len(slots):
        summary["warnings"].append(
            f"family_count={len(families)} is smaller than worker_count={len(slots)}; "
            "this run may not reach the GPU utilization target."
        )

    jsonl_path = args.results_root / "family_render_summary.jsonl"
    def run_queued_family(family: str) -> dict[str, Any]:
        slot = slot_queue.get()
        try:
            return _render_family(
                args,
                family,
                logs_root=logs_root,
                families_root=families_root,
                slot=slot,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "family": family,
                "status": "failed",
                "error": str(exc),
                "worker_slot": slot.slot_id,
                "gpu_id": slot.gpu_id,
                "cpu_cores": list(slot.cpu_cores),
                "cpu_threads": slot.cpu_threads,
            }
        finally:
            slot_queue.put(slot)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as executor:
        future_to_family = {
            executor.submit(run_queued_family, family): family
            for family in families
        }

        for future in concurrent.futures.as_completed(future_to_family):
            family = future_to_family[future]
            try:
                record = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                record = {
                    "family": family,
                    "status": "failed",
                    "error": str(exc),
                }
            summary["families"].append(record)
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                (
                    f"{family}: {record.get('status')} rc={record.get('render_returncode')} "
                    f"gpu={record.get('gpu_id')} slot={record.get('worker_slot')} "
                    f"videos={len(record.get('videos', []))} frames={record.get('frames_total')} "
                    f"actors={record.get('actor_count')} wall={float(record.get('wall_time_sec') or 0.0):.2f}s"
                ),
                flush=True,
            )

    samples = monitor.stop() if bool(args.util_monitor) else []
    utilization = _summarize_gpu_utilization(samples, target_gpu_util=float(args.target_gpu_util))
    _write_text(
        args.results_root / "gpu_utilization_summary.json",
        json.dumps(utilization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    summary["families"] = sorted(summary["families"], key=lambda item: str(item.get("family")))
    summary["family_count"] = len(summary["families"])
    summary["success_count"] = sum(1 for item in summary["families"] if item.get("status") == "success")
    summary["status"] = "success" if summary["success_count"] == summary["family_count"] else "failed"
    summary["gpu_utilization"] = utilization
    if utilization.get("target_met") is False:
        summary["warnings"].append(
            "Average sampled GPU utilization did not meet target; increase --jobs-per-gpu or "
            "--max-workers if CPU/RAM headroom remains, or raise resolution/batch size for benchmarking."
        )
    _write_text(
        args.results_root / "family_render_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"summary_status {summary['status']} {summary['success_count']} / {summary['family_count']}", flush=True)
    print(
        (
            f"gpu_util avg={utilization.get('avg_gpu_util_pct')} "
            f"target={utilization.get('target_gpu_util_pct')} met={utilization.get('target_met')}"
        ),
        flush=True,
    )
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
