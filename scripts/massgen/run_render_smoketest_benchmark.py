#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.render_executor import build_render_plans, load_render_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MassGen render smoke manifests and record speed/space/GPU usage."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--render-script", type=Path, default=None)
    parser.add_argument("--video-backend", choices=["nvenc", "cpu", "gpu"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimal-frames", type=int, default=None)
    parser.add_argument("--max-renders", type=int, default=0)
    parser.add_argument(
        "--renders-per-family-source-scene",
        type=int,
        default=0,
        help="If >0, cap selected manifests per (family, source, scene) group before --max-renders.",
    )
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--skip-expected-blocked", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-sample-interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of render manifests to execute concurrently on the selected device.",
    )
    parser.add_argument(
        "--group-same-scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Group compatible selected manifests by family/source/scene and render many "
            "label IDs in one renderer process per group."
        ),
    )
    parser.add_argument(
        "--group-max-labels-per-command",
        type=int,
        default=0,
        help=(
            "When --group-same-scene is enabled, split compatible scene groups into "
            "renderer commands with at most this many label plans. 0 keeps each "
            "compatible scene bucket in one command."
        ),
    )
    parser.add_argument(
        "--group-max-manifests-per-task",
        type=int,
        default=0,
        help=(
            "When --group-same-scene is enabled, split each family/source/scene "
            "group into independent outer scheduler tasks with at most this many "
            "manifests. 0 keeps one outer task per scene group."
        ),
    )
    parser.add_argument(
        "--actor-gpu-resident",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward --actor-gpu-resident to MassGen render jobs.",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_component(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("__")
    return "".join(safe).strip("_") or "unnamed"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _run_capture(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "".join(
            [
                f"command: {' '.join(cmd)}\n",
                f"started_at: {started}\n",
                f"wall_time_sec: {elapsed:.6f}\n",
                f"returncode: {completed.returncode}\n",
                "\n--- stdout ---\n",
                completed.stdout or "",
                "\n--- stderr ---\n",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return completed, elapsed


def _read_gpu_sample() -> dict[str, Any] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    line = (completed.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 9:
        return None
    return {
        "timestamp": parts[0],
        "index": int(parts[1]),
        "gpu_util_pct": float(parts[2]),
        "mem_util_pct": float(parts[3]),
        "memory_used_mib": float(parts[4]),
        "memory_free_mib": float(parts[5]),
        "memory_total_mib": float(parts[6]),
        "power_w": float(parts[7]),
        "temperature_c": float(parts[8]),
    }


class GpuMonitor:
    def __init__(self, interval_sec: float, *, log_path: Path | None = None) -> None:
        self.interval_sec = max(0.1, float(interval_sec))
        self.log_path = log_path
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMonitor":
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _read_gpu_sample()
            if sample is not None:
                self.samples.append(sample)
                if self.log_path is not None:
                    with self.log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(sample, sort_keys=True) + "\n")
            self._stop.wait(self.interval_sec)


def _gpu_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    gpu_utils = [float(item["gpu_util_pct"]) for item in samples]
    mem_utils = [float(item["mem_util_pct"]) for item in samples]
    used = [float(item["memory_used_mib"]) for item in samples]
    return {
        "sample_count": len(samples),
        "gpu_util_avg_pct": sum(gpu_utils) / len(gpu_utils),
        "gpu_util_max_pct": max(gpu_utils),
        "mem_util_avg_pct": sum(mem_utils) / len(mem_utils),
        "memory_used_min_mib": min(used),
        "memory_used_max_mib": max(used),
        "memory_used_delta_mib": max(used) - min(used),
    }


def _metric_payloads(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for metric in sorted((path / "metrics").glob("*.json")):
        try:
            payload = _load_json(metric)
        except Exception as exc:  # pylint: disable=broad-except
            payload = {"error": str(exc)}
        payload["_path"] = str(metric)
        payloads.append(payload)
    return payloads


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _valid_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [metric for metric in metrics if isinstance(metric, Mapping) and not metric.get("error")]


def _render_overhead_summary(
    metrics: list[dict[str, Any]],
    *,
    outer_render_elapsed_sec: float,
) -> dict[str, Any]:
    valid = _valid_metrics(metrics)
    frames_total = sum(_as_int(metric.get("frames_total")) for metric in valid)
    paths_total = sum(_as_int(metric.get("paths_total")) for metric in valid)
    paths_ok = sum(_as_int(metric.get("paths_ok")) for metric in valid)
    nested_duration_total_sec = sum(_as_float(metric.get("duration_total_sec")) for metric in valid)

    nested_render_loop_sec = 0.0
    nested_process_total_sec = 0.0
    nested_setup_load_process_sec = 0.0
    process_total_count = 0
    for metric in valid:
        lifecycle = metric.get("lifecycle_seconds")
        if isinstance(lifecycle, Mapping):
            render_loop = _as_float(lifecycle.get("render_loop_sec"), _as_float(metric.get("duration_total_sec")))
            process_total = _as_float(lifecycle.get("process_total_sec"))
            setup_total = _as_float(lifecycle.get("process_minus_render_loop_sec"))
            if setup_total <= 0.0 and process_total > 0.0:
                setup_total = max(0.0, process_total - render_loop)
            nested_render_loop_sec += render_loop
            nested_process_total_sec += process_total
            nested_setup_load_process_sec += setup_total
            if process_total > 0.0:
                process_total_count += 1
        else:
            nested_render_loop_sec += _as_float(metric.get("duration_total_sec"))

    outer = max(0.0, float(outer_render_elapsed_sec or 0.0))
    hidden_overhead_sec = max(0.0, outer - nested_render_loop_sec)
    process_launches = len(valid)
    has_complete_process_totals = bool(valid) and process_total_count == len(valid)
    nested_process_total_value: float | None = (
        nested_process_total_sec if has_complete_process_totals else None
    )
    nested_setup_value: float | None = (
        nested_setup_load_process_sec if has_complete_process_totals else None
    )
    process_wrapper_overhead_sec = (
        max(0.0, outer - nested_process_total_sec)
        if has_complete_process_totals
        else None
    )
    return {
        "outer_render_elapsed_sec": outer,
        "renderer_process_count": process_launches,
        "nested_paths_total": paths_total,
        "nested_paths_ok": paths_ok,
        "nested_frames_total": frames_total,
        "nested_duration_total_sec": nested_duration_total_sec,
        "nested_render_loop_sec": nested_render_loop_sec,
        "nested_process_total_sec": nested_process_total_value,
        "nested_setup_load_process_sec": nested_setup_value,
        "hidden_overhead_sec": hidden_overhead_sec,
        "hidden_overhead_pct": (
            (hidden_overhead_sec / outer) * 100.0 if outer > 0.0 else None
        ),
        "process_wrapper_overhead_sec": process_wrapper_overhead_sec,
        "process_launches_per_1000_frames": (
            (float(process_launches) / float(frames_total)) * 1000.0
            if frames_total > 0
            else None
        ),
        "setup_seconds_per_frame": (
            hidden_overhead_sec / float(frames_total) if frames_total > 0 else None
        ),
        "renderer_setup_seconds_per_frame": (
            nested_setup_load_process_sec / float(frames_total)
            if frames_total > 0 and has_complete_process_totals
            else None
        ),
    }


def _summarize_render_overhead(records: list[dict[str, Any]]) -> dict[str, Any]:
    overheads = [
        record.get("render_overhead")
        for record in records
        if record.get("status") == "success" and isinstance(record.get("render_overhead"), Mapping)
    ]
    if not overheads:
        return {}
    outer = sum(_as_float(item.get("outer_render_elapsed_sec")) for item in overheads)
    render_loop = sum(_as_float(item.get("nested_render_loop_sec")) for item in overheads)
    duration_total = sum(_as_float(item.get("nested_duration_total_sec")) for item in overheads)
    frames = sum(_as_int(item.get("nested_frames_total")) for item in overheads)
    paths = sum(_as_int(item.get("nested_paths_total")) for item in overheads)
    processes = sum(_as_int(item.get("renderer_process_count")) for item in overheads)
    process_totals = [
        _as_float(item.get("nested_process_total_sec"))
        for item in overheads
        if item.get("nested_process_total_sec") is not None
    ]
    setup_totals = [
        _as_float(item.get("nested_setup_load_process_sec"))
        for item in overheads
        if item.get("nested_setup_load_process_sec") is not None
    ]
    has_complete_process_totals = len(process_totals) == len(overheads)
    has_complete_setup_totals = len(setup_totals) == len(overheads)
    process_total = sum(process_totals) if has_complete_process_totals else None
    setup_total = sum(setup_totals) if has_complete_setup_totals else None
    hidden = max(0.0, outer - render_loop)
    return {
        "record_count": len(overheads),
        "outer_render_elapsed_sec": outer,
        "renderer_process_count": processes,
        "nested_paths_total": paths,
        "nested_frames_total": frames,
        "nested_duration_total_sec": duration_total,
        "nested_render_loop_sec": render_loop,
        "nested_process_total_sec": process_total,
        "nested_setup_load_process_sec": setup_total,
        "hidden_overhead_sec": hidden,
        "hidden_overhead_pct": (hidden / outer) * 100.0 if outer > 0.0 else None,
        "process_wrapper_overhead_sec": (
            max(0.0, outer - process_total) if process_total is not None else None
        ),
        "process_launches_per_1000_frames": (
            (float(processes) / float(frames)) * 1000.0 if frames > 0 else None
        ),
        "setup_seconds_per_frame": hidden / float(frames) if frames > 0 else None,
        "renderer_setup_seconds_per_frame": (
            setup_total / float(frames) if frames > 0 and setup_total is not None else None
        ),
    }


def _entry_output_root(results_root: Path, entry: Mapping[str, Any]) -> Path:
    return (
        results_root
        / "renders"
        / _safe_component(str(entry["family"]))
        / _safe_component(str(entry["source"]))
        / _safe_component(str(entry["scene"]))
        / Path(str(entry["render_manifest_json"])).stem.removesuffix(".render_manifest")
    )


def _entry_log_stem(entry: Mapping[str, Any]) -> Path:
    return (
        Path(_safe_component(str(entry["family"])))
        / _safe_component(str(entry["source"]))
        / _safe_component(str(entry["scene"]))
        / Path(str(entry["render_manifest_json"])).stem.removesuffix(".render_manifest")
    )


def _entry_group_key(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("family")),
        str(entry.get("source")),
        str(entry.get("scene")),
    )


def _group_output_root(
    results_root: Path,
    key: tuple[str, str, str],
    *,
    chunk_index: int | None = None,
) -> Path:
    family, source, scene = key
    root = (
        results_root
        / "grouped_renders"
        / _safe_component(family)
        / _safe_component(source)
        / _safe_component(scene)
    )
    if chunk_index is not None:
        root = root / f"chunk_{chunk_index:04d}"
    return root


def _group_log_stem(
    key: tuple[str, str, str],
    *,
    chunk_index: int | None = None,
) -> Path:
    family, source, scene = key
    stem = Path(_safe_component(family)) / _safe_component(source) / _safe_component(scene)
    if chunk_index is not None:
        stem = stem / f"chunk_{chunk_index:04d}"
    return stem


_COMMAND_OPTIONS_WITH_VALUE = {
    "--scenes-dir",
    "--tasks-dir",
    "--scene",
    "--output-dir",
    "--metrics-json",
    "--error-log",
    "--skip-completed-log",
    "--label-id",
    "--max-labels",
    "--stride",
    "--resample-step",
    "--path-handedness",
    "--look-ahead",
    "--look-down",
    "--height-offset",
    "--resolution",
    "--fov-deg",
    "--znear",
    "--zfar",
    "--device",
    "--sh-degree",
    "--gaussian-model",
    "--video-fps",
    "--minimal-frames",
    "--view-mode",
    "--path-progress-space-interval-sec",
    "--video-backend",
    "--video-nvenc-preset",
    "--video-nvenc-bitrate",
    "--depth-bit-depth",
    "--ply-transform-backend",
    "--cl-light-mode",
    "--cl-shading-model",
    "--cl-strength",
    "--cl-color",
    "--cl-ambient",
    "--cl-base-scale",
    "--cl-diffuse",
    "--cl-specular",
    "--cl-shininess",
    "--cl-range",
    "--cl-offset",
    "--cl-light-world",
    "--cl-light-center-z",
    "--cl-normal-smooth",
    "--cl-normal-filter",
    "--cl-normal-kernel",
    "--cl-normal-sigma-range",
    "--cl-normal-sigma-domain",
    "--cl-shadow-bias",
    "--cl-shadow-strength",
    "--cl-shadow-pcf",
    "--cl-shadow-compare",
    "--npc-count",
    "--npc-max-count",
    "--npc-density-coverage",
    "--npc-priority",
    "--npc-density-mode",
    "--npc-zone-ratio",
    "--npc-max-range",
    "--npc-free-threshold",
    "--npc-placement-backend",
    "--npc-seed",
    "--npc-actor-root",
    "--npc-frame-pool-size",
    "--job-slot",
    "--job-name",
    "--job-actor-id",
    "--light-mode",
    "--light-strength",
    "--light-radius",
    "--light-center",
    "--light-jitter",
    "--light-temp-k",
    "--light-seed",
    "--actor-seq-dir",
    "--actor-plan-json",
    "--label-actor-plan-json",
    "--actor-pattern",
    "--actor-height",
    "--actor-speed",
    "--actor-fps",
    "--follow-distance",
    "--follow-buffer",
    "--actor-foot-offset",
    "--animation-cycle-mod",
    "--actor-cull-margin-m",
    "--actor-gpu-cache-mb",
    "--actor-gpu-sh-mode",
}

_GROUP_IGNORED_OPTIONS = {
    "--label-id",
    "--metrics-json",
    "--actor-plan-json",
    "--label-actor-plan-json",
}


def _option_values(command: list[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item == option and index + 1 < len(command):
            values.append(str(command[index + 1]))
            index += 2
            continue
        index += 1
    return values


def _single_option_value(command: list[str], option: str) -> str | None:
    values = _option_values(command, option)
    return values[-1] if values else None


def _command_without_group_specific_options(command: list[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item in _GROUP_IGNORED_OPTIONS:
            index += 2 if item in _COMMAND_OPTIONS_WITH_VALUE else 1
            continue
        out.append(item)
        if item in _COMMAND_OPTIONS_WITH_VALUE and index + 1 < len(command):
            out.append(command[index + 1])
            index += 2
        else:
            index += 1
    return out


def _replace_command_option(command: list[str], option: str, values: list[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item == option:
            index += 2 if item in _COMMAND_OPTIONS_WITH_VALUE else 1
            continue
        out.append(item)
        if item in _COMMAND_OPTIONS_WITH_VALUE and index + 1 < len(command):
            out.append(command[index + 1])
            index += 2
        else:
            index += 1
    for value in values:
        out.extend([option, str(value)])
    return out


def _build_grouped_render_commands(
    plans: list[Mapping[str, Any]],
    *,
    metrics_root: Path,
    max_labels_per_command: int = 0,
) -> list[list[str]]:
    buckets: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for plan in plans:
        if plan.get("blockers") or plan.get("robot_overlay_commands"):
            continue
        command = [str(item) for item in plan.get("command", [])]
        if not command:
            continue
        key = tuple(_command_without_group_specific_options(command))
        buckets.setdefault(key, []).append(plan)

    grouped_commands: list[list[str]] = []
    group_index = 0
    chunk_size = int(max_labels_per_command or 0)
    for bucket_plans in buckets.values():
        plan_chunks = (
            [
                bucket_plans[index : index + chunk_size]
                for index in range(0, len(bucket_plans), chunk_size)
            ]
            if chunk_size > 0
            else [bucket_plans]
        )
        for group_plans in plan_chunks:
            first_command = [str(item) for item in group_plans[0].get("command", [])]
            command = _command_without_group_specific_options(first_command)
            metrics_json = metrics_root / f"group_{group_index:04d}.json"
            command.extend(["--metrics-json", str(metrics_json)])
            for plan in group_plans:
                plan_command = [str(item) for item in plan.get("command", [])]
                label_ids = _option_values(plan_command, "--label-id")
                if not label_ids:
                    continue
                for label_id in label_ids:
                    command.extend(["--label-id", label_id])
                actor_plan = _single_option_value(plan_command, "--actor-plan-json")
                if actor_plan:
                    for label_id in label_ids:
                        command.extend(["--label-actor-plan-json", f"{label_id}={actor_plan}"])
            grouped_commands.append(command)
            group_index += 1
    return grouped_commands


def _passes_filters(entry: Mapping[str, Any], families: list[str] | None, sources: list[str] | None) -> bool:
    if families and str(entry.get("family")) not in set(families):
        return False
    if sources and str(entry.get("source")) not in set(sources):
        return False
    return True


def _render_entry(args: argparse.Namespace, entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    manifest_path = Path(str(entry["render_manifest_json"]))
    if not manifest_path.is_absolute():
        manifest_path = args.package_root / manifest_path
    output_root = _entry_output_root(args.results_root, entry)
    log_stem = _entry_log_stem(entry)
    plan_json = output_root / "render_plan.json"
    base_cmd = [
        str(args.python_bin),
        "scripts/massgen/render_manifest_jobs.py",
        "--manifest-json",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--write-inputs",
        "--video-backend",
        str(args.video_backend),
        "--device",
        str(args.device),
        "--json",
    ]
    if args.render_script is not None:
        base_cmd.extend(["--render-script", str(args.render_script)])
    if args.minimal_frames is not None and int(args.minimal_frames) > 0:
        base_cmd.extend(["--minimal-frames", str(int(args.minimal_frames))])
    if bool(args.actor_gpu_resident):
        base_cmd.append("--actor-gpu-resident")

    plan_completed, plan_elapsed = _run_capture(
        base_cmd,
        cwd=args.repo_root,
        log_path=args.results_root / "logs" / f"{log_stem}_plan.log",
    )
    try:
        plan_payload = json.loads(plan_completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        plan_payload = {"status": "invalid", "job_count": 0, "error": str(exc)}
    _write_json(plan_json, plan_payload)

    render_elapsed = 0.0
    render_returncode = int(plan_completed.returncode or 2)
    samples: list[dict[str, Any]] = []
    if plan_completed.returncode == 0 and plan_payload.get("status") == "ready":
        render_cmd = [*base_cmd, "--execute"]
        if int(args.workers) > 1:
            render_completed, render_elapsed = _run_capture(
                render_cmd,
                cwd=args.repo_root,
                log_path=args.results_root / "logs" / f"{log_stem}_render.log",
            )
        else:
            with GpuMonitor(float(args.gpu_sample_interval_sec)) as monitor:
                render_completed, render_elapsed = _run_capture(
                    render_cmd,
                    cwd=args.repo_root,
                    log_path=args.results_root / "logs" / f"{log_stem}_render.log",
                )
                samples = monitor.samples
        render_returncode = int(render_completed.returncode)

    metrics = _metric_payloads(output_root)
    videos = sorted(output_root.glob("renders/**/*.mp4"))
    output_bytes = _dir_size_bytes(output_root)
    first_metric = metrics[0] if metrics else {}
    render_overhead = _render_overhead_summary(
        metrics,
        outer_render_elapsed_sec=render_elapsed,
    )
    return {
        "entry_index": int(entry.get("entry_index", index)),
        "family": entry.get("family"),
        "source": entry.get("source"),
        "scene": entry.get("scene"),
        "scenario_id": entry.get("scenario_id"),
        "video_backend": args.video_backend,
        "minimal_frames": args.minimal_frames,
        "expected_renderer_blocked": bool(entry.get("expected_renderer_blocked")),
        "plan_status": plan_payload.get("status"),
        "plan_returncode": int(plan_completed.returncode),
        "plan_elapsed_sec": plan_elapsed,
        "render_returncode": render_returncode,
        "render_elapsed_sec": render_elapsed,
        "status": "success" if render_returncode == 0 and videos else str(plan_payload.get("status")),
        "job_count": plan_payload.get("job_count"),
        "frames_total": render_overhead.get("nested_frames_total") or first_metric.get("frames_total"),
        "duration_total_sec": (
            render_overhead.get("nested_duration_total_sec")
            or first_metric.get("duration_total_sec")
        ),
        "time_per_frame_sec": first_metric.get("time_per_frame_sec"),
        "paths_ok": render_overhead.get("nested_paths_ok") or first_metric.get("paths_ok"),
        "renderer_process_count": render_overhead.get("renderer_process_count"),
        "render_overhead": render_overhead,
        "metrics": metrics,
        "videos": [str(path) for path in videos],
        "output_root": str(output_root),
        "output_bytes": output_bytes,
        "gpu_summary": _gpu_summary(samples),
    }


def _render_group(
    args: argparse.Namespace,
    key: tuple[str, str, str],
    entries: list[Mapping[str, Any]],
    index: int,
    *,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> dict[str, Any]:
    output_root = _group_output_root(args.results_root, key, chunk_index=chunk_index)
    log_stem = _group_log_stem(key, chunk_index=chunk_index)
    plans_root = output_root / "render_plans"
    metrics_root = output_root / "metrics"
    all_plans: list[Mapping[str, Any]] = []
    plan_elapsed = 0.0
    plan_errors: list[str] = []
    plan_statuses: list[str] = []

    for entry_offset, entry in enumerate(entries):
        manifest_path = Path(str(entry["render_manifest_json"]))
        if not manifest_path.is_absolute():
            manifest_path = args.package_root / manifest_path
        t0 = time.perf_counter()
        try:
            manifest = load_render_manifest(manifest_path)
            plan_payload = build_render_plans(
                manifest,
                manifest_path=manifest_path,
                output_root=output_root,
                render_script=args.render_script if args.render_script is not None else REPO_ROOT / "render_label_paths_telesim.py",
                python_bin=str(args.python_bin),
                write_inputs=True,
                video_backend=str(args.video_backend),
                device=str(args.device),
                minimal_frames=args.minimal_frames,
                actor_gpu_resident=bool(args.actor_gpu_resident),
            )
        except Exception as exc:  # pylint: disable=broad-except
            plan_payload = {"status": "invalid", "job_count": 0, "plans": [], "error": str(exc)}
            plan_errors.append(f"{manifest_path}: {exc}")
        plan_elapsed += time.perf_counter() - t0
        plan_statuses.append(str(plan_payload.get("status")))
        plan_path = plans_root / f"{entry_offset:04d}_{Path(str(entry['render_manifest_json'])).stem}.json"
        _write_json(plan_path, plan_payload)
        for plan in plan_payload.get("plans", []):
            if isinstance(plan, Mapping):
                all_plans.append(plan)

    blocked = [plan for plan in all_plans if plan.get("blockers")]
    grouped_commands = (
        []
        if blocked
        else _build_grouped_render_commands(
            all_plans,
            metrics_root=metrics_root,
            max_labels_per_command=args.group_max_labels_per_command,
        )
    )
    render_elapsed = 0.0
    render_returncode = 2 if blocked or plan_errors else 0
    if not blocked and not plan_errors and grouped_commands:
        for command_index, command in enumerate(grouped_commands):
            completed, elapsed = _run_capture(
                command,
                cwd=args.repo_root,
                log_path=args.results_root / "logs" / f"{log_stem}_group_{command_index:04d}.log",
            )
            render_elapsed += elapsed
            render_returncode = int(completed.returncode)
            if render_returncode != 0:
                break

    metrics = _metric_payloads(output_root)
    videos = sorted(output_root.glob("renders/**/*.mp4"))
    output_bytes = _dir_size_bytes(output_root)
    render_overhead = _render_overhead_summary(
        metrics,
        outer_render_elapsed_sec=render_elapsed,
    )
    family, source, scene = key
    status = "success" if render_returncode == 0 and videos else "failed"
    if blocked:
        status = "blocked"
    elif plan_errors:
        status = "invalid"
    return {
        "entry_index": int(index),
        "record_type": "grouped_same_scene",
        "entry_count": len(entries),
        "family": family,
        "source": source,
        "scene": scene,
        "scenario_id": None,
        "video_backend": args.video_backend,
        "minimal_frames": args.minimal_frames,
        "expected_renderer_blocked": False,
        "plan_status": "ready" if all(status == "ready" for status in plan_statuses) else ",".join(sorted(set(plan_statuses))),
        "plan_returncode": 0 if not plan_errors else 2,
        "plan_elapsed_sec": plan_elapsed,
        "plan_errors": plan_errors,
        "blocked_plan_count": len(blocked),
        "render_returncode": render_returncode,
        "render_elapsed_sec": render_elapsed,
        "status": status,
        "job_count": len(all_plans),
        "grouped_command_count": len(grouped_commands),
        "group_max_labels_per_command": int(args.group_max_labels_per_command or 0),
        "group_max_manifests_per_task": int(args.group_max_manifests_per_task or 0),
        "group_chunk_index": chunk_index,
        "group_chunk_count": chunk_count,
        "frames_total": render_overhead.get("nested_frames_total"),
        "duration_total_sec": render_overhead.get("nested_duration_total_sec"),
        "time_per_frame_sec": (
            float(render_overhead.get("nested_duration_total_sec") or 0.0)
            / float(render_overhead.get("nested_frames_total") or 1)
            if render_overhead.get("nested_frames_total")
            else None
        ),
        "paths_ok": render_overhead.get("nested_paths_ok"),
        "renderer_process_count": render_overhead.get("renderer_process_count"),
        "render_overhead": render_overhead,
        "metrics": metrics,
        "videos": [str(path) for path in videos],
        "output_root": str(output_root),
        "output_bytes": output_bytes,
        "gpu_summary": {},
    }


def _group_entry_tasks(
    grouped: Mapping[tuple[str, str, str], list[Mapping[str, Any]]],
    *,
    max_manifests_per_task: int,
) -> list[tuple[tuple[str, str, str], list[Mapping[str, Any]], int, int | None, int | None]]:
    tasks: list[tuple[tuple[str, str, str], list[Mapping[str, Any]], int, int | None, int | None]] = []
    chunk_size = int(max_manifests_per_task or 0)
    for key, group_entries in grouped.items():
        if chunk_size <= 0:
            tasks.append((key, group_entries, len(tasks), None, None))
            continue
        chunks = [
            group_entries[index : index + chunk_size]
            for index in range(0, len(group_entries), chunk_size)
        ]
        chunk_count = len(chunks)
        for chunk_index, chunk_entries in enumerate(chunks):
            tasks.append((key, chunk_entries, len(tasks), chunk_index, chunk_count))
    return tasks


def main() -> int:
    args = _parse_args()
    args.package_root = args.package_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    if args.render_script is not None:
        args.render_script = args.render_script.expanduser().resolve()
    if args.clean and args.results_root.exists():
        shutil.rmtree(args.results_root)
    args.results_root.mkdir(parents=True, exist_ok=True)

    package_index = _load_json(args.package_root / "smoketest_package_index.json")
    entries = [entry for entry in package_index.get("entries", []) if isinstance(entry, Mapping)]
    selected: list[Mapping[str, Any]] = []
    per_group_counts: dict[tuple[str, str, str], int] = {}
    for entry in entries:
        if not _passes_filters(entry, args.family, args.source):
            continue
        if bool(args.skip_expected_blocked) and bool(entry.get("expected_renderer_blocked")):
            continue
        group_key = (
            str(entry.get("family")),
            str(entry.get("source")),
            str(entry.get("scene")),
        )
        if int(args.renders_per_family_source_scene) > 0:
            group_count = per_group_counts.get(group_key, 0)
            if group_count >= int(args.renders_per_family_source_scene):
                continue
            per_group_counts[group_key] = group_count + 1
        selected.append(entry)
        if int(args.max_renders) > 0 and len(selected) >= int(args.max_renders):
            break

    summary: dict[str, Any] = {
        "schema_version": "navdp_massgen_render_smoketest_benchmark.v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "package_root": str(args.package_root),
        "results_root": str(args.results_root),
        "repo_root": str(args.repo_root),
        "python_bin": str(args.python_bin),
        "render_script": str(args.render_script) if args.render_script is not None else None,
        "video_backend": str(args.video_backend),
        "device": str(args.device),
        "minimal_frames": args.minimal_frames,
        "max_renders": int(args.max_renders),
        "renders_per_family_source_scene": int(args.renders_per_family_source_scene),
        "workers": int(args.workers),
        "skip_expected_blocked": bool(args.skip_expected_blocked),
        "actor_gpu_resident": bool(args.actor_gpu_resident),
        "execution_mode": "grouped_same_scene" if bool(args.group_same_scene) else "per_manifest",
        "group_max_labels_per_command": int(args.group_max_labels_per_command),
        "group_max_manifests_per_task": int(args.group_max_manifests_per_task),
        "selected_count": len(selected),
        "records": [],
    }
    jsonl_path = args.results_root / "render_records.jsonl"
    gpu_samples_path = args.results_root / "gpu_samples.jsonl"
    benchmark_t0 = time.perf_counter()
    with GpuMonitor(float(args.gpu_sample_interval_sec), log_path=gpu_samples_path) as run_monitor:
        if bool(args.group_same_scene):
            grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
            for entry in selected:
                grouped.setdefault(_entry_group_key(entry), []).append(entry)
            group_items = list(grouped.items())
            group_tasks = _group_entry_tasks(
                grouped,
                max_manifests_per_task=int(args.group_max_manifests_per_task),
            )
            summary["scene_group_count"] = len(group_items)
            summary["execution_record_count"] = len(group_tasks)
            if int(args.workers) <= 1:
                for completed_count, (key, group_entries, index, chunk_index, chunk_count) in enumerate(group_tasks, start=1):
                    record = _render_group(
                        args,
                        key,
                        group_entries,
                        index,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )
                    summary["records"].append(record)
                    with jsonl_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(
                        f"{completed_count}/{len(group_tasks)} group={record['family']} "
                        f"{record['source']} {record['scene']} entries={record['entry_count']} "
                        f"chunk={record.get('group_chunk_index')} "
                        f"status={record['status']} frames={record.get('frames_total')} "
                        f"render={record['render_elapsed_sec']:.2f}s bytes={record['output_bytes']}",
                        flush=True,
                    )
            else:
                with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
                    futures = {
                        pool.submit(
                            _render_group,
                            args,
                            key,
                            group_entries,
                            index,
                            chunk_index=chunk_index,
                            chunk_count=chunk_count,
                        ): index
                        for key, group_entries, index, chunk_index, chunk_count in group_tasks
                    }
                    completed_count = 0
                    for future in as_completed(futures):
                        index = futures[future]
                        record = future.result()
                        completed_count += 1
                        summary["records"].append(record)
                        with jsonl_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                        print(
                            f"{completed_count}/{len(group_tasks)} group={index} "
                            f"{record['family']} {record['source']} {record['scene']} "
                            f"entries={record['entry_count']} chunk={record.get('group_chunk_index')} "
                            f"status={record['status']} "
                            f"frames={record.get('frames_total')} render={record['render_elapsed_sec']:.2f}s "
                            f"bytes={record['output_bytes']}",
                            flush=True,
                        )
        elif int(args.workers) <= 1:
            for index, entry in enumerate(selected):
                record = _render_entry(args, entry, index)
                summary["records"].append(record)
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"{index + 1}/{len(selected)} {record['family']} {record['source']} {record['scene']} "
                    f"status={record['status']} frames={record.get('frames_total')} "
                    f"render={record['render_elapsed_sec']:.2f}s bytes={record['output_bytes']}",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
                futures = {
                    pool.submit(_render_entry, args, entry, index): index
                    for index, entry in enumerate(selected)
                }
                completed_count = 0
                for future in as_completed(futures):
                    index = futures[future]
                    record = future.result()
                    completed_count += 1
                    summary["records"].append(record)
                    with jsonl_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(
                        f"{completed_count}/{len(selected)} entry={index} {record['family']} "
                        f"{record['source']} {record['scene']} status={record['status']} "
                        f"frames={record.get('frames_total')} render={record['render_elapsed_sec']:.2f}s "
                        f"bytes={record['output_bytes']}",
                        flush=True,
                    )
        summary["gpu_samples_jsonl"] = str(gpu_samples_path)
        summary["gpu_run_summary"] = _gpu_summary(run_monitor.samples)
    summary["benchmark_wall_sec"] = time.perf_counter() - benchmark_t0

    success = [record for record in summary["records"] if record.get("status") == "success"]
    total_frames = sum(int(record.get("frames_total") or 0) for record in success)
    total_render_sec = sum(float(record.get("render_elapsed_sec") or 0.0) for record in success)
    total_output_bytes = sum(int(record.get("output_bytes") or 0) for record in summary["records"])
    summary["record_count"] = len(summary["records"])
    summary["success_count"] = len(success)
    summary["total_frames"] = total_frames
    summary["total_render_sec"] = total_render_sec
    summary["total_output_bytes"] = total_output_bytes
    summary["avg_time_per_frame_sec"] = total_render_sec / total_frames if total_frames else None
    summary["render_overhead_summary"] = _summarize_render_overhead(summary["records"])
    summary["status"] = "success" if success else "failed"
    _write_json(args.results_root / "benchmark_summary.json", summary)
    print(
        f"summary_status {summary['status']} success={summary['success_count']}/{summary['record_count']} "
        f"frames={total_frames} render_sec={total_render_sec:.2f} bytes={total_output_bytes}",
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
