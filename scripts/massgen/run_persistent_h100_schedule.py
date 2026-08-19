#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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

_SMOKE_MODULE_PATH = REPO_ROOT / "scripts" / "massgen" / "run_render_smoketest_benchmark.py"
_SMOKE_SPEC = importlib.util.spec_from_file_location(
    "navdp_run_render_smoketest_benchmark",
    _SMOKE_MODULE_PATH,
)
if _SMOKE_SPEC is None or _SMOKE_SPEC.loader is None:
    raise ImportError(f"Unable to import smoke benchmark helpers from {_SMOKE_MODULE_PATH}")
smoke = importlib.util.module_from_spec(_SMOKE_SPEC)
_SMOKE_SPEC.loader.exec_module(smoke)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute an h100_persistent_schedule.v1 JSON as a Phase-A "
            "persistent-worker bridge. Chunks are scene/resource ordered and "
            "run through the existing grouped renderer path."
        )
    )
    parser.add_argument("--schedule-json", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of GPU assignments to execute concurrently.",
    )
    parser.add_argument(
        "--group-max-labels-per-command",
        type=int,
        default=0,
        help="Split compatible labels inside a chunk into renderer commands of this size.",
    )
    parser.add_argument("--gpu-sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--preemptible-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write summary records without executing renderer commands.",
    )
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


def _run_capture_env(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str],
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, **dict(env)},
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "".join(
            [
                f"command: {' '.join(cmd)}\n",
                f"started_at: {started}\n",
                f"wall_time_sec: {elapsed:.6f}\n",
                f"returncode: {completed.returncode}\n",
                f"env_overrides: {json.dumps(dict(env), sort_keys=True)}\n",
                "\n--- stdout ---\n",
                completed.stdout or "",
                "\n--- stderr ---\n",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return completed, elapsed


def _chunk_output_root(results_root: Path, gpu_id: str, chunk: Mapping[str, Any]) -> Path:
    return (
        results_root
        / "persistent_schedule_renders"
        / f"gpu_{smoke._safe_component(gpu_id)}"
        / smoke._safe_component(str(chunk.get("scene_id") or "scene"))
        / smoke._safe_component(str(chunk.get("chunk_id") or "chunk"))
    )


def _chunk_log_stem(gpu_id: str, chunk: Mapping[str, Any]) -> Path:
    return (
        Path("persistent_schedule")
        / f"gpu_{smoke._safe_component(gpu_id)}"
        / smoke._safe_component(str(chunk.get("scene_id") or "scene"))
        / smoke._safe_component(str(chunk.get("chunk_id") or "chunk"))
    )


def _plans_from_chunk(chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    plans = chunk.get("plans")
    if isinstance(plans, list) and plans:
        return [dict(plan) for plan in plans if isinstance(plan, Mapping)]

    out: list[dict[str, Any]] = []
    for item in chunk.get("work_items", []):
        if not isinstance(item, Mapping):
            continue
        out.append(
            {
                "status": "ready",
                "job_id": item.get("job_id"),
                "scene_id": item.get("scene_id"),
                "command": list(item.get("command") or []),
                "env": dict(item.get("env") or {}),
                "label_path": item.get("label_path"),
                "actor_plan_path": item.get("actor_plan_path"),
                "human_actor_ids": list(item.get("human_actor_ids") or []),
                "peer_robot_ids": list(item.get("peer_robot_ids") or []),
                "mission_families": list(item.get("mission_families") or []),
                "blockers": [],
                "robot_overlay_commands": [],
            }
        )
    return out


def _rewrite_plan_for_chunk(plan: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    job_id = str(plan.get("job_id") or "job")
    command = [str(item) for item in plan.get("command", [])]
    command = smoke._replace_command_option(command, "--output-dir", [str(output_root / "renders")])
    command = smoke._replace_command_option(
        command,
        "--metrics-json",
        [str(output_root / "metrics" / f"{smoke._safe_component(job_id)}.json")],
    )
    rewritten = dict(plan)
    rewritten["command"] = command
    outputs = dict(rewritten.get("outputs") or {})
    outputs["render_dir"] = str(output_root / "renders" / str(plan.get("scene_id") or ""))
    outputs["metrics_json"] = str(output_root / "metrics" / f"{smoke._safe_component(job_id)}.json")
    rewritten["outputs"] = outputs
    return rewritten


def _chunk_env(gpu_id: str, plans: list[Mapping[str, Any]]) -> dict[str, str]:
    env: dict[str, str] = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    for plan in plans:
        plan_env = plan.get("env")
        if isinstance(plan_env, Mapping):
            env.update({str(key): str(value) for key, value in plan_env.items()})
    return env


def _render_chunk(
    args: argparse.Namespace,
    *,
    gpu_id: str,
    chunk: Mapping[str, Any],
    assignment_index: int,
    chunk_index: int,
) -> dict[str, Any]:
    final_output_root = _chunk_output_root(args.results_root, gpu_id, chunk)
    output_root, done_record = smoke._prepare_task_output_root(
        final_output_root,
        preemptible=bool(args.preemptible_output),
        resume=bool(args.resume),
    )
    if done_record is not None:
        return done_record

    raw_plans = _plans_from_chunk(chunk)
    plans = [_rewrite_plan_for_chunk(plan, output_root=output_root) for plan in raw_plans]
    blockers = [
        f"{plan.get('job_id')}: {blocker}"
        for plan in plans
        for blocker in (plan.get("blockers") or [])
    ]
    overlay_plans = [
        str(plan.get("job_id"))
        for plan in plans
        if plan.get("robot_overlay_commands")
    ]
    if overlay_plans:
        blockers.append(
            "robot overlay plans are not supported by the Phase-A schedule runner: "
            + ", ".join(overlay_plans)
        )

    grouped_commands = (
        []
        if blockers
        else smoke._build_grouped_render_commands(
            plans,
            metrics_root=output_root / "metrics",
            max_labels_per_command=int(args.group_max_labels_per_command),
        )
    )

    render_elapsed = 0.0
    render_returncode = 2 if blockers else 0
    env = _chunk_env(gpu_id, plans)
    if not blockers and grouped_commands and not bool(args.dry_run):
        for command_index, command in enumerate(grouped_commands):
            completed, elapsed = _run_capture_env(
                command,
                cwd=args.repo_root,
                env=env,
                log_path=args.results_root
                / "logs"
                / f"{_chunk_log_stem(gpu_id, chunk)}_cmd_{command_index:04d}.log",
            )
            render_elapsed += elapsed
            render_returncode = int(completed.returncode)
            if render_returncode != 0:
                break

    metrics = smoke._metric_payloads(output_root)
    videos = sorted(output_root.glob("renders/**/*.mp4"))
    render_overhead = smoke._render_overhead_summary(
        metrics,
        outer_render_elapsed_sec=render_elapsed,
    )
    output_bytes = smoke._dir_size_bytes(output_root)
    status = "success" if render_returncode == 0 and (videos or bool(args.dry_run)) else "failed"
    if blockers:
        status = "blocked"
    record: dict[str, Any] = {
        "assignment_index": int(assignment_index),
        "chunk_index": int(chunk_index),
        "record_type": "persistent_schedule_chunk",
        "gpu_id": str(gpu_id),
        "chunk_id": chunk.get("chunk_id"),
        "scene": chunk.get("scene_id"),
        "job_ids": list(chunk.get("job_ids") or []),
        "job_count": len(plans),
        "grouped_command_count": len(grouped_commands),
        "group_max_labels_per_command": int(args.group_max_labels_per_command or 0),
        "estimated_vram_bytes": chunk.get("estimated_vram_bytes"),
        "estimated_ram_bytes": chunk.get("estimated_ram_bytes"),
        "frame_count_hint": chunk.get("frame_count_hint"),
        "render_returncode": render_returncode,
        "render_elapsed_sec": render_elapsed,
        "status": status,
        "blockers": blockers,
        "frames_total": render_overhead.get("nested_frames_total"),
        "duration_total_sec": render_overhead.get("nested_duration_total_sec"),
        "paths_ok": render_overhead.get("nested_paths_ok"),
        "renderer_process_count": render_overhead.get("renderer_process_count"),
        "render_overhead": render_overhead,
        "metrics": metrics,
        "videos": [str(path) for path in videos],
        "output_root": str(output_root),
        "final_output_root": str(final_output_root),
        "preemptible_output": bool(args.preemptible_output),
        "output_bytes": output_bytes,
    }
    if status == "success":
        committed_root = smoke._commit_task_output_root(
            work_root=output_root,
            final_root=final_output_root,
            record=record,
            preemptible=bool(args.preemptible_output),
        )
        record["output_root"] = str(committed_root)
        record["final_output_root"] = str(committed_root)
        if bool(args.preemptible_output):
            marker_record = smoke._load_done_record(committed_root)
            if marker_record is not None:
                marker_record["skipped_existing"] = False
                record.update(marker_record)
    return record


def _run_assignment(
    args: argparse.Namespace,
    assignment: Mapping[str, Any],
    *,
    assignment_index: int,
    jsonl_lock: threading.Lock,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    gpu_id = str(assignment.get("gpu_id") or assignment_index)
    chunks = [chunk for chunk in assignment.get("chunks", []) if isinstance(chunk, Mapping)]
    records: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        record = _render_chunk(
            args,
            gpu_id=gpu_id,
            chunk=chunk,
            assignment_index=assignment_index,
            chunk_index=chunk_index,
        )
        records.append(record)
        with jsonl_lock:
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"gpu={gpu_id} chunk={record.get('chunk_id')} status={record['status']} "
            f"jobs={record.get('job_count')} frames={record.get('frames_total')} "
            f"render={float(record.get('render_elapsed_sec') or 0.0):.2f}s "
            f"bytes={record.get('output_bytes')}",
            flush=True,
        )
    return records


def main() -> int:
    args = _parse_args()
    args.schedule_json = args.schedule_json.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.clean and args.results_root.exists():
        shutil.rmtree(args.results_root)
    args.results_root.mkdir(parents=True, exist_ok=True)
    stale_tmp_removed = (
        smoke._cleanup_stale_tmp_outputs(args.results_root)
        if bool(args.preemptible_output) and bool(args.resume)
        else 0
    )

    schedule = _load_json(args.schedule_json)
    if schedule.get("schema_version") != "h100_persistent_schedule.v1":
        print(f"Unsupported schedule schema: {schedule.get('schema_version')}", file=sys.stderr)
        return 2
    assignments = [
        assignment
        for assignment in schedule.get("assignments", [])
        if isinstance(assignment, Mapping)
    ]
    if not assignments:
        print("Schedule has no assignments.", file=sys.stderr)
        return 2

    summary: dict[str, Any] = {
        "schema_version": "navdp_massgen_persistent_h100_schedule_run.v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "schedule_json": str(args.schedule_json),
        "schedule_work_item_count": schedule.get("work_item_count"),
        "schedule_chunk_count": schedule.get("chunk_count"),
        "includes_execution": bool(schedule.get("includes_execution")),
        "results_root": str(args.results_root),
        "repo_root": str(args.repo_root),
        "workers": int(args.workers),
        "group_max_labels_per_command": int(args.group_max_labels_per_command),
        "preemptible_output": bool(args.preemptible_output),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "stale_tmp_removed": stale_tmp_removed,
        "records": [],
    }
    jsonl_path = args.results_root / "render_records.jsonl"
    gpu_samples_path = args.results_root / "gpu_samples.jsonl"
    jsonl_lock = threading.Lock()
    benchmark_t0 = time.perf_counter()
    with smoke.GpuMonitor(float(args.gpu_sample_interval_sec), log_path=gpu_samples_path) as monitor:
        if int(args.workers) <= 1 or len(assignments) <= 1:
            for assignment_index, assignment in enumerate(assignments):
                summary["records"].extend(
                    _run_assignment(
                        args,
                        assignment,
                        assignment_index=assignment_index,
                        jsonl_lock=jsonl_lock,
                        jsonl_path=jsonl_path,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=min(int(args.workers), len(assignments))) as pool:
                futures = {
                    pool.submit(
                        _run_assignment,
                        args,
                        assignment,
                        assignment_index=assignment_index,
                        jsonl_lock=jsonl_lock,
                        jsonl_path=jsonl_path,
                    ): assignment_index
                    for assignment_index, assignment in enumerate(assignments)
                }
                for future in as_completed(futures):
                    summary["records"].extend(future.result())
        summary["gpu_samples"] = monitor.samples

    summary["benchmark_wall_sec"] = time.perf_counter() - benchmark_t0
    success_records = [
        record for record in summary["records"] if record.get("status") == "success"
    ]
    summary["status"] = (
        "success"
        if len(success_records) == len(summary["records"]) and summary["records"]
        else "failed"
    )
    summary["success_count"] = len(success_records)
    summary["record_count"] = len(summary["records"])
    summary["total_render_sec"] = sum(
        float(record.get("render_elapsed_sec") or 0.0) for record in success_records
    )
    summary["total_output_bytes"] = sum(
        int(record.get("output_bytes") or 0) for record in success_records
    )
    summary["total_frames"] = sum(
        int(record.get("frames_total") or 0) for record in success_records
    )
    summary["gpu_run_summary"] = smoke._gpu_summary(summary["gpu_samples"])
    summary["render_overhead_summary"] = smoke._summarize_render_overhead(summary["records"])
    _write_json(args.results_root / "benchmark_summary.json", summary)
    print(
        f"summary_status {summary['status']} success={summary['success_count']}/{summary['record_count']} "
        f"frames={summary['total_frames']} render_sec={summary['total_render_sec']:.2f} "
        f"bytes={summary['total_output_bytes']}",
        flush=True,
    )
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
