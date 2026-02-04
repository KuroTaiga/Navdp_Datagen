#!/usr/bin/env python3
"""Parallel dispatcher for render_label_paths_telesim.py.

This is a lightweight TeleSim3D-backed alternative to parallel_render_paths.py.
It mirrors a subset of the CLI to keep existing entry scripts working, while
ignoring unsupported features with warnings.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

STATUS_NOT_RUN = 0
STATUS_DONE = 1
STATUS_RETRY = 2
STATUS_SKIP = 3


def _resolve_label_directory(scene_task_dir: Path) -> Path | None:
    label_paths_dir = scene_task_dir / "label_paths"
    if label_paths_dir.is_dir() and any(label_paths_dir.glob("*.json")):
        return label_paths_dir
    if scene_task_dir.is_dir() and any(scene_task_dir.glob("*.json")):
        return scene_task_dir
    return None


def _count_planned_labels(tasks_dir: Path | None, scene_id: str, label_ids: list[str] | None) -> int | None:
    if label_ids:
        return len(label_ids)
    if tasks_dir is None:
        return None
    scene_dir = tasks_dir / scene_id
    label_dir = _resolve_label_directory(scene_dir)
    if label_dir is None:
        return None
    json_paths = [
        p
        for p in label_dir.glob("*.json")
        if not p.name.endswith("_detailed.json") and p.name != "summary.json"
    ]
    return len(json_paths)


def _extract_rendered_labels(metrics_path: Path | None) -> int | None:
    if metrics_path is None or not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload.get("paths_total"), int):
        return int(payload["paths_total"])
    if isinstance(payload.get("paths"), list):
        return len(payload["paths"])
    return None


def _discover_scenes(tasks_dir: Path) -> list[str]:
    if not tasks_dir.is_dir():
        return []
    return [p.name for p in sorted(tasks_dir.iterdir()) if p.is_dir()]


def _build_command(
    *,
    render_script: Path,
    scenes_dir: Path,
    tasks_dir: Path,
    scene_id: str,
    output_dir: Path,
    metrics_path: Path | None,
    extra_args: Iterable[str],
    minimal_frames: int | None,
    exclude_detailed: bool | None,
    max_labels: int | None,
    label_ids: list[str] | None,
    actor_args: list[str] | None,
    skip_completed_log: Path | None,
    resume: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(render_script),
        "--scenes-dir",
        str(scenes_dir),
        "--tasks-dir",
        str(tasks_dir),
        "--scene",
        scene_id,
        "--output-dir",
        str(output_dir),
    ]
    if metrics_path is not None:
        cmd.extend(["--metrics-json", str(metrics_path)])
    if minimal_frames is not None:
        cmd.extend(["--minimal-frames", str(minimal_frames)])
    if exclude_detailed is not None:
        cmd.append("--exclude-detailed-labels" if exclude_detailed else "--no-exclude-detailed-labels")
    if max_labels is not None:
        cmd.extend(["--max-labels", str(max_labels)])
    if label_ids:
        for label_id in label_ids:
            cmd.extend(["--label-id", str(label_id)])
    if actor_args:
        cmd.extend(actor_args)
    if skip_completed_log is not None:
        cmd.extend(["--skip-completed-log", str(skip_completed_log)])
    if resume:
        cmd.append("--resume")
    cmd.extend(list(extra_args))
    return cmd


def _write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_status_map(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_status_entries(status_map: dict, entries: list[dict]) -> dict:
    for entry in entries:
        scene = str(entry.get("scene_id") or entry.get("scene") or "")
        label = str(entry.get("label_id") or entry.get("label") or "")
        if not scene or not label:
            continue
        scene_map = status_map.setdefault(scene, {})
        scene_map[label] = {
            "status": int(entry.get("status", STATUS_NOT_RUN)),
            "error": entry.get("error"),
        }
    return status_map


def _load_assignment_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_actor_jobs(manifest: dict) -> list[dict]:
    actors = {entry["id"]: entry for entry in (manifest.get("actors") or [])}
    jobs: dict[tuple[str, str], dict] = {}
    for assign in manifest.get("assignments") or []:
        scene = str(assign.get("scene"))
        label = str(assign.get("label"))
        actor_id = str(assign.get("actor_id"))
        actor_entry = actors.get(actor_id) or {}
        actor_dir = assign.get("actor_dir") or actor_entry.get("directory")
        if not scene or not label or not actor_id or not actor_dir:
            continue
        key = (scene, actor_id)
        if key not in jobs:
            jobs[key] = {
                "scene": scene,
                "actor_id": actor_id,
                "actor_dir": actor_dir,
                "labels": [],
                "actor": actor_entry,
                "assignment": assign,
            }
        jobs[key]["labels"].append(label)
    return list(jobs.values())


def _actor_args_from_job(job: dict) -> list[str]:
    actor = job.get("actor") or {}
    assign = job.get("assignment") or {}
    args = [
        "--actor-seq-dir",
        str(job["actor_dir"]),
        "--actor-pattern",
        str(actor.get("pattern") or "*.ply"),
        "--actor-height",
        str(actor.get("height") or 1.7),
        "--actor-foot-offset",
        str(assign.get("actor_foot_offset") or actor.get("foot_offset") or 0.0),
        "--actor-speed",
        str(actor.get("speed") or 1.3),
        "--actor-fps",
        str(actor.get("fps") or 10.0),
        "--follow-distance",
        str(actor.get("follow_distance") or 1.5),
        "--follow-buffer",
        str(actor.get("follow_buffer") or 0.0),
        "--animation-cycle-mod",
        str(actor.get("animation_cycle_mod") or 3),
        "--job-actor-id",
        str(job.get("actor_id")),
    ]
    if actor.get("loop") is False:
        args.append("--actor-no-loop")
    return args


def _run_command(cmd: list[str], procs: list[subprocess.Popen]) -> int:
    proc = subprocess.Popen(cmd, start_new_session=True)
    procs.append(proc)
    return proc.wait()


def _terminate_processes(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
    time.sleep(1.0)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel TeleSim3D label rendering dispatcher.")
    root_dir = Path(__file__).absolute().parent
    parser.add_argument("--render-script", type=Path, default=root_dir / "render_label_paths_telesim.py")
    parser.add_argument("--scenes-dir", type=Path, default=root_dir / "data" / "scenes")
    parser.add_argument("--tasks-dir", type=Path, default=root_dir / "data" / "tasks")
    parser.add_argument("--output-dir", type=Path, default=root_dir / "data" / "tmp" / "test_telesim3d")
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--render-extra-args", action="append", default=[])
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--per-job-metrics-dir", type=Path, default=None)
    parser.add_argument("--progress-json", type=Path, default=None)
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument("--error-log", type=Path, default=None)
    parser.add_argument("--minimal-frames", type=int, default=None)
    parser.add_argument("--exclude-detailed-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--label-id", action="append", default=None)
    parser.add_argument("--assignment-manifest", type=Path, default=None)
    parser.add_argument("--fpv-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fpv-follow-distance", type=float, default=None)
    parser.add_argument("--skip-completed-log", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retry-cuda-oom", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-oom-retry-delay", type=float, default=None)
    parser.add_argument("--cuda-oom-max-retries", type=int, default=None)
    parser.add_argument("--worker-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--job-slot", type=int, default=None)
    parser.add_argument("--job-name", default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unsupported args: {' '.join(unknown)}", file=sys.stderr)

    jobs: list[dict] = []
    if args.assignment_manifest is not None and args.assignment_manifest.is_file():
        manifest = _load_assignment_manifest(args.assignment_manifest)
        jobs = _build_actor_jobs(manifest)
    else:
        scenes = args.scene or _discover_scenes(args.tasks_dir)
        if not scenes:
            print(f"[ERROR] No scenes found under {args.tasks_dir}", file=sys.stderr)
            return 1
        for scene_id in scenes:
            jobs.append(
                {
                    "scene": scene_id,
                    "labels": args.label_id,
                    "actor_args": None,
                }
            )

    extra_args: list[str] = []
    for snippet in args.render_extra_args:
        if snippet:
            extra_args.extend(shlex.split(snippet))

    # Planned label counts per job/scene to report progress.
    scene_planned: dict[str, int] = {}
    total_planned = 0
    for job in jobs:
        planned = _count_planned_labels(args.tasks_dir, job["scene"], job.get("labels"))
        job["planned_labels"] = planned
        if planned is not None:
            scene_planned[job["scene"]] = scene_planned.get(job["scene"], 0) + planned
            total_planned += planned

    total_jobs = len(jobs)
    completed_jobs = 0
    scene_done: dict[str, int] = {}
    total_done = 0
    results: list[dict] = []

    def _update_progress(last_scene: str | None = None) -> None:
        payload = {
            "total": total_jobs,
            "completed": min(completed_jobs, total_jobs),
            "last_scene": last_scene,
            "timestamp": time.time(),
        }
        _write_json(args.progress_json, payload)

    def _update_status_from_metrics(metrics_path: Path | None) -> None:
        if args.status_json is None or metrics_path is None or not metrics_path.is_file():
            return
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        entries = payload.get("path_statuses") or []
        if not entries:
            return
        status_map = _load_status_map(args.status_json)
        status_map = _merge_status_entries(status_map, entries)
        _write_json(args.status_json, status_map)

    procs: list[subprocess.Popen] = []

    def _print_plans() -> None:
        if not scene_planned:
            return
        print("[PLAN] Planned labels per scene:")
        for scene_id, planned in sorted(scene_planned.items()):
            print(f"  - {scene_id}: {planned} planned")
        if total_planned > 0:
            print(f"[PLAN] Overall planned labels: {total_planned}")

    def _note_completion(
        scene_id: str,
        metrics_path: Path | None,
        planned_labels: int | None,
        *,
        count_job: bool = True,
    ) -> None:
        nonlocal completed_jobs, total_done
        if count_job:
            completed_jobs += 1
        rendered = _extract_rendered_labels(metrics_path)
        if rendered is None:
            rendered = planned_labels if planned_labels is not None else 0
        scene_done[scene_id] = scene_done.get(scene_id, 0) + rendered
        total_done += rendered
        scene_total = scene_planned.get(scene_id)
        scene_progress = (
            f"{scene_done[scene_id]}/{scene_total}" if scene_total not in (None, 0) else f"{scene_done[scene_id]}"
        )
        overall_progress = (
            f"{total_done}/{total_planned}" if total_planned not in (None, 0) else f"{total_done}"
        )
        print(f"[PROGRESS] Scene {scene_id}: {scene_progress} | Overall: {overall_progress}")
        _update_progress(scene_id)

    def _handle_terminate(_signum: int, _frame: object | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_terminate)
    signal.signal(signal.SIGTERM, _handle_terminate)
    try:
        _print_plans()
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {}
            for job in jobs:
                scene_id = job["scene"]
                metrics_path = None
                if args.per_job_metrics_dir is not None:
                    metrics_path = args.per_job_metrics_dir / f"{scene_id}_{job.get('actor_id','base')}_metrics.json"
                elif args.metrics_json is not None and len(jobs) == 1:
                    metrics_path = args.metrics_json
                actor_args = job.get("actor_args")
                if actor_args is None and job.get("actor_id"):
                    actor_args = _actor_args_from_job(job)
                cmd = _build_command(
                    render_script=args.render_script,
                    scenes_dir=args.scenes_dir,
                    tasks_dir=args.tasks_dir,
                    scene_id=scene_id,
                    output_dir=args.output_dir,
                    metrics_path=metrics_path,
                    extra_args=extra_args,
                    minimal_frames=args.minimal_frames,
                    exclude_detailed=args.exclude_detailed_labels,
                    max_labels=args.max_labels,
                    label_ids=job.get("labels") or args.label_id,
                    actor_args=actor_args,
                    skip_completed_log=args.skip_completed_log,
                    resume=bool(args.resume),
                )
                futures[executor.submit(_run_command, cmd, procs)] = {
                    "scene": scene_id,
                    "cmd": cmd,
                    "metrics_path": metrics_path,
                    "planned_labels": job.get("planned_labels"),
                }

            for future in as_completed(futures):
                info = futures[future]
                returncode = future.result()
                results.append(
                    {
                        "scene": info["scene"],
                        "returncode": returncode,
                        "cmd": info["cmd"],
                    }
                )
                _note_completion(info["scene"], info.get("metrics_path"), info.get("planned_labels"))
                _update_status_from_metrics(info.get("metrics_path"))
    except KeyboardInterrupt:
        _terminate_processes(procs)
        raise

    if args.retry_cuda_oom and args.status_json is not None:
        retry_delay = float(args.cuda_oom_retry_delay or 0.0)
        remaining = args.cuda_oom_max_retries if args.cuda_oom_max_retries is not None else -1
        retry_pass = 0
        try:
            while remaining != 0:
                status_map = _load_status_map(args.status_json)
                retry_jobs: list[dict] = []
                for job in jobs:
                    labels = job.get("labels") or []
                    if not labels:
                        continue
                    pending = [
                        label for label in labels
                        if status_map.get(job["scene"], {}).get(label, {}).get("status") == STATUS_RETRY
                    ]
                    if pending:
                        retry_job = dict(job)
                        retry_job["labels"] = pending
                        retry_jobs.append(retry_job)
                if not retry_jobs:
                    break
                retry_pass += 1
                if retry_delay > 0:
                    time.sleep(retry_delay)
                retry_results: list[dict] = []
                with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
                    futures = {}
                    for job in retry_jobs:
                        scene_id = job["scene"]
                        metrics_path = None
                        if args.per_job_metrics_dir is not None:
                            metrics_path = args.per_job_metrics_dir / f"{scene_id}_{job.get('actor_id','base')}_oom{retry_pass}.json"
                        cmd = _build_command(
                            render_script=args.render_script,
                            scenes_dir=args.scenes_dir,
                            tasks_dir=args.tasks_dir,
                            scene_id=scene_id,
                            output_dir=args.output_dir,
                            metrics_path=metrics_path,
                            extra_args=extra_args,
                            minimal_frames=args.minimal_frames,
                            exclude_detailed=args.exclude_detailed_labels,
                            max_labels=args.max_labels,
                            label_ids=job.get("labels"),
                            actor_args=job.get("actor_args"),
                            skip_completed_log=args.skip_completed_log,
                            resume=bool(args.resume),
                        )
                        futures[executor.submit(_run_command, cmd, procs)] = {
                            "scene": scene_id,
                            "cmd": cmd,
                            "metrics_path": metrics_path,
                            "planned_labels": job.get("planned_labels"),
                        }
                    for future in as_completed(futures):
                        info = futures[future]
                        returncode = future.result()
                        retry_results.append(
                            {"scene": info["scene"], "returncode": returncode, "cmd": info["cmd"]}
                        )
                        _note_completion(
                            info["scene"],
                            info.get("metrics_path"),
                            info.get("planned_labels"),
                            count_job=False,
                        )
                        _update_status_from_metrics(info.get("metrics_path"))
                results.extend(retry_results)
                if remaining > 0:
                    remaining -= 1
        except KeyboardInterrupt:
            _terminate_processes(procs)
            raise

    if args.report_out is not None:
        _write_json(args.report_out, {"results": results})

    failures = [r for r in results if r.get("returncode") not in (0, None)]
    if failures and args.error_log is not None:
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        with args.error_log.open("a", encoding="utf-8") as handle:
            for entry in failures:
                handle.write(f"{entry['scene']} failed with {entry['returncode']}\n")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
