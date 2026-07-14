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
import shutil
import signal
import subprocess
import sys
import threading
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


def _count_planned_labels(
    tasks_dir: Path | None,
    scene_id: str,
    label_ids: list[str] | None,
    *,
    exclude_detailed: bool,
    max_labels: int | None,
) -> int | None:
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
        if (not exclude_detailed or not p.name.endswith("_detailed.json")) and p.name != "summary.json"
    ]
    json_paths.sort()
    if max_labels is not None and max_labels > 0:
        json_paths = json_paths[:max_labels]
    return len(json_paths)


def _extract_rendered_labels(metrics_path: Path | None) -> int | None:
    if metrics_path is None or not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Newer metrics include paths_done, which counts rendered + skipped (resume) + fatal.
    # This is the right signal for "work completed" in resume runs where outputs already exist.
    if isinstance(payload.get("paths_done"), int):
        return int(payload["paths_done"])
    if isinstance(payload.get("paths_total"), int):
        return int(payload["paths_total"])
    if isinstance(payload.get("paths"), list):
        return len(payload["paths"])
    return None


def _extract_ok_labels(metrics_path: Path | None) -> int | None:
    if metrics_path is None or not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload.get("paths_ok"), int):
        return int(payload["paths_ok"])
    if isinstance(payload.get("paths"), list):
        # paths payload is populated only for successful renders.
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
    if minimal_frames is not None and minimal_frames > 0:
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


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days:
        return f"{days}d{hrs:02d}h{mins:02d}m{sec:02d}s"
    if hrs:
        return f"{hrs}h{mins:02d}m{sec:02d}s"
    return f"{mins}m{sec:02d}s"


def _format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "-"
    try:
        value = float(num_bytes)
    except (TypeError, ValueError):
        return "-"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f}{units[idx]}"


def _normalize_label_id(label: str) -> str:
    # Manifest / CLI sometimes includes "123" and sometimes "123.json".
    p = Path(str(label))
    return p.stem if p.suffix == ".json" else str(label)


def _list_scene_output_mp4_labels(output_dir: Path, scene_id: str) -> set[str]:
    scene_out = output_dir / scene_id
    if not scene_out.is_dir():
        return set()
    labels: set[str] = set()
    try:
        with os.scandir(scene_out) as it:
            for ent in it:
                if ent.is_file() and ent.name.endswith(".mp4"):
                    labels.add(Path(ent.name).stem)
    except FileNotFoundError:
        return set()
    except Exception:
        return labels
    return labels


def _list_scene_task_labels(
    tasks_dir: Path,
    scene_id: str,
    *,
    exclude_detailed: bool,
    max_labels: int | None,
) -> list[str]:
    scene_dir = tasks_dir / scene_id
    label_dir = _resolve_label_directory(scene_dir)
    if label_dir is None:
        return []
    labels = []
    for p in sorted(label_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        if exclude_detailed and p.name.endswith("_detailed.json"):
            continue
        labels.append(p.stem)
    if max_labels is not None and max_labels > 0:
        labels = labels[:max_labels]
    return labels


def _dir_size_bytes(path: Path) -> int | None:
    """
    Best-effort directory size.
    Uses `du -sb` when available (fast on Linux), otherwise falls back to Python walk.
    """
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return int(out.split()[0])
    except Exception:
        pass
    try:
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
        return total
    except Exception:
        return None


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
    parser.add_argument(
        "--minimal-frames",
        type=int,
        default=None,
        help="If >0, truncate each rendered path to the first N frames (0/omit for full length).",
    )
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
    parser.add_argument(
        "--monitor-interval-sec",
        type=float,
        default=60.0,
        help="How often to print overall progress/ETA/space stats (default: 60s).",
    )
    parser.add_argument(
        "--avg-path-length-m",
        type=float,
        default=15.6,
        help="Average path length for derived speed reporting (default: 15.6m).",
    )
    parser.add_argument(
        "--step-distance-m",
        type=float,
        default=0.05,
        help="Meters per frame for derived FPS reporting (default: 0.05m).",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unsupported args: {' '.join(unknown)}", file=sys.stderr)

    print(
        "[START] "
        f"workers={int(args.workers)} resume={bool(args.resume)} "
        f"tasks_dir={args.tasks_dir} scenes_dir={args.scenes_dir} output_dir={args.output_dir}",
        flush=True,
    )

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

    total_jobs = len(jobs)
    completed_jobs = 0
    # Planned label counts per scene to report progress.
    scene_planned: dict[str, int] = {}
    total_planned = 0
    scene_done: dict[str, int] = {}
    total_done = 0
    total_ok = 0  # used for throughput/ETA; excludes resume skips
    results: list[dict] = []
    worker_count = max(1, int(args.workers))
    start_ts = time.time()
    last_disk_ts = 0.0
    last_disk_bytes: int | None = None
    oom_retry_passes = 0
    scene_output_bytes: dict[str, int] = {}

    # Optional resume prefilter: compute pending labels once and only spawn workers for those.
    # This avoids thousands of slow per-label existence checks on network mounts.
    if args.resume:
        prefilter_skipped = 0
        resume_scan_threshold = 32
        prefilter_t0 = time.monotonic()
        last_prefilter_note = prefilter_t0
        for idx, job in enumerate(jobs, start=1):
            scene_id = job["scene"]
            planned_labels = job.get("labels")
            if planned_labels:
                planned_ids = [_normalize_label_id(l) for l in planned_labels]
            else:
                planned_ids = _list_scene_task_labels(
                    args.tasks_dir,
                    scene_id,
                    exclude_detailed=bool(args.exclude_detailed_labels),
                    max_labels=args.max_labels,
                )

            # Track planned counts for plan/progress reporting.
            planned_n = len(planned_ids)
            job["planned_labels"] = planned_n
            scene_planned[scene_id] = scene_planned.get(scene_id, 0) + planned_n
            total_planned += planned_n

            scene_out = Path(args.output_dir) / scene_id
            existing = set()
            scene_bytes = 0
            if len(planned_ids) <= resume_scan_threshold:
                # For quick tests (few labels), probing those exact paths is faster than scanning
                # a directory with thousands of mp4s on some mounts.
                for lid in planned_ids:
                    mp4_path = scene_out / f"{lid}.mp4"
                    if mp4_path.is_file():
                        existing.add(lid)
                        try:
                            scene_bytes += int(mp4_path.stat().st_size)
                        except OSError:
                            continue
            elif scene_out.is_dir():
                try:
                    with os.scandir(scene_out) as it:
                        for ent in it:
                            if ent.is_file() and ent.name.endswith(".mp4"):
                                existing.add(Path(ent.name).stem)
                                try:
                                    scene_bytes += int(ent.stat().st_size)
                                except OSError:
                                    continue
                except Exception:
                    existing = _list_scene_output_mp4_labels(Path(args.output_dir), scene_id)
            pending_ids = [lid for lid in planned_ids if lid not in existing]
            skipped = max(0, len(planned_ids) - len(pending_ids))

            job["prefilter_planned_labels"] = len(planned_ids)
            job["prefilter_pending_labels"] = len(pending_ids)
            job["prefilter_skipped_outputs_exist"] = skipped
            job["skip_job"] = (len(pending_ids) == 0)
            job["prefilter_scene_mp4_total_bytes"] = scene_bytes
            if scene_bytes:
                scene_output_bytes[scene_id] = int(scene_bytes)

            if skipped:
                scene_done[scene_id] = scene_done.get(scene_id, 0) + skipped
                total_done += skipped
                prefilter_skipped += skipped

            # Only override labels when there is pending work; an empty list would mean "render all".
            if pending_ids:
                job["labels"] = pending_ids

            # Periodic heartbeat so `tee` users know we're alive and in prefilter IO.
            now = time.monotonic()
            if now - last_prefilter_note >= 5.0:
                print(
                    "[PREFILTER] "
                    f"scanned_scenes~={idx}/{len(jobs)} last_scene={scene_id} "
                    f"planned={planned_n} pending={len(pending_ids)} skipped={skipped} "
                    f"elapsed={_format_seconds(now - prefilter_t0)}",
                    flush=True,
                )
                last_prefilter_note = now

        if prefilter_skipped:
            print(
                f"[PREFILTER] resume=true skipped_outputs_exist={prefilter_skipped} (counted as done for progress; excluded from speed)",
                flush=True,
            )
        if scene_output_bytes:
            # Best-effort: used for disk estimation without expensive `du` walks.
            last_disk_bytes = int(sum(scene_output_bytes.values()))

        # Run heavier jobs earlier to reduce tail idle time (still 1 scene per worker).
        jobs.sort(key=lambda j: int(j.get("prefilter_pending_labels") or 0), reverse=True)
    else:
        # Non-resume: still compute planned counts for progress display.
        for job in jobs:
            planned = _count_planned_labels(
                args.tasks_dir,
                job["scene"],
                job.get("labels"),
                exclude_detailed=bool(args.exclude_detailed_labels),
                max_labels=args.max_labels,
            )
            job["planned_labels"] = planned
            if planned is not None:
                scene_planned[job["scene"]] = scene_planned.get(job["scene"], 0) + planned
                total_planned += planned

    def _update_progress(last_scene: str | None = None) -> None:
        stats = _compute_stats(time.time())
        payload = {
            "total": total_jobs,
            "completed": min(completed_jobs, total_jobs),
            "paths_total": (total_planned if total_planned > 0 else None),
            "paths_done": total_done,
            "paths_ok": total_ok,
            "workers": worker_count,
            "elapsed_sec": stats.get("elapsed_sec"),
            "speed_paths_per_sec": stats.get("speed_paths_per_sec"),
            "eta_sec": stats.get("eta_sec"),
            "disk_usage_bytes": stats.get("disk_usage_bytes"),
            "disk_free_bytes": stats.get("disk_free_bytes"),
            "disk_est_total_bytes": stats.get("disk_est_total_bytes"),
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
        print(f"[PLAN] Workers: {worker_count}")
        print("[PLAN] Planned labels per scene:")
        for scene_id, planned in sorted(scene_planned.items()):
            print(f"  - {scene_id}: {planned} planned")
        if total_planned > 0:
            print(f"[PLAN] Overall planned labels: {total_planned}")

    def _compute_stats(now_ts: float) -> dict:
        elapsed = max(0.0, now_ts - start_ts)
        # Throughput: use successful renders only (exclude resume skips + fatals).
        speed_paths = (total_ok / elapsed) if elapsed > 0 and total_ok > 0 else None
        remaining = (total_planned - total_done) if total_planned > 0 else None
        eta_sec = (remaining / speed_paths) if remaining is not None and speed_paths and speed_paths > 0 else None

        avg_len_m = float(args.avg_path_length_m or 0.0)
        step_m = float(args.step_distance_m or 0.0)
        frames_per_path = (avg_len_m / step_m) if avg_len_m > 0 and step_m > 0 else None

        speed_mps = (speed_paths * avg_len_m) if speed_paths and avg_len_m > 0 else None
        speed_fps = (speed_paths * frames_per_path) if speed_paths and frames_per_path else None

        nonlocal last_disk_ts, last_disk_bytes
        usage_bytes = None
        free_bytes = None
        total_bytes = None
        est_total_bytes = None
        if args.output_dir is not None:
            try:
                du = shutil.disk_usage(str(args.output_dir))
                free_bytes = int(du.free)
                total_bytes = int(du.total)
            except Exception:
                free_bytes = None
                total_bytes = None
            # Prefer fast, metrics-based estimate of "output bytes" when available.
            usage_bytes = last_disk_bytes

        if usage_bytes is not None and total_done > 0 and total_planned > 0:
            bytes_per_path = usage_bytes / float(total_done)
            est_total_bytes = int(bytes_per_path * float(total_planned))

        return {
            "elapsed_sec": elapsed,
            "speed_paths_per_sec": speed_paths,
            "speed_m_per_sec": speed_mps,
            "speed_frames_per_sec": speed_fps,
            "eta_sec": eta_sec,
            "disk_usage_bytes": usage_bytes,
            "disk_free_bytes": free_bytes,
            "disk_total_bytes": total_bytes,
            "disk_est_total_bytes": est_total_bytes,
        }

    def _note_completion(
        scene_id: str,
        metrics_path: Path | None,
        planned_labels: int | None,
        *,
        count_job: bool = True,
    ) -> None:
        nonlocal completed_jobs, total_done, total_ok
        nonlocal last_disk_bytes, scene_output_bytes
        if count_job:
            completed_jobs += 1
        rendered = _extract_rendered_labels(metrics_path)
        if rendered is None:
            rendered = planned_labels if planned_labels is not None else 0
        ok = _extract_ok_labels(metrics_path)
        if ok is None:
            # Fallback: if we can't read metrics, assume all completions were "ok".
            ok = int(rendered)
        # Output size accounting from per-scene metrics (mp4 totals).
        scene_bytes = None
        if metrics_path is not None and metrics_path.is_file():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(payload.get("scene_mp4_total_bytes"), int):
                    scene_bytes = int(payload["scene_mp4_total_bytes"])
            except Exception:
                scene_bytes = None
        if scene_bytes is not None:
            scene_output_bytes[scene_id] = int(scene_bytes)
            last_disk_bytes = int(sum(scene_output_bytes.values()))
        scene_done[scene_id] = scene_done.get(scene_id, 0) + rendered
        total_done += rendered
        total_ok += ok
        scene_total = scene_planned.get(scene_id)
        scene_progress = (
            f"{scene_done[scene_id]}/{scene_total}" if scene_total not in (None, 0) else f"{scene_done[scene_id]}"
        )
        overall_progress = (
            f"{total_done}/{total_planned}" if total_planned not in (None, 0) else f"{total_done}"
        )
        stats = _compute_stats(time.time())
        speed = stats.get("speed_paths_per_sec")
        speed_mps = stats.get("speed_m_per_sec")
        speed_fps = stats.get("speed_frames_per_sec")
        eta_sec = stats.get("eta_sec")
        disk_used = stats.get("disk_usage_bytes")
        disk_free = stats.get("disk_free_bytes")
        disk_est_total = stats.get("disk_est_total_bytes")
        if speed is not None:
            print(
                "[PROGRESS] "
                f"scene={scene_id} paths={scene_progress} overall={overall_progress} workers={worker_count} "
                f"speed={speed:.3f} paths/s ({(speed_mps or 0.0):.2f} m/s, {(speed_fps or 0.0):.1f} fps) "
                f"eta={_format_seconds(eta_sec)} "
                f"disk={_format_bytes(disk_used)}/{_format_bytes(disk_free)} est_total={_format_bytes(disk_est_total)}",
                flush=True,
            )
        else:
            print(
                "[PROGRESS] "
                f"scene={scene_id} paths={scene_progress} overall={overall_progress} workers={worker_count} "
                f"eta={_format_seconds(eta_sec)} "
                f"disk={_format_bytes(disk_used)}/{_format_bytes(disk_free)} est_total={_format_bytes(disk_est_total)}",
                flush=True,
            )
        _update_progress(scene_id)

    def _count_running_procs() -> int:
        return sum(1 for p in procs if p.poll() is None)

    def _collect_unresolved_ooms(status_map: dict) -> dict[str, list[str]]:
        unresolved: dict[str, list[str]] = {}
        for scene, labels in (status_map or {}).items():
            if not isinstance(labels, dict):
                continue
            for label, info in labels.items():
                if not isinstance(info, dict):
                    continue
                if int(info.get("status", STATUS_NOT_RUN)) == STATUS_RETRY:
                    unresolved.setdefault(str(scene), []).append(str(label))
        for scene in unresolved:
            unresolved[scene].sort()
        return unresolved

    def _handle_terminate(_signum: int, _frame: object | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_terminate)
    signal.signal(signal.SIGTERM, _handle_terminate)
    monitor_stop = threading.Event()
    monitor_thread: threading.Thread | None = None
    try:
        _print_plans()

        def _monitor_loop() -> None:
            interval = float(args.monitor_interval_sec or 0.0)
            if interval <= 0:
                return
            while not monitor_stop.is_set():
                time.sleep(interval)
                if monitor_stop.is_set():
                    break
                stats = _compute_stats(time.time())
                speed = stats.get("speed_paths_per_sec")
                eta_sec = stats.get("eta_sec")
                disk_used = stats.get("disk_usage_bytes")
                disk_free = stats.get("disk_free_bytes")
                disk_est_total = stats.get("disk_est_total_bytes")
                overall_progress = (
                    f"{total_done}/{total_planned}" if total_planned not in (None, 0) else f"{total_done}"
                )
                running = _count_running_procs()
                if speed is not None:
                    print(
                        "[MON] "
                        f"jobs={completed_jobs}/{total_jobs} running={running} paths={overall_progress} workers={worker_count} "
                        f"speed={speed:.3f} paths/s eta={_format_seconds(eta_sec)} "
                        f"disk={_format_bytes(disk_used)}/{_format_bytes(disk_free)} est_total={_format_bytes(disk_est_total)}",
                        flush=True,
                    )
                else:
                    print(
                        "[MON] "
                        f"jobs={completed_jobs}/{total_jobs} running={running} paths={overall_progress} workers={worker_count} "
                        f"eta={_format_seconds(eta_sec)} "
                        f"disk={_format_bytes(disk_used)}/{_format_bytes(disk_free)} est_total={_format_bytes(disk_est_total)}",
                        flush=True,
                    )

        monitor_thread = threading.Thread(target=_monitor_loop, name="telesim-progress", daemon=True)
        monitor_thread.start()

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for job in jobs:
                scene_id = job["scene"]
                if job.get("skip_job"):
                    # Nothing pending for this scene based on prefilter; don't spawn a worker.
                    completed_jobs += 1
                    scene_total = scene_planned.get(scene_id)
                    scene_progress = (
                        f"{scene_done.get(scene_id, 0)}/{scene_total}"
                        if scene_total not in (None, 0)
                        else f"{scene_done.get(scene_id, 0)}"
                    )
                    overall_progress = (
                        f"{total_done}/{total_planned}" if total_planned not in (None, 0) else f"{total_done}"
                    )
                    stats = _compute_stats(time.time())
                    speed = stats.get("speed_paths_per_sec")
                    eta_sec = stats.get("eta_sec")
                    disk_used = stats.get("disk_usage_bytes")
                    disk_free = stats.get("disk_free_bytes")
                    disk_est_total = stats.get("disk_est_total_bytes")
                    if speed is not None:
                        print(
                            "[PROGRESS] "
                            f"scene={scene_id} paths={scene_progress} overall={overall_progress} workers={worker_count} "
                            f"speed={speed:.3f} paths/s eta={_format_seconds(eta_sec)} "
                            f"disk={_format_bytes(disk_used)}/{_format_bytes(disk_free)} est_total={_format_bytes(disk_est_total)}",
                            flush=True,
                        )
                    else:
                        print(
                            "[PROGRESS] "
                            f"scene={scene_id} paths={scene_progress} overall={overall_progress} workers={worker_count} "
                            f"eta={_format_seconds(eta_sec)} "
                            f"disk={_format_bytes(disk_used)}/{_format_bytes(disk_free)} est_total={_format_bytes(disk_est_total)}",
                            flush=True,
                        )
                    _update_progress(scene_id)
                    continue
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
                    label_ids=(job.get("labels") if job.get("labels") is not None else args.label_id),
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
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
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
                oom_retry_passes = retry_pass
                if retry_delay > 0:
                    time.sleep(retry_delay)
                retry_results: list[dict] = []
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
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
            monitor_stop.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=2.0)
            _terminate_processes(procs)
            raise

    if args.report_out is not None:
        unresolved_ooms = {}
        if args.status_json is not None and args.status_json.is_file():
            unresolved_ooms = _collect_unresolved_ooms(_load_status_map(args.status_json))
        _write_json(
            args.report_out,
            {
                "results": results,
                "oom_retry_passes": oom_retry_passes,
                "oom_unresolved": unresolved_ooms,
            },
        )

    failures = [r for r in results if r.get("returncode") not in (0, None)]
    if failures and args.error_log is not None:
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        with args.error_log.open("a", encoding="utf-8") as handle:
            for entry in failures:
                handle.write(f"{entry['scene']} failed with {entry['returncode']}\n")

    # Even when subprocesses return 0, TeleSim may mark some labels as RETRY (cuda_oom).
    # Surface this explicitly after the retry loop terminates.
    if args.status_json is not None and args.status_json.is_file():
        unresolved_ooms = _collect_unresolved_ooms(_load_status_map(args.status_json))
        if unresolved_ooms:
            remaining_labels = sum(len(v) for v in unresolved_ooms.values())
            remaining_scenes = len(unresolved_ooms)
            max_show = 5
            examples = []
            for scene_id, labels in sorted(unresolved_ooms.items()):
                examples.append(f"{scene_id}:{','.join(labels[:max_show])}{'...' if len(labels) > max_show else ''}")
                if len(examples) >= 6:
                    break
            print(
                f"[OOM][WARN] Unresolved cuda_oom after retry passes={oom_retry_passes} "
                f"(scenes={remaining_scenes} labels={remaining_labels}). Examples: {' | '.join(examples)}",
                file=sys.stderr,
                flush=True,
            )
            if args.error_log is not None:
                args.error_log.parent.mkdir(parents=True, exist_ok=True)
                with args.error_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"[OOM] Unresolved cuda_oom after retry passes={oom_retry_passes} "
                        f"scenes={remaining_scenes} labels={remaining_labels}\n"
                    )

    monitor_stop.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=2.0)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
