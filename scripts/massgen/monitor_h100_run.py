#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detached progress monitor for H100 MassGen runs. Writes a compact "
            "progress markdown file and JSONL snapshots independently of the "
            "local Codex/session lifetime."
        )
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/team/telenav/code/Navdp_Datagen"))
    parser.add_argument("--python-bin", type=Path, default=Path("/team/telenav/code/conda_envs/navdp_cuda121/bin/python"))
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--stage-window-min", type=float, default=10.0)
    parser.add_argument("--baseline-summary", action="append", default=None)
    parser.add_argument("--title", default="MassGen Persistent H100 Pipeline")
    parser.add_argument("--exit-when-complete", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _pid_alive(pid: str) -> bool:
    if not pid.strip().isdigit():
        return False
    completed = subprocess.run(["ps", "-p", pid.strip(), "-o", "pid="], text=True, capture_output=True, check=False)
    return bool(completed.stdout.strip())


def _driver_status(root: Path) -> dict[str, Any]:
    pid_path = root / "driver.pid"
    pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.is_file() else ""
    ps_text = ""
    if pid:
        completed = subprocess.run(
            ["ps", "-p", pid, "-o", "pid=,etime=,stat=,%cpu=,%mem="],
            text=True,
            capture_output=True,
            check=False,
        )
        ps_text = completed.stdout.strip()
    return {"pid": pid, "alive": _pid_alive(pid), "ps": ps_text}


def _renderer_count() -> int:
    completed = subprocess.run(
        ["pgrep", "-fc", "/team/telenav/code/Navdp_Datagen/render_label_paths_telesim.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        return int(completed.stdout.strip() or "0")
    except ValueError:
        return 0


def _schedule_stats(root: Path) -> dict[str, Any]:
    schedule = _load_json(root / "persistent_schedule.json")
    assignments = schedule.get("assignments") if isinstance(schedule.get("assignments"), list) else []
    chunk_count = 0
    for assignment in assignments:
        if isinstance(assignment, dict) and isinstance(assignment.get("chunks"), list):
            chunk_count += len(assignment["chunks"])
    return {"assignment_count": len(assignments), "chunk_count": chunk_count}


def _plan_stats(root: Path) -> dict[str, Any]:
    plan = _load_json(root / "aggregate_render_plan.json")
    return {
        "selected_entry_count": plan.get("selected_entry_count"),
        "job_count": plan.get("job_count"),
        "status": plan.get("status"),
    }


def _record_stats(root: Path) -> dict[str, Any]:
    records_path = root / "run_persistent" / "render_records.jsonl"
    statuses: dict[str, int] = {}
    frames = 0
    paths_ok = 0
    output_bytes = 0
    render_sec = 0.0
    records = 0
    try:
        with records_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records += 1
                status = str(record.get("status") or "unknown")
                statuses[status] = statuses.get(status, 0) + 1
                frames += int(record.get("frames_total") or 0)
                paths_ok += int(record.get("paths_ok") or 0)
                output_bytes += int(record.get("output_bytes") or 0)
                render_sec += float(record.get("render_elapsed_sec") or 0.0)
    except OSError:
        pass
    return {
        "record_count": records,
        "statuses": statuses,
        "frames_total": frames,
        "paths_ok": paths_ok,
        "output_bytes": output_bytes,
        "summed_render_sec": render_sec,
    }


def _materialized_stats(root: Path) -> dict[str, Any]:
    materialized = root / "materialized"
    if not materialized.exists():
        return {
            "exists": False,
            "label_paths": 0,
            "actor_plans": 0,
        }
    return {
        "exists": True,
        "label_paths": sum(1 for _ in materialized.rglob("label_paths/*.json")),
        "actor_plans": sum(1 for _ in materialized.rglob("actor_plans/*.json")),
    }


def _summary_stats(root: Path) -> dict[str, Any]:
    summary = _load_json(root / "run_persistent" / "benchmark_summary.json")
    return {
        "status": summary.get("status"),
        "record_count": summary.get("record_count"),
        "success_count": summary.get("success_count"),
        "failure_count": summary.get("failure_count"),
        "total_frames": summary.get("total_frames"),
        "benchmark_wall_sec": summary.get("benchmark_wall_sec"),
    }


def _nvidia_snapshot() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    rows = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) < 5:
            continue
        try:
            rows.append(
                {
                    "nvidia_timestamp": row[0].strip(),
                    "gpu_index": row[1].strip(),
                    "gpu_util_pct": float(row[2].strip()),
                    "memory_used_mb": float(row[3].strip()),
                    "memory_total_mb": float(row[4].strip()),
                }
            )
        except ValueError:
            continue
    return rows


def _latest_snapshot(root: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(root),
        "driver": _driver_status(root),
        "renderer_count": _renderer_count(),
        "plan": _plan_stats(root),
        "schedule": _schedule_stats(root),
        "materialized": _materialized_stats(root),
        "records": _record_stats(root),
        "summary": _summary_stats(root),
        "gpu": _nvidia_snapshot(),
        "gpu_sample_lines": _line_count(root / "run_persistent" / "gpu_samples.jsonl"),
        "report_exists": (root / "report_persistent" / "REPORT.md").is_file(),
    }


def _write_progress_md(path: Path, snapshot: dict[str, Any]) -> None:
    records = snapshot["records"]
    schedule = snapshot["schedule"]
    materialized = snapshot["materialized"]
    summary = snapshot["summary"]
    gpu_lines = []
    for gpu in snapshot.get("gpu") or []:
        used = float(gpu.get("memory_used_mb") or 0.0) / 1024.0
        total = float(gpu.get("memory_total_mb") or 0.0) / 1024.0
        gpu_lines.append(
            f"- GPU {gpu.get('gpu_index')}: util {gpu.get('gpu_util_pct')}%, VRAM {used:.2f}/{total:.2f} GiB"
        )
    if not gpu_lines:
        gpu_lines = ["- GPU: unavailable"]

    total_chunks = int(schedule.get("chunk_count") or 0)
    done_chunks = int(records.get("record_count") or 0)
    pct = (100.0 * done_chunks / total_chunks) if total_chunks else 0.0
    status = summary.get("status") or ("running" if snapshot["driver"].get("alive") else "not_running")
    text = "\n".join(
        [
            "# H100 Run Progress",
            "",
            f"Updated: `{snapshot['generated_at']}`",
            f"Results root: `{snapshot['results_root']}`",
            "",
            "## Status",
            "",
            f"- Status: `{status}`",
            f"- Driver: `{snapshot['driver'].get('ps') or 'not running'}`",
            f"- Render processes: `{snapshot['renderer_count']}`",
            f"- Chunk records: `{done_chunks} / {total_chunks}` ({pct:.1f}%)",
            f"- Record statuses: `{json.dumps(records.get('statuses', {}), sort_keys=True)}`",
            f"- Rendered paths: `{records.get('paths_ok')}`",
            f"- Frames so far: `{records.get('frames_total')}`",
            f"- Summed render seconds so far: `{float(records.get('summed_render_sec') or 0.0):.1f}`",
            f"- Materialized label paths: `{materialized.get('label_paths')}`",
            f"- Materialized actor plans: `{materialized.get('actor_plans')}`",
            f"- 10 Hz GPU sample lines: `{snapshot.get('gpu_sample_lines')}`",
            f"- Final report exists: `{snapshot.get('report_exists')}`",
            "",
            "## Plan",
            "",
            f"- Selected entries: `{snapshot['plan'].get('selected_entry_count')}`",
            f"- Render work items: `{snapshot['plan'].get('job_count')}`",
            f"- Schedule assignments: `{schedule.get('assignment_count')}`",
            "",
            "## GPU",
            "",
            *gpu_lines,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_report(args: argparse.Namespace) -> int:
    run_root = args.results_root / "run_persistent"
    if not (run_root / "benchmark_summary.json").is_file():
        return 0
    if (args.results_root / "report_persistent" / "REPORT.md").is_file():
        return 0
    command = [
        str(args.python_bin),
        "scripts/massgen/report_persistent_h100_schedule_run.py",
        "--run-root",
        str(run_root),
        "--output-root",
        str(args.results_root / "report_persistent"),
        "--title",
        str(args.title),
        "--stage-window-min",
        str(float(args.stage_window_min)),
    ]
    natural_json = args.results_root / "natural_length_projection.json"
    if natural_json.is_file():
        command.extend(["--natural-length-json", str(natural_json)])
    for baseline in args.baseline_summary or []:
        command.extend(["--baseline-summary", str(Path(baseline).expanduser())])
    completed = subprocess.run(
        command,
        cwd=args.repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = args.results_root / "monitor" / "report_from_monitor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"command: {' '.join(command)}\nreturncode: {completed.returncode}\n\n{completed.stdout}",
        encoding="utf-8",
    )
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()
    root = args.results_root.expanduser().resolve()
    monitor_root = root / "monitor"
    monitor_root.mkdir(parents=True, exist_ok=True)
    snapshot_jsonl = monitor_root / "progress_snapshots.jsonl"
    progress_md = monitor_root / "PROGRESS.md"
    poll_sec = max(5.0, float(args.poll_sec))

    while True:
        snapshot = _latest_snapshot(root)
        with snapshot_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        _write_progress_md(progress_md, snapshot)

        driver_alive = bool(snapshot["driver"].get("alive"))
        summary_status = snapshot.get("summary", {}).get("status")
        if not driver_alive and summary_status:
            _run_report(args)
            snapshot = _latest_snapshot(root)
            with snapshot_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            _write_progress_md(progress_md, snapshot)
            if bool(args.exit_when_complete):
                return 0

        time.sleep(poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
