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
        "frames_total": first_metric.get("frames_total"),
        "duration_total_sec": first_metric.get("duration_total_sec"),
        "time_per_frame_sec": first_metric.get("time_per_frame_sec"),
        "paths_ok": first_metric.get("paths_ok"),
        "metrics": metrics,
        "videos": [str(path) for path in videos],
        "output_root": str(output_root),
        "output_bytes": output_bytes,
        "gpu_summary": _gpu_summary(samples),
    }


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
        "selected_count": len(selected),
        "records": [],
    }
    jsonl_path = args.results_root / "render_records.jsonl"
    gpu_samples_path = args.results_root / "gpu_samples.jsonl"
    benchmark_t0 = time.perf_counter()
    with GpuMonitor(float(args.gpu_sample_interval_sec), log_path=gpu_samples_path) as run_monitor:
        if int(args.workers) <= 1:
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
