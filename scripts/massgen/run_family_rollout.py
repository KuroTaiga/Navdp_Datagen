#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one render per MassGen family package and collect reproducible logs."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--video-backend", default="cpu", choices=["cpu", "nvenc", "gpu"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimal-frames", type=int, default=None)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--limit", type=int, default=1, help="Maximum manifest jobs to render per family.")
    parser.add_argument("--retry", type=int, default=1)
    return parser.parse_args()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_capture(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - t0
    lines = [
        f"command: {' '.join(cmd)}\n",
        f"started_at: {started}\n",
        f"wall_time_sec: {elapsed:.6f}\n",
        f"returncode: {completed.returncode}\n",
        "\n--- stdout ---\n",
        completed.stdout or "",
        "\n--- stderr ---\n",
        completed.stderr or "",
    ]
    _write_text(log_path, "".join(lines))
    return completed, elapsed


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _collect_family(family_dir: Path, plan_payload: Mapping[str, Any], render_rc: int, wall_time_sec: float, retry_count: int) -> dict[str, Any]:
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
    }


def _copy_package_metadata(package_root: Path, results_root: Path) -> None:
    for name in ("family_index.json", "action_catalog_5880_avatar.json"):
        src = package_root / name
        if src.exists():
            shutil.copy2(src, results_root / name)


def _render_family(args: argparse.Namespace, family: str, logs_root: Path, families_root: Path) -> dict[str, Any]:
    src_family = args.package_root / family
    family_dir = families_root / family
    if family_dir.exists():
        shutil.rmtree(family_dir)
    shutil.copytree(src_family, family_dir)

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
    if args.minimal_frames is not None and int(args.minimal_frames) > 0:
        base_cmd.extend(["--minimal-frames", str(int(args.minimal_frames))])
    if args.limit is not None and int(args.limit) > 0:
        base_cmd.extend(["--limit", str(int(args.limit))])

    plan_completed, plan_elapsed = _run_capture(base_cmd, cwd=args.repo_root, log_path=logs_root / f"{family}_plan.log")
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
        render_completed, render_elapsed = _run_capture(render_cmd, cwd=args.repo_root, log_path=logs_root / f"{family}_render.log")
        render_rc = int(render_completed.returncode)
        while render_rc != 0 and retry_count < int(args.retry):
            retry_count += 1
            retry_completed, render_elapsed = _run_capture(
                render_cmd,
                cwd=args.repo_root,
                log_path=logs_root / f"{family}_render_retry{retry_count}.log",
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
        ),
    )
    return _collect_family(family_dir, plan_payload, render_rc, render_elapsed, retry_count)


def main() -> int:
    args = _parse_args()
    args.package_root = args.package_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()

    if args.results_root.exists():
        shutil.rmtree(args.results_root)
    logs_root = args.results_root / "logs"
    families_root = args.results_root / "families"
    logs_root.mkdir(parents=True, exist_ok=True)
    families_root.mkdir(parents=True, exist_ok=True)
    _copy_package_metadata(args.package_root, args.results_root)

    families = _family_names(args.package_root, args.family)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(args.repo_root),
        "package_root": str(args.package_root),
        "results_root": str(args.results_root),
        "python_bin": str(args.python_bin),
        "video_backend": str(args.video_backend),
        "device": str(args.device),
        "minimal_frames": args.minimal_frames,
        "limit": args.limit,
        "families": [],
    }
    jsonl_path = args.results_root / "family_render_summary.jsonl"
    for family in families:
        record = _render_family(args, family, logs_root, families_root)
        summary["families"].append(record)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            (
                f"{family}: {record['status']} rc={record['render_returncode']} "
                f"videos={len(record['videos'])} frames={record['frames_total']} "
                f"actors={record['actor_count']} wall={record['wall_time_sec']:.2f}s"
            ),
            flush=True,
        )

    summary["family_count"] = len(summary["families"])
    summary["success_count"] = sum(1 for item in summary["families"] if item.get("status") == "success")
    summary["status"] = "success" if summary["success_count"] == summary["family_count"] else "failed"
    _write_text(
        args.results_root / "family_render_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"summary_status {summary['status']} {summary['success_count']} / {summary['family_count']}", flush=True)
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
