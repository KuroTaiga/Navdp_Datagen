#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.render_executor import (  # noqa: E402
    build_render_plans,
    execute_render_plans,
    load_render_manifest,
)


STAGE_KEYS = (
    "actor_gpu_cache_upload_sec",
    "actor_visibility_sec",
    "actor_transform_sec",
    "actor_tensor_pack_sec",
    "actor_merge_update_sec",
    "gaussian_render_sec",
    "gpu_readback_sec",
    "perframe_depth_sec",
    "perframe_png_sec",
    "mp4_write_sec",
    "h264_encode_sec",
    "h264_mux_sec",
    "video_close_sec",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run comparable baseline/optimized MassGen human-only render jobs. "
            "Baseline uses exact CPU actor transforms; optimized enables the GPU-resident actor cache."
        )
    )
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, default=None)
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--render-script", type=Path, default=REPO_ROOT / "render_label_paths_telesim.py")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--robot-id", action="append", default=None)
    parser.add_argument("--sensor", dest="sensor_names", action="append", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--video-backend", default="nvenc", choices=["cpu", "nvenc", "gpu"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimal-frames", type=int, default=120)
    parser.add_argument("--save-depth-maps", action="store_true")
    parser.add_argument("--save-rgb-frames", action="store_true")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _load_metrics(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.is_file():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _sum_stage_seconds(metrics: Mapping[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for path_entry in metrics.get("paths") or []:
        if not isinstance(path_entry, Mapping):
            continue
        for key, value in (path_entry.get("stage_seconds") or {}).items():
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            totals[key] = totals.get(key, 0.0) + seconds
    return totals


def _first_metrics_path(plan_payload: Mapping[str, Any]) -> str | None:
    for plan in plan_payload.get("plans", []):
        if not isinstance(plan, Mapping):
            continue
        outputs = plan.get("outputs", {})
        if isinstance(outputs, Mapping) and outputs.get("metrics_json"):
            return str(outputs["metrics_json"])
    return None


def _run_variant(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    name: str,
    actor_gpu_resident: bool,
) -> dict[str, Any]:
    variant_root = args.output_root / name
    plan_payload = build_render_plans(
        manifest,
        manifest_path=args.manifest_json,
        output_root=variant_root,
        scenes_dir=args.scenes_dir,
        tasks_dir=args.tasks_dir,
        render_script=args.render_script,
        python_bin=str(args.python_bin),
        families=args.family,
        job_ids=args.job_id,
        robot_ids=args.robot_id,
        sensor_names=args.sensor_names,
        limit=args.limit,
        write_inputs=True,
        video_backend=str(args.video_backend),
        device=str(args.device),
        save_depth_maps=bool(args.save_depth_maps),
        save_rgb_frames=bool(args.save_rgb_frames),
        minimal_frames=int(args.minimal_frames) if int(args.minimal_frames) > 0 else None,
        actor_gpu_resident=actor_gpu_resident,
        save_actor_metadata=True,
    )
    metrics_path = _first_metrics_path(plan_payload)
    start = time.perf_counter()
    returncode = 0 if args.dry_run else execute_render_plans(plan_payload)
    wall_time_sec = time.perf_counter() - start
    metrics = _load_metrics(metrics_path)
    return {
        "name": name,
        "actor_gpu_resident": actor_gpu_resident,
        "status": plan_payload.get("status"),
        "job_count": plan_payload.get("job_count"),
        "returncode": int(returncode),
        "ok": int(returncode) == 0 and plan_payload.get("status") == "ready",
        "wall_time_sec": wall_time_sec,
        "output_root": str(variant_root),
        "metrics_path": metrics_path,
        "frames_total": int(metrics.get("frames_total") or 0),
        "paths_ok": int(metrics.get("paths_ok") or 0),
        "paths_fatal": int(metrics.get("paths_fatal") or 0),
        "stage_totals": _sum_stage_seconds(metrics),
        "plans": plan_payload.get("plans", []),
    }


def _delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline <= 0.0:
        return "-"
    diff = value - baseline
    return f"{diff:+.3f}s ({(diff / baseline) * 100.0:+.1f}%)"


def _stage(run: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    totals = run.get("stage_totals") or {}
    found = False
    value = 0.0
    for key in keys:
        if key in totals:
            found = True
            value += float(totals[key])
    return value if found else None


def _write_report(output_root: Path, payload: Mapping[str, Any]) -> Path:
    runs = [run for run in payload.get("runs", []) if isinstance(run, Mapping)]
    baseline = runs[0] if runs else {}
    baseline_wall = float(baseline.get("wall_time_sec") or 0.0) if baseline else None
    lines = [
        "# Simple Actor Render Benchmark",
        "",
        f"- Manifest: `{payload.get('manifest_json')}`",
        f"- Output root: `{payload.get('output_root')}`",
        f"- Dry run: `{payload.get('dry_run')}`",
        "",
        "| Variant | OK | Jobs | Frames | Wall sec | Wall vs baseline | Actor stages sec | Actor stages vs baseline | Metrics |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    baseline_actor = _stage(
        baseline,
        (
            "actor_gpu_cache_upload_sec",
            "actor_visibility_sec",
            "actor_transform_sec",
            "actor_tensor_pack_sec",
            "actor_merge_update_sec",
        ),
    )
    for run in runs:
        wall = float(run.get("wall_time_sec") or 0.0)
        actor_total = _stage(
            run,
            (
                "actor_gpu_cache_upload_sec",
                "actor_visibility_sec",
                "actor_transform_sec",
                "actor_tensor_pack_sec",
                "actor_merge_update_sec",
            ),
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run.get('name')}`",
                    "yes" if run.get("ok") else f"no rc={run.get('returncode')}",
                    str(run.get("job_count")),
                    str(run.get("frames_total")),
                    f"{wall:.3f}",
                    _delta(wall, baseline_wall),
                    "-" if actor_total is None else f"{actor_total:.3f}",
                    _delta(actor_total, baseline_actor),
                    f"`{run.get('metrics_path')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Stage Totals", ""])
    for run in runs:
        lines.append(f"### {run.get('name')}")
        lines.append("")
        totals = run.get("stage_totals") or {}
        if not totals:
            lines.append("- No metrics captured.")
        else:
            for key in STAGE_KEYS:
                if key in totals:
                    lines.append(f"- `{key}`: {float(totals[key]):.3f}s")
        lines.append("")
    report_path = output_root / "simple_actor_benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_render_manifest(args.manifest_json)
    runs = [
        _run_variant(
            args=args,
            manifest=manifest,
            name="baseline_cpu_actor",
            actor_gpu_resident=False,
        ),
        _run_variant(
            args=args,
            manifest=manifest,
            name="optimized_gpu_actor_cache",
            actor_gpu_resident=True,
        ),
    ]
    summary_path = args.output_root / "simple_actor_benchmark_summary.json"
    payload = {
        "manifest_json": str(args.manifest_json),
        "output_root": str(args.output_root),
        "summary_json": str(summary_path),
        "dry_run": bool(args.dry_run),
        "variants": ["baseline_cpu_actor", "optimized_gpu_actor_cache"],
        "runs": runs,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = _write_report(args.output_root, payload)
    print(f"[BENCH] summary={summary_path}", flush=True)
    print(f"[BENCH] report={report_path}", flush=True)
    return 1 if any(not run["ok"] for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
