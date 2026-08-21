#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


_GPU_TIMESTAMP_TZ = timezone(timedelta(hours=8))

_EXCLUSIVE_LIFECYCLE_STAGES = (
    "python_import_sec",
    "argument_parse_sec",
    "output_preflight_sec",
    "scene_metadata_load_sec",
    "label_collect_sec",
    "manifest_plan_sec",
    "precheck_sec",
    "scene_ply_load_sec",
    "scene_asset_build_sec",
    "renderer_init_sec",
    "actor_plan_load_sec",
    "actor_sequence_load_sec",
    "actor_gpu_cache_upload_sec",
    "path_prepare_sec",
    "render_loop_sec",
    "writer_close_sec",
    "output_bookkeeping_sec",
    "renderer_shutdown_sec",
)

_STAGE_COLORS = {
    "startup/precheck": "#94a3b8",
    "scene+renderer init": "#8b5cf6",
    "actor load": "#f97316",
    "actor GPU upload": "#eab308",
    "path prepare": "#06b6d4",
    "render loop": "#16a34a",
    "write/close": "#64748b",
    "between chunks": "#cbd5e1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graphs/tables/report for run_persistent_h100_schedule.py outputs."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Report output root. Defaults to <run-root>/report.",
    )
    parser.add_argument("--title", default="Persistent H100 Schedule Run Report")
    parser.add_argument("--stage-window-min", type=float, default=4.0)
    parser.add_argument(
        "--natural-length-json",
        type=Path,
        default=None,
        help="Optional natural_length_projection.json to include in the report.",
    )
    parser.add_argument(
        "--baseline-summary",
        action="append",
        default=None,
        help="Optional comparison summary JSON. Can be passed multiple times.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_gpu_timestamp(value: str) -> float:
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=_GPU_TIMESTAMP_TZ).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unsupported GPU timestamp: {value}")


def _gpu_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    for item in _load_jsonl(path):
        if "timestamp" not in item:
            continue
        epoch = _parse_gpu_timestamp(str(item["timestamp"]))
        samples.append(
            {
                **item,
                "epoch_sec": epoch,
                "gpu_util_pct": float(item.get("gpu_util_pct") or 0.0),
                "memory_used_mib": float(item.get("memory_used_mib") or 0.0),
            }
        )
    if samples:
        first = samples[0]["epoch_sec"]
        for sample in samples:
            sample["t_rel_sec"] = float(sample["epoch_sec"]) - first
    return samples


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p90": _pct(values, 0.90),
        "p95": _pct(values, 0.95),
        "max": max(values),
    }


def _gpu_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    util = [float(sample["gpu_util_pct"]) for sample in samples]
    memory = [float(sample["memory_used_mib"]) for sample in samples]
    return {
        "sample_count": len(samples),
        "duration_sec": samples[-1]["epoch_sec"] - samples[0]["epoch_sec"] if len(samples) > 1 else 0.0,
        "gpu_util_pct": _summary(util),
        "samples_ge50_pct": 100.0 * sum(1 for value in util if value >= 50.0) / len(util),
        "samples_ge80_pct": 100.0 * sum(1 for value in util if value >= 80.0) / len(util),
        "memory_used_gib": _summary([value / 1024.0 for value in memory]),
    }


def _stage_group(stage: str) -> str:
    if stage in {
        "python_import_sec",
        "argument_parse_sec",
        "output_preflight_sec",
        "scene_metadata_load_sec",
        "label_collect_sec",
        "manifest_plan_sec",
        "precheck_sec",
    }:
        return "startup/precheck"
    if stage in {"scene_ply_load_sec", "scene_asset_build_sec", "renderer_init_sec"}:
        return "scene+renderer init"
    if stage in {"actor_plan_load_sec", "actor_sequence_load_sec"}:
        return "actor load"
    if stage == "actor_gpu_cache_upload_sec":
        return "actor GPU upload"
    if stage == "path_prepare_sec":
        return "path prepare"
    if stage == "render_loop_sec":
        return "render loop"
    if stage in {"writer_close_sec", "output_bookkeeping_sec", "renderer_shutdown_sec"}:
        return "write/close"
    return stage


def _stage_intervals(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    chunk_bounds_by_assignment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records = sorted(
        [record for record in summary.get("records", []) if isinstance(record, Mapping)],
        key=lambda item: (str(item.get("assignment_id") or item.get("gpu_id") or ""), int(item.get("chunk_index") or 0)),
    )
    for record in records:
        metrics = [metric for metric in record.get("metrics", []) if isinstance(metric, Mapping)]
        if not metrics:
            continue
        lifecycle = metrics[0].get("lifecycle_seconds")
        if not isinstance(lifecycle, Mapping):
            continue
        process_start = lifecycle.get("process_start_sec")
        process_total = lifecycle.get("process_total_sec")
        try:
            cursor = float(process_start)
        except (TypeError, ValueError):
            continue
        assignment_id = str(record.get("assignment_id") or record.get("gpu_id") or "")
        if process_total is not None:
            try:
                chunk_bounds_by_assignment[assignment_id].append(
                    {
                        "start_epoch_sec": cursor,
                        "end_epoch_sec": cursor + max(0.0, float(process_total)),
                        "chunk_index": int(record.get("chunk_index") or 0),
                        "chunk_id": str(record.get("chunk_id") or ""),
                        "gpu_id": str(record.get("gpu_id") or ""),
                    }
                )
            except (TypeError, ValueError):
                pass
        for stage in _EXCLUSIVE_LIFECYCLE_STAGES:
            try:
                duration = max(0.0, float(lifecycle.get(stage) or 0.0))
            except (TypeError, ValueError):
                continue
            if duration <= 0.0:
                continue
            intervals.append(
                {
                    "assignment_id": assignment_id,
                    "gpu_id": str(record.get("gpu_id") or ""),
                    "chunk_index": int(record.get("chunk_index") or 0),
                    "chunk_id": str(record.get("chunk_id") or ""),
                    "stage": stage,
                    "stage_group": _stage_group(stage),
                    "start_epoch_sec": cursor,
                    "end_epoch_sec": cursor + duration,
                    "duration_sec": duration,
                }
            )
            cursor += duration
    for assignment_id, bounds in chunk_bounds_by_assignment.items():
        bounds = sorted(bounds, key=lambda item: (float(item["start_epoch_sec"]), int(item["chunk_index"])))
        for left, right in zip(bounds, bounds[1:]):
            start = float(left["end_epoch_sec"])
            end = float(right["start_epoch_sec"])
            if end <= start:
                continue
            intervals.append(
                {
                    "assignment_id": assignment_id,
                    "gpu_id": str(right.get("gpu_id") or left.get("gpu_id") or ""),
                    "chunk_index": int(left.get("chunk_index") or 0),
                    "chunk_id": f"{left.get('chunk_id')}->{right.get('chunk_id')}",
                    "stage": "between_chunks",
                    "stage_group": "between chunks",
                    "start_epoch_sec": start,
                    "end_epoch_sec": end,
                    "duration_sec": end - start,
                }
            )
    return intervals


def _stage_summary(
    *,
    intervals: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    window_start: float,
    window_end: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clipped: list[dict[str, Any]] = []
    duration_by_group: Counter[str] = Counter()
    for interval in intervals:
        start = max(float(interval["start_epoch_sec"]), window_start)
        end = min(float(interval["end_epoch_sec"]), window_end)
        if end <= start:
            continue
        item = dict(interval)
        item["start_clip"] = start
        item["end_clip"] = end
        item["duration_clip_sec"] = end - start
        clipped.append(item)
        duration_by_group[str(item["stage_group"])] += end - start

    priority = [
        "render loop",
        "actor GPU upload",
        "path prepare",
        "scene+renderer init",
        "actor load",
        "write/close",
        "startup/precheck",
        "between chunks",
    ]
    sample_values_by_group: dict[str, list[float]] = defaultdict(list)
    window_samples = [sample for sample in samples if window_start <= float(sample["epoch_sec"]) <= window_end]
    for sample in window_samples:
        active_groups = {
            str(interval["stage_group"])
            for interval in clipped
            if float(interval["start_clip"]) <= float(sample["epoch_sec"]) <= float(interval["end_clip"])
        }
        for group in active_groups:
            sample_values_by_group[group].append(float(sample["gpu_util_pct"]))
        if not active_groups:
            sample_values_by_group["unmarked/idle"].append(float(sample["gpu_util_pct"]))
        else:
            active_groups_sorted = sorted(
                active_groups,
                key=lambda group: priority.index(group) if group in priority else 99,
            )
            sample_values_by_group[f"priority:{active_groups_sorted[0]}"].append(
                float(sample["gpu_util_pct"])
            )

    rows: list[dict[str, Any]] = []
    window_sec = max(0.001, window_end - window_start)
    groups = sorted(set(duration_by_group) | {group for group in sample_values_by_group if not group.startswith("priority:")})
    for group in groups:
        values = sample_values_by_group.get(group, [])
        rows.append(
            {
                "stage_group": group,
                "worker_seconds_in_window": round(float(duration_by_group.get(group, 0.0)), 3),
                "worker_seconds_per_window_sec": round(float(duration_by_group.get(group, 0.0)) / window_sec, 3),
                "sample_count_when_active": len(values),
                "avg_gpu_pct_when_active": round(sum(values) / len(values), 3) if values else "",
                "p90_gpu_pct_when_active": round(_pct(values, 0.90), 3) if values else "",
                "max_gpu_pct_when_active": round(max(values), 3) if values else "",
            }
        )
    rows.sort(key=lambda row: float(row["worker_seconds_in_window"]), reverse=True)
    return rows, clipped


def _worker_lane_rows(clipped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    chunks: dict[str, set[str]] = defaultdict(set)
    for interval in clipped:
        assignment_id = str(interval["assignment_id"])
        totals[assignment_id][str(interval["stage_group"])] += float(interval["duration_clip_sec"])
        chunks[assignment_id].add(str(interval["chunk_id"]))
    rows: list[dict[str, Any]] = []
    for assignment_id in sorted(totals):
        total = sum(totals[assignment_id].values())
        rows.append(
            {
                "assignment_id": assignment_id,
                "chunk_count_seen": len(chunks[assignment_id]),
                "worker_seconds": round(total, 3),
                "render_loop_sec": round(float(totals[assignment_id].get("render loop", 0.0)), 3),
                "actor_load_sec": round(float(totals[assignment_id].get("actor load", 0.0)), 3),
                "write_close_sec": round(float(totals[assignment_id].get("write/close", 0.0)), 3),
                "startup_precheck_sec": round(float(totals[assignment_id].get("startup/precheck", 0.0)), 3),
                "scene_renderer_init_sec": round(float(totals[assignment_id].get("scene+renderer init", 0.0)), 3),
            }
        )
    return rows


def _plot_gpu_timeline(samples: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    if samples:
        x = [float(sample["t_rel_sec"]) / 60.0 for sample in samples]
        ax.plot(x, [float(sample["gpu_util_pct"]) for sample in samples], color="#2563eb", label="GPU util %")
        ax.axhline(80, color="#dc2626", linestyle="--", linewidth=1.0, label="80% target")
        ax.set_ylim(0, 105)
        ax2 = ax.twinx()
        ax2.plot(
            x,
            [float(sample["memory_used_mib"]) / 1024.0 for sample in samples],
            color="#334155",
            alpha=0.6,
            label="VRAM GiB",
        )
        ax2.set_ylabel("VRAM used GiB")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
    ax.set_title("GPU/VRAM Timeline")
    ax.set_xlabel("minutes from trace start")
    ax.set_ylabel("GPU util %")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_stage_overlay(
    *,
    samples: list[dict[str, Any]],
    clipped: list[dict[str, Any]],
    window_start: float,
    window_min: float,
    path: Path,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.25])
    ax = fig.add_subplot(grid[0, 0])
    ax2 = ax.twinx()
    window_end = window_start + window_min * 60.0
    window_samples = [sample for sample in samples if window_start <= float(sample["epoch_sec"]) <= window_end]
    if window_samples:
        x = [(float(sample["epoch_sec"]) - window_start) / 60.0 for sample in window_samples]
        ax.plot(x, [float(sample["gpu_util_pct"]) for sample in window_samples], color="#2563eb", label="GPU util %")
        ax.axhline(80, color="#dc2626", linestyle="--", linewidth=1.0, label="80% target")
        ax2.plot(
            x,
            [float(sample["memory_used_mib"]) / 1024.0 for sample in window_samples],
            color="#334155",
            alpha=0.55,
            label="VRAM GiB",
        )
    for interval in clipped:
        if interval["stage_group"] in {"actor load", "render loop", "scene+renderer init", "write/close"}:
            ax.axvspan(
                (float(interval["start_clip"]) - window_start) / 60.0,
                (float(interval["end_clip"]) - window_start) / 60.0,
                color=_STAGE_COLORS.get(str(interval["stage_group"]), "#999999"),
                alpha=0.06,
                linewidth=0,
            )
    ax.set_xlim(0, window_min)
    ax.set_ylim(0, 105)
    ax.set_title(title or f"First {window_min:g} minutes: GPU/VRAM with worker lifecycle lanes")
    ax.set_ylabel("GPU util %")
    ax2.set_ylabel("VRAM used GiB")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    ax_lane = fig.add_subplot(grid[1, 0], sharex=ax)
    assignments = sorted({str(interval["assignment_id"]) for interval in clipped})
    y_by_assignment = {assignment_id: len(assignments) - index for index, assignment_id in enumerate(assignments)}
    for interval in clipped:
        y = y_by_assignment[str(interval["assignment_id"])]
        ax_lane.broken_barh(
            [
                (
                    (float(interval["start_clip"]) - window_start) / 60.0,
                    (float(interval["end_clip"]) - float(interval["start_clip"])) / 60.0,
                )
            ],
            (y - 0.35, 0.7),
            facecolors=_STAGE_COLORS.get(str(interval["stage_group"]), "#999999"),
            edgecolors="none",
        )
    ax_lane.set_yticks([y_by_assignment[item] for item in assignments])
    ax_lane.set_yticklabels(assignments)
    ax_lane.set_xlabel("minutes from trace start")
    ax_lane.set_ylabel("assignment lane")
    legend_groups = [
        "startup/precheck",
        "scene+renderer init",
        "actor load",
        "actor GPU upload",
        "path prepare",
        "render loop",
        "write/close",
    ]
    handles = [Patch(facecolor=_STAGE_COLORS[group], label=group) for group in legend_groups]
    ax_lane.legend(handles=handles, ncol=4, fontsize=8, loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_overhead_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(summary.get("records", []), key=lambda item: (str(item.get("assignment_id") or ""), int(item.get("chunk_index") or 0))):
        if not isinstance(record, Mapping):
            continue
        overhead = record.get("render_overhead")
        if not isinstance(overhead, Mapping):
            overhead = {}
        rows.append(
            {
                "assignment_id": record.get("assignment_id") or record.get("gpu_id"),
                "gpu_id": record.get("gpu_id"),
                "chunk_index": record.get("chunk_index"),
                "chunk_id": record.get("chunk_id"),
                "status": record.get("status"),
                "jobs": record.get("job_count"),
                "frames": record.get("frames_total"),
                "render_elapsed_sec": round(float(record.get("render_elapsed_sec") or 0.0), 3),
                "render_loop_sec": round(float(overhead.get("nested_render_loop_sec") or 0.0), 3),
                "setup_load_process_sec": round(float(overhead.get("nested_setup_load_process_sec") or 0.0), 3),
                "output_mib": round(float(record.get("output_bytes") or 0.0) / (1024.0 * 1024.0), 3),
            }
        )
    return rows


def _comparison_rows(paths: list[str] | None, current_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_text in paths or []:
        path = Path(path_text)
        if not path.is_file():
            continue
        payload = _load_json(path)
        rows.append(_comparison_row(path.stem, payload))
    rows.append(_comparison_row("current", current_summary))
    return rows


def _comparison_row(name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    gpu = summary.get("gpu_run_summary")
    if not isinstance(gpu, Mapping):
        gpu = {}
    overhead = summary.get("render_overhead_summary")
    if not isinstance(overhead, Mapping):
        overhead = {}
    frames = float(summary.get("total_frames") or 0.0)
    wall = float(summary.get("benchmark_wall_sec") or 0.0)
    return {
        "run": name,
        "status": summary.get("status"),
        "frames": int(frames),
        "wall_sec": round(wall, 3) if wall else "",
        "frames_per_wall_sec": round(frames / wall, 3) if wall else "",
        "gpu_avg_pct": round(float(gpu.get("gpu_util_avg_pct") or 0.0), 3) if gpu else "",
        "gpu_max_pct": round(float(gpu.get("gpu_util_max_pct") or 0.0), 3) if gpu else "",
        "renderer_processes": int(overhead.get("renderer_process_count") or 0),
        "setup_sec_per_frame": round(float(overhead.get("setup_seconds_per_frame") or 0.0), 6) if overhead else "",
        "launches_per_1000_frames": round(float(overhead.get("process_launches_per_1000_frames") or 0.0), 3) if overhead else "",
    }


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _human_seconds(value: float | int | None) -> str:
    seconds = float(value or 0.0)
    if seconds >= 3600:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_root = (args.output_root or (run_root / "report")).expanduser().resolve()
    graphs_root = output_root / "assets" / "graphs"
    tables_root = output_root / "assets" / "tables"
    metrics_root = output_root / "metrics"
    output_root.mkdir(parents=True, exist_ok=True)

    summary = _load_json(run_root / "benchmark_summary.json")
    samples = _gpu_samples(run_root / "gpu_samples.jsonl")
    gpu_summary = _gpu_summary(samples)
    intervals = _stage_intervals(summary)
    interval_start = min((float(interval["start_epoch_sec"]) for interval in intervals), default=0.0)
    interval_end = max((float(interval["end_epoch_sec"]) for interval in intervals), default=0.0)
    window_start = samples[0]["epoch_sec"] if samples else interval_start
    requested_window_end = window_start + float(args.stage_window_min) * 60.0
    observed_window_end = samples[-1]["epoch_sec"] if samples else max(requested_window_end, interval_end)
    window_end = min(requested_window_end, observed_window_end)
    actual_window_min = max(0.0, window_end - window_start) / 60.0
    stage_rows, clipped = _stage_summary(
        intervals=intervals,
        samples=samples,
        window_start=window_start,
        window_end=window_end,
    )
    lane_rows = _worker_lane_rows(clipped)
    full_window_start = window_start
    full_window_end = max(observed_window_end, interval_end)
    full_window_min = max(0.0, full_window_end - full_window_start) / 60.0
    full_stage_rows, full_clipped = _stage_summary(
        intervals=intervals,
        samples=samples,
        window_start=full_window_start,
        window_end=full_window_end,
    )
    full_lane_rows = _worker_lane_rows(full_clipped)
    chunk_rows = _render_overhead_rows(summary)
    comparison_rows = _comparison_rows(args.baseline_summary, summary)
    assignment_ids = [
        str(item)
        for item in (summary.get("schedule_assignment_ids") or [])
        if item is not None
    ]
    if not assignment_ids:
        assignment_ids = sorted(
            {
                str(interval.get("assignment_id"))
                for interval in intervals
                if interval.get("assignment_id")
            }
        )
    physical_gpu_ids = [
        str(item)
        for item in (summary.get("physical_gpu_ids") or [])
        if item is not None
    ]
    if not physical_gpu_ids:
        physical_gpu_ids = sorted(
            {
                str(interval.get("gpu_id"))
                for interval in intervals
                if interval.get("gpu_id")
            }
        )
    assignment_count = summary.get("schedule_assignment_count") or len(assignment_ids)
    natural_payload = (
        _load_json(args.natural_length_json.expanduser().resolve())
        if args.natural_length_json is not None
        else None
    )

    _write_csv(tables_root / "first_window_stage_summary.csv", stage_rows)
    _write_csv(tables_root / "first_window_worker_lanes.csv", lane_rows)
    _write_csv(tables_root / "full_run_stage_summary.csv", full_stage_rows)
    _write_csv(tables_root / "full_run_worker_lanes.csv", full_lane_rows)
    _write_csv(tables_root / "chunk_timing_breakdown.csv", chunk_rows)
    _write_csv(tables_root / "comparison_summary.csv", comparison_rows)
    _plot_gpu_timeline(samples, graphs_root / "gpu_vram_timeline.png")
    _plot_stage_overlay(
        samples=samples,
        clipped=clipped,
        window_start=window_start,
        window_min=actual_window_min or float(args.stage_window_min),
        path=graphs_root / "first_window_stage_overlay.png",
    )
    _plot_stage_overlay(
        samples=samples,
        clipped=full_clipped,
        window_start=full_window_start,
        window_min=full_window_min,
        path=graphs_root / "full_run_stage_overlay.png",
        title="Full run: GPU/VRAM with worker lifecycle lanes",
    )

    rollup = {
        "schema_version": "persistent_h100_schedule_run_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "run": {
            "status": summary.get("status"),
            "workers": summary.get("workers"),
            "schedule_assignment_count": assignment_count,
            "schedule_assignment_ids": assignment_ids,
            "physical_gpu_ids": physical_gpu_ids,
            "record_count": summary.get("record_count"),
            "success_count": summary.get("success_count"),
            "total_frames": summary.get("total_frames"),
            "benchmark_wall_sec": summary.get("benchmark_wall_sec"),
            "total_render_sec": summary.get("total_render_sec"),
            "render_overhead_summary": summary.get("render_overhead_summary"),
        },
        "gpu_summary": gpu_summary,
        "requested_first_window_min": float(args.stage_window_min),
        "actual_first_window_min": actual_window_min,
        "actual_full_window_min": full_window_min,
        "first_window_stage_rows": stage_rows,
        "first_window_worker_lanes": lane_rows,
        "full_run_stage_rows": full_stage_rows,
        "full_run_worker_lanes": full_lane_rows,
        "natural_length_projection": natural_payload,
    }
    _write_json(metrics_root / "summary_metrics.json", rollup)

    gpu_util = gpu_summary.get("gpu_util_pct", {}) if isinstance(gpu_summary, Mapping) else {}
    memory = gpu_summary.get("memory_used_gib", {}) if isinstance(gpu_summary, Mapping) else {}
    stage_table_rows = stage_rows[:8]
    lane_table_rows = lane_rows[:12]
    full_stage_table_rows = full_stage_rows[:8]
    full_lane_table_rows = full_lane_rows[:12]
    natural_section = ""
    if natural_payload:
        overall = natural_payload.get("overall", {})
        natural_frames = ((overall.get("natural_frames") or {}).get("sum") if isinstance(overall, Mapping) else None)
        capped_frames = ((overall.get("capped_frames") or {}).get("sum") if isinstance(overall, Mapping) else None)
        if natural_frames and capped_frames:
            natural_section = (
                "\n## Natural-Length Input\n\n"
                f"- capped frames in selected inputs: {int(capped_frames):,}\n"
                f"- natural frames in selected inputs: {int(natural_frames):,}\n"
                f"- natural/capped multiplier: {float(natural_frames) / float(capped_frames):.2f}x\n"
            )
    report = f"""# {args.title}

Generated: {datetime.now().isoformat(timespec="seconds")}

## Scope

- Run root: `{run_root}`
- Workers requested: `{summary.get("workers")}`
- Schedule assignments: `{assignment_count}`
- Assignment ids: `{", ".join(assignment_ids)}`
- Physical GPU ids: `{", ".join(physical_gpu_ids)}`

## Result

| metric | value |
| --- | --- |
| status | {summary.get("status")} |
| chunks success / total | {summary.get("success_count")} / {summary.get("record_count")} |
| frames | {int(summary.get("total_frames") or 0):,} |
| benchmark wall | {_human_seconds(summary.get("benchmark_wall_sec"))} |
| summed render time | {_human_seconds(summary.get("total_render_sec"))} |
| GPU avg / max | {float(gpu_util.get("mean") or 0.0):.2f}% / {float(gpu_util.get("max") or 0.0):.2f}% |
| GPU samples >=80% | {float(gpu_summary.get("samples_ge80_pct") or 0.0):.2f}% |
| peak VRAM | {float(memory.get("max") or 0.0):.2f} GiB |

## First {(actual_window_min or float(args.stage_window_min)):.2f}-Minute Stage Diagnostic

[Stage overlay](assets/graphs/first_window_stage_overlay.png) shows worker lifecycle lanes under the GPU/VRAM trace. Stage durations are worker-seconds, so totals can exceed wall time when multiple workers overlap.

{_markdown_table(stage_table_rows, ["stage_group", "worker_seconds_in_window", "worker_seconds_per_window_sec", "sample_count_when_active", "avg_gpu_pct_when_active", "max_gpu_pct_when_active"])}

## Full-Run Stage Diagnostic

[Full-run stage overlay](assets/graphs/full_run_stage_overlay.png) shows the whole GPU/VRAM trace with one lifecycle lane per worker.

{_markdown_table(full_stage_table_rows, ["stage_group", "worker_seconds_in_window", "worker_seconds_per_window_sec", "sample_count_when_active", "avg_gpu_pct_when_active", "max_gpu_pct_when_active"])}

## Worker Lanes

{_markdown_table(lane_table_rows, ["assignment_id", "chunk_count_seen", "worker_seconds", "render_loop_sec", "actor_load_sec", "write_close_sec", "scene_renderer_init_sec"])}

## Full-Run Worker Lanes

{_markdown_table(full_lane_table_rows, ["assignment_id", "chunk_count_seen", "worker_seconds", "render_loop_sec", "actor_load_sec", "write_close_sec", "scene_renderer_init_sec"])}

## Comparison

{_markdown_table(comparison_rows, ["run", "status", "frames", "wall_sec", "frames_per_wall_sec", "gpu_avg_pct", "gpu_max_pct", "renderer_processes", "setup_sec_per_frame", "launches_per_1000_frames"])}
{natural_section}
## Artifacts

- [GPU/VRAM timeline](assets/graphs/gpu_vram_timeline.png)
- [First-window stage overlay](assets/graphs/first_window_stage_overlay.png)
- [Full-run stage overlay](assets/graphs/full_run_stage_overlay.png)
- [summary_metrics.json](metrics/summary_metrics.json)
- [first_window_stage_summary.csv](assets/tables/first_window_stage_summary.csv)
- [first_window_worker_lanes.csv](assets/tables/first_window_worker_lanes.csv)
- [full_run_stage_summary.csv](assets/tables/full_run_stage_summary.csv)
- [full_run_worker_lanes.csv](assets/tables/full_run_worker_lanes.csv)
- [chunk_timing_breakdown.csv](assets/tables/chunk_timing_breakdown.csv)
- [comparison_summary.csv](assets/tables/comparison_summary.csv)
"""
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    print(output_root / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
