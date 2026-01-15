#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _iter_records(payload: dict) -> list[dict]:
    if isinstance(payload, dict) and isinstance(payload.get("paths"), list):
        return payload["paths"]
    if isinstance(payload, dict) and "frames" in payload and "duration_sec" in payload:
        return [payload]
    return []


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-path metrics JSON files.")
    parser.add_argument("--metrics-dir", type=Path, required=True, help="Directory with metrics JSON files.")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for metrics files.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    metrics_dir = args.metrics_dir
    if not metrics_dir.is_dir():
        raise SystemExit(f"Metrics directory not found: {metrics_dir}")

    files = sorted(metrics_dir.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No metrics files matched {args.pattern} under {metrics_dir}")

    fps_values: list[float] = []
    durations: list[float] = []
    frames: list[int] = []
    stage_totals: dict[str, float] = {}
    path_count = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = _iter_records(payload)
        for record in records:
            frames_val = int(record.get("frames", 0) or 0)
            duration_val = float(record.get("duration_sec", 0.0) or 0.0)
            fps_val = record.get("frames_per_sec", None)
            if fps_val is None and frames_val > 0 and duration_val > 0:
                fps_val = frames_val / duration_val
            if fps_val is not None:
                fps_values.append(float(fps_val))
            if duration_val > 0:
                durations.append(duration_val)
            if frames_val > 0:
                frames.append(frames_val)
            stage_seconds = record.get("stage_seconds", {}) or {}
            for key, value in stage_seconds.items():
                stage_totals[key] = stage_totals.get(key, 0.0) + float(value or 0.0)
            path_count += 1

    total_frames = int(sum(frames))
    total_duration = float(sum(durations))
    fps_overall = (total_frames / total_duration) if total_duration > 0 else None
    fps_mean = (sum(fps_values) / len(fps_values)) if fps_values else None

    stage_ratios = {}
    if total_duration > 0:
        stage_ratios = {k: v / total_duration for k, v in stage_totals.items()}

    report = {
        "metrics_dir": str(metrics_dir),
        "files": len(files),
        "paths": path_count,
        "total_frames": total_frames,
        "total_duration_sec": total_duration,
        "fps_overall": fps_overall,
        "fps_mean": fps_mean,
        "fps_median": _percentile(fps_values, 50),
        "fps_p10": _percentile(fps_values, 10),
        "fps_p90": _percentile(fps_values, 90),
        "stage_seconds": stage_totals,
        "stage_ratios": stage_ratios,
    }

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
