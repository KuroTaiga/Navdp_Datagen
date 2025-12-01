#!/usr/bin/env python3
"""Summarize navigation dataset statistics and emit charts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
import os


_MPL_CACHE = Path(__file__).resolve().parent / ".matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

try:  # matplotlib is only needed for chart generation
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover - matplotlib is optional
    plt = None
    _HAVE_MPL = False


DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "data" / "selected_33w"


@dataclass(frozen=True)
class PathMetrics:
    scene_id: str
    label_id: str
    label_name: str | None
    file_path: Path
    path_length: float
    keypoint_count: int
    raster_steps: int | None
    displacement: float
    tortuosity: float | None


def resolve_label_directory(scene_dir: Path) -> Path | None:
    """Mirror the logic used in render_label_paths.py."""

    label_dir = scene_dir / "label_paths"
    if label_dir.is_dir():
        return label_dir
    if scene_dir.is_dir() and any(scene_dir.glob("*.json")):
        return scene_dir
    return None


def _percentile(sorted_vals: Sequence[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(sorted_vals[int(pos)])
    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    fraction = pos - lower
    return float(lower_val + (upper_val - lower_val) * fraction)


def describe(values: Iterable[float]) -> dict:
    seq = [float(v) for v in values if v is not None]
    if not seq:
        return {"count": 0}
    sorted_seq = sorted(seq)
    result = {
        "count": len(sorted_seq),
        "min": float(sorted_seq[0]),
        "max": float(sorted_seq[-1]),
        "mean": statistics.fmean(sorted_seq),
        "median": statistics.median(sorted_seq),
        "p10": _percentile(sorted_seq, 0.10),
        "p90": _percentile(sorted_seq, 0.90),
    }
    if len(sorted_seq) > 1:
        result["stddev"] = statistics.pstdev(sorted_seq)
    return result


def load_path_metrics(scene_id: str, json_path: Path) -> PathMetrics | None:
    try:
        payload = json.loads(json_path.read_text())
    except Exception as exc:  # pragma: no cover - IO errors are recorded elsewhere
        raise RuntimeError(f"Failed to read {json_path}: {exc}") from exc

    path_section = payload.get("path", {}) or {}
    keypoints_world = path_section.get("keypoints_world") or []
    coords: list[tuple[float, float]] = []
    for entry in keypoints_world:
        x = entry.get("x")
        y = entry.get("y")
        if x is None or y is None:
            continue
        coords.append((float(x), float(y)))

    if len(coords) < 2:
        raise RuntimeError(f"Path {json_path} does not contain enough keypoints to measure length.")

    path_length = sum(
        math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords[:-1], coords[1:])
    )
    displacement = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
    tortuosity = (path_length / displacement) if displacement > 0 else None

    raster_steps = None
    raster_pixels = path_section.get("raster_pixel")
    if isinstance(raster_pixels, list):
        raster_steps = len(raster_pixels)

    label_payload = payload.get("label", {}) or {}
    label_id = str(label_payload.get("ins_id", json_path.stem))
    label_name = label_payload.get("label")

    return PathMetrics(
        scene_id=scene_id,
        label_id=label_id,
        label_name=label_name,
        file_path=json_path,
        path_length=path_length,
        keypoint_count=len(coords),
        raster_steps=raster_steps,
        displacement=displacement,
        tortuosity=tortuosity,
    )


def build_charts(
    *,
    output_dir: Path,
    path_lengths: list[float],
    scene_summaries: list[dict],
    top_scenes: int,
    hist_bins: int,
) -> dict[str, str]:
    if not _HAVE_MPL:
        return {}

    charts: dict[str, str] = {}
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if path_lengths:
        plt.figure(figsize=(8, 4.5))
        plt.hist(path_lengths, bins=hist_bins, color="#3b82f6", edgecolor="black", alpha=0.8)
        plt.title("Path length distribution")
        plt.xlabel("Path length (world units)")
        plt.ylabel("Count")
        plt.grid(alpha=0.2, linestyle="--")
        hist_path = charts_dir / "path_length_histogram.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        charts["path_length_histogram"] = str(hist_path)

    if scene_summaries:
        limited = sorted(scene_summaries, key=lambda item: item["path_count"], reverse=True)[:top_scenes]
        if limited:
            plt.figure(figsize=(10, 4.5))
            labels = [entry["scene_id"] for entry in limited]
            counts = [entry["path_count"] for entry in limited]
            plt.bar(range(len(limited)), counts, color="#10b981")
            plt.xticks(range(len(limited)), labels, rotation=60, ha="right", fontsize=8)
            plt.ylabel("Path count")
            plt.title(f"Top {len(limited)} scenes by path count")
            plt.tight_layout()
            top_path = charts_dir / "scene_path_count_top.png"
            plt.savefig(top_path, dpi=150)
            plt.close()
            charts["top_scenes_by_path_count"] = str(top_path)

        avg_lengths = [entry["path_length_stats"].get("mean") for entry in scene_summaries if entry["path_count"] > 0]
        counts_all = [entry["path_count"] for entry in scene_summaries if entry["path_count"] > 0]
        if avg_lengths and counts_all and len(avg_lengths) == len(counts_all):
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(counts_all, avg_lengths, alpha=0.5, color="#f97316")
            plt.xlabel("Path count per scene")
            plt.ylabel("Average path length (world units)")
            plt.title("Path count vs. average length per scene")
            plt.grid(alpha=0.2, linestyle=":")
            plt.tight_layout()
            scatter_path = charts_dir / "count_vs_avg_length.png"
            plt.savefig(scatter_path, dpi=150)
            plt.close()
            charts["path_count_vs_avg_length"] = str(scatter_path)

    return charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NavDP data generation tasks directory")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="Directory that contains per-scene task folders (default: ./data/selected_33w).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./analysis/datagen"),
        help="Directory where analysis JSON and charts will be written.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of bins for the path length histogram.",
    )
    parser.add_argument(
        "--top-scenes",
        type=int,
        default=20,
        help="Number of scenes to highlight in the top-N chart.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="datagen_analysis.json",
        help="Base filename for the JSON report.",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Optional limit on the number of scenes to scan (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_dir = args.tasks_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")

    per_scene_files: dict[str, list[Path]] = {}
    for scene_idx, scene_dir in enumerate(sorted(tasks_dir.iterdir()), start=1):
        if not scene_dir.is_dir():
            continue
        if args.limit_scenes and len(per_scene_files) >= args.limit_scenes:
            break
        label_dir = resolve_label_directory(scene_dir)
        if label_dir is None:
            continue
        json_files = sorted(label_dir.glob("*.json"))
        if json_files:
            per_scene_files[scene_dir.name] = json_files

    path_records: list[PathMetrics] = []
    failures: list[dict[str, str]] = []

    for scene_id, files in per_scene_files.items():
        for json_path in files:
            try:
                metrics = load_path_metrics(scene_id, json_path)
            except Exception as exc:  # pragma: no cover - logged for later review
                failures.append({"scene_id": scene_id, "file": str(json_path), "error": str(exc)})
                continue
            path_records.append(metrics)

    paths_by_scene: dict[str, list[PathMetrics]] = defaultdict(list)
    for record in path_records:
        paths_by_scene[record.scene_id].append(record)

    scene_summaries: list[dict] = []
    for scene_id in sorted(paths_by_scene):
        scene_paths = paths_by_scene[scene_id]
        lengths = [item.path_length for item in scene_paths]
        displacements = [item.displacement for item in scene_paths]
        raster_steps = [item.raster_steps for item in scene_paths if item.raster_steps is not None]
        tortuosity = [item.tortuosity for item in scene_paths if item.tortuosity is not None]
        summary = {
            "scene_id": scene_id,
            "path_count": len(scene_paths),
            "path_length_stats": describe(lengths),
            "displacement_stats": describe(displacements),
            "raster_step_stats": describe(raster_steps),
            "tortuosity_stats": describe(tortuosity),
        }
        scene_summaries.append(summary)

    path_lengths = [item.path_length for item in path_records]
    displacements_all = [item.displacement for item in path_records]
    raster_steps_all = [item.raster_steps for item in path_records if item.raster_steps is not None]
    tortuosity_all = [item.tortuosity for item in path_records if item.tortuosity is not None]

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks_dir": str(tasks_dir),
        "output_dir": str(output_dir),
        "scene_count": len(scene_summaries),
        "total_paths": len(path_records),
        "overall_path_length_stats": describe(path_lengths),
        "overall_displacement_stats": describe(displacements_all),
        "overall_raster_step_stats": describe(raster_steps_all),
        "overall_tortuosity_stats": describe(tortuosity_all),
        "scene_path_count_stats": describe([entry["path_count"] for entry in scene_summaries]),
        "per_scene": scene_summaries,
        "failures": failures,
    }

    longest_paths = sorted(path_records, key=lambda item: item.path_length, reverse=True)[:10]
    analysis["longest_paths"] = [
        {
            "scene_id": record.scene_id,
            "label_id": record.label_id,
            "label_name": record.label_name,
            "length": record.path_length,
            "file": str(record.file_path),
        }
        for record in longest_paths
    ]

    shortest_paths = sorted(path_records, key=lambda item: item.path_length)[:10]
    analysis["shortest_paths"] = [
        {
            "scene_id": record.scene_id,
            "label_id": record.label_id,
            "label_name": record.label_name,
            "length": record.path_length,
            "file": str(record.file_path),
        }
        for record in shortest_paths
    ]

    charts = build_charts(
        output_dir=output_dir,
        path_lengths=path_lengths,
        scene_summaries=scene_summaries,
        top_scenes=args.top_scenes,
        hist_bins=args.hist_bins,
    )
    analysis["charts"] = charts

    json_path = output_dir / args.json_name
    json_path.write_text(json.dumps(analysis, indent=2))

    def fmt_stat(stat_key: str, stats_dict: dict) -> str:
        if stats_dict.get("count", 0) == 0:
            return "n/a"
        value = stats_dict.get(stat_key)
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    length_stats = analysis["overall_path_length_stats"]
    count_stats = analysis["scene_path_count_stats"]

    print(f"Analyzed {analysis['scene_count']} scene(s) with {analysis['total_paths']} path(s) from {tasks_dir}.")
    if analysis["scene_count"]:
        print(
            "Paths per scene -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", count_stats),
                median=fmt_stat("median", count_stats),
                mean=fmt_stat("mean", count_stats),
                max=fmt_stat("max", count_stats),
            )
        )
    if length_stats.get("count", 0):
        print(
            "Path length (world units) -> min={min}, mid={median}, ave={mean}, max={max}".format(
                min=fmt_stat("min", length_stats),
                median=fmt_stat("median", length_stats),
                mean=fmt_stat("mean", length_stats),
                max=fmt_stat("max", length_stats),
            )
        )
    if charts:
        print("Charts written to: ")
        for label, path in charts.items():
            print(f"  - {label}: {path}")
    if failures:
        print(f"Encountered {len(failures)} unreadable/invalid path file(s); see JSON for details.")

    print(f"Full report: {json_path}")


if __name__ == "__main__":
    main()
