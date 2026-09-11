#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


JsonDict = dict[str, Any]
ACTION_NAMES = ("move", "stop", "turn_left", "turn_right")
ACTION_COLORS = {
    "move": "#4D908E",
    "stop": "#C33C54",
    "turn_left": "#6C5CE7",
    "turn_right": "#E07A3F",
}
EXPERIMENT_COLORS = {
    "budget_2": "#6C757D",
    "budget_4": "#3F88C5",
    "budget_8": "#087E8B",
    "event_aware": "#C33C54",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a completed POI policy matrix survey.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--representative-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif-fps", type=int, default=6)
    return parser.parse_args()


def _read(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _pretty(value: str) -> str:
    return value.replace("_", " ").replace(":", ": ").title()


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _write_action_graph(payload: Mapping[str, Any], output_dir: Path) -> None:
    experiments = payload["experiments"]
    populations = ("all_source_frames", "poi_centers", "independent_windows", "unique_retained_frames")
    fig, axes = plt.subplots(len(experiments), 1, figsize=(13, 2.8 * len(experiments)), dpi=120)
    axes = np.atleast_1d(axes)
    for ax, experiment in zip(axes, experiments):
        bottom = np.zeros(len(populations), dtype=float)
        for action in ACTION_NAMES:
            values = np.asarray([
                float(experiment["totals"]["actions"][population]["ratios"].get(action, 0.0))
                for population in populations
            ])
            ax.barh(range(len(populations)), values, left=bottom, color=ACTION_COLORS[action], label=_pretty(action))
            bottom += values
        ax.set_yticks(range(len(populations)), [_pretty(item) for item in populations])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(_pretty(str(experiment["experiment"])), loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Action distributions by policy and retained population", x=0.13, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "action_distributions.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_episode_graph(payload: Mapping[str, Any], output_dir: Path) -> None:
    experiments = payload["experiments"]
    signals = sorted({
        signal
        for experiment in experiments
        for signal in experiment["totals"]["navigation_signals"]["eligible_center_episodes"]
    })
    if not signals:
        return
    fig, ax = plt.subplots(figsize=(15, max(7.0, len(signals) * 0.55)), dpi=120)
    y = np.arange(len(signals), dtype=float)
    width = 0.8 / max(1, len(experiments))
    for index, experiment in enumerate(experiments):
        nav = experiment["totals"]["navigation_signals"]
        values = [
            float(nav["poi_covered_episodes"].get(signal, 0))
            / max(1, int(nav["eligible_center_episodes"].get(signal, 0)))
            for signal in signals
        ]
        ax.barh(
            y + (index - (len(experiments) - 1) / 2) * width,
            values,
            height=width * 0.92,
            color=EXPERIMENT_COLORS.get(str(experiment["experiment"]), "#6C757D"),
            label=_pretty(str(experiment["experiment"])),
        )
    ax.set_yticks(y, [_pretty(signal) for signal in signals])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Eligible contiguous episodes covered by at least one scored POI")
    ax.set_title("Semantic episode coverage", loc="left", fontsize=17, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, ncol=min(4, len(experiments)), loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_episode_coverage.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_navigation_signal_graph(payload: Mapping[str, Any], output_dir: Path) -> None:
    experiments = payload["experiments"]
    signals = sorted({
        signal
        for experiment in experiments
        for signal in experiment["totals"]["navigation_signals"]["all_source_frames"]
    })
    if not signals:
        return
    fig, axes = plt.subplots(1, 2, figsize=(17, max(7.0, len(signals) * 0.55)), dpi=120)
    y = np.arange(len(signals), dtype=float)
    source = experiments[0]["totals"]
    source_frames = max(1, int(source["source_frame_count"]))
    source_values = [
        int(source["navigation_signals"]["all_source_frames"].get(signal, 0)) / source_frames
        for signal in signals
    ]
    axes[0].barh(y, source_values, color="#6C757D")
    axes[0].set_title("All sampled source frames", loc="left", fontweight="bold")
    axes[0].set_xlabel("Signal-bearing frames / source frames")
    width = 0.8 / max(1, len(experiments))
    for index, experiment in enumerate(experiments):
        totals = experiment["totals"]
        poi_count = max(1, int(totals["selected_poi_count"]))
        values = [
            int(totals["navigation_signals"]["poi_center_frames"].get(signal, 0)) / poi_count
            for signal in signals
        ]
        axes[1].barh(
            y + (index - (len(experiments) - 1) / 2) * width,
            values,
            height=width * 0.92,
            color=EXPERIMENT_COLORS.get(str(experiment["experiment"]), "#6C757D"),
            label=_pretty(str(experiment["experiment"])),
        )
    axes[1].set_title("Scored POI centers", loc="left", fontweight="bold")
    axes[1].set_xlabel("Signal-bearing POIs / POI centers (signals may overlap)")
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_yticks(y, [_pretty(signal) for signal in signals])
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("Navigation-signal frame distributions", x=0.18, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "navigation_signal_distributions.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_volume_graph(payload: Mapping[str, Any], output_dir: Path) -> None:
    experiments = payload["experiments"]
    names = [str(item["experiment"]) for item in experiments]
    retained = [float(item["totals"]["retention"]["unique_retained_frame_percent"]) for item in experiments]
    poi_per_path = [
        float(item["totals"]["selected_poi_count"]) / max(1, int(item["totals"]["sampled_source_path_count"]))
        for item in experiments
    ]
    critical_signals = (
        "instruction_turn", "semantic_stop", "room_transition", "collision_avoidance",
        "social_law_decision", "human_interaction_decision", "traffic_wait",
        "target_human_visible", "route_deviation", "planned_execution_mismatch",
    )
    mean_coverage = []
    for item in experiments:
        nav = item["totals"]["navigation_signals"]
        values = [
            int(nav["poi_covered_episodes"].get(signal, 0))
            / max(1, int(nav["eligible_center_episodes"].get(signal, 0)))
            for signal in critical_signals
            if int(nav["eligible_center_episodes"].get(signal, 0)) > 0
        ]
        mean_coverage.append(100.0 * sum(values) / max(1, len(values)))
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    for name, x, y, size in zip(names, retained, mean_coverage, poi_per_path):
        ax.scatter(x, y, s=100 + 35 * size, color=EXPERIMENT_COLORS.get(name, "#6C757D"), edgecolor="white", linewidth=1.5)
        ax.annotate(f"{_pretty(name)}\n{size:.1f} POIs/path", (x, y), xytext=(7, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Unique retained source frames (%)")
    ax.set_ylabel("Mean critical semantic-episode coverage (%)")
    ax.set_title("Coverage versus retained volume", loc="left", fontsize=17, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_vs_volume.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _selected_sets(experiment: Mapping[str, Any], job_id: str) -> tuple[set[int], set[int], set[int]]:
    retained: set[int] = set()
    pois: set[int] = set()
    anchors: set[int] = set()
    for chunk in experiment.get("chunks", []):
        if str(chunk.get("source_job_id") or "") != job_id:
            continue
        retained.update(int(value) for value in chunk.get("source_frame_indices", []))
        target = int(chunk.get("target_frame", 0))
        if str(chunk.get("target_role")) == "mandatory_anchor":
            anchors.add(target)
        else:
            pois.add(target)
    return retained, pois, anchors


def _draw_path(ax: plt.Axes, representative: Mapping[str, Any], experiment_name: str, progress: float = 1.0) -> None:
    path = representative["source_path"]
    points = path["points"]
    xy = np.asarray([point["xy"] for point in points], dtype=float)
    experiment = representative["experiments"][experiment_name]
    retained, pois, anchors = _selected_sets(experiment, str(path["source_job_id"]))
    reveal = max(1, int(math.ceil(len(points) * progress)))
    shown_xy = xy[:reveal]
    ax.plot(xy[:, 0], xy[:, 1], color="#B8C0C8", linewidth=1.2, label="Full original path", zorder=1)
    ax.plot(shown_xy[:, 0], shown_xy[:, 1], color="#49545E", linewidth=1.8, alpha=0.8, zorder=2)
    retained_now = sorted(index for index in retained if index < reveal and 0 <= index < len(xy))
    if retained_now:
        ax.scatter(xy[retained_now, 0], xy[retained_now, 1], s=11, color="#55C2FF", alpha=0.75, label="Retained frames/windows", zorder=3)
    pois_now = sorted(index for index in pois if index < reveal and 0 <= index < len(xy))
    if pois_now:
        ax.scatter(xy[pois_now, 0], xy[pois_now, 1], s=55, marker="o", color="#087E8B", edgecolor="white", linewidth=0.8, label="Scored POIs", zorder=5)
    anchors_now = sorted(index for index in anchors if index < reveal and 0 <= index < len(xy))
    if anchors_now:
        ax.scatter(xy[anchors_now, 0], xy[anchors_now, 1], s=75, marker="D", color="#C33C54", edgecolor="white", linewidth=0.8, label="Mandatory endpoints/checkpoints", zorder=6)
    if reveal > 0:
        ax.scatter(shown_xy[-1, 0], shown_xy[-1, 1], s=35, color="#111827", marker=">", zorder=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(_pretty(experiment_name), loc="left", fontweight="bold")
    ax.set_xlabel("BEV x (m)")
    ax.set_ylabel("BEV y (m)")
    ax.grid(alpha=0.15)


def _write_representative(representative_path: Path, output_dir: Path, gif_fps: int) -> JsonDict:
    representative = _read(representative_path)
    cohort = representative["cohort"]
    name = _safe(f"{cohort['corpus']}__{cohort['family']}__{cohort['dataset']}__{cohort['scene']}")
    target_dir = output_dir / "representatives" / name
    target_dir.mkdir(parents=True, exist_ok=True)
    experiments = tuple(representative["experiments"])
    cols = 2
    rows = int(math.ceil(len(experiments) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows), dpi=120)
    axes = np.atleast_1d(axes).ravel()
    for ax, experiment in zip(axes, experiments):
        _draw_path(ax, representative, experiment)
    for ax in axes[len(experiments):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(f"{cohort['family']} — {cohort['corpus']} / {cohort['scene']}", x=0.08, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = target_dir / "bev_policy_comparison.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    chosen = "event_aware" if "event_aware" in experiments else experiments[-1]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=100)
    frame_count = 24
    def update(frame: int) -> None:
        ax.clear()
        _draw_path(ax, representative, chosen, min(1.0, (frame + 1) / frame_count))
        ax.legend(frameon=False, loc="best")
    animation = FuncAnimation(fig, update, frames=frame_count + 4, interval=1000 / max(1, gif_fps))
    gif_path = target_dir / "bev_event_aware.gif"
    animation.save(gif_path, writer=PillowWriter(fps=max(1, gif_fps)))
    plt.close(fig)

    instruction_path = target_dir / "instructions_and_metadata.json"
    instruction_path.write_text(
        json.dumps({
            "cohort": cohort,
            "source_path": {key: value for key, value in representative["source_path"].items() if key != "points"},
            "instructions": representative.get("instructions", {}),
            "temporal_contract": {
                "past_frames": 32,
                "current_frames": 1,
                "future_frames": 0,
                "target_window_index": 32,
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"cohort": cohort, "png": str(png_path), "gif": str(gif_path), "metadata": str(instruction_path)}


def main() -> int:
    args = _parse_args()
    payload = _read(args.summary_json.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_action_graph(payload, output_dir)
    _write_navigation_signal_graph(payload, output_dir)
    _write_episode_graph(payload, output_dir)
    _write_volume_graph(payload, output_dir)
    representatives = [
        _write_representative(path, output_dir, int(args.gif_fps))
        for path in sorted(args.representative_dir.resolve().glob("*.json"))
    ]
    summary = {
        "schema_version": "navdp_poi_policy_matrix_visualization/v1.0",
        "summary_json": str(args.summary_json.resolve()),
        "action_distribution_graph": str(output_dir / "action_distributions.png"),
        "navigation_signal_distribution_graph": str(output_dir / "navigation_signal_distributions.png"),
        "semantic_episode_coverage_graph": str(output_dir / "semantic_episode_coverage.png"),
        "coverage_volume_graph": str(output_dir / "coverage_vs_volume.png"),
        "representatives": representatives,
        "gaussian_rgb_rendering_executed": False,
    }
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
