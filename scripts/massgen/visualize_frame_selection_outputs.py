#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


JsonDict = dict[str, Any]

SOURCE_COLOR = "#697681"
UNSELECTED_SOURCE_COLOR = "#C7CED4"
WINDOW_COLORS = ("#087E8B", "#E07A3F", "#6C5CE7", "#2A9D5B")
CENTER_COLOR = "#C33C54"
ANCHOR_COLOR = "#D99000"
HUMAN_COLOR = "#3A7D44"
BUCKET_COLORS = {
    "critical_margin": "#C33C54",
    "human_interaction": "#E07A3F",
    "route_decision": "#6C5CE7",
    "representative_motion": "#087E8B",
}
ACTION_COLORS = {
    "move": "#087E8B",
    "stop": "#C33C54",
    "turn_left": "#E07A3F",
    "turn_right": "#6C5CE7",
}
ROLE_COLORS = {
    "frame_of_interest": "#087E8B",
    "mandatory_anchor": "#C33C54",
}
NAVIGATION_SIGNAL_COLORS = {
    "instruction_turn": "#6C5CE7",
    "semantic_stop": "#C33C54",
    "instruction_section_boundary": "#3F88C5",
    "room_transition": "#087E8B",
    "collision_avoidance": "#D1495B",
    "social_law_decision": "#E07A3F",
    "human_interaction_decision": "#2A9D5B",
    "traffic_wait": "#D99000",
    "mission_endpoint": "#7A5195",
    "relevant_human_visible": "#4D908E",
    "target_human_visible": "#277DA1",
    "route_deviation": "#B56576",
    "planned_execution_mismatch": "#8C564B",
}


@dataclass(frozen=True)
class PathSeries:
    series_id: str
    points: tuple[tuple[float, float], ...]
    target_bucket: str | None = None
    target_action: str | None = None
    target_frame: int | None = None
    target_window_index: int | None = None
    target_role: str | None = None


@dataclass(frozen=True)
class FamilyVisualization:
    family: str
    source_paths: tuple[PathSeries, ...]
    unselected_source_paths: tuple[PathSeries, ...]
    selected_paths: tuple[PathSeries, ...]
    human_paths: tuple[PathSeries, ...]
    candidate_count: int
    selected_count: int
    unique_frame_count: int
    available_buckets: Mapping[str, int]
    selected_buckets: Mapping[str, int]
    available_actions: Mapping[str, int]
    selected_actions: Mapping[str, int]
    retained_window_actions: Mapping[str, int]
    selected_roles: Mapping[str, int]
    available_navigation_signals: Mapping[str, int]
    selected_navigation_signals: Mapping[str, int]
    available_navigation_signal_episodes: Mapping[str, int]
    selected_interest_navigation_signal_episodes: Mapping[str, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize source paths, selected frame windows, and sampling distributions."
    )
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest-root",
        type=Path,
        action="append",
        default=None,
        help="Additional local directory used to resolve source-manifest basenames. Repeatable.",
    )
    parser.add_argument("--gif-fps", type=int, default=5)
    parser.add_argument("--reveal-frames", type=int, default=12)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument(
        "--per-family",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a GIF and final-frame PNG for each ready family.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or "family"


def _points(trajectory: Any) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for point in trajectory if isinstance(trajectory, list) else []:
        if not isinstance(point, Mapping):
            continue
        position = point.get("position", [])
        if not isinstance(position, Sequence) or len(position) < 2:
            continue
        try:
            x = float(position[0])
            y = float(position[1])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            out.append((x, y))
    return tuple(out)


def _local_manifest_path(
    pilot_root: Path,
    remote_path: str,
    source_manifest_roots: Sequence[Path],
) -> Path:
    candidate = Path(remote_path)
    if candidate.is_file():
        return candidate
    for root in (pilot_root / "source_render_manifests", *source_manifest_roots):
        local = root / candidate.name
        if local.is_file():
            return local
    raise FileNotFoundError(f"cannot resolve source manifest: {remote_path}")


def _family_data(
    pilot_root: Path,
    record: Mapping[str, Any],
    source_manifest_roots: Sequence[Path],
) -> FamilyVisualization:
    family = str(record["mission_family"])
    family_root = pilot_root / _safe_id(family)
    selection = _read_json(family_root / "selection.json")
    selection_config = selection.get("config", {})
    selection_config = selection_config if isinstance(selection_config, Mapping) else {}
    selected_source_job_ids = {
        str(item.get("source_job_id"))
        for item in selection.get("distribution", {})
        .get("source_path_sampling", {})
        .get("selected_source_paths", [])
        if isinstance(item, Mapping)
    }
    source_paths: list[PathSeries] = []
    unselected_source_paths: list[PathSeries] = []
    human_paths: list[PathSeries] = []
    for raw_path in record.get("matched_source_manifests", []):
        manifest = _read_json(
            _local_manifest_path(pilot_root, str(raw_path), source_manifest_roots)
        )
        for job in manifest.get("jobs", []):
            if not isinstance(job, Mapping):
                continue
            camera = job.get("camera", {})
            trajectory = camera.get("trajectory", []) if isinstance(camera, Mapping) else []
            points = _points(trajectory)
            if points:
                series = PathSeries(str(job.get("job_id", "source_job")), points)
                if not selected_source_job_ids or series.series_id in selected_source_job_ids:
                    source_paths.append(series)
                else:
                    unselected_source_paths.append(series)
        actors = manifest.get("actors", {})
        humans = actors.get("humans", []) if isinstance(actors, Mapping) else []
        for actor in humans:
            if not isinstance(actor, Mapping):
                continue
            points = _points(actor.get("trajectory", []))
            if points:
                human_paths.append(PathSeries(str(actor.get("actor_id", "human")), points))

    selected_paths: list[PathSeries] = []
    render_manifest_root = family_root / "render_manifests"
    for manifest_path in sorted(render_manifest_root.glob("*.json")):
        manifest = _read_json(manifest_path)
        for job in manifest.get("jobs", []):
            if not isinstance(job, Mapping):
                continue
            camera = job.get("camera", {})
            trajectory = camera.get("trajectory", []) if isinstance(camera, Mapping) else []
            points = _points(trajectory)
            if not points:
                continue
            frame_selection = job.get("frame_selection", {})
            frame_selection = frame_selection if isinstance(frame_selection, Mapping) else {}
            target_frame = frame_selection.get("target_source_frame_id")
            target_window_index = frame_selection.get("target_window_index")
            if target_window_index is None:
                target_window_index = selection_config.get("past_frames")
            selected_paths.append(
                PathSeries(
                    str(job.get("job_id", "selected_window")),
                    points,
                    target_bucket=str(frame_selection.get("target_bucket", "unknown")),
                    target_action=str(frame_selection.get("target_action_name", "unknown")),
                    target_frame=int(target_frame) if target_frame is not None else None,
                    target_window_index=(
                        int(target_window_index) if target_window_index is not None else None
                    ),
                    target_role=str(frame_selection.get("target_role", "frame_of_interest")),
                )
            )

    summary = selection.get("selection_summary", {})
    distribution = selection.get("distribution", {})
    mandatory_count = int(summary.get("mandatory_anchor_count", 0))
    selected_count = int(summary.get("selected_target_count", 0))
    return FamilyVisualization(
        family=family,
        source_paths=tuple(source_paths),
        unselected_source_paths=tuple(unselected_source_paths),
        selected_paths=tuple(selected_paths),
        human_paths=tuple(human_paths),
        candidate_count=int(summary.get("candidate_count", 0)),
        selected_count=selected_count,
        unique_frame_count=int(summary.get("unique_source_render_frame_count", 0)),
        available_buckets=dict(distribution.get("available_bucket_counts", {})),
        selected_buckets=dict(summary.get("selected_bucket_counts", {})),
        available_actions=dict(distribution.get("available_action_counts", {})),
        selected_actions=dict(
            summary.get("selected_center_action_counts", summary.get("selected_action_counts", {}))
        ),
        retained_window_actions=dict(summary.get("selected_window_action_counts", {})),
        selected_roles={
            "frame_of_interest": max(0, selected_count - mandatory_count),
            "mandatory_anchor": mandatory_count,
        },
        available_navigation_signals=dict(
            distribution.get("available_navigation_signal_counts", {})
        ),
        selected_navigation_signals=dict(
            summary.get("selected_navigation_signal_counts", {})
        ),
        available_navigation_signal_episodes=dict(
            distribution.get("available_navigation_signal_episode_counts", {})
        ),
        selected_interest_navigation_signal_episodes=dict(
            summary.get("selected_interest_navigation_signal_episode_counts", {})
        ),
    )


def load_pilot(
    pilot_root: Path,
    *,
    source_manifest_roots: Sequence[Path] = (),
) -> tuple[FamilyVisualization, ...]:
    plan = _read_json(pilot_root / "pilot_plan.json")
    records = [
        record
        for record in plan.get("families", [])
        if isinstance(record, Mapping) and record.get("status") == "ready"
    ]
    families = tuple(
        _family_data(pilot_root, record, source_manifest_roots)
        for record in records
    )
    if not families:
        raise ValueError("pilot plan has no ready mission-family records")
    return families


def _pretty(value: str) -> str:
    return str(value).replace("navigate_with_social_constraints:", "social:").replace("_", " ")


def _bounds(family: FamilyVisualization) -> tuple[float, float, float, float]:
    points = [
        point
        for series in (
            *family.source_paths,
            *family.unselected_source_paths,
            *family.selected_paths,
            *family.human_paths,
        )
        for point in series.points
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if not xs or not ys:
        return (-1.0, 1.0, -1.0, 1.0)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * 0.08, 0.5)
    ypad = max((ymax - ymin) * 0.08, 0.5)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _draw_bev(ax: plt.Axes, family: FamilyVisualization, fraction: float) -> None:
    ax.clear()
    for series in family.unselected_source_paths:
        xy = np.asarray(series.points)
        ax.plot(xy[:, 0], xy[:, 1], color=UNSELECTED_SOURCE_COLOR, linewidth=1.2, alpha=0.65)
    for series in family.source_paths:
        xy = np.asarray(series.points)
        ax.plot(xy[:, 0], xy[:, 1], color=SOURCE_COLOR, linewidth=2.0, alpha=0.75)
        ax.scatter(xy[0, 0], xy[0, 1], color=SOURCE_COLOR, marker="o", s=22, zorder=4)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=SOURCE_COLOR, marker="s", s=24, zorder=4)
    for series in family.human_paths:
        xy = np.asarray(series.points)
        ax.plot(xy[:, 0], xy[:, 1], color=HUMAN_COLOR, linewidth=1.3, linestyle="--", alpha=0.8)
        ax.scatter(xy[0, 0], xy[0, 1], color=HUMAN_COLOR, marker="^", s=28, zorder=4)
    for index, series in enumerate(family.selected_paths):
        xy = np.asarray(series.points)
        if fraction > 0.0:
            point_count = max(2, min(len(xy), int(np.ceil(len(xy) * fraction))))
            ax.plot(
                xy[:point_count, 0],
                xy[:point_count, 1],
                color=WINDOW_COLORS[index % len(WINDOW_COLORS)],
                linewidth=4.0,
                alpha=0.95,
                solid_capstyle="round",
                zorder=5,
            )
        if fraction >= 1.0:
            target_index = (
                int(series.target_window_index)
                if series.target_window_index is not None
                else len(xy) // 2
            )
            center = xy[min(max(0, target_index), len(xy) - 1)]
            is_anchor = series.target_role == "mandatory_anchor"
            ax.scatter(
                center[0],
                center[1],
                color=ANCHOR_COLOR if is_anchor else CENTER_COLOR,
                edgecolor="white",
                linewidth=1.2,
                marker="D" if is_anchor else "o",
                s=76 if is_anchor else 72,
                zorder=7,
            )

    xmin, xmax, ymin, ymax = _bounds(family)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D6DCE2", linewidth=0.7, alpha=0.75)
    ax.set_facecolor("#F7F9FA")
    ax.set_xlabel("World x position (m)")
    ax.set_ylabel("World y position (m)")
    ax.set_title(
        _pretty(family.family),
        loc="left",
        y=1.075,
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.015,
        (
            f"{len(family.source_paths)} selected whole paths "
            f"({len(family.source_paths) + len(family.unselected_source_paths)} eligible) | "
            f"{family.candidate_count:,} eligible centers | "
            f"{family.selected_count} selected windows | {family.unique_frame_count:,} unique frames"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#49545E",
        va="bottom",
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=UNSELECTED_SOURCE_COLOR, linewidth=1.2, label="Unselected eligible path"),
            Line2D([0], [0], color=SOURCE_COLOR, linewidth=2, label="Selected whole path"),
            Line2D([0], [0], color=WINDOW_COLORS[0], linewidth=4, label="Selected context window"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CENTER_COLOR,
                   markeredgecolor="white", markersize=8, label="Frame of interest"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=ANCHOR_COLOR,
                   markeredgecolor="white", markersize=8, label="Mandatory endpoint"),
            Line2D([0], [0], color=HUMAN_COLOR, linewidth=1.3, linestyle="--", label="Human track"),
        ],
        loc="best",
        frameon=True,
        framealpha=0.94,
        fontsize=8,
    )


def write_family_bev(family: FamilyVisualization, output_root: Path, config: argparse.Namespace) -> None:
    family_root = output_root / "families" / _safe_id(family.family)
    family_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=110)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.84)
    fractions = [*np.linspace(0.0, 1.0, max(2, int(config.reveal_frames))), *([1.0] * int(config.hold_frames))]

    def update(frame_index: int) -> None:
        _draw_bev(ax, family, float(fractions[frame_index]))

    animation = FuncAnimation(fig, update, frames=len(fractions), interval=1000 / config.gif_fps)
    animation.save(family_root / "bev_sampling.gif", writer=PillowWriter(fps=config.gif_fps))
    _draw_bev(ax, family, 1.0)
    fig.savefig(family_root / "bev_sampling.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_combined_bev(
    families: Sequence[FamilyVisualization],
    output_root: Path,
    config: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=110)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.84)
    family_fractions = [
        (family, fraction)
        for family in families
        for fraction in [
            *np.linspace(0.0, 1.0, max(2, int(config.reveal_frames))),
            *([1.0] * int(config.hold_frames)),
        ]
    ]

    def update(frame_index: int) -> None:
        family, fraction = family_fractions[frame_index]
        _draw_bev(ax, family, float(fraction))

    animation = FuncAnimation(fig, update, frames=len(family_fractions), interval=1000 / config.gif_fps)
    animation.save(output_root / "all_mission_families_bev_sampling.gif", writer=PillowWriter(fps=config.gif_fps))
    plt.close(fig)


def _normalized(counts: Mapping[str, int], categories: Sequence[str]) -> np.ndarray:
    values = np.asarray([float(counts.get(category, 0)) for category in categories], dtype=float)
    total = float(values.sum())
    return values / total if total > 0 else values


def _stacked_distribution(
    ax: plt.Axes,
    families: Sequence[FamilyVisualization],
    *,
    categories: Sequence[str],
    colors: Mapping[str, str],
    counts_attribute: str,
    title: str,
) -> None:
    y = np.arange(len(families))
    left = np.zeros(len(families), dtype=float)
    totals: list[int] = []
    for family in families:
        counts = getattr(family, counts_attribute)
        totals.append(sum(int(value) for value in counts.values()))
    for category in categories:
        values = np.asarray(
            [_normalized(getattr(family, counts_attribute), categories)[categories.index(category)] for family in families]
        )
        ax.barh(y, values, left=left, color=colors[category], height=0.68, label=_pretty(category))
        left += values
    ax.set_xlim(0.0, 1.12)
    ax.set_yticks(y, [_pretty(family.family) for family in families])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color="#D6DCE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Share within mission cohort")
    for index, total in enumerate(totals):
        ax.text(1.015, index, f"n={total:,}", va="center", fontsize=8, color="#49545E")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def write_distributions(families: Sequence[FamilyVisualization], output_root: Path) -> None:
    bucket_categories = tuple(BUCKET_COLORS)
    action_categories = tuple(ACTION_COLORS)
    role_categories = tuple(ROLE_COLORS)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=120, sharey=True)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.08, top=0.90, wspace=0.14, hspace=0.26)
    _stacked_distribution(
        axes[0, 0], families, categories=bucket_categories, colors=BUCKET_COLORS,
        counts_attribute="available_buckets", title="Interest buckets before sampling",
    )
    _stacked_distribution(
        axes[0, 1], families, categories=bucket_categories, colors=BUCKET_COLORS,
        counts_attribute="selected_buckets", title="Interest buckets after sampling",
    )
    _stacked_distribution(
        axes[0, 2], families, categories=role_categories, colors=ROLE_COLORS,
        counts_attribute="selected_roles", title="Selected center roles",
    )
    _stacked_distribution(
        axes[1, 0], families, categories=action_categories, colors=ACTION_COLORS,
        counts_attribute="available_actions", title="Navigation actions before sampling",
    )
    _stacked_distribution(
        axes[1, 1], families, categories=action_categories, colors=ACTION_COLORS,
        counts_attribute="selected_actions", title="Actions at selected centers",
    )
    _stacked_distribution(
        axes[1, 2], families, categories=action_categories, colors=ACTION_COLORS,
        counts_attribute="retained_window_actions", title="Actions in retained context windows",
    )
    for ax in axes[:, 1:].flat:
        ax.tick_params(axis="y", labelleft=False)
    bucket_legend = [Patch(color=BUCKET_COLORS[item], label=_pretty(item)) for item in bucket_categories]
    action_legend = [Patch(color=ACTION_COLORS[item], label=_pretty(item)) for item in action_categories]
    role_legend = [Patch(color=ROLE_COLORS[item], label=_pretty(item)) for item in role_categories]
    fig.legend(handles=bucket_legend, loc="upper center", ncol=4, bbox_to_anchor=(0.47, 0.935), frameon=False)
    fig.legend(handles=role_legend, loc="upper right", ncol=2, bbox_to_anchor=(0.98, 0.935), frameon=False)
    fig.legend(handles=action_legend, loc="center", ncol=4, bbox_to_anchor=(0.59, 0.49), frameon=False)
    fig.suptitle(
        "Frame-selection distributions across ready mission cohorts",
        x=0.20,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.20,
        0.955,
        "Center selection is interest-focused; retained-window actions count every context frame. Each row is normalized independently.",
        ha="left",
        fontsize=10,
        color="#49545E",
    )
    fig.savefig(output_root / "sampling_distribution_graphs.png", dpi=150, facecolor="white")
    plt.close(fig)


def write_navigation_signal_coverage(
    families: Sequence[FamilyVisualization],
    output_root: Path,
) -> None:
    categories = tuple(
        signal
        for signal in NAVIGATION_SIGNAL_COLORS
        if any(
            int(family.available_navigation_signals.get(signal, 0)) > 0
            or int(family.selected_navigation_signals.get(signal, 0)) > 0
            for family in families
        )
    )
    if not categories:
        return

    fig_height = max(7.0, 0.42 * len(categories) + 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(17, fig_height), dpi=120)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.10, top=0.86, wspace=0.28)
    y = np.arange(len(categories))
    available = np.asarray(
        [
            sum(int(family.available_navigation_signals.get(signal, 0)) for family in families)
            for signal in categories
        ],
        dtype=float,
    )
    selected = np.asarray(
        [
            sum(int(family.selected_navigation_signals.get(signal, 0)) for family in families)
            for signal in categories
        ],
        dtype=float,
    )
    colors = [NAVIGATION_SIGNAL_COLORS[signal] for signal in categories]

    axes[0].barh(y, available, color=colors, alpha=0.32, label="Eligible centers")
    axes[0].barh(y, selected, color=colors, alpha=0.95, label="All selected centers")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Center count (log scale)")
    axes[0].set_title("Navigation-signal populations", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    available_episodes = np.asarray(
        [
            sum(
                int(family.available_navigation_signal_episodes.get(signal, 0))
                for family in families
            )
            for signal in categories
        ],
        dtype=float,
    )
    selected_interest_episodes = np.asarray(
        [
            sum(
                int(family.selected_interest_navigation_signal_episodes.get(signal, 0))
                for family in families
            )
            for signal in categories
        ],
        dtype=float,
    )
    retention = np.divide(
        selected_interest_episodes,
        available_episodes,
        out=np.zeros_like(selected_interest_episodes),
        where=available_episodes > 0,
    )
    axes[1].barh(y, retention, color=colors)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_xlabel("Episodes covered by scored POIs / eligible episodes")
    axes[1].set_title("Distinct-event coverage", loc="left", fontweight="bold")
    for index, (picked, total) in enumerate(
        zip(selected_interest_episodes, available_episodes)
    ):
        axes[1].text(
            min(1.0, retention[index]) + 0.01,
            index,
            f"{int(picked):,}/{int(total):,}",
            va="center",
            fontsize=8,
            color="#49545E",
        )
    axes[1].set_xlim(0.0, max(0.1, float(retention.max()) * 1.30))

    labels = [_pretty(signal) for signal in categories]
    for ax in axes:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D6DCE2", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(
        "Metadata-grounded point-selection coverage",
        x=0.25,
        y=0.97,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.25,
        0.925,
        "Left: signal-bearing frames. Right: contiguous signal episodes covered by at least one scored POI.",
        ha="left",
        fontsize=10,
        color="#49545E",
    )
    fig.savefig(
        output_root / "navigation_signal_coverage.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    pilot_root = args.pilot_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest_roots = tuple(
        path.resolve() for path in (args.source_manifest_root or [])
    )
    families = load_pilot(
        pilot_root,
        source_manifest_roots=source_manifest_roots,
    )
    write_combined_bev(families, output_root, args)
    if args.per_family:
        for family in families:
            write_family_bev(family, output_root, args)
    write_distributions(families, output_root)
    write_navigation_signal_coverage(families, output_root)
    signal_graph = output_root / "navigation_signal_coverage.png"
    summary = {
        "ready_family_count": len(families),
        "combined_bev_gif": str(output_root / "all_mission_families_bev_sampling.gif"),
        "distribution_graphs": str(output_root / "sampling_distribution_graphs.png"),
        "navigation_signal_graph": str(signal_graph) if signal_graph.is_file() else None,
        "per_family_output_root": str(output_root / "families") if args.per_family else None,
        "gaussian_rendering_executed": False,
    }
    with (output_root / "visualization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
