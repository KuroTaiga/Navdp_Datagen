#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.frame_selection import (  # noqa: E402
    DEFAULT_ACTION_RATIOS,
    DEFAULT_FUTURE_FRAMES,
    DEFAULT_PAST_FRAMES,
    FrameSelectionConfig,
    _action_name,
    _job_trajectory,
    _navigation_signals,
    select_frame_interest_windows,
)
from scripts.analysis.survey_path_sampling_corpus import (  # noqa: E402
    _describe,
    _merge_center_window_composition,
    _progress_records,
    _rank,
    _ratios,
    _sample_scenario_paths,
    _sum_counts,
    _training_weights,
)


JsonDict = dict[str, Any]
ACTION_NAMES = ("move", "stop", "turn_left", "turn_right")
EXPERIMENTS = {
    "budget_2": {"average_scored_pois_per_path": 2, "semantic_episode_policy": "none"},
    "budget_4": {"average_scored_pois_per_path": 4, "semantic_episode_policy": "none"},
    "budget_8": {"average_scored_pois_per_path": 8, "semantic_episode_policy": "none"},
    "event_aware": {
        "average_scored_pois_per_path": 0,
        "semantic_episode_policy": "guarantee_representative",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed and event-aware POI policies on one deterministic sample of "
            "complete source paths from every corpus/family/scene cohort."
        )
    )
    parser.add_argument("--formal-fast-root", type=Path, required=True)
    parser.add_argument("--formal-slow-root", type=Path, required=True)
    parser.add_argument("--formal-social-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths-per-cohort", type=int, default=50)
    parser.add_argument(
        "--scenes-per-family",
        type=int,
        default=0,
        help="0 surveys every available cohort; positive values are for smoke tests only.",
    )
    parser.add_argument("--past-frames", type=int, default=DEFAULT_PAST_FRAMES)
    parser.add_argument("--future-frames", type=int, default=DEFAULT_FUTURE_FRAMES)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=119)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=tuple(EXPERIMENTS),
        help="Repeat to select a subset; defaults to all four experiments.",
    )
    return parser.parse_args()


def _compact_chunk(chunk: Mapping[str, Any]) -> JsonDict:
    return {
        "source_manifest_index": int(chunk.get("source_manifest_index", 0)),
        "source_manifest_path": chunk.get("source_manifest_path"),
        "source_job_id": str(chunk.get("source_job_id") or ""),
        "scene_id": str(chunk.get("scene_id") or ""),
        "viewpoint_robot_id": str(chunk.get("viewpoint_robot_id") or ""),
        "mission_families": list(chunk.get("mission_families", [])),
        "target_frame": int(chunk.get("target_frame", 0)),
        "target_source_frame_id": int(chunk.get("target_source_frame_id", 0)),
        "target_window_index": int(chunk.get("target_window_index", DEFAULT_PAST_FRAMES)),
        "target_time_s": float(chunk.get("target_time_s", 0.0)),
        "target_action_name": str(chunk.get("target_action_name") or "unknown"),
        "target_bucket": str(chunk.get("target_bucket") or "unknown"),
        "target_role": str(chunk.get("target_role") or "frame_of_interest"),
        "mandatory_anchor_types": list(chunk.get("mandatory_anchor_types", [])),
        "navigation_signals": list(chunk.get("navigation_signals", [])),
        "source_frame_indices": [
            int(value) for value in chunk.get("window", {}).get("source_frame_indices", [])
        ],
        "render_frame_count": int(
            chunk.get("window", {}).get("render_frame_count", 0)
        ),
    }


def _source_population(
    manifests: Sequence[Mapping[str, Any]],
    selected_paths: Sequence[Mapping[str, Any]],
) -> JsonDict:
    action_counts: Counter[str] = Counter()
    signal_frame_counts: Counter[str] = Counter()
    signal_episode_counts: Counter[str] = Counter()
    path_lengths: list[int] = []
    missing_navigation_frames = 0
    malformed_paths = 0
    path_payloads: list[JsonDict] = []
    for record in selected_paths:
        manifest_index = int(record.get("source_manifest_index", -1))
        job_id = str(record.get("source_job_id") or "")
        if manifest_index < 0 or manifest_index >= len(manifests):
            malformed_paths += 1
            continue
        jobs = [
            job
            for job in manifests[manifest_index].get("jobs", [])
            if isinstance(job, Mapping) and str(job.get("job_id") or "") == job_id
        ]
        if len(jobs) != 1:
            malformed_paths += 1
            continue
        trajectory = _job_trajectory(jobs[0], densify_frame_gaps=True)
        if not trajectory:
            malformed_paths += 1
            continue
        path_lengths.append(len(trajectory))
        active_previous: set[str] = set()
        points: list[JsonDict] = []
        for index, point in enumerate(trajectory):
            action = _action_name(trajectory, index)
            signals = set(_navigation_signals(trajectory, index))
            action_counts[action] += 1
            signal_frame_counts.update(signals)
            signal_episode_counts.update(signals - active_previous)
            active_previous = signals
            metadata = point.get("metadata")
            navigation = (
                metadata.get("navigation")
                if isinstance(metadata, Mapping) and isinstance(metadata.get("navigation"), Mapping)
                else point.get("navigation")
            )
            if not isinstance(navigation, Mapping) or not navigation:
                missing_navigation_frames += 1
            position = point.get("position")
            if isinstance(position, Sequence) and not isinstance(position, (str, bytes)):
                xy = [float(position[0]), float(position[1])] if len(position) >= 2 else [0.0, 0.0]
            else:
                pose = point.get("map_pose") if isinstance(point.get("map_pose"), Mapping) else {}
                xy = [float(pose.get("x", 0.0)), float(pose.get("y", 0.0))]
            points.append(
                {
                    "frame": index,
                    "source_frame_id": int(point.get("frame", index)),
                    "xy": xy,
                    "action": action,
                    "navigation_signals": sorted(signals),
                }
            )
        path_payloads.append(
            {
                "source_manifest_index": manifest_index,
                "source_manifest_path": record.get("source_manifest_path"),
                "source_job_id": job_id,
                "scene_id": record.get("scene_id"),
                "viewpoint_robot_id": record.get("viewpoint_robot_id"),
                "mission_families": list(record.get("mission_families", [])),
                "points": points,
            }
        )
    return {
        "path_lengths": path_lengths,
        "action_counts": dict(sorted(action_counts.items())),
        "navigation_signal_frame_counts": dict(sorted(signal_frame_counts.items())),
        "navigation_signal_episode_counts": dict(sorted(signal_episode_counts.items())),
        "missing_navigation_frame_count": missing_navigation_frames,
        "malformed_path_count": malformed_paths,
        "path_payloads": path_payloads,
    }


def _experiment_record(
    experiment: str,
    family: str,
    scene: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    manifest_paths: Sequence[str],
    source: Mapping[str, Any] | None,
    *,
    args: argparse.Namespace,
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    definition = EXPERIMENTS[experiment]
    available_source_paths = sum(
        len([job for job in manifest.get("jobs", []) if isinstance(job, Mapping)])
        for manifest in manifests
    )
    requested_source_paths = min(int(args.paths_per_cohort), available_source_paths)
    selector_family = (
        "navigate_with_social_constraints"
        if family.startswith("navigate_with_social_constraints:")
        else family
    )
    selection = select_frame_interest_windows(
        manifests,
        manifest_paths=manifest_paths,
        config=FrameSelectionConfig(
            target_count=(
                int(definition["average_scored_pois_per_path"]) * requested_source_paths
            ),
            past_frames=int(args.past_frames),
            future_frames=int(args.future_frames),
            mission_families=(selector_family,),
            max_source_paths=int(args.paths_per_cohort),
            max_targets_per_job=0,
            preserve_mission_endpoints=True,
            endpoint_window_policy="clamp",
            target_action_ratios=DEFAULT_ACTION_RATIOS,
            enforce_action_minimums=True,
            semantic_episode_policy=str(definition["semantic_episode_policy"]),
            seed=int(args.seed),
            split="survey",
        ),
    )
    summary = selection["selection_summary"]
    distribution = selection["distribution"]
    selected_paths = distribution["source_path_sampling"]["selected_source_paths"]
    if source is None:
        source = _source_population(manifests, selected_paths)
    source_frames = sum(int(value) for value in source["path_lengths"])
    actions = {
        "all_source_frames": dict(source["action_counts"]),
        "poi_centers": dict(summary.get("selected_interest_action_counts", {})),
        "all_selected_centers": dict(summary.get("selected_center_action_counts", {})),
        "mandatory_anchors": dict(summary.get("mandatory_anchor_action_counts", {})),
        "independent_windows": dict(summary.get("selected_window_action_counts", {})),
        "unique_retained_frames": dict(summary.get("selected_unique_window_action_counts", {})),
    }
    available_anchor_types = dict(distribution.get("available_mandatory_anchor_type_counts", {}))
    selected_anchor_types = dict(summary.get("mandatory_anchor_type_counts", {}))
    available_anchor_count = sum(available_anchor_types.values())
    covered_anchor_count = sum(
        min(int(count), int(selected_anchor_types.get(anchor_type, 0)))
        for anchor_type, count in available_anchor_types.items()
    )
    selected_center_count = int(summary.get("selected_target_count", 0))
    repeated_frames = int(summary.get("requested_window_frame_renders", 0))
    unique_frames = int(summary.get("unique_source_render_frame_count", 0))
    record = {
        "experiment": experiment,
        "corpus": str(scene.get("root_kind") or ""),
        "family": family,
        "dataset": str(scene.get("dataset") or ""),
        "scene": str(scene.get("scene") or ""),
        "cohort_id": str(scene.get("cohort_id") or ""),
        "generated_path_count": int(scene.get("accepted_paths", 0)),
        "sampled_scenario_count": len(manifests),
        "available_loaded_source_path_count": available_source_paths,
        "sampled_source_path_count": len(selected_paths),
        "source_path_frame_stats": _describe(source["path_lengths"]),
        "source_frame_count": source_frames,
        "eligible_center_count": int(summary.get("candidate_count", 0)),
        "selected_poi_count": int(summary.get("selected_interest_target_count", 0)),
        "requested_poi_count": int(summary.get("requested_target_count", 0)),
        "mandatory_anchor_count": int(summary.get("mandatory_anchor_count", 0)),
        "selected_center_count": selected_center_count,
        "independent_window_count": selected_center_count,
        "independent_window_frame_count": repeated_frames,
        "unique_retained_frame_count": unique_frames,
        "retention": {
            "poi_center_percent": 100.0 * int(summary.get("selected_interest_target_count", 0)) / max(1, source_frames),
            "all_center_percent": 100.0 * selected_center_count / max(1, source_frames),
            "independent_window_frame_percent": 100.0 * repeated_frames / max(1, source_frames),
            "unique_retained_frame_percent": 100.0 * unique_frames / max(1, source_frames),
        },
        "actions": {
            name: {"counts": counts, "ratios": _ratios(counts)}
            for name, counts in actions.items()
        },
        "navigation_signals": {
            "all_source_frames": dict(source["navigation_signal_frame_counts"]),
            "all_source_episodes": dict(source["navigation_signal_episode_counts"]),
            "eligible_center_frames": dict(distribution.get("available_navigation_signal_counts", {})),
            "eligible_center_episodes": dict(distribution.get("available_navigation_signal_episode_counts", {})),
            "selected_center_frames": dict(summary.get("selected_navigation_signal_counts", {})),
            "poi_center_frames": dict(summary.get("selected_interest_navigation_signal_counts", {})),
            "selected_center_covered_episodes": dict(summary.get("selected_navigation_signal_episode_counts", {})),
            "poi_covered_episodes": dict(summary.get("selected_interest_navigation_signal_episode_counts", {})),
        },
        "endpoint_coverage": {
            "available_anchor_type_counts": available_anchor_types,
            "selected_anchor_type_counts": selected_anchor_types,
            "available_anchor_count": available_anchor_count,
            "covered_anchor_count": covered_anchor_count,
            "coverage_ratio": covered_anchor_count / max(1, available_anchor_count),
        },
        "semantic_episode_policy": definition["semantic_episode_policy"],
        "guaranteed_episode_representative_count": int(
            distribution.get("guaranteed_episode_representative_count", 0)
        ),
        "action_deficits": dict(summary.get("action_deficits", {})),
        "scored_center_window_composition": dict(
            summary.get("scored_center_window_composition", {})
        ),
        "missing_navigation_frame_count": int(source["missing_navigation_frame_count"]),
        "malformed_path_count": int(source["malformed_path_count"]),
        "errors": [],
    }
    compact_manifest = {
        "schema_version": "navdp_poi_survey_selection/v1.0",
        "cohort": {
            key: record[key]
            for key in ("corpus", "family", "dataset", "scene", "cohort_id")
        },
        "experiment": experiment,
        "config": selection["config"],
        "source_manifests": selection["source_manifests"],
        "source_path_sampling": distribution["source_path_sampling"],
        "selection_summary": summary,
        "chunks": [_compact_chunk(chunk) for chunk in selection["chunks"]],
        "renderer_independent": True,
    }
    representative = {
        "cohort": compact_manifest["cohort"],
        "experiment": experiment,
        "config": selection["config"],
        "chunks": compact_manifest["chunks"],
    }
    return record, compact_manifest, representative, dict(source)


def _survey_cohort(task: tuple[str, Mapping[str, Any], argparse.Namespace, tuple[str, ...], bool]) -> JsonDict:
    family, scene, args, experiments, representative = task
    manifests, manifest_paths, load_errors = _sample_scenario_paths(
        scene,
        paths_per_scene=int(args.paths_per_cohort),
        fps=float(args.fps),
        seed=int(args.seed),
    )
    records: list[JsonDict] = []
    compact_manifests: list[JsonDict] = []
    representative_payload: JsonDict | None = None
    source: JsonDict | None = None
    for experiment in experiments:
        try:
            record, compact_manifest, selection_rep, source = _experiment_record(
                experiment,
                family,
                scene,
                manifests,
                manifest_paths,
                source,
                args=args,
            )
            record["errors"] = list(load_errors)
            records.append(record)
            compact_manifests.append(compact_manifest)
            if representative and representative_payload is None and source["path_payloads"]:
                chosen_path = source["path_payloads"][0]
                representative_payload = {
                    "schema_version": "navdp_poi_survey_representative/v1.0",
                    "cohort": selection_rep["cohort"],
                    "source_path": chosen_path,
                    "instructions": {
                        "missions": manifests[int(chosen_path["source_manifest_index"])].get("missions", []),
                        "navigation_supervision": manifests[int(chosen_path["source_manifest_index"])].get("navigation_supervision", {}),
                    },
                    "experiments": {},
                }
            if representative_payload is not None:
                chosen = representative_payload["source_path"]
                representative_payload["experiments"][experiment] = {
                    **selection_rep,
                    "chunks": [
                        chunk
                        for chunk in selection_rep["chunks"]
                        if int(chunk["source_manifest_index"]) == int(chosen["source_manifest_index"])
                        and str(chunk["source_job_id"]) == str(chosen["source_job_id"])
                    ],
                }
        except Exception as exc:  # keep the full matrix running and report the cohort failure
            records.append(
                {
                    "experiment": experiment,
                    "corpus": str(scene.get("root_kind") or ""),
                    "family": family,
                    "dataset": str(scene.get("dataset") or ""),
                    "scene": str(scene.get("scene") or ""),
                    "cohort_id": str(scene.get("cohort_id") or ""),
                    "generated_path_count": int(scene.get("accepted_paths", 0)),
                    "sampled_scenario_count": len(manifests),
                    "sampled_source_path_count": 0,
                    "source_frame_count": 0,
                    "errors": [*load_errors, f"{type(exc).__name__}: {exc}"],
                }
            )
    return {
        "records": records,
        "compact_manifests": compact_manifests,
        "representative": representative_payload,
        "path_lengths": list(source["path_lengths"]) if source is not None else [],
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    valid = [record for record in records if "actions" in record]
    source_frames = sum(int(record.get("source_frame_count", 0)) for record in valid)
    actions = {
        name: _sum_counts([record["actions"][name]["counts"] for record in valid])
        for name in (
            "all_source_frames",
            "poi_centers",
            "all_selected_centers",
            "mandatory_anchors",
            "independent_windows",
            "unique_retained_frames",
        )
    }
    signal_names = (
        "all_source_frames",
        "all_source_episodes",
        "eligible_center_frames",
        "eligible_center_episodes",
        "selected_center_frames",
        "poi_center_frames",
        "selected_center_covered_episodes",
        "poi_covered_episodes",
    )
    navigation = {
        name: _sum_counts([record["navigation_signals"][name] for record in valid])
        for name in signal_names
    }
    available_anchors = _sum_counts(
        [record["endpoint_coverage"]["available_anchor_type_counts"] for record in valid]
    )
    selected_anchors = _sum_counts(
        [record["endpoint_coverage"]["selected_anchor_type_counts"] for record in valid]
    )
    return {
        "cohort_count": len(records),
        "successful_cohort_count": len(valid),
        "error_cohort_count": sum(1 for record in records if record.get("errors")),
        "sampled_scenario_count": sum(int(record.get("sampled_scenario_count", 0)) for record in valid),
        "sampled_source_path_count": sum(int(record.get("sampled_source_path_count", 0)) for record in valid),
        "source_frame_count": source_frames,
        "selected_poi_count": sum(int(record.get("selected_poi_count", 0)) for record in valid),
        "mandatory_anchor_count": sum(int(record.get("mandatory_anchor_count", 0)) for record in valid),
        "selected_center_count": sum(int(record.get("selected_center_count", 0)) for record in valid),
        "independent_window_count": sum(int(record.get("independent_window_count", 0)) for record in valid),
        "independent_window_frame_count": sum(int(record.get("independent_window_frame_count", 0)) for record in valid),
        "unique_retained_frame_count": sum(int(record.get("unique_retained_frame_count", 0)) for record in valid),
        "retention": {
            "poi_center_percent": 100.0 * sum(int(record.get("selected_poi_count", 0)) for record in valid) / max(1, source_frames),
            "independent_window_frame_percent": 100.0 * sum(int(record.get("independent_window_frame_count", 0)) for record in valid) / max(1, source_frames),
            "unique_retained_frame_percent": 100.0 * sum(int(record.get("unique_retained_frame_count", 0)) for record in valid) / max(1, source_frames),
        },
        "actions": {name: {"counts": counts, "ratios": _ratios(counts)} for name, counts in actions.items()},
        "navigation_signals": navigation,
        "endpoint_coverage": {
            "available_anchor_type_counts": available_anchors,
            "selected_anchor_type_counts": selected_anchors,
            "available_anchor_count": sum(available_anchors.values()),
            "covered_anchor_count": sum(
                min(int(count), int(selected_anchors.get(key, 0)))
                for key, count in available_anchors.items()
            ),
            "coverage_ratio": sum(
                min(int(count), int(selected_anchors.get(key, 0)))
                for key, count in available_anchors.items()
            ) / max(1, sum(available_anchors.values())),
        },
        "missing_navigation_frame_count": sum(int(record.get("missing_navigation_frame_count", 0)) for record in valid),
        "malformed_path_count": sum(int(record.get("malformed_path_count", 0)) for record in valid),
        "action_deficits": _sum_counts([record.get("action_deficits", {}) for record in valid]),
        "scored_center_window_composition": _merge_center_window_composition(valid),
        "training_center_sampling": {
            "population": "poi_centers",
            "mandatory_anchor_policy": "retained_separate_pool",
            "target_action_ratios": dict(DEFAULT_ACTION_RATIOS),
            "per_example_action_weights": _training_weights(actions["poi_centers"]),
        },
    }


def _coverage(covered: Mapping[str, int], available: Mapping[str, int], signal: str) -> float:
    return float(covered.get(signal, 0)) / max(1, int(available.get(signal, 0)))


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    config = payload["config"]
    lines = [
        "# POI Policy Matrix Survey",
        "",
        "## Scope",
        "",
        f"- Seed: `{config['seed']}`.",
        f"- Cohorts: every available `(corpus, mission family, scene)` cohort; up to `{config['paths_per_cohort']}` complete paths each.",
        f"- Causal package: `{config['past_frames']}` past frames plus current, `{config['future_frames']}` future frames (`{config['window_frame_count']}` total).",
        "- Mandatory mission/checkpoint/sub-mission/route endpoints are additive.",
        "- No Gaussian or RGB rendering was performed.",
        "",
        "## Generation Population",
        "",
        "| Corpus | Mission family | Scene cohorts | Generated paths | Mean paths/scene |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for group in payload["generation"]["groups"]:
        lines.append(
            f"| `{group['corpus']}` | `{group['family']}` | {group['scene_count']:,} | "
            f"{group['generated_path_count']:,} | "
            f"{float(group['generated_paths_per_scene']['mean'] or 0.0):.2f} |"
        )
    stats = payload["sampled_path_frame_stats"]
    lines.extend([
        "",
        "## Sampled Path Lengths",
        "",
        "| Paths | Mean | Median | P25 | P75 | P90 | P95 | P99 | Min | Max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {int(stats['count']):,} | {float(stats['mean'] or 0.0):.2f} | "
        f"{float(stats['median'] or 0.0):.2f} | {float(stats.get('p25') or 0.0):.0f} | "
        f"{float(stats.get('p75') or 0.0):.0f} | {float(stats.get('p90') or 0.0):.0f} | "
        f"{float(stats.get('p95') or 0.0):.0f} | {float(stats.get('p99') or 0.0):.0f} | "
        f"{float(stats['min'] or 0.0):.0f} | {float(stats['max'] or 0.0):.0f} |",
        "",
        "## Comparative Results",
        "",
        "| Experiment | Paths | Source frames | POIs | Anchors | Centers/windows | Independent window frames | Window/source | Unique frames | Unique retention | Endpoint coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for experiment in payload["experiments"]:
        totals = experiment["totals"]
        lines.append(
            f"| `{experiment['experiment']}` | {totals['sampled_source_path_count']:,} | "
            f"{totals['source_frame_count']:,} | {totals['selected_poi_count']:,} | "
            f"{totals['mandatory_anchor_count']:,} | {totals['selected_center_count']:,} | "
            f"{totals['independent_window_frame_count']:,} | "
            f"{totals['retention']['independent_window_frame_percent']:.2f}% | "
            f"{totals['unique_retained_frame_count']:,} | "
            f"{totals['retention']['unique_retained_frame_percent']:.2f}% | "
            f"{100.0 * totals['endpoint_coverage']['coverage_ratio']:.2f}% |"
        )
    lines.extend([
        "",
        "## Action Distributions",
        "",
        "Percentages are reported separately for source frames, POI labels, independent causal windows, and overlap-deduplicated retained frames.",
        "",
        "| Experiment / population | Move | Stop | Left | Right |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for experiment in payload["experiments"]:
        for population in ("all_source_frames", "poi_centers", "independent_windows", "unique_retained_frames"):
            ratios = experiment["totals"]["actions"][population]["ratios"]
            values = [100.0 * float(ratios.get(action, 0.0)) for action in ACTION_NAMES]
            lines.append(
                f"| `{experiment['experiment']}` / {population.replace('_', ' ')} | "
                + " | ".join(f"{value:.2f}%" for value in values)
                + " |"
            )
    signals = sorted(
        {
            signal
            for experiment in payload["experiments"]
            for signal in experiment["totals"]["navigation_signals"]["eligible_center_episodes"]
        }
    )
    lines.extend([
        "",
        "## Navigation Signal Frames and Episodes",
        "",
        "Signal-frame percentages may overlap because one frame can carry multiple signals.",
        "",
        "| Signal | Source signal frames | Source-frame share | Contiguous source episodes | Eligible center episodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    reference = payload["experiments"][0]["totals"]
    reference_nav = reference["navigation_signals"]
    for signal in signals:
        signal_frames = int(reference_nav["all_source_frames"].get(signal, 0))
        lines.append(
            f"| `{signal}` | {signal_frames:,} | "
            f"{100.0 * signal_frames / max(1, reference['source_frame_count']):.2f}% | "
            f"{int(reference_nav['all_source_episodes'].get(signal, 0)):,} | "
            f"{int(reference_nav['eligible_center_episodes'].get(signal, 0)):,} |"
        )
    for coverage_population, title in (
        ("selected_center_covered_episodes", "Semantic Episode Coverage by Any Selected Center"),
        ("poi_covered_episodes", "Semantic Episode Coverage by Scored POIs Excluding Mandatory Anchors"),
    ):
        lines.extend([
            "",
            f"## {title}",
            "",
            "| Signal | " + " | ".join(f"`{item['experiment']}`" for item in payload["experiments"]) + " |",
            "| --- | " + " | ".join("---:" for _ in payload["experiments"]) + " |",
        ])
        for signal in signals:
            values = []
            for experiment in payload["experiments"]:
                nav = experiment["totals"]["navigation_signals"]
                covered = int(nav[coverage_population].get(signal, 0))
                available = int(nav["eligible_center_episodes"].get(signal, 0))
                values.append(
                    f"{covered:,}/{available:,} "
                    f"({100.0 * covered / max(1, available):.2f}%)"
                )
            lines.append(f"| `{signal}` | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Per Corpus and Mission Family",
        "",
        "| Experiment | Corpus | Family | Cohorts | Paths | Mean frames/path | POIs/path | Unique retention | Errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for experiment in payload["experiments"]:
        for family in experiment["families"]:
            totals = family["totals"]
            lines.append(
                f"| `{experiment['experiment']}` | `{family['corpus']}` | `{family['family']}` | "
                f"{totals['cohort_count']:,} | {totals['sampled_source_path_count']:,} | "
                f"{float(family['source_path_frame_stats']['mean'] or 0.0):.2f} | "
                f"{totals['selected_poi_count'] / max(1, totals['sampled_source_path_count']):.2f} | "
                f"{totals['retention']['unique_retained_frame_percent']:.2f}% | "
                f"{totals['error_cohort_count']:,} |"
            )
    lines.extend([
        "",
        "## Data Quality",
        "",
    ])
    for experiment in payload["experiments"]:
        totals = experiment["totals"]
        lines.append(
            f"- `{experiment['experiment']}`: {totals['error_cohort_count']:,} cohorts with errors, "
            f"{totals['malformed_path_count']:,} malformed selected paths, and "
            f"{totals['missing_navigation_frame_count']:,} selected-source frames missing navigation metadata."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_row(record: Mapping[str, Any]) -> JsonDict:
    row = {
        key: record.get(key)
        for key in (
            "experiment", "corpus", "family", "dataset", "scene", "cohort_id",
            "generated_path_count", "sampled_scenario_count", "sampled_source_path_count",
            "source_frame_count", "selected_poi_count", "mandatory_anchor_count",
            "selected_center_count", "independent_window_count",
            "independent_window_frame_count", "unique_retained_frame_count",
            "missing_navigation_frame_count", "malformed_path_count",
        )
    }
    retention = record.get("retention", {})
    for key, value in retention.items():
        row[key] = value
    for population in ("all_source_frames", "poi_centers", "independent_windows", "unique_retained_frames"):
        ratios = record.get("actions", {}).get(population, {}).get("ratios", {})
        for action in ACTION_NAMES:
            row[f"{population}_{action}_ratio"] = ratios.get(action, 0.0)
    nav = record.get("navigation_signals", {})
    for key in ("all_source_frames", "all_source_episodes", "eligible_center_episodes", "poi_covered_episodes"):
        row[f"navigation_{key}_json"] = json.dumps(nav.get(key, {}), sort_keys=True)
    row["endpoint_coverage_ratio"] = record.get("endpoint_coverage", {}).get("coverage_ratio", 0.0)
    row["errors_json"] = json.dumps(record.get("errors", []), sort_keys=True)
    return row


def main() -> int:
    args = _parse_args()
    experiments = tuple(args.experiment or EXPERIMENTS)
    roots = (
        (args.formal_fast_root.resolve(), "fast"),
        (args.formal_slow_root.resolve(), "slow"),
        (args.formal_social_root.resolve(), "social"),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "selection_manifests"
    representative_dir = output_dir / "representatives"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    representative_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        experiment: gzip.open(manifest_dir / f"{experiment}.jsonl.gz", "wt", encoding="utf-8")
        for experiment in experiments
    }
    try:
        scene_records = _progress_records(roots)
        for scene in scene_records:
            scene["corpus"] = str(scene["root_kind"])
            scene["cohort_id"] = "/".join(
                (str(scene["root_kind"]), str(scene["family"]), str(scene["dataset"]), str(scene["scene"]))
            )
        groups: defaultdict[tuple[str, str], list[JsonDict]] = defaultdict(list)
        for scene in scene_records:
            groups[(str(scene["root_kind"]), str(scene["family"]))].append(scene)

        all_records: list[JsonDict] = []
        path_lengths: list[int] = []
        family_path_lengths: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        family_records: defaultdict[tuple[str, str, str], list[JsonDict]] = defaultdict(list)
        representatives: list[JsonDict] = []
        for (corpus, family), scenes in sorted(groups.items()):
            nonempty = [scene for scene in scenes if int(scene.get("accepted_paths", 0)) > 0]
            ranked = sorted(nonempty, key=lambda item: _rank(int(args.seed), corpus, family, item["scene_key"]))
            if int(args.scenes_per_family) > 0:
                ranked = ranked[: int(args.scenes_per_family)]
            representative_id = str(ranked[0]["cohort_id"]) if ranked else ""
            tasks = [
                (family, scene, args, experiments, str(scene["cohort_id"]) == representative_id)
                for scene in ranked
            ]
            worker_count = min(max(1, int(args.workers)), max(1, len(tasks)))
            if worker_count > 1 and len(tasks) > 1:
                with ProcessPoolExecutor(max_workers=worker_count) as pool:
                    results: Iterable[JsonDict] = pool.map(_survey_cohort, tasks, chunksize=1)
                    for result in results:
                        path_lengths.extend(result["path_lengths"])
                        family_path_lengths[(corpus, family)].extend(result["path_lengths"])
                        for record in result["records"]:
                            all_records.append(record)
                            family_records[(str(record["experiment"]), corpus, family)].append(record)
                        for manifest in result["compact_manifests"]:
                            handles[str(manifest["experiment"])].write(json.dumps(manifest, separators=(",", ":")) + "\n")
                        if result["representative"] is not None:
                            representatives.append(result["representative"])
            else:
                for task in tasks:
                    result = _survey_cohort(task)
                    path_lengths.extend(result["path_lengths"])
                    family_path_lengths[(corpus, family)].extend(result["path_lengths"])
                    for record in result["records"]:
                        all_records.append(record)
                        family_records[(str(record["experiment"]), corpus, family)].append(record)
                    for manifest in result["compact_manifests"]:
                        handles[str(manifest["experiment"])].write(json.dumps(manifest, separators=(",", ":")) + "\n")
                    if result["representative"] is not None:
                        representatives.append(result["representative"])
            print(f"surveyed corpus={corpus} family={family} cohorts={len(tasks)}", flush=True)

        for index, representative in enumerate(representatives):
            cohort = representative["cohort"]
            safe = "__".join(
                str(cohort[key]).replace("/", "_").replace(":", "_")
                for key in ("corpus", "family", "dataset", "scene")
            )
            (representative_dir / f"{index:03d}_{safe}.json").write_text(
                json.dumps(representative, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        experiment_payloads: list[JsonDict] = []
        for experiment in experiments:
            experiment_records = [record for record in all_records if record["experiment"] == experiment]
            families = [
                {
                    "corpus": corpus,
                    "family": family,
                    "source_path_frame_stats": _describe(family_path_lengths[(corpus, family)]),
                    "totals": _aggregate_records(records),
                }
                for (name, corpus, family), records in sorted(family_records.items())
                if name == experiment
            ]
            experiment_payloads.append(
                {
                    "experiment": experiment,
                    "definition": EXPERIMENTS[experiment],
                    "totals": _aggregate_records(experiment_records),
                    "families": families,
                }
            )

        generation_groups = []
        for (corpus, family), scenes in sorted(groups.items()):
            counts = [int(scene.get("accepted_paths", 0)) for scene in scenes]
            generation_groups.append(
                {
                    "corpus": corpus,
                    "family": family,
                    "scene_count": len(scenes),
                    "generated_path_count": sum(counts),
                    "generated_paths_per_scene": _describe(counts),
                }
            )
        payload = {
            "schema_version": "navdp_poi_policy_matrix_survey/v1.0",
            "config": {
                "roots": [{"corpus": layout, "path": str(root)} for root, layout in roots],
                "paths_per_cohort": int(args.paths_per_cohort),
                "scenes_per_family": int(args.scenes_per_family),
                "past_frames": int(args.past_frames),
                "future_frames": int(args.future_frames),
                "window_frame_count": int(args.past_frames) + 1 + int(args.future_frames),
                "fps": float(args.fps),
                "seed": int(args.seed),
                "workers": int(args.workers),
                "experiments": list(experiments),
                "sampling_policy": "deterministic_hash_rank_complete_paths_per_corpus_family_scene",
                "gaussian_rgb_rendering_executed": False,
            },
            "generation": {
                "cohort_count": len(scene_records),
                "generated_path_count": sum(int(scene.get("accepted_paths", 0)) for scene in scene_records),
                "groups": generation_groups,
                "scenes": scene_records,
            },
            "sampled_path_frame_stats": _describe(path_lengths),
            "experiments": experiment_payloads,
            "cohort_results": all_records,
            "selection_manifest_files": {
                experiment: str(manifest_dir / f"{experiment}.jsonl.gz")
                for experiment in experiments
            },
            "representative_files": [str(path) for path in sorted(representative_dir.glob("*.json"))],
        }
        (output_dir / "poi_policy_matrix.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"cohort_results"}
        }
        summary_payload["generation"] = {
            key: value
            for key, value in payload["generation"].items()
            if key != "scenes"
        }
        (output_dir / "poi_policy_matrix_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows = [_csv_row(record) for record in all_records]
        with (output_dir / "poi_policy_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = tuple(rows[0]) if rows else ("experiment",)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        _write_markdown(payload, output_dir / "poi_policy_matrix.md")
        print(json.dumps({
            "output_dir": str(output_dir),
            "cohorts": len(scene_records),
            "sampled_paths": experiment_payloads[0]["totals"]["sampled_source_path_count"] if experiment_payloads else 0,
            "errors": sum(item["totals"]["error_cohort_count"] for item in experiment_payloads),
        }, sort_keys=True))
        return 0
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
