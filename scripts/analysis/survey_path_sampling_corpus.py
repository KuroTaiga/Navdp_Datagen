#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navdp_datagen.massgen.frame_selection import (  # noqa: E402
    DEFAULT_ACTION_RATIOS,
    FrameSelectionConfig,
    select_frame_interest_windows,
)
from utils.massgen_render_manifest import scenario_file_to_render_manifest  # noqa: E402


JsonDict = dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey current Pathplanner output counts and frame-selection retention."
    )
    parser.add_argument("--formal-fast-root", type=Path, required=True)
    parser.add_argument("--formal-slow-root", type=Path, required=True)
    parser.add_argument("--formal-social-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-scene-csv", type=Path, required=True)
    parser.add_argument("--output-sampling-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--scenes-per-family", type=int, default=25)
    parser.add_argument("--paths-per-scene", type=int, default=50)
    parser.add_argument("--scored-pois-per-source-path", type=int, default=2)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rank(seed: int, *parts: object) -> str:
    text = "::".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _progress_paths(root: Path, layout: str) -> list[Path]:
    if layout in {"fast", "social"}:
        return sorted(root.glob("*/*/*/mass_generation_progress.json"))
    return sorted(root.glob("*/*/*/*/mass_generation_progress.json"))


def _scene_identity(root: Path, path: Path, layout: str) -> tuple[str, str, str]:
    parts = path.parent.relative_to(root).parts
    if layout in {"fast", "social"}:
        return str(parts[0]), str(parts[1]), str(parts[2])
    return str(parts[1]), str(parts[2]), str(parts[3])


def _progress_records(roots: Sequence[tuple[Path, str]]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for root, layout in roots:
        for path in _progress_paths(root, layout):
            try:
                payload = _read_json(path)
                family, dataset, scene = _scene_identity(root, path, layout)
                accepted = int(payload.get("accepted_total") or 0)
                target = int(payload.get("accepted_target") or 0)
            except (OSError, ValueError, json.JSONDecodeError, IndexError):
                continue
            records.append(
                {
                    "root": str(root),
                    "root_kind": layout,
                    "family": family,
                    "dataset": dataset,
                    "scene": scene,
                    "scene_key": f"{dataset}/{scene}",
                    "accepted_paths": accepted,
                    "target_paths": target,
                    "status": str(payload.get("status") or "unknown"),
                    "updated_at": payload.get("updated_at"),
                    "scene_dir": str(path.parent),
                }
            )
    return records


def _describe(values: Sequence[int | float]) -> JsonDict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None, "total": 0}

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _sum_counts(values: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: Counter[str] = Counter()
    for value in values:
        counts.update({str(key): int(count) for key, count in value.items()})
    return dict(sorted(counts.items()))


def _ratios(counts: Mapping[str, int]) -> JsonDict:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return {}
    return {key: int(value) / total for key, value in sorted(counts.items())}


def _sample_scenario_paths(
    scene: Mapping[str, Any],
    *,
    paths_per_scene: int,
    fps: float,
    seed: int,
) -> tuple[list[JsonDict], list[str], list[str]]:
    manifest_path = Path(str(scene["scene_dir"])) / "mass_example_manifest.json"
    errors: list[str] = []
    try:
        source_manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [], [], [f"{manifest_path}: {exc}"]
    examples = [item for item in source_manifest.get("examples", []) if isinstance(item, Mapping)]
    ranked = sorted(
        examples,
        key=lambda item: _rank(seed, scene["scene_key"], item.get("scenario_id"), item.get("path")),
    )
    manifests: list[JsonDict] = []
    manifest_paths: list[str] = []
    available_jobs = 0
    for example in ranked:
        path = Path(str(example.get("path") or ""))
        if not path.is_file():
            errors.append(f"missing scenario: {path}")
            continue
        try:
            render_manifest = scenario_file_to_render_manifest(path, fps=float(fps))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        manifests.append(render_manifest)
        manifest_paths.append(str(path))
        available_jobs += len(
            [job for job in render_manifest.get("jobs", []) if isinstance(job, Mapping)]
        )
        if available_jobs >= int(paths_per_scene):
            break
    return manifests, manifest_paths, errors


def _survey_scene(
    family: str,
    scene: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> JsonDict:
    manifests, manifest_paths, errors = _sample_scenario_paths(
        scene,
        paths_per_scene=int(args.paths_per_scene),
        fps=float(args.fps),
        seed=int(args.seed),
    )
    available_source_paths = sum(len(manifest.get("jobs", [])) for manifest in manifests)
    requested_source_paths = min(int(args.paths_per_scene), available_source_paths)
    selector_family = (
        "navigate_with_social_constraints"
        if family.startswith("navigate_with_social_constraints:")
        else family
    )
    selection = select_frame_interest_windows(
        manifests,
        manifest_paths=manifest_paths,
        config=FrameSelectionConfig(
            target_count=int(args.scored_pois_per_source_path) * requested_source_paths,
            mission_families=(selector_family,),
            max_source_paths=int(args.paths_per_scene),
            max_targets_per_job=0,
            preserve_mission_endpoints=True,
            endpoint_window_policy="clamp",
            target_action_ratios=DEFAULT_ACTION_RATIOS,
            enforce_action_minimums=True,
            seed=int(args.seed),
            split="survey",
        ),
    )
    summary = selection["selection_summary"]
    selected_paths = selection["distribution"]["source_path_sampling"]["selected_source_paths"]
    source_frame_counts = [int(item["whole_path_frame_count"]) for item in selected_paths]
    source_frames = sum(source_frame_counts)
    unique_frames = int(summary["unique_source_render_frame_count"])
    repeated_frames = int(summary["requested_window_frame_renders"])
    center_actions = dict(summary.get("selected_center_action_counts", {}))
    poi_actions = dict(summary.get("selected_interest_action_counts", {}))
    anchor_actions = dict(summary.get("mandatory_anchor_action_counts", {}))
    retained_actions = dict(summary.get("selected_window_action_counts", {}))
    unique_actions = dict(summary.get("selected_unique_window_action_counts", {}))
    return {
        "family": family,
        "dataset": str(scene.get("dataset") or ""),
        "scene": str(scene.get("scene") or ""),
        "scene_key": str(scene.get("scene_key") or ""),
        "generated_scenario_count": int(scene.get("accepted_paths", 0)),
        "sampled_scenario_count": len(manifests),
        "available_loaded_source_path_count": available_source_paths,
        "sampled_source_path_count": len(selected_paths),
        "source_path_frame_stats": _describe(source_frame_counts),
        "source_frame_count": source_frames,
        "eligible_center_count": int(summary.get("candidate_count", 0)),
        "selected_interest_count": int(summary.get("selected_interest_target_count", 0)),
        "mandatory_anchor_count": int(summary.get("mandatory_anchor_count", 0)),
        "selected_center_count": int(summary.get("selected_target_count", 0)),
        "selected_center_percent_of_source_frames": (
            int(summary.get("selected_target_count", 0)) / source_frames * 100.0 if source_frames else 0.0
        ),
        "repeated_window_frame_count": repeated_frames,
        "repeated_window_percent_of_source_frames": (
            repeated_frames / source_frames * 100.0 if source_frames else 0.0
        ),
        "unique_retained_frame_count": unique_frames,
        "unique_retained_percent_of_source_frames": (
            unique_frames / source_frames * 100.0 if source_frames else 0.0
        ),
        "actions": {
            "selected_centers": {"counts": center_actions, "ratios": _ratios(center_actions)},
            "scored_poi_centers": {"counts": poi_actions, "ratios": _ratios(poi_actions)},
            "mandatory_anchors": {"counts": anchor_actions, "ratios": _ratios(anchor_actions)},
            "repeated_window_frames": {"counts": retained_actions, "ratios": _ratios(retained_actions)},
            "unique_retained_frames": {"counts": unique_actions, "ratios": _ratios(unique_actions)},
        },
        "target_action_counts": dict(selection["distribution"].get("target_action_counts", {})),
        "action_deficits": dict(summary.get("action_deficits", {})),
        "training_center_sampling": dict(summary.get("training_center_sampling", {})),
        "scored_center_window_composition": dict(
            summary.get("scored_center_window_composition", {})
        ),
        "errors": errors,
        "_source_frame_counts": source_frame_counts,
    }


def _merge_center_window_composition(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    merged: dict[str, JsonDict] = {}
    for record in records:
        for center_action, values in record.get("scored_center_window_composition", {}).items():
            target = merged.setdefault(
                str(center_action),
                {
                    "window_count": 0,
                    "window_frame_count": 0,
                    "matching_action_frame_count": 0,
                    "window_action_counts": Counter(),
                    "windows_with_matching_action_at_least": Counter(),
                },
            )
            target["window_count"] += int(values.get("window_count", 0))
            target["window_frame_count"] += int(values.get("window_frame_count", 0))
            target["matching_action_frame_count"] += int(values.get("matching_action_frame_count", 0))
            target["window_action_counts"].update(values.get("window_action_counts", {}))
            target["windows_with_matching_action_at_least"].update(
                values.get("windows_with_matching_action_at_least", {})
            )
    out: JsonDict = {}
    for center_action, values in sorted(merged.items()):
        window_count = int(values["window_count"])
        action_counts = dict(sorted(values["window_action_counts"].items()))
        out[center_action] = {
            "window_count": window_count,
            "window_frame_count": int(values["window_frame_count"]),
            "window_action_counts": action_counts,
            "window_action_ratios": _ratios(action_counts),
            "matching_action_frame_count": int(values["matching_action_frame_count"]),
            "mean_matching_action_frames": (
                int(values["matching_action_frame_count"]) / window_count if window_count else 0.0
            ),
            "windows_with_matching_action_at_least": dict(
                sorted(values["windows_with_matching_action_at_least"].items())
            ),
        }
    return out


def _training_weights(counts: Mapping[str, int]) -> JsonDict:
    available = [action for action in DEFAULT_ACTION_RATIOS if int(counts.get(action, 0)) > 0]
    ratio_total = sum(float(DEFAULT_ACTION_RATIOS[action]) for action in available)
    ratios = {
        action: float(DEFAULT_ACTION_RATIOS[action]) / ratio_total
        for action in available
    }
    return {
        action: float(ratios[action]) / float(counts[action])
        for action in available
    }


def _survey_family(
    family: str,
    scenes: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
) -> tuple[JsonDict, list[JsonDict]]:
    nonempty = [scene for scene in scenes if int(scene.get("accepted_paths", 0)) > 0]
    selected_scenes = sorted(
        nonempty,
        key=lambda item: _rank(int(args.seed), family, item["scene_key"]),
    )[: int(args.scenes_per_family)]
    tasks = [(family, scene, args) for scene in selected_scenes]
    worker_count = min(max(1, int(args.workers)), max(1, len(tasks)))
    if worker_count > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            scene_results = list(pool.map(_survey_scene_task, tasks))
    else:
        scene_results = [_survey_scene_task(task) for task in tasks]
    source_frame_counts = [
        count
        for record in scene_results
        for count in record.pop("_source_frame_counts", [])
    ]
    source_frames = sum(source_frame_counts)
    actions = {
        name: _sum_counts([record["actions"][name]["counts"] for record in scene_results])
        for name in (
            "selected_centers",
            "scored_poi_centers",
            "mandatory_anchors",
            "repeated_window_frames",
            "unique_retained_frames",
        )
    }
    family_result = {
        "family": family,
        "sampled_scene_count": sum(1 for record in scene_results if record["sampled_source_path_count"]),
        "sampled_scenario_count": sum(record["sampled_scenario_count"] for record in scene_results),
        "sampled_source_path_count": sum(record["sampled_source_path_count"] for record in scene_results),
        "source_path_frame_stats": _describe(source_frame_counts),
        "source_frame_count": source_frames,
        "selected_interest_count": sum(record["selected_interest_count"] for record in scene_results),
        "mandatory_anchor_count": sum(record["mandatory_anchor_count"] for record in scene_results),
        "selected_center_count": sum(record["selected_center_count"] for record in scene_results),
        "repeated_window_frame_count": sum(record["repeated_window_frame_count"] for record in scene_results),
        "unique_retained_frame_count": sum(record["unique_retained_frame_count"] for record in scene_results),
        "unique_retained_percent_of_source_frames": (
            sum(record["unique_retained_frame_count"] for record in scene_results) / source_frames * 100.0
            if source_frames else 0.0
        ),
        "actions": {
            name: {"counts": counts, "ratios": _ratios(counts)}
            for name, counts in actions.items()
        },
        "action_deficits": _sum_counts([record["action_deficits"] for record in scene_results]),
        "training_center_sampling": {
            "population": "scored_poi_centers",
            "mandatory_anchor_policy": "retained_separate_pool",
            "target_action_ratios": dict(DEFAULT_ACTION_RATIOS),
            "per_example_action_weights": _training_weights(actions["scored_poi_centers"]),
        },
        "scored_center_window_composition": _merge_center_window_composition(scene_results),
        "errors": [error for record in scene_results for error in record["errors"]],
    }
    return family_result, scene_results


def _survey_scene_task(task: tuple[str, Mapping[str, Any], argparse.Namespace]) -> JsonDict:
    family, scene, args = task
    return _survey_scene(family, scene, args=args)


def _percent(ratios: Mapping[str, float], action: str) -> str:
    return f"{100.0 * float(ratios.get(action, 0.0)):.2f}%"


def _write_report(payload: Mapping[str, Any], output_path: Path) -> None:
    config = payload["config"]
    totals = payload["sampling"]["totals"]
    lines = [
        "# Anchor-Aware Frame-Sampling Survey",
        "",
        "## Scope",
        "",
        f"- Deterministic seed: `{config['seed']}`.",
        f"- Random scene cohorts per family: up to `{config['scenes_per_family']}`.",
        f"- Random robot source paths per selected scene: up to `{config['paths_per_scene']}`.",
        f"- Scored POI budget: `{config['scored_pois_per_source_path']}` per sampled source path.",
        f"- Scene survey workers: `{config['workers']}`.",
        "- Mandatory mission/checkpoint/path endpoints remain additive and are reported separately.",
        "- Every selected center contributes a 32+1+32 frame context window; no rendering was executed.",
        "",
        "## Aggregate Results",
        "",
        f"- Sampled scenes: **{totals['sampled_scene_count']:,}**.",
        f"- Sampled scenarios: **{totals['sampled_scenario_count']:,}**.",
        f"- Sampled robot source paths: **{totals['sampled_source_path_count']:,}**.",
        f"- Source frames: **{totals['source_frame_count']:,}**.",
        f"- Scored POI centers: **{totals['selected_interest_count']:,}**.",
        f"- Mandatory anchors: **{totals['mandatory_anchor_count']:,}**.",
        f"- Unique retained frames: **{totals['unique_retained_frame_count']:,}** "
        f"({100.0 * totals['unique_retained_frame_count'] / max(1, totals['source_frame_count']):.2f}% of source).",
        "",
        "| Population | Move | Stop | Left | Right |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Scored POI centers", "scored_poi_center_actions"),
        ("Mandatory anchors", "mandatory_anchor_actions"),
        ("All selected centers", "selected_center_actions"),
        ("Repeated 65-frame windows", "repeated_window_actions"),
        ("Unique retained frames", "unique_retained_actions"),
    ):
        ratios = totals[key]["ratios"]
        lines.append(
            f"| {label} | {_percent(ratios, 'move')} | {_percent(ratios, 'stop')} | "
            f"{_percent(ratios, 'turn_left')} | {_percent(ratios, 'turn_right')} |"
        )
    lines.extend(
        [
            "",
            "## Per-Family Results",
            "",
            "| Mission family | Scenes | Source paths | POIs | Anchors | Unique retention | Unique move / stop / left / right |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in payload["sampling"]["families"]:
        ratios = record["actions"]["unique_retained_frames"]["ratios"]
        spread = " / ".join(
            _percent(ratios, action)
            for action in ("move", "stop", "turn_left", "turn_right")
        )
        lines.append(
            f"| `{record['family']}` | {record['sampled_scene_count']:,} | "
            f"{record['sampled_source_path_count']:,} | {record['selected_interest_count']:,} | "
            f"{record['mandatory_anchor_count']:,} | "
            f"{record['unique_retained_percent_of_source_frames']:.2f}% | {spread} |"
        )
    lines.extend(
        [
            "",
            "## Turn-Window Dilution",
            "",
            "These rows condition on the scored center action. They show how many matching-action frames occur inside each 65-frame input.",
            "",
            "| Center action | Windows | Mean matching frames | At least 5 matching frames | At least 10 matching frames |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for action in ("turn_left", "turn_right", "stop", "move"):
        values = totals["scored_center_window_composition"].get(action, {})
        windows = int(values.get("window_count", 0))
        thresholds = values.get("windows_with_matching_action_at_least", {})
        lines.append(
            f"| `{action}` | {windows:,} | {float(values.get('mean_matching_action_frames', 0.0)):.2f} | "
            f"{int(thresholds.get('5', 0)):,} ({100.0 * int(thresholds.get('5', 0)) / max(1, windows):.2f}%) | "
            f"{int(thresholds.get('10', 0)):,} ({100.0 * int(thresholds.get('10', 0)) / max(1, windows):.2f}%) |"
        )
    deficits = totals["action_deficits"]
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Aggregate scored-center action deficits: `{json.dumps(deficits, sort_keys=True)}`.",
            f"- Scene cohorts with errors: **{totals['scene_error_count']:,}**.",
            "- Training balance should use scored center labels or the emitted per-action weights; raw context-frame percentages are not the center-label distribution.",
            "- `render_frame_index` remains the frame-reuse contract for avoiding duplicate rendering across overlapping windows.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    roots = (
        (args.formal_fast_root.resolve(), "fast"),
        (args.formal_slow_root.resolve(), "slow"),
        (args.formal_social_root.resolve(), "social"),
    )
    scene_records = _progress_records(roots)
    scenes_by_family: defaultdict[str, list[JsonDict]] = defaultdict(list)
    for record in scene_records:
        scenes_by_family[str(record["family"])].append(record)

    family_generation: list[JsonDict] = []
    family_sampling: list[JsonDict] = []
    sampling_scenes: list[JsonDict] = []
    for family in sorted(scenes_by_family):
        scenes = scenes_by_family[family]
        path_counts = [int(item["accepted_paths"]) for item in scenes]
        family_generation.append(
            {
                "family": family,
                "scene_count": len(scenes),
                "generated_path_count": sum(path_counts),
                "generated_paths_per_scene": _describe(path_counts),
                "finalized_scene_count": sum(
                    1
                    for item in scenes
                    if item["status"] == "complete"
                ),
                "target_met_scene_count": sum(
                    1
                    for item in scenes
                    if int(item["accepted_paths"]) >= int(item["target_paths"])
                ),
                "target_path_count": sum(int(item["target_paths"]) for item in scenes),
            }
        )
        family_result, scene_results = _survey_family(family, scenes, args=args)
        family_sampling.append(family_result)
        sampling_scenes.extend(scene_results)
        print(
            f"surveyed family={family} scenes={family_result['sampled_scene_count']} "
            f"paths={family_result['sampled_source_path_count']}",
            flush=True,
        )

    selected_center_actions = _sum_counts(
        [item["actions"]["selected_centers"]["counts"] for item in family_sampling]
    )
    scored_poi_center_actions = _sum_counts(
        [item["actions"]["scored_poi_centers"]["counts"] for item in family_sampling]
    )
    mandatory_anchor_actions = _sum_counts(
        [item["actions"]["mandatory_anchors"]["counts"] for item in family_sampling]
    )
    retained_actions = _sum_counts(
        [item["actions"]["repeated_window_frames"]["counts"] for item in family_sampling]
    )
    unique_actions = _sum_counts(
        [item["actions"]["unique_retained_frames"]["counts"] for item in family_sampling]
    )
    payload = {
        "schema_version": "navdp_path_sampling_corpus_survey/v0.2",
        "config": {
            "roots": [str(root) for root, _layout in roots],
            "scenes_per_family": int(args.scenes_per_family),
            "paths_per_scene": int(args.paths_per_scene),
            "scored_pois_per_source_path": int(args.scored_pois_per_source_path),
            "fps": float(args.fps),
            "seed": int(args.seed),
            "workers": int(args.workers),
            "sampling_scope": "deterministic_stratified_survey_not_full_corpus_export",
        },
        "generation": {
            "family_count": len(family_generation),
            "scene_record_count": len(scene_records),
            "generated_path_count": sum(item["generated_path_count"] for item in family_generation),
            "families": family_generation,
            "scenes": scene_records,
        },
        "sampling": {
            "families": family_sampling,
            "totals": {
                "sampled_scenario_count": sum(item["sampled_scenario_count"] for item in family_sampling),
                "sampled_scene_count": sum(item["sampled_scene_count"] for item in family_sampling),
                "sampled_source_path_count": sum(item["sampled_source_path_count"] for item in family_sampling),
                "source_frame_count": sum(item["source_frame_count"] for item in family_sampling),
                "selected_interest_count": sum(item["selected_interest_count"] for item in family_sampling),
                "mandatory_anchor_count": sum(item["mandatory_anchor_count"] for item in family_sampling),
                "selected_center_count": sum(item["selected_center_count"] for item in family_sampling),
                "repeated_window_frame_count": sum(item["repeated_window_frame_count"] for item in family_sampling),
                "unique_retained_frame_count": sum(item["unique_retained_frame_count"] for item in family_sampling),
                "selected_center_actions": {
                    "counts": selected_center_actions,
                    "ratios": _ratios(selected_center_actions),
                },
                "scored_poi_center_actions": {
                    "counts": scored_poi_center_actions,
                    "ratios": _ratios(scored_poi_center_actions),
                },
                "mandatory_anchor_actions": {
                    "counts": mandatory_anchor_actions,
                    "ratios": _ratios(mandatory_anchor_actions),
                },
                "repeated_window_actions": {
                    "counts": retained_actions,
                    "ratios": _ratios(retained_actions),
                },
                "unique_retained_actions": {
                    "counts": unique_actions,
                    "ratios": _ratios(unique_actions),
                },
                "action_deficits": _sum_counts(
                    [item["action_deficits"] for item in family_sampling]
                ),
                "training_center_sampling": {
                    "population": "scored_poi_centers",
                    "mandatory_anchor_policy": "retained_separate_pool",
                    "target_action_ratios": dict(DEFAULT_ACTION_RATIOS),
                    "per_example_action_weights": _training_weights(scored_poi_center_actions),
                },
                "scored_center_window_composition": _merge_center_window_composition(
                    family_sampling
                ),
                "scene_error_count": sum(1 for item in sampling_scenes if item["errors"]),
            },
            "scenes": sampling_scenes,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_scene_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_scene_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(scene_records[0].keys()) if scene_records else ("family",))
        writer.writeheader()
        writer.writerows(scene_records)
    args.output_sampling_csv.parent.mkdir(parents=True, exist_ok=True)
    sampling_rows = []
    for item in sampling_scenes:
        row = {
            key: item[key]
            for key in (
                "family",
                "dataset",
                "scene",
                "scene_key",
                "generated_scenario_count",
                "sampled_scenario_count",
                "sampled_source_path_count",
                "source_frame_count",
                "selected_interest_count",
                "mandatory_anchor_count",
                "selected_center_count",
                "repeated_window_frame_count",
                "unique_retained_frame_count",
                "unique_retained_percent_of_source_frames",
            )
        }
        for population in ("scored_poi_centers", "mandatory_anchors", "unique_retained_frames"):
            for action in ("move", "stop", "turn_left", "turn_right"):
                row[f"{population}_{action}_ratio"] = item["actions"][population]["ratios"].get(action, 0.0)
        row["error_count"] = len(item["errors"])
        sampling_rows.append(row)
    with args.output_sampling_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = tuple(sampling_rows[0].keys()) if sampling_rows else ("family",)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sampling_rows)
    _write_report(payload, args.output_report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "scene_csv": str(args.output_scene_csv),
                "sampling_csv": str(args.output_sampling_csv),
                "report": str(args.output_report),
                "families": len(family_generation),
                "sampled_source_paths": payload["sampling"]["totals"]["sampled_source_path_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
