from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from navdp_datagen.massgen.frame_selection import (
    DEFAULT_ACTION_RATIOS,
    DEFAULT_FUTURE_FRAMES,
    DEFAULT_PAST_FRAMES,
    DEFAULT_SEED,
    FrameSelectionConfig,
    apply_frame_selection_to_manifest,
    load_json,
    select_frame_interest_windows,
    write_json,
)


JsonDict = dict[str, Any]

MISSION_FAMILY_PILOT_SCHEMA_VERSION = "navdp_mission_family_frame_pilot/v0.1"
SOCIAL_FAMILY_VARIANT_LAW_IDS = {
    "navigate_with_social_constraints:personal_space": "L1_personal_space",
    "navigate_with_social_constraints:pedestrian_yield": "L2_non_obstruction_yield",
    "navigate_with_social_constraints:group_integrity": "L3_group_integrity",
    "navigate_with_social_constraints:queue_order": "L4_queue_order",
}


@dataclass(frozen=True)
class MissionFamilyPilotConfig:
    mission_families: tuple[str, ...]
    targets_per_family: int = 2
    past_frames: int = DEFAULT_PAST_FRAMES
    future_frames: int = DEFAULT_FUTURE_FRAMES
    edge_window_policy: str = "reject"
    min_target_spacing_frames: int = 16
    max_targets_per_job: int = 0
    split: str = "pilot"
    seed: int = DEFAULT_SEED
    target_bucket_ratios: Mapping[str, float] | None = None
    target_action_ratios: Mapping[str, float] | None = None
    action_deficit_weight: float = 0.35
    python_bin: str = sys.executable
    remote_sources_connected: bool = False
    densify_frame_gaps: bool = True
    max_source_paths: int = 0
    preserve_mission_endpoints: bool = True
    endpoint_window_policy: str = "clamp"
    render_repo_root: str | Path | None = None

    @property
    def window_frame_count(self) -> int:
        return int(self.past_frames) + 1 + int(self.future_frames)


def prepare_mission_family_pilot(
    manifest_paths: Sequence[str | Path],
    *,
    output_root: str | Path,
    config: MissionFamilyPilotConfig,
) -> JsonDict:
    """Write a non-executing, render-ready pilot plan partitioned by mission family."""

    if not config.mission_families:
        raise ValueError("at least one mission family is required")
    if int(config.targets_per_family) <= 0:
        raise ValueError("targets_per_family must be positive")
    if config.window_frame_count <= 0:
        raise ValueError("pilot window size must be positive")

    root = Path(output_root).resolve()
    normalized_paths = [Path(path).resolve() for path in manifest_paths]
    manifests = [load_json(path) for path in normalized_paths]
    family_records = [
        _prepare_family(
            family,
            manifests=manifests,
            manifest_paths=normalized_paths,
            output_root=root,
            config=config,
        )
        for family in config.mission_families
    ]
    status_counts = _count_values(record["status"] for record in family_records)
    plan: JsonDict = {
        "schema_version": MISSION_FAMILY_PILOT_SCHEMA_VERSION,
        "status": _overall_status(family_records),
        "execution_authorized": False,
        "source_mode": "filesystem_render_manifests",
        "remote_sources_connected": bool(config.remote_sources_connected),
        "config": {
            "mission_families": list(config.mission_families),
            "targets_per_family": int(config.targets_per_family),
            "past_frames": int(config.past_frames),
            "future_frames": int(config.future_frames),
            "window_frame_count": config.window_frame_count,
            "edge_window_policy": str(config.edge_window_policy),
            "min_target_spacing_frames": int(config.min_target_spacing_frames),
            "max_targets_per_job": int(config.max_targets_per_job),
            "split": str(config.split),
            "seed": int(config.seed),
            "densify_frame_gaps": bool(config.densify_frame_gaps),
            "max_source_paths": int(config.max_source_paths),
            "preserve_mission_endpoints": bool(config.preserve_mission_endpoints),
            "endpoint_window_policy": str(config.endpoint_window_policy),
            "render_repo_root": str(
                Path(config.render_repo_root).resolve()
                if config.render_repo_root is not None
                else Path(__file__).resolve().parents[2]
            ),
        },
        "source_manifest_count": len(normalized_paths),
        "source_manifests": [str(path) for path in normalized_paths],
        "summary": {
            "family_count": len(family_records),
            "status_counts": status_counts,
            "selected_target_count": sum(
                int(record.get("selected_target_count", 0)) for record in family_records
            ),
            "selected_interest_target_count": sum(
                int(record.get("selected_interest_target_count", 0)) for record in family_records
            ),
            "mandatory_anchor_count": sum(
                int(record.get("mandatory_anchor_count", 0)) for record in family_records
            ),
            "requested_window_frame_renders": sum(
                int(record.get("requested_window_frame_renders", 0)) for record in family_records
            ),
            "unique_source_render_frame_count": sum(
                int(record.get("unique_source_render_frame_count", 0)) for record in family_records
            ),
        },
        "acceptance_contract": {
            "selection": [
                "the selected center belongs to the requested mission family",
                f"every selected center has exactly {config.window_frame_count} source frames",
                "stationary frame samples are preserved",
                "assigned sub-mission completions and trajectory endpoints are mandatory anchors",
                "whole selected paths are scored and may contribute multiple interest centers",
            ],
            "render_plan": [
                "plan status is ready",
                "planned job count equals selected target count",
                "there are no asset or selected-window truncation blockers",
            ],
            "render_output": [
                f"every selected job produces exactly {config.window_frame_count} temporal frames",
                "camera metadata retains source-frame and center-frame identities",
                "human actor metadata and peer-robot overlay outputs are present when enabled",
            ],
        },
        "families": family_records,
    }
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "pilot_plan.json"
    plan["pilot_plan_path"] = str(plan_path)
    write_json(plan_path, plan)
    return plan


def _prepare_family(
    family: str,
    *,
    manifests: Sequence[Mapping[str, Any]],
    manifest_paths: Sequence[Path],
    output_root: Path,
    config: MissionFamilyPilotConfig,
) -> JsonDict:
    family = str(family)
    all_matched = [
        (manifest, path)
        for manifest, path in zip(manifests, manifest_paths)
        if _manifest_has_family(manifest, family)
    ]
    selection_family = _selection_family(family)
    dedicated = [
        (manifest, path)
        for manifest, path in all_matched
        if _manifest_family_set(manifest) == {selection_family}
    ]
    matched = dedicated or all_matched
    source_selection_policy = "dedicated_single_family" if dedicated else "all_matching_sources"
    family_root = output_root / _safe_id(family)
    base: JsonDict = {
        "mission_family": family,
        "requested_target_count": int(config.targets_per_family),
        "matched_source_manifest_count": len(matched),
        "matched_source_manifests": [str(path) for _manifest, path in matched],
        "source_selection_policy": source_selection_policy,
        "selected_target_count": 0,
        "selected_interest_target_count": 0,
        "mandatory_anchor_count": 0,
        "requested_window_frame_renders": 0,
        "unique_source_render_frame_count": 0,
        "selection_json": None,
        "selected_render_manifests": [],
        "commands": [],
    }
    if not matched:
        base.update(
            {
                "status": "waiting_for_source_manifest",
                "next_action": "connect or copy a render manifest containing this mission family",
            }
        )
        return base

    family_manifests = [manifest for manifest, _path in matched]
    family_paths = [str(path) for _manifest, path in matched]
    selection_config = FrameSelectionConfig(
        target_count=int(config.targets_per_family),
        past_frames=int(config.past_frames),
        future_frames=int(config.future_frames),
        edge_window_policy=str(config.edge_window_policy),
        min_target_spacing_frames=int(config.min_target_spacing_frames),
        max_targets_per_job=int(config.max_targets_per_job),
        split=str(config.split),
        seed=int(config.seed),
        mission_families=(selection_family,),
        densify_frame_gaps=bool(config.densify_frame_gaps),
        max_source_paths=int(config.max_source_paths),
        preserve_mission_endpoints=bool(config.preserve_mission_endpoints),
        endpoint_window_policy=str(config.endpoint_window_policy),
        target_bucket_ratios=config.target_bucket_ratios,
        target_action_ratios=dict(config.target_action_ratios or DEFAULT_ACTION_RATIOS),
        action_deficit_weight=float(config.action_deficit_weight),
    )
    selection = select_frame_interest_windows(
        family_manifests,
        manifest_paths=family_paths,
        config=selection_config,
    )
    selection["pilot_mission_family"] = family
    for chunk in selection.get("chunks", []):
        if isinstance(chunk, dict):
            chunk["pilot_mission_family"] = family
    _validate_family_selection(
        selection,
        family=selection_family,
        pilot_family=family,
        window_frame_count=config.window_frame_count,
    )
    selection_path = family_root / "selection.json"
    selection["selection_path"] = str(selection_path)
    write_json(selection_path, selection)

    summary = selection["selection_summary"]
    selected_count = int(summary["selected_target_count"])
    selected_interest_count = int(summary.get("selected_interest_target_count", selected_count))
    render_manifests: list[str] = []
    commands: list[JsonDict] = []
    for source_index, (manifest, source_path) in enumerate(matched):
        selected_manifest = apply_frame_selection_to_manifest(
            manifest,
            selection,
            manifest_path=source_path,
        )
        selected_jobs = [job for job in selected_manifest.get("jobs", []) if isinstance(job, Mapping)]
        if not selected_jobs:
            continue
        selected_path = (
            family_root
            / "render_manifests"
            / f"{source_index:03d}_{_safe_id(source_path.stem)}.selected.json"
        )
        write_json(selected_path, selected_manifest)
        render_manifests.append(str(selected_path))
        render_output_root = family_root / "render_jobs" / f"source_{source_index:03d}"
        repo_root = (
            Path(config.render_repo_root).resolve()
            if config.render_repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        plan_argv = [
            str(config.python_bin),
            str(repo_root / "scripts" / "massgen" / "render_manifest_jobs.py"),
            "--manifest-json",
            str(selected_path),
            "--output-root",
            str(render_output_root),
            "--write-inputs",
            "--json",
        ]
        commands.append(
            {
                "source_manifest": str(source_path),
                "selected_render_manifest": str(selected_path),
                "selected_window_job_count": len(selected_jobs),
                "working_directory": str(repo_root),
                "plan_argv": plan_argv,
                "execute_argv": [*plan_argv, "--execute"],
                "execution_authorized": False,
            }
        )

    status = "ready"
    next_action = "run each plan_argv after local or remote scene assets are connected"
    if selected_interest_count == 0 and selected_count == 0:
        status = "no_eligible_65_frame_window"
        next_action = "provide trajectories with enough frames for the configured temporal window"
    elif selected_interest_count < int(config.targets_per_family):
        status = "insufficient_candidates"
        next_action = "inspect candidate coverage before accepting a reduced family pilot"

    base.update(
        {
            "status": status,
            "next_action": next_action,
            "selected_target_count": selected_count,
            "selected_interest_target_count": selected_interest_count,
            "mandatory_anchor_count": int(summary.get("mandatory_anchor_count", 0)),
            "requested_window_frame_renders": int(summary["requested_window_frame_renders"]),
            "unique_source_render_frame_count": int(summary["unique_source_render_frame_count"]),
            "available_bucket_counts": selection["distribution"]["available_bucket_counts"],
            "available_action_counts": selection["distribution"]["available_action_counts"],
            "selected_bucket_counts": summary["selected_bucket_counts"],
            "selected_action_counts": summary["selected_action_counts"],
            "selection_json": str(selection_path),
            "selected_render_manifests": render_manifests,
            "commands": commands,
        }
    )
    return base


def _manifest_has_family(manifest: Mapping[str, Any], family: str) -> bool:
    selection_family = _selection_family(family)
    if selection_family in _manifest_family_set(manifest):
        required_law = SOCIAL_FAMILY_VARIANT_LAW_IDS.get(family)
        if required_law is None:
            return True
        return required_law in {str(item) for item in manifest.get("social_law_ids", [])}
    if selection_family in {
        str(item)
        for job in manifest.get("jobs", [])
        if isinstance(job, Mapping)
        for item in job.get("mission_families", [])
    }:
        required_law = SOCIAL_FAMILY_VARIANT_LAW_IDS.get(family)
        if required_law is None:
            return True
        return required_law in {str(item) for item in manifest.get("social_law_ids", [])}
    return False


def _selection_family(family: str) -> str:
    if family in SOCIAL_FAMILY_VARIANT_LAW_IDS:
        return "navigate_with_social_constraints"
    return str(family)


def _manifest_family_set(manifest: Mapping[str, Any]) -> set[str]:
    families = {str(item) for item in manifest.get("mission_families", [])}
    if families:
        return families
    return {
        str(item)
        for job in manifest.get("jobs", [])
        if isinstance(job, Mapping)
        for item in job.get("mission_families", [])
    }


def _validate_family_selection(
    selection: Mapping[str, Any],
    *,
    family: str,
    pilot_family: str,
    window_frame_count: int,
) -> None:
    for chunk in selection.get("chunks", []):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"{family}: selection chunk must be an object")
        chunk_families = {str(item) for item in chunk.get("mission_families", [])}
        if family not in chunk_families:
            raise ValueError(f"{family}: selected chunk does not belong to the requested family")
        if str(chunk.get("pilot_mission_family")) != pilot_family:
            raise ValueError(f"{pilot_family}: selected chunk is missing its pilot-family identity")
        window = chunk.get("window", {})
        source_frames = window.get("source_frame_indices", []) if isinstance(window, Mapping) else []
        if len(source_frames) != int(window_frame_count):
            raise ValueError(
                f"{family}: selected chunk has {len(source_frames)} frames; "
                f"expected {window_frame_count}"
            )
        render_contract = chunk.get("render_contract", {})
        if not isinstance(render_contract, Mapping) or not render_contract.get(
            "preserve_stationary_frames"
        ):
            raise ValueError(f"{family}: selected chunk does not preserve stationary frames")


def _count_values(values: Iterable[str]) -> JsonDict:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _overall_status(records: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(record.get("status")) for record in records}
    if statuses == {"ready"}:
        return "ready_to_plan"
    if statuses <= {"waiting_for_source_manifest"}:
        return "waiting_for_sources"
    return "partially_ready"


def _safe_id(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)]
    return "".join(safe).strip("_") or "family"
