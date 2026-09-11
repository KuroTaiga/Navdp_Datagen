from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

FRAME_SELECTION_SCHEMA_VERSION = "navdp_frame_interest_selection/v0.1"
DEFAULT_PAST_FRAMES = 32
DEFAULT_FUTURE_FRAMES = 0
DEFAULT_SEED = 20260907
INTEREST_BUCKETS = (
    "critical_margin",
    "human_interaction",
    "route_decision",
    "representative_motion",
)
ACTION_NAMES = ("stop", "move", "turn_left", "turn_right")

DEFAULT_BUCKET_PRIOR_WEIGHTS = {
    "critical_margin": 1.80,
    "human_interaction": 1.55,
    "route_decision": 1.30,
    "representative_motion": 0.70,
}
DEFAULT_BUCKET_MIN_SHARE = {
    "critical_margin": 0.15,
    "human_interaction": 0.15,
    "route_decision": 0.15,
    "representative_motion": 0.10,
}
DEFAULT_BUCKET_MAX_SHARE = {
    "critical_margin": 0.40,
    "human_interaction": 0.35,
    "route_decision": 0.30,
    "representative_motion": 0.25,
}
DEFAULT_ACTION_RATIOS = {
    "stop": 0.15,
    "move": 0.55,
    "turn_left": 0.15,
    "turn_right": 0.15,
}
SEMANTIC_EPISODE_POLICIES = ("none", "guarantee_representative")
DEFAULT_SEMANTIC_EPISODE_POLICY = "guarantee_representative"
GUARANTEED_EPISODE_SIGNALS = (
    "instruction_turn",
    "semantic_stop",
    "instruction_section_boundary",
    "room_transition",
    "collision_avoidance",
    "social_law_decision",
    "human_interaction_decision",
    "traffic_wait",
    "target_human_visible",
    "route_deviation",
    "planned_execution_mismatch",
)


@dataclass(frozen=True)
class FrameSelectionConfig:
    target_count: int = 0
    past_frames: int = DEFAULT_PAST_FRAMES
    future_frames: int = DEFAULT_FUTURE_FRAMES
    edge_window_policy: str = "reject"
    min_target_spacing_frames: int = 16
    max_targets_per_job: int = 0
    seed: int = DEFAULT_SEED
    split: str = "train"
    mission_families: tuple[str, ...] = ()
    densify_frame_gaps: bool = True
    max_source_paths: int = 0
    preserve_mission_endpoints: bool = True
    endpoint_window_policy: str = "clamp"
    target_bucket_ratios: Mapping[str, float] | None = None
    target_action_ratios: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTION_RATIOS))
    action_deficit_weight: float = 0.35
    bucket_deficit_weight: float = 0.35
    enforce_action_minimums: bool = True
    semantic_episode_policy: str = DEFAULT_SEMANTIC_EPISODE_POLICY

    @property
    def window_frame_count(self) -> int:
        return int(self.past_frames) + 1 + int(self.future_frames)

    def to_json(self) -> JsonDict:
        return {
            "target_count": int(self.target_count),
            "past_frames": int(self.past_frames),
            "future_frames": int(self.future_frames),
            "window_frame_count": self.window_frame_count,
            "edge_window_policy": str(self.edge_window_policy),
            "min_target_spacing_frames": int(self.min_target_spacing_frames),
            "max_targets_per_job": int(self.max_targets_per_job),
            "seed": int(self.seed),
            "split": str(self.split),
            "mission_families": list(self.mission_families),
            "densify_frame_gaps": bool(self.densify_frame_gaps),
            "max_source_paths": int(self.max_source_paths),
            "preserve_mission_endpoints": bool(self.preserve_mission_endpoints),
            "endpoint_window_policy": str(self.endpoint_window_policy),
            "target_bucket_ratios": (
                dict(self.target_bucket_ratios) if self.target_bucket_ratios is not None else None
            ),
            "target_action_ratios": dict(self.target_action_ratios),
            "action_deficit_weight": float(self.action_deficit_weight),
            "bucket_deficit_weight": float(self.bucket_deficit_weight),
            "enforce_action_minimums": bool(self.enforce_action_minimums),
            "semantic_episode_policy": str(self.semantic_episode_policy),
            "distribution_policy": "anchor_aware_semantic_center_balance/v0.5",
        }


@dataclass(frozen=True)
class FrameCandidate:
    manifest_index: int
    manifest_path: str | None
    job_id: str
    scene_id: str
    viewpoint_robot_id: str
    source_frame_index: int
    source_frame_id: int
    source_window_indices: tuple[int, ...]
    source_window_actions: tuple[str, ...]
    window_edge_policy: str
    time_s: float
    position: tuple[float, float, float]
    yaw_rad: float
    motion_state: str
    navigation: Mapping[str, Any]
    navigation_signals: tuple[str, ...]
    mission_families: tuple[str, ...]
    human_actor_ids: tuple[str, ...]
    peer_robot_ids: tuple[str, ...]
    target_action_name: str
    target_bucket: str
    bucket_scores: Mapping[str, float]
    interest_score: float
    reasons: tuple[str, ...]
    mandatory_anchor_types: tuple[str, ...]
    stable_key: str


def load_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_frame_interest_windows_from_paths(
    manifest_paths: Sequence[str | Path],
    *,
    config: FrameSelectionConfig | None = None,
) -> JsonDict:
    manifests: list[JsonDict] = []
    paths: list[str] = []
    for raw_path in manifest_paths:
        path = Path(raw_path)
        manifests.append(load_json(path))
        paths.append(str(path))
    return select_frame_interest_windows(manifests, manifest_paths=paths, config=config)


def select_frame_interest_windows(
    manifests: Sequence[Mapping[str, Any]],
    *,
    manifest_paths: Sequence[str | None] | None = None,
    config: FrameSelectionConfig | None = None,
) -> JsonDict:
    cfg = config or FrameSelectionConfig()
    _validate_config(cfg)
    paths = list(manifest_paths or [None] * len(manifests))
    selected_source_paths = _select_source_paths(manifests, paths, cfg)
    selected_source_path_keys = {
        (int(record["source_manifest_index"]), str(record["source_job_id"]))
        for record in selected_source_paths
    }
    candidates = _build_candidates(
        manifests,
        paths,
        cfg,
        selected_source_path_keys=selected_source_path_keys,
    )
    interest_candidates = [candidate for candidate in candidates if not candidate.mandatory_anchor_types]
    bucket_counts = _count_by(candidates, lambda candidate: candidate.target_bucket)
    interest_bucket_counts = _count_by(
        interest_candidates,
        lambda candidate: candidate.target_bucket,
    )
    action_candidate_counts = _count_by(candidates, lambda candidate: candidate.target_action_name)
    interest_action_candidate_counts = _count_by(
        interest_candidates,
        lambda candidate: candidate.target_action_name,
    )
    mission_family_candidate_counts = _count_memberships(
        candidates,
        lambda candidate: candidate.mission_families,
    )
    navigation_signal_candidate_counts = _count_memberships(
        candidates,
        lambda candidate: candidate.navigation_signals,
    )
    available_mandatory_anchor_type_counts = _count_memberships(
        candidates,
        lambda candidate: candidate.mandatory_anchor_types,
    )
    requested_target_count = (
        int(cfg.target_count)
        if int(cfg.target_count) > 0
        else len(interest_candidates)
    )
    requested_target_count = min(requested_target_count, len(interest_candidates))
    episode_representatives = _semantic_episode_representatives(candidates, cfg)
    target_count = min(
        len(interest_candidates),
        max(requested_target_count, len(episode_representatives)),
    )
    bucket_ratios = _derive_bucket_ratios(
        interest_bucket_counts,
        override=cfg.target_bucket_ratios,
    )
    bucket_targets = _integer_targets(bucket_ratios, target_count)
    action_ratios = _normalize_ratios(cfg.target_action_ratios, ACTION_NAMES)
    action_targets = _capacity_constrained_targets(
        action_ratios,
        target_count,
        capacities=interest_action_candidate_counts,
    )
    selected = _select_candidates(
        candidates,
        target_count=target_count,
        bucket_targets=bucket_targets,
        action_targets=action_targets,
        action_candidate_counts=interest_action_candidate_counts,
        episode_representatives=episode_representatives,
        config=cfg,
    )
    (
        available_navigation_signal_episode_counts,
        selected_navigation_signal_episode_counts,
        selected_interest_navigation_signal_episode_counts,
    ) = _navigation_signal_episode_coverage(candidates, selected)
    chunks = [
        _chunk_record(candidate, index=index, config=cfg)
        for index, candidate in enumerate(selected)
    ]
    return {
        "schema_version": FRAME_SELECTION_SCHEMA_VERSION,
        "config": cfg.to_json(),
        "source_manifests": _source_manifest_records(manifests, paths),
        "distribution": {
            "basis": (
                "Randomly choose complete source paths, score every frame on each selected path, "
                "balance scored centers with availability-aware action minimums, and retain "
                "mandatory mission/section endpoints in a separate additive pool."
            ),
            "source_path_sampling": {
                "policy": "deterministic_seeded_whole_path",
                "available_source_path_count": _eligible_source_path_count(manifests, cfg),
                "selected_source_path_count": len(selected_source_paths),
                "max_source_paths": int(cfg.max_source_paths),
                "whole_path_scoring": True,
                "multiple_targets_per_source_path_allowed": int(cfg.max_targets_per_job) == 0,
                "selected_source_paths": selected_source_paths,
            },
            "available_bucket_counts": bucket_counts,
            "available_interest_bucket_counts": interest_bucket_counts,
            "available_action_counts": action_candidate_counts,
            "available_interest_action_counts": interest_action_candidate_counts,
            "available_mission_family_counts": mission_family_candidate_counts,
            "available_navigation_signal_counts": navigation_signal_candidate_counts,
            "available_mandatory_anchor_type_counts": available_mandatory_anchor_type_counts,
            "available_navigation_signal_episode_counts": (
                available_navigation_signal_episode_counts
            ),
            "target_bucket_ratios": bucket_ratios,
            "target_bucket_counts": bucket_targets,
            "target_action_ratios": action_ratios,
            "target_action_counts": action_targets,
            "action_selection_policy": "availability_aware_scored_center_minimums",
            "semantic_episode_policy": str(cfg.semantic_episode_policy),
            "guaranteed_episode_representative_count": len(episode_representatives),
        },
        "selection_summary": _selection_summary(
            candidates=candidates,
            selected=selected,
            chunks=chunks,
            target_count=target_count,
            bucket_targets=bucket_targets,
            action_targets=action_targets,
            available_navigation_signal_episode_counts=(
                available_navigation_signal_episode_counts
            ),
            selected_navigation_signal_episode_counts=(
                selected_navigation_signal_episode_counts
            ),
            selected_interest_navigation_signal_episode_counts=(
                selected_interest_navigation_signal_episode_counts
            ),
            config=cfg,
        ),
        "chunks": chunks,
        "render_frame_index": _render_frame_index(chunks),
    }


def apply_frame_selection_to_manifest(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> JsonDict:
    """Return a render manifest whose jobs are selected temporal windows."""

    source_chunks = _chunks_for_manifest(selection, manifest_path=manifest_path)
    out = deepcopy(dict(manifest))
    out.setdefault("source", {})
    if isinstance(out["source"], dict):
        out["source"]["frame_selection_schema_version"] = str(selection.get("schema_version") or "")
        out["source"]["frame_selection_manifest_path"] = str(selection.get("selection_path") or "")
        if manifest_path is not None:
            out["source"]["parent_render_manifest_path"] = str(manifest_path)
    out.setdefault("metadata", {})
    if isinstance(out["metadata"], dict):
        out["metadata"]["frame_selection"] = {
            "enabled": True,
            "selection_schema_version": str(selection.get("schema_version") or ""),
            "selected_window_count": len(source_chunks),
            "window_frame_count": int(
                selection.get("config", {}).get("window_frame_count")
                or DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
            ),
        }
    rendering_contract = out.get("rendering_metadata_contract")
    if not isinstance(rendering_contract, dict):
        rendering_contract = {}
        out["rendering_metadata_contract"] = rendering_contract
    rendering_contract.update(
        {
            "schema_version": str(
                rendering_contract.get("schema_version") or "navdp_rendering_metadata/v1.0"
            ),
            "manifest_job_scope": "selected_windows",
            "source_jobs_contain_complete_trajectories": False,
            "complete_robot_tracks_retained": True,
            "complete_robot_tracks_path": "actors.robots[*].trajectory",
            "selection_policy": "producer_selection_manifest",
            "preserve_stationary_samples": True,
        }
    )

    jobs = [job for job in manifest.get("jobs", []) if isinstance(job, Mapping)]
    job_by_id = {str(job.get("job_id")): job for job in jobs}
    selected_jobs: list[JsonDict] = []
    for chunk in source_chunks:
        source_job_id = str(chunk.get("source_job_id") or chunk.get("job_id") or "")
        source_job = job_by_id.get(source_job_id)
        if source_job is None:
            continue
        selected_jobs.append(_windowed_job(source_job, chunk))
    out["jobs"] = selected_jobs
    out["warnings"] = list(out.get("warnings", [])) if isinstance(out.get("warnings"), list) else []
    if not selected_jobs:
        out["warnings"].append("frame selection did not match any jobs in this render manifest")
    timing = dict(out.get("timing", {})) if isinstance(out.get("timing"), Mapping) else {}
    original_timing = dict(timing)
    timing.update(
        {
            "selection_mode": "frame_interest_windows",
            "selected_window_count": len(selected_jobs),
            "window_frame_count": int(
                selection.get("config", {}).get("window_frame_count")
                or DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
            ),
            "original_timing": original_timing,
        }
    )
    out["timing"] = timing
    return out


def _validate_config(config: FrameSelectionConfig) -> None:
    if int(config.past_frames) < 0 or int(config.future_frames) < 0:
        raise ValueError("past_frames and future_frames must be non-negative")
    if int(config.window_frame_count) <= 0:
        raise ValueError("window_frame_count must be positive")
    if str(config.edge_window_policy) not in {"reject", "clamp"}:
        raise ValueError("edge_window_policy must be 'reject' or 'clamp'")
    if str(config.endpoint_window_policy) not in {"reject", "clamp"}:
        raise ValueError("endpoint_window_policy must be 'reject' or 'clamp'")
    if int(config.target_count) < 0:
        raise ValueError("target_count must be non-negative")
    if int(config.max_source_paths) < 0:
        raise ValueError("max_source_paths must be non-negative")
    if str(config.semantic_episode_policy) not in SEMANTIC_EPISODE_POLICIES:
        raise ValueError(
            f"semantic_episode_policy must be one of {SEMANTIC_EPISODE_POLICIES}"
        )
    if any(not str(family).strip() for family in config.mission_families):
        raise ValueError("mission_families cannot contain empty values")


def _job_mission_families(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[str, ...]:
    families = tuple(str(item) for item in job.get("mission_families", []) if str(item))
    if families:
        return families
    return tuple(str(item) for item in manifest.get("mission_families", []) if str(item))


def _eligible_source_path_records(
    manifests: Sequence[Mapping[str, Any]],
    manifest_paths: Sequence[str | None],
    config: FrameSelectionConfig,
) -> list[JsonDict]:
    requested_families = {str(item) for item in config.mission_families}
    records: list[JsonDict] = []
    for manifest_index, manifest in enumerate(manifests):
        manifest_path = manifest_paths[manifest_index] if manifest_index < len(manifest_paths) else None
        for job in manifest.get("jobs", []):
            if not isinstance(job, Mapping):
                continue
            families = _job_mission_families(manifest, job)
            if requested_families and requested_families.isdisjoint(families):
                continue
            job_id = str(job.get("job_id") or "")
            trajectory = _job_trajectory(job, densify_frame_gaps=bool(config.densify_frame_gaps))
            if not trajectory:
                continue
            random_key = hashlib.sha256(
                f"{int(config.seed)}::{manifest_path or manifest_index}::{job_id}".encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "source_manifest_index": int(manifest_index),
                    "source_manifest_path": manifest_path,
                    "source_job_id": job_id,
                    "scene_id": str(job.get("scene_id") or manifest.get("source", {}).get("scene_id") or ""),
                    "viewpoint_robot_id": str(job.get("viewpoint_robot_id") or ""),
                    "mission_families": list(families),
                    "whole_path_frame_count": len(trajectory),
                    "random_rank_key": random_key[:16],
                }
            )
    return records


def _select_source_paths(
    manifests: Sequence[Mapping[str, Any]],
    manifest_paths: Sequence[str | None],
    config: FrameSelectionConfig,
) -> list[JsonDict]:
    records = _eligible_source_path_records(manifests, manifest_paths, config)
    ranked = sorted(records, key=lambda record: (str(record["random_rank_key"]), str(record["source_job_id"])))
    if int(config.max_source_paths) > 0:
        ranked = ranked[: int(config.max_source_paths)]
    return sorted(
        ranked,
        key=lambda record: (int(record["source_manifest_index"]), str(record["source_job_id"])),
    )


def _eligible_source_path_count(
    manifests: Sequence[Mapping[str, Any]],
    config: FrameSelectionConfig,
) -> int:
    return len(_eligible_source_path_records(manifests, [None] * len(manifests), config))


def _build_candidates(
    manifests: Sequence[Mapping[str, Any]],
    manifest_paths: Sequence[str | None],
    config: FrameSelectionConfig,
    *,
    selected_source_path_keys: set[tuple[int, str]],
) -> list[FrameCandidate]:
    candidates: list[FrameCandidate] = []
    for manifest_index, manifest in enumerate(manifests):
        manifest_path = manifest_paths[manifest_index] if manifest_index < len(manifest_paths) else None
        actors = manifest.get("actors", {}) if isinstance(manifest.get("actors"), Mapping) else {}
        humans = [human for human in actors.get("humans", []) if isinstance(human, Mapping)]
        human_tracks = [
            _position_samples(human.get("trajectory", []), fallback=human.get("start_pose"))
            for human in humans
        ]
        human_tracks = [track for track in human_tracks if track]
        jobs = [job for job in manifest.get("jobs", []) if isinstance(job, Mapping)]
        event_times = _event_times(manifest)
        for job in jobs:
            job_id = str(job.get("job_id") or "")
            if (manifest_index, job_id) not in selected_source_path_keys:
                continue
            trajectory = _job_trajectory(job, densify_frame_gaps=bool(config.densify_frame_gaps))
            if not trajectory:
                continue
            action_names = tuple(
                _action_name(trajectory, index) for index in range(len(trajectory))
            )
            navigation_signals_by_index = tuple(
                _navigation_signals(trajectory, index) for index in range(len(trajectory))
            )
            peer_tracks = [
                _position_samples(track.get("trajectory", []))
                for track in job.get("peer_robot_pose_tracks", [])
                if isinstance(track, Mapping)
            ]
            peer_tracks = [track for track in peer_tracks if track]
            mandatory_anchors = _mandatory_anchor_indices(
                manifest,
                job,
                trajectory,
                enabled=bool(config.preserve_mission_endpoints),
            )
            human_ids = tuple(str(item) for item in job.get("human_actor_ids", []) if str(item))
            peer_robot_ids = tuple(str(item) for item in job.get("peer_robot_ids", []) if str(item))
            family_tuple = _job_mission_families(manifest, job)
            for frame_index, point in enumerate(trajectory):
                source_window_indices = _window_indices(frame_index, len(trajectory), config)
                window_edge_policy = str(config.edge_window_policy)
                anchor_types = mandatory_anchors.get(frame_index, ())
                if not source_window_indices and anchor_types:
                    source_window_indices = _window_indices(
                        frame_index,
                        len(trajectory),
                        config,
                        policy=str(config.endpoint_window_policy),
                    )
                    window_edge_policy = str(config.endpoint_window_policy)
                if not source_window_indices:
                    continue
                target_action_name = action_names[frame_index]
                navigation_signals = navigation_signals_by_index[frame_index]
                source_window_actions = tuple(
                    action_names[int(source_index)]
                    for source_index in source_window_indices
                )
                point_position = _point_position(point)
                point_time = _point_time(point, frame_index)
                bucket_scores = _bucket_scores(
                    trajectory,
                    frame_index,
                    event_times=event_times,
                    target_action_name=target_action_name,
                    navigation_signals=navigation_signals,
                    human_min=_min_track_distance(point_position, point_time, human_tracks),
                    peer_min=_min_track_distance(point_position, point_time, peer_tracks),
                )
                target_bucket = max(INTEREST_BUCKETS, key=lambda bucket: bucket_scores.get(bucket, 0.0))
                reasons = _candidate_reasons(
                    target_bucket,
                    bucket_scores,
                    target_action_name,
                    navigation_signals,
                )
                score = _interest_score(bucket_scores)
                source_frame_id = _int_or(point.get("frame"), frame_index)
                stable_key = _stable_key(
                    manifest_path,
                    manifest_index,
                    str(job.get("job_id") or ""),
                    frame_index,
                    source_frame_id,
                )
                candidates.append(
                    FrameCandidate(
                        manifest_index=manifest_index,
                        manifest_path=manifest_path,
                        job_id=job_id,
                        scene_id=str(job.get("scene_id") or manifest.get("source", {}).get("scene_id") or ""),
                        viewpoint_robot_id=str(job.get("viewpoint_robot_id") or ""),
                        source_frame_index=frame_index,
                        source_frame_id=source_frame_id,
                        source_window_indices=tuple(int(item) for item in source_window_indices),
                        source_window_actions=source_window_actions,
                        window_edge_policy=window_edge_policy,
                        time_s=point_time,
                        position=point_position,
                        yaw_rad=float(point.get("yaw_rad", 0.0) or 0.0),
                        motion_state=str(point.get("motion_state") or ""),
                        navigation=_point_navigation(point),
                        navigation_signals=navigation_signals,
                        mission_families=family_tuple,
                        human_actor_ids=human_ids,
                        peer_robot_ids=peer_robot_ids,
                        target_action_name=target_action_name,
                        target_bucket=target_bucket,
                        bucket_scores=bucket_scores,
                        interest_score=score,
                        reasons=tuple((*reasons, *(f"mandatory:{item}" for item in anchor_types))),
                        mandatory_anchor_types=tuple(anchor_types),
                        stable_key=stable_key,
                    )
                )
    return candidates


def _derive_bucket_ratios(
    available_counts: Mapping[str, int],
    *,
    override: Mapping[str, float] | None,
) -> JsonDict:
    available = [bucket for bucket in INTEREST_BUCKETS if int(available_counts.get(bucket, 0)) > 0]
    if not available:
        return {}
    if override:
        return _normalize_ratios(override, available)
    total = float(sum(int(available_counts.get(bucket, 0)) for bucket in available))
    raw = {
        bucket: math.sqrt(float(available_counts.get(bucket, 0)) / total) * DEFAULT_BUCKET_PRIOR_WEIGHTS[bucket]
        for bucket in available
    }
    return _bounded_normalize(
        raw,
        lower={bucket: DEFAULT_BUCKET_MIN_SHARE[bucket] for bucket in available},
        upper={bucket: DEFAULT_BUCKET_MAX_SHARE[bucket] for bucket in available},
    )


def _bounded_normalize(
    raw: Mapping[str, float],
    *,
    lower: Mapping[str, float],
    upper: Mapping[str, float],
) -> JsonDict:
    active = {key for key, value in raw.items() if value > 0.0}
    if not active:
        return {}
    lower_sum = sum(max(0.0, float(lower.get(key, 0.0))) for key in active)
    if lower_sum >= 1.0:
        return {
            key: round(max(0.0, float(lower.get(key, 0.0))) / lower_sum, 6)
            for key in sorted(active)
        }
    assigned: dict[str, float] = {}
    remaining = 1.0
    while active:
        denom = sum(float(raw[key]) for key in active)
        if denom <= 0.0:
            even = remaining / float(len(active))
            for key in list(active):
                assigned[key] = even
            break
        changed = False
        for key in list(active):
            value = remaining * float(raw[key]) / denom
            low = max(0.0, float(lower.get(key, 0.0)))
            high = min(1.0, float(upper.get(key, 1.0)))
            if value < low:
                assigned[key] = low
                remaining -= low
                active.remove(key)
                changed = True
            elif value > high:
                assigned[key] = high
                remaining -= high
                active.remove(key)
                changed = True
        if not changed:
            for key in list(active):
                assigned[key] = remaining * float(raw[key]) / denom
            break
    norm = sum(max(0.0, value) for value in assigned.values())
    return {key: round(max(0.0, assigned[key]) / norm, 6) for key in sorted(assigned)} if norm > 0.0 else {}


def _normalize_ratios(ratios: Mapping[str, float], keys: Sequence[str]) -> JsonDict:
    values = {str(key): max(0.0, float(ratios.get(key, 0.0))) for key in keys}
    total = sum(values.values())
    if total <= 0.0:
        even = 1.0 / float(len(keys)) if keys else 0.0
        return {str(key): round(even, 6) for key in keys}
    return {key: round(value / total, 6) for key, value in values.items() if value > 0.0}


def _integer_targets(ratios: Mapping[str, float], total: int) -> JsonDict:
    if total <= 0 or not ratios:
        return {}
    raw = {key: float(value) * float(total) for key, value in ratios.items()}
    out = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(out.values())
    remainders = sorted(raw.items(), key=lambda item: (item[1] - math.floor(item[1]), item[0]), reverse=True)
    for key, _value in remainders[:remaining]:
        out[key] += 1
    return dict(sorted(out.items()))


def _capacity_constrained_targets(
    ratios: Mapping[str, float],
    total: int,
    *,
    capacities: Mapping[str, int],
) -> JsonDict:
    """Allocate integer action goals and redistribute unavailable capacity."""

    remaining = max(0, int(total))
    targets = {str(key): 0 for key in ratios}
    active = {
        str(key)
        for key, value in ratios.items()
        if float(value) > 0.0 and int(capacities.get(key, 0)) > 0
    }
    while remaining > 0 and active:
        normalized = _normalize_ratios(ratios, sorted(active))
        proposed = _integer_targets(normalized, remaining)
        saturated = False
        for key in sorted(active):
            capacity_left = int(capacities.get(key, 0)) - int(targets.get(key, 0))
            requested = int(proposed.get(key, 0))
            if requested >= capacity_left:
                targets[key] += max(0, capacity_left)
                remaining -= max(0, capacity_left)
                active.remove(key)
                saturated = True
        if saturated:
            continue
        for key, count in proposed.items():
            targets[key] += int(count)
            remaining -= int(count)
        break
    return dict(sorted((key, value) for key, value in targets.items() if value > 0))


def _select_candidates(
    candidates: Sequence[FrameCandidate],
    *,
    target_count: int,
    bucket_targets: Mapping[str, int],
    action_targets: Mapping[str, int],
    action_candidate_counts: Mapping[str, int],
    episode_representatives: Sequence[FrameCandidate],
    config: FrameSelectionConfig,
) -> list[FrameCandidate]:
    selected: list[FrameCandidate] = []
    selected_keys: set[str] = set()
    per_job_counts: dict[str, int] = {}
    selected_by_job: dict[str, list[int]] = {}
    action_counts: dict[str, int] = {name: 0 for name in ACTION_NAMES}
    bucket_counts: dict[str, int] = {name: 0 for name in INTEREST_BUCKETS}
    candidate_pool = _CandidatePool(candidates, config)

    mandatory = sorted(
        (candidate for candidate in candidates if candidate.mandatory_anchor_types),
        key=lambda item: (item.manifest_index, item.scene_id, item.job_id, item.source_frame_index),
    )
    for candidate in mandatory:
        _mark_selected(
            candidate,
            selected,
            selected_keys,
            selected_by_job,
            per_job_counts,
            action_counts,
            bucket_counts,
        )

    for candidate in episode_representatives:
        if candidate.stable_key in selected_keys:
            continue
        _mark_selected(
            candidate,
            selected,
            selected_keys,
            selected_by_job,
            per_job_counts,
            action_counts,
            bucket_counts,
        )

    if config.enforce_action_minimums:
        action_order = sorted(
            action_targets,
            key=lambda action: (
                float(action_candidate_counts.get(action, 0))
                / max(1, int(action_targets.get(action, 0))),
                action,
            ),
        )
        for action in action_order:
            while (
                _interest_selected_count(selected) < target_count
                and int(action_counts.get(action, 0)) < int(action_targets.get(action, 0))
            ):
                candidate = _best_candidate(
                    candidate_pool,
                    selected_keys=selected_keys,
                    selected_by_job=selected_by_job,
                    per_job_counts=per_job_counts,
                    action_counts=action_counts,
                    bucket_counts=bucket_counts,
                    bucket_targets=bucket_targets,
                    config=config,
                    bucket=None,
                    target_action=action,
                    enforce_spacing=True,
                )
                if candidate is None:
                    break
                _mark_selected(
                    candidate,
                    selected,
                    selected_keys,
                    selected_by_job,
                    per_job_counts,
                    action_counts,
                    bucket_counts,
                )

    for bucket in INTEREST_BUCKETS:
        while (
            _interest_selected_count(selected) < target_count
            and int(bucket_counts.get(bucket, 0)) < int(bucket_targets.get(bucket, 0))
        ):
            candidate = _best_candidate(
                candidate_pool,
                selected_keys=selected_keys,
                selected_by_job=selected_by_job,
                per_job_counts=per_job_counts,
                action_counts=action_counts,
                bucket_counts=bucket_counts,
                bucket_targets=bucket_targets,
                config=config,
                bucket=bucket,
                target_action=None,
                enforce_spacing=True,
            )
            if candidate is None:
                break
            _mark_selected(
                candidate,
                selected,
                selected_keys,
                selected_by_job,
                per_job_counts,
                action_counts,
                bucket_counts,
            )

    while _interest_selected_count(selected) < target_count:
        candidate = _best_candidate(
            candidate_pool,
            selected_keys=selected_keys,
            selected_by_job=selected_by_job,
            per_job_counts=per_job_counts,
            action_counts=action_counts,
            bucket_counts=bucket_counts,
            bucket_targets=bucket_targets,
            config=config,
            bucket=None,
            target_action=None,
            enforce_spacing=True,
        )
        if candidate is None:
            candidate = _best_candidate(
                candidate_pool,
                selected_keys=selected_keys,
                selected_by_job=selected_by_job,
                per_job_counts=per_job_counts,
                action_counts=action_counts,
                bucket_counts=bucket_counts,
                bucket_targets=bucket_targets,
                config=config,
                bucket=None,
                target_action=None,
                enforce_spacing=False,
            )
        if candidate is None:
            break
        _mark_selected(
            candidate,
            selected,
            selected_keys,
            selected_by_job,
            per_job_counts,
            action_counts,
            bucket_counts,
        )

    return sorted(
        selected,
        key=lambda item: (item.manifest_index, item.scene_id, item.job_id, item.source_frame_index),
    )


def _semantic_episode_representatives(
    candidates: Sequence[FrameCandidate],
    config: FrameSelectionConfig,
) -> list[FrameCandidate]:
    """Choose sparse scored centers that explicitly cover important signal episodes.

    One apex representative is retained per contiguous episode. Long episodes only
    receive approach/completion representatives when their surrounding semantic
    context actually changes, preventing long stationary intervals from becoming
    one-POI-per-frame expansions.
    """

    if str(config.semantic_episode_policy) == "none":
        return []
    guaranteed_signals = set(GUARANTEED_EPISODE_SIGNALS)
    by_path_signal: dict[tuple[int, str, str], list[FrameCandidate]] = {}
    for candidate in candidates:
        if candidate.mandatory_anchor_types:
            continue
        for signal in candidate.navigation_signals:
            if signal in guaranteed_signals:
                by_path_signal.setdefault(
                    (candidate.manifest_index, candidate.job_id, signal), []
                ).append(candidate)

    selected: dict[str, FrameCandidate] = {}
    for path_signal, members in sorted(by_path_signal.items()):
        ordered = sorted(members, key=lambda item: item.source_frame_index)
        episode: list[FrameCandidate] = []
        previous_frame: int | None = None
        for candidate in ordered:
            if previous_frame is not None and candidate.source_frame_index != previous_frame + 1:
                _add_episode_representatives(episode, selected)
                episode = []
            episode.append(candidate)
            previous_frame = candidate.source_frame_index
        _add_episode_representatives(episode, selected)
    return sorted(
        selected.values(),
        key=lambda item: (item.manifest_index, item.scene_id, item.job_id, item.source_frame_index),
    )


def _add_episode_representatives(
    episode: Sequence[FrameCandidate],
    selected: dict[str, FrameCandidate],
) -> None:
    if not episode:
        return

    def priority(candidate: FrameCandidate) -> tuple[float, int, int]:
        return (
            float(candidate.interest_score) + 0.02 * len(candidate.navigation_signals),
            -abs(candidate.source_frame_index - episode[len(episode) // 2].source_frame_index),
            -candidate.source_frame_index,
        )

    apex = max(episode, key=priority)
    selected[apex.stable_key] = apex
    if len(episode) < 64:
        return
    first_signature = _episode_context_signature(episode[0])
    apex_signature = _episode_context_signature(apex)
    last_signature = _episode_context_signature(episode[-1])
    if first_signature != apex_signature:
        selected[episode[0].stable_key] = episode[0]
    if last_signature != apex_signature:
        selected[episode[-1].stable_key] = episode[-1]


def _episode_context_signature(candidate: FrameCandidate) -> tuple[object, ...]:
    decision = candidate.navigation.get("decision")
    decision = decision if isinstance(decision, Mapping) else {}
    room = candidate.navigation.get("current_room")
    room = room if isinstance(room, Mapping) else {}
    visible = candidate.navigation.get("visible_human_ids")
    visible_ids = (
        tuple(sorted(str(item) for item in visible))
        if isinstance(visible, Sequence) and not isinstance(visible, (str, bytes))
        else ()
    )
    return (
        candidate.target_action_name,
        str(candidate.navigation.get("instruction_section_id") or ""),
        str(room.get("room_id") or ""),
        str(decision.get("primary_reason") or ""),
        visible_ids,
    )


class _CandidatePool:
    """Monotonic action/bucket queues for sublinear repeated best-candidate lookup."""

    def __init__(
        self,
        candidates: Sequence[FrameCandidate],
        config: FrameSelectionConfig,
    ) -> None:
        cells: dict[tuple[str, str], list[FrameCandidate]] = {}
        for candidate in candidates:
            if candidate.mandatory_anchor_types:
                continue
            cells.setdefault(
                (candidate.target_action_name, candidate.target_bucket), []
            ).append(candidate)
        self.cells = {
            key: sorted(
                values,
                key=lambda candidate: (
                    _static_candidate_priority(candidate, config),
                    candidate.stable_key,
                ),
                reverse=True,
            )
            for key, values in cells.items()
        }
        self.offsets = {
            True: {key: 0 for key in self.cells},
            False: {key: 0 for key in self.cells},
        }

    def best(
        self,
        *,
        selected_keys: set[str],
        selected_by_job: Mapping[str, Sequence[int]],
        per_job_counts: Mapping[str, int],
        action_counts: Mapping[str, int],
        bucket_counts: Mapping[str, int],
        bucket_targets: Mapping[str, int],
        config: FrameSelectionConfig,
        bucket: str | None,
        target_action: str | None,
        enforce_spacing: bool,
    ) -> FrameCandidate | None:
        best: tuple[float, FrameCandidate] | None = None
        offsets = self.offsets[bool(enforce_spacing)]
        for cell, values in self.cells.items():
            action_name, bucket_name = cell
            if bucket is not None and bucket_name != bucket:
                continue
            if target_action is not None and action_name != target_action:
                continue
            offset = offsets[cell]
            while offset < len(values):
                candidate = values[offset]
                job_key = _candidate_job_key(candidate)
                invalid = candidate.stable_key in selected_keys
                invalid = invalid or (
                    int(config.max_targets_per_job) > 0
                    and per_job_counts.get(job_key, 0) >= int(config.max_targets_per_job)
                )
                invalid = invalid or (
                    enforce_spacing
                    and _violates_spacing(
                        candidate,
                        selected_by_job,
                        int(config.min_target_spacing_frames),
                    )
                )
                if not invalid:
                    break
                offset += 1
            offsets[cell] = offset
            if offset >= len(values):
                continue
            candidate = values[offset]
            priority = _selection_priority(
                candidate,
                action_counts=action_counts,
                bucket_counts=bucket_counts,
                bucket_targets=bucket_targets,
                config=config,
            )
            if best is None or priority > best[0]:
                best = (priority, candidate)
        return best[1] if best is not None else None


def _best_candidate(
    candidate_pool: _CandidatePool,
    *,
    selected_keys: set[str],
    selected_by_job: Mapping[str, Sequence[int]],
    per_job_counts: Mapping[str, int],
    action_counts: Mapping[str, int],
    bucket_counts: Mapping[str, int],
    bucket_targets: Mapping[str, int],
    config: FrameSelectionConfig,
    bucket: str | None,
    target_action: str | None,
    enforce_spacing: bool,
) -> FrameCandidate | None:
    return candidate_pool.best(
        selected_keys=selected_keys,
        selected_by_job=selected_by_job,
        per_job_counts=per_job_counts,
        action_counts=action_counts,
        bucket_counts=bucket_counts,
        bucket_targets=bucket_targets,
        config=config,
        bucket=bucket,
        target_action=target_action,
        enforce_spacing=enforce_spacing,
    )


def _static_candidate_priority(
    candidate: FrameCandidate,
    config: FrameSelectionConfig,
) -> float:
    jitter_key = f"{int(config.seed)}::{candidate.stable_key}"
    jitter_seed = int.from_bytes(
        hashlib.sha256(jitter_key.encode("utf-8")).digest()[:8], "big"
    )
    jitter = float(jitter_seed) / float(1 << 64) * 1e-6
    return float(candidate.interest_score) + jitter


def _selection_priority(
    candidate: FrameCandidate,
    *,
    action_counts: Mapping[str, int],
    bucket_counts: Mapping[str, int],
    bucket_targets: Mapping[str, int],
    config: FrameSelectionConfig,
) -> float:
    selected_total = max(1, sum(action_counts.values()))
    action_ratios = _normalize_ratios(config.target_action_ratios, ACTION_NAMES)
    current_share = float(action_counts.get(candidate.target_action_name, 0)) / float(selected_total)
    target_share = float(action_ratios.get(candidate.target_action_name, 0.0))
    deficit = max(0.0, target_share - current_share)
    bucket_target = int(bucket_targets.get(candidate.target_bucket, 0))
    bucket_deficit = max(0, bucket_target - int(bucket_counts.get(candidate.target_bucket, 0)))
    bucket_deficit_share = float(bucket_deficit) / float(max(1, bucket_target))
    return (
        _static_candidate_priority(candidate, config)
        + float(config.action_deficit_weight) * deficit
        + float(config.bucket_deficit_weight) * bucket_deficit_share
    )


def _candidate_job_key(candidate: FrameCandidate) -> str:
    return f"{int(candidate.manifest_index)}::{candidate.job_id}"


def _violates_spacing(
    candidate: FrameCandidate,
    selected_by_job: Mapping[str, Sequence[int]],
    spacing: int,
) -> bool:
    if spacing <= 0:
        return False
    for selected_index in selected_by_job.get(_candidate_job_key(candidate), []):
        if abs(int(candidate.source_frame_index) - int(selected_index)) < spacing:
            return True
    return False


def _mark_selected(
    candidate: FrameCandidate,
    selected: list[FrameCandidate],
    selected_keys: set[str],
    selected_by_job: dict[str, list[int]],
    per_job_counts: dict[str, int],
    action_counts: dict[str, int],
    bucket_counts: dict[str, int],
) -> None:
    job_key = _candidate_job_key(candidate)
    selected.append(candidate)
    selected_keys.add(candidate.stable_key)
    selected_by_job.setdefault(job_key, []).append(candidate.source_frame_index)
    if not candidate.mandatory_anchor_types:
        per_job_counts[job_key] = per_job_counts.get(job_key, 0) + 1
    if not candidate.mandatory_anchor_types:
        action_counts[candidate.target_action_name] = action_counts.get(candidate.target_action_name, 0) + 1
        bucket_counts[candidate.target_bucket] = bucket_counts.get(candidate.target_bucket, 0) + 1


def _interest_selected_count(selected: Sequence[FrameCandidate]) -> int:
    return sum(1 for candidate in selected if not candidate.mandatory_anchor_types)


def _chunk_record(candidate: FrameCandidate, *, index: int, config: FrameSelectionConfig) -> JsonDict:
    window_indices = [int(item) for item in candidate.source_window_indices]
    chunk_id = (
        f"{_safe_id(candidate.job_id)}__m{int(candidate.manifest_index):03d}"
        f"__foi_{candidate.source_frame_index:06d}"
    )
    window_job_id = chunk_id
    return {
        "chunk_id": chunk_id,
        "window_job_id": window_job_id,
        "selection_index": int(index),
        "split": str(config.split),
        "source_manifest_index": int(candidate.manifest_index),
        "source_manifest_path": candidate.manifest_path,
        "source_job_id": candidate.job_id,
        "scene_id": candidate.scene_id,
        "viewpoint_robot_id": candidate.viewpoint_robot_id,
        "mission_families": list(candidate.mission_families),
        "human_actor_ids": list(candidate.human_actor_ids),
        "peer_robot_ids": list(candidate.peer_robot_ids),
        "target_frame": int(candidate.source_frame_index),
        "target_source_frame_id": int(candidate.source_frame_id),
        "target_window_index": int(config.past_frames),
        "target_time_s": float(candidate.time_s),
        "target_action_name": candidate.target_action_name,
        "target_bucket": candidate.target_bucket,
        "target_navigation": dict(candidate.navigation),
        "navigation_signals": list(candidate.navigation_signals),
        "target_role": "mandatory_anchor" if candidate.mandatory_anchor_types else "frame_of_interest",
        "mandatory_anchor_types": list(candidate.mandatory_anchor_types),
        "window_action_counts": _count_by(candidate.source_window_actions, lambda action: action),
        "bucket_scores": {key: round(float(value), 6) for key, value in candidate.bucket_scores.items()},
        "interest_score": round(float(candidate.interest_score), 6),
        "selection_reasons": list(candidate.reasons),
        "window": {
            "past_frames": int(config.past_frames),
            "future_frames": int(config.future_frames),
            "edge_policy": candidate.window_edge_policy,
            "source_frame_indices": [int(item) for item in window_indices],
            "render_frame_count": len(window_indices),
        },
        "render_contract": {
            "must_render_all_window_frames": True,
            "preserve_stationary_frames": True,
            "densify_frame_gaps": bool(config.densify_frame_gaps),
            "renderer_frame_indices": list(range(len(window_indices))),
            "target_renderer_frame_index": int(config.past_frames),
        },
    }


def _selection_summary(
    *,
    candidates: Sequence[FrameCandidate],
    selected: Sequence[FrameCandidate],
    chunks: Sequence[Mapping[str, Any]],
    target_count: int,
    bucket_targets: Mapping[str, int],
    action_targets: Mapping[str, int],
    available_navigation_signal_episode_counts: Mapping[str, int],
    selected_navigation_signal_episode_counts: Mapping[str, int],
    selected_interest_navigation_signal_episode_counts: Mapping[str, int],
    config: FrameSelectionConfig,
) -> JsonDict:
    selected_bucket_counts = _count_by(selected, lambda candidate: candidate.target_bucket)
    selected_action_counts = _count_by(selected, lambda candidate: candidate.target_action_name)
    selected_interest = [candidate for candidate in selected if not candidate.mandatory_anchor_types]
    selected_interest_bucket_counts = _count_by(
        selected_interest,
        lambda candidate: candidate.target_bucket,
    )
    selected_interest_action_counts = _count_by(
        selected_interest,
        lambda candidate: candidate.target_action_name,
    )
    mandatory = [candidate for candidate in selected if candidate.mandatory_anchor_types]
    mandatory_anchor_action_counts = _count_by(
        mandatory,
        lambda candidate: candidate.target_action_name,
    )
    scored_center_window_composition = _center_window_composition(selected_interest)
    selected_window_action_counts = _count_memberships(
        selected,
        lambda candidate: candidate.source_window_actions,
    )
    unique_source_actions: dict[tuple[int, str, int], str] = {}
    for candidate in selected:
        for frame_index, action in zip(candidate.source_window_indices, candidate.source_window_actions):
            unique_source_actions.setdefault(
                (int(candidate.manifest_index), candidate.job_id, int(frame_index)),
                str(action),
            )
    selected_unique_window_action_counts = _count_by(
        list(unique_source_actions.values()),
        lambda action: action,
    )
    mandatory_anchor_type_counts = _count_memberships(
        selected,
        lambda candidate: candidate.mandatory_anchor_types,
    )
    selected_mission_family_counts = _count_memberships(
        selected,
        lambda candidate: candidate.mission_families,
    )
    available_navigation_signal_counts = _count_memberships(
        candidates,
        lambda candidate: candidate.navigation_signals,
    )
    selected_navigation_signal_counts = _count_memberships(
        selected,
        lambda candidate: candidate.navigation_signals,
    )
    selected_interest_navigation_signal_counts = _count_memberships(
        selected_interest,
        lambda candidate: candidate.navigation_signals,
    )
    selected_targets_per_source_path = _count_by(
        selected,
        lambda candidate: _candidate_job_key(candidate),
    )
    unique_source_frames = {
        (
            str(chunk.get("source_manifest_index")),
            str(chunk.get("source_job_id")),
            int(frame_index),
        )
        for chunk in chunks
        for frame_index in chunk.get("window", {}).get("source_frame_indices", [])
    }
    requested_window_frames = sum(
        int(chunk.get("window", {}).get("render_frame_count") or 0) for chunk in chunks
    )
    return {
        "candidate_count": len(candidates),
        "selected_target_count": len(selected),
        "selected_interest_target_count": _interest_selected_count(selected),
        "requested_target_count": int(config.target_count),
        "selection_interest_target_count": int(target_count),
        "mandatory_anchor_count": sum(1 for candidate in selected if candidate.mandatory_anchor_types),
        "mandatory_anchor_type_counts": mandatory_anchor_type_counts,
        "mandatory_anchor_action_counts": mandatory_anchor_action_counts,
        "window_frame_count": int(config.window_frame_count),
        "requested_window_frame_renders": requested_window_frames,
        "unique_source_render_frame_count": len(unique_source_frames),
        "selected_bucket_counts": selected_bucket_counts,
        "selected_interest_bucket_counts": selected_interest_bucket_counts,
        "selected_action_counts": selected_action_counts,
        "selected_center_action_counts": selected_action_counts,
        "selected_interest_action_counts": selected_interest_action_counts,
        "selected_interest_action_ratios": _count_ratios(selected_interest_action_counts),
        "scored_center_window_composition": scored_center_window_composition,
        "selected_window_action_counts": selected_window_action_counts,
        "selected_unique_window_action_counts": selected_unique_window_action_counts,
        "selected_window_action_ratios": _count_ratios(selected_window_action_counts),
        "selected_unique_window_action_ratios": _count_ratios(selected_unique_window_action_counts),
        "selected_mission_family_counts": selected_mission_family_counts,
        "available_navigation_signal_counts": available_navigation_signal_counts,
        "selected_navigation_signal_counts": selected_navigation_signal_counts,
        "selected_interest_navigation_signal_counts": selected_interest_navigation_signal_counts,
        "available_navigation_signal_episode_counts": dict(
            available_navigation_signal_episode_counts
        ),
        "selected_navigation_signal_episode_counts": dict(
            selected_navigation_signal_episode_counts
        ),
        "selected_interest_navigation_signal_episode_counts": dict(
            selected_interest_navigation_signal_episode_counts
        ),
        "selected_targets_per_source_path": selected_targets_per_source_path,
        "bucket_deficits": {
            bucket: max(
                0,
                int(bucket_targets.get(bucket, 0))
                - int(selected_interest_bucket_counts.get(bucket, 0)),
            )
            for bucket in sorted(bucket_targets)
        },
        "action_deficits": {
            action: max(
                0,
                int(action_targets.get(action, 0))
                - int(selected_interest_action_counts.get(action, 0)),
            )
            for action in sorted(action_targets)
        },
        "training_center_sampling": _training_center_sampling_plan(
            selected_interest_action_counts,
            target_ratios=config.target_action_ratios,
            mandatory_anchor_count=len(mandatory),
        ),
        "render_frames_per_target": (
            round(requested_window_frames / float(len(selected)), 6) if selected else 0.0
        ),
        "unique_render_frames_per_target": (
            round(len(unique_source_frames) / float(len(selected)), 6) if selected else 0.0
        ),
    }


def _training_center_sampling_plan(
    counts: Mapping[str, int],
    *,
    target_ratios: Mapping[str, float],
    mandatory_anchor_count: int,
) -> JsonDict:
    available = [action for action in ACTION_NAMES if int(counts.get(action, 0)) > 0]
    ratios = _normalize_ratios(target_ratios, available)
    per_example_weights = {
        action: round(float(ratios[action]) / float(counts[action]), 12)
        for action in available
    }
    return {
        "population": "scored_poi_centers",
        "mandatory_anchor_policy": "retained_separate_pool",
        "mandatory_anchor_count": int(mandatory_anchor_count),
        "target_action_ratios": ratios,
        "per_example_action_weights": per_example_weights,
        "weight_contract": (
            "Sampling each scored center with its action weight yields the target action mix in expectation; "
            "mandatory anchors remain addressable through their separate target_role pool."
        ),
    }


def _center_window_composition(
    selected: Sequence[FrameCandidate],
) -> JsonDict:
    grouped: dict[str, list[FrameCandidate]] = {}
    for candidate in selected:
        grouped.setdefault(candidate.target_action_name, []).append(candidate)
    out: JsonDict = {}
    for center_action, candidates in sorted(grouped.items()):
        window_counts = _count_memberships(
            candidates,
            lambda candidate: candidate.source_window_actions,
        )
        matching_counts = [
            sum(1 for action in candidate.source_window_actions if action == center_action)
            for candidate in candidates
        ]
        out[center_action] = {
            "window_count": len(candidates),
            "window_frame_count": sum(len(candidate.source_window_actions) for candidate in candidates),
            "window_action_counts": window_counts,
            "window_action_ratios": _count_ratios(window_counts),
            "matching_action_frame_count": sum(matching_counts),
            "mean_matching_action_frames": round(
                sum(matching_counts) / float(len(matching_counts)),
                6,
            ),
            "windows_with_matching_action_at_least": {
                str(threshold): sum(1 for count in matching_counts if count >= threshold)
                for threshold in (1, 5, 10)
            },
        }
    return out


def _render_frame_index(chunks: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[str, JsonDict] = {}
    for chunk in chunks:
        key = f"{chunk.get('source_manifest_index')}::{chunk.get('source_job_id')}"
        group = grouped.setdefault(
            key,
            {
                "source_manifest_index": chunk.get("source_manifest_index"),
                "source_manifest_path": chunk.get("source_manifest_path"),
                "source_job_id": chunk.get("source_job_id"),
                "scene_id": chunk.get("scene_id"),
                "target_frames": [],
                "all_render_frames": [],
                "frame_roles": {},
            },
        )
        target = int(chunk.get("target_frame"))
        group["target_frames"].append(target)
        for frame_index in chunk.get("window", {}).get("source_frame_indices", []):
            frame_index = int(frame_index)
            roles = group["frame_roles"].setdefault(str(frame_index), [])
            if frame_index == target and "target" not in roles:
                roles.append("target")
            if "context_window" not in roles:
                roles.append("context_window")
            group["all_render_frames"].append(frame_index)
    for group in grouped.values():
        group["target_frames"] = sorted(set(int(item) for item in group["target_frames"]))
        group["all_render_frames"] = sorted(set(int(item) for item in group["all_render_frames"]))
    return dict(sorted(grouped.items()))


def _source_manifest_records(
    manifests: Sequence[Mapping[str, Any]],
    paths: Sequence[str | None],
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for index, manifest in enumerate(manifests):
        path = paths[index] if index < len(paths) else None
        records.append(
            {
                "source_manifest_index": index,
                "source_manifest_path": path,
                "schema_version": manifest.get("schema_version"),
                "scenario_id": manifest.get("source", {}).get("scenario_id"),
                "scene_id": manifest.get("source", {}).get("scene_id"),
                "mission_families": list(manifest.get("mission_families", [])),
                "job_count": len([job for job in manifest.get("jobs", []) if isinstance(job, Mapping)]),
                "fingerprint": _manifest_fingerprint(manifest),
            }
        )
    return records


def _chunks_for_manifest(
    selection: Mapping[str, Any],
    *,
    manifest_path: str | Path | None,
) -> list[Mapping[str, Any]]:
    chunks = [chunk for chunk in selection.get("chunks", []) if isinstance(chunk, Mapping)]
    if manifest_path is None:
        indexes = {
            int(chunk.get("source_manifest_index", 0) or 0)
            for chunk in chunks
        }
        if not indexes or indexes == {0}:
            return chunks
        return [chunk for chunk in chunks if int(chunk.get("source_manifest_index", 0) or 0) == 0]
    path_text = str(manifest_path)
    exact = [chunk for chunk in chunks if str(chunk.get("source_manifest_path") or "") == path_text]
    if exact:
        return exact
    return [
        chunk
        for chunk in chunks
        if Path(str(chunk.get("source_manifest_path") or "")).name == Path(path_text).name
    ]


def _windowed_job(source_job: Mapping[str, Any], chunk: Mapping[str, Any]) -> JsonDict:
    job = deepcopy(dict(source_job))
    render_contract = chunk.get("render_contract", {})
    densify_frame_gaps = bool(
        render_contract.get("densify_frame_gaps", False)
        if isinstance(render_contract, Mapping)
        else False
    )
    source_trajectory = _job_trajectory(
        source_job,
        densify_frame_gaps=densify_frame_gaps,
    )
    selected_points: list[JsonDict] = []
    source_indices = [int(item) for item in chunk.get("window", {}).get("source_frame_indices", [])]
    for window_frame, source_index in enumerate(source_indices):
        if not source_trajectory:
            continue
        clamped = min(max(0, source_index), len(source_trajectory) - 1)
        point = deepcopy(dict(source_trajectory[clamped]))
        metadata = dict(point.get("metadata", {})) if isinstance(point.get("metadata"), Mapping) else {}
        metadata.update(
            {
                "source_frame_index": int(source_index),
                "source_frame_id": int(point.get("frame", source_index) or source_index),
                "window_frame_index": int(window_frame),
                "frame_selection_chunk_id": chunk.get("chunk_id"),
                "frame_selection_target_frame": chunk.get("target_frame"),
            }
        )
        point["sample_index"] = int(window_frame)
        point["frame"] = int(window_frame)
        point["metadata"] = metadata
        selected_points.append(point)
    camera = deepcopy(dict(job.get("camera", {}))) if isinstance(job.get("camera"), Mapping) else {}
    camera["trajectory"] = selected_points
    camera["preserve_frame_samples"] = True
    camera["metadata"] = {
        **(dict(camera.get("metadata", {})) if isinstance(camera.get("metadata"), Mapping) else {}),
        "frame_selection_chunk_id": chunk.get("chunk_id"),
        "source_job_id": chunk.get("source_job_id"),
        "source_frame_indices": source_indices,
        "target_frame": chunk.get("target_frame"),
        "target_source_frame_id": chunk.get("target_source_frame_id"),
        "target_window_index": chunk.get("target_window_index"),
        "target_time_s": chunk.get("target_time_s"),
        "target_action_name": chunk.get("target_action_name"),
        "target_bucket": chunk.get("target_bucket"),
        "target_navigation": dict(chunk.get("target_navigation", {})),
        "navigation_signals": list(chunk.get("navigation_signals", [])),
        "target_role": chunk.get("target_role"),
        "mandatory_anchor_types": list(chunk.get("mandatory_anchor_types", [])),
        "preserve_frame_samples": True,
    }
    job["camera"] = camera
    job["frame_catalog"] = {
        "selection_policy": "producer_selection_manifest",
        "trajectory_scope": "selected_past_context_window",
        "complete_source_trajectory": False,
        "complete_source_trajectory_path": "actors.robots[*].trajectory",
        "source_job_id": chunk.get("source_job_id"),
        "sample_count": len(selected_points),
        "sample_index_range": [0, len(selected_points) - 1] if selected_points else None,
        "source_frame_indices": source_indices,
        "target_frame": chunk.get("target_frame"),
        "target_window_index": chunk.get("target_window_index"),
        "trajectory_path": "camera.trajectory",
        "point_supervision_path": "camera.trajectory[*].metadata.navigation",
        "preserve_stationary_samples": True,
    }
    job["job_id"] = str(chunk.get("window_job_id") or chunk.get("chunk_id") or job.get("job_id"))
    outputs = deepcopy(dict(job.get("outputs", {}))) if isinstance(job.get("outputs"), Mapping) else {}
    outputs.update(
        {
            "stem": job["job_id"],
            "video_name": f"{job['job_id']}.mp4",
            "camera_metadata_name": f"{job['job_id']}_camera.json",
            "actor_debug_name": f"{job['job_id']}_actors.json",
        }
    )
    job["outputs"] = outputs
    job["frame_selection"] = {
        "chunk_id": chunk.get("chunk_id"),
        "source_job_id": chunk.get("source_job_id"),
        "target_frame": chunk.get("target_frame"),
        "target_source_frame_id": chunk.get("target_source_frame_id"),
        "target_window_index": chunk.get("target_window_index"),
        "target_time_s": chunk.get("target_time_s"),
        "target_action_name": chunk.get("target_action_name"),
        "target_bucket": chunk.get("target_bucket"),
        "target_navigation": dict(chunk.get("target_navigation", {})),
        "navigation_signals": list(chunk.get("navigation_signals", [])),
        "target_role": chunk.get("target_role"),
        "mandatory_anchor_types": list(chunk.get("mandatory_anchor_types", [])),
        "window_action_counts": dict(chunk.get("window_action_counts", {})),
        "source_frame_indices": source_indices,
        "window_frame_count": len(selected_points),
        "preserve_stationary_frames": True,
    }
    return job


def _job_trajectory(
    job: Mapping[str, Any],
    *,
    densify_frame_gaps: bool = False,
) -> list[Mapping[str, Any]]:
    camera = job.get("camera", {})
    if not isinstance(camera, Mapping):
        return []
    trajectory = camera.get("trajectory", [])
    if not isinstance(trajectory, list):
        return []
    points = [point for point in trajectory if isinstance(point, Mapping)]
    return _densify_trajectory_by_frame(points) if densify_frame_gaps else points


def _densify_trajectory_by_frame(
    trajectory: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(trajectory) < 2:
        return list(trajectory)

    by_frame: dict[int, Mapping[str, Any]] = {}
    for index, point in enumerate(trajectory):
        by_frame[_int_or(point.get("frame"), index)] = point
    framed = sorted(by_frame.items())
    if len(framed) < 2 or all(right[0] - left[0] == 1 for left, right in zip(framed, framed[1:])):
        return [point for _frame, point in framed]

    dense: list[JsonDict] = []
    for segment_index, ((left_frame, left), (right_frame, right)) in enumerate(
        zip(framed, framed[1:])
    ):
        if segment_index == 0:
            dense.append(_dense_endpoint(left, dense_index=0, source_frame=left_frame))
        gap = int(right_frame) - int(left_frame)
        if gap <= 0:
            continue
        left_position = _point_position(left)
        right_position = _point_position(right)
        left_time = _point_time(left, left_frame)
        right_time = _point_time(right, right_frame)
        left_yaw = _point_yaw(left, 0, (left, right))
        right_yaw = _point_yaw(right, 1, (left, right))
        yaw_delta = _wrap_angle(right_yaw - left_yaw)
        for step in range(1, gap + 1):
            alpha = float(step) / float(gap)
            if step == gap:
                dense.append(
                    _dense_endpoint(
                        right,
                        dense_index=len(dense),
                        source_frame=right_frame,
                    )
                )
                continue
            nearest = left if alpha < 0.5 else right
            point = deepcopy(dict(nearest))
            metadata = dict(point.get("metadata", {})) if isinstance(point.get("metadata"), Mapping) else {}
            metadata.update(
                {
                    "frame_selection_interpolated": True,
                    "interpolation_source_frames": [int(left_frame), int(right_frame)],
                    "interpolation_alpha": round(alpha, 8),
                    "original_sample_index": point.get("sample_index"),
                }
            )
            point.update(
                {
                    "sample_index": len(dense),
                    "frame": int(left_frame + step),
                    "t": float(left_time + (right_time - left_time) * alpha),
                    "time_s": float(left_time + (right_time - left_time) * alpha),
                    "position": [
                        float(left_position[axis] + (right_position[axis] - left_position[axis]) * alpha)
                        for axis in range(3)
                    ],
                    "yaw_rad": _wrap_angle(left_yaw + yaw_delta * alpha),
                    "metadata": metadata,
                }
            )
            dense.append(point)
    return dense


def _dense_endpoint(
    source: Mapping[str, Any],
    *,
    dense_index: int,
    source_frame: int,
) -> JsonDict:
    point = deepcopy(dict(source))
    metadata = dict(point.get("metadata", {})) if isinstance(point.get("metadata"), Mapping) else {}
    metadata.setdefault("original_sample_index", point.get("sample_index"))
    point["sample_index"] = int(dense_index)
    point["frame"] = int(source_frame)
    point["metadata"] = metadata
    return point


def _window_indices(
    target_index: int,
    trajectory_len: int | None,
    config: FrameSelectionConfig,
    *,
    policy: str | None = None,
) -> list[int]:
    edge_policy = str(policy or config.edge_window_policy)
    start = int(target_index) - int(config.past_frames)
    end = int(target_index) + int(config.future_frames)
    if trajectory_len is not None and edge_policy == "reject":
        if start < 0 or end >= int(trajectory_len):
            return []
    if trajectory_len is None or edge_policy == "reject":
        return list(range(start, end + 1))
    return [min(max(0, index), int(trajectory_len) - 1) for index in range(start, end + 1)]


def _mandatory_anchor_indices(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> dict[int, tuple[str, ...]]:
    if not enabled or not trajectory:
        return {}

    anchors: dict[int, list[str]] = {}

    def add(time_s: float, anchor_type: str) -> None:
        index = min(
            range(len(trajectory)),
            key=lambda item: abs(_point_time(trajectory[item], item) - float(time_s)),
        )
        labels = anchors.setdefault(index, [])
        if anchor_type not in labels:
            labels.append(anchor_type)

    def add_index(index: int, anchor_type: str) -> None:
        labels = anchors.setdefault(int(index), [])
        if anchor_type not in labels:
            labels.append(anchor_type)

    anchors[len(trajectory) - 1] = ["trajectory_end"]
    stop_episode_end: int | None = None
    for index in range(len(trajectory)):
        if _action_name(trajectory, index) == "stop":
            stop_episode_end = index
            continue
        if stop_episode_end is not None:
            add_index(stop_episode_end, "stopping_point")
            stop_episode_end = None
    if stop_episode_end is not None:
        add_index(stop_episode_end, "stopping_point")
    assigned_mission_ids = {
        str(item) for item in job.get("assigned_mission_ids", []) if str(item)
    }
    viewpoint_robot_id = str(
        job.get("viewpoint_robot_id")
        or (job.get("camera", {}).get("source_actor_id") if isinstance(job.get("camera"), Mapping) else "")
        or ""
    )
    completed_missions: set[str] = set()
    for event in manifest.get("events", []):
        if not isinstance(event, Mapping) or event.get("t") is None:
            continue
        event_type = str(event.get("event_type") or "").lower().replace("-", "_")
        if not any(
            token in event_type
            for token in (
                "completion",
                "complete",
                "robot_eos",
                "mission_end",
                "section_end",
                "checkpoint",
                "route_end",
                "sub_mission_end",
            )
        ):
            continue
        mission_id = str(event.get("mission_id") or "")
        if assigned_mission_ids and mission_id and mission_id not in assigned_mission_ids:
            continue
        event_actor_id = str(event.get("actor_id") or event.get("robot_id") or "")
        if event_actor_id and viewpoint_robot_id and event_actor_id != viewpoint_robot_id:
            continue
        label = f"mission_section_end:{mission_id}" if mission_id else f"event:{event_type}"
        add(float(event.get("t") or 0.0), label)
        if mission_id:
            completed_missions.add(mission_id)

    for mission in manifest.get("missions", []):
        if not isinstance(mission, Mapping):
            continue
        mission_id = str(mission.get("mission_id") or "")
        if assigned_mission_ids and mission_id not in assigned_mission_ids:
            continue
        if mission_id in completed_missions:
            continue
        metadata = mission.get("metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        completion_time = next(
            (
                float(metadata[key])
                for key in ("robot_eos_t", "expected_completion_t", "robot_arrival_t")
                if metadata.get(key) is not None
            ),
            None,
        )
        if completion_time is not None:
            label = f"mission_section_end:{mission_id}" if mission_id else "mission_section_end"
            add(completion_time, label)

    return {index: tuple(labels) for index, labels in sorted(anchors.items())}


def _bucket_scores(
    trajectory: Sequence[Mapping[str, Any]],
    frame_index: int,
    *,
    event_times: Sequence[float],
    target_action_name: str,
    navigation_signals: Sequence[str],
    human_min: float | None,
    peer_min: float | None,
) -> JsonDict:
    point = trajectory[frame_index]
    time_s = _point_time(point, frame_index)
    position = _point_position(point)
    navigation = _point_navigation(point)
    decision = navigation.get("decision") if isinstance(navigation.get("decision"), Mapping) else {}
    decision_reason = str(decision.get("primary_reason") or "")
    section_type = str(navigation.get("instruction_section_type") or "")
    navigation_signal_set = set(navigation_signals)
    turn_delta = abs(_turn_delta(trajectory, frame_index))
    event_score = _event_proximity_score(time_s, event_times)
    stop_transition_score = _stop_transition_score(trajectory, frame_index)

    human_interaction = _distance_score(human_min, close=0.7, far=5.0)
    route_decision = max(
        _clamp01(turn_delta / 0.9),
        stop_transition_score,
        event_score * 0.85,
    )
    closest = min(value for value in (human_min, peer_min) if value is not None) if (human_min is not None or peer_min is not None) else None
    critical_margin = max(
        _distance_score(closest, close=0.55, far=2.5),
        0.35 if target_action_name == "stop" and human_min is not None and human_min < 3.0 else 0.0,
        event_score,
    )
    if decision_reason in {"HUMAN_COLLISION_AVOIDANCE", "ROBOT_COLLISION_AVOIDANCE"}:
        critical_margin = max(critical_margin, 0.98)
    if decision_reason in {
        "L2_PEDESTRIAN_YIELD",
        "HUMAN_GUIDANCE_APPROACH",
        "HUMAN_GUIDANCE_CONVERSATION",
        "TARGET_HUMAN_APPROACH",
    }:
        human_interaction = max(human_interaction, 0.90)
    elif decision_reason.startswith(("L1_", "L3_", "L4_")):
        human_interaction = max(human_interaction, 0.65)
    if "relevant_human_visible" in navigation_signal_set:
        human_interaction = max(human_interaction, 0.75)
    if "target_human_visible" in navigation_signal_set:
        human_interaction = max(human_interaction, 0.82)
    if "social_law_decision" in navigation_signal_set:
        human_interaction = max(human_interaction, 0.72)
    if "human_interaction_decision" in navigation_signal_set:
        human_interaction = max(human_interaction, 0.90)
    if "traffic_wait" in navigation_signal_set:
        human_interaction = max(human_interaction, 0.78)
    if "route_deviation" in navigation_signal_set:
        critical_margin = max(critical_margin, 0.90)
    if "planned_execution_mismatch" in navigation_signal_set:
        critical_margin = max(critical_margin, 0.85)
    if section_type == "turn" or "instruction_turn" in navigation_signal_set:
        route_decision = max(route_decision, 0.90)
    if "semantic_stop" in navigation_signal_set:
        route_decision = max(route_decision, 0.88)
    if "instruction_section_boundary" in navigation_signal_set:
        route_decision = max(route_decision, 0.82)
    if "room_transition" in navigation_signal_set:
        route_decision = max(route_decision, 0.95)
    if "mission_endpoint" in navigation_signal_set:
        route_decision = max(route_decision, 0.92)
    representative = 0.25
    if target_action_name == "move":
        representative += 0.20
    if human_min is None or human_min > 4.0:
        representative += 0.15
    if turn_delta < 0.15 and stop_transition_score < 0.2 and event_score < 0.2:
        representative += 0.15
    if max(critical_margin, human_interaction, route_decision) < 0.35:
        representative = max(representative, 0.55)

    return {
        "critical_margin": round(_clamp01(critical_margin), 6),
        "human_interaction": round(_clamp01(human_interaction), 6),
        "route_decision": round(_clamp01(route_decision), 6),
        "representative_motion": round(_clamp01(representative), 6),
    }


def _interest_score(bucket_scores: Mapping[str, float]) -> float:
    return max(
        float(bucket_scores.get("critical_margin", 0.0)) * 1.10,
        float(bucket_scores.get("human_interaction", 0.0)) * 1.00,
        float(bucket_scores.get("route_decision", 0.0)) * 0.95,
        float(bucket_scores.get("representative_motion", 0.0)) * 0.70,
    )


def _candidate_reasons(
    target_bucket: str,
    bucket_scores: Mapping[str, float],
    action_name: str,
    navigation_signals: Sequence[str],
) -> tuple[str, ...]:
    reasons = [target_bucket]
    if float(bucket_scores.get("critical_margin", 0.0)) >= 0.55:
        reasons.append("low_margin_or_safety_event")
    if float(bucket_scores.get("human_interaction", 0.0)) >= 0.55:
        reasons.append("near_human_or_target_actor")
    if float(bucket_scores.get("route_decision", 0.0)) >= 0.55:
        reasons.append("turn_stop_transition_or_event")
    reasons.extend(f"navigation:{signal}" for signal in navigation_signals)
    reasons.append(f"action:{action_name}")
    return tuple(dict.fromkeys(reasons))


def _navigation_signals(
    trajectory: Sequence[Mapping[str, Any]],
    frame_index: int,
) -> tuple[str, ...]:
    navigation = _point_navigation(trajectory[frame_index])
    previous_navigation = (
        _point_navigation(trajectory[frame_index - 1]) if frame_index > 0 else {}
    )
    signals: list[str] = []
    section_type = str(navigation.get("instruction_section_type") or "")
    if section_type == "turn":
        signals.append("instruction_turn")
    elif section_type == "stop":
        signals.append("semantic_stop")

    section_id = str(navigation.get("instruction_section_id") or "")
    previous_section_id = str(previous_navigation.get("instruction_section_id") or "")
    if frame_index > 0 and section_id and section_id != previous_section_id:
        signals.append("instruction_section_boundary")

    room = navigation.get("current_room")
    room = room if isinstance(room, Mapping) else {}
    previous_room = previous_navigation.get("current_room")
    previous_room = previous_room if isinstance(previous_room, Mapping) else {}
    room_id = str(room.get("room_id") or "")
    previous_room_id = str(previous_room.get("room_id") or "")
    if frame_index > 0 and room_id != previous_room_id and (room_id or previous_room_id):
        signals.append("room_transition")

    decision = navigation.get("decision")
    decision = decision if isinstance(decision, Mapping) else {}
    decision_reason = str(decision.get("primary_reason") or "")
    secondary_reasons = decision.get("secondary_reasons")
    decision_reasons = [decision_reason]
    if isinstance(secondary_reasons, Sequence) and not isinstance(secondary_reasons, (str, bytes)):
        decision_reasons.extend(str(reason) for reason in secondary_reasons)
    if decision_reason in {"HUMAN_COLLISION_AVOIDANCE", "ROBOT_COLLISION_AVOIDANCE"}:
        signals.append("collision_avoidance")
    if any(reason.startswith(("L1_", "L2_", "L3_", "L4_")) for reason in decision_reasons):
        signals.append("social_law_decision")
    if decision_reason in {
        "HUMAN_GUIDANCE_APPROACH",
        "HUMAN_GUIDANCE_CONVERSATION",
        "TARGET_HUMAN_APPROACH",
    }:
        signals.append("human_interaction_decision")
    if decision_reason == "TRAFFIC_RESERVATION_WAIT":
        signals.append("traffic_wait")
    if decision_reason == "MISSION_ENDPOINT_STOP":
        signals.append("mission_endpoint")

    visible_human_ids = navigation.get("visible_human_ids")
    if (
        isinstance(visible_human_ids, Sequence)
        and not isinstance(visible_human_ids, (str, bytes))
        and any(str(human_id) for human_id in visible_human_ids)
    ):
        signals.append("relevant_human_visible")

    target_human = navigation.get("target_human")
    target_human = target_human if isinstance(target_human, Mapping) else {}
    visibility = str(target_human.get("visibility") or "")
    if visibility in {"geometry_estimated_visible", "render_verified_visible", "seen"}:
        signals.append("target_human_visible")

    try:
        deviation_m = float(navigation.get("deviation_from_planned_route_m") or 0.0)
    except (TypeError, ValueError):
        deviation_m = 0.0
    if deviation_m >= 0.25:
        signals.append("route_deviation")
    planned_action = str(navigation.get("planned_action") or "")
    executed_action = str(navigation.get("executed_action") or "")
    if planned_action and executed_action and planned_action != executed_action:
        signals.append("planned_execution_mismatch")
    return tuple(dict.fromkeys(signals))


def _action_name(trajectory: Sequence[Mapping[str, Any]], frame_index: int) -> str:
    point = trajectory[frame_index]
    motion_state = str(point.get("motion_state") or "").lower()
    if any(token in motion_state for token in ("idle", "stop", "wait", "yield", "done")):
        return "stop"
    previous_pos = _point_position(trajectory[max(0, frame_index - 1)])
    next_pos = _point_position(trajectory[min(len(trajectory) - 1, frame_index + 1)])
    speed_proxy = math.dist(previous_pos[:2], next_pos[:2])
    if speed_proxy < 1e-3:
        return "stop"
    delta = _turn_delta(trajectory, frame_index)
    if abs(delta) >= 0.30:
        return "turn_left" if delta > 0.0 else "turn_right"
    return "move"


def _turn_delta(trajectory: Sequence[Mapping[str, Any]], frame_index: int) -> float:
    left = trajectory[max(0, frame_index - 1)]
    right = trajectory[min(len(trajectory) - 1, frame_index + 1)]
    current = trajectory[frame_index]
    yaw_left = _point_yaw(left, max(0, frame_index - 1), trajectory)
    yaw_right = _point_yaw(right, min(len(trajectory) - 1, frame_index + 1), trajectory)
    if frame_index > 0 and frame_index + 1 < len(trajectory):
        return _wrap_angle(yaw_right - yaw_left)
    return _wrap_angle(_point_yaw(current, frame_index, trajectory) - yaw_left)


def _point_yaw(point: Mapping[str, Any], index: int, trajectory: Sequence[Mapping[str, Any]]) -> float:
    if point.get("yaw_rad") is not None:
        return float(point.get("yaw_rad") or 0.0)
    if index + 1 < len(trajectory):
        start = _point_position(point)
        end = _point_position(trajectory[index + 1])
    elif index > 0:
        start = _point_position(trajectory[index - 1])
        end = _point_position(point)
    else:
        return 0.0
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if math.hypot(dx, dy) < 1e-5:
        return 0.0
    return math.atan2(dy, dx)


def _stop_transition_score(trajectory: Sequence[Mapping[str, Any]], frame_index: int) -> float:
    center_action = _coarse_motion(trajectory[frame_index])
    lo = max(0, frame_index - 3)
    hi = min(len(trajectory), frame_index + 4)
    if any(_coarse_motion(trajectory[index]) != center_action for index in range(lo, hi)):
        return 0.75
    return 0.0


def _coarse_motion(point: Mapping[str, Any]) -> str:
    motion_state = str(point.get("motion_state") or "").lower()
    if any(token in motion_state for token in ("idle", "stop", "wait", "yield", "done")):
        return "stop"
    return "move"


def _event_times(manifest: Mapping[str, Any]) -> list[float]:
    times: list[float] = []
    for event in manifest.get("events", []):
        if isinstance(event, Mapping) and event.get("t") is not None:
            times.append(float(event.get("t") or 0.0))
    for mission in manifest.get("missions", []):
        if not isinstance(mission, Mapping):
            continue
        for key in ("release_time_s", "deadline_s"):
            if mission.get(key) is not None:
                times.append(float(mission.get(key) or 0.0))
    return sorted(times)


def _event_proximity_score(time_s: float, event_times: Sequence[float]) -> float:
    if not event_times:
        return 0.0
    nearest = min(abs(float(time_s) - float(event_time)) for event_time in event_times)
    if nearest >= 2.0:
        return 0.0
    return _clamp01((2.0 - nearest) / 2.0)


def _min_track_distance(
    position: tuple[float, float, float],
    time_s: float,
    tracks: Sequence[Sequence[tuple[float, tuple[float, float, float]]]],
) -> float | None:
    distances: list[float] = []
    for track in tracks:
        actor_position = _position_from_samples(track, time_s)
        distances.append(math.dist(position[:2], actor_position[:2]))
    return min(distances) if distances else None


def _position_at_time(
    trajectory: Any,
    time_s: float,
    *,
    fallback: Any = None,
) -> tuple[float, float, float] | None:
    samples = _position_samples(trajectory, fallback=fallback)
    if not samples:
        return None
    return _position_from_samples(samples, time_s)


def _position_samples(
    trajectory: Any,
    *,
    fallback: Any = None,
) -> list[tuple[float, tuple[float, float, float]]]:
    points = (
        [point for point in trajectory if isinstance(point, Mapping)]
        if isinstance(trajectory, list)
        else []
    )
    if not points:
        if isinstance(fallback, Mapping):
            return [
                (
                    0.0,
                    (
                        float(fallback.get("x", 0.0) or 0.0),
                        float(fallback.get("y", 0.0) or 0.0),
                        float(fallback.get("z", 0.0) or 0.0),
                    ),
                )
            ]
        return []
    return sorted(
        (_point_time(point, index), _point_position(point))
        for index, point in enumerate(points)
    )


def _position_from_samples(
    samples: Sequence[tuple[float, tuple[float, float, float]]],
    time_s: float,
) -> tuple[float, float, float]:
    if float(time_s) <= samples[0][0]:
        return samples[0][1]
    if float(time_s) >= samples[-1][0]:
        return samples[-1][1]
    right = bisect_left(samples, float(time_s), key=lambda item: item[0])
    left = max(0, right - 1)
    t0, p0 = samples[left]
    t1, p1 = samples[right]
    alpha = (float(time_s) - t0) / max(t1 - t0, 1e-6)
    return tuple(
        float(p0[axis] + (p1[axis] - p0[axis]) * alpha)
        for axis in range(3)
    )  # type: ignore[return-value]


def _distance_score(value: float | None, *, close: float, far: float) -> float:
    if value is None:
        return 0.0
    if value <= close:
        return 1.0
    if value >= far:
        return 0.0
    return _clamp01((far - value) / max(far - close, 1e-6))


def _point_position(point: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = point.get("position")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
        return (
            float(raw[0]),
            float(raw[1]),
            float(raw[2]) if len(raw) > 2 else 0.0,
        )
    pose = point.get("map_pose") if isinstance(point.get("map_pose"), Mapping) else {}
    return (
        float(pose.get("x", 0.0) or 0.0),
        float(pose.get("y", 0.0) or 0.0),
        0.0,
    )


def _point_navigation(point: Mapping[str, Any]) -> JsonDict:
    direct = point.get("navigation")
    if isinstance(direct, Mapping):
        return dict(direct)
    metadata = point.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("navigation"), Mapping):
        return dict(metadata["navigation"])
    return {}


def _point_time(point: Mapping[str, Any], index: int) -> float:
    value = point.get("time_s", point.get("t"))
    return float(value) if value is not None else float(index)


def _count_by(items: Sequence[Any], key_fn) -> JsonDict:
    counts: dict[str, int] = {}
    for item in items:
        key = str(key_fn(item))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_memberships(items: Sequence[Any], key_fn) -> JsonDict:
    counts: dict[str, int] = {}
    for item in items:
        for raw_key in key_fn(item):
            key = str(raw_key)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _navigation_signal_episode_coverage(
    candidates: Sequence[FrameCandidate],
    selected: Sequence[FrameCandidate],
) -> tuple[JsonDict, JsonDict, JsonDict]:
    selected_keys = {candidate.stable_key for candidate in selected}
    selected_interest_keys = {
        candidate.stable_key
        for candidate in selected
        if not candidate.mandatory_anchor_types
    }
    memberships = sorted(
        (
            candidate.manifest_index,
            candidate.job_id,
            signal,
            candidate.source_frame_index,
            candidate.stable_key,
        )
        for candidate in candidates
        for signal in candidate.navigation_signals
    )
    available: Counter[str] = Counter()
    selected_covered: Counter[str] = Counter()
    selected_interest_covered: Counter[str] = Counter()
    episode_key: tuple[int, str, str] | None = None
    previous_frame: int | None = None
    episode_selected = False
    episode_interest_selected = False

    def finish_episode() -> None:
        if episode_key is None:
            return
        signal = episode_key[2]
        available[signal] += 1
        if episode_selected:
            selected_covered[signal] += 1
        if episode_interest_selected:
            selected_interest_covered[signal] += 1

    for manifest_index, job_id, signal, frame_index, stable_key in memberships:
        key = (manifest_index, job_id, signal)
        if key != episode_key or previous_frame is None or frame_index != previous_frame + 1:
            finish_episode()
            episode_key = key
            episode_selected = False
            episode_interest_selected = False
        episode_selected = episode_selected or stable_key in selected_keys
        episode_interest_selected = (
            episode_interest_selected or stable_key in selected_interest_keys
        )
        previous_frame = frame_index
    finish_episode()
    return (
        dict(sorted(available.items())),
        dict(sorted(selected_covered.items())),
        dict(sorted(selected_interest_covered.items())),
    )


def _count_ratios(counts: Mapping[str, int]) -> JsonDict:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return {}
    return {
        str(key): round(int(value) / float(total), 6)
        for key, value in sorted(counts.items())
    }


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stable_key(
    manifest_path: str | None,
    manifest_index: int,
    job_id: str,
    source_frame_index: int,
    source_frame_id: int,
) -> str:
    text = "::".join(
        [
            str(manifest_path or manifest_index),
            str(job_id),
            str(source_frame_index),
            str(source_frame_id),
        ]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "job"


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
