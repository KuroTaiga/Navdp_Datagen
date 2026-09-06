from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JsonDict = dict[str, Any]

MANIFEST_SCHEMA_VERSION = "massgen_render_manifest/v0.1"
DEFAULT_RENDER_BACKEND = "gsplat"
DEFAULT_FPS = 10.0
DEFAULT_HUMAN_RADIUS_M = 0.4
DEFAULT_HUMAN_HEIGHT_M = 1.8
DEFAULT_HUMAN_CULL_MARGIN_M = 0.25
DEFAULT_ROBOT_RADIUS_M = 0.3
DEFAULT_ROBOT_HEIGHT_M = 1.3
DEFAULT_ROBOT_CULL_MARGIN_M = 0.25
DEFAULT_ROBOT_GLB = "assets/robots/g1_29dof_mode_16.glb"

ACTIVE_MASS_MISSION_FAMILIES = (
    "human_guided_uncertain_region",
    "serve_queue",
    "dense_dynamic_humans",
    "dense_dynamic_combined",
    "dense_dynamic_avoidance",
    "dense_multi_robot",
    "mission_stream",
    "interruption_recovery",
    "multi_robot_handoff",
    "escort_and_rendezvous",
    "implicit_need_fulfillment",
    "conflict_resolution",
    "deliver_to_human",
    "navigate_with_social_constraints",
)
SCHEMA_ONLY_MISSION_FAMILIES = (
    "human_guided_person_disambiguation",
    "human_guided_route_correction",
)
MULTI_ROBOT_MISSION_FAMILIES = (
    "dense_multi_robot",
    "dense_dynamic_combined",
    "mission_stream",
    "multi_robot_handoff",
)

def load_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_action_catalog(path: str | Path | None) -> dict[str, JsonDict]:
    if path is None:
        return {}
    payload = load_json(path)
    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError(f"{path} must contain an 'actions' list")
    catalog: dict[str, JsonDict] = {}
    for index, raw in enumerate(actions):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: action #{index} must be an object")
        action_id = _str_or_none(raw.get("action_id") or raw.get("id") or raw.get("sequence_id"))
        if action_id is None:
            raise ValueError(f"{path}: action #{index} is missing action_id")
        catalog[action_id] = dict(raw)
    return catalog


def scenario_file_to_render_manifest(
    scenario_path: str | Path,
    *,
    action_catalog_path: str | Path | None = None,
    fps: float = DEFAULT_FPS,
    render_backend: str = DEFAULT_RENDER_BACKEND,
    default_robot_glb: str = DEFAULT_ROBOT_GLB,
    visibility_culling: bool = True,
    human_cull_margin_m: float = DEFAULT_HUMAN_CULL_MARGIN_M,
    robot_cull_margin_m: float = DEFAULT_ROBOT_CULL_MARGIN_M,
) -> JsonDict:
    scenario_path = Path(scenario_path)
    return scenario_to_render_manifest(
        load_json(scenario_path),
        source_path=scenario_path,
        action_catalog=load_action_catalog(action_catalog_path),
        fps=fps,
        render_backend=render_backend,
        default_robot_glb=default_robot_glb,
        visibility_culling=visibility_culling,
        human_cull_margin_m=human_cull_margin_m,
        robot_cull_margin_m=robot_cull_margin_m,
    )


def scenario_to_render_manifest(
    scenario: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    action_catalog: Mapping[str, Mapping[str, Any]] | None = None,
    fps: float = DEFAULT_FPS,
    render_backend: str = DEFAULT_RENDER_BACKEND,
    default_robot_glb: str = DEFAULT_ROBOT_GLB,
    visibility_culling: bool = True,
    human_cull_margin_m: float = DEFAULT_HUMAN_CULL_MARGIN_M,
    robot_cull_margin_m: float = DEFAULT_ROBOT_CULL_MARGIN_M,
) -> JsonDict:
    """Convert a Pathplanner scenario JSON into a renderer-owned manifest.

    The manifest is intentionally declarative: GPU workers can render each job
    without importing Pathplanner or reinterpreting mission-family internals.
    """

    scenario_id = _required_str(scenario, "scenario_id", "scenario")
    scene_id = _required_str(scenario, "scene_id", "scenario")
    fps = _positive_float(fps, "fps")
    source_path_str = str(source_path) if source_path is not None else None

    missions = [_mission_record(item, index) for index, item in enumerate(_list(scenario.get("missions")))]
    mission_families = _stable_unique(str(mission["mission_type"]) for mission in missions)
    social_law_ids = _social_law_ids(scenario, missions)
    scenario_end_s = _scenario_end_time_s(scenario, missions)
    frame_count = int(math.floor(scenario_end_s * fps + 1e-6)) + 1
    action_catalog = dict(action_catalog or {})

    humans = [
        _human_actor_record(
            item,
            index,
            scenario_end_s=scenario_end_s,
            fps=fps,
            action_catalog=action_catalog,
        )
        for index, item in enumerate(_list(scenario.get("humans")))
    ]
    _attach_human_mission_bindings(humans, missions, scenario)
    robots = [
        _robot_actor_record(
            item,
            index,
            fps=fps,
            default_robot_glb=default_robot_glb,
        )
        for index, item in enumerate(_list(scenario.get("robots")))
    ]
    if not robots:
        raise ValueError(f"{scenario_id}: at least one robot is required to build render jobs")

    robot_ids = [str(robot["actor_id"]) for robot in robots]
    training_robot_ids = _training_robot_ids(scenario, missions, robot_ids, mission_families)
    jobs = [
        _render_job(
            scenario_id=scenario_id,
            scene_id=scene_id,
            viewpoint_robot_id=robot_id,
            robots=robots,
            humans=humans,
            missions=missions,
            mission_families=mission_families,
            render_backend=render_backend,
            visibility_culling=visibility_culling,
            human_cull_margin_m=human_cull_margin_m,
            robot_cull_margin_m=robot_cull_margin_m,
        )
        for robot_id in training_robot_ids
    ]

    warnings = _manifest_warnings(
        scenario_id=scenario_id,
        mission_families=mission_families,
        robots=robots,
        humans=humans,
        action_catalog=action_catalog,
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "kind": "pathplanner_scenario",
            "scenario_path": source_path_str,
            "scenario_schema_version": str(scenario.get("schema_version", "")),
            "scenario_id": scenario_id,
            "scene_id": scene_id,
        },
        "render_backend": render_backend,
        "mission_families": mission_families,
        "social_law_ids": social_law_ids,
        "scene_assets": dict(_mapping_or_empty(scenario.get("scene_assets"))),
        "timing": {
            "fps": fps,
            "start_time_s": 0.0,
            "end_time_s": scenario_end_s,
            "frame_count": frame_count,
        },
        "render_layers": {
            "scene_gaussians": {
                "enabled": True,
                "backend": render_backend,
            },
            "humans": {
                "enabled": bool(humans),
                "representation": "gaussian_ply_sequence",
                "visibility_culling": {
                    "enabled": bool(visibility_culling),
                    "margin_m": float(human_cull_margin_m),
                    "cull_before_ply_merge": True,
                },
            },
            "peer_robots": {
                "enabled": len(robots) > 1,
                "representation": "glb_overlay",
                "visibility_culling": {
                    "enabled": bool(visibility_culling),
                    "margin_m": float(robot_cull_margin_m),
                    "cull_before_glb_render": True,
                },
                "depth_compose": True,
            },
        },
        "actors": {
            "humans": humans,
            "robots": robots,
        },
        "missions": missions,
        "events": _events(scenario),
        "jobs": jobs,
        "warnings": warnings,
    }


def _required_str(payload: Mapping[str, Any], key: str, path: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _positive_float(value: Any, name: str) -> float:
    out = float(value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _mission_record(raw: Any, index: int) -> JsonDict:
    mission = _mapping(raw, f"missions[{index}]")
    mission_type = _required_str(mission, "mission_type", f"missions[{index}]")
    mission_id = _required_str(mission, "mission_id", f"missions[{index}]")
    return {
        "mission_id": mission_id,
        "mission_type": mission_type,
        "release_time_s": float(mission.get("release_time", 0.0) or 0.0),
        "deadline_s": _optional_float(mission.get("deadline")),
        "priority": int(mission.get("priority", 0) or 0),
        "assigned_robot_id": _str_or_none(mission.get("assigned_robot_id")),
        "target_human_id": _str_or_none(mission.get("target_human_id")),
        "target_object_id": _str_or_none(mission.get("target_object_id")),
        "target_region_id": _str_or_none(mission.get("target_region_id")),
        "success_conditions": [str(item) for item in _list(mission.get("success_conditions"))],
        "social_law_ids": [str(item) for item in _list(mission.get("social_law_ids"))],
        "metadata": dict(_mapping_or_empty(mission.get("metadata"))),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _events(scenario: Mapping[str, Any]) -> list[JsonDict]:
    event_log = _mapping_or_empty(scenario.get("event_log"))
    events = _list(event_log.get("events"))
    out: list[JsonDict] = []
    for index, raw in enumerate(events):
        event = _mapping(raw, f"event_log.events[{index}]")
        out.append(
            {
                "event_id": _str_or_none(event.get("event_id")) or f"event_{index:04d}",
                "event_type": _str_or_none(event.get("event_type")) or "unknown",
                "t": float(event.get("t", 0.0) or 0.0),
                "actor_id": _str_or_none(event.get("actor_id")),
                "mission_id": _str_or_none(event.get("mission_id")),
                "payload": dict(_mapping_or_empty(event.get("payload"))),
            }
        )
    return out


def _social_law_ids(scenario: Mapping[str, Any], missions: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for mission in missions:
        values.extend(str(item) for item in _list(mission.get("social_law_ids")))
    for raw in _list(scenario.get("social_structures")):
        if isinstance(raw, Mapping):
            values.extend(str(item) for item in _list(raw.get("law_ids")))
    values.extend(_collect_string_lists(scenario.get("metadata"), keys=("social_law_ids",)))
    return _stable_unique(values)


def _scenario_end_time_s(scenario: Mapping[str, Any], missions: Sequence[Mapping[str, Any]]) -> float:
    times = [0.0]
    for mission in missions:
        times.append(float(mission.get("release_time_s", 0.0) or 0.0))
        deadline = mission.get("deadline_s")
        if deadline is not None:
            times.append(float(deadline))
    for event in _events(scenario):
        times.append(float(event["t"]))
    for group_name in ("robots", "humans"):
        for actor in _list(scenario.get(group_name)):
            if not isinstance(actor, Mapping):
                continue
            for point in _list(actor.get("trajectory")):
                if isinstance(point, Mapping):
                    times.append(float(point.get("t", 0.0) or 0.0))
            for segment in _list(actor.get("behavior_timeline")):
                if isinstance(segment, Mapping):
                    times.append(float(segment.get("end_time", 0.0) or 0.0))
    return max(times)


def _human_actor_record(
    raw: Any,
    index: int,
    *,
    scenario_end_s: float,
    fps: float,
    action_catalog: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    human = _mapping(raw, f"humans[{index}]")
    human_id = _required_str(human, "human_id", f"humans[{index}]")
    role = str(human.get("role", "human"))
    tags = [str(item) for item in _list(human.get("tags"))]
    attributes = dict(_mapping_or_empty(human.get("attributes")))
    appearance = dict(_mapping_or_empty(human.get("appearance")))
    scale = float(appearance.get("scale", 1.0) or 1.0)
    radius_xy_m = float(
        attributes.get("render_radius_xy_m")
        or human.get("render_radius_xy_m")
        or DEFAULT_HUMAN_RADIUS_M
    ) * scale
    height_m = float(
        attributes.get("render_height_m")
        or human.get("render_height_m")
        or DEFAULT_HUMAN_HEIGHT_M
    ) * scale

    return {
        "actor_id": human_id,
        "kind": "human",
        "role": role,
        "tags": tags,
        "attributes": attributes,
        "appearance": appearance,
        "start_pose": _pose_record(human.get("start_map_pose"), f"humans[{index}].start_map_pose"),
        "trajectory": _trajectory_records(
            human,
            actor_path=f"humans[{index}]",
            fps=fps,
            default_motion_state="idle",
        ),
        "action_segments": _human_action_segments(
            human,
            human_id=human_id,
            role=role,
            tags=tags,
            scenario_end_s=scenario_end_s,
            action_catalog=action_catalog,
        ),
        "mission_bindings": [],
        "visibility_bounds": {
            "type": "grounded_cylinder",
            "radius_xy_m": radius_xy_m,
            "height_m": height_m,
        },
        "social_defaults": dict(_mapping_or_empty(human.get("social_defaults"))),
        "metadata": dict(_mapping_or_empty(human.get("metadata"))),
    }


def _robot_actor_record(
    raw: Any,
    index: int,
    *,
    fps: float,
    default_robot_glb: str,
) -> JsonDict:
    robot = _mapping(raw, f"robots[{index}]")
    robot_id = _required_str(robot, "robot_id", f"robots[{index}]")
    embodiment = dict(_mapping_or_empty(robot.get("embodiment")))
    radius_m = float(embodiment.get("footprint_radius") or DEFAULT_ROBOT_RADIUS_M)
    height_m = float(embodiment.get("height") or DEFAULT_ROBOT_HEIGHT_M)
    metadata = dict(_mapping_or_empty(robot.get("metadata")))
    asset = dict(_mapping_or_empty(metadata.get("render_asset")))
    glb_path = (
        _str_or_none(asset.get("glb_path"))
        or _str_or_none(embodiment.get("render_glb_path"))
        or default_robot_glb
    )
    return {
        "actor_id": robot_id,
        "kind": "robot",
        "robot_type": str(robot.get("robot_type", "ground")),
        "capabilities": [str(item) for item in _list(robot.get("capabilities"))],
        "embodiment": embodiment,
        "asset": {
            "glb_path": glb_path,
            "urdf_path": _str_or_none(asset.get("urdf_path") or embodiment.get("source_config_path")),
        },
        "start_pose": _pose_record(robot.get("start_map_pose"), f"robots[{index}].start_map_pose"),
        "trajectory": _trajectory_records(
            robot,
            actor_path=f"robots[{index}]",
            fps=fps,
            default_motion_state="idle",
        ),
        "visibility_bounds": {
            "type": "grounded_cylinder",
            "radius_xy_m": radius_m,
            "height_m": height_m,
        },
        "metadata": metadata,
    }


def _attach_human_mission_bindings(
    humans: Sequence[JsonDict],
    missions: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
) -> None:
    human_by_id = {str(human["actor_id"]): human for human in humans}
    for mission in missions:
        target_human_id = _str_or_none(mission.get("target_human_id"))
        if target_human_id and target_human_id in human_by_id:
            _append_human_binding(
                human_by_id[target_human_id],
                mission=mission,
                binding_role="target_human",
                action_hint=_mission_target_action_hint(str(mission["mission_type"])),
            )

    for structure in _list(scenario.get("social_structures")):
        if not isinstance(structure, Mapping):
            continue
        structure_type = str(structure.get("structure_type", ""))
        for human_id in _list(structure.get("human_ids")):
            human = human_by_id.get(str(human_id))
            if human is None:
                continue
            _append_human_binding(
                human,
                mission=None,
                binding_role=f"social_structure:{structure_type}",
                action_hint=_social_structure_action_hint(structure_type),
                social_structure_id=_str_or_none(structure.get("structure_id")),
            )

    guided_mission_ids = [
        str(mission["mission_id"])
        for mission in missions
        if mission.get("mission_type") == "human_guided_uncertain_region"
    ]
    if guided_mission_ids:
        for human in humans:
            role_and_tags = " ".join([str(human.get("role", "")), *human.get("tags", [])]).lower()
            if "informant" not in role_and_tags and "guidance" not in role_and_tags:
                continue
            for mission_id in guided_mission_ids:
                _append_human_binding(
                    human,
                    mission={"mission_id": mission_id, "mission_type": "human_guided_uncertain_region"},
                    binding_role="guidance_informant",
                    action_hint="wave",
                )


def _append_human_binding(
    human: JsonDict,
    *,
    mission: Mapping[str, Any] | None,
    binding_role: str,
    action_hint: str,
    social_structure_id: str | None = None,
) -> None:
    binding = {
        "binding_role": binding_role,
        "action_hint": action_hint,
        "mission_id": _str_or_none(mission.get("mission_id")) if mission else None,
        "mission_type": _str_or_none(mission.get("mission_type")) if mission else None,
        "social_structure_id": social_structure_id,
    }
    existing = human.setdefault("mission_bindings", [])
    if binding not in existing:
        existing.append(binding)


def _mission_target_action_hint(mission_type: str) -> str:
    if mission_type == "deliver_to_human":
        return "receive_item"
    if mission_type == "serve_queue":
        return "queue_wait"
    if mission_type == "human_guided_uncertain_region":
        return "wave"
    if mission_type == "escort_and_rendezvous":
        return "walk"
    if mission_type in {
        "interruption_recovery",
        "multi_robot_handoff",
        "implicit_need_fulfillment",
        "conflict_resolution",
    }:
        return "wave"
    return "stand"


def _social_structure_action_hint(structure_type: str) -> str:
    if structure_type == "queue":
        return "queue_wait"
    if structure_type == "pedestrian_flow":
        return "walk"
    if structure_type == "f_formation":
        return "wave"
    return "stand"


def _pose_record(raw_pose: Any, path: str) -> JsonDict:
    pose = _mapping(raw_pose, path)
    return {
        "x": float(pose.get("x", 0.0) or 0.0),
        "y": float(pose.get("y", 0.0) or 0.0),
        "yaw_rad": float(pose.get("yaw", pose.get("yaw_rad", 0.0)) or 0.0),
    }


def _trajectory_records(
    actor: Mapping[str, Any],
    *,
    actor_path: str,
    fps: float,
    default_motion_state: str,
) -> list[JsonDict]:
    points = _list(actor.get("trajectory"))
    if not points:
        start_pose = _pose_record(actor.get("start_map_pose"), f"{actor_path}.start_map_pose")
        return [
            {
                "sample_index": 0,
                "frame": 0,
                "t": 0.0,
                "position": [start_pose["x"], start_pose["y"], 0.0],
                "yaw_rad": start_pose["yaw_rad"],
                "motion_state": default_motion_state,
                "metadata": {"source": "start_map_pose_fallback"},
            }
        ]
    records: list[JsonDict] = []
    for index, raw_point in enumerate(points):
        point = _mapping(raw_point, f"{actor_path}.trajectory[{index}]")
        pose = _pose_record(point.get("map_pose"), f"{actor_path}.trajectory[{index}].map_pose")
        t = float(point.get("t", 0.0) or 0.0)
        records.append(
            {
                "sample_index": index,
                "frame": int(round(t * fps)),
                "t": t,
                "position": [pose["x"], pose["y"], 0.0],
                "yaw_rad": pose["yaw_rad"],
                "motion_state": str(point.get("motion_state", default_motion_state) or default_motion_state),
                "action_sequence_id": _str_or_none(point.get("action_sequence_id")),
                "metadata": dict(_mapping_or_empty(point.get("metadata"))),
            }
        )
    return sorted(records, key=lambda item: (float(item["t"]), int(item["sample_index"])))


def _human_action_segments(
    human: Mapping[str, Any],
    *,
    human_id: str,
    role: str,
    tags: Sequence[str],
    scenario_end_s: float,
    action_catalog: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    timeline = _list(human.get("behavior_timeline"))
    sequences = {
        str(sequence["sequence_id"]): dict(sequence)
        for sequence in _list(human.get("action_sequences"))
        if isinstance(sequence, Mapping) and sequence.get("sequence_id")
    }
    if not timeline:
        action_id = _infer_human_action_id(
            behavior_label=None,
            motion_state=_dominant_motion_state(human),
            role=role,
            tags=tags,
            sequence_id=None,
        )
        return [
            _action_segment_record(
                human_id=human_id,
                start_time_s=0.0,
                end_time_s=max(scenario_end_s, 0.0),
                action_sequence_id=f"{human_id}_{action_id}",
                behavior_state={},
                social_structure_id=None,
                sequence={},
                action_id=action_id,
                action_catalog=action_catalog,
            )
        ]
    segments: list[JsonDict] = []
    for index, raw_segment in enumerate(timeline):
        segment = _mapping(raw_segment, f"{human_id}.behavior_timeline[{index}]")
        behavior = dict(_mapping_or_empty(segment.get("behavior_state")))
        sequence_id = _str_or_none(segment.get("action_sequence_id"))
        sequence = sequences.get(sequence_id or "", {})
        action_id = _action_id_for_segment(
            segment,
            sequence=sequence,
            role=role,
            tags=tags,
            motion_state=_motion_state_in_interval(
                human,
                start_time_s=float(segment.get("start_time", 0.0) or 0.0),
                end_time_s=float(segment.get("end_time", scenario_end_s) or scenario_end_s),
            ),
        )
        segments.append(
            _action_segment_record(
                human_id=human_id,
                start_time_s=float(segment.get("start_time", 0.0) or 0.0),
                end_time_s=float(segment.get("end_time", scenario_end_s) or scenario_end_s),
                action_sequence_id=sequence_id or f"{human_id}_{action_id}_{index:03d}",
                behavior_state=behavior,
                social_structure_id=_str_or_none(segment.get("social_structure_id")),
                sequence=sequence,
                action_id=action_id,
                action_catalog=action_catalog,
            )
        )
    return sorted(segments, key=lambda item: (float(item["start_time_s"]), float(item["end_time_s"])))


def _action_segment_record(
    *,
    human_id: str,
    start_time_s: float,
    end_time_s: float,
    action_sequence_id: str,
    behavior_state: Mapping[str, Any],
    social_structure_id: str | None,
    sequence: Mapping[str, Any],
    action_id: str,
    action_catalog: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    catalog_entry = dict(action_catalog.get(action_id, {}))
    source = str(sequence.get("source") or catalog_entry.get("source") or "catalog_or_inferred")
    generator_config = dict(
        _mapping_or_empty(sequence.get("generator_config") or catalog_entry.get("generator_config"))
    )
    source_prompt = str(sequence.get("source_prompt") or catalog_entry.get("source_prompt") or "")
    generation_seed = sequence.get("generation_seed", catalog_entry.get("generation_seed"))
    manifest_path = _str_or_none(sequence.get("manifest_path") or catalog_entry.get("default_manifest_path"))
    ply_frame_dir = _str_or_none(sequence.get("ply_frame_dir") or catalog_entry.get("default_ply_frame_dir"))
    smplx_frame_dir = _str_or_none(
        sequence.get("smplx_frame_dir") or catalog_entry.get("default_smplx_frame_dir")
    )
    pre_generated = bool(sequence.get("pre_generated", catalog_entry.get("pre_generated", True)))
    requires_generation = _action_requires_generation(
        source=source,
        pre_generated=pre_generated,
        ply_frame_dir=ply_frame_dir,
        manifest_path=manifest_path,
    )
    return {
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "action_sequence_id": action_sequence_id,
        "behavior_state": dict(behavior_state),
        "social_structure_id": social_structure_id,
        "render_action_id": action_id,
        "action_label": str(sequence.get("action_label") or catalog_entry.get("display_name") or action_id),
        "asset": {
            "manifest_path": manifest_path,
            "ply_frame_dir": ply_frame_dir,
            "smplx_frame_dir": smplx_frame_dir,
            "source": source,
            "pre_generated": pre_generated,
            "requires_generation": requires_generation,
            "loop": bool(sequence.get("loop", catalog_entry.get("loop", True))),
            "root_motion_mode": str(
                sequence.get("root_motion_mode")
                or catalog_entry.get("root_motion_mode")
                or ("follow_map_path" if action_id == "walk" else "stationary")
            ),
            "fps": _optional_float(sequence.get("fps", catalog_entry.get("fps"))),
            "frame_count": _optional_int(sequence.get("frame_count", catalog_entry.get("frame_count"))),
            "duration_s": _optional_float(sequence.get("duration", catalog_entry.get("duration"))),
        },
        "generation_request": {
            "enabled": requires_generation,
            "generator": _action_generator_name(source),
            "instruction": source_prompt or str(sequence.get("action_label") or action_id),
            "input_style": _generation_input_style(generator_config),
            "keypoints": _action_keypoints(sequence, catalog_entry, generator_config),
            "seed": _optional_int(generation_seed),
            "generator_config": generator_config,
            "output_contract": {
                "manifest_path": manifest_path,
                "ply_frame_dir": ply_frame_dir,
                "smplx_frame_dir": smplx_frame_dir,
            },
        },
        "metadata": {
            "human_id": human_id,
            "sequence_declared_in_scenario": bool(sequence),
            "catalog_action_found": bool(catalog_entry),
        },
    }


def _action_requires_generation(
    *,
    source: str,
    pre_generated: bool,
    ply_frame_dir: str | None,
    manifest_path: str | None,
) -> bool:
    source_norm = source.strip().lower()
    if source_norm in {"kimodo", "stmc", "generated_on_the_fly", "motion_generator"}:
        return True
    return not pre_generated or (ply_frame_dir is None and manifest_path is None)


def _action_generator_name(source: str) -> str | None:
    source_norm = source.strip().lower()
    if source_norm in {"kimodo", "stmc"}:
        return source_norm
    if source_norm in {"generated_on_the_fly", "motion_generator"}:
        return source_norm
    return None


def _generation_input_style(generator_config: Mapping[str, Any]) -> str:
    keypoints = _action_keypoints({}, {}, generator_config)
    if keypoints is not None:
        return "text_with_keypoints"
    return "text"


def _action_keypoints(
    sequence: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
    generator_config: Mapping[str, Any],
) -> Any:
    for payload in (sequence, catalog_entry, generator_config):
        for key in (
            "keypoints",
            "keypoint_constraints",
            "motion_keypoints",
            "pose_keypoints",
            "waypoints",
        ):
            value = payload.get(key)
            if value:
                return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _action_id_for_segment(
    segment: Mapping[str, Any],
    *,
    sequence: Mapping[str, Any],
    role: str,
    tags: Sequence[str],
    motion_state: str | None,
) -> str:
    behavior_state = _mapping_or_empty(segment.get("behavior_state"))
    candidates = [
        _str_or_none(sequence.get("action_label")),
        _str_or_none(sequence.get("sequence_id")),
        _str_or_none(segment.get("action_sequence_id")),
        _str_or_none(behavior_state.get("state_label")),
        _str_or_none(behavior_state.get("social_role")),
        motion_state,
        role,
        *tags,
    ]
    return _infer_human_action_id_from_labels(candidates, role=role, tags=tags)


def _infer_human_action_id(
    *,
    behavior_label: str | None,
    motion_state: str | None,
    role: str,
    tags: Sequence[str],
    sequence_id: str | None,
) -> str:
    return _infer_human_action_id_from_labels(
        [behavior_label, motion_state, role, sequence_id, *tags],
        role=role,
        tags=tags,
    )


def _infer_human_action_id_from_labels(
    labels: Iterable[str | None],
    *,
    role: str,
    tags: Sequence[str],
) -> str:
    role_and_tags = " ".join([role, *tags]).lower()
    tokens: list[str] = []
    for label in labels:
        if not label:
            continue
        tokens.extend(_label_tokens(label))
    token_set = set(tokens)
    if "walk" in token_set or "walking" in token_set or "moving" in token_set:
        return "walk"
    if token_set.intersection({"receive", "receive_item"}):
        return "receive_item"
    if token_set.intersection({"talk", "talking", "gesture", "gesturing", "guidance", "informant", "wave", "waving"}):
        return "wave"
    if token_set.intersection({"yield", "yield_stop"}):
        return "yield_stop"
    if "queue" in role_and_tags or "queue" in token_set or "queueing" in token_set:
        return "queue_wait"
    if token_set.intersection({"idle", "stand", "standing", "stopped", "waiting", "wait"}):
        return "stand"
    return "stand"


def _label_tokens(label: str) -> list[str]:
    lowered = label.strip().lower()
    if not lowered:
        return []
    tokens = [lowered]
    tokens.extend(part for part in re.split(r"[^a-z0-9]+", lowered) if part)
    return tokens


def _dominant_motion_state(human: Mapping[str, Any]) -> str | None:
    states = [
        str(point.get("motion_state"))
        for point in _list(human.get("trajectory"))
        if isinstance(point, Mapping) and point.get("motion_state")
    ]
    if not states:
        return None
    return max(set(states), key=states.count)


def _motion_state_in_interval(
    human: Mapping[str, Any],
    *,
    start_time_s: float,
    end_time_s: float,
) -> str | None:
    states = [
        str(point.get("motion_state"))
        for point in _list(human.get("trajectory"))
        if (
            isinstance(point, Mapping)
            and point.get("motion_state")
            and start_time_s <= float(point.get("t", 0.0) or 0.0) <= end_time_s
        )
    ]
    if not states:
        return _dominant_motion_state(human)
    return max(set(states), key=states.count)


def _training_robot_ids(
    scenario: Mapping[str, Any],
    missions: Sequence[Mapping[str, Any]],
    robot_ids: Sequence[str],
    mission_families: Sequence[str],
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(
        _collect_string_lists(
            scenario.get("metadata"),
            keys=("training_robot_ids", "active_robot_ids"),
        )
    )
    for mission in missions:
        candidates.extend(
            _collect_string_lists(
                mission.get("metadata"),
                keys=("training_robot_ids", "active_robot_ids"),
            )
        )
    has_explicit_robot_set = bool(candidates)
    if not candidates:
        for mission in missions:
            assigned = _str_or_none(mission.get("assigned_robot_id"))
            if assigned is not None:
                candidates.append(assigned)
    if (
        not has_explicit_robot_set
        and any(family in MULTI_ROBOT_MISSION_FAMILIES for family in mission_families)
    ):
        candidates.extend(robot_ids)
    if not candidates:
        candidates.extend(robot_ids)
    known = set(robot_ids)
    ordered = [robot_id for robot_id in _stable_unique(candidates) if robot_id in known]
    return ordered or list(robot_ids)


def _collect_string_lists(value: Any, *, keys: Sequence[str]) -> list[str]:
    out: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, list):
                out.extend(str(entry) for entry in item)
            elif isinstance(item, (Mapping, list)):
                out.extend(_collect_string_lists(item, keys=keys))
    elif isinstance(value, list):
        for item in value:
            out.extend(_collect_string_lists(item, keys=keys))
    return out


def _render_job(
    *,
    scenario_id: str,
    scene_id: str,
    viewpoint_robot_id: str,
    robots: Sequence[Mapping[str, Any]],
    humans: Sequence[Mapping[str, Any]],
    missions: Sequence[Mapping[str, Any]],
    mission_families: Sequence[str],
    render_backend: str,
    visibility_culling: bool,
    human_cull_margin_m: float,
    robot_cull_margin_m: float,
) -> JsonDict:
    robot_by_id = {str(robot["actor_id"]): robot for robot in robots}
    ego_robot = robot_by_id[viewpoint_robot_id]
    peer_robot_ids = [str(robot["actor_id"]) for robot in robots if str(robot["actor_id"]) != viewpoint_robot_id]
    peer_robot_pose_tracks = [
        {
            "actor_id": peer_robot_id,
            "asset": dict(_mapping_or_empty(robot_by_id[peer_robot_id].get("asset"))),
            "visibility_bounds": dict(_mapping_or_empty(robot_by_id[peer_robot_id].get("visibility_bounds"))),
            "trajectory": list(robot_by_id[peer_robot_id]["trajectory"]),
        }
        for peer_robot_id in peer_robot_ids
    ]
    assigned_mission_ids = [
        str(mission["mission_id"])
        for mission in missions
        if mission.get("assigned_robot_id") in (None, viewpoint_robot_id)
        or mission.get("mission_type") == "mission_stream"
    ]
    job_id = f"{scenario_id}__view_{viewpoint_robot_id}"
    return {
        "job_id": job_id,
        "scene_id": scene_id,
        "viewpoint_robot_id": viewpoint_robot_id,
        "mission_families": list(mission_families),
        "assigned_mission_ids": assigned_mission_ids,
        "camera": {
            "mode": "robot_fpv",
            "source_actor_id": viewpoint_robot_id,
            "trajectory": list(ego_robot["trajectory"]),
        },
        "human_actor_ids": [str(human["actor_id"]) for human in humans],
        "peer_robot_ids": peer_robot_ids,
        "peer_robot_pose_tracks": peer_robot_pose_tracks,
        "render_options": {
            "backend": render_backend,
            "save_camera_metadata": True,
            "human_visibility_culling": {
                "enabled": bool(visibility_culling),
                "margin_m": float(human_cull_margin_m),
            },
            "peer_robot_visibility_culling": {
                "enabled": bool(visibility_culling),
                "margin_m": float(robot_cull_margin_m),
            },
            "peer_robot_depth_compose": True,
        },
        "outputs": {
            "stem": job_id,
            "video_name": f"{job_id}.mp4",
            "camera_metadata_name": f"{job_id}_camera.json",
            "actor_debug_name": f"{job_id}_actors.json",
        },
    }


def _manifest_warnings(
    *,
    scenario_id: str,
    mission_families: Sequence[str],
    robots: Sequence[Mapping[str, Any]],
    humans: Sequence[Mapping[str, Any]],
    action_catalog: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    active = set(ACTIVE_MASS_MISSION_FAMILIES)
    schema_only = set(SCHEMA_ONLY_MISSION_FAMILIES)
    for family in mission_families:
        if family in schema_only:
            warnings.append(
                f"{scenario_id}: {family} is schema-only in Pathplanner; renderer behavior is not finalized"
            )
        elif family not in active:
            warnings.append(f"{scenario_id}: unknown mission family {family!r}")
    for robot in robots:
        if len(robot.get("trajectory", [])) < 2:
            warnings.append(f"{scenario_id}: robot {robot['actor_id']} has fewer than two trajectory samples")
    for human in humans:
        for segment in human.get("action_segments", []):
            if not isinstance(segment, Mapping):
                continue
            action_id = str(segment.get("render_action_id"))
            asset = _mapping_or_empty(segment.get("asset"))
            if action_catalog and not segment.get("metadata", {}).get("catalog_action_found"):
                warnings.append(
                    f"{scenario_id}: human {human['actor_id']} action {action_id!r} is not in the action catalog"
                )
            if asset.get("requires_generation"):
                warnings.append(
                    f"{scenario_id}: human {human['actor_id']} action {action_id!r} "
                    "requires action generation before rendering"
                )
            elif not asset.get("ply_frame_dir"):
                warnings.append(
                    f"{scenario_id}: human {human['actor_id']} action {action_id!r} has no ply_frame_dir"
                )
    return warnings
