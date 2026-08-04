from __future__ import annotations

import pytest

from utils.massgen_render_manifest import (
    ACTIVE_MASS_MISSION_FAMILIES,
    DEFAULT_RENDER_BACKEND,
    scenario_to_render_manifest,
)


def _action_catalog() -> dict[str, dict[str, object]]:
    action_ids = ("stand", "walk", "wave", "queue_wait", "receive_item", "yield_stop")
    return {
        action_id: {
            "action_id": action_id,
            "default_manifest_path": f"assets/human_actions/{action_id}/manifest.json",
            "default_ply_frame_dir": f"assets/human_actions/{action_id}/ply_frames",
            "default_smplx_frame_dir": f"assets/human_actions/{action_id}/smplx_frames",
            "loop": action_id not in {"receive_item"},
            "root_motion_mode": "follow_map_path" if action_id == "walk" else "stationary",
        }
        for action_id in action_ids
    }


def _pose(x: float, y: float, yaw: float = 0.0) -> dict[str, float]:
    return {"x": x, "y": y, "yaw": yaw}


def _trajectory(start_x: float, *, motion_state: str = "moving") -> list[dict[str, object]]:
    return [
        {"t": 0.0, "map_pose": _pose(start_x, 0.0, 0.0), "motion_state": "idle"},
        {"t": 3.0, "map_pose": _pose(start_x + 1.0, 1.0, 0.5), "motion_state": motion_state},
    ]


def _robot(robot_id: str, index: int) -> dict[str, object]:
    return {
        "robot_id": robot_id,
        "robot_type": "ground",
        "start_map_pose": _pose(float(index), 0.0),
        "embodiment": {
            "embodiment_id": "ground_robot_default",
            "footprint_radius": 0.3,
            "height": 1.3,
        },
        "trajectory": _trajectory(float(index)),
    }


def _human(
    human_id: str,
    *,
    role: str = "bystander",
    tags: list[str] | None = None,
    state_label: str = "waiting",
    motion_state: str = "idle",
) -> dict[str, object]:
    return {
        "human_id": human_id,
        "role": role,
        "start_map_pose": _pose(2.0, 1.0),
        "tags": tags or [],
        "appearance": {
            "appearance_id": f"appearance_{human_id}",
            "canonical_ply_path": f"assets/humans/{human_id}/canonical.ply",
            "canonical_smplx_json_path": f"assets/humans/{human_id}/canonical_smplx.json",
            "scale": 1.0,
        },
        "trajectory": _trajectory(2.0, motion_state=motion_state),
        "behavior_timeline": [
            {
                "start_time": 0.0,
                "end_time": 3.0,
                "behavior_state": {
                    "state_label": state_label,
                    "social_role": role,
                    "interaction_state": "available",
                },
            }
        ],
    }


def _scenario(
    mission_family: str,
    *,
    robots: list[dict[str, object]] | None = None,
    humans: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
    social_structures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    robots = robots or [_robot("robot_0", 0)]
    humans = humans if humans is not None else [_human("human_0")]
    target_human_id = "human_0" if humans and mission_family in {"deliver_to_human", "serve_queue"} else None
    return {
        "scenario_id": f"scenario_{mission_family}",
        "scene_id": "scene_001",
        "schema_version": "0.1",
        "scene_assets": {"splat_model_path": "scenes/scene_001/point_cloud.ply"},
        "robots": robots,
        "humans": humans,
        "missions": [
            {
                "mission_id": f"mission_{mission_family}_001",
                "mission_type": mission_family,
                "release_time": 0.0,
                "deadline": 5.0,
                "assigned_robot_id": robots[0]["robot_id"],
                "target_human_id": target_human_id,
                "target_region_id": "region_goal",
                "social_law_ids": (
                    ["L4_queue_order"] if mission_family == "navigate_with_social_constraints" else []
                ),
                "metadata": {},
            }
        ],
        "social_structures": social_structures or [],
        "event_log": {"events": [{"event_id": "evt_done", "event_type": "completion", "t": 5.0}]},
        "metadata": metadata or {},
    }


@pytest.mark.parametrize("mission_family", ACTIVE_MASS_MISSION_FAMILIES)
def test_active_mission_family_gets_gsplat_render_jobs(mission_family: str) -> None:
    robot_count = 2 if mission_family in {"dense_multi_robot", "dense_dynamic_combined", "mission_stream"} else 1
    scenario = _scenario(
        mission_family,
        robots=[_robot(f"robot_{idx}", idx) for idx in range(robot_count)],
        humans=[] if mission_family == "dense_multi_robot" else [_human("human_0")],
    )

    manifest = scenario_to_render_manifest(scenario, action_catalog=_action_catalog())

    assert manifest["render_backend"] == DEFAULT_RENDER_BACKEND
    assert manifest["render_layers"]["scene_gaussians"]["backend"] == "gsplat"
    assert mission_family in manifest["mission_families"]
    assert len(manifest["jobs"]) == robot_count
    assert not any("unknown mission family" in warning for warning in manifest["warnings"])


def test_dense_combined_manifest_tracks_dynamic_humans_and_peer_robots() -> None:
    scenario = _scenario(
        "dense_dynamic_combined",
        robots=[_robot("robot_alpha", 0), _robot("robot_beta", 1), _robot("robot_gamma", 2)],
        humans=[
            _human("human_pedestrian", role="pedestrian", tags=["walking_pedestrian"], state_label="walking"),
            _human("human_queue", role="queue_participant", tags=["queue_participant"], state_label="waiting"),
        ],
        social_structures=[
            {
                "structure_id": "social_queue_001",
                "structure_type": "queue",
                "human_ids": ["human_queue"],
                "law_ids": ["L4_queue_order"],
            }
        ],
    )

    manifest = scenario_to_render_manifest(scenario, action_catalog=_action_catalog())

    assert len(manifest["jobs"]) == 3
    for job in manifest["jobs"]:
        assert len(job["peer_robot_ids"]) == 2
        assert len(job["peer_robot_pose_tracks"]) == 2
        assert all(track["trajectory"] for track in job["peer_robot_pose_tracks"])
        assert job["render_options"]["human_visibility_culling"]["enabled"]
        assert job["render_options"]["peer_robot_visibility_culling"]["enabled"]
    human_actions = {
        segment["render_action_id"]
        for human in manifest["actors"]["humans"]
        for segment in human["action_segments"]
    }
    assert {"walk", "queue_wait"}.issubset(human_actions)
    queue_human = next(human for human in manifest["actors"]["humans"] if human["actor_id"] == "human_queue")
    assert queue_human["mission_bindings"][0]["action_hint"] == "queue_wait"


def test_mission_specific_human_action_hints_are_preserved() -> None:
    deliver_manifest = scenario_to_render_manifest(
        _scenario("deliver_to_human", humans=[_human("human_0", tags=["target"])]),
        action_catalog=_action_catalog(),
    )
    target = deliver_manifest["actors"]["humans"][0]
    assert target["mission_bindings"][0]["action_hint"] == "receive_item"

    guided_manifest = scenario_to_render_manifest(
        _scenario(
            "human_guided_uncertain_region",
            humans=[_human("human_informant", role="informant", tags=["guidance_source"])],
        ),
        action_catalog=_action_catalog(),
    )
    informant = guided_manifest["actors"]["humans"][0]
    assert any(binding["action_hint"] == "wave" for binding in informant["mission_bindings"])
    assert informant["action_segments"][0]["render_action_id"] == "wave"


def test_explicit_training_robot_ids_control_viewpoints() -> None:
    scenario = _scenario(
        "mission_stream",
        robots=[_robot("robot_alpha", 0), _robot("robot_beta", 1), _robot("robot_gamma", 2)],
        metadata={"mission_stream": {"training_robot_ids": ["robot_beta", "robot_gamma"]}},
    )

    manifest = scenario_to_render_manifest(scenario, action_catalog=_action_catalog())

    assert [job["viewpoint_robot_id"] for job in manifest["jobs"]] == ["robot_beta", "robot_gamma"]
    assert manifest["jobs"][0]["peer_robot_ids"] == ["robot_alpha", "robot_gamma"]
    assert [track["actor_id"] for track in manifest["jobs"][0]["peer_robot_pose_tracks"]] == [
        "robot_alpha",
        "robot_gamma",
    ]


def test_generated_kimodo_action_preserves_text_and_keypoints() -> None:
    human = _human("human_informant", role="informant", tags=["guidance_source"])
    human["action_sequences"] = [
        {
            "sequence_id": "seq_guidance_kimodo",
            "action_label": "gesture",
            "source": "kimodo",
            "source_prompt": "stand still, raise the right hand, then point toward the corridor",
            "pre_generated": False,
            "generation_seed": 1234,
            "generator_config": {
                "duration_s": 2.5,
                "keypoints": [
                    {"t": 0.0, "body": "neutral_stand"},
                    {"t": 1.0, "right_hand": [0.35, 0.0, 1.45]},
                    {"t": 2.0, "right_hand": [0.65, 0.4, 1.35]},
                ],
            },
        }
    ]
    human["behavior_timeline"][0]["action_sequence_id"] = "seq_guidance_kimodo"

    manifest = scenario_to_render_manifest(
        _scenario("human_guided_uncertain_region", humans=[human]),
        action_catalog=_action_catalog(),
    )

    segment = manifest["actors"]["humans"][0]["action_segments"][0]
    assert segment["render_action_id"] == "wave"
    assert segment["asset"]["requires_generation"]
    assert segment["generation_request"]["enabled"]
    assert segment["generation_request"]["generator"] == "kimodo"
    assert segment["generation_request"]["input_style"] == "text_with_keypoints"
    assert segment["generation_request"]["seed"] == 1234
    assert segment["generation_request"]["keypoints"][1]["right_hand"] == [0.35, 0.0, 1.45]
    assert any("requires action generation before rendering" in warning for warning in manifest["warnings"])


def test_generated_stmc_action_can_be_text_only() -> None:
    human = _human("human_service", role="queue_participant", tags=["queue_participant"])
    human["action_sequences"] = [
        {
            "sequence_id": "seq_service_stmc",
            "action_label": "receive item",
            "source": "stmc",
            "source_prompt": "wait in line, step forward, and accept a small item",
            "pre_generated": False,
            "generator_config": {"duration_s": 4.0},
        }
    ]
    human["behavior_timeline"][0]["action_sequence_id"] = "seq_service_stmc"

    manifest = scenario_to_render_manifest(
        _scenario("serve_queue", humans=[human]),
        action_catalog=_action_catalog(),
    )

    segment = manifest["actors"]["humans"][0]["action_segments"][0]
    assert segment["render_action_id"] == "receive_item"
    assert segment["asset"]["requires_generation"]
    assert segment["generation_request"]["generator"] == "stmc"
    assert segment["generation_request"]["input_style"] == "text"
    assert segment["generation_request"]["instruction"] == "wait in line, step forward, and accept a small item"
