from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from navdp_datagen.massgen.render_executor import build_render_plans, format_plan_text
from navdp_datagen.massgen.run_config import prepare_render_run


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _pose(x: float, y: float, yaw: float = 0.0) -> dict[str, float]:
    return {"x": x, "y": y, "yaw": yaw}


def _write_png_header(path: Path, *, width: int = 32, height: int = 32) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + int(width).to_bytes(4, "big")
        + int(height).to_bytes(4, "big")
        + b"\x08\x00\x00\x00\x00"
    )


def _write_scene(scene_dir: Path) -> Path:
    scene_dir.mkdir(parents=True)
    (scene_dir / "occupancy.json").write_text(
        json.dumps(
            {
                "scale": 0.5,
                "min": [0.0, 0.0, 0.0],
                "max": [16.0, 16.0, 3.0],
                "lower": [0.0, 0.0, 0.0],
                "upper": [16.0, 16.0, 2.4],
            }
        ),
        encoding="utf-8",
    )
    _write_png_header(scene_dir / "occupancy.png")
    ply_path = scene_dir / "3dgs_raw.ply"
    ply_path.write_text("ply\n", encoding="utf-8")
    return ply_path


def _write_action_catalog(tmp_path: Path) -> Path:
    action_root = tmp_path / "actions"
    actions = []
    for action_id, root_motion_mode in (
        ("receive_item", "stationary"),
        ("stand", "stationary"),
        ("wave", "stationary"),
        ("walk", "follow_map_path"),
        ("yield_stop", "stationary"),
        ("queue_wait", "stationary"),
    ):
        ply_dir = action_root / action_id / "ply_frames"
        ply_dir.mkdir(parents=True)
        actions.append(
            {
                "action_id": action_id,
                "default_ply_frame_dir": str(ply_dir),
                "pre_generated": True,
                "root_motion_mode": root_motion_mode,
            }
        )
    catalog = {"actions": actions}
    path = tmp_path / "action_catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def _write_kimodo_smplx_dir(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "walking_kimodo"
    frame_dir.mkdir()
    frame = {
        "root_pose": [0.0, 0.0, 0.0],
        "body_pose": [[0.0, 0.0, 0.0] for _ in range(21)],
        "trans": [0.0, 0.0, 1.0],
    }
    frame["body_pose"][3] = [0.5, 0.0, 0.0]
    frame["body_pose"][4] = [0.1, 0.0, 0.0]
    (frame_dir / "000000.json").write_text(json.dumps(frame), encoding="utf-8")
    return frame_dir


def _write_simple_scenario(
    tmp_path: Path,
    scene_ply: Path,
    *,
    scenario_id: str = "deliver_easy_001",
    mission_type: str = "deliver_to_human",
    human_role: str = "target",
    human_tags: list[str] | None = None,
    behavior_label: str = "receive_item",
    social_law_ids: list[str] | None = None,
    moving_human: bool = False,
) -> Path:
    human_tags = human_tags or ["target", behavior_label]
    human_trajectory = [
        {"t": 0.0, "map_pose": _pose(6.0, 4.5), "motion_state": "moving" if moving_human else "idle"},
        {
            "t": 2.0,
            "map_pose": _pose(8.0 if moving_human else 6.0, 5.5 if moving_human else 4.5, 0.2),
            "motion_state": "moving" if moving_human else "idle",
        },
    ]
    mission = {
        "mission_id": f"mission_{scenario_id}",
        "mission_type": mission_type,
        "release_time": 0.0,
        "deadline": 2.0,
        "assigned_robot_id": "robot_alpha",
    }
    if mission_type in {"deliver_to_human", "serve_queue"}:
        mission["target_human_id"] = "human_target"
    if social_law_ids:
        mission["social_law_ids"] = social_law_ids
    scenario = {
        "scenario_id": scenario_id,
        "scene_id": "scene_001",
        "schema_version": "0.1",
        "scene_assets": {"splat_model_path": str(scene_ply)},
        "robots": [
            {
                "robot_id": "robot_alpha",
                "robot_type": "ground",
                "start_map_pose": _pose(1.0, 1.0),
                "trajectory": [
                    {"t": 0.0, "map_pose": _pose(1.0, 1.0), "motion_state": "idle"},
                    {"t": 1.0, "map_pose": _pose(3.0, 2.0, 0.2), "motion_state": "moving"},
                    {"t": 2.0, "map_pose": _pose(5.0, 4.0, 0.4), "motion_state": "moving"},
                ],
            }
        ],
        "humans": [
            {
                "human_id": "human_target",
                "role": human_role,
                "tags": human_tags,
                "start_map_pose": _pose(6.0, 4.5),
                "trajectory": human_trajectory,
                "behavior_timeline": [
                    {
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "behavior_state": {"state_label": behavior_label},
                    }
                ],
            }
        ],
        "missions": [mission],
    }
    if social_law_ids:
        scenario["social_law_ids"] = social_law_ids
    path = tmp_path / f"{scenario_id}.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    return path


def _prepared_manifest(
    tmp_path: Path,
    *,
    scenario_id: str = "deliver_easy_001",
    mission_type: str = "deliver_to_human",
    human_role: str = "target",
    human_tags: list[str] | None = None,
    behavior_label: str = "receive_item",
    social_law_ids: list[str] | None = None,
    moving_human: bool = False,
) -> tuple[dict, Path, Path]:
    scene_ply = _write_scene(tmp_path / "scenes" / "scene_001")
    scenario_json = _write_simple_scenario(
        tmp_path,
        scene_ply,
        scenario_id=scenario_id,
        mission_type=mission_type,
        human_role=human_role,
        human_tags=human_tags,
        behavior_label=behavior_label,
        social_law_ids=social_law_ids,
        moving_human=moving_human,
    )
    action_catalog_json = _write_action_catalog(tmp_path)
    result = prepare_render_run(
        {
            "scenario_json": str(scenario_json),
            "action_catalog_json": str(action_catalog_json),
            "output_root": str(tmp_path / "out"),
            "sensor_profile": "navdp_legacy_fpv",
            "selected_sensors": ["fpv_rgb"],
            "strict_assets": False,
        },
        config_path=tmp_path / "config.json",
        write_outputs=False,
    )
    return result["manifest"], scenario_json, tmp_path / "out"


def test_deliver_to_human_executor_materializes_label_and_plans_renderer_command(tmp_path) -> None:
    manifest, scenario_json, output_root = _prepared_manifest(tmp_path)

    plan = build_render_plans(
        manifest,
        manifest_path=scenario_json,
        output_root=output_root,
        families=["deliver_to_human"],
        write_inputs=True,
        python_bin=sys.executable,
    )

    assert plan["job_count"] == 1
    job_plan = plan["plans"][0]
    assert job_plan["job_id"] == "deliver_easy_001__view_robot_alpha"
    assert job_plan["scene_id"] == "scene_001"
    assert job_plan["sensor_names"] == ["fpv_rgb"]
    assert job_plan["human_actor_ids"] == ["human_target"]
    assert job_plan["env"] == {"GAUSSIAN_RENDER_BACKEND": "gsplat"}
    assert job_plan["status"] == "ready"
    assert "--no-mirror-translation" in job_plan["command"]
    assert "--follow-distance" in job_plan["command"]
    assert "--label-id" in job_plan["command"]
    assert "deliver_easy_001__view_robot_alpha" in job_plan["command"]
    assert "--actor-plan-json" in job_plan["command"]
    assert "--actor-seq-dir" not in job_plan["command"]
    assert "--save-actor-metadata" in job_plan["command"]
    assert Path(job_plan["label_path"]).is_file()
    assert Path(job_plan["actor_plan_path"]).is_file()
    label_payload = json.loads(Path(job_plan["label_path"]).read_text(encoding="utf-8"))
    assert label_payload["metadata"]["coordinate_frame"] == "pathplanner_left_handed"
    assert label_payload["metadata"]["source_coordinate_frame"] == "pathplanner_left_handed"
    assert label_payload["metadata"]["coordinate_transform"] == "identity_xy"
    assert label_payload["metadata"]["coordinate_pipeline"][0]["stage"] == "massgen_render_manifest"
    assert (
        label_payload["metadata"]["coordinate_pipeline"][1]["operation"]
        == "identity XY into raster_world"
    )
    assert label_payload["path"]["raster_world"][0]["x"] == 1.0
    assert label_payload["path"]["raster_world"][0]["y"] == 1.0
    assert label_payload["path"]["raster_pixel"][0] == [2, 30]
    assert len(label_payload["path"]["raster_world"]) == 3
    actor_payload = json.loads(Path(job_plan["actor_plan_path"]).read_text(encoding="utf-8"))
    assert actor_payload["schema_version"] == "massgen_actor_bundle.v1"
    assert actor_payload["coordinate_pipeline"][0]["stage"] == "massgen_render_manifest"
    assert (
        actor_payload["coordinate_pipeline"][1]["operation"]
        == "identity XY into actor frame positions"
    )
    assert len(actor_payload["actors"]) == 1
    actor = actor_payload["actors"][0]
    assert actor["actor_id"] == "human_target"
    assert actor["action"]["render_action_id"] == "receive_item"
    assert actor["frames"][0]["position"][:2] == [6.0, 4.5]
    assert len(actor["frames"]) == 3


def test_stationary_human_actor_yaw_preserves_planner_facing(tmp_path) -> None:
    planner_yaw = math.radians(97.0)
    scene_ply = _write_scene(tmp_path / "scenes" / "scene_001")
    scenario_json = _write_simple_scenario(
        tmp_path,
        scene_ply,
        scenario_id="queue_facing_easy_001",
        mission_type="serve_queue",
        human_tags=["queue_participant", "queue_wait", "stationary"],
        behavior_label="queue_wait",
    )
    scenario = json.loads(scenario_json.read_text(encoding="utf-8"))
    human = scenario["humans"][0]
    human["start_map_pose"]["yaw"] = planner_yaw
    for frame in human["trajectory"]:
        frame["map_pose"]["yaw"] = planner_yaw
    scenario_json.write_text(json.dumps(scenario), encoding="utf-8")
    action_catalog_json = _write_action_catalog(tmp_path)
    result = prepare_render_run(
        {
            "scenario_json": str(scenario_json),
            "action_catalog_json": str(action_catalog_json),
            "output_root": str(tmp_path / "out"),
            "sensor_profile": "navdp_legacy_fpv",
            "selected_sensors": ["fpv_rgb"],
            "strict_assets": False,
        },
        config_path=tmp_path / "config.json",
        write_outputs=False,
    )
    manifest = result["manifest"]
    output_root = tmp_path / "out"

    plan = build_render_plans(
        manifest,
        manifest_path=scenario_json,
        output_root=output_root,
        families=["serve_queue"],
        write_inputs=True,
        python_bin=sys.executable,
    )

    actor_payload = json.loads(Path(plan["plans"][0]["actor_plan_path"]).read_text(encoding="utf-8"))
    actor_yaw = actor_payload["actors"][0]["frames"][0]["yaw_rad"]

    expected_actor_yaw = math.atan2(math.cos(planner_yaw), math.sin(planner_yaw)) + math.pi
    old_actor_yaw = math.atan2(-math.cos(planner_yaw), -math.sin(planner_yaw)) + math.pi
    assert abs(_wrap_angle(actor_yaw - expected_actor_yaw)) < 1e-9
    assert abs(abs(_wrap_angle(old_actor_yaw - expected_actor_yaw)) - math.pi) < 1e-9


def test_executor_family_selection_and_text_output(tmp_path) -> None:
    manifest, scenario_json, output_root = _prepared_manifest(tmp_path)

    empty_plan = build_render_plans(
        manifest,
        manifest_path=scenario_json,
        output_root=output_root,
        families=["serve_queue"],
    )
    assert empty_plan["job_count"] == 0

    deliver_plan = build_render_plans(
        manifest,
        manifest_path=scenario_json,
        output_root=output_root,
        families=["deliver_to_human"],
    )
    text = format_plan_text(deliver_plan)
    assert "MassGen render plan" in text
    assert "deliver_easy_001__view_robot_alpha" in text
    assert "GAUSSIAN_RENDER_BACKEND=gsplat" in text
    assert "label path not written" in text
    assert "actor plan not written" in text


def test_executor_supports_one_human_simple_family_rollout(tmp_path) -> None:
    cases = [
        {
            "scenario_id": "guided_easy_001",
            "mission_type": "human_guided_uncertain_region",
            "family": "human_guided_uncertain_region",
            "human_role": "informant",
            "human_tags": ["informant", "guidance"],
            "behavior_label": "wave",
            "expected_action": "wave",
            "moving_human": False,
        },
        {
            "scenario_id": "serve_queue_easy_001",
            "mission_type": "serve_queue",
            "family": "serve_queue",
            "human_role": "queue_member",
            "human_tags": ["queue", "queue_wait"],
            "behavior_label": "queue_wait",
            "expected_action": "queue_wait",
            "moving_human": False,
        },
        {
            "scenario_id": "personal_space_easy_001",
            "mission_type": "navigate_with_social_constraints",
            "family": "navigate_with_social_constraints:personal_space",
            "human_role": "pedestrian",
            "human_tags": ["personal_space"],
            "behavior_label": "stand",
            "social_law_ids": ["personal_space"],
            "expected_action": "stand",
            "moving_human": False,
        },
        {
            "scenario_id": "queue_order_easy_001",
            "mission_type": "navigate_with_social_constraints",
            "family": "navigate_with_social_constraints:queue_order",
            "human_role": "queue_member",
            "human_tags": ["queue_order", "queue_wait"],
            "behavior_label": "queue_wait",
            "social_law_ids": ["queue_order"],
            "expected_action": "queue_wait",
            "moving_human": False,
        },
        {
            "scenario_id": "group_integrity_easy_001",
            "mission_type": "navigate_with_social_constraints",
            "family": "navigate_with_social_constraints:group_integrity",
            "human_role": "group_member",
            "human_tags": ["group_integrity", "stand"],
            "behavior_label": "stand",
            "social_law_ids": ["group_integrity"],
            "expected_action": "stand",
            "moving_human": False,
        },
        {
            "scenario_id": "pedestrian_yield_easy_001",
            "mission_type": "navigate_with_social_constraints",
            "family": "navigate_with_social_constraints:pedestrian_yield",
            "human_role": "pedestrian",
            "human_tags": ["pedestrian_yield", "walk"],
            "behavior_label": "walk",
            "social_law_ids": ["pedestrian_yield"],
            "expected_action": "walk",
            "moving_human": True,
        },
        {
            "scenario_id": "dense_humans_easy_001",
            "mission_type": "dense_dynamic_humans",
            "family": "dense_dynamic_humans",
            "human_role": "pedestrian",
            "human_tags": ["dense", "walk"],
            "behavior_label": "walk",
            "expected_action": "walk",
            "moving_human": True,
        },
        {
            "scenario_id": "dense_avoidance_easy_001",
            "mission_type": "dense_dynamic_avoidance",
            "family": "dense_dynamic_avoidance",
            "human_role": "pedestrian",
            "human_tags": ["avoidance", "walk"],
            "behavior_label": "walk",
            "expected_action": "walk",
            "moving_human": True,
        },
    ]

    for index, case in enumerate(cases):
        case_tmp = tmp_path / f"case_{index}"
        case_tmp.mkdir()
        manifest, scenario_json, output_root = _prepared_manifest(
            case_tmp,
            scenario_id=case["scenario_id"],
            mission_type=case["mission_type"],
            human_role=case["human_role"],
            human_tags=case["human_tags"],
            behavior_label=case["behavior_label"],
            social_law_ids=case.get("social_law_ids"),
            moving_human=case["moving_human"],
        )

        plan = build_render_plans(
            manifest,
            manifest_path=scenario_json,
            output_root=output_root,
            families=[case["family"]],
            write_inputs=True,
            python_bin=sys.executable,
        )

        assert plan["status"] == "ready"
        assert plan["job_count"] == 1
        job_plan = plan["plans"][0]
        assert job_plan["status"] == "ready"
        actor_payload = json.loads(Path(job_plan["actor_plan_path"]).read_text(encoding="utf-8"))
        assert actor_payload["schema_version"] == "massgen_actor_bundle.v1"
        assert actor_payload["actors"][0]["action"]["render_action_id"] == case["expected_action"]
        assert actor_payload["actors"][0]["actor_id"] == "human_target"
        assert "--actor-plan-json" in job_plan["command"]


def test_executor_supports_multi_human_human_only_bundle(tmp_path) -> None:
    cases = [
        {
            "scenario_id": "serve_queue_multi_easy_001",
            "mission_type": "serve_queue",
            "family": "serve_queue",
            "human_role": "queue_member",
            "human_tags": ["queue", "queue_wait"],
            "behavior_label": "queue_wait",
            "moving_human": False,
        },
        {
            "scenario_id": "queue_order_multi_easy_001",
            "mission_type": "navigate_with_social_constraints",
            "family": "navigate_with_social_constraints:queue_order",
            "human_role": "queue_member",
            "human_tags": ["queue_order", "queue_wait"],
            "behavior_label": "queue_wait",
            "social_law_ids": ["queue_order"],
            "moving_human": False,
        },
        {
            "scenario_id": "dense_humans_multi_easy_001",
            "mission_type": "dense_dynamic_humans",
            "family": "dense_dynamic_humans",
            "human_role": "pedestrian",
            "human_tags": ["dense", "walk"],
            "behavior_label": "walk",
            "moving_human": True,
        },
    ]
    for index, case in enumerate(cases):
        case_tmp = tmp_path / f"multi_case_{index}"
        case_tmp.mkdir()
        manifest, scenario_json, output_root = _prepared_manifest(
            case_tmp,
            scenario_id=case["scenario_id"],
            mission_type=case["mission_type"],
            human_role=case["human_role"],
            human_tags=case["human_tags"],
            behavior_label=case["behavior_label"],
            social_law_ids=case.get("social_law_ids"),
            moving_human=case["moving_human"],
        )
        second_human = json.loads(json.dumps(manifest["actors"]["humans"][0]))
        second_human["actor_id"] = "human_peer"
        second_human["start_pose"]["x"] = 7.0
        second_human["start_pose"]["y"] = 5.0
        for point in second_human["trajectory"]:
            point["position"][0] = 7.0
            point["position"][1] = 5.0
        for segment in second_human["action_segments"]:
            segment["metadata"]["human_id"] = "human_peer"
        manifest["actors"]["humans"].append(second_human)
        manifest["jobs"][0]["human_actor_ids"].append("human_peer")

        plan = build_render_plans(
            manifest,
            manifest_path=scenario_json,
            output_root=output_root,
            families=[case["family"]],
            write_inputs=True,
            python_bin=sys.executable,
        )

        assert plan["status"] == "ready"
        job_plan = plan["plans"][0]
        assert job_plan["status"] == "ready"
        assert job_plan["human_actor_ids"] == ["human_target", "human_peer"]
        assert "--actor-plan-json" in job_plan["command"]
        assert "--actor-seq-dir" not in job_plan["command"]
        actor_payload = json.loads(Path(job_plan["actor_plan_path"]).read_text(encoding="utf-8"))
        assert actor_payload["schema_version"] == "massgen_actor_bundle.v1"
        assert {actor["actor_id"] for actor in actor_payload["actors"]} == {"human_target", "human_peer"}
        assert len(actor_payload["actors"]) == 2
        assert all(len(actor["frames"]) == 3 for actor in actor_payload["actors"])


def test_executor_supports_multi_sequence_same_human_bundle(tmp_path) -> None:
    manifest, scenario_json, output_root = _prepared_manifest(
        tmp_path,
        scenario_id="guided_multi_action_easy_001",
        mission_type="human_guided_uncertain_region",
        human_role="informant",
        human_tags=["informant", "guidance"],
        behavior_label="wave",
    )
    human = manifest["actors"]["humans"][0]
    first_segment = json.loads(json.dumps(human["action_segments"][0]))
    second_segment = json.loads(json.dumps(human["action_segments"][0]))
    first_segment["render_action_id"] = "stand"
    first_segment["action_sequence_id"] = "human_target_stand_000"
    first_segment["start_time_s"] = 0.0
    first_segment["end_time_s"] = 1.0
    first_segment["asset"]["ply_frame_dir"] = str(tmp_path / "actions" / "stand" / "ply_frames")
    second_segment["render_action_id"] = "wave"
    second_segment["action_sequence_id"] = "human_target_wave_001"
    second_segment["start_time_s"] = 1.0
    second_segment["end_time_s"] = 2.0
    second_segment["asset"]["ply_frame_dir"] = str(tmp_path / "actions" / "wave" / "ply_frames")
    human["action_segments"] = [first_segment, second_segment]

    plan = build_render_plans(
        manifest,
        manifest_path=scenario_json,
        output_root=output_root,
        families=["human_guided_uncertain_region"],
        write_inputs=True,
        python_bin=sys.executable,
    )

    assert plan["status"] == "ready"
    job_plan = plan["plans"][0]
    assert job_plan["status"] == "ready"
    actor_payload = json.loads(Path(job_plan["actor_plan_path"]).read_text(encoding="utf-8"))
    assert actor_payload["schema_version"] == "massgen_actor_bundle.v1"
    assert len(actor_payload["actors"]) == 2
    assert {actor["actor_id"] for actor in actor_payload["actors"]} == {"human_target"}
    assert {actor["action"]["render_action_id"] for actor in actor_payload["actors"]} == {"stand", "wave"}
    stand = next(actor for actor in actor_payload["actors"] if actor["action"]["render_action_id"] == "stand")
    wave = next(actor for actor in actor_payload["actors"] if actor["action"]["render_action_id"] == "wave")
    assert [frame["active"] for frame in stand["frames"]] == [True, True, False]
    assert [frame["active"] for frame in wave["frames"]] == [False, True, True]
    assert stand["action"]["animation_frame_policy"] == "first_frame_static"
    assert wave["action"]["animation_frame_policy"] == "first_frame_static"
    assert {frame["animation_frame_index"] for frame in stand["frames"]} == {0}
    assert {frame["animation_frame_index"] for frame in wave["frames"]} == {0}
    assert "uses 2 renderer action segments" in "\n".join(job_plan["warnings"])


def test_executor_plans_chained_peer_robot_overlays(tmp_path) -> None:
    manifest, scenario_json, output_root = _prepared_manifest(tmp_path)
    robot_glb = tmp_path / "robot.glb"
    robot_glb.write_bytes(b"glb")
    robot_urdf = tmp_path / "g1.urdf"
    robot_urdf.write_text("<robot name='g1' />", encoding="utf-8")
    overlay_script = tmp_path / "render_glb_robot_overlay.py"
    overlay_script.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    kimodo_smplx_dir = _write_kimodo_smplx_dir(tmp_path)

    manifest["jobs"][0]["peer_robot_ids"] = ["robot_beta", "robot_gamma"]
    manifest["jobs"][0]["peer_robot_pose_tracks"] = [
        {
            "actor_id": "robot_beta",
            "asset": {"glb_path": str(robot_glb), "urdf_path": str(robot_urdf)},
            "trajectory": [
                {"t": 0.0, "position": [2.0, 2.0, 0.0], "yaw_rad": 0.0},
                {"t": 1.0, "position": [3.0, 2.0, 0.0], "yaw_rad": 0.0},
                {"t": 2.0, "position": [4.0, 2.0, 0.0], "yaw_rad": 0.0},
            ],
        },
        {
            "actor_id": "robot_gamma",
            "asset": {"glb_path": str(robot_glb), "urdf_path": str(robot_urdf)},
            "trajectory": [
                {"t": 0.0, "position": [2.0, 3.0, 0.0], "yaw_rad": 0.0},
                {"t": 1.0, "position": [3.0, 3.0, 0.0], "yaw_rad": 0.0},
                {"t": 2.0, "position": [4.0, 3.0, 0.0], "yaw_rad": 0.0},
            ],
        },
    ]

    plan = build_render_plans(
        manifest,
        manifest_path=scenario_json,
        output_root=output_root,
        families=["deliver_to_human"],
        write_inputs=True,
        python_bin=sys.executable,
        robot_overlay_script=overlay_script,
        robot_glb=robot_glb,
        robot_urdf=robot_urdf,
        kimodo_smplx_dir=kimodo_smplx_dir,
    )

    assert plan["status"] == "ready"
    job_plan = plan["plans"][0]
    assert job_plan["status"] == "ready"
    assert "--rgb-frames" in job_plan["command"]
    assert len(job_plan["robot_overlay_commands"]) == 2
    first, second = job_plan["robot_overlay_commands"]
    assert first["actor_id"] == "robot_beta"
    assert second["actor_id"] == "robot_gamma"
    assert first["video"] is None
    assert second["video"].endswith("__with_peer_robots.mp4")
    assert second["input_frames_dir"] == first["output_frames_dir"]
    assert Path(first["poses_json"]).is_file()
    assert Path(first["amo_poses_json"]).is_file()
    pose_payload = json.loads(Path(first["poses_json"]).read_text(encoding="utf-8"))
    amo_payload = json.loads(Path(first["amo_poses_json"]).read_text(encoding="utf-8"))
    assert pose_payload["schema_version"] == "massgen_robot_overlay_poses.v1"
    assert len(pose_payload["frames"]) == 3
    assert amo_payload["schema_version"] == "g1_amo_retarget.v1"
    assert amo_payload["frames"][0]["joint_positions"]["left_shoulder_roll_joint"] == 0.18
