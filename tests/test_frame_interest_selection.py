from __future__ import annotations

import json
from pathlib import Path

import pytest

from navdp_datagen.massgen.frame_selection import (
    DEFAULT_FUTURE_FRAMES,
    DEFAULT_PAST_FRAMES,
    FrameSelectionConfig,
    apply_frame_selection_to_manifest,
    select_frame_interest_windows,
)
from navdp_datagen.massgen.frame_selection_pilot import (
    MissionFamilyPilotConfig,
    prepare_mission_family_pilot,
)
from navdp_datagen.massgen.render_executor import build_render_plans
from utils.massgen_render_manifest import ACTIVE_MASS_MISSION_FAMILIES


def _write_png_header(path: Path, *, width: int = 64, height: int = 64) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + int(width).to_bytes(4, "big")
        + int(height).to_bytes(4, "big")
        + b"\x08\x00\x00\x00\x00"
    )


def _scene(tmp_path: Path) -> Path:
    scene_dir = tmp_path / "scenes" / "scene_001"
    scene_dir.mkdir(parents=True)
    (scene_dir / "occupancy.json").write_text(
        json.dumps(
            {
                "scale": 0.5,
                "min": [0.0, 0.0, 0.0],
                "max": [32.0, 32.0, 3.0],
                "lower": [0.0, 0.0, 0.0],
                "upper": [32.0, 32.0, 2.4],
            }
        ),
        encoding="utf-8",
    )
    _write_png_header(scene_dir / "occupancy.png")
    ply = scene_dir / "point_cloud.ply"
    ply.write_text("ply\n", encoding="utf-8")
    return ply


def _trajectory(count: int = 100) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for index in range(count):
        if 47 <= index <= 53:
            x = 4.7
            state = "stopped"
        else:
            x = index * 0.1
            state = "moving"
        points.append(
            {
                "sample_index": index,
                "frame": index,
                "t": index * 0.1,
                "position": [x, 0.0, 0.0],
                "yaw_rad": 0.0,
                "motion_state": state,
            }
        )
    return points


def _manifest(
    scene_ply: Path,
    *,
    mission_family: str = "dense_dynamic_avoidance",
    scenario_id: str = "scenario_001",
) -> dict[str, object]:
    trajectory = _trajectory()
    job_id = f"{scenario_id}__view_robot_alpha"
    return {
        "schema_version": "massgen_render_manifest/v0.1",
        "source": {
            "kind": "pathplanner_scenario",
            "scenario_id": scenario_id,
            "scene_id": "scene_001",
        },
        "render_backend": "gsplat",
        "mission_families": [mission_family],
        "social_law_ids": ["pedestrian_yield"],
        "scene_assets": {"splat_model_path": str(scene_ply)},
        "timing": {"fps": 10.0, "start_time_s": 0.0, "end_time_s": 9.9, "frame_count": 100},
        "actors": {"humans": [], "robots": []},
        "missions": [
            {
                "mission_id": "mission_001",
                "mission_type": mission_family,
                "release_time_s": 1.0,
                "deadline_s": 8.0,
            }
        ],
        "events": [{"event_id": "evt_yield", "event_type": "yield", "t": 5.0}],
        "jobs": [
            {
                "job_id": job_id,
                "scene_id": "scene_001",
                "viewpoint_robot_id": "robot_alpha",
                "mission_families": [mission_family],
                "assigned_mission_ids": ["mission_001"],
                "camera": {"mode": "robot_fpv", "source_actor_id": "robot_alpha", "trajectory": trajectory},
                "human_actor_ids": [],
                "peer_robot_ids": [],
                "peer_robot_pose_tracks": [],
                "render_options": {"backend": "gsplat"},
                "outputs": {"stem": job_id},
            }
        ],
        "warnings": [],
    }


def test_frame_interest_selection_emits_past_and_current_windows(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))
    config = FrameSelectionConfig(target_count=4, seed=1234, preserve_mission_endpoints=False)

    selection = select_frame_interest_windows([manifest], manifest_paths=["manifest.json"], config=config)

    assert selection["schema_version"] == "navdp_frame_interest_selection/v0.1"
    assert selection["selection_summary"]["selected_target_count"] == 4
    expected_window_frames = DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
    assert expected_window_frames == 33
    assert selection["selection_summary"]["window_frame_count"] == expected_window_frames
    assert (
        selection["selection_summary"]["requested_window_frame_renders"]
        == 4 * expected_window_frames
    )
    for chunk in selection["chunks"]:
        frames = chunk["window"]["source_frame_indices"]
        assert len(frames) == DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
        assert chunk["target_frame"] in frames
        assert chunk["render_contract"]["preserve_stationary_frames"] is True


def test_apply_frame_selection_rewrites_jobs_to_selected_windows(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))
    config = FrameSelectionConfig(target_count=2, seed=7, preserve_mission_endpoints=False)
    selection = select_frame_interest_windows([manifest], manifest_paths=["manifest.json"], config=config)

    selected_manifest = apply_frame_selection_to_manifest(manifest, selection, manifest_path="manifest.json")

    assert selected_manifest["timing"]["selection_mode"] == "frame_interest_windows"
    assert len(selected_manifest["jobs"]) == 2
    for job in selected_manifest["jobs"]:
        assert job["job_id"].startswith("scenario_001__view_robot_alpha__m000__foi_")
        assert (
            len(job["camera"]["trajectory"])
            == DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
        )
        assert job["camera"]["preserve_frame_samples"] is True
        assert job["frame_selection"]["preserve_stationary_frames"] is True
        assert job["camera"]["trajectory"][0]["metadata"]["source_frame_index"] >= 0


def test_selected_render_plan_preserves_stationary_frame_samples(tmp_path: Path) -> None:
    scene_ply = _scene(tmp_path)
    manifest = _manifest(scene_ply)
    config = FrameSelectionConfig(
        target_count=1,
        seed=12,
        target_bucket_ratios={"route_decision": 1.0},
        min_target_spacing_frames=0,
    )
    selection = select_frame_interest_windows([manifest], manifest_paths=["manifest.json"], config=config)
    chunk = dict(selection["chunks"][0])
    source_indices = list(range(50 - DEFAULT_PAST_FRAMES, 50 + DEFAULT_FUTURE_FRAMES + 1))
    chunk.update(
        {
            "chunk_id": "scenario_001__view_robot_alpha__m000__foi_000050",
            "window_job_id": "scenario_001__view_robot_alpha__m000__foi_000050",
            "target_frame": 50,
            "target_source_frame_id": 50,
            "target_time_s": 5.0,
        }
    )
    chunk["window"] = {
        **chunk["window"],
        "source_frame_indices": source_indices,
        "render_frame_count": len(source_indices),
    }
    selection["chunks"] = [chunk]
    selected_manifest = apply_frame_selection_to_manifest(manifest, selection, manifest_path="manifest.json")

    plan = build_render_plans(
        selected_manifest,
        manifest_path=tmp_path / "manifest.json",
        output_root=tmp_path / "out",
        scenes_dir=scene_ply.parent.parent,
        write_inputs=True,
        python_bin="python3",
    )

    assert plan["status"] == "ready"
    label_path = Path(plan["plans"][0]["label_path"])
    label = json.loads(label_path.read_text(encoding="utf-8"))
    assert label["metadata"]["preserve_frame_samples"] is True
    assert (
        len(label["path"]["raster_world"])
        == DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
    )
    assert sum(1 for point in label["path"]["raster_world"] if point["x"] == 4.7) == 4


def test_family_filter_excludes_other_manifest_jobs(tmp_path: Path) -> None:
    scene_ply = _scene(tmp_path)
    requested = _manifest(
        scene_ply,
        mission_family="deliver_to_human",
        scenario_id="scenario_deliver",
    )
    other = _manifest(
        scene_ply,
        mission_family="dense_multi_robot",
        scenario_id="scenario_multi",
    )
    config = FrameSelectionConfig(
        target_count=2,
        mission_families=("deliver_to_human",),
        preserve_mission_endpoints=False,
    )

    selection = select_frame_interest_windows(
        [requested, other],
        manifest_paths=["deliver.json", "multi.json"],
        config=config,
    )

    assert selection["config"]["mission_families"] == ["deliver_to_human"]
    assert selection["selection_summary"]["selected_target_count"] == 2
    assert all(chunk["mission_families"] == ["deliver_to_human"] for chunk in selection["chunks"])


def test_sparse_source_frame_ids_are_densified_for_temporal_windows(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))
    job = manifest["jobs"][0]
    job["camera"]["trajectory"] = [
        {
            "sample_index": 0,
            "frame": 0,
            "t": 0.0,
            "position": [0.0, 0.0, 0.0],
            "yaw_rad": 0.0,
            "motion_state": "moving",
        },
        {
            "sample_index": 1,
            "frame": 100,
            "t": 10.0,
            "position": [10.0, 0.0, 0.0],
            "yaw_rad": 0.0,
            "motion_state": "moving",
        },
    ]

    sparse_selection = select_frame_interest_windows(
        [manifest],
        manifest_paths=["manifest.json"],
        config=FrameSelectionConfig(
            target_count=1,
            densify_frame_gaps=False,
            preserve_mission_endpoints=False,
        ),
    )
    dense_selection = select_frame_interest_windows(
        [manifest],
        manifest_paths=["manifest.json"],
        config=FrameSelectionConfig(
            target_count=1,
            densify_frame_gaps=True,
            preserve_mission_endpoints=False,
        ),
    )

    assert sparse_selection["selection_summary"]["candidate_count"] == 0
    assert dense_selection["selection_summary"]["selected_target_count"] == 1
    assert dense_selection["chunks"][0]["render_contract"]["densify_frame_gaps"] is True

    selected_manifest = apply_frame_selection_to_manifest(
        manifest,
        dense_selection,
        manifest_path="manifest.json",
    )
    selected_points = selected_manifest["jobs"][0]["camera"]["trajectory"]
    source_frame_ids = [point["metadata"]["source_frame_id"] for point in selected_points]
    expected_window_frames = DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
    assert len(selected_points) == expected_window_frames
    assert source_frame_ids == list(
        range(source_frame_ids[0], source_frame_ids[0] + expected_window_frames)
    )
    assert any(point["metadata"].get("frame_selection_interpolated") for point in selected_points)


def test_final_trajectory_endpoint_is_a_mandatory_past_and_current_anchor(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))

    selection = select_frame_interest_windows(
        [manifest],
        manifest_paths=["manifest.json"],
        config=FrameSelectionConfig(target_count=1, edge_window_policy="reject"),
    )

    chunk = next(
        chunk for chunk in selection["chunks"] if chunk["target_role"] == "mandatory_anchor"
    )
    assert selection["selection_summary"]["mandatory_anchor_count"] == 1
    assert selection["selection_summary"]["selected_interest_target_count"] == 1
    assert selection["selection_summary"]["selected_target_count"] == 2
    assert sum(
        selection["selection_summary"]["selected_interest_action_counts"].values()
    ) == 1
    assert sum(
        selection["selection_summary"]["selected_interest_bucket_counts"].values()
    ) == 1
    assert chunk["target_frame"] == 99
    assert chunk["target_role"] == "mandatory_anchor"
    assert chunk["mandatory_anchor_types"] == ["trajectory_end"]
    assert chunk["window"]["edge_policy"] == "reject"
    assert chunk["window"]["source_frame_indices"][0] == 99 - DEFAULT_PAST_FRAMES
    assert chunk["window"]["source_frame_indices"][32] == 99
    assert chunk["window"]["source_frame_indices"][-1] == 99
    assert len(chunk["window"]["source_frame_indices"]) == DEFAULT_PAST_FRAMES + 1


def test_assigned_sub_mission_completions_are_always_selected(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path), mission_family="mission_stream")
    manifest["missions"] = [
        {"mission_id": "child_a", "mission_type": "deliver_to_human"},
        {"mission_id": "child_b", "mission_type": "navigate_with_social_constraints"},
    ]
    manifest["events"] = [
        {
            "event_id": "complete_a",
            "event_type": "completion",
            "mission_id": "child_a",
            "actor_id": "robot_alpha",
            "t": 4.0,
        },
        {
            "event_id": "complete_b",
            "event_type": "robot_eos",
            "mission_id": "child_b",
            "actor_id": "robot_alpha",
            "t": 7.0,
        },
    ]
    manifest["jobs"][0]["assigned_mission_ids"] = ["child_a", "child_b"]

    selection = select_frame_interest_windows(
        [manifest],
        manifest_paths=["manifest.json"],
        config=FrameSelectionConfig(target_count=1),
    )

    mandatory_chunks = [
        chunk for chunk in selection["chunks"] if chunk["target_role"] == "mandatory_anchor"
    ]
    chunks_by_frame = {chunk["target_frame"]: chunk for chunk in mandatory_chunks}
    assert set(chunks_by_frame) == {40, 70, 99}
    assert "mission_section_end:child_a" in chunks_by_frame[40]["mandatory_anchor_types"]
    assert "mission_section_end:child_b" in chunks_by_frame[70]["mandatory_anchor_types"]
    assert "trajectory_end" in chunks_by_frame[99]["mandatory_anchor_types"]
    assert selection["selection_summary"]["mandatory_anchor_count"] == 3
    assert selection["selection_summary"]["selected_interest_target_count"] == 1
    assert selection["selection_summary"]["selected_target_count"] == 4


def test_summary_separates_center_actions_from_retained_window_actions(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))
    selection = select_frame_interest_windows(
        [manifest],
        manifest_paths=["manifest.json"],
        config=FrameSelectionConfig(
            target_count=1,
            preserve_mission_endpoints=False,
            target_bucket_ratios={"route_decision": 1.0},
            min_target_spacing_frames=0,
        ),
    )

    summary = selection["selection_summary"]
    assert summary["selected_center_action_counts"] == summary["selected_action_counts"]
    assert sum(summary["selected_window_action_counts"].values()) == DEFAULT_PAST_FRAMES + 1
    assert summary["selected_window_action_counts"]["move"] > 0
    assert sum(summary["selected_window_action_ratios"].values()) == pytest.approx(1.0)


def test_scored_center_action_minimums_exclude_mandatory_anchors(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))
    trajectory = []
    for index in range(220):
        motion_state = "moving"
        yaw = 0.0
        if 45 <= index < 85:
            motion_state = "stopped"
        elif 85 <= index < 130:
            yaw = (index - 85) * 0.2
        elif 130 <= index < 175:
            yaw = 9.0 - (index - 130) * 0.2
        if index == 219:
            motion_state = "done"
        trajectory.append(
            {
                "sample_index": index,
                "frame": index,
                "t": index * 0.1,
                "position": [index * 0.1, 0.0, 0.0],
                "yaw_rad": yaw,
                "motion_state": motion_state,
            }
        )
    manifest["jobs"][0]["camera"]["trajectory"] = trajectory

    selection = select_frame_interest_windows(
        [manifest],
        manifest_paths=["manifest.json"],
        config=FrameSelectionConfig(
            target_count=20,
            min_target_spacing_frames=0,
            target_bucket_ratios={"route_decision": 1.0},
        ),
    )

    assert selection["distribution"]["target_action_counts"] == {
        "move": 11,
        "stop": 3,
        "turn_left": 3,
        "turn_right": 3,
    }
    summary = selection["selection_summary"]
    assert summary["selected_interest_action_counts"] == {
        "move": 11,
        "stop": 3,
        "turn_left": 3,
        "turn_right": 3,
    }
    assert summary["mandatory_anchor_action_counts"] == {"stop": 1}
    assert summary["action_deficits"] == {
        "move": 0,
        "stop": 0,
        "turn_left": 0,
        "turn_right": 0,
    }
    assert summary["training_center_sampling"]["population"] == "scored_poi_centers"
    assert summary["training_center_sampling"]["mandatory_anchor_policy"] == "retained_separate_pool"
    assert summary["scored_center_window_composition"]["turn_left"]["window_count"] == 3


def test_seeded_source_path_sampling_keeps_complete_paths(tmp_path: Path) -> None:
    scene_ply = _scene(tmp_path)
    manifests = [
        _manifest(scene_ply, scenario_id=f"scenario_{index}")
        for index in range(4)
    ]
    paths = [f"manifest_{index}.json" for index in range(4)]
    config = FrameSelectionConfig(
        target_count=4,
        max_source_paths=2,
        preserve_mission_endpoints=False,
        seed=123,
    )

    first = select_frame_interest_windows(manifests, manifest_paths=paths, config=config)
    second = select_frame_interest_windows(manifests, manifest_paths=paths, config=config)

    first_sampling = first["distribution"]["source_path_sampling"]
    second_sampling = second["distribution"]["source_path_sampling"]
    assert first_sampling == second_sampling
    assert first_sampling["available_source_path_count"] == 4
    assert first_sampling["selected_source_path_count"] == 2
    assert first_sampling["whole_path_scoring"] is True
    assert all(
        record["whole_path_frame_count"] == 100
        for record in first_sampling["selected_source_paths"]
    )
    selected_jobs = {
        record["source_job_id"] for record in first_sampling["selected_source_paths"]
    }
    assert {chunk["source_job_id"] for chunk in first["chunks"]} <= selected_jobs


def test_one_whole_path_can_contribute_multiple_interest_centers(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))

    selection = select_frame_interest_windows(
        [manifest],
        manifest_paths=["manifest.json"],
        config=FrameSelectionConfig(
            target_count=4,
            max_source_paths=1,
            max_targets_per_job=0,
            min_target_spacing_frames=8,
            preserve_mission_endpoints=False,
        ),
    )

    assert selection["selection_summary"]["selected_target_count"] == 4
    assert len({chunk["source_job_id"] for chunk in selection["chunks"]}) == 1
    assert selection["distribution"]["source_path_sampling"][
        "multiple_targets_per_source_path_allowed"
    ] is True


def test_empty_family_pilot_records_remote_source_wait_state(tmp_path: Path) -> None:
    output_root = tmp_path / "pilot"
    config = MissionFamilyPilotConfig(
        mission_families=tuple(ACTIVE_MASS_MISSION_FAMILIES),
        targets_per_family=1,
    )

    plan = prepare_mission_family_pilot([], output_root=output_root, config=config)

    assert plan["status"] == "waiting_for_sources"
    assert plan["execution_authorized"] is False
    assert plan["remote_sources_connected"] is False
    assert plan["summary"]["family_count"] == len(ACTIVE_MASS_MISSION_FAMILIES)
    assert plan["summary"]["status_counts"] == {
        "waiting_for_source_manifest": len(ACTIVE_MASS_MISSION_FAMILIES)
    }
    assert (output_root / "pilot_plan.json").is_file()


def test_family_pilot_prefers_dedicated_manifest_over_mixed_stream(tmp_path: Path) -> None:
    scene_ply = _scene(tmp_path)
    dedicated = _manifest(
        scene_ply,
        mission_family="deliver_to_human",
        scenario_id="scenario_deliver",
    )
    mixed = _manifest(
        scene_ply,
        mission_family="mission_stream",
        scenario_id="scenario_stream",
    )
    mixed["mission_families"] = ["mission_stream", "deliver_to_human"]
    mixed["jobs"][0]["mission_families"] = ["mission_stream", "deliver_to_human"]
    dedicated_path = tmp_path / "dedicated_render_manifest.json"
    mixed_path = tmp_path / "mixed_render_manifest.json"
    dedicated_path.write_text(json.dumps(dedicated), encoding="utf-8")
    mixed_path.write_text(json.dumps(mixed), encoding="utf-8")

    plan = prepare_mission_family_pilot(
        [mixed_path, dedicated_path],
        output_root=tmp_path / "pilot",
        config=MissionFamilyPilotConfig(
            mission_families=("deliver_to_human",),
            targets_per_family=1,
        ),
    )

    record = plan["families"][0]
    assert record["source_selection_policy"] == "dedicated_single_family"
    assert record["matched_source_manifests"] == [str(dedicated_path.resolve())]
    selection = json.loads(Path(record["selection_json"]).read_text(encoding="utf-8"))
    assert selection["chunks"][0]["source_manifest_path"] == str(dedicated_path.resolve())


@pytest.mark.parametrize(
    ("pilot_family", "law_id"),
    [
        ("navigate_with_social_constraints:personal_space", "L1_personal_space"),
        ("navigate_with_social_constraints:pedestrian_yield", "L2_non_obstruction_yield"),
        ("navigate_with_social_constraints:group_integrity", "L3_group_integrity"),
        ("navigate_with_social_constraints:queue_order", "L4_queue_order"),
    ],
)
def test_social_family_variant_selects_matching_law_manifest(
    tmp_path: Path,
    pilot_family: str,
    law_id: str,
) -> None:
    scene_ply = _scene(tmp_path)
    manifest = _manifest(
        scene_ply,
        mission_family="navigate_with_social_constraints",
        scenario_id=f"scenario_{law_id}",
    )
    manifest["social_law_ids"] = [law_id]
    manifest_path = tmp_path / f"{law_id}_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = prepare_mission_family_pilot(
        [manifest_path],
        output_root=tmp_path / "pilot",
        config=MissionFamilyPilotConfig(
            mission_families=(pilot_family,),
            targets_per_family=1,
        ),
    )

    record = plan["families"][0]
    assert record["status"] == "ready"
    selection = json.loads(Path(record["selection_json"]).read_text(encoding="utf-8"))
    assert selection["pilot_mission_family"] == pilot_family
    assert selection["chunks"][0]["pilot_mission_family"] == pilot_family


@pytest.mark.parametrize("mission_family", ACTIVE_MASS_MISSION_FAMILIES)
def test_family_pilot_prepares_one_poi_plus_endpoint_render_tests(
    tmp_path: Path,
    mission_family: str,
) -> None:
    scene_ply = _scene(tmp_path)
    manifest = _manifest(
        scene_ply,
        mission_family=mission_family,
        scenario_id=f"scenario_{mission_family}",
    )
    manifest_path = tmp_path / f"{mission_family}_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_root = tmp_path / "pilot"
    config = MissionFamilyPilotConfig(
        mission_families=(mission_family,),
        targets_per_family=1,
    )

    plan = prepare_mission_family_pilot(
        [manifest_path],
        output_root=output_root,
        config=config,
    )

    record = plan["families"][0]
    assert plan["status"] == "ready_to_plan"
    assert record["status"] == "ready"
    assert record["selected_interest_target_count"] == 1
    assert record["mandatory_anchor_count"] == 1
    assert record["selected_target_count"] == 2
    assert record["requested_window_frame_renders"] == 2 * (DEFAULT_PAST_FRAMES + 1)
    assert len(record["selected_render_manifests"]) == 1
    assert record["commands"][0]["execution_authorized"] is False
    assert "--execute" not in record["commands"][0]["plan_argv"]
    assert "--execute" in record["commands"][0]["execute_argv"]

    selection = json.loads(Path(record["selection_json"]).read_text(encoding="utf-8"))
    assert all(chunk["mission_families"] == [mission_family] for chunk in selection["chunks"])
    assert all(
        len(chunk["window"]["source_frame_indices"]) == DEFAULT_PAST_FRAMES + 1
        for chunk in selection["chunks"]
    )
