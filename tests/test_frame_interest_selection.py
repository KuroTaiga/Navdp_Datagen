from __future__ import annotations

import json
from pathlib import Path

from navdp_datagen.massgen.frame_selection import (
    DEFAULT_FUTURE_FRAMES,
    DEFAULT_PAST_FRAMES,
    FrameSelectionConfig,
    apply_frame_selection_to_manifest,
    select_frame_interest_windows,
)
from navdp_datagen.massgen.render_executor import build_render_plans


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


def _manifest(scene_ply: Path) -> dict[str, object]:
    trajectory = _trajectory()
    return {
        "schema_version": "massgen_render_manifest/v0.1",
        "source": {
            "kind": "pathplanner_scenario",
            "scenario_id": "scenario_001",
            "scene_id": "scene_001",
        },
        "render_backend": "gsplat",
        "mission_families": ["dense_dynamic_avoidance"],
        "social_law_ids": ["pedestrian_yield"],
        "scene_assets": {"splat_model_path": str(scene_ply)},
        "timing": {"fps": 10.0, "start_time_s": 0.0, "end_time_s": 9.9, "frame_count": 100},
        "actors": {"humans": [], "robots": []},
        "missions": [
            {
                "mission_id": "mission_001",
                "mission_type": "dense_dynamic_avoidance",
                "release_time_s": 1.0,
                "deadline_s": 8.0,
            }
        ],
        "events": [{"event_id": "evt_yield", "event_type": "yield", "t": 5.0}],
        "jobs": [
            {
                "job_id": "scenario_001__view_robot_alpha",
                "scene_id": "scene_001",
                "viewpoint_robot_id": "robot_alpha",
                "mission_families": ["dense_dynamic_avoidance"],
                "assigned_mission_ids": ["mission_001"],
                "camera": {"mode": "robot_fpv", "source_actor_id": "robot_alpha", "trajectory": trajectory},
                "human_actor_ids": [],
                "peer_robot_ids": [],
                "peer_robot_pose_tracks": [],
                "render_options": {"backend": "gsplat"},
                "outputs": {"stem": "scenario_001__view_robot_alpha"},
            }
        ],
        "warnings": [],
    }


def test_frame_interest_selection_emits_65_frame_windows(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))
    config = FrameSelectionConfig(target_count=4, seed=1234)

    selection = select_frame_interest_windows([manifest], manifest_paths=["manifest.json"], config=config)

    assert selection["schema_version"] == "navdp_frame_interest_selection/v0.1"
    assert selection["selection_summary"]["selected_target_count"] == 4
    assert selection["selection_summary"]["window_frame_count"] == 65
    assert selection["selection_summary"]["requested_window_frame_renders"] == 4 * 65
    for chunk in selection["chunks"]:
        frames = chunk["window"]["source_frame_indices"]
        assert len(frames) == DEFAULT_PAST_FRAMES + 1 + DEFAULT_FUTURE_FRAMES
        assert chunk["target_frame"] in frames
        assert chunk["render_contract"]["preserve_stationary_frames"] is True


def test_apply_frame_selection_rewrites_jobs_to_selected_windows(tmp_path: Path) -> None:
    manifest = _manifest(_scene(tmp_path))
    config = FrameSelectionConfig(target_count=2, seed=7)
    selection = select_frame_interest_windows([manifest], manifest_paths=["manifest.json"], config=config)

    selected_manifest = apply_frame_selection_to_manifest(manifest, selection, manifest_path="manifest.json")

    assert selected_manifest["timing"]["selection_mode"] == "frame_interest_windows"
    assert len(selected_manifest["jobs"]) == 2
    for job in selected_manifest["jobs"]:
        assert job["job_id"].startswith("scenario_001__view_robot_alpha__m000__foi_")
        assert len(job["camera"]["trajectory"]) == 65
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
    assert len(label["path"]["raster_world"]) == 65
    assert sum(1 for point in label["path"]["raster_world"] if point[0] == 4.7) == 7
