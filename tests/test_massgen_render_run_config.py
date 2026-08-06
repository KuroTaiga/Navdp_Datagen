from __future__ import annotations

import json

from navdp_datagen.massgen.run_config import prepare_render_run


def _pose(x: float, y: float, yaw: float = 0.0) -> dict[str, float]:
    return {"x": x, "y": y, "yaw": yaw}


def _robot(robot_id: str, x: float) -> dict[str, object]:
    return {
        "robot_id": robot_id,
        "robot_type": "ground",
        "start_map_pose": _pose(x, 0.0),
        "trajectory": [
            {"t": 0.0, "map_pose": _pose(x, 0.0), "motion_state": "idle"},
            {"t": 1.0, "map_pose": _pose(x + 0.5, 0.5, 0.2), "motion_state": "moving"},
        ],
    }


def _scenario(scene_asset_name: str) -> dict[str, object]:
    return {
        "scenario_id": "scenario_dense_multi_robot",
        "scene_id": "scene_001",
        "schema_version": "0.1",
        "scene_assets": {"splat_model_path": scene_asset_name},
        "robots": [_robot("robot_alpha", 0.0), _robot("robot_beta", 1.0)],
        "humans": [],
        "missions": [
            {
                "mission_id": "mission_001",
                "mission_type": "dense_multi_robot",
                "release_time": 0.0,
                "deadline": 1.0,
                "metadata": {"training_robot_ids": ["robot_alpha", "robot_beta"]},
            }
        ],
    }


def test_prepare_render_run_attaches_default_sensor_profile_and_writes_outputs(tmp_path) -> None:
    scene_asset = tmp_path / "scene.ply"
    scene_asset.write_text("ply\n", encoding="utf-8")
    scenario_json = tmp_path / "scenario.json"
    scenario_json.write_text(json.dumps(_scenario(scene_asset.name)), encoding="utf-8")
    output_root = tmp_path / "out"
    manifest_json = output_root / "render_manifest.json"
    summary_json = output_root / "summary.json"

    result = prepare_render_run(
        {
            "scenario_json": str(scenario_json),
            "output_root": str(output_root),
            "manifest_json": str(manifest_json),
            "summary_json": str(summary_json),
            "sensor_profile": "g1_head_fpv_default",
            "selected_sensors": ["head_rgbd"],
            "strict_assets": True,
        },
        config_path=tmp_path / "config.json",
        write_outputs=True,
    )

    manifest = result["manifest"]
    summary = result["summary"]

    assert manifest_json.is_file()
    assert summary["status"] == "ready"
    assert summary["job_count"] == 2
    assert summary["sensor_rig_id"] == "g1_head_fpv_default"
    assert summary["selected_sensors"] == ["head_rgbd"]
    assert "g1_head_fpv_default" in manifest["sensor_rigs"]
    assert manifest["jobs"][0]["sensors"][0]["sensor_name"] == "head_rgbd"
    assert any("provisional fallback profile" in warning for warning in summary["warnings"])


def test_prepare_render_run_blocks_when_strict_assets_missing(tmp_path) -> None:
    scenario_json = tmp_path / "scenario.json"
    scenario_json.write_text(json.dumps(_scenario("missing_scene.ply")), encoding="utf-8")

    result = prepare_render_run(
        {
            "scenario_json": str(scenario_json),
            "output_root": str(tmp_path / "out"),
            "sensor_profile": "navdp_legacy_fpv",
            "strict_assets": True,
        },
        config_path=tmp_path / "config.json",
    )

    summary = result["summary"]

    assert summary["status"] == "blocked"
    assert any("scene_assets.splat_model_path not found" in error for error in summary["preflight"]["errors"])


def test_prepare_render_run_uses_imported_sensor_rig(tmp_path) -> None:
    scene_asset = tmp_path / "scene.ply"
    scene_asset.write_text("ply\n", encoding="utf-8")
    scenario_json = tmp_path / "scenario.json"
    scenario_json.write_text(json.dumps(_scenario(scene_asset.name)), encoding="utf-8")
    rig_json = tmp_path / "rig.json"
    rig_json.write_text(
        json.dumps(
            {
                "rig_id": "custom_rig",
                "sensors": [
                    {
                        "name": "front_rgb",
                        "resolution": [800, 600],
                        "fov_y_deg": 65.0,
                        "modalities": ["rgb"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_render_run(
        {
            "scenario_json": str(scenario_json),
            "output_root": str(tmp_path / "out"),
            "sensor_rig_json": str(rig_json),
            "selected_sensors": ["front_rgb"],
            "strict_assets": True,
        },
        config_path=tmp_path / "config.json",
    )

    manifest = result["manifest"]

    assert result["summary"]["status"] == "ready"
    assert result["summary"]["sensor_rig_id"] == "custom_rig"
    assert manifest["jobs"][0]["sensors"] == [
        {
            "rig_id": "custom_rig",
            "sensor_name": "front_rgb",
            "type": "camera",
            "modalities": ["rgb"],
            "profile": "imported",
        }
    ]
