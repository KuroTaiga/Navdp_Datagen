from __future__ import annotations

import json

import pytest

from navdp_datagen.sensors import (
    load_sensor_rig,
    pinhole_from_fov_y,
    sensor_profile_by_name,
)


def test_navdp_legacy_intrinsics_match_documented_defaults() -> None:
    intrinsics = pinhole_from_fov_y(960, 720, 70.0)

    assert intrinsics["width"] == 960
    assert intrinsics["height"] == 720
    assert intrinsics["fx"] == pytest.approx(514.133, rel=1e-5)
    assert intrinsics["fy"] == pytest.approx(514.133, rel=1e-5)
    assert intrinsics["cx"] == 480
    assert intrinsics["cy"] == 360
    assert intrinsics["fov_x_deg"] == pytest.approx(86.067, rel=1e-4)


def test_g1_profile_is_provisional_and_selectable() -> None:
    profile = sensor_profile_by_name("g1_head_fpv_default")
    sensor = profile["sensors"][0]

    assert profile["source"]["provisional"] is True
    assert sensor["name"] == "head_rgbd"
    assert sensor["transform"]["translation_m"] == [0.1, 0.0, 1.2]
    assert sensor["transform"]["rotation_rpy_deg"] == [0.0, -5.0, 0.0]
    assert sensor["clipping_range_m"] == [0.05, 30.0]


def test_imported_sensor_rig_json_normalizes_isaacsim_style_camera(tmp_path) -> None:
    rig_path = tmp_path / "rig.json"
    rig_path.write_text(
        json.dumps(
            {
                "robot_id": "robot_alpha",
                "format": "isaacsim_export_json",
                "cameras": [
                    {
                        "camera_name": "front_rgb",
                        "camera_prim_path": "/World/G1/head/front_rgb",
                        "local_position": [0.1, 0.0, 1.2],
                        "local_rotation_rpy_deg": [0.0, -5.0, 0.0],
                        "resolution": [640, 480],
                        "fov_y_deg": 70.0,
                        "clipping_range": [0.05, 30.0],
                        "modalities": ["rgb", "depth"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rig = load_sensor_rig(rig_path)

    assert rig["rig_id"] == "robot_alpha"
    assert rig["source"]["format"] == "isaacsim_export_json"
    assert rig["sensors"][0]["name"] == "front_rgb"
    assert rig["sensors"][0]["prim_path"] == "/World/G1/head/front_rgb"
    assert rig["sensors"][0]["intrinsics"]["width"] == 640
    assert rig["sensors"][0]["intrinsics"]["height"] == 480
    assert rig["sensors"][0]["modalities"] == ["rgb", "depth"]
