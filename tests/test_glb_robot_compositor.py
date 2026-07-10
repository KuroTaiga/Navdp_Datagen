import math

import numpy as np

from utils.glb_robot_compositor import (
    camera_metadata_to_pyrender_pose,
    compose_rgba_over_rgb,
    decode_quantized_depth,
    parse_robot_poses,
    validate_pose_constraints,
)


def test_parse_robot_pose_position_and_yaw_deg():
    poses = parse_robot_poses(
        {"frames": [{"frame": 3, "position": [1.0, 2.0], "yaw_deg": 90.0}]},
        default_z=0.5,
        foot_offset=0.25,
    )

    pose = poses[3]
    np.testing.assert_allclose(pose.transform[:3, 3], [1.0, 2.0, 0.75])
    np.testing.assert_allclose(pose.transform[:2, :2], [[0.0, -1.0], [1.0, 0.0]], atol=1e-6)
    assert math.isclose(pose.yaw_rad, math.pi * 0.5)


def test_pose_constraint_report_flags_speed_and_yaw():
    poses = parse_robot_poses(
        [
            {"frame": 0, "position": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
            {"frame": 1, "position": [10.0, 0.0, 0.0], "yaw_deg": 180.0},
        ]
    )

    report = validate_pose_constraints(
        poses,
        fps=10.0,
        max_speed_mps=1.0,
        max_yaw_rate_radps=math.radians(90.0),
    )

    assert not report.ok
    assert any("speed" in item for item in report.violations)
    assert any("yaw rate" in item for item in report.violations)


def test_camera_metadata_to_pyrender_pose_transposes_and_flips_forward():
    camera_to_world_transposed = np.eye(4, dtype=np.float64)
    camera_to_world_transposed[3, :3] = [1.0, 2.0, 3.0]
    payload = {"camera_to_world": camera_to_world_transposed.tolist()}

    pose = camera_metadata_to_pyrender_pose(payload)

    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose[:3, 1], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(pose[:3, 2], [0.0, 0.0, -1.0])
    assert math.isclose(float(np.linalg.det(pose[:3, :3])), 1.0)


def test_compose_rgba_over_rgb_respects_depth_gate():
    base = np.full((1, 2, 3), 10, dtype=np.uint8)
    overlay = np.array([[[200, 0, 0, 255], [0, 200, 0, 255]]], dtype=np.uint8)
    mesh_depth = np.array([[2.0, 5.0]], dtype=np.float32)
    base_depth = np.array([[3.0, 4.0]], dtype=np.float32)

    composed = compose_rgba_over_rgb(
        base,
        overlay,
        mesh_depth_m=mesh_depth,
        base_depth_m=base_depth,
    )

    np.testing.assert_array_equal(composed[0, 0], [200, 0, 0])
    np.testing.assert_array_equal(composed[0, 1], [10, 10, 10])


def test_decode_quantized_depth_uses_renderer_step_sizes():
    raw = np.array([[1000]], dtype=np.uint16)
    decoded_16 = decode_quantized_depth(raw, bit_depth=16)
    decoded_8 = decode_quantized_depth(raw.astype(np.uint8), bit_depth=8)

    np.testing.assert_allclose(decoded_16, [[1.0]])
    np.testing.assert_allclose(decoded_8, [[232 * 0.04]])
