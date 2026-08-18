from __future__ import annotations

import json

from scripts.smoke.run_g1_robot_follow_example import _base_outputs_complete, _generate_robot_poses


def test_g1_pose_generation_falls_back_to_camera_metadata(tmp_path) -> None:
    follow_path = tmp_path / "bad_follow.json"
    follow_path.write_text('{"frames": [', encoding="utf-8")
    camera_path = tmp_path / "camera.json"
    camera_path.write_text(
        json.dumps(
            {
                "frames": [
                    {"camera_center_world": [0.0, 0.0, 1.3]},
                    {"camera_center_world": [1.0, 0.0, 1.3]},
                ]
            }
        ),
        encoding="utf-8",
    )
    pose_path = tmp_path / "poses.json"

    count = _generate_robot_poses(
        follow_metadata_path=follow_path,
        camera_metadata_path=camera_path,
        output_path=pose_path,
        floor_z=0.0,
        foot_offset=0.0,
        yaw_offset_rad=0.0,
        forward_axis="x",
        follow_distance=1.5,
    )

    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    assert count == 2
    assert payload["frames"][0]["position"][:2] == [1.5, 0.0]
    assert payload["frames"][1]["position"][:2] == [2.5, 0.0]


def test_base_outputs_complete_requires_camera_rgb_and_depth(tmp_path) -> None:
    frames_dir = tmp_path / "base" / "scene_001" / "label_001"
    frames_dir.mkdir(parents=True)
    assert not _base_outputs_complete(tmp_path / "base", "scene_001", "label_001")

    (tmp_path / "base" / "scene_001" / "label_001_camera.json").write_text("{}", encoding="utf-8")
    (frames_dir / "frame_0000.png").write_bytes(b"png")
    assert not _base_outputs_complete(tmp_path / "base", "scene_001", "label_001")

    (frames_dir / "frame_0000_depth.png").write_bytes(b"png")
    assert _base_outputs_complete(tmp_path / "base", "scene_001", "label_001")
