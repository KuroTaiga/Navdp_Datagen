from __future__ import annotations

import json
from pathlib import Path

from scripts.render.assets.retarget_smplx_kimodo_to_g1 import (
    G1_JOINT_LIMITS,
    load_smplx_frame_paths,
    retarget_smplx_frames,
)


def _smplx_frame(*, left_knee: float, right_knee: float) -> dict:
    body_pose = [[0.0, 0.0, 0.0] for _ in range(21)]
    body_pose[3] = [left_knee, 0.0, 0.0]
    body_pose[4] = [right_knee, 0.0, 0.0]
    body_pose[15] = [0.2, -0.3, -0.4]
    body_pose[16] = [0.25, 0.35, 0.45]
    return {
        "root_pose": [0.0, 0.0, 0.0],
        "body_pose": body_pose,
        "trans": [0.0, 0.0, 1.0],
    }


def test_kimodo_smplx_retarget_writes_g1_joint_positions(tmp_path: Path) -> None:
    frame_dir = tmp_path / "walking_kimodo"
    frame_dir.mkdir()
    (frame_dir / "000000.json").write_text(json.dumps(_smplx_frame(left_knee=0.5, right_knee=0.1)), encoding="utf-8")
    (frame_dir / "000001.json").write_text(json.dumps(_smplx_frame(left_knee=0.2, right_knee=0.8)), encoding="utf-8")

    payload = retarget_smplx_frames(load_smplx_frame_paths(frame_dir), frame_count=3)

    assert payload["schema_version"] == "g1_amo_retarget.v1"
    assert payload["source_format"] == "kimodo_smplx_axis_angle"
    assert len(payload["frames"]) == 3
    assert payload["frames"][0]["source_frame_index"] == 0
    assert payload["frames"][2]["source_frame_index"] == 0
    joints = payload["frames"][1]["joint_positions"]
    assert joints["right_knee_joint"] == 0.8
    assert set(joints) == set(G1_JOINT_LIMITS)
    assert all(
        lower <= joints[name] <= upper
        for name, (lower, upper) in G1_JOINT_LIMITS.items()
    )


def test_kimodo_smplx_retarget_clamps_to_g1_limits(tmp_path: Path) -> None:
    frame_path = tmp_path / "000000.json"
    frame_path.write_text(json.dumps(_smplx_frame(left_knee=9.0, right_knee=-2.0)), encoding="utf-8")

    payload = retarget_smplx_frames(load_smplx_frame_paths(frame_path), frame_count=1)
    joints = payload["frames"][0]["joint_positions"]

    assert joints["left_knee_joint"] == G1_JOINT_LIMITS["left_knee_joint"][1]
    assert joints["right_knee_joint"] == 0.0
