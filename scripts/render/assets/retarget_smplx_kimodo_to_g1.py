#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SMPLX_BODY_JOINTS = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


G1_JOINT_LIMITS = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.972222054, 1.972222054),
    "left_wrist_pitch_joint": (-1.614429558, 1.614429558),
    "left_wrist_yaw_joint": (-1.614429558, 1.614429558),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.972222054, 1.972222054),
    "right_wrist_pitch_joint": (-1.614429558, 1.614429558),
    "right_wrist_yaw_joint": (-1.614429558, 1.614429558),
}


def _natural_json_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (10**9, stem)


def load_smplx_frame_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.json"), key=_natural_json_key)
    else:
        paths = [input_path]
    if not paths:
        raise FileNotFoundError(f"No SMPL-X/Kimodo JSON frames found under {input_path}")
    return paths


def _load_frame(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    body_pose = payload.get("body_pose")
    if not isinstance(body_pose, list) or len(body_pose) < len(SMPLX_BODY_JOINTS):
        raise ValueError(f"{path} does not look like a Kimodo SMPL-X frame with 21 body joints")
    return payload


def _axis_angle(payload: Mapping[str, Any], joint_name: str) -> tuple[float, float, float]:
    idx = SMPLX_BODY_JOINTS.index(joint_name)
    raw = payload["body_pose"][idx]
    if not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"SMPL-X joint {joint_name} must be a 3-vector")
    return float(raw[0]), float(raw[1]), float(raw[2])


def _root_pose(payload: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = payload.get("root_pose", [0.0, 0.0, 0.0])
    if not isinstance(raw, Sequence) or len(raw) != 3:
        return 0.0, 0.0, 0.0
    return float(raw[0]), float(raw[1]), float(raw[2])


def _translation(payload: Mapping[str, Any]) -> list[float] | None:
    raw = payload.get("trans")
    if not isinstance(raw, Sequence) or len(raw) < 3:
        return None
    return [float(raw[0]), float(raw[1]), float(raw[2])]


def _clamp(name: str, value: float) -> float:
    lower, upper = G1_JOINT_LIMITS[name]
    return min(max(float(value), lower), upper)


def _soft_clip(value: float, limit: float) -> float:
    return min(max(float(value), -float(limit)), float(limit))


def _retarget_frame(payload: Mapping[str, Any], *, scale: float) -> dict[str, float]:
    left_hip = _axis_angle(payload, "left_hip")
    right_hip = _axis_angle(payload, "right_hip")
    left_knee = _axis_angle(payload, "left_knee")
    right_knee = _axis_angle(payload, "right_knee")
    left_ankle = _axis_angle(payload, "left_ankle")
    right_ankle = _axis_angle(payload, "right_ankle")
    spine3 = _axis_angle(payload, "spine3")
    root = _root_pose(payload)

    # This is a visual smoke-test retarget. Kimodo gives SMPL-X axis-angle
    # joints, while the renderer consumes G1 revolute joints. We keep the leg
    # swing visible but clamp to the URDF.
    left_knee_flex = max(0.0, left_knee[0]) * scale
    right_knee_flex = max(0.0, right_knee[0]) * scale
    left_arm_pitch = _soft_clip(-0.35 * right_hip[0] * scale, 0.5)
    right_arm_pitch = _soft_clip(-0.35 * left_hip[0] * scale, 0.5)
    joints = {
        "left_hip_pitch_joint": 0.75 * left_hip[0] * scale,
        "left_hip_roll_joint": 0.45 * left_hip[1] * scale,
        "left_hip_yaw_joint": 0.45 * left_hip[2] * scale,
        "left_knee_joint": left_knee_flex,
        "left_ankle_pitch_joint": (-0.45 * left_knee_flex + 0.25 * left_ankle[0]) * scale,
        "left_ankle_roll_joint": 0.35 * left_ankle[1] * scale,
        "right_hip_pitch_joint": 0.75 * right_hip[0] * scale,
        "right_hip_roll_joint": 0.45 * right_hip[1] * scale,
        "right_hip_yaw_joint": 0.45 * right_hip[2] * scale,
        "right_knee_joint": right_knee_flex,
        "right_ankle_pitch_joint": (-0.45 * right_knee_flex + 0.25 * right_ankle[0]) * scale,
        "right_ankle_roll_joint": 0.35 * right_ankle[1] * scale,
        "waist_yaw_joint": (0.15 * root[2] + 0.25 * spine3[2]) * scale,
        "left_shoulder_pitch_joint": left_arm_pitch,
        "left_shoulder_roll_joint": 0.18,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.35,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": right_arm_pitch,
        "right_shoulder_roll_joint": -0.18,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.35,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    return {name: _clamp(name, value) for name, value in joints.items()}


def retarget_smplx_frames(
    frame_paths: Iterable[Path],
    *,
    frame_count: int | None = None,
    stride: int = 1,
    cycle: bool = True,
    scale: float = 1.0,
) -> dict[str, Any]:
    paths = list(frame_paths)
    if not paths:
        raise ValueError("frame_paths must not be empty")
    if stride <= 0:
        raise ValueError("stride must be positive")
    source_frames = [_load_frame(path) for path in paths]
    output_count = int(frame_count) if frame_count is not None else math.ceil(len(source_frames) / stride)
    if output_count <= 0:
        raise ValueError("frame_count must be positive")

    frames: list[dict[str, Any]] = []
    for out_idx in range(output_count):
        source_idx = out_idx * int(stride)
        if cycle:
            source_idx %= len(source_frames)
        elif source_idx >= len(source_frames):
            break
        payload = source_frames[source_idx]
        entry: dict[str, Any] = {
            "frame": out_idx,
            "source_frame_index": source_idx,
            "source_path": str(paths[source_idx]),
            "joint_positions": _retarget_frame(payload, scale=float(scale)),
        }
        source_translation = _translation(payload)
        if source_translation is not None:
            entry["source_smplx_translation"] = source_translation
        frames.append(entry)

    return {
        "schema_version": "g1_amo_retarget.v1",
        "source_format": "kimodo_smplx_axis_angle",
        "retarget_policy": {
            "type": "heuristic_visual_probe",
            "smplx_body_joint_order": SMPLX_BODY_JOINTS,
            "scale": float(scale),
            "stride": int(stride),
            "cycle": bool(cycle),
            "notes": (
                "Kimodo frames are SMPL-X body axis-angle poses, not native G1 AMO. "
                "This mapping preserves visible walking leg motion for renderer validation, "
                "uses conservative G1-native arm swing to avoid SMPL-X shoulder-axis sign errors, "
                "and clamps values to g1_29dof_mode_16 URDF limits."
            ),
        },
        "joint_names": list(G1_JOINT_LIMITS),
        "frames": frames,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget Kimodo SMPL-X JSON frames to G1 joint poses for AMO overlay tests.")
    parser.add_argument("--input", type=Path, required=True, help="Kimodo SMPL-X JSON frame or directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output G1 AMO/joint pose JSON.")
    parser.add_argument("--frame-count", type=int, default=None, help="Number of output frames. Defaults to source length / stride.")
    parser.add_argument("--stride", type=int, default=1, help="Source frame stride.")
    parser.add_argument("--no-cycle", action="store_true", help="Do not cycle source frames when frame-count exceeds source length.")
    parser.add_argument("--scale", type=float, default=1.0, help="Joint amplitude scale before URDF-limit clamping.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = retarget_smplx_frames(
        load_smplx_frame_paths(args.input),
        frame_count=args.frame_count,
        stride=int(args.stride),
        cycle=not bool(args.no_cycle),
        scale=float(args.scale),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote {len(payload['frames'])} G1 AMO frame(s) to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
