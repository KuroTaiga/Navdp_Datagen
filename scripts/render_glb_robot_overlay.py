#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.glb_robot_compositor import (
    GlbRobotRenderer,
    compose_rgba_over_rgb,
    decode_quantized_depth,
    parse_robot_joint_poses,
    parse_robot_poses,
    validate_pose_constraints,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_joint_names(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = _load_json(path)
    if isinstance(payload, list):
        names = payload
    else:
        names = (
            payload.get("joint_names")
            or payload.get("amo_joint_names")
            or payload.get("dof_names")
            or payload.get("qpos_names")
        )
    if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
        raise ValueError(f"{path} must contain a list of joint names")
    return list(names)


def _load_joint_positions(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object mapping joint names to values")
    return {str(name): float(value) for name, value in payload.items()}


def _frame_map(camera_payload: dict) -> dict[int, dict]:
    frames = camera_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("camera metadata must contain a 'frames' list")
    out: dict[int, dict] = {}
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"camera frame #{idx} must be an object")
        out[int(frame.get("frame", idx))] = frame
    return out


def _load_depth(path: Path | None, *, bit_depth: int) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    import imageio.v2 as imageio  # pylint: disable=import-outside-toplevel

    return decode_quantized_depth(imageio.imread(path), bit_depth=bit_depth)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay a GLB robot render onto existing Gaussian Splatting RGB frames."
    )
    parser.add_argument("--camera-json", type=Path, required=True, help="Path to <label>_camera.json.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing frame_XXXX.png files.")
    parser.add_argument("--robot-glb", type=Path, required=True, help="Robot GLB/GLTF asset to render.")
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=None,
        help=(
            "Optional URDF matching the GLB. Required for per-frame AMO/joint articulation because "
            "the default GLB is otherwise rendered as a static mesh."
        ),
    )
    parser.add_argument(
        "--robot-package-root",
        type=Path,
        default=None,
        help="Package root used to resolve package:// URDF resources for articulation.",
    )
    parser.add_argument(
        "--bind-joint-positions-json",
        type=Path,
        default=None,
        help=(
            "Joint pose used when the GLB was exported. Defaults to zero joints; pass the same mapping "
            "used by convert_urdf_visuals_to_glb.py --joint-positions-json if nonzero."
        ),
    )
    parser.add_argument(
        "--poses-json",
        type=Path,
        required=True,
        help=(
            "Per-frame robot poses. Accepts {'frames':[...]} or a raw list. Each entry may contain "
            "frame, position/translation/xyz, yaw_rad/yaw_deg/quaternion_wxyz, transform, and optional "
            "joint_positions/joints/amo_pose/qpos."
        ),
    )
    parser.add_argument(
        "--amo-poses-json",
        type=Path,
        default=None,
        help=(
            "Optional separate per-frame AMO/joint pose JSON. Frames are merged onto --poses-json by frame id."
        ),
    )
    parser.add_argument(
        "--joint-names-json",
        type=Path,
        default=None,
        help=(
            "Joint-name list for list-valued AMO poses. Not needed when AMO poses are dictionaries or the "
            "pose JSON contains joint_names/amo_joint_names/dof_names."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for composited PNG frames.")
    parser.add_argument("--frame-prefix", default="frame", help="Frame filename prefix (default: frame).")
    parser.add_argument("--image-ext", default="png", help="Input/output image extension (default: png).")
    parser.add_argument(
        "--compose-mode",
        choices=("foreground", "depth"),
        default="foreground",
        help="foreground always overlays visible robot pixels; depth respects saved GS depth maps.",
    )
    parser.add_argument("--depth-bit-depth", type=int, default=16, choices=(8, 10, 12, 16))
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Directory containing frame_XXXX_depth.png. Defaults to --frames-dir for depth mode.",
    )
    parser.add_argument("--depth-bias-m", type=float, default=0.02, help="Depth compare tolerance in meters.")
    parser.add_argument(
        "--output-rotation-k",
        type=int,
        default=2,
        help="Rotate GLB render by k*90 degrees before compositing. Existing GS perspective frames use 2.",
    )
    parser.add_argument("--target-height", type=float, default=None, help="Normalize robot mesh height in meters.")
    parser.add_argument("--glb-up-axis", choices=("y", "z"), default="y", help="Source GLB up axis.")
    parser.add_argument("--default-z", type=float, default=0.0, help="Pose Z used when poses are 2D.")
    parser.add_argument("--foot-offset", type=float, default=0.0, help="Vertical offset added to robot poses.")
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0, help="Additional yaw applied to all robot poses.")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS used for constraint validation.")
    parser.add_argument("--max-speed-mps", type=float, default=None, help="Optional max robot speed.")
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=None, help="Optional max yaw rate.")
    parser.add_argument(
        "--constraint-mode",
        choices=("warn", "error"),
        default="warn",
        help="How to handle max speed/yaw-rate violations.",
    )
    parser.add_argument(
        "--pyopengl-platform",
        default="egl",
        help="Headless OpenGL platform passed through PYOPENGL_PLATFORM (default: egl).",
    )
    parser.add_argument("--video", type=Path, default=None, help="Optional MP4 output path.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing output frames/video (default: on). Use --no-overwrite to skip existing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import imageio.v2 as imageio  # pylint: disable=import-outside-toplevel

    camera_payload = _load_json(args.camera_json)
    camera_frames = _frame_map(camera_payload)
    joint_names = _load_joint_names(args.joint_names_json)
    poses = parse_robot_poses(
        _load_json(args.poses_json),
        yaw_offset_rad=math.radians(float(args.yaw_offset_deg)),
        foot_offset=float(args.foot_offset),
        default_z=float(args.default_z),
        joint_names=joint_names,
    )
    if args.amo_poses_json is not None:
        amo_poses = parse_robot_joint_poses(_load_json(args.amo_poses_json), joint_names=joint_names)
        missing_pose_frames = sorted(frame for frame in amo_poses if frame not in poses)
        if missing_pose_frames:
            print(
                "[WARN] AMO poses without matching robot base poses were ignored: "
                + ", ".join(str(frame) for frame in missing_pose_frames[:10]),
                flush=True,
            )
        for frame_idx, joints in amo_poses.items():
            if frame_idx in poses:
                merged = dict(poses[frame_idx].joint_positions or {})
                merged.update(joints)
                poses[frame_idx] = replace(poses[frame_idx], joint_positions=merged)
    has_joint_poses = any(pose.joint_positions for pose in poses.values())
    if has_joint_poses and args.robot_urdf is None:
        print(
            "[WARN] AMO/joint poses were provided but --robot-urdf is missing; "
            "the GLB robot will render in its static mesh pose.",
            flush=True,
        )

    max_yaw_rate = (
        math.radians(float(args.max_yaw_rate_deg_s))
        if args.max_yaw_rate_deg_s is not None
        else None
    )
    report = validate_pose_constraints(
        poses,
        fps=float(args.fps),
        max_speed_mps=args.max_speed_mps,
        max_yaw_rate_radps=max_yaw_rate,
    )
    if not report.ok:
        message = "\n".join(report.violations)
        if args.constraint_mode == "error":
            raise SystemExit(f"[ERROR] Robot pose constraints failed:\n{message}")
        print(f"[WARN] Robot pose constraints failed:\n{message}", flush=True)

    if args.compose_mode == "depth":
        depth_dir = args.depth_dir or args.frames_dir
    else:
        depth_dir = None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_frame = camera_frames[min(camera_frames)]
    resolution = first_frame["resolution"]
    renderer = GlbRobotRenderer(
        args.robot_glb,
        width=int(resolution["width"]),
        height=int(resolution["height"]),
        target_height=args.target_height,
        up_axis=args.glb_up_axis,
        articulation_urdf_path=args.robot_urdf,
        articulation_package_root=args.robot_package_root,
        bind_joint_positions=_load_joint_positions(args.bind_joint_positions_json),
        pyopengl_platform=args.pyopengl_platform,
    )

    written: list[Path] = []
    rendered_count = 0
    skipped_count = 0
    try:
        for frame_idx in sorted(camera_frames):
            if frame_idx not in poses:
                continue
            rgb_path = args.frames_dir / f"{args.frame_prefix}_{frame_idx:04d}.{args.image_ext}"
            if not rgb_path.is_file():
                print(f"[WARN] Missing RGB frame: {rgb_path}", flush=True)
                continue
            out_path = args.output_dir / rgb_path.name
            if out_path.exists() and not args.overwrite:
                written.append(out_path)
                skipped_count += 1
                continue

            base_rgb = imageio.imread(rgb_path)
            if base_rgb.ndim == 2:
                base_rgb = np.repeat(base_rgb[..., None], 3, axis=2)
            if base_rgb.shape[2] > 3:
                base_rgb = base_rgb[..., :3]
            mesh = renderer.render(
                camera_frame=camera_frames[frame_idx],
                robot_transform=poses[frame_idx].transform,
                joint_positions=poses[frame_idx].joint_positions,
            )
            overlay = np.rot90(mesh.rgba, k=int(args.output_rotation_k))
            mesh_depth = np.rot90(mesh.depth_m, k=int(args.output_rotation_k))
            base_depth = None
            if depth_dir is not None:
                depth_path = depth_dir / f"{args.frame_prefix}_{frame_idx:04d}_depth.png"
                base_depth = _load_depth(depth_path, bit_depth=int(args.depth_bit_depth))

            composed = compose_rgba_over_rgb(
                base_rgb,
                overlay,
                mesh_depth_m=mesh_depth if args.compose_mode == "depth" else None,
                base_depth_m=base_depth if args.compose_mode == "depth" else None,
                depth_bias_m=float(args.depth_bias_m),
            )
            imageio.imwrite(out_path, composed)
            written.append(out_path)
            rendered_count += 1
    finally:
        renderer.close()

    if args.video is not None:
        if args.video.exists() and not args.overwrite:
            print(f"[SKIP] Video exists: {args.video}", flush=True)
        else:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(args.video, fps=float(args.fps)) as writer:
                for frame_path in written:
                    writer.append_data(imageio.imread(frame_path))

    print(
        f"[DONE] Rendered {rendered_count} frame(s), skipped {skipped_count} existing frame(s) "
        f"under {args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
