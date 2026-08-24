#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render.assets.retarget_smplx_kimodo_to_g1 import (  # noqa: E402
    load_smplx_frame_paths,
    retarget_smplx_frames,
)


def _natural_label_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), stem
    return 10**9, stem


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("[CMD] " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _find_scene(tasks_dir: Path, scene_prefix: str) -> str:
    matches = sorted(
        path.name
        for path in tasks_dir.iterdir()
        if path.is_dir() and path.name.startswith(scene_prefix)
    )
    if not matches:
        raise FileNotFoundError(f"No scene matching {scene_prefix!r} found under {tasks_dir}")
    return matches[0]


def _select_labels(tasks_dir: Path, scene_id: str, count: int) -> list[str]:
    scene_dir = tasks_dir / scene_id
    json_files = sorted(
        (
            path
            for path in scene_dir.glob("*.json")
            if path.name != "summary.json" and not path.name.endswith("_detailed.json")
        ),
        key=_natural_label_key,
    )
    labels = [path.stem for path in json_files[:count]]
    if len(labels) < count:
        raise RuntimeError(f"Only found {len(labels)} label JSONs for {scene_id}; requested {count}")
    return labels


def _load_scene_floor_z(scenes_dir: Path, scene_id: str) -> float:
    occupancy_path = scenes_dir / scene_id / "occupancy.json"
    with occupancy_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    lower = payload.get("lower")
    if isinstance(lower, list) and len(lower) >= 3:
        return float(lower[2])
    minimum = payload.get("min")
    if isinstance(minimum, list) and len(minimum) >= 3:
        return float(minimum[2])
    return 0.0


def _yaw_from_direction(dx: float, dy: float, *, forward_axis: str) -> float:
    norm = math.hypot(dx, dy)
    if norm < 1e-8:
        return 0.0
    dx /= norm
    dy /= norm
    axis = forward_axis.strip().lower()
    if axis in {"x", "+x"}:
        return math.atan2(dy, dx)
    if axis == "-x":
        return math.atan2(-dy, -dx)
    if axis in {"y", "+y"}:
        return math.atan2(-dx, dy)
    if axis == "-y":
        return math.atan2(dx, -dy)
    raise ValueError(f"Unsupported forward axis: {forward_axis!r}")


def _generate_robot_poses(
    *,
    follow_metadata_path: Path,
    output_path: Path,
    floor_z: float,
    foot_offset: float,
    yaw_offset_rad: float,
    forward_axis: str,
    follow_distance: float,
) -> int:
    with follow_metadata_path.open("r", encoding="utf-8") as fh:
        follow_payload = json.load(fh)
    frames = follow_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Missing frames in follow metadata: {follow_metadata_path}")

    person_xy: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        value = frame.get("person_world")
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(f"Frame {idx} missing person_world in {follow_metadata_path}")
        person_xy.append(np.asarray([float(value[0]), float(value[1])], dtype=np.float64))

    pose_frames: list[dict[str, Any]] = []
    last_yaw = yaw_offset_rad
    z = float(floor_z) + float(foot_offset)
    for idx, xy in enumerate(person_xy):
        if len(person_xy) == 1:
            direction = np.array([0.0, 1.0], dtype=np.float64)
        elif idx < len(person_xy) - 1:
            direction = person_xy[idx + 1] - xy
        else:
            direction = xy - person_xy[idx - 1]
        if float(np.linalg.norm(direction)) > 1e-8:
            last_yaw = (
                _yaw_from_direction(
                    float(direction[0]),
                    float(direction[1]),
                    forward_axis=forward_axis,
                )
                + yaw_offset_rad
            )
        pose_frames.append(
            {
                "frame": int(idx),
                "position": [float(xy[0]), float(xy[1]), z],
                "yaw_rad": float(last_yaw),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "source_follow_metadata": str(follow_metadata_path),
        "follow_distance": float(follow_distance),
        "floor_z": float(floor_z),
        "foot_offset": float(foot_offset),
        "yaw_offset_rad": float(yaw_offset_rad),
        "forward_axis": str(forward_axis),
        "frames": pose_frames,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return len(pose_frames)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small G1 robot follow-camera example from the first 0001_* "
            "scene in data/interiorGS_0500_42."
        )
    )
    parser.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "data" / "interiorGS_0500_42")
    parser.add_argument("--scenes-dir", type=Path, default=REPO_ROOT / "data" / "scenes")
    parser.add_argument("--scene-prefix", default="0001_", help="Scene prefix to select (default: 0001_).")
    parser.add_argument("--path-count", type=int, default=10, help="Number of label paths to render.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data2" / "g1_robot_follow_example")
    parser.add_argument("--urdf", type=Path, default=REPO_ROOT / "data" / "g1_description" / "g1_29dof_mode_16.urdf")
    parser.add_argument("--robot-glb", type=Path, default=REPO_ROOT / "assets" / "robots" / "g1_29dof_mode_16.glb")
    parser.add_argument(
        "--robot-glb-up-axis",
        choices=("y", "z"),
        default="z",
        help="Up axis of the robot GLB. Generated G1 URDF GLBs are Z-up (default: z).",
    )
    parser.add_argument(
        "--robot-forward-axis",
        choices=("x", "+x", "-x", "y", "+y", "-y"),
        default="x",
        help="Robot model forward axis before yaw. Unitree/G1 URDF convention is +X (default: x).",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(960, 720), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--follow-distance", type=float, default=1.5)
    parser.add_argument("--robot-foot-offset", type=float, default=0.0)
    parser.add_argument("--robot-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--kimodo-smplx-dir",
        type=Path,
        default=None,
        help="Optional Kimodo SMPL-X frame directory used to drive G1 AMO/joint animation.",
    )
    parser.add_argument("--kimodo-stride", type=int, default=1, help="Source SMPL-X frame stride for AMO retargeting.")
    parser.add_argument("--kimodo-scale", type=float, default=1.0, help="Retargeted AMO joint amplitude scale.")
    parser.add_argument("--compose-mode", choices=("foreground", "depth"), default="foreground")
    parser.add_argument("--depth-bit-depth", type=int, choices=(8, 10, 12, 16), default=16)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child render/conversion commands (default: current interpreter).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing base/overlay outputs (default: on). Use --no-overwrite to skip existing files where possible.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-render-arg",
        action="append",
        default=[],
        help="Additional single argument forwarded to render_label_paths.py; repeat as needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tasks_dir = args.tasks_dir.resolve()
    scenes_dir = args.scenes_dir.resolve()
    output_dir = args.output_dir.resolve()
    base_dir = output_dir / "base_gs"
    pose_dir = output_dir / "robot_poses"
    robot_dir = output_dir / "robot_overlay"
    python_exe = str(args.python)
    kimodo_frame_paths = None
    if args.kimodo_smplx_dir is not None:
        kimodo_frame_paths = load_smplx_frame_paths(args.kimodo_smplx_dir.resolve())

    scene_id = _find_scene(tasks_dir, str(args.scene_prefix))
    labels = _select_labels(tasks_dir, scene_id, int(args.path_count))
    floor_z = _load_scene_floor_z(scenes_dir, scene_id)

    if not args.robot_glb.is_file():
        _run(
            [
                python_exe,
                "scripts/render/assets/convert_urdf_visuals_to_glb.py",
                "--urdf",
                str(args.urdf),
                "--output",
                str(args.robot_glb),
            ],
            dry_run=bool(args.dry_run),
        )

    render_cmd = [
        python_exe,
        "render_label_paths.py",
        "--scenes-dir",
        str(scenes_dir),
        "--tasks-dir",
        str(tasks_dir),
        "--scene",
        scene_id,
        "--output-dir",
        str(base_dir),
        "--view-mode",
        "forward",
        "--follow-distance",
        str(float(args.follow_distance)),
        "--resolution",
        str(int(args.resolution[0])),
        str(int(args.resolution[1])),
        "--rgb-frames",
        "--save-depth-maps",
        "--save-camera-metadata",
        "--save-follow-metadata",
        "--limit-camera-to-follow",
        "--no-video",
        "--skip-summary",
    ]
    if args.overwrite:
        render_cmd.append("--overwrite")
    for label in labels:
        render_cmd.extend(["--label-id", label])
    render_cmd.extend(args.extra_render_arg)
    _run(render_cmd, dry_run=bool(args.dry_run))

    overlay_outputs: list[dict[str, Any]] = []
    yaw_offset_rad = math.radians(float(args.robot_yaw_offset_deg))
    for label in labels:
        follow_metadata_path = base_dir / scene_id / f"{label}_follow_path.json"
        camera_json = base_dir / scene_id / f"{label}_camera.json"
        frames_dir = base_dir / scene_id / label
        pose_json = pose_dir / scene_id / f"{label}_robot_poses.json"
        amo_json = pose_dir / scene_id / f"{label}_g1_amo_from_kimodo.json"
        out_frames = robot_dir / scene_id / label
        out_video = robot_dir / scene_id / f"{label}_g1_robot.mp4"

        if args.dry_run:
            pose_frame_count = 0
        elif pose_json.exists() and not args.overwrite:
            with pose_json.open("r", encoding="utf-8") as fh:
                pose_payload = json.load(fh)
            pose_frame_count = len(pose_payload.get("frames", []))
        else:
            pose_frame_count = _generate_robot_poses(
                follow_metadata_path=follow_metadata_path,
                output_path=pose_json,
                floor_z=floor_z,
                foot_offset=float(args.robot_foot_offset),
                yaw_offset_rad=yaw_offset_rad,
                forward_axis=str(args.robot_forward_axis),
                follow_distance=float(args.follow_distance),
            )

        use_kimodo_amo = kimodo_frame_paths is not None
        if use_kimodo_amo and not args.dry_run and (args.overwrite or not amo_json.exists()):
            amo_payload = retarget_smplx_frames(
                kimodo_frame_paths,
                frame_count=int(pose_frame_count),
                stride=int(args.kimodo_stride),
                scale=float(args.kimodo_scale),
            )
            amo_json.parent.mkdir(parents=True, exist_ok=True)
            amo_json.write_text(json.dumps(amo_payload, indent=2), encoding="utf-8")

        overlay_cmd = [
            python_exe,
            "scripts/render/assets/render_glb_robot_overlay.py",
            "--camera-json",
            str(camera_json),
            "--frames-dir",
            str(frames_dir),
            "--robot-glb",
            str(args.robot_glb),
            "--poses-json",
            str(pose_json),
            "--output-dir",
            str(out_frames),
            "--compose-mode",
            str(args.compose_mode),
            "--depth-bit-depth",
            str(int(args.depth_bit_depth)),
            "--glb-up-axis",
            str(args.robot_glb_up_axis),
        ]
        if use_kimodo_amo:
            overlay_cmd.extend(
                [
                    "--robot-urdf",
                    str(args.urdf),
                    "--robot-package-root",
                    str(args.urdf.parent),
                    "--amo-poses-json",
                    str(amo_json),
                ]
            )
        if args.compose_mode == "depth":
            overlay_cmd.extend(["--depth-dir", str(frames_dir)])
        if args.video:
            overlay_cmd.extend(["--video", str(out_video)])
        if args.overwrite:
            overlay_cmd.append("--overwrite")
        else:
            overlay_cmd.append("--no-overwrite")
        _run(overlay_cmd, dry_run=bool(args.dry_run))

        overlay_outputs.append(
            {
                "label": label,
                "pose_json": str(pose_json),
                "amo_json": str(amo_json) if use_kimodo_amo else None,
                "pose_frames": int(pose_frame_count),
                "frames_dir": str(out_frames),
                "video": str(out_video) if args.video else None,
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene": scene_id,
        "labels": labels,
        "tasks_dir": str(tasks_dir),
        "scenes_dir": str(scenes_dir),
        "base_render_dir": str(base_dir),
        "robot_glb": str(args.robot_glb),
        "robot_urdf": str(args.urdf),
        "robot_glb_up_axis": str(args.robot_glb_up_axis),
        "robot_forward_axis": str(args.robot_forward_axis),
        "kimodo_smplx_dir": str(args.kimodo_smplx_dir) if args.kimodo_smplx_dir is not None else None,
        "follow_distance": float(args.follow_distance),
        "compose_mode": str(args.compose_mode),
        "outputs": overlay_outputs,
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[DONE] G1 robot follow example manifest: {manifest_path}", flush=True)
    else:
        print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
