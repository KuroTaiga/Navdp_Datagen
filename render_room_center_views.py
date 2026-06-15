#!/usr/bin/env python3
"""Render four 90-degree room-center views for each scene."""

from __future__ import annotations

import json
import math
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch

from arguments import PipelineParams
from gaussian_renderer import render
from scene import GaussianModel
from utils.render_utils import build_perspective_camera


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    center_xy: np.ndarray
    floor_z: float
    source: str
    points_xy: list[list[float]]
    z_range: list[float] | None
    related_info: dict


def _natural_key(value: str) -> tuple[int, object]:
    if re.fullmatch(r"-?\d+", value):
        return 0, int(value)
    return 1, value


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _points_from_room_box(points: object) -> tuple[np.ndarray, np.ndarray]:
    xy: list[tuple[float, float]] = []
    z_values: list[float] = []
    if not isinstance(points, list):
        raise ValueError("room_box entry is not a list")
    for point in points:
        if not isinstance(point, dict):
            continue
        if "x" not in point or "y" not in point:
            continue
        xy.append((float(point["x"]), float(point["y"])))
        if "z" in point:
            z_values.append(float(point["z"]))
    if not xy:
        raise ValueError("room_box entry has no x/y points")
    return np.array(xy, dtype=np.float32), np.array(z_values, dtype=np.float32)


def _bbox_center(xy: np.ndarray) -> np.ndarray:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    return ((mins + maxs) * 0.5).astype(np.float32)


def _polygon_centroid(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] < 3:
        return xy.mean(axis=0).astype(np.float32)
    x = xy[:, 0]
    y = xy[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area2 = float(cross.sum())
    if abs(area2) < 1e-6:
        return xy.mean(axis=0).astype(np.float32)
    cx = float(((x + x_next) * cross).sum() / (3.0 * area2))
    cy = float(((y + y_next) * cross).sum() / (3.0 * area2))
    return np.array([cx, cy], dtype=np.float32)


def _structure_room_points(room: dict) -> tuple[list, str] | tuple[None, None]:
    if isinstance(room.get("profile"), list):
        return room["profile"], "profile"
    if isinstance(room.get("boundary"), list):
        return room["boundary"], "boundary"
    return None, None


def _floor_z_from_occupancy(scene_dir: Path) -> float:
    occ_path = scene_dir / "occupancy.json"
    if not occ_path.is_file():
        return 0.0
    occ = _load_json(occ_path)
    if not isinstance(occ, dict):
        return 0.0
    for key in ("lower", "min"):
        values = occ.get(key)
        if isinstance(values, list) and len(values) >= 3:
            return float(values[2])
    return 0.0


def load_room_specs(scene_dir: Path) -> list[RoomSpec]:
    structure_rooms: list[dict] = []
    structure_path = scene_dir / "structure.json"
    if structure_path.is_file():
        structure = _load_json(structure_path)
        if isinstance(structure, dict) and isinstance(structure.get("rooms"), list):
            structure_rooms = [room for room in structure["rooms"] if isinstance(room, dict)]

    labels_path = scene_dir / "labels.json"
    if labels_path.is_file():
        labels = _load_json(labels_path)
        rooms_by_id: dict[str, RoomSpec] = {}
        if isinstance(labels, list):
            for item in labels:
                if not isinstance(item, dict):
                    continue
                if str(item.get("label", "")).lower() != "room":
                    continue
                room_box = item.get("room_box")
                if not isinstance(room_box, dict):
                    continue
                for room_id, points in room_box.items():
                    xy, z_values = _points_from_room_box(points)
                    floor_z = float(z_values.min()) if z_values.size else _floor_z_from_occupancy(scene_dir)
                    room_id_str = str(room_id)
                    related_info = {
                        "room_box": points,
                    }
                    if re.fullmatch(r"\d+", room_id_str):
                        structure_idx = int(room_id_str) - 1
                        if 0 <= structure_idx < len(structure_rooms):
                            related_info["structure_room"] = structure_rooms[structure_idx]
                    rooms_by_id[str(room_id)] = RoomSpec(
                        room_id=room_id_str,
                        center_xy=_bbox_center(xy),
                        floor_z=floor_z,
                        source="labels.json:room_box",
                        points_xy=xy.astype(float).tolist(),
                        z_range=[float(z_values.min()), float(z_values.max())] if z_values.size else None,
                        related_info=_json_safe(related_info),
                    )
        if rooms_by_id:
            return [rooms_by_id[key] for key in sorted(rooms_by_id, key=_natural_key)]

    if structure_path.is_file():
        floor_z = _floor_z_from_occupancy(scene_dir)
        specs: list[RoomSpec] = []
        for idx, room in enumerate(structure_rooms, start=1):
            points, points_key = _structure_room_points(room)
            if points is None:
                continue
            xy = np.array(points, dtype=np.float32)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            related_info = {"structure_room": _json_safe(room), "structure_room_points_key": points_key}
            specs.append(
                RoomSpec(
                    room_id=str(idx),
                    center_xy=_polygon_centroid(xy[:, :2]),
                    floor_z=floor_z,
                    source="structure.json:rooms",
                    points_xy=xy[:, :2].astype(float).tolist(),
                    z_range=None,
                    related_info=related_info,
                )
            )
        if specs:
            return specs

    raise FileNotFoundError(f"No room metadata found in {scene_dir}")


def find_ply_file(scene_dir: Path) -> Path:
    for name in ("3dgs_raw.ply", "3dgs_compressed.ply"):
        preferred = scene_dir / name
        if preferred.is_file():
            return preferred
    candidates = sorted(scene_dir.glob("*.ply"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .ply file found in {scene_dir}")


def iter_scene_dirs(scenes_dir: Path, selected_scenes: set[str] | None) -> Iterable[Path]:
    for scene_dir in sorted(path for path in scenes_dir.iterdir() if path.is_dir()):
        if selected_scenes is not None and scene_dir.name not in selected_scenes:
            continue
        yield scene_dir


def _room_bounds_xy(points_xy: list[list[float]]) -> dict:
    xy = np.array(points_xy, dtype=np.float32)
    return {
        "min": [float(xy[:, 0].min()), float(xy[:, 1].min())],
        "max": [float(xy[:, 0].max()), float(xy[:, 1].max())],
    }


def write_room_reference_json(
    scene_dir: Path,
    scene_output_dir: Path,
    room: RoomSpec,
    position: np.ndarray,
    view_records: list[dict],
    args,
) -> None:
    payload = {
        "scene_id": scene_dir.name,
        "room_id": room.room_id,
        "source": room.source,
        "bev_plane": "xy",
        "vertical_axis": "z",
        "center_xy": [float(room.center_xy[0]), float(room.center_xy[1])],
        "floor_z": float(room.floor_z),
        "camera_height_above_floor": float(args.camera_height),
        "camera_position_xyz": [float(position[0]), float(position[1]), float(position[2])],
        "room_bounds_xy": _room_bounds_xy(room.points_xy),
        "room_points_xy": room.points_xy,
        "room_z_range": room.z_range,
        "render_settings": {
            "width": int(args.width),
            "height": int(args.height),
            "fov_deg": float(args.fov_deg),
            "znear": float(args.znear),
            "zfar": float(args.zfar),
            "look_down": float(args.look_down),
            "start_yaw_deg": float(args.start_yaw_deg),
            "rotate_k": int(args.rotate_k),
            "sh_degree": int(args.sh_degree),
            "antialiasing": bool(args.antialiasing),
            "white_background": bool(args.white_background),
        },
        "views": view_records,
        "related_info": room.related_info,
    }
    reference_path = scene_output_dir / f"{room.room_id}.json"
    with reference_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2)
        fh.write("\n")


def render_room_views(
    scene_dir: Path,
    scene_output_dir: Path,
    pipeline: PipelineParams,
    device: torch.device,
    bg_color: torch.Tensor,
    args,
) -> tuple[int, int]:
    rooms = load_room_specs(scene_dir)
    ply_path = find_ply_file(scene_dir)

    print(f"[SCENE] {scene_dir.name}: loading {ply_path.name} ({len(rooms)} rooms)", flush=True)
    gaussians = GaussianModel(sh_degree=int(args.sh_degree))
    gaussians.load_ply(str(ply_path))

    scene_output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    skipped = 0
    for room in rooms:
        position = np.array(
            [
                float(room.center_xy[0]),
                float(room.center_xy[1]),
                float(room.floor_z) + float(args.camera_height),
            ],
            dtype=np.float32,
        )
        print(
            f"  room {room.room_id}: center_xy=({position[0]:.3f}, {position[1]:.3f}) "
            f"z={position[2]:.3f} source={room.source}",
            flush=True,
        )
        view_records: list[dict] = []
        for idx in range(4):
            output_path = scene_output_dir / f"{room.room_id}_{idx:02d}.png"

            yaw_rad = math.radians(float(args.start_yaw_deg) + idx * 90.0)
            forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)
            target = position + forward
            target[2] = position[2] - float(args.look_down)
            view_records.append(
                {
                    "index": idx,
                    "yaw_deg": float(args.start_yaw_deg) + idx * 90.0,
                    "image": output_path.name,
                    "target_xyz": [float(target[0]), float(target[1]), float(target[2])],
                    "forward_xyz": [float(forward[0]), float(forward[1]), float(forward[2])],
                }
            )

            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue

            camera = build_perspective_camera(
                position=position,
                target=target,
                width=int(args.width),
                height=int(args.height),
                fov_deg=float(args.fov_deg),
                znear=float(args.znear),
                zfar=float(args.zfar),
                device=device,
            )

            with torch.no_grad():
                img_pkg = render(camera, gaussians, pipeline, bg_color=bg_color)
                image = img_pkg["render"].detach().cpu().numpy()
                rgb = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                if int(args.rotate_k) != 0:
                    rgb = np.rot90(rgb, k=int(args.rotate_k))
                imageio.imwrite(output_path, rgb)
            rendered += 1
        write_room_reference_json(
            scene_dir=scene_dir,
            scene_output_dir=scene_output_dir,
            room=room,
            position=position,
            view_records=view_records,
            args=args,
        )
    return rendered, skipped


def parse_args():
    parser = ArgumentParser(description="Render 4 horizontal RGB images from each room center.")
    parser.add_argument("--scenes-dir", type=Path, default=Path("./data/scenes"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ./data2/<scenes-dir-basename>_room_img",
    )
    parser.add_argument("--scene", action="append", default=None, help="Scene ID to render; may be repeated.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--znear", type=float, default=0.001)
    parser.add_argument("--zfar", type=float, default=30.0)
    parser.add_argument("--look-down", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--rotate-k",
        type=int,
        default=2,
        help="np.rot90 count applied before saving; 2 matches render_first_frame.py.",
    )
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--antialiasing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    bg_group = parser.add_mutually_exclusive_group()
    bg_group.add_argument("--white-background", dest="white_background", action="store_true", default=True)
    bg_group.add_argument("--black-background", dest="white_background", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes_dir = args.scenes_dir
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: gaussian_renderer.render creates CUDA tensors.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("./data2") / f"{scenes_dir.name}_room_img"

    selected_scenes = set(args.scene) if args.scene else None
    scene_dirs = list(iter_scene_dirs(scenes_dir, selected_scenes))
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[: int(args.max_scenes)]
    if not scene_dirs:
        print(f"No scene directories selected under {scenes_dir}.", flush=True)
        return

    pipeline_parser = ArgumentParser(description="Pipeline parameters")
    pipeline = PipelineParams(pipeline_parser)
    pipeline.antialiasing = bool(args.antialiasing)

    device = torch.device("cuda")
    bg = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    bg_color = torch.tensor(bg, dtype=torch.float32, device=device)

    total_rendered = 0
    total_skipped = 0
    failures = 0
    print(f"[INFO] BEV/floor plane is x-y; z is vertical.", flush=True)
    print(f"[INFO] scenes_dir={scenes_dir} output_dir={output_dir}", flush=True)
    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{scene_idx}/{len(scene_dirs)}] {scene_dir.name}", flush=True)
        try:
            rendered, skipped = render_room_views(
                scene_dir=scene_dir,
                scene_output_dir=output_dir / scene_dir.name,
                pipeline=pipeline,
                device=device,
                bg_color=bg_color,
                args=args,
            )
            total_rendered += rendered
            total_skipped += skipped
        except Exception as exc:  # pylint: disable=broad-except
            failures += 1
            print(f"  [ERROR] {scene_dir.name}: {exc}", flush=True)

    print(
        f"[DONE] rendered={total_rendered} skipped_existing={total_skipped} failed_scenes={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
