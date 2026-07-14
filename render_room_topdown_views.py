#!/usr/bin/env python3
"""Render masked top-down room views with per-room Gaussian cropping."""

from __future__ import annotations

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn

from arguments import PipelineParams
from gaussian_renderer import render_or
from render_room_center_views import (
    _json_safe,
    find_ply_file,
    iter_scene_dirs,
    load_room_specs,
)
from scene import GaussianModel
from utils.sh_utils import SH2RGB


class OrthoMiniCam:
    def __init__(
        self,
        width: int,
        height: int,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
        half_width: float,
        half_height: float,
        znear: float,
        zfar: float,
    ) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVy = 1.0
        self.FoVx = 1.0
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self._half_width = half_width
        self._half_height = half_height
        view_inv = torch.inverse(world_view_transform)
        self.camera_center = view_inv[3][:3]

    def get_full_proj_transform(self, orthographic: bool = False):
        if not orthographic:
            return self.full_proj_transform
        return self._half_width, self._half_height, self.full_proj_transform


def _clean_polygon_xy(points_xy: list[list[float]]) -> np.ndarray:
    polygon: list[list[float]] = []
    for point in points_xy:
        xy = [float(point[0]), float(point[1])]
        if not polygon or not np.allclose(polygon[-1], xy, atol=1e-5):
            polygon.append(xy)
    if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1], atol=1e-5):
        polygon.pop()

    deduped: list[list[float]] = []
    for xy in polygon:
        if not any(np.allclose(xy, existing, atol=1e-5) for existing in deduped):
            deduped.append(xy)
    if len(deduped) < 3:
        raise ValueError("Need at least 3 unique room polygon vertices for top-down masking.")
    return np.array(deduped, dtype=np.float32)


def _room_polygon_xy(room) -> tuple[np.ndarray, str]:
    structure_room = room.related_info.get("structure_room")
    if isinstance(structure_room, dict):
        for key in ("profile", "boundary"):
            if isinstance(structure_room.get(key), list):
                try:
                    return _clean_polygon_xy(structure_room[key]), f"structure_room.{key}"
                except ValueError:
                    pass
    return _clean_polygon_xy(room.points_xy), "labels.room_box"


def _points_in_polygon_torch(points_xy: torch.Tensor, polygon_xy: np.ndarray) -> torch.Tensor:
    poly = torch.tensor(polygon_xy, dtype=points_xy.dtype, device=points_xy.device)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = torch.zeros(points_xy.shape[0], dtype=torch.bool, device=points_xy.device)
    xj = poly[-1, 0]
    yj = poly[-1, 1]
    eps = torch.tensor(1e-12, dtype=points_xy.dtype, device=points_xy.device)
    for idx in range(poly.shape[0]):
        xi = poly[idx, 0]
        yi = poly[idx, 1]
        crosses = (yi > y) != (yj > y)
        x_at_y = (xj - xi) * (y - yi) / (yj - yi + eps) + xi
        inside ^= crosses & (x < x_at_y)
        xj = xi
        yj = yi
    return inside


def _points_in_polygon_numpy(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = np.zeros(points_xy.shape[0], dtype=bool)
    xj, yj = polygon_xy[-1]
    for xi, yi in polygon_xy:
        crosses = (yi > y) != (yj > y)
        x_at_y = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        inside ^= crosses & (x < x_at_y)
        xj, yj = xi, yi
    return inside


def _subset_gaussians(source: GaussianModel, keep_mask: torch.Tensor, sh_degree: int) -> GaussianModel:
    subset = GaussianModel(sh_degree=sh_degree)
    subset.active_sh_degree = min(source.active_sh_degree, subset.max_sh_degree)
    subset.spatial_lr_scale = source.spatial_lr_scale
    subset._xyz = nn.Parameter(source._xyz[keep_mask].detach().clone().requires_grad_(True))
    subset._features_dc = nn.Parameter(source._features_dc[keep_mask].detach().clone().requires_grad_(True))
    subset._features_rest = nn.Parameter(source._features_rest[keep_mask].detach().clone().requires_grad_(True))
    subset._opacity = nn.Parameter(source._opacity[keep_mask].detach().clone().requires_grad_(True))
    subset._scaling = nn.Parameter(source._scaling[keep_mask].detach().clone().requires_grad_(True))
    subset._rotation = nn.Parameter(source._rotation[keep_mask].detach().clone().requires_grad_(True))
    subset.max_radii2D = torch.zeros((subset.get_xyz.shape[0]), device=source.get_xyz.device)
    return subset


def _room_bounds(polygon_xy: np.ndarray, padding: float) -> tuple[float, float, float, float]:
    min_xy = polygon_xy.min(axis=0)
    max_xy = polygon_xy.max(axis=0)
    left = float(min_xy[0] - padding)
    right = float(max_xy[0] + padding)
    bottom = float(min_xy[1] - padding)
    top = float(max_xy[1] + padding)
    return left, right, bottom, top


def _build_topdown_camera(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    camera_z: float,
    floor_z: float,
    width: int,
    height: int,
    device: torch.device,
) -> OrthoMiniCam:
    cx = 0.5 * (left + right)
    cy = 0.5 * (bottom + top)
    half_width = 0.5 * (right - left)
    half_height = 0.5 * (top - bottom)
    znear = 0.01
    zfar = max(camera_z - floor_z + 2.0, 1.0)

    world_view_np = np.array(
        [
            [-1.0, 0.0, 0.0, cx],
            [0.0, -1.0, 0.0, cy],
            [0.0, 0.0, -1.0, camera_z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    left_cam = -half_width
    right_cam = half_width
    top_cam = -half_height
    bottom_cam = half_height
    projection_np = np.array(
        [
            [2.0 / (right_cam - left_cam), 0.0, 0.0, -(right_cam + left_cam) / (right_cam - left_cam)],
            [0.0, 2.0 / (top_cam - bottom_cam), 0.0, -(top_cam + bottom_cam) / (top_cam - bottom_cam)],
            [0.0, 0.0, -2.0 / (zfar - znear), -(zfar + znear) / (zfar - znear)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    world_view_transform = torch.tensor(world_view_np, device=device).transpose(0, 1)
    projection_matrix = torch.tensor(projection_np, device=device).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)).squeeze(0)
    return OrthoMiniCam(
        width=width,
        height=height,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        half_width=half_width,
        half_height=half_height,
        znear=znear,
        zfar=zfar,
    )


def _room_image_mask(
    polygon_xy: np.ndarray,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    width: int,
    height: int,
    rotate_k: int,
) -> np.ndarray:
    xs = left + (np.arange(width, dtype=np.float32) + 0.5) * ((right - left) / width)
    ys = top - (np.arange(height, dtype=np.float32) + 0.5) * ((top - bottom) / height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    mask = _points_in_polygon_numpy(points, polygon_xy).reshape(height, width)
    if rotate_k:
        mask = np.rot90(mask, k=rotate_k)
    return mask


def _apply_mask(rgb: np.ndarray, mask: np.ndarray, bg_color: list[int], alpha: bool) -> np.ndarray:
    if alpha:
        alpha_channel = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha_channel[..., None]], axis=-1)
        rgba[~mask, :3] = np.array(bg_color, dtype=np.uint8)
        return rgba
    out = rgb.copy()
    out[~mask] = np.array(bg_color, dtype=np.uint8)
    return out


def _render_points_bev(
    xyz: np.ndarray,
    rgb: np.ndarray,
    polygon_xy: np.ndarray,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    width: int,
    height: int,
    rotate_k: int,
    bg_color: list[int],
    point_size: int,
    alpha: bool,
) -> np.ndarray:
    if xyz.size == 0:
        canvas = np.full((height, width, 3), np.array(bg_color, dtype=np.uint8), dtype=np.uint8)
    else:
        canvas = np.full((height, width, 3), np.array(bg_color, dtype=np.uint8), dtype=np.uint8)
        u = np.floor((xyz[:, 0] - left) / max(right - left, 1e-8) * width).astype(np.int32)
        v = np.floor((top - xyz[:, 1]) / max(top - bottom, 1e-8) * height).astype(np.int32)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if np.any(valid):
            u = u[valid]
            v = v[valid]
            colors = rgb[valid]
            order = np.argsort(xyz[valid, 2])
            radius = max(int(point_size), 1) // 2
            for idx in order:
                uu = int(u[idx])
                vv = int(v[idx])
                color = colors[idx]
                y0 = max(vv - radius, 0)
                y1 = min(vv + radius + 1, height)
                x0 = max(uu - radius, 0)
                x1 = min(uu + radius + 1, width)
                canvas[y0:y1, x0:x1] = color

    if rotate_k:
        canvas = np.rot90(canvas, k=rotate_k)
    mask = _room_image_mask(
        polygon_xy,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        width=width,
        height=height,
        rotate_k=rotate_k,
    )
    return _apply_mask(canvas, mask, bg_color, alpha)


def render_scene_topdowns(
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
    total_points = int(gaussians.get_xyz.shape[0])
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    for room in rooms:
        output_path = scene_output_dir / f"{room.room_id}_topdown.png"
        points_output_path = scene_output_dir / f"{room.room_id}_topdown_points.png"
        metadata_path = scene_output_dir / f"{room.room_id}_topdown.json"
        polygon_xy, polygon_source = _room_polygon_xy(room)
        left, right, bottom, top = _room_bounds(polygon_xy, float(args.padding))
        floor_z = float(room.floor_z)
        camera_z = floor_z + float(args.camera_height)
        max_z = floor_z + float(args.cut_above) - float(args.ceiling_epsilon)

        xyz = gaussians.get_xyz
        inside_room = _points_in_polygon_torch(xyz[:, :2], polygon_xy)
        within_height = (xyz[:, 2] >= floor_z - float(args.below_floor_tolerance)) & (xyz[:, 2] <= max_z)
        keep_mask = inside_room & within_height
        kept_points = int(keep_mask.sum().item())
        print(
            f"  room {room.room_id}: kept {kept_points}/{total_points} points "
            f"(z <= {max_z:.3f}, camera_z={camera_z:.3f})",
            flush=True,
        )

        metadata = {
            "scene_id": scene_dir.name,
            "room_id": room.room_id,
            "source": room.source,
            "bev_plane": "xy",
            "vertical_axis": "z",
            "topdown_image": output_path.name,
            "camera_position_xyz": [0.5 * (left + right), 0.5 * (bottom + top), camera_z],
            "camera_height_above_floor": float(args.camera_height),
            "floor_z": floor_z,
            "point_filter": {
                "total_points": total_points,
                "kept_points": kept_points,
                "inside_room_polygon": int(inside_room.sum().item()),
                "within_height": int(within_height.sum().item()),
                "min_z": floor_z - float(args.below_floor_tolerance),
                "max_z": max_z,
                "requested_cut_above": float(args.cut_above),
                "ceiling_epsilon": float(args.ceiling_epsilon),
            },
            "orthographic_bounds_xy": {
                "left": left,
                "right": right,
                "bottom": bottom,
                "top": top,
                "padding": float(args.padding),
            },
            "room_polygon_xy": polygon_xy.astype(float).tolist(),
            "room_polygon_source": polygon_source,
            "render_settings": {
                "mode": args.mode,
                "width": int(args.width),
                "height": int(args.height),
                "rotate_k": int(args.rotate_k),
                "sh_degree": int(args.sh_degree),
                "antialiasing": bool(args.antialiasing),
                "alpha": bool(args.alpha),
                "mask_outside_room": True,
                "point_size": int(args.point_size),
            },
            "related_info": room.related_info,
        }

        if kept_points == 0:
            metadata["status"] = "skipped_no_points_after_filter"
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(_json_safe(metadata), fh, indent=2)
                fh.write("\n")
            skipped += 1
            continue

        primary_output_path = points_output_path if args.mode == "both" else output_path
        if primary_output_path.exists() and not args.overwrite:
            metadata["status"] = "skipped_existing"
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(_json_safe(metadata), fh, indent=2)
                fh.write("\n")
            skipped += 1
            continue

        bg_uint8 = [int(round(v * 255.0)) for v in args.bg_color]
        if args.mode in ("render", "both"):
            room_gaussians = _subset_gaussians(gaussians, keep_mask, int(args.sh_degree))
            camera = _build_topdown_camera(
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                camera_z=camera_z,
                floor_z=floor_z,
                width=int(args.width),
                height=int(args.height),
                device=device,
            )
            with torch.no_grad():
                img_pkg = render_or(
                    camera,
                    room_gaussians,
                    pipeline,
                    bg_color=bg_color,
                    orthographic=True,
                    antialiasing=pipeline.antialiasing,
                )
                image = img_pkg["render"].detach().cpu().numpy()
                rgb = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                if int(args.rotate_k) != 0:
                    rgb = np.rot90(rgb, k=int(args.rotate_k))
                mask = _room_image_mask(
                    polygon_xy,
                    left=left,
                    right=right,
                    bottom=bottom,
                    top=top,
                    width=int(args.width),
                    height=int(args.height),
                    rotate_k=int(args.rotate_k),
                )
                output = _apply_mask(rgb, mask, bg_uint8, bool(args.alpha))
                imageio.imwrite(output_path, output)
            metadata["topdown_image"] = output_path.name

        if args.mode in ("points", "both"):
            with torch.no_grad():
                kept_xyz = gaussians.get_xyz[keep_mask].detach().cpu().numpy()
                kept_rgb = SH2RGB(gaussians._features_dc[keep_mask, 0, :]).detach().cpu().numpy()
                kept_rgb = (np.clip(kept_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            point_output = _render_points_bev(
                kept_xyz,
                kept_rgb,
                polygon_xy,
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                width=int(args.width),
                height=int(args.height),
                rotate_k=int(args.rotate_k),
                bg_color=bg_uint8,
                point_size=int(args.point_size),
                alpha=bool(args.alpha),
            )
            if args.mode == "points":
                imageio.imwrite(output_path, point_output)
                metadata["topdown_image"] = output_path.name
            else:
                imageio.imwrite(points_output_path, point_output)
                metadata["topdown_points_image"] = points_output_path.name

        metadata["status"] = "rendered"
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(metadata), fh, indent=2)
            fh.write("\n")
        rendered += 1
    return rendered, skipped


def parse_args():
    parser = ArgumentParser(description="Render masked top-down room images.")
    parser.add_argument("--scenes-dir", type=Path, default=Path("./data/scenes"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ./data2/<scenes-dir-basename>_room_topdown",
    )
    parser.add_argument("--scene", action="append", default=None, help="Scene ID to render; may be repeated.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--camera-height", type=float, default=5.0)
    parser.add_argument("--cut-above", type=float, default=3.0)
    parser.add_argument(
        "--ceiling-epsilon",
        type=float,
        default=0.02,
        help="Subtract this from floor_z + cut_above so ceiling planes at the cutoff are removed.",
    )
    parser.add_argument("--below-floor-tolerance", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--rotate-k", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--mode", choices=["points", "render", "both"], default="points")
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--antialiasing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--alpha", action="store_true", help="Save transparent pixels outside the room mask.")
    parser.add_argument("--bg-color", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes_dir = args.scenes_dir
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: gaussian_renderer.render_or creates CUDA tensors.")
    if any(v < 0.0 or v > 1.0 for v in args.bg_color):
        raise ValueError("--bg-color values must be in [0, 1].")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("./data2") / f"{scenes_dir.name}_room_topdown"

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
    bg_color = torch.tensor(args.bg_color, dtype=torch.float32, device=device)
    print("[INFO] Top-down BEV plane is x-y; z is vertical.", flush=True)
    print(
        f"[INFO] camera_height={args.camera_height} cut_above={args.cut_above} "
        f"output_dir={output_dir}",
        flush=True,
    )

    total_rendered = 0
    total_skipped = 0
    failures = 0
    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{scene_idx}/{len(scene_dirs)}] {scene_dir.name}", flush=True)
        try:
            rendered, skipped = render_scene_topdowns(
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
        f"[DONE] rendered={total_rendered} skipped={total_skipped} failed_scenes={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
