#!/usr/bin/env python3
"""Render one Gaussian-Splatting view from a generated camera JSON.

This is a small command-line wrapper around the GraphDECO-style renderer used by
the local gaussian_splatting project.  It is intended to be called from
render_single_item_ply_augmentations.py with --renderer-backend external_gs.
"""

from __future__ import annotations

import json
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import imageio
import numpy as np
import torch

from arguments import PipelineParams
from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix


def parse_rgb(value: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--background must contain 3 comma-separated values, e.g. 1,1,1")
    rgb = tuple(float(p) for p in parts)
    return tuple(max(0.0, min(1.0, v)) for v in rgb)  # type: ignore[return-value]


def focal_to_fov(focal: float, pixels: int) -> float:
    if focal <= 0 or pixels <= 0:
        raise ValueError(f"Invalid focal/pixels: focal={focal}, pixels={pixels}")
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def load_camera_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "world_to_camera" not in data:
        raise KeyError(f"Camera JSON missing required field 'world_to_camera': {path}")
    return data


def get_nested(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_camera_size(data: Dict[str, Any], width_arg: int, height_arg: int) -> Tuple[int, int]:
    width = int(width_arg or get_nested(data, ("resolution", "width"), 0) or data.get("width", 0) or 0)
    height = int(height_arg or get_nested(data, ("resolution", "height"), 0) or data.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Camera width/height are missing. Pass --width and --height or include resolution in JSON.")
    return width, height


def resolve_camera_fov(data: Dict[str, Any], width: int, height: int, fov_deg_arg: float) -> Tuple[float, float]:
    if fov_deg_arg > 0:
        fov = math.radians(float(fov_deg_arg))
        return fov, fov

    intr = data.get("intrinsics", {}) or {}
    fx = float(intr.get("fx", data.get("fx", 0.0)) or 0.0)
    fy = float(intr.get("fy", data.get("fy", 0.0)) or 0.0)
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera JSON missing fx/fy. Pass --fov-deg or include intrinsics.fx/fy.")
    return focal_to_fov(fx, width), focal_to_fov(fy, height)


def build_camera(data: Dict[str, Any], width: int, height: int, fovx: float, fovy: float, device: torch.device) -> MiniCam:
    # render_single_item_ply_augmentations.py writes row-vector matrices:
    # p_cam = p_world_h @ world_to_camera. GraphDECO MiniCam stores the
    # transposed column-convention matrix, which is the same 4x4 array here.
    world_to_camera_row = np.asarray(data["world_to_camera"], dtype=np.float32)
    if world_to_camera_row.shape != (4, 4):
        raise ValueError(f"world_to_camera must be 4x4, got {world_to_camera_row.shape}")

    world_view_transform = torch.tensor(world_to_camera_row, dtype=torch.float32, device=device)
    projection_matrix = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=fovx,
        fovY=fovy,
    ).transpose(0, 1).to(device)
    full_proj_transform = world_view_transform @ projection_matrix

    return MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def save_rendered_image(rendered: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = (rendered.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image[:3].transpose(1, 2, 0)
    imageio.imwrite(output_path, image)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Render a single GS image from a ply and camera JSON.")
    parser.add_argument("--ply", required=True, type=Path, help="Path to 3dgs_compressed.ply or compatible GS ply.")
    parser.add_argument("--camera-json", required=True, type=Path, help="Camera JSON with world_to_camera/intrinsics.")
    parser.add_argument("--output", required=True, type=Path, help="Output image path.")
    parser.add_argument("--width", type=int, default=0, help="Override output width.")
    parser.add_argument("--height", type=int, default=0, help="Override output height.")
    parser.add_argument("--fov-deg", type=float, default=0.0, help="Override both horizontal and vertical FOV in degrees.")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--background", type=str, default="1,1,1", help="RGB background in 0..1, e.g. 1,1,1.")
    parser.add_argument("--quiet", action="store_true")
    PipelineParams(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.ply.exists():
        raise FileNotFoundError(f"PLY not found: {args.ply}")
    if not args.camera_json.exists():
        raise FileNotFoundError(f"Camera JSON not found: {args.camera_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_payload = load_camera_payload(args.camera_json)
    width, height = resolve_camera_size(camera_payload, args.width, args.height)
    fovx, fovy = resolve_camera_fov(camera_payload, width, height, args.fov_deg)

    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(str(args.ply))
    camera = build_camera(camera_payload, width, height, fovx, fovy, device)
    bg_color = torch.tensor(parse_rgb(args.background), dtype=torch.float32, device=device)

    with torch.no_grad():
        rendered = render(camera, gaussians, args, bg_color=bg_color)["render"]
    save_rendered_image(rendered, args.output)

    if not args.quiet:
        print(
            "Rendered GS image "
            f"ply={args.ply} camera={args.camera_json} output={args.output} "
            f"size={width}x{height} fov=({math.degrees(fovx):.2f},{math.degrees(fovy):.2f}) device={device}"
        )


if __name__ == "__main__":
    main()
