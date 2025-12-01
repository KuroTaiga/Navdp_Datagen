#!/usr/bin/env python3
"""Batch renderer / verifier for InteriorGS datasets.

For each dataset directory (up to a limit of 100) under
    NavDP/navdp_api/gaussian_splatting/data/scenes
the script renders an orthographic image using the Gaussian-splatting renderer
and writes the result to
    NavDP/navdp_api/gaussian_splatting/data/results
as `<dataset>.png`.

When the optional `--verification` flag is provided, the script also produces a
side-by-side comparison between the occupancy map and the rendered result in
    NavDP/navdp_api/gaussian_splatting/data/verification.
"""

from __future__ import annotations

import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch

from arguments import PipelineParams
from gaussian_renderer import render_or
from scene import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix


BASE_DIR = Path(__file__).resolve().parent
INTERIOR_GS_DIR = BASE_DIR / "data" / "scenes"
RESULTS_DIR = BASE_DIR / "data" / "results"
VERIFICATION_DIR = BASE_DIR / "data" / "verification"
VIDEO_FRAMES_DIR = BASE_DIR / "data" / "video_frames"
DEFAULT_VIDEO_STEP_DEG = 10
DEFAULT_VIDEO_FOV_DEG = 60.0
VIDEO_FRAME_RESOLUTION = 100


def read_png_size(path: Path) -> tuple[int, int]:
    """Read the width/height of a PNG by inspecting the IHDR chunk."""

    with path.open("rb") as fh:
        header = fh.read(8)
        if header != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a valid PNG file")
        length = int.from_bytes(fh.read(4), "big")
        chunk_type = fh.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"{path} missing IHDR chunk")
        width = int.from_bytes(fh.read(4), "big")
        height = int.from_bytes(fh.read(4), "big")
        _ = fh.read(length - 8)  # remaining IHDR payload (not used)
    return width, height


def load_occupancy_metadata(dataset_dir: Path) -> dict:
    """Load occupancy.json and derive the world-space bounds."""

    occ_json = dataset_dir / "occupancy.json"
    if not occ_json.is_file():
        raise FileNotFoundError(f"Missing occupancy.json in {dataset_dir}")

    with occ_json.open("r", encoding="utf-8") as fh:
        occ = json.load(fh)

    scale = float(occ.get("scale", 1.0))
    min_x, min_y, min_z = map(float, occ.get("min", (0.0, 0.0, 0.0)))
    max_x, max_y, max_z = map(float, occ.get("max", (0.0, 0.0, 0.0)))

    lower = occ.get("lower") or [min_x, min_y, min_z]
    upper = occ.get("upper") or [max_x, max_y, max_z]
    lower_z = float(lower[2])
    upper_z = float(upper[2])

    occ_png = dataset_dir / "occupancy.png"
    if not occ_png.is_file():
        raise FileNotFoundError(f"Missing occupancy.png in {dataset_dir}")

    width_px, height_px = read_png_size(occ_png)

    left = min_x
    right = left + width_px * scale
    top = max_y
    bottom = top - height_px * scale

    return {
        "width": int(width_px),
        "height": int(height_px),
        "scale": scale,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "lower_z": lower_z,
        "upper_z": upper_z,
        "occupancy_png": occ_png,
    }


class OrthoMiniCam:
    """Minimal camera wrapper, matching nav_render's orthographic setup."""

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


def _generate_video_angles(step_deg: int) -> list[tuple[int, int]]:
    frames: list[tuple[int, int]] = []
    for pitch in range(180, 90 - step_deg, -step_deg):
        frames.append((pitch, 0))
    for yaw in range(step_deg, 360 + step_deg, step_deg):
        frames.append((90, yaw))
    return frames


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _compose_view_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    rot_t = rotation.transpose()
    view[:3, :3] = rot_t
    view[:3, 3] = -rot_t @ position
    return view


def render_video_frames(
    dataset_name: str,
    meta: dict,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    device: torch.device,
    bg_color: torch.Tensor,
    *,
    step_deg: int = DEFAULT_VIDEO_STEP_DEG,
    fov_deg: float = DEFAULT_VIDEO_FOV_DEG,
) -> None:
    width = meta["width"]
    height = meta["height"]
    left_world = meta["left"]
    right_world = meta["right"]
    top_world = meta["top"]
    bottom_world = meta["bottom"]
    lower_z = meta["lower_z"]
    upper_z = meta["upper_z"]

    cx = (left_world + right_world) * 0.5
    cy = (top_world + bottom_world) * 0.5
    cz = upper_z + 1.0
    camera_pos = np.array([cx, cy, cz], dtype=np.float32)

    span_xy = max(right_world - left_world, top_world - bottom_world)
    span_z = max(upper_z - lower_z, 0.0)
    extent = max(span_xy, span_z, 1.0)
    znear = 0.01
    zfar = extent * 4.0

    frame_width = VIDEO_FRAME_RESOLUTION
    frame_height = VIDEO_FRAME_RESOLUTION

    fovy = math.radians(fov_deg)
    aspect = max(frame_width / frame_height, 1e-6)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * aspect)

    frames_dir = VIDEO_FRAMES_DIR / dataset_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_params = _generate_video_angles(step_deg)
    if not frame_params:
        return

    print(
        f"    Rendering video frames ({len(frame_params)} frames, step {step_deg}°)…",
        flush=True,
    )

    projection_matrix = getProjectionMatrix(
        znear=znear,
        zfar=zfar,
        fovX=fovx,
        fovY=fovy,
    ).to(device).transpose(0, 1)

    for idx, (pitch_deg, yaw_deg) in enumerate(frame_params, start=1):
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        rotation = _rotation_z(yaw_rad) @ _rotation_y(pitch_rad)
        view_np = _compose_view_matrix(camera_pos, rotation)
        world_view_transform = torch.from_numpy(view_np).to(device).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)
        ).squeeze(0)

        camera = MiniCam(
            width=frame_width,
            height=frame_height,
            fovy=fovy,
            fovx=fovx,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
        )

        img_pkg = render_or(
            camera,
            gaussians,
            pipeline,
            bg_color=bg_color,
            orthographic=False,
        )

        render = img_pkg["render"].detach().cpu().numpy()
        render_uint8 = (
            np.clip(render, 0.0, 1.0) * 255.0
        ).astype(np.uint8).transpose(1, 2, 0)
        render_uint8 = np.rot90(render_uint8, k=-1)

        frame_path = frames_dir / f"frame_{idx:03d}_pitch{pitch_deg:03d}_yaw{yaw_deg:03d}.png"
        imageio.imwrite(frame_path, render_uint8)

    print(f"    Video frames saved to {frames_dir}", flush=True)

def find_ply_file(dataset_dir: Path) -> Path:
    """Return the first plausible .ply file inside the dataset directory."""

    candidates: Iterable[Path] = sorted(dataset_dir.glob("*.ply"))
    for cand in candidates:
        if cand.name.endswith(".ply"):
            return cand
    raise FileNotFoundError(f"No .ply file found in {dataset_dir}")


def build_camera(meta: dict, device: torch.device) -> OrthoMiniCam:
    width = meta["width"]
    height = meta["height"]

    left_world = meta["left"]
    right_world = meta["right"]
    top_world = meta["top"]
    bottom_world = meta["bottom"]
    lower_z = meta["lower_z"]
    upper_z = meta["upper_z"]

    half_width = 0.5 * (right_world - left_world)
    half_height = 0.5 * (top_world - bottom_world)
    cx = (left_world + right_world) * 0.5
    cy = (top_world + bottom_world) * 0.5
    cz = upper_z + 1.0

    znear = 0.01
    zfar = (cz - lower_z) + 1.0

    world_view_np = np.array(
        [
            [-1.0, 0.0, 0.0, cx],
            [0.0, -1.0, 0.0, cy],
            [0.0, 0.0, -1.0, cz],
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


def render_dataset(
    dataset_dir: Path,
    results_dir: Path,
    pipeline: PipelineParams,
    device: torch.device,
    bg_color: torch.Tensor,
    *,
    verification: bool = False,
) -> None:
    meta = load_occupancy_metadata(dataset_dir)
    ply_path = find_ply_file(dataset_dir)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))

    camera = build_camera(meta, device)
    img_pkg = render_or(camera, gaussians, pipeline, bg_color=bg_color, orthographic=True)
    rendered = img_pkg["render"].detach().cpu().numpy()
    render_uint8 = (np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
    render_uint8 = np.rot90(render_uint8, k=-1)

    dataset_name = dataset_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    render_path = results_dir / f"{dataset_name}.png"
    imageio.imwrite(render_path, render_uint8)

    render_video_frames(
        dataset_name,
        meta,
        gaussians,
        pipeline,
        device,
        bg_color,
    )

    if not verification:
        return

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    occ_img = imageio.imread(meta["occupancy_png"])

    occ_for_concat = occ_img
    if occ_for_concat.ndim == 2:
        occ_for_concat = np.stack([occ_for_concat] * 3, axis=-1)
    occ_for_concat = np.rot90(occ_for_concat, k=-1)
    if occ_for_concat.shape[0] != render_uint8.shape[0] or occ_for_concat.shape[1] != render_uint8.shape[1]:
        raise ValueError(
            f"Dimension mismatch for {dataset_name}: occupancy {occ_for_concat.shape[:2]}, render {render_uint8.shape[:2]}"
        )

    side_by_side = np.concatenate([occ_for_concat, render_uint8], axis=1)
    side_path = VERIFICATION_DIR / f"{dataset_name}.png"
    imageio.imwrite(side_path, side_by_side)


def main() -> None:
    cli_parser = ArgumentParser(description="Batch renderer for InteriorGS datasets")
    cli_parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Maximum number of dataset folders to process (default: all)",
    )
    cli_parser.add_argument(
        "--verification",
        action="store_true",
        help="Also export occupancy vs. render comparisons (default: off)",
    )
    args = cli_parser.parse_args()
    limit = args.max_datasets
    verification_enabled = bool(args.verification)

    if not INTERIOR_GS_DIR.is_dir():
        raise FileNotFoundError(f"InteriorGS directory not found at {INTERIOR_GS_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(d for d in INTERIOR_GS_DIR.iterdir() if d.is_dir())
    if limit is not None:
        dataset_dirs = dataset_dirs[:limit]
    if not dataset_dirs:
        print("No dataset directories found – aborting.")
        return

    pipeline_parser = ArgumentParser(description="Batch verification pipeline")
    pipeline = PipelineParams(pipeline_parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)

    for idx, dataset in enumerate(dataset_dirs, start=1):
        print(f"[{idx}/{len(dataset_dirs)}] Processing {dataset.name}…", flush=True)
        try:
            render_dataset(
                dataset,
                RESULTS_DIR,
                pipeline,
                device,
                bg_color,
                verification=verification_enabled,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ⚠️  Failed on {dataset.name}: {exc}")


if __name__ == "__main__":
    main()
