#!/usr/bin/env python3
import torch
import numpy as np
import imageio.v2 as imageio
import math
import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is in PYTHONPATH
# Usually needed if the script is running from a subdirectory
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from arguments import PipelineParams
from scene import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render_or
from utils.graphics_utils import getProjectionMatrix

# ==========================================
# Math Utilities
# ==========================================


def qvec2rotmat(qvec):
    """Convert Quaternion (w,x,y,z) to 3x3 Rotation Matrix."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# ==========================================
# Helper Functions (Metadata & Transform)
# ==========================================


def read_png_size(path: Path) -> tuple[int, int]:
    """Read PNG header to get width and height."""
    if not path.exists():
        raise FileNotFoundError(f"PNG file not found: {path}")
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
    return width, height


def load_occupancy_metadata(scene_dir: Path) -> dict:
    """Load metadata from occupancy.json and occupancy.png in the scene directory."""
    occ_json = scene_dir / "occupancy.json"
    if not occ_json.is_file():
        raise FileNotFoundError(f"Missing occupancy.json in {scene_dir}")

    with occ_json.open("r", encoding="utf-8") as fh:
        occ = json.load(fh)

    scale = float(occ.get("scale", 1.0))
    min_x, min_y, min_z = map(float, occ.get("min", (0.0, 0.0, 0.0)))

    occ_png = scene_dir / "occupancy.png"
    width_px, height_px = read_png_size(occ_png)

    left = min_x
    right = left + width_px * scale
    # Note: occupancy coordinates typically have Y pointing down, so top is max Y
    top = occ.get("max", [0, 0, 0])[1]
    bottom = top - height_px * scale

    return {
        "width": int(width_px),
        "height": int(height_px),
        "scale": scale,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "lower_z": float(min_z),
    }


def load_raster_world_points(json_path: Path, swap_xy: bool = False):
    """Load raster_world and raster_pixel points from JSON for affine calculation."""
    if not json_path.exists():
        raise FileNotFoundError(f"Reference JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    path_data = payload.get("path", {})
    raster_world = path_data.get("raster_world")
    raster_pixel = path_data.get("raster_pixel")

    if not raster_world or not raster_pixel:
        raise ValueError(f"Reference JSON missing raster data.")

    points = []
    pixels = []
    for entry, pix in zip(raster_world, raster_pixel):
        x = float(entry["x"])
        y = float(entry["y"])
        if swap_xy:
            points.append(np.array([y, x], dtype=np.float32))
        else:
            points.append(np.array([x, y], dtype=np.float32))
        pixels.append((int(pix[0]), int(pix[1])))
    return points, pixels


def derive_affine_transform(points, pixels, meta):
    """Calculate affine transform parameters using Least Squares."""
    n = len(points)
    if n < 2:
        raise ValueError(f"Need at least 2 points to calculate affine transform.")

    scale = float(meta["scale"])
    left = float(meta["left"])
    top = float(meta["top"])

    sum_x = sum(pt[0] for pt in points)
    sum_y = sum(pt[1] for pt in points)
    sum_x2 = sum(pt[0] * pt[0] for pt in points)
    sum_y2 = sum(pt[1] * pt[1] for pt in points)
    sum_map_x = 0.0
    sum_map_y = 0.0
    sum_x_map_x = 0.0
    sum_y_map_y = 0.0

    for pt, pix in zip(points, pixels):
        # Convert pixel coordinates to scene physical coordinates (Map Coordinates)
        map_x = left + int(pix[0]) * scale
        map_y = top - int(pix[1]) * scale

        sum_map_x += map_x
        sum_map_y += map_y
        sum_x_map_x += pt[0] * map_x
        sum_y_map_y += pt[1] * map_y

    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y

    if abs(denom_x) < 1e-8 or abs(denom_y) < 1e-8:
        raise ValueError("Cannot solve affine transform.")

    a_x = (n * sum_x_map_x - sum_x * sum_map_x) / denom_x
    b_x = (sum_map_x - a_x * sum_x) / n
    a_y = (n * sum_y_map_y - sum_y * sum_map_y) / denom_y
    b_y = (sum_map_y - a_y * sum_y) / n
    return a_x, b_x, a_y, b_y


def transform_point_3d(
    pt_xyz: np.ndarray, affine: tuple, meta: dict, mirror_translation: bool
) -> np.ndarray:
    """
    3D Coordinate Transformation Pipeline:
    Nav Coords -> Affine Transform -> Mirror Translation -> GS Coords
    - X, Y: Transformed
    - Z: Passed through unchanged
    """
    a_x, b_x, a_y, b_y = affine

    # 1. Affine Transform
    x_aff = a_x * pt_xyz[0] + b_x
    y_aff = a_y * pt_xyz[1] + b_y

    # 2. Mirror Translation (Coordinate System Correction)
    if mirror_translation:
        center_x = 0.5 * (meta["left"] + meta["right"])
        center_y = 0.5 * (meta["top"] + meta["bottom"])
        x_final = center_x * 2.0 - x_aff
        y_final = center_y * 2.0 - y_aff
    else:
        x_final = x_aff
        y_final = y_aff

    return np.array([x_final, y_final, pt_xyz[2]], dtype=np.float32)


# ==========================================
# Camera Builder
# ==========================================


def build_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        # Fallback if target equals eye
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward /= forward_norm

    z_axis = forward
    x_axis = np.cross(z_axis, up)  # Right vector

    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)  # Down vector (OpenCV style)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = x_axis
    view[1, :3] = y_axis
    view[2, :3] = z_axis
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def build_perspective_camera(
    position, target, width, height, fov_deg, znear, zfar, device
):
    fovy = math.radians(fov_deg)
    aspect = width / max(height, 1)
    fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * aspect)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    view = build_look_at(position, target, up)
    world_view = torch.from_numpy(view).to(device).transpose(0, 1)
    projection = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
        .to(device)
        .transpose(0, 1)
    )
    full_proj = (world_view.unsqueeze(0) @ projection.unsqueeze(0)).squeeze(0)

    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)


# ==========================================
# Main
# ==========================================


def main():
    parser = argparse.ArgumentParser(
        description="Render a single view using NavDP coords (2 modes)."
    )

    # Required file paths
    parser.add_argument("--ply", type=Path, required=True, help="Path to .ply file")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Scene root dir (containing occupancy.json)",
    )
    parser.add_argument(
        "--ref-json",
        type=Path,
        required=True,
        help="Reference trajectory JSON for affine calc",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output.png"), help="Output image path"
    )

    # --- Input Modes (Mutually Exclusive) ---
    group = parser.add_mutually_exclusive_group(required=True)

    # Mode 1: Pos + Target (2D)
    # Input: --pos x y --target x y
    group.add_argument("--pos", type=float, nargs=2, help="Mode 1: Camera Pos X Y")

    # Mode 2: Pose (3D Pos + Quaternion)
    # Input: --pose x y z w x y z
    group.add_argument(
        "--pose", type=float, nargs=7, help="Mode 2: Pos(x,y,z) + Quat(w,x,y,z)"
    )

    # Target argument for Mode 1 (non-exclusive group, checked in logic)
    parser.add_argument(
        "--target", type=float, nargs=2, help="Mode 1: Look-at Target X Y"
    )

    # Transformation Options
    parser.add_argument(
        "--swap-xy", action="store_true", help="Swap X/Y in raster world input"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Disable mirror translation"
    )

    # Intrinsics
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--bg-color", type=float, nargs=3, default=[1.0, 1.0, 1.0])

    args = parser.parse_args()

    # Logical validation
    if args.pos and args.target is None:
        parser.error("Mode 1 requires both --pos and --target.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("Warning: CUDA not found, using CPU (slow).")

    # 1. Prepare Transforms
    print(f"[1/5] Loading metadata...")
    try:
        meta = load_occupancy_metadata(args.scene_dir)
        points, pixels = load_raster_world_points(args.ref_json, swap_xy=args.swap_xy)
        affine = derive_affine_transform(points, pixels, meta)
        print(f"      Affine: ax={affine[0]:.4f}, ay={affine[2]:.4f}")
    except Exception as e:
        print(f"Error preparing transform: {e}")
        return

    mirror_translation = not args.no_mirror
    fixed_z = 1.3  # Fixed height override

    # 2. Calculate Start and End points in Nav coordinate system
    # We transform both the camera position and the "look-at target".
    # This allows the LookAt function to handle the rotation changes caused by mirroring automatically.

    if args.pose:
        # --- Mode 2: Pose (XYZ + Quat) ---
        print("[Mode 2] Using Pose input (Position + Quaternion).")
        raw_x, raw_y, raw_z_input, qw, qx, qy, qz = args.pose

        # Camera Position in Nav coords (forcing fixed Z)
        nav_pos = np.array([raw_x, raw_y, fixed_z], dtype=np.float32)

        # Calculate viewing direction vector
        rot_mat = qvec2rotmat([qw, qx, qy, qz])

        # Define local forward vector.
        # Assuming typical conventions where view direction is +Z.
        local_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Forward vector in World coords
        world_forward = rot_mat @ local_forward

        # Generate a virtual target point (1 meter away)
        nav_target = nav_pos + world_forward

        print(f"      Nav Pos:    {nav_pos}")
        print(f"      Nav Forward: {world_forward}")
        print(f"      Nav Target: {nav_target} (Virtual)")

    else:
        # --- Mode 1: Pos + Target ---
        print("[Mode 1] Using Pos + Target input.")
        nav_pos = np.array([args.pos[0], args.pos[1], fixed_z], dtype=np.float32)
        nav_target = np.array(
            [args.target[0], args.target[1], fixed_z], dtype=np.float32
        )

    # 3. Coordinate Transformation (Nav -> GS)
    gs_pos = transform_point_3d(nav_pos, affine, meta, mirror_translation)
    gs_target = transform_point_3d(nav_target, affine, meta, mirror_translation)

    print("-" * 40)
    print(f"GS Camera Pos:    {gs_pos}")
    print(f"GS Camera Target: {gs_target}")
    print("-" * 40)

    # 4. Load Model
    print(f"[3/5] Loading Gaussian model...")
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(str(args.ply))

    # 5. Render
    print(f"[4/5] Rendering...")
    camera = build_perspective_camera(
        position=gs_pos,
        target=gs_target,
        width=args.width,
        height=args.height,
        fov_deg=args.fov,
        znear=0.01,
        zfar=100.0,
        device=device,
    )

    pipeline_parser = argparse.ArgumentParser()
    pipeline_args = PipelineParams(pipeline_parser)
    bg_color = torch.tensor(args.bg_color, dtype=torch.float32, device=device)

    with torch.no_grad():
        img_pkg = render_or(
            camera, gaussians, pipeline_args, bg_color=bg_color, orthographic=False
        )

    render = img_pkg["render"].detach().cpu().numpy()
    render_img = (np.clip(render, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)

    # Rotate image if necessary (based on data coordinates)
    # render_img = np.rot90(render_img, k=2)

    imageio.imwrite(args.output, render_img)
    print(f"Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
