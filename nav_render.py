import math
import os
import numpy as np
import torch
import imageio
from gaussian_renderer import render_or
from scene import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import fov2focal, focal2fov
from arguments import PipelineParams
from argparse import ArgumentParser

from datetime import datetime


def ts():
    """Return local timestamp with millisecond precision."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


# Camera intrinsics
W, H = 640, 480
intrinsic = np.array([
    [360.0, 0.0, 350.0],
    [0.0, 360.0, 230.0],
    [0.0, 0.0, 1.0]
])
focal_length_x = intrinsic[0, 0]
focal_length_y = intrinsic[1, 1]

# Load Gaussian model
gaussians = GaussianModel(sh_degree=3)
# gaussians.load_ply("/home/tianhang/tianshi_gong/Nav/merged_scene_noroof.ply")
gaussians.load_ply(
    "/home/tianhang/tianshi_gong/floor30/floor30_rot_v1.ply"
)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    C2W[:3, 3] = t
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrixwithPrincipalPointOffset(znear, zfar, fovX, fovY, fx, fy, cx, cy, w, h):
    """Construct perspective projection matrix accounting for principal-point offset."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top_c = tanHalfFovY * znear
    bottom_c = -top_c
    right_c = tanHalfFovX * znear
    left_c = -right_c

    dx = (cx - w / 2) / fx * znear
    dy = (cy - h / 2) / fy * znear

    top = top_c + dy
    bottom = bottom_c + dy
    left = left_c + dx
    right = right_c + dx

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = z_sign * (right + left) / (right - left)
    P[1, 2] = z_sign * (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def render_and_save(pose_matrix, save_dir="./nav_data"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]
    K = intrinsic
    znear, zfar = 0.01, 10.0
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0

    fovx = focal2fov(focal_length_x, W)
    fovy = focal2fov(focal_length_y, H)

    world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).to(device)
    projection_matrix = getProjectionMatrixwithPrincipalPointOffset(
        znear=znear,
        zfar=zfar,
        fovX=fovx,
        fovY=fovy,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        w=W,
        h=H,
    ).transpose(0, 1).to(device)

    full_proj_transform = (world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)).squeeze(0)

    camera = MiniCam(
        width=W,
        height=H,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )

    parser = ArgumentParser(description="Testing script parameters")
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    img_pkg = render_or(camera, gaussians, PipelineParams(parser), bg_color=bg_color)
    rendered = img_pkg["render"].detach().cpu().numpy()
    depth = img_pkg["depth"].detach().cpu().numpy()
    rgb = (rendered.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)

    np.save(os.path.join(save_dir, "rgb_image.npy"), rgb)
    np.save(os.path.join(save_dir, "depth_image.npy"), depth)
    imageio.imwrite(os.path.join(save_dir, "render_image.png"), rgb)
    print(f"渲染图像已保存到 {os.path.join(save_dir, 'render_image.png')}")

    return rgb


if __name__ == "__main__":
    default_pose = np.eye(4, dtype=np.float32)
    render_and_save(default_pose)
