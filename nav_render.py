import math
import os
import json
import numpy as np
import torch
import imageio
from gaussian_renderer import render_or
from scene import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import  fov2focal, focal2fov
from arguments import PipelineParams
from argparse import ArgumentParser


from datetime import datetime

def ts():
    # 本地时间，精确到毫秒：20250105_173412_123
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

# 相机内参设置
W, H = 600, 480
intrinsic = np.array([
    [100.0, 0.0, 310.0],
    [0.0, 100.0, 230.0],
    [0.0, 0.0, 1.0]
])
focal_length_x = intrinsic[0, 0]
focal_length_y = intrinsic[1, 1]

# 加载高斯点云模型
gaussians = GaussianModel(sh_degree=3)
# gaussians.load_ply("/home/tianhang/tianshi_gong/Nav/merged_scene_noroof.ply")
PLY_PATH = "/home/tianhang/tianshi_gong/Nav/merged_scene_noroof.ply"
gaussians.load_ply(PLY_PATH)


def read_png_size(path):
    with open(path, "rb") as f:
        header = f.read(8)
        if header != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a valid PNG file")
        length = int.from_bytes(f.read(4), "big")
        chunk_type = f.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"{path} missing IHDR chunk")
        width = int.from_bytes(f.read(4), "big")
        height = int.from_bytes(f.read(4), "big")
        return width, height


def load_occupancy_metadata(ply_path):
    data_dir = os.path.dirname(ply_path)
    occ_json = os.path.join(data_dir, "occupancy.json")
    if not os.path.isfile(occ_json):
        raise FileNotFoundError(f"Occupancy metadata not found at {occ_json}")

    with open(occ_json, "r") as f:
        occ = json.load(f)

    scale = float(occ.get("scale", 1.0))
    min_x, min_y, min_z = map(float, occ.get("min", [0, 0, 0]))
    max_x, max_y, max_z = map(float, occ.get("max", [0, 0, 0]))
    lower = occ.get("lower", [min_x, min_y, min_z])
    upper = occ.get("upper", [max_x, max_y, max_z])
    lower_z = float(lower[2])
    upper_z = float(upper[2])

    occ_png = os.path.join(data_dir, "occupancy.png")
    if not os.path.isfile(occ_png):
        raise FileNotFoundError(f"Occupancy map not found at {occ_png}")
    width_px, height_px = read_png_size(occ_png)

    left = float(min_x)
    right = left + width_px * scale
    top = float(max_y)
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
    }


class OrthoMiniCam:
    def __init__(self, width, height, world_view_transform, full_proj_transform,
                 half_width, half_height, znear, zfar):
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

    def get_full_proj_transform(self, orthographic=False):
        if not orthographic:
            return self.full_proj_transform
        return self._half_width, self._half_height, self.full_proj_transform

# === 相机转换矩阵 ===
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
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

# === 投影矩阵 ===
def getProjectionMatrixwithPrincipalPointOffset(znear, zfar, fovX, fovY, fx, fy, cx, cy, w, h):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top_c = tanHalfFovY * znear
    bottom_c = -top_c
    right_c = tanHalfFovX * znear
    left_c = -right_c

    dx = (cx - w/2) / fx * znear
    dy = (cy - h/2) / fy * znear

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

# === 主函数接口 ===
def render_and_save(pose_matrix, save_dir="./nav_data", orthographic=False):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)

    if orthographic:
        occ_meta = load_occupancy_metadata(PLY_PATH)
        width = occ_meta["width"]
        height = occ_meta["height"]

        left_world = occ_meta["left"]
        right_world = occ_meta["right"]
        top_world = occ_meta["top"]
        bottom_world = occ_meta["bottom"]

        half_width = 0.5 * (right_world - left_world)
        half_height = 0.5 * (top_world - bottom_world)
        cx = (left_world + right_world) * 0.5
        cy = (top_world + bottom_world) * 0.5
        lower_z = occ_meta["lower_z"]
        upper_z = occ_meta["upper_z"]
        cz = upper_z + 1.0
        znear = 0.01
        zfar = (cz - lower_z) + 1.0

        world_view_np = np.array([
            [-1.0, 0.0, 0.0,  cx],
            [0.0, -1.0, 0.0,  cy],
            [0.0, 0.0, -1.0,  cz],
            [0.0, 0.0, 0.0,  1.0],
        ], dtype=np.float32)

        left_cam = -half_width
        right_cam = half_width
        top_cam = -half_height
        bottom_cam = half_height

        projection_np = np.array([
            [2.0 / (right_cam - left_cam), 0.0, 0.0, -(right_cam + left_cam) / (right_cam - left_cam)],
            [0.0, 2.0 / (top_cam - bottom_cam), 0.0, -(top_cam + bottom_cam) / (top_cam - bottom_cam)],
            [0.0, 0.0, -2.0 / (zfar - znear), -(zfar + znear) / (zfar - znear)],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        world_view_transform = torch.tensor(world_view_np, device=device).transpose(0, 1)
        projection_matrix = torch.tensor(projection_np, device=device).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)).squeeze(0)

        camera = OrthoMiniCam(
            width=width,
            height=height,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            half_width=half_width,
            half_height=half_height,
            znear=znear,
            zfar=zfar,
        )

        img_pkg = render_or(
            camera,
            gaussians,
            pipeline,
            bg_color=bg_color,
            orthographic=False,
        )

    else:
        width = W
        height = H
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        K = intrinsic
        znear, zfar = 0.01, 10.0
        trans = np.array([0,0,0])
        scale = 1.0

        fovx = focal2fov(focal_length_x, width)
        fovy = focal2fov(focal_length_y, height)

        world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).to(device)
        projection_matrix = getProjectionMatrixwithPrincipalPointOffset(znear, zfar, fovx, fovy,
            fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], w=width, h=height).transpose(0,1).to(device)

        full_proj_transform = (world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)).squeeze(0)

        camera = MiniCam(
            width=width,
            height=height,
            fovy=fovy,
            fovx=fovx,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform
        )

        img_pkg = render_or(
            camera,
            gaussians,
            pipeline,
            bg_color=bg_color,
            orthographic=False,
        )
    rendered = img_pkg["render"].detach().cpu().numpy()
    depth = img_pkg['depth'].detach().cpu().numpy()
    rgb = (rendered.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)

    np.save(os.path.join(save_dir, "rgb_image.npy"), rgb)
    np.save(os.path.join(save_dir, "depth_image.npy"), depth)
    # fname = f"render_{ts()}.png"
    # imageio.imwrite(os.path.join(save_dir, fname), rgb)
    imageio.imwrite(os.path.join(save_dir, "render_image.png"), rgb)
    print(f"渲染图像已保存到 {os.path.join(save_dir, 'render_image.png')}")

    return rgb


# if __name__ == "__main__":
#     default_pose = np.eye(4, dtype=np.float32)
#     render_and_save(default_pose)

if __name__ == "__main__":
    default_pose = np.eye(4, dtype=np.float32)
    flip_rx = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0,-1]], dtype=np.float32)
    default_pose[:3, :3] = default_pose[:3, :3] @ flip_rx
    default_pose[2, 3] = 0  # 这里把 z 平移改成你想要的值
    render_and_save(default_pose, orthographic=True)
