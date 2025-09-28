import matplotlib.pyplot as plt
import torch
import imageio
from gaussian_renderer import render_or
from scene import GaussianModel
from scene.cameras import MiniCam
from arguments import PipelineParams
from argparse import ArgumentParser
import os
from pathlib import Path
import numpy as np
import open3d as o3d

# === Helper: world→view transform (Y‑up or Z‑up) ===
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
    #import pdb; pdb.set_trace()
    return np.float32(Rt)


# -----------------------------------------------------------------------
# Render BEV using Gaussian‑Splatting renderer, camera centre fixed at (0,0)
# -----------------------------------------------------------------------
def render_topdown_gaussian_render(
    cloud_path: str | Path,
    save_path: str | Path,
    img_res: tuple[int, int] = (1024, 1024),
    bg_color: tuple[float, float, float] = (1, 1, 1),
    up_axis: str = "y",
    padding: float = 0.2,
    cam_height_offset: float | None = None,
):
    """
    Render an orthographic bird‑eye image via the Gaussian‑Splatting renderer,
    **forcing the camera to be centred at the world origin (0,0)**.

    Parameters
    ----------
    cloud_path : str | Path
        Path to Gaussian‑splat .ply file  (must contain per‑Gauss attributes).
    save_path : str | Path
        PNG output path.
    img_res : (int,int)
        Output resolution (width, height).
    bg_color : (r,g,b) in [0,1].
    up_axis : {"y","z"}, default "y"
        Which world axis is "up".
    padding : float
        Extra margin (scene units) added around scene bounds.
    cam_height_offset : float | None
        If given, use this absolute height above origin; otherwise the code
        selects `max_abs(up_axis_range)+10`.
    """
    # ---------- I/O ----------
    cloud_path = Path(cloud_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Load Gaussian model ----------
    gaussians = GaussianModel(sh_degree=3)
    if cloud_path.suffix != ".ply":
        raise ValueError("Expected .ply exported by Gaussian‑Splatting.")
    gaussians.load_ply(str(cloud_path))
    device = gaussians.get_xyz.device

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    if xyz.size == 0:
        raise ValueError("Loaded Gaussian model contains no valid points.")

    x_min, y_min, z_min = xyz.min(axis=0)
    x_max, y_max, z_max = xyz.max(axis=0)

    # ---------- Camera placement (fixed origin) ----------
    if up_axis.lower() == "y":
        half_x = max(abs(x_min), abs(x_max)) + padding
        half_z = max(abs(z_min), abs(z_max)) + padding
        abs_y = max(abs(y_min), abs(y_max))
        cam_h = cam_height_offset if cam_height_offset is not None else abs_y + 10.0
        cam_pos = np.array([0.0, cam_h, 0.0], dtype=np.float32)

        # Y‑up: right=+X, up=+Z, forward=−Y  (pure Rx(−90°))
        c2w_R = np.array([[1, 0,  0],
                          [0, 0,  1],
                          [0,-1,  0]], dtype=np.float32)

        left, right = -half_x, half_x
        bottom, top = -half_z, half_z
        ground_half_extent = max(half_x, half_z)
    else:  # Z‑up
        half_x = max(abs(x_min), abs(x_max)) + padding
        half_y = max(abs(y_min), abs(y_max)) + padding
        abs_z = max(abs(z_min), abs(z_max))
        cam_h = cam_height_offset if cam_height_offset is not None else abs_z + 10.0
        cam_pos = np.array([5.5, -1.75, cam_h], dtype=np.float32)

        # Z‑up camera for BEV (Rx(−90°)+Yaw) ‑‑ matches earlier code
        c2w_R = np.array([[ -1,  0,  0],
                          [0,  1,  0],
                          [ 0,  0, -1]], dtype=np.float32)

        left, right = -half_x, half_x
        bottom, top = -half_y, half_y
        ground_half_extent = max(half_x, half_y)

    # ---------- Helpers ----------
    def _get_ortho(l, r, b, t, n, f, dev):
        """Column‑major OpenGL‑style orthographic projection."""
        return torch.tensor(
            [[ 2/(r-l),      0,          0, -(r+l)/(r-l)],
             [      0,  2/(t-b),        0, -(t+b)/(t-b)],
             [      0,       0, -2/(f-n), -(f+n)/(f-n)],
             [      0,       0,        0,           1]],
            dtype=torch.float32,
            device=dev,
        )

    # ---------- Build matrices ----------
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = c2w_R
    pose[:3, 3] = cam_pos

    world_view = torch.tensor(
        getWorld2View2(c2w_R, cam_pos), dtype=torch.float32, device=device
    ).T  # column‑major

    near = 0.01
    far = cam_h + ground_half_extent + padding
    ortho = _get_ortho(left, right, bottom, top, near, far, device)
    full_proj = ortho @ world_view

    # ---------- Camera object ----------
    W, H = img_res
    camera = MiniCam(
        width=W,
        height=H,
        fovy=1.0,                # unused in ortho mode
        fovx=1.0,
        znear=near,
        zfar=far,
        world_view_transform=world_view,
        full_proj_transform=full_proj
    )

    # ---------- Render ----------
    parser = ArgumentParser()
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    img_pkg = render_or(camera, gaussians, PipelineParams(parser),
                        bg_color=bg, orthographic=True)
    rendered = img_pkg["render"].detach().cpu().numpy()
    rgb = (rendered.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)

    imageio.imwrite(save_path, rgb)
    print(
        f"✅ BEV saved to {save_path.resolve()}  "
        f"(centre@origin, up={up_axis}, size={img_res})"
    )


def render_topdown_gaussian_cloud(
    cloud_path: str | Path,
    save_path: str | Path,
    img_res: tuple[int, int] = (1024, 1024),
    bg_color: tuple[float, float, float] = (1, 1, 1),   # 白色背景
    point_size: float = 1.5,
    flip_y: bool = True,
    up_axis: str = "z",
):
    """
    Render an orthographic (bird-eye) image from a Gaussian point cloud.

    Parameters
    ----------
    cloud_path : str | Path
        Path to the Gaussian point-cloud file (.ply / .pcd / .npy with dict).
    save_path : str | Path
        Path where the top-down PNG will be saved.
    img_res : (int, int), default (1024,1024)
        Output image resolution in (width, height).
    bg_color : (r,g,b) float triple, default white
        Matplotlib background color in range [0,1].
    point_size : float, default 1.5
        Matplotlib scatter marker size.
    flip_y : bool, default True
        Whether to flip Y so that +Y points upward in the image.
    up_axis : {"z", "y"}, default "z"
        Axis that represents "up" in the world coordinate.  If "y", the ground plane
        is X‑Z; if "z", the ground plane is X‑Y.
    """
    cloud_path = Path(cloud_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 读取点云 ----------
    if cloud_path.suffix == ".npy":
        data = np.load(cloud_path, allow_pickle=True).item()
        pts = data["xyz"]           # (N,3)
        colors = data.get("rgb", None)  # (N,3) or None
    else:  # .ply / .pcd 等
        pcd = o3d.io.read_point_cloud(str(cloud_path))
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    if pts.size == 0:
        raise ValueError("Point cloud is empty.")

    # ---------- 2. 正交投影（丢弃 up_axis） ----------
    if up_axis.lower() == "y":
        ground = pts[:, [0, 2]]           # (x, z)
    else:  # "z"
        ground = pts[:, :2]               # (x, y)

    if flip_y:   # 将 +Y/+Z 朝上
        ground[:, 1] *= -1

    # ---------- 3. 归一化到图像坐标 ----------
    min_ground = ground.min(axis=0)
    max_ground = ground.max(axis=0)
    span = (max_ground - min_ground).max()  # 取最大边做等比缩放，保证不变形

    # 留一点边距
    pad = 0.02 * span
    min_ground -= pad
    max_ground += pad
    span = (max_ground - min_ground).max()

    # 将坐标映射到 [0,1]
    norm_ground = (ground - min_ground) / span

    # ---------- 4. 使用 Matplotlib 绘制 ----------
    w, h = img_res
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])        # 无边框
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")

    if colors is not None and len(colors) == len(pts):
        ax.scatter(
            norm_ground[:, 0],
            norm_ground[:, 1],
            s=point_size,
            c=colors,
            marker="o",
            linewidths=0,
        )
    else:
        ax.scatter(
            norm_ground[:, 0],
            norm_ground[:, 1],
            s=point_size,
            c="black",
            marker="o",
            linewidths=0,
        )

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"✅ Saved top-down view to {save_path.resolve()} (up_axis={up_axis})")


# ------------------------------ CLI 例子 ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render top-down image from Gaussian point cloud.")
    parser.add_argument("--in_pcd", required=True, help="Input point-cloud path (.ply / .npy).")
    parser.add_argument("--out_img", required=True, help="Output PNG path.")
    parser.add_argument("--res", type=int, nargs=2, default=(345, 312), metavar=("W", "H"),
                        help="Image resolution (width height).")
    parser.add_argument("--size", type=float, default=1.5, help="Dot size.")
    parser.add_argument("--up_axis", choices=["y", "z"], default="z",
                        help="Which axis is up in world coordinates.")
    parser.add_argument("--mode", choices=["points", "render"], default="render",
                        help="BEV style: 'points' for scatter plot, 'render' for Gaussian renderer.")
    args = parser.parse_args()

    if args.mode == "points":
        render_topdown_gaussian_cloud(
            cloud_path=args.in_pcd,
            save_path=args.out_img,
            img_res=tuple(args.res),
            point_size=args.size,
            up_axis=args.up_axis,
        )
    else:
        render_topdown_gaussian_render(
            cloud_path=args.in_pcd,
            save_path=args.out_img,
            img_res=tuple(args.res),
            up_axis=args.up_axis,
        )