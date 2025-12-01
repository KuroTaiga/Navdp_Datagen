import torch
import numpy as np
import imageio
import os
from arguments import PipelineParams
from argparse import ArgumentParser
from scene.cameras import MiniCam

# 手动创建一个空 parser
parser = ArgumentParser(description="Testing script parameters")
pipeline = PipelineParams(parser)

# 这些模块来自 GraphDECO 的代码结构
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载高斯点云 ===
gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply("/home/tianhang/tianshi_gong/Nav/merged_scene_noroof.ply")  # ← 请替换为你的 .ply 路径


# === 打印高斯点坐标范围 ===
xyz = gaussians.get_xyz.detach().cpu().numpy()
print("📦 Gaussian 点云坐标范围:")
print(f"X: {xyz[:,0].min():.3f} → {xyz[:,0].max():.3f}")
print(f"Y: {xyz[:,1].min():.3f} → {xyz[:,1].max():.3f}")
print(f"Z: {xyz[:,2].min():.3f} → {xyz[:,2].max():.3f}")


# === 相机参数（手动设定） ===
H, W = 640, 640           # 图像高度与宽度
fov_deg = 60              # 相机视场角（越小越“长焦”）
focal = fov2focal(fov_deg, W)

R = np.eye(3)
t = np.array([0.0, 0.0, 0.0])


# === 构建相机对象 ===
world_view_transform = torch.tensor(getWorld2View2(R, t), dtype=torch.float32).transpose(0, 1).to(device)
projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_deg, fovY=fov_deg).transpose(0, 1).to(device)
full_proj_transform = world_view_transform @ projection_matrix

camera = MiniCam(
    width=W,
    height=H,
    fovy=fov_deg,
    fovx=fov_deg,
    znear=0.01,
    zfar=100.0,
    world_view_transform=world_view_transform,
    full_proj_transform=full_proj_transform
)
# === 设置背景色并渲染 ===
bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)  # 白色背景
rendered = render(camera, gaussians, pipeline, bg_color=bg_color)["render"]

# === 保存图像 ===
out_img = (rendered.clamp(0,1).detach().cpu().numpy() * 255).astype(np.uint8)
out_img = out_img.transpose(1, 2, 0)
#import pdb; pdb.set_trace()
imageio.imwrite("./output/render_output.png", out_img)
print("✅ 渲染完成，保存至 ./output/render_output.png")