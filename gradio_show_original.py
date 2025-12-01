# import math
# import os
# import numpy as np
# import torch
# import gradio as gr
# import cv2
# import imageio
# from datetime import datetime
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from arguments import PipelineParams
# from argparse import ArgumentParser

# from gradio_model import BicycleModel
# from gaussian_renderer import render_or
# from scene import GaussianModel
# from scene.cameras import MiniCam
# from utils.graphics_utils import  fov2focal, focal2fov
# import shutil
# import atexit
# import gradio.processing_utils

# gradio.processing_utils.CACHE_DIR = "./gradio_cache"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #============

# def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = R.transpose()
 
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0

#     C2W = np.linalg.inv(Rt)
#     C2W[:3, 3] = t
#     cam_center = C2W[:3, 3]
#     cam_center = (cam_center + translate) * scale
#     C2W[:3, 3] = cam_center
#     Rt = np.linalg.inv(C2W)
#     #import pdb; pdb.set_trace()
#     return np.float32(Rt)
# #============

# def getProjectionMatrixwithPrincipalPointOffset(znear, zfar, fovX, fovY, fx, fy, cx, cy, w, h):
#     '''
#         Reference for refleecting principal point shift to calculate the projection matrix
#         http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
#     '''
    
#     tanHalfFovY = math.tan((fovY / 2))
#     tanHalfFovX = math.tan((fovX / 2))
#     top_c = tanHalfFovY * znear
#     bottom_c = -top_c
#     right_c = tanHalfFovX * znear
#     left_c = -right_c

#     # Project difference between camera center and half of dimension to znear plane
#     dx = (cx - w/2) / fx * znear
#     dy = (cy - h/2) / fy * znear

#     top = top_c + dy
#     bottom = bottom_c + dy 
#     left = left_c + dx
#     right = right_c + dx    
#     P = torch.zeros(4, 4)
#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = z_sign * (right + left) / (right - left)
#     P[1, 2] = z_sign * (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)

#     return P

# # ====== 相机参数初始化 ======
# W, H = 640, 480


# intrinsic = np.array([
#     [360.0, 0.0, 350.0],
#     [0.0, 360.0, 230.0],
#     [0.0, 0.0, 1.0]
# ])

# focal_length_x = intrinsic[0, 0]
# focal_length_y = intrinsic[1, 1]


# # ====== 加载点云并获取中心点 ======
# gaussians = GaussianModel(sh_degree=3)
# gaussians.load_ply("/home/tianhang/tianshi_gong/floor30/floor30geo_cut.ply")
# center = gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()
# print("Point cloud center:", center)


# # ====== 初始相机状态 ======

# trans = np.eye(4)
# # trans[:3, :3] = [[0, 0, 1], 
# #                 [-1, 0, 0], 
# #                 [0, -1, 0]]
# # trans[:3, 3] = [0.42992, 2.382528, -1.040942]

# # trans[:3, :3] = [[1, 0, 0],
# #                         [0, 0, 1],
# #                         [0, -1, 0]]
# # trans[:3, 3] =[2.970080, -5.017472, 0.459058]

# trans[:4, :4] = [
#     [1.000000, 0.000000, 0.000000, 0],
#     [0.000000, -1.000000, 0.000000, 0],
#     [0.000000, 0.000000, -1.000000, 10],
#     [0.000000, 0.000000, 0.000000, 1.000000],
# ]



# bicycle_model = BicycleModel()
# camera_pose = np.zeros([10000, 3, 4])
# count = 1

# # ====== 渲染单帧 ======
# def gen_one(pose_trans, transformation_matrix):
#     parser = ArgumentParser(description="Testing script parameters")
#     pipeline = PipelineParams(parser)
#     R = transformation_matrix[:3, :3]
#     t = pose_trans[:3, 3]
    
    
#     K = intrinsic

   
#     zfar = 100.0
#     znear = 0.01

#     trans = np.array([0,0,0])
#     scale = 1.0

#     fovy=focal2fov(focal_length_y, H)
#     fovx=focal2fov(focal_length_x, W)
    
#     world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).cuda()
#     projection_matrix = getProjectionMatrixwithPrincipalPointOffset(znear=znear, 
#                                                                             zfar=zfar, 
#                                                                             fovX=fovx, 
#                                                                             fovY=fovy,
#                                                                             fx=K[0, 0],
#                                                                             fy=K[1, 1],
#                                                                             cx=K[0, 2],
#                                                                             cy=K[1, 2],
#                                                                             w=W,
#                                                                             h=H).transpose(0,1).cuda()
#     full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
#     camera_center = world_view_transform.inverse()[3, :3]
#     camera = MiniCam(
#         width=W,
#         height=H,
#         fovy=focal2fov(focal_length_y, H),
#         fovx=focal2fov(focal_length_x, W),
#         znear=znear,
#         zfar=zfar,
#         world_view_transform=world_view_transform,
#         full_proj_transform=full_proj_transform
#     )

#     bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
#     img_pkg = render_or(camera, gaussians, pipeline, bg_color=bg_color, orthographic=True)
#     rendered = img_pkg["render"]
#     depth = img_pkg['depth']
#     norm_depth = depth / depth.max() if depth.max() > 0 else depth
   
    
#     np.save("./output/rgb_image.npy", rendered.detach().cpu().numpy())
#     # np.save("./output/depth_image.npy", norm_depth.detach().cpu().numpy())
#     out_img = (rendered.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)


#     # # 保存为 PNG 文件
#     # # Clamp and convert to numpy uint8
#     # # Convert from (C, H, W) to (H, W, C)
#     out_img_save = out_img.transpose(1, 2, 0)

#     # Now safe to save
#     imageio.imwrite("./output/render_image.png", out_img_save)
#     return out_img.transpose(1, 2, 0)

# # ====== 绘制轨迹图 ======
# def draw_traj(camera_pose_, count):
#     #print(f"Drawing trajectory for {count} poses")
#     camera_pose_ = camera_pose_[:count, :, :]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for pose in camera_pose_:
#         R = pose[:, :3]
#         T = pose[:, 3]
#         direction = R[:, 0]
#         ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=0.1, color='blue')
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     width, height = fig.get_size_inches() * fig.get_dpi()
#     image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
#     return image

# # ====== 更新函数 ======
# def update_pose(direction, dt, pedal, steering):
#     global trans, count
#     step = 0.25
#     if direction == 'z+': trans[2, 3] += step
#     elif direction == 'z-': trans[2, 3] -= step
#     elif direction == 'y+': trans[1, 3] += step
#     elif direction == 'y-': trans[1, 3] -= step
#     elif direction == 'x+': trans[0, 3] += step
#     elif direction == 'x-': trans[0, 3] -= step
#     elif direction == 'rot_x+15':
#         angle = np.pi / 12
#         Rx = np.array([
#             [1, 0, 0],
#             [0, np.cos(angle), -np.sin(angle)],
#             [0, np.sin(angle),  np.cos(angle)]
#         ])
#         trans[:3, :3] =  Rx @ trans[:3, :3]
 
        
#     elif direction == 'rot_y+15':
#         angle = np.pi / 12
#         Ry = np.array([
#             [np.cos(angle), 0, np.sin(angle)],
#             [0, 1, 0],
#             [-np.sin(angle), 0, np.cos(angle)]
#         ])

#         trans[:3, :3] =  Ry @ trans[:3, :3] 
        

#     elif direction == 'rot_z+15':
#         angle = np.pi / 12
#         Rz = np.array([
#             [np.cos(angle), -np.sin(angle), 0],
#             [np.sin(angle),  np.cos(angle), 0],
#             [0, 0, 1]
#         ])
#         trans[:3, :3] =  Rz @ trans[:3, :3] 
 
        
#     elif direction == 'reset':
#         trans[:, :] = 0
#         bicycle_model.reset(0, 0, 0, 0, 0)
#         count = 0

#     camera_pose[count] = trans[:3, :4]
#     count += 1

#     print("[")
#     for row in trans:
#         print("    [" + ", ".join(f"{v:.6f}" for v in row) + "],")
#     print("]")

#     res_img = gen_one(trans, trans)
#     traj = draw_traj(camera_pose, count)
#     return res_img, traj

# # ====== Gradio UI ======
# iface = gr.Interface(
#     fn=update_pose,
#     inputs=[
#         gr.Radio(["z+", "z-", "y+", "y-", "x+", "x-", "rot_x+15", "rot_y+15", "rot_z+15", "reset"], label="Direction"),
#         gr.Number(label="Time Delta (dt)", value=0.1),
#         gr.Slider(label="Pedal Position (-1 to 1)", minimum=-1, maximum=1, step=0.01, value=0),
#         gr.Slider(label="Steering Angle (-30 to 30)", minimum=-30, maximum=30, step=0.1, value=0),
#     ],
#     outputs=["image", "image"],
#     live=False,
# )

# iface.queue().launch(share=True, inbrowser=True, server_name="0.0.0.0", server_port=17800)

# @atexit.register
# def cleanup_gradio_cache():
#     cache_dir = os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio")
#     if os.path.exists(cache_dir):
#         print(f"Cleaning up cache at {cache_dir}...")
#         shutil.rmtree(cache_dir, ignore_errors=True)




import math
import os
import numpy as np
import torch
import gradio as gr
import cv2
import imageio
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from arguments import PipelineParams
from argparse import ArgumentParser

from gradio_model import BicycleModel
from gaussian_renderer import render_or
from scene import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import  fov2focal, focal2fov
import shutil
import atexit
import gradio.processing_utils

gradio.processing_utils.CACHE_DIR = "./gradio_cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#============

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
#============

def getProjectionMatrixwithPrincipalPointOffset(znear, zfar, fovX, fovY, fx, fy, cx, cy, w, h):
    '''
        Reference for refleecting principal point shift to calculate the projection matrix
        http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    '''
    
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top_c = tanHalfFovY * znear
    bottom_c = -top_c
    right_c = tanHalfFovX * znear
    left_c = -right_c

    # Project difference between camera center and half of dimension to znear plane
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

# ====== 相机参数初始化 ======
W, H = 640, 480


intrinsic = np.array([
    [360.0, 0.0, 350.0],
    [0.0, 360.0, 230.0],
    [0.0, 0.0, 1.0]
])

focal_length_x = intrinsic[0, 0]
focal_length_y = intrinsic[1, 1]


# ====== 加载点云并获取中心点 ======
gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply("/home/tianhang/tianshi_gong/floor30/floor30_rot_v1.ply")
# gaussians.load_ply("/home/tianhang/NavDP/navdp_api/gaussian_splatting/data/InteriorGS/0008_840170/3dgs_compressed.ply")
center = gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()
print("Point cloud center:", center)


# ====== 初始相机状态 ======

trans = np.eye(4)

trans[:3, :3] = [[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]]
trans[:3, 3] =[5.930335998535156, -8.14944076538086, 2.289952278137207]

#[[5.930335998535156, -8.14944076538086, 2.289952278137207], [7.732640266418457, -6.1651225090026855, 2.289952278137207], [9.534944534301758, -4.150276184082031, 2.289952278137207], [10.02370548248291, 0.8563117980957031, 2.289952278137207], [8.618518829345703, 2.7490463256835938, 2.289952278137207], [7.213332176208496, 3.054326057434082, 2.289952278137207], [2.020251512527466, 3.6648855209350586, 2.24098539352417], [-0.3013609051704407, 3.6648855209350586, 2.289952278137207], [-3.203376531600952, 4.06174898147583, 2.289952278137207], [-5.922106742858887, 4.550196647644043, 2.289952278137207], [-8.121529579162598, 5.069172382354736, 2.289952278137207], [-9.007408142089844, 6.076595783233643, 2.289952278137207], [-8.21317195892334, 8.976753234863281, 2.289952278137207]]


bicycle_model = BicycleModel()
camera_pose = np.zeros([10000, 3, 4])
count = 1

# ====== 渲染单帧 ======
def gen_one(pose_trans, transformation_matrix):
    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    R = transformation_matrix[:3, :3]
    t = pose_trans[:3, 3]
    
    
    K = intrinsic

   
    zfar = 100.0
    znear = 0.01

    trans = np.array([0,0,0])
    scale = 1.0

    fovy=focal2fov(focal_length_y, H)
    fovx=focal2fov(focal_length_x, W)
    
    world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrixwithPrincipalPointOffset(znear=znear, 
                                                                            zfar=zfar, 
                                                                            fovX=fovx, 
                                                                            fovY=fovy,
                                                                            fx=K[0, 0],
                                                                            fy=K[1, 1],
                                                                            cx=K[0, 2],
                                                                            cy=K[1, 2],
                                                                            w=W,
                                                                            h=H).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    camera = MiniCam(
        width=W,
        height=H,
        fovy=focal2fov(focal_length_y, H),
        fovx=focal2fov(focal_length_x, W),
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform
    )

    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    img_pkg = render_or(camera, gaussians, pipeline, bg_color=bg_color, orthographic=True)
    rendered = img_pkg["render"]
    depth = img_pkg['depth']
    norm_depth = depth / depth.max() if depth.max() > 0 else depth
    #import pdb; pdb.set_trace()
    
    np.save("./output/rgb_image.npy", rendered.detach().cpu().numpy())
    # np.save("./output/depth_image.npy", norm_depth.detach().cpu().numpy())
    out_img = (rendered.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    # import imageio

    # # 保存为 PNG 文件
    # # Clamp and convert to numpy uint8
    # # Convert from (C, H, W) to (H, W, C)
    out_img_save = out_img.transpose(1, 2, 0)

    # Now safe to save
    imageio.imwrite("./output/render_image.png", out_img_save)
    return out_img.transpose(1, 2, 0)

# ====== 绘制轨迹图 ======
def draw_traj(camera_pose_, count):
    #print(f"Drawing trajectory for {count} poses")
    camera_pose_ = camera_pose_[:count, :, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pose in camera_pose_:
        R = pose[:, :3]
        T = pose[:, 3]
        direction = R[:, 0]
        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=0.1, color='blue')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image

# ====== 更新函数 ======
def update_pose(direction, dt, pedal, steering):
    global trans, count
    step = 0.25
    if direction == 'z+': trans[2, 3] += step
    elif direction == 'z-': trans[2, 3] -= step
    elif direction == 'y+': trans[1, 3] += step
    elif direction == 'y-': trans[1, 3] -= step
    elif direction == 'x+': trans[0, 3] += step
    elif direction == 'x-': trans[0, 3] -= step
    elif direction == 'rot_x+15':
        angle = np.pi / 12
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
        trans[:3, :3] =  Rx @ trans[:3, :3]
 
        
    elif direction == 'rot_y+15':
        angle = np.pi / 12
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

        trans[:3, :3] =  Ry @ trans[:3, :3] 
        

    elif direction == 'rot_z+15':
        angle = np.pi / 12
        Rz = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        trans[:3, :3] =  Rz @ trans[:3, :3] 
 
        
    elif direction == 'reset':
        trans[:, :] = 0
        bicycle_model.reset(0, 0, 0, 0, 0)
        count = 0


    camera_pose[count] = trans[:3, :4]
    count += 1

    print("[")
    for row in trans:
        print("    [" + ", ".join(f"{v:.6f}" for v in row) + "],")
    print("]")

    res_img = gen_one(trans, trans)
    traj = draw_traj(camera_pose, count)
    return res_img, traj

# ====== Gradio UI ======
iface = gr.Interface(
    fn=update_pose,
    inputs=[
        gr.Radio(["z+", "z-", "y+", "y-", "x+", "x-", "rot_x+15", "rot_y+15", "rot_z+15", "reset"], label="Direction"),
        gr.Number(label="Time Delta (dt)", value=0.1),
        gr.Slider(label="Pedal Position (-1 to 1)", minimum=-1, maximum=1, step=0.01, value=0),
        gr.Slider(label="Steering Angle (-30 to 30)", minimum=-30, maximum=30, step=0.1, value=0),
    ],
    outputs=["image", "image"],
    live=False,
)

iface.queue().launch(share=True, inbrowser=True, server_name="0.0.0.0", server_port=17800)

@atexit.register
def cleanup_gradio_cache():
    cache_dir = os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio")
    if os.path.exists(cache_dir):
        print(f"Cleaning up cache at {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)