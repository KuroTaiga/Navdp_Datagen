import numpy as np

# 1. 读取 camera_pose.npy 文件
npy_path = "./gradio_data/camera_pose.npy"  # <-- 替换成你的实际路径
poses = np.load(npy_path)

# 2. 输出文件路径
output_txt = "./gradio_data/camera_poses_readable.txt"

# 3. 写入为人类可读格式
with open(output_txt, "w") as f:
    for i, pose in enumerate(poses):
        if np.all(pose == 0):  # 跳过全为 0 的空 pose（未初始化）
            continue
        f.write(f"--- Pose {i} ---\n")
        f.write("Rotation (3x3):\n")
        for row in pose[:, :3]:
            f.write("  " + " ".join(f"{v:.6f}" for v in row) + "\n")
        f.write("Translation (x y z):\n")
        translation = pose[:, 3]
        f.write("  " + " ".join(f"{v:.6f}" for v in translation) + "\n")
        f.write("\n")

print(f"Saved readable poses to: {output_txt}")