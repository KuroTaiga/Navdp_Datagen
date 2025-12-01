import os
import json
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from pathlib import Path

def rotate_image_180(image):
    return cv2.rotate(image, cv2.ROTATE_180)

def flip_image_vertical(image):
    # 上下镜像
    return cv2.flip(image, 0)

def project_scene_to_bev(scene_dir, output_dir):
    occ_png_path = scene_dir / "occupancy.png"
    occ_json_path = scene_dir / "occupancy.json"
    ply_path = next(scene_dir.glob("*.ply"), None)
    scene_id = scene_dir.name
    
    if not occ_png_path.exists():
        print(f"Warning: {scene_id} missing occupancy.png, skipped")
        return
    if not occ_json_path.exists():
        print(f"Warning: {scene_id} missing occupancy.json, skipped")
        return
    if not ply_path:
        print(f"Warning: {scene_id} missing PLY file, skipped")
        return

    output_dir = Path(output_dir) / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(occ_json_path, 'r') as f:
        occ_data = json.load(f)
    
    scale = occ_data["scale"]
    min_x, min_y = occ_data["min"][0], occ_data["min"][1]
    max_x, max_y = occ_data["max"][0], occ_data["max"][1]
    
    bev_width = int((max_x - min_x) / scale)
    bev_height = int((max_y - min_y) / scale)

    mask = cv2.imread(str(occ_png_path))
    #mask = flip_image_vertical(mask)
    #mask = rotate_image_180(mask)
    
    if mask.shape[0] != bev_height or mask.shape[1] != bev_width:
        mask = cv2.resize(mask, (bev_width, bev_height), interpolation=cv2.INTER_NEAREST)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    h, w = mask.shape[:2]

    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex']
    xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)

    xy = xyz[:, :2]
    
    #u = np.round((xy[:, 0] - min_x) / scale).astype(np.int64)
    u = np.round((max_x - xy[:, 0]) / scale).astype(np.int64)
    v = np.round((xy[:, 1] - min_y) / scale).astype(np.int64)
    #v = np.round((max_y - xy[:, 1]) / scale).astype(np.int64)

    
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_valid = u[in_bounds]
    v_valid = v[in_bounds]
    xy_valid = xy[in_bounds]

    mask_samples = mask[v_valid, u_valid]
    is_white = np.all(mask_samples >= 250, axis=1)
    xy_kept = xy_valid[~is_white]

    ply_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    ply_array = np.empty(len(xy_kept), dtype=ply_dtype)
    ply_array['x'] = xy_kept[:, 0]
    ply_array['y'] = xy_kept[:, 1]
    ply_array['z'] = 0.0
    ply_array['red'] = 0
    ply_array['green'] = 0
    ply_array['blue'] = 255

    ply_element = PlyElement.describe(ply_array, 'vertex')
    output_ply = output_dir / f"{scene_id}_bev.ply"
    PlyData([ply_element], text=False).write(output_ply)

    bev_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    bev_image_bgr = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR)
    bev_image_bgr[v_valid[~is_white], u_valid[~is_white]] = [255, 0, 0]
    
    output_png = output_dir / f"{scene_id}_bev.png"
    cv2.imwrite(str(output_png), bev_image_bgr)

    print(f"Processed: {scene_id} -> {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Project scenes to BEV view and save results")
    parser.add_argument("--input-dir", default="./data/scenes", help="Root directory of scenes")
    parser.add_argument("--output-dir", default="/mnt/nas/jiankundong/scenes_projection_debug", help="Output directory")
    parser.add_argument("--max-scenes", type=int, help="Maximum number of scenes to process (optional)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    scene_dirs.sort()

    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]

    print(f"Starting processing {len(scene_dirs)} scenes...")
    for scene_dir in scene_dirs:
        project_scene_to_bev(scene_dir, args.output_dir)
    print("All scenes processed successfully")

if __name__ == "__main__":
    main()
