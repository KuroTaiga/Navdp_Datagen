#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
默认输出为 VLNPE 格式数据集（参考 scripts/generate_vlnpe_dataset.py）。
如需旧版“两张带箭头图片 + dataset.json/jsonl”，请加 --legacy。

Legacy 模式（--legacy）保留的旧行为：
  1) occupancy.png 颜色反转：白 -> 灰(128,128,128)；非白(含灰/黑) -> 白
  2) BEV 只逆时针旋转 90°，不变色
  3) 文件名按样本顺序统一编号：bev_map_{i:06d}.png、map_{i:06d}.png
  4) 同时输出 JSON（数组）与 JSONL（逐行）
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Any, Optional
from glob import glob
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "outputs" / "Process_65k_py" / "traning_65k"

# ================= dataset 文本构造（保留原语义） =================
def _build_sample_from_json_core(input_json_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_px = data.get("start", {}).get("pixel")
    if not (isinstance(start_px, list) and len(start_px) == 2):
        raise ValueError(f"[{input_json_path}] Invalid or missing 'start.pixel' in input JSON")
    start_x, start_y = int(start_px[0]), int(start_px[1])

    keypoints = data.get("path", {}).get("keypoints_pixel", [])
    if not (isinstance(keypoints, list) and all(isinstance(pt, list) and len(pt) == 2 for pt in keypoints)):
        raise ValueError(f"[{input_json_path}] Invalid or missing 'path.keypoints_pixel' in input JSON")

    pts = [(int(p[0]), int(p[1])) for p in keypoints]
    n_pts = len(pts)

    instruction_text = data.get("instruction", "")
    if not isinstance(instruction_text, str):
        instruction_text = str(instruction_text)

    user_content = (
        "<image><image> You are given TWO images. Image 1 is a BEV (top-down) map to help you locate the goal mentioned in the instruction. "
        "Image 2 is a NavMesh where ONLY gray pixels are traversable. The starting position is indicated by a white circle with blue triangular arrow icon in it on both image. "
        f"START at pixel (x, y) = ({start_x}, {start_y}) on Image 2 and generate a collision-free path according to the instruction: "
        f"\\\"{instruction_text}\\\". "
        "All waypoints MUST lie on gray pixels of the NavMesh (Image 2), and the FIRST waypoint MUST be exactly "
        f"({start_x}, {start_y}). Return the path as an array of integer pixel coordinates (x, y) on Image 2. "
        f"Return exactly {n_pts} points."
    )
    pts_str = ", ".join(f"({x}, {y})" for x, y in pts)
    assistant_content = f"The collision-free {n_pts} points path is: [{pts_str}]"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": []   # 占位，稍后写入真实图片相对路径（先 bev_map 再 map）
    }

# ================= 绘制 & 方向估计 =================
def load_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_by_dotted(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"Key '{part}' not found while resolving '{dotted}'")
            cur = cur[part]
        elif isinstance(cur, list):
            idx = int(part)
            if not (0 <= idx < len(cur)):
                raise IndexError(f"Index {idx} out of range while resolving '{dotted}'")
            cur = cur[idx]
        else:
            raise KeyError(f"Cannot descend into non-container while resolving '{dotted}': type={type(cur)}")
    return cur

def to_xy(pt: Any) -> Tuple[float, float]:
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    if isinstance(pt, dict):
        if "pixel" in pt:
            return to_xy(pt["pixel"])
        for kx, ky in (("x", "y"), ("u", "v"), ("col", "row"), ("w", "h")):
            if kx in pt and ky in pt:
                return float(pt[kx]), float(pt[ky])
        vals = [v for v in pt.values() if isinstance(v, (int, float))]
        if len(vals) >= 2:
            return float(vals[0]), float(vals[1])
    raise ValueError(f"Unsupported point format: {pt!r}")

def extract_path_list_strict(data: dict, key: str):
    val = get_by_dotted(data, key)
    if not isinstance(val, list) or len(val) == 0:
        raise ValueError(f"'{key}' is not a non-empty list (type={type(val)}).")
    return val

def extract_start_flexible(data: dict, explicit_key: Optional[str], path_list):
    if explicit_key:
        return to_xy(get_by_dotted(data, explicit_key))
    for cand in ("start", "start_pixel", "start_xy", "start_point", "origin", "start_uv"):
        if isinstance(data, dict) and cand in data:
            v = data[cand]
            if isinstance(v, dict) and "pixel" in v:
                return to_xy(v["pixel"])
            return to_xy(v)
    try:
        return to_xy(get_by_dotted(data, "start.pixel"))
    except Exception:
        pass
    return to_xy(path_list[0])

def initial_direction(start: Tuple[float, float], path_pts, min_seg_len: float = 1.0, window: int = 8):
    sx, sy = start
    pts = [to_xy(p) for p in path_pts if p is not None]
    if len(pts) < 2:
        return (0.0, -1.0)
    pts_w = pts[: max(2, min(window + 1, len(pts)))]
    segs = []
    for i in range(len(pts_w) - 1):
        dx = pts_w[i + 1][0] - pts_w[i][0]
        dy = pts_w[i + 1][1] - pts_w[i][1]
        L = math.hypot(dx, dy)
        if L >= min_seg_len:
            segs.append((dx, dy, L))
    if not segs:
        for x, y in pts_w[1:]:
            dx, dy = x - sx, y - sy
            if math.hypot(dx, dy) >= min_seg_len:
                return dx, dy
        return (0.0, -1.0)
    sum_x = sum((dx / L) * L for dx, _, L in segs)
    sum_y = sum((dy / L) * L for _, dy, L in segs)
    vx, vy = sum_x, sum_y
    fwd_dx = pts_w[-1][0] - pts_w[0][0]
    fwd_dy = pts_w[-1][1] - pts_w[0][1]
    if (vx * fwd_dx + vy * fwd_dy) < 0:
        vx, vy = -vx, -vy
    if math.hypot(vx, vy) < 1e-6:
        vx = fwd_dx
        vy = fwd_dy
        if math.hypot(vx, vy) < 1e-6:
            return (0.0, -1.0)
    return (vx, vy)

def rotation_degrees_for_up_to_vec(dx: float, dy: float, base_up_deg: float = 90.0, fine_offset_deg: float = 0.0) -> float:
    return math.degrees(math.atan2(dy, dx)) + base_up_deg + fine_offset_deg

def paste_center(base: Image.Image, overlay: Image.Image, center_xy: Tuple[float, float]) -> None:
    ox, oy = overlay.size
    cx, cy = center_xy
    left = int(round(cx - ox / 2))
    top = int(round(cy - oy / 2))
    left = max(min(left, base.width - 1), -ox + 1)
    top = max(min(top, base.height - 1), -oy + 1)
    base.alpha_composite(overlay, (left, top))

def _transform_occupancy_colors(scene_rgba: Image.Image) -> Image.Image:
    """
    occupancy.png 颜色反转：
      - 纯白 -> 灰(128,128,128)
      - 其它（含灰/黑等） -> 纯白
    """
    arr = np.array(scene_rgba)
    rgb, a = arr[..., :3], arr[..., 3:]
    white_mask = (rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] == 255)
    rgb[:, :, :] = 255
    rgb[white_mask] = np.array([128, 128, 128], dtype=np.uint8)
    out = np.concatenate([rgb, a], axis=-1)
    return Image.fromarray(out, mode="RGBA")

def process_one(json_path: str, scene_img: str, arrow_path: str,
                scale: float, path_key: str, start_key: Optional[str],
                min_seg_len: float, dir_window: int, fine_offset_deg: float,
                base_rotate_ccw_deg: float = 0.0, transform_fn=None) -> Image.Image:
    data = load_json(json_path)
    path_list = extract_path_list_strict(data, path_key)
    start_xy = extract_start_flexible(data, start_key, path_list)
    dx, dy = initial_direction(start_xy, path_list, min_seg_len=min_seg_len, window=dir_window)
    rot_deg = rotation_degrees_for_up_to_vec(dx, dy, base_up_deg=90.0, fine_offset_deg=fine_offset_deg)

    scene = Image.open(scene_img).convert("RGBA")
    if transform_fn is not None:
        scene = transform_fn(scene)
    if abs(base_rotate_ccw_deg) > 1e-6:
        scene = scene.rotate(base_rotate_ccw_deg, expand=True, resample=Image.BICUBIC)

    arrow = Image.open(arrow_path).convert("RGBA")
    if scale != 1.0:
        new_w = max(1, int(round(arrow.width * scale)))
        new_h = max(1, int(round(arrow.height * scale)))
        arrow = arrow.resize((new_w, new_h), resample=Image.BICUBIC)
    arrow_rot = arrow.rotate(-rot_deg, expand=True, resample=Image.BICUBIC)

    composed = scene.copy()
    paste_center(composed, arrow_rot, start_xy)
    return composed

# ================= 搜集 json =================
def collect_jsons(input_path: str):
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".json":
        return [str(p)]
    if p.is_dir():
        # 如果当前目录下直接有 json，就读它们
        jsons_here = sorted(glob(str(p / "*.json")))
        if jsons_here:
            return jsons_here

        # 否则递归查找所有子目录下的 json 文件（不限于 label_paths）
        return sorted(glob(str(p / "*" / "*.json")))
    return []


# ================= 主流程 =================
def legacy_main():
    parser = argparse.ArgumentParser(
        description="Overlay arrows and build dataset.json & dataset.jsonl with exact-matching image paths."
    )

    # 必填
    parser.add_argument("--base_dir", required=True, help="全量根目录 / 单场景目录 / label_paths 目录 / 单个 .json")

    # 你的数据根（可改）
    parser.add_argument("--scenes_root", default=str(REPO_ROOT / "scenes"),
                        help="场景底图根目录（<scene_id>/occupancy.png）")
    parser.add_argument("--bev_root", default=str(REPO_ROOT / "bev_map"),
                        help="BEV 底图根目录（<scene_id>.png）")
    parser.add_argument("--arrow", default=str(REPO_ROOT / "Arrow" / "100x100.png"),
                        help="箭头 PNG（朝上）")

    # ====== 默认输出路径（已按你的要求修改）======
    parser.add_argument("--map_outdir", default=str(DATA_ROOT / "test_dataset_30k" / "navllm"),
                        help="两类图片统一输出目录")
    parser.add_argument("--dataset_outdir", default=str(DATA_ROOT / "test_dataset_30k"),
                        help="dataset（json/jsonl）输出目录")
    # images 相对前缀（相对 dataset_outdir）
    parser.add_argument("--images_prefix", default="navllm", help="写入 dataset 的图片相对前缀")

    parser.add_argument("--outfile", default="dataset.json", help="JSON（数组）文件名")
    parser.add_argument("--outfile_jsonl", default="dataset.jsonl", help="JSON Lines 文件名")
    parser.add_argument("--path-key", default="path.raster_pixel", help="JSON 中路径列表键（点式）")
    parser.add_argument("--start-key", default=None, help="起点键（点式，如 'start.pixel'），默认自动回退")
    parser.add_argument("--scale", type=float, default=0.15, help="箭头缩放")
    parser.add_argument("--min_seg_len", type=float, default=1.0, help="方向估计最短段长度")
    parser.add_argument("--dir_window", type=int, default=8, help="方向估计窗口点数")
    parser.add_argument("--fine_offset_deg", type=float, default=0.0, help="微调角（度）")

    # 统一编号起点
    parser.add_argument("--start-index", type=int, default=1,
                        help="样本编号起始（用于 bev_map_000001.png / map_000001.png 的 1）")

    parser.add_argument("--verbose", action="store_true", help="打印处理细节")
    args = parser.parse_args()

    os.makedirs(args.map_outdir, exist_ok=True)
    os.makedirs(args.dataset_outdir, exist_ok=True)

    json_list = collect_jsons(args.base_dir)

    # 稳定排序
    def sort_key(p: str):
        parts = Path(p).parts
        try:
            idx = parts.index("label_paths")
            scene_id = parts[idx - 1] if idx - 1 >= 0 else ""
        except ValueError:
            scene_id = Path(p).parent.parent.name
        return (scene_id, os.path.basename(p))
    json_list.sort(key=sort_key)

    samples = []
    saved_occ = 0
    saved_bev = 0
    cur_index = max(1, int(args.start_index))

    for jp in json_list:
        idx_str = f"{cur_index:06d}"

        # 用 scene_id 找到底图（命名不再用）
        parts = Path(jp).parts
        try:
            if "label_paths" in parts:
                ix = parts.index("label_paths")
                scene_id = parts[ix - 1] if ix - 1 >= 0 else Path(jp).parent.parent.name
            else:
                scene_id = Path(jp).parent.name
        except Exception:
            scene_id = Path(jp).parent.name


        occ_img = os.path.join(args.scenes_root, scene_id, "occupancy.png")
        bev_img = os.path.join(args.bev_root, f"{scene_id}.png")

        out_occ = None
        out_bev = None

        # 1) occupancy（颜色反转）→ map_编号.png
        if os.path.isfile(occ_img):
            try:
                composed_occ = process_one(
                    json_path=jp, scene_img=occ_img, arrow_path=args.arrow,
                    scale=args.scale, path_key=args.path_key, start_key=args.start_key,
                    min_seg_len=args.min_seg_len, dir_window=args.dir_window,
                    fine_offset_deg=args.fine_offset_deg, base_rotate_ccw_deg=0.0,
                    transform_fn=_transform_occupancy_colors,
                )
                out_occ = os.path.join(args.map_outdir, f"map_{idx_str}.png")
                if os.path.exists(out_occ) and args.verbose:
                    print(f"[WARN] 将覆盖已存在文件: {out_occ}")
                composed_occ.save(out_occ)
                saved_occ += 1
                if args.verbose:
                    print(f"[OCC OK] {jp} -> {out_occ}")
            except Exception as e:
                if args.verbose:
                    print(f"[OCC 失败] {jp}: {e}")
        else:
            if args.verbose:
                print(f"[OCC 跳过] 找不到底图: {occ_img}")

        # 2) BEV（旋转 90°）→ bev_map_编号.png
        if os.path.isfile(bev_img):
            try:
                composed_bev = process_one(
                    json_path=jp, scene_img=bev_img, arrow_path=args.arrow,
                    scale=args.scale, path_key=args.path_key, start_key=args.start_key,
                    min_seg_len=args.min_seg_len, dir_window=args.dir_window,
                    fine_offset_deg=args.fine_offset_deg, base_rotate_ccw_deg=90.0,
                    transform_fn=None,
                )
                out_bev = os.path.join(args.map_outdir, f"bev_map_{idx_str}.png")
                if os.path.exists(out_bev) and args.verbose:
                    print(f"[WARN] 将覆盖已存在文件: {out_bev}")
                composed_bev.save(out_bev)
                saved_bev += 1
                if args.verbose:
                    print(f"[BEV OK] {jp} -> {out_bev}")
            except Exception as e:
                if args.verbose:
                    print(f"[BEV 失败] {jp}: {e}")
        else:
            if args.verbose:
                print(f"[BEV 跳过] 找不到底图: {bev_img}")

        # 3) dataset 条目（先 bev_map 再 map；写相对 prefix）
        try:
            sample = _build_sample_from_json_core(jp)
            bev_rel = os.path.join(args.images_prefix, f"bev_map_{idx_str}.png").replace("\\", "/")
            occ_rel = os.path.join(args.images_prefix, f"map_{idx_str}.png").replace("\\", "/")

            images_list = []
            if out_bev and os.path.isfile(out_bev):
                images_list.append(bev_rel)
            if out_occ and os.path.isfile(out_occ):
                images_list.append(occ_rel)

            sample["images"] = images_list
            samples.append(sample)
        except Exception as e:
            if args.verbose:
                print(f"[SAMPLE 跳过] {jp}: {e}")

        cur_index += 1

    # 4) 输出 dataset.json / dataset.jsonl
    if not samples:
        print("[Error] 未生成任何样本，已退出。")
        sys.exit(2)

    out_json = os.path.join(args.dataset_outdir, args.outfile)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    out_jsonl = os.path.join(args.dataset_outdir, args.outfile_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")

    print(f"[OK] map 输出 {saved_occ} 张，bev_map 输出 {saved_bev} 张；样本 {len(samples)} 条")
    print(f"JSON  输出：{out_json}")
    print(f"JSONL 输出：{out_jsonl}")
    print("提示：images 为相对路径 '{args.images_prefix}/...'，顺序固定为先 bev_map 再 map。")

def main():
    if "--legacy" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--legacy"]
        return legacy_main()
    try:
        from generate_vlnpe_dataset import main as vlnpe_main
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to load VLNPE generator: {exc}") from exc
    return vlnpe_main()


if __name__ == "__main__":
    main()
