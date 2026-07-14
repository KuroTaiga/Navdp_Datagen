#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 VLNPE/LeRobot 格式数据集转换为 NavLLM 格式
每帧生成一条记录：单张图片 + 接下来5步action
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import pandas as pd


def rgb_to_pil(rgb_array: np.ndarray) -> Image.Image:
    if rgb_array.ndim == 4:
        rgb_array = rgb_array[0]
    if rgb_array.ndim == 3:
        rgb_array = rgb_array[0] if rgb_array.shape[0] == 3 else rgb_array
    
    if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
        rgb_array = (rgb_array * 255).astype(np.uint8)
    
    if rgb_array.max() > 255:
        rgb_array = (rgb_array / rgb_array.max() * 255).astype(np.uint8)
    
    if len(rgb_array.shape) == 2:
        rgb_array = np.stack([rgb_array] * 3, axis=-1)
    
    return Image.fromarray(rgb_array)


def process_episode(
    episode_dir: Path,
    scene_id: str,
    episode_index: int,
    output_dir: Path,
    num_future_actions: int = 5
) -> List[Dict]:
    try:
        meta_dir = episode_dir / "meta"
        data_dir = episode_dir / "data" / "chunk-000"
        video_dir = episode_dir / "videos" / "chunk-000" / "observation.images.rgb"
        
        if not meta_dir.exists() or not video_dir.exists():
            return []
        
        episodes_jsonl = meta_dir / "episodes.jsonl"
        if not episodes_jsonl.exists():
            return []
        
        with open(episodes_jsonl, 'r') as f:
            lines = f.readlines()
        
        if episode_index >= len(lines):
            return []
        
        ep_info = json.loads(lines[episode_index].strip())
        instruction_text = ep_info.get('instruction_text', 'Navigate to the goal.')
        finish_status = ep_info.get('finish_status', 'unknown')
        fail_reason = ep_info.get('fail_reason', '')
        episode_id = ep_info.get('episode_id', str(episode_index))
        
        rgb_npy = video_dir / "rgb.npy"
        if not rgb_npy.exists():
            return []
        
        rgb_frames = np.load(str(rgb_npy))
        
        parquet_files = list(data_dir.glob("episode_*.parquet"))
        if not parquet_files:
            return []
        
        df = pd.read_parquet(str(parquet_files[0]))
        
        if episode_index not in df['episode_index'].unique():
            return []
        
        df = df[df['episode_index'] == episode_index]
        
        if 'observation.robot_position' in df.columns:
            positions = np.array([p for p in df['observation.robot_position'].tolist()])
        elif 'observation.camera_position' in df.columns:
            positions = np.array([p for p in df['observation.camera_position'].tolist()])
        else:
            print(f"[Debug] 可用列: {df.columns.tolist()}")
            return []
        
        actions = df['observation.action'].tolist() if 'observation.action' in df.columns else []
        
        if rgb_frames.ndim == 4:
            num_frames = rgb_frames.shape[0]
        else:
            num_frames = 1
        
        samples = []
        
        for frame_idx in range(num_frames):
            img_filename = f"{scene_id}_{episode_id}_{frame_idx:04d}.jpg"
            img_output_path = output_dir / "navllm" / img_filename
            
            if rgb_frames.ndim == 4:
                frame = rgb_frames[frame_idx]
            else:
                frame = rgb_frames
            
            img = rgb_to_pil(frame)
            img.save(img_output_path)
            
            rel_path = str(Path("navllm") / img_filename)
            
            start_x, start_y = int(positions[frame_idx][0]), int(positions[frame_idx][1])
            
            future_actions = []
            for i in range(num_future_actions):
                action_idx = frame_idx + i
                if action_idx < len(actions):
                    future_actions.append(str(actions[action_idx]))
                else:
                    future_actions.append("0")
            
            actions_str = ",".join(future_actions)
            
            user_content = (
                f"<image> You are given a FIRST-PERSON VIEW image. "
                f"Based on this image and the instruction: \"{instruction_text}\", "
                f"predict the next {num_future_actions} navigation actions. "
                f"Return ONLY the sequence of {num_future_actions} actions."
            )
            
            assistant_content = f"<action>{actions_str}</action>"
            
            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [rel_path],
                "metadata": {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "finish_status": finish_status,
                    "fail_reason": fail_reason,
                    "num_frames": num_frames,
                    "num_samples": num_frames,
                    "current_frame": frame_idx,
                    "future_actions_count": len(future_actions)
                }
            }
            
            samples.append(sample)
        
        return samples
        
    except Exception as e:
        print(f"[Error] 处理失败 {scene_id}/{episode_index}: {e}")
        import traceback
        traceback.print_exc()
        return []


def convert_vlnpe_to_navllm(
    input_dir: str,
    output_dir: str,
    max_episodes: Optional[int] = None,
    filter_success_only: bool = False,
    num_future_actions: int = 5
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    images_dir = output_path / "navllm"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    scene_dirs = []
    for subdir in input_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            digit_dirs = [d for d in subdir.iterdir() if d.is_dir() and d.name.isdigit()]
            if digit_dirs:
                scene_dirs.append(subdir)
    
    print(f"[Info] 找到 {len(scene_dirs)} 个场景")
    
    samples = []
    processed = 0
    skipped = 0
    
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        episode_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        
        print(f"[Info] 场景 {scene_id}: {len(episode_dirs)} 个轨迹目录")
        
        for episode_dir in sorted(episode_dirs):
            print(f"[Debug] 处理 {episode_dir.name}...", flush=True)
            try:
                episode_samples = process_episode(
                    episode_dir=episode_dir,
                    scene_id=scene_id,
                    episode_index=0,
                    output_dir=output_path,
                    num_future_actions=num_future_actions
                )
            except Exception as e:
                print(f"[Error] 异常: {e}")
                import traceback
                traceback.print_exc()
                episode_samples = []
            
            if episode_samples:
                for sample in episode_samples:
                    if filter_success_only and sample['metadata']['finish_status'] != 'success':
                        skipped += 1
                        continue
                    samples.append(sample)
                    processed += 1
                
                print(f"[Debug] 成功 - 共 {processed} 条记录", flush=True)
                
                if processed % 500 == 0:
                    print(f"[Progress] 已处理 {processed} 条记录")
                
                if max_episodes is not None and max_episodes > 0 and processed >= max_episodes:
                    print(f"[Debug] 达到最大数量限制: {max_episodes}")
                    break
            else:
                skipped += 1
                print(f"[Debug] 失败 - 共 {skipped} 个跳过", flush=True)
        
        print(f"[Debug] 场景 {scene_id} 完成", flush=True)
        
        if max_episodes is not None and max_episodes > 0 and processed >= max_episodes:
            break
    
    print(f"[Complete] 处理完成: {processed} 条记录, {skipped} 个跳过")
    
    if not samples:
        print("[Error] 未生成任何样本")
        return
    
    output_json = output_path / "dataset.json"
    output_jsonl = output_path / "dataset.jsonl"
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write('\n')
    
    print(f"[OK] 输出文件:")
    print(f"  - JSON: {output_json}")
    print(f"  - JSONL: {output_jsonl}")
    print(f"  - 图片: {images_dir}")
    print(f"  - 总样本数: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="将 VLNPE/LeRobot 格式数据集转换为 NavLLM 格式（单张图片 + 未来N步action）"
    )
    
    parser.add_argument(
        "--input_dir", 
        required=True,
        help="VLNPE/LeRobot 格式数据集根目录"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="最大处理的轨迹数（默认: 全部）"
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="只保留成功完成的轨迹"
    )
    parser.add_argument(
        "--future_actions",
        type=int,
        default=5,
        help="每帧预测的未来action数量（默认: 5）"
    )
    
    args = parser.parse_args()
    
    convert_vlnpe_to_navllm(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        filter_success_only=args.success_only,
        num_future_actions=args.future_actions
    )


if __name__ == "__main__":
    main()
