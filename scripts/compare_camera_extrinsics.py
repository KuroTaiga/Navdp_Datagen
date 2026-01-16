#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def _parse_frame_index(path: Path) -> int | None:
    name = path.stem
    if not name.startswith("frame_") or not name.endswith("_camera"):
        return None
    parts = name.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _load_camera_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _matrix_from_payload(payload: dict) -> np.ndarray | None:
    matrix = payload.get("camera_to_world")
    if matrix is not None:
        return np.array(matrix, dtype=np.float64)
    matrix = payload.get("world_to_camera")
    if matrix is None:
        return None
    return np.linalg.inv(np.array(matrix, dtype=np.float64))


def _center_from_payload(payload: dict, camera_to_world: np.ndarray | None) -> np.ndarray | None:
    center = payload.get("camera_center_world")
    if center is not None:
        return np.array(center, dtype=np.float64)
    if camera_to_world is None or camera_to_world.shape[0] < 4:
        return None
    return np.array(camera_to_world[3][:3], dtype=np.float64)


def _rotation_from_camera_to_world(camera_to_world: np.ndarray | None) -> np.ndarray | None:
    if camera_to_world is None or camera_to_world.shape[0] < 3:
        return None
    return np.array(camera_to_world[:3, :3], dtype=np.float64)


def _rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a.T @ rot_b
    trace = float(np.trace(rel))
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _gather_frame_map(path_dir: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for candidate in path_dir.glob("frame_*_camera.json"):
        idx = _parse_frame_index(candidate)
        if idx is None:
            continue
        mapping[idx] = candidate
    return mapping


def _iter_scene_dirs(root: Path, scenes: set[str] | None) -> Iterable[Path]:
    if scenes:
        for scene in sorted(scenes):
            scene_dir = root / scene
            if scene_dir.is_dir():
                yield scene_dir
        return
    for scene_dir in sorted(root.iterdir()):
        if scene_dir.is_dir():
            yield scene_dir


def _iter_path_dirs(scene_dir: Path) -> Iterable[Path]:
    for entry in sorted(scene_dir.iterdir()):
        if entry.is_dir():
            yield entry


def compare_roots(
    *,
    golden_root: Path,
    candidate_root: Path,
    scenes: set[str] | None,
    top_k: int,
) -> dict:
    results: dict[str, dict] = {}
    totals = {
        "frames_compared": 0,
        "pos_err_sum": 0.0,
        "pos_err_max": 0.0,
        "rot_err_sum": 0.0,
        "rot_err_max": 0.0,
        "paths_compared": 0,
        "paths_missing_golden": 0,
        "paths_missing_candidate": 0,
    }

    for scene_dir in _iter_scene_dirs(candidate_root, scenes):
        scene_id = scene_dir.name
        golden_scene = golden_root / scene_id
        scene_entry: dict[str, dict] = {}
        if not golden_scene.is_dir():
            results[scene_id] = {"missing_in_golden": True, "paths": {}}
            totals["paths_missing_golden"] += 1
            continue

        for path_dir in _iter_path_dirs(scene_dir):
            path_id = path_dir.name
            golden_path = golden_scene / path_id
            if not golden_path.is_dir():
                scene_entry[path_id] = {"missing_in_golden": True}
                totals["paths_missing_golden"] += 1
                continue

            cand_map = _gather_frame_map(path_dir)
            gold_map = _gather_frame_map(golden_path)
            if not cand_map:
                scene_entry[path_id] = {"missing_in_candidate": True, "frames_compared": 0}
                totals["paths_missing_candidate"] += 1
                continue
            if not gold_map:
                scene_entry[path_id] = {"missing_in_golden": True, "frames_compared": 0}
                totals["paths_missing_golden"] += 1
                continue

            shared_frames = sorted(set(cand_map) & set(gold_map))
            pos_errors: list[float] = []
            rot_errors: list[float] = []
            worst_entries: list[tuple[float, int, float, float]] = []

            for frame_idx in shared_frames:
                cand_payload = _load_camera_payload(cand_map[frame_idx])
                gold_payload = _load_camera_payload(gold_map[frame_idx])
                cand_cam = _matrix_from_payload(cand_payload)
                gold_cam = _matrix_from_payload(gold_payload)
                cand_center = _center_from_payload(cand_payload, cand_cam)
                gold_center = _center_from_payload(gold_payload, gold_cam)
                cand_rot = _rotation_from_camera_to_world(cand_cam)
                gold_rot = _rotation_from_camera_to_world(gold_cam)
                if cand_center is None or gold_center is None or cand_rot is None or gold_rot is None:
                    continue
                pos_err = float(np.linalg.norm(cand_center - gold_center))
                rot_err = _rotation_angle_deg(gold_rot, cand_rot)
                pos_errors.append(pos_err)
                rot_errors.append(rot_err)
                worst_entries.append((pos_err, frame_idx, pos_err, rot_err))

            frames_compared = len(pos_errors)
            if frames_compared == 0:
                scene_entry[path_id] = {"frames_compared": 0}
                continue

            pos_err_sum = float(np.sum(pos_errors))
            rot_err_sum = float(np.sum(rot_errors))
            pos_err_max = float(np.max(pos_errors))
            rot_err_max = float(np.max(rot_errors))
            worst_entries.sort(reverse=True)
            top = [
                {
                    "frame": idx,
                    "pos_err_m": float(pos_err),
                    "rot_err_deg": float(rot_err),
                }
                for _, idx, pos_err, rot_err in worst_entries[: max(top_k, 0)]
            ]
            scene_entry[path_id] = {
                "frames_compared": frames_compared,
                "frames_missing_in_candidate": len(set(gold_map) - set(cand_map)),
                "frames_missing_in_golden": len(set(cand_map) - set(gold_map)),
                "pos_err_mean_m": pos_err_sum / frames_compared,
                "pos_err_max_m": pos_err_max,
                "rot_err_mean_deg": rot_err_sum / frames_compared,
                "rot_err_max_deg": rot_err_max,
                "worst_frames": top,
            }

            totals["frames_compared"] += frames_compared
            totals["pos_err_sum"] += pos_err_sum
            totals["rot_err_sum"] += rot_err_sum
            totals["pos_err_max"] = max(totals["pos_err_max"], pos_err_max)
            totals["rot_err_max"] = max(totals["rot_err_max"], rot_err_max)
            totals["paths_compared"] += 1

        results[scene_id] = {
            "missing_in_golden": False,
            "paths": scene_entry,
        }

    frames_compared = totals["frames_compared"]
    summary = {
        "frames_compared": frames_compared,
        "paths_compared": totals["paths_compared"],
        "paths_missing_in_golden": totals["paths_missing_golden"],
        "paths_missing_in_candidate": totals["paths_missing_candidate"],
        "pos_err_mean_m": (totals["pos_err_sum"] / frames_compared) if frames_compared else None,
        "pos_err_max_m": totals["pos_err_max"] if frames_compared else None,
        "rot_err_mean_deg": (totals["rot_err_sum"] / frames_compared) if frames_compared else None,
        "rot_err_max_deg": totals["rot_err_max"] if frames_compared else None,
    }
    return {"summary": summary, "scenes": results}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare per-frame camera extrinsics against a golden dataset."
    )
    parser.add_argument("--golden-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        default=None,
        help="Optional scene ID to compare (repeatable).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many worst frames to report per path (default: 5).",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    scenes = set(args.scenes) if args.scenes else None
    report = compare_roots(
        golden_root=args.golden_root,
        candidate_root=args.candidate_root,
        scenes=scenes,
        top_k=args.top_k,
    )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report["summary"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
