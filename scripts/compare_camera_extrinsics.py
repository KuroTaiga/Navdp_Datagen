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


def _extract_pose_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray] | None:
    cam = _matrix_from_payload(payload)
    center = _center_from_payload(payload, cam)
    rot = _rotation_from_camera_to_world(cam)
    if center is None or rot is None:
        return None
    return center, rot


def _extract_extrinsics(payload: dict) -> dict:
    return {
        "camera_center_world": payload.get("camera_center_world"),
        "camera_to_world": payload.get("camera_to_world"),
        "world_to_camera": payload.get("world_to_camera"),
    }


def _rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a.T @ rot_b
    trace = float(np.trace(rel))
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _position_error(cand_center: np.ndarray, gold_center: np.ndarray) -> float:
    # Ignore Z because camera heights differ between datasets (e.g., 1.0m vs 1.3m),
    # so we only care about XY drift when validating frame ordering.
    delta = cand_center - gold_center
    if delta.shape[0] >= 3:
        delta = delta.copy()
        delta[2] = 0.0
    return float(np.linalg.norm(delta))


def _gather_frame_payload_map(path_source: Path) -> dict[int, dict]:
    """
    Return per-frame payloads for a path source.

    Supported formats:
      - New: <scene>/{label}_camera.json, with {"frames": [{"frame": i, ...camera fields...}, ...]}
      - Legacy: <scene>/<label>/frame_XXXX_camera.json per frame
    """
    mapping: dict[int, dict] = {}
    if path_source.is_file() and path_source.name.endswith("_camera.json"):
        payload = _load_camera_payload(path_source)
        frames = payload.get("frames") or []
        if isinstance(frames, list):
            for entry in frames:
                if not isinstance(entry, dict):
                    continue
                try:
                    idx = int(entry.get("frame", 0))
                except Exception:
                    continue
                mapping[idx] = entry
        return mapping

    if path_source.is_dir():
        for candidate in path_source.glob("frame_*_camera.json"):
            idx = _parse_frame_index(candidate)
            if idx is None:
                continue
            mapping[idx] = _load_camera_payload(candidate)
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


def _path_id_from_source(path_source: Path) -> str:
    if path_source.is_file() and path_source.name.endswith("_camera.json"):
        stem = path_source.stem
        if stem.endswith("_camera"):
            return stem[: -len("_camera")]
        return stem
    return path_source.name


def _resolve_path_source(scene_dir: Path, path_id: str) -> Path | None:
    candidate_json = scene_dir / f"{path_id}_camera.json"
    if candidate_json.is_file():
        return candidate_json
    candidate_dir = scene_dir / path_id
    if candidate_dir.is_dir():
        return candidate_dir
    return None


def _iter_path_sources(scene_dir: Path) -> Iterable[Path]:
    camera_jsons = sorted(p for p in scene_dir.glob("*_camera.json") if p.is_file())
    if camera_jsons:
        yield from camera_jsons
        return
    for entry in sorted(scene_dir.iterdir()):
        if entry.is_dir():
            yield entry


def compare_roots(
    *,
    golden_root: Path,
    candidate_root: Path,
    scenes: set[str] | None,
    top_k: int,
    search_window: int,
    window_rot_weight: float,
    include_frames: bool,
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
    window_totals = {
        "frames_compared": 0,
        "frames_missing": 0,
        "pos_err_sum": 0.0,
        "pos_err_max": 0.0,
        "rot_err_sum": 0.0,
        "rot_err_max": 0.0,
        "offset_histogram": {},
        "nonzero_offsets": 0,
    }

    for scene_dir in _iter_scene_dirs(candidate_root, scenes):
        scene_id = scene_dir.name
        golden_scene = golden_root / scene_id
        scene_entry: dict[str, dict] = {}
        if not golden_scene.is_dir():
            results[scene_id] = {"missing_in_golden": True, "paths": {}}
            totals["paths_missing_golden"] += 1
            continue

        for path_source in _iter_path_sources(scene_dir):
            path_id = _path_id_from_source(path_source)
            golden_source = _resolve_path_source(golden_scene, path_id)
            if golden_source is None:
                scene_entry[path_id] = {"missing_in_golden": True}
                totals["paths_missing_golden"] += 1
                continue

            cand_payloads = _gather_frame_payload_map(path_source)
            gold_payloads = _gather_frame_payload_map(golden_source)
            if not cand_payloads:
                scene_entry[path_id] = {"missing_in_candidate": True, "frames_compared": 0}
                totals["paths_missing_candidate"] += 1
                continue
            if not gold_payloads:
                scene_entry[path_id] = {"missing_in_golden": True, "frames_compared": 0}
                totals["paths_missing_golden"] += 1
                continue

            shared_frames = sorted(set(cand_payloads) & set(gold_payloads))
            pos_errors: list[float] = []
            rot_errors: list[float] = []
            worst_entries: list[tuple[float, int, float, float]] = []
            frames_detail: list[dict] = []
            cand_pose = {idx: _extract_pose_from_payload(payload) for idx, payload in cand_payloads.items()}
            gold_pose = {idx: _extract_pose_from_payload(payload) for idx, payload in gold_payloads.items()}

            for frame_idx in shared_frames:
                cand_data = cand_pose.get(frame_idx)
                gold_data = gold_pose.get(frame_idx)
                if cand_data is None or gold_data is None:
                    continue
                cand_center, cand_rot = cand_data
                gold_center, gold_rot = gold_data
                pos_err = _position_error(cand_center, gold_center)
                rot_err = _rotation_angle_deg(gold_rot, cand_rot)
                pos_errors.append(pos_err)
                rot_errors.append(rot_err)
                worst_entries.append((pos_err, frame_idx, pos_err, rot_err))
                if include_frames:
                    frames_detail.append(
                        {
                            "frame": frame_idx,
                            "status": "compared",
                            "pos_err_m": float(pos_err),
                            "rot_err_deg": float(rot_err),
                            "candidate": _extract_extrinsics(cand_payloads.get(frame_idx, {})),
                            "golden": _extract_extrinsics(gold_payloads.get(frame_idx, {})),
                        }
                    )

            frames_compared = len(pos_errors)
            if frames_compared == 0:
                empty_entry = {"frames_compared": 0}
                if include_frames:
                    all_frames = sorted(set(cand_payloads) | set(gold_payloads))
                    for frame_idx in all_frames:
                        status = "compared"
                        cand_payload = cand_payloads.get(frame_idx, {})
                        gold_payload = gold_payloads.get(frame_idx, {})
                        if frame_idx not in cand_payloads:
                            status = "candidate_missing"
                        elif frame_idx not in gold_payloads:
                            status = "golden_missing"
                        frames_detail.append(
                            {
                                "frame": frame_idx,
                                "status": status,
                                "candidate": _extract_extrinsics(cand_payload),
                                "golden": _extract_extrinsics(gold_payload),
                            }
                        )
                    empty_entry["frames"] = frames_detail
                scene_entry[path_id] = empty_entry
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
            path_entry = {
                "frames_compared": frames_compared,
                "frames_missing_in_candidate": len(set(gold_map) - set(cand_map)),
                "frames_missing_in_golden": len(set(cand_map) - set(gold_map)),
                "pos_err_mean_m": pos_err_sum / frames_compared,
                "pos_err_max_m": pos_err_max,
                "rot_err_mean_deg": rot_err_sum / frames_compared,
                "rot_err_max_deg": rot_err_max,
                "worst_frames": top,
            }
            if include_frames:
                if frames_detail:
                    compared_frames = {entry["frame"] for entry in frames_detail}
                else:
                    compared_frames = set()
                all_frames = sorted(set(cand_map) | set(gold_map))
                for frame_idx in all_frames:
                    if frame_idx in compared_frames:
                        continue
                    status = "compared"
                    cand_payload = cand_payloads.get(frame_idx, {})
                    gold_payload = gold_payloads.get(frame_idx, {})
                    if frame_idx not in cand_map:
                        status = "candidate_missing"
                    elif frame_idx not in gold_map:
                        status = "golden_missing"
                    frames_detail.append(
                        {
                            "frame": frame_idx,
                            "status": status,
                            "candidate": _extract_extrinsics(cand_payload),
                            "golden": _extract_extrinsics(gold_payload),
                        }
                    )
                path_entry["frames"] = sorted(frames_detail, key=lambda item: item["frame"])
            scene_entry[path_id] = path_entry
            if search_window > 0:
                window_matches = []
                max_offset = max(0, int(search_window))
                for cand_idx in sorted(cand_map.keys()):
                    cand_data = cand_pose.get(cand_idx)
                    if cand_data is None:
                        window_matches.append(
                            {
                                "frame": cand_idx,
                                "status": "candidate_missing",
                            }
                        )
                        window_totals["frames_missing"] += 1
                        continue
                    cand_center, cand_rot = cand_data
                    best = None
                    best_score = None
                    best_pos = None
                    best_rot = None
                    best_offset = None
                    for offset in range(-max_offset, max_offset + 1):
                        gold_idx = cand_idx + offset
                        gold_data = gold_pose.get(gold_idx)
                        if gold_data is None:
                            continue
                        gold_center, gold_rot = gold_data
                        pos_err = _position_error(cand_center, gold_center)
                        rot_err = _rotation_angle_deg(gold_rot, cand_rot)
                        score = pos_err + (window_rot_weight * rot_err)
                        if best_score is None or score < best_score:
                            best_score = score
                            best = gold_idx
                            best_pos = pos_err
                            best_rot = rot_err
                            best_offset = offset
                    if best is None:
                        window_matches.append(
                            {
                                "frame": cand_idx,
                                "status": "golden_missing",
                            }
                        )
                        window_totals["frames_missing"] += 1
                        continue
                    window_matches.append(
                        {
                            "frame": cand_idx,
                            "best_match": best,
                            "offset": int(best_offset),
                            "pos_err_m": float(best_pos),
                            "rot_err_deg": float(best_rot),
                        }
                    )
                    window_totals["frames_compared"] += 1
                    window_totals["pos_err_sum"] += float(best_pos)
                    window_totals["rot_err_sum"] += float(best_rot)
                    window_totals["pos_err_max"] = max(window_totals["pos_err_max"], float(best_pos))
                    window_totals["rot_err_max"] = max(window_totals["rot_err_max"], float(best_rot))
                    if best_offset != 0:
                        window_totals["nonzero_offsets"] += 1
                    hist = window_totals["offset_histogram"]
                    key = str(int(best_offset))
                    hist[key] = int(hist.get(key, 0)) + 1

                scene_entry[path_id]["window_search"] = {
                    "window": max_offset,
                    "rot_weight": window_rot_weight,
                    "frames_compared": sum(1 for m in window_matches if m.get("best_match") is not None),
                    "frames_missing": sum(1 for m in window_matches if m.get("best_match") is None),
                    "offset_histogram": window_totals["offset_histogram"],
                    "matches": window_matches,
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
    report = {"summary": summary, "scenes": results}
    if search_window > 0:
        window_frames = window_totals["frames_compared"]
        report["window_summary"] = {
            "window": int(search_window),
            "rot_weight": float(window_rot_weight),
            "frames_compared": window_frames,
            "frames_missing": window_totals["frames_missing"],
            "pos_err_mean_m": (window_totals["pos_err_sum"] / window_frames) if window_frames else None,
            "pos_err_max_m": window_totals["pos_err_max"] if window_frames else None,
            "rot_err_mean_deg": (window_totals["rot_err_sum"] / window_frames) if window_frames else None,
            "rot_err_max_deg": window_totals["rot_err_max"] if window_frames else None,
            "offset_histogram": window_totals["offset_histogram"],
            "nonzero_offsets": window_totals["nonzero_offsets"],
        }
    return report


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
    parser.add_argument(
        "--search-window",
        type=int,
        default=0,
        help="Search +/- N frames around each candidate frame for best match (default: 0).",
    )
    parser.add_argument(
        "--window-rot-weight",
        type=float,
        default=0.0,
        help="Rotation error weight when selecting best match in search window (default: 0).",
    )
    parser.add_argument(
        "--per-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include per-frame comparison entries in the output JSON (default: off).",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    scenes = set(args.scenes) if args.scenes else None
    report = compare_roots(
        golden_root=args.golden_root,
        candidate_root=args.candidate_root,
        scenes=scenes,
        top_k=args.top_k,
        search_window=int(args.search_window),
        window_rot_weight=float(args.window_rot_weight),
        include_frames=bool(args.per_frame),
    )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report["summary"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
