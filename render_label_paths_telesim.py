#!/usr/bin/env python3
"""Render NavDP label paths with TeleSim3D's Gaussian renderer.

This is a lightweight TeleSim3D-backed alternative to render_label_paths.py.
It mirrors NavDP path preprocessing (affine mapping, stride, mirroring) and
writes MP4 + optional camera metadata for quick pipeline validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

_script_path = Path(__file__).absolute()
REPO_ROOT = _script_path.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TELESIM_ROOT = REPO_ROOT / "TeleSim3D"
if str(TELESIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TELESIM_ROOT))
TELESIM_GAUSSIAN_ROOT = TELESIM_ROOT / "gaussian-splatting"
if TELESIM_GAUSSIAN_ROOT.exists():
    sys.path.insert(0, str(TELESIM_GAUSSIAN_ROOT))

from utils.render_utils import (  # type: ignore
    build_look_at,
    deduplicate_points,
    derive_affine_transform,
    load_occupancy_metadata,
    load_raster_world_points,
    sample_points,
)
from utils.ply_transform_utils import apply_transform_to_frame, build_transform_matrix, rotation_matrix_z_np
from lighting.lighting_utils import LightFilterConfig, apply_light_filter, stable_hash_seed
from lighting.shading import CameraLightConfig, apply_camera_light_shading, intrinsics_from_camera
from tele_sim.core.viewer import Pose
from tele_sim.rendering import GaussianRendererBackend, GaussianRendererConfig
from tele_sim.rendering.gaussian_transform import GaussianPly, _matrix_to_quat_wxyz  # type: ignore
from tele_sim.scene.assets import SceneAsset
from gaussian_renderer import render as render_gaussians  # type: ignore
from utils.telesim_actor_utils import (  # type: ignore
    ActorOptions,
    ActorRuntime,
    CombinedGaussianModel,
    DEFAULT_ACTOR_PATTERN,
    DEFAULT_ACTOR_SPEED,
    DEFAULT_VIDEO_FPS,
    build_path_metadata,
    load_actor_sequence,
    actor_data_to_tensors,
)

LOGGER = logging.getLogger("render_label_paths_telesim")
logging.basicConfig(level=logging.INFO, format="%(message)s")

STABILIZE_WINDOW = 5
FORWARD_SMOOTH_BLEND = 0.35
EPS = 1e-6
STATUS_NOT_RUN = 0
STATUS_DONE = 1
STATUS_RETRY = 2
STATUS_SKIP = 3


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days:
        return f"{days}d{hrs:02d}h{mins:02d}m{sec:02d}s"
    if hrs:
        return f"{hrs}h{mins:02d}m{sec:02d}s"
    return f"{mins}m{sec:02d}s"


def _format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "-"
    try:
        value = float(num_bytes)
    except (TypeError, ValueError):
        return "-"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f}{units[idx]}"


def _dir_size_bytes(path: Path) -> int | None:
    """Best-effort directory size (fast path via du, fallback via stat walk)."""
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return int(out.split()[0])
    except Exception:
        pass
    try:
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total
    except Exception:
        return None


def _is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or "cublas_status_alloc_failed" in message
        or "cudnn_status_alloc_failed" in message
        or "out of memory" in message
    )


def _apply_light_filter_if_enabled(
    render: np.ndarray,
    light_config: LightFilterConfig | None,
    *,
    frame_index: int,
    seed_offset: int,
) -> np.ndarray:
    if light_config is None or not light_config.enabled():
        return render
    return apply_light_filter(
        render,
        light_config,
        frame_index=frame_index,
        seed_offset=seed_offset,
    )


def _apply_camera_light_if_enabled(
    render: np.ndarray,
    depth_inv: np.ndarray | None,
    camera,
    cl_config: CameraLightConfig | None,
    *,
    cl_light_world: np.ndarray | None,
) -> np.ndarray:
    if cl_config is None or not cl_config.active():
        return render
    if depth_inv is None:
        return render
    fx, fy, cx, cy = intrinsics_from_camera(camera)
    frame_config = cl_config
    if cl_light_world is not None:
        cam_to_world = torch.inverse(camera.world_view_transform).detach().cpu().numpy()
        world_to_cam_rot = cam_to_world[:3, :3].T
        cam_center = camera.camera_center.detach().cpu().numpy()
        offset_cam = world_to_cam_rot @ (cl_light_world - cam_center)
        frame_config = CameraLightConfig(
            enabled=cl_config.enabled,
            strength=cl_config.strength,
            color=cl_config.color,
            ambient=cl_config.ambient,
            diffuse=cl_config.diffuse,
            specular=cl_config.specular,
            shininess=cl_config.shininess,
            range_m=cl_config.range_m,
            offset_cam=(float(offset_cam[0]), float(offset_cam[1]), float(offset_cam[2])),
            normal_smooth=cl_config.normal_smooth,
            shadow_enabled=cl_config.shadow_enabled,
            shadow_bias=cl_config.shadow_bias,
            shadow_strength=cl_config.shadow_strength,
            shadow_pcf_radius=cl_config.shadow_pcf_radius,
            light_mode=cl_config.light_mode,
            shading_model=cl_config.shading_model,
            shadow_compare=cl_config.shadow_compare,
            normal_filter=cl_config.normal_filter,
            normal_kernel=cl_config.normal_kernel,
            normal_sigma_range=cl_config.normal_sigma_range,
            normal_sigma_domain=cl_config.normal_sigma_domain,
            base_scale=cl_config.base_scale,
            light_reverse=cl_config.light_reverse,
        )
    return apply_camera_light_shading(
        render,
        depth_inv,
        config=frame_config,
        camera_fx=fx,
        camera_fy=fy,
        camera_cx=cx,
        camera_cy=cy,
    )


class PathSampler:
    """Lightweight sampler to query positions/tangents along a 2D polyline."""

    def __init__(self, points: Sequence[np.ndarray]):
        if len(points) < 2:
            raise ValueError("PathSampler requires at least two points.")
        raw = np.asarray(points, dtype=np.float32)
        diffs = raw[1:] - raw[:-1]
        lengths = np.linalg.norm(diffs, axis=1)
        valid = lengths > 1e-6
        cleaned = [raw[0]]
        vectors: list[np.ndarray] = []
        seg_lengths: list[float] = []
        for idx, ok in enumerate(valid):
            if not ok:
                continue
            cleaned.append(raw[idx + 1])
            vectors.append(diffs[idx])
            seg_lengths.append(float(lengths[idx]))
        self.points = np.asarray(cleaned, dtype=np.float32)
        self.segment_vectors = np.asarray(vectors, dtype=np.float32)
        self.segment_lengths = np.asarray(seg_lengths, dtype=np.float32)
        self.cumulative = np.concatenate(
            [np.array([0.0], dtype=np.float32), np.cumsum(self.segment_lengths)]
        )

    @property
    def total_length(self) -> float:
        return float(self.cumulative[-1])

    def position_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            direction = self.segment_vectors[0] / self.segment_lengths[0]
            return self.points[0] + direction * distance
        total = self.total_length
        if distance >= total:
            direction = self.segment_vectors[-1] / self.segment_lengths[-1]
            return self.points[-1] + direction * (distance - total)
        seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
        seg_offset = distance - self.cumulative[seg_idx]
        ratio = seg_offset / self.segment_lengths[seg_idx]
        return self.points[seg_idx] + self.segment_vectors[seg_idx] * ratio

    def direction_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            vec = self.segment_vectors[0]
        elif distance >= self.total_length:
            vec = self.segment_vectors[-1]
        else:
            seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
            vec = self.segment_vectors[seg_idx]
        norm = np.linalg.norm(vec)
        if norm < EPS:
            return np.array([0.0, 1.0], dtype=np.float32)
        return vec / norm


def resample_path_by_distance(points: Sequence[np.ndarray], step: float) -> list[np.ndarray]:
    if step <= 0.0 or len(points) < 2:
        return list(points)
    sampler = PathSampler(points)
    total = sampler.total_length
    if total <= step:
        return list(points)
    distances: list[float] = []
    dist = 0.0
    while dist < total:
        distances.append(dist)
        dist += step
    distances.append(total)
    resampled = [sampler.position_at(float(d)) for d in distances]
    deduped = [resampled[0]]
    for point in resampled[1:]:
        if np.linalg.norm(point - deduped[-1]) > 1e-4:
            deduped.append(point)
    return [np.array([pt[0], pt[1]], dtype=np.float32) for pt in deduped]


def forward_direction(points: Sequence[np.ndarray], idx: int, window: int = 1) -> np.ndarray:
    if len(points) == 1:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    accum = np.zeros(2, dtype=np.float32)
    count = 0
    max_step = max(1, int(window))
    for step in range(1, max_step + 1):
        nxt = min(idx + step, len(points) - 1)
        delta = points[nxt][:2] - points[idx][:2]
        if np.linalg.norm(delta) > 1e-4:
            accum += delta
            count += 1
    for step in range(1, max_step + 1):
        prev = max(idx - step, 0)
        delta = points[idx][:2] - points[prev][:2]
        if np.linalg.norm(delta) > 1e-4:
            accum += delta
            count += 1
    if count == 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    direction_xy = accum / float(count)
    norm = np.linalg.norm(direction_xy)
    if norm < 1e-4:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([direction_xy[0] / norm, direction_xy[1] / norm, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class PreparedPath:
    path_xy: list[np.ndarray]
    raw_points: list[np.ndarray]
    floor_z: float
    ceiling: float


def prepare_path_data(
    json_path: Path,
    meta: dict,
    *,
    stride: int,
    resample_step: float,
    mirror_translation: bool,
    swap_xy: bool = False,
    handedness: str = "left",
    negate_xy: bool = False,
) -> PreparedPath:
    raw_points, raster_pixels = load_raster_world_points(
        json_path,
        swap_xy=swap_xy,
        handedness=handedness,
        negate_xy=negate_xy,
    )
    a_x, b_x, a_y, b_y = derive_affine_transform(raw_points, raster_pixels, meta)
    transformed = [
        np.array([a_x * pt[0] + b_x, a_y * pt[1] + b_y], dtype=np.float32) for pt in raw_points
    ]
    points_xy = deduplicate_points(transformed)
    sampled_xy = sample_points(points_xy, stride)
    if len(sampled_xy) < 2:
        sampled_xy = points_xy

    if mirror_translation:
        center_x = 0.5 * (meta["left"] + meta["right"])
        center_y = 0.5 * (meta["top"] + meta["bottom"])
        sampled_xy = [
            np.array([center_x * 2.0 - pt[0], center_y * 2.0 - pt[1]], dtype=np.float32)
            for pt in sampled_xy
        ]
    if resample_step > 0.0:
        resampled = resample_path_by_distance(sampled_xy, resample_step)
        if len(resampled) >= 2:
            sampled_xy = resampled
    return PreparedPath(
        path_xy=[np.array([pt[0], pt[1], 0.0], dtype=np.float32) for pt in sampled_xy],
        raw_points=raw_points,
        floor_z=float(meta["lower_z"]),
        ceiling=float(meta["upper_z"]),
    )


def _ensure_bev(
    scene_dir: Path,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    meta: dict,
) -> tuple[Path, float]:
    occ_png = scene_dir / "occupancy.png"
    if occ_png.exists():
        meters_per_pixel = float(meta.get("scale", 0.05))
        return occ_png, meters_per_pixel
    width = max(float(bounds_max[0] - bounds_min[0]), 1e-3)
    depth = max(float(bounds_max[1] - bounds_min[1]), 1e-3)
    meters_per_pixel = max(width, depth) / 512.0
    bev_image = np.zeros((512, 512, 3), dtype=np.uint8)
    bev_path = Path(tempfile.gettempdir()) / f"navdp_blank_bev_{scene_dir.name}.png"
    imageio.imwrite(bev_path, bev_image)
    return bev_path, float(meters_per_pixel)


def build_scene_asset(scene_dir: Path, gaussian_model: Path, meta: dict) -> SceneAsset:
    ply_path = gaussian_model.expanduser().resolve()
    ply = GaussianPly.read(ply_path)
    xs = ply.data["x"].astype(np.float64)
    ys = ply.data["y"].astype(np.float64)
    zs = ply.data["z"].astype(np.float64)
    bounds_min = (float(xs.min()), float(ys.min()), float(zs.min()))
    bounds_max = (float(xs.max()), float(ys.max()), float(zs.max()))
    bev_path, meters_per_pixel = _ensure_bev(scene_dir, bounds_min, bounds_max, meta)

    metadata_payload = {
        "scene_id": ply_path.stem,
        "source": "navdp_path_renderer_telesim",
        "note": "Auto-generated metadata for TeleSim3D NavDP path rendering.",
    }
    metadata_path = Path(tempfile.gettempdir()) / f"navdp_scene_meta_{ply_path.stem}.json"
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    navmesh_placeholder = ply_path.with_suffix(".navmesh")
    return SceneAsset(
        scene_id=ply_path.stem,
        metadata_path=metadata_path,
        scene_glb=ply_path,
        dataset_config=None,
        navmesh_path=navmesh_placeholder,
        bev_path=bev_path,
        meters_per_pixel=meters_per_pixel if meters_per_pixel > 0 else float(meta.get("scale", 0.05)),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        resolved_slice_height=float(meta.get("lower_z", bounds_min[2])),
        scene_metadata_path=None,
        splat_model_path=ply_path,
        splat_bev_path=None,
    )


def build_renderer(asset: SceneAsset, args: argparse.Namespace) -> GaussianRendererBackend:
    cfg = GaussianRendererConfig(
        scene_asset=asset,
        model_path=asset.splat_model_path,
        frame_size=tuple(args.resolution),
        device=args.device,
        vertical_fov_degrees=args.fov_deg,
        z_near=args.znear,
        z_far=args.zfar,
        antialiasing=args.antialiasing,
        sh_degree=args.sh_degree,
        separate_sh=args.separate_sh,
        use_trained_exposure=args.use_trained_exposure,
    )
    return GaussianRendererBackend(cfg)


def resolve_scene_dir(root: Path, scene_id: str) -> Path:
    exact = root / scene_id
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"{scene_id}*"))
    if not matches:
        raise FileNotFoundError(f"No scene matching '{scene_id}' under {root}")
    LOGGER.info("Scene '%s' resolved to %s", scene_id, matches[0].name)
    return matches[0]


def resolve_label_directory(scene_task_dir: Path) -> Path | None:
    label_paths_dir = scene_task_dir / "label_paths"
    if label_paths_dir.is_dir():
        return label_paths_dir
    if scene_task_dir.is_dir() and any(scene_task_dir.glob("*.json")):
        return scene_task_dir
    return None


def collect_labels(
    label_dir: Path,
    label_ids: Sequence[str] | None,
    max_labels: int | None,
    exclude_detailed: bool,
) -> list[Path]:
    if label_ids:
        resolved: list[Path] = []
        for label_id in label_ids:
            label_path = Path(label_id)
            if label_path.suffix != ".json":
                label_path = label_path.with_suffix(".json")
            if not label_path.is_file():
                candidate = label_dir / label_path.name
                if candidate.is_file():
                    label_path = candidate
            resolved.append(label_path)
        return resolved

    candidates = sorted(p for p in label_dir.glob("*.json"))
    if exclude_detailed:
        candidates = [p for p in candidates if not p.name.endswith("_detailed.json")]
    if max_labels is not None and max_labels > 0:
        candidates = candidates[:max_labels]
    return candidates


def _parse_resume_log(log_path: Path) -> dict[str, set[str]]:
    """Parse a render log to detect completed labels per scene."""
    completed: dict[str, set[str]] = {}
    current_scene: str | None = None
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "Rendering" in line and "scene" in line:
                parts = line.strip().split()
                if parts and parts[-1]:
                    current_scene = parts[-1]
                    completed.setdefault(current_scene, set())
                continue
            if line.strip().startswith("->") or line.strip().startswith("->"):
                label = line.strip().replace("->", "").strip()
                if current_scene and label:
                    completed.setdefault(current_scene, set()).add(label)
                continue
            if line.strip().startswith("->") or line.strip().startswith("  ->"):
                label = line.strip().replace("->", "").strip()
                if current_scene and label:
                    completed.setdefault(current_scene, set()).add(label)
    except Exception:
        return completed
    return completed


def _label_already_rendered(output_dir: Path, scene_id: str, label_id: str) -> bool:
    video_path = output_dir / scene_id / f"{label_id}.mp4"
    if video_path.is_file():
        return True
    frames_dir = output_dir / scene_id / label_id
    if frames_dir.is_dir() and any(frames_dir.glob("frame_*")):
        return True
    return False


def build_camera_poses(
    path_xy: Sequence[np.ndarray],
    *,
    floor_z: float,
    ceiling: float,
    follow_distance: float,
    height_offset: float,
    look_ahead: float,
    look_down: float,
    stabilize: bool,
) -> list[tuple[Pose, np.ndarray]]:
    sampler = PathSampler([pt[:2] for pt in path_xy])
    distances = list(sampler.cumulative)
    total_length = sampler.total_length
    follow = max(float(follow_distance), 0.0)
    max_cam_dist = max(total_length - follow, 0.0)

    camera_positions: list[np.ndarray] = []
    for dist in distances:
        cam_dist = min(dist, max_cam_dist)
        xy = sampler.position_at(cam_dist)
        camera_positions.append(np.array([xy[0], xy[1], ceiling + height_offset], dtype=np.float32))
        if cam_dist >= max_cam_dist - 1e-6:
            break

    poses: list[tuple[Pose, np.ndarray]] = []
    direction_window = STABILIZE_WINDOW if stabilize else 1
    prev_forward: np.ndarray | None = None
    for idx, cam_pos in enumerate(camera_positions):
        fwd = forward_direction(camera_positions, idx, window=direction_window)
        if np.linalg.norm(fwd[:2]) < EPS:
            fwd = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if stabilize and prev_forward is not None:
            blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + fwd * FORWARD_SMOOTH_BLEND
            bnorm = float(np.linalg.norm(blended))
            if bnorm > EPS:
                fwd = (blended / bnorm).astype(np.float32)
        prev_forward = fwd.copy()
        target_xy = cam_pos[:2] + fwd[:2] * look_ahead
        target_z = max(cam_pos[2] - abs(look_down), floor_z + 0.05)
        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

        view = build_look_at(cam_pos, target, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        rot_world = view[:3, :3].T.astype(np.float32)
        quat = _matrix_to_quat_wxyz(rot_world[None, ...])[0]
        pose = Pose(position=tuple(cam_pos.astype(float)), orientation=tuple(quat.astype(float)))
        poses.append((pose, cam_pos))
    return poses


def build_actor_follow_plans(
    path_xy: Sequence[np.ndarray],
    *,
    floor_z: float,
    ceiling: float,
    follow_distance: float,
    height_offset: float,
    look_ahead: float,
    look_down: float,
    stabilize: bool,
    actor_runtime: ActorRuntime,
) -> tuple[list[tuple[Pose, np.ndarray]], list[np.ndarray], list[int]]:
    sampler = PathSampler([pt[:2] for pt in path_xy])
    distances = list(sampler.cumulative)
    total_length = sampler.total_length

    follow_distance_m = max(float(actor_runtime.options.follow_distance), 0.0)
    max_camera_distance = max(total_length - follow_distance_m, 0.0)
    actor_ground_z = float(floor_z + actor_runtime.options.foot_offset)

    cycle_mod = max(1, int(getattr(actor_runtime.options, "animation_cycle_mod", 1)))
    anim_step = (actor_runtime.options.fps / float(DEFAULT_VIDEO_FPS)) * cycle_mod
    anim_cursor = 0.0
    num_actor_frames = len(actor_runtime.sequence.frames)

    camera_positions: list[np.ndarray] = []
    actor_plans: list[np.ndarray] = []
    actor_indices: list[int] = []
    cached_direction = np.array([0.0, 1.0], dtype=np.float32)
    prev_actor_dir: np.ndarray | None = None

    for dist in distances:
        camera_distance = min(dist, max_camera_distance)
        actor_distance = min(camera_distance + follow_distance_m, total_length)

        direction_xy = sampler.direction_at(actor_distance)
        if np.linalg.norm(direction_xy) < 1e-6:
            direction_xy = cached_direction
        actor_dir = direction_xy.copy()
        if stabilize and prev_actor_dir is not None:
            blended_actor = prev_actor_dir * (1.0 - FORWARD_SMOOTH_BLEND) + actor_dir * FORWARD_SMOOTH_BLEND
            norm_actor = np.linalg.norm(blended_actor)
            if norm_actor > EPS:
                actor_dir = blended_actor / norm_actor
        if np.linalg.norm(direction_xy) >= 1e-6:
            cached_direction = actor_dir
        prev_actor_dir = actor_dir

        theta = math.atan2(actor_dir[0], actor_dir[1]) + math.pi
        rotation_np = rotation_matrix_z_np(theta)

        actor_pos_xy = sampler.position_at(actor_distance)
        translation_vec = np.array([actor_pos_xy[0], actor_pos_xy[1], actor_ground_z], dtype=np.float64)
        transform = build_transform_matrix(rotation_np, translation_vec)

        if actor_runtime.options.loop:
            anim_idx = int(anim_cursor) % num_actor_frames
        else:
            anim_idx = min(int(anim_cursor), num_actor_frames - 1)
        anim_cursor += anim_step

        actor_plans.append(transform)
        actor_indices.append(anim_idx)

        cam_xy = sampler.position_at(camera_distance)
        camera_positions.append(np.array([cam_xy[0], cam_xy[1], ceiling + height_offset], dtype=np.float32))
        if camera_distance >= max_camera_distance - 1e-6:
            break

    poses: list[tuple[Pose, np.ndarray]] = []
    direction_window = STABILIZE_WINDOW if stabilize else 1
    prev_forward: np.ndarray | None = None
    for idx, cam_pos in enumerate(camera_positions):
        fwd = forward_direction(camera_positions, idx, window=direction_window)
        if np.linalg.norm(fwd[:2]) < EPS:
            fwd = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if stabilize and prev_forward is not None:
            blended = prev_forward * (1.0 - FORWARD_SMOOTH_BLEND) + fwd * FORWARD_SMOOTH_BLEND
            bnorm = float(np.linalg.norm(blended))
            if bnorm > EPS:
                fwd = (blended / bnorm).astype(np.float32)
        prev_forward = fwd.copy()
        target_xy = cam_pos[:2] + fwd[:2] * look_ahead
        target_z = max(cam_pos[2] - abs(look_down), floor_z + 0.05)
        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
        view = build_look_at(cam_pos, target, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        rot_world = view[:3, :3].T.astype(np.float32)
        quat = _matrix_to_quat_wxyz(rot_world[None, ...])[0]
        pose = Pose(position=tuple(cam_pos.astype(float)), orientation=tuple(quat.astype(float)))
        poses.append((pose, cam_pos))

    return poses, actor_plans, actor_indices


def _render_custom_gaussians(
    renderer: GaussianRendererBackend,
    pose: Pose,
    gaussians,
) -> tuple[np.ndarray, np.ndarray | None, object]:
    camera = renderer._pose_to_camera(pose)  # pylint: disable=protected-access
    with torch.no_grad():
        result = render_gaussians(
            camera,
            gaussians,
            renderer._pipe,  # pylint: disable=protected-access
            renderer._background,  # pylint: disable=protected-access
            scaling_modifier=renderer._scaling_modifier,  # pylint: disable=protected-access
            separate_sh=renderer._separate_sh,  # pylint: disable=protected-access
            use_trained_exp=renderer._use_trained_exposure,  # pylint: disable=protected-access
        )
    image = result["render"].permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    depth_inv = None
    if result.get("depth") is not None:
        depth_inv = result["depth"].detach().cpu().numpy()
    return (image * 255.0).astype(np.uint8), depth_inv, camera


def _serialize_camera(
    *,
    renderer: GaussianRendererBackend,
    pose: Pose,
    frame_size: tuple[int, int],
    fov_y_rad: float,
) -> dict:
    matrices = renderer.camera_matrices(pose)
    world_view = np.asarray(matrices["world_view"], dtype=np.float64)
    projection = np.asarray(matrices["full_projection"], dtype=np.float64)
    camera_to_world = np.linalg.inv(world_view)
    camera_center = camera_to_world[3][:3].tolist()
    w, h = frame_size
    fovx = 2.0 * math.atan(math.tan(fov_y_rad * 0.5) * (w / float(h)))
    fx = w / (2.0 * math.tan(fovx * 0.5))
    fy = h / (2.0 * math.tan(fov_y_rad * 0.5))

    return {
        "type": "perspective",
        "resolution": {"width": w, "height": h},
        "fov": {
            "x_rad": float(fovx),
            "y_rad": float(fov_y_rad),
            "x_deg": math.degrees(float(fovx)),
            "y_deg": math.degrees(float(fov_y_rad)),
        },
        "znear": float(matrices["intrinsics"]["znear"]),
        "zfar": float(matrices["intrinsics"]["zfar"]),
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(w * 0.5),
            "cy": float(h * 0.5),
            "half_width": None,
            "half_height": None,
        },
        "camera_center_world": camera_center,
        "world_to_camera": world_view.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "projection_matrix": projection.tolist(),
    }


def _write_camera_metadata(
    *,
    frames_dir: Path,
    frame_prefix: str,
    frame_idx: int,
    payload: dict,
) -> None:
    cam_json_path = frames_dir / f"{frame_prefix}_{frame_idx:04d}_camera.json"
    cam_json_path.write_text(json.dumps(payload, indent=2))


def _quantize_depth(depth_m: np.ndarray, *, bit_depth: int) -> np.ndarray:
    if bit_depth <= 0:
        raise ValueError("bit_depth must be positive.")
    depth = depth_m.astype(np.float32, copy=False)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(depth.max()) if depth.size else 0.0
    if max_val <= 1e-6:
        max_val = 1.0
    depth_norm = np.clip(depth / max_val, 0.0, 1.0)
    scale = (1 << bit_depth) - 1
    return (depth_norm * scale + 0.5).astype(np.uint16 if bit_depth > 8 else np.uint8)


def _save_depth_map(
    *,
    depth_inv: np.ndarray,
    frames_dir: Path,
    frame_prefix: str,
    frame_idx: int,
    rotate_180: bool,
    bit_depth: int = 16,
) -> None:
    if depth_inv.ndim > 2:
        depth_inv = np.squeeze(depth_inv)
    with np.errstate(divide="ignore"):
        depth_m = np.where(depth_inv > 0.0, 1.0 / depth_inv, 0.0)
    if rotate_180:
        depth_m = np.flipud(np.fliplr(depth_m))
    depth_quant = _quantize_depth(depth_m, bit_depth=bit_depth)
    depth_png_path = frames_dir / f"{frame_prefix}_{frame_idx:04d}_depth.png"
    imageio.imwrite(depth_png_path, depth_quant)


def _write_video_frames(
    *,
    video_path: Path,
    frames: Iterable[np.ndarray],
    fps: int,
    encode_time_accum: list[float],
) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(video_path), fps=fps) as writer:
        for frame in frames:
            start = time.monotonic()
            writer.append_data(frame)
            encode_time_accum[0] += time.monotonic() - start


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_label(
    *,
    renderer: GaussianRendererBackend,
    prepared: PreparedPath,
    output_dir: Path,
    label_id: str,
    args: argparse.Namespace,
) -> dict:
    poses = build_camera_poses(
        prepared.path_xy,
        floor_z=prepared.floor_z,
        ceiling=prepared.ceiling,
        follow_distance=args.follow_distance,
        height_offset=args.height_offset,
        look_ahead=args.look_ahead,
        look_down=args.look_down,
        stabilize=args.stabilize,
    )

    if args.minimal_frames is not None and args.minimal_frames > 0:
        poses = poses[: args.minimal_frames]

    scene_dir = output_dir / args.scene
    _safe_mkdir(scene_dir)
    frames_dir = scene_dir / label_id
    if args.save_camera_metadata:
        _safe_mkdir(frames_dir)

    video_path = scene_dir / f"{label_id}.mp4"
    frame_prefix = "frame"

    render_time = 0.0
    encode_time = 0.0
    frames_rendered = 0
    fov_y_rad = math.radians(float(args.fov_deg))

    light_config: LightFilterConfig | None = getattr(args, "light_config", None)
    cl_config: CameraLightConfig | None = getattr(args, "cl_config", None)
    cl_light_world = getattr(args, "cl_light_world", None)
    light_seed_offset = getattr(args, "light_seed_offset", 0)

    def _frame_iter():
        nonlocal render_time, frames_rendered
        for idx, (pose, _) in enumerate(poses):
            start = time.monotonic()
            rgb, depth_inv, camera = _render_custom_gaussians(renderer, pose, renderer._gaussians)  # pylint: disable=protected-access
            render_time += time.monotonic() - start
            rgb = _apply_light_filter_if_enabled(
                rgb,
                light_config,
                frame_index=idx,
                seed_offset=light_seed_offset,
            )
            rgb = _apply_camera_light_if_enabled(
                rgb,
                depth_inv,
                camera,
                cl_config,
                cl_light_world=cl_light_world,
            )
            if args.rotate_180:
                rgb = np.flipud(np.fliplr(rgb))
            frames_rendered += 1
            if args.save_camera_metadata:
                cam_payload = _serialize_camera(
                    renderer=renderer,
                    pose=pose,
                    frame_size=tuple(args.resolution),
                    fov_y_rad=fov_y_rad,
                )
                _write_camera_metadata(
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=idx,
                    payload=cam_payload,
                )
            if args.save_depth_maps and depth_inv is not None:
                _save_depth_map(
                    depth_inv=depth_inv,
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=idx,
                    rotate_180=args.rotate_180,
                )
            if args.rgb_frames:
                frame_path = frames_dir / f"{frame_prefix}_{idx:04d}.png"
                imageio.imwrite(frame_path, rgb)
            yield rgb

    if args.video:
        encode_holder = [0.0]
        _write_video_frames(
            video_path=video_path,
            frames=_frame_iter(),
            fps=args.video_fps,
            encode_time_accum=encode_holder,
        )
        encode_time = encode_holder[0]
    else:
        for _ in _frame_iter():
            pass

    duration = render_time + encode_time
    return {
        "scene_id": args.scene,
        "label_id": label_id,
        "frames": frames_rendered,
        "duration_sec": duration,
        "stage_seconds": {
            "render": render_time,
            "encode": encode_time,
        },
    }


def _camera_xy_sequence(
    prepared: PreparedPath,
    args: argparse.Namespace,
    actor_runtime: ActorRuntime | None,
) -> list[np.ndarray]:
    if actor_runtime is None:
        poses = build_camera_poses(
            prepared.path_xy,
            floor_z=prepared.floor_z,
            ceiling=prepared.ceiling,
            follow_distance=args.follow_distance,
            height_offset=args.height_offset,
            look_ahead=args.look_ahead,
            look_down=args.look_down,
            stabilize=args.stabilize,
        )
        return [pos[:2].copy() for _, pos in poses]
    poses, _, _ = build_actor_follow_plans(
        prepared.path_xy,
        floor_z=prepared.floor_z,
        ceiling=prepared.ceiling,
        follow_distance=args.follow_distance,
        height_offset=args.height_offset,
        look_ahead=args.look_ahead,
        look_down=args.look_down,
        stabilize=args.stabilize,
        actor_runtime=actor_runtime,
    )
    return [pos[:2].copy() for _, pos in poses]


def render_label_with_actor(
    *,
    renderer: GaussianRendererBackend,
    prepared: PreparedPath,
    output_dir: Path,
    label_id: str,
    args: argparse.Namespace,
    actor_runtime: ActorRuntime,
) -> dict:
    poses, actor_transforms, actor_indices = build_actor_follow_plans(
        prepared.path_xy,
        floor_z=prepared.floor_z,
        ceiling=prepared.ceiling,
        follow_distance=args.follow_distance,
        height_offset=args.height_offset,
        look_ahead=args.look_ahead,
        look_down=args.look_down,
        stabilize=args.stabilize,
        actor_runtime=actor_runtime,
    )

    if args.minimal_frames is not None and args.minimal_frames > 0:
        poses = poses[: args.minimal_frames]
        actor_transforms = actor_transforms[: args.minimal_frames]
        actor_indices = actor_indices[: args.minimal_frames]

    scene_dir = output_dir / args.scene
    _safe_mkdir(scene_dir)
    frames_dir = scene_dir / label_id
    if args.save_camera_metadata:
        _safe_mkdir(frames_dir)

    video_path = scene_dir / f"{label_id}.mp4"
    frame_prefix = "frame"

    render_time = 0.0
    encode_time = 0.0
    frames_rendered = 0
    fov_y_rad = math.radians(float(args.fov_deg))

    base_gaussians = renderer._gaussians  # pylint: disable=protected-access
    scene_rest_dim = int(base_gaussians.get_features_rest.shape[1])
    combined_model: CombinedGaussianModel | None = None
    combined_actor_size: int | None = None

    light_config: LightFilterConfig | None = getattr(args, "light_config", None)
    cl_config: CameraLightConfig | None = getattr(args, "cl_config", None)
    cl_light_world = getattr(args, "cl_light_world", None)
    light_seed_offset = getattr(args, "light_seed_offset", 0)

    def _frame_iter():
        nonlocal render_time, frames_rendered, combined_model, combined_actor_size
        for idx, ((pose, _), transform, actor_idx) in enumerate(
            zip(poses, actor_transforms, actor_indices)
        ):
            sequence_frame = actor_runtime.sequence.frames[actor_idx]
            # Note: apply_transform_to_frame expects (ActorSequenceFrame, ActorSequence, transform).
            actor_data = apply_transform_to_frame(
                sequence_frame,
                actor_runtime.sequence,
                transform,
            )
            actor_render = actor_data_to_tensors(
                actor_data,
                actor_runtime.sequence,
                device=base_gaussians.get_xyz.device,
                target_rest_dim=scene_rest_dim,
            )
            current_actor_size = int(actor_render.xyz.shape[0])
            if combined_model is None or combined_actor_size != current_actor_size:
                combined_actor_size = current_actor_size
                combined_model = CombinedGaussianModel(base_gaussians, actor_render)
            else:
                combined_model.update_actor(actor_render)

            start = time.monotonic()
            rgb, depth_inv, camera = _render_custom_gaussians(renderer, pose, combined_model)
            render_time += time.monotonic() - start
            rgb = _apply_light_filter_if_enabled(
                rgb,
                light_config,
                frame_index=idx,
                seed_offset=light_seed_offset,
            )
            rgb = _apply_camera_light_if_enabled(
                rgb,
                depth_inv,
                camera,
                cl_config,
                cl_light_world=cl_light_world,
            )
            if args.rotate_180:
                rgb = np.flipud(np.fliplr(rgb))
            frames_rendered += 1
            if args.save_camera_metadata:
                cam_payload = _serialize_camera(
                    renderer=renderer,
                    pose=pose,
                    frame_size=tuple(args.resolution),
                    fov_y_rad=fov_y_rad,
                )
                _write_camera_metadata(
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=idx,
                    payload=cam_payload,
                )
            if args.save_depth_maps and depth_inv is not None:
                _save_depth_map(
                    depth_inv=depth_inv,
                    frames_dir=frames_dir,
                    frame_prefix=frame_prefix,
                    frame_idx=idx,
                    rotate_180=args.rotate_180,
                )
            if args.rgb_frames:
                frame_path = frames_dir / f"{frame_prefix}_{idx:04d}.png"
                imageio.imwrite(frame_path, rgb)
            yield rgb

    if args.video:
        encode_holder = [0.0]
        _write_video_frames(
            video_path=video_path,
            frames=_frame_iter(),
            fps=args.video_fps,
            encode_time_accum=encode_holder,
        )
        encode_time = encode_holder[0]
    else:
        for _ in _frame_iter():
            pass

    duration = render_time + encode_time
    return {
        "scene_id": args.scene,
        "label_id": label_id,
        "frames": frames_rendered,
        "duration_sec": duration,
        "stage_seconds": {
            "render": render_time,
            "encode": encode_time,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render NavDP label paths with TeleSim3D.")
    parser.add_argument("--scenes-dir", type=Path, default=REPO_ROOT / "data" / "scenes")
    parser.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "data" / "tasks")
    parser.add_argument("--scene", required=True, help="Scene ID to render.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "tmp" / "test_telesim3d")
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--error-log", type=Path, default=None)
    parser.add_argument("--skip-completed-log", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-id", action="append", default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--exclude-detailed-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resample-step", type=float, default=0.0)
    parser.add_argument("--path-handedness", choices=["left", "right", "auto"], default="left")
    parser.add_argument("--swap-xy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--negate-xy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--negate-raster-world-xy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --negate-xy.",
    )
    parser.add_argument("--mirror-translation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--look-ahead", type=float, default=2.0)
    parser.add_argument("--look-down", type=float, default=0.1)
    parser.add_argument("--height-offset", type=float, default=0.3)
    parser.add_argument("--resolution", type=int, nargs=2, default=(960, 720), metavar=("W", "H"))
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--znear", type=float, default=0.001)
    parser.add_argument("--zfar", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--antialiasing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--separate-sh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-trained-exposure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stabilize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false")
    parser.set_defaults(rotate_180=True)
    parser.add_argument("--gaussian-model", type=Path, default=None)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--rgb-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-camera-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimal-frames", type=int, default=None)
    parser.add_argument("--view-mode", default="forward")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--path-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-path progress (scene/label/done/eta/space). Default: on.",
    )
    parser.add_argument(
        "--path-progress-space-interval-sec",
        type=float,
        default=5.0,
        help="How often to refresh disk usage stats for per-path progress (default: 5s).",
    )
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--video-nvenc-preset", default=None)
    parser.add_argument("--video-nvenc-bitrate", default=None)
    parser.add_argument("--gpu-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-BEV", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-depth-maps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-follow-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-bit-depth", type=int, default=16)
    parser.add_argument("--no-validate-path-bounds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--navdp-ply-per-scene", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ply-transform-backend", default=None)
    parser.add_argument("--cl-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cl-light-mode", default="headlight")
    parser.add_argument("--cl-shading-model", default="classic")
    parser.add_argument("--cl-strength", type=float, default=1.0)
    parser.add_argument("--cl-color", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--cl-ambient", type=float, default=0.2)
    parser.add_argument("--cl-base-scale", type=float, default=1.0)
    parser.add_argument("--cl-diffuse", type=float, default=1.0)
    parser.add_argument("--cl-specular", type=float, default=0.2)
    parser.add_argument("--cl-shininess", type=float, default=16.0)
    parser.add_argument("--cl-range", type=float, default=0.0)
    parser.add_argument("--cl-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--cl-light-reverse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cl-light-world", type=float, nargs=3, default=None)
    parser.add_argument("--cl-light-center", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cl-light-center-z", type=float, default=0.0)
    parser.add_argument("--cl-normal-smooth", type=float, default=0.0)
    parser.add_argument("--cl-normal-filter", default="box")
    parser.add_argument("--cl-normal-kernel", type=int, default=2)
    parser.add_argument("--cl-normal-sigma-range", type=float, default=0.1)
    parser.add_argument("--cl-normal-sigma-domain", type=float, default=1.0)
    parser.add_argument("--cl-shadow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cl-shadow-bias", type=float, default=0.02)
    parser.add_argument("--cl-shadow-strength", type=float, default=0.2)
    parser.add_argument("--cl-shadow-pcf", type=float, default=0)
    parser.add_argument("--cl-shadow-compare", default="z")
    parser.add_argument("--npc-render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--npc-count", type=int, default=0)
    parser.add_argument("--npc-max-count", type=int, default=0)
    parser.add_argument("--npc-density-coverage", type=float, default=0.0)
    parser.add_argument("--npc-priority", default=None)
    parser.add_argument("--npc-density-mode", default=None)
    parser.add_argument("--npc-zone-ratio", default=None)
    parser.add_argument("--npc-max-range", type=float, default=0.0)
    parser.add_argument("--npc-free-threshold", type=int, default=0)
    parser.add_argument("--npc-placement-backend", default=None)
    parser.add_argument("--npc-seed", type=int, default=0)
    parser.add_argument("--npc-free-white", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--npc-rotate-mask-180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--npc-actor-root", type=Path, default=None)
    parser.add_argument("--npc-frame-pool-size", type=int, default=0)
    parser.add_argument("--job-slot", type=int, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--job-actor-id", default=None)
    parser.add_argument("--light-mode", default="none")
    parser.add_argument("--light-strength", type=float, default=0.0)
    parser.add_argument("--light-radius", type=float, default=0.45)
    parser.add_argument("--light-center", type=float, nargs=2, default=(0.5, 0.5))
    parser.add_argument("--light-jitter", type=float, default=0.0)
    parser.add_argument("--light-temp-k", type=float, default=0.0)
    parser.add_argument("--light-vignette", type=float, default=0.0)
    parser.add_argument("--light-seed", type=int, default=0)
    parser.add_argument("--actor-seq-dir", type=Path, default=None)
    parser.add_argument("--actor-pattern", default=DEFAULT_ACTOR_PATTERN)
    parser.add_argument("--actor-height", type=float, default=1.7)
    parser.add_argument("--actor-speed", type=float, default=DEFAULT_ACTOR_SPEED)
    parser.add_argument("--actor-fps", type=float, default=float(DEFAULT_VIDEO_FPS))
    parser.add_argument("--follow-distance", type=float, default=1.5)
    parser.add_argument("--follow-buffer", type=float, default=0.0)
    parser.add_argument("--actor-foot-offset", type=float, default=0.0)
    parser.add_argument("--animation-cycle-mod", type=int, default=3)
    parser.add_argument("--actor-no-loop", dest="actor_loop", action="store_false")
    parser.set_defaults(actor_loop=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        LOGGER.warning("Ignoring unsupported args: %s", " ".join(unknown))
    return args


def main() -> int:
    args = parse_args()
    if args.view_mode not in ("forward", ""):
        LOGGER.warning("view-mode '%s' is not supported; using 'forward'.", args.view_mode)
    if args.path_handedness == "auto":
        args.path_handedness = "left"
    if getattr(args, "negate_raster_world_xy", False):
        args.negate_xy = True
    if args.light_mode != "none":
        args.light_config = LightFilterConfig(
            mode=str(args.light_mode),
            strength=float(args.light_strength),
            radius_frac=float(args.light_radius),
            center_xy=(float(args.light_center[0]), float(args.light_center[1])),
            center_jitter=float(args.light_jitter),
            temp_k=float(args.light_temp_k),
            vignette=float(args.light_vignette),
            seed=int(args.light_seed),
        )
        args.light_seed_offset = stable_hash_seed(f"{args.scene}:{args.light_seed}")
    else:
        args.light_config = None
        args.light_seed_offset = 0
    if args.cl_enable:
        range_m = None if float(args.cl_range) <= 0.0 else float(args.cl_range)
        args.cl_config = CameraLightConfig(
            enabled=True,
            strength=float(args.cl_strength),
            color=(float(args.cl_color[0]), float(args.cl_color[1]), float(args.cl_color[2])),
            ambient=float(args.cl_ambient),
            diffuse=float(args.cl_diffuse),
            specular=float(args.cl_specular),
            shininess=float(args.cl_shininess),
            range_m=range_m,
            offset_cam=(float(args.cl_offset[0]), float(args.cl_offset[1]), float(args.cl_offset[2])),
            normal_smooth=int(args.cl_normal_smooth),
            shadow_enabled=bool(args.cl_shadow),
            shadow_bias=float(args.cl_shadow_bias),
            shadow_strength=float(args.cl_shadow_strength),
            shadow_pcf_radius=int(args.cl_shadow_pcf),
            light_mode=str(args.cl_light_mode),
            shading_model=str(args.cl_shading_model),
            shadow_compare=str(args.cl_shadow_compare),
            normal_filter=str(args.cl_normal_filter),
            normal_kernel=int(args.cl_normal_kernel),
            normal_sigma_range=float(args.cl_normal_sigma_range),
            normal_sigma_domain=float(args.cl_normal_sigma_domain),
            base_scale=float(args.cl_base_scale),
            light_reverse=bool(args.cl_light_reverse),
        )
    else:
        args.cl_config = None

    # Preflight output directory once. If a file blocks any parent component (e.g. "navdata"),
    # mkdir will raise FileExistsError with that filename; failing fast avoids spamming per-path errors.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        LOGGER.error(
            "Output directory cannot be created (a file exists where a directory is expected): "
            "blocked_path=%s output_dir=%s",
            getattr(exc, "filename", None) or str(exc),
            args.output_dir,
        )
        return 2
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Output directory cannot be created: output_dir=%s error=%s", args.output_dir, exc)
        return 2

    scene_dir = resolve_scene_dir(args.scenes_dir, args.scene)
    scene_id = scene_dir.name
    args.scene = scene_id
    tasks_scene_dir = resolve_scene_dir(args.tasks_dir, scene_id)
    label_dir = resolve_label_directory(tasks_scene_dir)
    if label_dir is None:
        raise FileNotFoundError(f"No label JSONs under {tasks_scene_dir}")

    gaussian_model = (
        args.gaussian_model
        if args.gaussian_model is not None
        else (scene_dir / "3dgs_raw.ply")
    )
    if not gaussian_model.exists():
        candidates = sorted(scene_dir.glob("*.ply"))
        if not candidates:
            raise FileNotFoundError(f"No Gaussian PLY found under {scene_dir}")
        gaussian_model = candidates[0]
        LOGGER.warning("gaussian_model not found; using %s", gaussian_model.name)

    meta = load_occupancy_metadata(scene_dir)
    asset = build_scene_asset(scene_dir, gaussian_model, meta)
    renderer = build_renderer(asset, args)

    label_paths = collect_labels(label_dir, args.label_id, args.max_labels, args.exclude_detailed_labels)
    completed_labels: set[str] = set()
    if args.skip_completed_log is not None and args.skip_completed_log.is_file():
        completed_labels = _parse_resume_log(args.skip_completed_log).get(scene_id, set())
    if not label_paths:
        LOGGER.warning("No label paths found under %s", label_dir)
        return 1

    if args.verbose:
        LOGGER.info("Rendering %d labels for scene %s", len(label_paths), scene_dir.name)

    paths_payload: list[dict] = []
    total_frames = 0
    total_duration = 0.0
    total_encode = 0.0
    path_statuses: dict[tuple[str, str], dict] = {}
    paths_planned = len(label_paths)
    paths_done = 0  # done includes ok/skip/fatal; OOM is still pending
    paths_ok = 0
    paths_skipped = 0
    paths_fatal = 0
    paths_oom = 0
    paths_attempted = 0
    progress_t0 = time.monotonic()
    last_space_ts = 0.0
    last_scene_bytes: int | None = None
    last_free_bytes: int | None = None

    def record_path_status(label_id: str, status: int, error: str | None = None) -> None:
        path_statuses[(scene_id, label_id)] = {
            "scene_id": scene_id,
            "label_id": label_id,
            "status": int(status),
            "error": error,
        }

    def _log_path_progress(label_id: str, *, status: str, frames: int | None = None) -> None:
        if not args.path_progress:
            return
        nonlocal last_space_ts, last_scene_bytes, last_free_bytes
        now = time.monotonic()
        elapsed = max(1e-6, now - progress_t0)
        # Don't count fatal errors towards throughput; they can be fast failures and skew speed/ETA.
        completed_for_speed = paths_ok + paths_skipped
        speed_paths = (completed_for_speed / elapsed) if completed_for_speed > 0 else None
        remaining = max(0, paths_planned - paths_done)
        eta_sec = (remaining / speed_paths) if speed_paths and speed_paths > 0 else None

        space_interval = float(getattr(args, "path_progress_space_interval_sec", 5.0) or 0.0)
        if space_interval <= 0:
            space_interval = 0.0
        if last_space_ts <= 0.0 or (space_interval > 0.0 and (now - last_space_ts) >= space_interval):
            scene_out_dir = args.output_dir / scene_id
            last_scene_bytes = _dir_size_bytes(scene_out_dir) if scene_out_dir.exists() else 0
            try:
                last_free_bytes = int(shutil.disk_usage(str(args.output_dir)).free)
            except Exception:
                last_free_bytes = None
            last_space_ts = now

        est_total = None
        if last_scene_bytes is not None and paths_done > 0:
            est_total = int((float(last_scene_bytes) / float(paths_done)) * float(paths_planned))

        extra = f" frames={frames}" if frames is not None else ""
        if speed_paths is not None:
            LOGGER.info(
                "[PATH] scene=%s label=%s status=%s done=%d/%d ok=%d skip=%d fatal=%d oom=%d "
                "speed=%.3f paths/s eta=%s space=%s free=%s est_total=%s%s",
                scene_id,
                label_id,
                status,
                paths_done,
                paths_planned,
                paths_ok,
                paths_skipped,
                paths_fatal,
                paths_oom,
                speed_paths,
                _format_seconds(eta_sec),
                _format_bytes(last_scene_bytes),
                _format_bytes(last_free_bytes),
                _format_bytes(est_total),
                extra,
            )
        else:
            LOGGER.info(
                "[PATH] scene=%s label=%s status=%s done=%d/%d ok=%d skip=%d fatal=%d oom=%d "
                "eta=%s space=%s free=%s est_total=%s%s",
                scene_id,
                label_id,
                status,
                paths_done,
                paths_planned,
                paths_ok,
                paths_skipped,
                paths_fatal,
                paths_oom,
                _format_seconds(eta_sec),
                _format_bytes(last_scene_bytes),
                _format_bytes(last_free_bytes),
                _format_bytes(est_total),
                extra,
            )

    actor_runtime: ActorRuntime | None = None
    if args.actor_seq_dir is not None:
        actor_options = ActorOptions(
            sequence_dir=args.actor_seq_dir,
            pattern=args.actor_pattern,
            height=float(args.actor_height),
            speed=float(args.actor_speed),
            fps=float(args.actor_fps),
            loop=bool(args.actor_loop),
            foot_offset=float(args.actor_foot_offset),
            follow_distance=float(args.follow_distance),
            buffer_distance=float(args.follow_buffer),
            animation_cycle_mod=int(args.animation_cycle_mod),
        )
        if actor_options.fps <= 0.0:
            raise ValueError("actor_fps must be positive.")
        if actor_options.speed <= 0.0:
            raise ValueError("actor_speed must be positive.")
        if actor_options.buffer_distance > actor_options.follow_distance:
            raise ValueError("follow_buffer must be <= follow_distance.")
        actor_sequence = load_actor_sequence(actor_options, debug=bool(args.verbose))
        actor_runtime = ActorRuntime(options=actor_options, sequence=actor_sequence)

    for path_file in label_paths:
        label_id = path_file.stem
        if label_id in completed_labels:
            if args.verbose:
                LOGGER.info("  -> %s (skip: completed in log)", label_id)
            record_path_status(label_id, STATUS_DONE, error="skipped_completed_log")
            paths_skipped += 1
            paths_done += 1
            _log_path_progress(label_id, status="skip_completed_log")
            continue
        if args.resume and _label_already_rendered(args.output_dir, scene_id, label_id):
            if args.verbose:
                LOGGER.info("  -> %s (skip: outputs exist)", label_id)
            record_path_status(label_id, STATUS_DONE, error="skipped_outputs_exist")
            paths_skipped += 1
            paths_done += 1
            _log_path_progress(label_id, status="skip_outputs_exist")
            continue
        if args.verbose:
            LOGGER.info("  -> %s", label_id)
        prepared = prepare_path_data(
            path_file,
            meta,
            stride=args.stride,
            resample_step=args.resample_step,
            mirror_translation=args.mirror_translation,
            swap_xy=args.swap_xy,
            handedness=args.path_handedness,
            negate_xy=args.negate_xy,
        )
        try:
            paths_attempted += 1
            record_path_status(label_id, STATUS_RETRY, error="started")
            if actor_runtime is not None:
                path_metrics = render_label_with_actor(
                    renderer=renderer,
                    prepared=prepared,
                    output_dir=args.output_dir,
                    label_id=label_id,
                    args=args,
                    actor_runtime=actor_runtime,
                )
            else:
                path_metrics = render_label(
                    renderer=renderer,
                    prepared=prepared,
                    output_dir=args.output_dir,
                    label_id=label_id,
                    args=args,
                )
            record_path_status(label_id, STATUS_DONE, error=None)
            paths_payload.append(path_metrics)
            paths_ok += 1
            paths_done += 1
            total_frames += int(path_metrics.get("frames", 0))
            total_duration += float(path_metrics.get("duration_sec", 0.0))
            stage = path_metrics.get("stage_seconds") or {}
            total_encode += float(stage.get("encode", 0.0))
            _log_path_progress(label_id, status="ok", frames=int(path_metrics.get("frames", 0) or 0))

            if args.save_follow_metadata:
                cam_seq = _camera_xy_sequence(prepared, args, actor_runtime)
                metadata_payload = build_path_metadata(
                    scene_id=scene_id,
                    label_id=label_id,
                    path_xy=prepared.path_xy,
                    camera_xy_seq=cam_seq,
                    meta=meta,
                    follow_distance=float(args.follow_distance),
                    limit_to_follow=actor_runtime is not None,
                )
                metadata_path = args.output_dir / scene_id / f"{label_id}_follow_path.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pylint: disable=broad-except
            is_oom = _is_cuda_oom_error(exc)
            record_path_status(
                label_id,
                STATUS_RETRY if is_oom else STATUS_SKIP,
                error="cuda_oom" if is_oom else "fatal",
            )
            mp4_path = args.output_dir / scene_id / f"{label_id}.mp4"
            LOGGER.warning(
                "Rendering failed scene=%s label=%s json=%s mp4=%s error=%s",
                scene_id,
                label_id,
                path_file,
                mp4_path,
                exc,
            )
            if is_oom:
                paths_oom += 1
                _log_path_progress(label_id, status="oom")
            else:
                paths_fatal += 1
                paths_done += 1
                _log_path_progress(label_id, status="fatal")
            if args.error_log is not None:
                args.error_log.parent.mkdir(parents=True, exist_ok=True)
                with args.error_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"Scene={scene_id} Label={path_file.name} Error={exc}\n")

    renderer.shutdown()

    metrics = {
        "paths_planned": paths_planned,
        "paths_done": paths_done,
        "paths_ok": paths_ok,
        "paths_skipped": paths_skipped,
        "paths_fatal": paths_fatal,
        "paths_oom": paths_oom,
        "paths_attempted": paths_attempted,
        "paths_total": len(paths_payload),
        "frames_total": total_frames,
        "duration_total_sec": total_duration,
        "time_per_frame_sec": (total_duration / total_frames) if total_frames > 0 else None,
        "h264_encode_total_sec": total_encode,
        "h264_encode_sec_per_frame": (total_encode / total_frames) if total_frames > 0 else None,
        "h264_encode_fps": (total_frames / total_encode) if total_encode > 0 else None,
        "h264_mux_total_sec": 0.0,
        "h264_mux_sec_per_frame": 0.0,
        "h264_mux_sec_per_path": (0.0 if not paths_payload else 0.0),
        "vram_peak_max_bytes": 0.0,
        "vram_avg_max_worker_bytes": 0.0,
        "paths": paths_payload,
        "path_statuses": list(path_statuses.values()),
    }

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Be explicit: this script runs per-scene (the parallel dispatcher fans out scenes),
    # so logging should always include scene context.
    outputs_dir = args.output_dir / scene_id
    LOGGER.info(
        "Done scene=%s planned=%d done=%d ok=%d skip=%d fatal=%d oom=%d outputs=%s",
        scene_id,
        paths_planned,
        paths_done,
        paths_ok,
        paths_skipped,
        paths_fatal,
        paths_oom,
        outputs_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
