#!/usr/bin/env python3
"""Render NavDP label paths with TeleSim3D's Gaussian renderer.

This is a lightweight TeleSim3D-backed alternative to render_label_paths.py.
It mirrors NavDP path preprocessing (affine mapping, stride, mirroring) and
writes MP4 + optional camera metadata for quick pipeline validation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
    ActorRenderFrame,
    ActorRuntime,
    CombinedGaussianModel,
    DEFAULT_ACTOR_PATTERN,
    DEFAULT_ACTOR_SPEED,
    DEFAULT_VIDEO_FPS,
    GpuActorSequence,
    MultiActorCombinedGaussianModel,
    build_gpu_actor_sequence,
    build_path_metadata,
    load_actor_sequence,
    actor_data_to_tensors,
    transform_gpu_actor_frame,
)
from utils.actor_visibility import sphere_visible_in_camera, transformed_actor_sphere
from utils.video_writer_utils import VideoWriterBackend, make_video_writer

LOGGER = logging.getLogger("render_label_paths_telesim")
logging.basicConfig(level=logging.INFO, format="%(message)s")

STABILIZE_WINDOW = 5
FORWARD_SMOOTH_BLEND = 0.35
GPU_VIDEO_FORMAT = "ABGR"
EPS = 1e-6
STATUS_NOT_RUN = 0
STATUS_DONE = 1
STATUS_RETRY = 2
STATUS_SKIP = 3
TELESIM_STAGE_KEYS = (
    "actor_gpu_cache_upload_sec",
    "actor_visibility_sec",
    "actor_transform_sec",
    "actor_tensor_pack_sec",
    "actor_merge_update_sec",
    "gaussian_render_sec",
    "gpu_readback_sec",
    "perframe_light_sec",
    "camera_metadata_sec",
    "perframe_depth_sec",
    "perframe_png_sec",
    "mp4_write_sec",
    "h264_encode_sec",
    "h264_mux_sec",
    "video_close_sec",
)
TELESIM_ALIAS_STAGE_KEYS = ("render", "encode", "measured_total_sec")


def _new_stage_seconds() -> dict[str, float]:
    return {stage: 0.0 for stage in TELESIM_STAGE_KEYS}


@contextlib.contextmanager
def _measure_stage(stage_seconds: dict[str, float], stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        stage_seconds[stage] = stage_seconds.get(stage, 0.0) + (time.perf_counter() - start)


def _finalize_stage_seconds(stage_seconds: dict[str, float]) -> dict[str, float]:
    render_sec = (
        stage_seconds.get("gaussian_render_sec", 0.0)
        + stage_seconds.get("gpu_readback_sec", 0.0)
    )
    encode_sec = (
        stage_seconds.get("mp4_write_sec", 0.0)
        + stage_seconds.get("h264_encode_sec", 0.0)
    )
    measured_total_sec = sum(
        float(value)
        for key, value in stage_seconds.items()
        if key not in TELESIM_ALIAS_STAGE_KEYS
    )
    stage_seconds["render"] = render_sec
    stage_seconds["encode"] = encode_sec
    stage_seconds["measured_total_sec"] = measured_total_sec
    return dict(stage_seconds)


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


def _strict_gpu_backends_enabled() -> bool:
    return str(os.getenv("STRICT_GPU_BACKENDS", "")).lower() in (
        "1",
        "true",
        "yes",
        "on",
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
    """
    Resume check policy: ONLY check the direct `output_dir/<scene>/<label>.mp4`.

    Intentionally does not probe any nested directories (e.g. `output_dir/<scene>/<label>/frame_*`)
    to keep resume checks fast on remote / large mounts and to match the "direct area only" rule.
    """
    video_path = output_dir / scene_id / f"{label_id}.mp4"
    if video_path.is_file():
        return True
    return False


def _scan_existing_scene_mp4s(output_dir: Path, scene_id: str) -> tuple[dict[str, int], int]:
    """
    Fast resume helper: list already-rendered labels by scanning the output scene directory once.

    This avoids doing thousands of per-label `stat()` calls against (often slow) mounts when
    resuming and most outputs already exist.
    """
    scene_out = output_dir / scene_id
    if not scene_out.is_dir():
        return {}, 0
    sizes: dict[str, int] = {}
    total = 0
    try:
        with os.scandir(scene_out) as it:
            for ent in it:
                if ent.is_file() and ent.name.endswith(".mp4"):
                    label = Path(ent.name).stem
                    try:
                        sz = int(ent.stat().st_size)
                    except OSError:
                        sz = 0
                    sizes[label] = sz
                    total += sz
    except FileNotFoundError:
        return {}, 0
    except Exception:
        # Best-effort: resume checks should never crash rendering.
        return sizes, total
    return sizes, total


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


@dataclass(frozen=True)
class ActorMotionPlan:
    actor_id: str | None
    sequence_dir: Path | None
    frames: tuple[dict, ...]
    z_mode: str
    yaw_offset_rad: float
    actor_height_m: float | None = None
    actor_fps: float | None = None
    loop: bool | None = None


def load_actor_motion_plan(path: Path) -> ActorMotionPlan:
    plans = load_actor_motion_plans(path)
    if len(plans) != 1:
        raise ValueError(f"Expected exactly one actor plan in {path}, found {len(plans)}")
    return plans[0]


def load_actor_motion_plans(path: Path) -> tuple[ActorMotionPlan, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Actor plan must be a JSON object: {path}")
    actors = payload.get("actors")
    if isinstance(actors, list):
        plans = [
            _actor_motion_plan_from_payload(actor, path=path, index=index)
            for index, actor in enumerate(actors)
            if isinstance(actor, Mapping)
        ]
        if not plans:
            raise ValueError(f"Actor plan bundle must contain at least one actor object: {path}")
        return tuple(plans)
    return (_actor_motion_plan_from_payload(payload, path=path, index=0),)


def _actor_motion_plan_from_payload(
    payload: Mapping,
    *,
    path: Path,
    index: int,
) -> ActorMotionPlan:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Actor plan {index} must contain a non-empty frames list: {path}")
    sequence_dir = payload.get("sequence_dir")
    usable_frames = tuple(dict(frame) for frame in frames if isinstance(frame, Mapping))
    if not usable_frames:
        raise ValueError(f"Actor plan {index} has no usable frame objects: {path}")
    actor_height_m = payload.get("actor_height_m")
    actor_fps = payload.get("actor_fps")
    loop = payload.get("loop")
    return ActorMotionPlan(
        actor_id=str(payload["actor_id"]) if payload.get("actor_id") else None,
        sequence_dir=Path(sequence_dir).expanduser() if isinstance(sequence_dir, str) and sequence_dir else None,
        frames=usable_frames,
        z_mode=str(payload.get("z_mode") or "floor"),
        yaw_offset_rad=float(payload.get("yaw_offset_rad", 0.0) or 0.0),
        actor_height_m=float(actor_height_m) if actor_height_m is not None else None,
        actor_fps=float(actor_fps) if actor_fps is not None else None,
        loop=bool(loop) if loop is not None else None,
    )


@dataclass(frozen=True)
class ActorRenderRuntime:
    plan: ActorMotionPlan
    runtime: ActorRuntime
    gpu_sequence: GpuActorSequence | None = None
    gpu_error: str | None = None
    gpu_cache_upload_sec: float = 0.0


def build_actor_motion_plan_transforms(
    plan: ActorMotionPlan,
    *,
    frame_count: int,
    floor_z: float,
    actor_runtime: ActorRuntime,
) -> tuple[list[np.ndarray], list[int]]:
    if frame_count <= 0:
        return [], []
    if not plan.frames:
        raise ValueError("Actor motion plan has no usable frames.")

    actor_ground_z = float(floor_z + actor_runtime.options.foot_offset)
    cycle_mod = max(1, int(getattr(actor_runtime.options, "animation_cycle_mod", 1)))
    anim_step = (actor_runtime.options.fps / float(DEFAULT_VIDEO_FPS)) * cycle_mod
    anim_cursor = 0.0
    num_actor_frames = len(actor_runtime.sequence.frames)
    transforms: list[np.ndarray] = []
    actor_indices: list[int] = []

    for frame_idx in range(frame_count):
        raw_frame = plan.frames[min(frame_idx, len(plan.frames) - 1)]
        raw_position = raw_frame.get("position")
        if not isinstance(raw_position, Sequence) or len(raw_position) < 2:
            raise ValueError(f"Actor motion frame {frame_idx} is missing position[x,y].")
        x = float(raw_position[0])
        y = float(raw_position[1])
        if plan.z_mode == "absolute" and len(raw_position) >= 3:
            z = float(raw_position[2])
        else:
            z = actor_ground_z
        yaw_rad = float(raw_frame.get("yaw_rad", 0.0) or 0.0) + plan.yaw_offset_rad
        transform = build_transform_matrix(
            rotation_matrix_z_np(yaw_rad),
            np.array([x, y, z], dtype=np.float64),
        )
        transforms.append(transform)

        if raw_frame.get("animation_frame_index") is not None:
            anim_idx = int(raw_frame["animation_frame_index"]) % max(1, num_actor_frames)
        elif actor_runtime.options.loop:
            anim_idx = int(anim_cursor) % max(1, num_actor_frames)
        else:
            anim_idx = min(int(anim_cursor), max(0, num_actor_frames - 1))
        actor_indices.append(anim_idx)
        anim_cursor += anim_step

    return transforms, actor_indices


def _render_custom_gaussians(
    renderer: GaussianRendererBackend,
    pose: Pose,
    gaussians,
    *,
    need_depth_inv: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, object]:
    return _render_custom_gaussians_ex(
        renderer,
        pose,
        gaussians,
        return_render_tensor=False,
        need_depth_inv=need_depth_inv,
    )


def _render_tensor_to_gpu_format(render_tensor: torch.Tensor, *, gpu_format: str) -> torch.Tensor:
    render_uint8_gpu = (render_tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    render_uint8_gpu = render_uint8_gpu.permute(1, 2, 0)
    alpha = torch.full(
        (render_uint8_gpu.shape[0], render_uint8_gpu.shape[1], 1),
        255,
        device=render_uint8_gpu.device,
        dtype=torch.uint8,
    )
    fmt = gpu_format.upper()
    if fmt == "ABGR":
        # PyNvVideoCodec expects little-endian ABGR; RGBA byte order matches that.
        return torch.cat([render_uint8_gpu, alpha], dim=2)
    if fmt == "ARGB":
        # PyNvVideoCodec expects little-endian ARGB; BGRA byte order matches that.
        render_uint8_gpu = render_uint8_gpu[..., [2, 1, 0]]
        return torch.cat([render_uint8_gpu, alpha], dim=2)
    raise ValueError(f"Unsupported GPU video format: {gpu_format}")


def _render_custom_gaussians_ex(
    renderer: GaussianRendererBackend,
    pose: Pose,
    gaussians,
    *,
    return_render_tensor: bool,
    need_depth_inv: bool,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | None, object]:
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
    render_tensor = result["render"]
    depth_inv = None
    if need_depth_inv and result.get("depth") is not None:
        depth_inv = result["depth"].detach().cpu().numpy()
    if return_render_tensor:
        return render_tensor, depth_inv, camera
    image = render_tensor.permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8), depth_inv, camera


def _serialize_camera(
    *,
    renderer: GaussianRendererBackend,
    pose: Pose,
    frame_size: tuple[int, int],
    fov_y_rad: float,
) -> dict:
    w, h = frame_size
    if hasattr(renderer, "camera_matrices"):
        matrices = renderer.camera_matrices(pose)
        world_view = np.asarray(matrices["world_view"], dtype=np.float64)
        projection = np.asarray(matrices["full_projection"], dtype=np.float64)
        znear = float(matrices["intrinsics"]["znear"])
        zfar = float(matrices["intrinsics"]["zfar"])
        fovx = 2.0 * math.atan(math.tan(fov_y_rad * 0.5) * (w / float(h)))
        camera_to_world = np.linalg.inv(world_view)
        camera_center = camera_to_world[3][:3].tolist()
    else:
        camera = renderer._pose_to_camera(pose)  # pylint: disable=protected-access
        world_view = camera.world_view_transform.detach().cpu().numpy().T.astype(np.float64)
        projection = camera.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        znear = float(camera.znear)
        zfar = float(camera.zfar)
        fovx = float(getattr(camera, "FoVx", 2.0 * math.atan(math.tan(fov_y_rad * 0.5) * (w / float(h)))))
        fov_y_rad = float(getattr(camera, "FoVy", fov_y_rad))
        camera_to_world = np.linalg.inv(world_view)
        camera_center = camera_to_world[:3, 3].tolist()
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
        "znear": znear,
        "zfar": zfar,
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
    scene_dir: Path,
    label_id: str,
    camera_frames: Sequence[dict],
) -> Path:
    """
    Persist camera metadata JSON for one label path.

    New format (preferred): one file per path at the same level as the MP4:
      <output_dir>/<scene>/<label>_camera.json
    """
    out_path = scene_dir / f"{label_id}_camera.json"
    payload = {
        "dataset_root": str(scene_dir.parent),
        "scene": str(scene_dir.name),
        "label": str(label_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames": list(camera_frames),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


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
    raise RuntimeError("_write_video_frames is deprecated; use make_video_writer directly.")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_gaussian_model(scene_dir: Path, explicit_model: Path | None) -> Path:
    if explicit_model is not None:
        return explicit_model

    preferred_plys = [
        scene_dir / "3dgs_raw.ply",
        scene_dir / "3dgs_decompressed.ply",
        scene_dir / "decompressed.ply",
        scene_dir / "debug-decompressed.ply",
        scene_dir / "point_cloud.ply",
        scene_dir / "3dgs_compressed.ply",
    ]
    for path in preferred_plys:
        if path.exists():
            if path.name == "3dgs_compressed.ply":
                LOGGER.warning(
                    "Using packed compressed Gaussian PLY as a last resort: %s. "
                    "Prefer 3dgs_raw.ply or unpack it to 3dgs_decompressed.ply first.",
                    path,
                )
            return path
    return preferred_plys[0]


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
    if args.save_depth_maps or args.rgb_frames:
        _safe_mkdir(frames_dir)

    video_path = scene_dir / f"{label_id}.mp4"
    frame_prefix = "frame"

    stage_seconds = _new_stage_seconds()
    frames_rendered = 0
    fov_y_rad = math.radians(float(args.fov_deg))

    light_config: LightFilterConfig | None = getattr(args, "light_config", None)
    cl_config: CameraLightConfig | None = getattr(args, "cl_config", None)
    cl_light_world = getattr(args, "cl_light_world", None)
    light_seed_offset = getattr(args, "light_seed_offset", 0)

    video_backend = VideoWriterBackend(str(args.video_backend or VideoWriterBackend.NVENC.value))
    use_gpu_video = bool(args.video) and video_backend == VideoWriterBackend.GPU
    video_stage = "h264_encode_sec" if use_gpu_video else "mp4_write_sec"
    need_depth_inv = bool(args.save_depth_maps) or (cl_config is not None and cl_config.active())
    camera_frames: list[dict] = []
    if use_gpu_video and (
        (light_config is not None and light_config.enabled())
        or (cl_config is not None and cl_config.active())
    ):
        raise RuntimeError(
            "video-backend=gpu does not support light filters / camera light shading; "
            "use --video-backend nvenc|cpu or disable --light-mode / --cl-enable."
        )

    def _render_frames(writer=None):
        nonlocal frames_rendered
        for idx, (pose, _) in enumerate(poses):
            if use_gpu_video:
                with _measure_stage(stage_seconds, "gaussian_render_sec"):
                    render_tensor, depth_inv, camera = _render_custom_gaussians_ex(
                        renderer,
                        pose,
                        renderer._gaussians,  # pylint: disable=protected-access
                        return_render_tensor=True,
                        need_depth_inv=need_depth_inv,
                    )
                rgb = None
                if args.rgb_frames:
                    with _measure_stage(stage_seconds, "gpu_readback_sec"):
                        rgb = (
                            (render_tensor.clamp(0.0, 1.0) * 255.0)
                            .to(torch.uint8)
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                with _measure_stage(stage_seconds, "gaussian_render_sec"):
                    rgb, depth_inv, camera = _render_custom_gaussians(
                        renderer,
                        pose,
                        renderer._gaussians,  # pylint: disable=protected-access
                        need_depth_inv=need_depth_inv,
                    )
            if not use_gpu_video:
                if light_config is not None and light_config.enabled():
                    with _measure_stage(stage_seconds, "perframe_light_sec"):
                        rgb = _apply_light_filter_if_enabled(
                            rgb,
                            light_config,
                            frame_index=idx,
                            seed_offset=light_seed_offset,
                        )
                if cl_config is not None and cl_config.active():
                    with _measure_stage(stage_seconds, "perframe_light_sec"):
                        rgb = _apply_camera_light_if_enabled(
                            rgb,
                            depth_inv,
                            camera,
                            cl_config,
                            cl_light_world=cl_light_world,
                        )
            if args.rotate_180:
                if use_gpu_video:
                    if rgb is not None:
                        rgb = np.flipud(np.fliplr(rgb))
                else:
                    rgb = np.flipud(np.fliplr(rgb))
            frames_rendered += 1
            if args.save_camera_metadata:
                with _measure_stage(stage_seconds, "camera_metadata_sec"):
                    cam_payload = _serialize_camera(
                        renderer=renderer,
                        pose=pose,
                        frame_size=tuple(args.resolution),
                        fov_y_rad=fov_y_rad,
                    )
                    camera_frames.append({"frame": int(idx), **cam_payload})
            if args.save_depth_maps and depth_inv is not None:
                with _measure_stage(stage_seconds, "perframe_depth_sec"):
                    _save_depth_map(
                        depth_inv=depth_inv,
                        frames_dir=frames_dir,
                        frame_prefix=frame_prefix,
                        frame_idx=idx,
                        rotate_180=args.rotate_180,
                    )
            if args.rgb_frames:
                frame_path = frames_dir / f"{frame_prefix}_{idx:04d}.png"
                if rgb is None:
                    raise RuntimeError("rgb frame requested but was not produced.")
                with _measure_stage(stage_seconds, "perframe_png_sec"):
                    imageio.imwrite(frame_path, rgb)

            if args.video and writer is not None:
                with _measure_stage(stage_seconds, video_stage):
                    if use_gpu_video:
                        frame_gpu = _render_tensor_to_gpu_format(
                            render_tensor, gpu_format=GPU_VIDEO_FORMAT
                        )
                        if args.rotate_180:
                            frame_gpu = frame_gpu.flip((0, 1))
                        writer.append_data(frame_gpu.contiguous())
                    else:
                        writer.append_data(rgb)

    if args.video:
        w, h = int(args.resolution[0]), int(args.resolution[1])
        writer = make_video_writer(
            video_path,
            fps=args.video_fps,
            backend=video_backend,
            nvenc_preset=args.video_nvenc_preset,
            nvenc_bitrate=args.video_nvenc_bitrate,
            width=w,
            height=h,
            gpu_format=GPU_VIDEO_FORMAT,
            encode_timer=(
                (lambda: _measure_stage(stage_seconds, "h264_encode_sec"))
                if use_gpu_video
                else None
            ),
            mux_timer=(
                (lambda: _measure_stage(stage_seconds, "h264_mux_sec"))
                if use_gpu_video
                else None
            ),
        )
        try:
            _render_frames(writer=writer)
        finally:
            close_timer = (
                contextlib.nullcontext()
                if use_gpu_video
                else _measure_stage(stage_seconds, "video_close_sec")
            )
            with close_timer:
                writer.close()
    else:
        _render_frames(writer=None)

    if args.save_camera_metadata:
        with _measure_stage(stage_seconds, "camera_metadata_sec"):
            _write_camera_metadata(scene_dir=scene_dir, label_id=label_id, camera_frames=camera_frames)

    stage_payload = _finalize_stage_seconds(stage_seconds)
    duration = stage_payload["measured_total_sec"]
    return {
        "scene_id": args.scene,
        "label_id": label_id,
        "frames": frames_rendered,
        "duration_sec": duration,
        "stage_seconds": stage_payload,
    }


def _camera_xy_sequence(
    prepared: PreparedPath,
    args: argparse.Namespace,
    actor_runtime: ActorRuntime | None,
) -> list[np.ndarray]:
    if actor_runtime is None or getattr(args, "actor_motion_plan", None) is not None:
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


def _load_actor_render_runtime(
    *,
    plan: ActorMotionPlan,
    args: argparse.Namespace,
    renderer: GaussianRendererBackend,
) -> ActorRenderRuntime:
    sequence_dir = plan.sequence_dir or args.actor_seq_dir
    if sequence_dir is None:
        raise ValueError("--actor-plan-json requires --actor-seq-dir or sequence_dir in every actor plan.")
    actor_options = ActorOptions(
        sequence_dir=sequence_dir,
        pattern=args.actor_pattern,
        height=float(plan.actor_height_m if plan.actor_height_m is not None else args.actor_height),
        speed=float(args.actor_speed),
        fps=float(plan.actor_fps if plan.actor_fps is not None else args.actor_fps),
        loop=bool(plan.loop if plan.loop is not None else args.actor_loop),
        foot_offset=float(args.actor_foot_offset),
        follow_distance=float(args.follow_distance),
        buffer_distance=float(args.follow_buffer),
        animation_cycle_mod=int(args.animation_cycle_mod),
    )
    _validate_actor_options(actor_options)
    actor_sequence = load_actor_sequence(actor_options, debug=bool(args.verbose))
    actor_runtime = ActorRuntime(options=actor_options, sequence=actor_sequence)

    gpu_actor_sequence: GpuActorSequence | None = None
    actor_gpu_error: str | None = None
    actor_gpu_cache_upload_sec = 0.0
    if bool(getattr(args, "actor_gpu_resident", False)):
        base_gaussians = renderer._gaussians  # pylint: disable=protected-access
        base_device = base_gaussians.get_xyz.device
        actor_label = plan.actor_id or str(sequence_dir)
        if base_device.type != "cuda":
            actor_gpu_error = f"scene gaussians are on {base_device}, not CUDA"
            LOGGER.warning("Actor GPU-resident cache disabled for %s: %s", actor_label, actor_gpu_error)
        else:
            cache_mb = float(getattr(args, "actor_gpu_cache_mb", 0.0) or 0.0)
            memory_cap_mb = cache_mb if cache_mb > 0.0 else None
            sh_mode = str(getattr(args, "actor_gpu_sh_mode", "copy"))
            scene_rest_dim = int(base_gaussians.get_features_rest.shape[1])
            start = time.perf_counter()
            try:
                gpu_actor_sequence = build_gpu_actor_sequence(
                    actor_runtime.sequence,
                    device=base_device,
                    target_rest_dim=scene_rest_dim,
                    memory_cap_mb=memory_cap_mb,
                    sh_mode=sh_mode,
                )
                actor_gpu_cache_upload_sec = time.perf_counter() - start
                LOGGER.info(
                    "Actor GPU-resident cache enabled: actor=%s frames=%d size=%s sh_mode=%s upload=%.3fs",
                    actor_label,
                    gpu_actor_sequence.frame_count,
                    _format_bytes(gpu_actor_sequence.bytes_allocated),
                    gpu_actor_sequence.sh_mode,
                    actor_gpu_cache_upload_sec,
                )
            except Exception as exc:  # pylint: disable=broad-except
                actor_gpu_cache_upload_sec = time.perf_counter() - start
                actor_gpu_error = str(exc)
                if _strict_gpu_backends_enabled():
                    raise RuntimeError(
                        f"Actor GPU-resident cache failed in strict mode: {actor_gpu_error}"
                    ) from exc
                LOGGER.warning(
                    "Actor GPU-resident cache disabled for %s: %s; falling back to exact CPU/PLY transform path.",
                    actor_label,
                    actor_gpu_error,
                )

    return ActorRenderRuntime(
        plan=plan,
        runtime=actor_runtime,
        gpu_sequence=gpu_actor_sequence,
        gpu_error=actor_gpu_error,
        gpu_cache_upload_sec=actor_gpu_cache_upload_sec,
    )


def _validate_actor_options(actor_options: ActorOptions) -> None:
    if actor_options.fps <= 0.0:
        raise ValueError("actor_fps must be positive.")
    if actor_options.speed <= 0.0:
        raise ValueError("actor_speed must be positive.")
    if actor_options.buffer_distance > actor_options.follow_distance:
        raise ValueError("follow_buffer must be <= follow_distance.")


def _actor_render_frame(
    *,
    render_runtime: ActorRenderRuntime,
    actor_idx: int,
    transform: np.ndarray,
    base_gaussians,
    scene_rest_dim: int,
    stage_seconds: dict[str, float],
) -> ActorRenderFrame:
    if render_runtime.gpu_sequence is not None:
        with _measure_stage(stage_seconds, "actor_transform_sec"):
            return transform_gpu_actor_frame(
                render_runtime.gpu_sequence,
                actor_idx,
                transform,
            )
    sequence_frame = render_runtime.runtime.sequence.frames[actor_idx]
    with _measure_stage(stage_seconds, "actor_transform_sec"):
        actor_data = apply_transform_to_frame(
            sequence_frame,
            render_runtime.runtime.sequence,
            transform,
        )
    with _measure_stage(stage_seconds, "actor_tensor_pack_sec"):
        return actor_data_to_tensors(
            actor_data,
            render_runtime.runtime.sequence,
            device=base_gaussians.get_xyz.device,
            target_rest_dim=scene_rest_dim,
        )


def render_label_with_actor_plans(
    *,
    renderer: GaussianRendererBackend,
    prepared: PreparedPath,
    output_dir: Path,
    label_id: str,
    args: argparse.Namespace,
    actor_render_runtimes: Sequence[ActorRenderRuntime],
) -> dict:
    if not actor_render_runtimes:
        raise ValueError("actor_render_runtimes cannot be empty")

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
    actor_tracks = [
        build_actor_motion_plan_transforms(
            render_runtime.plan,
            frame_count=len(poses),
            floor_z=prepared.floor_z,
            actor_runtime=render_runtime.runtime,
        )
        for render_runtime in actor_render_runtimes
    ]
    if args.minimal_frames is not None and args.minimal_frames > 0:
        poses = poses[: args.minimal_frames]
        actor_tracks = [
            (transforms[: args.minimal_frames], indices[: args.minimal_frames])
            for transforms, indices in actor_tracks
        ]

    scene_dir = output_dir / args.scene
    _safe_mkdir(scene_dir)
    frames_dir = scene_dir / label_id
    if args.save_depth_maps or args.rgb_frames:
        _safe_mkdir(frames_dir)

    video_path = scene_dir / f"{label_id}.mp4"
    frame_prefix = "frame"

    stage_seconds = _new_stage_seconds()
    stage_seconds["actor_gpu_cache_upload_sec"] = sum(
        max(0.0, float(render_runtime.gpu_cache_upload_sec or 0.0))
        for render_runtime in actor_render_runtimes
    )
    frames_rendered = 0
    fov_y_rad = math.radians(float(args.fov_deg))

    base_gaussians = renderer._gaussians  # pylint: disable=protected-access
    scene_rest_dim = int(base_gaussians.get_features_rest.shape[1])
    combined_model: MultiActorCombinedGaussianModel | None = None
    combined_actor_signature: tuple[int, ...] | None = None
    actor_cull_enabled = bool(getattr(args, "actor_visibility_culling", False))
    actor_cull_margin_m = float(getattr(args, "actor_cull_margin_m", 0.0) or 0.0)
    actor_culled_frames = 0
    actor_candidate_frames = 0
    actor_visible_frames = 0
    actor_rendered_frames = 0
    actor_frame_metadata: list[dict] = []
    actor_gpu_requested = bool(getattr(args, "actor_gpu_resident", False))

    light_config: LightFilterConfig | None = getattr(args, "light_config", None)
    cl_config: CameraLightConfig | None = getattr(args, "cl_config", None)
    cl_light_world = getattr(args, "cl_light_world", None)
    light_seed_offset = getattr(args, "light_seed_offset", 0)

    video_backend = VideoWriterBackend(str(args.video_backend or VideoWriterBackend.NVENC.value))
    use_gpu_video = bool(args.video) and video_backend == VideoWriterBackend.GPU
    video_stage = "h264_encode_sec" if use_gpu_video else "mp4_write_sec"
    need_depth_inv = bool(args.save_depth_maps) or (cl_config is not None and cl_config.active())
    camera_frames: list[dict] = []
    if use_gpu_video and (
        (light_config is not None and light_config.enabled())
        or (cl_config is not None and cl_config.active())
    ):
        raise RuntimeError(
            "video-backend=gpu does not support light filters / camera light shading; "
            "use --video-backend nvenc|cpu or disable --light-mode / --cl-enable."
        )

    def _render_frames(writer=None):
        nonlocal frames_rendered, combined_model, combined_actor_signature
        nonlocal actor_culled_frames, actor_candidate_frames, actor_visible_frames, actor_rendered_frames
        for idx, (pose, _) in enumerate(poses):
            render_gaussians = base_gaussians
            visible_actor_frames: list[ActorRenderFrame] = []
            frame_actor_metadata: list[dict] = []
            camera_for_cull = None
            for render_runtime, (actor_transforms, actor_indices) in zip(actor_render_runtimes, actor_tracks):
                if idx >= len(actor_transforms) or idx >= len(actor_indices):
                    continue
                transform = actor_transforms[idx]
                actor_idx = actor_indices[idx]
                actor_visible = True
                actor_candidate_frames += 1
                if actor_cull_enabled:
                    if camera_for_cull is None:
                        camera_for_cull = renderer._pose_to_camera(pose)  # pylint: disable=protected-access
                    with _measure_stage(stage_seconds, "actor_visibility_sec"):
                        sphere = transformed_actor_sphere(
                            transform,
                            radius_xy_m=render_runtime.runtime.sequence.radius_xy,
                            height_m=render_runtime.runtime.sequence.height,
                            margin_m=actor_cull_margin_m,
                        )
                        visibility = sphere_visible_in_camera(
                            center_world=sphere.center_world,
                            radius_m=sphere.radius_m,
                            world_view_transform=camera_for_cull.world_view_transform.detach().cpu().numpy(),
                            fov_x_rad=float(camera_for_cull.FoVx),
                            fov_y_rad=float(camera_for_cull.FoVy),
                            znear=float(camera_for_cull.znear),
                            zfar=float(camera_for_cull.zfar),
                            matrix_is_transposed=True,
                        )
                    actor_visible = bool(visibility.visible)
                    if not actor_visible:
                        actor_culled_frames += 1
                if actor_visible:
                    actor_visible_frames += 1
                    actor_render = _actor_render_frame(
                        render_runtime=render_runtime,
                        actor_idx=actor_idx,
                        transform=transform,
                        base_gaussians=base_gaussians,
                        scene_rest_dim=scene_rest_dim,
                        stage_seconds=stage_seconds,
                    )
                    visible_actor_frames.append(actor_render)
                    actor_rendered_frames += 1
                frame_actor_metadata.append(
                    {
                        "actor_id": render_runtime.plan.actor_id,
                        "candidate": True,
                        "visible": bool(actor_visible),
                        "rendered": bool(actor_visible),
                        "animation_frame_index": int(actor_idx),
                    }
                )

            if visible_actor_frames:
                with _measure_stage(stage_seconds, "actor_merge_update_sec"):
                    signature = tuple(int(frame.xyz.shape[0]) for frame in visible_actor_frames)
                    if combined_model is None or combined_actor_signature != signature:
                        combined_actor_signature = signature
                        combined_model = MultiActorCombinedGaussianModel(base_gaussians, visible_actor_frames)
                    else:
                        combined_model.update_actors(visible_actor_frames)
                render_gaussians = combined_model
            else:
                combined_actor_signature = None

            actor_frame_metadata.append({"frame": int(idx), "actors": frame_actor_metadata})

            if use_gpu_video:
                with _measure_stage(stage_seconds, "gaussian_render_sec"):
                    render_tensor, depth_inv, camera = _render_custom_gaussians_ex(
                        renderer,
                        pose,
                        render_gaussians,
                        return_render_tensor=True,
                        need_depth_inv=need_depth_inv,
                    )
                rgb = None
                if args.rgb_frames:
                    with _measure_stage(stage_seconds, "gpu_readback_sec"):
                        rgb = (
                            (render_tensor.clamp(0.0, 1.0) * 255.0)
                            .to(torch.uint8)
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                with _measure_stage(stage_seconds, "gaussian_render_sec"):
                    rgb, depth_inv, camera = _render_custom_gaussians(
                        renderer,
                        pose,
                        render_gaussians,
                        need_depth_inv=need_depth_inv,
                    )
            if not use_gpu_video:
                if light_config is not None and light_config.enabled():
                    with _measure_stage(stage_seconds, "perframe_light_sec"):
                        rgb = _apply_light_filter_if_enabled(
                            rgb,
                            light_config,
                            frame_index=idx,
                            seed_offset=light_seed_offset,
                        )
                if cl_config is not None and cl_config.active():
                    with _measure_stage(stage_seconds, "perframe_light_sec"):
                        rgb = _apply_camera_light_if_enabled(
                            rgb,
                            depth_inv,
                            camera,
                            cl_config,
                            cl_light_world=cl_light_world,
                        )
            if args.rotate_180:
                if use_gpu_video:
                    if rgb is not None:
                        rgb = np.flipud(np.fliplr(rgb))
                else:
                    rgb = np.flipud(np.fliplr(rgb))
            frames_rendered += 1
            if args.save_camera_metadata:
                with _measure_stage(stage_seconds, "camera_metadata_sec"):
                    cam_payload = _serialize_camera(
                        renderer=renderer,
                        pose=pose,
                        frame_size=tuple(args.resolution),
                        fov_y_rad=fov_y_rad,
                    )
                    camera_frames.append({"frame": int(idx), **cam_payload})
            if args.save_depth_maps and depth_inv is not None:
                with _measure_stage(stage_seconds, "perframe_depth_sec"):
                    _save_depth_map(
                        depth_inv=depth_inv,
                        frames_dir=frames_dir,
                        frame_prefix=frame_prefix,
                        frame_idx=idx,
                        rotate_180=args.rotate_180,
                    )
            if args.rgb_frames:
                frame_path = frames_dir / f"{frame_prefix}_{idx:04d}.png"
                if rgb is None:
                    raise RuntimeError("rgb frame requested but was not produced.")
                with _measure_stage(stage_seconds, "perframe_png_sec"):
                    imageio.imwrite(frame_path, rgb)

            if args.video and writer is not None:
                with _measure_stage(stage_seconds, video_stage):
                    if use_gpu_video:
                        frame_gpu = _render_tensor_to_gpu_format(
                            render_tensor, gpu_format=GPU_VIDEO_FORMAT
                        )
                        if args.rotate_180:
                            frame_gpu = frame_gpu.flip((0, 1))
                        writer.append_data(frame_gpu.contiguous())
                    else:
                        writer.append_data(rgb)

    if args.video:
        w, h = int(args.resolution[0]), int(args.resolution[1])
        writer = make_video_writer(
            video_path,
            fps=args.video_fps,
            backend=video_backend,
            nvenc_preset=args.video_nvenc_preset,
            nvenc_bitrate=args.video_nvenc_bitrate,
            width=w,
            height=h,
            gpu_format=GPU_VIDEO_FORMAT,
            encode_timer=(
                (lambda: _measure_stage(stage_seconds, "h264_encode_sec"))
                if use_gpu_video
                else None
            ),
            mux_timer=(
                (lambda: _measure_stage(stage_seconds, "h264_mux_sec"))
                if use_gpu_video
                else None
            ),
        )
        try:
            _render_frames(writer=writer)
        finally:
            close_timer = (
                contextlib.nullcontext()
                if use_gpu_video
                else _measure_stage(stage_seconds, "video_close_sec")
            )
            with close_timer:
                writer.close()
    else:
        _render_frames(writer=None)

    if args.save_camera_metadata:
        with _measure_stage(stage_seconds, "camera_metadata_sec"):
            _write_camera_metadata(scene_dir=scene_dir, label_id=label_id, camera_frames=camera_frames)

    actor_metadata_path: Path | None = None
    if bool(getattr(args, "save_actor_metadata", False)):
        actor_metadata_path = scene_dir / f"{label_id}_actors.json"
        actor_metadata_path.write_text(
            json.dumps(
                {
                    "scene_id": args.scene,
                    "label_id": label_id,
                    "actor_plan_json": (
                        str(getattr(args, "actor_plan_json"))
                        if getattr(args, "actor_plan_json", None) is not None
                        else None
                    ),
                    "actor_ids": [
                        render_runtime.plan.actor_id for render_runtime in actor_render_runtimes
                    ],
                    "candidate_frames": actor_candidate_frames,
                    "visible_frames": actor_visible_frames,
                    "culled_frames": actor_culled_frames,
                    "rendered_frames": actor_rendered_frames,
                    "frames": actor_frame_metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    stage_payload = _finalize_stage_seconds(stage_seconds)
    duration = stage_payload["measured_total_sec"]
    gpu_sequences = [item.gpu_sequence for item in actor_render_runtimes if item.gpu_sequence is not None]
    actor_gpu_errors = {
        str(item.plan.actor_id or index): item.gpu_error
        for index, item in enumerate(actor_render_runtimes)
        if item.gpu_error
    }
    return {
        "scene_id": args.scene,
        "label_id": label_id,
        "frames": frames_rendered,
        "duration_sec": duration,
        "stage_seconds": stage_payload,
        "actor_culled_frames": actor_culled_frames,
        "actor_candidate_frames": actor_candidate_frames,
        "actor_visible_frames": actor_visible_frames,
        "actor_rendered_frames": actor_rendered_frames,
        "actor_metadata_path": str(actor_metadata_path) if actor_metadata_path is not None else None,
        "actor_count": len(actor_render_runtimes),
        "actor_ids": [item.plan.actor_id for item in actor_render_runtimes],
        "actor_gpu_resident_requested": actor_gpu_requested,
        "actor_gpu_resident": len(gpu_sequences) == len(actor_render_runtimes),
        "actor_gpu_cache_bytes": sum(item.bytes_allocated for item in gpu_sequences),
        "actor_gpu_sh_mode": (
            gpu_sequences[0].sh_mode
            if gpu_sequences
            else str(getattr(args, "actor_gpu_sh_mode", "copy"))
        ),
        "actor_gpu_error": actor_gpu_errors or None,
    }


def render_label_with_actor(
    *,
    renderer: GaussianRendererBackend,
    prepared: PreparedPath,
    output_dir: Path,
    label_id: str,
    args: argparse.Namespace,
    actor_runtime: ActorRuntime,
    gpu_actor_sequence: GpuActorSequence | None = None,
    actor_gpu_error: str | None = None,
    actor_gpu_cache_upload_sec: float = 0.0,
) -> dict:
    actor_motion_plan: ActorMotionPlan | None = getattr(args, "actor_motion_plan", None)
    if actor_motion_plan is not None:
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
        actor_transforms, actor_indices = build_actor_motion_plan_transforms(
            actor_motion_plan,
            frame_count=len(poses),
            floor_z=prepared.floor_z,
            actor_runtime=actor_runtime,
        )
    else:
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
    if args.save_depth_maps or args.rgb_frames:
        _safe_mkdir(frames_dir)

    video_path = scene_dir / f"{label_id}.mp4"
    frame_prefix = "frame"

    stage_seconds = _new_stage_seconds()
    stage_seconds["actor_gpu_cache_upload_sec"] = max(
        0.0,
        float(actor_gpu_cache_upload_sec or 0.0),
    )
    frames_rendered = 0
    fov_y_rad = math.radians(float(args.fov_deg))

    base_gaussians = renderer._gaussians  # pylint: disable=protected-access
    scene_rest_dim = int(base_gaussians.get_features_rest.shape[1])
    combined_model: CombinedGaussianModel | None = None
    combined_actor_size: int | None = None
    actor_cull_enabled = bool(getattr(args, "actor_visibility_culling", False))
    actor_cull_margin_m = float(getattr(args, "actor_cull_margin_m", 0.0) or 0.0)
    actor_culled_frames = 0
    actor_candidate_frames = 0
    actor_visible_frames = 0
    actor_rendered_frames = 0
    actor_frame_metadata: list[dict] = []
    actor_id = str(getattr(args, "job_actor_id", "") or "")
    if actor_motion_plan is not None and actor_motion_plan.actor_id:
        actor_id = actor_motion_plan.actor_id
    actor_gpu_requested = bool(getattr(args, "actor_gpu_resident", False))

    light_config: LightFilterConfig | None = getattr(args, "light_config", None)
    cl_config: CameraLightConfig | None = getattr(args, "cl_config", None)
    cl_light_world = getattr(args, "cl_light_world", None)
    light_seed_offset = getattr(args, "light_seed_offset", 0)

    video_backend = VideoWriterBackend(str(args.video_backend or VideoWriterBackend.NVENC.value))
    use_gpu_video = bool(args.video) and video_backend == VideoWriterBackend.GPU
    video_stage = "h264_encode_sec" if use_gpu_video else "mp4_write_sec"
    need_depth_inv = bool(args.save_depth_maps) or (cl_config is not None and cl_config.active())
    camera_frames: list[dict] = []
    if use_gpu_video and (
        (light_config is not None and light_config.enabled())
        or (cl_config is not None and cl_config.active())
    ):
        raise RuntimeError(
            "video-backend=gpu does not support light filters / camera light shading; "
            "use --video-backend nvenc|cpu or disable --light-mode / --cl-enable."
        )

    def _render_frames(writer=None):
        nonlocal frames_rendered, combined_model, combined_actor_size, actor_culled_frames
        nonlocal actor_candidate_frames, actor_visible_frames, actor_rendered_frames
        for idx, ((pose, _), transform, actor_idx) in enumerate(
            zip(poses, actor_transforms, actor_indices)
        ):
            render_gaussians = base_gaussians
            actor_visible = True
            actor_candidate_frames += 1
            if actor_cull_enabled:
                with _measure_stage(stage_seconds, "actor_visibility_sec"):
                    camera_for_cull = renderer._pose_to_camera(pose)  # pylint: disable=protected-access
                    sphere = transformed_actor_sphere(
                        transform,
                        radius_xy_m=actor_runtime.sequence.radius_xy,
                        height_m=actor_runtime.sequence.height,
                        margin_m=actor_cull_margin_m,
                    )
                    visibility = sphere_visible_in_camera(
                        center_world=sphere.center_world,
                        radius_m=sphere.radius_m,
                        world_view_transform=camera_for_cull.world_view_transform.detach().cpu().numpy(),
                        fov_x_rad=float(camera_for_cull.FoVx),
                        fov_y_rad=float(camera_for_cull.FoVy),
                        znear=float(camera_for_cull.znear),
                        zfar=float(camera_for_cull.zfar),
                        matrix_is_transposed=True,
                    )
                actor_visible = bool(visibility.visible)
                if not actor_visible:
                    actor_culled_frames += 1
            if actor_visible:
                actor_visible_frames += 1
                if gpu_actor_sequence is not None:
                    with _measure_stage(stage_seconds, "actor_transform_sec"):
                        actor_render = transform_gpu_actor_frame(
                            gpu_actor_sequence,
                            actor_idx,
                            transform,
                        )
                else:
                    sequence_frame = actor_runtime.sequence.frames[actor_idx]
                    # Note: apply_transform_to_frame expects (ActorSequenceFrame, ActorSequence, transform).
                    with _measure_stage(stage_seconds, "actor_transform_sec"):
                        actor_data = apply_transform_to_frame(
                            sequence_frame,
                            actor_runtime.sequence,
                            transform,
                        )
                    with _measure_stage(stage_seconds, "actor_tensor_pack_sec"):
                        actor_render = actor_data_to_tensors(
                            actor_data,
                            actor_runtime.sequence,
                            device=base_gaussians.get_xyz.device,
                            target_rest_dim=scene_rest_dim,
                        )
                with _measure_stage(stage_seconds, "actor_merge_update_sec"):
                    current_actor_size = int(actor_render.xyz.shape[0])
                    if combined_model is None or combined_actor_size != current_actor_size:
                        combined_actor_size = current_actor_size
                        combined_model = CombinedGaussianModel(base_gaussians, actor_render)
                    else:
                        combined_model.update_actor(actor_render)
                render_gaussians = combined_model
                actor_rendered_frames += 1
            else:
                combined_actor_size = None
            actor_frame_metadata.append(
                {
                    "frame": int(idx),
                    "actors": [
                        {
                            "actor_id": actor_id or None,
                            "candidate": True,
                            "visible": bool(actor_visible),
                            "rendered": bool(actor_visible),
                            "animation_frame_index": int(actor_idx),
                        }
                    ],
                }
            )

            if use_gpu_video:
                with _measure_stage(stage_seconds, "gaussian_render_sec"):
                    render_tensor, depth_inv, camera = _render_custom_gaussians_ex(
                        renderer,
                        pose,
                        render_gaussians,
                        return_render_tensor=True,
                        need_depth_inv=need_depth_inv,
                    )
                rgb = None
                if args.rgb_frames:
                    with _measure_stage(stage_seconds, "gpu_readback_sec"):
                        rgb = (
                            (render_tensor.clamp(0.0, 1.0) * 255.0)
                            .to(torch.uint8)
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                with _measure_stage(stage_seconds, "gaussian_render_sec"):
                    rgb, depth_inv, camera = _render_custom_gaussians(
                        renderer,
                        pose,
                        render_gaussians,
                        need_depth_inv=need_depth_inv,
                    )
            if not use_gpu_video:
                if light_config is not None and light_config.enabled():
                    with _measure_stage(stage_seconds, "perframe_light_sec"):
                        rgb = _apply_light_filter_if_enabled(
                            rgb,
                            light_config,
                            frame_index=idx,
                            seed_offset=light_seed_offset,
                        )
                if cl_config is not None and cl_config.active():
                    with _measure_stage(stage_seconds, "perframe_light_sec"):
                        rgb = _apply_camera_light_if_enabled(
                            rgb,
                            depth_inv,
                            camera,
                            cl_config,
                            cl_light_world=cl_light_world,
                        )
            if args.rotate_180:
                if use_gpu_video:
                    if rgb is not None:
                        rgb = np.flipud(np.fliplr(rgb))
                else:
                    rgb = np.flipud(np.fliplr(rgb))
            frames_rendered += 1
            if args.save_camera_metadata:
                with _measure_stage(stage_seconds, "camera_metadata_sec"):
                    cam_payload = _serialize_camera(
                        renderer=renderer,
                        pose=pose,
                        frame_size=tuple(args.resolution),
                        fov_y_rad=fov_y_rad,
                    )
                    camera_frames.append({"frame": int(idx), **cam_payload})
            if args.save_depth_maps and depth_inv is not None:
                with _measure_stage(stage_seconds, "perframe_depth_sec"):
                    _save_depth_map(
                        depth_inv=depth_inv,
                        frames_dir=frames_dir,
                        frame_prefix=frame_prefix,
                        frame_idx=idx,
                        rotate_180=args.rotate_180,
                    )
            if args.rgb_frames:
                frame_path = frames_dir / f"{frame_prefix}_{idx:04d}.png"
                if rgb is None:
                    raise RuntimeError("rgb frame requested but was not produced.")
                with _measure_stage(stage_seconds, "perframe_png_sec"):
                    imageio.imwrite(frame_path, rgb)

            if args.video and writer is not None:
                with _measure_stage(stage_seconds, video_stage):
                    if use_gpu_video:
                        frame_gpu = _render_tensor_to_gpu_format(
                            render_tensor, gpu_format=GPU_VIDEO_FORMAT
                        )
                        if args.rotate_180:
                            frame_gpu = frame_gpu.flip((0, 1))
                        writer.append_data(frame_gpu.contiguous())
                    else:
                        writer.append_data(rgb)

    if args.video:
        w, h = int(args.resolution[0]), int(args.resolution[1])
        writer = make_video_writer(
            video_path,
            fps=args.video_fps,
            backend=video_backend,
            nvenc_preset=args.video_nvenc_preset,
            nvenc_bitrate=args.video_nvenc_bitrate,
            width=w,
            height=h,
            gpu_format=GPU_VIDEO_FORMAT,
            encode_timer=(
                (lambda: _measure_stage(stage_seconds, "h264_encode_sec"))
                if use_gpu_video
                else None
            ),
            mux_timer=(
                (lambda: _measure_stage(stage_seconds, "h264_mux_sec"))
                if use_gpu_video
                else None
            ),
        )
        try:
            _render_frames(writer=writer)
        finally:
            close_timer = (
                contextlib.nullcontext()
                if use_gpu_video
                else _measure_stage(stage_seconds, "video_close_sec")
            )
            with close_timer:
                writer.close()
    else:
        _render_frames(writer=None)

    if args.save_camera_metadata:
        with _measure_stage(stage_seconds, "camera_metadata_sec"):
            _write_camera_metadata(scene_dir=scene_dir, label_id=label_id, camera_frames=camera_frames)

    actor_metadata_path: Path | None = None
    if bool(getattr(args, "save_actor_metadata", False)):
        actor_metadata_path = scene_dir / f"{label_id}_actors.json"
        actor_metadata_path.write_text(
            json.dumps(
                {
                    "scene_id": args.scene,
                    "label_id": label_id,
                    "actor_plan_json": (
                        str(getattr(args, "actor_plan_json"))
                        if getattr(args, "actor_plan_json", None) is not None
                        else None
                    ),
                    "candidate_frames": actor_candidate_frames,
                    "visible_frames": actor_visible_frames,
                    "culled_frames": actor_culled_frames,
                    "rendered_frames": actor_rendered_frames,
                    "frames": actor_frame_metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    stage_payload = _finalize_stage_seconds(stage_seconds)
    duration = stage_payload["measured_total_sec"]
    return {
        "scene_id": args.scene,
        "label_id": label_id,
        "frames": frames_rendered,
        "duration_sec": duration,
        "stage_seconds": stage_payload,
        "actor_culled_frames": actor_culled_frames,
        "actor_candidate_frames": actor_candidate_frames,
        "actor_visible_frames": actor_visible_frames,
        "actor_rendered_frames": actor_rendered_frames,
        "actor_metadata_path": str(actor_metadata_path) if actor_metadata_path is not None else None,
        "actor_gpu_resident_requested": actor_gpu_requested,
        "actor_gpu_resident": gpu_actor_sequence is not None,
        "actor_gpu_cache_bytes": (
            gpu_actor_sequence.bytes_allocated if gpu_actor_sequence is not None else 0
        ),
        "actor_gpu_sh_mode": (
            gpu_actor_sequence.sh_mode
            if gpu_actor_sequence is not None
            else str(getattr(args, "actor_gpu_sh_mode", "copy"))
        ),
        "actor_gpu_error": actor_gpu_error,
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
    parser.add_argument(
        "--minimal-frames",
        type=int,
        default=None,
        help="If >0, truncate the render to the first N frames (0/omit for full length).",
    )
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
    parser.add_argument(
        "--video-backend",
        choices=[backend.value for backend in VideoWriterBackend],
        default=VideoWriterBackend.NVENC.value,
        help="Video encoder backend: cpu (libx264), nvenc (ffmpeg h264_nvenc), gpu (PyNvVideoCodec).",
    )
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
    parser.add_argument(
        "--actor-plan-json",
        type=Path,
        default=None,
        help=(
            "Optional manifest actor motion plan. When provided, the actor is placed "
            "from the plan and the camera remains on the label path."
        ),
    )
    parser.add_argument("--actor-pattern", default=DEFAULT_ACTOR_PATTERN)
    parser.add_argument("--actor-height", type=float, default=1.7)
    parser.add_argument("--actor-speed", type=float, default=DEFAULT_ACTOR_SPEED)
    parser.add_argument("--actor-fps", type=float, default=float(DEFAULT_VIDEO_FPS))
    parser.add_argument("--follow-distance", type=float, default=1.5)
    parser.add_argument("--follow-buffer", type=float, default=0.0)
    parser.add_argument("--actor-foot-offset", type=float, default=0.0)
    parser.add_argument("--animation-cycle-mod", type=int, default=3)
    parser.add_argument(
        "--actor-visibility-culling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip composing the Gaussian actor on frames where its approximate "
            "bounding sphere is outside the camera frustum. Default: off."
        ),
    )
    parser.add_argument(
        "--actor-cull-margin-m",
        type=float,
        default=0.25,
        help="Extra bounding-sphere margin for --actor-visibility-culling.",
    )
    parser.add_argument(
        "--actor-gpu-resident",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache canonical actor frames on CUDA and transform actor xyz/rotation in torch. "
            "Opt-in because SH-rest coefficients are copied instead of CPU-rotated."
        ),
    )
    parser.add_argument(
        "--actor-gpu-cache-mb",
        type=float,
        default=2048.0,
        help="Maximum CUDA memory to use for the actor frame cache; <=0 disables the cap.",
    )
    parser.add_argument(
        "--actor-gpu-sh-mode",
        choices=["copy"],
        default="copy",
        help="SH-rest handling for --actor-gpu-resident. copy is fast but not exact SH rotation.",
    )
    parser.add_argument("--actor-no-loop", dest="actor_loop", action="store_false")
    parser.set_defaults(actor_loop=True)
    parser.add_argument(
        "--save-actor-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-frame actor candidate/visible/culled/rendered metadata.",
    )
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
    if bool(args.resume) and not bool(args.video):
        LOGGER.warning(
            "resume=true but video=false; resume checks only consider %s/<scene>/<label>.mp4 and will not scan subfolders.",
            args.output_dir,
        )
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

    gaussian_model = _resolve_gaussian_model(scene_dir, args.gaussian_model)
    if not gaussian_model.exists():
        candidates = sorted(scene_dir.glob("*.ply"))
        if not candidates:
            raise FileNotFoundError(f"No Gaussian PLY found under {scene_dir}")
        gaussian_model = candidates[0]
        LOGGER.warning("gaussian_model not found; using %s", gaussian_model.name)

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
    total_mux = 0.0
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
    precheck_t0 = time.monotonic()
    mp4_sizes: dict[str, int] = {}
    mp4_total_bytes = 0
    # Scanning the entire output scene directory is great for large resume runs, but can be
    # slower than per-label checks for "quick test" jobs (e.g. 1 label per scene).
    resume_scan_threshold = 32

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
        # Don't count "already done" skips (resume) or fatal errors towards throughput; they can
        # be fast metadata checks / failures and skew speed/ETA.
        completed_for_speed = paths_ok
        speed_paths = (completed_for_speed / elapsed) if completed_for_speed > 0 else None
        remaining = max(0, paths_planned - paths_done)
        eta_sec = (remaining / speed_paths) if speed_paths and speed_paths > 0 else None

        # Space reporting: avoid `du` / directory walks on large mounts. We maintain total mp4 bytes
        # via a one-time scene scan (resume) and per-path mp4 stats (rendered).
        space_interval = float(getattr(args, "path_progress_space_interval_sec", 5.0) or 0.0)
        if space_interval < 0:
            space_interval = 0.0
        if space_interval == 0.0:
            last_free_bytes = None
        elif last_space_ts <= 0.0 or (now - last_space_ts) >= space_interval:
            try:
                last_free_bytes = int(shutil.disk_usage(str(args.output_dir)).free)
            except Exception:
                last_free_bytes = None
            last_space_ts = now

        est_total = None
        # Prefer averaging by "have mp4 bytes" count, since fatal paths don't produce outputs.
        have_bytes_count = max(0, len(mp4_sizes))
        if last_scene_bytes is not None and have_bytes_count > 0:
            est_total = int((float(last_scene_bytes) / float(have_bytes_count)) * float(paths_planned))

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

    # Pre-check which paths we will actually run BEFORE loading the scene into GPU memory.
    # This makes resume runs much faster when most outputs already exist.
    pending_label_paths: list[Path] = []
    pre_skipped_completed_log = 0
    pre_skipped_outputs_exist = 0
    if args.resume:
        if paths_planned > resume_scan_threshold:
            mp4_sizes, mp4_total_bytes = _scan_existing_scene_mp4s(args.output_dir, scene_id)
            last_scene_bytes = mp4_total_bytes
        else:
            mp4_sizes = {}
            mp4_total_bytes = 0
            last_scene_bytes = 0
    else:
        mp4_sizes = {}
        mp4_total_bytes = 0
        last_scene_bytes = 0
    existing_mp4_labels = set(mp4_sizes.keys())
    for path_file in label_paths:
        label_id = path_file.stem
        if label_id in completed_labels:
            record_path_status(label_id, STATUS_DONE, error="skipped_completed_log")
            paths_skipped += 1
            paths_done += 1
            pre_skipped_completed_log += 1
            continue
        if args.resume and (
            label_id in existing_mp4_labels
            or _label_already_rendered(args.output_dir, scene_id, label_id)
        ):
            record_path_status(label_id, STATUS_DONE, error="skipped_outputs_exist")
            paths_skipped += 1
            paths_done += 1
            pre_skipped_outputs_exist += 1
            continue
        pending_label_paths.append(path_file)

    if args.path_progress:
        precheck_sec = max(0.0, time.monotonic() - precheck_t0)
        LOGGER.info(
            "[PRECHECK] scene=%s planned=%d pending=%d skipped_completed_log=%d skipped_outputs_exist=%d precheck_sec=%.2f",
            scene_id,
            paths_planned,
            len(pending_label_paths),
            pre_skipped_completed_log,
            pre_skipped_outputs_exist,
            precheck_sec,
        )

    if not pending_label_paths:
        # Nothing to do for this scene; write metrics/status for resume accounting and exit early.
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
            "time_per_frame_sec": None,
            "h264_encode_total_sec": total_encode,
            "h264_encode_sec_per_frame": None,
            "h264_encode_fps": None,
            "h264_mux_total_sec": total_mux,
            "h264_mux_sec_per_frame": None,
            "h264_mux_sec_per_path": None,
            "vram_peak_max_bytes": 0.0,
            "vram_avg_max_worker_bytes": 0.0,
            "paths": paths_payload,
            "path_statuses": list(path_statuses.values()),
        }
        if args.metrics_json is not None:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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

    # From here on we have pending work; now load metadata + scene assets.
    meta = load_occupancy_metadata(scene_dir)
    asset = build_scene_asset(scene_dir, gaussian_model, meta)
    renderer = build_renderer(asset, args)

    actor_motion_plans: tuple[ActorMotionPlan, ...] = ()
    actor_render_runtimes: tuple[ActorRenderRuntime, ...] = ()
    if args.actor_plan_json is not None:
        actor_motion_plans = load_actor_motion_plans(args.actor_plan_json)
        args.actor_motion_plans = actor_motion_plans
        args.actor_motion_plan = actor_motion_plans[0] if len(actor_motion_plans) == 1 else None
    else:
        args.actor_motion_plans = ()
        args.actor_motion_plan = None

    actor_runtime: ActorRuntime | None = None
    if actor_motion_plans:
        actor_render_runtimes = tuple(
            _load_actor_render_runtime(plan=plan, args=args, renderer=renderer)
            for plan in actor_motion_plans
        )
    legacy_actor_render_runtime: ActorRenderRuntime | None = None
    if not actor_motion_plans and args.actor_seq_dir is not None:
        legacy_actor_render_runtime = _load_actor_render_runtime(
            plan=ActorMotionPlan(
                actor_id=str(getattr(args, "job_actor_id", "") or "") or None,
                sequence_dir=args.actor_seq_dir,
                frames=({"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},),
                z_mode="floor",
                yaw_offset_rad=0.0,
                actor_height_m=float(args.actor_height),
                actor_fps=float(args.actor_fps),
                loop=bool(args.actor_loop),
            ),
            args=args,
            renderer=renderer,
        )
        actor_runtime = legacy_actor_render_runtime.runtime
    actor_gpu_cache_metric_pending = (
        legacy_actor_render_runtime.gpu_cache_upload_sec if legacy_actor_render_runtime is not None else 0.0
    )
    for path_file in pending_label_paths:
        label_id = path_file.stem
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
            if actor_render_runtimes:
                path_metrics = render_label_with_actor_plans(
                    renderer=renderer,
                    prepared=prepared,
                    output_dir=args.output_dir,
                    label_id=label_id,
                    args=args,
                    actor_render_runtimes=actor_render_runtimes,
                )
            elif actor_runtime is not None and legacy_actor_render_runtime is not None:
                path_metrics = render_label_with_actor(
                    renderer=renderer,
                    prepared=prepared,
                    output_dir=args.output_dir,
                    label_id=label_id,
                    args=args,
                    actor_runtime=actor_runtime,
                    gpu_actor_sequence=legacy_actor_render_runtime.gpu_sequence,
                    actor_gpu_error=legacy_actor_render_runtime.gpu_error,
                    actor_gpu_cache_upload_sec=actor_gpu_cache_metric_pending,
                )
                actor_gpu_cache_metric_pending = 0.0
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
            # Update output size accounting cheaply (mp4 only).
            if bool(args.video):
                mp4_path = args.output_dir / scene_id / f"{label_id}.mp4"
                try:
                    new_size = int(mp4_path.stat().st_size)
                except OSError:
                    new_size = 0
                old_size = int(mp4_sizes.get(label_id, 0))
                mp4_sizes[label_id] = new_size
                mp4_total_bytes += (new_size - old_size)
                last_scene_bytes = mp4_total_bytes
            total_frames += int(path_metrics.get("frames", 0))
            total_duration += float(path_metrics.get("duration_sec", 0.0))
            stage = path_metrics.get("stage_seconds") or {}
            total_encode += float(stage.get("encode", 0.0))
            total_mux += float(stage.get("h264_mux_sec", 0.0))
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
        "scene_mp4_total_bytes": mp4_total_bytes,
        "scene_mp4_count": len(mp4_sizes),
        "paths_total": len(paths_payload),
        "frames_total": total_frames,
        "duration_total_sec": total_duration,
        "time_per_frame_sec": (total_duration / total_frames) if total_frames > 0 else None,
        "h264_encode_total_sec": total_encode,
        "h264_encode_sec_per_frame": (total_encode / total_frames) if total_frames > 0 else None,
        "h264_encode_fps": (total_frames / total_encode) if total_encode > 0 else None,
        "h264_mux_total_sec": total_mux,
        "h264_mux_sec_per_frame": (total_mux / total_frames) if total_frames > 0 else None,
        "h264_mux_sec_per_path": (total_mux / len(paths_payload)) if paths_payload else None,
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
