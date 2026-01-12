from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CameraLightConfig:
    enabled: bool
    strength: float
    color: Tuple[float, float, float]
    ambient: float
    diffuse: float
    specular: float
    shininess: float
    range_m: float | None
    offset_cam: Tuple[float, float, float]
    shadow_enabled: bool
    shadow_bias: float
    shadow_strength: float

    def active(self) -> bool:
        return self.enabled


def intrinsics_from_camera(camera) -> tuple[float, float, float, float]:
    width = int(camera.image_width)
    height = int(camera.image_height)
    fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    cx = width * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy


def _inverse_depth_to_meters(depth_inv: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(depth_inv > 0.0, 1.0 / depth_inv, 0.0)
    return depth.astype(np.float32, copy=False)


def _compute_camera_points(
    depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    h, w = depth_m.shape
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)
    z = depth_m
    x = (uu - cx) / fx * z
    y = (cy - vv) / fy * z
    return np.stack((x, y, z), axis=-1)


def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return vectors / norm


def _compute_normals(points_cam: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(points_cam)
    dy = np.zeros_like(points_cam)
    dx[:, 1:-1, :] = points_cam[:, 2:, :] - points_cam[:, :-2, :]
    dx[:, 0, :] = points_cam[:, 1, :] - points_cam[:, 0, :]
    dx[:, -1, :] = points_cam[:, -1, :] - points_cam[:, -2, :]

    dy[1:-1, :, :] = points_cam[2:, :, :] - points_cam[:-2, :, :]
    dy[0, :, :] = points_cam[1, :, :] - points_cam[0, :, :]
    dy[-1, :, :] = points_cam[-1, :, :] - points_cam[-2, :, :]

    normals = np.cross(dx, dy)
    normals = _normalize_vectors(normals)
    normals[normals[..., 2] < 0.0] *= -1.0
    return normals


def _compute_shadow_mask(
    points_cam: np.ndarray,
    shadow_depth_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    offset_cam: Tuple[float, float, float],
    bias: float,
) -> np.ndarray:
    h, w = shadow_depth_m.shape
    offset = np.array(offset_cam, dtype=np.float32)
    points_light = points_cam - offset[None, None, :]
    z = points_light[..., 2]
    valid = z > 0.0
    u = fx * (points_light[..., 0] / z) + cx
    v = cy - fy * (points_light[..., 1] / z)
    u_i = np.rint(u).astype(np.int32)
    v_i = np.rint(v).astype(np.int32)
    in_bounds = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
    mask = valid & in_bounds
    shadow = np.zeros_like(z, dtype=bool)
    if np.any(mask):
        depth_ref = shadow_depth_m[v_i[mask], u_i[mask]]
        shadow[mask] = (depth_ref > 0.0) & (z[mask] > depth_ref + bias)
    return shadow


def apply_camera_light_shading(
    render: np.ndarray,
    depth_inv: np.ndarray,
    *,
    config: CameraLightConfig,
    camera_fx: float,
    camera_fy: float,
    camera_cx: float,
    camera_cy: float,
    shadow_depth_inv: np.ndarray | None = None,
    shadow_fx: float | None = None,
    shadow_fy: float | None = None,
    shadow_cx: float | None = None,
    shadow_cy: float | None = None,
) -> np.ndarray:
    if not config.active():
        return render

    depth_m = _inverse_depth_to_meters(depth_inv)
    if depth_m.ndim != 2:
        depth_m = np.squeeze(depth_m)
    if depth_m.ndim != 2:
        return render
    valid = depth_m > 0.0

    if render.ndim == 3 and render.shape[0] in (3, 4):
        base = np.transpose(render[:3, :, :], (1, 2, 0)).astype(np.float32, copy=False)
        layout = "chw"
    else:
        base = render.astype(np.float32, copy=False)
        layout = "hwc"

    points_cam = _compute_camera_points(depth_m, camera_fx, camera_fy, camera_cx, camera_cy)
    normals = _compute_normals(points_cam)

    offset = np.array(config.offset_cam, dtype=np.float32)
    light_pos = offset[None, None, :]
    l_vec = light_pos - points_cam
    l_dir = _normalize_vectors(l_vec)
    v_dir = _normalize_vectors(-points_cam)

    diff = np.maximum(np.sum(normals * l_dir, axis=-1), 0.0)
    half_vec = _normalize_vectors(l_dir + v_dir)
    spec = np.maximum(np.sum(normals * half_vec, axis=-1), 0.0) ** max(config.shininess, 1.0)

    att = 1.0
    if config.range_m is not None and config.range_m > 0.0:
        dist = np.linalg.norm(l_vec, axis=-1)
        att = 1.0 / (1.0 + (dist / config.range_m) ** 2)

    shadow_factor = 1.0
    if (
        config.shadow_enabled
        and shadow_depth_inv is not None
        and shadow_fx is not None
        and shadow_fy is not None
        and shadow_cx is not None
        and shadow_cy is not None
    ):
        shadow_depth_m = _inverse_depth_to_meters(shadow_depth_inv)
        shadow = _compute_shadow_mask(
            points_cam,
            shadow_depth_m,
            fx=shadow_fx,
            fy=shadow_fy,
            cx=shadow_cx,
            cy=shadow_cy,
            offset_cam=config.offset_cam,
            bias=float(config.shadow_bias),
        )
        shadow_factor = np.where(shadow, float(config.shadow_strength), 1.0)

    light_rgb = np.array(config.color, dtype=np.float32)
    diffuse_term = config.diffuse * diff * att * shadow_factor
    spec_term = config.specular * spec * att * shadow_factor
    spec_term = np.where(valid, spec_term, 0.0)
    shade = config.ambient + config.strength * diffuse_term
    shade = np.where(valid, shade, 1.0)
    lit = base * shade[..., None]
    lit += config.strength * spec_term[..., None] * light_rgb[None, None, :]
    lit = np.clip(lit, 0.0, 1.0)

    if layout == "chw":
        return np.transpose(lit, (2, 0, 1))
    return lit
