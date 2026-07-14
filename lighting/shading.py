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
    normal_smooth: int
    shadow_enabled: bool
    shadow_bias: float
    shadow_strength: float
    shadow_pcf_radius: int
    light_mode: str = "headlight"
    shading_model: str = "classic"
    shadow_compare: str = "z"
    normal_filter: str = "box"
    normal_kernel: int = 2
    normal_sigma_range: float = 0.1
    normal_sigma_domain: float = 1.0
    base_scale: float = 1.0
    light_reverse: bool = False

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


def _shadow_depth_to_radial(
    depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    points = _compute_camera_points(depth_m, fx, fy, cx, cy)
    return np.linalg.norm(points, axis=-1)


def _box_blur_sum(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    pad = np.pad(img, ((radius, radius), (radius, radius)), mode="edge")
    csum = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    k = 2 * radius + 1
    return csum[k:, k:] - csum[:-k, k:] - csum[k:, :-k] + csum[:-k, :-k]


def _box_blur(img: np.ndarray, radius: int, *, weights: np.ndarray | None = None) -> np.ndarray:
    if radius <= 0:
        return img
    if weights is None:
        norm = float((2 * radius + 1) ** 2)
        return _box_blur_sum(img, radius) / max(norm, 1.0)
    weights = weights.astype(np.float32, copy=False)
    num = _box_blur_sum(img * weights, radius)
    den = _box_blur_sum(weights, radius)
    return np.divide(num, den, out=img.copy(), where=den > 1e-6)


def _bilateral_filter(
    img: np.ndarray,
    radius: int,
    *,
    sigma_range: float,
    sigma_domain: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if radius <= 0:
        return img
    if sigma_range <= 0.0 or sigma_domain <= 0.0:
        return img
    if valid_mask is None:
        valid_mask = img > 0.0

    img = img.astype(np.float32, copy=False)
    h, w = img.shape
    pad = radius
    padded = np.pad(img, pad, mode="edge")
    center = padded[pad:pad + h, pad:pad + w]

    total = np.zeros_like(center)
    weights = np.zeros_like(center)
    sigma_range_sq = sigma_range * sigma_range
    sigma_domain_sq = sigma_domain * sigma_domain
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            spatial_dist = float(dx * dx + dy * dy)
            spatial_w = math.exp(-spatial_dist / (2.0 * sigma_domain_sq))
            range_w = np.exp(-((neighbor - center) ** 2) / (2.0 * sigma_range_sq))
            neighbor_valid = (neighbor > 0.0).astype(np.float32)
            wgt = spatial_w * range_w * neighbor_valid
            total += neighbor * wgt
            weights += wgt
    out = np.divide(total, weights, out=center.copy(), where=weights > 1e-6)
    out = np.where(valid_mask, out, 0.0)
    return out


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


def _compute_shadow_factor(
    points_cam: np.ndarray,
    shadow_depth_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    offset_cam: Tuple[float, float, float],
    bias: float,
    strength: float,
    pcf_radius: int,
    compare_mode: str = "z",
) -> np.ndarray:
    h, w = shadow_depth_m.shape
    offset = np.array(offset_cam, dtype=np.float32)
    points_light = points_cam - offset[None, None, :]
    z = points_light[..., 2]
    valid = z > 0.0
    z_safe = np.where(valid, z, 1.0)
    u = fx * (points_light[..., 0] / z_safe) + cx
    v = cy - fy * (points_light[..., 1] / z_safe)
    u_i = np.rint(u).astype(np.int32)
    v_i = np.rint(v).astype(np.int32)
    in_bounds = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
    mask = valid & in_bounds
    if not np.any(mask):
        return np.ones_like(z, dtype=np.float32)

    if compare_mode == "radial":
        depth = np.linalg.norm(points_light, axis=-1)
        shadow_depth_ref = _shadow_depth_to_radial(shadow_depth_m, fx, fy, cx, cy)
    else:
        depth = z
        shadow_depth_ref = shadow_depth_m

    if pcf_radius <= 0:
        shadow = np.zeros_like(z, dtype=bool)
        depth_ref = shadow_depth_ref[v_i[mask], u_i[mask]]
        shadow[mask] = (depth_ref > 0.0) & (depth[mask] > depth_ref + bias)
        return np.where(shadow, strength, 1.0).astype(np.float32)

    shadow_hits = np.zeros_like(z, dtype=np.float32)
    sample_counts = np.zeros_like(z, dtype=np.float32)
    for dy in range(-pcf_radius, pcf_radius + 1):
        for dx in range(-pcf_radius, pcf_radius + 1):
            u_s = u_i + dx
            v_s = v_i + dy
            in_bounds = (u_s >= 0) & (u_s < w) & (v_s >= 0) & (v_s < h)
            mask_s = mask & in_bounds
            if not np.any(mask_s):
                continue
            depth_ref = shadow_depth_ref[v_s[mask_s], u_s[mask_s]]
            hit = (depth_ref > 0.0) & (depth[mask_s] > depth_ref + bias)
            shadow_hits[mask_s] += hit.astype(np.float32)
            sample_counts[mask_s] += 1.0
    ratio = np.divide(shadow_hits, sample_counts, out=np.zeros_like(shadow_hits), where=sample_counts > 0.0)
    return 1.0 - ratio * (1.0 - strength)


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
    valid = np.isfinite(depth_m) & (depth_m > 0.0)

    normal_filter = str(config.normal_filter).lower()
    normal_smooth = max(0, int(config.normal_smooth))
    if normal_filter == "box" and normal_smooth > 0:
        weights = valid.astype(np.float32)
        depth_m = _box_blur(depth_m, normal_smooth, weights=weights)
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
    elif normal_filter == "bilateral":
        kernel = max(0, int(config.normal_kernel))
        depth_m = _bilateral_filter(
            depth_m,
            kernel,
            sigma_range=float(config.normal_sigma_range),
            sigma_domain=float(config.normal_sigma_domain),
            valid_mask=valid,
        )
        valid = np.isfinite(depth_m) & (depth_m > 0.0)

    if render.ndim == 3 and render.shape[0] in (3, 4):
        base = np.transpose(render[:3, :, :], (1, 2, 0)).astype(np.float32, copy=False)
        layout = "chw"
    else:
        base = render.astype(np.float32, copy=False)
        layout = "hwc"

    base_scale = max(0.0, float(config.base_scale))
    if base_scale != 1.0:
        base = base * base_scale

    points_cam = _compute_camera_points(depth_m, camera_fx, camera_fy, camera_cx, camera_cy)
    normals = _compute_normals(points_cam)

    offset = np.array(config.offset_cam, dtype=np.float32)
    light_pos = offset[None, None, :]
    l_vec = light_pos - points_cam
    l_dir = _normalize_vectors(l_vec)
    if bool(config.light_reverse):
        l_dir = -l_dir
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
        if shadow_depth_m.ndim != 2:
            shadow_depth_m = np.squeeze(shadow_depth_m)
        if shadow_depth_m.ndim != 2:
            shadow_depth_m = None
        if shadow_depth_m is not None:
            shadow_factor = _compute_shadow_factor(
                points_cam,
                shadow_depth_m,
                fx=shadow_fx,
                fy=shadow_fy,
                cx=shadow_cx,
                cy=shadow_cy,
                offset_cam=config.offset_cam,
                bias=float(config.shadow_bias),
                strength=float(config.shadow_strength),
                pcf_radius=max(0, int(config.shadow_pcf_radius)),
                compare_mode=str(config.shadow_compare).lower(),
            )
            shadow_factor = np.where(valid, shadow_factor, 1.0)

    light_rgb = np.array(config.color, dtype=np.float32)
    ambient = max(float(config.ambient), 0.0)
    fallback = base * ambient
    diffuse_term = config.diffuse * diff * att * shadow_factor
    diffuse_term = np.where(valid, diffuse_term, 0.0)
    if str(config.shading_model).lower() == "lambert":
        lit = base * ambient
        lit += base * (config.strength * diffuse_term)[..., None] * light_rgb[None, None, :]
    else:
        spec_term = config.specular * spec * att * shadow_factor
        spec_term = np.where(valid, spec_term, 0.0)
        lit = base * ambient
        lit += base * (config.strength * diffuse_term)[..., None] * light_rgb[None, None, :]
        lit += (config.strength * spec_term)[..., None] * light_rgb[None, None, :]
    lit = np.where(valid[..., None], lit, fallback)
    lit = np.clip(lit, 0.0, 1.0)

    if layout == "chw":
        return np.transpose(lit, (2, 0, 1))
    return lit
