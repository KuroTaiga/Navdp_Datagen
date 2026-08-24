from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class BoundingSphere:
    center_world: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class VisibilityResult:
    visible: bool
    reason: str
    center_camera: np.ndarray


def transformed_actor_sphere(
    transform: np.ndarray,
    *,
    radius_xy_m: float,
    height_m: float,
    margin_m: float = 0.0,
) -> BoundingSphere:
    """Approximate a grounded actor cylinder as a world-space bounding sphere."""

    mat = np.asarray(transform, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError("actor transform must be 4x4")
    radius_xy = max(float(radius_xy_m), 0.0)
    height = max(float(height_m), 0.0)
    local_center = np.array([0.0, 0.0, height * 0.5, 1.0], dtype=np.float64)
    center = (mat @ local_center)[:3]
    local_radius = math.hypot(radius_xy, height * 0.5)
    linear = mat[:3, :3]
    if linear.size:
        scale = float(np.linalg.svd(linear, compute_uv=False)[0])
    else:
        scale = 1.0
    return BoundingSphere(
        center_world=center.astype(np.float64, copy=False),
        radius_m=max(local_radius * scale + float(margin_m), 0.0),
    )


def sphere_visible_in_camera(
    *,
    center_world: Iterable[float],
    radius_m: float,
    world_view_transform: np.ndarray,
    fov_x_rad: float,
    fov_y_rad: float,
    znear: float,
    zfar: float,
    matrix_is_transposed: bool = True,
) -> VisibilityResult:
    """Conservative perspective-frustum test for a world-space sphere.

    NavDP/GraphDeco camera metadata stores ``world_view_transform`` transposed.
    The default ``matrix_is_transposed=True`` matches that convention.
    """

    view = np.asarray(world_view_transform, dtype=np.float64)
    if view.shape != (4, 4):
        raise ValueError("world_view_transform must be 4x4")
    if matrix_is_transposed:
        view = view.T

    center = np.asarray(list(center_world), dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("center_world must contain exactly 3 values")
    radius = max(float(radius_m), 0.0)
    center_h = np.array([center[0], center[1], center[2], 1.0], dtype=np.float64)
    camera = (view @ center_h)[:3]
    x, y, z = map(float, camera)

    if z + radius < float(znear):
        return VisibilityResult(False, "before_near", camera)
    if z - radius > float(zfar):
        return VisibilityResult(False, "beyond_far", camera)
    if z + radius <= EPS:
        return VisibilityResult(False, "behind_camera", camera)

    half_w = max(z, 0.0) * math.tan(float(fov_x_rad) * 0.5) + radius
    half_h = max(z, 0.0) * math.tan(float(fov_y_rad) * 0.5) + radius
    if x < -half_w:
        return VisibilityResult(False, "left_of_frustum", camera)
    if x > half_w:
        return VisibilityResult(False, "right_of_frustum", camera)
    if y < -half_h:
        return VisibilityResult(False, "below_frustum", camera)
    if y > half_h:
        return VisibilityResult(False, "above_frustum", camera)
    return VisibilityResult(True, "visible", camera)


def sphere_visible_in_camera_frame(
    frame: dict[str, Any],
    *,
    center_world: Iterable[float],
    radius_m: float,
) -> VisibilityResult:
    """Visibility test using one serialized NavDP camera metadata frame."""

    fov = frame.get("fov") or {}
    return sphere_visible_in_camera(
        center_world=center_world,
        radius_m=radius_m,
        world_view_transform=np.asarray(frame["world_to_camera"], dtype=np.float64),
        fov_x_rad=float(fov["x_rad"]),
        fov_y_rad=float(fov["y_rad"]),
        znear=float(frame.get("znear", 0.0)),
        zfar=float(frame.get("zfar", float("inf"))),
        matrix_is_transposed=True,
    )
