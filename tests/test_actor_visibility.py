from __future__ import annotations

import math

import numpy as np

from utils.actor_visibility import (
    sphere_visible_in_camera,
    sphere_visible_in_camera_frame,
    transformed_actor_sphere,
)


def _world_view_transform() -> np.ndarray:
    # Standard NavDP metadata stores the view matrix transposed. This camera is
    # at the origin and looks along +Z in camera space.
    return np.eye(4, dtype=np.float64).T


def test_sphere_visible_in_center_of_camera_frustum() -> None:
    result = sphere_visible_in_camera(
        center_world=[0.0, 0.0, 5.0],
        radius_m=0.5,
        world_view_transform=_world_view_transform(),
        fov_x_rad=math.radians(90.0),
        fov_y_rad=math.radians(60.0),
        znear=0.1,
        zfar=10.0,
        matrix_is_transposed=True,
    )

    assert result.visible
    assert result.reason == "visible"
    np.testing.assert_allclose(result.center_camera, [0.0, 0.0, 5.0])


def test_sphere_outside_camera_frustum_side_is_culled() -> None:
    result = sphere_visible_in_camera(
        center_world=[8.0, 0.0, 5.0],
        radius_m=0.25,
        world_view_transform=_world_view_transform(),
        fov_x_rad=math.radians(60.0),
        fov_y_rad=math.radians(60.0),
        znear=0.1,
        zfar=10.0,
        matrix_is_transposed=True,
    )

    assert not result.visible
    assert result.reason == "right_of_frustum"


def test_sphere_behind_camera_is_culled() -> None:
    result = sphere_visible_in_camera(
        center_world=[0.0, 0.0, -2.0],
        radius_m=0.25,
        world_view_transform=_world_view_transform(),
        fov_x_rad=math.radians(90.0),
        fov_y_rad=math.radians(90.0),
        znear=0.1,
        zfar=10.0,
        matrix_is_transposed=True,
    )

    assert not result.visible
    assert result.reason == "before_near"


def test_sphere_visible_in_serialized_camera_frame() -> None:
    frame = {
        "world_to_camera": _world_view_transform().tolist(),
        "fov": {"x_rad": math.radians(90.0), "y_rad": math.radians(90.0)},
        "znear": 0.1,
        "zfar": 20.0,
    }

    assert sphere_visible_in_camera_frame(
        frame,
        center_world=[1.0, 1.0, 4.0],
        radius_m=0.25,
    ).visible


def test_transformed_actor_sphere_uses_grounded_cylinder_extents() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [2.0, 3.0, 0.5]

    sphere = transformed_actor_sphere(
        transform,
        radius_xy_m=0.4,
        height_m=1.8,
        margin_m=0.1,
    )

    np.testing.assert_allclose(sphere.center_world, [2.0, 3.0, 1.4])
    assert math.isclose(sphere.radius_m, math.hypot(0.4, 0.9) + 0.1)
