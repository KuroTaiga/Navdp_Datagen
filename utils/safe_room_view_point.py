"""Select safer XY camera points for room-center view rendering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SafeRoomViewPoint:
    original_xy: tuple[float, float]
    selected_xy: tuple[float, float]
    selected_pixel: tuple[int, int]
    status: str
    manual_verification_required: bool
    search_radius_m: Optional[float]
    reasons: list[str]
    collided_label_ids: list[str]
    collided_structure_ids: list[str]


@dataclass(frozen=True)
class _OccupancyMeta:
    scale: float
    left: float
    top: float
    width: int
    height: int


@dataclass(frozen=True)
class _LabelFootprint:
    item_id: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float | None
    max_z: float | None


@dataclass(frozen=True)
class _StructureShape:
    item_id: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _ValidationResult:
    valid: bool
    reasons: list[str]
    collided_label_ids: list[str]
    collided_structure_ids: list[str]


def _load_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_occupancy(scene_dir: Path) -> tuple[np.ndarray, _OccupancyMeta]:
    occ_path = scene_dir / "occupancy.png"
    meta_path = scene_dir / "occupancy.json"
    if not occ_path.is_file():
        raise FileNotFoundError(f"Missing occupancy map: {occ_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing occupancy metadata: {meta_path}")

    image = np.asarray(Image.open(occ_path).convert("L"), dtype=np.float32)
    h, w = image.shape[:2]

    occ = _load_json(meta_path)
    if not isinstance(occ, dict):
        raise ValueError(f"Invalid occupancy metadata: {meta_path}")
    scale = float(occ.get("scale", 1.0))
    min_values = occ.get("min", occ.get("lower", [0.0, 0.0, 0.0]))
    max_values = occ.get("max", occ.get("upper", [0.0, 0.0, 0.0]))
    if not isinstance(min_values, list) or len(min_values) < 2:
        min_values = [0.0, 0.0]
    if not isinstance(max_values, list) or len(max_values) < 2:
        max_values = [0.0, 0.0]
    meta = _OccupancyMeta(
        scale=scale,
        left=float(min_values[0]),
        top=float(max_values[1]),
        width=int(w),
        height=int(h),
    )
    return image, meta


def world_to_pixel_left_handed(xy: Sequence[float], meta: _OccupancyMeta) -> tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    u = int(round((x - meta.left) / meta.scale))
    v = int(round((meta.top - y) / meta.scale))
    return u, v


def pixel_to_world_left_handed(pixel: Sequence[int], meta: _OccupancyMeta) -> tuple[float, float]:
    u, v = int(pixel[0]), int(pixel[1])
    return float(meta.left + u * meta.scale), float(meta.top - v * meta.scale)


def _point_in_polygon(xy: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    if len(polygon) < 3:
        return True
    x, y = xy
    inside = False
    prev = polygon[-1]
    x0, y0 = float(prev[0]), float(prev[1])
    for point in polygon:
        x1, y1 = float(point[0]), float(point[1])
        crosses = (y1 > y) != (y0 > y)
        if crosses:
            x_at_y = (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1
            if x < x_at_y:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def _distance_to_segment(
    xy: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    p = np.array(xy, dtype=np.float64)
    a = np.array(start, dtype=np.float64)
    b = np.array(end, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _distance_to_polyline_or_polygon(xy: tuple[float, float], points: Sequence[tuple[float, float]], closed: bool) -> float:
    if len(points) == 1:
        return math.dist(xy, points[0])
    limit = len(points) if closed else len(points) - 1
    distances = [
        _distance_to_segment(xy, points[idx], points[(idx + 1) % len(points)])
        for idx in range(limit)
    ]
    return min(distances) if distances else float("inf")


def _xy_from_value(value: object) -> tuple[float, float] | None:
    if isinstance(value, dict) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _points_from_any(value: object) -> list[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            vals = [float(item) for item in value]
            return [(vals[0], vals[1]), (vals[2], vals[3])]
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        points = []
        for item in value:
            xy = _xy_from_value(item)
            if xy is not None:
                points.append(xy)
        return points
    return []


def _unique_points(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    unique: list[tuple[float, float]] = []
    for x, y in points:
        xy = (float(x), float(y))
        if not any(math.isclose(xy[0], old[0], abs_tol=1e-6) and math.isclose(xy[1], old[1], abs_tol=1e-6) for old in unique):
            unique.append(xy)
    return tuple(unique)


def _load_label_footprints(scene_dir: Path) -> list[_LabelFootprint]:
    labels = _load_json(scene_dir / "labels.json")
    if not isinstance(labels, list):
        return []
    footprints: list[_LabelFootprint] = []
    for idx, item in enumerate(labels):
        if not isinstance(item, dict):
            continue
        points = _points_from_any(item.get("bounding_box"))
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        zs = [
            float(point["z"])
            for point in item.get("bounding_box", [])
            if isinstance(point, dict) and "z" in point
        ]
        item_id = str(item.get("ins_id") or f"{item.get('label', 'label')}_{idx}")
        footprints.append(
            _LabelFootprint(
                item_id=item_id,
                min_x=min(xs),
                max_x=max(xs),
                min_y=min(ys),
                max_y=max(ys),
                min_z=min(zs) if zs else None,
                max_z=max(zs) if zs else None,
            )
        )
    return footprints


def _structure_item_id(parent_key: str, item: object, index: int) -> str:
    if isinstance(item, dict):
        for key in ("id", "ins_id", "label", "type"):
            if item.get(key) is not None:
                return f"{parent_key}:{item[key]}:{index}"
    return f"{parent_key}:{index}"


def _extract_structure_points(item: object) -> tuple[tuple[float, float], ...]:
    if isinstance(item, dict):
        for start_key, end_key in (("start", "end"), ("p1", "p2")):
            start = _xy_from_value(item.get(start_key))
            end = _xy_from_value(item.get(end_key))
            if start is not None and end is not None:
                return (start, end)
        for key in ("points", "profile", "boundary", "location", "bounding_box", "bbox"):
            points = _points_from_any(item.get(key))
            if points:
                return _unique_points(points)
    return _unique_points(_points_from_any(item))


def _include_structure_item(parent_key: str, item: object) -> bool:
    key = parent_key.lower()
    if key in {"walls", "wall", "windows", "window", "doors", "door"}:
        return True
    item_type = ""
    if isinstance(item, dict):
        item_type = str(item.get("type", "")).lower()
    return any(word in item_type for word in ("wall", "window", "door"))


def _load_structure_shapes(scene_dir: Path) -> list[_StructureShape]:
    structure = _load_json(scene_dir / "structure.json")
    if not isinstance(structure, dict):
        return []
    shapes: list[_StructureShape] = []
    for parent_key, value in structure.items():
        items = value if isinstance(value, list) else [value]
        for idx, item in enumerate(items):
            if not _include_structure_item(parent_key, item):
                continue
            points = _extract_structure_points(item)
            if points:
                shapes.append(_StructureShape(_structure_item_id(parent_key, item, idx), points))
    return shapes


def _has_free_clearance(
    occupancy: np.ndarray,
    pixel: tuple[int, int],
    threshold: int,
    clearance_px: int,
) -> bool:
    u, v = pixel
    if clearance_px <= 0:
        return True
    if u - clearance_px < 0 or u + clearance_px >= occupancy.shape[1]:
        return False
    if v - clearance_px < 0 or v + clearance_px >= occupancy.shape[0]:
        return False
    y_grid, x_grid = np.ogrid[-clearance_px : clearance_px + 1, -clearance_px : clearance_px + 1]
    mask = (x_grid * x_grid + y_grid * y_grid) <= clearance_px * clearance_px
    window = occupancy[v - clearance_px : v + clearance_px + 1, u - clearance_px : u + clearance_px + 1]
    return bool(np.all(window[mask] >= float(threshold)))


def _validate_candidate(
    xy: tuple[float, float],
    occupancy: np.ndarray,
    meta: _OccupancyMeta,
    room_polygon: Optional[Sequence[Sequence[float]]],
    label_footprints: Sequence[_LabelFootprint],
    structure_shapes: Sequence[_StructureShape],
    occupancy_threshold: int,
    occupancy_clearance_m: float,
    structure_margin_m: float,
    object_margin_m: float,
    camera_z: float | None,
    object_vertical_clearance_m: float,
) -> _ValidationResult:
    reasons: list[str] = []
    collided_label_ids: list[str] = []
    collided_structure_ids: list[str] = []

    pixel = world_to_pixel_left_handed(xy, meta)
    u, v = pixel
    if not (0 <= u < meta.width and 0 <= v < meta.height):
        reasons.append("outside_occupancy")
    else:
        if float(occupancy[v, u]) < float(occupancy_threshold):
            reasons.append("non_free_occupancy")
        clearance_px = int(math.ceil(max(0.0, occupancy_clearance_m) / meta.scale))
        if not _has_free_clearance(occupancy, pixel, occupancy_threshold, clearance_px):
            reasons.append("insufficient_occupancy_clearance")

    if room_polygon is not None and not _point_in_polygon(xy, room_polygon):
        reasons.append("outside_room_polygon")

    x, y = xy
    low_clear_label_ids: list[str] = []
    for footprint in label_footprints:
        if (
            footprint.min_x - object_margin_m <= x <= footprint.max_x + object_margin_m
            and footprint.min_y - object_margin_m <= y <= footprint.max_y + object_margin_m
        ):
            if (
                camera_z is not None
                and footprint.max_z is not None
                and footprint.max_z + object_vertical_clearance_m < float(camera_z)
            ):
                low_clear_label_ids.append(footprint.item_id)
                continue
            collided_label_ids.append(footprint.item_id)
    if collided_label_ids:
        reasons.append("overlaps_object_bbox")
    elif low_clear_label_ids:
        reasons = [
            reason
            for reason in reasons
            if reason not in {"non_free_occupancy", "insufficient_occupancy_clearance"}
        ]

    for shape in structure_shapes:
        points = shape.points
        if len(points) == 1:
            collides = math.dist(xy, points[0]) <= structure_margin_m
        elif len(points) == 2:
            collides = _distance_to_segment(xy, points[0], points[1]) <= structure_margin_m
        else:
            collides = _point_in_polygon(xy, points) or _distance_to_polyline_or_polygon(xy, points, closed=True) <= structure_margin_m
        if collides:
            collided_structure_ids.append(shape.item_id)
    if collided_structure_ids:
        reasons.append("overlaps_structure")

    return _ValidationResult(
        valid=not reasons,
        reasons=reasons,
        collided_label_ids=collided_label_ids,
        collided_structure_ids=collided_structure_ids,
    )


def _candidate_pixels_by_distance(origin: tuple[int, int], radius_px: int) -> list[tuple[int, int]]:
    ou, ov = origin
    candidates: list[tuple[int, int, int]] = []
    radius_sq = radius_px * radius_px
    for dv in range(-radius_px, radius_px + 1):
        for du in range(-radius_px, radius_px + 1):
            dist_sq = du * du + dv * dv
            if dist_sq == 0 or dist_sq > radius_sq:
                continue
            candidates.append((dist_sq, ou + du, ov + dv))
    candidates.sort(key=lambda item: item[0])
    return [(u, v) for _, u, v in candidates]


def _radius_status_value(radius_m: float) -> str:
    return f"{float(radius_m):.1f}"


def choose_safe_room_view_point(
    scene_dir: Path,
    original_xy: tuple[float, float],
    *,
    room_polygon: Optional[Sequence[Sequence[float]]] = None,
    occupancy_threshold: int = 200,
    occupancy_clearance_m: float = 0.10,
    structure_margin_m: float = 0.08,
    object_margin_m: float = 0.03,
    camera_z: float | None = None,
    object_vertical_clearance_m: float = 0.20,
    search_radii_m: Sequence[float] = (0.5, 1.0),
) -> SafeRoomViewPoint:
    """Return the original point if valid, otherwise the nearest valid free point."""

    occupancy, meta = _load_occupancy(scene_dir)
    labels = _load_label_footprints(scene_dir)
    structures = _load_structure_shapes(scene_dir)
    original_xy = (float(original_xy[0]), float(original_xy[1]))
    original_pixel = world_to_pixel_left_handed(original_xy, meta)
    original_check = _validate_candidate(
        original_xy,
        occupancy,
        meta,
        room_polygon,
        labels,
        structures,
        occupancy_threshold,
        occupancy_clearance_m,
        structure_margin_m,
        object_margin_m,
        camera_z,
        object_vertical_clearance_m,
    )
    if original_check.valid:
        return SafeRoomViewPoint(
            original_xy=original_xy,
            selected_xy=original_xy,
            selected_pixel=original_pixel,
            status="original_valid",
            manual_verification_required=False,
            search_radius_m=None,
            reasons=[],
            collided_label_ids=[],
            collided_structure_ids=[],
        )

    prefixed_reasons = [f"original_{reason}" for reason in original_check.reasons]
    for radius_m in search_radii_m:
        radius_px = int(math.ceil(float(radius_m) / meta.scale))
        for pixel in _candidate_pixels_by_distance(original_pixel, radius_px):
            u, v = pixel
            if not (0 <= u < meta.width and 0 <= v < meta.height):
                continue
            if float(occupancy[v, u]) < float(occupancy_threshold):
                continue
            candidate_xy = pixel_to_world_left_handed(pixel, meta)
            check = _validate_candidate(
                candidate_xy,
                occupancy,
                meta,
                room_polygon,
                labels,
                structures,
                occupancy_threshold,
                occupancy_clearance_m,
                structure_margin_m,
                object_margin_m,
                camera_z,
                object_vertical_clearance_m,
            )
            if check.valid:
                return SafeRoomViewPoint(
                    original_xy=original_xy,
                    selected_xy=candidate_xy,
                    selected_pixel=(int(u), int(v)),
                    status=f"adjusted_within_{_radius_status_value(float(radius_m))}m",
                    manual_verification_required=False,
                    search_radius_m=float(radius_m),
                    reasons=prefixed_reasons,
                    collided_label_ids=original_check.collided_label_ids,
                    collided_structure_ids=original_check.collided_structure_ids,
                )

    return SafeRoomViewPoint(
        original_xy=original_xy,
        selected_xy=original_xy,
        selected_pixel=original_pixel,
        status="manual_verification_required",
        manual_verification_required=True,
        search_radius_m=None,
        reasons=prefixed_reasons,
        collided_label_ids=original_check.collided_label_ids,
        collided_structure_ids=original_check.collided_structure_ids,
    )
