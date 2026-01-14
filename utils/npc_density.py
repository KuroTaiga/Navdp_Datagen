from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

import imageio.v2 as imageio
import numpy as np

try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - optional dependency
    _ndimage = None

from utils.render_utils import load_occupancy_metadata, world_to_pixel

EPS = 1e-6


class NPCPlacementBackend(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


_GPU_BACKEND_FAILED = False
_GPU_FALLBACK_REPORTED = False


@dataclass(frozen=True)
class CameraWedge:
    """2D ground-plane slice of the camera FOV."""

    origin_xy: np.ndarray
    forward_xy: np.ndarray
    fov_deg: float
    max_range: float
    min_range: float = 0.0


@dataclass(frozen=True)
class NPCDensityConfig:
    """Knobs for NPC placement inside a camera wedge."""

    clearance_radius: float = 0.30  # meters
    min_center_distance: float = 1.0  # meters (camera -> NPC center, before adding radius)
    max_distance_from_camera: float | None = None  # meters (None => use wedge.max_range)
    target_coverage: float | None = None  # fraction of wedge area to occupy (0..1)
    max_npcs: int | None = None  # hard cap regardless of coverage target
    allow_blocking: bool = False  # if False, avoid blocking camera->goal line
    max_resamples: int = 50  # per requested NPC
    free_pixel_min: int = 250  # occupancy pixels considered free (>= threshold if free_is_white else <= threshold)
    free_is_white: bool = True  # True: free if >= threshold; False: free if <= threshold
    coverage_mode: Literal["area", "angular"] = "angular"  # "area" = wedge area, "angular" = FOV angular coverage
    desired_count: int | None = None  # guiding count
    priority: Literal["coverage", "count"] = "coverage"  # which requirement is treated as hard
    zone_weights: tuple[float, float, float] = (1.0, 2.0, 1.0)  # near:mid:far count ratios (soft)


@dataclass(frozen=True)
class NPCPlacementResult:
    positions_xy: list[np.ndarray]
    requested_count: int
    achieved_coverage: float
    target_coverage: float | None
    attempts: int
    rejected_blocking: int
    rejected_clearance: int
    rejected_oob: int
    shortfall: int
    accepted_indices: list[int]


def rgb_to_luma_u8(mask: np.ndarray) -> np.ndarray:
    """Convert RGB mask to uint8 luma (Rec. 709); pass through non-RGB arrays."""
    if mask.ndim == 3:
        return np.round(
            0.2126 * mask[..., 0] + 0.7152 * mask[..., 1] + 0.0722 * mask[..., 2]
        ).astype(np.uint8)
    return mask


def load_free_space_mask(dataset_dir: Path, *, threshold: int = 128, free_is_white: bool = True) -> tuple[np.ndarray, dict]:
    """
    Load occupancy.png and return a boolean free-space mask plus metadata.
    If free_is_white: free if grayscale >= threshold.
    If not free_is_white: free if grayscale <= threshold.
    """
    meta = load_occupancy_metadata(dataset_dir)
    occ_png = dataset_dir / "occupancy.png"
    if not occ_png.is_file():
        raise FileNotFoundError(f"Missing occupancy.png in {dataset_dir}")

    mask = rgb_to_luma_u8(imageio.imread(occ_png))
    free = mask >= threshold if free_is_white else mask <= threshold
    return free.astype(bool), meta


def load_wall_mask(dataset_dir: Path, *, threshold: int = 128, white_is_inside: bool = True) -> np.ndarray:
    """Load wall_mask.png and return a boolean mask of allowed placement."""
    wall_path = dataset_dir / "wall_mask.png"
    if not wall_path.is_file():
        raise FileNotFoundError(f"Missing wall_mask.png in {dataset_dir}")

    mask = rgb_to_luma_u8(imageio.imread(wall_path))
    allowed = mask >= threshold if white_is_inside else mask <= threshold
    return allowed.astype(bool)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        raise ValueError("Cannot normalise zero-length vector.")
    return (vec / norm).astype(np.float32)


def _rotate(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y = float(vec[0]), float(vec[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def _wedge_area(fov_rad: float, r_min: float, r_max: float) -> float:
    if r_max <= r_min or fov_rad <= 0.0:
        return 0.0
    return 0.5 * fov_rad * (r_max * r_max - r_min * r_min)


def _disc_area(radius: float) -> float:
    if radius <= 0.0:
        return 0.0
    return math.pi * radius * radius

def _disc_angular_width(radius: float, distance: float) -> float:
    """
    Angular span (radians) of a disc of radius r at range d, seen from origin.
    Formula: theta = 2 * asin(r / d). Small-angle approx: ~2r/d.
    """
    d = max(distance, radius + EPS)
    ratio = min(1.0, max(0.0, radius / d))
    return 2.0 * math.asin(ratio)


def _angle_between(forward: np.ndarray, vec: np.ndarray) -> float:
    """Signed angle between forward and vec in radians."""
    f = _normalize(forward)
    v = _normalize(vec)
    dot = float(np.clip(np.dot(f, v), -1.0, 1.0))
    det = float(f[0] * v[1] - f[1] * v[0])
    return math.atan2(det, dot)


def _effective_disc_span(radius: float, distance: float, fov_half: float, center_angle: float) -> float:
    """
    Limit disc angular span by the remaining FOV if the center is near the edge.
    """
    raw = _disc_angular_width(radius, distance)
    available_half = max(0.0, fov_half - abs(center_angle))
    return min(raw, 2.0 * available_half)


def _blocks_goal(camera_xy: np.ndarray, goal_xy: np.ndarray, candidate_xy: np.ndarray, radius: float) -> bool:
    """Return True if the candidate disc overlaps the camera->goal segment."""
    seg = goal_xy - camera_xy
    seg_len = float(np.linalg.norm(seg))
    if seg_len < EPS:
        return False
    t = float(np.dot(candidate_xy - camera_xy, seg)) / (seg_len * seg_len)
    if t < 0.0 or t > 1.0:
        return False
    closest = camera_xy + t * seg
    return float(np.linalg.norm(candidate_xy - closest)) <= radius


def _inside_mask(meta: dict, free_mask: np.ndarray, xy: np.ndarray, radius_m: float, free_pixel_min: int) -> bool:
    """Check if xy is in-bounds and clears occupancy (using a circular footprint)."""
    h, w = free_mask.shape[:2]
    u, v = world_to_pixel(meta, xy)
    radius_px = int(math.ceil(radius_m / float(meta["scale"])))
    if u < radius_px or v < radius_px or u >= w - radius_px or v >= h - radius_px:
        return False
    # Build a circular mask to test clearance
    uu = np.arange(u - radius_px, u + radius_px + 1)
    vv = np.arange(v - radius_px, v + radius_px + 1)
    du = uu[np.newaxis, :] - u
    dv = vv[:, np.newaxis] - v
    circle = (du * du + dv * dv) <= (radius_px * radius_px)
    region = free_mask[v - radius_px : v + radius_px + 1, u - radius_px : u + radius_px + 1]
    if region.dtype != bool:
        region = region >= free_pixel_min
    if region.shape != circle.shape:
        return False
    # Occupancy is free if all covered pixels are above the threshold
    return bool(np.all(region[circle]))

def _inside_center_mask(meta: dict, center_mask: np.ndarray, xy: np.ndarray, free_pixel_min: int) -> bool:
    """Check if xy is in-bounds and allowed by a pre-cleared center mask."""
    h, w = center_mask.shape[:2]
    u, v = world_to_pixel(meta, xy)
    if u < 0 or v < 0 or u >= w or v >= h:
        return False
    if center_mask.dtype == bool:
        return bool(center_mask[v, u])
    return bool(center_mask[v, u] >= free_pixel_min)


def _estimate_target_count(
    *,
    wedge_area: float,
    disc_area: float,
    target_coverage: float | None,
    max_npcs: int | None,
    coverage_mode: str,
    fov_rad: float,
    r_min: float,
    r_max: float,
    radius: float,
) -> int:
    if coverage_mode == "area":
        if disc_area <= 0.0 or wedge_area <= 0.0:
            return 0
    else:
        if fov_rad <= 0.0:
            return 0
    target = 0
    if target_coverage is not None and target_coverage > 0.0:
        clamped = min(1.0, max(0.0, target_coverage))
        if coverage_mode == "area":
            target = max(1, math.ceil(clamped * wedge_area / disc_area))
        else:
            # Conservative: use farthest distance to estimate smallest angular span.
            r_use = max(r_min, r_max, radius + EPS)
            ang = _disc_angular_width(radius, r_use)
            if ang > 0.0:
                target = max(1, math.ceil(clamped * fov_rad / ang))
    if max_npcs is not None and max_npcs > 0:
        target = max_npcs if target == 0 else min(target, max_npcs)
    return target


def estimate_npc_target_count(
    *,
    wedge: CameraWedge,
    config: NPCDensityConfig,
    radius_m: float | None = None,
) -> tuple[int, int | None]:
    """Estimate the desired NPC count and coverage cap for a wedge."""
    base_min = max(config.min_center_distance, wedge.min_range)
    r_max = (
        wedge.max_range
        if config.max_distance_from_camera is None
        else min(wedge.max_range, config.max_distance_from_camera)
    )
    if r_max <= base_min:
        return 0, 0

    fov_rad = math.radians(wedge.fov_deg)
    wedge_area = _wedge_area(fov_rad, base_min, r_max)
    radius = float(radius_m) if radius_m is not None else float(config.clearance_radius)
    disc_area = _disc_area(radius)

    coverage_cap: int | None = None
    if config.target_coverage is not None and config.target_coverage > 0.0:
        cov = min(1.0, max(0.0, config.target_coverage))
        if config.coverage_mode == "area":
            coverage_cap = (
                int(math.floor(cov * wedge_area / disc_area))
                if wedge_area > 0.0 and disc_area > 0.0
                else 0
            )
        else:
            ang = _disc_angular_width(radius, max(r_max, radius + EPS))
            ang = min(ang, fov_rad)
            coverage_cap = (
                int(math.floor(cov * fov_rad / ang))
                if ang > 0.0 and fov_rad > 0.0
                else 0
            )
        coverage_cap = max(0, coverage_cap)

    desired = config.desired_count if (config.desired_count is not None and config.desired_count > 0) else None
    count_priority = config.priority == "count" and desired is not None
    target_count = 0
    if count_priority:
        target_count = desired
    elif config.priority == "count":
        if coverage_cap is not None:
            target_count = coverage_cap
    else:  # coverage priority
        if coverage_cap is not None:
            target_count = coverage_cap
        elif desired is not None:
            target_count = desired

    if coverage_cap is not None and target_count > coverage_cap and not count_priority:
        target_count = coverage_cap
    if config.max_npcs is not None and config.max_npcs > 0:
        target_count = min(target_count, config.max_npcs)

    return target_count, coverage_cap


def _zone_bounds(r_min: float, r_max: float) -> list[tuple[float, float]]:
    """Split radial band into 3 equal spans (near, mid, far)."""
    if r_max <= r_min:
        return [(r_min, r_min), (r_min, r_min), (r_min, r_min)]
    span = r_max - r_min
    step = span / 3.0
    return [
        (r_min, r_min + step),
        (r_min + step, r_min + 2.0 * step),
        (r_min + 2.0 * step, r_max),
    ]


def _distribute_counts(total: int, weights: Sequence[float]) -> list[int]:
    """Allocate counts across zones based on weights, preserving total."""
    if total <= 0:
        return [0, 0, 0]
    w = np.array(weights, dtype=np.float32)
    w = np.where(w < 0.0, 0.0, w)
    if np.allclose(w.sum(), 0.0):
        w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    w = w / w.sum()
    raw = w * float(total)
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        frac = raw - base
        order = np.argsort(-frac)
        for idx in range(remainder):
            base[order[idx % len(base)]] += 1
    return base.tolist()


def _plan_npc_positions_cpu(
    *,
    wedge: CameraWedge,
    free_mask: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    config: NPCDensityConfig,
    goal_xy: np.ndarray | None = None,
    exclude_discs: Sequence[tuple[np.ndarray, float]] | None = None,
    radii_m: Sequence[float] | None = None,
    center_mask: np.ndarray | None = None,
    center_mask_is_bloomed: bool = False,
) -> NPCPlacementResult:
    """
    Sample NPC positions inside the camera wedge with disc clearance.

    The sampler targets a coverage fraction (if provided) and falls back to max_npcs as a cap.
    It rejects candidates that collide with obstacles/other NPCs, fall outside the mask, or
    block the camera->goal segment when blocking is disabled. Optional exclude_discs are treated
    as occupied (collide but do not contribute to coverage).
    If center_mask is provided, it is treated as a pre-cleared mask of allowed centers.
    When center_mask_is_bloomed is True, the clearance check against free_mask is skipped.
    """
    forward_xy = _normalize(wedge.forward_xy)
    fov_rad = math.radians(wedge.fov_deg)
    base_min = max(config.min_center_distance, wedge.min_range)
    r_max = (
        wedge.max_range
        if config.max_distance_from_camera is None
        else min(wedge.max_range, config.max_distance_from_camera)
    )
    if r_max <= base_min:
        return NPCPlacementResult(
            positions_xy=[],
            requested_count=0,
            achieved_coverage=0.0,
            target_coverage=config.target_coverage,
            attempts=0,
            rejected_blocking=0,
            rejected_clearance=0,
            rejected_oob=0,
            shortfall=0,
            accepted_indices=[],
        )

    base_radius = max(float(config.clearance_radius), EPS)
    target_count, coverage_cap = estimate_npc_target_count(
        wedge=wedge,
        config=config,
        radius_m=base_radius,
    )
    if radii_m is not None:
        radii = [max(float(r), EPS) for r in radii_m]
        if config.max_npcs is not None and config.max_npcs > 0:
            radii = radii[: config.max_npcs]
        target_count = len(radii)
    else:
        radii = [base_radius] * target_count

    if target_count <= 0 or not radii:
        return NPCPlacementResult(
            positions_xy=[],
            requested_count=0,
            achieved_coverage=0.0,
            target_coverage=config.target_coverage,
            attempts=0,
            rejected_blocking=0,
            rejected_clearance=0,
            rejected_oob=0,
            shortfall=0,
            accepted_indices=[],
        )

    zones = _zone_bounds(base_min, r_max)
    effective_weights = config.zone_weights if target_count >= 12 else (1.0, 1.0, 1.0)
    zone_counts = _distribute_counts(target_count, effective_weights)

    placements: list[tuple[np.ndarray, float, int]] = []
    rejected_blocking = rejected_clearance = rejected_oob = 0
    attempts = 0
    max_attempts = max(config.max_resamples * max(target_count, 1), config.max_resamples)
    radius_idx = 0

    for zone_idx, need in enumerate(zone_counts):
        z_min, z_max = zones[zone_idx]
        while need > 0 and attempts < max_attempts and radius_idx < len(radii):
            radius = radii[radius_idx]
            min_center = base_min + radius
            if min_center > r_max:
                rejected_oob += 1
                radius_idx += 1
                need -= 1
                continue
            z_min_eff = max(z_min, min_center)
            z_max_eff = max(z_min_eff, z_max)
            attempts += 1
            angle = rng.uniform(-0.5 * fov_rad, 0.5 * fov_rad)
            r = (
                math.sqrt(rng.uniform(z_min_eff * z_min_eff, z_max_eff * z_max_eff))
                if z_max_eff > z_min_eff
                else z_min_eff
            )
            direction = _rotate(forward_xy, angle)
            candidate = wedge.origin_xy + direction * r

            if center_mask is not None:
                if not _inside_center_mask(meta, center_mask, candidate, config.free_pixel_min):
                    rejected_oob += 1
                    continue
            if not center_mask_is_bloomed:
                # Use the raw free mask for clearance so bloom/center masks don't double-shrink space.
                if not _inside_mask(meta, free_mask, candidate, radius, config.free_pixel_min):
                    rejected_oob += 1
                    continue

            too_close = any(
                float(np.linalg.norm(candidate - placed)) < (radius + placed_radius - EPS)
                for placed, placed_radius, _ in placements
            )
            if not too_close and exclude_discs:
                for center, exclude_radius in exclude_discs:
                    if float(np.linalg.norm(candidate - center)) < (exclude_radius + radius - EPS):
                        too_close = True
                        break
            if too_close:
                rejected_clearance += 1
                continue

            if (
                not config.allow_blocking
                and goal_xy is not None
                and _blocks_goal(wedge.origin_xy, goal_xy, candidate, radius)
            ):
                rejected_blocking += 1
                continue

            placements.append((candidate.astype(np.float32), radius, radius_idx))
            radius_idx += 1
            need -= 1

    achieved_coverage = 0.0
    if placements:
        if config.coverage_mode == "area":
            wedge_area = _wedge_area(fov_rad, base_min, r_max)
            if wedge_area > 0.0:
                covered_area = sum(_disc_area(r) for _, r, _ in placements)
                achieved_coverage = min(1.0, covered_area / wedge_area)
        else:
            total_angle = 0.0
            fov_half = 0.5 * fov_rad
            for pos, radius, _ in placements:
                offset = pos - wedge.origin_xy
                dist = float(np.linalg.norm(offset))
                center_angle = _angle_between(forward_xy, offset)
                span = _effective_disc_span(
                    radius,
                    max(dist, radius + EPS),
                    fov_half,
                    center_angle,
                )
                total_angle += span
            if fov_rad > 0.0:
                achieved_coverage = min(1.0, total_angle / fov_rad)
    shortfall = max(0, target_count - len(placements))

    return NPCPlacementResult(
        positions_xy=[pos for pos, _, _ in placements],
        requested_count=target_count,
        achieved_coverage=achieved_coverage,
        target_coverage=config.target_coverage,
        attempts=attempts,
        rejected_blocking=rejected_blocking,
        rejected_clearance=rejected_clearance,
        rejected_oob=rejected_oob,
        shortfall=shortfall,
        accepted_indices=[idx for _, _, idx in placements],
    )


def _plan_npc_positions_gpu(
    *,
    wedge: CameraWedge,
    free_mask: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    config: NPCDensityConfig,
    goal_xy: np.ndarray | None = None,
    exclude_discs: Sequence[tuple[np.ndarray, float]] | None = None,
    radii_m: Sequence[float] | None = None,
    center_mask: np.ndarray | None = None,
    center_mask_is_bloomed: bool = False,
) -> NPCPlacementResult:
    try:
        import torch
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("GPU NPC placement requires torch.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("GPU NPC placement requires a CUDA device.")

    forward_xy = _normalize(wedge.forward_xy)
    fov_rad = math.radians(wedge.fov_deg)
    base_min = max(config.min_center_distance, wedge.min_range)
    r_max = (
        wedge.max_range
        if config.max_distance_from_camera is None
        else min(wedge.max_range, config.max_distance_from_camera)
    )
    if r_max <= base_min:
        return NPCPlacementResult(
            positions_xy=[],
            requested_count=0,
            achieved_coverage=0.0,
            target_coverage=config.target_coverage,
            attempts=0,
            rejected_blocking=0,
            rejected_clearance=0,
            rejected_oob=0,
            shortfall=0,
            accepted_indices=[],
        )

    base_radius = max(float(config.clearance_radius), EPS)
    target_count, _coverage_cap = estimate_npc_target_count(
        wedge=wedge,
        config=config,
        radius_m=base_radius,
    )
    if radii_m is not None:
        radii = [max(float(r), EPS) for r in radii_m]
        if config.max_npcs is not None and config.max_npcs > 0:
            radii = radii[: config.max_npcs]
        target_count = len(radii)
    else:
        radii = [base_radius] * target_count

    if target_count <= 0 or not radii:
        return NPCPlacementResult(
            positions_xy=[],
            requested_count=0,
            achieved_coverage=0.0,
            target_coverage=config.target_coverage,
            attempts=0,
            rejected_blocking=0,
            rejected_clearance=0,
            rejected_oob=0,
            shortfall=0,
            accepted_indices=[],
        )

    zones = _zone_bounds(base_min, r_max)
    effective_weights = config.zone_weights if target_count >= 12 else (1.0, 1.0, 1.0)
    zone_counts = _distribute_counts(target_count, effective_weights)

    placements: list[tuple[np.ndarray, float, int]] = []
    rejected_blocking = rejected_clearance = rejected_oob = 0
    attempts = 0
    max_attempts = max(config.max_resamples * max(target_count, 1), config.max_resamples)
    radius_idx = 0

    device = torch.device("cuda")
    free_mask_c = np.ascontiguousarray(free_mask)
    free_mask_t = torch.as_tensor(free_mask_c, device=device)
    center_mask_c = np.ascontiguousarray(center_mask) if center_mask is not None else None
    center_mask_t = torch.as_tensor(center_mask_c, device=device) if center_mask_c is not None else None
    circle_cache: dict[int, torch.Tensor] = {}

    def _inside_center_mask_torch(xy: np.ndarray) -> bool:
        u, v = world_to_pixel(meta, xy)
        h, w = center_mask_t.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return False
        value = center_mask_t[v, u]
        if center_mask_t.dtype == torch.bool:
            return bool(value.item())
        return bool((value >= config.free_pixel_min).item())

    def _inside_mask_torch(xy: np.ndarray, radius_m: float) -> bool:
        u, v = world_to_pixel(meta, xy)
        radius_px = int(math.ceil(radius_m / float(meta["scale"])))
        h, w = free_mask_t.shape[:2]
        if u < radius_px or v < radius_px or u >= w - radius_px or v >= h - radius_px:
            return False
        circle = circle_cache.get(radius_px)
        if circle is None:
            offsets = torch.arange(-radius_px, radius_px + 1, device=device)
            du = offsets.unsqueeze(0)
            dv = offsets.unsqueeze(1)
            circle = (du * du + dv * dv) <= (radius_px * radius_px)
            circle_cache[radius_px] = circle
        region = free_mask_t[v - radius_px : v + radius_px + 1, u - radius_px : u + radius_px + 1]
        if region.shape != circle.shape:
            return False
        if region.dtype != torch.bool:
            region = region >= config.free_pixel_min
        return bool(torch.all(region[circle]).item())

    for zone_idx, need in enumerate(zone_counts):
        z_min, z_max = zones[zone_idx]
        while need > 0 and attempts < max_attempts and radius_idx < len(radii):
            radius = radii[radius_idx]
            min_center = base_min + radius
            if min_center > r_max:
                rejected_oob += 1
                radius_idx += 1
                need -= 1
                continue
            z_min_eff = max(z_min, min_center)
            z_max_eff = max(z_min_eff, z_max)
            attempts += 1
            angle = rng.uniform(-0.5 * fov_rad, 0.5 * fov_rad)
            r = (
                math.sqrt(rng.uniform(z_min_eff * z_min_eff, z_max_eff * z_max_eff))
                if z_max_eff > z_min_eff
                else z_min_eff
            )
            direction = _rotate(forward_xy, angle)
            candidate = wedge.origin_xy + direction * r

            if center_mask_t is not None:
                if not _inside_center_mask_torch(candidate):
                    rejected_oob += 1
                    continue
            if not center_mask_is_bloomed:
                if not _inside_mask_torch(candidate, radius):
                    rejected_oob += 1
                    continue

            too_close = any(
                float(np.linalg.norm(candidate - placed)) < (radius + placed_radius - EPS)
                for placed, placed_radius, _ in placements
            )
            if not too_close and exclude_discs:
                for center, exclude_radius in exclude_discs:
                    if float(np.linalg.norm(candidate - center)) < (exclude_radius + radius - EPS):
                        too_close = True
                        break
            if too_close:
                rejected_clearance += 1
                continue

            if (
                not config.allow_blocking
                and goal_xy is not None
                and _blocks_goal(wedge.origin_xy, goal_xy, candidate, radius)
            ):
                rejected_blocking += 1
                continue

            placements.append((candidate.astype(np.float32), radius, radius_idx))
            radius_idx += 1
            need -= 1

    achieved_coverage = 0.0
    if placements:
        if config.coverage_mode == "area":
            wedge_area = _wedge_area(fov_rad, base_min, r_max)
            if wedge_area > 0.0:
                covered_area = sum(_disc_area(r) for _, r, _ in placements)
                achieved_coverage = min(1.0, covered_area / wedge_area)
        else:
            total_angle = 0.0
            fov_half = 0.5 * fov_rad
            for pos, radius, _ in placements:
                offset = pos - wedge.origin_xy
                dist = float(np.linalg.norm(offset))
                center_angle = _angle_between(forward_xy, offset)
                span = _effective_disc_span(
                    radius,
                    max(dist, radius + EPS),
                    fov_half,
                    center_angle,
                )
                total_angle += span
            if fov_rad > 0.0:
                achieved_coverage = min(1.0, total_angle / fov_rad)
    shortfall = max(0, target_count - len(placements))

    return NPCPlacementResult(
        positions_xy=[pos for pos, _, _ in placements],
        requested_count=target_count,
        achieved_coverage=achieved_coverage,
        target_coverage=config.target_coverage,
        attempts=attempts,
        rejected_blocking=rejected_blocking,
        rejected_clearance=rejected_clearance,
        rejected_oob=rejected_oob,
        shortfall=shortfall,
        accepted_indices=[idx for _, _, idx in placements],
    )


def plan_npc_positions(
    *,
    wedge: CameraWedge,
    free_mask: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    config: NPCDensityConfig,
    goal_xy: np.ndarray | None = None,
    exclude_discs: Sequence[tuple[np.ndarray, float]] | None = None,
    radii_m: Sequence[float] | None = None,
    center_mask: np.ndarray | None = None,
    center_mask_is_bloomed: bool = False,
    backend: NPCPlacementBackend | str = NPCPlacementBackend.CPU,
) -> NPCPlacementResult:
    """
    Sample NPC positions inside the camera wedge with disc clearance.

    The sampler targets a coverage fraction (if provided) and falls back to max_npcs as a cap.
    It rejects candidates that collide with obstacles/other NPCs, fall outside the mask, or
    block the camera->goal segment when blocking is disabled. Optional exclude_discs are treated
    as occupied (collide but do not contribute to coverage).
    If center_mask is provided, it is treated as a pre-cleared mask of allowed centers.
    When center_mask_is_bloomed is True, the clearance check against free_mask is skipped.
    """
    global _GPU_BACKEND_FAILED, _GPU_FALLBACK_REPORTED
    backend_value = (
        backend.value if isinstance(backend, NPCPlacementBackend) else str(backend).lower()
    )
    if backend_value == NPCPlacementBackend.CPU.value:
        return _plan_npc_positions_cpu(
            wedge=wedge,
            free_mask=free_mask,
            meta=meta,
            rng=rng,
            config=config,
            goal_xy=goal_xy,
            exclude_discs=exclude_discs,
            radii_m=radii_m,
            center_mask=center_mask,
            center_mask_is_bloomed=center_mask_is_bloomed,
        )
    if backend_value == NPCPlacementBackend.GPU.value:
        if _GPU_BACKEND_FAILED:
            return _plan_npc_positions_cpu(
                wedge=wedge,
                free_mask=free_mask,
                meta=meta,
                rng=rng,
                config=config,
                goal_xy=goal_xy,
                exclude_discs=exclude_discs,
                radii_m=radii_m,
                center_mask=center_mask,
                center_mask_is_bloomed=center_mask_is_bloomed,
            )
        try:
            return _plan_npc_positions_gpu(
                wedge=wedge,
                free_mask=free_mask,
                meta=meta,
                rng=rng,
                config=config,
                goal_xy=goal_xy,
                exclude_discs=exclude_discs,
                radii_m=radii_m,
                center_mask=center_mask,
                center_mask_is_bloomed=center_mask_is_bloomed,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _GPU_BACKEND_FAILED = True
            if not _GPU_FALLBACK_REPORTED:
                print(
                    f"[WARN] GPU NPC placement failed ({exc}); falling back to CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                _GPU_FALLBACK_REPORTED = True
            return _plan_npc_positions_cpu(
                wedge=wedge,
                free_mask=free_mask,
                meta=meta,
                rng=rng,
                config=config,
                goal_xy=goal_xy,
                exclude_discs=exclude_discs,
                radii_m=radii_m,
                center_mask=center_mask,
                center_mask_is_bloomed=center_mask_is_bloomed,
            )
    raise ValueError(f"Unknown NPC placement backend: {backend}")


def estimate_coverage_for_positions(
    positions_xy: Sequence[np.ndarray],
    *,
    wedge: CameraWedge,
    config: NPCDensityConfig,
) -> float:
    """Compute coverage achieved by already-placed NPCs."""
    wedge_area = _wedge_area(
        math.radians(wedge.fov_deg),
        max(wedge.min_range, config.min_center_distance),
        wedge.max_range,
    )
    disc_area = _disc_area(config.clearance_radius)
    if wedge_area <= 0.0 or disc_area <= 0.0:
        return 0.0
    return min(1.0, len(list(positions_xy)) * disc_area / wedge_area)


def compute_clearance_distance(free_mask: np.ndarray) -> np.ndarray | None:
    """Return per-pixel distance (in pixels) to the nearest blocked cell."""
    if _ndimage is None:
        return None
    return _ndimage.distance_transform_edt(free_mask)
