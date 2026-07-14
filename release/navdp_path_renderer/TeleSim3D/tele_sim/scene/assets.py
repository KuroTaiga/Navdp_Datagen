"""Scene asset loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


class SceneAssetError(RuntimeError):
    """Raised when a scene asset bundle is incomplete or malformed."""


Vec3 = Tuple[float, float, float]


def _as_vec3(candidate: Sequence[object], *, label: str) -> Vec3:
    if len(candidate) != 3:
        raise SceneAssetError(f"{label} must contain exactly 3 elements; received {candidate!r}")
    try:
        return float(candidate[0]), float(candidate[1]), float(candidate[2])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise SceneAssetError(f"{label} must contain numeric values") from exc


@dataclass(frozen=True)
class SceneAsset:
    """Immutable description of scene resources produced by the asset builder."""

    scene_id: str
    metadata_path: Path
    scene_glb: Path
    dataset_config: Optional[Path]
    navmesh_path: Path
    bev_path: Path
    meters_per_pixel: float
    bounds_min: Vec3
    bounds_max: Vec3
    resolved_slice_height: float
    scene_metadata_path: Optional[Path]
    splat_model_path: Optional[Path]
    splat_bev_path: Optional[Path]

    @property
    def bev_origin_world(self) -> Vec3:
        """Return the world-space origin for BEV pixel (0, 0)."""

        x, _, z = self.bounds_min
        return x, self.resolved_slice_height, z

    def bev_pixel_to_world(self, pixel_x: float, pixel_y: float, *, height: Optional[float] = None) -> Vec3:
        """Map BEV pixel coordinates back to world-space."""

        origin_x, default_y, origin_z = self.bev_origin_world
        world_x = origin_x + pixel_x * self.meters_per_pixel
        world_z = origin_z + pixel_y * self.meters_per_pixel
        world_y = height if height is not None else default_y
        return world_x, world_y, world_z

    def world_to_bev_pixel(self, world_x: float, world_z: float) -> Tuple[float, float]:
        """Inverse mapping from world coordinates to BEV pixel indices."""

        origin_x, _, origin_z = self.bev_origin_world
        pixel_x = (world_x - origin_x) / self.meters_per_pixel
        pixel_y = (world_z - origin_z) / self.meters_per_pixel
        return pixel_x, pixel_y

    @staticmethod
    def from_nav_metadata(metadata_path: Path, *, scene_id: Optional[str] = None) -> "SceneAsset":
        """Load a scene asset description from the nav asset metadata JSON."""

        if not metadata_path.exists():
            raise SceneAssetError(f"Metadata file not found: {metadata_path}")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SceneAssetError(f"Failed to parse metadata JSON at {metadata_path}") from exc
        if not isinstance(payload, dict):
            raise SceneAssetError("Metadata JSON must contain an object")

        scene_glb_raw = payload.get("scene_glb")
        navmesh_raw = payload.get("navmesh_path")
        bev_raw = payload.get("bev_path")
        meters_per_pixel = payload.get("meters_per_pixel")
        bounds_min_raw = payload.get("bounds_min")
        bounds_max_raw = payload.get("bounds_max")
        resolved_slice_height = payload.get("resolved_slice_height")

        if not isinstance(scene_glb_raw, str):
            raise SceneAssetError("Metadata missing 'scene_glb' path")
        if not isinstance(navmesh_raw, str):
            raise SceneAssetError("Metadata missing 'navmesh_path'")
        if not isinstance(bev_raw, str):
            raise SceneAssetError("Metadata missing 'bev_path'")
        if not isinstance(meters_per_pixel, (int, float)):
            raise SceneAssetError("Metadata missing numeric 'meters_per_pixel'")
        if not isinstance(bounds_min_raw, list) or not isinstance(bounds_max_raw, list):
            raise SceneAssetError("Metadata missing 'bounds_min'/'bounds_max' vectors")
        if not isinstance(resolved_slice_height, (int, float)):
            raise SceneAssetError("Metadata missing numeric 'resolved_slice_height'")

        dataset_config_raw = payload.get("dataset_config")
        scene_metadata_raw = payload.get("scene_metadata_path")
        splat_bev_raw = payload.get("splat_bev_path")
        splat_model_raw = payload.get("splat_model_path")

        dataset_config = Path(dataset_config_raw).resolve() if isinstance(dataset_config_raw, str) else None
        scene_metadata_path = Path(scene_metadata_raw).resolve() if isinstance(scene_metadata_raw, str) else None
        splat_bev_path = Path(splat_bev_raw).resolve() if isinstance(splat_bev_raw, str) else None
        splat_model_path = Path(splat_model_raw).resolve() if isinstance(splat_model_raw, str) else None

        scene_glb = Path(scene_glb_raw).resolve()
        navmesh_path = Path(navmesh_raw).resolve()
        bev_path = Path(bev_raw).resolve()
        bounds_min = _as_vec3(bounds_min_raw, label="bounds_min")
        bounds_max = _as_vec3(bounds_max_raw, label="bounds_max")

        if scene_id is None:
            scene_id = scene_glb.stem.replace(" ", "_")

        return SceneAsset(
            scene_id=scene_id,
            metadata_path=metadata_path.resolve(),
            scene_glb=scene_glb,
            dataset_config=dataset_config,
            navmesh_path=navmesh_path,
            bev_path=bev_path,
            meters_per_pixel=float(meters_per_pixel),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resolved_slice_height=float(resolved_slice_height),
            scene_metadata_path=scene_metadata_path,
            splat_model_path=splat_model_path,
            splat_bev_path=splat_bev_path,
        )
