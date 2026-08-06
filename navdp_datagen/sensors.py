from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]


def pinhole_from_fov_y(width: int, height: int, fov_y_deg: float) -> JsonDict:
    """Return centered square-pixel pinhole intrinsics from vertical FOV."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if fov_y_deg <= 0.0 or fov_y_deg >= 180.0:
        raise ValueError("fov_y_deg must be in (0, 180)")
    fov_y_rad = math.radians(float(fov_y_deg))
    fy = (height / 2.0) / math.tan(fov_y_rad / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    fov_x_rad = 2.0 * math.atan((width / 2.0) / fx)
    return {
        "model": "pinhole",
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fov_x_deg": math.degrees(fov_x_rad),
        "fov_y_deg": float(fov_y_deg),
        "distortion_model": "none",
        "distortion_coefficients": [],
    }


def pinhole_from_openusd_camera(
    *,
    width: int,
    height: int,
    focal_length: float = 50.0,
    horizontal_aperture: float = 20.955,
    vertical_aperture: float = 15.2908,
) -> JsonDict:
    """Return pinhole intrinsics from OpenUSD camera optical attributes."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if focal_length <= 0.0 or horizontal_aperture <= 0.0 or vertical_aperture <= 0.0:
        raise ValueError("OpenUSD camera optics must be positive")
    fx = width * float(focal_length) / float(horizontal_aperture)
    fy = height * float(focal_length) / float(vertical_aperture)
    cx = width / 2.0
    cy = height / 2.0
    fov_x_rad = 2.0 * math.atan(float(horizontal_aperture) / (2.0 * float(focal_length)))
    fov_y_rad = 2.0 * math.atan(float(vertical_aperture) / (2.0 * float(focal_length)))
    return {
        "model": "pinhole",
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fov_x_deg": math.degrees(fov_x_rad),
        "fov_y_deg": math.degrees(fov_y_rad),
        "distortion_model": "none",
        "distortion_coefficients": [],
        "openusd_camera": {
            "focal_length": float(focal_length),
            "horizontal_aperture": float(horizontal_aperture),
            "vertical_aperture": float(vertical_aperture),
        },
    }


def _camera_sensor(
    *,
    name: str,
    profile_name: str,
    intrinsics: Mapping[str, Any],
    translation_m: list[float],
    rotation_rpy_deg: list[float],
    clipping_range_m: list[float],
    modalities: list[str],
    rate_hz: float,
    notes: str,
) -> JsonDict:
    return {
        "name": name,
        "type": "camera",
        "profile": profile_name,
        "enabled": True,
        "prim_path": None,
        "frame": "robot_base",
        "transform": {
            "translation_m": list(translation_m),
            "rotation_rpy_deg": list(rotation_rpy_deg),
            "convention": "+X forward, +Y left, +Z up",
        },
        "intrinsics": dict(intrinsics),
        "clipping_range_m": list(clipping_range_m),
        "rate_hz": float(rate_hz),
        "modalities": list(modalities),
        "notes": notes,
    }


def default_sensor_profiles() -> dict[str, JsonDict]:
    legacy_intrinsics = pinhole_from_fov_y(960, 720, 70.0)
    return {
        "navdp_legacy_fpv": {
            "rig_id": "navdp_legacy_fpv",
            "profile": "navdp_legacy_fpv",
            "source": {
                "kind": "fallback_profile",
                "provisional": False,
                "doc": "docs/camera_sensor_defaults.md",
            },
            "sensors": [
                _camera_sensor(
                    name="fpv_rgb",
                    profile_name="navdp_legacy_fpv",
                    intrinsics=legacy_intrinsics,
                    translation_m=[0.0, 0.0, 0.3],
                    rotation_rpy_deg=[0.0, 0.0, 0.0],
                    clipping_range_m=[0.001, 30.0],
                    modalities=["rgb", "depth", "camera_metadata"],
                    rate_hz=10.0,
                    notes="Legacy path-follow camera; vertical placement is occupancy_upper_z + 0.3m.",
                )
            ],
        },
        "g1_head_fpv_default": {
            "rig_id": "g1_head_fpv_default",
            "profile": "g1_head_fpv_default",
            "source": {
                "kind": "fallback_profile",
                "provisional": True,
                "doc": "docs/camera_sensor_defaults.md",
            },
            "sensors": [
                _camera_sensor(
                    name="head_rgbd",
                    profile_name="g1_head_fpv_default",
                    intrinsics=legacy_intrinsics,
                    translation_m=[0.10, 0.0, 1.20],
                    rotation_rpy_deg=[0.0, -5.0, 0.0],
                    clipping_range_m=[0.05, 30.0],
                    modalities=["rgb", "depth", "camera_metadata"],
                    rate_hz=10.0,
                    notes="Provisional G1 head camera fallback until an imported USD rig is available.",
                )
            ],
        },
        "openusd_camera_fallback": {
            "rig_id": "openusd_camera_fallback",
            "profile": "openusd_camera_fallback",
            "source": {
                "kind": "fallback_profile",
                "provisional": True,
                "doc": "docs/camera_sensor_defaults.md",
            },
            "sensors": [
                _camera_sensor(
                    name="usd_camera",
                    profile_name="openusd_camera_fallback",
                    intrinsics=pinhole_from_openusd_camera(width=960, height=720),
                    translation_m=[0.0, 0.0, 0.0],
                    rotation_rpy_deg=[0.0, 0.0, 0.0],
                    clipping_range_m=[1.0, 1_000_000.0],
                    modalities=["rgb", "camera_metadata"],
                    rate_hz=10.0,
                    notes="Raw OpenUSD optics fallback for camera prims missing authored optics.",
                )
            ],
        },
    }


def sensor_profile_by_name(name: str) -> JsonDict:
    profiles = default_sensor_profiles()
    if name not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"unknown sensor profile {name!r}; expected one of: {known}")
    return deepcopy(profiles[name])


def load_sensor_rig(path: str | Path) -> JsonDict:
    rig_path = Path(path)
    payload = json.loads(rig_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{rig_path} must contain a JSON object")
    return normalize_sensor_rig(payload, source_path=rig_path)


def normalize_sensor_rig(payload: Mapping[str, Any], *, source_path: str | Path | None = None) -> JsonDict:
    """Normalize a renderer-neutral or IsaacSim-style exported rig JSON."""

    rig_id = str(payload.get("rig_id") or payload.get("name") or payload.get("robot_id") or "imported_sensor_rig")
    sensors_raw = payload.get("sensors") or payload.get("cameras") or []
    if not isinstance(sensors_raw, list) or not sensors_raw:
        raise ValueError("sensor rig must contain a non-empty 'sensors' or 'cameras' list")
    sensors = [_normalize_sensor(item, index) for index, item in enumerate(sensors_raw)]
    return {
        "rig_id": rig_id,
        "profile": str(payload.get("profile") or "imported"),
        "robot_id": payload.get("robot_id"),
        "source": {
            "kind": str(payload.get("source_kind") or "imported_json"),
            "path": str(source_path) if source_path is not None else None,
            "format": str(payload.get("format") or payload.get("schema_version") or "navdp_sensor_rig_json"),
            "provisional": False,
        },
        "sensors": sensors,
        "metadata": dict(payload.get("metadata") or {}),
    }


def _normalize_sensor(raw: Any, index: int) -> JsonDict:
    if not isinstance(raw, Mapping):
        raise ValueError(f"sensors[{index}] must be an object")
    name = str(raw.get("name") or raw.get("sensor_name") or raw.get("camera_name") or f"sensor_{index}")
    sensor_type = str(raw.get("type") or raw.get("sensor_type") or "camera")
    if sensor_type != "camera":
        raise ValueError(f"sensors[{index}] has unsupported type {sensor_type!r}; only camera is supported now")
    width, height = _resolution(raw)
    intrinsics = _intrinsics(raw, width=width, height=height)
    return {
        "name": name,
        "type": "camera",
        "profile": str(raw.get("profile") or "imported"),
        "enabled": bool(raw.get("enabled", True)),
        "prim_path": raw.get("prim_path") or raw.get("camera_prim_path") or raw.get("path"),
        "frame": str(raw.get("frame") or raw.get("parent_frame") or "robot_base"),
        "transform": {
            "translation_m": _float_list(
                raw.get("translation_m")
                or raw.get("local_position")
                or raw.get("position")
                or [0.0, 0.0, 0.0],
                length=3,
                field=f"sensors[{index}].translation_m",
            ),
            "rotation_rpy_deg": _float_list(
                raw.get("rotation_rpy_deg")
                or raw.get("local_rotation_rpy_deg")
                or raw.get("orientation_rpy_deg")
                or [0.0, 0.0, 0.0],
                length=3,
                field=f"sensors[{index}].rotation_rpy_deg",
            ),
            "convention": str(raw.get("transform_convention") or "+X forward, +Y left, +Z up"),
        },
        "intrinsics": intrinsics,
        "clipping_range_m": _clipping_range(raw),
        "rate_hz": float(raw.get("rate_hz") or raw.get("frequency") or 10.0),
        "modalities": [str(item) for item in raw.get("modalities", ["rgb"])],
        "notes": str(raw.get("notes") or ""),
        "metadata": dict(raw.get("metadata") or {}),
    }


def _resolution(raw: Mapping[str, Any]) -> tuple[int, int]:
    resolution = raw.get("resolution")
    if isinstance(resolution, list) and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])
    width = raw.get("width")
    height = raw.get("height")
    if width is not None and height is not None:
        return int(width), int(height)
    render_product = raw.get("render_product")
    if isinstance(render_product, Mapping):
        rp_resolution = render_product.get("resolution")
        if isinstance(rp_resolution, list) and len(rp_resolution) == 2:
            return int(rp_resolution[0]), int(rp_resolution[1])
    return 960, 720


def _intrinsics(raw: Mapping[str, Any], *, width: int, height: int) -> JsonDict:
    explicit = raw.get("intrinsics")
    if isinstance(explicit, Mapping):
        out = dict(explicit)
        out.setdefault("model", "pinhole")
        out.setdefault("width", width)
        out.setdefault("height", height)
        return out
    if raw.get("fx") is not None and raw.get("fy") is not None:
        fx = float(raw["fx"])
        fy = float(raw["fy"])
        return {
            "model": "pinhole",
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": float(raw.get("cx", width / 2.0)),
            "cy": float(raw.get("cy", height / 2.0)),
            "distortion_model": str(raw.get("distortion_model") or "none"),
            "distortion_coefficients": list(raw.get("distortion_coefficients") or []),
        }
    if raw.get("fov_y_deg") is not None:
        return pinhole_from_fov_y(width, height, float(raw["fov_y_deg"]))
    if raw.get("focal_length") is not None:
        return pinhole_from_openusd_camera(
            width=width,
            height=height,
            focal_length=float(raw.get("focal_length", 50.0)),
            horizontal_aperture=float(raw.get("horizontal_aperture", 20.955)),
            vertical_aperture=float(raw.get("vertical_aperture", 15.2908)),
        )
    return pinhole_from_fov_y(width, height, 70.0)


def _clipping_range(raw: Mapping[str, Any]) -> list[float]:
    clipping = raw.get("clipping_range_m") or raw.get("clipping_range")
    if isinstance(clipping, list) and len(clipping) == 2:
        return [float(clipping[0]), float(clipping[1])]
    return [0.001, 30.0]


def _float_list(value: Any, *, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a list of {length} numbers")
    return [float(item) for item in value]
