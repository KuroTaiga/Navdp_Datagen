from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from navdp_datagen.sensors import load_sensor_rig, sensor_profile_by_name
from utils.massgen_render_manifest import (
    DEFAULT_FPS,
    DEFAULT_RENDER_BACKEND,
    DEFAULT_ROBOT_GLB,
    scenario_file_to_render_manifest,
    write_json,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_render_run_config(path: str | Path) -> JsonDict:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return dict(payload)


def prepare_render_run(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    write_outputs: bool = False,
) -> JsonDict:
    """Build a MassGen manifest, attach sensors, and preflight local inputs."""

    base_dir = Path(config_path).resolve().parent if config_path is not None else REPO_ROOT
    scenario_json = _required_path(config, "scenario_json", base_dir=base_dir, must_exist=True)
    action_catalog_json = _optional_path(config, "action_catalog_json", base_dir=base_dir, must_exist=True)
    output_root = _output_root(config, base_dir=base_dir, scenario_json=scenario_json)
    manifest_json = _manifest_path(config, base_dir=base_dir, output_root=output_root)
    summary_json = _optional_path(config, "summary_json", base_dir=base_dir, must_exist=False)

    render_backend = str(config.get("render_backend", DEFAULT_RENDER_BACKEND))
    fps = float(config.get("fps", DEFAULT_FPS))
    default_robot_glb = str(config.get("default_robot_glb", DEFAULT_ROBOT_GLB))
    manifest = scenario_file_to_render_manifest(
        scenario_json,
        action_catalog_path=action_catalog_json,
        fps=fps,
        render_backend=render_backend,
        default_robot_glb=default_robot_glb,
        visibility_culling=bool(config.get("visibility_culling", True)),
        human_cull_margin_m=float(config.get("human_cull_margin_m", 0.25)),
        robot_cull_margin_m=float(config.get("robot_cull_margin_m", 0.25)),
    )

    sensor_rig = _load_configured_sensor_rig(config, base_dir=base_dir)
    selected_sensors = _selected_sensor_names(config, sensor_rig=sensor_rig)
    _attach_sensors(manifest, sensor_rig=sensor_rig, selected_sensors=selected_sensors)

    preflight = preflight_manifest(
        manifest,
        output_root=output_root,
        manifest_json=manifest_json,
        config=config,
        base_dir=base_dir,
    )
    summary = _summary(
        config=config,
        manifest=manifest,
        output_root=output_root,
        manifest_json=manifest_json,
        sensor_rig=sensor_rig,
        selected_sensors=selected_sensors,
        preflight=preflight,
    )

    if write_outputs:
        write_json(manifest_json, manifest)
        if summary_json is not None:
            write_json(summary_json, summary)

    return {
        "manifest": manifest,
        "summary": summary,
        "manifest_json": str(manifest_json),
        "summary_json": str(summary_json) if summary_json is not None else None,
    }


def prepare_render_run_from_config_path(
    config_path: str | Path,
    *,
    write_outputs: bool = False,
) -> JsonDict:
    config_path = Path(config_path)
    return prepare_render_run(
        load_render_run_config(config_path),
        config_path=config_path,
        write_outputs=write_outputs,
    )


def preflight_manifest(
    manifest: Mapping[str, Any],
    *,
    output_root: Path,
    manifest_json: Path,
    config: Mapping[str, Any],
    base_dir: Path,
) -> JsonDict:
    errors: list[str] = []
    warnings: list[str] = []

    if output_root.exists() and not output_root.is_dir():
        errors.append(f"output_root is not a directory: {output_root}")
    output_parent = _nearest_existing_parent(output_root)
    if not os.access(output_parent, os.W_OK):
        errors.append(f"output_root parent is not writable: {output_parent}")
    manifest_parent = _nearest_existing_parent(manifest_json.parent)
    if not os.access(manifest_parent, os.W_OK):
        errors.append(f"manifest_json parent is not writable: {manifest_parent}")

    strict_assets = bool(config.get("strict_assets", False))
    for label, raw_path in _asset_paths(manifest):
        path = _resolve_path(raw_path, base_dir=base_dir)
        if path.exists():
            continue
        message = f"{label} not found: {raw_path}"
        if strict_assets:
            errors.append(message)
        else:
            warnings.append(message)

    for rig in manifest.get("sensor_rigs", {}).values():
        if isinstance(rig, Mapping) and rig.get("source", {}).get("provisional"):
            warnings.append(f"sensor rig {rig.get('rig_id')} is a provisional fallback profile")

    return {
        "status": "blocked" if errors else "ready",
        "errors": errors,
        "warnings": warnings,
    }


def _summary(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_root: Path,
    manifest_json: Path,
    sensor_rig: Mapping[str, Any],
    selected_sensors: list[str],
    preflight: Mapping[str, Any],
) -> JsonDict:
    jobs = list(manifest.get("jobs", []))
    return {
        "status": preflight["status"],
        "scenario_id": manifest.get("source", {}).get("scenario_id"),
        "scene_id": manifest.get("source", {}).get("scene_id"),
        "mission_families": list(manifest.get("mission_families", [])),
        "render_backend": manifest.get("render_backend"),
        "timing": dict(manifest.get("timing", {})),
        "output_root": str(output_root),
        "manifest_json": str(manifest_json),
        "job_count": len(jobs),
        "jobs": [
            {
                "job_id": job.get("job_id"),
                "viewpoint_robot_id": job.get("viewpoint_robot_id"),
                "sensor_names": [sensor.get("sensor_name") for sensor in job.get("sensors", [])],
                "peer_robot_count": len(job.get("peer_robot_ids", [])),
                "human_count": len(job.get("human_actor_ids", [])),
            }
            for job in jobs
            if isinstance(job, Mapping)
        ],
        "sensor_rig_id": sensor_rig.get("rig_id"),
        "selected_sensors": selected_sensors,
        "gpu_devices": [str(item) for item in config.get("gpu_devices", [])],
        "workers": int(config.get("workers", 1)),
        "preflight": dict(preflight),
        "warnings": list(manifest.get("warnings", [])) + list(preflight.get("warnings", [])),
    }


def _attach_sensors(
    manifest: JsonDict,
    *,
    sensor_rig: Mapping[str, Any],
    selected_sensors: list[str],
) -> None:
    rig = deepcopy(dict(sensor_rig))
    rig_id = str(rig["rig_id"])
    sensor_by_name = {str(sensor["name"]): sensor for sensor in rig.get("sensors", [])}
    missing = [name for name in selected_sensors if name not in sensor_by_name]
    if missing:
        raise ValueError(f"selected_sensors not found in rig {rig_id}: {', '.join(missing)}")
    manifest["sensor_rigs"] = {rig_id: rig}
    job_sensors = [
        {
            "rig_id": rig_id,
            "sensor_name": name,
            "type": sensor_by_name[name].get("type"),
            "modalities": list(sensor_by_name[name].get("modalities", [])),
            "profile": sensor_by_name[name].get("profile"),
        }
        for name in selected_sensors
    ]
    for job in manifest.get("jobs", []):
        if isinstance(job, dict):
            job["sensors"] = deepcopy(job_sensors)


def _load_configured_sensor_rig(config: Mapping[str, Any], *, base_dir: Path) -> JsonDict:
    sensor_rig_json = _optional_path(config, "sensor_rig_json", base_dir=base_dir, must_exist=True)
    if sensor_rig_json is not None:
        return load_sensor_rig(sensor_rig_json)
    profile_name = str(config.get("sensor_profile", "navdp_legacy_fpv"))
    return sensor_profile_by_name(profile_name)


def _selected_sensor_names(config: Mapping[str, Any], *, sensor_rig: Mapping[str, Any]) -> list[str]:
    sensors = [sensor for sensor in sensor_rig.get("sensors", []) if isinstance(sensor, Mapping)]
    explicit = config.get("selected_sensors")
    if explicit is None:
        return [str(sensor["name"]) for sensor in sensors if sensor.get("enabled", True)]
    if not isinstance(explicit, list) or not explicit:
        raise ValueError("selected_sensors must be a non-empty list when provided")
    return [str(item) for item in explicit]


def _asset_paths(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    scene_assets = manifest.get("scene_assets", {})
    if isinstance(scene_assets, Mapping):
        for key, value in scene_assets.items():
            if isinstance(value, str) and _looks_like_path_key(str(key)):
                out.append((f"scene_assets.{key}", value))
    actors = manifest.get("actors", {})
    if isinstance(actors, Mapping):
        for robot in actors.get("robots", []):
            if not isinstance(robot, Mapping):
                continue
            robot_id = str(robot.get("actor_id"))
            asset = robot.get("asset", {})
            if isinstance(asset, Mapping):
                for key in ("glb_path", "urdf_path"):
                    value = asset.get(key)
                    if isinstance(value, str) and value:
                        out.append((f"robots.{robot_id}.asset.{key}", value))
        for human in actors.get("humans", []):
            if not isinstance(human, Mapping):
                continue
            human_id = str(human.get("actor_id"))
            for index, segment in enumerate(human.get("action_segments", [])):
                if not isinstance(segment, Mapping):
                    continue
                asset = segment.get("asset", {})
                if isinstance(asset, Mapping):
                    for key, value in asset.items():
                        if isinstance(value, str) and _looks_like_path_key(str(key)):
                            out.append((f"humans.{human_id}.action_segments[{index}].asset.{key}", value))
    return out


def _looks_like_path_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_path") or lowered.endswith("_file") or lowered in {"path", "uri"}


def _required_path(
    config: Mapping[str, Any],
    key: str,
    *,
    base_dir: Path,
    must_exist: bool,
) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config.{key} must be a non-empty string")
    path = _resolve_path(value, base_dir=base_dir)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"config.{key} does not exist: {path}")
    return path


def _optional_path(
    config: Mapping[str, Any],
    key: str,
    *,
    base_dir: Path,
    must_exist: bool,
) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"config.{key} must be a string when provided")
    path = _resolve_path(value, base_dir=base_dir)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"config.{key} does not exist: {path}")
    return path


def _output_root(config: Mapping[str, Any], *, base_dir: Path, scenario_json: Path) -> Path:
    value = config.get("output_root")
    if isinstance(value, str) and value:
        return _resolve_path(value, base_dir=base_dir)
    return REPO_ROOT / "out" / "massgen_runs" / scenario_json.stem


def _manifest_path(config: Mapping[str, Any], *, base_dir: Path, output_root: Path) -> Path:
    value = config.get("manifest_json")
    if isinstance(value, str) and value:
        return _resolve_path(value, base_dir=base_dir)
    return output_root / "render_manifest.json"


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (REPO_ROOT / path).resolve()


def _nearest_existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current
