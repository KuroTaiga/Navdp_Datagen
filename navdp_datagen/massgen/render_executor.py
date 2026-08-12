from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.massgen_render_manifest import load_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDER_SCRIPT = REPO_ROOT / "render_label_paths_telesim.py"
DEFAULT_VIDEO_BACKEND = "nvenc"
DEFAULT_DEVICE = "cuda"


def load_render_manifest(path: str | Path) -> JsonDict:
    manifest = load_json(path)
    if not isinstance(manifest.get("jobs"), list):
        raise ValueError(f"{path} must contain a MassGen render manifest with a jobs list")
    return manifest


def build_render_plans(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    output_root: str | Path,
    scenes_dir: str | Path | None = None,
    tasks_dir: str | Path | None = None,
    render_script: str | Path = DEFAULT_RENDER_SCRIPT,
    python_bin: str = sys.executable,
    families: Sequence[str] | None = None,
    job_ids: Sequence[str] | None = None,
    robot_ids: Sequence[str] | None = None,
    sensor_names: Sequence[str] | None = None,
    limit: int | None = None,
    write_inputs: bool = False,
    video_backend: str = DEFAULT_VIDEO_BACKEND,
    device: str = DEFAULT_DEVICE,
    save_depth_maps: bool = False,
    save_rgb_frames: bool = False,
    minimal_frames: int | None = None,
    actor_gpu_resident: bool = False,
    save_actor_metadata: bool = True,
) -> JsonDict:
    """Build dry-run or executable plans for manifest jobs.

    The executor currently bridges manifest robot trajectories into the existing
    label-path renderer. It deliberately reports human-composition blockers
    instead of silently producing camera-only renders for human mission families.
    """

    manifest_base = Path(manifest_path).resolve().parent if manifest_path is not None else REPO_ROOT
    output_root = Path(output_root).expanduser().resolve()
    tasks_root = Path(tasks_dir).expanduser().resolve() if tasks_dir is not None else output_root / "render_inputs" / "tasks"
    render_output_root = output_root / "renders"
    metrics_root = output_root / "metrics"
    selected_jobs = _select_jobs(
        manifest,
        families=families,
        job_ids=job_ids,
        robot_ids=robot_ids,
        sensor_names=sensor_names,
        limit=limit,
    )
    plans = [
        _build_job_plan(
            manifest,
            job,
            manifest_base=manifest_base,
            output_root=output_root,
            render_output_root=render_output_root,
            metrics_root=metrics_root,
            tasks_root=tasks_root,
            scenes_dir=Path(scenes_dir).expanduser().resolve() if scenes_dir is not None else None,
            render_script=Path(render_script).expanduser().resolve(),
            python_bin=python_bin,
            write_inputs=write_inputs,
            video_backend=video_backend,
            device=device,
            save_depth_maps=save_depth_maps,
            save_rgb_frames=save_rgb_frames,
            minimal_frames=minimal_frames,
            actor_gpu_resident=actor_gpu_resident,
            save_actor_metadata=save_actor_metadata,
        )
        for job in selected_jobs
    ]
    return {
        "status": "blocked" if any(plan["blockers"] for plan in plans) else "ready",
        "write_inputs": bool(write_inputs),
        "manifest": {
            "schema_version": manifest.get("schema_version"),
            "scenario_id": manifest.get("source", {}).get("scenario_id"),
            "scene_id": manifest.get("source", {}).get("scene_id"),
            "mission_families": list(manifest.get("mission_families", [])),
            "render_backend": manifest.get("render_backend"),
        },
        "selection": {
            "families": list(families or []),
            "job_ids": list(job_ids or []),
            "robot_ids": list(robot_ids or []),
            "sensor_names": list(sensor_names or []),
            "limit": limit,
        },
        "output_root": str(output_root),
        "tasks_dir": str(tasks_root),
        "render_output_root": str(render_output_root),
        "job_count": len(plans),
        "plans": plans,
    }


def execute_render_plans(plan_payload: Mapping[str, Any]) -> int:
    plans = [plan for plan in plan_payload.get("plans", []) if isinstance(plan, Mapping)]
    if not plans:
        return 1
    blocked = [plan for plan in plans if plan.get("blockers")]
    if blocked:
        return 2
    for plan in plans:
        command = [str(item) for item in plan["command"]]
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in plan.get("env", {}).items()})
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def format_plan_text(plan_payload: Mapping[str, Any]) -> str:
    lines = [
        (
            f"MassGen render plan: status={plan_payload.get('status')} "
            f"jobs={plan_payload.get('job_count')} output_root={plan_payload.get('output_root')}"
        )
    ]
    for plan in plan_payload.get("plans", []):
        if not isinstance(plan, Mapping):
            continue
        lines.append("")
        lines.append(
            f"Job {plan.get('job_id')} scene={plan.get('scene_id')} "
            f"robot={plan.get('viewpoint_robot_id')} status={plan.get('status')}"
        )
        lines.append(f"  label_path: {plan.get('label_path')}")
        lines.append(f"  sensors: {', '.join(plan.get('sensor_names', [])) or '-'}")
        lines.append(f"  humans: {', '.join(plan.get('human_actor_ids', [])) or '-'}")
        if plan.get("blockers"):
            lines.append("  blockers:")
            for blocker in plan.get("blockers", []):
                lines.append(f"  - {blocker}")
        if plan.get("warnings"):
            lines.append("  warnings:")
            for warning in plan.get("warnings", []):
                lines.append(f"  - {warning}")
        env_prefix = " ".join(
            f"{shlex.quote(str(key))}={shlex.quote(str(value))}"
            for key, value in sorted(plan.get("env", {}).items())
        )
        command = shlex.join([str(item) for item in plan.get("command", [])])
        lines.append(f"  command: {env_prefix} {command}".rstrip())
    return "\n".join(lines)


def _select_jobs(
    manifest: Mapping[str, Any],
    *,
    families: Sequence[str] | None,
    job_ids: Sequence[str] | None,
    robot_ids: Sequence[str] | None,
    sensor_names: Sequence[str] | None,
    limit: int | None,
) -> list[Mapping[str, Any]]:
    family_filter = {str(item) for item in (families or []) if str(item)}
    job_filter = {str(item) for item in (job_ids or []) if str(item)}
    robot_filter = {str(item) for item in (robot_ids or []) if str(item)}
    sensor_filter = {str(item) for item in (sensor_names or []) if str(item)}
    selected: list[Mapping[str, Any]] = []
    for job in manifest.get("jobs", []):
        if not isinstance(job, Mapping):
            continue
        if job_filter and str(job.get("job_id")) not in job_filter:
            continue
        if robot_filter and str(job.get("viewpoint_robot_id")) not in robot_filter:
            continue
        job_families = {str(item) for item in job.get("mission_families", [])}
        if family_filter and not any(_family_matches(requested, job_families, manifest) for requested in family_filter):
            continue
        job_sensors = {
            str(sensor.get("sensor_name"))
            for sensor in job.get("sensors", [])
            if isinstance(sensor, Mapping) and sensor.get("sensor_name")
        }
        if sensor_filter and not (sensor_filter & job_sensors):
            continue
        selected.append(job)
        if limit is not None and int(limit) > 0 and len(selected) >= int(limit):
            break
    return selected


def _family_matches(requested: str, job_families: set[str], manifest: Mapping[str, Any]) -> bool:
    if requested in job_families:
        return True
    if ":" not in requested:
        return False
    base, _, subfamily = requested.partition(":")
    if base not in job_families:
        return False
    token = subfamily.strip().lower()
    social_ids = " ".join(str(item).lower() for item in manifest.get("social_law_ids", []))
    missions = " ".join(json.dumps(item, sort_keys=True).lower() for item in manifest.get("missions", []))
    return token in social_ids or token in missions


def _build_job_plan(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    manifest_base: Path,
    output_root: Path,
    render_output_root: Path,
    metrics_root: Path,
    tasks_root: Path,
    scenes_dir: Path | None,
    render_script: Path,
    python_bin: str,
    write_inputs: bool,
    video_backend: str,
    device: str,
    save_depth_maps: bool,
    save_rgb_frames: bool,
    minimal_frames: int | None,
    actor_gpu_resident: bool,
    save_actor_metadata: bool,
) -> JsonDict:
    job_id = str(job.get("job_id") or job.get("outputs", {}).get("stem") or "massgen_job")
    scene_id = str(job.get("scene_id") or manifest.get("source", {}).get("scene_id") or "")
    blockers: list[str] = []
    warnings: list[str] = []
    scene_dir, scene_root, gaussian_model = _resolve_scene_paths(
        manifest,
        scene_id=scene_id,
        scenes_dir=scenes_dir,
        base_dir=manifest_base,
    )
    label_path = tasks_root / scene_id / "label_paths" / f"{job_id}.json"
    actor_plan_path: Path | None = None
    if write_inputs:
        try:
            _write_label_path(job, label_path=label_path, scene_dir=scene_dir)
        except Exception as exc:  # pylint: disable=broad-except
            blockers.append(f"unable to materialize label path: {exc}")
    else:
        warnings.append("label path not written; pass --write-inputs before --execute")

    sensor_details = _job_sensor_details(manifest, job)
    if len(sensor_details) > 1:
        blockers.append("multiple sensors per job are not executable yet; select one sensor")
    sensor = sensor_details[0] if sensor_details else {}
    resolution = _sensor_resolution(sensor)
    fov_y = _sensor_fov_y_deg(sensor)
    znear, zfar = _sensor_clipping(sensor)
    fps = float(manifest.get("timing", {}).get("fps", 10.0) or 10.0)
    command = [
        str(python_bin),
        str(render_script),
        "--scenes-dir",
        str(scene_root),
        "--tasks-dir",
        str(tasks_root),
        "--scene",
        scene_id,
        "--label-id",
        job_id,
        "--output-dir",
        str(render_output_root),
        "--metrics-json",
        str(metrics_root / f"{job_id}.json"),
        "--path-handedness",
        "left",
        "--no-mirror-translation",
        "--video-backend",
        str(video_backend),
        "--video-fps",
        str(max(1, int(round(fps)))),
        "--follow-distance",
        "0",
        "--resolution",
        str(resolution[0]),
        str(resolution[1]),
        "--fov-deg",
        _float_arg(fov_y),
        "--znear",
        _float_arg(znear),
        "--zfar",
        _float_arg(zfar),
        "--device",
        str(device),
        "--save-camera-metadata",
    ]
    if gaussian_model is not None:
        command.extend(["--gaussian-model", str(gaussian_model)])
    if save_depth_maps:
        command.append("--save-depth-maps")
    if save_rgb_frames:
        command.append("--rgb-frames")
    if minimal_frames is not None and int(minimal_frames) > 0:
        command.extend(["--minimal-frames", str(int(minimal_frames))])
    human_ids = [str(item) for item in job.get("human_actor_ids", [])]
    if job.get("peer_robot_ids"):
        blockers.append(
            "peer robot rendering is not connected in the simple human-only renderer path yet"
        )
    actor_plan_payload, actor_plan_path, human_blockers, human_warnings = _human_actor_plan_bundle(
        manifest,
        job,
        human_ids,
        base_dir=manifest_base,
        actor_plan_path=tasks_root / scene_id / "actor_plans" / f"{job_id}.json",
        frame_count_hint=len(_trajectory_world_points(job.get("camera", {}).get("trajectory", []))),
    )
    blockers.extend(human_blockers)
    warnings.extend(human_warnings)
    if actor_plan_payload is not None and actor_plan_path is not None:
        if write_inputs:
            try:
                _write_actor_plan(actor_plan_payload, actor_plan_path)
            except Exception as exc:  # pylint: disable=broad-except
                blockers.append(f"unable to materialize actor plan: {exc}")
        else:
            warnings.append("actor plan not written; pass --write-inputs before --execute")
        command.extend(["--actor-plan-json", str(actor_plan_path)])
        if actor_gpu_resident:
            command.append("--actor-gpu-resident")
        if save_actor_metadata:
            command.append("--save-actor-metadata")
    render_options = job.get("render_options", {})
    if isinstance(render_options, Mapping):
        cull = render_options.get("human_visibility_culling", {})
        if isinstance(cull, Mapping) and cull.get("enabled"):
            command.extend(["--actor-visibility-culling", "--actor-cull-margin-m", _float_arg(cull.get("margin_m", 0.25))])
    return {
        "status": "blocked" if blockers else "ready",
        "job_id": job_id,
        "scene_id": scene_id,
        "viewpoint_robot_id": job.get("viewpoint_robot_id"),
        "mission_families": list(job.get("mission_families", [])),
        "assigned_mission_ids": list(job.get("assigned_mission_ids", [])),
        "sensor_names": [str(sensor.get("name") or sensor.get("sensor_name")) for sensor in sensor_details],
        "human_actor_ids": human_ids,
        "peer_robot_ids": list(job.get("peer_robot_ids", [])),
        "scene_dir": str(scene_dir),
        "scene_root": str(scene_root),
        "gaussian_model": str(gaussian_model) if gaussian_model is not None else None,
        "label_path": str(label_path),
        "actor_plan_path": str(actor_plan_path) if actor_plan_path is not None else None,
        "outputs": {
            "render_dir": str(render_output_root / scene_id),
            "metrics_json": str(metrics_root / f"{job_id}.json"),
        },
        "env": {"GAUSSIAN_RENDER_BACKEND": "gsplat"},
        "command": command,
        "blockers": blockers,
        "warnings": warnings,
        "metadata": {
            "current_executor_boundary": (
                "camera label materialization plus human Gaussian actor bundles; "
                "peer robot GLB composition is still in progress"
            ),
            "output_root": str(output_root),
        },
    }


def _resolve_scene_paths(
    manifest: Mapping[str, Any],
    *,
    scene_id: str,
    scenes_dir: Path | None,
    base_dir: Path,
) -> tuple[Path, Path, Path | None]:
    gaussian_model = _scene_gaussian_model(manifest, base_dir=base_dir)
    if scenes_dir is not None:
        scene_dir = _resolve_scene_dir(scenes_dir, scene_id)
        return scene_dir, scenes_dir, gaussian_model
    if gaussian_model is not None:
        scene_dir = gaussian_model.parent
        return scene_dir, scene_dir.parent, gaussian_model
    scene_root = REPO_ROOT / "data" / "scenes"
    return scene_root / scene_id, scene_root, None


def _resolve_scene_dir(root: Path, scene_id: str) -> Path:
    direct = root / scene_id
    if direct.exists() or root.name != scene_id:
        return direct
    return root


def _scene_gaussian_model(manifest: Mapping[str, Any], *, base_dir: Path) -> Path | None:
    scene_assets = manifest.get("scene_assets", {})
    if not isinstance(scene_assets, Mapping):
        return None
    preferred = (
        "splat_model_path",
        "gaussian_model_path",
        "gaussian_ply_path",
        "point_cloud_path",
        "ply_path",
    )
    for key in preferred:
        value = scene_assets.get(key)
        if isinstance(value, str) and value:
            return _resolve_path(value, base_dir=base_dir)
    for key, value in scene_assets.items():
        if isinstance(value, str) and value and (key.endswith("_path") or value.endswith(".ply")):
            return _resolve_path(value, base_dir=base_dir)
    return None


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [(base_dir / path).resolve(), (REPO_ROOT / path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _write_label_path(job: Mapping[str, Any], *, label_path: Path, scene_dir: Path) -> None:
    trajectory = job.get("camera", {}).get("trajectory", [])
    if not isinstance(trajectory, list):
        raise ValueError("job.camera.trajectory must be a list")
    points = _trajectory_world_points(trajectory)
    if len(points) < 2:
        raise ValueError("job.camera.trajectory must contain at least two distinct positions")
    meta = _load_occupancy_metadata_for_label(scene_dir)
    raster_world = [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in points]
    raster_pixel = [list(_world_to_pixel(meta, float(x), float(y))) for x, y, _ in points]
    payload = {
        "ins_id": str(job.get("job_id") or label_path.stem),
        "scene_id": str(job.get("scene_id") or scene_dir.name),
        "source": "massgen_render_executor",
        "path": {
            "raster_world": raster_world,
            "raster_pixel": raster_pixel,
        },
        "metadata": {
            "viewpoint_robot_id": job.get("viewpoint_robot_id"),
            "mission_families": list(job.get("mission_families", [])),
            "assigned_mission_ids": list(job.get("assigned_mission_ids", [])),
        },
    }
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_occupancy_metadata_for_label(scene_dir: Path) -> JsonDict:
    occ_json = scene_dir / "occupancy.json"
    if not occ_json.is_file():
        raise FileNotFoundError(f"Missing occupancy.json in {scene_dir}")
    occ = json.loads(occ_json.read_text(encoding="utf-8"))
    if not isinstance(occ, Mapping):
        raise ValueError(f"Invalid occupancy.json in {scene_dir}")
    occ_png = scene_dir / "occupancy.png"
    if not occ_png.is_file():
        raise FileNotFoundError(f"Missing occupancy.png in {scene_dir}")
    width_px, height_px = _read_png_size(occ_png)
    scale = float(occ.get("scale", 1.0))
    min_x, min_y, min_z = map(float, occ.get("min", (0.0, 0.0, 0.0)))
    max_x, max_y, max_z = map(float, occ.get("max", (0.0, 0.0, 0.0)))
    lower = occ.get("lower") or [min_x, min_y, min_z]
    upper = occ.get("upper") or [max_x, max_y, max_z]
    return {
        "width": int(width_px),
        "height": int(height_px),
        "scale": scale,
        "left": min_x,
        "right": min_x + int(width_px) * scale,
        "top": max_y,
        "bottom": max_y - int(height_px) * scale,
        "lower_z": float(lower[2]),
        "upper_z": float(upper[2]),
    }


def _read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(8)
        if header != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a valid PNG file")
        length = int.from_bytes(handle.read(4), "big")
        chunk_type = handle.read(4)
        if length < 8 or chunk_type != b"IHDR":
            raise ValueError(f"{path} missing IHDR chunk")
        width = int.from_bytes(handle.read(4), "big")
        height = int.from_bytes(handle.read(4), "big")
    return width, height


def _world_to_pixel(meta: Mapping[str, Any], x: float, y: float) -> tuple[int, int]:
    u = int(round((float(x) - float(meta["left"])) / float(meta["scale"])))
    v = int(round((float(meta["top"]) - float(y)) / float(meta["scale"])))
    return u, v


def _trajectory_world_points(trajectory: Sequence[Any]) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for item in trajectory:
        if not isinstance(item, Mapping):
            continue
        raw_position = item.get("position")
        if not isinstance(raw_position, Sequence) or len(raw_position) < 2:
            continue
        x = float(raw_position[0])
        y = float(raw_position[1])
        z = float(raw_position[2]) if len(raw_position) > 2 else 0.0
        if points and math.dist(points[-1][:2], (x, y)) < 1e-4:
            continue
        points.append((x, y, z))
    return points


def _job_sensor_details(manifest: Mapping[str, Any], job: Mapping[str, Any]) -> list[JsonDict]:
    rigs = manifest.get("sensor_rigs", {})
    out: list[JsonDict] = []
    for selected in job.get("sensors", []):
        if not isinstance(selected, Mapping):
            continue
        rig_id = str(selected.get("rig_id") or "")
        sensor_name = str(selected.get("sensor_name") or "")
        rig = rigs.get(rig_id) if isinstance(rigs, Mapping) else None
        if not isinstance(rig, Mapping):
            out.append({"sensor_name": sensor_name, "rig_id": rig_id})
            continue
        found = None
        for sensor in rig.get("sensors", []):
            if isinstance(sensor, Mapping) and str(sensor.get("name")) == sensor_name:
                found = dict(sensor)
                break
        if found is None:
            out.append({"sensor_name": sensor_name, "rig_id": rig_id})
        else:
            found["sensor_name"] = sensor_name
            found["rig_id"] = rig_id
            out.append(found)
    return out


def _sensor_resolution(sensor: Mapping[str, Any]) -> tuple[int, int]:
    intrinsics = sensor.get("intrinsics", {})
    if isinstance(intrinsics, Mapping):
        width = intrinsics.get("width")
        height = intrinsics.get("height")
        if width and height:
            return int(width), int(height)
    return 960, 720


def _sensor_fov_y_deg(sensor: Mapping[str, Any]) -> float:
    intrinsics = sensor.get("intrinsics", {})
    if not isinstance(intrinsics, Mapping):
        return 70.0
    value = intrinsics.get("fov_y_deg")
    if value is not None:
        return float(value)
    height = float(intrinsics.get("height") or 720)
    fy = intrinsics.get("fy")
    if fy:
        return math.degrees(2.0 * math.atan(height / (2.0 * float(fy))))
    return 70.0


def _sensor_clipping(sensor: Mapping[str, Any]) -> tuple[float, float]:
    clipping = sensor.get("clipping_range_m")
    if isinstance(clipping, Sequence) and len(clipping) >= 2:
        return float(clipping[0]), float(clipping[1])
    return 0.001, 30.0


def _write_actor_plan(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _human_actor_plan_bundle(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    human_ids: Sequence[str],
    *,
    base_dir: Path,
    actor_plan_path: Path,
    frame_count_hint: int,
) -> tuple[JsonDict | None, Path | None, list[str], list[str]]:
    if not human_ids:
        return None, None, [], []

    blockers: list[str] = []
    warnings: list[str] = []
    humans = manifest.get("actors", {}).get("humans", [])
    human_by_id = {
        str(human.get("actor_id")): human
        for human in humans
        if isinstance(human, Mapping) and human.get("actor_id")
    }

    camera_times = _trajectory_times(job.get("camera", {}).get("trajectory", []))
    if not camera_times:
        camera_times = [float(index) for index in range(max(1, int(frame_count_hint)))]
    actor_plans: list[JsonDict] = []
    for human_id in human_ids:
        human_id = str(human_id)
        human = human_by_id.get(human_id)
        if human is None:
            blockers.append(f"job references unknown human actor {human_id}")
            continue
        actor_payloads, actor_blockers, actor_warnings = _human_actor_plans(
            manifest,
            job,
            human,
            human_id,
            base_dir=base_dir,
            camera_times=camera_times,
        )
        blockers.extend(actor_blockers)
        warnings.extend(actor_warnings)
        actor_plans.extend(actor_payloads)

    if blockers:
        return None, None, blockers, warnings
    if not actor_plans:
        return None, None, ["job has no renderer-ready human actor plans"], warnings

    payload: JsonDict = {
        "schema_version": "massgen_actor_bundle.v1",
        "job_id": str(job.get("job_id") or ""),
        "scene_id": str(job.get("scene_id") or manifest.get("source", {}).get("scene_id") or ""),
        "source": "massgen_render_executor",
        "actors": actor_plans,
    }
    return payload, actor_plan_path, blockers, warnings


def _human_actor_plans(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    human: Mapping[str, Any],
    human_id: str,
    *,
    base_dir: Path,
    camera_times: Sequence[float],
) -> tuple[list[JsonDict], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    segments = [
        segment
        for segment in human.get("action_segments", [])
        if isinstance(segment, Mapping)
    ]
    if not segments:
        return [], [f"human {human_id} has no action segments"], warnings

    human_frames = _sample_human_motion_frames(human, camera_times)
    if not human_frames:
        return [], [f"human {human_id} has no usable trajectory or start_pose"], warnings

    bounds = human.get("visibility_bounds", {})
    actor_height_m = 1.7
    if isinstance(bounds, Mapping) and bounds.get("height_m") is not None:
        actor_height_m = float(bounds.get("height_m") or actor_height_m)

    actor_plans: list[JsonDict] = []
    manifest_fps = float(manifest.get("timing", {}).get("fps", 10.0) or 10.0)
    usable_segment_count = 0
    for index, segment in enumerate(segments):
        asset = segment.get("asset", {})
        if not isinstance(asset, Mapping):
            blockers.append(f"human {human_id} action segment {index} has no asset object")
            continue
        action_id = str(segment.get("render_action_id", "unknown"))
        if asset.get("requires_generation"):
            blockers.append(f"human {human_id} action {action_id!r} requires generation before rendering")
            continue
        ply_frame_dir = asset.get("ply_frame_dir")
        if not isinstance(ply_frame_dir, str) or not ply_frame_dir:
            blockers.append(f"human {human_id} action segment {index} has no ply_frame_dir")
            continue
        sequence_dir = _resolve_path(ply_frame_dir, base_dir=base_dir)
        if not sequence_dir.is_dir():
            blockers.append(f"human {human_id} action PLY directory does not exist: {sequence_dir}")
            continue
        usable_segment_count += 1
        segment_fps = manifest_fps
        if asset.get("fps") is not None:
            segment_fps = float(asset.get("fps") or segment_fps)
        start_time_s = _optional_float(segment.get("start_time_s"))
        end_time_s = _optional_float(segment.get("end_time_s"))
        segment_frames = _frames_with_segment_activity(
            human_frames,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            is_only_segment=len(segments) == 1,
        )
        payload: JsonDict = {
            "schema_version": "massgen_actor_plan.v1",
            "actor_id": human_id,
            "track_id": f"{human_id}__segment_{index:03d}",
            "job_id": str(job.get("job_id") or ""),
            "scene_id": str(job.get("scene_id") or manifest.get("source", {}).get("scene_id") or ""),
            "sequence_dir": str(sequence_dir),
            "actor_height_m": actor_height_m,
            "actor_fps": segment_fps,
            "loop": bool(asset.get("loop", True)),
            "z_mode": "floor",
            "yaw_offset_rad": 0.0,
            "source": "massgen_render_executor",
            "action": {
                "render_action_id": segment.get("render_action_id"),
                "action_sequence_id": segment.get("action_sequence_id"),
                "start_time_s": segment.get("start_time_s"),
                "end_time_s": segment.get("end_time_s"),
            },
            "frames": segment_frames,
        }
        actor_plans.append(payload)

    if blockers:
        return [], blockers, warnings
    if usable_segment_count == 0 or not actor_plans:
        return [], [f"human {human_id} has no renderer-ready action segment"], warnings
    if usable_segment_count > 1:
        warnings.append(
            f"human {human_id} uses {usable_segment_count} renderer action segments; inactive frames are skipped"
        )
    return actor_plans, blockers, warnings


def _trajectory_times(trajectory: Any) -> list[float]:
    if not isinstance(trajectory, Sequence):
        return []
    times: list[float] = []
    for index, item in enumerate(trajectory):
        if not isinstance(item, Mapping):
            continue
        value = item.get("time_s", item.get("t"))
        times.append(float(value) if value is not None else float(index))
    return times


def _sample_human_motion_frames(human: Mapping[str, Any], times: Sequence[float]) -> list[JsonDict]:
    samples = _human_motion_samples(human)
    if not samples:
        return []
    return [
        {
            "frame": int(index),
            "time_s": float(time_s),
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "yaw_rad": float(yaw_rad),
        }
        for index, time_s in enumerate(times)
        for position, yaw_rad in [_interpolate_human_motion(samples, float(time_s))]
    ]


def _frames_with_segment_activity(
    frames: Sequence[Mapping[str, Any]],
    *,
    start_time_s: float | None,
    end_time_s: float | None,
    is_only_segment: bool,
) -> list[JsonDict]:
    tagged: list[JsonDict] = []
    for frame in frames:
        item = dict(frame)
        if is_only_segment:
            item["active"] = True
        else:
            time_s = float(item.get("time_s", item.get("frame", 0.0)) or 0.0)
            if start_time_s is not None and time_s < start_time_s:
                item["active"] = False
            elif end_time_s is not None and time_s > end_time_s:
                item["active"] = False
            else:
                item["active"] = True
        tagged.append(item)
    return tagged


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _human_motion_samples(human: Mapping[str, Any]) -> list[tuple[float, tuple[float, float, float], float]]:
    samples: list[tuple[float, tuple[float, float, float], float]] = []
    trajectory = human.get("trajectory", [])
    if isinstance(trajectory, Sequence):
        for index, raw in enumerate(trajectory):
            if not isinstance(raw, Mapping):
                continue
            raw_position = raw.get("position")
            if not isinstance(raw_position, Sequence) or len(raw_position) < 2:
                continue
            x = float(raw_position[0])
            y = float(raw_position[1])
            z = float(raw_position[2]) if len(raw_position) > 2 else 0.0
            time_s = float(raw.get("time_s", raw.get("t", index)) or 0.0)
            yaw = float(raw.get("yaw_rad", 0.0) or 0.0)
            samples.append((time_s, (x, y, z), yaw))
    if not samples:
        start_pose = human.get("start_pose")
        if isinstance(start_pose, Mapping):
            samples.append(
                (
                    0.0,
                    (
                        float(start_pose.get("x", 0.0) or 0.0),
                        float(start_pose.get("y", 0.0) or 0.0),
                        0.0,
                    ),
                    float(start_pose.get("yaw_rad", 0.0) or 0.0),
                )
            )
    return sorted(samples, key=lambda item: item[0])


def _interpolate_human_motion(
    samples: Sequence[tuple[float, tuple[float, float, float], float]],
    time_s: float,
) -> tuple[tuple[float, float, float], float]:
    if len(samples) == 1 or time_s <= samples[0][0]:
        return samples[0][1], _actor_yaw_for_sample(samples, 0)
    if time_s >= samples[-1][0]:
        return samples[-1][1], _actor_yaw_for_sample(samples, len(samples) - 1)
    for index in range(len(samples) - 1):
        t0, p0, _ = samples[index]
        t1, p1, _ = samples[index + 1]
        if t0 <= time_s <= t1:
            span = max(t1 - t0, 1e-6)
            alpha = (time_s - t0) / span
            position = tuple(float(p0[axis] + (p1[axis] - p0[axis]) * alpha) for axis in range(3))
            direction = (p1[0] - p0[0], p1[1] - p0[1])
            if math.hypot(direction[0], direction[1]) > 1e-4:
                return position, math.atan2(direction[0], direction[1]) + math.pi
            return position, _actor_yaw_for_sample(samples, index)
    return samples[-1][1], _actor_yaw_for_sample(samples, len(samples) - 1)


def _actor_yaw_for_sample(
    samples: Sequence[tuple[float, tuple[float, float, float], float]],
    index: int,
) -> float:
    current = samples[index][1]
    if index + 1 < len(samples):
        nxt = samples[index + 1][1]
        dx, dy = nxt[0] - current[0], nxt[1] - current[1]
        if math.hypot(dx, dy) > 1e-4:
            return math.atan2(dx, dy) + math.pi
    if index > 0:
        prev = samples[index - 1][1]
        dx, dy = current[0] - prev[0], current[1] - prev[1]
        if math.hypot(dx, dy) > 1e-4:
            return math.atan2(dx, dy) + math.pi
    return float(samples[index][2]) + math.pi


def _float_arg(value: Any) -> str:
    return f"{float(value):.8g}"
