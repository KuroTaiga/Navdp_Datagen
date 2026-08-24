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
from scripts.render.assets.retarget_smplx_kimodo_to_g1 import (
    load_smplx_frame_paths,
    retarget_smplx_frames,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDER_SCRIPT = REPO_ROOT / "render_label_paths_telesim.py"
DEFAULT_VIDEO_BACKEND = "nvenc"
DEFAULT_DEVICE = "cuda"
DEFAULT_ROBOT_OVERLAY_SCRIPT = REPO_ROOT / "scripts" / "render" / "assets" / "render_glb_robot_overlay.py"
DEFAULT_ROBOT_GLB = REPO_ROOT / "assets" / "robots" / "g1_29dof_mode_16.glb"
DEFAULT_ROBOT_URDF = REPO_ROOT / "data" / "g1_description" / "g1_29dof_mode_16.urdf"
DEFAULT_KIMODO_SMPLX_DIR = REPO_ROOT / "assets" / "walking_kimodo"


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
    robot_overlay_script: str | Path = DEFAULT_ROBOT_OVERLAY_SCRIPT,
    robot_glb: str | Path = DEFAULT_ROBOT_GLB,
    robot_urdf: str | Path = DEFAULT_ROBOT_URDF,
    kimodo_smplx_dir: str | Path = DEFAULT_KIMODO_SMPLX_DIR,
    robot_compose_mode: str = "depth",
    robot_glb_up_axis: str = "z",
    robot_target_height: float | None = None,
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
            robot_overlay_script=Path(robot_overlay_script).expanduser().resolve(),
            robot_glb=Path(robot_glb).expanduser().resolve(),
            robot_urdf=Path(robot_urdf).expanduser().resolve(),
            kimodo_smplx_dir=Path(kimodo_smplx_dir).expanduser().resolve(),
            robot_compose_mode=robot_compose_mode,
            robot_glb_up_axis=robot_glb_up_axis,
            robot_target_height=robot_target_height,
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
        overlay_input_error = _validate_robot_overlay_base_inputs(plan)
        if overlay_input_error is not None:
            print(overlay_input_error, file=sys.stderr)
            return 3
        for overlay in plan.get("robot_overlay_commands", []):
            if not isinstance(overlay, Mapping):
                continue
            overlay_command = [str(item) for item in overlay.get("command", [])]
            if not overlay_command:
                continue
            completed = _run_overlay_command_with_retry(overlay_command, env=env, attempts=2)
            if completed.returncode != 0:
                return int(completed.returncode)
    return 0


def _run_overlay_command_with_retry(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    attempts: int,
) -> subprocess.CompletedProcess:
    last: subprocess.CompletedProcess | None = None
    for attempt in range(max(1, int(attempts))):
        last = subprocess.run(command, env=env, check=False)
        if last.returncode == 0:
            return last
        if attempt + 1 < attempts:
            print(
                f"robot overlay command failed with {last.returncode}; retrying once",
                file=sys.stderr,
            )
    assert last is not None
    return last


def _validate_robot_overlay_base_inputs(plan: Mapping[str, Any]) -> str | None:
    if not plan.get("robot_overlay_commands"):
        return None
    paths = plan.get("robot_overlay_paths", {})
    if not isinstance(paths, Mapping):
        return f"missing robot overlay paths for job {plan.get('job_id')}"
    camera_json = paths.get("camera_json")
    if isinstance(camera_json, str) and not Path(camera_json).is_file():
        return f"missing camera metadata before robot overlays for job {plan.get('job_id')}: {camera_json}"
    base_frames_dir = paths.get("base_frames_dir")
    if isinstance(base_frames_dir, str):
        base_path = Path(base_frames_dir)
        if not base_path.is_dir():
            return f"missing base RGB frame directory before robot overlays for job {plan.get('job_id')}: {base_frames_dir}"
        if not any(base_path.glob("frame_*.png")):
            return f"base RGB frame directory is empty before robot overlays for job {plan.get('job_id')}: {base_frames_dir}"
    return None


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
        for overlay in plan.get("robot_overlay_commands", []):
            if not isinstance(overlay, Mapping):
                continue
            overlay_command = shlex.join([str(item) for item in overlay.get("command", [])])
            lines.append(
                f"  robot_overlay[{overlay.get('actor_id')}]: {overlay_command}".rstrip()
            )
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
    robot_overlay_script: Path,
    robot_glb: Path,
    robot_urdf: Path,
    kimodo_smplx_dir: Path,
    robot_compose_mode: str,
    robot_glb_up_axis: str,
    robot_target_height: float | None,
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
    needs_robot_overlay = bool(job.get("peer_robot_ids") or job.get("peer_robot_pose_tracks"))
    effective_save_rgb_frames = bool(save_rgb_frames or needs_robot_overlay)
    effective_save_depth_maps = bool(save_depth_maps or (needs_robot_overlay and str(robot_compose_mode) == "depth"))
    if effective_save_depth_maps and "--save-depth-maps" not in command:
        command.append("--save-depth-maps")
    if effective_save_rgb_frames:
        command.append("--rgb-frames")
    if minimal_frames is not None and int(minimal_frames) > 0:
        command.extend(["--minimal-frames", str(int(minimal_frames))])
    human_ids = [str(item) for item in job.get("human_actor_ids", [])]
    robot_overlay_commands, robot_overlay_paths, robot_blockers, robot_warnings = _peer_robot_overlay_bundle(
        manifest,
        job,
        base_dir=manifest_base,
        render_output_root=render_output_root,
        tasks_root=tasks_root,
        scene_id=scene_id,
        label_id=job_id,
        python_bin=python_bin,
        robot_overlay_script=robot_overlay_script,
        robot_glb=robot_glb,
        robot_urdf=robot_urdf,
        kimodo_smplx_dir=kimodo_smplx_dir,
        robot_compose_mode=robot_compose_mode,
        robot_glb_up_axis=robot_glb_up_axis,
        robot_target_height=robot_target_height,
        depth_bit_depth=16,
        write_inputs=write_inputs,
        fps=fps,
        scene_dir=scene_dir,
        frame_count_hint=len(_trajectory_world_points(job.get("camera", {}).get("trajectory", []))),
    )
    blockers.extend(robot_blockers)
    warnings.extend(robot_warnings)
    actor_plan_payload, actor_plan_path, human_blockers, human_warnings = _human_actor_plan_bundle(
        manifest,
        job,
        human_ids,
        base_dir=manifest_base,
        actor_plan_path=tasks_root / scene_id / "actor_plans" / f"{job_id}.json",
        scene_dir=scene_dir,
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
        "robot_overlay_commands": robot_overlay_commands,
        "robot_overlay_paths": robot_overlay_paths,
        "blockers": blockers,
        "warnings": warnings,
        "metadata": {
            "current_executor_boundary": (
                "camera label materialization, human Gaussian actor bundles, "
                "and peer robot GLB overlay commands"
            ),
            "output_root": str(output_root),
        },
    }


def _peer_robot_overlay_bundle(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    base_dir: Path,
    render_output_root: Path,
    tasks_root: Path,
    scene_id: str,
    label_id: str,
    python_bin: str,
    robot_overlay_script: Path,
    robot_glb: Path,
    robot_urdf: Path,
    kimodo_smplx_dir: Path,
    robot_compose_mode: str,
    robot_glb_up_axis: str,
    robot_target_height: float | None,
    depth_bit_depth: int,
    write_inputs: bool,
    fps: float,
    scene_dir: Path,
    frame_count_hint: int,
) -> tuple[list[JsonDict], JsonDict, list[str], list[str]]:
    tracks = [
        track
        for track in job.get("peer_robot_pose_tracks", [])
        if isinstance(track, Mapping)
    ]
    if not tracks:
        return [], {}, [], []

    blockers: list[str] = []
    warnings: list[str] = []
    commands: list[JsonDict] = []
    track_outputs: list[JsonDict] = []
    if not robot_overlay_script.is_file():
        blockers.append(f"robot overlay script does not exist: {robot_overlay_script}")
    default_glb = _resolve_existing_or_candidate(robot_glb, base_dir=base_dir)
    default_urdf = _resolve_existing_or_candidate(robot_urdf, base_dir=base_dir)
    if not default_glb.is_file():
        blockers.append(f"default robot GLB does not exist: {default_glb}")
    if not default_urdf.is_file():
        blockers.append(f"default robot URDF does not exist: {default_urdf}")
    if not kimodo_smplx_dir.exists():
        blockers.append(f"Kimodo SMPL-X directory does not exist: {kimodo_smplx_dir}")

    try:
        coord_meta = _load_occupancy_metadata_for_label(scene_dir)
    except Exception as exc:  # pylint: disable=broad-except
        blockers.append(f"unable to load scene occupancy metadata for robot coordinates: {exc}")
        coord_meta = None

    camera_times = _trajectory_times(job.get("camera", {}).get("trajectory", []))
    if not camera_times:
        camera_times = [float(index) for index in range(max(1, int(frame_count_hint)))]

    base_frames_dir = render_output_root / scene_id / label_id
    previous_frames_dir = base_frames_dir
    camera_json = render_output_root / scene_id / f"{label_id}_camera.json"
    depth_dir = base_frames_dir
    overlay_root = render_output_root / scene_id / f"{label_id}__peer_robots"
    robot_inputs_root = tasks_root / scene_id / "robot_overlays" / label_id
    final_video = render_output_root / scene_id / f"{label_id}__with_peer_robots.mp4"

    source_frame_paths = None
    if not blockers:
        try:
            source_frame_paths = load_smplx_frame_paths(kimodo_smplx_dir)
        except Exception as exc:  # pylint: disable=broad-except
            blockers.append(f"unable to load Kimodo SMPL-X frames for robot AMO: {exc}")

    for index, track in enumerate(tracks):
        actor_id = str(track.get("actor_id") or f"peer_robot_{index:03d}")
        safe_actor_id = _safe_filename_token(actor_id)
        pose_json = robot_inputs_root / f"{safe_actor_id}_poses.json"
        amo_json = robot_inputs_root / f"{safe_actor_id}_g1_amo_from_kimodo.json"
        out_frames = overlay_root / f"{index:02d}_{safe_actor_id}"
        asset = track.get("asset", {})
        if isinstance(asset, Mapping):
            glb_path = _resolve_existing_or_candidate(
                _str_path_or_default(asset.get("glb_path"), default_glb),
                base_dir=base_dir,
            )
            urdf_path = _resolve_existing_or_candidate(
                _str_path_or_default(asset.get("urdf_path"), default_urdf),
                base_dir=base_dir,
            )
        else:
            glb_path = default_glb
            urdf_path = default_urdf
        if not glb_path.is_file():
            blockers.append(f"peer robot {actor_id} GLB does not exist: {glb_path}")
        if not urdf_path.is_file():
            blockers.append(f"peer robot {actor_id} URDF does not exist: {urdf_path}")

        if write_inputs and coord_meta is not None and source_frame_paths is not None:
            try:
                pose_payload = _robot_pose_payload(
                    manifest,
                    job,
                    track,
                    actor_id=actor_id,
                    camera_times=camera_times,
                    coord_meta=coord_meta,
                    frame_count_hint=frame_count_hint,
                )
                pose_json.parent.mkdir(parents=True, exist_ok=True)
                pose_json.write_text(
                    json.dumps(pose_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                amo_payload = retarget_smplx_frames(
                    source_frame_paths,
                    frame_count=len(pose_payload["frames"]),
                )
                amo_json.write_text(json.dumps(amo_payload, indent=2), encoding="utf-8")
            except Exception as exc:  # pylint: disable=broad-except
                blockers.append(f"unable to materialize peer robot {actor_id} overlay inputs: {exc}")
        else:
            warnings.append("robot overlay inputs not written; pass --write-inputs before --execute")

        command = [
            str(python_bin),
            str(robot_overlay_script),
            "--camera-json",
            str(camera_json),
            "--frames-dir",
            str(previous_frames_dir),
            "--robot-glb",
            str(glb_path),
            "--robot-urdf",
            str(urdf_path),
            "--robot-package-root",
            str(urdf_path.parent),
            "--poses-json",
            str(pose_json),
            "--amo-poses-json",
            str(amo_json),
            "--output-dir",
            str(out_frames),
            "--compose-mode",
            str(robot_compose_mode),
            "--depth-bit-depth",
            str(int(depth_bit_depth)),
            "--glb-up-axis",
            str(robot_glb_up_axis),
            "--fps",
            _float_arg(fps),
            "--overwrite",
        ]
        if str(robot_compose_mode) == "depth":
            command.extend(["--depth-dir", str(depth_dir)])
        if robot_target_height is not None:
            command.extend(["--target-height", _float_arg(robot_target_height)])
        if index == len(tracks) - 1:
            command.extend(["--video", str(final_video)])
        commands.append(
            {
                "actor_id": actor_id,
                "command": command,
                "input_frames_dir": str(previous_frames_dir),
                "output_frames_dir": str(out_frames),
                "video": str(final_video) if index == len(tracks) - 1 else None,
                "poses_json": str(pose_json),
                "amo_poses_json": str(amo_json),
                "robot_glb": str(glb_path),
                "robot_urdf": str(urdf_path),
            }
        )
        track_outputs.append(
            {
                "actor_id": actor_id,
                "poses_json": str(pose_json),
                "amo_poses_json": str(amo_json),
                "frames_dir": str(out_frames),
            }
        )
        previous_frames_dir = out_frames

    paths: JsonDict = {
        "camera_json": str(camera_json),
        "base_frames_dir": str(base_frames_dir),
        "overlay_root": str(overlay_root),
        "final_video": str(final_video),
        "tracks": track_outputs,
    }
    return commands, paths, blockers, warnings


def _str_path_or_default(value: Any, default: Path) -> Path:
    if isinstance(value, str) and value:
        return Path(value)
    return default


def _resolve_existing_or_candidate(path: Path, *, base_dir: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    for candidate in ((base_dir / expanded).resolve(), (REPO_ROOT / expanded).resolve()):
        if candidate.exists():
            return candidate
    return (base_dir / expanded).resolve()


def _safe_filename_token(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return out or "robot"


def _robot_pose_payload(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    track: Mapping[str, Any],
    *,
    actor_id: str,
    camera_times: Sequence[float],
    coord_meta: Mapping[str, Any],
    frame_count_hint: int,
) -> JsonDict:
    samples = _robot_motion_samples(track, coord_meta=coord_meta)
    if not samples:
        raise ValueError(f"peer robot {actor_id} has no usable trajectory")
    times = list(camera_times) or [float(index) for index in range(max(1, int(frame_count_hint)))]
    frames = [
        {
            "frame": int(index),
            "time_s": float(time_s),
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "yaw_rad": float(yaw_rad),
        }
        for index, time_s in enumerate(times)
        for position, yaw_rad in [_interpolate_robot_motion(samples, float(time_s))]
    ]
    return {
        "schema_version": "massgen_robot_overlay_poses.v1",
        "source": "massgen_render_executor",
        "job_id": str(job.get("job_id") or ""),
        "scene_id": str(job.get("scene_id") or manifest.get("source", {}).get("scene_id") or ""),
        "actor_id": actor_id,
        "coordinate_pipeline": _coordinate_pipeline_metadata(coord_meta)["label_path"],
        "frames": frames,
    }


def _robot_motion_samples(
    track: Mapping[str, Any],
    *,
    coord_meta: Mapping[str, Any],
) -> list[tuple[float, tuple[float, float, float], float]]:
    samples: list[tuple[float, tuple[float, float, float], float]] = []
    for index, raw in enumerate(track.get("trajectory", [])):
        if not isinstance(raw, Mapping):
            continue
        raw_position = raw.get("position")
        if not isinstance(raw_position, Sequence) or len(raw_position) < 2:
            continue
        x, y = _pathplanner_xy_for_telesim_label(
            coord_meta,
            float(raw_position[0]),
            float(raw_position[1]),
        )
        z = float(raw_position[2]) if len(raw_position) > 2 else float(coord_meta.get("lower_z", 0.0))
        time_s = float(raw.get("time_s", raw.get("t", index)) or 0.0)
        yaw_rad = float(raw.get("yaw_rad", 0.0) or 0.0)
        samples.append((time_s, (x, y, z), yaw_rad))
    return sorted(samples, key=lambda item: item[0])


def _interpolate_robot_motion(
    samples: Sequence[tuple[float, tuple[float, float, float], float]],
    time_s: float,
) -> tuple[tuple[float, float, float], float]:
    if len(samples) == 1 or time_s <= samples[0][0]:
        return samples[0][1], _robot_yaw_for_sample(samples, 0)
    if time_s >= samples[-1][0]:
        return samples[-1][1], _robot_yaw_for_sample(samples, len(samples) - 1)
    for index in range(len(samples) - 1):
        t0, p0, yaw0 = samples[index]
        t1, p1, yaw1 = samples[index + 1]
        if t0 <= time_s <= t1:
            span = max(t1 - t0, 1e-6)
            alpha = (time_s - t0) / span
            position = tuple(float(p0[axis] + (p1[axis] - p0[axis]) * alpha) for axis in range(3))
            direction = (p1[0] - p0[0], p1[1] - p0[1])
            if math.hypot(direction[0], direction[1]) > 1e-4:
                return position, math.atan2(direction[1], direction[0])
            return position, _lerp_angle(yaw0, yaw1, alpha)
    return samples[-1][1], _robot_yaw_for_sample(samples, len(samples) - 1)


def _robot_yaw_for_sample(
    samples: Sequence[tuple[float, tuple[float, float, float], float]],
    index: int,
) -> float:
    current = samples[index][1]
    if index + 1 < len(samples):
        nxt = samples[index + 1][1]
        dx, dy = nxt[0] - current[0], nxt[1] - current[1]
        if math.hypot(dx, dy) > 1e-4:
            return math.atan2(dy, dx)
    if index > 0:
        prev = samples[index - 1][1]
        dx, dy = current[0] - prev[0], current[1] - prev[1]
        if math.hypot(dx, dy) > 1e-4:
            return math.atan2(dy, dx)
    return float(samples[index][2])


def _lerp_angle(start: float, end: float, alpha: float) -> float:
    delta = _wrap_angle(float(end) - float(start))
    return _wrap_angle(float(start) + delta * float(alpha))


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
    raster_pixel = [list(_scene_xy_to_pixel(meta, float(x), float(y))) for x, y, _ in points]
    coordinate_pipeline = _coordinate_pipeline_metadata(meta)
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
            "coordinate_frame": "pathplanner_left_handed",
            "source_coordinate_frame": "pathplanner_left_handed",
            "coordinate_transform": "identity_xy",
            "coordinate_pipeline": coordinate_pipeline["label_path"],
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


def _pathplanner_xy_for_telesim_label(meta: Mapping[str, Any], x: float, y: float) -> tuple[float, float]:
    """Return MassGen scenario XY in the frame consumed by TeleSim label rendering.

    Existing CHINGMU Pathplanner scenario coordinates align with the BEV scene
    frame used by the label renderer. Mirroring them here moves paths to the
    opposite side of the occupancy map; the renderer already receives matching
    raster_pixel values for the unmirrored coordinates.
    """

    _ = meta
    return float(x), float(y)


def _coordinate_pipeline_metadata(meta: Mapping[str, Any]) -> JsonDict:
    return {
        "source_frame": "pathplanner_left_handed_map_pose",
        "target_frame": "telesim_scene_xy",
        "occupancy": {
            "left": float(meta["left"]),
            "right": float(meta["right"]),
            "top": float(meta["top"]),
            "bottom": float(meta["bottom"]),
            "scale": float(meta["scale"]),
            "width_px": int(meta["width"]),
            "height_px": int(meta["height"]),
        },
        "label_path": [
            {
                "stage": "massgen_render_manifest",
                "operation": "copy robot trajectory map_pose to camera trajectory position",
                "x": "position[0] = map_pose.x",
                "y": "position[1] = map_pose.y",
                "z": "position[2] = 0.0",
                "yaw": "yaw_rad = map_pose.yaw",
            },
            {
                "stage": "massgen_render_executor.label_path",
                "operation": "identity XY into raster_world",
                "x": "raster_world.x = position[0]",
                "y": "raster_world.y = position[1]",
                "z": "raster_world.z = position[2]",
                "reason": "CHINGMU Pathplanner scenario XY already aligns with TeleSim scene BEV.",
            },
            {
                "stage": "massgen_render_executor.label_path",
                "operation": "compute raster_pixel from occupancy metadata",
                "u": "round((raster_world.x - left) / scale)",
                "v": "round((top - raster_world.y) / scale)",
            },
            {
                "stage": "render_label_paths_telesim.prepare_path_data",
                "operation": "derive affine from raster_world to raster_pixel and render with --no-mirror-translation",
            },
        ],
        "actor_plan": [
            {
                "stage": "massgen_render_manifest",
                "operation": "copy human trajectory map_pose to actor trajectory position",
                "x": "position[0] = map_pose.x",
                "y": "position[1] = map_pose.y",
                "z": "position[2] = 0.0",
                "yaw": "yaw_rad = map_pose.yaw",
            },
            {
                "stage": "massgen_render_executor.actor_plan",
                "operation": "identity XY into actor frame positions",
                "x": "actor_frame.position[0] = position[0]",
                "y": "actor_frame.position[1] = position[1]",
                "z": "actor_frame.position[2] = position[2]",
                "reason": "Actor positions share the same TeleSim scene XY as camera label paths.",
            },
            {
                "stage": "massgen_render_executor.actor_plan",
                "operation": "convert Pathplanner yaw to aligned human PLY sample yaw",
                "yaw": "atan2(-cos(pathplanner_yaw), -sin(pathplanner_yaw))",
            },
            {
                "stage": "render_label_paths_telesim.build_actor_motion_plan_transforms",
                "operation": "apply actor position directly as world translation; z uses floor when z_mode=floor",
            },
        ],
    }


def _pathplanner_yaw_to_gs_actor_sample_yaw(yaw_rad: float) -> float:
    """Convert conventional Pathplanner yaw to the sample yaw used by human actor placement."""

    planner_yaw = float(yaw_rad)
    gs_dx = -math.cos(planner_yaw)
    gs_dy = -math.sin(planner_yaw)
    return math.atan2(gs_dx, gs_dy)


def _scene_xy_to_pixel(meta: Mapping[str, Any], x: float, y: float) -> tuple[int, int]:
    u = int(round((float(x) - float(meta["left"])) / float(meta["scale"])))
    v = int(round((float(meta["top"]) - float(y)) / float(meta["scale"])))
    return u, v


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


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
    scene_dir: Path,
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
    try:
        coord_meta = _load_occupancy_metadata_for_label(scene_dir)
    except Exception as exc:  # pylint: disable=broad-except
        return None, None, [f"unable to load scene occupancy metadata for actor coordinates: {exc}"], warnings
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
            coord_meta=coord_meta,
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
        "coordinate_pipeline": _coordinate_pipeline_metadata(coord_meta)["actor_plan"],
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
    coord_meta: Mapping[str, Any],
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

    human_frames = _sample_human_motion_frames(human, camera_times, coord_meta=coord_meta)
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
        root_motion_mode = str(asset.get("root_motion_mode") or "").strip().lower()
        freeze_animation_frame = root_motion_mode != "follow_map_path"
        start_time_s = _optional_float(segment.get("start_time_s"))
        end_time_s = _optional_float(segment.get("end_time_s"))
        segment_frames = _frames_with_segment_activity(
            human_frames,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            is_only_segment=len(segments) == 1,
            animation_frame_index=0 if freeze_animation_frame else None,
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
                "root_motion_mode": asset.get("root_motion_mode"),
                "animation_frame_policy": (
                    "first_frame_static" if freeze_animation_frame else "advance_sequence"
                ),
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


def _sample_human_motion_frames(
    human: Mapping[str, Any],
    times: Sequence[float],
    *,
    coord_meta: Mapping[str, Any],
) -> list[JsonDict]:
    samples = _human_motion_samples(human, coord_meta=coord_meta)
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
    animation_frame_index: int | None = None,
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
        if animation_frame_index is not None:
            item["animation_frame_index"] = int(animation_frame_index)
        tagged.append(item)
    return tagged


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _human_motion_samples(
    human: Mapping[str, Any],
    *,
    coord_meta: Mapping[str, Any],
) -> list[tuple[float, tuple[float, float, float], float]]:
    samples: list[tuple[float, tuple[float, float, float], float]] = []
    trajectory = human.get("trajectory", [])
    if isinstance(trajectory, Sequence):
        for index, raw in enumerate(trajectory):
            if not isinstance(raw, Mapping):
                continue
            raw_position = raw.get("position")
            if not isinstance(raw_position, Sequence) or len(raw_position) < 2:
                continue
            x, y = _pathplanner_xy_for_telesim_label(
                coord_meta,
                float(raw_position[0]),
                float(raw_position[1]),
            )
            z = float(raw_position[2]) if len(raw_position) > 2 else 0.0
            time_s = float(raw.get("time_s", raw.get("t", index)) or 0.0)
            yaw = _pathplanner_yaw_to_gs_actor_sample_yaw(float(raw.get("yaw_rad", 0.0) or 0.0))
            samples.append((time_s, (x, y, z), yaw))
    if not samples:
        start_pose = human.get("start_pose")
        if isinstance(start_pose, Mapping):
            samples.append(
                (
                    0.0,
                    (*_pathplanner_xy_for_telesim_label(
                        coord_meta,
                        float(start_pose.get("x", 0.0) or 0.0),
                        float(start_pose.get("y", 0.0) or 0.0),
                    ), 0.0),
                    _pathplanner_yaw_to_gs_actor_sample_yaw(float(start_pose.get("yaw_rad", 0.0) or 0.0)),
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
