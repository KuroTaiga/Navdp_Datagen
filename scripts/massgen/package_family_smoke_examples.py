#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.massgen_render_manifest import scenario_file_to_render_manifest, write_json  # noqa: E402
from navdp_datagen.sensors import pinhole_from_fov_y  # noqa: E402


DEFAULT_SOURCE_ROOT = (
    Path("/Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner")
    / "tmp"
    / "chingmu3_0011_859081_all_missions_500_v1"
)
DEFAULT_VIS_ROOT = (
    Path("/Users/dongjk/ProjectFiles/Navdp_Datagen_Pathplanner")
    / "tmp"
    / "chingmu3_0011_859081_all_missions_500_v1_visual"
    / "bev"
)
DEFAULT_ACTOR_DIR = (
    "/home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting/"
    "debug_npc_ply/0001_839920/73"
)
DEFAULT_HUMAN_GS_SOURCE_ROOT = (
    "/home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting/"
    "data/human_gs_source"
)
DEFAULT_REMOTE_SCENE_PLY = "/mnt/DATA/dongjk/navdp_data/scenes/0030_839913/3dgs_compressed.ply"
PLANNER_VISUAL_SUFFIXES = (".png", ".gif")

FAMILY_SOURCES = {
    "deliver_to_human": ("deliver_to_human", "deliver_to_human"),
    "serve_queue": ("serve_queue", "serve_queue"),
    "human_guided_uncertain_region": (
        "human_guided_uncertain_region",
        "human_guided_uncertain_region",
    ),
    "dense_dynamic_humans": ("dense_dynamic_humans", "dense_dynamic_humans"),
    "dense_dynamic_avoidance": ("dense_dynamic_avoidance", "dense_dynamic_avoidance"),
    "personal_space": (
        "navigate_with_social_constraints/personal_space_L1",
        "navigate_with_social_constraints:personal_space",
    ),
    "queue_order": (
        "navigate_with_social_constraints/queue_order_L4",
        "navigate_with_social_constraints:queue_order",
    ),
    "group_integrity": (
        "navigate_with_social_constraints/group_integrity_L3",
        "navigate_with_social_constraints:group_integrity",
    ),
    "pedestrian_yield": (
        "navigate_with_social_constraints/pedestrian_yield_L2",
        "navigate_with_social_constraints:pedestrian_yield",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package one human-renderable MassGen example per mission family."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VIS_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action-catalog-json", type=Path, default=None)
    parser.add_argument("--actor-ply-frame-dir", default=DEFAULT_ACTOR_DIR)
    parser.add_argument(
        "--actor-source-root",
        default=None,
        help=(
            "Optional root whose child directories are stable human identities. "
            "Each mission human is assigned one child directory and keeps it for all actions."
        ),
    )
    parser.add_argument(
        "--actor-source-id",
        action="append",
        default=None,
        help=(
            "Human identity subdirectory under --actor-source-root. Repeat to use a remote "
            "root that is not visible on this machine."
        ),
    )
    parser.add_argument("--remote-scene-id", default="0030_839913")
    parser.add_argument("--remote-scene-ply", default=DEFAULT_REMOTE_SCENE_PLY)
    parser.add_argument(
        "--preserve-scene",
        action="store_true",
        help="Keep each scenario scene_id and only inject a splat path if missing.",
    )
    parser.add_argument(
        "--no-thin-trajectories",
        action="store_true",
        help="Keep full robot and human trajectories instead of reducing them to six points.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--scenario-override",
        action="append",
        default=None,
        metavar="FAMILY=PATH",
        help="Override the selected source scenario for a family key. Repeat as needed.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_scenario(source_root: Path, source_rel: str, *, visual_root: Path | None = None) -> Path:
    json_dir = source_root / source_rel / "jsons"
    candidates = sorted(path for path in json_dir.glob("*.json") if not path.name.endswith("_cornercase_metadata.json"))
    if not candidates:
        raise FileNotFoundError(f"No scenario JSON files found in {json_dir}")
    if visual_root is not None:
        by_stem = {path.stem: path for path in candidates}
        visual_dir = visual_root / source_rel / "visualizations"
        visual_stems = sorted(
            {
                path.name.removesuffix("_bev_trajectory.png")
                for path in visual_dir.glob("*_bev_trajectory.png")
                if (visual_dir / f"{path.name.removesuffix('_bev_trajectory.png')}_bev_trajectory.gif").is_file()
            }
        )
        for stem in visual_stems:
            if stem in by_stem:
                return by_stem[stem]
    return candidates[0]


def _scenario_overrides(raw_items: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for raw in raw_items or []:
        if "=" not in raw:
            raise ValueError(f"--scenario-override must be FAMILY=PATH, got: {raw}")
        family_key, path = raw.split("=", 1)
        family_key = family_key.strip()
        path = path.strip()
        if not family_key or not path:
            raise ValueError(f"--scenario-override must be FAMILY=PATH, got: {raw}")
        overrides[family_key] = Path(path).expanduser()
    return overrides


def _matching_visuals(visual_root: Path, source_rel: str, scenario_stem: str) -> list[Path]:
    vis_dir = visual_root / source_rel / "visualizations"
    paths: list[Path] = []
    for suffix in PLANNER_VISUAL_SUFFIXES:
        path = vis_dir / f"{scenario_stem}_bev_trajectory{suffix}"
        if path.is_file():
            paths.append(path)
    return paths


def _scenario_times(scenario: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for robot in scenario.get("robots", []):
        if not isinstance(robot, Mapping):
            continue
        for point in robot.get("trajectory", []):
            if isinstance(point, Mapping) and point.get("t") is not None:
                values.append(float(point.get("t") or 0.0))
    for mission in scenario.get("missions", []):
        if isinstance(mission, Mapping):
            for key in ("release_time", "deadline"):
                if mission.get(key) is not None:
                    values.append(float(mission.get(key) or 0.0))
    return values


def _thin_trajectory(points: list[Any], *, max_points: int = 6) -> list[Any]:
    if len(points) <= max_points:
        return points
    indices = sorted({round(index * (len(points) - 1) / (max_points - 1)) for index in range(max_points)})
    return [points[int(index)] for index in indices]


def _actor_identity_dirs(
    *,
    actor_source_root: str | None,
    actor_source_ids: list[str] | None,
    actor_ply_frame_dir: str,
) -> list[str]:
    if actor_source_root:
        root = Path(actor_source_root).expanduser()
        ids = [str(item).strip() for item in (actor_source_ids or []) if str(item).strip()]
        if ids:
            return [
                str(Path(item).expanduser()) if Path(item).is_absolute() else str(root / item)
                for item in ids
            ]
        if root.is_dir():
            dirs = [
                path
                for path in sorted(root.iterdir(), key=lambda item: _natural_identity_key(item.name))
                if path.is_dir() and any(path.glob("*.ply"))
            ]
            if dirs:
                return [str(path) for path in dirs]
        raise FileNotFoundError(
            "No actor identities found under "
            f"{root}; pass --actor-source-id for remote roots that are not locally mounted."
        )
    return [str(actor_ply_frame_dir)]


def _natural_identity_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _human_actor_assignment(humans: Any, actor_dirs: list[str]) -> dict[str, str]:
    if not actor_dirs:
        raise ValueError("At least one actor identity directory is required.")
    assignments: dict[str, str] = {}
    index = 0
    if not isinstance(humans, list):
        return assignments
    for human in humans:
        if not isinstance(human, Mapping):
            continue
        human_id = str(human.get("human_id") or human.get("actor_id") or f"human_{index:03d}")
        assignments[human_id] = actor_dirs[index % len(actor_dirs)]
        index += 1
    return assignments


def _scenario_for_smoke(
    scenario: Mapping[str, Any],
    *,
    scene_id: str | None,
    scene_ply: str | None,
    actor_ply_frame_dir: str,
    actor_ply_frame_dirs: list[str] | None = None,
    thin_trajectories: bool = True,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(scenario))
    if scene_id:
        payload["scene_id"] = scene_id
    scene_assets = dict(payload.get("scene_assets") or {})
    if scene_ply:
        scene_assets["splat_model_path"] = scene_ply
    payload["scene_assets"] = scene_assets
    for robot in payload.get("robots", []):
        if thin_trajectories and isinstance(robot, dict) and isinstance(robot.get("trajectory"), list):
            robot["trajectory"] = _thin_trajectory(robot["trajectory"])
    actor_dirs = actor_ply_frame_dirs or [actor_ply_frame_dir]
    actor_assignments = _human_actor_assignment(payload.get("humans", []), actor_dirs)
    for human in payload.get("humans", []):
        if thin_trajectories and isinstance(human, dict) and isinstance(human.get("trajectory"), list):
            human["trajectory"] = _thin_trajectory(human["trajectory"])
        if isinstance(human, dict):
            human_id = str(human.get("human_id") or human.get("actor_id") or "")
            assigned_actor_dir = actor_assignments.get(human_id, actor_ply_frame_dir)
            human.setdefault("metadata", {})
            if isinstance(human["metadata"], dict):
                human["metadata"]["renderer_actor_identity_dir"] = assigned_actor_dir
                human["metadata"]["renderer_actor_identity_id"] = Path(assigned_actor_dir).name
            for sequence in human.get("action_sequences", []):
                if not isinstance(sequence, dict):
                    continue
                sequence["ply_frame_dir"] = assigned_actor_dir
                sequence["manifest_path"] = None
                sequence["smplx_frame_dir"] = None
                sequence["pre_generated"] = True
                sequence["source"] = str(sequence.get("source") or "approved_action_codex")
                sequence["renderer_actor_identity_id"] = Path(assigned_actor_dir).name
    return payload


def _action_catalog_for_actor(path: Path, *, actor_ply_frame_dir: str) -> Path:
    actions = []
    for action_id, root_motion_mode in (
        ("receive_item", "stationary"),
        ("stand", "stationary"),
        ("wave", "stationary"),
        ("walk", "follow_map_path"),
        ("yield_stop", "stationary"),
        ("queue_wait", "stationary"),
    ):
        actions.append(
            {
                "action_id": action_id,
                "default_ply_frame_dir": actor_ply_frame_dir,
                "pre_generated": True,
                "loop": True,
                "root_motion_mode": root_motion_mode,
            }
        )
    payload = {"actions": actions}
    _write_json(path, payload)
    return path


def _scene_extent_from_scenario(scenario: Mapping[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    def add_pose(raw: Any) -> None:
        if not isinstance(raw, Mapping):
            return
        if raw.get("x") is not None and raw.get("y") is not None:
            xs.append(float(raw["x"]))
            ys.append(float(raw["y"]))

    for robot in scenario.get("robots", []):
        if not isinstance(robot, Mapping):
            continue
        add_pose(robot.get("start_map_pose"))
        for point in robot.get("trajectory", []):
            if isinstance(point, Mapping):
                add_pose(point.get("map_pose"))
    for human in scenario.get("humans", []):
        if not isinstance(human, Mapping):
            continue
        add_pose(human.get("start_map_pose"))
        for point in human.get("trajectory", []):
            if isinstance(point, Mapping):
                add_pose(point.get("map_pose"))
    if not xs or not ys:
        return -10.0, -10.0, 10.0, 10.0
    pad = 2.0
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _pose_xy(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, Mapping):
        return None
    if raw.get("x") is None or raw.get("y") is None:
        return None
    return float(raw["x"]), float(raw["y"])


def _draw_line(draw: Any, points: list[tuple[float, float]], color: tuple[int, int, int], width: int) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)
    for x, y in points:
        r = max(2, width + 1)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _generate_bev(scenario: Mapping[str, Any], out_path: Path, *, title: str) -> None:
    from PIL import Image, ImageDraw

    width = 900
    height = 700
    min_x, min_y, max_x, max_y = _scene_extent_from_scenario(scenario)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    def project(x: float, y: float) -> tuple[float, float]:
        px = 45.0 + ((x - min_x) / span_x) * (width - 90.0)
        py = height - 45.0 - ((y - min_y) / span_y) * (height - 90.0)
        return px, py

    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    for gx in range(0, width, 75):
        draw.line((gx, 0, gx, height), fill=(225, 225, 220), width=1)
    for gy in range(0, height, 75):
        draw.line((0, gy, width, gy), fill=(225, 225, 220), width=1)
    draw.rectangle((40, 40, width - 40, height - 40), outline=(70, 70, 70), width=2)
    draw.text((48, 14), title, fill=(20, 20, 20))

    colors = [
        (37, 99, 235),
        (220, 38, 38),
        (5, 150, 105),
        (147, 51, 234),
        (234, 88, 12),
        (8, 145, 178),
        (190, 18, 60),
    ]
    for index, robot in enumerate(scenario.get("robots", [])):
        if not isinstance(robot, Mapping):
            continue
        points: list[tuple[float, float]] = []
        for point in robot.get("trajectory", []):
            if isinstance(point, Mapping):
                xy = _pose_xy(point.get("map_pose"))
                if xy is not None:
                    points.append(project(*xy))
        _draw_line(draw, points, colors[index % len(colors)], 4)
        if points:
            draw.text((points[0][0] + 6, points[0][1] + 6), str(robot.get("robot_id", "robot")), fill=colors[index % len(colors)])

    for index, human in enumerate(scenario.get("humans", [])):
        if not isinstance(human, Mapping):
            continue
        points = []
        for point in human.get("trajectory", []):
            if isinstance(point, Mapping):
                xy = _pose_xy(point.get("map_pose"))
                if xy is not None:
                    points.append(project(*xy))
        color = colors[(index + 2) % len(colors)]
        _draw_line(draw, points, color, 3)
        if points:
            draw.text((points[0][0] + 6, points[0][1] - 16), str(human.get("human_id", "human")), fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _prune_peer_robots_for_human_only(manifest: dict[str, Any]) -> None:
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            continue
        job["peer_robot_ids"] = []
        job["peer_robot_pose_tracks"] = []


def _attach_smoke_sensor(manifest: dict[str, Any], *, width: int = 320, height: int = 240) -> None:
    rig_id = "massgen_smoke_fpv_320x240"
    sensor_name = "fpv_rgb"
    manifest["sensor_rigs"] = {
        rig_id: {
            "rig_id": rig_id,
            "profile": rig_id,
            "source": {
                "kind": "massgen_family_smoke_package",
                "provisional": False,
            },
            "sensors": [
                {
                    "name": sensor_name,
                    "type": "camera",
                    "profile": rig_id,
                    "enabled": True,
                    "frame": "robot_base",
                    "transform": {
                        "translation_m": [0.0, 0.0, 0.3],
                        "rotation_rpy_deg": [0.0, 0.0, 0.0],
                        "convention": "+X forward, +Y left, +Z up",
                    },
                    "intrinsics": pinhole_from_fov_y(width, height, 70.0),
                    "clipping_range_m": [0.001, 30.0],
                    "rate_hz": 10.0,
                    "modalities": ["rgb", "camera_metadata"],
                    "notes": "Low-resolution smoke camera for 5880 family rollout.",
                }
            ],
        }
    }
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            continue
        job["sensors"] = [
            {
                "rig_id": rig_id,
                "sensor_name": sensor_name,
                "type": "camera",
                "modalities": ["rgb", "camera_metadata"],
                "profile": rig_id,
            }
        ]


def _copy_or_generate_visual(
    *,
    visual_root: Path,
    source_rel: str,
    scenario: Mapping[str, Any],
    scenario_stem: str,
    family_dir: Path,
    family_key: str,
) -> dict[str, Any]:
    copied = _matching_visuals(visual_root, source_rel, scenario_stem)
    generated_path = family_dir / "example_visualization.png"
    visual_paths: list[str] = []
    if copied:
        for source_path in copied:
            copied_path = family_dir / source_path.name
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, copied_path)
            visual_paths.append(str(copied_path))
        return {
            "primary": str(next((family_dir / path.name for path in copied if path.suffix.lower() == ".png"), family_dir / copied[0].name)),
            "paths": sorted(visual_paths),
            "source": "planner_exact",
        }
    _generate_bev(scenario, generated_path, title=family_key)
    return {
        "primary": str(generated_path),
        "paths": [str(generated_path)],
        "source": "generated_from_scenario",
    }


def main() -> int:
    args = _parse_args()
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    scenario_overrides = _scenario_overrides(args.scenario_override)
    actor_identity_dirs = _actor_identity_dirs(
        actor_source_root=args.actor_source_root,
        actor_source_ids=args.actor_source_id,
        actor_ply_frame_dir=str(args.actor_ply_frame_dir),
    )
    action_catalog = _action_catalog_for_actor(
        out_root / "action_catalog_5880_avatar.json",
        actor_ply_frame_dir=actor_identity_dirs[0],
    )
    index: dict[str, Any] = {
        "source_root": str(args.source_root),
        "visual_root": str(args.visual_root),
        "output_root": str(out_root),
        "remote_scene_id": None if args.preserve_scene else str(args.remote_scene_id),
        "remote_scene_ply": str(args.remote_scene_ply) if args.remote_scene_ply else None,
        "preserve_scene": bool(args.preserve_scene),
        "thin_trajectories": not bool(args.no_thin_trajectories),
        "actor_source_root": str(args.actor_source_root) if args.actor_source_root else None,
        "actor_identity_dirs": actor_identity_dirs,
        "action_catalog_json": str(action_catalog),
        "families": [],
    }
    for family_key, (source_rel, render_family) in FAMILY_SOURCES.items():
        scenario_path = scenario_overrides.get(family_key)
        if scenario_path is None:
            scenario_path = _candidate_scenario(args.source_root, source_rel, visual_root=args.visual_root)
        if not scenario_path.is_file():
            raise FileNotFoundError(f"Scenario override for {family_key} not found: {scenario_path}")
        scenario = _load_json(scenario_path)
        smoke_scenario = _scenario_for_smoke(
            scenario,
            scene_id=None if args.preserve_scene else str(args.remote_scene_id),
            scene_ply=str(args.remote_scene_ply) if args.remote_scene_ply else None,
            actor_ply_frame_dir=actor_identity_dirs[0],
            actor_ply_frame_dirs=actor_identity_dirs,
            thin_trajectories=not bool(args.no_thin_trajectories),
        )
        family_dir = out_root / family_key
        family_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scenario_path, family_dir / "example_original.json")
        _write_json(family_dir / "example_smoke.json", smoke_scenario)
        manifest = scenario_file_to_render_manifest(
            family_dir / "example_smoke.json",
            action_catalog_path=action_catalog,
        )
        if "dense_dynamic_combined" not in render_family and "dense_multi_robot" not in render_family:
            _prune_peer_robots_for_human_only(manifest)
        _attach_smoke_sensor(manifest)
        write_json(family_dir / "render_manifest.json", manifest)
        visual_artifacts = _copy_or_generate_visual(
            visual_root=args.visual_root,
            source_rel=source_rel,
            scenario=scenario,
            scenario_stem=scenario_path.stem,
            family_dir=family_dir,
            family_key=family_key,
        )
        family_info = {
            "family_key": family_key,
            "render_family": render_family,
            "source_rel": source_rel,
            "source_scenario_json": str(scenario_path),
            "example_original_json": str(family_dir / "example_original.json"),
            "example_smoke_json": str(family_dir / "example_smoke.json"),
            "render_manifest_json": str(family_dir / "render_manifest.json"),
            "example_visualization": visual_artifacts["primary"],
            "example_visual_artifacts": visual_artifacts["paths"],
            "example_visual_source": visual_artifacts["source"],
            "scenario_id": scenario.get("scenario_id"),
            "mission_families": manifest.get("mission_families", []),
            "job_count": len(manifest.get("jobs", [])),
            "human_count": len(manifest.get("actors", {}).get("humans", [])),
            "human_actor_identity_dirs": [
                human.get("metadata", {}).get("renderer_actor_identity_dir")
                for human in smoke_scenario.get("humans", [])
                if isinstance(human, Mapping)
            ],
            "robot_count": len(manifest.get("actors", {}).get("robots", [])),
        }
        _write_json(family_dir / "family_package.json", family_info)
        index["families"].append(family_info)
        if len(index["families"]) >= int(args.limit) * len(FAMILY_SOURCES):
            break
    _write_json(out_root / "family_index.json", index)
    print(f"Wrote {len(index['families'])} family package(s) to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
