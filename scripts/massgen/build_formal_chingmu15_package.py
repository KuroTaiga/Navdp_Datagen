#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.massgen.package_family_smoke_examples import (  # noqa: E402
    _action_catalog_for_actor,
    _attach_smoke_sensor,
    _prune_peer_robots_for_human_only,
    _scenario_for_smoke,
)
from utils.massgen_render_manifest import scenario_file_to_render_manifest, write_json  # noqa: E402


DEFAULT_GENERATION_ROOT = Path("/private_lxh/dongjk/navdata/mass_generation_runs/formal_fast_v1")
DEFAULT_REMOTE_SCENE_ROOT = "/mnt/DATA/dongjk/navdp_data"
DEFAULT_REMOTE_HUMAN_ROOT = "/mnt/DATA/dongjk/navdp_data/human_gs_source"
DEFAULT_MISSIONS_PER_FAMILY_SCENE = 100
DEFAULT_SEED = 20260825

RENDERABLE_FAMILIES = (
    "deliver_to_human",
    "dense_dynamic_avoidance",
    "dense_dynamic_humans",
    "human_guided_uncertain_region",
    "navigate_with_social_constraints:pedestrian_yield",
    "navigate_with_social_constraints:queue_order",
    "serve_queue",
)

DEFAULT_CHINGMU15_SCENES = {
    "CHINGMU_rescaled_1": (
        "0052_858885",
        "0034_858867",
        "0017_858849",
        "0073_858906",
        "0070_858903",
    ),
    "CHINGMU_rescaled_2": (
        "0032_858992",
        "0067_859028",
        "0005_858965",
        "0006_858966",
        "0062_859023",
    ),
    "CHINGMU_rescaled_3": (
        "0011_859081",
        "0004_859074",
        "0002_859072",
        "0010_859080",
        "0013_859083",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the formal H100 CHINGMU workload package from full MassGen "
            "generation outputs: 7 renderable families x 15 CHINGMU scenes x "
            "N missions per family/scene."
        )
    )
    parser.add_argument("--generation-root", type=Path, default=DEFAULT_GENERATION_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--missions-per-family-scene", type=int, default=DEFAULT_MISSIONS_PER_FAMILY_SCENE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--remote-scene-root", default=DEFAULT_REMOTE_SCENE_ROOT)
    parser.add_argument("--remote-human-root", default=DEFAULT_REMOTE_HUMAN_ROOT)
    parser.add_argument("--actor-source-id", action="append", default=None)
    parser.add_argument("--sensor-width", type=int, default=320)
    parser.add_argument("--sensor-height", type=int, default=240)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _safe_component(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("__")
    return "".join(safe).strip("_") or "unnamed"


def _sample_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256(("::".join([str(base_seed), *parts])).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _mission_jsons(json_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in json_dir.glob("*.json")
        if not path.name.endswith("_cornercase_metadata.json")
    )


def _scene_ply_path(remote_scene_root: str, source: str, scene: str) -> str:
    root = remote_scene_root.rstrip("/")
    if source == "InteriorGS":
        return f"{root}/scenes/{scene}/3dgs_compressed.ply"
    return f"{root}/{source}/{scene}/3dgs_raw.ply"


def _actor_dirs(remote_human_root: str, actor_source_ids: list[str] | None) -> list[str]:
    ids = [str(item).strip() for item in actor_source_ids or [] if str(item).strip()]
    if not ids:
        ids = [
            "7611",
            "1018",
            "10395",
            "10719",
            "10971",
            "11600",
            "11801",
            "12055",
        ]
    return [f"{remote_human_root.rstrip('/')}/{item}" for item in ids]


def _json_dir(scene_dir: Path, family: str) -> Path:
    candidates = [
        path
        for path in scene_dir.rglob("jsons")
        if path.is_dir()
        and any(child.suffix == ".json" and not child.name.endswith("_cornercase_metadata.json") for child in path.iterdir())
    ]
    if not candidates:
        raise FileNotFoundError(f"No mission jsons found under {scene_dir}")
    preferred = [path for path in candidates if family.split(":", 1)[0] in path.as_posix()]
    return sorted(preferred or candidates)[0]


def _copy_scene_metadata(src_scene_dir: Path, dst_scene_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for name in (
        "mass_example_manifest.json",
        "mass_generation_report.json",
        "mass_generation_report.md",
        "mass_generation_progress.json",
    ):
        src = src_scene_dir / name
        if src.is_file():
            dst = dst_scene_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _copy_selected_pair(
    *,
    mission_path: Path,
    local_json_dir: Path,
    package_json_dir: Path,
) -> tuple[Path, Path | None]:
    package_json_dir.mkdir(parents=True, exist_ok=True)
    mission_dst = package_json_dir / mission_path.name
    shutil.copy2(mission_path, mission_dst)
    metadata_src = local_json_dir / f"{mission_path.stem}_cornercase_metadata.json"
    metadata_dst = None
    if metadata_src.is_file():
        metadata_dst = package_json_dir / metadata_src.name
        shutil.copy2(metadata_src, metadata_dst)
    return mission_dst, metadata_dst


def main() -> int:
    args = _parse_args()
    generation_root = args.generation_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    missions_per_group = int(args.missions_per_family_scene)
    if missions_per_group <= 0:
        raise SystemExit("--missions-per-family-scene must be positive")
    if not generation_root.is_dir():
        raise SystemExit(f"missing generation root: {generation_root}")
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    if output_root.exists():
        raise SystemExit(f"output already exists, refusing to overwrite: {output_root}")
    output_root.mkdir(parents=True)

    actor_dirs = _actor_dirs(str(args.remote_human_root), args.actor_source_id)
    action_catalog = _action_catalog_for_actor(
        output_root / "action_catalog_5880_avatar.json",
        actor_ply_frame_dir=actor_dirs[0],
    )

    selected_root = output_root / f"selected_paths_{missions_per_group}_per_family_scene"
    scenario_root = output_root / "render_scenarios"
    manifest_root = output_root / "render_manifests"
    copied_scene_root = output_root / "scene_generation_metadata"

    index: dict[str, Any] = {
        "schema_version": "navdp_massgen_render_smoketest_package.v0.2",
        "source_root": str(generation_root),
        "remote_scene_root": str(args.remote_scene_root),
        "remote_human_root": str(args.remote_human_root),
        "actor_identity_dirs": actor_dirs,
        "action_catalog_json": _rel(action_catalog, output_root),
        "missions_per_family_scene": missions_per_group,
        "scenarios_per_scene": missions_per_group,
        "seed": int(args.seed),
        "sensor": {"width": int(args.sensor_width), "height": int(args.sensor_height)},
        "selected_scene_set": DEFAULT_CHINGMU15_SCENES,
        "families": list(RENDERABLE_FAMILIES),
        "entries": [],
        "renderer_limitations": [
            "dense_multi_robot and dense_dynamic_combined are excluded from this formal renderable workload.",
        ],
    }

    copied_metadata_cache: set[tuple[str, str, str]] = set()
    for family in RENDERABLE_FAMILIES:
        for source, scenes in DEFAULT_CHINGMU15_SCENES.items():
            for scene in scenes:
                src_scene_dir = generation_root / family / source / scene
                local_json_dir = _json_dir(src_scene_dir, family)
                missions = _mission_jsons(local_json_dir)
                if len(missions) < missions_per_group:
                    raise ValueError(
                        f"{local_json_dir} has {len(missions)} missions, expected at least {missions_per_group}"
                    )
                rng = random.Random(_sample_seed(int(args.seed), family, source, scene))
                selected = sorted(rng.sample(missions, missions_per_group))

                safe_family = _safe_component(family)
                safe_source = _safe_component(source)
                safe_scene = _safe_component(scene)
                json_rel_dir = local_json_dir.relative_to(src_scene_dir)
                package_json_dir = selected_root / safe_family / safe_source / safe_scene / json_rel_dir
                scenario_dir = scenario_root / safe_family / safe_source / safe_scene
                manifest_dir = manifest_root / safe_family / safe_source / safe_scene
                metadata_key = (family, source, scene)
                copied_metadata: list[Path] = []
                if metadata_key not in copied_metadata_cache:
                    copied_metadata = _copy_scene_metadata(
                        src_scene_dir,
                        copied_scene_root / safe_family / safe_source / safe_scene,
                    )
                    copied_metadata_cache.add(metadata_key)

                for rank, mission_path in enumerate(selected):
                    selected_mission_path, selected_metadata_path = _copy_selected_pair(
                        mission_path=mission_path,
                        local_json_dir=local_json_dir,
                        package_json_dir=package_json_dir,
                    )
                    scenario = _load_json(mission_path)
                    scene_ply = _scene_ply_path(str(args.remote_scene_root), source, scene)
                    smoke_scenario = _scenario_for_smoke(
                        scenario,
                        scene_id=None,
                        scene_ply=scene_ply,
                        actor_ply_frame_dir=actor_dirs[0],
                        actor_ply_frame_dirs=actor_dirs,
                        thin_trajectories=False,
                    )
                    scenario_out = scenario_dir / mission_path.name
                    write_json(scenario_out, smoke_scenario)
                    manifest = scenario_file_to_render_manifest(
                        scenario_out,
                        action_catalog_path=action_catalog,
                    )
                    _prune_peer_robots_for_human_only(manifest)
                    _attach_smoke_sensor(
                        manifest,
                        width=int(args.sensor_width),
                        height=int(args.sensor_height),
                    )
                    manifest_out = manifest_dir / f"{mission_path.stem}.render_manifest.json"
                    write_json(manifest_out, manifest)
                    index["entries"].append(
                        {
                            "entry_index": len(index["entries"]),
                            "rank_in_scene": rank,
                            "family": family,
                            "source": source,
                            "scene": scene,
                            "scenario_id": smoke_scenario.get("scenario_id"),
                            "mission_count_in_scene": len(missions),
                            "selected_original_json": _rel(selected_mission_path, output_root),
                            "selected_metadata_json": _rel(selected_metadata_path, output_root),
                            "render_scenario_json": _rel(scenario_out, output_root),
                            "render_manifest_json": _rel(manifest_out, output_root),
                            "scene_gaussian_ply": scene_ply,
                            "scene_generation_metadata": [_rel(path, output_root) for path in copied_metadata],
                            "robot_count": len(smoke_scenario.get("robots", [])),
                            "human_count": len(smoke_scenario.get("humans", [])),
                            "job_count": len(manifest.get("jobs", [])),
                            "expected_renderer_blocked": False,
                        }
                    )

    entries = index["entries"]
    index["entry_count"] = len(entries)
    index["selected_entry_count"] = len(entries)
    index["family_count"] = len({entry["family"] for entry in entries})
    index["source_scene_count"] = len({(entry["source"], entry["scene"]) for entry in entries})
    index["family_source_scene_count"] = len(
        {(entry["family"], entry["source"], entry["scene"]) for entry in entries}
    )
    write_json(output_root / "smoketest_package_index.json", index)
    print(
        f"Wrote {index['entry_count']} selected render scenarios "
        f"across {index['family_source_scene_count']} family/source scenes to {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
