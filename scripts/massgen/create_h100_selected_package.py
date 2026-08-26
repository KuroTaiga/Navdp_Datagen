#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SELECTED_SCENES = {
    "CHINGMU_rescaled_1": [
        "0052_858885",
        "0034_858867",
        "0017_858849",
        "0073_858906",
        "0070_858903",
    ],
    "CHINGMU_rescaled_2": [
        "0032_858992",
        "0067_859028",
        "0005_858965",
        "0006_858966",
        "0062_859023",
    ],
    "CHINGMU_rescaled_3": [
        "0011_859081",
        "0004_859074",
        "0002_859072",
        "0010_859080",
        "0013_859083",
    ],
    "InteriorGS": [
        "0183_840105",
        "0753_840985",
        "0635_839968",
        "0087_839962",
        "0139_840055",
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the H100 formal-test MassGen package variant: selected "
            "renderable scenes, scene paths rewritten to /team/telenav/navsources, "
            "and human actors rewritten to the filtered no-waving action bundle."
        )
    )
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260824)
    parser.add_argument("--expected-entry-count", type=int, default=1000)
    parser.add_argument(
        "--scene-selection",
        choices=["default", "all"],
        default="default",
        help="default uses the curated 20-scene list; all uses every renderable package scene.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Maximum number of source/scene pairs to select after filters. 0 means no cap.",
    )
    parser.add_argument(
        "--max-entries-per-family",
        type=int,
        default=0,
        help=(
            "Maximum selected entries per mission family after scene filters. "
            "0 means no cap. Entries are distributed round-robin across the "
            "remaining source/scene groups for that family."
        ),
    )
    parser.add_argument(
        "--require-readable-scene-ply",
        action="store_true",
        help="Drop source/scene groups whose rewritten scene_gaussian_ply cannot be read by plyfile.",
    )
    parser.add_argument(
        "--skip-family",
        action="append",
        default=["dense_dynamic_combined", "dense_multi_robot"],
    )
    parser.add_argument("--navsources-root", type=Path, default=Path("/team/telenav/navsources"))
    parser.add_argument(
        "--interiorgs-root",
        type=Path,
        default=None,
        help=(
            "Optional package-local InteriorGS scene root. When set, "
            "legacy /mnt/DATA/.../scenes/<scene> paths are rewritten here "
            "instead of <navsources-root>/InteriorGS."
        ),
    )
    parser.add_argument(
        "--actor-root",
        type=Path,
        default=Path(
            "/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/"
            "grouped_actions/use_default_no_waving"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_rel(src_root: Path, dst_root: Path, rel: str | None) -> str | None:
    if not rel:
        return rel
    rel_path = Path(rel)
    src = src_root / rel_path
    dst = dst_root / rel_path
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return rel


def _rewrite_scene_path(value: Any, *, navsources_root: Path, interiorgs_root: Path | None = None) -> Any:
    if not isinstance(value, str):
        return value
    prefix = "/mnt/DATA/dongjk/navdp_data/"
    if not value.startswith(prefix):
        return value
    rest = value[len(prefix) :]
    parts = rest.split("/")
    if not parts:
        return value
    if parts[0] == "scenes" and len(parts) >= 2:
        root = interiorgs_root or (navsources_root / "InteriorGS")
        return str((root / Path(*parts[1:])).resolve())
    if parts[0].startswith("CHINGMU_rescaled_"):
        return str((navsources_root / Path(rest)).resolve())
    return value


def _actor_map(actor_root: Path) -> dict[str, Path]:
    return {
        "walk": actor_root
        / "stmc/outputs/subject_035__049_a_person_walks_straight_forward_at_a_relaxed_pace",
        "queue_wait": actor_root
        / "stmc/outputs/subject_008__055_a_person_rocks_the_body_slowly_from_the_heels_to_the_toes",
        "stand": actor_root
        / "kimodo/outputs/subject_035__007_a_person_stands_still_looks_left_looks_right_and_returns_the_head_to_face_forward",
    }


def _patch_manifest(
    manifest: dict[str, Any],
    *,
    src_root: Path,
    dst_root: Path,
    navsources_root: Path,
    interiorgs_root: Path | None,
    actor_paths: dict[str, Path],
    skip_families: set[str],
) -> tuple[dict[str, Any], Counter[str]]:
    counts: Counter[str] = Counter()
    scene_assets = manifest.get("scene_assets")
    if isinstance(scene_assets, dict):
        for key, value in list(scene_assets.items()):
            new_value = _rewrite_scene_path(
                value,
                navsources_root=navsources_root,
                interiorgs_root=interiorgs_root,
            )
            if new_value != value:
                scene_assets[key] = new_value
                counts["scene_path_rewrites"] += 1

    source = manifest.get("source")
    if isinstance(source, dict):
        scenario_path = source.get("scenario_path")
        if isinstance(scenario_path, str):
            try:
                rel = Path(scenario_path).expanduser().resolve().relative_to(src_root)
                source["scenario_path"] = str((dst_root / rel).resolve())
                counts["scenario_path_rewrites"] += 1
            except ValueError:
                for marker in (f"/{src_root.name}/", "/package_baseline_50/"):
                    if marker in scenario_path:
                        rel = scenario_path.split(marker, 1)[1]
                        source["scenario_path"] = str((dst_root / rel).resolve())
                        counts["scenario_path_rewrites"] += 1
                        break

    for human in manifest.get("actors", {}).get("humans") or []:
        if not isinstance(human, dict):
            continue
        for segment in human.get("action_segments") or []:
            if not isinstance(segment, dict):
                continue
            action = str(segment.get("action_label") or segment.get("render_action_id") or "")
            target = actor_paths.get(action)
            if target is None:
                counts[f"unknown_action:{action}"] += 1
                continue
            asset = segment.get("asset")
            if not isinstance(asset, dict):
                continue
            old_dir = asset.get("ply_frame_dir")
            if old_dir != str(target):
                asset["legacy_ply_frame_dir"] = old_dir
                asset["ply_frame_dir"] = str(target)
                asset["source"] = "h100_good_actions_no_waving"
                counts[f"actor_rewrite:{action}"] += 1
            generation_request = segment.get("generation_request")
            if isinstance(generation_request, dict):
                contract = generation_request.get("output_contract")
                if isinstance(contract, dict):
                    contract["legacy_ply_frame_dir"] = contract.get("ply_frame_dir")
                    contract["ply_frame_dir"] = str(target)

    manifest["h100_path_patch"] = {
        "navsources_root": str(navsources_root),
        "interiorgs_root": str(interiorgs_root) if interiorgs_root is not None else None,
        "actor_map": {key: str(value) for key, value in actor_paths.items()},
        "skip_families": sorted(skip_families),
    }
    return manifest, counts


def _validate_manifest(manifest: dict[str, Any], rel: str) -> list[str]:
    errors: list[str] = []
    scene_assets = manifest.get("scene_assets") if isinstance(manifest.get("scene_assets"), dict) else {}
    splat = (
        scene_assets.get("splat_model_path")
        or scene_assets.get("gaussian_model_path")
        or scene_assets.get("gaussian_ply_path")
    )
    if isinstance(splat, str) and not Path(splat).is_file():
        errors.append(f"{rel}: missing splat {splat}")
    scene_dir = scene_assets.get("scene_dir")
    if isinstance(scene_dir, str):
        for name in ("occupancy.json", "occupancy.png"):
            if not (Path(scene_dir) / name).is_file():
                errors.append(f"{rel}: missing {name} under {scene_dir}")
    for human in manifest.get("actors", {}).get("humans") or []:
        if not isinstance(human, dict):
            continue
        for segment in human.get("action_segments") or []:
            if not isinstance(segment, dict):
                continue
            asset = segment.get("asset") if isinstance(segment.get("asset"), dict) else {}
            path = asset.get("ply_frame_dir")
            if isinstance(path, str) and not Path(path).is_dir():
                errors.append(f"{rel}: missing actor dir {path}")
    return errors


def _selected_scene_keys(
    entries: list[Any],
    *,
    scene_selection: str,
    max_scenes: int,
    skip_families: set[str],
) -> set[tuple[str, str]]:
    if scene_selection == "default":
        keys = {
            (str(source), str(scene))
            for source, scenes in DEFAULT_SELECTED_SCENES.items()
            for scene in scenes
        }
        if max_scenes > 0:
            return set(sorted(keys)[:max_scenes])
        return keys

    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("family"))
        if family in skip_families or entry.get("expected_renderer_blocked"):
            continue
        key = (str(entry.get("source")), str(entry.get("scene")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(key)
        if max_scenes > 0 and len(selected) >= max_scenes:
            break
    return set(selected)


def _can_read_scene_ply(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing scene ply: {path}"
    try:
        from plyfile import PlyData  # type: ignore

        ply = PlyData.read(str(path))
        counts = {element.name: int(element.count) for element in ply.elements}
        return True, json.dumps(counts, sort_keys=True)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _filter_readable_scene_keys(
    entries: list[Any],
    *,
    scene_keys: set[tuple[str, str]],
    navsources_root: Path,
    interiorgs_root: Path | None,
    skip_families: set[str],
) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    scene_paths: dict[tuple[str, str], Path] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("family"))
        if family in skip_families or entry.get("expected_renderer_blocked"):
            continue
        key = (str(entry.get("source")), str(entry.get("scene")))
        if key not in scene_keys or key in scene_paths:
            continue
        scene_path = _rewrite_scene_path(
            entry.get("scene_gaussian_ply"),
            navsources_root=navsources_root,
            interiorgs_root=interiorgs_root,
        )
        if isinstance(scene_path, str):
            scene_paths[key] = Path(scene_path)

    readable: set[tuple[str, str]] = set()
    results: list[dict[str, Any]] = []
    for source, scene in sorted(scene_paths):
        path = scene_paths[(source, scene)]
        ok, message = _can_read_scene_ply(path)
        if ok:
            readable.add((source, scene))
        results.append(
            {
                "source": source,
                "scene": scene,
                "scene_gaussian_ply": str(path),
                "readable": ok,
                "message": message,
            }
        )
    return scene_keys.intersection(readable), results


def _candidate_entries(
    entries: list[Any],
    *,
    scene_keys: set[tuple[str, str]],
    skip_families: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("family"))
        source = str(entry.get("source"))
        scene = str(entry.get("scene"))
        if family in skip_families or entry.get("expected_renderer_blocked"):
            continue
        if (source, scene) not in scene_keys:
            continue
        selected.append(entry)
    return selected


def _cap_entries_per_family(
    entries: list[dict[str, Any]],
    *,
    max_entries_per_family: int,
) -> list[dict[str, Any]]:
    cap = int(max_entries_per_family or 0)
    if cap <= 0:
        return entries

    family_order: list[str] = []
    groups_by_family: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    for entry in entries:
        family = str(entry.get("family"))
        group = (str(entry.get("source")), str(entry.get("scene")))
        if family not in groups_by_family:
            groups_by_family[family] = {}
            family_order.append(family)
        groups_by_family[family].setdefault(group, []).append(entry)

    selected: list[dict[str, Any]] = []
    for family in family_order:
        groups = groups_by_family[family]
        group_order = list(groups)
        positions = {group: 0 for group in group_order}
        family_selected: list[dict[str, Any]] = []
        while len(family_selected) < cap:
            progressed = False
            for group in group_order:
                if len(family_selected) >= cap:
                    break
                pos = positions[group]
                group_entries = groups[group]
                if pos >= len(group_entries):
                    continue
                family_selected.append(group_entries[pos])
                positions[group] = pos + 1
                progressed = True
            if not progressed:
                break
        selected.extend(family_selected)
    return selected


def main() -> int:
    args = _parse_args()
    src_root = args.source_package.expanduser().resolve()
    dst_root = args.output_package.expanduser().resolve()
    navsources_root = args.navsources_root.expanduser().resolve()
    interiorgs_root = args.interiorgs_root.expanduser().resolve() if args.interiorgs_root is not None else None
    actor_paths = _actor_map(args.actor_root.expanduser().resolve())
    skip_families = {str(item) for item in args.skip_family or []}

    if not src_root.is_dir():
        raise SystemExit(f"missing source package: {src_root}")
    if dst_root.exists():
        raise SystemExit(f"output already exists, refusing to overwrite: {dst_root}")
    for action, path in actor_paths.items():
        if not path.is_dir():
            raise SystemExit(f"missing actor target for {action}: {path}")
        if not any(path.glob("*.ply")):
            raise SystemExit(f"actor target has no PLY frames for {action}: {path}")

    package_index = _load_json(src_root / "smoketest_package_index.json")
    scene_keys = _selected_scene_keys(
        package_index.get("entries") or [],
        scene_selection=str(args.scene_selection),
        max_scenes=max(0, int(args.max_scenes or 0)),
        skip_families=skip_families,
    )
    scene_readability: list[dict[str, Any]] = []
    if bool(args.require_readable_scene_ply):
        scene_keys, scene_readability = _filter_readable_scene_keys(
            package_index.get("entries") or [],
            scene_keys=scene_keys,
            navsources_root=navsources_root,
            interiorgs_root=interiorgs_root,
            skip_families=skip_families,
        )
    candidate_entries = _candidate_entries(
        package_index.get("entries") or [],
        scene_keys=scene_keys,
        skip_families=skip_families,
    )
    candidate_entries = _cap_entries_per_family(
        candidate_entries,
        max_entries_per_family=max(0, int(args.max_entries_per_family or 0)),
    )
    selected_entries: list[dict[str, Any]] = []
    patch_counts: Counter[str] = Counter()
    validation_errors: list[str] = []
    group_counts: Counter[tuple[str, str, str]] = Counter()
    source_counts: Counter[str] = Counter()

    for entry in candidate_entries:
        family = str(entry.get("family"))
        source = str(entry.get("source"))
        scene = str(entry.get("scene"))

        patched_entry = dict(entry)
        patched_entry["entry_index"] = len(selected_entries)
        patched_entry["scene_gaussian_ply"] = _rewrite_scene_path(
            patched_entry.get("scene_gaussian_ply"),
            navsources_root=navsources_root,
            interiorgs_root=interiorgs_root,
        )
        _copy_rel(src_root, dst_root, patched_entry.get("render_scenario_json"))
        _copy_rel(src_root, dst_root, patched_entry.get("selected_metadata_json"))
        _copy_rel(src_root, dst_root, patched_entry.get("selected_original_json"))
        for rel in patched_entry.get("scene_generation_metadata") or []:
            _copy_rel(src_root, dst_root, rel)

        manifest_rel = patched_entry.get("render_manifest_json")
        if not manifest_rel:
            raise SystemExit(f"missing render_manifest_json for selected entry: {entry}")
        manifest = _load_json(src_root / manifest_rel)
        manifest, counts = _patch_manifest(
            manifest,
            src_root=src_root,
            dst_root=dst_root,
            navsources_root=navsources_root,
            interiorgs_root=interiorgs_root,
            actor_paths=actor_paths,
            skip_families=skip_families,
        )
        patch_counts.update(counts)
        out_manifest = dst_root / manifest_rel
        _write_json(out_manifest, manifest)
        validation_errors.extend(_validate_manifest(manifest, str(manifest_rel)))

        selected_entries.append(patched_entry)
        group_counts[(family, source, scene)] += 1
        source_counts[source] += 1

    if validation_errors:
        dst_root.mkdir(parents=True, exist_ok=True)
        (dst_root / "validation_errors.txt").write_text("\n".join(validation_errors) + "\n", encoding="utf-8")
        raise SystemExit(f"validation failed with {len(validation_errors)} errors")
    if int(args.expected_entry_count or 0) > 0 and len(selected_entries) != int(args.expected_entry_count):
        raise SystemExit(f"expected {args.expected_entry_count} selected entries, got {len(selected_entries)}")

    dst_root.mkdir(parents=True, exist_ok=True)
    for top_json in src_root.glob("*.json"):
        if top_json.name == "smoketest_package_index.json":
            continue
        shutil.copy2(top_json, dst_root / top_json.name)

    selected_sources: dict[str, list[str]] = {}
    for source, scene in sorted(scene_keys):
        selected_sources.setdefault(source, []).append(scene)
    new_index = dict(package_index)
    new_index["entries"] = selected_entries
    new_index["entry_count"] = len(selected_entries)
    new_index["selected_entry_count"] = len(selected_entries)
    new_index["family_source_scene_count"] = len(group_counts)
    new_index["h100_selection"] = {
        "seed": int(args.selection_seed),
        "scene_selection": str(args.scene_selection),
        "max_scenes": int(args.max_scenes or 0),
        "max_entries_per_family": int(args.max_entries_per_family or 0),
        "require_readable_scene_ply": bool(args.require_readable_scene_ply),
        "sources": selected_sources,
        "skip_families": sorted(skip_families),
        "actor_map": {key: str(value) for key, value in actor_paths.items()},
        "navsources_root": str(navsources_root),
        "interiorgs_root": str(interiorgs_root) if interiorgs_root is not None else None,
    }
    _write_json(dst_root / "smoketest_package_index.json", new_index)
    if scene_readability:
        _write_json(dst_root / "scene_ply_readability.json", {"scenes": scene_readability})

    family_entries = Counter(str(entry.get("family")) for entry in selected_entries)
    family_groups = Counter(key[0] for key in group_counts)
    report = {
        "source_package": str(src_root),
        "output_package": str(dst_root),
        "selected_entry_count": len(selected_entries),
        "selected_group_count": len(group_counts),
        "scene_selection": str(args.scene_selection),
        "max_scenes": int(args.max_scenes or 0),
        "max_entries_per_family": int(args.max_entries_per_family or 0),
        "require_readable_scene_ply": bool(args.require_readable_scene_ply),
        "selected_sources": selected_sources,
        "scene_readability": scene_readability,
        "skip_families": sorted(skip_families),
        "family_entry_counts": dict(sorted(family_entries.items())),
        "family_group_counts": dict(sorted(family_groups.items())),
        "source_entry_counts": dict(sorted(source_counts.items())),
        "patch_counts": dict(sorted(patch_counts.items())),
        "actor_map": {key: str(value) for key, value in actor_paths.items()},
        "interiorgs_root": str(interiorgs_root) if interiorgs_root is not None else None,
    }
    _write_json(dst_root / "h100_selection_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
