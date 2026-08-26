#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ACTOR_ROOT = Path(
    "/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/"
    "grouped_actions/use_default_no_waving"
)
LOCOMOTION_KEYWORDS = (
    "walk",
    "jog",
    "run",
    "jump",
    "hop",
    "kick",
    "step",
    "shuffle",
    "march",
    "circle",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a narrow H100 package variant and randomize human action "
            "asset directories from the available grouped avatar action sources."
        )
    )
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--actor-root", type=Path, default=DEFAULT_ACTOR_ROOT)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--max-entries-per-family-source-scene", type=int, default=0)
    parser.add_argument("--min-action-pool-size", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _has_animation_frames(path: Path) -> bool:
    for pattern in ("frame_*.ply", "[0-9][0-9][0-9][0-9][0-9][0-9].ply"):
        if any(child.is_file() for child in path.glob(pattern)):
            return True
    plys = [child for child in path.glob("*.ply") if child.is_file()]
    return len(plys) > 1


def _discover_actor_dirs(actor_root: Path) -> list[Path]:
    candidates: list[Path] = []
    output_roots = [path for path in actor_root.glob("*/outputs") if path.is_dir()]
    if (actor_root / "outputs").is_dir():
        output_roots.append(actor_root / "outputs")
    for output_root in sorted(output_roots):
        for child in sorted(output_root.iterdir()):
            if child.is_dir() and _has_animation_frames(child):
                candidates.append(child.resolve())
    if not candidates and _has_animation_frames(actor_root):
        candidates.append(actor_root.resolve())
    return sorted(dict.fromkeys(candidates), key=lambda path: path.as_posix())


def _pool_for_action(
    action_label: str,
    actor_dirs: list[Path],
    *,
    min_pool_size: int,
) -> list[Path]:
    action = action_label.lower()
    if any(keyword in action for keyword in ("walk", "route", "move")):
        pool = [path for path in actor_dirs if "walk" in path.name.lower()]
    else:
        pool = [
            path
            for path in actor_dirs
            if not any(keyword in path.name.lower() for keyword in LOCOMOTION_KEYWORDS)
        ]
    if len(pool) < max(1, min_pool_size):
        pool = actor_dirs
    return pool


def _passes_filters(
    entry: Mapping[str, Any],
    *,
    families: set[str] | None,
    sources: set[str] | None,
    scenes: set[str] | None,
) -> bool:
    if families and str(entry.get("family")) not in families:
        return False
    if sources and str(entry.get("source")) not in sources:
        return False
    if scenes and str(entry.get("scene")) not in scenes:
        return False
    if entry.get("expected_renderer_blocked"):
        return False
    return True


def _select_entries(
    entries: list[Any],
    *,
    families: set[str] | None,
    sources: set[str] | None,
    scenes: set[str] | None,
    max_entries_per_group: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    group_counts: Counter[tuple[str, str, str]] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _passes_filters(entry, families=families, sources=sources, scenes=scenes):
            continue
        key = (str(entry.get("family")), str(entry.get("source")), str(entry.get("scene")))
        if max_entries_per_group > 0 and group_counts[key] >= max_entries_per_group:
            continue
        group_counts[key] += 1
        selected.append(entry)
    return selected


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


def _choice_seed(base_seed: int, *parts: object) -> int:
    text = "::".join(str(part) for part in (base_seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _randomize_manifest_assets(
    manifest: dict[str, Any],
    *,
    actor_pools: Mapping[str, list[Path]],
    all_actor_dirs: list[Path],
    actor_root: Path,
    seed: int,
) -> tuple[dict[str, Any], Counter[str], set[str]]:
    counts: Counter[str] = Counter()
    assigned: set[str] = set()
    scenario_id = str(manifest.get("source", {}).get("scenario_id") or "")
    humans = manifest.get("actors", {}).get("humans") or []
    for human_index, human in enumerate(humans):
        if not isinstance(human, dict):
            continue
        human_id = str(human.get("actor_id") or human.get("human_id") or human_index)
        for segment_index, segment in enumerate(human.get("action_segments") or []):
            if not isinstance(segment, dict):
                continue
            action_label = str(
                segment.get("action_label")
                or segment.get("render_action_id")
                or segment.get("action_id")
                or "unknown"
            )
            pool = list(actor_pools.get(action_label) or all_actor_dirs)
            if not pool:
                counts[f"missing_pool:{action_label}"] += 1
                continue
            choice = random.Random(
                _choice_seed(seed, scenario_id, human_id, segment_index, action_label)
            ).choice(pool)
            asset = segment.get("asset")
            if not isinstance(asset, dict):
                asset = {}
                segment["asset"] = asset
            old_dir = asset.get("ply_frame_dir")
            asset["legacy_ply_frame_dir"] = old_dir
            asset["ply_frame_dir"] = str(choice)
            asset["source"] = "envtest_randomized_human_assets"
            asset["asset_selection_seed"] = int(seed)
            asset["asset_selection_pool"] = action_label
            generation_request = segment.get("generation_request")
            if isinstance(generation_request, dict):
                contract = generation_request.get("output_contract")
                if isinstance(contract, dict):
                    contract["legacy_ply_frame_dir"] = contract.get("ply_frame_dir")
                    contract["ply_frame_dir"] = str(choice)
            assigned.add(str(choice))
            counts[f"segment:{action_label}"] += 1
    manifest["envtest_human_asset_randomization"] = {
        "seed": int(seed),
        "actor_pool_sizes": {key: len(value) for key, value in sorted(actor_pools.items())},
        "actor_root": str(actor_root),
    }
    return manifest, counts, assigned


def _manifest_action_labels(manifest: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    for human in manifest.get("actors", {}).get("humans") or []:
        if not isinstance(human, Mapping):
            continue
        for segment in human.get("action_segments") or []:
            if not isinstance(segment, Mapping):
                continue
            label = str(
                segment.get("action_label")
                or segment.get("render_action_id")
                or segment.get("action_id")
                or "unknown"
            )
            labels.add(label)
    return labels


def main() -> int:
    args = _parse_args()
    src_root = args.source_package.expanduser().resolve()
    dst_root = args.output_package.expanduser().resolve()
    actor_root = args.actor_root.expanduser().resolve()
    if not src_root.is_dir():
        raise SystemExit(f"missing source package: {src_root}")
    if not actor_root.is_dir():
        raise SystemExit(f"missing actor root: {actor_root}")
    if args.clean and dst_root.exists():
        shutil.rmtree(dst_root)
    if dst_root.exists():
        raise SystemExit(f"output package already exists: {dst_root}")

    index = _load_json(src_root / "smoketest_package_index.json")
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"{src_root / 'smoketest_package_index.json'} has no entries list")
    selected = _select_entries(
        entries,
        families={str(item) for item in args.family} if args.family else None,
        sources={str(item) for item in args.source} if args.source else None,
        scenes={str(item) for item in args.scene} if args.scene else None,
        max_entries_per_group=max(0, int(args.max_entries_per_family_source_scene or 0)),
    )
    if not selected:
        raise SystemExit("selection produced no entries")

    actor_dirs = _discover_actor_dirs(actor_root)
    if not actor_dirs:
        raise SystemExit(f"no actor action directories with PLY frames under {actor_root}")

    all_action_labels: set[str] = set()
    for entry in selected:
        manifest_rel = entry.get("render_manifest_json")
        if not manifest_rel:
            continue
        all_action_labels.update(_manifest_action_labels(_load_json(src_root / str(manifest_rel))))
    actor_pools = {
        action_label: _pool_for_action(
            action_label,
            actor_dirs,
            min_pool_size=max(1, int(args.min_action_pool_size)),
        )
        for action_label in sorted(all_action_labels)
    }

    dst_root.mkdir(parents=True)
    for top_json in src_root.glob("*.json"):
        if top_json.name != "smoketest_package_index.json":
            shutil.copy2(top_json, dst_root / top_json.name)

    new_entries: list[dict[str, Any]] = []
    patch_counts: Counter[str] = Counter()
    assigned_dirs: set[str] = set()
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    group_counts: Counter[tuple[str, str, str]] = Counter()
    missing_files: list[str] = []

    for original in selected:
        entry = dict(original)
        entry["entry_index"] = len(new_entries)
        for key in (
            "render_scenario_json",
            "selected_metadata_json",
            "selected_original_json",
        ):
            try:
                entry[key] = _copy_rel(src_root, dst_root, entry.get(key))
            except OSError as exc:
                missing_files.append(f"{entry.get(key)}: {exc}")
        for rel in entry.get("scene_generation_metadata") or []:
            try:
                _copy_rel(src_root, dst_root, rel)
            except OSError as exc:
                missing_files.append(f"{rel}: {exc}")

        manifest_rel = entry.get("render_manifest_json")
        if not manifest_rel:
            missing_files.append(f"missing render_manifest_json for {entry.get('scenario_id')}")
            continue
        manifest = _load_json(src_root / str(manifest_rel))
        manifest, counts, manifest_assigned = _randomize_manifest_assets(
            manifest,
            actor_pools=actor_pools,
            all_actor_dirs=actor_dirs,
            actor_root=actor_root,
            seed=int(args.seed),
        )
        patch_counts.update(counts)
        assigned_dirs.update(manifest_assigned)
        _write_json(dst_root / str(manifest_rel), manifest)

        new_entries.append(entry)
        family = str(entry.get("family"))
        source = str(entry.get("source"))
        scene = str(entry.get("scene"))
        family_counts[family] += 1
        source_counts[source] += 1
        group_counts[(family, source, scene)] += 1

    if missing_files:
        _write_json(dst_root / "missing_files.json", {"missing_files": missing_files})
        raise SystemExit(f"missing {len(missing_files)} referenced package files")

    new_index = dict(index)
    new_index["entries"] = new_entries
    new_index["entry_count"] = len(new_entries)
    new_index["selected_entry_count"] = len(new_entries)
    new_index["family_count"] = len(family_counts)
    new_index["source_scene_count"] = len({(entry.get("source"), entry.get("scene")) for entry in new_entries})
    new_index["family_source_scene_count"] = len(group_counts)
    new_index["envtest_human_asset_randomization"] = {
        "seed": int(args.seed),
        "source_package": str(src_root),
        "actor_root": str(actor_root),
        "actor_dir_count": len(actor_dirs),
        "assigned_actor_dir_count": len(assigned_dirs),
        "families": sorted(family_counts),
        "sources": sorted(source_counts),
        "scenes": sorted({str(entry.get("scene")) for entry in new_entries}),
        "action_pool_sizes": {key: len(value) for key, value in sorted(actor_pools.items())},
        "max_entries_per_family_source_scene": int(args.max_entries_per_family_source_scene or 0),
    }
    _write_json(dst_root / "smoketest_package_index.json", new_index)

    report = {
        "source_package": str(src_root),
        "output_package": str(dst_root),
        "actor_root": str(actor_root),
        "seed": int(args.seed),
        "selected_entry_count": len(new_entries),
        "selected_group_count": len(group_counts),
        "family_entry_counts": dict(sorted(family_counts.items())),
        "source_entry_counts": dict(sorted(source_counts.items())),
        "patch_counts": dict(sorted(patch_counts.items())),
        "action_pool_sizes": {key: len(value) for key, value in sorted(actor_pools.items())},
        "assigned_actor_dir_count": len(assigned_dirs),
        "assigned_actor_dirs_sample": sorted(assigned_dirs)[:50],
    }
    _write_json(dst_root / "human_asset_randomization_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
