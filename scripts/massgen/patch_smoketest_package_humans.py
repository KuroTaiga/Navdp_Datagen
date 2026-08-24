#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REMOTE_HUMAN_ROOT = "/mnt/DATA/dongjk/navdp_data/human_gs_source"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch a MassGen smoke-test package to use available human GS avatar folders."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--remote-human-root", default=DEFAULT_REMOTE_HUMAN_ROOT)
    parser.add_argument("--actor-source-id", action="append", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _avatar_dirs(remote_human_root: str, actor_source_ids: list[str]) -> list[str]:
    root = remote_human_root.rstrip("/")
    ids = [str(item).strip() for item in actor_source_ids if str(item).strip()]
    if not ids:
        raise ValueError("at least one --actor-source-id is required")
    return [f"{root}/{actor_id}" for actor_id in ids]


def _collect_old_dirs(package_root: Path, remote_human_root: str, index: Mapping[str, Any]) -> list[str]:
    from_index = index.get("actor_identity_dirs")
    if isinstance(from_index, list):
        old = [str(item) for item in from_index if isinstance(item, str) and item]
        if old:
            return old

    pattern = re.compile(re.escape(remote_human_root.rstrip("/")) + r"/[^/\"'\s]+")
    found: set[str] = set()
    for path in package_root.rglob("*.json"):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return sorted(found)


def _rewrite(value: Any, mapping: Mapping[str, str]) -> tuple[Any, int]:
    if isinstance(value, str):
        out = value
        changes = 0
        for old, new in mapping.items():
            if old in out:
                out = out.replace(old, new)
                changes += 1
        return out, changes
    if isinstance(value, list):
        out_list = []
        changes = 0
        for item in value:
            out_item, item_changes = _rewrite(item, mapping)
            out_list.append(out_item)
            changes += item_changes
        return out_list, changes
    if isinstance(value, dict):
        out_dict: dict[str, Any] = {}
        changes = 0
        for key, item in value.items():
            out_item, item_changes = _rewrite(item, mapping)
            out_dict[key] = out_item
            changes += item_changes
        return out_dict, changes
    return value, 0


def main() -> int:
    args = _parse_args()
    package_root = args.package_root.expanduser().resolve()
    index_path = package_root / "smoketest_package_index.json"
    index = _load_json(index_path)
    new_dirs = _avatar_dirs(args.remote_human_root, args.actor_source_id)
    old_dirs = _collect_old_dirs(package_root, args.remote_human_root, index)
    if len(new_dirs) < len(old_dirs):
        raise ValueError(f"need at least {len(old_dirs)} replacement avatars, got {len(new_dirs)}")
    mapping = dict(zip(old_dirs, new_dirs))

    changed_files = 0
    string_replacements = 0
    for path in sorted(package_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewritten, changes = _rewrite(payload, mapping)
        if changes:
            _write_json(path, rewritten)
            changed_files += 1
            string_replacements += changes

    index = _load_json(index_path)
    index["actor_identity_dirs"] = new_dirs
    index["human_asset_patch"] = {
        "source_actor_identity_dirs": old_dirs,
        "replacement_actor_identity_dirs": new_dirs,
        "mapping": mapping,
    }
    _write_json(index_path, index)
    print(
        f"patched {changed_files} JSON files with {string_replacements} avatar string replacements "
        f"under {package_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
