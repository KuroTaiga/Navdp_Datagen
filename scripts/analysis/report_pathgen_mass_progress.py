#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_DEFAULT = 500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Pathplanner mass-generation progress by mission family "
            "from mass_generation_progress.json files."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="Generation root to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--target-default",
        type=int,
        default=TARGET_DEFAULT,
        help="Target count to assume for initialized scenes without a progress JSON.",
    )
    parser.add_argument(
        "--active-dir-limit",
        type=int,
        default=20,
        help="Maximum active output dirs to print.",
    )
    return parser.parse_args()


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _family_label(root: Path, path: Path) -> str:
    rel = path.parent.relative_to(root)
    parts = rel.parts
    if root.name == "formal_fast_v1":
        return parts[0] if parts else "unknown"
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else "unknown"


def _active_generation_counts(roots: list[Path]) -> tuple[dict[str, int], list[str]]:
    counts: defaultdict[str, int] = defaultdict(int)
    active_dirs: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return {}, []
    for child in proc.iterdir():
        if not child.name.isdigit():
            continue
        try:
            cmd = (child / "cmdline").read_bytes().decode("utf-8", "ignore").split("\0")
        except OSError:
            continue
        if "generate-mass-examples" not in " ".join(cmd):
            continue
        output_dir = None
        for index, item in enumerate(cmd):
            if item == "--output-dir" and index + 1 < len(cmd):
                output_dir = cmd[index + 1]
                break
        if output_dir is None:
            continue
        active_dirs.append(output_dir)
        output_path = Path(output_dir)
        for root in roots:
            try:
                label = _family_label(root, output_path / "mass_generation_progress.json")
            except ValueError:
                continue
            counts[label] += 1
            break
    return dict(counts), active_dirs


def _format_eta(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _iter_progress_paths(root: Path) -> list[Path]:
    if root.name == "formal_fast_v1":
        return sorted(root.glob("*/*/*/mass_generation_progress.json"))
    if root.name == "formal_slow_v1":
        return sorted(root.glob("*/*/*/*/mass_generation_progress.json"))
    return sorted(root.rglob("mass_generation_progress.json"))


def _iter_scene_manifest_paths(root: Path) -> list[Path]:
    if root.name == "formal_fast_v1":
        return sorted(root.glob("*/*/*/scene_manifest.json"))
    if root.name == "formal_slow_v1":
        return sorted(root.glob("*/*/*/*/scene_manifest.json"))
    return sorted(root.rglob("scene_manifest.json"))


def main() -> int:
    args = _parse_args()
    roots = args.root or [
        Path("/private_lxh/dongjk/navdata/mass_generation_runs/formal_fast_v1"),
        Path("/private_lxh/dongjk/navdata/mass_generation_runs/formal_slow_v1"),
    ]
    roots = [root for root in roots if root.exists()]

    summary: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scenes": 0,
            "complete_scenes": 0,
            "accepted": 0,
            "target": 0,
            "attempted": 0,
            "elapsed_s": 0.0,
            "last_update": None,
            "statuses": defaultdict(int),
            "complete_scene_seconds": [],
            "roots": set(),
        }
    )
    all_complete_scene_seconds: list[float] = []
    seen_dirs: set[str] = set()

    for root in roots:
        for progress_path in _iter_progress_paths(root):
            payload = _load_json(progress_path)
            if payload is None:
                continue
            label = _family_label(root, progress_path)
            seen_dirs.add(str(progress_path.parent))
            family_payload = None
            family_progress = payload.get("family_progress")
            if isinstance(family_progress, dict):
                if label in family_progress:
                    family_payload = family_progress[label]
                elif len(family_progress) == 1:
                    family_payload = next(iter(family_progress.values()))
            if not isinstance(family_payload, dict):
                family_payload = {}

            accepted = int(family_payload.get("accepted", payload.get("accepted_total", 0)) or 0)
            target = int(
                family_payload.get("target", payload.get("accepted_target", args.target_default))
                or args.target_default
            )
            status = str(
                payload.get("status") or ("complete" if accepted >= target else "running")
            )
            elapsed_s = float(payload.get("elapsed_s") or 0.0)
            updated_at = _parse_ts(payload.get("updated_at"))

            item = summary[label]
            item["scenes"] += 1
            if accepted >= target or status == "complete":
                item["complete_scenes"] += 1
            item["accepted"] += accepted
            item["target"] += target
            item["attempted"] += int(payload.get("completed_attempts") or 0)
            item["elapsed_s"] += elapsed_s
            item["statuses"][status] += 1
            item["roots"].add(root.name)
            if updated_at and (item["last_update"] is None or updated_at > item["last_update"]):
                item["last_update"] = updated_at
            if accepted >= target and elapsed_s > 0:
                item["complete_scene_seconds"].append(elapsed_s)
                all_complete_scene_seconds.append(elapsed_s)

    for root in roots:
        for scene_manifest in _iter_scene_manifest_paths(root):
            scene_dir = str(scene_manifest.parent)
            if scene_dir in seen_dirs:
                continue
            try:
                label = _family_label(
                    root,
                    scene_manifest.parent / "mass_generation_progress.json",
                )
            except ValueError:
                label = "unknown"
            item = summary[label]
            item["scenes"] += 1
            item["target"] += args.target_default
            item["statuses"]["not_started"] += 1
            item["roots"].add(root.name)

    active_counts, active_dirs = _active_generation_counts(roots)
    global_median = (
        statistics.median(all_complete_scene_seconds) if all_complete_scene_seconds else None
    )

    print("PATHGEN_PROGRESS_SNAPSHOT " + datetime.now(timezone.utc).isoformat())
    print("roots=" + ",".join(str(root) for root in roots))
    print("active_workers_by_family=" + json.dumps(active_counts, sort_keys=True))
    print("active_output_dirs=" + json.dumps(active_dirs[: args.active_dir_limit], indent=2))
    print(
        "family,accepted,target,percent,scenes_complete,scenes_total,"
        "status_counts,active_workers,eta"
    )

    for label in sorted(summary):
        item = summary[label]
        target = int(item["target"] or 0)
        accepted = int(item["accepted"] or 0)
        percent = (accepted / target * 100.0) if target else 0.0
        remaining = max(target - accepted, 0)
        if remaining <= 0:
            eta = "done"
        else:
            median_scene = (
                statistics.median(item["complete_scene_seconds"])
                if item["complete_scene_seconds"]
                else global_median
            )
            sec_per_target = median_scene / args.target_default if median_scene else None
            active_workers = active_counts.get(label, 0)
            if sec_per_target and active_workers > 0:
                eta = _format_eta(remaining * sec_per_target / active_workers)
            elif sec_per_target:
                eta = "pending; serial " + _format_eta(remaining * sec_per_target)
            else:
                eta = "unknown"
        statuses = json.dumps(dict(sorted(item["statuses"].items())), sort_keys=True)
        print(
            f"{label},{accepted},{target},{percent:.2f}%,"
            f"{item['complete_scenes']},{item['scenes']},{statuses},"
            f"{active_counts.get(label, 0)},{eta}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
