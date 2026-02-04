#!/usr/bin/env python3
"""
Compare overlapping navigation paths between two dataset folders.

A "path" is identified by its start/goal pair within the same scene.
Only numeric JSON filenames are considered (e.g., 112.json). Scene folders
must match NNNN_* or NNNNN_* with 1 <= N <= 10000.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple


SCENE_RE = re.compile(r"^(\d{4,5})_")


Pair = Tuple[Tuple[float, float], Tuple[float, float]]


def is_scene_name(name: str) -> bool:
    match = SCENE_RE.match(name)
    if not match:
        return False
    try:
        number = int(match.group(1))
    except ValueError:
        return False
    return 1 <= number <= 10000


def iter_scene_dirs(root: str) -> Iterable[Tuple[str, str]]:
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise RuntimeError(f"Failed to read folder: {root}: {exc}") from exc
    for entry in entries:
        if entry.is_dir() and is_scene_name(entry.name):
            yield entry.path, entry.name


def is_numeric_json(filename: str) -> bool:
    if not filename.endswith(".json"):
        return False
    stem = filename[:-5]
    return stem.isdigit()


def get_xy(value: object) -> Tuple[float, float]:
    if isinstance(value, dict):
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise KeyError("Unsupported coordinate format")


def extract_pair(data: dict, coord: str, round_ndigits: Optional[int]) -> Pair:
    if coord == "pixel":
        start = get_xy(data["start"]["pixel"])
        goal = get_xy(data["goal"]["pixel"])
        return (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
    if coord == "world":
        start = get_xy(data["start"]["world"])
        goal = get_xy(data["goal"]["world"])
        if round_ndigits is not None:
            start = (round(start[0], round_ndigits), round(start[1], round_ndigits))
            goal = (round(goal[0], round_ndigits), round(goal[1], round_ndigits))
        return start, goal
    raise ValueError(f"Unknown coord type: {coord}")


def count_files(scene_counts: Dict[str, Counter]) -> int:
    return sum(sum(counter.values()) for counter in scene_counts.values())


def count_unique(scene_pairs: Dict[str, set]) -> int:
    return sum(len(pairs) for pairs in scene_pairs.values())


def load_dataset(
    root: str, coord: str, round_ndigits: Optional[int]
) -> Tuple[Dict[str, set], Dict[str, Counter], Dict[str, int]]:
    scene_pairs: Dict[str, set] = {}
    scene_counts: Dict[str, Counter] = {}
    stats = {
        "files": 0,
        "bad_json": 0,
        "missing_fields": 0,
    }

    for scene_path, scene_name in iter_scene_dirs(root):
        pairs_counter: Counter = Counter()
        for dirpath, _, filenames in os.walk(scene_path):
            for filename in filenames:
                if not is_numeric_json(filename):
                    continue
                stats["files"] += 1
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    stats["bad_json"] += 1
                    continue
                try:
                    pair = extract_pair(data, coord, round_ndigits)
                except (KeyError, TypeError, ValueError):
                    stats["missing_fields"] += 1
                    continue
                pairs_counter[pair] += 1

        if pairs_counter:
            scene_pairs[scene_name] = set(pairs_counter.keys())
            scene_counts[scene_name] = pairs_counter

    return scene_pairs, scene_counts, stats


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.2f}%"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare overlapping start/goal pairs across two datasets."
    )
    parser.add_argument("folder_a", help="First dataset folder")
    parser.add_argument("folder_b", help="Second dataset folder")
    parser.add_argument(
        "--coord",
        choices=["pixel", "world"],
        default="pixel",
        help="Use pixel or world coordinates for start/goal pairs",
    )
    parser.add_argument(
        "--round",
        dest="round_ndigits",
        type=int,
        default=6,
        help="Rounding digits for world coordinates (ignored for pixel)",
    )
    parser.add_argument(
        "--per-scene",
        action="store_true",
        help="Print per-scene overlap details",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit per-scene rows (0 means no limit)",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.folder_a):
        print(f"Folder not found: {args.folder_a}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.folder_b):
        print(f"Folder not found: {args.folder_b}", file=sys.stderr)
        return 2

    round_ndigits = None if args.coord == "pixel" else args.round_ndigits

    pairs_a, counts_a, stats_a = load_dataset(
        args.folder_a, args.coord, round_ndigits
    )
    pairs_b, counts_b, stats_b = load_dataset(
        args.folder_b, args.coord, round_ndigits
    )

    scenes_common = sorted(set(pairs_a.keys()) & set(pairs_b.keys()))

    unique_overlap = 0
    file_overlap_a = 0
    file_overlap_b = 0
    per_scene_rows = []

    for scene in scenes_common:
        overlap = pairs_a[scene] & pairs_b[scene]
        unique_overlap += len(overlap)
        file_overlap_a += sum(counts_a[scene][pair] for pair in overlap)
        file_overlap_b += sum(counts_b[scene][pair] for pair in overlap)

        if args.per_scene:
            per_scene_rows.append(
                (
                    scene,
                    len(overlap),
                    len(pairs_a[scene]),
                    len(pairs_b[scene]),
                    sum(counts_a[scene].values()),
                    sum(counts_b[scene].values()),
                )
            )

    files_a = count_files(counts_a)
    files_b = count_files(counts_b)
    unique_a = count_unique(pairs_a)
    unique_b = count_unique(pairs_b)

    print(f"Dataset A: {args.folder_a}")
    print(f"  scenes (valid): {len(pairs_a)}")
    print(f"  path files (numeric json with start/goal): {files_a}")
    print(f"  unique start/goal pairs: {unique_a}")
    print(
        f"  skipped json (bad/missing): "
        f"{stats_a['bad_json'] + stats_a['missing_fields']}"
    )
    print(f"Dataset B: {args.folder_b}")
    print(f"  scenes (valid): {len(pairs_b)}")
    print(f"  path files (numeric json with start/goal): {files_b}")
    print(f"  unique start/goal pairs: {unique_b}")
    print(
        f"  skipped json (bad/missing): "
        f"{stats_b['bad_json'] + stats_b['missing_fields']}"
    )
    print(f"Common scenes: {len(scenes_common)}")
    print(
        f"Unique overlaps: {unique_overlap} "
        f"(A: {format_rate(unique_overlap, unique_a)}, "
        f"B: {format_rate(unique_overlap, unique_b)})"
    )
    print(
        f"A path files with match in B: {file_overlap_a} "
        f"({format_rate(file_overlap_a, files_a)})"
    )
    print(
        f"B path files with match in A: {file_overlap_b} "
        f"({format_rate(file_overlap_b, files_b)})"
    )

    if args.per_scene:
        if args.limit > 0:
            per_scene_rows = per_scene_rows[: args.limit]
        if per_scene_rows:
            print("Per-scene overlaps:")
            for row in per_scene_rows:
                scene, uniq_overlap, uniq_a_scene, uniq_b_scene, files_a_scene, files_b_scene = row
                print(
                    f"  {scene}: unique_overlap={uniq_overlap} "
                    f"unique_a={uniq_a_scene} unique_b={uniq_b_scene} "
                    f"files_a={files_a_scene} files_b={files_b_scene}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
