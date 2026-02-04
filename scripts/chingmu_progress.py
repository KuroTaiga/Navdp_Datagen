#!/usr/bin/env python3
"""
Progress report: planned paths (JSON) vs generated videos (MP4), per scene and overall.

Default roots:
  planned_root = ./data/CHINGMU_75_rescaled_0800_42_iter1/
  video_root   = ./navdata/CHINGMU_0800/

Usage:
  ./chingmu_progress.py
  ./chingmu_progress.py --csv ./report/chingmu_progress.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_PLANNED_ROOT = Path("./data/CHINGMU_75_rescaled_0800_42_iter1/")
DEFAULT_VIDEO_ROOT = Path("./navdata/CHINGMU_0800/")


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def count_files_one_level(root: Path, suffix: str, label: str) -> Dict[str, int]:
    """
    Count files with suffix `suffix` directly under each immediate subfolder of root.
    Only scans: root/<scene_id>/*<suffix>
    Does NOT descend further.
    Prints progress as it scans each scene folder.
    """
    counts: Dict[str, int] = {}

    if not root.exists():
        eprint(f"[{label}] Root not found: {root}")
        return counts

    scene_dirs = [p for p in root.iterdir() if p.is_dir()]
    total = len(scene_dirs)

    eprint(f"[{label}] Scanning root: {root}")
    eprint(f"[{label}] Found {total} scene folders")

    t0 = time.time()
    for i, scene_dir in enumerate(scene_dirs, 1):
        n = 0
        for p in scene_dir.iterdir():  # non-recursive
            if p.is_file() and p.suffix.lower() == suffix:
                n += 1
        counts[scene_dir.name] = n

        # per-scene progress line
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0.0
        eta = (total - i) / rate if rate > 0 else 0.0
        eprint(
            f"[{label}] {i:>5}/{total}  scene={scene_dir.name}  {suffix}={n}  "
            f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s"
        )

    elapsed = time.time() - t0
    eprint(f"[{label}] Done. Scenes={total}, elapsed={elapsed:.1f}s")
    return counts


def pct(planned: int, done: int) -> float:
    if planned <= 0:
        return 0.0
    return min(100.0, (done / planned) * 100.0)


def status(planned: int, done: int) -> str:
    if planned <= 0:
        return "NO_PLAN"
    return "DONE" if done >= planned else "IN_PROGRESS"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--planned-root",
        type=Path,
        default=DEFAULT_PLANNED_ROOT,
        help=f"Root containing <scene_id>/*.json (default: {DEFAULT_PLANNED_ROOT})",
    )
    ap.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help=f"Root containing <scene_id>/*.mp4 (default: {DEFAULT_VIDEO_ROOT})",
    )
    ap.add_argument("--csv", type=Path, default=None, help="Optional: write report as CSV to this path")
    args = ap.parse_args()

    eprint("=== chingmu_progress.py ===")
    eprint(f"Planned root: {args.planned_root}")
    eprint(f"Video root:   {args.video_root}")
    eprint("")

    planned = count_files_one_level(args.planned_root, ".json", label="PLAN")
    eprint("")
    videos = count_files_one_level(args.video_root, ".mp4", label="MP4")
    eprint("")

    scenes = sorted(set(planned.keys()) | set(videos.keys()))
    rows: list[Tuple[str, int, int, float, str]] = []

    total_planned = 0
    total_done = 0

    eprint(f"[MERGE] Merging counts across {len(scenes)} scenes")
    for idx, scene in enumerate(scenes, 1):
        p = planned.get(scene, 0)
        d = videos.get(scene, 0)
        total_planned += p
        total_done += d
        rows.append((scene, p, d, pct(p, d), status(p, d)))

        if idx % 50 == 0 or idx == len(scenes):
            eprint(f"[MERGE] {idx}/{len(scenes)} scenes processed")

    overall_pct = pct(total_planned, total_done)

    # stdout report
    print("=== Roots ===")
    print(f"Planned root: {args.planned_root}")
    print(f"Video root:   {args.video_root}")
    print()
    print("=== Overall ===")
    print(f"Planned(JSON): {total_planned}")
    print(f"Done(MP4):     {total_done}")
    print(f"Progress:      {total_done}/{total_planned} ({overall_pct:.2f}%)")
    print()

    scene_w = max(8, max((len(s) for s in scenes), default=8))
    print(f"{'SCENE':<{scene_w}}  {'PLANNED':>8}  {'MP4':>8}  {'PCT':>8}  {'STATUS':>12}")
    print("-" * (scene_w + 8 + 8 + 8 + 12 + 10))
    for scene, p, d, pc, st in rows:
        print(f"{scene:<{scene_w}}  {p:>8}  {d:>8}  {pc:>7.2f}%  {st:>12}")

    if args.csv:
        import csv

        eprint(f"[CSV] Writing {args.csv}")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "planned_json", "done_mp4", "pct", "status"])
            for scene, p, d, pc, st in rows:
                w.writerow([scene, p, d, f"{pc:.2f}", st])
            w.writerow([])
            w.writerow(["OVERALL", total_planned, total_done, f"{overall_pct:.2f}", ""])
        eprint("[CSV] Done")

    eprint("=== Done ===")


if __name__ == "__main__":
    main()
