#!/usr/bin/env python3
"""
Progress report: planned paths (JSON) vs generated videos (MP4), per scene and overall.

Strict rules:
- Planned files must be named exactly: <number>.json  (e.g., 0.json, 12.json, 000123.json)
  Anything else is ignored (and can be reported).
- Video files counted are *.mp4 (not strict on mp4 naming, since you only asked strict JSON).

Default roots:
  planned_root = ./data/CHINGMU_75_rescaled_0800_42_iter1/
  video_root   = ./navdata/CHINGMU_0800/

Usage:
  ./chingmu_progress.py
  ./chingmu_progress.py --csv ./report/chingmu_progress.csv
  ./chingmu_progress.py --report-ignored  # prints ignored json filenames per scene (first N)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_PLANNED_ROOT = Path("./data/CHINGMU_75_rescaled_0800_42_iter1/")
DEFAULT_VIDEO_ROOT = Path("./navdata/CHINGMU_0800/")

# strict: only digits + .json, nothing else
STRICT_JSON_RE = re.compile(r"^[0-9]+\.json$")


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def count_planned_json_strict(
    root: Path,
    report_ignored: bool = False,
    ignored_cap: int = 10,
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Counts only files matching <number>.json directly under root/<scene_id>/.
    Non-recursive. Prints progress as it scans.
    Returns: (counts, ignored_examples_per_scene)
    """
    counts: Dict[str, int] = {}
    ignored: Dict[str, List[str]] = {}

    if not root.exists():
        eprint(f"[PLAN] Root not found: {root}")
        return counts, ignored

    scene_dirs = [p for p in root.iterdir() if p.is_dir()]
    total = len(scene_dirs)

    eprint(f"[PLAN] Scanning root: {root}")
    eprint(f"[PLAN] Found {total} scene folders")
    t0 = time.time()

    for i, scene_dir in enumerate(scene_dirs, 1):
        ok = 0
        bad_examples: List[str] = []
        bad_total = 0

        for p in scene_dir.iterdir():  # non-recursive
            if not p.is_file():
                continue
            name = p.name
            if STRICT_JSON_RE.fullmatch(name):
                ok += 1
            else:
                # only track "json-like" ignored to avoid noise
                if p.suffix.lower() == ".json":
                    bad_total += 1
                    if report_ignored and len(bad_examples) < ignored_cap:
                        bad_examples.append(name)

        counts[scene_dir.name] = ok
        if report_ignored and bad_total > 0:
            ignored[scene_dir.name] = bad_examples

        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0.0
        eta = (total - i) / rate if rate > 0 else 0.0
        extra = f"  ignored_json={bad_total}" if report_ignored else ""
        eprint(
            f"[PLAN] {i:>5}/{total}  scene={scene_dir.name}  strict_json={ok}{extra}  "
            f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s"
        )

    elapsed = time.time() - t0
    eprint(f"[PLAN] Done. Scenes={total}, elapsed={elapsed:.1f}s")
    return counts, ignored


def count_mp4_one_level(root: Path) -> Dict[str, int]:
    """
    Count *.mp4 directly under root/<scene_id>/ (non-recursive). Prints progress.
    """
    counts: Dict[str, int] = {}

    if not root.exists():
        eprint(f"[MP4] Root not found: {root}")
        return counts

    scene_dirs = [p for p in root.iterdir() if p.is_dir()]
    total = len(scene_dirs)

    eprint(f"[MP4] Scanning root: {root}")
    eprint(f"[MP4] Found {total} scene folders")
    t0 = time.time()

    for i, scene_dir in enumerate(scene_dirs, 1):
        n = 0
        for p in scene_dir.iterdir():  # non-recursive
            if p.is_file() and p.suffix.lower() == ".mp4":
                n += 1
        counts[scene_dir.name] = n

        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0.0
        eta = (total - i) / rate if rate > 0 else 0.0
        eprint(
            f"[MP4]  {i:>5}/{total}  scene={scene_dir.name}  mp4={n}  "
            f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s"
        )

    elapsed = time.time() - t0
    eprint(f"[MP4] Done. Scenes={total}, elapsed={elapsed:.1f}s")
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
    ap.add_argument("--planned-root", type=Path, default=DEFAULT_PLANNED_ROOT)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--csv", type=Path, default=None, help="Optional: write report as CSV")
    ap.add_argument(
        "--report-ignored",
        action="store_true",
        help="Report ignored .json filenames that are NOT strictly <number>.json (prints first N per scene)",
    )
    ap.add_argument(
        "--ignored-cap",
        type=int,
        default=10,
        help="Max ignored filenames to print per scene (when --report-ignored is set)",
    )
    args = ap.parse_args()

    eprint("=== chingmu_progress.py ===")
    eprint(f"Planned root: {args.planned_root}")
    eprint(f"Video root:   {args.video_root}")
    eprint("Strict planned json format: <number>.json")
    eprint("")

    planned, ignored = count_planned_json_strict(
        args.planned_root, report_ignored=args.report_ignored, ignored_cap=args.ignored_cap
    )
    eprint("")
    videos = count_mp4_one_level(args.video_root)
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
    print("Planned json rule: <number>.json (strict)")
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

    if args.report_ignored and ignored:
        print("\n=== Ignored planned .json (NOT <number>.json) ===")
        for scene in sorted(ignored.keys()):
            ex = ignored[scene]
            if ex:
                more = "..." if len(ex) >= args.ignored_cap else ""
                print(f"{scene}: {', '.join(ex)}{more}")

    if args.csv:
        import csv

        eprint(f"[CSV] Writing {args.csv}")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "planned_json_strict", "done_mp4", "pct", "status"])
            for scene, p, d, pc, st in rows:
                w.writerow([scene, p, d, f"{pc:.2f}", st])
            w.writerow([])
            w.writerow(["OVERALL", total_planned, total_done, f"{overall_pct:.2f}", ""])
        eprint("[CSV] Done")

    eprint("=== Done ===")


if __name__ == "__main__":
    main()
