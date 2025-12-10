#!/usr/bin/env python3
"""
Check rendered videos under a NAS directory and report any label whose MP4 is missing
or suspiciously small. Uses the assignment manifest to map each (scene, label) to
the responsible actor so we can identify problematic avatars quickly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify rendered videos and report undersized outputs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Assignment manifest JSON (output of random_actor_assignments.py).",
    )
    parser.add_argument(
        "--nas-root",
        type=Path,
        required=True,
        help="Root directory that should contain {scene}/{label}.mp4 outputs.",
    )
    parser.add_argument(
        "--min-mb",
        type=float,
        default=1.0,
        help="Minimum acceptable MP4 size in megabytes (default: 1.0 MB).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the JSON summary to stdout as well.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def bytes_threshold(min_mb: float) -> int:
    return int(float(min_mb) * 1024 * 1024)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    assignments = manifest.get("assignments", [])
    if not assignments:
        raise SystemExit("Manifest does not contain any assignments.")

    threshold = bytes_threshold(args.min_mb)
    nas_root = args.nas_root

    per_scene = defaultdict(lambda: {"expected": 0, "success": 0, "failed": 0, "skipped": 0})
    per_actor = defaultdict(lambda: {"expected": 0, "success": 0, "failed": 0, "skipped": 0})
    failures: list[dict] = []
    skipped: list[dict] = []

    for entry in assignments:
        scene = entry["scene"]
        label = entry["label"]
        actor_id = entry["actor_id"]
        video_path = nas_root / scene / f"{label}.mp4"

        per_scene[scene]["expected"] += 1
        per_actor[actor_id]["expected"] += 1

        exists = video_path.is_file()
        size = video_path.stat().st_size if exists else 0
        if exists and size >= threshold:
            per_scene[scene]["success"] += 1
            per_actor[actor_id]["success"] += 1
            continue

        if not exists:
            per_scene[scene]["skipped"] += 1
            per_actor[actor_id]["skipped"] += 1
            skipped.append(
                {
                    "scene": scene,
                    "label": label,
                    "actor_id": actor_id,
                    "video_path": str(video_path),
                }
            )
            continue

        per_scene[scene]["failed"] += 1
        per_actor[actor_id]["failed"] += 1
        failures.append(
            {
                "scene": scene,
                "label": label,
                "actor_id": actor_id,
                "video_path": str(video_path),
                "exists": exists,
                "size_bytes": size,
            }
        )

    per_actor_summary = {actor_id: dict(stats) for actor_id, stats in per_actor.items()}
    per_scene_summary = {scene_id: dict(stats) for scene_id, stats in per_scene.items()}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "nas_root": str(nas_root.resolve()),
        "threshold_bytes": threshold,
        "threshold_mb": args.min_mb,
        "total_assignments": len(assignments),
        "failure_count": len(failures),
        "skipped_count": len(skipped),
        "failures": failures,
        "skipped": skipped,
        "per_actor": per_actor_summary,
        "per_scene": per_scene_summary,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"[REPORT] JSON summary written to {args.json_out}")

    print(f"[SUMMARY] Checked {len(assignments)} videos under {nas_root}")
    print(f"[SUMMARY] Threshold: {args.min_mb:.2f} MB ({threshold} bytes)")
    print(f"[SUMMARY] Failures (too small): {len(failures)}")
    print(f"[SUMMARY] Skipped (missing files): {len(skipped)}")
    if failures:
        print("Actor_ID  Scene           Label   Size(Bytes)  Exists")
        for item in failures:
            print(
                f"{item['actor_id']:>8}  {item['scene']:<13}  {item['label']:<8}  "
                f"{item['size_bytes']:>11}  {item['exists']}"
            )
    if skipped:
        print("\nSkipped videos (missing MP4s):")
        print("Actor_ID  Scene           Label   Video Path")
        for item in skipped:
            print(
                f"{item['actor_id']:>8}  {item['scene']:<13}  {item['label']:<8}  {item['video_path']}"
            )

    print("\nPer-actor stats (expected / success / failed / skipped):")
    for actor_id in sorted(per_actor_summary.keys()):
        stats = per_actor_summary[actor_id]
        print(
            f"  {actor_id:>8}: {stats['expected']:>4} / {stats['success']:>4} / {stats['failed']:>4} / {stats['skipped']:>4}"
        )

    print("\nPer-scene stats (expected / success / failed / skipped):")
    for scene_id in sorted(per_scene_summary.keys()):
        stats = per_scene_summary[scene_id]
        print(
            f"  {scene_id}: {stats['expected']:>4} / {stats['success']:>4} / {stats['failed']:>4} / {stats['skipped']:>4}"
        )

    if args.print_json:
        print("\nJSON summary:\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
