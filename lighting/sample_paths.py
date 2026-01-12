#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


def _has_raster_world(payload: dict[str, Any]) -> bool:
    path_data = payload.get("path")
    if isinstance(path_data, dict):
        raster = path_data.get("raster_world")
        if isinstance(raster, list) and len(raster) >= 2:
            return True
    raster = payload.get("raster_world")
    return isinstance(raster, list) and len(raster) >= 2


def _sample_label(scene_dir: Path, rng: random.Random) -> Path | None:
    json_files = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix == ".json"]
    rng.shuffle(json_files)
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _has_raster_world(payload):
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample random path JSONs from task outputs.")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Task output root containing scene subdirectories.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of paths to sample (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for sampling (default: 12345).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the sample list JSON.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Optional plain-text list (scene label) output.",
    )
    parser.add_argument(
        "--allow-duplicate-scenes",
        action="store_true",
        help="Allow multiple samples from the same scene (default: unique scenes).",
    )
    args = parser.parse_args()

    tasks_dir = args.tasks_dir
    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks dir not found: {tasks_dir}")
    rng = random.Random(int(args.seed))

    scenes = [p for p in tasks_dir.iterdir() if p.is_dir()]
    rng.shuffle(scenes)
    samples = []
    tried = 0
    while scenes and len(samples) < int(args.count):
        scene_dir = scenes.pop(0)
        tried += 1
        label_path = _sample_label(scene_dir, rng)
        if label_path is None:
            continue
        samples.append(
            {
                "scene": scene_dir.name,
                "label": label_path.stem,
                "path": str(label_path),
            }
        )
        if args.allow_duplicate_scenes:
            scenes.append(scene_dir)
            rng.shuffle(scenes)

    if len(samples) < int(args.count):
        raise SystemExit(
            f"Only sampled {len(samples)} paths from {tried} scenes; reduce --count or check tasks dir."
        )

    payload = {
        "tasks_dir": str(tasks_dir),
        "count": int(args.count),
        "seed": int(args.seed),
        "samples": samples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text("\n".join(f"{s['scene']} {s['label']}" for s in samples) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
