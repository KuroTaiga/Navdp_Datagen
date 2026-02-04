#!/usr/bin/env python3
"""
Quick smoke test for the TeleSim "following" pipeline.

What it does:
1) Picks a scene (default: first scene under --tasks-dir matching --scene-prefix, e.g. "0001_")
2) Builds a *trimmed* actor-assignment manifest containing only the first N label paths for that scene
3) Runs run_random_human_datagen_telesim.sh using that manifest (unless --no-run)

This avoids running the full dataset while still exercising the actor-follow + TeleSim renderer path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from subprocess import run
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _discover_scene(tasks_dir: Path, prefix: str) -> str:
    if not tasks_dir.is_dir():
        raise SystemExit(f"[ERROR] tasks dir not found: {tasks_dir}")
    scenes = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not scenes:
        raise SystemExit(f"[ERROR] no scenes matching prefix {prefix!r} under {tasks_dir}")
    return scenes[0]


def _find_default_source_manifest() -> Path | None:
    candidates = [
        REPO_ROOT / "data" / "actor_assignments_w_ban_CHINGMU.json",
        REPO_ROOT / "data" / "actor_assignments_w_ban_65k_1.json",
        REPO_ROOT / "data" / "actor_assignments_w_ban_65k_2.json",
        REPO_ROOT / "data" / "actor_assignments_w_ban_65k.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _trim_manifest(*, manifest: dict[str, Any], scene_id: str, limit: int) -> dict[str, Any]:
    assignments = [a for a in (manifest.get("assignments") or []) if str(a.get("scene")) == scene_id]
    if not assignments:
        raise SystemExit(f"[ERROR] source manifest has no assignments for scene {scene_id!r}")

    def _sort_key(a: dict[str, Any]) -> tuple[int, str]:
        try:
            order = int(a.get("order_index", 0))
        except Exception:
            order = 0
        return (order, str(a.get("label", "")))

    assignments.sort(key=_sort_key)

    picked: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for a in assignments:
        label = str(a.get("label", ""))
        if not label or label in seen_labels:
            continue
        picked.append(a)
        seen_labels.add(label)
        if len(picked) >= limit:
            break

    if not picked:
        raise SystemExit(f"[ERROR] no label assignments selected for scene {scene_id!r}")

    actor_ids = {str(a.get("actor_id", "")) for a in picked if a.get("actor_id") is not None}
    actors = [a for a in (manifest.get("actors") or []) if str(a.get("id", "")) in actor_ids]

    out: dict[str, Any] = {
        "actors": actors,
        "assignments": picked,
        # Keep a bit of provenance if present.
        "generated_at": manifest.get("generated_at"),
        "seed": manifest.get("seed"),
        "tasks_root": manifest.get("tasks_root"),
        "scenes_root": manifest.get("scenes_root"),
        "trimmed_for_scene": scene_id,
        "trimmed_label_limit": int(limit),
    }
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quick TeleSim following pipeline smoke test.")
    ap.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "data" / "CHINGMU_75_rescaled_0800_42_iter1")
    ap.add_argument("--scenes-dir", type=Path, default=REPO_ROOT / "data" / "CHINGMU_scenes_rescaled")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "navdata" / "following_quick_30paths")
    ap.add_argument("--scene-prefix", type=str, default="0001_", help="Prefix used to select the first scene.")
    ap.add_argument("--scene", type=str, default=None, help="Override scene id (skips prefix discovery).")
    ap.add_argument("--num-paths", type=int, default=30, help="How many label paths to run (default: 30).")
    ap.add_argument("--workers", type=int, default=1, help="Workers passed to the TeleSim dispatcher (default: 1).")
    ap.add_argument("--source-manifest", type=Path, default=None, help="Existing large assignment manifest to trim.")
    ap.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Where to write the trimmed manifest (default: data/tmp/following_<scene>_<N>paths_manifest.json).",
    )
    ap.add_argument(
        "--run-script",
        type=Path,
        default=REPO_ROOT / "run_random_human_datagen_telesim.sh",
        help="TeleSim following runner script to execute.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only write the trimmed manifest and print info.")
    ap.add_argument("--no-run", action="store_true", help="Alias for --dry-run.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_paths <= 0:
        raise SystemExit("--num-paths must be > 0")

    scene_id = args.scene or _discover_scene(args.tasks_dir, args.scene_prefix)
    source_manifest = args.source_manifest or _find_default_source_manifest()
    if source_manifest is None:
        raise SystemExit(
            "[ERROR] no source manifest found. Pass --source-manifest, or create one via random_actor_assignments.py."
        )
    if not source_manifest.is_file():
        raise SystemExit(f"[ERROR] source manifest not found: {source_manifest}")

    manifest = _load_json(source_manifest)
    trimmed = _trim_manifest(manifest=manifest, scene_id=scene_id, limit=int(args.num_paths))

    out_manifest = args.out_manifest
    if out_manifest is None:
        out_manifest = REPO_ROOT / "data" / "tmp" / f"following_{scene_id}_{int(args.num_paths)}paths_manifest.json"
    _write_json(out_manifest, trimmed)

    labels = [str(a.get("label")) for a in (trimmed.get("assignments") or [])]
    print(f"[OK] Scene: {scene_id}")
    print(f"[OK] Trimmed manifest: {out_manifest}")
    print(f"[OK] Labels selected: {len(labels)}")
    if labels:
        preview = ", ".join(labels[: min(10, len(labels))])
        suffix = " ..." if len(labels) > 10 else ""
        print(f"[OK] First labels: {preview}{suffix}")

    if args.dry_run or args.no_run:
        return 0

    if not args.run_script.is_file():
        raise SystemExit(f"[ERROR] run script not found: {args.run_script}")

    env = dict(os.environ)
    env["SCENE_ID"] = scene_id
    env["TASKS_DIR"] = str(args.tasks_dir)
    env["SCENES_DIR"] = str(args.scenes_dir)
    env["OUTPUT_DIR"] = str(args.output_dir)
    env["WORKERS"] = str(int(args.workers))
    env["ASSIGNMENTS_OUT"] = str(out_manifest)

    print(f"[RUN] {args.run_script} (scene={scene_id}, paths={len(labels)}, workers={args.workers})", flush=True)
    proc = run(["bash", str(args.run_script)], env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

