#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sampled path JSONs.")
    parser.add_argument("--sample-json", type=Path, required=True, help="Sample list JSON.")
    parser.add_argument("--scenes-dir", type=Path, required=True, help="Scene reconstruction root.")
    parser.add_argument("--tasks-dir", type=Path, required=True, help="Task output root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory root.")
    parser.add_argument(
        "--render-script",
        type=Path,
        default=None,
        help="Path to render_label_paths.py (default: alongside this script).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory to write per-path metrics JSON.",
    )
    parser.add_argument(
        "--render-extra-args",
        type=str,
        default="",
        help="Extra CLI args forwarded to render_label_paths.py.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --overwrite to render_label_paths.py (default: true).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running.",
    )
    args = parser.parse_args()

    sample_path = args.sample_json
    if not sample_path.is_file():
        raise SystemExit(f"Sample list not found: {sample_path}")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []
    if not samples:
        raise SystemExit("Sample list is empty.")

    render_script = args.render_script
    if render_script is None:
        render_script = Path(__file__).resolve().parents[1] / "render_label_paths.py"
    if not render_script.is_file():
        raise SystemExit(f"Render script not found: {render_script}")

    extra_args = shlex.split(args.render_extra_args) if args.render_extra_args else []
    if args.overwrite:
        extra_args.append("--overwrite")

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    for entry in samples:
        scene_id = entry["scene"]
        label_id = entry["label"]
        metrics_path = args.metrics_dir / f"{scene_id}_{label_id}.json"
        cmd = [
            sys.executable,
            str(render_script),
            "--scenes-dir",
            str(args.scenes_dir),
            "--tasks-dir",
            str(args.tasks_dir),
            "--scene",
            str(scene_id),
            "--label-id",
            str(label_id),
            "--output-dir",
            str(args.output_dir),
            "--metrics-json",
            str(metrics_path),
        ]
        cmd.extend(extra_args)
        print(" ".join(shlex.quote(part) for part in cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f"Render failed for scene={scene_id} label={label_id}")


if __name__ == "__main__":
    main()
