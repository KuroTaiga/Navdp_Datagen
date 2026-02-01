#!/usr/bin/env python3
"""Parallel dispatcher for render_label_paths_telesim.py.

This is a lightweight TeleSim3D-backed alternative to parallel_render_paths.py.
It fans out scene renders across a thread pool and forwards extra args to the
per-scene renderer.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def _discover_scenes(tasks_dir: Path) -> list[str]:
    if not tasks_dir.is_dir():
        return []
    return [p.name for p in sorted(tasks_dir.iterdir()) if p.is_dir()]


def _build_command(
    *,
    render_script: Path,
    scenes_dir: Path,
    tasks_dir: Path,
    scene_id: str,
    output_dir: Path,
    extra_args: Iterable[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(render_script),
        "--scenes-dir",
        str(scenes_dir),
        "--tasks-dir",
        str(tasks_dir),
        "--scene",
        scene_id,
        "--output-dir",
        str(output_dir),
    ]
    cmd.extend(list(extra_args))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel TeleSim3D label rendering dispatcher.")
    root_dir = Path(__file__).absolute().parent
    parser.add_argument("--render-script", type=Path, default=root_dir / "render_label_paths_telesim.py")
    parser.add_argument("--scenes-dir", type=Path, default=root_dir / "data" / "scenes")
    parser.add_argument("--tasks-dir", type=Path, default=root_dir / "data" / "tasks")
    parser.add_argument("--output-dir", type=Path, default=root_dir / "data" / "tmp" / "test_telesim3d")
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--render-extra-args", action="append", default=[])
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    scenes = args.scene or _discover_scenes(args.tasks_dir)
    if not scenes:
        print(f"[ERROR] No scenes found under {args.tasks_dir}", file=sys.stderr)
        return 1

    extra_args: list[str] = []
    for snippet in args.render_extra_args:
        if snippet:
            extra_args.extend(shlex.split(snippet))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {}
        for scene_id in scenes:
            cmd = _build_command(
                render_script=args.render_script,
                scenes_dir=args.scenes_dir,
                tasks_dir=args.tasks_dir,
                scene_id=scene_id,
                output_dir=args.output_dir,
                extra_args=extra_args,
            )
            futures[executor.submit(subprocess.run, cmd, check=False)] = {
                "scene": scene_id,
                "cmd": cmd,
            }
        for future in as_completed(futures):
            info = futures[future]
            proc = future.result()
            results.append(
                {
                    "scene": info["scene"],
                    "returncode": proc.returncode,
                    "cmd": info["cmd"],
                }
            )

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    failures = [r for r in results if r.get("returncode") not in (0, None)]
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
