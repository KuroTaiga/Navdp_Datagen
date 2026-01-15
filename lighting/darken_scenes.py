#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils import gaussian_ply_utils as ply_utils  # noqa: E402


def _iter_scenes(scenes_dir: Path, scene_ids: Iterable[str] | None) -> list[Path]:
    if scene_ids:
        return [scenes_dir / scene_id for scene_id in scene_ids]
    return [p for p in sorted(scenes_dir.iterdir()) if p.is_dir()]


def _scale_sh_coeffs(data: np.ndarray, scale: float) -> tuple[int, int]:
    names = list(data.dtype.names or [])
    dc_fields = [f"f_dc_{i}" for i in range(3) if f"f_dc_{i}" in names]
    rest_fields = [name for name in names if name.startswith("f_rest_")]
    if not dc_fields and not rest_fields:
        raise ValueError("PLY file does not contain f_dc_* or f_rest_* fields.")
    for field in dc_fields + rest_fields:
        data[field] = (data[field].astype(np.float32) * scale).astype(data.dtype[field])
    return len(dc_fields), len(rest_fields)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise RuntimeError(f"Output scene dir already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "skip":
        return
    if mode == "link":
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create darkened/brightened scene PLYs.")
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=BASE_DIR / "data" / "scenes",
        help="Input scene directory root (default: data/scenes).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory root for modified scenes.",
    )
    parser.add_argument(
        "--scene",
        nargs="+",
        default=None,
        help="Optional scene id(s) to process (default: all).",
    )
    parser.add_argument(
        "--ply-name",
        type=str,
        default="3dgs_compressed.ply",
        help="PLY filename inside each scene (default: 3dgs_compressed.ply).",
    )
    scale_group = parser.add_mutually_exclusive_group(required=True)
    scale_group.add_argument(
        "--scale",
        type=float,
        help="Scale applied to f_dc_* and f_rest_* coefficients (e.g., 0.5 for darker).",
    )
    scale_group.add_argument(
        "--ev",
        type=float,
        help="Exposure value delta (scale = 2**ev). Negative values darken.",
    )
    parser.add_argument(
        "--other-mode",
        choices=("link", "copy", "skip"),
        default="link",
        help="How to handle non-PLY assets (default: link).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output scenes if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing outputs.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    scale = float(args.scale) if args.scale is not None else 2.0 ** float(args.ev)
    scenes_dir = args.scenes_dir
    output_root = args.output_dir
    if not scenes_dir.is_dir():
        raise SystemExit(f"Scenes dir not found: {scenes_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = _iter_scenes(scenes_dir, args.scene)
    report = {
        "scenes_dir": str(scenes_dir),
        "output_dir": str(output_root),
        "ply_name": args.ply_name,
        "scale": scale,
        "other_mode": args.other_mode,
        "processed": [],
        "skipped": [],
    }

    for scene_dir in scenes:
        if not scene_dir.is_dir():
            report["skipped"].append({"scene": scene_dir.name, "reason": "missing_dir"})
            continue
        ply_path = scene_dir / args.ply_name
        if not ply_path.is_file():
            report["skipped"].append({"scene": scene_dir.name, "reason": "missing_ply"})
            continue

        out_scene_dir = output_root / scene_dir.name
        if args.dry_run:
            report["processed"].append(
                {"scene": scene_dir.name, "ply": str(ply_path), "output": str(out_scene_dir), "dry_run": True}
            )
            continue

        _prepare_output_dir(out_scene_dir, overwrite=args.overwrite)
        ply = ply_utils.GaussianPly.read(ply_path)
        data = ply.data.copy()
        dc_count, rest_count = _scale_sh_coeffs(data, scale)
        ply.write(data, out_scene_dir / args.ply_name)

        for entry in scene_dir.iterdir():
            if entry.name == args.ply_name:
                continue
            _link_or_copy(entry, out_scene_dir / entry.name, mode=args.other_mode)

        report["processed"].append(
            {
                "scene": scene_dir.name,
                "ply": str(ply_path),
                "output": str(out_scene_dir),
                "dc_fields": dc_count,
                "rest_fields": rest_count,
            }
        )

    report["total_processed"] = len(report["processed"])
    report["total_skipped"] = len(report["skipped"])
    print(json.dumps(report, indent=2))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
