#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from lighting.lighting_utils import LightFilterConfig, apply_light_filter  # noqa: E402


DEFAULT_PRESETS = [
    {"name": "dawn", "scale": 0.75, "temp_k": 3200.0},
    {"name": "morning", "scale": 0.9, "temp_k": 4500.0},
    {"name": "noon", "scale": 1.05, "temp_k": 6500.0},
    {"name": "afternoon", "scale": 1.0, "temp_k": 5600.0},
    {"name": "golden_hour", "scale": 0.85, "temp_k": 3000.0},
    {"name": "dusk", "scale": 0.6, "temp_k": 2600.0},
    {"name": "blue_hour", "scale": 0.65, "temp_k": 9000.0},
    {"name": "night", "scale": 0.4, "temp_k": 2200.0},
]


def _collect_mp4s(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def _load_mp4_list(list_path: Path, root: Path) -> list[Path]:
    if not list_path.is_file():
        raise SystemExit(f"MP4 list not found: {list_path}")
    paths: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = root / candidate
        paths.append(candidate)
    return paths


def _normalize_presets(presets: list[dict]) -> list[dict]:
    seen: set[str] = set()
    normalized: list[dict] = []
    for entry in presets:
        if not isinstance(entry, dict):
            raise SystemExit("Preset entries must be JSON objects.")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise SystemExit("Preset name is required.")
        if "/" in name or "\\" in name:
            raise SystemExit(f"Invalid preset name: {name}")
        if name in seen:
            raise SystemExit(f"Duplicate preset name: {name}")
        scale = entry.get("scale")
        ev = entry.get("ev")
        if scale is None:
            if ev is None:
                raise SystemExit(f"Preset {name} must set scale or ev.")
            scale = 2.0 ** float(ev)
        scale = float(scale)
        if scale <= 0.0:
            raise SystemExit(f"Preset {name} scale must be > 0.")
        temp_k = float(entry.get("temp_k", 0.0))
        vignette = float(entry.get("vignette", 0.0))
        normalized.append(
            {
                "name": name,
                "scale": scale,
                "temp_k": temp_k,
                "vignette": vignette,
            }
        )
        seen.add(name)
    return normalized


def _load_presets_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("presets", [])
    if not isinstance(data, list):
        raise SystemExit("Preset JSON must be a list of objects or {\"presets\": [...]}.")
    return _normalize_presets(data)


def _select_presets(presets: list[dict], names: list[str]) -> list[dict]:
    if not names:
        return presets
    name_map = {preset["name"]: preset for preset in presets}
    missing = [name for name in names if name not in name_map]
    if missing:
        raise SystemExit(f"Unknown presets: {', '.join(missing)}")
    return [name_map[name] for name in names]


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame * 255.0, 0.0, 255.0)
    return frame.astype(np.uint8)


def _copy_other_files(
    input_dir: Path,
    output_dirs: list[Path],
    mp4s: list[Path],
    *,
    overwrite: bool,
) -> int:
    copied = 0
    mp4_set = {path.resolve() for path in mp4s}
    for path in input_dir.rglob("*"):
        if path.is_dir():
            continue
        if path.resolve() in mp4_set:
            continue
        rel = path.relative_to(input_dir)
        for out_dir in output_dirs:
            dst = out_dir / rel
            if dst.exists() and not overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied += 1
    return copied


def _process_mp4_task(
    src_path_str: str,
    rel_path_str: str,
    tone_items: list[tuple[str, str, float, float, float]],
    *,
    overwrite: bool,
    max_frames: int | None,
) -> dict:
    src_path = Path(src_path_str)
    rel_path = Path(rel_path_str)
    start = time.perf_counter()
    reader = imageio.get_reader(src_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 10)
    writers: dict[str, imageio.core.format.Writer] = {}
    configs: dict[str, LightFilterConfig] = {}
    outputs: list[str] = []
    try:
        for name, out_dir_str, scale, temp_k, vignette in tone_items:
            out_path = Path(out_dir_str) / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                continue
            writers[name] = imageio.get_writer(out_path, fps=fps)
            configs[name] = LightFilterConfig(
                mode="global",
                strength=float(scale) - 1.0,
                radius_frac=0.0,
                center_xy=(0.5, 0.5),
                center_jitter=0.0,
                temp_k=float(temp_k),
                vignette=float(vignette),
                seed=0,
            )
            outputs.append(name)

        if not writers:
            return {
                "src": str(src_path),
                "rel": str(rel_path),
                "frames": 0,
                "outputs": [],
                "elapsed_sec": 0.0,
                "skipped": True,
            }

        frames = 0
        for frame_index, frame in enumerate(reader):
            if max_frames is not None and frames >= max_frames:
                break
            for name, writer in writers.items():
                filtered = apply_light_filter(frame, configs[name], frame_index=frame_index)
                writer.append_data(_frame_to_uint8(filtered))
            frames += 1
    finally:
        reader.close()
        for writer in writers.values():
            writer.close()
    elapsed = time.perf_counter() - start
    return {
        "src": str(src_path),
        "rel": str(rel_path),
        "frames": frames,
        "outputs": outputs,
        "elapsed_sec": elapsed,
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build time-of-day lighting variants (tone + brightness) from MP4 datasets."
    )
    parser.add_argument("input", type=Path, help="Input dataset directory.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern for MP4s under input (default: *.mp4).",
    )
    parser.add_argument(
        "--mp4-list",
        type=Path,
        default=None,
        help="Optional text file with one MP4 path per line (relative to input or absolute).",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Subset of preset names to render (default: all).",
    )
    parser.add_argument(
        "--preset-json",
        type=Path,
        default=None,
        help="Optional JSON list of presets (overrides built-in defaults).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for output datasets (default: input parent).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output MP4s if they already exist.",
    )
    parser.add_argument(
        "--other-mode",
        choices=("copy", "skip"),
        default="skip",
        help="How to handle non-MP4 files after MP4 processing (default: skip).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N MP4s (default: 10).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most this many MP4s.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most this many frames per MP4.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    input_dir = args.input
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    if args.preset_json is not None:
        presets = _load_presets_json(args.preset_json)
    else:
        presets = _normalize_presets(DEFAULT_PRESETS)
    presets = _select_presets(presets, args.presets or [])
    if not presets:
        raise SystemExit("No presets selected.")

    output_root = args.output_root if args.output_root is not None else input_dir.parent
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = {
        preset["name"]: output_root / f"{input_dir.name}_{preset['name']}"
        for preset in presets
    }
    for out_dir in output_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.mp4_list is not None:
        mp4s = _load_mp4_list(args.mp4_list, input_dir)
    else:
        print(f"[SCAN] collecting mp4s under {input_dir}...", flush=True)
        mp4s = _collect_mp4s(input_dir, args.pattern)
    if not mp4s:
        raise SystemExit(f"No MP4s matched {args.pattern} under {input_dir}")
    if args.max_files is not None:
        mp4s = mp4s[: int(args.max_files)]

    tone_items = [
        (
            preset["name"],
            str(output_dirs[preset["name"]]),
            float(preset["scale"]),
            float(preset["temp_k"]),
            float(preset.get("vignette", 0.0)),
        )
        for preset in presets
    ]
    tasks = [(str(path), str(path.relative_to(input_dir))) for path in mp4s]

    start = time.perf_counter()
    reports: list[dict] = []
    total_frames = 0
    files_skipped = 0
    outputs_per_preset = {preset["name"]: 0 for preset in presets}

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [
                executor.submit(
                    _process_mp4_task,
                    src,
                    rel,
                    tone_items,
                    overwrite=args.overwrite,
                    max_frames=args.max_frames,
                )
                for src, rel in tasks
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                report = future.result()
                reports.append(report)
                total_frames += report["frames"]
                if report.get("skipped"):
                    files_skipped += 1
                for name in report.get("outputs", []):
                    outputs_per_preset[name] += 1
                if args.progress_every > 0 and idx % int(args.progress_every) == 0:
                    print(f"[PROGRESS] processed {idx}/{len(tasks)} mp4s", flush=True)
    else:
        for idx, (src, rel) in enumerate(tasks, start=1):
            report = _process_mp4_task(
                src,
                rel,
                tone_items,
                overwrite=args.overwrite,
                max_frames=args.max_frames,
            )
            reports.append(report)
            total_frames += report["frames"]
            if report.get("skipped"):
                files_skipped += 1
            for name in report.get("outputs", []):
                outputs_per_preset[name] += 1
            if args.progress_every > 0 and idx % int(args.progress_every) == 0:
                print(f"[PROGRESS] processed {idx}/{len(tasks)} mp4s", flush=True)

    other_copied = 0
    if args.other_mode == "copy":
        other_copied = _copy_other_files(
            input_dir,
            list(output_dirs.values()),
            mp4s,
            overwrite=args.overwrite,
        )

    elapsed = time.perf_counter() - start
    summary = {
        "input": str(input_dir),
        "pattern": args.pattern,
        "output_root": str(output_root),
        "output_dirs": {name: str(path) for name, path in output_dirs.items()},
        "presets": presets,
        "files_found": len(mp4s),
        "files_processed": len(reports),
        "files_skipped": files_skipped,
        "outputs_per_preset": outputs_per_preset,
        "frames_processed": total_frames,
        "elapsed_sec": elapsed,
        "other_files_copied": other_copied,
        "per_file": reports,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
