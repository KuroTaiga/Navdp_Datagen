#!/usr/bin/env python3
"""Lean entry point for rendering NavDP paths in one Gaussian scene."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


RELEASE_DIR = Path(__file__).resolve().parent
DEFAULT_RENDER_SCRIPT = RELEASE_DIR / "render_label_paths_telesim.py"


def _positive_int_pair(values: list[str]) -> tuple[int, int]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected WIDTH HEIGHT")
    width, height = int(values[0]), int(values[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(f"Could not symlink scene directory {src} -> {dst}: {exc}") from exc


def _validate_path_json(path_json: Path) -> None:
    with path_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    path_payload = payload.get("path")
    if not isinstance(path_payload, dict):
        raise ValueError(f"{path_json} must contain a 'path' object")
    if not path_payload.get("raster_world") or not path_payload.get("raster_pixel"):
        raise ValueError(
            f"{path_json} must contain path.raster_world and path.raster_pixel; "
            "the renderer uses both to map planned-path coordinates into the splat world"
        )


def _is_renderable_path_json(path_json: Path) -> bool:
    try:
        _validate_path_json(path_json)
    except Exception:
        return False
    return True


def _collect_path_jsons(
    paths_dir: Path,
    *,
    exclude_detailed: bool,
    max_labels: int | None,
) -> list[Path]:
    candidates = sorted(paths_dir.glob("*.json"))
    if exclude_detailed:
        candidates = [p for p in candidates if not p.name.endswith("_detailed.json")]
    renderable = [p for p in candidates if _is_renderable_path_json(p)]
    if max_labels is not None and max_labels > 0:
        renderable = renderable[:max_labels]
    return renderable


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Render one or more planned paths through one Gaussian splatting scene."
    )
    parser.add_argument("--scene-dir", type=Path, required=True, help="Scene directory with occupancy.json/png and a PLY.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--path-json", type=Path, help="Single path JSON containing path.raster_world and path.raster_pixel.")
    input_group.add_argument("--paths-dir", type=Path, help="Directory of path JSONs to render.")
    parser.add_argument("--output-dir", type=Path, default=RELEASE_DIR / "outputs")
    parser.add_argument("--scene-id", default=None, help="Output scene ID. Defaults to scene directory name.")
    parser.add_argument("--label-id", default=None, help="Output label ID. Defaults to path JSON stem.")
    parser.add_argument("--gaussian-model", type=Path, default=None, help="Override Gaussian PLY path. Defaults to scene-dir/3dgs_raw.ply or first scene PLY.")
    parser.add_argument("--render-script", type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", nargs=2, default=("960", "720"), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-backend", choices=("cpu", "nvenc", "gpu"), default="nvenc")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-depth-maps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-camera-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--height-offset", type=float, default=0.3)
    parser.add_argument("--follow-distance", type=float, default=0.0)
    parser.add_argument("--look-ahead", type=float, default=2.0)
    parser.add_argument("--look-down", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resample-step", type=float, default=0.0)
    parser.add_argument("--minimal-frames", type=int, default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--exclude-detailed-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path-handedness", choices=("left", "right", "auto"), default="left")
    parser.add_argument("--swap-xy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--negate-xy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mirror-translation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--antialiasing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--backend", default="gsplat", choices=("gsplat", "diff-gaussian"))
    parser.add_argument("--dry-run", action="store_true")
    args, renderer_args = parser.parse_known_args()
    args.resolution = _positive_int_pair(list(args.resolution))
    return args, renderer_args


def main() -> int:
    args, renderer_args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    path_json = args.path_json.expanduser().resolve() if args.path_json else None
    paths_dir = args.paths_dir.expanduser().resolve() if args.paths_dir else None
    render_script = args.render_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    scene_id = args.scene_id or scene_dir.name

    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory not found: {scene_dir}")
    if not (scene_dir / "occupancy.json").is_file():
        raise FileNotFoundError(f"missing scene occupancy.json: {scene_dir / 'occupancy.json'}")
    if not (scene_dir / "occupancy.png").is_file():
        raise FileNotFoundError(f"missing scene occupancy.png: {scene_dir / 'occupancy.png'}")
    if not render_script.is_file():
        raise FileNotFoundError(f"renderer script not found: {render_script}")
    if path_json is not None:
        if not path_json.is_file():
            raise FileNotFoundError(f"path JSON not found: {path_json}")
        _validate_path_json(path_json)
        path_files = [path_json]
    else:
        if paths_dir is None or not paths_dir.is_dir():
            raise FileNotFoundError(f"paths directory not found: {paths_dir}")
        path_files = _collect_path_jsons(
            paths_dir,
            exclude_detailed=bool(args.exclude_detailed_labels),
            max_labels=args.max_labels,
        )
        if not path_files:
            raise FileNotFoundError(f"no renderable path JSONs found under: {paths_dir}")

    gaussian_model = args.gaussian_model.expanduser().resolve() if args.gaussian_model else None
    if gaussian_model is not None and not gaussian_model.is_file():
        raise FileNotFoundError(f"Gaussian model not found: {gaussian_model}")

    with tempfile.TemporaryDirectory(prefix="navdp_render_layout_") as tmp:
        tmp_root = Path(tmp)
        scenes_root = tmp_root / "scenes"
        tasks_root = tmp_root / "tasks"
        _link_dir(scene_dir, scenes_root / scene_id)
        task_label_dir = tasks_root / scene_id / "label_paths"
        label_ids: list[str] = []
        for path_file in path_files:
            label_id = args.label_id if path_json is not None and args.label_id else path_file.stem
            label_ids.append(label_id)
            _link_or_copy_file(path_file, task_label_dir / f"{label_id}.json")

        cmd = [
            sys.executable,
            str(render_script),
            "--scenes-dir",
            str(scenes_root),
            "--tasks-dir",
            str(tasks_root),
            "--scene",
            scene_id,
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--resolution",
            str(args.resolution[0]),
            str(args.resolution[1]),
            "--fov-deg",
            str(args.fov_deg),
            "--video-fps",
            str(args.video_fps),
            "--video-backend",
            args.video_backend,
            "--height-offset",
            str(args.height_offset),
            "--follow-distance",
            str(args.follow_distance),
            "--look-ahead",
            str(args.look_ahead),
            "--look-down",
            str(args.look_down),
            "--stride",
            str(args.stride),
            "--resample-step",
            str(args.resample_step),
            "--path-handedness",
            args.path_handedness,
            "--sh-degree",
            str(args.sh_degree),
            "--no-show-BEV",
            "--path-progress",
        ]
        if path_json is not None:
            cmd.extend(["--label-id", label_ids[0]])
        if args.max_labels is not None:
            cmd.extend(["--max-labels", str(args.max_labels)])
        if gaussian_model is not None:
            cmd.extend(["--gaussian-model", str(gaussian_model)])
        if args.minimal_frames is not None:
            cmd.extend(["--minimal-frames", str(args.minimal_frames)])
        cmd.append("--video" if args.video else "--no-video")
        cmd.append("--rgb-frames" if args.rgb_frames else "--no-rgb-frames")
        cmd.append("--save-depth-maps" if args.save_depth_maps else "--no-save-depth-maps")
        cmd.append("--save-camera-metadata" if args.save_camera_metadata else "--no-save-camera-metadata")
        cmd.append("--exclude-detailed-labels" if args.exclude_detailed_labels else "--no-exclude-detailed-labels")
        cmd.append("--swap-xy" if args.swap_xy else "--no-swap-xy")
        cmd.append("--negate-xy" if args.negate_xy else "--no-negate-xy")
        cmd.append("--mirror-translation" if args.mirror_translation else "--no-mirror-translation")
        cmd.append("--overwrite" if args.overwrite else "--no-overwrite")
        cmd.append("--antialiasing" if args.antialiasing else "--no-antialiasing")
        cmd.extend(renderer_args)

        env = os.environ.copy()
        env["GAUSSIAN_RENDER_BACKEND"] = args.backend
        print("[RUN]", " ".join(cmd), flush=True)
        if args.dry_run:
            return 0
        return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
