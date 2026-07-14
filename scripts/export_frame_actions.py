#!/usr/bin/env python3
# Example:
#   python3 scripts/export_frame_actions.py ./data1/33w_key2
# Example with optional tweaks:
#   python3 scripts/export_frame_actions.py ./data1/33w_key2 --scenes-dir ./data/scenes --max-next 8 --move-threshold-deg 10 --turn-threshold-deg 15 --turn-threshold-scale 0.5 --clean
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from argparse import BooleanOptionalAction


FRAME_RE = re.compile(r"^frame_(\d+)_camera\.json$")
CAMERA_PATH_SUFFIX = "_camera.json"


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        header = fh.read(8)
        if header != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a valid PNG file")
        length = int.from_bytes(fh.read(4), "big")
        chunk_type = fh.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"{path} missing IHDR chunk")
        width = int.from_bytes(fh.read(4), "big")
        height = int.from_bytes(fh.read(4), "big")
        _ = fh.read(length - 8)
    return width, height


def load_occupancy_metadata(scene_dir: Path) -> dict:
    occ_json = scene_dir / "occupancy.json"
    if not occ_json.is_file():
        raise FileNotFoundError(f"Missing occupancy.json in {scene_dir}")
    with occ_json.open("r", encoding="utf-8") as fh:
        occ = json.load(fh)

    scale = float(occ.get("scale", 1.0))
    min_x, min_y, min_z = map(float, occ.get("min", (0.0, 0.0, 0.0)))
    max_x, max_y, max_z = map(float, occ.get("max", (0.0, 0.0, 0.0)))

    lower = occ.get("lower") or [min_x, min_y, min_z]
    upper = occ.get("upper") or [max_x, max_y, max_z]
    lower_z = float(lower[2])
    upper_z = float(upper[2])

    occ_png = scene_dir / "occupancy.png"
    if not occ_png.is_file():
        raise FileNotFoundError(f"Missing occupancy.png in {scene_dir}")
    width_px, height_px = read_png_size(occ_png)

    left = min_x
    right = left + width_px * scale
    top = max_y
    bottom = top - height_px * scale

    return {
        "width": int(width_px),
        "height": int(height_px),
        "scale": scale,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "lower_z": lower_z,
        "upper_z": upper_z,
    }


def world_to_pixel(meta: dict, x: float, y: float) -> tuple[int, int]:
    u = int(round((x - float(meta["left"])) / float(meta["scale"])))
    v = int(round((float(meta["top"]) - y) / float(meta["scale"])))
    return u, v


def parse_frame_index(path: Path) -> int | None:
    match = FRAME_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _label_from_camera_filename(path: Path) -> str | None:
    if not path.name.endswith(CAMERA_PATH_SUFFIX):
        return None
    stem = path.stem
    if not stem.endswith("_camera"):
        return None
    return stem[: -len("_camera")]


def iter_labels(
    scene_dir: Path,
    label_filter: set[str] | None,
) -> Iterable[tuple[str, str, Path]]:
    """
    Yield (label_id, mode, path) for one scene.

    mode:
      - "path_json": scene_dir/{label}_camera.json
      - "legacy_dir": scene_dir/{label}/frame_XXXX_camera.json
    """
    camera_jsons: list[tuple[str, Path]] = []
    for p in sorted(scene_dir.glob(f"*{CAMERA_PATH_SUFFIX}")):
        if not p.is_file():
            continue
        label = _label_from_camera_filename(p)
        if not label:
            continue
        if label_filter and label not in label_filter:
            continue
        camera_jsons.append((label, p))
    if camera_jsons:
        for label, p in camera_jsons:
            yield label, "path_json", p
        return

    for child in sorted(scene_dir.iterdir()):
        if child.is_dir():
            if label_filter and child.name not in label_filter:
                continue
            yield child.name, "legacy_dir", child


def _iter_camera_payloads_from_path_json(path_json: Path) -> list[dict]:
    payload = json.loads(path_json.read_text(encoding="utf-8"))
    frames = payload.get("frames") or []
    ordered = list(frames) if isinstance(frames, list) else []
    ordered.sort(key=lambda f: int(f.get("frame", 0)))
    return ordered


def signed_angle_delta(a: float, b: float) -> float:
    delta = b - a
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def compute_yaw_delta_series(frames: list[dict], step: int) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for k in range(len(frames) - step):
        idx = int(frames[k]["frame"])
        y0 = float(frames[k]["yaw"])
        y1 = float(frames[k + step]["yaw"])
        deg = abs(math.degrees(signed_angle_delta(y0, y1)))
        xs.append(idx)
        ys.append(deg)
    return xs, ys


def compute_yaw_window_series(frames: list[dict], window: int) -> list[dict]:
    step = max(1, int(window))
    series: list[dict] = []
    for idx in range(len(frames) - step):
        start = frames[idx]
        end = frames[idx + step]
        delta = signed_angle_delta(start["yaw"], end["yaw"])
        delta_deg = math.degrees(delta)
        series.append(
            {
                "frame": start["frame"],
                "next_frame": end["frame"],
                "delta_yaw_deg": delta_deg,
                "abs_delta_yaw_deg": abs(delta_deg),
            }
        )
    return series


def parse_window_steps(raw: str) -> list[int]:
    if raw is None:
        raise ValueError("window steps string is required")
    steps: list[int] = []
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError as exc:
            raise ValueError(f"invalid window step '{chunk}'") from exc
        if value <= 0:
            raise ValueError(f"window step must be positive (got {value})")
        if value not in steps:
            steps.append(value)
    if not steps:
        raise ValueError("no valid window steps provided")
    return steps


def draw_arrow(
    draw,
    start: tuple[float, float],
    direction: tuple[float, float],
    *,
    length_px: int,
    width_px: int,
    color: tuple[int, int, int],
) -> None:
    dx, dy = direction
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return
    dx /= norm
    dy /= norm
    end = (start[0] + dx * length_px, start[1] + dy * length_px)
    draw.line([start, end], fill=color, width=width_px)
    head_len = max(4.0, length_px * 0.35)
    angle = math.atan2(dy, dx)
    left = (
        end[0] - head_len * math.cos(angle - math.pi / 6),
        end[1] - head_len * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_len * math.cos(angle + math.pi / 6),
        end[1] - head_len * math.sin(angle + math.pi / 6),
    )
    draw.line([end, left], fill=color, width=width_px)
    draw.line([end, right], fill=color, width=width_px)


def render_bev_debug(
    occ_png: Path,
    frames: list[dict],
    output_path: Path,
    *,
    arrow_step: int,
    arrow_len_px: int,
    arrow_width_px: int,
    mirror_center: bool,
) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(occ_png).convert("RGB")
    draw = ImageDraw.Draw(img)
    step = max(1, int(arrow_step))
    color = (255, 0, 0)
    radius = 2

    for idx, frame in enumerate(frames):
        if idx % step != 0:
            continue
        pixel = frame.get("pixel")
        if pixel is None:
            continue
        u, v = pixel
        if not (0 <= u < img.width and 0 <= v < img.height):
            continue
        fx, fy = frame["forward"]
        # Convert world-space forward to pixel-space direction.
        direction = (fx, -fy)
        if mirror_center:
            u = (img.width - 1) - u
            v = (img.height - 1) - v
            direction = (-direction[0], -direction[1])
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color)
        draw_arrow(
            draw,
            (u, v),
            direction,
            length_px=int(arrow_len_px),
            width_px=int(arrow_width_px),
            color=color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def render_plot(
    xs: list[int],
    ys: list[float],
    title: str,
    out_path: Path,
    mpl_config_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel("Yaw delta (degrees)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def render_yaw_windows_plot(
    yaw_windows: dict[str, list[dict]],
    out_path: Path,
    mpl_config_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    for window, series in sorted(yaw_windows.items(), key=lambda item: int(item[0])):
        xs = [entry["frame"] for entry in series]
        ys = [entry["abs_delta_yaw_deg"] for entry in series]
        plt.plot(xs, ys, linewidth=1.2, label=f"step={window}")
    plt.title("Yaw Delta (Sliding Window)")
    plt.xlabel("Frame index")
    plt.ylabel("Abs yaw delta (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_scene(
    scene_dir: Path,
    scenes_meta_dir: Path,
    output_template: str,
    plot_name: str,
    max_next: int,
    skip_frames: int,
    move_threshold_deg: float,
    turn_threshold_deg: float,
    ahead_dot_eps: float,
    action_yaw_window: int,
    turn_threshold_scale: float,
    verbose: bool,
    plots: bool,
    clean: bool,
    label_filter: set[str] | None,
    skip_actions: bool,
    debug_yaw: bool,
    debug_bev: bool,
    debug_yaw_template: str,
    debug_bev_template: str,
    debug_yaw_plot_template: str,
    debug_yaw_window_steps: list[int],
    debug_output_dir: Path | None,
    debug_clean: bool,
    debug_arrow_step: int,
    debug_arrow_len_px: int,
    debug_arrow_width_px: int,
    debug_bev_mirror_center: bool,
    debug_yaw_plot: bool,
    skip_existing: bool,
) -> tuple[str, int, int]:
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    scene_id = scene_dir.name
    meta = None
    scene_meta_dir = scenes_meta_dir / scene_id
    try:
        meta = load_occupancy_metadata(scene_meta_dir)
        log(f"[scene] {scene_id}: loaded occupancy metadata from {scene_meta_dir}")
    except FileNotFoundError:
        meta = None
        log(f"[scene] {scene_id}: occupancy metadata missing at {scene_meta_dir}")

    outputs_written = 0
    labels_seen = 0
    step = max(1, int(skip_frames) + 1)
    action_window = max(1, int(action_yaw_window))
    action_turn_threshold = float(turn_threshold_deg) * float(turn_threshold_scale)

    plots_root = scene_dir.parent / "plots"
    if clean:
        legacy_plot_scene = plots_root / scene_id
        if legacy_plot_scene.exists():
            for child in legacy_plot_scene.glob("*"):
                if child.is_file():
                    child.unlink()

    debug_root = (debug_output_dir / scene_id) if debug_output_dir is not None else scene_dir
    if debug_clean and debug_output_dir is not None and debug_root.exists():
        if debug_root.is_dir():
            shutil.rmtree(debug_root)
        else:
            debug_root.unlink()
    if debug_yaw or debug_bev:
        debug_root.mkdir(parents=True, exist_ok=True)

    for label_name, mode, label_path in iter_labels(scene_dir, label_filter):
        labels_seen += 1
        output_name = output_template.replace("{label}", label_name)
        output_path = scene_dir / output_name
        if skip_existing and not debug_clean:
            expected_paths = []
            if not skip_actions:
                expected_paths.append(output_path)
            if debug_yaw:
                expected_paths.append(debug_root / debug_yaw_template.replace("{label}", label_name))
                if debug_yaw_plot:
                    expected_paths.append(
                        debug_root / debug_yaw_plot_template.replace("{label}", label_name)
                    )
            if debug_bev:
                expected_paths.append(debug_root / debug_bev_template.replace("{label}", label_name))
            if expected_paths and all(path.exists() for path in expected_paths):
                log(f"[skip] {scene_id}/{label_name}: outputs already exist")
                continue
        if clean:
            if mode == "legacy_dir":
                legacy_json = label_path / "frame_actions.json"
                if legacy_json.exists():
                    legacy_json.unlink()
            if output_path.exists():
                output_path.unlink()

        camera_payloads: list[tuple[int, dict]] = []
        if mode == "path_json":
            for entry in _iter_camera_payloads_from_path_json(label_path):
                camera_payloads.append((int(entry.get("frame", 0)), entry))
        else:
            camera_files = []
            for path in label_path.iterdir():
                idx = parse_frame_index(path)
                if idx is not None:
                    camera_files.append((idx, path))
            if not camera_files:
                log(f"[label] {scene_id}/{label_name}: no camera frames found")
                continue
            camera_files.sort(key=lambda item: item[0])
            for frame_idx, cam_path in camera_files:
                with cam_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                camera_payloads.append((int(frame_idx), payload))

        if not camera_payloads:
            log(f"[label] {scene_id}/{label_name}: no camera frames found")
            continue

        frames = []
        for frame_idx, payload in camera_payloads:
            cam_center = payload.get("camera_center_world")
            cam_to_world = payload.get("camera_to_world")
            if not cam_center or not cam_to_world:
                log(f"[frame] {scene_id}/{label_name}/frame={frame_idx}: missing camera data")
                continue

            x, y, z = map(float, cam_center[:3])
            forward_row = cam_to_world[2]
            fx = float(forward_row[0])
            fy = float(forward_row[1])
            norm = math.hypot(fx, fy)
            if norm < 1e-6:
                fx, fy = 0.0, 1.0
            else:
                fx, fy = fx / norm, fy / norm
            yaw = math.atan2(fy, fx)

            pixel = None
            if meta is not None:
                u, v = world_to_pixel(meta, x, y)
                pixel = [int(u), int(v)]

            frames.append(
                {
                    "frame": int(frame_idx),
                    "world": [x, y, z],
                    "pixel": pixel,
                    "forward": [fx, fy],
                    "yaw": yaw,
                }
            )

        if len(frames) < 1:
            log(f"[label] {scene_id}/{label_name}: no usable frames after parsing")
            continue
        log(f"[label] {scene_id}/{label_name}: parsed {len(frames)} frames")

        if debug_yaw:
            debug_name = debug_yaw_template.replace("{label}", label_name)
            debug_path = debug_root / debug_name
            yaw_windows = {
                str(step): compute_yaw_window_series(frames, step)
                for step in sorted(debug_yaw_window_steps)
            }
            debug_payload = {
                "dataset_root": str(scene_dir.parent),
                "scene": scene_id,
                "label": label_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "frames": [
                    {
                        "frame": frame["frame"],
                        "yaw_rad": frame["yaw"],
                        "yaw_deg": math.degrees(frame["yaw"]),
                        "world": frame["world"],
                        "pixel": frame["pixel"],
                        "forward": frame["forward"],
                    }
                    for frame in frames
                ],
                "yaw_windows": yaw_windows,
            }
            debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
            outputs_written += 1
            log(f"[debug] wrote yaw report to {debug_path}")
            if debug_yaw_plot:
                plot_name = debug_yaw_plot_template.replace("{label}", label_name)
                plot_path = debug_root / plot_name
                mpl_config_dir = debug_root / ".mplconfig"
                render_yaw_windows_plot(yaw_windows, plot_path, mpl_config_dir)
                outputs_written += 1
                log(f"[debug] wrote yaw window plot to {plot_path}")

        if debug_bev:
            if meta is None:
                log(f"[debug] skipping BEV overlay for {scene_id}/{label_name}: missing metadata")
            else:
                occ_png = scene_meta_dir / "occupancy.png"
                if not occ_png.is_file():
                    log(f"[debug] skipping BEV overlay for {scene_id}/{label_name}: {occ_png} missing")
                else:
                    debug_name = debug_bev_template.replace("{label}", label_name)
                    debug_path = debug_root / debug_name
                    render_bev_debug(
                        occ_png,
                        frames,
                        debug_path,
                        arrow_step=debug_arrow_step,
                        arrow_len_px=debug_arrow_len_px,
                        arrow_width_px=debug_arrow_width_px,
                        mirror_center=debug_bev_mirror_center,
                    )
                    outputs_written += 1
                    log(f"[debug] wrote BEV overlay to {debug_path}")

        actions: list[str] = []
        per_frame: list[dict] = []
        for i, frame in enumerate(frames):
            if i == len(frames) - 1:
                actions.append("stop")
                continue

            curr = frame
            nxt = frames[i + 1]
            yaw_idx = min(i + action_window, len(frames) - 1)
            yaw_target = frames[yaw_idx]
            delta_yaw = signed_angle_delta(curr["yaw"], yaw_target["yaw"])
            angle_deg = abs(math.degrees(delta_yaw))

            dx = nxt["world"][0] - curr["world"][0]
            dy = nxt["world"][1] - curr["world"][1]
            dot = dx * curr["forward"][0] + dy * curr["forward"][1]
            ahead = dot > float(ahead_dot_eps)

            if angle_deg >= action_turn_threshold:
                actions.append("turn left" if delta_yaw > 0 else "turn right")
            elif angle_deg <= float(move_threshold_deg) and ahead:
                actions.append("move")
            else:
                actions.append("move" if ahead else "stop")

        action_codes = {
            "stop": 0,
            "move": 1,
            "turn left": 2,
            "turn right": 3,
        }

        for i, frame in enumerate(frames):
            next_actions = []
            for j in range(1, max_next + 1):
                if i + j < len(actions):
                    next_actions.append(action_codes[actions[i + j]])
                else:
                    next_actions.append(action_codes["stop"])

            per_frame.append(
                {
                    "frame": frame["frame"],
                    "world": frame["world"],
                    "pixel": frame["pixel"],
                    "curr_action": action_codes[actions[i]],
                    "next_actions": next_actions,
                }
            )

        payload = {
            "dataset_root": str(scene_dir.parent),
            "scene": scene_id,
            "label": label_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "max_next": max_next,
            "move_threshold_deg": float(move_threshold_deg),
            "turn_threshold_deg": float(turn_threshold_deg),
            "turn_threshold_scale": float(turn_threshold_scale),
            "action_turn_threshold_deg": action_turn_threshold,
            "action_yaw_window": action_window,
            "frames": per_frame,
        }
        if not skip_actions:
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            outputs_written += 1
            log(f"[write] {output_path}")
        else:
            log(f"[skip] action output suppressed for {scene_id}/{label_name}")

        if plots and len(frames) > step:
            plot_name_expanded = plot_name
            if "{skip}" in plot_name_expanded:
                plot_name_expanded = plot_name_expanded.replace("{skip}", str(skip_frames))
            plot_name_final = (
                plot_name_expanded.replace("{label}", label_name)
                if "{label}" in plot_name_expanded
                else plot_name_expanded
            )
            plot_path = plots_root / scene_id / plot_name_final
            xs, ys = compute_yaw_delta_series(frames, step)
            title = f"Yaw Delta vs Frame (skip {skip_frames} frames)"
            mpl_config_dir = scene_dir.parent / ".mplconfig"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            render_plot(xs, ys, title, plot_path, mpl_config_dir)
            log(f"[plot] {plot_path}")

    return scene_id, labels_seen, outputs_written


def process_scene_args(args: tuple) -> tuple[str, int, int]:
    return process_scene(*args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export per-frame actions from a rendered dataset."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory of the rendered dataset (scene folders underneath).",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="{label}_actions.json",
        help="Output filename template placed under each scene (default: {label}_actions.json).",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="yaw_delta_{label}_skip{skip}.png",
        help="Per-path plot filename; supports {label} and {skip} placeholders.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Optional scene filter. Repeat to restrict processing.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label filter. Repeat to restrict processing.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "scenes",
        help="Directory containing scene occupancy metadata.",
    )
    parser.add_argument(
        "--max-next",
        type=int,
        default=8,
        help="Number of future actions to include per frame (default: 8).",
    )
    parser.add_argument(
        "--action-yaw-window",
        type=int,
        default=5,
        help="Frame lookahead for yaw delta when classifying turns (default: 5).",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=4,
        help="Number of frames to skip when plotting yaw deltas (default: 4).",
    )
    parser.add_argument(
        "--move-threshold-deg",
        type=float,
        default=10.0,
        help="Max yaw change for a move action (default: 10).",
    )
    parser.add_argument(
        "--turn-threshold-deg",
        type=float,
        default=15.0,
        help="Min yaw change for a turn action (default: 15).",
    )
    parser.add_argument(
        "--turn-threshold-scale",
        type=float,
        default=0.5,
        help="Scale factor applied to --turn-threshold-deg for action classification (default: 0.5).",
    )
    parser.add_argument(
        "--ahead-dot-eps",
        type=float,
        default=1e-6,
        help="Minimum forward dot-product to consider the next frame ahead.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel scene workers (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable plot generation (default: off).",
    )
    parser.add_argument(
        "--skip-actions",
        action="store_true",
        help="Skip writing action JSON outputs (debug-only runs).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip labels when expected outputs already exist.",
    )
    parser.add_argument(
        "--debug-yaw",
        action="store_true",
        help="Write per-frame yaw debug JSON for each label (default: off).",
    )
    parser.add_argument(
        "--debug-bev",
        action="store_true",
        help="Write BEV occupancy overlay with camera arrows (default: off).",
    )
    parser.add_argument(
        "--debug-yaw-template",
        type=str,
        default="{label}_yaw_debug.json",
        help="Filename template for yaw debug output (default: {label}_yaw_debug.json).",
    )
    parser.add_argument(
        "--debug-bev-template",
        type=str,
        default="{label}_yaw_bev.png",
        help="Filename template for BEV debug output (default: {label}_yaw_bev.png).",
    )
    parser.add_argument(
        "--debug-yaw-plot",
        action=BooleanOptionalAction,
        default=True,
        help="Write yaw window plot PNGs when --debug-yaw is set (default: True).",
    )
    parser.add_argument(
        "--debug-yaw-plot-template",
        type=str,
        default="{label}_yaw_windows.png",
        help="Filename template for yaw window plots (default: {label}_yaw_windows.png).",
    )
    parser.add_argument(
        "--debug-yaw-window-steps",
        type=str,
        default="5",
        help="Comma-separated frame steps for yaw window deltas (default: 5).",
    )
    parser.add_argument(
        "--debug-output-dir",
        type=Path,
        default=Path("analysis/frame_action_debug"),
        help="Directory for debug outputs (default: analysis/frame_action_debug).",
    )
    parser.add_argument(
        "--debug-clean",
        action="store_true",
        help="Remove existing debug outputs under --debug-output-dir before writing.",
    )
    parser.add_argument(
        "--debug-arrow-step",
        type=int,
        default=1,
        help="Draw arrows every N frames on the BEV overlay (default: 1).",
    )
    parser.add_argument(
        "--debug-arrow-len-px",
        type=int,
        default=10,
        help="Arrow length in pixels for BEV overlay (default: 10).",
    )
    parser.add_argument(
        "--debug-arrow-width-px",
        type=int,
        default=2,
        help="Arrow width in pixels for BEV overlay (default: 2).",
    )
    parser.add_argument(
        "--debug-bev-mirror-center",
        action=BooleanOptionalAction,
        default=True,
        help="Mirror BEV debug overlay around image center (default: True).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous outputs before writing new results.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    max_next = max(0, int(args.max_next))
    if args.action_yaw_window <= 0:
        raise SystemExit("[ERROR] --action-yaw-window must be a positive integer.")
    if args.turn_threshold_scale <= 0:
        raise SystemExit("[ERROR] --turn-threshold-scale must be a positive number.")
    try:
        debug_yaw_window_steps = parse_window_steps(args.debug_yaw_window_steps)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    skip_scene_names = {"plots", ".mplconfig"}
    scene_filter = set(args.scene) if args.scene else None
    label_filter = set(args.label) if args.label else None
    scene_dirs = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir() or p.name in skip_scene_names:
            continue
        if scene_filter and p.name not in scene_filter:
            continue
        scene_dirs.append(p)
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {dataset_root}")

    if args.verbose:
        print(
            f"[start] scenes={len(scene_dirs)} workers={args.workers} skip_frames={args.skip_frames} plots={args.plots}",
            flush=True,
        )

    from multiprocessing import Pool

    total_labels = 0
    outputs_written = 0
    work_items = [
        (
            scene_dir,
            args.scenes_dir,
            args.output_template,
            args.plot_name,
            max_next,
            args.skip_frames,
            args.move_threshold_deg,
            args.turn_threshold_deg,
            args.ahead_dot_eps,
            args.action_yaw_window,
            args.turn_threshold_scale,
            args.verbose,
            args.plots,
            args.clean,
            label_filter,
            args.skip_actions,
            args.debug_yaw,
            args.debug_bev,
            args.debug_yaw_template,
            args.debug_bev_template,
            args.debug_yaw_plot_template,
            debug_yaw_window_steps,
            args.debug_output_dir,
            args.debug_clean,
            args.debug_arrow_step,
            args.debug_arrow_len_px,
            args.debug_arrow_width_px,
            args.debug_bev_mirror_center,
            args.debug_yaw_plot,
            args.skip_existing,
        )
        for scene_dir in scene_dirs
    ]

    if args.workers <= 1:
        for item in work_items:
            scene_id, labels_seen, written = process_scene_args(item)
            total_labels += labels_seen
            outputs_written += written
            if args.verbose:
                print(
                    f"[progress] scene={scene_id} labels={labels_seen} outputs={written}",
                    flush=True,
                )
    else:
        with Pool(processes=args.workers) as pool:
            for scene_id, labels_seen, written in pool.imap_unordered(process_scene_args, work_items):
                total_labels += labels_seen
                outputs_written += written
                if args.verbose:
                    print(
                        f"[progress] scene={scene_id} labels={labels_seen} outputs={written}",
                        flush=True,
                    )

    if outputs_written == 0:
        raise RuntimeError(f"No camera frame JSONs found under {dataset_root}")
    if args.verbose:
        print(
            f"[summary] scenes={len(scene_dirs)} labels={total_labels} outputs={outputs_written}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
