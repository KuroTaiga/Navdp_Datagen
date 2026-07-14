#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from argparse import BooleanOptionalAction
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from lighting.lighting_utils import (  # noqa: E402
    LightFilterConfig,
    apply_light_filter,
    color_temperature_to_rgb,
)
from utils.video_writer_utils import VideoWriterBackend, make_video_writer  # noqa: E402


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


def _iter_mp4s(root: Path, pattern: str, *, progress_every: int = 0):
    start = time.perf_counter()
    for idx, path in enumerate(root.rglob(pattern), start=1):
        yield path
        if progress_every > 0 and idx % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(f"[SCAN] discovered {idx} files ({rate:.1f}/s)...", flush=True)


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


def _format_bytes(value: float | int | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}TB"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0.0, float(seconds))
    mins, sec = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{sec:02d}s"
    return f"{mins:02d}m{sec:02d}s"


def _format_rate(count: float | int, elapsed: float, suffix: str) -> str:
    if elapsed <= 0:
        return "-"
    return f"{float(count) / elapsed:.2f}{suffix}"


def _disk_usage(path: Path) -> tuple[int, int, int]:
    usage = shutil.disk_usage(path)
    return usage.total, usage.used, usage.free


def _done_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".done.json")


def _load_done(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _done_matches(
    done: dict | None,
    *,
    src_stat: os.stat_result,
    preset_name: str,
    scale: float,
    temp_k: float,
    vignette: float,
    max_frames: int | None,
) -> bool:
    if not done:
        return False
    try:
        if int(done.get("src_size", -1)) != int(src_stat.st_size):
            return False
        if int(done.get("src_mtime_ns", -1)) != int(src_stat.st_mtime_ns):
            return False
        if str(done.get("preset", "")) != str(preset_name):
            return False
        if abs(float(done.get("scale", 0.0)) - float(scale)) > 1e-6:
            return False
        if abs(float(done.get("temp_k", 0.0)) - float(temp_k)) > 1e-6:
            return False
        if abs(float(done.get("vignette", 0.0)) - float(vignette)) > 1e-6:
            return False
        if done.get("max_frames") is None and max_frames is None:
            return True
        return int(done.get("max_frames", -1)) == int(max_frames or -1)
    except Exception:
        return False


def _write_done(
    path: Path,
    *,
    src_path: Path,
    src_stat: os.stat_result,
    preset_name: str,
    scale: float,
    temp_k: float,
    vignette: float,
    max_frames: int | None,
    frames: int,
    output_bytes: int,
) -> None:
    payload = {
        "src": str(src_path),
        "src_size": int(src_stat.st_size),
        "src_mtime_ns": int(src_stat.st_mtime_ns),
        "preset": str(preset_name),
        "scale": float(scale),
        "temp_k": float(temp_k),
        "vignette": float(vignette),
        "max_frames": int(max_frames) if max_frames is not None else None,
        "frames": int(frames),
        "output_bytes": int(output_bytes),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, indent=2))


def _safe_unlink(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception:
        return False


def _cleanup_latest_outputs(output_dirs: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    candidates: list[Path] = []
    for out_dir in output_dirs:
        if not out_dir.is_dir():
            continue
        candidates.extend(out_dir.rglob("*.mp4"))
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    removed: list[Path] = []
    for path in candidates[:count]:
        _safe_unlink(_done_path(path))
        if _safe_unlink(path):
            removed.append(path)
    return removed


def _progress_line(
    *,
    completed: int,
    total: int | None,
    scanned: int | None,
    elapsed: float,
    frames_total: int,
    files_skipped: int,
    outputs_written: int,
    output_bytes: int,
    output_root: Path,
    files_with_outputs: int,
) -> str:
    rate = _format_rate(completed, elapsed, "/s")
    fps = _format_rate(frames_total, elapsed, " fps")
    eta = None
    if total is not None and completed > 0:
        avg_sec = elapsed / completed
        eta = avg_sec * max(total - completed, 0)
    avg_out_bytes = (
        output_bytes / files_with_outputs if files_with_outputs > 0 else None
    )
    est_remaining = (
        avg_out_bytes * max(total - completed, 0)
        if (avg_out_bytes is not None and total is not None)
        else None
    )
    total_bytes, _, free_bytes = _disk_usage(output_root)
    if total is None:
        head = f"[PROGRESS] {completed} done"
        if scanned is not None:
            head += f" (scanned {scanned})"
    else:
        head = f"[PROGRESS] {completed}/{total} files"
    return (
        f"{head} (skipped {files_skipped}) | "
        f"rate {rate} | eta {_format_eta(eta)} | "
        f"frames {frames_total} ({fps}) | "
        f"outputs {outputs_written} | "
        f"out {_format_bytes(output_bytes)} (avg {_format_bytes(avg_out_bytes)}/file, "
        f"est rem {_format_bytes(est_remaining)}) | "
        f"disk free {_format_bytes(free_bytes)}/{_format_bytes(total_bytes)}"
    )


GPU_VIDEO_FORMAT = "ABGR"


def _torch_from_frame(frame: np.ndarray, device) -> "torch.Tensor":
    import torch

    if frame.ndim != 3:
        raise ValueError(f"Expected frame with 3 dims, got shape {frame.shape}")
    if frame.shape[2] == 4:
        frame = frame[..., :3]
    tensor = torch.as_tensor(frame, device=device)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    else:
        tensor = tensor.to(torch.float32)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
    return tensor


def _prepare_vignette_base(height: int, width: int, device) -> tuple["torch.Tensor", float]:
    import torch

    center_x = 0.5 * float(max(width - 1, 1))
    center_y = 0.5 * float(max(height - 1, 1))
    yy = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    rr = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    corners = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, float(height - 1)],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
        ],
        device=device,
        dtype=torch.float32,
    )
    corner_dist = torch.sqrt((corners[:, 0] - center_x) ** 2 + (corners[:, 1] - center_y) ** 2)
    rmax = float(corner_dist.max().item()) if corner_dist.numel() else 1.0
    if rmax <= 0.0:
        rmax = 1.0
    return rr, rmax


def _apply_global_filter_gpu(
    frame: "torch.Tensor",
    *,
    scale: float,
    temp_rgb: "torch.Tensor | None",
    vignette: float,
    rr: "torch.Tensor | None",
    rmax: float,
) -> "torch.Tensor":
    import torch

    out = frame * float(scale)
    if vignette > 0.0 and rr is not None:
        mask = 1.0 - float(vignette) * (rr / float(rmax)) ** 2
        out = out * mask.clamp(0.0, 1.0).unsqueeze(-1)
    if temp_rgb is not None:
        out = out * temp_rgb
    return out.clamp(0.0, 1.0)


def _to_gpu_video_frame(img: "torch.Tensor", gpu_format: str) -> "torch.Tensor":
    import torch

    gpu_format = gpu_format.upper()
    render_uint8 = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    if gpu_format == "ABGR":
        rgb = render_uint8
    elif gpu_format == "ARGB":
        rgb = render_uint8[..., [2, 1, 0]]
    else:
        raise ValueError(f"Unsupported GPU video format: {gpu_format}")
    alpha = torch.full(
        (rgb.shape[0], rgb.shape[1], 1),
        255,
        device=rgb.device,
        dtype=torch.uint8,
    )
    return torch.cat([rgb, alpha], dim=2).contiguous()


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
    resume: bool,
    compute_backend: str,
    video_backend: str,
    video_nvenc_preset: str | None,
    video_nvenc_bitrate: str | None,
    max_frames: int | None,
) -> dict:
    src_path = Path(src_path_str)
    rel_path = Path(rel_path_str)
    start = time.perf_counter()
    src_stat = src_path.stat()
    reader = imageio.get_reader(src_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 10)
    writers: dict[str, imageio.core.format.Writer] = {}
    configs: dict[str, LightFilterConfig] = {}
    outputs_written: list[str] = []
    outputs_skipped: list[str] = []
    outputs_removed: list[str] = []
    out_paths: dict[str, Path] = {}
    output_bytes = 0
    tone_by_name = {name: (scale, temp_k, vignette) for name, _, scale, temp_k, vignette in tone_items}
    use_gpu_video = str(video_backend).lower() == VideoWriterBackend.GPU.value
    use_gpu_compute = str(compute_backend).lower() == "gpu" or use_gpu_video
    torch_device = None
    rr = None
    rmax = 1.0
    temp_rgb_map: dict[str, "torch.Tensor | None"] = {}
    tone_scale_map: dict[str, float] = {}
    tone_vignette_map: dict[str, float] = {}
    success = False
    try:
        for name, out_dir_str, scale, temp_k, vignette in tone_items:
            out_path = Path(out_dir_str) / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                if resume:
                    done_path = _done_path(out_path)
                    done = _load_done(done_path)
                    if done is None:
                        outputs_skipped.append(name)
                        try:
                            output_bytes += out_path.stat().st_size
                        except Exception:
                            pass
                        continue
                    if _done_matches(
                        done,
                        src_stat=src_stat,
                        preset_name=name,
                        scale=scale,
                        temp_k=temp_k,
                        vignette=vignette,
                        max_frames=max_frames,
                    ):
                        outputs_skipped.append(name)
                        try:
                            output_bytes += out_path.stat().st_size
                        except Exception:
                            pass
                        continue
                    removed = _safe_unlink(out_path)
                    _safe_unlink(done_path)
                    if not removed and out_path.exists():
                        outputs_skipped.append(name)
                        continue
                    outputs_removed.append(name)
                else:
                    outputs_skipped.append(name)
                    continue
            elif resume:
                done_path = _done_path(out_path)
                if done_path.exists() and not out_path.exists():
                    _safe_unlink(done_path)
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
            outputs_written.append(name)
            out_paths[name] = out_path
            tone_scale_map[name] = float(scale)
            tone_vignette_map[name] = float(vignette)

        if not out_paths:
            return {
                "src": str(src_path),
                "rel": str(rel_path),
                "frames": 0,
                "outputs_written": [],
                "outputs_skipped": outputs_skipped,
                "outputs_removed": outputs_removed,
                "output_bytes": output_bytes,
                "elapsed_sec": 0.0,
                "skipped": True,
            }

        frames = 0
        for frame_index, frame in enumerate(reader):
            if max_frames is not None and frames >= max_frames:
                break
            if not writers:
                height, width = frame.shape[:2]
                for name, out_path in out_paths.items():
                    writers[name] = make_video_writer(
                        out_path,
                        fps=fps,
                        backend=video_backend,
                        nvenc_preset=video_nvenc_preset,
                        nvenc_bitrate=video_nvenc_bitrate,
                        width=width,
                        height=height,
                        gpu_format=GPU_VIDEO_FORMAT,
                    )
                if use_gpu_compute:
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError("GPU compute requested but CUDA is not available.")
                    torch_device = torch.device("cuda")
                    rr, rmax = _prepare_vignette_base(height, width, torch_device)
                    for name, (_, temp_k, _) in tone_by_name.items():
                        if temp_k > 0.0:
                            temp = color_temperature_to_rgb(float(temp_k))
                            temp_rgb_map[name] = torch.tensor(
                                temp, device=torch_device, dtype=torch.float32
                            )
                        else:
                            temp_rgb_map[name] = None

            if use_gpu_compute:
                import torch

                if torch_device is None:
                    raise RuntimeError("GPU device not initialized.")
                frame_tensor = _torch_from_frame(frame, torch_device)
                for name, writer in writers.items():
                    out = _apply_global_filter_gpu(
                        frame_tensor,
                        scale=tone_scale_map[name],
                        temp_rgb=temp_rgb_map.get(name),
                        vignette=tone_vignette_map[name],
                        rr=rr,
                        rmax=rmax,
                    )
                    if use_gpu_video:
                        writer.append_data(_to_gpu_video_frame(out, GPU_VIDEO_FORMAT))
                    else:
                        writer.append_data(
                            (out.clamp(0.0, 1.0) * 255.0)
                            .to(torch.uint8)
                            .cpu()
                            .numpy()
                        )
            else:
                for name, writer in writers.items():
                    filtered = apply_light_filter(frame, configs[name], frame_index=frame_index)
                    writer.append_data(_frame_to_uint8(filtered))
            frames += 1
        success = True
    finally:
        reader.close()
        for writer in writers.values():
            writer.close()
        if success:
            for name, out_path in out_paths.items():
                if not out_path.exists():
                    continue
                size = out_path.stat().st_size
                output_bytes += size
                scale, temp_k, vignette = tone_by_name.get(name, (0.0, 0.0, 0.0))
                _write_done(
                    _done_path(out_path),
                    src_path=src_path,
                    src_stat=src_stat,
                    preset_name=name,
                    scale=scale,
                    temp_k=temp_k,
                    vignette=vignette,
                    max_frames=max_frames,
                    frames=frames,
                    output_bytes=size,
                )
    elapsed = time.perf_counter() - start
    return {
        "src": str(src_path),
        "rel": str(rel_path),
        "frames": frames,
        "outputs_written": outputs_written,
        "outputs_skipped": outputs_skipped,
        "outputs_removed": outputs_removed,
        "output_bytes": output_bytes,
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
        "--resume",
        action=BooleanOptionalAction,
        default=True,
        help="Resume by skipping verified outputs and cleaning mismatched files (default: on).",
    )
    parser.add_argument(
        "--cleanup-latest",
        type=int,
        default=0,
        help="Delete the most recent N output MP4s before processing (default: 0).",
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
        "--compute-backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Compute backend for filtering (default: cpu).",
    )
    parser.add_argument(
        "--video-backend",
        choices=[backend.value for backend in VideoWriterBackend],
        default=VideoWriterBackend.CPU.value,
        help="Video writer backend (default: cpu).",
    )
    parser.add_argument(
        "--video-nvenc-preset",
        type=str,
        default=None,
        help="NVENC preset when --video-backend=nvenc (example: p4, slow).",
    )
    parser.add_argument(
        "--video-nvenc-bitrate",
        type=str,
        default=None,
        help="NVENC bitrate when --video-backend=nvenc (example: 10M).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N MP4s (default: 10).",
    )
    parser.add_argument(
        "--scan-mode",
        choices=("stream", "sorted"),
        default="stream",
        help="MP4 discovery mode: stream starts processing while scanning; sorted matches old behavior.",
    )
    parser.add_argument(
        "--scan-progress-every",
        type=int,
        default=0,
        help="Print scan progress every N discovered files (stream mode only; default: 0).",
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

    if str(args.video_backend).lower() == VideoWriterBackend.GPU.value and str(args.compute_backend).lower() != "gpu":
        print("[WARN] video-backend=gpu forces compute-backend=gpu.", flush=True)
        args.compute_backend = "gpu"
    if str(args.compute_backend).lower() == "gpu" or str(args.video_backend).lower() == VideoWriterBackend.GPU.value:
        if int(args.workers) > 1:
            print(
                "[WARN] GPU backend with multiple workers can oversubscribe the device; "
                "consider --workers 1.",
                flush=True,
            )

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
    cleanup_removed = 0
    if int(args.cleanup_latest) > 0:
        removed = _cleanup_latest_outputs(list(output_dirs.values()), int(args.cleanup_latest))
        cleanup_removed = len(removed)
        print(
            f"[CLEANUP] Removed {len(removed)} output MP4(s) (latest first).",
            flush=True,
        )

    max_files = int(args.max_files) if args.max_files is not None else None
    scan_mode = str(args.scan_mode)
    total_known: int | None = None
    mp4s: list[Path] | None = None

    if args.mp4_list is not None:
        mp4s = _load_mp4_list(args.mp4_list, input_dir)
        if max_files is not None:
            mp4s = mp4s[:max_files]
        if not mp4s:
            raise SystemExit(f"No MP4s listed in {args.mp4_list}")
        mp4_iter = iter(mp4s)
        total_known = len(mp4s)
    elif scan_mode == "sorted":
        print(f"[SCAN] collecting mp4s under {input_dir} (sorted)...", flush=True)
        mp4s = _collect_mp4s(input_dir, args.pattern)
        if not mp4s:
            raise SystemExit(f"No MP4s matched {args.pattern} under {input_dir}")
        if max_files is not None:
            mp4s = mp4s[:max_files]
        mp4_iter = iter(mp4s)
        total_known = len(mp4s)
    else:
        print(f"[SCAN] streaming mp4 discovery under {input_dir}...", flush=True)
        mp4_iter = _iter_mp4s(
            input_dir,
            args.pattern,
            progress_every=max(0, int(args.scan_progress_every)),
        )

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
    capture_mp4s = args.other_mode == "copy"
    mp4s_for_copy: list[Path] = []

    start = time.perf_counter()
    reports: list[dict] = []
    total_frames = 0
    files_skipped = 0
    outputs_per_preset = {preset["name"]: 0 for preset in presets}
    outputs_skipped_per_preset = {preset["name"]: 0 for preset in presets}
    outputs_removed_per_preset = {preset["name"]: 0 for preset in presets}
    outputs_written_total = 0
    output_bytes_total = 0
    files_with_outputs = 0
    scanned = 0
    completed = 0

    progress_every = max(1, int(args.progress_every)) if int(args.progress_every) > 0 else 0
    if int(args.workers) > 1:
        max_inflight = max(4, int(args.workers) * 4)
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            inflight = set()

            def handle_done(done_future) -> None:
                nonlocal completed, total_frames, files_skipped, outputs_written_total, output_bytes_total, files_with_outputs
                report = done_future.result()
                reports.append(report)
                total_frames += report["frames"]
                if report.get("skipped"):
                    files_skipped += 1
                for name in report.get("outputs_written", []):
                    outputs_per_preset[name] += 1
                for name in report.get("outputs_skipped", []):
                    outputs_skipped_per_preset[name] += 1
                for name in report.get("outputs_removed", []):
                    outputs_removed_per_preset[name] += 1
                outputs_written_total += len(report.get("outputs_written", []))
                output_bytes_total += int(report.get("output_bytes", 0))
                if report.get("output_bytes", 0) > 0:
                    files_with_outputs += 1
                completed += 1
                if progress_every and completed % progress_every == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        _progress_line(
                            completed=completed,
                            total=total_known,
                            scanned=scanned,
                            elapsed=elapsed,
                            frames_total=total_frames,
                            files_skipped=files_skipped,
                            outputs_written=outputs_written_total,
                            output_bytes=output_bytes_total,
                            output_root=output_root,
                            files_with_outputs=files_with_outputs,
                        ),
                        flush=True,
                    )

            for path in mp4_iter:
                if max_files is not None and scanned >= max_files:
                    break
                scanned += 1
                if capture_mp4s:
                    mp4s_for_copy.append(path)
                rel = path.relative_to(input_dir)
                inflight.add(
                    executor.submit(
                        _process_mp4_task,
                        str(path),
                        str(rel),
                        tone_items,
                        overwrite=args.overwrite,
                        resume=bool(args.resume),
                        compute_backend=str(args.compute_backend),
                        video_backend=str(args.video_backend),
                        video_nvenc_preset=args.video_nvenc_preset,
                        video_nvenc_bitrate=args.video_nvenc_bitrate,
                        max_frames=args.max_frames,
                    )
                )
                if len(inflight) >= max_inflight:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for future in done:
                        handle_done(future)

            if scanned == 0:
                raise SystemExit(f"No MP4s matched {args.pattern} under {input_dir}")
            if total_known is None:
                total_known = scanned
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    handle_done(future)
    else:
        for path in mp4_iter:
            if max_files is not None and scanned >= max_files:
                break
            scanned += 1
            if capture_mp4s:
                mp4s_for_copy.append(path)
            report = _process_mp4_task(
                str(path),
                str(path.relative_to(input_dir)),
                tone_items,
                overwrite=args.overwrite,
                resume=bool(args.resume),
                compute_backend=str(args.compute_backend),
                video_backend=str(args.video_backend),
                video_nvenc_preset=args.video_nvenc_preset,
                video_nvenc_bitrate=args.video_nvenc_bitrate,
                max_frames=args.max_frames,
            )
            reports.append(report)
            total_frames += report["frames"]
            if report.get("skipped"):
                files_skipped += 1
            for name in report.get("outputs_written", []):
                outputs_per_preset[name] += 1
            for name in report.get("outputs_skipped", []):
                outputs_skipped_per_preset[name] += 1
            for name in report.get("outputs_removed", []):
                outputs_removed_per_preset[name] += 1
            outputs_written_total += len(report.get("outputs_written", []))
            output_bytes_total += int(report.get("output_bytes", 0))
            if report.get("output_bytes", 0) > 0:
                files_with_outputs += 1
            completed += 1
            if progress_every and completed % progress_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    _progress_line(
                        completed=completed,
                        total=total_known,
                        scanned=scanned,
                        elapsed=elapsed,
                        frames_total=total_frames,
                        files_skipped=files_skipped,
                        outputs_written=outputs_written_total,
                        output_bytes=output_bytes_total,
                        output_root=output_root,
                        files_with_outputs=files_with_outputs,
                    ),
                    flush=True,
                )
        if scanned == 0:
            raise SystemExit(f"No MP4s matched {args.pattern} under {input_dir}")
        if total_known is None:
            total_known = scanned

    if completed and progress_every and completed % progress_every != 0:
        elapsed = time.perf_counter() - start
        print(
            _progress_line(
                completed=completed,
                total=total_known,
                scanned=scanned,
                elapsed=elapsed,
                frames_total=total_frames,
                files_skipped=files_skipped,
                outputs_written=outputs_written_total,
                output_bytes=output_bytes_total,
                output_root=output_root,
                files_with_outputs=files_with_outputs,
            ),
            flush=True,
        )

    other_copied = 0
    if args.other_mode == "copy":
        other_copied = _copy_other_files(
            input_dir,
            list(output_dirs.values()),
            (mp4s if mp4s is not None else mp4s_for_copy),
            overwrite=args.overwrite,
        )

    elapsed = time.perf_counter() - start
    summary = {
        "input": str(input_dir),
        "pattern": args.pattern,
        "output_root": str(output_root),
        "output_dirs": {name: str(path) for name, path in output_dirs.items()},
        "presets": presets,
        "resume": bool(args.resume),
        "cleanup_latest": int(args.cleanup_latest),
        "cleanup_latest_removed": cleanup_removed,
        "compute_backend": str(args.compute_backend),
        "video_backend": str(args.video_backend),
        "video_nvenc_preset": args.video_nvenc_preset,
        "video_nvenc_bitrate": args.video_nvenc_bitrate,
        "files_found": int(total_known or 0),
        "files_processed": len(reports),
        "files_skipped": files_skipped,
        "outputs_per_preset": outputs_per_preset,
        "outputs_skipped_per_preset": outputs_skipped_per_preset,
        "outputs_removed_per_preset": outputs_removed_per_preset,
        "outputs_written_total": outputs_written_total,
        "output_bytes_accounted": output_bytes_total,
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
