#!/usr/bin/env python3
"""Benchmark TeleSim render output modes across video backends.

The helper intentionally runs each backend/mode in an isolated output directory
so the MP4s, logs, and metrics are easy to inspect after a host-side run.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDER_SCRIPT = REPO_ROOT / "render_label_paths_telesim.py"

MODE_FLAGS = {
    "video_only": ["--video", "--no-rgb-frames", "--no-save-depth-maps"],
    "video_rgb": ["--video", "--rgb-frames", "--no-save-depth-maps"],
    "video_depth": ["--video", "--no-rgb-frames", "--save-depth-maps"],
    "video_rgb_depth": ["--video", "--rgb-frames", "--save-depth-maps"],
    "rgb_only": ["--no-video", "--rgb-frames", "--no-save-depth-maps"],
    "depth_only": ["--no-video", "--no-rgb-frames", "--save-depth-maps"],
    "rgb_depth_only": ["--no-video", "--rgb-frames", "--save-depth-maps"],
}
VIDEO_MODES = {
    "video_only",
    "video_rgb",
    "video_depth",
    "video_rgb_depth",
}
DEFAULT_MODES = "video_only,video_rgb,video_depth,rgb_only,depth_only"
DEFAULT_BACKENDS = "nvenc,cpu,gpu"

STAGE_ORDER = (
    "actor_gpu_cache_upload_sec",
    "actor_visibility_sec",
    "actor_transform_sec",
    "actor_tensor_pack_sec",
    "actor_merge_update_sec",
    "gaussian_render_sec",
    "gpu_readback_sec",
    "perframe_light_sec",
    "camera_metadata_sec",
    "perframe_depth_sec",
    "perframe_png_sec",
    "mp4_write_sec",
    "h264_encode_sec",
    "h264_mux_sec",
    "video_close_sec",
    "render",
    "encode",
)
ALIAS_STAGE_KEYS = {"render", "encode", "measured_total_sec"}


@dataclass(frozen=True)
class RunSpec:
    mode: str
    backend: str

    @property
    def name(self) -> str:
        return f"{self.mode}__{self.backend}"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or suffix == "TiB":
            return f"{size:.1f} {suffix}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _pct_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline <= 0.0:
        return "-"
    delta = value - baseline
    return f"{delta:+.3f}s ({(delta / baseline) * 100.0:+.1f}%)"


def _discover_scene(scenes_dir: Path, tasks_dir: Path) -> str:
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"scenes_dir does not exist: {scenes_dir}")
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks_dir does not exist: {tasks_dir}")
    task_scenes = {path.name for path in tasks_dir.iterdir() if path.is_dir()}
    scene_dirs = {path.name for path in scenes_dir.iterdir() if path.is_dir()}
    for scene in sorted(task_scenes & scene_dirs):
        return scene
    raise FileNotFoundError(
        f"Could not auto-select a scene shared by {scenes_dir} and {tasks_dir}; pass --scene."
    )


def _build_run_specs(
    *,
    modes: Iterable[str],
    backends: Iterable[str],
    repeat_no_video_per_backend: bool,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    backend_list = list(backends)
    for mode in modes:
        if mode not in MODE_FLAGS:
            known = ", ".join(sorted(MODE_FLAGS))
            raise ValueError(f"Unknown mode '{mode}'. Known modes: {known}")
        if mode in VIDEO_MODES or repeat_no_video_per_backend:
            specs.extend(RunSpec(mode=mode, backend=backend) for backend in backend_list)
        else:
            specs.append(RunSpec(mode=mode, backend="no_video"))
    return specs


def _sum_stage_seconds(metrics: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    alias_totals: dict[str, float] = {}
    for path_entry in metrics.get("paths") or []:
        for stage, value in (path_entry.get("stage_seconds") or {}).items():
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            if stage in ALIAS_STAGE_KEYS:
                if stage != "measured_total_sec":
                    alias_totals[stage] = alias_totals.get(stage, 0.0) + seconds
                continue
            totals[stage] = totals.get(stage, 0.0) + seconds
    return totals or alias_totals


def _format_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _find_output_videos(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return [str(path) for path in sorted(output_dir.rglob("*.mp4"))]


def _output_bytes(paths: Iterable[str]) -> int:
    total = 0
    for path_str in paths:
        try:
            total += Path(path_str).stat().st_size
        except OSError:
            pass
    return total


def _load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.is_file():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _command_for_run(
    *,
    args: argparse.Namespace,
    spec: RunSpec,
    scene: str,
    run_output_dir: Path,
    metrics_path: Path,
    error_log: Path,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        str(args.python_bin),
        str(args.render_script),
        "--scenes-dir",
        str(args.scenes_dir),
        "--tasks-dir",
        str(args.tasks_dir),
        "--scene",
        scene,
        "--output-dir",
        str(run_output_dir),
        "--metrics-json",
        str(metrics_path),
        "--error-log",
        str(error_log),
        "--overwrite",
        "--no-resume",
        "--minimal-frames",
        str(args.minimal_frames),
        "--resolution",
        str(args.resolution[0]),
        str(args.resolution[1]),
    ]
    if args.max_labels is not None:
        cmd.extend(["--max-labels", str(args.max_labels)])
    for label_id in args.label_id or []:
        cmd.extend(["--label-id", label_id])
    cmd.extend(MODE_FLAGS[spec.mode])
    if spec.mode in VIDEO_MODES or spec.backend != "no_video":
        cmd.extend(["--video-backend", spec.backend if spec.backend != "no_video" else args.backends[0]])
    if args.video_nvenc_preset:
        cmd.extend(["--video-nvenc-preset", args.video_nvenc_preset])
    if args.video_nvenc_bitrate:
        cmd.extend(["--video-nvenc-bitrate", args.video_nvenc_bitrate])
    if args.camera_metadata:
        cmd.append("--save-camera-metadata")
    else:
        cmd.append("--no-save-camera-metadata")
    if args.path_progress:
        cmd.append("--path-progress")
    else:
        cmd.append("--no-path-progress")
    cmd.extend(passthrough)
    return cmd


def _run_one(
    *,
    args: argparse.Namespace,
    spec: RunSpec,
    scene: str,
    output_root: Path,
    passthrough: list[str],
) -> dict:
    run_output_dir = output_root / "runs" / spec.name
    metrics_path = output_root / "metrics" / f"{spec.name}.json"
    log_path = output_root / "logs" / f"{spec.name}.log"
    error_log = output_root / "logs" / f"{spec.name}.errors.log"
    if args.clean_run_dirs and run_output_dir.exists():
        shutil.rmtree(run_output_dir)
    if args.clean_run_dirs:
        for stale_file in (metrics_path, error_log):
            try:
                stale_file.unlink()
            except FileNotFoundError:
                pass
    run_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _command_for_run(
        args=args,
        spec=spec,
        scene=scene,
        run_output_dir=run_output_dir,
        metrics_path=metrics_path,
        error_log=error_log,
        passthrough=passthrough,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.ffmpeg_bin is not None:
        env["IMAGEIO_FFMPEG_EXE"] = str(args.ffmpeg_bin)
        env["FFMPEG_BIN"] = str(args.ffmpeg_bin)
    if args.strict_gpu_backends:
        env["STRICT_GPU_BACKENDS"] = "1"
    if spec.backend == "gpu":
        env.setdefault("GPU_VIDEO_DISABLE_BFRAMES", "1")
        env.setdefault("GPU_VIDEO_CLONE", "1")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + _format_cmd(cmd) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            cmd,
            cwd=str(args.cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time_sec = time.perf_counter() - start
    metrics = _load_metrics(metrics_path)
    videos = _find_output_videos(run_output_dir)
    stage_totals = _sum_stage_seconds(metrics)
    frames_total = metrics.get("frames_total")
    paths_ok = int(metrics.get("paths_ok") or 0)
    paths_fatal = int(metrics.get("paths_fatal") or 0)
    paths_oom = int(metrics.get("paths_oom") or 0)
    paths_attempted = int(metrics.get("paths_attempted") or 0)
    metrics_ok = (paths_fatal == 0 and paths_oom == 0)
    if metrics and paths_attempted > 0:
        metrics_ok = metrics_ok and paths_ok > 0
    time_per_frame = (
        (wall_time_sec / float(frames_total))
        if frames_total not in (None, 0) and wall_time_sec > 0.0
        else None
    )
    return {
        "name": spec.name,
        "mode": spec.mode,
        "backend": spec.backend,
        "returncode": int(completed.returncode),
        "ok": completed.returncode == 0 and metrics_ok,
        "wall_time_sec": wall_time_sec,
        "wall_time_per_frame_sec": time_per_frame,
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "error_log": str(error_log),
        "output_dir": str(run_output_dir),
        "videos": videos,
        "video_bytes": _output_bytes(videos),
        "frames_total": int(frames_total or 0),
        "paths_total": int(metrics.get("paths_total") or 0),
        "paths_ok": paths_ok,
        "paths_fatal": paths_fatal,
        "paths_oom": paths_oom,
        "paths_attempted": paths_attempted,
        "duration_total_sec": metrics.get("duration_total_sec"),
        "time_per_frame_sec": metrics.get("time_per_frame_sec"),
        "h264_encode_total_sec": metrics.get("h264_encode_total_sec"),
        "h264_mux_total_sec": metrics.get("h264_mux_total_sec"),
        "stage_totals": stage_totals,
        "command": cmd,
    }


def _stage_value(run: dict, *stages: str) -> float | None:
    totals = run.get("stage_totals") or {}
    found = False
    value = 0.0
    for stage in stages:
        if stage in totals:
            found = True
            value += float(totals.get(stage) or 0.0)
    return value if found else None


def _stage_value_with_fallback(
    run: dict,
    stages: tuple[str, ...],
    fallback_stages: tuple[str, ...],
) -> float | None:
    value = _stage_value(run, *stages)
    return value if value is not None else _stage_value(run, *fallback_stages)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(output_root: Path, payload: dict) -> Path:
    report_path = output_root / "benchmark_report.md"
    runs = payload["runs"]
    run_rows = []
    for run in runs:
        status = "ok" if run["ok"] else f"failed ({run['returncode']})"
        run_rows.append(
            [
                run["name"],
                run["mode"],
                run["backend"],
                status,
                str(run["frames_total"]),
                (
                    f"ok={run.get('paths_ok', 0)} "
                    f"fatal={run.get('paths_fatal', 0)} "
                    f"oom={run.get('paths_oom', 0)}"
                ),
                _format_seconds(run["wall_time_sec"]),
                _format_seconds(run["wall_time_per_frame_sec"]),
                _format_bytes(run["video_bytes"]),
                str(len(run["videos"])),
            ]
        )

    stage_rows = []
    for run in runs:
        totals = run.get("stage_totals") or {}
        for stage in STAGE_ORDER:
            value = totals.get(stage)
            if value is None or abs(float(value)) < 1e-9:
                continue
            per_frame = (
                float(value) / float(run["frames_total"])
                if run["frames_total"] > 0
                else None
            )
            stage_rows.append(
                [
                    run["name"],
                    stage,
                    _format_seconds(float(value)),
                    _format_seconds(per_frame),
                ]
            )

    delta_rows = []
    by_mode: dict[str, dict[str, dict]] = {}
    for run in runs:
        if not run["ok"]:
            continue
        by_mode.setdefault(run["mode"], {})[run["backend"]] = run
    for mode, backend_runs in sorted(by_mode.items()):
        baseline = backend_runs.get(payload["baseline_backend"])
        if baseline is None:
            continue
        for backend, run in sorted(backend_runs.items()):
            if backend == payload["baseline_backend"]:
                continue
            delta_rows.append(
                [
                    mode,
                    backend,
                    _pct_delta(run["wall_time_sec"], baseline["wall_time_sec"]),
                    _pct_delta(
                        _stage_value_with_fallback(
                            run,
                            ("gaussian_render_sec", "gpu_readback_sec"),
                            ("render",),
                        ),
                        _stage_value_with_fallback(
                            baseline,
                            ("gaussian_render_sec", "gpu_readback_sec"),
                            ("render",),
                        ),
                    ),
                    _pct_delta(
                        _stage_value_with_fallback(
                            run,
                            ("mp4_write_sec", "h264_encode_sec"),
                            ("encode",),
                        ),
                        _stage_value_with_fallback(
                            baseline,
                            ("mp4_write_sec", "h264_encode_sec"),
                            ("encode",),
                        ),
                    ),
                    _pct_delta(
                        _stage_value(run, "video_close_sec"),
                        _stage_value(baseline, "video_close_sec"),
                    ),
                    _pct_delta(
                        _stage_value(run, "perframe_png_sec"),
                        _stage_value(baseline, "perframe_png_sec"),
                    ),
                    _pct_delta(
                        _stage_value(run, "perframe_depth_sec"),
                        _stage_value(baseline, "perframe_depth_sec"),
                    ),
                ]
            )

    video_rows = []
    for run in runs:
        for video_path in run["videos"]:
            video_rows.append([run["name"], run["backend"], video_path])

    lines = [
        "# Output Backend Benchmark",
        "",
        f"- Scene: `{payload['scene']}`",
        f"- Baseline backend: `{payload['baseline_backend']}`",
        f"- Output root: `{payload['output_root']}`",
        f"- Metrics JSON: `{payload['summary_json']}`",
        "",
        "## Runs",
        "",
        _markdown_table(
            [
                "Run",
                "Mode",
                "Backend",
                "Status",
                "Frames",
                "Paths",
                "Wall sec",
                "Wall sec/frame",
                "Video bytes",
                "Videos",
            ],
            run_rows,
        ),
        "",
        "## Stage Totals",
        "",
        _markdown_table(
            ["Run", "Stage", "Seconds", "Seconds/frame"],
            stage_rows or [["-", "-", "-", "-"]],
        ),
        "",
        "## Backend Deltas",
        "",
        _markdown_table(
            [
                "Mode",
                "Backend",
                "Wall vs baseline",
                "Render/readback vs baseline",
                "Video write/encode vs baseline",
                "Video close vs baseline",
                "PNG write vs baseline",
                "Depth write vs baseline",
            ],
            delta_rows or [["-", "-", "-", "-", "-", "-", "-", "-"]],
        ),
        "",
        "## Output Videos",
        "",
        _markdown_table(
            ["Run", "Backend", "Video"],
            video_rows or [["-", "-", "No MP4 outputs were produced."]],
        ),
        "",
        "## Logs",
        "",
        _markdown_table(
            ["Run", "Log", "Metrics"],
            [[run["name"], run["log_path"], run["metrics_path"]] for run in runs],
        ),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run TeleSim output backend benchmarks and summarize per-stage timings.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--render-script", type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--baseline-backend", default=None)
    parser.add_argument("--repeat-no-video-per-backend", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label-id", action="append", default=None)
    parser.add_argument("--max-labels", type=int, default=1)
    parser.add_argument("--minimal-frames", type=int, default=120)
    parser.add_argument("--resolution", type=int, nargs=2, default=(960, 720), metavar=("W", "H"))
    parser.add_argument("--video-nvenc-preset", default=None)
    parser.add_argument("--video-nvenc-bitrate", default=None)
    parser.add_argument("--ffmpeg-bin", type=Path, default=None)
    parser.add_argument("--camera-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-gpu-backends", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-run-dirs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    args.backends = _split_csv(args.backends)
    args.modes = _split_csv(args.modes)
    if not args.backends:
        raise ValueError("--backends must include at least one backend")
    if not args.modes:
        raise ValueError("--modes must include at least one mode")
    if args.baseline_backend is None:
        args.baseline_backend = args.backends[0]
    return args, passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(sys.argv[1:] if argv is None else argv)
    scene = args.scene or _discover_scene(args.scenes_dir, args.tasks_dir)
    specs = _build_run_specs(
        modes=args.modes,
        backends=args.backends,
        repeat_no_video_per_backend=bool(args.repeat_no_video_per_backend),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for spec in specs:
            cmd = _command_for_run(
                args=args,
                spec=spec,
                scene=scene,
                run_output_dir=args.output_root / "runs" / spec.name,
                metrics_path=args.output_root / "metrics" / f"{spec.name}.json",
                error_log=args.output_root / "logs" / f"{spec.name}.errors.log",
                passthrough=passthrough,
            )
            print("$ " + _format_cmd(cmd))
        return 0

    runs = []
    for spec in specs:
        print(f"[BENCH] run={spec.name}", flush=True)
        result = _run_one(
            args=args,
            spec=spec,
            scene=scene,
            output_root=args.output_root,
            passthrough=passthrough,
        )
        runs.append(result)
        status = "ok" if result["ok"] else f"failed rc={result['returncode']}"
        print(
            "[BENCH] "
            f"run={spec.name} status={status} wall={result['wall_time_sec']:.3f}s "
            f"frames={result['frames_total']} videos={len(result['videos'])}",
            flush=True,
        )
        if not result["ok"] and args.fail_fast:
            break

    summary_path = args.output_root / "benchmark_summary.json"
    payload = {
        "scene": scene,
        "output_root": str(args.output_root),
        "summary_json": str(summary_path),
        "baseline_backend": args.baseline_backend,
        "modes": args.modes,
        "backends": args.backends,
        "runs": runs,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = _write_report(args.output_root, payload)
    print(f"[BENCH] summary={summary_path}", flush=True)
    print(f"[BENCH] report={report_path}", flush=True)
    return 1 if any(not run["ok"] for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
