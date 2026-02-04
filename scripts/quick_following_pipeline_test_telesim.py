#!/usr/bin/env python3
"""
Quick smoke test for the TeleSim "following" pipeline.

Default behavior (as requested):
- sample 30 different scenes at random (seed=1)
- pick 1 path per scene at random
- run the TeleSim dispatcher on that small manifest
- record:
  - average VRAM usage per worker process / per scene (via nvidia-smi PID memory polling)
  - speed + ETA extrapolated to a default 40,000 paths, assuming avg path length 15.6m
    and step distance 0.05m/frame (=> ~312 frames per path)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROGRESS_RE = re.compile(
    r"\[PROGRESS\]\s+Scene\s+(?P<scene>\S+):\s+(?P<scene_done>\d+)/(?P<scene_planned>\d+)\s+\|\s+Overall:\s+(?P<overall_done>\d+)/(?P<overall_planned>\d+)"
)


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


def _resolve_label_directory(scene_task_dir: Path) -> Path | None:
    label_paths_dir = scene_task_dir / "label_paths"
    if label_paths_dir.is_dir() and any(label_paths_dir.glob("*.json")):
        return label_paths_dir
    if scene_task_dir.is_dir() and any(scene_task_dir.glob("*.json")):
        return scene_task_dir
    return None


def _label_exists(tasks_dir: Path, scene_id: str, label_id: str, *, exclude_detailed: bool) -> bool:
    if exclude_detailed and label_id.endswith("_detailed"):
        return False
    scene_dir = tasks_dir / scene_id
    label_dir = _resolve_label_directory(scene_dir)
    if label_dir is None:
        return False
    return (label_dir / f"{label_id}.json").is_file()


def _trim_manifest_random(
    *,
    manifest: dict[str, Any],
    tasks_dir: Path,
    scene_prefix: str,
    num_scenes: int,
    paths_per_scene: int,
    seed: int,
    exclude_detailed: bool,
) -> dict[str, Any]:
    assignments_all = manifest.get("assignments") or []
    if not isinstance(assignments_all, list) or not assignments_all:
        raise SystemExit("[ERROR] source manifest has no assignments.")

    assignments_by_scene: dict[str, list[dict[str, Any]]] = {}
    for a in assignments_all:
        scene = str(a.get("scene") or "")
        if not scene:
            continue
        assignments_by_scene.setdefault(scene, []).append(a)

    if not tasks_dir.is_dir():
        raise SystemExit(f"[ERROR] tasks dir not found: {tasks_dir}")

    # Candidate scenes must exist on disk AND in the assignment manifest.
    candidates = [
        p.name
        for p in tasks_dir.iterdir()
        if p.is_dir()
        and (not scene_prefix or p.name.startswith(scene_prefix))
        and p.name in assignments_by_scene
    ]
    if not candidates:
        raise SystemExit(
            f"[ERROR] no candidate scenes under {tasks_dir} with prefix={scene_prefix!r} that also exist in the "
            "source manifest. This usually means the manifest was generated for a different TASKS_DIR. "
            "Pass a matching --source-manifest, or use --auto-generate-source-manifest."
        )

    rng = random.Random(int(seed))
    rng.shuffle(candidates)

    picked: list[dict[str, Any]] = []
    picked_scenes: list[str] = []

    for scene_id in candidates:
        scene_assignments = assignments_by_scene.get(scene_id) or []
        viable = []
        for a in scene_assignments:
            label = str(a.get("label") or "")
            if not label:
                continue
            if _label_exists(tasks_dir, scene_id, label, exclude_detailed=exclude_detailed):
                viable.append(a)
        if not viable:
            continue

        rng.shuffle(viable)
        selected = viable[: int(paths_per_scene)]
        if not selected:
            continue

        picked.extend(selected)
        picked_scenes.append(scene_id)
        if len(picked_scenes) >= int(num_scenes):
            break

    if len(picked_scenes) < int(num_scenes):
        raise SystemExit(
            f"[ERROR] only found {len(picked_scenes)} viable scenes (wanted {num_scenes}). "
            f"Try lowering --num-scenes or changing --scene-prefix."
        )

    actor_ids = {str(a.get("actor_id", "")) for a in picked if a.get("actor_id") is not None}
    actors = [a for a in (manifest.get("actors") or []) if str(a.get("id", "")) in actor_ids]

    return {
        "actors": actors,
        "assignments": picked,
        "generated_at": manifest.get("generated_at"),
        "seed": manifest.get("seed"),
        "tasks_root": manifest.get("tasks_root"),
        "scenes_root": manifest.get("scenes_root"),
        "trimmed_seed": int(seed),
        "trimmed_scene_prefix": scene_prefix,
        "trimmed_num_scenes": int(num_scenes),
        "trimmed_paths_per_scene": int(paths_per_scene),
        "trimmed_scenes": picked_scenes,
    }


def _nvidia_smi_pid_mem_mb() -> dict[int, int]:
    """Return pid -> used_memory_MB (summed across GPUs)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}

    usage: dict[int, int] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(float(parts[1]))
        except Exception:
            continue
        usage[pid] = usage.get(pid, 0) + mem
    return usage


def _mean_int(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _child_pids(ppid: int) -> set[int]:
    try:
        out = subprocess.check_output(["ps", "-o", "pid=", "--ppid", str(ppid)], text=True)
    except Exception:
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.add(int(line))
        except Exception:
            continue
    return pids


def _pid_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return []
    parts = [p for p in raw.split(b"\0") if p]
    return [p.decode(errors="ignore") for p in parts]


def _extract_arg(cmdline: list[str], flag: str) -> str | None:
    for i, token in enumerate(cmdline):
        if token == flag and i + 1 < len(cmdline):
            return cmdline[i + 1]
    return None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quick TeleSim following pipeline smoke test.")
    ap.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "data" / "CHINGMU_75_rescaled_0800_42_iter1")
    ap.add_argument("--scenes-dir", type=Path, default=REPO_ROOT / "data" / "CHINGMU_scenes_rescaled")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "navdata" / "following_quick_random30_seed1")

    ap.add_argument("--scene-prefix", type=str, default="", help="Optional prefix to filter scenes (default: all).")
    ap.add_argument("--num-scenes", type=int, default=30, help="How many scenes to sample (default: 30).")
    ap.add_argument("--paths-per-scene", type=int, default=1, help="How many paths per scene (default: 1).")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for sampling (default: 1).")

    ap.add_argument("--workers", type=int, default=6, help="Workers passed to the TeleSim dispatcher (default: 6).")
    ap.add_argument("--conda-env", type=str, default=os.environ.get("CONDA_ENV", "cuda121"))
    ap.add_argument(
        "--use-conda-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the TeleSim dispatcher inside conda (default: on).",
    )
    ap.add_argument("--minimal-frames", type=int, default=38, help="Forwarded to renderer (default: 38).")
    ap.add_argument(
        "--exclude-detailed-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude *_detailed.json (default: on).",
    )

    ap.add_argument("--source-manifest", type=Path, default=None, help="Existing large assignment manifest to trim.")
    ap.add_argument(
        "--auto-generate-source-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If no matching source manifest is found, generate one via scripts/generate_assignment_manifest.sh (default: off).",
    )
    ap.add_argument(
        "--actor-root",
        type=Path,
        default=REPO_ROOT / "data" / "human_gs_source",
        help="Actor root used when auto-generating a source manifest (default: ./data/human_gs_source).",
    )
    ap.add_argument(
        "--ban-list",
        type=Path,
        default=None,
        help="Optional ban list used when auto-generating a source manifest (default: <actor-root>/BanList.txt if exists).",
    )
    ap.add_argument(
        "--generated-source-manifest",
        type=Path,
        default=None,
        help="Where to write an auto-generated source manifest (default: data/tmp/following_autogen_seedS.json).",
    )
    ap.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Where to write the trimmed manifest (default: data/tmp/following_quick_randomN_seedS_manifest.json).",
    )

    ap.add_argument("--per-job-metrics-dir", type=Path, default=REPO_ROOT / "analysis" / "quick_following_telesim_metrics")
    ap.add_argument("--progress-json", type=Path, default=REPO_ROOT / "analysis" / "quick_following_telesim_progress.json")
    ap.add_argument("--status-json", type=Path, default=REPO_ROOT / "analysis" / "quick_following_telesim_status.json")
    ap.add_argument("--report-json", type=Path, default=REPO_ROOT / "analysis" / "quick_following_telesim_report.json")

    ap.add_argument(
        "--total-paths",
        type=int,
        default=40000,
        help="Assumed total paths for ETA extrapolation (default: 40000).",
    )
    ap.add_argument(
        "--avg-path-length-m",
        type=float,
        default=15.6,
        help="Assumed average path length in meters (default: 15.6).",
    )
    ap.add_argument(
        "--step-distance-m",
        type=float,
        default=0.05,
        help="Meters per frame when converting length->frames (default: 0.05).",
    )
    ap.add_argument(
        "--monitor-interval-sec",
        type=float,
        default=1.0,
        help="Sampling interval for nvidia-smi VRAM usage (default: 1.0s).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only write the trimmed manifest and print info.")
    ap.add_argument("--no-run", action="store_true", help="Alias for --dry-run.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_scenes <= 0:
        raise SystemExit("--num-scenes must be > 0")
    if args.paths_per_scene <= 0:
        raise SystemExit("--paths-per-scene must be > 0")

    source_manifest = args.source_manifest or _find_default_source_manifest()
    if source_manifest is None or not source_manifest.is_file():
        if not args.auto_generate_source_manifest:
            raise SystemExit(
                "[ERROR] no source manifest found. Pass --source-manifest, or run with "
                "--auto-generate-source-manifest --actor-root <path>."
            )
        gen_out = args.generated_source_manifest
        if gen_out is None:
            gen_out = REPO_ROOT / "data" / "tmp" / f"following_autogen_seed{int(args.seed)}.json"
        ban_list = args.ban_list
        if ban_list is None:
            candidate = args.actor_root / "BanList.txt"
            if candidate.is_file():
                ban_list = candidate
        gen_script = REPO_ROOT / "scripts" / "generate_assignment_manifest.sh"
        if not gen_script.is_file():
            raise SystemExit(f"[ERROR] missing generator script: {gen_script}")
        env = dict(os.environ)
        env["CONDA_ENV"] = str(args.conda_env)
        env["ACTOR_ROOT"] = str(args.actor_root)
        env["ASSIGNMENTS_OUT"] = str(gen_out)
        env["SCENES_DIR"] = str(args.scenes_dir)
        env["TASKS_DIR"] = str(args.tasks_dir)
        env["SEED"] = str(int(args.seed))
        env["EXCLUDE_DETAILED_LABELS"] = "true" if args.exclude_detailed_labels else "false"
        if ban_list is not None:
            env["BAN_LIST"] = str(ban_list)
        print(f"[GEN] Generating source manifest: {gen_out}", flush=True)
        rc = subprocess.call(["bash", str(gen_script)], env=env)
        if rc != 0 or not gen_out.is_file():
            raise SystemExit(f"[ERROR] failed to generate source manifest at {gen_out} (rc={rc})")
        source_manifest = gen_out

    manifest = _load_json(source_manifest)
    trimmed = _trim_manifest_random(
        manifest=manifest,
        tasks_dir=args.tasks_dir,
        scene_prefix=str(args.scene_prefix),
        num_scenes=int(args.num_scenes),
        paths_per_scene=int(args.paths_per_scene),
        seed=int(args.seed),
        exclude_detailed=bool(args.exclude_detailed_labels),
    )

    out_manifest = args.out_manifest
    if out_manifest is None:
        out_manifest = (
            REPO_ROOT
            / "data"
            / "tmp"
            / f"following_quick_random{int(args.num_scenes)}_seed{int(args.seed)}_manifest.json"
        )
    _write_json(out_manifest, trimmed)

    assignments = trimmed.get("assignments") or []
    scenes = [str(a.get("scene")) for a in assignments]
    labels = [str(a.get("label")) for a in assignments]
    unique_scenes = sorted(set(scenes))

    print(f"[OK] Scenes: {len(unique_scenes)} (seed={args.seed}, prefix={args.scene_prefix!r})")
    print(f"[OK] Paths selected: {len(labels)} ({args.paths_per_scene} per scene)")
    print(f"[OK] Trimmed manifest: {out_manifest}")
    if labels:
        preview = ", ".join(f"{s}:{l}" for s, l in list(zip(scenes, labels))[: min(10, len(labels))])
        suffix = " ..." if len(labels) > 10 else ""
        print(f"[OK] First labels: {preview}{suffix}")

    if args.dry_run or args.no_run:
        return 0

    dispatcher = REPO_ROOT / "parallel_render_paths_telesim.py"
    render_script = REPO_ROOT / "render_label_paths_telesim.py"
    if not dispatcher.is_file():
        raise SystemExit(f"[ERROR] dispatcher not found: {dispatcher}")
    if not render_script.is_file():
        raise SystemExit(f"[ERROR] render script not found: {render_script}")

    cmd: list[str] = []
    if args.use_conda_run:
        if not shutil.which("conda"):
            raise SystemExit("[ERROR] conda not found but --use-conda-run was enabled.")
        cmd.extend(["conda", "run", "--no-capture-output", "-n", str(args.conda_env), "python"])
    else:
        cmd.append("python3")

    cmd.extend(
        [
            str(dispatcher),
            "--render-script",
            str(render_script),
            "--scenes-dir",
            str(args.scenes_dir),
            "--tasks-dir",
            str(args.tasks_dir),
            "--workers",
            str(int(args.workers)),
            "--minimal-frames",
            str(int(args.minimal_frames)),
            "--output-dir",
            str(args.output_dir),
            "--assignment-manifest",
            str(out_manifest),
            "--progress-json",
            str(args.progress_json),
            "--status-json",
            str(args.status_json),
            "--per-job-metrics-dir",
            str(args.per_job_metrics_dir),
        ]
    )
    if args.exclude_detailed_labels:
        cmd.append("--exclude-detailed-labels")
    else:
        cmd.append("--no-exclude-detailed-labels")

    # Render flags: align with following defaults, but keep this smoke test simple.
    render_extra_args = [
        "--overwrite",
        "--stabilize",
        "--height-offset",
        "0.3",
        "--no-show-BEV",
        "--video",
        "--no-rgb-frames",
        "--no-save-depth-maps",
        "--save-camera-metadata",
        "--save-follow-metadata",
        "--no-antialiasing",
    ]
    cmd.extend(["--render-extra-args", " ".join(render_extra_args)])

    args.per_job_metrics_dir.mkdir(parents=True, exist_ok=True)
    args.progress_json.parent.mkdir(parents=True, exist_ok=True)
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    pid_to_scene: dict[int, str] = {}
    pid_to_actor: dict[int, str] = {}
    pid_samples_mb: dict[int, list[int]] = {}
    pid_start_time: dict[int, float] = {}
    scene_durations: dict[str, list[float]] = {}
    overall_done = 0
    overall_planned = len(assignments)

    started_at = time.time()
    stop_evt = threading.Event()
    lock = threading.Lock()

    def _sampler() -> None:
        while not stop_evt.is_set():
            current = _child_pids(proc.pid)
            # Track child workers + map pid->scene/actor via cmdline.
            for pid in current:
                with lock:
                    if pid not in pid_start_time:
                        pid_start_time[pid] = time.time()
                    if pid not in pid_to_scene or pid not in pid_to_actor:
                        cmdline = _pid_cmdline(pid)
                        scene = _extract_arg(cmdline, "--scene")
                        actor = _extract_arg(cmdline, "--job-actor-id")
                        if scene:
                            pid_to_scene[pid] = scene
                        if actor:
                            pid_to_actor[pid] = actor

            # Close out durations for exited children.
            with lock:
                ended = [pid for pid in pid_start_time.keys() if pid not in current]
                now = time.time()
                for pid in ended:
                    start = pid_start_time.pop(pid, None)
                    if start is None:
                        continue
                    scene = pid_to_scene.get(pid)
                    if scene:
                        scene_durations.setdefault(scene, []).append(now - start)

            usage = _nvidia_smi_pid_mem_mb()
            with lock:
                for pid, mem_mb in usage.items():
                    pid_samples_mb.setdefault(pid, []).append(int(mem_mb))
            time.sleep(max(0.05, float(args.monitor_interval_sec)))

    print(f"[RUN] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None

    sampler_thread = threading.Thread(target=_sampler, name="vram-sampler", daemon=True)
    sampler_thread.start()

    avg_frames = float(args.avg_path_length_m) / max(1e-9, float(args.step_distance_m))

    for line in proc.stdout:
        print(line, end="", flush=True)

        m = _PROGRESS_RE.search(line)
        if m:
            overall_done = int(m.group("overall_done"))
            elapsed = time.time() - started_at
            if elapsed > 0 and overall_done > 0:
                speed_paths = overall_done / elapsed
                eta_sec = (max(0, int(args.total_paths) - overall_done) / speed_paths) if speed_paths > 0 else None
                mps = speed_paths * float(args.avg_path_length_m)
                fps = speed_paths * avg_frames
                eta_str = f"{eta_sec:.1f}s" if eta_sec is not None else "-"
                print(
                    f"[STATS] paths={overall_done}/{overall_planned} speed={speed_paths:.3f} paths/s "
                    f"({mps:.2f} m/s, {fps:.1f} fps) eta={eta_str}",
                    flush=True,
                )

    returncode = proc.wait()
    stop_evt.set()
    sampler_thread.join(timeout=5.0)
    elapsed = time.time() - started_at

    with lock:
        pid_avg_mb: dict[int, float] = {}
        for pid, samples in pid_samples_mb.items():
            avg = _mean_int(samples)
            if avg is not None:
                pid_avg_mb[pid] = avg

        scene_to_avgs: dict[str, list[float]] = {}
        for pid, scene in pid_to_scene.items():
            avg = pid_avg_mb.get(pid)
            if avg is None:
                continue
            scene_to_avgs.setdefault(scene, []).append(float(avg))
        scene_avg_mb: dict[str, float] = {
            scene: float(statistics.fmean(avgs)) for scene, avgs in sorted(scene_to_avgs.items()) if avgs
        }

        scene_duration_avg: dict[str, float] = {
            scene: float(statistics.fmean(vals)) for scene, vals in sorted(scene_durations.items()) if vals
        }

    speed_paths = (overall_done / elapsed) if elapsed > 0 else None
    eta_sec = None
    if speed_paths and speed_paths > 0:
        eta_sec = max(0.0, (float(args.total_paths) - float(overall_done)) / speed_paths)
    speed_mps = (speed_paths * float(args.avg_path_length_m)) if speed_paths is not None else None
    speed_fps = (speed_paths * avg_frames) if speed_paths is not None else None

    report: dict[str, Any] = {
        "started_at": started_at,
        "elapsed_sec": elapsed,
        "returncode": int(returncode),
        "seed": int(args.seed),
        "scene_prefix": str(args.scene_prefix),
        "num_scenes": int(args.num_scenes),
        "paths_per_scene": int(args.paths_per_scene),
        "paths_planned": int(overall_planned),
        "paths_done": int(overall_done),
        "assumed_total_paths": int(args.total_paths),
        "assumed_avg_path_length_m": float(args.avg_path_length_m),
        "assumed_step_distance_m": float(args.step_distance_m),
        "assumed_avg_frames_per_path": float(avg_frames),
        "speed_paths_per_sec": float(speed_paths) if speed_paths is not None else None,
        "speed_m_per_sec": float(speed_mps) if speed_mps is not None else None,
        "speed_frames_per_sec": float(speed_fps) if speed_fps is not None else None,
        "eta_sec": float(eta_sec) if eta_sec is not None else None,
        "avg_vram_mb_per_pid": {str(k): v for k, v in sorted(pid_avg_mb.items())},
        "avg_vram_mb_per_scene": {k: v for k, v in sorted(scene_avg_mb.items())},
        "duration_sec_per_scene": {k: v for k, v in sorted(scene_duration_avg.items())},
        "pid_to_scene": {str(k): v for k, v in sorted(pid_to_scene.items())},
        "pid_to_actor": {str(k): v for k, v in sorted(pid_to_actor.items())},
        "out_manifest": str(out_manifest),
        "output_dir": str(args.output_dir),
        "progress_json": str(args.progress_json),
        "status_json": str(args.status_json),
        "per_job_metrics_dir": str(args.per_job_metrics_dir),
    }
    _write_json(args.report_json, report)
    print(f"[OK] Wrote report: {args.report_json}", flush=True)

    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
