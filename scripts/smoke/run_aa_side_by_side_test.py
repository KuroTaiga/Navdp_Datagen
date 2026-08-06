#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
from pathlib import Path


def _resolve_ffmpeg_bin() -> str | None:
    return (
        os.getenv("IMAGEIO_FFMPEG_EXE")
        or os.getenv("FFMPEG_BIN")
        or shutil.which("ffmpeg")
    )


def _pick_scene_id(tasks_dir: Path) -> str:
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"TASKS_DIR does not exist: {tasks_dir}")
    for name in sorted(p.name for p in tasks_dir.iterdir() if p.is_dir()):
        if name.startswith("0001_"):
            return name
    raise FileNotFoundError(f"No 0001_* scene found under {tasks_dir}")


def _pick_first_mp4(root: Path) -> Path | None:
    mp4s = sorted(root.rglob("*.mp4"))
    return mp4s[0] if mp4s else None


def _run_render(
    python_bin: str,
    script_dir: Path,
    scene_id: str,
    scenes_dir: Path,
    tasks_dir: Path,
    output_dir: Path,
    max_labels: int,
    width: int,
    height: int,
    video_backend: str,
    sh_degree: int,
    aa_flag: str,
) -> None:
    cmd = [
        python_bin,
        str(script_dir / "render_label_paths_telesim.py"),
        "--scene",
        scene_id,
        "--scenes-dir",
        str(scenes_dir),
        "--tasks-dir",
        str(tasks_dir),
        "--output-dir",
        str(output_dir),
        "--max-labels",
        str(max_labels),
        "--video",
        "--no-rgb-frames",
        "--no-save-depth-maps",
        "--save-camera-metadata",
        "--no-save-follow-metadata",
        "--video-backend",
        video_backend,
        "--resolution",
        str(width),
        str(height),
        "--sh-degree",
        str(sh_degree),
        aa_flag,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    python_bin = os.getenv("PYTHON_BIN") or sys.executable
    scenes_dir = Path(os.getenv("SCENES_DIR", repo_root / "data" / "scenes"))
    tasks_dir = Path(os.getenv("TASKS_DIR", repo_root / "data" / "CHINGMU_75_rescaled_0800_42_iter1"))
    output_root = Path(os.getenv("OUTPUT_ROOT", repo_root / "data" / "aa_side_by_side_test"))
    scene_id = os.getenv("SCENE_ID") or _pick_scene_id(tasks_dir)
    max_labels = int(os.getenv("MAX_LABELS", "30"))
    width = int(os.getenv("RESOLUTION_WIDTH", "640"))
    height = int(os.getenv("RESOLUTION_HEIGHT", "480"))
    video_backend = os.getenv("VIDEO_BACKEND", "cpu")
    sh_degree = int(os.getenv("SH_DEGREE", "-1"))

    aa_on_dir = output_root / "aa_on"
    aa_off_dir = output_root / "aa_off"
    compare_out = output_root / "aa_side_by_side.mp4"
    ffmpeg_bin = _resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        print("[ERROR] ffmpeg is required for side-by-side comparison but was not found in PATH.", file=sys.stderr)
        return 1

    print(f"[CONFIG] SCENE_ID={scene_id}")
    print(f"[CONFIG] SCENES_DIR={scenes_dir}")
    print(f"[CONFIG] TASKS_DIR={tasks_dir}")
    print(f"[CONFIG] OUTPUT_ROOT={output_root}")
    print(f"[CONFIG] MAX_LABELS={max_labels}")
    print(f"[CONFIG] RESOLUTION={width}x{height}")

    output_root.mkdir(parents=True, exist_ok=True)

    _run_render(
        python_bin,
        repo_root,
        scene_id,
        scenes_dir,
        tasks_dir,
        aa_on_dir,
        max_labels,
        width,
        height,
        video_backend,
        sh_degree,
        "--antialiasing",
    )
    _run_render(
        python_bin,
        repo_root,
        scene_id,
        scenes_dir,
        tasks_dir,
        aa_off_dir,
        max_labels,
        width,
        height,
        video_backend,
        sh_degree,
        "--no-antialiasing",
    )

    aa_on_video = _pick_first_mp4(aa_on_dir)
    aa_off_video = _pick_first_mp4(aa_off_dir)
    if not aa_on_video or not aa_off_video:
        print(f"[ERROR] Could not find mp4 outputs in {aa_on_dir} or {aa_off_dir}.", file=sys.stderr)
        return 1

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(aa_on_video),
        "-i",
        str(aa_off_video),
        "-filter_complex",
        "[0:v]drawtext=text='AA ON':x=12:y=12:fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5[v0]; "
        "[1:v]drawtext=text='AA OFF':x=12:y=12:fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5[v1]; "
        "[v0][v1]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-shortest",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        str(compare_out),
    ]
    subprocess.run(cmd, check=True)
    print(f"[DONE] Side-by-side video: {compare_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
