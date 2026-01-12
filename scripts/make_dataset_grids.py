#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    print("Missing dependency: opencv-python", file=sys.stderr)
    raise

try:
    from PIL import Image, ImageOps
except ImportError as exc:
    print("Missing dependency: Pillow", file=sys.stderr)
    raise


RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENES_ROOT = BASE_DIR / "data" / "scenes"
DEFAULT_VIDEOS_ROOT = BASE_DIR / "data2" / "0500_fpv"


def list_videos(videos_root: Path) -> list[Path]:
    return sorted(videos_root.rglob("*.mp4"))


def list_occupancy_images(scenes_root: Path) -> list[Path]:
    return sorted(p for p in scenes_root.glob("*/occupancy.png") if p.is_file())


def path_sampler(paths: list[Path], rng: random.Random):
    if not paths:
        return
    pool = list(paths)
    while True:
        rng.shuffle(pool)
        for path in pool:
            yield path


def read_random_frame(video_path: Path, rng: random.Random) -> Image.Image | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for _ in range(5):
            if frame_count > 0:
                frame_idx = rng.randrange(frame_count)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
    finally:
        cap.release()
    return None


def fit_image(img: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    if mode == "cover":
        return ImageOps.fit(img, size, method=RESAMPLE)
    if mode == "contain":
        return ImageOps.pad(img, size, method=RESAMPLE, color=(0, 0, 0))
    raise ValueError(f"Unknown fit mode: {mode}")


def build_video_grid(
    video_paths: list[Path],
    grid_size: int,
    cell_size: tuple[int, int],
    rng: random.Random,
) -> Image.Image:
    if not video_paths:
        raise RuntimeError("No mp4 files found for the video grid.")
    total = grid_size * grid_size
    grid = Image.new("RGB", (cell_size[0] * grid_size, cell_size[1] * grid_size), color=(0, 0, 0))
    sampler = path_sampler(video_paths, rng)
    filled = 0
    attempts = 0
    max_attempts = max(50, total * 12)
    while filled < total and attempts < max_attempts:
        attempts += 1
        path = next(sampler)
        frame = read_random_frame(path, rng)
        if frame is None:
            continue
        frame = fit_image(frame, cell_size, mode="cover")
        row = filled // grid_size
        col = filled % grid_size
        grid.paste(frame, (col * cell_size[0], row * cell_size[1]))
        filled += 1
    if filled < total:
        raise RuntimeError(
            f"Only filled {filled}/{total} video cells. Check video files for readability."
        )
    return grid


def build_occupancy_grid(
    occ_paths: list[Path],
    grid_size: int,
    cell_size: tuple[int, int],
    rng: random.Random,
) -> Image.Image:
    if not occ_paths:
        raise RuntimeError("No occupancy.png files found for the occupancy grid.")
    total = grid_size * grid_size
    grid = Image.new("RGB", (cell_size[0] * grid_size, cell_size[1] * grid_size), color=(0, 0, 0))
    sampler = path_sampler(occ_paths, rng)
    filled = 0
    attempts = 0
    max_attempts = max(50, total * 3)
    while filled < total and attempts < max_attempts:
        attempts += 1
        path = next(sampler)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = fit_image(img, cell_size, mode="contain")
        except OSError:
            continue
        row = filled // grid_size
        col = filled % grid_size
        grid.paste(img, (col * cell_size[0], row * cell_size[1]))
        filled += 1
    if filled < total:
        raise RuntimeError(
            f"Only filled {filled}/{total} occupancy cells. Check occupancy files."
        )
    return grid


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create grid images from random video frames and occupancy maps for dataset preview."
        )
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help=f"Root to search for .mp4 files (default: {DEFAULT_VIDEOS_ROOT})",
    )
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=DEFAULT_SCENES_ROOT,
        help=f"Scenes root containing per-scene occupancy.png (default: {DEFAULT_SCENES_ROOT})",
    )
    parser.add_argument(
        "--video-grid",
        type=int,
        default=20,
        help="Grid size for video frames (default: 20 for 20x20)",
    )
    parser.add_argument(
        "--occ-grid",
        type=int,
        default=30,
        help="Grid size for occupancy images (default: 30 for 30x30)",
    )
    parser.add_argument(
        "--video-cell",
        type=int,
        default=128,
        help="Cell size in pixels for video frames (default: 128)",
    )
    parser.add_argument(
        "--occ-cell",
        type=int,
        default=96,
        help="Cell size in pixels for occupancy images (default: 96)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory (default: current working directory)",
    )
    parser.add_argument(
        "--video-output",
        type=str,
        default=None,
        help="Filename for the video grid image (default: video_grid_<N>x<N>.png)",
    )
    parser.add_argument(
        "--occ-output",
        type=str,
        default=None,
        help="Filename for the occupancy grid image (default: occupancy_grid_<N>x<N>.png)",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip generating the video grid image",
    )
    parser.add_argument(
        "--skip-occupancy",
        action="store_true",
        help="Skip generating the occupancy grid image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.video_grid <= 0 or args.occ_grid <= 0:
        print("Grid sizes must be positive.", file=sys.stderr)
        sys.exit(1)

    videos_root = args.videos_root
    scenes_root = args.scenes_root

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_output = args.video_output or f"video_grid_{args.video_grid}x{args.video_grid}.png"
    occ_output = args.occ_output or f"occupancy_grid_{args.occ_grid}x{args.occ_grid}.png"

    if not args.skip_video:
        video_paths = list_videos(videos_root)
        print(f"Found {len(video_paths)} video(s) under {videos_root}")
        if len(video_paths) < args.video_grid * args.video_grid:
            print("Warning: fewer videos than grid cells; duplicates will be used.", file=sys.stderr)
        grid = build_video_grid(
            video_paths,
            grid_size=args.video_grid,
            cell_size=(args.video_cell, args.video_cell),
            rng=rng,
        )
        out_path = args.output_dir / video_output
        grid.save(out_path)
        print(f"Saved video grid -> {out_path}")

    if not args.skip_occupancy:
        occ_paths = list_occupancy_images(scenes_root)
        print(f"Found {len(occ_paths)} occupancy image(s) under {scenes_root}")
        if not occ_paths:
            raise RuntimeError(
                f"No occupancy.png files found under {scenes_root}. "
                "Expected ./data/scenes/<scene_id>/occupancy.png."
            )
        if len(occ_paths) < args.occ_grid * args.occ_grid:
            print("Warning: fewer occupancy maps than grid cells; duplicates will be used.", file=sys.stderr)
        grid = build_occupancy_grid(
            occ_paths,
            grid_size=args.occ_grid,
            cell_size=(args.occ_cell, args.occ_cell),
            rng=rng,
        )
        out_path = args.output_dir / occ_output
        grid.save(out_path)
        print(f"Saved occupancy grid -> {out_path}")


if __name__ == "__main__":
    main()
