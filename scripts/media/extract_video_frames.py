#!/usr/bin/env python3
"""Extract a video into per-frame image files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_VIDEO = Path("data2/0500_fpv/0463_840754/138.mp4")
DEFAULT_OUT_DIR = Path("out/imgs_from_vid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Break down a video into individual images per frame."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Input video path (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame_",
        help="Output filename prefix (default: frame_)",
    )
    parser.add_argument(
        "--ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Image extension/format (default: jpg)",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Save one frame every N frames (default: 1 = save all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = args.video

    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_idx = 0
    out_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if in_idx % args.every_n == 0:
            out_name = f"{args.prefix}{out_idx:06d}.{args.ext}"
            out_path = out_dir / out_name
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            out_idx += 1

        in_idx += 1

    cap.release()
    print(
        f"Done. Read {in_idx} frames"
        + (f" (reported total: {total})" if total > 0 else "")
        + f", wrote {out_idx} images to: {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
