"""Video writer helpers with pluggable backends."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import sys

import imageio.v2 as imageio


class VideoWriterBackend(str, Enum):
    CPU = "cpu"
    NVENC = "nvenc"


_NVENC_FALLBACK_REPORTED = False
_CPU_H264_FALLBACK_REPORTED = False


def make_video_writer(
    path: Path,
    *,
    fps: float,
    backend: VideoWriterBackend | str = VideoWriterBackend.CPU,
    nvenc_preset: str | None = None,
    nvenc_bitrate: str | None = None,
    pixel_format: str = "yuv420p",
):
    backend_value = (
        backend.value if isinstance(backend, VideoWriterBackend) else str(backend).lower()
    )
    if backend_value == VideoWriterBackend.CPU.value:
        try:
            return imageio.get_writer(
                path,
                mode="I",
                fps=fps,
                codec="libx264",
                pixelformat=pixel_format,
            )
        except Exception as exc:  # pylint: disable=broad-except
            global _CPU_H264_FALLBACK_REPORTED
            if not _CPU_H264_FALLBACK_REPORTED:
                print(
                    f"[WARN] CPU H.264 encoder failed ({exc}); falling back to default CPU encoder.",
                    file=sys.stderr,
                    flush=True,
                )
                _CPU_H264_FALLBACK_REPORTED = True
            return imageio.get_writer(path, mode="I", fps=fps)
    if backend_value == VideoWriterBackend.NVENC.value:
        output_params = []
        if nvenc_preset:
            output_params.extend(["-preset", nvenc_preset])
        if nvenc_bitrate:
            output_params.extend(["-b:v", nvenc_bitrate])
        kwargs = {
            "mode": "I",
            "fps": fps,
            "codec": "h264_nvenc",
            "pixelformat": pixel_format,
        }
        if output_params:
            kwargs["output_params"] = output_params
        try:
            return imageio.get_writer(path, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            global _NVENC_FALLBACK_REPORTED
            if not _NVENC_FALLBACK_REPORTED:
                print(
                    f"[WARN] NVENC video backend failed ({exc}); falling back to CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                _NVENC_FALLBACK_REPORTED = True
            return imageio.get_writer(path, mode="I", fps=fps)
    raise ValueError(f"Unknown video backend: {backend}")
