"""Video writer helpers with pluggable backends."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import contextlib
import os
import shutil
import subprocess
import sys

import imageio.v2 as imageio


class VideoWriterBackend(str, Enum):
    CPU = "cpu"
    NVENC = "nvenc"
    GPU = "gpu"


_NVENC_FALLBACK_REPORTED = False
_CPU_H264_FALLBACK_REPORTED = False
_STRICT_GPU_BACKENDS = os.getenv("STRICT_GPU_BACKENDS", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_GPU_VIDEO_SYNC_MODE = (os.getenv("GPU_VIDEO_SYNC", "both") or "both").lower()
_GPU_VIDEO_SYNC_BEFORE = _GPU_VIDEO_SYNC_MODE in ("1", "true", "yes", "on", "before", "both")
_GPU_VIDEO_SYNC_AFTER = _GPU_VIDEO_SYNC_MODE in ("1", "true", "yes", "on", "after", "both")
_GPU_VIDEO_RETAIN_FRAMES = int(os.getenv("GPU_VIDEO_RETAIN_FRAMES", "8") or 0)
_VIDEO_GOP_FRAMES_ENV = os.getenv("VIDEO_GOP_FRAMES") or os.getenv("GPU_VIDEO_GOP_FRAMES")
_GPU_VIDEO_DISABLE_BFRAMES = os.getenv("GPU_VIDEO_DISABLE_BFRAMES", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_GPU_VIDEO_CLONE = os.getenv("GPU_VIDEO_CLONE", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _resolve_gop_frames(fps: float, gop_frames: int | None = None) -> int:
    if gop_frames is not None and int(gop_frames) > 0:
        return int(gop_frames)
    if _VIDEO_GOP_FRAMES_ENV:
        try:
            env_value = int(_VIDEO_GOP_FRAMES_ENV)
        except ValueError:
            env_value = 0
        if env_value > 0:
            return env_value
    return max(1, int(round(float(fps))))


def _resolve_ffmpeg_bin() -> str | None:
    return (
        os.getenv("IMAGEIO_FFMPEG_EXE")
        or os.getenv("FFMPEG_BIN")
        or shutil.which("ffmpeg")
    )


class GpuVideoWriter:
    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        width: int,
        height: int,
        nvenc_preset: str | None = None,
        nvenc_bitrate: str | None = None,
        pixel_format: str = "ABGR",
        ffmpeg_bin: str | None = None,
        encode_timer=None,
        mux_timer=None,
        gop_frames: int | None = None,
    ) -> None:
        try:
            import PyNvVideoCodec as nvc  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "PyNvVideoCodec is required for video-backend=gpu."
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError("GPU video writer requires valid width/height.")
        self._nvc = nvc
        self._path = path
        self._fps = fps
        self._pixel_format = pixel_format.upper()
        self._ffmpeg_bin = ffmpeg_bin or _resolve_ffmpeg_bin()
        if not self._ffmpeg_bin:
            raise RuntimeError("ffmpeg binary not found for GPU muxing.")
        self._bitstream_path = path.with_suffix(".h264")
        self._bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        self._bitstream_handle = self._bitstream_path.open("wb")
        self._encode_timer = encode_timer
        self._mux_timer = mux_timer
        self._retain_frames = _GPU_VIDEO_RETAIN_FRAMES
        self._recent_frames: list = []
        resolved_gop = _resolve_gop_frames(fps, gop_frames)
        base_params: dict[str, str] = {
            "codec": "h264",
            "fps": str(int(round(fps))),
            "gop": str(resolved_gop),
            "idrperiod": str(resolved_gop),
            "repeatspspps": "1",
        }
        if nvenc_preset:
            base_params["preset"] = nvenc_preset
        if nvenc_bitrate:
            base_params["bitrate"] = nvenc_bitrate
        encoder_params = dict(base_params)
        if _GPU_VIDEO_DISABLE_BFRAMES:
            encoder_params["bf"] = "0"
        try:
            self._encoder = nvc.CreateEncoder(
                int(width),
                int(height),
                self._pixel_format,
                False,
                **encoder_params,
            )
        except Exception as exc:  # pylint: disable=broad-except
            if _GPU_VIDEO_DISABLE_BFRAMES:
                encoder_params = dict(base_params)
                encoder_params["bframes"] = "0"
                try:
                    self._encoder = nvc.CreateEncoder(
                        int(width),
                        int(height),
                        self._pixel_format,
                        False,
                        **encoder_params,
                    )
                except Exception as exc_alt:  # pylint: disable=broad-except
                    if _STRICT_GPU_BACKENDS:
                        raise
                    print(
                        f"[WARN] GPU encoder b-frame disable failed ({exc_alt}); "
                        "falling back to default encoder params.",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._encoder = nvc.CreateEncoder(
                        int(width),
                        int(height),
                        self._pixel_format,
                        False,
                        **base_params,
                    )
            else:
                raise exc
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def append_data(self, frame) -> None:
        if self._closed:
            raise RuntimeError("GPU video writer is closed.")
        if not hasattr(frame, "__cuda_array_interface__"):
            raise TypeError("GPU video writer expects a CUDA tensor frame.")
        if _GPU_VIDEO_SYNC_BEFORE:
            import torch
            torch.cuda.synchronize()
        if _GPU_VIDEO_CLONE:
            import torch
            if isinstance(frame, torch.Tensor):
                frame = frame.clone()
                if not frame.is_contiguous():
                    frame = frame.contiguous()
        if self._retain_frames > 0:
            # Retain recent frames to avoid reuse while encoder is busy.
            self._recent_frames.append(frame)
            if len(self._recent_frames) > self._retain_frames:
                self._recent_frames.pop(0)
        bitstream = self._encoder.Encode(frame)
        if bitstream:
            self._bitstream_handle.write(bitstream)
        if _GPU_VIDEO_SYNC_AFTER:
            import torch
            torch.cuda.synchronize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._encode_timer is not None:
            with self._encode_timer():
                tail = self._encoder.EndEncode()
                if tail:
                    self._bitstream_handle.write(tail)
        else:
            tail = self._encoder.EndEncode()
            if tail:
                self._bitstream_handle.write(tail)
        self._bitstream_handle.close()
        mux_cmd = [
            self._ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-r",
            str(int(round(self._fps))),
            "-f",
            "h264",
            "-i",
            str(self._bitstream_path),
            "-c",
            "copy",
            str(self._path),
        ]
        mux_ctx = self._mux_timer() if self._mux_timer is not None else contextlib.nullcontext()
        with mux_ctx:
            subprocess.run(mux_cmd, check=True)
        try:
            self._bitstream_path.unlink()
        except OSError:
            pass


def make_video_writer(
    path: Path,
    *,
    fps: float,
    backend: VideoWriterBackend | str = VideoWriterBackend.CPU,
    nvenc_preset: str | None = None,
    nvenc_bitrate: str | None = None,
    pixel_format: str = "yuv420p",
    width: int | None = None,
    height: int | None = None,
    gpu_format: str = "ABGR",
    gop_frames: int | None = None,
    encode_timer=None,
    mux_timer=None,
):
    backend_value = (
        backend.value if isinstance(backend, VideoWriterBackend) else str(backend).lower()
    )
    if backend_value == VideoWriterBackend.CPU.value:
        resolved_gop = _resolve_gop_frames(fps, gop_frames)
        try:
            return imageio.get_writer(
                path,
                mode="I",
                fps=fps,
                codec="libx264",
                pixelformat=pixel_format,
                output_params=[
                    "-g",
                    str(resolved_gop),
                    "-keyint_min",
                    str(resolved_gop),
                    "-sc_threshold",
                    "0",
                ],
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
        resolved_gop = _resolve_gop_frames(fps, gop_frames)
        output_params = [
            "-g",
            str(resolved_gop),
            "-keyint_min",
            str(resolved_gop),
            "-forced-idr",
            "1",
            "-sc_threshold",
            "0",
        ]
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
            if _STRICT_GPU_BACKENDS:
                raise RuntimeError(
                    f"NVENC video backend failed in strict mode: {exc}"
                ) from exc
            global _NVENC_FALLBACK_REPORTED
            if not _NVENC_FALLBACK_REPORTED:
                print(
                    f"[WARN] NVENC video backend failed ({exc}); falling back to CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                _NVENC_FALLBACK_REPORTED = True
            return imageio.get_writer(path, mode="I", fps=fps)
    if backend_value == VideoWriterBackend.GPU.value:
        if width is None or height is None:
            raise ValueError("GPU video backend requires width and height.")
        return GpuVideoWriter(
            path,
            fps=fps,
            width=width,
            height=height,
            nvenc_preset=nvenc_preset,
            nvenc_bitrate=nvenc_bitrate,
            pixel_format=gpu_format,
            gop_frames=gop_frames,
            encode_timer=encode_timer,
            mux_timer=mux_timer,
        )
    raise ValueError(f"Unknown video backend: {backend}")
