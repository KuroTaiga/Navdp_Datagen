from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class LightFilterConfig:
    mode: str
    strength: float
    radius_frac: float
    center_xy: Tuple[float, float]
    center_jitter: float
    temp_k: float
    vignette: float
    seed: int

    def enabled(self) -> bool:
        return self.mode != "none"


@dataclass
class LumaStats:
    bins: int = 256
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    hist: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))

    def update_from_luma(self, luma: np.ndarray) -> None:
        if luma.size == 0:
            return
        luma = np.clip(luma, 0.0, 1.0)
        self.sum += float(luma.sum())
        self.sum_sq += float((luma * luma).sum())
        self.count += int(luma.size)
        hist, _ = np.histogram(luma, bins=self.bins, range=(0.0, 1.0))
        self.hist += hist.astype(np.int64)

    def update_from_frame(self, frame: np.ndarray, pixel_step: int = 1) -> None:
        luma = compute_luma(frame, pixel_step=pixel_step)
        self.update_from_luma(luma)

    def percentile(self, pct: float) -> float | None:
        if self.count <= 0:
            return None
        pct = max(0.0, min(float(pct), 100.0))
        target = (pct / 100.0) * self.count
        cumulative = np.cumsum(self.hist)
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), len(self.hist) - 1)
        return (idx + 0.5) / float(len(self.hist))

    def finalize(self) -> dict:
        if self.count <= 0:
            return {}
        mean = self.sum / float(self.count)
        var = max((self.sum_sq / float(self.count)) - mean * mean, 0.0)
        std = math.sqrt(var)
        return {
            "luma_mean": mean,
            "luma_std": std,
            "luma_p05": self.percentile(5),
            "luma_p50": self.percentile(50),
            "luma_p95": self.percentile(95),
            "luma_log2_mean": math.log2(mean + 1e-6),
            "pixels_sampled": int(self.count),
        }


def stable_hash_seed(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def compute_luma(frame: np.ndarray, pixel_step: int = 1) -> np.ndarray:
    img = frame
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        luma = img.astype(np.float32, copy=False)
    else:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = img.astype(np.float32, copy=False)
        if img.max() > 1.5:
            img = img / 255.0
        luma = (
            0.2126 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2]
        )
    if pixel_step > 1:
        luma = luma[::pixel_step, ::pixel_step]
    if luma.max() > 1.5:
        luma = luma / 255.0
    return np.clip(luma, 0.0, 1.0)


def color_temperature_to_rgb(temp_k: float) -> np.ndarray:
    if temp_k <= 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    temp = temp_k / 100.0
    if temp <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(max(temp, 1e-6)) - 161.1195681661
        if temp <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * math.log(max(temp - 10.0, 1e-6)) - 305.0447927307
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        b = 255.0
    rgb = np.array([r, g, b], dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    return rgb


def _to_hwc_float(img: np.ndarray) -> tuple[np.ndarray, str]:
    layout = "hwc"
    out = img
    if out.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {out.shape}")
    if out.shape[0] in (1, 3, 4) and out.shape[2] not in (1, 3, 4):
        out = np.transpose(out, (1, 2, 0))
        layout = "chw"
    if out.shape[2] == 4:
        out = out[..., :3]
    out = out.astype(np.float32, copy=False)
    if out.max() > 1.5:
        out = out / 255.0
    return out, layout


def _from_hwc(img: np.ndarray, layout: str) -> np.ndarray:
    if layout == "chw":
        return np.transpose(img, (2, 0, 1))
    return img


def apply_light_filter(
    img: np.ndarray,
    cfg: LightFilterConfig,
    *,
    frame_index: int = 0,
    seed_offset: int = 0,
) -> np.ndarray:
    if not cfg.enabled():
        return img
    img_hwc, layout = _to_hwc_float(img)
    height, width, _ = img_hwc.shape

    center_x = float(cfg.center_xy[0]) * max(width - 1, 1)
    center_y = float(cfg.center_xy[1]) * max(height - 1, 1)
    if cfg.center_jitter > 0.0:
        jitter_px = float(cfg.center_jitter) * float(min(width, height))
        seed = int(cfg.seed) + int(seed_offset) + int(frame_index)
        rng = np.random.default_rng(seed)
        center_x += float(rng.uniform(-jitter_px, jitter_px))
        center_y += float(rng.uniform(-jitter_px, jitter_px))
    center_x = min(max(center_x, 0.0), float(width - 1))
    center_y = min(max(center_y, 0.0), float(height - 1))

    out = img_hwc
    rr = None
    if cfg.mode in ("disk", "cl"):
        radius_px = max(float(cfg.radius_frac) * float(min(width, height)), 1.0)
        yy, xx = np.ogrid[:height, :width]
        rr = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        mask = np.exp(-((rr / radius_px) ** 2))
        gain = 1.0 + float(cfg.strength) * mask
        out = out * gain[..., None]
    elif cfg.mode == "global":
        out = out * (1.0 + float(cfg.strength))

    if cfg.vignette > 0.0:
        if rr is None:
            yy, xx = np.ogrid[:height, :width]
            rr = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        corners = np.array(
            [
                [0.0, 0.0],
                [0.0, float(height - 1)],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
            ]
        )
        corner_dist = np.sqrt((corners[:, 0] - center_x) ** 2 + (corners[:, 1] - center_y) ** 2)
        rmax = float(corner_dist.max()) if corner_dist.size else 1.0
        if rmax <= 0.0:
            rmax = 1.0
        vignette = 1.0 - float(cfg.vignette) * (rr / rmax) ** 2
        out = out * np.clip(vignette, 0.0, 1.0)[..., None]

    if cfg.temp_k > 0.0:
        temp_rgb = color_temperature_to_rgb(cfg.temp_k)
        out = out * temp_rgb[None, None, :]

    out = np.clip(out, 0.0, 1.0)
    return _from_hwc(out, layout)
