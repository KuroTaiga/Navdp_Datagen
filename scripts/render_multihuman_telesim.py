#!/usr/bin/env python3
"""Render one scene with one camera path and multiple ping-pong walking humans.

This script is designed for a single-scene demo where:
- Camera follows one planned path label.
- Multiple human avatars each follow their own planned path labels.
- Human movement is ping-pong (start->end, then end->start, repeat).

It writes:
- One output video.
- One JSON manifest that records paths + selected avatars + settings.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from argparse import BooleanOptionalAction
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_SCRIPT_PATH = Path(__file__).absolute()
REPO_ROOT = _SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_label_paths_telesim import (  # type: ignore
    GPU_VIDEO_FORMAT,
    PreparedPath,
    PathSampler,
    _render_custom_gaussians,
    _serialize_camera,
    build_camera_poses,
    build_renderer,
    build_scene_asset,
    prepare_path_data,
    resolve_scene_dir,
)
from utils.ply_transform_utils import apply_transform_to_frame, build_transform_matrix, rotation_matrix_z_np
from utils.render_utils import build_look_at, load_occupancy_metadata
from utils.telesim_actor_utils import (  # type: ignore
    ActorOptions,
    ActorRenderFrame,
    ActorRuntime,
    DEFAULT_ACTOR_PATTERN,
    DEFAULT_ACTOR_SPEED,
    DEFAULT_VIDEO_FPS,
    actor_data_to_tensors,
    list_actor_frame_paths_in_dir,
    load_actor_sequence,
)
from utils.video_writer_utils import VideoWriterBackend, make_video_writer
from utils.general_utils import inverse_sigmoid
from tele_sim.core.viewer import Pose  # type: ignore
from tele_sim.rendering.gaussian_transform import _matrix_to_quat_wxyz  # type: ignore

DISTANCE_TOLERANCE_M = 1e-3


@dataclass
class HumanTrack:
    label_id: str
    path_json: Path
    prepared: PreparedPath
    sampler: PathSampler
    runtime: ActorRuntime
    actor_id: str
    actor_dir: Path
    prev_direction_xy: np.ndarray
    phase_offset_frames: int = 0
    planned_positions_xy: list[np.ndarray] | None = None
    planned_directions_xy: list[np.ndarray] | None = None


@dataclass(frozen=True)
class CameraSpeedPlan:
    camera_indices: list[int]
    achieved_min_distance_m: float
    achieved_min_human_human_distance_m: float
    frame_count: int
    holds: int
    avg_step: float
    target_distance_met: bool
    target_human_human_distance_met: bool


def _normalize_xy(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        return v / n
    if fallback is not None:
        f = np.asarray(fallback, dtype=np.float32)
        fn = float(np.linalg.norm(f))
        if fn > 1e-6:
            return f / fn
    return np.array([0.0, 1.0], dtype=np.float32)


def _smoothed_directions_from_positions(
    positions_xy: Sequence[np.ndarray],
    *,
    window: int,
    ema_alpha: float,
) -> list[np.ndarray]:
    if not positions_xy:
        return []
    pts = [np.asarray(p, dtype=np.float32) for p in positions_xy]
    n = len(pts)
    w = max(1, int(window))
    alpha = float(np.clip(ema_alpha, 0.0, 1.0))
    raw_dirs: list[np.ndarray] = []
    prev = np.array([0.0, 1.0], dtype=np.float32)
    for i in range(n):
        left = max(0, i - w)
        right = min(n - 1, i + w)
        d = pts[right] - pts[left]
        d = _normalize_xy(d, fallback=prev)
        raw_dirs.append(d)
        prev = d
    out: list[np.ndarray] = []
    prev_smooth: np.ndarray | None = None
    for d in raw_dirs:
        if prev_smooth is None:
            s = d
        else:
            blended = prev_smooth * (1.0 - alpha) + d * alpha
            s = _normalize_xy(blended, fallback=prev_smooth)
        out.append(s)
        prev_smooth = s
    return out


def _build_camera_poses_from_positions(
    *,
    camera_positions: Sequence[np.ndarray],
    floor_z: float,
    look_ahead: float,
    look_down: float,
    facing_window: int,
    facing_ema_alpha: float,
) -> list[tuple[Pose, np.ndarray]]:
    if not camera_positions:
        return []
    pos3 = [np.asarray(p, dtype=np.float32) for p in camera_positions]
    pos2 = [p[:2].copy() for p in pos3]
    dirs = _smoothed_directions_from_positions(
        pos2,
        window=facing_window,
        ema_alpha=facing_ema_alpha,
    )
    poses: list[tuple[Pose, np.ndarray]] = []
    for pos, direction_xy in zip(pos3, dirs):
        target_xy = pos[:2] + direction_xy * float(look_ahead)
        target_z = max(float(pos[2]) - abs(float(look_down)), float(floor_z) + 0.05)
        target = np.array([float(target_xy[0]), float(target_xy[1]), float(target_z)], dtype=np.float32)
        view = build_look_at(pos, target, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        rot_world = view[:3, :3].T.astype(np.float32)
        quat = _matrix_to_quat_wxyz(rot_world[None, ...])[0]
        pose = Pose(position=tuple(pos.astype(float)), orientation=tuple(quat.astype(float)))
        poses.append((pose, pos))
    return poses


class MultiActorCombinedGaussianModel:
    """Scene gaussians + N actor slices concatenated in GPU memory."""

    def __init__(self, base, actor_frames: Sequence[ActorRenderFrame]):
        if not actor_frames:
            raise ValueError("actor_frames cannot be empty.")

        device = base.get_xyz.device
        base_xyz = base._xyz.detach()
        base_dc = base._features_dc.detach()
        base_rest = base._features_rest.detach()
        base_opacity = base._opacity.detach()
        base_scaling = base._scaling.detach()
        base_rotation = base._rotation.detach()

        self.base_size = int(base_xyz.shape[0])
        self.actor_sizes = tuple(int(f.xyz.shape[0]) for f in actor_frames)
        total_actor = int(sum(self.actor_sizes))
        total = self.base_size + total_actor

        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree = base.max_sh_degree

        self._xyz = torch.empty((total, 3), device=device, dtype=base_xyz.dtype)
        self._features_dc = torch.empty(
            (total, base_dc.shape[1], base_dc.shape[2]),
            device=device,
            dtype=base_dc.dtype,
        )

        if base_rest.shape[1] > 0:
            self._features_rest = torch.empty(
                (total, base_rest.shape[1], base_rest.shape[2]),
                device=device,
                dtype=base_rest.dtype,
            )
        else:
            self._features_rest = torch.zeros((total, 0, 0), device=device, dtype=base_rest.dtype)

        self._opacity = torch.empty((total, 1), device=device, dtype=base_opacity.dtype)
        self._scaling = torch.empty((total, base_scaling.shape[1]), device=device, dtype=base_scaling.dtype)
        self._rotation = torch.empty((total, base_rotation.shape[1]), device=device, dtype=base_rotation.dtype)

        self._xyz[: self.base_size] = base_xyz
        self._features_dc[: self.base_size] = base_dc
        if self._features_rest.shape[1] > 0:
            self._features_rest[: self.base_size] = base_rest
        self._opacity[: self.base_size] = base_opacity
        self._scaling[: self.base_size] = base_scaling
        self._rotation[: self.base_size] = base_rotation

        self._actor_slices: list[slice] = []
        cursor = self.base_size
        for frame in actor_frames:
            size = int(frame.xyz.shape[0])
            seg = slice(cursor, cursor + size)
            self._actor_slices.append(seg)
            cursor += size

        self.update_actors(actor_frames)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def update_actors(self, actor_frames: Sequence[ActorRenderFrame]) -> None:
        if len(actor_frames) != len(self._actor_slices):
            raise ValueError("Actor count mismatch while updating combined model.")
        for seg, frame in zip(self._actor_slices, actor_frames):
            self._xyz[seg] = frame.xyz
            self._features_dc[seg] = frame.features_dc
            if self._features_rest.shape[1] > 0:
                self._features_rest[seg] = frame.features_rest
            self._opacity[seg] = frame.opacity
            if frame.scaling.shape[1] == 0:
                self._scaling[seg] = 0.0
            else:
                self._scaling[seg] = frame.scaling
            self._rotation[seg] = frame.rotation

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        return self._features_rest

    @property
    def get_features(self) -> torch.Tensor:
        if self._features_rest.shape[1] == 0:
            return self._features_dc
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)


def _load_ban_list(path: Path | None) -> set[str]:
    banned: set[str] = set()
    if path is None or not path.is_file():
        return banned
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        banned.add(token)
    return banned


def _flatten_csv_tokens(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        for token in str(raw).split(","):
            t = token.strip()
            if t:
                out.append(t)
    return out


def _parse_label_ids(raw: str | Sequence[str]) -> list[str]:
    tokens: list[str] = []
    if isinstance(raw, str):
        tokens = [tok for tok in raw.split(",")]
    else:
        for item in raw:
            tokens.extend(str(item).split(","))

    out: list[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if t.endswith(".json"):
            t = t[: -len(".json")]
        t = t.rstrip(" .;")
        if not t:
            continue
        out.append(t)
    return out


def _resolve_actor_dirs(
    *,
    actor_root: Path,
    actor_ids: Sequence[str],
    needed: int,
    seed: int,
    pattern: str,
    ban_list: set[str],
) -> list[Path]:
    selected: list[Path] = []
    seen: set[str] = set()

    def _maybe_add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            return
        if resolved.name in ban_list:
            return
        if not resolved.is_dir():
            return
        if not list_actor_frame_paths_in_dir(resolved, pattern=pattern):
            return
        seen.add(key)
        selected.append(resolved)

    for actor_id in actor_ids:
        p = Path(actor_id)
        if not p.is_absolute():
            p = actor_root / actor_id
        _maybe_add(p)

    if len(selected) >= needed:
        return selected[:needed]

    candidates = []
    if actor_root.is_dir():
        for child in sorted(actor_root.iterdir()):
            if child.is_dir() and child.name not in ban_list:
                if list_actor_frame_paths_in_dir(child, pattern=pattern):
                    candidates.append(child.resolve())

    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    for c in candidates:
        _maybe_add(c)
        if len(selected) >= needed:
            break

    if len(selected) < needed:
        raise RuntimeError(
            f"Not enough actor sequences under {actor_root}; needed={needed}, found={len(selected)}"
        )
    return selected[:needed]


def _resolve_gaussian_model(scene_dir: Path, override: Path | None) -> Path:
    def _has_xyz_fields(path: Path) -> bool:
        try:
            from plyfile import PlyData  # local import to keep module load lightweight

            ply = PlyData.read(str(path))
            names = ply["vertex"].data.dtype.names or ()
            return all(k in names for k in ("x", "y", "z"))
        except Exception:
            return False

    if override is not None:
        path = override.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Gaussian model not found: {path}")
        if not _has_xyz_fields(path):
            raise ValueError(
                f"Gaussian model is not TeleSim-compatible (missing x/y/z fields): {path}. "
                "Use a raw/expanded Gaussian PLY."
            )
        return path

    for name in ("3dgs_compressed.ply", "point_cloud.ply", "3dgs_raw.ply"):
        candidate = scene_dir / name
        if candidate.is_file() and _has_xyz_fields(candidate):
            return candidate

    candidates = sorted(scene_dir.glob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"No .ply Gaussian model found under scene dir: {scene_dir}")
    for candidate in candidates:
        if _has_xyz_fields(candidate):
            return candidate
    return candidates[0]


def _pingpong_index(step_idx: int, num_points: int) -> tuple[int, int]:
    if num_points <= 1:
        return 0, 1
    period = 2 * num_points - 2
    m = int(step_idx) % period
    if m < num_points:
        return m, 1
    return period - m, -1


def _build_human_actor_frame(
    *,
    human: HumanTrack,
    frame_idx: int,
    floor_z: float,
    human_step: int,
    scene_rest_dim: int,
    device: torch.device,
) -> ActorRenderFrame:
    if (
        human.planned_positions_xy is not None
        and human.planned_directions_xy is not None
        and frame_idx < len(human.planned_positions_xy)
        and frame_idx < len(human.planned_directions_xy)
    ):
        pos_xy = human.planned_positions_xy[frame_idx]
        direction_xy = human.planned_directions_xy[frame_idx]
    else:
        point_count = int(len(human.sampler.cumulative))
        step_idx = int(frame_idx) * max(1, int(human_step))
        point_idx, sign = _pingpong_index(step_idx, point_count)
        distance = float(human.sampler.cumulative[point_idx])
        pos_xy = human.sampler.position_at(distance)
        direction_xy = human.sampler.direction_at(distance)
        if sign < 0:
            direction_xy = -direction_xy
        direction_xy = _normalize_xy(direction_xy, fallback=human.prev_direction_xy)

    direction_xy = _normalize_xy(direction_xy, fallback=human.prev_direction_xy)
    human.prev_direction_xy = direction_xy

    theta = math.atan2(float(direction_xy[0]), float(direction_xy[1])) + math.pi
    rotation_np = rotation_matrix_z_np(theta)
    actor_ground_z = float(floor_z + human.runtime.options.foot_offset)
    translation_vec = np.array([float(pos_xy[0]), float(pos_xy[1]), actor_ground_z], dtype=np.float64)
    transform = build_transform_matrix(rotation_np, translation_vec)

    seq = human.runtime.sequence
    seq_len = len(seq.frames)
    if seq_len <= 0:
        raise RuntimeError(f"Actor {human.actor_id} has no animation frames.")

    # Sequential animation playback with no frame skipping:
    # each rendered frame advances by exactly one PLY frame.
    if human.runtime.options.loop:
        anim_idx = int(frame_idx) % seq_len
    else:
        anim_idx = min(int(frame_idx), seq_len - 1)

    sequence_frame = seq.frames[anim_idx]
    actor_data = apply_transform_to_frame(sequence_frame, seq, transform)
    return actor_data_to_tensors(
        actor_data,
        seq,
        device=device,
        target_rest_dim=scene_rest_dim,
    )


def _human_position_xy_at_frame(human: HumanTrack, frame_idx: int, human_step: int) -> np.ndarray:
    point_count = int(len(human.sampler.cumulative))
    step_idx = (int(frame_idx) + int(human.phase_offset_frames)) * max(1, int(human_step))
    point_idx, _ = _pingpong_index(step_idx, point_count)
    distance = float(human.sampler.cumulative[point_idx])
    return human.sampler.position_at(distance)


def _candidate_camera_advances(
    *,
    preferred_step: int,
    max_step: int,
    allow_hold: bool,
) -> list[int]:
    preferred = max(1, int(preferred_step))
    max_s = max(preferred, int(max_step))
    order: list[int] = []
    order.append(preferred)
    for s in range(preferred + 1, max_s + 1):
        order.append(s)
    for s in range(preferred - 1, -1, -1):
        order.append(s)
    if not allow_hold:
        order = [s for s in order if s > 0]
    deduped: list[int] = []
    seen: set[int] = set()
    for s in order:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _plan_camera_speed_indices(
    *,
    camera_poses: Sequence[tuple[object, np.ndarray]],
    humans: Sequence[HumanTrack],
    human_step: int,
    min_distance_m: float,
    min_human_human_distance_m: float,
    strict_human_human_distance: bool,
    preferred_step: int,
    max_step: int,
    allow_hold: bool,
    max_frames: int,
) -> CameraSpeedPlan:
    if not camera_poses:
        raise RuntimeError("Camera pose sequence is empty.")
    if min_distance_m <= 0.0:
        idxs = list(range(len(camera_poses)))
        return CameraSpeedPlan(
            camera_indices=idxs,
            achieved_min_distance_m=float("inf"),
            achieved_min_human_human_distance_m=float("inf"),
            frame_count=len(idxs),
            holds=0,
            avg_step=1.0,
            target_distance_met=True,
            target_human_human_distance_met=True,
        )

    last_idx = len(camera_poses) - 1
    camera_xy = [np.asarray(pos[:2], dtype=np.float32) for _, pos in camera_poses]
    advances = _candidate_camera_advances(
        preferred_step=preferred_step,
        max_step=max_step,
        allow_hold=allow_hold,
    )
    if not advances:
        raise RuntimeError("No camera speed candidates available for planning.")

    current_idx = 0
    frame_idx = 0
    planned: list[int] = []
    min_seen = float("inf")
    min_hh_seen = float("inf")
    holds = 0
    step_sum = 0.0
    target_met = True
    hh_target_met = True

    while True:
        if max_frames > 0 and frame_idx >= max_frames:
            raise RuntimeError(
                f"Camera speed planner exceeded max frames ({max_frames}) before reaching end of path."
            )

        human_xy = [
            _human_position_xy_at_frame(h, frame_idx=frame_idx, human_step=human_step)
            for h in humans
        ]
        if len(human_xy) >= 2:
            hh_min = float("inf")
            for i in range(len(human_xy)):
                for j in range(i + 1, len(human_xy)):
                    d = float(np.linalg.norm(human_xy[i] - human_xy[j]))
                    if d < hh_min:
                        hh_min = d
            min_hh_seen = min(min_hh_seen, hh_min)
            if hh_min + DISTANCE_TOLERANCE_M < float(min_human_human_distance_m):
                if bool(strict_human_human_distance):
                    raise RuntimeError(
                        "Unable to satisfy human-human spacing constraint at frame "
                        f"{frame_idx}: best={hh_min:.3f}m required={float(min_human_human_distance_m):.3f}m "
                        f"(tol={DISTANCE_TOLERANCE_M:.3f}m)."
                    )
                hh_target_met = False
        else:
            hh_min = float("inf")
            min_hh_seen = min(min_hh_seen, hh_min)

        candidate_rows: list[tuple[int, int, float, float]] = []
        # (cand_idx, step_taken, local_min_cam_human, local_min_human_human)
        for adv in advances:
            cand_idx = min(current_idx + int(adv), last_idx)
            step_taken = int(cand_idx - current_idx)
            cam_xy = camera_xy[cand_idx]
            local_min = float(
                min(np.linalg.norm(cam_xy - hx) for hx in human_xy)
            ) if human_xy else float("inf")
            candidate_rows.append((cand_idx, step_taken, local_min, hh_min))

        chosen_idx: int
        chosen_step: int
        chosen_min_dist: float
        target_candidates = [
            r for r in candidate_rows if r[2] + DISTANCE_TOLERANCE_M >= float(min_distance_m)
        ]
        if target_candidates:
            chosen_idx, chosen_step, chosen_min_dist, _ = min(
                target_candidates,
                key=lambda r: (abs(int(r[1]) - int(preferred_step)), -float(r[2]), -int(r[1])),
            )
        else:
            target_met = False
            chosen_idx, chosen_step, chosen_min_dist, _ = max(
                candidate_rows,
                key=lambda r: (float(r[2]), -abs(int(r[1]) - int(preferred_step)), int(r[1])),
            )

        if chosen_step == 0:
            holds += 1
        step_sum += float(chosen_step)
        current_idx = chosen_idx
        planned.append(current_idx)
        min_seen = min(min_seen, chosen_min_dist)
        frame_idx += 1

        if current_idx >= last_idx:
            break

    avg_step = (step_sum / float(len(planned))) if planned else 0.0
    return CameraSpeedPlan(
        camera_indices=planned,
        achieved_min_distance_m=float(min_seen),
        achieved_min_human_human_distance_m=float(min_hh_seen),
        frame_count=len(planned),
        holds=int(holds),
        avg_step=float(avg_step),
        target_distance_met=bool(target_met),
        target_human_human_distance_met=bool(hh_target_met),
    )


def _plan_with_phase_search(
    *,
    camera_poses: Sequence[tuple[object, np.ndarray]],
    humans: Sequence[HumanTrack],
    human_step: int,
    min_camera_human_distance_m: float,
    min_human_human_distance_m: float,
    strict_human_human_distance: bool,
    preferred_step: int,
    max_step: int,
    allow_hold: bool,
    max_frames: int,
    phase_search_trials: int,
    seed: int,
) -> tuple[CameraSpeedPlan, list[int]]:
    rng = random.Random(int(seed))
    n = len(humans)
    hs = max(1, int(human_step))

    periods: list[int] = []
    for h in humans:
        points = int(len(h.sampler.cumulative))
        steps_period = max(1, 2 * points - 2)
        frames_period = max(1, int(math.ceil(steps_period / float(hs))))
        periods.append(frames_period)

    candidate_offsets: list[list[int]] = []
    candidate_offsets.append([0] * n)
    if n > 0:
        spread = []
        for i in range(n):
            p = periods[i]
            spread.append(int((i * p) / max(1, n)))
        candidate_offsets.append(spread)
    for _ in range(max(0, int(phase_search_trials))):
        candidate_offsets.append(
            [rng.randrange(periods[i]) if periods[i] > 1 else 0 for i in range(n)]
        )

    best_plan: CameraSpeedPlan | None = None
    best_offsets: list[int] | None = None
    last_error: str | None = None

    for offsets in candidate_offsets:
        for h, off in zip(humans, offsets):
            h.phase_offset_frames = int(off)
            h.planned_positions_xy = None
            h.planned_directions_xy = None
        try:
            plan = _plan_camera_speed_indices(
                camera_poses=camera_poses,
                humans=humans,
                human_step=hs,
                min_distance_m=float(min_camera_human_distance_m),
                min_human_human_distance_m=float(min_human_human_distance_m),
                strict_human_human_distance=bool(strict_human_human_distance),
                preferred_step=max(1, int(preferred_step)),
                max_step=max(1, int(max_step)),
                allow_hold=bool(allow_hold),
                max_frames=max(0, int(max_frames)),
            )
        except RuntimeError as exc:
            last_error = str(exc)
            continue

        if best_plan is None:
            best_plan = plan
            best_offsets = list(offsets)
        else:
            # Prefer plans that meet constraints first, then maximize clearances.
            curr = (
                int(bool(plan.target_distance_met)),
                int(bool(plan.target_human_human_distance_met)),
                float(plan.achieved_min_distance_m),
                float(plan.achieved_min_human_human_distance_m),
                -int(plan.holds),
                -int(plan.frame_count),
            )
            best = (
                int(bool(best_plan.target_distance_met)),
                int(bool(best_plan.target_human_human_distance_met)),
                float(best_plan.achieved_min_distance_m),
                float(best_plan.achieved_min_human_human_distance_m),
                -int(best_plan.holds),
                -int(best_plan.frame_count),
            )
            if curr > best:
                best_plan = plan
                best_offsets = list(offsets)

        if best_plan.target_distance_met and best_plan.target_human_human_distance_met:
            break

    if best_plan is None or best_offsets is None:
        suffix = f" Last planner error: {last_error}" if last_error else ""
        raise RuntimeError(
            "Failed to find a valid camera/human schedule with current constraints." + suffix
        )

    for h, off in zip(humans, best_offsets):
        h.phase_offset_frames = int(off)
        h.planned_positions_xy = None
        h.planned_directions_xy = None
    return best_plan, best_offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one scene with one camera path and multiple ping-pong walking humans (TeleSim)."
    )

    parser.add_argument("--scenes-dir", type=Path, default=REPO_ROOT / "data" / "scenes")
    parser.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "data" / "interiorGS_0500_42")
    parser.add_argument("--scene", default="0001_839920")

    parser.add_argument("--camera-label", default="78", help="Label id for camera path (default: 78).")
    parser.add_argument(
        "--human-labels",
        default="82,97,141,247,330,103,648,1028",
        help="Comma-separated label ids for human paths (default: 82,97,141,247,330,103,648,1028).",
    )

    parser.add_argument("--actor-root", type=Path, default=REPO_ROOT / "data" / "human_gs_source")
    parser.add_argument(
        "--avatar-id",
        action="append",
        default=None,
        help="Optional actor id(s) under actor-root. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--ban-list", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data1" / "multihuman_demo")
    parser.add_argument("--scene-name", default=None, help="Output folder name (default: scene id).")
    parser.add_argument("--video-name", default=None, help="Video filename (default: <camera_label>_multihuman.mp4).")
    parser.add_argument("--manifest-name", default="multihuman_demo.json")

    parser.add_argument("--overwrite", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Only write manifest; do not render video.")

    parser.add_argument("--gaussian-model", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, nargs=2, default=[960, 720])
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--znear", type=float, default=0.001)
    parser.add_argument("--zfar", type=float, default=30.0)
    parser.add_argument("--antialiasing", action=BooleanOptionalAction, default=False)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--separate-sh", action=BooleanOptionalAction, default=False)
    parser.add_argument("--use-trained-exposure", action=BooleanOptionalAction, default=False)

    parser.add_argument("--mirror-translation", action=BooleanOptionalAction, default=True)
    parser.add_argument("--path-handedness", choices=["left", "right", "auto"], default="left")
    parser.add_argument("--swap-xy", action=BooleanOptionalAction, default=False)
    parser.add_argument("--negate-xy", action=BooleanOptionalAction, default=False)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resample-step", type=float, default=0.0)

    parser.add_argument("--camera-follow-distance", type=float, default=0.0)
    parser.add_argument("--height-offset", type=float, default=0.3)
    parser.add_argument("--look-ahead", type=float, default=2.0)
    parser.add_argument("--look-down", type=float, default=0.1)
    parser.add_argument("--stabilize", action=BooleanOptionalAction, default=True)
    parser.add_argument("--minimal-frames", type=int, default=0)
    parser.add_argument(
        "--facing-window",
        type=int,
        default=5,
        help="Window (in frames) used for heading estimation for both camera and humans (default: 5).",
    )
    parser.add_argument(
        "--facing-ema-alpha",
        type=float,
        default=0.35,
        help="EMA alpha used to smooth heading vectors for both camera and humans (default: 0.35).",
    )
    parser.add_argument(
        "--min-camera-human-distance",
        type=float,
        default=1.0,
        help="Minimum allowed XY distance (meters) between camera and any human per frame (default: 1.0).",
    )
    parser.add_argument(
        "--min-human-human-distance",
        type=float,
        default=0.5,
        help="Minimum allowed XY distance (meters) between any pair of humans per frame (default: 0.5).",
    )
    parser.add_argument(
        "--strict-human-human-distance",
        action=BooleanOptionalAction,
        default=False,
        help="If true, fail planning when human-human distance target cannot be met; otherwise best-effort (default: false).",
    )
    parser.add_argument(
        "--camera-speed-ratio-max",
        type=float,
        default=2.5,
        help="Maximum camera speed ratio relative to human speed (default: 2.5).",
    )
    parser.add_argument(
        "--camera-preferred-step",
        type=int,
        default=1,
        help="Preferred camera path-index advance per frame before speed adjustment (default: 1).",
    )
    parser.add_argument(
        "--camera-max-step",
        type=int,
        default=4,
        help="Maximum camera path-index advance per frame for clearance planning (default: 4).",
    )
    parser.add_argument(
        "--camera-allow-hold",
        action=BooleanOptionalAction,
        default=True,
        help="Allow camera to stay on the same path index for a frame when needed for clearance (default: true).",
    )
    parser.add_argument(
        "--camera-planner-max-frames",
        type=int,
        default=200000,
        help="Safety cap on planned output frame count while enforcing clearance (default: 200000).",
    )
    parser.add_argument(
        "--phase-search-trials",
        type=int,
        default=240,
        help="Number of random human phase-offset trials for scheduling search (default: 240).",
    )

    parser.add_argument("--actor-pattern", default=DEFAULT_ACTOR_PATTERN)
    parser.add_argument("--actor-height", type=float, default=1.7)
    parser.add_argument("--actor-speed", type=float, default=DEFAULT_ACTOR_SPEED)
    parser.add_argument("--actor-fps", type=float, default=float(DEFAULT_VIDEO_FPS))
    parser.add_argument("--actor-foot-offset", type=float, default=0.0)
    parser.add_argument("--animation-cycle-mod", type=int, default=3)
    parser.add_argument("--actor-loop", action=BooleanOptionalAction, default=True)
    parser.add_argument(
        "--human-step",
        type=int,
        default=1,
        help="Path-point advance per frame for each human (default: 1).",
    )

    parser.add_argument("--video", action=BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--rotate-180", action=BooleanOptionalAction, default=True)
    parser.add_argument(
        "--video-backend",
        choices=[v.value for v in VideoWriterBackend],
        default=VideoWriterBackend.CPU.value,
    )
    parser.add_argument("--video-nvenc-preset", default=None)
    parser.add_argument("--video-nvenc-bitrate", default=None)

    parser.add_argument("--save-camera-metadata", action=BooleanOptionalAction, default=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.scenes_dir = args.scenes_dir.expanduser().resolve()
    args.tasks_dir = args.tasks_dir.expanduser().resolve()
    args.actor_root = args.actor_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if args.path_handedness == "auto":
        args.path_handedness = "left"

    scene_id = str(args.scene)
    scene_name = str(args.scene_name or scene_id)
    output_dir = args.output_root / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = args.video_name or f"{str(args.camera_label)}_multihuman.mp4"
    video_path = output_dir / video_name
    manifest_path = output_dir / str(args.manifest_name)

    if video_path.exists() and not bool(args.overwrite) and not args.dry_run:
        raise FileExistsError(f"Output video exists and overwrite is disabled: {video_path}")

    scene_dir = resolve_scene_dir(args.scenes_dir, scene_id)
    tasks_scene_dir = args.tasks_dir / scene_id
    if not tasks_scene_dir.is_dir():
        raise FileNotFoundError(f"Task scene dir not found: {tasks_scene_dir}")

    camera_label = str(args.camera_label)
    human_labels = _parse_label_ids(args.human_labels)
    if len(human_labels) < 1:
        raise ValueError("At least one human path label is required.")

    camera_json = tasks_scene_dir / f"{camera_label}.json"
    if not camera_json.is_file():
        raise FileNotFoundError(f"Camera path JSON not found: {camera_json}")

    human_jsons: list[Path] = []
    for label in human_labels:
        path = tasks_scene_dir / f"{label}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Human path JSON not found: {path}")
        human_jsons.append(path)

    default_ban = args.actor_root / "BanList.txt"
    ban_path = args.ban_list.expanduser().resolve() if args.ban_list is not None else (default_ban if default_ban.is_file() else None)
    banned = _load_ban_list(ban_path)

    actor_ids = _flatten_csv_tokens(args.avatar_id)
    actor_dirs = _resolve_actor_dirs(
        actor_root=args.actor_root,
        actor_ids=actor_ids,
        needed=len(human_labels),
        seed=int(args.seed),
        pattern=str(args.actor_pattern),
        ban_list=banned,
    )

    gaussian_model = _resolve_gaussian_model(scene_dir, args.gaussian_model)
    meta = load_occupancy_metadata(scene_dir)

    camera_prepared = prepare_path_data(
        camera_json,
        meta,
        stride=max(1, int(args.stride)),
        resample_step=float(args.resample_step),
        mirror_translation=bool(args.mirror_translation),
        swap_xy=bool(args.swap_xy),
        handedness=str(args.path_handedness),
        negate_xy=bool(args.negate_xy),
    )

    camera_poses = build_camera_poses(
        camera_prepared.path_xy,
        floor_z=camera_prepared.floor_z,
        ceiling=camera_prepared.ceiling,
        follow_distance=float(args.camera_follow_distance),
        height_offset=float(args.height_offset),
        look_ahead=float(args.look_ahead),
        look_down=float(args.look_down),
        stabilize=bool(args.stabilize),
    )
    if int(args.minimal_frames) > 0:
        camera_poses = camera_poses[: int(args.minimal_frames)]
    if not camera_poses:
        raise RuntimeError("Camera pose sequence is empty.")

    human_tracks: list[HumanTrack] = []
    for idx, (label_id, path_json, actor_dir) in enumerate(zip(human_labels, human_jsons, actor_dirs)):
        prepared = prepare_path_data(
            path_json,
            meta,
            stride=max(1, int(args.stride)),
            resample_step=float(args.resample_step),
            mirror_translation=bool(args.mirror_translation),
            swap_xy=bool(args.swap_xy),
            handedness=str(args.path_handedness),
            negate_xy=bool(args.negate_xy),
        )
        if len(prepared.path_xy) < 2:
            raise RuntimeError(f"Human path has fewer than 2 points: {path_json}")

        actor_options = ActorOptions(
            sequence_dir=actor_dir,
            pattern=str(args.actor_pattern),
            height=float(args.actor_height),
            follow_distance=1.5,
            buffer_distance=0.0,
            speed=float(args.actor_speed),
            fps=float(args.actor_fps),
            loop=bool(args.actor_loop),
            foot_offset=float(args.actor_foot_offset),
            animation_cycle_mod=max(1, int(args.animation_cycle_mod)),
        )
        actor_sequence = load_actor_sequence(actor_options, debug=False)
        runtime = ActorRuntime(options=actor_options, sequence=actor_sequence)

        human_tracks.append(
            HumanTrack(
                label_id=str(label_id),
                path_json=path_json,
                prepared=prepared,
                sampler=PathSampler([pt[:2] for pt in prepared.path_xy]),
                runtime=runtime,
                actor_id=actor_dir.name,
                actor_dir=actor_dir,
                prev_direction_xy=np.array([0.0, 1.0], dtype=np.float32),
            )
        )

    human_step_effective = max(1, int(args.human_step))
    max_step_ratio_capped = max(
        1,
        int(math.floor(float(args.camera_speed_ratio_max) * float(human_step_effective))),
    )
    max_camera_step = min(max(1, int(args.camera_max_step)), max_step_ratio_capped)
    camera_plan, phase_offsets = _plan_with_phase_search(
        camera_poses=camera_poses,
        humans=human_tracks,
        human_step=human_step_effective,
        min_camera_human_distance_m=float(args.min_camera_human_distance),
        min_human_human_distance_m=float(args.min_human_human_distance),
        strict_human_human_distance=bool(args.strict_human_human_distance),
        preferred_step=max(1, int(args.camera_preferred_step)),
        max_step=max_camera_step,
        allow_hold=bool(args.camera_allow_hold),
        max_frames=max(0, int(args.camera_planner_max_frames)),
        phase_search_trials=max(0, int(args.phase_search_trials)),
        seed=int(args.seed),
    )
    planned_camera_positions = [camera_poses[i][1].copy() for i in camera_plan.camera_indices]
    planned_camera_poses = _build_camera_poses_from_positions(
        camera_positions=planned_camera_positions,
        floor_z=float(camera_prepared.floor_z),
        look_ahead=float(args.look_ahead),
        look_down=float(args.look_down),
        facing_window=max(1, int(args.facing_window)),
        facing_ema_alpha=float(args.facing_ema_alpha),
    )
    planned_frames = len(planned_camera_poses)
    for human in human_tracks:
        h_positions = [
            _human_position_xy_at_frame(
                human,
                frame_idx=fi,
                human_step=max(1, int(args.human_step)),
            )
            for fi in range(planned_frames)
        ]
        h_dirs = _smoothed_directions_from_positions(
            h_positions,
            window=max(1, int(args.facing_window)),
            ema_alpha=float(args.facing_ema_alpha),
        )
        human.planned_positions_xy = h_positions
        human.planned_directions_xy = h_dirs

    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene": scene_id,
        "scene_name": scene_name,
        "paths": {
            "camera_label": camera_label,
            "camera_json": str(camera_json),
            "camera_world_xy": [[float(p[0]), float(p[1])] for p in camera_prepared.path_xy],
            "humans": [
                {
                    "slot": i,
                    "label": t.label_id,
                    "json": str(t.path_json),
                    "world_xy": [[float(p[0]), float(p[1])] for p in t.prepared.path_xy],
                    "movement_mode": "ping_pong",
                    "step_per_frame": int(max(1, int(args.human_step))),
                }
                for i, t in enumerate(human_tracks)
            ],
        },
        "avatars": [
            {
                "slot": i,
                "actor_id": t.actor_id,
                "actor_dir": str(t.actor_dir),
                "pattern": str(args.actor_pattern),
                "height": float(args.actor_height),
                "speed": float(args.actor_speed),
                "fps": float(args.actor_fps),
                "foot_offset": float(args.actor_foot_offset),
                "loop": bool(args.actor_loop),
                "animation_cycle_mod": int(max(1, int(args.animation_cycle_mod))),
                "animation_frame_mode": "sequential_no_skip",
                "phase_offset_frames": int(t.phase_offset_frames),
            }
            for i, t in enumerate(human_tracks)
        ],
        "render": {
            "scenes_dir": str(args.scenes_dir),
            "tasks_dir": str(args.tasks_dir),
            "scene_dir": str(scene_dir),
            "gaussian_model": str(gaussian_model),
            "output_dir": str(output_dir),
            "video": str(video_path),
            "video_backend": str(args.video_backend),
            "video_fps": int(args.video_fps),
            "rotate_180": bool(args.rotate_180),
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
            "fov_deg": float(args.fov_deg),
            "height_offset": float(args.height_offset),
            "look_ahead": float(args.look_ahead),
            "look_down": float(args.look_down),
            "mirror_translation": bool(args.mirror_translation),
            "path_handedness": str(args.path_handedness),
            "swap_xy": bool(args.swap_xy),
            "negate_xy": bool(args.negate_xy),
            "stride": int(args.stride),
            "resample_step": float(args.resample_step),
            "facing_smoothing": {
                "window_frames": int(max(1, int(args.facing_window))),
                "ema_alpha": float(args.facing_ema_alpha),
                "method": "windowed_direction_plus_ema",
            },
            "camera_speed_planner": {
                "enabled": True,
                "min_camera_human_distance_m": float(args.min_camera_human_distance),
                "min_human_human_distance_m": float(args.min_human_human_distance),
                "strict_human_human_distance": bool(args.strict_human_human_distance),
                "preferred_step": int(max(1, int(args.camera_preferred_step))),
                "max_step": int(max_camera_step),
                "max_step_user": int(max(1, int(args.camera_max_step))),
                "max_step_ratio_cap": int(max_step_ratio_capped),
                "camera_speed_ratio_max": float(args.camera_speed_ratio_max),
                "allow_hold": bool(args.camera_allow_hold),
                "max_frames": int(max(0, int(args.camera_planner_max_frames))),
                "phase_search_trials": int(max(0, int(args.phase_search_trials))),
                "phase_offsets_frames": [int(x) for x in phase_offsets],
                "planned_frames": int(camera_plan.frame_count),
                "holds": int(camera_plan.holds),
                "avg_step": float(camera_plan.avg_step),
                "achieved_min_distance_m": float(camera_plan.achieved_min_distance_m),
                "achieved_min_human_human_distance_m": float(
                    camera_plan.achieved_min_human_human_distance_m
                ),
                "target_distance_met": bool(camera_plan.target_distance_met),
                "target_human_human_distance_met": bool(
                    camera_plan.target_human_human_distance_met
                ),
            },
        },
        "runtime": {
            "frames": int(planned_frames),
            "dry_run": bool(args.dry_run),
            "seed": int(args.seed),
            "ban_list": (str(ban_path) if ban_path is not None else None),
        },
    }

    if args.dry_run:
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        print(f"[DRYRUN] Wrote manifest: {manifest_path}")
        return 0

    asset = build_scene_asset(scene_dir, gaussian_model, meta)
    renderer = build_renderer(asset, args)

    base_gaussians = renderer._gaussians  # pylint: disable=protected-access
    scene_rest_dim = int(base_gaussians.get_features_rest.shape[1])
    device = torch.device(str(args.device))

    frames_rendered = 0
    camera_frames: list[dict] = []
    combined_model: MultiActorCombinedGaussianModel | None = None
    combined_sizes: tuple[int, ...] | None = None

    t0 = time.monotonic()

    if bool(args.video):
        backend = VideoWriterBackend(str(args.video_backend))
        with make_video_writer(
            video_path,
            fps=float(args.video_fps),
            backend=backend,
            nvenc_preset=args.video_nvenc_preset,
            nvenc_bitrate=args.video_nvenc_bitrate,
            width=int(args.resolution[0]),
            height=int(args.resolution[1]),
            gpu_format=GPU_VIDEO_FORMAT,
        ) as writer:
            for idx, (pose, _) in enumerate(planned_camera_poses):
                actor_frames: list[ActorRenderFrame] = []
                for human in human_tracks:
                    actor_frame = _build_human_actor_frame(
                        human=human,
                        frame_idx=idx,
                        floor_z=float(camera_prepared.floor_z),
                        human_step=max(1, int(args.human_step)),
                        scene_rest_dim=scene_rest_dim,
                        device=device,
                    )
                    actor_frames.append(actor_frame)

                frame_sizes = tuple(int(a.xyz.shape[0]) for a in actor_frames)
                if combined_model is None or combined_sizes != frame_sizes:
                    combined_model = MultiActorCombinedGaussianModel(base_gaussians, actor_frames)
                    combined_sizes = frame_sizes
                else:
                    combined_model.update_actors(actor_frames)

                rgb, _, camera = _render_custom_gaussians(
                    renderer,
                    pose,
                    combined_model,
                    need_depth_inv=False,
                )
                if bool(args.rotate_180):
                    rgb = np.flipud(np.fliplr(rgb))
                writer.append_data(rgb)

                if bool(args.save_camera_metadata):
                    cam_payload = _serialize_camera(
                        renderer=renderer,
                        pose=pose,
                        frame_size=(int(args.resolution[0]), int(args.resolution[1])),
                        fov_y_rad=math.radians(float(args.fov_deg)),
                    )
                    camera_frames.append({"frame": int(idx), **cam_payload})

                del camera
                frames_rendered += 1

    elapsed = time.monotonic() - t0

    if bool(args.save_camera_metadata):
        camera_meta_path = output_dir / f"{camera_label}_camera.json"
        camera_meta_payload = {
            "dataset_root": str(output_dir.parent),
            "scene": scene_name,
            "label": camera_label,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "frames": camera_frames,
        }
        camera_meta_path.write_text(json.dumps(camera_meta_payload, indent=2), encoding="utf-8")
        manifest_payload["render"]["camera_metadata"] = str(camera_meta_path)

    manifest_payload["runtime"].update(
        {
            "frames": int(frames_rendered),
            "elapsed_sec": float(elapsed),
            "fps_effective": (float(frames_rendered) / elapsed if elapsed > 0 else None),
            "dry_run": False,
        }
    )

    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"[OK] Video: {video_path}")
    print(f"[OK] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
