from __future__ import annotations

import json
import math
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utils.render_utils import (
    build_look_at,
    deduplicate_points,
    derive_affine_transform,
    load_raster_world_points,
    sample_points,
)

EPS = 1e-6


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def signed_angle_delta(a: float, b: float) -> float:
    delta = b - a
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


@dataclass(frozen=True)
class PreparedPath:
    path_xyz: list[np.ndarray]
    floor_z: float
    ceiling: float


class PathSampler:
    def __init__(self, points_xy: Sequence[np.ndarray]):
        if len(points_xy) < 2:
            raise ValueError("PathSampler requires at least two points.")
        raw = np.asarray(points_xy, dtype=np.float32)
        diffs = raw[1:] - raw[:-1]
        lengths = np.linalg.norm(diffs, axis=1)
        valid = lengths > 1e-6
        cleaned = [raw[0]]
        vectors: list[np.ndarray] = []
        seg_lengths: list[float] = []
        for idx, ok in enumerate(valid):
            if not ok:
                continue
            cleaned.append(raw[idx + 1])
            vectors.append(diffs[idx])
            seg_lengths.append(float(lengths[idx]))
        self.points = np.asarray(cleaned, dtype=np.float32)
        self.segment_vectors = np.asarray(vectors, dtype=np.float32)
        self.segment_lengths = np.asarray(seg_lengths, dtype=np.float32)
        self.cumulative = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(self.segment_lengths)])

    @property
    def total_length(self) -> float:
        return float(self.cumulative[-1])

    def position_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            direction = self.segment_vectors[0] / max(self.segment_lengths[0], EPS)
            return self.points[0] + direction * distance
        total = self.total_length
        if distance >= total:
            direction = self.segment_vectors[-1] / max(self.segment_lengths[-1], EPS)
            return self.points[-1] + direction * (distance - total)
        seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
        seg_offset = distance - self.cumulative[seg_idx]
        ratio = seg_offset / max(float(self.segment_lengths[seg_idx]), EPS)
        return self.points[seg_idx] + self.segment_vectors[seg_idx] * ratio


def resample_path_by_distance(points_xy: Sequence[np.ndarray], step: float) -> list[np.ndarray]:
    if step <= 0.0 or len(points_xy) < 2:
        return list(points_xy)
    sampler = PathSampler(points_xy)
    total = sampler.total_length
    if total <= step:
        return list(points_xy)
    distances: list[float] = []
    dist = 0.0
    while dist < total:
        distances.append(dist)
        dist += step
    distances.append(total)
    resampled = [sampler.position_at(float(d)) for d in distances]
    deduped = [resampled[0]]
    for point in resampled[1:]:
        if np.linalg.norm(point - deduped[-1]) > 1e-4:
            deduped.append(point)
    return [np.array([pt[0], pt[1]], dtype=np.float32) for pt in deduped]


def forward_direction(points_xyz: Sequence[np.ndarray], idx: int, window: int) -> np.ndarray:
    if len(points_xyz) == 1:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    accum = np.zeros(2, dtype=np.float32)
    count = 0
    max_step = max(1, int(window))
    for step in range(1, max_step + 1):
        nxt = min(idx + step, len(points_xyz) - 1)
        delta = points_xyz[nxt][:2] - points_xyz[idx][:2]
        if np.linalg.norm(delta) > 1e-4:
            accum += delta
            count += 1
    for step in range(1, max_step + 1):
        prev = max(idx - step, 0)
        delta = points_xyz[idx][:2] - points_xyz[prev][:2]
        if np.linalg.norm(delta) > 1e-4:
            accum += delta
            count += 1
    if count == 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    direction_xy = accum / float(count)
    norm = np.linalg.norm(direction_xy)
    if norm < 1e-4:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([direction_xy[0] / norm, direction_xy[1] / norm, 0.0], dtype=np.float32)


def prepare_path_data(
    json_path: Path,
    meta: dict[str, Any],
    *,
    stride: int,
    resample_step: float,
    mirror_translation: bool,
    swap_xy: bool,
    handedness: str,
    negate_xy: bool,
) -> PreparedPath:
    raw_points, raster_pixels = load_raster_world_points(
        json_path,
        swap_xy=swap_xy,
        handedness=handedness,
        negate_xy=negate_xy,
    )
    a_x, b_x, a_y, b_y = derive_affine_transform(raw_points, raster_pixels, meta)
    transformed = [np.array([a_x * pt[0] + b_x, a_y * pt[1] + b_y], dtype=np.float32) for pt in raw_points]
    points_xy = deduplicate_points(transformed)
    sampled_xy = sample_points(points_xy, max(1, int(stride)))
    if len(sampled_xy) < 2:
        sampled_xy = points_xy

    if mirror_translation:
        center_x = 0.5 * (float(meta["left"]) + float(meta["right"]))
        center_y = 0.5 * (float(meta["top"]) + float(meta["bottom"]))
        sampled_xy = [
            np.array([center_x * 2.0 - pt[0], center_y * 2.0 - pt[1]], dtype=np.float32) for pt in sampled_xy
        ]

    if resample_step > 0.0:
        resampled = resample_path_by_distance(sampled_xy, float(resample_step))
        if len(resampled) >= 2:
            sampled_xy = resampled

    return PreparedPath(
        path_xyz=[np.array([pt[0], pt[1], 0.0], dtype=np.float32) for pt in sampled_xy],
        floor_z=float(meta["lower_z"]),
        ceiling=float(meta["upper_z"]),
    )


def compute_projection_matrix(*, znear: float, zfar: float, fovx: float, fovy: float) -> np.ndarray:
    tan_half_fovy = math.tan(fovy / 2.0)
    tan_half_fovx = math.tan(fovx / 2.0)
    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right

    p = np.zeros((4, 4), dtype=np.float64)
    z_sign = 1.0
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def serialize_camera_frame(
    *,
    eye_world: np.ndarray,
    target_world: np.ndarray,
    frame_size: tuple[int, int],
    fov_y_rad: float,
    znear: float,
    zfar: float,
) -> dict[str, Any]:
    view = build_look_at(
        eye_world.astype(np.float32),
        target_world.astype(np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ).astype(np.float64)

    # Match render_label_paths(.py) convention: store TRANSPOSED matrices.
    world_to_camera = view.T
    camera_to_world = np.linalg.inv(world_to_camera)

    w, h = int(frame_size[0]), int(frame_size[1])
    fovx = 2.0 * math.atan(math.tan(fov_y_rad * 0.5) * (w / float(h)))
    fx = w / (2.0 * math.tan(fovx * 0.5))
    fy = h / (2.0 * math.tan(fov_y_rad * 0.5))

    proj = compute_projection_matrix(znear=float(znear), zfar=float(zfar), fovx=float(fovx), fovy=float(fov_y_rad))
    full_proj = world_to_camera @ proj.T

    cam_center = camera_to_world[3, :3].tolist()

    return {
        "type": "perspective",
        "resolution": {"width": w, "height": h},
        "fov": {
            "x_rad": float(fovx),
            "y_rad": float(fov_y_rad),
            "x_deg": math.degrees(float(fovx)),
            "y_deg": math.degrees(float(fov_y_rad)),
        },
        "znear": float(znear),
        "zfar": float(zfar),
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(w * 0.5),
            "cy": float(h * 0.5),
            "half_width": None,
            "half_height": None,
        },
        "camera_center_world": cam_center,
        "world_to_camera": world_to_camera.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "projection_matrix": full_proj.tolist(),
    }


def world_to_pixel(meta: dict[str, Any], x: float, y: float) -> tuple[int, int]:
    u = int(round((x - float(meta["left"])) / float(meta["scale"])))
    v = int(round((float(meta["top"]) - y) / float(meta["scale"])))
    return u, v


def build_camera_frame_payloads_for_path(
    *,
    prepared: PreparedPath,
    meta: dict[str, Any],
    follow_distance: float,
    height_offset: float,
    look_ahead: float,
    look_down: float,
    stabilize: bool,
    minimal_frames: int,
    resolution: tuple[int, int],
    fov_y_rad: float,
    znear: float,
    zfar: float,
) -> list[dict[str, Any]]:
    sampler = PathSampler([pt[:2] for pt in prepared.path_xyz])
    distances = list(sampler.cumulative)
    total_length = sampler.total_length
    follow = max(0.0, float(follow_distance))
    max_cam_dist = max(total_length - follow, 0.0)

    camera_positions: list[np.ndarray] = []
    for dist in distances:
        cam_dist = min(float(dist), max_cam_dist)
        xy = sampler.position_at(float(cam_dist))
        camera_positions.append(
            np.array([xy[0], xy[1], prepared.ceiling + float(height_offset)], dtype=np.float32)
        )
        if cam_dist >= max_cam_dist - 1e-6:
            break

    direction_window = 5 if stabilize else 1
    prev_forward: np.ndarray | None = None
    payloads: list[dict[str, Any]] = []
    for idx, eye in enumerate(camera_positions):
        fwd = forward_direction(camera_positions, idx, window=direction_window)
        if np.linalg.norm(fwd[:2]) < EPS:
            fwd = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if stabilize and prev_forward is not None:
            blended = prev_forward * (1.0 - 0.35) + fwd * 0.35
            bnorm = float(np.linalg.norm(blended))
            if bnorm > EPS:
                fwd = (blended / bnorm).astype(np.float32)
        prev_forward = fwd.copy()

        target_xy = eye[:2] + fwd[:2] * float(look_ahead)
        target_z = max(float(eye[2]) - abs(float(look_down)), float(prepared.floor_z + 0.05))
        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

        payloads.append(
            serialize_camera_frame(
                eye_world=eye,
                target_world=target,
                frame_size=(int(resolution[0]), int(resolution[1])),
                fov_y_rad=float(fov_y_rad),
                znear=float(znear),
                zfar=float(zfar),
            )
        )

        if minimal_frames > 0 and (idx + 1) >= int(minimal_frames):
            break

    return payloads


def action_payload_from_camera_payloads(
    *,
    camera_root: Path,
    scene_id: str,
    label_id: str,
    meta: dict[str, Any],
    camera_payloads: Sequence[dict[str, Any]],
    max_next: int,
    action_yaw_window: int,
    move_threshold_deg: float,
    turn_threshold_deg: float,
    turn_threshold_scale: float,
    ahead_dot_eps: float,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for idx, cam_payload in enumerate(camera_payloads):
        cam_center = cam_payload["camera_center_world"]
        cam_to_world = cam_payload["camera_to_world"]
        forward_row = cam_to_world[2]
        fx = float(forward_row[0])
        fy = float(forward_row[1])
        norm = math.hypot(fx, fy)
        if norm < 1e-6:
            fx, fy = 0.0, 1.0
        else:
            fx, fy = fx / norm, fy / norm
        yaw = math.atan2(fy, fx)
        x, y, z = map(float, cam_center[:3])
        u, v = world_to_pixel(meta, x, y)
        frames.append(
            {
                "frame": int(idx),
                "world": [x, y, z],
                "pixel": [int(u), int(v)],
                "forward": [float(fx), float(fy)],
                "yaw": float(yaw),
            }
        )

    action_window = max(1, int(action_yaw_window))
    action_turn_threshold = float(turn_threshold_deg) * float(turn_threshold_scale)

    actions: list[str] = []
    for i, frame in enumerate(frames):
        if i == len(frames) - 1:
            actions.append("stop")
            continue
        curr = frame
        nxt = frames[i + 1]
        yaw_idx = min(i + action_window, len(frames) - 1)
        yaw_target = frames[yaw_idx]
        delta_yaw = signed_angle_delta(float(curr["yaw"]), float(yaw_target["yaw"]))
        angle_deg = abs(math.degrees(delta_yaw))
        dx = float(nxt["world"][0]) - float(curr["world"][0])
        dy = float(nxt["world"][1]) - float(curr["world"][1])
        dot = dx * float(curr["forward"][0]) + dy * float(curr["forward"][1])
        ahead = dot > float(ahead_dot_eps)
        if angle_deg >= action_turn_threshold:
            actions.append("turn left" if delta_yaw > 0 else "turn right")
        elif angle_deg <= float(move_threshold_deg) and ahead:
            actions.append("move")
        else:
            actions.append("move" if ahead else "stop")

    action_codes = {"stop": 0, "move": 1, "turn left": 2, "turn right": 3}
    per_frame: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        next_actions: list[int] = []
        for j in range(1, max(1, int(max_next)) + 1):
            if i + j < len(actions):
                next_actions.append(int(action_codes[actions[i + j]]))
            else:
                next_actions.append(int(action_codes["stop"]))
        per_frame.append(
            {
                "frame": int(frame["frame"]),
                "world": frame["world"],
                "pixel": frame["pixel"],
                "curr_action": int(action_codes[actions[i]]),
                "next_actions": next_actions,
            }
        )

    return {
        "dataset_root": str(camera_root),
        "scene": scene_id,
        "label": label_id,
        "generated_at": utc_now_iso(),
        "max_next": int(max_next),
        "move_threshold_deg": float(move_threshold_deg),
        "turn_threshold_deg": float(turn_threshold_deg),
        "turn_threshold_scale": float(turn_threshold_scale),
        "action_turn_threshold_deg": float(action_turn_threshold),
        "action_yaw_window": int(action_window),
        "frames": per_frame,
    }


def _normalize_xy(vec: tuple[float, float]) -> tuple[float, float]:
    x, y = vec
    norm = (x * x + y * y) ** 0.5
    if norm <= 1e-9:
        return 0.0, 0.0
    return x / norm, y / norm


def _smooth_dirs(
    dirs: list[tuple[float, float] | None],
    window: int = 5,
) -> list[tuple[float, float] | None]:
    n = len(dirs)
    if n == 0 or window <= 1:
        return dirs
    half = window // 2
    yaws: list[float | None] = [None] * n
    last = None
    for i, d in enumerate(dirs):
        if d is not None:
            last = math.atan2(d[1], d[0])
            yaws[i] = last
        else:
            yaws[i] = None

    first_yaw = next((y for y in yaws if y is not None), None)
    if first_yaw is None:
        return dirs
    for i in range(n):
        if yaws[i] is None:
            yaws[i] = last if last is not None else first_yaw
        else:
            last = yaws[i]

    last = yaws[-1]
    for i in range(n - 1, -1, -1):
        if yaws[i] is None:
            yaws[i] = last
        else:
            last = yaws[i]

    smoothed: list[tuple[float, float] | None] = [None] * n
    sin = math.sin
    cos = math.cos
    atan2 = math.atan2
    for i in range(n):
        angles: list[float] = []
        for k in range(i - half, i + half + 1):
            kk = min(max(0, k), n - 1)
            angles.append(float(yaws[kk]))
        sum_sin = sum(sin(a) for a in angles)
        sum_cos = sum(cos(a) for a in angles)
        mean_angle = atan2(sum_sin, sum_cos)
        smoothed[i] = (math.cos(mean_angle), math.sin(mean_angle))
    return smoothed


def _compute_dirs(frames: list[dict[str, Any]]) -> list[tuple[float, float] | None]:
    dirs: list[tuple[float, float] | None] = [None] * len(frames)
    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]
        dx = float(curr["world"][0]) - float(prev["world"][0])
        dy = float(curr["world"][1]) - float(prev["world"][1])
        dirs[i] = _normalize_xy((dx, dy))
    return dirs


def _compute_prev_actions(
    *,
    frames: list[dict[str, Any]],
    dirs: list[tuple[float, float] | None],
    index: int,
    window: int,
    padding_action: int,
    padding_frame: int,
    turn_threshold_deg: float,
    forward_action: int,
    left_action: int,
    right_action: int,
) -> tuple[list[int], list[int]]:
    curr = index
    prev_actions: list[int] = []
    prev_frames: list[int] = []
    angle_carry = 0.0
    eps = 1e-6
    turn_threshold_rad = math.radians(float(turn_threshold_deg))

    while len(prev_actions) < window and curr > 0:
        block_end = curr
        block_start = max(0, block_end - window)
        angle_accum = angle_carry

        for i in range(block_end, block_start, -1):
            if i <= 0:
                continue
            d0 = dirs[i - 1]
            d1 = dirs[i]
            if d0 is None or d1 is None:
                continue
            yaw0 = math.atan2(d0[1], d0[0])
            yaw1 = math.atan2(d1[1], d1[0])
            angle_accum += signed_angle_delta(yaw0, yaw1)

            frame_id = int(frames[i].get("frame", i))
            action = forward_action
            if abs(angle_accum) + eps >= turn_threshold_rad:
                action = left_action if angle_accum > 0 else right_action
                angle_accum = 0.0

            if (
                prev_actions
                and action in (left_action, right_action)
                and prev_actions[-1] in (left_action, right_action)
                and prev_actions[-1] != action
                and len(prev_actions) < window
            ):
                prev_actions.append(int(forward_action))
                prev_frames.append(int(frame_id))
                if len(prev_actions) >= window:
                    break

            prev_actions.append(int(action))
            prev_frames.append(int(frame_id))
            if len(prev_actions) >= window:
                break

        angle_carry = angle_accum
        curr = block_start

    while len(prev_actions) < window:
        prev_actions.append(int(padding_action))
        prev_frames.append(int(padding_frame))

    return prev_actions, prev_frames


def reverse_action_payload_from_actions(
    *,
    action_payload: dict[str, Any],
    window: int,
    padding_action: int,
    padding_frame: int,
    turn_threshold_deg: float,
) -> dict[str, Any]:
    frames = list(action_payload.get("frames") or [])
    frames.sort(key=lambda f: int(f.get("frame", 0)))
    dirs = _compute_dirs(frames)
    dirs = _smooth_dirs(dirs, window=5)

    reverse_frames: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        frame_id = int(frame.get("frame", idx))
        prev_actions, prev_frames = _compute_prev_actions(
            frames=frames,
            dirs=dirs,
            index=idx,
            window=int(window),
            padding_action=int(padding_action),
            padding_frame=int(padding_frame),
            turn_threshold_deg=float(turn_threshold_deg),
            forward_action=1,
            left_action=2,
            right_action=3,
        )
        reverse_frames.append(
            {
                "frame": frame_id,
                "prev_actions": prev_actions,
                "prev_frames": prev_frames,
            }
        )

    return {
        "dataset_root": action_payload.get("dataset_root"),
        "scene": action_payload.get("scene"),
        "label": action_payload.get("label"),
        "generated_at": utc_now_iso(),
        "source_actions": None,
        "window": int(window),
        "padding_action": int(padding_action),
        "padding_frame": int(padding_frame),
        "step_distance": None,
        "turn_threshold_deg": float(turn_threshold_deg),
        "frames": reverse_frames,
    }


def write_tarball(tar_path: Path, roots: Sequence[Path]) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            tf.add(str(root), arcname=root.name, recursive=True)


def write_camera_metadata_frame(
    *,
    frames_dir: Path,
    frame_prefix: str,
    frame_idx: int,
    payload: dict[str, Any],
    overwrite: bool = True,
) -> Path:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cam_json_path = frames_dir / f"{frame_prefix}_{int(frame_idx):04d}_camera.json"
    if cam_json_path.exists() and not overwrite:
        return cam_json_path
    cam_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cam_json_path


def camera_metadata_path_for_label(*, scene_dir: Path, label_id: str) -> Path:
    return scene_dir / f"{str(label_id)}_camera.json"


def write_camera_metadata_for_path(
    *,
    scene_dir: Path,
    scene_id: str | None = None,
    label_id: str,
    camera_payloads: Sequence[dict[str, Any]],
    dataset_root: Path | None = None,
    overwrite: bool = True,
) -> Path:
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_path = camera_metadata_path_for_label(scene_dir=scene_dir, label_id=label_id)
    if out_path.exists() and not overwrite:
        return out_path
    frames: list[dict[str, Any]] = []
    for idx, payload in enumerate(camera_payloads):
        row = {"frame": int(idx)}
        row.update(payload)
        frames.append(row)
    out_payload: dict[str, Any] = {
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "scene": str(scene_id) if scene_id is not None else None,
        "label": str(label_id),
        "generated_at": utc_now_iso(),
        "frames": frames,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    return out_path


def write_tar_zst(
    *,
    out_path: Path,
    root_dir: Path,
    zstd_level: int = 3,
) -> None:
    """
    Create a <root_dir.name>.tar.zst archive at out_path.

    Uses system tar + zstd via a pipe for speed:
      tar -cf - <root.name> | zstd -T0 -<level> -o out.tar.zst
    """
    import shutil
    import subprocess

    root_dir = Path(root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    zstd = shutil.which("zstd")
    tar = shutil.which("tar")
    if not zstd:
        raise FileNotFoundError("zstd not found in PATH (required for .tar.zst).")
    if not tar:
        raise FileNotFoundError("tar not found in PATH (required for .tar.zst).")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parent = root_dir.parent
    name = root_dir.name

    tar_proc = subprocess.Popen(
        [tar, "-cf", "-", name],
        cwd=str(parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        zstd_proc = subprocess.Popen(
            [zstd, "-T0", f"-{int(zstd_level)}", "-o", str(out_path)],
            stdin=tar_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert tar_proc.stdout is not None
        tar_proc.stdout.close()
        z_out, z_err = zstd_proc.communicate()
        tar_out, tar_err = tar_proc.communicate()
    finally:
        if tar_proc.poll() is None:
            tar_proc.kill()
        if "zstd_proc" in locals() and zstd_proc.poll() is None:
            zstd_proc.kill()

    if tar_proc.returncode != 0:
        raise RuntimeError(f"tar failed (rc={tar_proc.returncode}): {tar_err.decode('utf-8', 'ignore')}")
    if zstd_proc.returncode != 0:
        raise RuntimeError(f"zstd failed (rc={zstd_proc.returncode}): {z_err.decode('utf-8', 'ignore')}")
