from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from plyfile import PlyElementParseError

from scene import GaussianModel
from utils import gaussian_ply_utils as ply_utils
from utils.general_utils import inverse_sigmoid
from utils.render_utils import world_to_pixel

EPS = 1e-6
DEFAULT_VIDEO_FPS = 10
DEFAULT_ACTOR_PATTERN = "*.ply"
DEFAULT_ACTOR_SPEED = 1.3

_DIGIT_PATTERN = re.compile(r"(\d+)")
_BROAD_ACTOR_PATTERNS = {"", "*", "*.ply"}
_ANIMATION_FRAME_PATTERNS = ("frame_*.ply", "[0-9][0-9][0-9][0-9][0-9][0-9].ply")


class PathSampler:
    """Lightweight sampler to query positions/tangents along a 2D polyline."""

    def __init__(self, points: Sequence[np.ndarray]):
        if len(points) < 2:
            raise ValueError("PathSampler requires at least two points.")
        raw = np.asarray(points, dtype=np.float32)
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
        self.cumulative = np.concatenate(
            [np.array([0.0], dtype=np.float32), np.cumsum(self.segment_lengths)]
        )

    @property
    def total_length(self) -> float:
        return float(self.cumulative[-1])

    def position_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            direction = self.segment_vectors[0] / self.segment_lengths[0]
            return self.points[0] + direction * distance
        total = self.total_length
        if distance >= total:
            direction = self.segment_vectors[-1] / self.segment_lengths[-1]
            return self.points[-1] + direction * (distance - total)
        seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
        seg_offset = distance - self.cumulative[seg_idx]
        ratio = seg_offset / self.segment_lengths[seg_idx]
        return self.points[seg_idx] + self.segment_vectors[seg_idx] * ratio


@dataclass(frozen=True)
class ActorOptions:
    sequence_dir: Path
    pattern: str
    height: float
    follow_distance: float
    buffer_distance: float
    speed: float
    fps: float
    loop: bool
    foot_offset: float
    animation_cycle_mod: int


@dataclass
class ActorSequenceFrame:
    base_data: np.ndarray


@dataclass
class ActorSequence:
    frames: list[ActorSequenceFrame]
    height: float
    hip_height: float
    radius_xy: float
    columns: dict[str, int]
    dtype: np.dtype
    feature_rest_names: list[str]
    scale_names: list[str]
    rot_names: list[str]
    rest_dim: int
    max_sh_degree: int
    uniform_scale: bool
    max_points: int


@dataclass(frozen=True)
class ActorRuntime:
    options: ActorOptions
    sequence: ActorSequence


@dataclass
class ActorRenderFrame:
    xyz: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor


def natural_sort_key(path: Path) -> list[object]:
    parts = _DIGIT_PATTERN.split(path.stem)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    key.append(path.suffix.lower())
    return key


def list_actor_frame_paths_in_dir(sequence_dir: Path, *, pattern: str = "*.ply") -> list[Path]:
    if not sequence_dir.is_dir():
        return []
    pattern = pattern or "*.ply"

    if pattern in _BROAD_ACTOR_PATTERNS:
        for animation_pattern in _ANIMATION_FRAME_PATTERNS:
            frames = [path for path in sequence_dir.glob(animation_pattern) if path.is_file()]
            frames = [path for path in frames if path.suffix.lower() == ".ply"]
            if frames:
                return sorted(frames, key=natural_sort_key)

    initial = [path for path in sequence_dir.glob(pattern) if path.is_file()]
    initial = [path for path in initial if path.suffix.lower() == ".ply"]
    if not initial:
        initial = [path for path in sequence_dir.glob("*.ply") if path.is_file()]
    return sorted(initial, key=natural_sort_key)


def list_actor_frame_paths(options: ActorOptions) -> list[Path]:
    return list_actor_frame_paths_in_dir(options.sequence_dir, pattern=options.pattern)


def list_actor_frame_paths_recursive(root: Path, *, pattern: str = "*.ply") -> list[Path]:
    if root is None or not root.is_dir():
        return []
    pattern = pattern or "*.ply"

    if pattern in _BROAD_ACTOR_PATTERNS:
        for animation_pattern in _ANIMATION_FRAME_PATTERNS:
            frames = [path for path in root.rglob(animation_pattern) if path.is_file()]
            frames = [path for path in frames if path.suffix.lower() == ".ply"]
            if frames:
                return sorted(frames, key=lambda p: p.as_posix())

    initial = [path for path in root.rglob(pattern) if path.is_file()]
    initial = [path for path in initial if path.suffix.lower() == ".ply"]
    if not initial:
        initial = [path for path in root.rglob("*.ply") if path.is_file()]
    return sorted(initial, key=lambda p: p.as_posix())


def _dir_has_ply(path: Path, *, pattern: str = "*.ply") -> bool:
    if not path.is_dir():
        return False
    return any(path.glob(pattern)) or any(path.glob("*.ply"))


def _list_actor_subdirs(root: Path, *, pattern: str = "*.ply") -> list[Path]:
    if root is None or not root.is_dir():
        return []
    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        if _dir_has_ply(current, pattern=pattern):
            found.append(current)
            dirnames[:] = []
    return sorted(found, key=lambda p: p.as_posix())


def _first_actor_subdir(root: Path) -> Path | None:
    candidates = _list_actor_subdirs(root, pattern="*.ply")
    return candidates[0] if candidates else None


def load_gaussian_ply(path: Path) -> ply_utils.GaussianPly:
    try:
        return ply_utils.GaussianPly.read(path)
    except PlyElementParseError as exc:
        raise ValueError(f"Unable to parse actor PLY: {path}") from exc


ACTOR_AXIS_ALIGNMENT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
HIP_HEIGHT_RATIO = 0.6


def _compute_frame_radius_xy(data: np.ndarray) -> float:
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return 0.0
    xy = np.stack((x, y), axis=1)
    center = np.median(xy, axis=0)
    dists = np.linalg.norm(xy - center[None, :], axis=1)
    return float(np.percentile(dists, 95))


def load_actor_sequence(
    options: ActorOptions,
    *,
    debug: bool = False,
) -> ActorSequence:
    if not options.sequence_dir.is_dir():
        raise FileNotFoundError(f"Actor sequence directory not found: {options.sequence_dir}")

    ply_files = list_actor_frame_paths(options)
    if not ply_files:
        raise FileNotFoundError(
            f"No actor frame PLY files found in {options.sequence_dir}"
        )

    alignment_transform = np.eye(4, dtype=np.float64)
    alignment_transform[:3, :3] = ACTOR_AXIS_ALIGNMENT_MATRIX

    actor_plys: list[ply_utils.GaussianPly] = []
    z_values: list[np.ndarray] = []

    for ply_path in ply_files:
        ply = load_gaussian_ply(Path(ply_path))
        ply_utils.apply_transform_inplace(
            ply,
            alignment_transform,
            rotate_normals=True,
            rotate_sh=True,
        )
        actor_plys.append(ply)
        z_values.append(ply.data["z"].astype(np.float64))

    combined_z = np.concatenate(z_values)
    raw_min_z = float(np.min(combined_z))
    raw_max_z = float(np.max(combined_z))
    measured_height = max(raw_max_z - raw_min_z, EPS)
    target_height = options.height if options.height > 0.0 else measured_height
    scale_factor = target_height / measured_height

    if debug:
        print(
            f"[DEBUG] Actor sequence: {len(actor_plys)} frames, "
            f"raw height {measured_height:.3f} m, applying scale {scale_factor:.3f}",
            flush=True,
        )

    if not math.isclose(scale_factor, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        scale_transform = np.eye(4, dtype=np.float64)
        scale_transform[:3, :3] *= scale_factor
        for ply in actor_plys:
            ply_utils.apply_transform_inplace(
                ply,
                scale_transform,
                rotate_normals=True,
                rotate_sh=True,
            )

    global_min_z = min(float(np.min(ply.data["z"])) for ply in actor_plys)
    translate_transform = np.eye(4, dtype=np.float64)
    translate_transform[2, 3] = -global_min_z
    for ply in actor_plys:
        ply_utils.apply_transform_inplace(
            ply,
            translate_transform,
            rotate_normals=False,
            rotate_sh=False,
        )

    global_max_z = max(float(np.max(ply.data["z"])) for ply in actor_plys)
    adjusted_height = max(global_max_z, EPS)
    hip_height = adjusted_height * HIP_HEIGHT_RATIO
    radius_xy = max(_compute_frame_radius_xy(ply.data) for ply in actor_plys)

    first_ply = actor_plys[0]
    dtype_names = list(first_ply.data.dtype.names or ())
    feature_rest_names = sorted(
        [name for name in dtype_names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rest_dim = len(feature_rest_names) // 3 if feature_rest_names else 0
    scale_names = sorted(
        [name for name in dtype_names if name.startswith("scale")],
        key=lambda name: int(name.split("_")[-1]) if "_" in name else 0,
    )
    rot_names = sorted(
        [name for name in dtype_names if name.startswith("rot_")],
        key=lambda name: int(name.split("_")[-1]),
    )

    if rest_dim * 3 != len(feature_rest_names):
        raise ValueError("Unexpected spherical harmonic coefficient layout in actor PLY.")
    if rot_names and len(rot_names) != 4:
        raise ValueError("Actor PLY must provide quaternion components rot_0..rot_3.")
    if scale_names and len(scale_names) not in (1, 3):
        raise ValueError("Actor PLY scales must appear as scale_0/1/2 or a single scale column.")

    max_sh_degree = int(round(math.sqrt(rest_dim + 1) - 1)) if rest_dim > 0 else 0
    frames = [ActorSequenceFrame(base_data=np.array(ply.data, copy=True)) for ply in actor_plys]

    return ActorSequence(
        frames=frames,
        height=float(adjusted_height),
        hip_height=float(hip_height),
        radius_xy=float(radius_xy),
        columns=dict(first_ply.columns),
        dtype=first_ply.data.dtype,
        feature_rest_names=feature_rest_names,
        scale_names=scale_names,
        rot_names=rot_names,
        rest_dim=rest_dim,
        max_sh_degree=max_sh_degree,
        uniform_scale=len(scale_names) == 1,
        max_points=max(frame.base_data.shape[0] for frame in frames),
    )


def actor_data_to_tensors(
    data: np.ndarray,
    sequence: ActorSequence,
    device: torch.device,
    *,
    target_rest_dim: int | None = None,
) -> ActorRenderFrame:
    xyz_np = np.stack((data["x"], data["y"], data["z"]), axis=1).astype(np.float32)
    xyz = torch.from_numpy(xyz_np).to(device)

    dc_names = [f"f_dc_0", f"f_dc_1", f"f_dc_2"]
    if not all(name in data.dtype.names for name in dc_names):
        missing = [name for name in dc_names if name not in data.dtype.names]
        raise KeyError(f"Actor PLY missing DC SH coefficients: {missing}")
    features_dc_np = np.stack([data[name] for name in dc_names], axis=1).astype(np.float32)
    features_dc = torch.from_numpy(features_dc_np[:, :, None]).to(device).transpose(1, 2).contiguous()

    source_rest_dim = sequence.rest_dim if sequence.rest_dim > 0 else 0
    expected_rest_dim = int(target_rest_dim) if target_rest_dim is not None else source_rest_dim
    if expected_rest_dim < 0:
        raise ValueError("target_rest_dim must be non-negative")

    if source_rest_dim > 0:
        rest_np = np.stack(
            [data[name] for name in sequence.feature_rest_names],
            axis=1,
        ).astype(np.float32)
        rest_np = rest_np.reshape(data.shape[0], 3, sequence.rest_dim)
        features_rest_src = torch.from_numpy(rest_np.transpose(0, 2, 1)).to(device)
    else:
        features_rest_src = torch.zeros(
            (data.shape[0], 0, 3),
            dtype=torch.float32,
            device=device,
        )

    if expected_rest_dim == features_rest_src.shape[1]:
        features_rest = features_rest_src
    else:
        features_rest = torch.zeros(
            (data.shape[0], expected_rest_dim, 3),
            dtype=torch.float32,
            device=device,
        )
        copy_dim = min(features_rest_src.shape[1], expected_rest_dim)
        if copy_dim > 0:
            features_rest[:, :copy_dim] = features_rest_src[:, :copy_dim]

    opacity_np = np.asarray(data["opacity"], dtype=np.float32).reshape(-1, 1)
    opacity = torch.from_numpy(opacity_np).to(device)

    if sequence.scale_names:
        if sequence.uniform_scale:
            scale_values = np.asarray(data[sequence.scale_names[0]], dtype=np.float32).reshape(-1, 1)
            scales_np = np.repeat(scale_values, 3, axis=1)
        else:
            scales_np = np.stack(
                [data[name] for name in sequence.scale_names],
                axis=1,
            ).astype(np.float32)
    else:
        scales_np = np.zeros((data.shape[0], 0), dtype=np.float32)
    scaling = torch.from_numpy(scales_np).to(device)

    rotation_np = np.stack(
        [data[name] for name in sequence.rot_names],
        axis=1,
    ).astype(np.float32)
    rotation = torch.from_numpy(rotation_np).to(device)

    return ActorRenderFrame(
        xyz=xyz.contiguous(),
        features_dc=features_dc.contiguous(),
        features_rest=features_rest.contiguous(),
        opacity=opacity.contiguous(),
        scaling=scaling.contiguous(),
        rotation=rotation.contiguous(),
    )


class CombinedGaussianModel:
    def __init__(self, base: GaussianModel, actor_frame: ActorRenderFrame):
        device = base.get_xyz.device
        dtype = base.get_xyz.dtype
        self.base_size = base.get_xyz.shape[0]
        actor_size = actor_frame.xyz.shape[0]

        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree = base.max_sh_degree

        self._xyz = torch.empty((self.base_size + actor_size, 3), device=device, dtype=dtype)
        self._xyz[: self.base_size] = base._xyz.detach()
        self._xyz[self.base_size :] = actor_frame.xyz

        dc_base = base._features_dc.detach()
        self._features_dc = torch.empty(
            (self.base_size + actor_size, dc_base.shape[1], dc_base.shape[2]),
            device=device,
            dtype=dc_base.dtype,
        )
        self._features_dc[: self.base_size] = dc_base
        self._features_dc[self.base_size :] = actor_frame.features_dc

        rest_base = base._features_rest.detach()
        if rest_base.shape[1] > 0:
            self._features_rest = torch.empty(
                (self.base_size + actor_size, rest_base.shape[1], rest_base.shape[2]),
                device=device,
                dtype=rest_base.dtype,
            )
            self._features_rest[: self.base_size] = rest_base
            self._features_rest[self.base_size :] = actor_frame.features_rest
        else:
            self._features_rest = torch.zeros(
                (self.base_size + actor_size, 0, 0),
                device=device,
                dtype=rest_base.dtype,
            )

        opacity_base = base._opacity.detach()
        self._opacity = torch.empty((self.base_size + actor_size, 1), device=device, dtype=opacity_base.dtype)
        self._opacity[: self.base_size] = opacity_base
        self._opacity[self.base_size :] = actor_frame.opacity

        scaling_base = base._scaling.detach()
        self._scaling = torch.empty((self.base_size + actor_size, scaling_base.shape[1]), device=device, dtype=scaling_base.dtype)
        self._scaling[: self.base_size] = scaling_base
        if actor_frame.scaling.shape[1] == 0:
            self._scaling[self.base_size :] = 0.0
        else:
            self._scaling[self.base_size :] = actor_frame.scaling

        rotation_base = base._rotation.detach()
        self._rotation = torch.empty((self.base_size + actor_size, rotation_base.shape[1]), device=device, dtype=rotation_base.dtype)
        self._rotation[: self.base_size] = rotation_base
        self._rotation[self.base_size :] = actor_frame.rotation

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def update_actor(self, actor_frame: ActorRenderFrame) -> None:
        self._xyz[self.base_size :] = actor_frame.xyz
        self._features_dc[self.base_size :] = actor_frame.features_dc
        if self._features_rest.shape[1] > 0:
            self._features_rest[self.base_size :] = actor_frame.features_rest
        self._opacity[self.base_size :] = actor_frame.opacity
        if actor_frame.scaling.shape[1] == 0:
            self._scaling[self.base_size :] = 0.0
        else:
            self._scaling[self.base_size :] = actor_frame.scaling
        self._rotation[self.base_size :] = actor_frame.rotation

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


def build_path_metadata(
    *,
    scene_id: str,
    label_id: str,
    path_xy: Sequence[np.ndarray],
    camera_xy_seq: Sequence[np.ndarray],
    meta: dict,
    follow_distance: float,
    limit_to_follow: bool,
) -> dict:
    follow_distance_m = max(float(follow_distance), 0.0)
    sampler = PathSampler(path_xy)
    cumulative = sampler.cumulative
    total_length = sampler.total_length
    max_camera_distance = max(total_length - follow_distance_m, 0.0)

    frames: list[dict] = []
    distances: list[float] = []

    for dist in cumulative:
        if len(distances) >= len(camera_xy_seq):
            break
        camera_distance = min(dist, max_camera_distance) if limit_to_follow else min(dist, total_length)
        distances.append(camera_distance)
        if limit_to_follow and camera_distance >= max_camera_distance - 1e-6:
            break

    while len(distances) < len(camera_xy_seq):
        distances.append(distances[-1] if distances else 0.0)

    points = sampler.points
    for frame_idx, (camera_xy, cam_dist) in enumerate(zip(camera_xy_seq, distances)):
        person_distance = min(cam_dist + follow_distance_m, total_length)
        person_xy = sampler.position_at(person_distance)

        between_world: list[list[float]] = []
        between_pixel: list[list[int]] = []
        for point_idx, point in enumerate(points):
            dist_val = cumulative[point_idx]
            if cam_dist < dist_val < person_distance - 1e-6:
                bw = [float(point[0]), float(point[1])]
                between_world.append(bw)
                pixel = world_to_pixel(meta, np.array(point[:2], dtype=np.float32))
                between_pixel.append([int(pixel[0]), int(pixel[1])])

        frame_entry = {
            "id": int(frame_idx),
            "camera_world": [float(camera_xy[0]), float(camera_xy[1])],
            "person_world": [float(person_xy[0]), float(person_xy[1])],
            "between_world": between_world,
            "between_pixel": between_pixel,
        }
        frames.append(frame_entry)

    return {
        "scene": scene_id,
        "label": label_id,
        "follow_distance": follow_distance_m,
        "frames": frames,
    }
