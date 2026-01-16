"""PLY transform helpers shared across rendering paths."""

from __future__ import annotations

import math
import os
import sys
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from utils import gaussian_ply_utils as ply_utils

if TYPE_CHECKING:
    from render_label_paths import ActorSequence, ActorSequenceFrame


class PlyTransformBackend(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


_GPU_BACKEND_FAILED = False
_GPU_FALLBACK_REPORTED = False
_STRICT_GPU_BACKENDS = os.getenv("STRICT_GPU_BACKENDS", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def rotation_matrix_z_np(theta: float) -> np.ndarray:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    if translation.shape != (3,):
        raise ValueError("translation must be length-3 vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _apply_transform_to_frame_cpu(
    base_frame: "ActorSequenceFrame",
    sequence: "ActorSequence",
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transform to a stored actor frame and return the mutated vertex array."""

    data = np.array(base_frame.base_data, copy=True)
    ply = ply_utils.GaussianPly(
        ply=None,
        vertex=None,
        data=data,
        columns=sequence.columns,
    )
    ply_utils.apply_transform_inplace(
        ply,
        transform,
        rotate_normals=True,
        rotate_sh=True,
    )
    return ply.data


def _apply_transform_to_frame_gpu(
    base_frame: "ActorSequenceFrame",
    sequence: "ActorSequence",
    transform: np.ndarray,
) -> np.ndarray:
    try:
        import torch
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("GPU PLY transform backend requires torch.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("GPU PLY transform backend requires a CUDA device.")
    if transform.shape != (4, 4):
        raise ValueError("transform must be a 4x4 matrix")

    data = np.array(base_frame.base_data, copy=True)
    if data.size == 0:
        return data

    a = transform[:3, :3].astype(np.float64)
    t = transform[:3, 3].astype(np.float64)
    rotation = a.T.copy()
    scale = float(np.sqrt((rotation @ rotation.T)[0, 0]))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid scale derived from transform: {scale}")
    rotation /= scale

    col = sequence.columns
    if not all(axis in col for axis in ("x", "y", "z")):
        raise KeyError("PLY is missing x/y/z columns")

    device = torch.device("cuda")
    rot_t = torch.tensor(rotation, device=device, dtype=torch.float32)
    t_t = torch.tensor(t, device=device, dtype=torch.float32)

    def _normalize_torch(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        return vec / norm

    def _quat_wxyz_to_matrix_torch(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return torch.stack(
            [
                1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
                2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
                2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
            ],
            dim=-1,
        ).reshape(quat.shape[:-1] + (3, 3))

    def _matrix_to_quat_wxyz_torch(matrix: torch.Tensor) -> torch.Tensor:
        tr = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
        quat = torch.empty(matrix.shape[:-2] + (4,), device=matrix.device, dtype=matrix.dtype)
        cond = tr > 0.0

        if torch.any(cond):
            m = matrix[cond]
            root = torch.sqrt(tr[cond] + 1.0)
            quat_cond = torch.zeros((m.shape[0], 4), device=matrix.device, dtype=matrix.dtype)
            quat_cond[:, 0] = 0.5 * root
            root = 0.5 / root
            quat_cond[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) * root
            quat_cond[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) * root
            quat_cond[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) * root
            quat[cond] = quat_cond

        if torch.any(~cond):
            m = matrix[~cond]
            diag = torch.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], dim=1)
            idx = torch.argmax(diag, dim=1)
            quat_cond = torch.zeros((m.shape[0], 4), device=matrix.device, dtype=matrix.dtype)

            mask0 = idx == 0
            if torch.any(mask0):
                r = m[mask0]
                root = torch.sqrt(
                    torch.clamp(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], min=0.0)
                ) * 2.0
                quat_cond[mask0, 0] = (r[:, 2, 1] - r[:, 1, 2]) / root
                quat_cond[mask0, 1] = 0.25 * root
                quat_cond[mask0, 2] = (r[:, 0, 1] + r[:, 1, 0]) / root
                quat_cond[mask0, 3] = (r[:, 0, 2] + r[:, 2, 0]) / root

            mask1 = idx == 1
            if torch.any(mask1):
                r = m[mask1]
                root = torch.sqrt(
                    torch.clamp(1.0 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2], min=0.0)
                ) * 2.0
                quat_cond[mask1, 0] = (r[:, 0, 2] - r[:, 2, 0]) / root
                quat_cond[mask1, 1] = (r[:, 0, 1] + r[:, 1, 0]) / root
                quat_cond[mask1, 2] = 0.25 * root
                quat_cond[mask1, 3] = (r[:, 1, 2] + r[:, 2, 1]) / root

            mask2 = idx == 2
            if torch.any(mask2):
                r = m[mask2]
                root = torch.sqrt(
                    torch.clamp(1.0 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2], min=0.0)
                ) * 2.0
                quat_cond[mask2, 0] = (r[:, 1, 0] - r[:, 0, 1]) / root
                quat_cond[mask2, 1] = (r[:, 0, 2] + r[:, 2, 0]) / root
                quat_cond[mask2, 2] = (r[:, 1, 2] + r[:, 2, 1]) / root
                quat_cond[mask2, 3] = 0.25 * root

            quat[~cond] = quat_cond

        return _normalize_torch(quat)

    xyz = torch.stack(
        [
            torch.from_numpy(data["x"]),
            torch.from_numpy(data["y"]),
            torch.from_numpy(data["z"]),
        ],
        dim=1,
    ).to(device=device, dtype=torch.float32)
    xyz = xyz @ rot_t.T
    xyz = xyz + t_t
    data["x"] = xyz[:, 0].cpu().numpy().astype(data.dtype["x"])
    data["y"] = xyz[:, 1].cpu().numpy().astype(data.dtype["y"])
    data["z"] = xyz[:, 2].cpu().numpy().astype(data.dtype["z"])

    scale_cols = ply_utils._find_scale_columns(col)
    if scale_cols:
        log_scale = math.log(scale)
        if len(scale_cols) == 3:
            for key in scale_cols:
                values = torch.from_numpy(data[key]).to(device=device, dtype=torch.float32)
                values = values + log_scale
                data[key] = values.cpu().numpy().astype(data.dtype[key])
        else:
            key = scale_cols[0]
            values = torch.from_numpy(data[key]).to(device=device, dtype=torch.float32)
            values = values + log_scale
            data[key] = values.cpu().numpy().astype(data.dtype[key])

    rot_keys = [f"rot_{i}" for i in range(4)]
    if all(key in col for key in rot_keys):
        quat = torch.stack(
            [torch.from_numpy(data[key]) for key in rot_keys],
            dim=1,
        ).to(device=device, dtype=torch.float32)
        quat = _normalize_torch(quat)
        local_rotation = _quat_wxyz_to_matrix_torch(quat)
        composed = torch.einsum("ij,njk->nik", rot_t, local_rotation)
        quat_world = _matrix_to_quat_wxyz_torch(composed)
        for idx, key in enumerate(rot_keys):
            data[key] = quat_world[:, idx].cpu().numpy().astype(data.dtype[key])

    if all(key in col for key in ("nx", "ny", "nz")):
        normals = torch.stack(
            [
                torch.from_numpy(data["nx"]),
                torch.from_numpy(data["ny"]),
                torch.from_numpy(data["nz"]),
            ],
            dim=1,
        ).to(device=device, dtype=torch.float32)
        normals = _normalize_torch(normals @ rot_t.T)
        data["nx"] = normals[:, 0].cpu().numpy().astype(data.dtype["nx"])
        data["ny"] = normals[:, 1].cpu().numpy().astype(data.dtype["ny"])
        data["nz"] = normals[:, 2].cpu().numpy().astype(data.dtype["nz"])

    field_names = [f"f_rest_{i}" for i in range(45)]
    if all(name in col for name in field_names):
        coeffs = np.stack([data[name] for name in field_names], axis=1).astype(np.float64)
        rotated = ply_utils._rotate_sh_coeffs(coeffs, rotation)
        for idx, name in enumerate(field_names):
            data[name] = rotated[:, idx].astype(data.dtype[name])

    return data


def apply_transform_to_frame(
    base_frame: "ActorSequenceFrame",
    sequence: "ActorSequence",
    transform: np.ndarray,
    *,
    backend: PlyTransformBackend | str = PlyTransformBackend.GPU,
) -> np.ndarray:
    global _GPU_BACKEND_FAILED, _GPU_FALLBACK_REPORTED
    backend_value = (
        backend.value if isinstance(backend, PlyTransformBackend) else str(backend).lower()
    )
    if backend_value == PlyTransformBackend.CPU.value:
        return _apply_transform_to_frame_cpu(base_frame, sequence, transform)
    if backend_value == PlyTransformBackend.GPU.value:
        if _GPU_BACKEND_FAILED:
            if _STRICT_GPU_BACKENDS:
                raise RuntimeError("GPU PLY transform backend unavailable in strict mode.")
            return _apply_transform_to_frame_cpu(base_frame, sequence, transform)
        try:
            return _apply_transform_to_frame_gpu(base_frame, sequence, transform)
        except Exception as exc:  # pylint: disable=broad-except
            _GPU_BACKEND_FAILED = True
            if _STRICT_GPU_BACKENDS:
                raise RuntimeError(
                    f"GPU PLY transform failed in strict mode: {exc}"
                ) from exc
            if not _GPU_FALLBACK_REPORTED:
                print(
                    f"[WARN] GPU PLY transform failed ({exc}); falling back to CPU.",
                    file=sys.stderr,
                    flush=True,
                )
                _GPU_FALLBACK_REPORTED = True
            return _apply_transform_to_frame_cpu(base_frame, sequence, transform)
    raise ValueError(f"Unknown PLY transform backend: {backend}")
