# Utilities for transforming and composing Gaussian splatting PLY files.

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return vec / norm


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    tr = np.trace(matrix, axis1=-2, axis2=-1)
    quat = np.empty(matrix.shape[:-2] + (4,), dtype=matrix.dtype)
    cond = tr > 0.0

    if np.any(cond):
        m = matrix[cond]
        root = np.sqrt(tr[cond] + 1.0)
        quat_cond = np.zeros((m.shape[0], 4), dtype=matrix.dtype)
        quat_cond[:, 0] = 0.5 * root
        root = 0.5 / root
        quat_cond[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) * root
        quat_cond[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) * root
        quat_cond[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) * root
        quat[cond] = quat_cond

    if np.any(~cond):
        m = matrix[~cond]
        i = np.argmax(
            np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1), axis=1
        )
        quat_cond = np.zeros((m.shape[0], 4), dtype=matrix.dtype)
        for idx in range(m.shape[0]):
            r = m[idx]
            if i[idx] == 0:
                root = np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2
                quat_cond[idx, 0] = (r[2, 1] - r[1, 2]) / root
                quat_cond[idx, 1] = 0.25 * root
                quat_cond[idx, 2] = (r[0, 1] + r[1, 0]) / root
                quat_cond[idx, 3] = (r[0, 2] + r[2, 0]) / root
            elif i[idx] == 1:
                root = np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) * 2
                quat_cond[idx, 0] = (r[0, 2] - r[2, 0]) / root
                quat_cond[idx, 1] = (r[0, 1] + r[1, 0]) / root
                quat_cond[idx, 2] = 0.25 * root
                quat_cond[idx, 3] = (r[1, 2] + r[2, 1]) / root
            else:
                root = np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) * 2
                quat_cond[idx, 0] = (r[1, 0] - r[0, 1]) / root
                quat_cond[idx, 1] = (r[0, 2] + r[2, 0]) / root
                quat_cond[idx, 2] = (r[1, 2] + r[2, 1]) / root
                quat_cond[idx, 3] = 0.25 * root
        quat[~cond] = quat_cond

    quat = _normalize(quat)
    return quat


C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def _sample_spherical_harmonics(directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = directions[:3, 0], directions[:3, 1], directions[:3, 2]
    s1 = np.zeros((3, 3), np.float64)
    s1[:, 0] = -C1 * y
    s1[:, 1] = C1 * z
    s1[:, 2] = -C1 * x

    x, y, z = directions[3:8, 0], directions[3:8, 1], directions[3:8, 2]
    xy, yz, xz = x * y, y * z, x * z
    xx, yy, zz = x * x, y * y, z * z
    s2 = np.zeros((5, 5), np.float64)
    s2[:, 0] = C2[0] * xy
    s2[:, 1] = C2[1] * yz
    s2[:, 2] = C2[2] * (2.0 * zz - xx - yy)
    s2[:, 3] = C2[3] * xz
    s2[:, 4] = C2[4] * (xx - yy)

    x, y, z = directions[8:15, 0], directions[8:15, 1], directions[8:15, 2]
    xx, yy, zz = x * x, y * y, z * z
    s3 = np.zeros((7, 7), np.float64)
    s3[:, 0] = C3[0] * y * (3 * xx - yy)
    s3[:, 1] = C3[1] * x * y * z
    s3[:, 2] = C3[2] * y * (4 * zz - xx - yy)
    s3[:, 3] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    s3[:, 4] = C3[4] * x * (4 * zz - xx - yy)
    s3[:, 5] = C3[5] * z * (xx - yy)
    s3[:, 6] = C3[6] * x * (xx - 3 * yy)
    return s1, s2, s3


def _rotate_sh_coeffs(sh_coeffs: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if sh_coeffs.size == 0:
        return sh_coeffs
    rng = np.random.default_rng(1234)
    dirs = rng.standard_normal((15, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-12, None)
    s1, s2, s3 = _sample_spherical_harmonics(dirs)
    dirs_rot = dirs @ rotation.T
    s1r, s2r, s3r = _sample_spherical_harmonics(dirs_rot)

    T1 = np.linalg.pinv(s1) @ s1r
    T2 = np.linalg.pinv(s2) @ s2r
    T3 = np.linalg.pinv(s3) @ s3r

    out = np.empty_like(sh_coeffs)
    out[:, 0:3] = sh_coeffs[:, 0:3] @ T1
    out[:, 3:8] = sh_coeffs[:, 3:8] @ T2
    out[:, 8:15] = sh_coeffs[:, 8:15] @ T3
    return out


def _find_scale_columns(columns: Mapping[str, int]) -> Tuple[str, ...]:
    candidates = [
        ("scale_0", "scale_1", "scale_2"),
        ("scales_0", "scales_1", "scales_2"),
    ]
    for triplet in candidates:
        if all(name in columns for name in triplet):
            return triplet
    if "scale" in columns:
        return ("scale",)
    return ()


@dataclass(frozen=True)
class GaussianPly:
    ply: PlyData | None
    vertex: PlyElement | None
    data: np.ndarray
    columns: Dict[str, int]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @classmethod
    def read(cls, path: Path) -> "GaussianPly":
        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        data = vertex.data.copy()
        columns = {name: idx for idx, name in enumerate(data.dtype.names or [])}
        return cls(ply=ply, vertex=vertex, data=data, columns=columns)

    def clone_empty_like(self, size: int) -> np.ndarray:
        return np.empty(size, dtype=self.dtype)

    def write(self, data: np.ndarray, path: Path) -> None:
        new_element = PlyElement.describe(data, "vertex")
        ply = PlyData([new_element], text=self.ply.text if self.ply is not None else False)
        if self.ply is not None:
            ply.comments = list(getattr(self.ply, "comments", []))
        ply.write(str(path))


def apply_transform_inplace(
    ply: GaussianPly,
    transform: np.ndarray,
    *,
    rotate_normals: bool = True,
    rotate_sh: bool = True,
    quat_order: str = "wxyz",
    translate: bool = True,
) -> None:
    if transform.shape != (4, 4):
        raise ValueError("transform must be a 4x4 matrix")

    a = transform[:3, :3].astype(np.float64)
    t = transform[:3, 3].astype(np.float64)

    rotation = a.T.copy()
    scale = float(np.sqrt((rotation @ rotation.T)[0, 0]))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid scale derived from transform: {scale}")
    rotation /= scale

    col = ply.columns
    if not all(axis in col for axis in ("x", "y", "z")):
        raise KeyError("PLY is missing x/y/z columns")

    xyz = np.stack(
        [
            ply.data["x"].astype(np.float64),
            ply.data["y"].astype(np.float64),
            ply.data["z"].astype(np.float64),
        ],
        axis=1,
    )

    xyz = xyz @ rotation.T
    if translate:
        xyz = xyz + t[None, :]

    ply.data["x"] = xyz[:, 0].astype(ply.dtype["x"])
    ply.data["y"] = xyz[:, 1].astype(ply.dtype["y"])
    ply.data["z"] = xyz[:, 2].astype(ply.dtype["z"])

    scale_cols = _find_scale_columns(col)
    if scale_cols:
        log_scale = math.log(scale)
        if len(scale_cols) == 3:
            for key in scale_cols:
                ply.data[key] = (
                    ply.data[key].astype(np.float64) + log_scale
                ).astype(ply.dtype[key])
        else:
            key = scale_cols[0]
            ply.data[key] = (
                ply.data[key].astype(np.float64) + log_scale
            ).astype(ply.dtype[key])

    rot_keys = [f"rot_{i}" for i in range(4)]
    if all(key in col for key in rot_keys):
        quat = np.stack([ply.data[key] for key in rot_keys], axis=1).astype(np.float64)
        quat = _normalize(quat)
        local_rotation = (
            _quat_wxyz_to_matrix(quat)
            if quat_order == "wxyz"
            else _quat_xyzw_to_matrix(quat)
        )
        composed = np.einsum("ij,njk->nik", rotation, local_rotation)
        quat_world = _matrix_to_quat_wxyz(composed)
        if quat_order == "xyzw":
            quat_world = np.stack(
                [
                    quat_world[:, 1],
                    quat_world[:, 2],
                    quat_world[:, 3],
                    quat_world[:, 0],
                ],
                axis=1,
            )
        for idx, key in enumerate(rot_keys):
            ply.data[key] = quat_world[:, idx].astype(ply.dtype[key])

    if rotate_normals and all(key in col for key in ("nx", "ny", "nz")):
        normals = np.stack(
            [
                ply.data["nx"].astype(np.float64),
                ply.data["ny"].astype(np.float64),
                ply.data["nz"].astype(np.float64),
            ],
            axis=1,
        )
        normals = _normalize(normals @ rotation.T)
        ply.data["nx"] = normals[:, 0].astype(ply.dtype["nx"])
        ply.data["ny"] = normals[:, 1].astype(ply.dtype["ny"])
        ply.data["nz"] = normals[:, 2].astype(ply.dtype["nz"])

    if rotate_sh:
        field_names = [f"f_rest_{i}" for i in range(45)]
        if all(name in col for name in field_names):
            coeffs = np.stack([ply.data[name] for name in field_names], axis=1).astype(np.float64)
            rotated = _rotate_sh_coeffs(coeffs, rotation)
            for idx, name in enumerate(field_names):
                ply.data[name] = rotated[:, idx].astype(ply.dtype[name])


def align_dtype(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    result = np.empty(data.shape[0], dtype=target_dtype)
    for name in target_dtype.names or []:
        if name in data.dtype.names:
            result[name] = data[name].astype(target_dtype[name])
        else:
            result[name] = 0
    return result


def concat_vertices(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("No arrays provided for concatenation")
    dtype = arrays[0].dtype
    return np.concatenate(arrays, axis=0).astype(dtype, copy=False)
