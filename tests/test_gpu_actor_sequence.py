from __future__ import annotations

import math

import numpy as np
import torch

from utils.ply_transform_utils import apply_transform_to_frame, build_transform_matrix, rotation_matrix_z_np
from utils.telesim_actor_utils import (
    ActorSequence,
    ActorSequenceFrame,
    actor_data_to_tensors,
    build_gpu_actor_sequence,
    transform_gpu_actor_frame,
)
from utils.sh_utils import RGB2SH


def _actor_sequence() -> ActorSequence:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )
    data = np.zeros(3, dtype=dtype)
    data["x"] = [0.0, 1.0, -0.5]
    data["y"] = [0.0, -1.0, 0.25]
    data["z"] = [0.2, 0.4, 1.0]
    data["f_dc_0"] = [0.1, 0.2, 0.3]
    data["f_dc_1"] = [0.4, 0.5, 0.6]
    data["f_dc_2"] = [0.7, 0.8, 0.9]
    data["opacity"] = [-2.0, 0.0, 1.0]
    data["scale_0"] = [-1.0, -1.2, -1.4]
    data["scale_1"] = [-1.1, -1.3, -1.5]
    data["scale_2"] = [-1.2, -1.4, -1.6]
    data["rot_0"] = 1.0
    columns = {name: idx for idx, name in enumerate(dtype.names or [])}
    return ActorSequence(
        frames=[ActorSequenceFrame(base_data=data)],
        height=1.8,
        hip_height=0.9,
        radius_xy=0.4,
        columns=columns,
        dtype=dtype,
        feature_rest_names=[],
        scale_names=["scale_0", "scale_1", "scale_2"],
        rot_names=["rot_0", "rot_1", "rot_2", "rot_3"],
        rest_dim=0,
        max_sh_degree=0,
        uniform_scale=False,
        max_points=3,
    )


def test_gpu_actor_sequence_transform_matches_exact_path_for_geometry() -> None:
    sequence = _actor_sequence()
    transform = build_transform_matrix(
        rotation_matrix_z_np(math.radians(37.0)),
        np.array([2.0, -3.0, 0.75], dtype=np.float64),
    )
    cpu_data = apply_transform_to_frame(
        sequence.frames[0],
        sequence,
        transform,
        backend="cpu",
    )
    cpu_render = actor_data_to_tensors(cpu_data, sequence, device=torch.device("cpu"))

    gpu_sequence = build_gpu_actor_sequence(
        sequence,
        device=torch.device("cpu"),
        target_rest_dim=0,
        memory_cap_mb=1.0,
    )
    gpu_render = transform_gpu_actor_frame(gpu_sequence, 0, transform)

    torch.testing.assert_close(gpu_render.xyz, cpu_render.xyz)
    torch.testing.assert_close(gpu_render.scaling, cpu_render.scaling)
    torch.testing.assert_close(gpu_render.features_dc, cpu_render.features_dc)
    torch.testing.assert_close(gpu_render.opacity, cpu_render.opacity)

    # Quaternion signs are equivalent, so compare the absolute unit-quaternion dot.
    quat_dot = torch.sum(gpu_render.rotation * cpu_render.rotation, dim=1).abs()
    torch.testing.assert_close(quat_dot, torch.ones_like(quat_dot))


def test_gpu_actor_sequence_memory_cap_is_enforced() -> None:
    sequence = _actor_sequence()

    with np.testing.assert_raises(MemoryError):
        build_gpu_actor_sequence(
            sequence,
            device=torch.device("cpu"),
            target_rest_dim=0,
            memory_cap_mb=0.000001,
        )


def test_actor_tensor_pack_accepts_rgb_axis_scale_schema() -> None:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("scale_x", "f4"),
            ("scale_y", "f4"),
            ("scale_z", "f4"),
            ("opacity", "f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ]
    )
    data = np.zeros(2, dtype=dtype)
    data["x"] = [0.0, 1.0]
    data["y"] = [0.5, -0.5]
    data["z"] = [0.2, 1.2]
    data["scale_x"] = [-1.0, -2.0]
    data["scale_y"] = [-1.1, -2.1]
    data["scale_z"] = [-1.2, -2.2]
    data["opacity"] = [-3.0, 0.0]
    data["r"] = [255, 128]
    data["g"] = [0, 128]
    data["b"] = [64, 128]
    sequence = ActorSequence(
        frames=[ActorSequenceFrame(base_data=data)],
        height=1.7,
        hip_height=0.85,
        radius_xy=0.3,
        columns={name: idx for idx, name in enumerate(dtype.names or [])},
        dtype=dtype,
        feature_rest_names=[],
        scale_names=["scale_x", "scale_y", "scale_z"],
        rot_names=[],
        rest_dim=0,
        max_sh_degree=0,
        uniform_scale=False,
        max_points=2,
    )

    transformed = apply_transform_to_frame(
        sequence.frames[0],
        sequence,
        build_transform_matrix(
            np.eye(3, dtype=np.float64) * 2.0,
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ),
        backend="cpu",
    )
    render = actor_data_to_tensors(transformed, sequence, device=torch.device("cpu"))

    expected_dc = RGB2SH(np.array([[1.0, 0.0, 64.0 / 255.0], [128.0 / 255.0] * 3], dtype=np.float32))
    torch.testing.assert_close(render.features_dc[:, 0, :], torch.from_numpy(expected_dc))
    torch.testing.assert_close(render.rotation, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
    np.testing.assert_allclose(
        transformed["scale_x"],
        data["scale_x"] + math.log(2.0),
        rtol=1e-6,
        atol=1e-6,
    )
