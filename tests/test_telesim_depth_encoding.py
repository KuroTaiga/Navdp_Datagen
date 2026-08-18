from __future__ import annotations

import numpy as np

from utils.glb_robot_compositor import decode_quantized_depth, quantize_depth_meters


def test_telesim_depth_encoding_matches_glb_compositor_decoder() -> None:
    depth_m = np.array([[0.0, 1.25, 12.345]], dtype=np.float32)

    encoded = quantize_depth_meters(depth_m, bit_depth=16)
    decoded = decode_quantized_depth(encoded, bit_depth=16)

    np.testing.assert_allclose(decoded, depth_m, atol=0.001)


def test_telesim_depth_encoding_clips_by_bit_depth_range() -> None:
    encoded = quantize_depth_meters(np.array([[100.0]], dtype=np.float32), bit_depth=8)
    decoded = decode_quantized_depth(encoded, bit_depth=8)

    np.testing.assert_allclose(decoded, [[10.2]], atol=1e-6)
