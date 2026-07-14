#!/usr/bin/env python3
"""Materialize packed 3DGS PLY scenes as standard GraphDeco PLY files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plyfile import PlyData


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
sys.path.insert(0, str(ROOT_DIR))

from scene.gaussian_model import GaussianModel  # noqa: E402


PACKED_PROPS = {
    "packed_position",
    "packed_rotation",
    "packed_scale",
    "packed_color",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack a compressed 3DGS PLY into a standard float-field PLY. "
            "The output can be passed to renderers as --gaussian-model."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Scene directory, compressed PLY file, or parent directory when "
            "--recursive is set."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PLY path. Defaults to 3dgs_decompressed.ply for a scene "
            "directory or sibling <stem>_decompressed.ply for another PLY."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find and unpack every */3dgs_compressed.ply below the input directory.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Initial spherical-harmonics degree. Default: 3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output PLY.",
    )
    parser.add_argument(
        "--copy-standard",
        action="store_true",
        help=(
            "Also rewrite already-standard PLY files through GaussianModel. "
            "By default they are reported and skipped."
        ),
    )
    return parser.parse_args()


def resolve_input_ply(input_path: Path) -> Path:
    if input_path.is_dir():
        ply_path = input_path / "3dgs_compressed.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"scene directory has no 3dgs_compressed.ply: {input_path}")
        return ply_path
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    return input_path


def default_output_path(input_path: Path, ply_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "3dgs_decompressed.ply"
    if ply_path.name == "3dgs_compressed.ply":
        return ply_path.with_name("3dgs_decompressed.ply")
    return ply_path.with_name(f"{ply_path.stem}_decompressed{ply_path.suffix}")


def is_packed_ply(ply_path: Path) -> bool:
    plydata = PlyData.read(ply_path)
    try:
        names = plydata["vertex"].data.dtype.names or ()
    except KeyError as exc:
        raise ValueError(f"PLY has no vertex element: {ply_path}") from exc
    return PACKED_PROPS.issubset(set(names))


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GaussianModel.load_ply() in this repo allocates tensors on CUDA. "
            "Run this script in a CUDA-capable PyTorch environment."
        )


def unpack_one(
    input_path: Path,
    output_path: Path | None,
    sh_degree: int,
    overwrite: bool,
    copy_standard: bool,
) -> str:
    ply_path = resolve_input_ply(input_path).resolve()
    out_path = (output_path or default_output_path(input_path, ply_path)).resolve()

    packed = is_packed_ply(ply_path)
    if not packed and not copy_standard:
        return f"[SKIP] already standard PLY: {ply_path}"

    if out_path.exists() and not overwrite:
        return f"[SKIP] output exists: {out_path}"

    require_cuda()
    model = GaussianModel(sh_degree)
    model.load_ply(str(ply_path))
    model.save_ply(str(out_path))

    mode = "unpacked" if packed else "rewrote standard"
    return f"[OK] {mode}: {ply_path} -> {out_path}"


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()

    if args.recursive:
        if args.output is not None:
            raise ValueError("--output cannot be used with --recursive")
        if not input_path.is_dir():
            raise ValueError("--recursive input must be a directory")
        ply_paths = sorted(input_path.rglob("3dgs_compressed.ply"))
        if not ply_paths:
            raise FileNotFoundError(f"no 3dgs_compressed.ply files found below {input_path}")
        for ply_path in ply_paths:
            print(
                unpack_one(
                    ply_path,
                    None,
                    args.sh_degree,
                    args.overwrite,
                    args.copy_standard,
                ),
                flush=True,
            )
        return 0

    print(
        unpack_one(
            input_path,
            args.output,
            args.sh_degree,
            args.overwrite,
            args.copy_standard,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
