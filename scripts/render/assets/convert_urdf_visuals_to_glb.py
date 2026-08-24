#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class VisualSpec:
    link: str
    mesh_path: Path
    transform: np.ndarray
    scale: np.ndarray
    rgba: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def _parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = [float(part) for part in text.split()]
    if len(values) != len(default):
        raise ValueError(f"Expected {len(default)} values, got {len(values)} in {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_to_matrix(origin: ET.Element | None) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if origin is None:
        return mat
    xyz = _parse_vec(origin.attrib.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_vec(origin.attrib.get("rpy"), (0.0, 0.0, 0.0))
    mat[:3, :3] = _rpy_to_matrix(rpy)
    mat[:3, 3] = xyz
    return mat


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.eye(4, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    rot = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    return mat


def _translation_matrix(xyz: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = xyz
    return mat


def _joint_motion(joint: JointSpec, value: float) -> np.ndarray:
    if joint.joint_type in {"fixed", "floating"}:
        return np.eye(4, dtype=np.float64)
    if joint.joint_type in {"revolute", "continuous"}:
        return _axis_angle_to_matrix(joint.axis, value)
    if joint.joint_type == "prismatic":
        return _translation_matrix(joint.axis * float(value))
    return np.eye(4, dtype=np.float64)


def _rgba_to_u8(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None:
        return None
    values = [float(part) for part in text.split()]
    if len(values) != 4:
        return None
    return tuple(int(round(max(0.0, min(1.0, v)) * 255.0)) for v in values)


def _load_materials(root: ET.Element) -> dict[str, tuple[int, int, int, int]]:
    materials: dict[str, tuple[int, int, int, int]] = {}
    for material in root.findall("material"):
        name = material.attrib.get("name")
        color = material.find("color")
        rgba = _rgba_to_u8(color.attrib.get("rgba") if color is not None else None)
        if name and rgba is not None:
            materials[name] = rgba
    return materials


def _resolve_mesh_path(filename: str, *, urdf_dir: Path, package_root: Path | None) -> Path:
    if filename.startswith("package://"):
        rest = filename[len("package://") :]
        parts = rest.split("/", 1)
        relative = parts[1] if len(parts) == 2 else parts[0]
        bases = [package_root, urdf_dir, urdf_dir.parent]
        for base in bases:
            if base is None:
                continue
            candidate = base / relative
            if candidate.is_file():
                return candidate
        return (package_root or urdf_dir) / relative
    path = Path(filename)
    if path.is_absolute():
        return path
    return urdf_dir / path


def parse_urdf_visuals(
    urdf_path: Path,
    *,
    package_root: Path | None = None,
) -> tuple[dict[str, list[VisualSpec]], list[JointSpec], set[str]]:
    root = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent
    materials = _load_materials(root)
    links = {link.attrib["name"] for link in root.findall("link") if "name" in link.attrib}
    visuals_by_link: dict[str, list[VisualSpec]] = {name: [] for name in links}

    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if not link_name:
            continue
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None or "filename" not in mesh.attrib:
                continue
            material = visual.find("material")
            rgba = None
            if material is not None:
                inline = material.find("color")
                rgba = _rgba_to_u8(inline.attrib.get("rgba") if inline is not None else None)
                if rgba is None:
                    material_name = material.attrib.get("name")
                    rgba = materials.get(material_name) if material_name else None
            visuals_by_link.setdefault(link_name, []).append(
                VisualSpec(
                    link=link_name,
                    mesh_path=_resolve_mesh_path(
                        mesh.attrib["filename"],
                        urdf_dir=urdf_dir,
                        package_root=package_root,
                    ),
                    transform=_origin_to_matrix(visual.find("origin")),
                    scale=_parse_vec(mesh.attrib.get("scale"), (1.0, 1.0, 1.0)),
                    rgba=rgba,
                )
            )

    joints: list[JointSpec] = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "fixed")
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        if parent is None or child is None:
            continue
        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin=_origin_to_matrix(joint.find("origin")),
                axis=_parse_vec(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            )
        )
    return visuals_by_link, joints, links


def compute_link_transforms(
    links: set[str],
    joints: list[JointSpec],
    *,
    joint_positions: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    joint_positions = joint_positions or {}
    children = {joint.child for joint in joints}
    roots = sorted(link for link in links if link not in children)
    root_link = roots[0] if roots else sorted(links)[0]
    by_parent: dict[str, list[JointSpec]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent, []).append(joint)

    transforms = {root_link: np.eye(4, dtype=np.float64)}
    stack = [root_link]
    while stack:
        parent = stack.pop()
        parent_tf = transforms[parent]
        for joint in by_parent.get(parent, []):
            value = float(joint_positions.get(joint.name, 0.0))
            transforms[joint.child] = parent_tf @ joint.origin @ _joint_motion(joint, value)
            stack.append(joint.child)
    for link in links:
        transforms.setdefault(link, np.eye(4, dtype=np.float64))
    return transforms


def _scale_matrix(scale: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = float(scale[0])
    mat[1, 1] = float(scale[1])
    mat[2, 2] = float(scale[2])
    return mat


def convert_urdf_visuals_to_glb(
    *,
    urdf_path: Path,
    output_path: Path,
    package_root: Path | None = None,
    joint_positions: dict[str, float] | None = None,
) -> int:
    try:
        import trimesh  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("URDF-to-GLB conversion requires trimesh.") from exc

    visuals_by_link, joints, links = parse_urdf_visuals(urdf_path, package_root=package_root)
    link_transforms = compute_link_transforms(links, joints, joint_positions=joint_positions)
    scene = trimesh.Scene()
    count = 0
    for link_name, visuals in visuals_by_link.items():
        for visual_idx, visual in enumerate(visuals):
            if not visual.mesh_path.is_file():
                print(f"[WARN] Missing mesh for {link_name}: {visual.mesh_path}", flush=True)
                continue
            mesh = trimesh.load_mesh(str(visual.mesh_path), process=False)
            if not hasattr(mesh, "vertices"):
                print(f"[WARN] Unsupported mesh for {link_name}: {visual.mesh_path}", flush=True)
                continue
            transform = link_transforms[link_name] @ visual.transform @ _scale_matrix(visual.scale)
            mesh = mesh.copy()
            mesh.apply_transform(transform)
            if visual.rgba is not None:
                mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=visual.rgba)
            scene.add_geometry(mesh, node_name=f"{link_name}_{visual_idx}", geom_name=f"{link_name}_{visual_idx}")
            count += 1
    if count == 0:
        raise RuntimeError(f"No visual meshes were exported from {urdf_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert URDF visual meshes to a static GLB.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("data/g1_description/g1_29dof_mode_16.urdf"),
        help="URDF path (default: data/g1_description/g1_29dof_mode_16.urdf).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GLB path (default: same path as URDF with .glb suffix).",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help="Package root used to resolve package:// mesh URIs.",
    )
    parser.add_argument(
        "--joint-positions-json",
        type=Path,
        default=None,
        help="Optional JSON object mapping joint names to radians/meters.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output or args.urdf.with_suffix(".glb")
    joint_positions = None
    if args.joint_positions_json is not None:
        with args.joint_positions_json.open("r", encoding="utf-8") as fh:
            joint_positions = {str(k): float(v) for k, v in json.load(fh).items()}
    count = convert_urdf_visuals_to_glb(
        urdf_path=args.urdf,
        output_path=output,
        package_root=args.package_root,
        joint_positions=joint_positions,
    )
    print(f"[DONE] Exported {count} visual mesh(es) to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
