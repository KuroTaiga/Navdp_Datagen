from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EPS = 1e-6

DEPTH_ENCODING_SPECS = {
    16: {"step_m": 0.001, "max_m": 0.001 * ((1 << 16) - 1)},
    12: {"step_m": 0.001, "max_m": 0.001 * ((1 << 12) - 1)},
    10: {"step_m": 0.002, "max_m": 0.002 * ((1 << 10) - 1)},
    8: {"step_m": 0.04, "max_m": 0.04 * ((1 << 8) - 1)},
}


@dataclass(frozen=True)
class RobotPose:
    frame: int
    transform: np.ndarray
    yaw_rad: float | None = None
    joint_positions: dict[str, float] | None = None


@dataclass(frozen=True)
class PoseConstraintReport:
    ok: bool
    violations: list[str]


@dataclass(frozen=True)
class MeshRenderResult:
    rgba: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True)
class _MeshNode:
    mesh: Any
    base_transform: np.ndarray
    node_name: str
    candidate_names: tuple[str, ...]
    link_name: str | None = None


def _translation_matrix(xyz: Iterable[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    values = list(xyz)
    if len(values) != 3:
        raise ValueError("translation must contain exactly 3 values")
    mat[:3, 3] = np.asarray(values, dtype=np.float64)
    return mat


def _scale_matrix(scale: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] *= float(scale)
    return mat


def _yaw_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _quaternion_wxyz_to_matrix(quat: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(quat), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must contain exactly 4 values")
    norm = float(np.linalg.norm(q))
    if norm < EPS:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    return mat


def _axis_alignment_matrix(up_axis: str) -> np.ndarray:
    normalized = up_axis.strip().lower()
    if normalized == "z":
        return np.eye(4, dtype=np.float64)
    if normalized == "y":
        # glTF commonly uses +Y as up and -Z as forward. Map that to NavDP's
        # +Z up, +Y forward world convention before applying path yaw.
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return mat
    raise ValueError(f"Unsupported up axis: {up_axis!r}; expected 'y' or 'z'")


def parse_robot_poses(
    payload: dict[str, Any] | list[Any],
    *,
    yaw_offset_rad: float = 0.0,
    foot_offset: float = 0.0,
    default_z: float = 0.0,
    joint_names: list[str] | None = None,
) -> dict[int, RobotPose]:
    """Parse per-frame robot poses from an IMO/AMO/controller JSON payload."""

    if isinstance(payload, dict):
        entries = payload.get("frames", payload.get("poses", []))
        joint_names = _joint_names_from_payload(payload, joint_names=joint_names)
        default_joint_positions = _joint_positions_from_payload(
            payload.get("default_joint_positions")
            or payload.get("default_joints")
            or payload.get("default_amo_pose"),
            joint_names=joint_names,
            frame_label="default joint positions",
            required=False,
        )
    else:
        entries = payload
        default_joint_positions = None
    poses: dict[int, RobotPose] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Pose entry #{idx} must be an object")
        frame = int(entry.get("frame", entry.get("id", idx)))
        transform_raw = entry.get("transform") or entry.get("matrix")
        yaw_rad: float | None = None
        if transform_raw is not None:
            transform = np.asarray(transform_raw, dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError(f"Pose frame {frame}: transform must be 4x4")
            if yaw_offset_rad:
                transform = transform @ _yaw_matrix(yaw_offset_rad)
        else:
            position = (
                entry.get("position")
                or entry.get("translation")
                or entry.get("xyz")
                or [entry.get("x", 0.0), entry.get("y", 0.0), entry.get("z", default_z)]
            )
            if len(position) == 2:
                position = [position[0], position[1], default_z]
            xyz = np.asarray(position, dtype=np.float64)
            if xyz.shape != (3,):
                raise ValueError(f"Pose frame {frame}: position must have 2 or 3 values")
            xyz[2] += float(foot_offset)

            if "yaw_rad" in entry:
                yaw_rad = float(entry["yaw_rad"])
                rot = _yaw_matrix(yaw_rad + yaw_offset_rad)
            elif "yaw_deg" in entry:
                yaw_rad = math.radians(float(entry["yaw_deg"]))
                rot = _yaw_matrix(yaw_rad + yaw_offset_rad)
            elif "quaternion_wxyz" in entry:
                rot = _quaternion_wxyz_to_matrix(entry["quaternion_wxyz"])
                if yaw_offset_rad:
                    rot = rot @ _yaw_matrix(yaw_offset_rad)
            else:
                yaw_rad = float(yaw_offset_rad)
                rot = _yaw_matrix(yaw_offset_rad)
            transform = _translation_matrix(xyz) @ rot
        frame_joint_positions = _joint_positions_from_entry(
            entry,
            joint_names=joint_names,
            frame_label=f"Pose frame {frame}",
        )
        if default_joint_positions is not None or frame_joint_positions is not None:
            merged_joint_positions = dict(default_joint_positions or {})
            merged_joint_positions.update(frame_joint_positions or {})
        else:
            merged_joint_positions = None
        poses[frame] = RobotPose(
            frame=frame,
            transform=transform,
            yaw_rad=yaw_rad,
            joint_positions=merged_joint_positions,
        )
    return poses


def parse_robot_joint_poses(
    payload: dict[str, Any] | list[Any],
    *,
    joint_names: list[str] | None = None,
) -> dict[int, dict[str, float]]:
    """Parse per-frame AMO/joint values without requiring base robot poses."""

    if isinstance(payload, dict):
        entries = payload.get("frames", payload.get("poses", []))
        joint_names = _joint_names_from_payload(payload, joint_names=joint_names)
    else:
        entries = payload
    joint_poses: dict[int, dict[str, float]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"AMO entry #{idx} must be an object")
        frame = int(entry.get("frame", entry.get("id", idx)))
        frame_joints = _joint_positions_from_entry(
            entry,
            joint_names=joint_names,
            frame_label=f"AMO frame {frame}",
        )
        if frame_joints is None:
            raise ValueError(f"AMO frame {frame}: missing joint_positions/joints/amo_pose/qpos")
        joint_poses[frame] = frame_joints
    return joint_poses


def _joint_names_from_payload(payload: dict[str, Any], *, joint_names: list[str] | None) -> list[str] | None:
    raw = (
        payload.get("joint_names")
        or payload.get("amo_joint_names")
        or payload.get("dof_names")
        or payload.get("qpos_names")
    )
    if raw is None:
        return joint_names
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("joint_names/amo_joint_names/dof_names must be a list of strings")
    if joint_names is not None and list(joint_names) != raw:
        raise ValueError("Joint names supplied by CLI do not match pose payload joint names")
    return list(raw)


def _joint_positions_from_entry(
    entry: dict[str, Any],
    *,
    joint_names: list[str] | None,
    frame_label: str,
) -> dict[str, float] | None:
    for key in ("joint_positions", "joints", "amo_pose", "amo", "qpos", "joint_values"):
        if key in entry:
            return _joint_positions_from_payload(
                entry[key],
                joint_names=joint_names,
                frame_label=f"{frame_label} {key}",
                required=True,
            )
    return None


def _joint_positions_from_payload(
    raw: Any,
    *,
    joint_names: list[str] | None,
    frame_label: str,
    required: bool,
) -> dict[str, float] | None:
    if raw is None:
        if required:
            raise ValueError(f"{frame_label}: missing joint position data")
        return None
    if isinstance(raw, dict):
        return {str(name): float(value) for name, value in raw.items()}
    if isinstance(raw, list):
        if joint_names is None:
            raise ValueError(f"{frame_label}: list-valued AMO pose requires joint_names")
        if len(raw) != len(joint_names):
            raise ValueError(
                f"{frame_label}: expected {len(joint_names)} joint values, got {len(raw)}"
            )
        return {name: float(value) for name, value in zip(joint_names, raw)}
    raise ValueError(f"{frame_label}: joint positions must be an object or list")


def validate_pose_constraints(
    poses: dict[int, RobotPose],
    *,
    fps: float,
    max_speed_mps: float | None = None,
    max_yaw_rate_radps: float | None = None,
) -> PoseConstraintReport:
    violations: list[str] = []
    if fps <= 0:
        raise ValueError("fps must be positive")
    ordered = [poses[key] for key in sorted(poses)]
    for prev, curr in zip(ordered, ordered[1:]):
        frame_delta = max(curr.frame - prev.frame, 1)
        dt = frame_delta / float(fps)
        prev_xyz = prev.transform[:3, 3]
        curr_xyz = curr.transform[:3, 3]
        speed = float(np.linalg.norm(curr_xyz - prev_xyz) / dt)
        if max_speed_mps is not None and speed > max_speed_mps + 1e-6:
            violations.append(
                f"frames {prev.frame}->{curr.frame}: speed {speed:.3f} m/s exceeds {max_speed_mps:.3f}"
            )
        if (
            max_yaw_rate_radps is not None
            and prev.yaw_rad is not None
            and curr.yaw_rad is not None
        ):
            delta = (curr.yaw_rad - prev.yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate = abs(float(delta / dt))
            if yaw_rate > max_yaw_rate_radps + 1e-6:
                violations.append(
                    f"frames {prev.frame}->{curr.frame}: yaw rate {math.degrees(yaw_rate):.3f} deg/s "
                    f"exceeds {math.degrees(max_yaw_rate_radps):.3f}"
                )
    return PoseConstraintReport(ok=not violations, violations=violations)


def camera_metadata_to_pyrender_pose(frame_payload: dict[str, Any]) -> np.ndarray:
    """Convert NavDP camera metadata to a standard OpenGL camera-to-world pose."""

    camera_to_world = np.asarray(frame_payload["camera_to_world"], dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must be a 4x4 matrix")
    pose = camera_to_world.T.copy()
    # NavDP/GS camera space uses +Z forward with an image-space Y direction
    # opposite to OpenGL. Pyrender cameras use -Z forward and +Y up.
    pose[:3, 1] *= -1.0
    pose[:3, 2] *= -1.0
    return pose


def compose_rgba_over_rgb(
    base_rgb: np.ndarray,
    overlay_rgba: np.ndarray,
    *,
    mesh_depth_m: np.ndarray | None = None,
    base_depth_m: np.ndarray | None = None,
    depth_bias_m: float = 0.0,
) -> np.ndarray:
    """Composite an RGBA mesh render over an RGB frame, optionally depth-gated."""

    if base_rgb.ndim != 3 or base_rgb.shape[2] != 3:
        raise ValueError("base_rgb must be HxWx3")
    if overlay_rgba.shape[:2] != base_rgb.shape[:2] or overlay_rgba.shape[2] != 4:
        raise ValueError("overlay_rgba must be HxWx4 and match base_rgb resolution")

    base = base_rgb.astype(np.float32)
    overlay = overlay_rgba.astype(np.float32)
    alpha = overlay[..., 3:4] / (255.0 if overlay_rgba.dtype == np.uint8 else 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    if mesh_depth_m is not None and base_depth_m is not None:
        if mesh_depth_m.shape != base_rgb.shape[:2] or base_depth_m.shape != base_rgb.shape[:2]:
            raise ValueError("depth maps must match base_rgb resolution")
        depth_mask = (mesh_depth_m > 0.0) & (
            (base_depth_m <= 0.0) | (mesh_depth_m <= base_depth_m + float(depth_bias_m))
        )
        alpha = alpha * depth_mask[..., None].astype(np.float32)

    overlay_rgb = overlay[..., :3]
    if overlay_rgba.dtype != np.uint8:
        overlay_rgb = overlay_rgb * 255.0
    composed = overlay_rgb * alpha + base * (1.0 - alpha)
    return np.clip(composed, 0.0, 255.0).astype(np.uint8)


def quantize_depth_meters(depth_m: np.ndarray, *, bit_depth: int) -> np.ndarray:
    if bit_depth not in DEPTH_ENCODING_SPECS:
        raise ValueError(f"Unsupported depth bit depth: {bit_depth}")
    spec = DEPTH_ENCODING_SPECS[bit_depth]
    depth = depth_m.astype(np.float32, copy=False)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_clipped = np.clip(depth, 0.0, float(spec["max_m"]))
    depth_quant = np.rint(depth_clipped / float(spec["step_m"]))
    return depth_quant.astype(np.uint16 if bit_depth > 8 else np.uint8)


def decode_quantized_depth(depth_image: np.ndarray, *, bit_depth: int) -> np.ndarray:
    if bit_depth not in DEPTH_ENCODING_SPECS:
        raise ValueError(f"Unsupported depth bit depth: {bit_depth}")
    return depth_image.astype(np.float32) * float(DEPTH_ENCODING_SPECS[bit_depth]["step_m"])


class GlbRobotRenderer:
    """Small pyrender-backed GLB renderer for robot foreground composition."""

    def __init__(
        self,
        glb_path: Path,
        *,
        width: int,
        height: int,
        target_height: float | None = None,
        up_axis: str = "y",
        articulation_urdf_path: Path | None = None,
        articulation_package_root: Path | None = None,
        bind_joint_positions: dict[str, float] | None = None,
        pyopengl_platform: str = "egl",
        ambient_light: tuple[float, float, float] = (0.35, 0.35, 0.35),
    ) -> None:
        if pyopengl_platform:
            os.environ.setdefault("PYOPENGL_PLATFORM", pyopengl_platform)
        try:
            import pyrender  # pylint: disable=import-outside-toplevel
            import trimesh  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "GLB robot rendering requires optional packages: trimesh, pyrender, PyOpenGL, pyglet."
            ) from exc

        self._pyrender = pyrender
        self._trimesh = trimesh
        self.width = int(width)
        self.height = int(height)
        self.ambient_light = np.asarray(ambient_light, dtype=np.float32)
        loaded = trimesh.load(str(glb_path), force="scene")
        self.mesh_nodes, self._bounds = self._load_mesh_nodes(loaded)
        self.link_bind_transforms: dict[str, np.ndarray] | None = None
        self.joints: list[Any] | None = None
        self.links: set[str] | None = None
        if articulation_urdf_path is not None:
            self._load_articulation(
                articulation_urdf_path,
                package_root=articulation_package_root,
                bind_joint_positions=bind_joint_positions,
            )
        self.normalizer = self._build_normalizer(target_height, up_axis)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)

    def close(self) -> None:
        self.renderer.delete()

    def _load_mesh_nodes(self, loaded: Any) -> tuple[list[_MeshNode], np.ndarray]:
        trimesh = self._trimesh
        pyrender = self._pyrender
        if isinstance(loaded, trimesh.Trimesh):
            bounds = np.asarray(loaded.bounds, dtype=np.float64)
            return [
                _MeshNode(
                    mesh=pyrender.Mesh.from_trimesh(loaded, smooth=True),
                    base_transform=np.eye(4, dtype=np.float64),
                    node_name="mesh",
                    candidate_names=("mesh",),
                )
            ], bounds

        nodes: list[_MeshNode] = []
        all_vertices: list[np.ndarray] = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geom = loaded.geometry[geometry_name]
            transform = np.asarray(transform, dtype=np.float64)
            nodes.append(
                _MeshNode(
                    mesh=pyrender.Mesh.from_trimesh(geom, smooth=True),
                    base_transform=transform,
                    node_name=str(node_name),
                    candidate_names=(str(node_name), str(geometry_name)),
                )
            )
            vertices = np.asarray(geom.vertices, dtype=np.float64)
            if vertices.size:
                vertices_h = np.concatenate(
                    [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                all_vertices.append((transform @ vertices_h.T).T[:, :3])
        if not nodes:
            raise ValueError("GLB scene contains no renderable geometry")
        if not all_vertices:
            bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        else:
            vertices = np.concatenate(all_vertices, axis=0)
            bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
        return nodes, bounds

    def _load_articulation(
        self,
        urdf_path: Path,
        *,
        package_root: Path | None,
        bind_joint_positions: dict[str, float] | None,
    ) -> None:
        try:
            from scripts.render.assets.convert_urdf_visuals_to_glb import (  # pylint: disable=import-outside-toplevel
                compute_link_transforms,
                parse_urdf_visuals,
            )
        except ImportError as exc:
            raise RuntimeError("URDF articulation requires scripts.convert_urdf_visuals_to_glb.") from exc

        _, joints, links = parse_urdf_visuals(urdf_path, package_root=package_root)
        bind_transforms = compute_link_transforms(
            links,
            joints,
            joint_positions=bind_joint_positions,
        )
        resolved_nodes: list[_MeshNode] = []
        unresolved: list[str] = []
        for node in self.mesh_nodes:
            link_name = _resolve_link_name(node.candidate_names, links)
            if link_name is None:
                unresolved.append(node.node_name)
            resolved_nodes.append(
                _MeshNode(
                    mesh=node.mesh,
                    base_transform=node.base_transform,
                    node_name=node.node_name,
                    candidate_names=node.candidate_names,
                    link_name=link_name,
                )
            )
        if unresolved:
            preview = ", ".join(unresolved[:5])
            raise ValueError(
                "Cannot map GLB mesh nodes to URDF links. Expected nodes named like "
                f"'<link>_<visual_idx>' from convert_urdf_visuals_to_glb.py; unresolved: {preview}"
            )
        self.mesh_nodes = resolved_nodes
        self.link_bind_transforms = bind_transforms
        self.joints = joints
        self.links = links

    def _link_transforms_for_joints(self, joint_positions: dict[str, float] | None) -> dict[str, np.ndarray] | None:
        if joint_positions is None:
            return None
        if self.joints is None or self.links is None or self.link_bind_transforms is None:
            return None
        from scripts.render.assets.convert_urdf_visuals_to_glb import compute_link_transforms  # pylint: disable=import-outside-toplevel

        return compute_link_transforms(
            self.links,
            self.joints,
            joint_positions=joint_positions,
        )

    def _build_normalizer(self, target_height: float | None, up_axis: str) -> np.ndarray:
        alignment = _axis_alignment_matrix(up_axis)
        corners = np.array(
            [
                [self._bounds[ix, 0], self._bounds[iy, 1], self._bounds[iz, 2], 1.0]
                for ix in (0, 1)
                for iy in (0, 1)
                for iz in (0, 1)
            ],
            dtype=np.float64,
        )
        aligned = (alignment @ corners.T).T[:, :3]
        min_z = float(aligned[:, 2].min())
        max_z = float(aligned[:, 2].max())
        measured_height = max(max_z - min_z, EPS)
        scale_factor = 1.0
        if target_height is not None and target_height > 0:
            scale_factor = float(target_height) / measured_height
        return _translation_matrix([0.0, 0.0, -min_z * scale_factor]) @ _scale_matrix(scale_factor) @ alignment

    def render(
        self,
        *,
        camera_frame: dict[str, Any],
        robot_transform: np.ndarray,
        joint_positions: dict[str, float] | None = None,
    ) -> MeshRenderResult:
        pyrender = self._pyrender
        resolution = camera_frame["resolution"]
        if int(resolution["width"]) != self.width or int(resolution["height"]) != self.height:
            raise ValueError("Camera metadata resolution does not match GLB renderer resolution")
        intrinsics = camera_frame["intrinsics"]
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=self.ambient_light)
        camera = pyrender.IntrinsicsCamera(
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
            znear=float(camera_frame.get("znear", 0.01)),
            zfar=float(camera_frame.get("zfar", 100.0)),
        )
        camera_pose = camera_metadata_to_pyrender_pose(camera_frame)
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=camera_pose)
        root = np.asarray(robot_transform, dtype=np.float64) @ self.normalizer
        posed_link_transforms = self._link_transforms_for_joints(joint_positions)
        for node in self.mesh_nodes:
            local_pose = node.base_transform
            if (
                posed_link_transforms is not None
                and self.link_bind_transforms is not None
                and node.link_name is not None
            ):
                local_pose = (
                    posed_link_transforms[node.link_name]
                    @ np.linalg.inv(self.link_bind_transforms[node.link_name])
                    @ node.base_transform
                )
            scene.add(node.mesh, pose=root @ local_pose)
        rgba, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return MeshRenderResult(rgba=rgba, depth_m=depth.astype(np.float32, copy=False))


def _resolve_link_name(candidate_names: Iterable[str], links: set[str]) -> str | None:
    for node_name in candidate_names:
        if node_name in links:
            return node_name
        prefix, sep, suffix = node_name.rpartition("_")
        if sep and suffix.isdigit() and prefix in links:
            return prefix
    return None
