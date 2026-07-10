from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EPS = 1e-6


@dataclass(frozen=True)
class RobotPose:
    frame: int
    transform: np.ndarray
    yaw_rad: float | None = None


@dataclass(frozen=True)
class PoseConstraintReport:
    ok: bool
    violations: list[str]


@dataclass(frozen=True)
class MeshRenderResult:
    rgba: np.ndarray
    depth_m: np.ndarray


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
) -> dict[int, RobotPose]:
    """Parse per-frame robot poses from an IMO/controller JSON payload."""

    entries = payload.get("frames", payload.get("poses", [])) if isinstance(payload, dict) else payload
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
        poses[frame] = RobotPose(frame=frame, transform=transform, yaw_rad=yaw_rad)
    return poses


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


def decode_quantized_depth(depth_image: np.ndarray, *, bit_depth: int) -> np.ndarray:
    steps = {
        16: 0.001,
        12: 0.001,
        10: 0.002,
        8: 0.04,
    }
    if bit_depth not in steps:
        raise ValueError(f"Unsupported depth bit depth: {bit_depth}")
    return depth_image.astype(np.float32) * steps[bit_depth]


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
        self.normalizer = self._build_normalizer(target_height, up_axis)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)

    def close(self) -> None:
        self.renderer.delete()

    def _load_mesh_nodes(self, loaded: Any) -> tuple[list[tuple[Any, np.ndarray]], np.ndarray]:
        trimesh = self._trimesh
        pyrender = self._pyrender
        if isinstance(loaded, trimesh.Trimesh):
            bounds = np.asarray(loaded.bounds, dtype=np.float64)
            return [(pyrender.Mesh.from_trimesh(loaded, smooth=True), np.eye(4, dtype=np.float64))], bounds

        nodes: list[tuple[Any, np.ndarray]] = []
        all_vertices: list[np.ndarray] = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geom = loaded.geometry[geometry_name]
            transform = np.asarray(transform, dtype=np.float64)
            nodes.append((pyrender.Mesh.from_trimesh(geom, smooth=True), transform))
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
        for mesh, base_transform in self.mesh_nodes:
            scene.add(mesh, pose=root @ base_transform)
        rgba, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return MeshRenderResult(rgba=rgba, depth_m=depth.astype(np.float32, copy=False))
