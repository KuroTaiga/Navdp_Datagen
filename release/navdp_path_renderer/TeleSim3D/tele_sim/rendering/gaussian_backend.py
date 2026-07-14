"""Gaussian Splatting renderer backend."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from tele_sim.scene.assets import SceneAsset

if TYPE_CHECKING:  # pragma: no cover
    from tele_sim.core.viewer import Pose


_HERE = Path(__file__).resolve()
_ROOT_CANDIDATES = [
    _HERE.parents[3],
    _HERE.parents[2] / "gaussian-splatting",
]
GAUSSIAN_ROOT = next((root for root in _ROOT_CANDIDATES if (root / "gaussian_renderer").exists()), None)
if GAUSSIAN_ROOT is None:  # pragma: no cover - repository integrity guard
    raise ImportError("Gaussian renderer modules are missing from the release package.")
if str(GAUSSIAN_ROOT) not in sys.path:  # pragma: no cover - idempotent
    sys.path.insert(0, str(GAUSSIAN_ROOT))

from gaussian_renderer import GaussianModel, render as render_gaussians  # type: ignore[attr-defined]
from scene.cameras import MiniCam  # type: ignore[attr-defined]
from utils.graphics_utils import getProjectionMatrix  # type: ignore[attr-defined]


ArrayLike = np.ndarray
Vec3 = Tuple[float, float, float]
LOGGER = logging.getLogger(__name__)


def _ensure_rgb(image: ArrayLike) -> ArrayLike:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., :3]
    return image


def _resize_image(image: ArrayLike, size: Tuple[int, int]) -> ArrayLike:
    width, height = size
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized)


def _quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Convert an (w, x, y, z) quaternion into a 3×3 rotation matrix."""

    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


def _resolve_ply_path(root: Path, *, iteration: Optional[int]) -> Path:
    root = root.expanduser().resolve()

    if root.is_file():
        LOGGER.info("Using provided Gaussian PLY: %s", root)
        return root

    candidates = sorted(root.rglob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"No PLY files found under {root}")

    LOGGER.info("Discovered %d PLY files under %s:", len(candidates), root)
    for idx, path in enumerate(candidates, start=1):
        LOGGER.info("  [%d] %s", idx, path)

    if iteration is not None:
        iteration_tag = f"iteration_{iteration}"
        filtered = [path for path in candidates if iteration_tag in str(path)]
        if filtered:
            LOGGER.info("Selected PLY matching iteration %s: %s", iteration, filtered[0])
            return filtered[0]
        LOGGER.warning(
            "Requested iteration %s not found; falling back to first available PLY.", iteration
        )

    LOGGER.info("Selected first available PLY: %s", candidates[0])
    return candidates[0]


@dataclass(frozen=True)
class GaussianRendererConfig:
    """Configuration for the Gaussian renderer backend."""

    scene_asset: SceneAsset
    model_path: Optional[Path] = None
    frame_size: Tuple[int, int] = (800, 600)
    device: str = "cuda"
    vertical_fov_degrees: float = 60.0
    z_near: float = 0.1
    z_far: float = 100.0
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scaling_modifier: float = 1.0
    antialiasing: bool = False
    debug: bool = False
    separate_sh: bool = False
    sh_degree: int = 3
    iteration: Optional[int] = None
    use_trained_exposure: bool = False
    invert_bev_y: bool = False


class GaussianRendererBackend:
    """Renderer backend powered by the Gaussian Splatting reference implementation."""

    def __init__(self, config: GaussianRendererConfig) -> None:
        self._asset = config.scene_asset
        self._frame_size = config.frame_size
        self._sh_degree = config.sh_degree
        self._device = torch.device(config.device)
        if self._device.type != "cuda":  # pragma: no cover - GPU-only guard
            raise RuntimeError("GaussianRendererBackend currently requires a CUDA device.")
        if not torch.cuda.is_available():  # pragma: no cover - runtime guard
            raise RuntimeError("CUDA device requested but not available. Check torch installation.")

        self._fovy = math.radians(config.vertical_fov_degrees)
        self._znear = config.z_near
        self._zfar = config.z_far
        self._scaling_modifier = config.scaling_modifier
        self._separate_sh = config.separate_sh
        self._use_trained_exposure = config.use_trained_exposure

        self._pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=config.debug,
            antialiasing=config.antialiasing,
        )
        self._background = torch.tensor(config.background_color, dtype=torch.float32, device=self._device)

        model_root = config.model_path or self._asset.splat_model_path
        if model_root is None:
            raise ValueError("GaussianRendererConfig requires a model_path or SceneAsset.splat_model_path.")

        ply_path = _resolve_ply_path(Path(model_root).expanduser().resolve(), iteration=config.iteration)
        self._gaussians = GaussianModel(config.sh_degree)
        self._gaussians.load_ply(str(ply_path), use_train_test_exp=config.use_trained_exposure)
        self._scene_ply_path = ply_path

        self._base_gaussians = self._clone_gaussians()
        self._base_gaussians_gpu = {k: v.clone().to(self._device) for k, v in self._base_gaussians.items()}
        LOGGER.info(
            "Loaded Gaussian scene (%s points) into GPU memory",
            self._base_gaussians["xyz"].shape[0],
        )
        self.apply_gaussians(self._base_gaussians_gpu)

        self._bev_image, self._bev_native_size = self._load_and_prepare_image(self._asset.bev_path)
        self._invert_bev_y = bool(config.invert_bev_y)

    def _load_and_prepare_image(self, path: Path) -> Tuple[ArrayLike, Tuple[int, int]]:
        raw = iio.imread(path)
        rgb = _ensure_rgb(raw)
        native_size = (rgb.shape[1], rgb.shape[0])  # (width, height)
        resized = _resize_image(rgb, self._frame_size)
        return resized.astype(np.uint8), native_size

    def _pose_to_pixel(self, pose: "Pose") -> Tuple[int, int]:
        pixel_x, pixel_y = self.world_to_bev_pixel(pose.position[0], pose.position[2])
        width = self._bev_image.shape[1]
        height = self._bev_image.shape[0]
        px = int(round(pixel_x))
        py = int(round(pixel_y))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        return px, py

    def world_to_bev_pixel(self, world_x: float, world_z: float) -> Tuple[float, float]:
        pixel_x, pixel_y = self._asset.world_to_bev_pixel(world_x, world_z)
        native_w, native_h = self._bev_native_size
        width = self._bev_image.shape[1]
        height = self._bev_image.shape[0]
        if native_w > 0 and native_h > 0:
            # Default to a 180° rotation so the BEV mask lines up with rendered poses.
            if self._invert_bev_y:
                pixel_y = (native_h - 1) - pixel_y
            else:
                pixel_x = (native_w - 1) - pixel_x
                pixel_y = (native_h - 1) - pixel_y
            scale_x = width / float(native_w)
            scale_y = height / float(native_h)
            pixel_x *= scale_x
            pixel_y *= scale_y
        return float(pixel_x), float(pixel_y)

    def bev_dimensions(self) -> Tuple[int, int]:
        return self._bev_image.shape[1], self._bev_image.shape[0]

    def bev_native_dimensions(self) -> Tuple[int, int]:
        return self._bev_native_size

    def set_bev_vertical_flip(self, enabled: bool) -> None:
        self._invert_bev_y = bool(enabled)

    def scene_ply_path(self) -> Optional[Path]:
        try:
            return Path(self._scene_ply_path)
        except AttributeError:
            return None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def sh_degree(self) -> int:
        return self._sh_degree

    def _clone_gaussians(self) -> Dict[str, torch.Tensor]:
        def _clone(t: torch.Tensor) -> torch.Tensor:
            return t.detach().clone().cpu()

        return {
            "xyz": _clone(self._gaussians._xyz),
            "features_dc": _clone(self._gaussians._features_dc),
            "features_rest": _clone(self._gaussians._features_rest),
            "opacity": _clone(self._gaussians._opacity),
            "scaling": _clone(self._gaussians._scaling),
            "rotation": _clone(self._gaussians._rotation),
        }

    def base_gaussians(self) -> Dict[str, torch.Tensor]:
        """Return a CPU snapshot of the scene gaussians as loaded from disk."""

        return {key: value.clone() for key, value in self._base_gaussians.items()}

    def base_gaussians_gpu(self) -> Dict[str, torch.Tensor]:
        """Return a GPU snapshot of the scene gaussians."""

        return {key: value.clone().to(self._device) for key, value in self._base_gaussians_gpu.items()}

    def apply_gaussians(self, tensors: Dict[str, torch.Tensor]) -> None:
        device_tensors = {key: tensor.to(self._device) for key, tensor in tensors.items()}
        self._gaussians._xyz = nn.Parameter(device_tensors["xyz"], requires_grad=False)
        self._gaussians._features_dc = nn.Parameter(device_tensors["features_dc"], requires_grad=False)
        self._gaussians._features_rest = nn.Parameter(device_tensors["features_rest"], requires_grad=False)
        self._gaussians._opacity = nn.Parameter(device_tensors["opacity"], requires_grad=False)
        self._gaussians._scaling = nn.Parameter(device_tensors["scaling"], requires_grad=False)
        self._gaussians._rotation = nn.Parameter(device_tensors["rotation"], requires_grad=False)
        self._gaussians.max_radii2D = torch.zeros((device_tensors["xyz"].shape[0],), device=self._device)
        self._current_gaussians = {key: value.clone() for key, value in device_tensors.items()}

    def restore_base_gaussians(self) -> None:
        self.apply_gaussians(self._base_gaussians_gpu)

    def _overlay_pose_marker(self, image: ArrayLike, pose: "Pose") -> ArrayLike:
        annotated = image.copy()
        px, py = self._pose_to_pixel(pose)
        height, width = annotated.shape[:2]
        if 0 <= px < width and 0 <= py < height:
            size = 4
            x_min = max(px - size, 0)
            x_max = min(px + size + 1, width)
            y_min = max(py - size, 0)
            y_max = min(py + size + 1, height)
            annotated[y_min:y_max, x_min:x_max] = [255, 64, 64]
        return annotated

    def _pose_to_camera(self, pose: "Pose") -> MiniCam:
        width, height = self._frame_size
        aspect_ratio = width / float(height)
        fovx = 2.0 * math.atan(math.tan(self._fovy * 0.5) * aspect_ratio)

        rotation_world = _quat_to_matrix(pose.orientation)
        translation_world = np.asarray(pose.position, dtype=np.float32)

        view_rotation = rotation_world.T
        view_translation = -view_rotation @ translation_world

        world_view = np.eye(4, dtype=np.float32)
        world_view[:3, :3] = view_rotation
        world_view[:3, 3] = view_translation

        world_view_torch = torch.from_numpy(world_view).to(self._device).transpose(0, 1)
        projection_torch = getProjectionMatrix(
            znear=self._znear,
            zfar=self._zfar,
            fovX=fovx,
            fovY=self._fovy,
        ).to(self._device).transpose(0, 1)
        full_proj = world_view_torch.unsqueeze(0).bmm(projection_torch.unsqueeze(0)).squeeze(0)

        return MiniCam(
            width=width,
            height=height,
            fovy=self._fovy,
            fovx=fovx,
            znear=self._znear,
            zfar=self._zfar,
            world_view_transform=world_view_torch,
            full_proj_transform=full_proj,
        )

    def _render_gaussians(self, pose: "Pose") -> ArrayLike:
        camera = self._pose_to_camera(pose)
        with torch.no_grad():
            result = render_gaussians(
                camera,
                self._gaussians,
                self._pipe,
                self._background,
                scaling_modifier=self._scaling_modifier,
                separate_sh=self._separate_sh,
                use_trained_exp=self._use_trained_exposure,
            )
        image = result["render"].permute(1, 2, 0).detach().cpu().numpy()
        image = np.clip(image, 0.0, 1.0)
        return (image * 255.0).astype(np.uint8)

    def render_rgb(self, pose: "Pose") -> ArrayLike:
        frame = self._render_gaussians(pose)
        return frame

    def render_bev(self, pose: "Pose") -> ArrayLike:
        return self._overlay_pose_marker(self._bev_image, pose)

    # Compatibility helpers for follow-dataset export

    def render_rgbd(self, pose: "Pose") -> Tuple[ArrayLike, Optional[ArrayLike]]:
        """Render RGB and a placeholder depth map."""

        return self.render_rgb(pose), None

    def camera_matrices(self, pose: "Pose") -> Dict[str, object]:
        """Return world/view/projection matrices and intrinsics."""

        cam = self._pose_to_camera(pose)
        intrinsics = {
            "fx": cam.full_proj_transform[0, 0].item(),
            "fy": cam.full_proj_transform[1, 1].item(),
            "cx": self._frame_size[0] * 0.5,
            "cy": self._frame_size[1] * 0.5,
            "width": self._frame_size[0],
            "height": self._frame_size[1],
            "znear": self._znear,
            "zfar": self._zfar,
        }
        return {
            "world_view": cam.world_view_transform.detach().cpu().numpy(),
            "full_projection": cam.full_proj_transform.detach().cpu().numpy(),
            "intrinsics": intrinsics,
        }

    def project_world_points(self, pose: "Pose", points: np.ndarray) -> np.ndarray:
        """Project world-space points into the current camera image plane."""

        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        camera = self._pose_to_camera(pose)
        full_proj = camera.full_proj_transform.detach().cpu().numpy()
        width, height = self._frame_size
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([points.astype(np.float32), ones])
        clip = homogeneous @ full_proj.T
        w = clip[:, 3]
        valid = w > 0
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float32)
        clip = clip[valid]
        w = w[valid]
        ndc = clip[:, :3] / w[:, None]
        mask = (
            (np.abs(ndc[:, 0]) <= 1.0)
            & (np.abs(ndc[:, 1]) <= 1.0)
            & (ndc[:, 2] >= -1.0)
            & (ndc[:, 2] <= 1.0)
        )
        if not np.any(mask):
            return np.empty((0, 2), dtype=np.float32)
        ndc = ndc[mask]
        x = ((ndc[:, 0] * 0.5) + 0.5) * width
        y = ((-ndc[:, 1] * 0.5) + 0.5) * height
        return np.stack([x, y], axis=1).astype(np.float32)

    def shutdown(self) -> None:
        if self._device.type == "cuda":  # pragma: no cover - depends on runtime
            torch.cuda.empty_cache()
