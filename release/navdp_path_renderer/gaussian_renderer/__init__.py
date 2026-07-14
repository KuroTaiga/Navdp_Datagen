#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import os
import inspect

import torch

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import fov2focal
from utils.sh_utils import eval_sh


_DIFF_GAUSSIAN_BACKEND = None


def _load_diff_gaussian_backend():
    global _DIFF_GAUSSIAN_BACKEND
    if _DIFF_GAUSSIAN_BACKEND is None:
        from diff_gaussian_rasterization import (  # pylint: disable=import-outside-toplevel
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        _DIFF_GAUSSIAN_BACKEND = (GaussianRasterizationSettings, GaussianRasterizer)
    return _DIFF_GAUSSIAN_BACKEND


def _prefer_gsplat() -> bool:
    backend = os.environ.get("GAUSSIAN_RENDER_BACKEND", "gsplat").strip().lower()
    return backend not in {"diff-gaussian", "diff_gaussian", "graphdeco", "legacy"}


def _can_use_gsplat(pipe, scaling_modifier, separate_sh) -> bool:
    return (
        not separate_sh
        and not pipe.compute_cov3D_python
        and float(scaling_modifier) == 1.0
    )


def _load_gsplat_rasterization():
    from gsplat import rasterization  # pylint: disable=import-outside-toplevel

    return rasterization


def _camera_intrinsics(viewpoint_camera, device: torch.device) -> torch.Tensor:
    width = float(viewpoint_camera.image_width)
    height = float(viewpoint_camera.image_height)
    fx = fov2focal(float(viewpoint_camera.FoVx), width)
    fy = fov2focal(float(viewpoint_camera.FoVy), height)
    return torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _compute_precomputed_colors(viewpoint_camera, pc: GaussianModel, pipe, override_color):
    if override_color is not None:
        return override_color
    if not pipe.convert_SHs_python:
        return None

    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    return torch.clamp_min(sh2rgb + 0.5, 0.0)


def _gsplat_render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
    antialiasing=None,
):
    if separate_sh:
        raise NotImplementedError("gsplat backend does not support separate SH tensors")
    if pipe.compute_cov3D_python:
        raise NotImplementedError("gsplat backend does not support precomputed 3D covariances")
    if scaling_modifier != 1.0:
        raise NotImplementedError("gsplat backend does not support scaling_modifier != 1.0")

    rasterization = _load_gsplat_rasterization()
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    means = pc.get_xyz
    device = means.device
    dtype = means.dtype

    colors_precomp = _compute_precomputed_colors(viewpoint_camera, pc, pipe, override_color)
    if colors_precomp is None:
        colors = pc.get_features.contiguous()
        sh_degree = pc.active_sh_degree
    else:
        colors = colors_precomp
        sh_degree = None

    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1).to(device=device, dtype=dtype)[None]
    Ks = _camera_intrinsics(viewpoint_camera, device).to(dtype=dtype)[None]
    backgrounds = bg_color.to(device=device, dtype=dtype)[None]

    render_colors, _, meta = rasterization(
        means=means,
        quats=pc.get_rotation,
        scales=pc.get_scaling,
        opacities=pc.get_opacity.squeeze(-1),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=float(getattr(viewpoint_camera, "znear", 0.01)),
        far_plane=float(getattr(viewpoint_camera, "zfar", 100.0)),
        backgrounds=backgrounds,
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+D",
        rasterize_mode="antialiased" if (pipe.antialiasing if antialiasing is None else antialiasing) else "classic",
    )

    rendered_image = render_colors[0, ..., :3].permute(2, 0, 1).clamp(0, 1)
    depth_image = render_colors[0, ..., 3].unsqueeze(0)

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )
        rendered_image = rendered_image.clamp(0, 1)

    radii = meta["radii"]
    screenspace_points = meta["means2d"]
    while radii.dim() > 1:
        radii = radii[0]
    while screenspace_points.dim() > 2:
        screenspace_points = screenspace_points[0]
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }

def _supports_orthographic() -> bool:
    try:
        GaussianRasterizationSettings, _ = _load_diff_gaussian_backend()
        params = inspect.signature(GaussianRasterizationSettings).parameters
        return "orthographic" in params
    except Exception:
        return False


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if _prefer_gsplat() and _can_use_gsplat(pipe, scaling_modifier, separate_sh):
        return _gsplat_render(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            scaling_modifier=scaling_modifier,
            separate_sh=separate_sh,
            override_color=override_color,
            use_trained_exp=use_trained_exp,
        )
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_kwargs = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )
    GaussianRasterizationSettings, GaussianRasterizer = _load_diff_gaussian_backend()
    raster_settings = GaussianRasterizationSettings(**raster_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out


def render_or(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, orthographic=False, antialiasing=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if _prefer_gsplat() and not orthographic and _can_use_gsplat(pipe, scaling_modifier, separate_sh):
        return _gsplat_render(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            scaling_modifier=scaling_modifier,
            separate_sh=separate_sh,
            override_color=override_color,
            use_trained_exp=use_trained_exp,
            antialiasing=antialiasing,
        )
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if not orthographic:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        full_proj_transform = viewpoint_camera.get_full_proj_transform(orthographic)
    else:
        tanfovx, tanfovy, full_proj_transform = viewpoint_camera.get_full_proj_transform(orthographic)

    if antialiasing is None:
        antialiasing = bool(pipe.antialiasing)

    raster_kwargs = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=antialiasing,
    )
    if _supports_orthographic():
        raster_kwargs["orthographic"] = orthographic
    GaussianRasterizationSettings, GaussianRasterizer = _load_diff_gaussian_backend()
    raster_settings = GaussianRasterizationSettings(**raster_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
