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
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        try:
            vertex_element = plydata["vertex"]
        except KeyError as exc:
            raise ValueError("PLY file does not contain a 'vertex' element") from exc

        vertex_data = vertex_element.data
        vertex_props = vertex_data.dtype.names if vertex_data.dtype.names is not None else ()

        if "packed_position" in vertex_props:
            try:
                chunk_element = plydata["chunk"]
            except KeyError as exc:
                raise ValueError("Compressed PLY is missing required 'chunk' element") from exc

            chunk_data = chunk_element.data
            vert_count = vertex_data.shape[0]
            chunk_size = 256
            indices = np.arange(vert_count, dtype=np.int64)
            chunk_indices = np.minimum(indices // chunk_size, len(chunk_data) - 1)

            def gather_chunk_fields(field_names):
                stacked = np.stack([chunk_data[name] for name in field_names], axis=1)
                return stacked[chunk_indices].astype(np.float32)

            pos_min = gather_chunk_fields(["min_x", "min_y", "min_z"])
            pos_max = gather_chunk_fields(["max_x", "max_y", "max_z"])
            scale_min = gather_chunk_fields(["min_scale_x", "min_scale_y", "min_scale_z"])
            scale_max = gather_chunk_fields(["max_scale_x", "max_scale_y", "max_scale_z"])
            color_min = gather_chunk_fields(["min_r", "min_g", "min_b"])
            color_max = gather_chunk_fields(["max_r", "max_g", "max_b"])

            mask_11 = np.uint32((1 << 11) - 1)
            mask_10 = np.uint32((1 << 10) - 1)
            inv_11 = 1.0 / float((1 << 11) - 1)
            inv_10 = 1.0 / float((1 << 10) - 1)

            packed_pos = vertex_data['packed_position'].astype(np.uint32)
            pos_norm = np.stack([
                ((packed_pos >> 21) & mask_11).astype(np.float32) * inv_11,
                ((packed_pos >> 11) & mask_10).astype(np.float32) * inv_10,
                (packed_pos & mask_11).astype(np.float32) * inv_11
            ], axis=1)
            xyz = pos_min + (pos_max - pos_min) * pos_norm

            packed_scale = vertex_data['packed_scale'].astype(np.uint32)
            scale_norm = np.stack([
                ((packed_scale >> 21) & mask_11).astype(np.float32) * inv_11,
                ((packed_scale >> 11) & mask_10).astype(np.float32) * inv_10,
                (packed_scale & mask_11).astype(np.float32) * inv_11
            ], axis=1)
            log_scales = scale_min + (scale_max - scale_min) * scale_norm

            packed_color = vertex_data['packed_color'].astype(np.uint32)
            r = ((packed_color >> 24) & 0xFF).astype(np.float32)
            g = ((packed_color >> 16) & 0xFF).astype(np.float32)
            b = ((packed_color >> 8) & 0xFF).astype(np.float32)
            a = (packed_color & 0xFF).astype(np.float32)
            color_norm = np.stack((r, g, b), axis=1) / 255.0
            colors = color_min + (color_max - color_min) * color_norm
            alpha = np.clip(a / 255.0, 1e-6, 1.0 - 1e-6)

            packed_rot = vertex_data['packed_rotation'].astype(np.uint32)
            sqrt2 = math.sqrt(2.0)
            r_comp = (((packed_rot >> 20) & mask_10).astype(np.float32) * inv_10 - 0.5) * sqrt2
            i_comp = (((packed_rot >> 10) & mask_10).astype(np.float32) * inv_10 - 0.5) * sqrt2
            o_comp = ((packed_rot & mask_10).astype(np.float32) * inv_10 - 0.5) * sqrt2
            sum_sq = r_comp * r_comp + i_comp * i_comp + o_comp * o_comp
            sum_sq = np.clip(sum_sq, 0.0, 1.0 - 1e-8)
            n_comp = np.sqrt(1.0 - sum_sq)
            variant = (packed_rot >> 30).astype(np.int32)

            x = np.empty(vert_count, dtype=np.float32)
            y = np.empty(vert_count, dtype=np.float32)
            z = np.empty(vert_count, dtype=np.float32)
            w = np.empty(vert_count, dtype=np.float32)

            mask0 = variant == 0
            x[mask0] = n_comp[mask0]
            y[mask0] = r_comp[mask0]
            z[mask0] = i_comp[mask0]
            w[mask0] = o_comp[mask0]

            mask1 = variant == 1
            x[mask1] = r_comp[mask1]
            y[mask1] = n_comp[mask1]
            z[mask1] = i_comp[mask1]
            w[mask1] = o_comp[mask1]

            mask2 = variant == 2
            x[mask2] = r_comp[mask2]
            y[mask2] = i_comp[mask2]
            z[mask2] = n_comp[mask2]
            w[mask2] = o_comp[mask2]

            mask3 = variant == 3
            x[mask3] = r_comp[mask3]
            y[mask3] = i_comp[mask3]
            z[mask3] = o_comp[mask3]
            w[mask3] = n_comp[mask3]

            x_new = z.copy()
            y_new = y.copy()
            z_new = x.copy()
            w_new = w.copy()
            norm = np.sqrt(x_new * x_new + y_new * y_new + z_new * z_new + w_new * w_new)
            norm[norm == 0.0] = 1.0
            x_new /= norm
            y_new /= norm
            z_new /= norm
            w_new /= norm
            rotations = np.stack((w_new, x_new, y_new, z_new), axis=1)

            rest_dim = (self.max_sh_degree + 1) ** 2 - 1
            if rest_dim > 0:
                try:
                    sh_element = plydata['sh']
                    sh_names = sh_element.data.dtype.names
                    sh_array = np.column_stack([sh_element.data[name] for name in sh_names]).astype(np.float32)
                    expected = rest_dim * 3
                    if sh_array.shape[1] != expected:
                        raise ValueError(f"Unexpected SH coefficient count: {sh_array.shape[1]} vs {expected}")
                    features_extra = sh_array / 255.0 * 2.0 - 1.0
                    features_extra = features_extra.reshape(vert_count, 3, rest_dim)
                except KeyError:
                    features_extra = np.zeros((vert_count, 3, rest_dim), dtype=np.float32)
            else:
                features_extra = np.zeros((vert_count, 3, 0), dtype=np.float32)

            xyz_tensor = torch.tensor(xyz, dtype=torch.float, device="cuda")
            log_scale_tensor = torch.tensor(log_scales, dtype=torch.float, device="cuda")
            rotation_tensor = torch.tensor(rotations, dtype=torch.float, device="cuda")
            alpha_tensor = torch.tensor(alpha.reshape(-1, 1), dtype=torch.float, device="cuda")
            alpha_tensor = torch.clamp(alpha_tensor, 1e-6, 1.0 - 1e-6)
            opacity_tensor = self.inverse_opacity_activation(alpha_tensor)

            colors_tensor = torch.tensor(colors, dtype=torch.float, device="cuda")
            # colors_tensor = torch.clamp(colors_tensor, 0.0, 1.0)
            features_dc = RGB2SH(colors_tensor).unsqueeze(-1)

            if features_extra.shape[2] > 0:
                features_rest_tensor = torch.tensor(features_extra, dtype=torch.float, device="cuda")
            else:
                features_rest_tensor = torch.zeros((vert_count, 3, features_extra.shape[2]), dtype=torch.float, device="cuda")

            self._xyz = nn.Parameter(xyz_tensor.requires_grad_(True))
            self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features_rest_tensor.transpose(1, 2).contiguous().requires_grad_(True))
            self._opacity = nn.Parameter(opacity_tensor.requires_grad_(True))
            self._scaling = nn.Parameter(log_scale_tensor.requires_grad_(True))
            self._rotation = nn.Parameter(rotation_tensor.requires_grad_(True))

        elif {"x", "y", "z"}.issubset(vertex_props):
            xyz = np.stack((vertex_data['x'], vertex_data['y'], vertex_data['z']), axis=1).astype(np.float32)
            opacities = vertex_data['opacity'].astype(np.float32)[..., np.newaxis]

            features_dc = np.stack([
                vertex_data['f_dc_0'],
                vertex_data['f_dc_1'],
                vertex_data['f_dc_2']
            ], axis=1).astype(np.float32)[..., np.newaxis]

            rest_dim = (self.max_sh_degree + 1) ** 2 - 1
            extra_f_names = [p.name for p in vertex_element.properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda name: int(name.split('_')[-1]))
            if extra_f_names:
                features_extra = np.stack([vertex_data[name] for name in extra_f_names], axis=1).astype(np.float32)
                features_extra = features_extra.reshape(features_extra.shape[0], 3, rest_dim)
            else:
                features_extra = np.zeros((xyz.shape[0], 3, rest_dim), dtype=np.float32)

            scale_names = sorted([p.name for p in vertex_element.properties if p.name.startswith("scale_")], key=lambda name: int(name.split('_')[-1]))
            scales = np.stack([vertex_data[name] for name in scale_names], axis=1).astype(np.float32)

            rot_names = sorted([p.name for p in vertex_element.properties if p.name.startswith("rot_")], key=lambda name: int(name.split('_')[-1]))
            rots = np.stack([vertex_data[name] for name in rot_names], axis=1).astype(np.float32)

            self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
            self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        else:
            raise ValueError("Unsupported PLY vertex format")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
