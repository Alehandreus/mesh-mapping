import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
from NeuRBF.img_sdf.network import rbf
from NeuRBF.img_sdf.provider import SDFDataset
from NeuRBF import util_network
from NeuRBF.configs import sdf
import trimesh
import gc


class RBFencoding(rbf):
    def __init__(self, cmin, cmax, s_dims, in_dim=2, out_dim=3, 
                num_layers=3, skips=[], hidden_dim=64, n_hidden_fl=20,
                num_levels_ref=16, level_dim_ref=2, base_resolution_ref=16, log2_hashmap_size_ref=24,
                num_levels=2, level_dim=2, base_resolution=16, log2_hashmap_size=24, desired_resolution=2048,
                rbf_type='nlin_s', n_kernel=64, point_nn_kernel=4, ks_alpha=1, 
                lc_init=[3e-1], lcb_init=None,
                w_init=None, b_init=None, a_init=None,
                sparse_embd_grad=True, act='relu', lc_act=None, 
                rbf_suffixes=None, kc_init_config=None, 
                rbf_lc0_normalize=True, pe_freqs=[], pe_lc0_freq=None, pe_hg0_freq=None,
                pe_lc0_rbf_freq=None, pe_lc0_rbf_keep=None, **kwargs):
        
        super().__init__(cmin, cmax, s_dims, in_dim, out_dim, 
                num_layers, skips, hidden_dim, n_hidden_fl,
                num_levels_ref, level_dim_ref, base_resolution_ref, log2_hashmap_size_ref,
                num_levels, level_dim, base_resolution, log2_hashmap_size, desired_resolution,
                rbf_type, n_kernel, point_nn_kernel, ks_alpha, 
                lc_init, lcb_init,
                w_init, b_init, a_init,
                sparse_embd_grad, act, lc_act, 
                rbf_suffixes, kc_init_config, 
                rbf_lc0_normalize, pe_freqs, pe_lc0_freq, pe_hg0_freq,
                pe_lc0_rbf_freq, pe_lc0_rbf_keep, **kwargs)
        
        self.n_output_dims = n_hidden_fl
        self.use_train_knn = False
    
    def forward(self, x_g, point_idx=None, **kwargs):
        suffix = '0'
        point_idx = torch.zeros(x_g.shape, device=x_g.device)
        if self.point_nn_kernel <= 0 or point_idx is None:  # Use all kernels for each point
            rbf_out = self.forward_rbf(x_g, None, suffix)  # [p nk]
            out = rbf_out @ self.lc0.weight  # [p hfl]
        else:
            kernel_idx = self.forward_kernel_idx(x_g, point_idx, suffix)
            rbf_out = self.forward_rbf(x_g, kernel_idx, suffix)  # [p k_topk]
            if self.rbf_lc0_normalize:
                rbf_out = rbf_out / (rbf_out.detach().sum(-1, keepdim=True) + 1e-8)

            out = self.lc0(kernel_idx)  # [p k_topk d_lc0]
            rbf_out = rbf_out[..., None]  # [p k_topk 1]
            if len(self.pe_lc0_rbf_freq) >= 2 and self.pe_lc0_rbf_keep < out.shape[-1]:
                if self.pe_lc0_rbf_keep > 0:
                    rbf_out = torch.cat([rbf_out.expand(-1, -1, self.pe_lc0_rbf_keep), 
                        torch.sin(rbf_out * self.pe_lc0_rbf_freqs[None, None])], -1)  # [p k_topk d_lc0]
                else:
                    rbf_out = torch.sin(rbf_out * self.pe_lc0_rbf_freqs[None, None])  # [p k_topk d_lc0]
            out = (out * rbf_out).sum(1)  # [p d_lc0]

        if self.num_levels > 0:
            out_hg = self.hg0(x_g / self.cmax_gpu.flip(-1)[None])  # [p d_hg0]
        else:
            out_hg = None
            
        if out_hg is not None:
            out = torch.cat([out_hg, out], -1)
        out = out + self.lcb0[None]

        h = out
        if self.lc_act == 0:
            pass
        elif self.lc_act == 1:
            h = F.relu(h, inplace=True)
        elif self.lc_act == 2:
            h = util_network.scaledsin_activation(h, self.a0[None])
        else:
            raise NotImplementedError

        if self.pe_x is not None:
            h = torch.cat((h, self.pe_x(x_g)), dim=-1)

        return h

class RayModel(nn.Module):
    def __init__(self, model_config, mesh, cfg):
        super().__init__()

        self.model_config = model_config

        mesh_min, mesh_max = mesh.get_bounds()
        self.mesh_min = nn.Parameter(torch.tensor(mesh_min, dtype=torch.float32), requires_grad=False)
        self.mesh_max = nn.Parameter(torch.tensor(mesh_max, dtype=torch.float32), requires_grad=False)

        self.network_config = model_config.network_config
        self.point_encoding_config = model_config.point_encoding_config
        self.uv_encoding_config = model_config.uv_encoding_config
        self.direction_encoding_config = model_config.direction_encoding_config

        # output dimentions of mlp head: presence of intersection (1), distance (1), normal (3) and color (3)
        self.n_output_dims = 8

        self.direction_encoding = tcnn.Encoding(3, self.direction_encoding_config)

        if model_config.encoding == "HashGrid":
            self.point_encoding = tcnn.Encoding(3, self.point_encoding_config)
        elif model_config.encoding == "RBF":
            opt = sdf.config
            gt_mesh = trimesh.load(cfg.fine_mesh_path, force='mesh')
            print(f"[INFO] mesh: {gt_mesh.vertices.shape} {gt_mesh.faces.shape} {cfg.fine_mesh_path}")

            opt.cmax = [1, 1, 1]  # ... y x
            opt.cmin = [-i for i in opt.cmax]  # ... y x
            s_dims = [opt.val_resolution] * 3
            n_train_point = opt.train_num_samples * opt.train_epoch_size
            desired_resolution=opt.desired_resolution
            in_dim = 3
            out_dim = 1
            opt.vmin = -(2**2*3)**0.5
            opt.vmax = (2**2*3)**0.5

            # Parameter of hashgrid part in RBF (in tiny-cuda-nn I didn't find similar)
            desired_resolution = 2048

            s_dims = [opt.val_resolution] * 3

            #out_dim is garbage here
            self.point_encoding = RBFencoding(opt.cmin, opt.cmax, s_dims, in_dim=3, out_dim=8, 
            num_layers=opt.num_layers, hidden_dim=opt.hidden_dim, n_hidden_fl=opt.n_hidden_fl, 
            num_levels_ref=opt.num_levels_ref, level_dim_ref=opt.level_dim_ref, 
            base_resolution_ref=opt.base_resolution_ref, log2_hashmap_size_ref=opt.log2_hashmap_size_ref, 
            num_levels=opt.num_levels, level_dim=opt.level_dim, base_resolution=opt.base_resolution,
            log2_hashmap_size=opt.log2_hashmap_size, desired_resolution=desired_resolution, 
            rbf_type=opt.rbf_type, n_kernel=opt.n_kernel, point_nn_kernel=opt.point_nn_kernel, ks_alpha=opt.ks_alpha, 
            lc_init=opt.lc_init, lcb_init=opt.lcb_init, 
            w_init=opt.w_init, b_init=opt.b_init, a_init=opt.a_init,
            sparse_embd_grad=False, act=opt.act, lc_act=opt.lc_act, rbf_suffixes=opt.rbf_suffixes, 
            kc_init_config=opt.kc_init_config, rbf_lc0_normalize=opt.rbf_lc0_normalize, 
            pe_freqs=opt.pe_freqs, pe_lc0_freq=opt.pe_lc0_freq, pe_hg0_freq=opt.pe_hg0_freq,
            pe_lc0_rbf_freq=opt.pe_lc0_rbf_freq, pe_lc0_rbf_keep=opt.pe_lc0_rbf_keep)
            
            train_dataset = SDFDataset(gt_mesh, opt.cmin, opt.cmax, s_dims, num_samples=opt.train_num_samples, 
            size=opt.train_epoch_size, presample=opt.train_presample, shuffle_mode=opt.train_shuffle_mode, 
            clip_sdf=opt.clip_sdf, mesh_fp=cfg.fine_mesh_path, device=cfg.device)

            util_network.init_rbf_params(self.point_encoding, train_dataset, opt.kc_init_config, opt.kw_init_config, device=0)
            if hasattr(train_dataset, 'points'):
                self.point_encoding.update_point_kernel_idx(train_dataset.points.view(-1, train_dataset.points.shape[-1]), device=cfg.device)
            util_network.fix_params(self.point_encoding, opt.fix_params)

        if model_config.encoding_type == "3d":
            self.mlp_input_dims = self.point_encoding.n_output_dims * 2 + self.direction_encoding.n_output_dims
        elif model_config.encoding_type == "3d+1":
            self.mlp_input_dims = self.point_encoding.n_output_dims * 3 + self.direction_encoding.n_output_dims
        
        self.network = tcnn.Network(self.mlp_input_dims, self.n_output_dims, self.network_config)
        self.omega_0 = 1
        with torch.no_grad():
            print(self.network.params.data.shape)
            self.network.params.data[:30720].uniform_(-1 / self.mlp_input_dims, 
                                             1 / self.mlp_input_dims)
            self.network.params.data[30720:].uniform_(-(6 / self.mlp_input_dims) ** 0.5 / self.omega_0, 
                                             (6 / self.mlp_input_dims) ** 0.5 / self.omega_0)

    def forward(self, points, points_inner, directions, *args, **kwargs):
        directions = (directions + 1) / 2
        directions_enc = self.direction_encoding(directions).float()

        if self.model_config.encoding_type == "3d":
            points = (points - self.mesh_min) / (self.mesh_max - self.mesh_min)
            points_enc = self.point_encoding(points).float()

            points_inner = (points_inner - self.mesh_min) / (self.mesh_max - self.mesh_min)
            points_inner_enc = self.point_encoding(points_inner).float()

            x = torch.cat([points_enc, points_inner_enc, directions_enc], dim=1)

        elif self.model_config.encoding_type == "3d+1":
            points = (points - self.mesh_min) / (self.mesh_max - self.mesh_min)
            points_enc = self.point_encoding(points).float()

            points_inner = (points_inner - self.mesh_min) / (self.mesh_max - self.mesh_min)
            points_inner_enc = self.point_encoding(points_inner).float()

            points_interp = (points + points_inner) / 2
            points_interp_enc = self.point_encoding(points_interp).float()

            x = torch.cat([points_enc, points_inner_enc, points_interp_enc, directions_enc], dim=1)    

        y = self.network(x).float()

        # skip_mask = torch.norm(points - points_inner, dim=1) < 1e-3

        has_intersection = y[:, 0]
        distance = y[:, 1]
        normal = y[:, 2:5]
        colors = y[:, 5:8]

        # has_intersection[skip_mask] = -1.0

        normalized_normal = torch.zeros(normal.shape, dtype=normal.dtype, device=normal.device)
        if (normal.norm(dim=1) > 1e-8).any():
            normalized_normal[normal.norm(dim=1) > 1e-8] = (normal / normal.norm(dim=1, keepdim=True))[normal.norm(dim=1) > 1e-8]
        
        return has_intersection, distance, normalized_normal, colors
