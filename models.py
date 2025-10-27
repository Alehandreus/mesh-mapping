import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn


class ResidualMap(nn.Module):
    def __init__(self, mesh):
        super().__init__()

        mesh_min, mesh_max = mesh.get_bounds()
        self.mesh_min = nn.Parameter(torch.tensor(mesh_min, dtype=torch.float32), requires_grad=False)
        self.mesh_max = nn.Parameter(torch.tensor(mesh_max, dtype=torch.float32), requires_grad=False)

        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": 8,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 2,
            "per_level_scale": 2,
            "fixed_point_pos": False,
        }
        self.network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }

        self.n_input_dims = 3
        self.encoding = TensoRFEncoder(mesh)
        # self.encoding = tcnn.Encoding(self.n_input_dims, self.encoding_config)
        self.n_encoder_dims = self.encoding.n_output_dims    
        self.n_output_dims = 3

        self.network = tcnn.Network(self.n_encoder_dims, self.n_output_dims, self.network_config)

    def forward(self, x):
        x = (x - self.mesh_min) / (self.mesh_max - self.mesh_min)
        x_enc = self.encoding(x).float()
        delta = self.network(x_enc)
        y = x + delta
        y = y * (self.mesh_max - self.mesh_min) + self.mesh_min
        return y


class Critic(nn.Module):
    def __init__(self, mesh):
        super().__init__()

        mesh_min, mesh_max = mesh.get_bounds()
        self.mesh_min = nn.Parameter(torch.tensor(mesh_min, dtype=torch.float32), requires_grad=False)
        self.mesh_max = nn.Parameter(torch.tensor(mesh_max, dtype=torch.float32), requires_grad=False)

        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": 8,
            "n_features_per_level": 2,
            "log2_hashmap_size": 18,
            "base_resolution": 2,
            "per_level_scale": 2,
            "fixed_point_pos": False,
        }
        self.network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 4,
        }

        self.n_input_dims = 3
        self.encoding = tcnn.Encoding(self.n_input_dims, self.encoding_config)
        self.n_encoder_dims = self.encoding.n_output_dims
        self.n_output_dims = 1

        self.network = tcnn.Network(self.n_encoder_dims, self.n_output_dims, self.network_config)        

    def forward(self, x):
        x = (x - self.mesh_min) / (self.mesh_max - self.mesh_min)
        x_enc = self.encoding(x).float()
        out = self.network(x_enc).squeeze(-1)
        return out



class TensoRFEncoder(nn.Module):
    def __init__(self, mesh):
        super().__init__()

        try:
            from TensoRF.models.tensoRF import TensorVM
        except ImportError:
            raise ImportError("Please clone TensoRF repository")

        self.encoder = TensorVM(
            # aabb=torch.tensor(mesh.get_bounds(), dtype=torch.float32, device='cuda'),
            aabb = torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ], dtype=torch.float32, device='cuda'),
            device='cuda',

            gridSize=[128] * 3,
            appearance_n_comp=4,
            density_n_comp=1,
        )

        self.n_output_dims = 3 * self.encoder.app_n_comp

    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((
            xyz_sampled[..., self.encoder.matMode[0]],
            xyz_sampled[..., self.encoder.matMode[1]],
            xyz_sampled[..., self.encoder.matMode[2]],
        )).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((
            xyz_sampled[..., self.encoder.vecMode[0]],
            xyz_sampled[..., self.encoder.vecMode[1]],
            xyz_sampled[..., self.encoder.vecMode[2]],
        ))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.encoder.plane_coef[:, :self.encoder.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.encoder.app_n_comp, -1)
        line_feats = F.grid_sample(self.encoder.line_coef[:, :self.encoder.app_n_comp], coordinate_line, align_corners=True).view(3 * self.encoder.app_n_comp, -1)

        return (plane_feats * line_feats).T

    def forward(self, x):
        return self.compute_appfeature(x)
        # a = self.encoder.compute_appfeature(x)
        return a
        # print(a.shape)
        # exit()