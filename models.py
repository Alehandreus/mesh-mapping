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
            "n_hidden_layers": 2,
        }

        self.n_input_dims = 3
        self.n_output_dims = 3

        self.encoding = tcnn.Encoding(self.n_input_dims, self.encoding_config)
        self.network = tcnn.Network(self.encoding.n_output_dims, self.n_output_dims, self.network_config)

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
        self.n_output_dims = 1

        self.encoding = tcnn.Encoding(self.n_input_dims, self.encoding_config)
        self.network = tcnn.Network(self.encoding.n_output_dims, self.n_output_dims, self.network_config)        

    def forward(self, x):
        x = (x - self.mesh_min) / (self.mesh_max - self.mesh_min)
        x_enc = self.encoding(x).float()
        out = self.network(x_enc).squeeze(-1)
        return out
