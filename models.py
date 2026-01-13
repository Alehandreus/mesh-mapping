import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn


class DisplacementModel(nn.Module):
    def __init__(self, cfg_model, mesh):
        super().__init__()

        mesh_min, mesh_max = mesh.get_bounds()
        self.mesh_min = nn.Parameter(torch.tensor(mesh_min, dtype=torch.float32), requires_grad=False)
        self.mesh_max = nn.Parameter(torch.tensor(mesh_max, dtype=torch.float32), requires_grad=False)

        self.encoding_config = cfg_model.encoding_config
        self.network_config = cfg_model.network_config

        self.n_input_dims = 3
        self.n_output_dims = 3

        self.network = tcnn.NetworkWithInputEncoding(
            n_input_dims=self.n_input_dims,
            n_output_dims=self.n_output_dims,
            encoding_config=self.encoding_config,
            network_config=self.network_config,
        )

    def forward(self, x, **kwargs):
        x = (x - self.mesh_min) / (self.mesh_max - self.mesh_min)
        delta = self.network(x).float()
        y = x + delta
        y = y * (self.mesh_max - self.mesh_min) + self.mesh_min
        return y
