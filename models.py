import torch
import torch.nn as nn

import tinycudann as tcnn


class RayModel(nn.Module):
    def __init__(self, model_config, mesh):
        super().__init__()

        self.model_config = model_config

        mesh_min, mesh_max = mesh.get_bounds()
        self.mesh_min = nn.Parameter(torch.tensor(mesh_min, dtype=torch.float32), requires_grad=False)
        self.mesh_max = nn.Parameter(torch.tensor(mesh_max, dtype=torch.float32), requires_grad=False)

        self.network_config = model_config.network_config
        self.point_encoding_config = model_config.point_encoding_config
        self.direction_encoding_config = model_config.direction_encoding_config

        # output dimentions of mlp head: presence of intersection (1), distance (1), normal (3) and color (3)
        self.n_output_dims = 8

        self.direction_encoding = tcnn.Encoding(3, self.direction_encoding_config)   

        if model_config.encoding_type == "3d":
            self.point_encoding = tcnn.Encoding(3, self.point_encoding_config)
            self.mlp_input_dims = self.point_encoding.n_output_dims * 2 + self.direction_encoding.n_output_dims
        elif model_config.encoding_type == "3d+1":
            self.point_encoding = tcnn.Encoding(3, self.point_encoding_config)
            self.mlp_input_dims = self.point_encoding.n_output_dims * 3 + self.direction_encoding.n_output_dims

        self.network = tcnn.Network(self.mlp_input_dims, self.n_output_dims, self.network_config)

    def forward(self, points, points_inner, directions, return_emb=False, inject_emb=None, *args, **kwargs):
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
        has_intersection = y[:, 0]
        distance = y[:, 1]
        normal = y[:, 2:5]
        colors = y[:, 5:8]

        return has_intersection, distance, normal, colors
