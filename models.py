import torch
import torch.nn as nn

import tinycudann as tcnn


class RayModel(nn.Module):
    def __init__(self, model_config, mesh):
        super().__init__()

        mesh_min, mesh_max = mesh.get_bounds()
        self.mesh_min = nn.Parameter(torch.tensor(mesh_min, dtype=torch.float32), requires_grad=False)
        self.mesh_max = nn.Parameter(torch.tensor(mesh_max, dtype=torch.float32), requires_grad=False)

        self.network_config = model_config.network_config
        self.point_encoding_config = model_config.point_encoding_config
        #self.uv_encoding_config = model_config.uv_encoding_config
        self.direction_encoding_config = model_config.direction_encoding_config

        # input dimentions for every encoder
        self.n_input_dims = 3

        # output dimentions of mlp head: presence of intersection (1), distance (1) and normal (3)
        self.n_output_dims = 5

        self.point_encoding = tcnn.Encoding(self.n_input_dims, self.point_encoding_config)
        #self.uv_encoding = tcnn.Encoding(self.n_input_dims - 1, self.uv_encoding_config)
        self.direction_encoding = tcnn.Encoding(self.n_input_dims, self.direction_encoding_config)

        self.mlp_input_dims = self.point_encoding.n_output_dims + self.direction_encoding.n_output_dims# + self.uv_encoding.n_output_dims
        self.network = tcnn.Network(self.mlp_input_dims, self.n_output_dims, self.network_config)


    def forward(self, points, directions, **kwargs):
        points = (points - self.mesh_min) / (self.mesh_max - self.mesh_min)
        points_enc = self.point_encoding(points).float()

        #u = torch.arccos(points[:, 2] / points.norm(dim=1))
        #v = torch.arctan(points[:, 1] / points[:, 0])
        #uv_enc = self.uv_encoding(torch.stack([u, v], dim=1)).float()

        directions = (directions + 1) / 2
        directions_enc = self.direction_encoding(directions).float()
       
        x = torch.cat([points_enc, directions_enc], dim=1)
        y = self.network(x)

        has_intersection = y[:, 0]
        distance = y[:, 1]
        normal = y[:, 2:]

        normalized_normal = torch.zeros(normal.shape, dtype=normal.dtype, device=normal.device)
        normalized_normal[normal.norm(dim=1) > 0] = normal / normal.norm(dim=1)[:, None]
        
        return has_intersection, distance, normalized_normal
