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
        self.uv_encoding_config = model_config.uv_encoding_config
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
        
        # self.network = tcnn.Network(self.mlp_input_dims, self.n_output_dims, self.network_config)
        self.network1 = nn.Sequential(
            nn.Linear(self.mlp_input_dims, self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),            
        )
        self.network2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.n_output_dims)
        )

        self.emb_encoder = nn.Sequential(
            tcnn.Encoding(3, {
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 4,
                "log2_hashmap_size": 22,
                "base_resolution": 16,
                "per_level_scale": 2,
                "fixed_point_pos": False,
            }),
        )
        self.emb_model1 = nn.Sequential(
            nn.Linear(8 * 4, self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),
        )
        self.emb_model2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], self.network_config["n_neurons"]),
            nn.LeakyReLU(),
            nn.Linear(self.network_config["n_neurons"], 3 + 3 + 3)
        )

    def forward_embedder(self, points):
        points_enc = self.point_encoding(points).float()
        emb = self.emb_model1(points_enc)
        out = self.emb_model2(emb)
        pred_point = out[:, :3]
        pred_normal = out[:, 3:6]
        pred_color = out[:, 6:9]
        return pred_point, pred_normal, pred_color, emb

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

        # y = self.network(x).float()

        emb = self.network1(x.float())
        if inject_emb is not None:
            emb = inject_emb
        y = self.network2(emb)

        has_intersection = y[:, 0]
        distance = y[:, 1]
        normal = y[:, 2:5]
        colors = y[:, 5:8]

        normalized_normal = torch.zeros(normal.shape, dtype=normal.dtype, device=normal.device)
        if (normal.norm(dim=1) > 1e-8).any():
            normalized_normal[normal.norm(dim=1) > 1e-8] = (normal / normal.norm(dim=1, keepdim=True))[normal.norm(dim=1) > 1e-8]

        if return_emb:        
            return has_intersection, distance, normalized_normal, colors, emb

        return has_intersection, distance, normalized_normal, colors
