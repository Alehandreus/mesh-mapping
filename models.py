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
            "log2_hashmap_size": 8,
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
        self.encoding = VertexEncoder(mesh)
        # self.encoding = TensoRFEncoder(mesh)
        # self.encoding = tcnn.Encoding(self.n_input_dims, self.encoding_config)
        # self.encoding = HashGridEncoding(self.n_input_dims, self.encoding_config)
        self.n_encoder_dims = self.encoding.n_output_dims
        # self.n_encoder_dims = self.n_input_dims
        self.n_output_dims = 3

        self.network = tcnn.Network(self.n_encoder_dims, self.n_output_dims, self.network_config)
        # self.network = tcnn.Network(3, self.n_output_dims, self.network_config)

    def forward(self, x, **kwargs):
        x = (x - self.mesh_min) / (self.mesh_max - self.mesh_min)
        x_enc = self.encoding(x, **kwargs).float()
        delta = self.network(x_enc).float()
        y = x + delta
        y = y * (self.mesh_max - self.mesh_min) + self.mesh_min

        # x = (x - self.mesh_min) / (self.mesh_max - self.mesh_min)
        # delta = self.network(x).float()
        # y = x + delta
        # y = y * (self.mesh_max - self.mesh_min) + self.mesh_min

        return y


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

    def forward(self, x, **kwargs):
        return self.compute_appfeature(x)
    

class VertexEncoder(nn.Module):
    def __init__(self, mesh):
        super().__init__()

        self.emb_size = 16
        self.n_output_dims = self.emb_size + 3

        self.n_vertices = mesh.get_num_vertices()
        self.embeddings = nn.Embedding(self.n_vertices, self.emb_size)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.1)

        self.faces = mesh.get_faces()
        self.faces = nn.Parameter(torch.from_numpy(self.faces).long(), requires_grad=False)

    def forward(self, x, face_idxs, barycentrics):
        v0_idxs = self.faces[face_idxs, 0]
        v1_idxs = self.faces[face_idxs, 1]
        v2_idxs = self.faces[face_idxs, 2]

        v0_emb = self.embeddings(v0_idxs)
        v1_emb = self.embeddings(v1_idxs)
        v2_emb = self.embeddings(v2_idxs)

        emb = (barycentrics[:, 0, None] * v0_emb +
               barycentrics[:, 1, None] * v1_emb +
               barycentrics[:, 2, None] * v2_emb)
        
        emb = torch.concatenate([
            emb,
            x,
        ], dim=-1)
        
        return emb
    

class HashGridEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoding = tcnn.Encoding(*args, **kwargs)
        self.n_output_dims = self.encoding.n_output_dims
    
    def forward(self, x, **kwargs):
        return self.encoding(x)
