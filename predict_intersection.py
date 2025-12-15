import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import tqdm
import math

import tinycudann as tcnn

from mesh_utils import Mesh, CPUBuilder, GPUTraverser
from mesh_utils import GPURayTracer, GPURayTracerAll

import cv2


# ==== Load and prepare BVH ==== #

fine_mesh = Mesh.from_file("/home/me/brain/mesh-mapping/models/monkey_fine.fbx")
rough_mesh = Mesh.from_file("/home/me/brain/mesh-mapping/models/monkey_rough.fbx")

img_size = 800
n_pixels = img_size * img_size
DEVICE = torch.device('cuda:0')


mesh_min, mesh_max = rough_mesh.get_bounds()
mesh_min = torch.from_numpy(mesh_min).float().to(DEVICE)
mesh_max = torch.from_numpy(mesh_max).float().to(DEVICE)


def intersect_update(orig, dir, center, radius, forward=False):
    o, d, c = orig, dir, center
    r = torch.as_tensor(radius, dtype=o.dtype, device=o.device)
    m = o - c
    a = (d*d).sum(-1); b = 2*(m*d).sum(-1); cterm = (m*m).sum(-1) - r*r
    disc = b*b - 4*a*cterm
    hit = (a>0) & (disc>=0)
    t = torch.full_like(a, float('nan'))
    sd = torch.zeros_like(a); sd[hit] = torch.sqrt(disc[hit])
    t1 = torch.empty_like(a); t2 = torch.empty_like(a)
    t1[hit] = (-b[hit]-sd[hit])/(2*a[hit]); t2[hit] = (-b[hit]+sd[hit])/(2*a[hit])
    if forward:
        t[hit] = torch.where((t1>0)&(t2>0), torch.minimum(t1,t2),
                             torch.where(t1>0, t1, torch.where(t2>0, t2, torch.maximum(t1,t2))))
    else:
        t[hit] = torch.where(torch.abs(t1)<=torch.abs(t2), t1, t2)
    new_o = o.clone(); new_o[hit] = o[hit] + t[hit, None] * d[hit]
    return new_o, t, hit


def rays_to_plucker4(origins, dirs, eps=1e-9):
    d = dirs / torch.clamp(dirs.norm(dim=-1, keepdim=True), min=eps)  # unit dir
    theta = torch.atan2(d[:,1], d[:,0])                               # azimuth
    phi = torch.atan2(d[:,2], torch.clamp(torch.sqrt(d[:,0]**2 + d[:,1]**2), min=eps))  # elevation
    e1 = torch.tensor([1.,0.,0.], device=d.device, dtype=d.dtype).expand_as(d)
    e2 = torch.tensor([0.,1.,0.], device=d.device, dtype=d.dtype).expand_as(d)
    c1, c2 = torch.cross(d, e1), torch.cross(d, e2)
    u = torch.where((c1.norm(dim=-1) >= c2.norm(dim=-1)).unsqueeze(-1), c1, c2)
    u = u / torch.clamp(u.norm(dim=-1, keepdim=True), min=eps)
    v = torch.cross(d, u)
    m = torch.cross(origins, d)
    mu, mv = (m*u).sum(-1), (m*v).sum(-1)
    return torch.stack([theta, phi, mu, mv], dim=-1)


def rays_to_unit_plucker4(orig, dirs, eps=1e-9):
    d = dirs / torch.clamp(dirs.norm(dim=-1, keepdim=True), min=eps)         # unit dir
    theta = torch.atan2(d[:,1], d[:,0]); phi = torch.atan2(d[:,2], torch.clamp(torch.sqrt(d[:,0]**2+d[:,1]**2), min=eps))
    t01 = (theta + torch.pi) / (2*torch.pi); p01 = (phi + 0.5*torch.pi) / torch.pi
    e1 = torch.tensor([1.,0.,0.], device=d.device, dtype=d.dtype).expand_as(d)
    e2 = torch.tensor([0.,1.,0.], device=d.device, dtype=d.dtype).expand_as(d)
    c1, c2 = torch.cross(d, e1), torch.cross(d, e2)
    u = torch.where((c1.norm(dim=-1) >= c2.norm(dim=-1)).unsqueeze(-1), c1, c2)
    u = u / torch.clamp(u.norm(dim=-1, keepdim=True), min=eps); v = torch.cross(d, u)
    m = torch.cross(orig, d); R = torch.clamp(orig.norm(dim=-1, keepdim=True), min=eps)
    mu01 = 0.5*((m*u).sum(-1, keepdim=True)/R + 1); mv01 = 0.5*((m*v).sum(-1, keepdim=True)/R + 1)
    return torch.cat([t01[:,None], p01[:,None], mu01, mv01], dim=-1)  # [N,4] in [0,1]


def sobel_filter(img):
    img = (img * 255).astype(np.uint8)
    edges = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
    edges = (edges > 0).astype(np.float32)
    return edges


class MyDataset:
    def __init__(self, fine_mesh, rough_mesh, batch_size, n_classes):
        self.n_classes = n_classes

        self.fine_mesh = fine_mesh
        fine_builder = CPUBuilder(fine_mesh)
        fine_bvh_data = fine_builder.build_bvh(5)
        self.fine_tracer = GPURayTracer(fine_bvh_data)

        self.rough_mesh = rough_mesh
        rough_builder = CPUBuilder(rough_mesh)
        rough_bvh_data = rough_builder.build_bvh(5)
        self.rough_tracer = GPURayTracer(rough_bvh_data)
        self.rough_tracer_all = GPURayTracerAll(rough_bvh_data)

        self.batch_size = batch_size

        mesh_min, mesh_max = rough_mesh.get_bounds()
        self.mesh_min = torch.from_numpy(mesh_min).float().to(DEVICE)
        self.mesh_max = torch.from_numpy(mesh_max).float().to(DEVICE)
        self.mesh_center = (self.mesh_min + self.mesh_max) * 0.5
        self.mesh_radius = torch.norm(self.mesh_max - self.mesh_min) * 0.5

        self.depths = torch.zeros(batch_size, dtype=torch.int, device=DEVICE)
        self.bbox_idxs = torch.zeros(batch_size, dtype=torch.uint32, device=DEVICE)
        self.history = torch.zeros(batch_size, 64, dtype=torch.uint32, device=DEVICE)
        self.mask = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
        self.t1 = torch.zeros(batch_size, dtype=torch.float32, device=DEVICE) + 1e9
        self.t2 = torch.zeros(batch_size, dtype=torch.float32, device=DEVICE) + 1e9
        self.normals = torch.zeros(batch_size, 3, dtype=torch.float32, device=DEVICE)

    def get_batch(self):
        directions = torch.randn((self.batch_size, 3), dtype=torch.float32, device=DEVICE)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        origins = torch.rand((self.batch_size, 3), dtype=torch.float32, device=DEVICE) * (mesh_max - mesh_min) + mesh_min
        origins, _, _ = intersect_update(origins, directions, self.mesh_center, self.mesh_radius)

        _, fine_t, _ = self.fine_tracer.trace(origins, directions)
        mask, t, _, n_hits = self.rough_tracer_all.trace_all(origins, directions, 10)
        fine_t_expanded = fine_t[:, None]
        diff = torch.abs(t - fine_t_expanded)
        min_indices = torch.argmin(diff, dim=1)
        min_indices[min_indices >= self.n_classes - 2] = self.n_classes - 2

        mask = mask[:, 0]
        min_indices[mask] = min_indices[mask] + 1
        min_indices[~mask] = 0

        self.mask, self.t1, _ = self.rough_tracer.trace(origins, directions)

        # print(self.mask[:10])
        # print(mask[:10])
        # print(min_indices[:10])

        # print(self.t1[:10])
        # print(t[:10, 0])

        return origins, directions, min_indices
        # return origins, directions, mask.long()
        # return origins, directions, self.mask.long()
    
    def get_batch_cam(self):
        mesh_min, mesh_max = rough_mesh.get_bounds()
        max_extent = max(mesh_max - mesh_min)

        center = (mesh_max + mesh_min) * 0.5

        cam_pos = np.array([
            center[0] + max_extent * 1.0,
            center[1] - max_extent * 1.5,
            center[2] + max_extent * 0.5,
        ])
        cam_poses = np.tile(cam_pos, (n_pixels, 1))
        cam_dir = (center - cam_pos) * 1.1

        x_dir = np.cross(cam_dir, np.array([0, 0, 1]), axis=0)
        x_dir = x_dir / np.linalg.norm(x_dir) * (max_extent / 2)

        y_dir = -np.cross(x_dir, cam_dir, axis=0)
        y_dir = y_dir / np.linalg.norm(y_dir) * (max_extent / 2)

        x_coords, y_coords = np.meshgrid(
            np.linspace(-1, 1, img_size),
            np.linspace(-1, 1, img_size),
        )

        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()

        dirs = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]

        d_cam_poses = torch.tensor(cam_poses, dtype=torch.float32, device=DEVICE)
        d_dirs = torch.tensor(dirs, dtype=torch.float32, device=DEVICE)
        d_cam_poses, _, _ = intersect_update(d_cam_poses, d_dirs, self.mesh_center, self.mesh_radius)

        d_dirs = d_dirs / torch.norm(d_dirs, dim=-1, keepdim=True)

        # self.mask, self.t1, _ = self.fine_tracer.trace(d_cam_poses, d_dirs)

        _, fine_t, _ = self.fine_tracer.trace(d_cam_poses, d_dirs)
        mask, t, _, n_hits = self.rough_tracer_all.trace_all(d_cam_poses, d_dirs, self.n_classes)
        fine_t_expanded = fine_t[:, None]
        diff = torch.abs(t - fine_t_expanded)
        min_indices = torch.argmin(diff, dim=1)

        mask = mask[:, 0]
        min_indices[mask] = min_indices[mask] + 1
        min_indices[~mask] = 0

        return d_cam_poses, d_dirs, min_indices

    

class Model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        from models import TensoRFEncoder
        from encoding import MultiResHashGrid

        self.n_points = 1

        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": 8,
            "n_features_per_level": 2,
            "log2_hashmap_size": 20,
            "base_resolution": 4,
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

        self.n_input_dims = 4
        self.n_output_dims = n_classes

        self.encoding = MultiResHashGrid(self.n_input_dims, n_levels=8, n_features_per_level=2, log2_hashmap_size=20, base_resolution=4, finest_resolution=1024)
        # self.encoding = tcnn.Encoding(self.n_input_dims, self.encoding_config)
        # self.network = tcnn.Network(self.encoding.n_output_dims, self.n_output_dims, self.network_config)
        # self.network = tcnn.Network(4, self.n_output_dims, self.network_config)

        dim = 64
        self.network = nn.Sequential(
            # nn.Linear(self.encoding.n_output_dims * self.n_points, dim),
            nn.LazyLinear(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            # nn.Linear(dim, dim),
            # nn.ReLU(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, origins, directions):
        x = rays_to_unit_plucker4(origins, directions)
        x = self.encoding(x).float()
        x = self.network(x).float()

        # x = torch.cat([origins, directions], dim=1)
        # x = x.reshape(-1, 3)
        # x = self.encoding(x).float()
        # x = x.reshape(-1, self.encoding.n_output_dims * 2)
        # x = self.network(x)

        # t = np.linspace(0, 4, num=self.n_points).astype(np.float32)
        # t = torch.linspace(0, 0.5, steps=self.n_points, device=origins.device)  # [n_points]

        # x = origins[:, None, :] + directions[:, None, :] * t[None, :, None]  # [N, n_points, 3]
        # x = x.reshape(-1, 3)  # [N * n_points, 3]
        # x = self.encoding(x).float()  # [N * n_points, n_enc_dims]
        # x = x.reshape(-1, self.encoding.n_output_dims * self.n_points)  # [N, n_enc_dims * n_points]
        # x = self.network(x)  # [N, n_classes]

        return x


n_batches = 10000
batch_size = 1000000
n_classes = 4

data = MyDataset(fine_mesh, rough_mesh, batch_size, n_classes)
model = Model(n_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch_i in (bar := tqdm.trange(n_batches)):
    origins, directions, idxs_true = data.get_batch()
    idxs_pred = model(origins, directions)
    # print(idxs_true)
    # print(F.log_softmax(idxs_pred, dim=-1))

    # print(idxs_pred.shape, idxs_true.shape)

    # loss = F.cross_entropy(idxs_pred, idxs_true)
    loss = F.nll_loss(F.log_softmax(idxs_pred, dim=-1), idxs_true)
    # loss = F.binary_cross_entropy_with_logits(idxs_pred.squeeze(-1), idxs_true.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    bar.set_description(f"loss: {loss.item():.4f}")

    if batch_i % 100 == 0:
        origins, directions, idxs_true = data.get_batch_cam()
        idxs_pred = model(origins, directions)

        # accuracy = ((mask_pred > 0) == mask_true).float().mean().item()
        # hits = mask_true.float().mean().item()
        # print(f"Eval acc: {accuracy:.4f}; hits: {hits:.4f}")

        # slight red
        pred_color = np.array([1.0, 0.2, 0.2])
        true_edge_color = np.array([0.2, 1.0, 0.2]) 

        idxs_pred = idxs_pred.argmax(dim=-1)

        # img_true = (idxs_true / (n_classes - 1)).cpu().numpy().reshape(img_size, img_size, 1) @ true_edge_color[None, None, :]
        # img_pred = (idxs_pred / (n_classes - 1)).cpu().numpy().reshape(img_size, img_size, 1) @ pred_color[None, None, :]
        img_true = (idxs_true / (n_classes - 1)).cpu().numpy().reshape(img_size, img_size)
        img_pred = (idxs_pred / (n_classes - 1)).cpu().numpy().reshape(img_size, img_size)
        mask_cat = np.concatenate([img_true, img_pred], axis=1) * 255
        image = Image.fromarray(mask_cat.astype(np.uint8))
        image.save(f'output.png')

        # save checkpoint
        torch.save(model.state_dict(), 'model_checkpoint.pth')
