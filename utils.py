import numpy as np
import torch
from dataclasses import dataclass
from mesh_utils import Mesh, GPURayTracer, GPUSampler


@dataclass
class MeshWrapper:
    mesh: Mesh
    sampler: GPUSampler
    ray_tracer: GPURayTracer

    @classmethod
    def from_file(cls, path, n_max_samples, scale):
        mesh = Mesh.from_file(path, build_bvh=True, max_leaf_size=5, smooth_normals=True, scale=scale)
        ray_tracer = GPURayTracer(mesh)
        sampler = GPUSampler(mesh, n_max_samples)
        return cls(mesh, sampler, ray_tracer)

# @torch.no_grad()
# def sample_points(sampler, batch_size, device):
#     points = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
#     barycentrics = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
#     face_idxs = torch.zeros((batch_size,), dtype=torch.uint32, device=device)

#     sampler.sample(points, barycentrics, face_idxs, batch_size)

#     return points, barycentrics, face_idxs.long()


def get_camera_rays(mesh, img_size, device, angle=0.0, distance_scale=1.0):
    """
    angle: rotation around the object in radians (positive = CCW around +Z).
           angle=0 keeps the original camera pose.
    """
    n_pixels = img_size * img_size

    mesh_min, mesh_max = mesh.get_bounds()
    mesh_min = np.stack([mesh_min[0], mesh_min[2], -mesh_max[1]], axis=0)
    mesh_max = np.stack([mesh_max[0], mesh_max[2], -mesh_min[1]], axis=0)

    max_extent = max(mesh_max - mesh_min)
    center = (mesh_max + mesh_min) * 0.5

    # --- original camera offset, but rotated around +Z by `angle`
    base_dx = max_extent * 1.0
    base_dy = -max_extent * 1.5
    r_xy = np.hypot(base_dx, base_dy)
    base_theta = np.arctan2(base_dy, base_dx)
    theta = base_theta + angle

    cam_pos = np.array([
        center[0] + r_xy * np.cos(theta) * distance_scale,
        center[1] + r_xy * np.sin(theta) * distance_scale,
        center[2] + max_extent * 0.5 * distance_scale,   # keep the same "slightly top" height
    ], dtype=np.float32)

    cam_poses = np.tile(cam_pos, (n_pixels, 1))

    # forward vector (keep same scaling as your original code)
    cam_dir = (center - cam_pos) * 0.9 / distance_scale

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    x_dir = np.cross(cam_dir, up)
    x_dir = x_dir / np.linalg.norm(x_dir) * (max_extent / 2)

    y_dir = -np.cross(x_dir, cam_dir)
    y_dir = y_dir / np.linalg.norm(y_dir) * (max_extent / 2)

    x_coords, y_coords = np.meshgrid(
        np.linspace(-1, 1, img_size),
        np.linspace(-1, 1, img_size),
    )
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    dirs = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]

    d_cam_poses = torch.from_numpy(cam_poses).float().to(device)
    d_dirs = torch.from_numpy(dirs).float().to(device)

    d_cam_poses = torch.stack([d_cam_poses[:, 0], d_cam_poses[:, 2], -d_cam_poses[:, 1]], dim=-1)
    d_dirs = torch.stack([d_dirs[:, 0], d_dirs[:, 2], -d_dirs[:, 1]], dim=-1)

    return d_cam_poses, d_dirs


def calculate_normals(faces, vertices, face_ids):
    v0_ids = faces[face_ids, 0]
    v1_ids = faces[face_ids, 1]
    v2_ids = faces[face_ids, 2]

    v0s = vertices[v0_ids]
    v1s = vertices[v1_ids]
    v2s = vertices[v2_ids]

    normals = np.cross(v1s - v0s, v2s - v0s, axis=1)
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def sample_directions(normals, device):
    thetas = np.random.uniform(0, np.pi, size=normals.shape[0])
    phis = np.random.uniform(0, np.pi, size=normals.shape[0])

    x_norm = np.sin(thetas) * np.cos(phis)
    y_norm = np.sin(thetas) * np.sin(phis)
    z_norm = np.cos(thetas)
    vector = np.stack([x_norm, y_norm, z_norm], axis=1)

    basis_x_norm = np.ones(normals.shape)
    mask = normals[:, 2] != 0
    basis_x_norm[mask, 2] = -(normals[mask, 0] + normals[mask, 1]) / normals[mask, 2]
    basis_x_norm[~mask, 2] = 0
    basis_x_norm = basis_x_norm / np.linalg.norm(basis_x_norm, axis=1)[:, None]
    basis_y_norm = -normals / np.linalg.norm(normals, axis=1)[:, None]
    basis_z_norm = np.cross(basis_x_norm, basis_y_norm, axis=1)
    basis_coefs = np.stack([basis_x_norm, basis_y_norm, basis_z_norm], axis=2)

    prop_ds = np.einsum('ijk,ik->ij', basis_coefs, vector)
    return torch.tensor(prop_ds, device=device)


def sample_directions_torch(normals, device):
    prop_ds = torch.randn(normals.shape, device=device)
    prop_ds = prop_ds / prop_ds.norm(dim=1, keepdim=True)
    return prop_ds
    # n = normals.shape[0]
    # thetas = torch.rand(n, device=device) * torch.pi
    # phis = torch.rand(n, device=device) * torch.pi

    # sin_thetas = torch.sin(thetas)
    # vector = torch.stack([
    #     sin_thetas * torch.cos(phis),
    #     sin_thetas * torch.sin(phis),
    #     torch.cos(thetas),
    # ], dim=1)

    # basis_x_norm = torch.ones_like(normals)
    # mask = normals[:, 2] != 0
    # basis_x_norm[mask, 2] = -(normals[mask, 0] + normals[mask, 1]) / normals[mask, 2]
    # basis_x_norm[~mask, 2] = 0
    # basis_x_norm = basis_x_norm / basis_x_norm.norm(dim=1, keepdim=True)
    # basis_y_norm = -normals / normals.norm(dim=1, keepdim=True)
    # basis_z_norm = torch.linalg.cross(basis_x_norm, basis_y_norm)
    # basis_coefs = torch.stack([basis_x_norm, basis_y_norm, basis_z_norm], dim=2)

    # prop_ds = torch.einsum('ijk,ik->ij', basis_coefs, vector)
    # return prop_ds


def sample_sphere(radius, center, sample_size):
    thetas = np.random.uniform(0, np.pi, size=sample_size)
    phis = np.random.uniform(0, 2 * np.pi, size=sample_size)

    x = np.sin(thetas) * np.cos(phis) * radius
    y = np.sin(thetas) * np.sin(phis) * radius
    z = np.cos(thetas) * radius

    points = np.stack([x, y, z], axis=1) + center[None, :]
    normals = points / np.linalg.norm(points, axis=1)[:, None]

    return points, normals


def sample_sphere_torch(radius, center, sample_size, device):
    thetas = torch.rand(sample_size, device=device) * torch.pi
    phis = torch.rand(sample_size, device=device) * 2 * torch.pi
    #radius = torch.rand(sample_size, device=device) * radius

    sin_thetas = torch.sin(thetas)
    x = sin_thetas * torch.cos(phis) * radius
    y = sin_thetas * torch.sin(phis) * radius
    z = torch.cos(thetas) * radius

    center_t = torch.as_tensor(center, dtype=torch.float32, device=device)
    points = torch.stack([x, y, z], dim=1) + center_t[None, :]
    normals = points / points.norm(dim=1, keepdim=True)

    return points, normals
