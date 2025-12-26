import numpy as np
import torch


@torch.no_grad()
def sample_points(sampler, batch_size, device):
    points = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    barycentrics = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    face_idxs = torch.zeros((batch_size,), dtype=torch.uint32, device=device)

    sampler.sample(points, barycentrics, face_idxs, batch_size)

    return points, barycentrics, face_idxs.long()


@torch.no_grad()
def point_query(traverser, points, device):
    t = torch.zeros((points.size(0),), dtype=torch.float32, device=device)
    closest_pts = torch.zeros((points.size(0), 3), dtype=torch.float32, device=device)
    barycentrics = torch.zeros((points.size(0), 3), dtype=torch.float32, device=device)
    face_idxs = torch.zeros((points.size(0),), dtype=torch.uint32, device=device)
    
    traverser.point_query(points, t, closest_pts, barycentrics, face_idxs)

    return t, closest_pts, barycentrics, face_idxs.long()


def chamfer_distance(a, b):
    """Compute bidirectional Chamfer distance between two point clouds a and b.
    a: (N, 3)
    b: (M, 3)
    Returns: scalar Chamfer distance
    """
    N, M = a.size(0), b.size(0)
    a_exp = a.unsqueeze(1).expand(N, M, 3)  # (N, M, 3)
    b_exp = b.unsqueeze(0).expand(N, M, 3)  # (N, M, 3)
    dists = torch.norm(a_exp - b_exp, dim=2)  # (N, M)

    min_a_to_b, _ = torch.min(dists, dim=1)  # (N,)
    min_b_to_a, _ = torch.min(dists, dim=0)  # (M,)

    cd = min_a_to_b.mean() + min_b_to_a.mean()
    return cd


def get_camera_rays(mesh, img_size, device):
    n_pixels = img_size * img_size

    mesh_min, mesh_max = mesh.get_bounds()
    max_extent = max(mesh_max - mesh_min)

    center = (mesh_max + mesh_min) * 0.5

    cam_pos = np.array([
        center[0] + max_extent * 1.0,
        center[1] - max_extent * 1.5,
        center[2] + max_extent * 0.5,
    ])
    cam_poses = np.tile(cam_pos, (n_pixels, 1))
    cam_dir = (center - cam_pos) * 0.9

    x_dir = np.cross(cam_dir, np.array([0, 0, 1]))
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

    return d_cam_poses, d_dirs
