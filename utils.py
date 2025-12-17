import numpy as np
import torch


def ray_mesh_intersect(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    batch_size: int = 1024,
    mode: str = "closest",
):
    """
    Ray / triangle mesh intersections using batched Möller–Trumbore.

    Args:
        ray_origins: (N, 3) float tensor of ray origins.
        ray_dirs:    (N, 3) float tensor of ray directions (NOT normalized).
        vertices:    (M, 3) float tensor of mesh vertices.
        faces:       (K, 3) long tensor of triangle vertex indices.
        batch_size:  Number of rays processed per batch.
        mode:        "closest" or "all".

    Returns:
        If mode == "closest":
            distances:   (N,)   float tensor, distance `t` along each ray
                         (inf if no hit).
            tri_indices: (N,)   long tensor, triangle index per ray
                         (-1 if no hit).

        If mode == "all":
            distances:   (N, K) float tensor, all intersection distances
                         sorted ascending per ray (inf where no hit).
            tri_indices: (N, K) long tensor, corresponding triangle indices
                         sorted to match distances (-1 where no hit).
    """
    if mode not in ("closest", "all"):
        raise ValueError("mode must be 'closest' or 'all'")

    assert ray_origins.shape == ray_dirs.shape
    assert ray_origins.shape[1] == 3
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3

    device = ray_origins.device
    dtype = ray_origins.dtype

    N = ray_origins.shape[0]
    K = faces.shape[0]

    # Triangle vertices (K, 3)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Triangle edges
    e1 = v1 - v0  # (K, 3)
    e2 = v2 - v0  # (K, 3)

    # Add ray dimension for broadcasting over triangles
    v0 = v0.unsqueeze(0)  # (1, K, 3)
    e1 = e1.unsqueeze(0)  # (1, K, 3)
    e2 = e2.unsqueeze(0)  # (1, K, 3)

    # Output buffers
    if mode == "closest":
        dist_out = torch.full((N,), float("inf"), device=device, dtype=dtype)
        tri_out = torch.full((N,), -1, device=device, dtype=torch.long)
    else:  # "all"
        dist_out = torch.full((N, K), float("inf"), device=device, dtype=dtype)
        tri_out = torch.full((N, K), -1, device=device, dtype=torch.long)

    # Numeric epsilon
    eps = 1e-8 if dtype in (torch.float32, torch.float64) else 1e-4

    # Process rays in batches
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ro = ray_origins[start:end].unsqueeze(1)  # (B, 1, 3)
        rd = ray_dirs[start:end].unsqueeze(1)     # (B, 1, 3)
        B = ro.shape[0]

        # Möller–Trumbore algorithm (batched over rays and triangles)
        pvec = torch.cross(rd, e2, dim=-1)        # (B, K, 3)
        det = (e1 * pvec).sum(-1)                 # (B, K)

        mask = det.abs() > eps                    # non-degenerate, non-parallel

        inv_det = torch.zeros_like(det)
        inv_det[mask] = 1.0 / det[mask]

        tvec = ro - v0                            # (B, K, 3)

        u = (tvec * pvec).sum(-1) * inv_det       # (B, K)
        mask = mask & (u >= 0.0) & (u <= 1.0)

        qvec = torch.cross(tvec, e1, dim=-1)      # (B, K, 3)

        v = (rd * qvec).sum(-1) * inv_det         # (B, K)
        mask = mask & (v >= 0.0) & (u + v <= 1.0)

        t = (e2 * qvec).sum(-1) * inv_det         # (B, K)
        mask = mask & (t >= 0.0)

        # Invalid intersections -> t = inf
        t_valid = t.clone()
        t_valid[~mask] = float("inf")

        if mode == "closest":
            # Min along triangles for each ray
            dist_batch, tri_batch = t_valid.min(dim=1)  # (B,)
            tri_batch = tri_batch.to(torch.long)

            # Mark rays with no valid intersections
            no_hit = ~torch.isfinite(dist_batch)
            tri_batch[no_hit] = -1

            dist_out[start:end] = dist_batch
            tri_out[start:end] = tri_batch

        else:  # "all"
            # Sort along triangles for each ray
            dist_sorted, order = torch.sort(t_valid, dim=1)  # (B, K)
            tri_sorted = order.to(torch.long)

            # Mark invalid (inf) entries with -1
            invalid = ~torch.isfinite(dist_sorted)
            tri_sorted[invalid] = -1

            dist_out[start:end] = dist_sorted
            tri_out[start:end] = tri_sorted

    return dist_out, tri_out


def closest_intersection_index_to_targets(
    t: torch.Tensor,
    hit_mask: torch.Tensor,
    target_value: torch.Tensor,
):
    """
    t           : (N_rays, N_faces) float, +inf where no hit
    hit_mask    : (N_rays, N_faces) bool
    target_value: (N_rays,) float, per-ray target distance

    Returns
    -------
    closest_idx : (N_rays,) long
        For each ray, the face index whose intersection distance is
        closest to the target_value for that ray.
        -1 where the ray has no intersections.
    num_hits    : (N_rays,) long
        Number of intersections per ray.
    """
    # Sanity check
    assert t.ndim == 2 and hit_mask.shape == t.shape
    assert target_value.ndim == 1 and target_value.shape[0] == t.shape[0]

    # Number of valid intersections per ray
    num_hits = hit_mask.sum(dim=1)   # (N_rays,)

    # Compute |t - target_value| for each ray/face
    diff = (t - target_value.unsqueeze(1)).abs()   # (N_rays, N_faces)

    # Argmin over faces for each ray
    closest_idx = diff.argmin(dim=1).to(torch.long)  # (N_rays,)

    # For rays with no hits at all, set index to -1
    closest_idx[num_hits == 0] = -1

    return closest_idx, num_hits


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



def write_pairs_as_obj(p0: torch.Tensor, p1: torch.Tensor, path: str) -> None:
    """
    p0: [N, 3] tensor
    p1: [N, 3] tensor
    Writes an OBJ with vertices and line segments (edges) connecting each pair.
    No faces.
    """
    if p0.ndim != 2 or p1.ndim != 2 or p0.shape != p1.shape or p0.shape[1] != 3:
        raise ValueError(f"Expected p0 and p1 to be [N,3] with same shape, got {p0.shape} and {p1.shape}")

    p0c = p0.detach().to(dtype=torch.float64, device="cpu")
    p1c = p1.detach().to(dtype=torch.float64, device="cpu")

    # Interleave vertices: (p0[0], p1[0], p0[1], p1[1], ...)
    verts = torch.stack((p0c, p1c), dim=1).reshape(-1, 3)  # [2N, 3]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# OBJ: point pairs as line segments\n")
        # vertices
        for x, y, z in verts.tolist():
            f.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")

        # edges as line elements; OBJ indices are 1-based
        # pair i uses vertices (2*i+1, 2*i+2)
        n = p0c.shape[0]
        for i in range(n):
            a = 2 * i + 1
            b = 2 * i + 2
            f.write(f"l {a} {b}\n")


def write_triples_as_obj(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, path: str) -> None:
    """
    p0, p1, p2: [N, 3] tensors (same shape)
    Writes an OBJ with vertices and line segments:
      - connects p0[i] -> p1[i]
      - connects p1[i] -> p2[i]
    Does NOT connect p0[i] -> p2[i].
    No faces.
    """
    for name, t in (("p0", p0), ("p1", p1), ("p2", p2)):
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"{name} must be [N,3], got {t.shape}")
    if p0.shape != p1.shape or p0.shape != p2.shape:
        raise ValueError(f"All tensors must have the same shape, got {p0.shape}, {p1.shape}, {p2.shape}")

    p0c = p0.detach().to(dtype=torch.float64, device="cpu")
    p1c = p1.detach().to(dtype=torch.float64, device="cpu")
    p2c = p2.detach().to(dtype=torch.float64, device="cpu")

    # Interleave per triple: (p0[i], p1[i], p2[i]) for i=0..N-1
    verts = torch.stack((p0c, p1c, p2c), dim=1).reshape(-1, 3)  # [3N, 3]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# OBJ: point triples as two line segments per triple (p0-p1, p1-p2)\n")

        # vertices
        for x, y, z in verts.tolist():
            f.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")

        # edges as line elements; OBJ indices are 1-based
        n = p0c.shape[0]
        for i in range(n):
            a = 3 * i + 1  # p0[i]
            b = 3 * i + 2  # p1[i]
            c = 3 * i + 3  # p2[i]
            f.write(f"l {a} {b}\n")
            f.write(f"l {b} {c}\n")
