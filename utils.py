import torch


@torch.no_grad()
def sample_points(sampler, batch_size, device):
    pts = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    sampler.sample(pts, batch_size)
    return pts


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