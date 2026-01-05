from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from utils import point_query


@dataclass
class RaytraceResult:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    normals: torch.Tensor


@dataclass
class RaytraceConfig:
    epochs: int = 3
    threshold: float = 0.01
    lr: float = 0.01
    snap_to_closest: bool = False


def get_raytrace_loss(cam_poses: torch.Tensor, dirs: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    distances = torch.cross(y - cam_poses, dirs, dim=1).norm(dim=1)
    if reduction == "mean":
        return distances.mean()
    return distances


def _optimize_hits(
    cam_poses: torch.Tensor,
    dirs: torch.Tensor,
    traverser,
    network,
    x0: torch.Tensor,
    *,
    config: Optional[RaytraceConfig] = None,
    verbose: bool = False,
) -> RaytraceResult:
    """
    Refine hit points along rays so that network outputs lie close to the ray.
    Returns per-hit tensors; caller must scatter them back to full image resolution.
    """
    device = x0.device
    config = config or RaytraceConfig()
    network.eval()

    accepted_x = torch.zeros_like(x0)
    accepted_y = torch.zeros_like(x0)
    accepted_mask = torch.zeros((x0.shape[0],), dtype=torch.bool, device=device)

    x = nn.Parameter(x0.clone(), requires_grad=True)
    _, sdf_closests, sdf_barycentrics, sdf_face_idxs = point_query(traverser, x.data, device)
    barycentrics = nn.Parameter(torch.zeros_like(sdf_barycentrics), requires_grad=True)
    barycentrics.data = sdf_barycentrics
    x.data = sdf_closests

    optimizer = torch.optim.Adam([x], lr=config.lr)

    for _ in range(config.epochs):
        with torch.no_grad():
            _, sdf_closests, sdf_barycentrics, sdf_face_idxs = point_query(traverser, x.data, device)
            barycentrics.data = sdf_barycentrics
            if config.snap_to_closest:
                x.data = sdf_closests

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            y = network(x, face_idxs=sdf_face_idxs, barycentrics=barycentrics)
            loss = get_raytrace_loss(cam_poses, dirs, y)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            y = network(x, face_idxs=sdf_face_idxs, barycentrics=barycentrics)
            per_ray_loss = get_raytrace_loss(cam_poses, dirs, y, reduction="none")
            mask = per_ray_loss < config.threshold

            accepted_x[mask] = x.data[mask]
            accepted_y[mask] = y[mask]
            accepted_mask[mask] = True

            if verbose:
                print("Loss:", per_ray_loss.mean().item())
                print(f"Accepted {accepted_mask.sum().item()} / {x.shape[0]}")

    normals = torch.zeros_like(accepted_x)
    normals[accepted_mask] = accepted_x[accepted_mask] - accepted_y[accepted_mask]
    normals[accepted_mask] = normals[accepted_mask] / (normals[accepted_mask].norm(dim=1, keepdim=True) + 1e-12)

    return RaytraceResult(x=accepted_x, y=accepted_y, mask=accepted_mask, normals=normals)


def raytrace_mesh(
    cam_poses: torch.Tensor,
    dirs: torch.Tensor,
    mesh,
    network,
    *,
    config: Optional[RaytraceConfig] = None,
    verbose: bool = False,
) -> RaytraceResult:
    initial_mask, t, _ = mesh.ray_tracer.trace(cam_poses, dirs)
    x0 = cam_poses + dirs * t[:, None]

    hits = _optimize_hits(
        cam_poses[initial_mask],
        dirs[initial_mask],
        mesh.traverser,
        network,
        x0[initial_mask],
        config=config,
        verbose=verbose,
    )

    device = cam_poses.device
    n_rays = cam_poses.shape[0]
    x = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    y = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    normals = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    mask = torch.zeros((n_rays,), dtype=torch.bool, device=device)

    hit_indices = initial_mask.nonzero(as_tuple=False).squeeze(1)
    accepted_indices = hit_indices[hits.mask]

    if accepted_indices.numel() > 0:
        x[accepted_indices] = hits.x[hits.mask]
        y[accepted_indices] = hits.y[hits.mask]
        normals[accepted_indices] = hits.normals[hits.mask]
        mask[accepted_indices] = True

    torch.cuda.empty_cache()
    return RaytraceResult(x=x, y=y, mask=mask, normals=normals)


def raytrace_inner_outer(
    cam_poses: torch.Tensor,
    dirs: torch.Tensor,
    inner_mesh,
    outer_mesh,
    inner_net,
    outer_net,
    *,
    config: Optional[RaytraceConfig] = None,
    verbose: bool = False,
) -> RaytraceResult:
    device = cam_poses.device
    n_rays = cam_poses.shape[0]
    x = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    y = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    normals = torch.zeros((n_rays, 3), dtype=torch.float32, device=device)
    mask = torch.zeros((n_rays,), dtype=torch.bool, device=device)

    outer = raytrace_mesh(cam_poses, dirs, outer_mesh, outer_net, config=config, verbose=verbose)
    inner = raytrace_mesh(cam_poses, dirs, inner_mesh, inner_net, config=config, verbose=verbose)

    for res in (outer, inner):  # inner overrides where both are valid to mimic previous ordering
        if res.mask.any():
            x[res.mask] = res.x[res.mask]
            y[res.mask] = res.y[res.mask]
            normals[res.mask] = res.normals[res.mask]
            mask = mask | res.mask

    return RaytraceResult(x=x, y=y, mask=mask, normals=normals)


# Backwards compatibility aliases
def do_raytrace_wrapper(cam_poses, dirs, mesh, net, verbose: bool = False, config: Optional[RaytraceConfig] = None):
    result = raytrace_mesh(cam_poses, dirs, mesh, net, config=config, verbose=verbose)
    return result.x, result.y, result.mask, result.normals, result.mask


def do_raytrace_wrapper_2(
    cam_poses,
    dirs,
    inner_mesh,
    outer_mesh,
    inner_net,
    outer_net,
    verbose: bool = False,
    config: Optional[RaytraceConfig] = None,
):
    result = raytrace_inner_outer(
        cam_poses, dirs, inner_mesh, outer_mesh, inner_net, outer_net, config=config, verbose=verbose
    )
    return result.x, result.y, result.mask, result.normals, result.mask
