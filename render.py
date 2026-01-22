import torch
import torch.nn as nn
from utils import get_camera_rays, MeshWrapper, point_query, ShellType, save_mesh_previews
from PIL import Image
import numpy as np
from dataclasses import dataclass
import os
from models import DisplacementModel


@dataclass
class RaytraceResult:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    normals: torch.Tensor


def get_raytrace_loss(cam_poses, dirs, y, reduction="mean", include_orig_dist=False, dist_scale=0.1):
    distances = torch.cross(y - cam_poses, dirs, dim=1).norm(dim=1)
    dist_to_origin = (y - cam_poses).norm(dim=1)
    if include_orig_dist:
        distances = distances + dist_scale * dist_to_origin
    if reduction == "mean":
        return distances.mean()
    return distances


def optimize_hits(cfg, cam_poses, dirs, traverser, network, x0, shell_type):
    """
    Refine hit points along rays so that network outputs lie close to the ray.
    Returns per-hit tensors; caller must scatter them back to full image resolution.
    """
    network.eval()

    threshold = cfg.render.inner_loss_threshold if shell_type == ShellType.INNER else cfg.render.outer_loss_threshold

    accepted_x = torch.zeros_like(x0)
    accepted_y = torch.zeros_like(x0)
    accepted_mask = torch.zeros((x0.shape[0],), dtype=torch.bool, device=cfg.device)
    prev_loss_dist = torch.full((x0.shape[0],), float('inf'), device=cfg.device)

    x = nn.Parameter(x0.clone(), requires_grad=True)
    _, sdf_closests, sdf_barycentrics, sdf_face_idxs = point_query(traverser, x.data, cfg.device)
    barycentrics = nn.Parameter(torch.zeros_like(sdf_barycentrics), requires_grad=True)
    barycentrics.data = sdf_barycentrics
    x.data = sdf_closests

    optimizer = torch.optim.Adam([x], lr=cfg.render.gd_lr)
    # optimizer = torch.optim.SGD([x, barycentrics], lr=config.lr, momentum=0.99)
    # optimizer = torch.optim.SGD([x, barycentrics], lr=cfg.render.gd_lr, momentum=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=cfg.render.gd_lr_scheduler_min, total_iters=cfg.render.gd_steps)

    with torch.no_grad():
        y = network(x)
        per_ray_loss = get_raytrace_loss(cam_poses, dirs, y, reduction="none", include_orig_dist=False)
        per_ray_loss_dist = get_raytrace_loss(cam_poses, dirs, y, reduction="none", include_orig_dist=True)
        mask = (per_ray_loss < threshold) & (per_ray_loss_dist < prev_loss_dist)
        prev_loss_dist[mask] = per_ray_loss_dist[mask]

        accepted_x[mask] = x.data[mask]
        accepted_y[mask] = y[mask]
        accepted_mask[mask] = True

        if cfg.render.verbose:
            print("Loss:", per_ray_loss.mean().item())
            print(f"Accepted {accepted_mask.sum().item()} / {x.shape[0]}")    

    for _ in range(cfg.render.gd_steps):
        optimizer.zero_grad()
        y = network(x)
        loss = get_raytrace_loss(cam_poses, dirs, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _, sdf_closests, sdf_barycentrics, _ = point_query(traverser, x.data, cfg.device)
            barycentrics.data = sdf_barycentrics
            x.data = sdf_closests

        with torch.no_grad():
            y = network(x)
            per_ray_loss = get_raytrace_loss(cam_poses, dirs, y, reduction="none", include_orig_dist=False)
            per_ray_loss_dist = get_raytrace_loss(cam_poses, dirs, y, reduction="none", include_orig_dist=True)
            mask = (per_ray_loss < threshold) & (per_ray_loss_dist < prev_loss_dist)
            prev_loss_dist[mask] = per_ray_loss_dist[mask]

            accepted_x[mask] = x.data[mask]
            accepted_y[mask] = y[mask]
            accepted_mask[mask] = True

            if cfg.render.verbose:
                print("Loss:", per_ray_loss.mean().item())
                print(f"Accepted {accepted_mask.sum().item()} / {x.shape[0]}")

    normals = torch.zeros_like(accepted_x)
    normals[accepted_mask] = accepted_y[accepted_mask] - accepted_x[accepted_mask]
    normals[accepted_mask] = normals[accepted_mask] / (normals[accepted_mask].norm(dim=1, keepdim=True) + 1e-12)
    if shell_type == ShellType.OUTER:
        normals[accepted_mask] = -normals[accepted_mask]

    return RaytraceResult(x=accepted_x, y=accepted_y, mask=accepted_mask, normals=normals)


def raytrace_single_shell(cfg, cam_poses, dirs, mesh, network, shell_type):
    initial_mask, t, _ = mesh.ray_tracer.trace(cam_poses, dirs)
    x0 = cam_poses + dirs * t[:, None]

    hits = optimize_hits(cfg, cam_poses[initial_mask], dirs[initial_mask], mesh.traverser, network, x0[initial_mask], shell_type)

    n_rays = cam_poses.shape[0]
    x = torch.zeros((n_rays, 3), dtype=torch.float32, device=cfg.device)
    y = torch.zeros((n_rays, 3), dtype=torch.float32, device=cfg.device)
    normals = torch.zeros((n_rays, 3), dtype=torch.float32, device=cfg.device)
    mask = torch.zeros((n_rays,), dtype=torch.bool, device=cfg.device)

    hit_indices = initial_mask.nonzero(as_tuple=False).squeeze(1)
    accepted_indices = hit_indices[hits.mask]

    if accepted_indices.numel() > 0:
        x[accepted_indices] = hits.x[hits.mask]
        y[accepted_indices] = hits.y[hits.mask]
        normals[accepted_indices] = hits.normals[hits.mask]
        mask[accepted_indices] = True

    torch.cuda.empty_cache()
    return RaytraceResult(x=x, y=y, mask=mask, normals=normals)


def shade_lambert(result, dirs, img_size):
    colors = torch.zeros((img_size * img_size,), dtype=torch.float32, device=dirs.device)
    colors[result.mask] = (-dirs[result.mask] * result.normals[result.mask]).sum(dim=1)
    colors = (colors + 1.0) * 0.5
    colors[~result.mask] = 0.0
    colors = colors.cpu().numpy().reshape(img_size, img_size)
    colors[np.isnan(colors)] = 0.0
    return colors, Image.fromarray((colors * 255).astype(np.uint8))


def render_entry(cfg):
    #### meshes ####
    orig_mesh  = MeshWrapper.from_file(cfg.orig_mesh,  n_max_samples=cfg.mesh_n_max_samples, scale=1.0)
    inner_mesh = MeshWrapper.from_file(cfg.inner_mesh, n_max_samples=cfg.mesh_n_max_samples, scale=1.0)
    outer_mesh = MeshWrapper.from_file(cfg.outer_mesh, n_max_samples=cfg.mesh_n_max_samples, scale=1.0)

    os.makedirs(cfg.train.previews_dir, exist_ok=True)
    save_mesh_previews({
        f"{cfg.train.previews_dir}/orig_preview.png":  orig_mesh.mesh,
        f"{cfg.train.previews_dir}/inner_preview.png": inner_mesh.mesh,
        f"{cfg.train.previews_dir}/outer_preview.png": outer_mesh.mesh,
    })    

    ##### model ####
    checkpoint = torch.load(cfg.render.model_checkpoint, weights_only=False)
    cfg_model = checkpoint["cfg_model"]
    inner_net = DisplacementModel(cfg_model, inner_mesh.mesh).to(cfg.device)
    outer_net = DisplacementModel(cfg_model, outer_mesh.mesh).to(cfg.device)
    print("Loading model state...")
    inner_net.load_state_dict(checkpoint["inner_net"]["model"])
    outer_net.load_state_dict(checkpoint["outer_net"]["model"])
    
    #### shoot camera rays ####
    cam_poses, dirs = get_camera_rays(orig_mesh.mesh, img_size=cfg.render.img_size, device=cfg.device, angle=cfg.render.angle, distance_scale=cfg.render.distance_scale)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    n_rays = cam_poses.shape[0]
    x = torch.zeros((n_rays, 3), dtype=torch.float32, device=cfg.device)
    y = torch.zeros((n_rays, 3), dtype=torch.float32, device=cfg.device)
    normals = torch.zeros((n_rays, 3), dtype=torch.float32, device=cfg.device)
    mask = torch.zeros((n_rays,), dtype=torch.bool, device=cfg.device)

    results = []

    if cfg.render.use_inner:
        inner_result = raytrace_single_shell(cfg, cam_poses, dirs, inner_mesh, inner_net, ShellType.INNER)
        results.append(inner_result)

    if cfg.render.use_outer:
        outer_result = raytrace_single_shell(cfg, cam_poses, dirs, outer_mesh, outer_net, ShellType.OUTER)
        results.append(outer_result)

    for res in results:
        if res.mask.any():
            x[res.mask] = res.x[res.mask]
            y[res.mask] = res.y[res.mask]
            normals[res.mask] = res.normals[res.mask]
            mask = mask | res.mask

    combined_result = RaytraceResult(x=x, y=y, mask=mask, normals=normals)

    img_size = cfg.render.img_size

    logs = dict()
    os.makedirs(cfg.render.output_dir, exist_ok=True)

    #### shade and save predicted image ####
    pred_colors, pred_colors_img = shade_lambert(combined_result, dirs, img_size)
    pred_colors_img.save(f"{cfg.render.output_dir}/pred_colors.png")
    logs["pred_colors"] = pred_colors

    #### the rest is computing metrics ####

    orig_mask, orig_t, orig_normals = orig_mesh.ray_tracer.trace(cam_poses, dirs)
    outer_mask, outer_t, _ = outer_mesh.ray_tracer.trace(cam_poses, dirs)
    inner_mask, inner_t, _ = inner_mesh.ray_tracer.trace(cam_poses, dirs)    

    orig_y = cam_poses + dirs * orig_t[:, None]
    inner_y = cam_poses + dirs * inner_t[:, None]
    outer_y = cam_poses + dirs * outer_t[:, None]

    _, inner_proj_y, _, _ = point_query(orig_mesh.traverser, inner_y, cfg.device)
    _, outer_proj_y, _, _ = point_query(orig_mesh.traverser, outer_y, cfg.device)

    inner_proj_normals = (inner_proj_y - inner_y) / (inner_y - inner_proj_y).norm(dim=1, keepdim=True)
    outer_proj_normals = (outer_y - outer_proj_y) / (outer_y - outer_proj_y).norm(dim=1, keepdim=True)

    if cfg.render.use_inner:
        pred_colors_inner, pred_colors_inner_img = shade_lambert(inner_result, dirs, img_size)
        pred_colors_inner_img.save(f"{cfg.render.output_dir}/pred_colors_inner.png")

        true_colors_nogd_inner, true_colors_nogd_inner_img = shade_lambert(
            RaytraceResult(x=inner_result.x, y=inner_result.y, mask=inner_result.mask, normals=inner_proj_normals),
            dirs, img_size,
        )
        true_colors_nogd_inner_img.save(f"{cfg.render.output_dir}/true_colors_nogd_inner.png")

        mse = np.square(pred_colors_inner - true_colors_nogd_inner).mean()
        print(f"Inner shell Pixel MSE: {mse}")
        logs["mse_nogd_inner"] = mse

        raytrace_loss = get_raytrace_loss(cam_poses[inner_mask], dirs[inner_mask], inner_result.y[inner_mask], reduction="none")
        loss_map = torch.zeros((img_size * img_size,), dtype=torch.float32, device=cfg.device)
        loss_map[inner_mask] = raytrace_loss
        loss_map = loss_map / raytrace_loss.max()
        loss_map = loss_map.cpu().numpy().reshape(img_size, img_size)
        loss_map_img = Image.fromarray((loss_map * 255).astype(np.uint8))
        loss_map_img.save(f"{cfg.render.output_dir}/raytrace_loss_inner.png")

    if cfg.render.use_outer:
        pred_colors_outer, pred_colors_outer_img = shade_lambert(outer_result, dirs, img_size)
        pred_colors_outer_img.save(f"{cfg.render.output_dir}/pred_colors_outer.png")

        true_colors_nogd_outer, true_colors_nogd_outer_img = shade_lambert(
            RaytraceResult(x=outer_result.x, y=outer_result.y, mask=outer_result.mask, normals=outer_proj_normals),
            dirs, img_size,
        )
        true_colors_nogd_outer_img.save(f"{cfg.render.output_dir}/true_colors_nogd_outer.png")

        mse = np.square(pred_colors_outer - true_colors_nogd_outer).mean()
        print(f"Outer shell Pixel MSE: {mse}")
        logs["mse_nogd_outer"] = mse

        raytrace_loss = get_raytrace_loss(cam_poses[outer_mask], dirs[outer_mask], outer_result.y[outer_mask], reduction="none")
        loss_map = torch.zeros((img_size * img_size,), dtype=torch.float32, device=cfg.device)
        loss_map[outer_mask] = raytrace_loss
        loss_map = loss_map / raytrace_loss.max()
        loss_map = loss_map.cpu().numpy().reshape(img_size, img_size)
        loss_map_img = Image.fromarray((loss_map * 255).astype(np.uint8))
        loss_map_img.save(f"{cfg.render.output_dir}/raytrace_loss_outer.png")

    gt_img = torch.zeros((img_size * img_size), dtype=torch.float32, device=cfg.device)
    gt_img[orig_mask] = (-dirs[orig_mask] * orig_normals[orig_mask]).sum(dim=-1).clamp(min=0.0)
    gt_img[orig_mask] = (gt_img[orig_mask] + 1.0) * 0.5
    gt_img = gt_img.cpu().numpy().reshape(img_size, img_size)
    gt_img_pil = Image.fromarray((gt_img * 255).astype(np.uint8))
    gt_img_pil.save(f"{cfg.render.output_dir}/gt_colors.png")

    return logs


if __name__ == "__main__":
    import cfg as cfg_module
    render_entry(cfg_module.cfg)
