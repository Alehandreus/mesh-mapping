import os
import numpy as np
import torch
import torch.autograd as autograd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import nn
import torch.nn.functional as F
import math
import copy
from tqdm import tqdm
from utils import MeshWrapper, sample_directions_torch, sample_sphere_torch, get_camera_rays
from visualization import save_mesh_previews, render_predictions

from models import RayModel
import time

# to make tensorflow shut up
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(cfg, model_config, model, averaged_model, optimizer, scheduler, run_name, step):
    checkpoint_data = {
        "model_config": model_config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "learning_rate_scheduler": scheduler.state_dict()
    }

    if cfg.train.use_averaged_model in ['EMA', 'SWA']:
        checkpoint_data["averaged_model"] = averaged_model.state_dict()
        copy_model = copy.deepcopy(averaged_model.module)
    else:
        copy_model = copy.deepcopy(model)
        
    copy_model.half()
    torch.save(copy_model.state_dict(), f"{cfg.train.checkpoints_path}/{run_name}_half.pt")
    
    checkpoint_name = f"{cfg.train.checkpoints_path}/{run_name}.pt"
    torch.save(checkpoint_data, checkpoint_name)

    if copy_model.model_config.encoding_type == "3d":
        point_encoding_params = copy_model.point_encoding.params.data.cpu().numpy().astype(np.float16)
        direction_encoding_params = copy_model.direction_encoding.params.data.cpu().numpy().astype(np.float16)
        network_params = copy_model.network.params.data.cpu().numpy().astype(np.float16)
        total_params = np.concatenate([point_encoding_params, direction_encoding_params, network_params])
        total_params.tofile(f"{cfg.train.checkpoints_path}/{run_name}.bin")


@torch.no_grad()
def eval_model(cfg, fine_mesh, outer_mesh, inner_mesh, model):
    model.eval()

    cam_poses, ds = get_camera_rays(fine_mesh.mesh, img_size=cfg.visualization.image_size, device=cfg.device, distance_scale=1.0)
    ds = ds / ds.norm(dim=1, keepdim=True)

    pred_intersection_global = torch.zeros((cam_poses.shape[0],), dtype=torch.float32, device=cfg.device)
    pred_t_global = torch.zeros((cam_poses.shape[0],), dtype=torch.float32, device=cfg.device)
    pred_normal_global = torch.zeros((cam_poses.shape[0], 3), dtype=torch.float32, device=cfg.device)

    x_outer_enter_mask, x_outer_enter_t, _, _ = outer_mesh.ray_tracer.trace(cam_poses, ds, allow_negative=False)
    x_outer_enter = cam_poses + ds * x_outer_enter_t[:, None]
    x_outer_enter = x_outer_enter[x_outer_enter_mask]

    mask_for_remaining_rays_global = x_outer_enter_mask.clone()

    # directions for rays left to trace
    ds_left = ds[x_outer_enter_mask]

    # distance already accumulated along traced segments
    accum_t = x_outer_enter_t[x_outer_enter_mask]

    while len(x_outer_enter) > 0:
        x_outer_enter = x_outer_enter + ds_left * 1e-3

        # intersect orig mesh, inner shell and outer shell again 
        x_outer_exit_mask, x_outer_exit_t, _, _ = outer_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
        x_outer_exit = x_outer_enter + ds_left * x_outer_exit_t[:, None]

        if (~x_outer_exit_mask).sum() > 0:
            # print(f"Warning no escape from outer shell! ({(~x_outer_exit_mask).sum().item()} rays)")
            x_outer_exit_t[~x_outer_exit_mask] = 1e-3
            x_outer_exit[~x_outer_exit_mask] = x_outer_enter[~x_outer_exit_mask] + ds_left[~x_outer_exit_mask] * x_outer_exit_t[~x_outer_exit_mask][:, None]

        x_inner_mask, x_inner_t, _, _ = inner_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
        x_inner_enter = x_outer_enter + ds_left * x_inner_t[:, None]
        x_inner_enter[~x_inner_mask] = 0

        input_enter_points = x_outer_enter
        input_exit_points = torch.zeros_like(input_enter_points)
        input_exit_points[x_inner_mask] = x_inner_enter[x_inner_mask]
        input_exit_points[~x_inner_mask] = x_outer_exit[~x_inner_mask]
        input_directions = ds_left

        pred_intersection, pred_t, pred_normal = model(
            input_enter_points,
            input_exit_points,
            input_directions,
        )
        pred_intersection_mask = (pred_intersection >= 0)

        pred_intersection_global[mask_for_remaining_rays_global] = pred_intersection
        pred_t_global[mask_for_remaining_rays_global] = pred_t + accum_t
        pred_normal_global[mask_for_remaining_rays_global] = pred_normal

        x_outer_exit = x_outer_exit + ds_left * 1e-3
        x_outer_enter_mask_new, x_outer_enter_t_new, _, _ = outer_mesh.ray_tracer.trace(x_outer_exit, ds_left, allow_negative=False)
        x_outer_enter_new = x_outer_exit + ds_left * x_outer_enter_t_new[:, None]        

        # prepare for next iteration
        mask_for_remaining_rays = ~pred_intersection_mask & x_outer_enter_mask_new
        mask_for_remaining_rays_global[mask_for_remaining_rays_global.clone()] = mask_for_remaining_rays
        x_outer_enter = x_outer_enter_new[mask_for_remaining_rays]
        ds_left = ds_left[mask_for_remaining_rays]
        accum_t = accum_t[mask_for_remaining_rays] + x_outer_exit_t[mask_for_remaining_rays] + x_outer_enter_t_new[mask_for_remaining_rays]

        print("Remaining rays to trace:", x_outer_enter.shape[0])

    # prepare inputs for rendering

    x_orig_mask, x_orig_t, x_orig_normals, _ = fine_mesh.ray_tracer.trace(cam_poses, ds, allow_negative=False)
    x_orig = cam_poses + ds * x_orig_t[:, None]
    x_orig[~x_orig_mask] = 0    

    true_mask = x_orig_mask.clone()
    points = cam_poses[true_mask] + ds[true_mask] * x_orig_t[true_mask][:, None]
    normals_traced = x_orig_normals.clone()

    whole_intersected_mask = (pred_intersection_global >= 0)

    predicted_points = (cam_poses + ds * pred_t_global[:, None])[whole_intersected_mask]

    whole_predicted_normal = pred_normal_global.clone()

    # true_mask, true_r, normals_traced, uvs = fine_mesh.ray_tracer.trace(cam_poses, ds)

    # model.eval()
    # render_iterations = 0
    # mask_iter = torch.ones(cam_poses.shape[0], dtype=torch.bool, device=cfg.device)
    # whole_intersected_mask = mask_iter.clone()
    # whole_predicted_r = torch.zeros(cam_poses.shape[0], device=cfg.device)
    # whole_predicted_normal = torch.zeros(cam_poses.shape, device=cfg.device)
    # x_src = cam_poses.clone()
    # while torch.sum(mask_iter) > 0 and render_iterations < 100:
    #     mask, t, normals, uvs_outer = outer_mesh.ray_tracer.trace(x_src[mask_iter], ds[mask_iter])
    #     x_src[mask_iter] = x_src[mask_iter] + ds[mask_iter] * t[:, None]

    #     if render_iterations == 0:
    #         whole_intersected_mask[mask_iter] = mask
    #     whole_mask_iter = mask_iter.clone()
    #     whole_mask_iter[mask_iter] = mask
    #     mask_iter = whole_mask_iter

    #     inner_mask, inner_t, _, uvs_inner = inner_mesh.ray_tracer.trace(x_src[mask_iter], ds[mask_iter])
    #     x_src_inner = torch.zeros_like(x_src[mask_iter])
    #     x_src_inner[inner_mask] = x_src[mask_iter][inner_mask] + ds[mask_iter][inner_mask] * inner_t[inner_mask][:, None]
    #     x_src_inner[~inner_mask] = 0
    #     inner_t[~inner_mask] = 0

    #     x_src_shifted = x_src[mask_iter] + ds[mask_iter] * 1e-3
    #     outer_mask, outer_t, _, uvs = outer_mesh.ray_tracer.trace(x_src_shifted, ds[mask_iter])
    #     x_src_outer = torch.zeros_like(x_src[mask_iter])
    #     x_src_outer[outer_mask] = x_src_shifted[outer_mask] + ds[mask_iter][outer_mask] * outer_t[outer_mask][:, None]
    #     x_src_outer[~outer_mask] = 0
    #     outer_t[~outer_mask] = 0
    #     x_src_inner[~inner_mask & outer_mask] = x_src_shifted[~inner_mask & outer_mask]

    #     x_src_shifted2 = x_src_shifted + ds[mask_iter] * outer_t[:, None]
    #     outer_mask2, outer_t2, _, uvs = outer_mesh.ray_tracer.trace(x_src_shifted, ds[mask_iter])

    #     predicted_intersection, predicted_r, predicted_normal = model(x_src[mask_iter], x_src_inner, ds[mask_iter], uvs_outer[mask])
    #     intersected_mask = predicted_intersection >= 0
    #     # intersected_mask = predicted_intersection >= -100
    #     whole_intersected_mask[mask_iter] = intersected_mask
    #     whole_predicted_r[mask_iter] = predicted_r
    #     whole_predicted_normal[mask_iter] = predicted_normal

    #     whole_mask_iter = mask_iter.clone()
    #     mask_iter_tmp = ~intersected_mask & inner_mask & outer_mask & (inner_t > outer_t2 + outer_t)
    #     #print(mask_iter_tmp.sum().item())
    #     whole_mask_iter[mask_iter] = mask_iter_tmp
    #     x_src[whole_mask_iter] = x_src_outer[mask_iter_tmp] + ds[mask_iter][mask_iter_tmp] * 1e-3
    #     mask_iter = whole_mask_iter
        
    #     render_iterations += 1

    # points = cam_poses[true_mask] + ds[true_mask] * true_r[true_mask][:, None] 
    # predicted_points = x_src[whole_intersected_mask] + ds[whole_intersected_mask] * whole_predicted_r[whole_intersected_mask][:, None]

    mse, psnr, accuracy = render_predictions(
        cfg, points, predicted_points, cam_poses, 
        normals_traced, whole_predicted_normal, 
        whole_intersected_mask, true_mask
    )
    return mse, psnr, accuracy



def train_model(cfg, fine_mesh, outer_mesh, inner_mesh, model, averaged_model, optimizer, scheduler, swa_scheduler, writer, model_config, run_name, step):
    mesh_min, mesh_max = outer_mesh.mesh.get_bounds()
    center = (mesh_min + mesh_max) / 2
    radius = np.sum((mesh_max - mesh_min) ** 2) ** 0.5 / 2
    print('Sample sphere radius =', radius)

    progress = tqdm(range(step, cfg.train.epochs))
    for epoch in progress:
        model.train()
        x, normals = sample_sphere_torch(radius + 0.1, center, cfg.train.sample_size, cfg.device)
        ds = sample_directions_torch(normals, cfg.device)

        #premask, t, _, uvs_outer = outer_mesh.ray_tracer.trace(x, ds, allow_negative=True)
        #x_src = x + ds * t[:, None]

        gt_intersection_mask_all = []
        gt_normals_all = []
        gt_distance_call = []

        input_enter_points_list = []
        input_exit_points_list = []
        input_directions = []

        # current point on the outer shell (enering another segment)
        x_outer_enter_mask, x_outer_enter_t, _, _ = outer_mesh.ray_tracer.trace(x, ds, allow_negative=False)
        x_outer_enter = x + ds * x_outer_enter_t[:, None]
        x_outer_enter = x_outer_enter[x_outer_enter_mask]

        # directions for rays left to trace
        ds_left = ds[x_outer_enter_mask]

        while len(x_outer_enter) > 0:
            # shift a bit into the outer shell
            x_outer_enter = x_outer_enter + ds_left * 1e-3

            # intersect orig mesh, inner shell and outer shell again 
            x_outer_exit_mask, x_outer_exit_t, _, _ = outer_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
            x_outer_exit = x_outer_enter + ds_left * x_outer_exit_t[:, None]

            if (~x_outer_exit_mask).sum() > 0:
                # print(f"Warning no escape from outer shell! ({(~x_outer_exit_mask).sum().item()} rays)")
                x_outer_exit_t[~x_outer_exit_mask] = 1e-3
                x_outer_exit[~x_outer_exit_mask] = x_outer_enter[~x_outer_exit_mask] + ds_left[~x_outer_exit_mask] * x_outer_exit_t[~x_outer_exit_mask][:, None]

            x_inner_mask, x_inner_t, _, _ = inner_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
            x_inner_enter = x_outer_enter + ds_left * x_inner_t[:, None]
            x_inner_enter[~x_inner_mask] = 0

            x_orig_mask, x_orig_t, x_orig_normals, _ = fine_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
            x_orig = x_outer_enter + ds_left * x_orig_t[:, None]
            x_orig[~x_orig_mask] = 0

            # ground truth for training
            gt_intersection_mask = x_orig_mask & (x_orig_t < x_outer_exit_t)
            gt_normals = x_orig_normals
            gt_distance = x_orig_t

            # input for model
            input_enter_points = x_outer_enter
            input_exit_points = torch.zeros_like(input_enter_points)
            input_exit_points[x_inner_mask] = x_inner_enter[x_inner_mask]
            input_exit_points[~x_inner_mask] = x_outer_exit[~x_inner_mask]

            # store results
            gt_intersection_mask_all.append(gt_intersection_mask)
            gt_normals_all.append(gt_normals)
            gt_distance_call.append(gt_distance)
            input_enter_points_list.append(input_enter_points)
            input_exit_points_list.append(input_exit_points)
            input_directions.append(ds_left)

            x_outer_exit = x_outer_exit + ds_left * 1e-3
            x_outer_enter_mask_new, x_outer_enter_t_new, _, _ = outer_mesh.ray_tracer.trace(x_outer_exit, ds_left, allow_negative=False)
            x_outer_enter_new = x_outer_exit + ds_left * x_outer_enter_t_new[:, None]

            # prepare for next iteration
            mask_for_remaining_rays = ~gt_intersection_mask & x_orig_mask & x_outer_exit_mask & x_outer_enter_mask_new
            x_outer_enter = x_outer_enter_new[mask_for_remaining_rays]
            ds_left = ds_left[mask_for_remaining_rays]

            # print('Remaining rays:', x_outer_enter.shape[0])
        
        gt_intersection_mask = torch.cat(gt_intersection_mask_all, dim=0)
        gt_normals = torch.cat(gt_normals_all, dim=0)
        gt_distance = torch.cat(gt_distance_call, dim=0)
        input_enter_points = torch.cat(input_enter_points_list, dim=0)
        input_exit_points = torch.cat(input_exit_points_list, dim=0)
        input_directions = torch.cat(input_directions, dim=0)

        predicted_intersection_mask, predicted_t, predicted_normals = model(
            input_enter_points,
            input_exit_points,
            input_directions,
        )

        cls_loss = F.binary_cross_entropy_with_logits(
            predicted_intersection_mask,
            gt_intersection_mask.float()
        )

        distance_loss = F.l1_loss(
            predicted_t[gt_intersection_mask],
            gt_distance[gt_intersection_mask]
        )

        normal_loss = -F.cosine_similarity(
            predicted_normals[gt_intersection_mask],
            gt_normals[gt_intersection_mask],
            dim=1,
            eps=1e-6
        ).mean()

        loss = cls_loss + normal_loss #+ distance_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if cfg.train.use_averaged_model in ['EMA', 'SWA']:
            averaged_model.update_parameters(model)
        else:
            averaged_model = model
        if cfg.train.use_averaged_model == 'SWA':
            swa_scheduler.step()
        scheduler.step()

        mse = 0
        psnr = 0
        accuracy = 0
        info = ""
        if epoch % cfg.visualization.render_interval == cfg.visualization.render_interval - 1:
            mse, psnr, accuracy = eval_model(cfg, fine_mesh, outer_mesh, inner_mesh, averaged_model)
            info += f"MSE={mse:.6f}, PSNR={psnr:.4f}"
        if epoch % cfg.train.checkpoints_interval == cfg.train.checkpoints_interval - 1:
            info += f" dist={distance_loss.item():.4f}, entr={cls_loss.item():.4f}, norm={normal_loss.item():.4f}, total={loss.item():.4f}"
            save_checkpoint(cfg, model_config, model, averaged_model, optimizer, scheduler, run_name, step)
        if info != "":
            progress.set_postfix_str(info)
        if cfg.train.tensorboard:
            writer.add_scalar("Total_loss", loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Normal_loss", normal_loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Distance_loss", distance_loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Entropy_loss", cls_loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch * cfg.train.sample_size)
            if mse > 0:
                writer.add_scalar("MSE", mse, epoch * cfg.train.sample_size)
            if psnr > 0:
                writer.add_scalar("PSNR", psnr, epoch * cfg.train.sample_size)            
            if accuracy > 0:
                writer.add_scalar("Accuracy", accuracy, epoch * cfg.train.sample_size)
            writer.flush()

    return model


def main(cfg):
    if cfg.train.model_start_checkpoint:
        checkpoint = torch.load(cfg.train.model_start_checkpoint, weights_only=False)
        model_config = checkpoint["model_config"]
    else:
        checkpoint = None
        model_config = cfg.model

    scale = cfg.scale
    fine_mesh = MeshWrapper.from_file(cfg.fine_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=scale)
    outer_mesh = MeshWrapper.from_file(cfg.outer_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=scale)
    inner_mesh = MeshWrapper.from_file(cfg.inner_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=scale)

    os.makedirs(cfg.visualization.preview_path, exist_ok=True)
    os.makedirs(cfg.visualization.render_path, exist_ok=True)
    os.makedirs(cfg.train.checkpoints_path, exist_ok=True)

    save_mesh_previews({
        f"{cfg.visualization.preview_path}/{cfg.visualization.fine_mesh_preview_name}":  fine_mesh.mesh,
        f"{cfg.visualization.preview_path}/{cfg.visualization.outer_mesh_preview_name}": outer_mesh.mesh,
        f"{cfg.visualization.preview_path}/{cfg.visualization.inner_mesh_preview_name}": inner_mesh.mesh,
    }, cfg.visualization.image_size)

    model = RayModel(model_config, outer_mesh.mesh).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    learing_rate_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=cfg.train.learning_rate * cfg.train.learning_rate_scheduler_min,
    )

    swa_scheduler = None
    if cfg.train.use_averaged_model == 'EMA':
        averaged_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.train.ema_decay))
    elif cfg.train.use_averaged_model == 'SWA':
        averaged_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.train.swa_learing_rate)
    else:
        averaged_model = None

    step = 0
    if checkpoint:
        print("Loading model state...")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        learing_rate_scheduler.load_state_dict(checkpoint["learning_rate_scheduler"])
        step = checkpoint["step"]

        if cfg.train.use_averaged_model in ['EMA', 'SWA']:
            averaged_model.load_state_dict(checkpoint["averaged_model"])
    
    writer = None
    run_name = f"run_{int(time.time())}"
    if cfg.train.tensorboard:
        if cfg.train.tensorboard_run_name:
            run_name = cfg.train.tensorboard_run_name
        writer = SummaryWriter(f"{cfg.train.tensorboard_path}/{run_name}")

    model = train_model(
        cfg, fine_mesh, outer_mesh, inner_mesh,
        model, averaged_model, optimizer, 
        learing_rate_scheduler, swa_scheduler, 
        writer, model_config, run_name, step
    )

    eval_model(cfg, fine_mesh, outer_mesh, inner_mesh, model)
    save_checkpoint(cfg, model_config, model, averaged_model, optimizer, learing_rate_scheduler, run_name, cfg.train.epochs)
    

if __name__ == "__main__":
    import config as cfg_module
    main(cfg_module.cfg)
