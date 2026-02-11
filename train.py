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
from visualization import save_mesh_previews, render_predictions, render_predictions_with_neural_renderer

from models import RayModel
import time

# to make tensorflow shut up
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter

eps = 1e-8


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

    cam_poses, ds = get_camera_rays(fine_mesh.mesh, img_size=cfg.visualization.image_size, device=cfg.device, distance_scale=1.0, angle=cfg.visualization.camera_angle)
    ds = ds / ds.norm(dim=1, keepdim=True)

    x_orig_mask, x_orig_t, x_orig_normals, _, colors = fine_mesh.ray_tracer.trace(cam_poses, ds, allow_negative=False)
    x_orig = cam_poses + ds * x_orig_t[:, None]
    x_orig[~x_orig_mask] = 0

    pred_intersection_global = torch.zeros((cam_poses.shape[0],), dtype=torch.bool, device=cfg.device)
    pred_t_global = torch.zeros((cam_poses.shape[0],), dtype=torch.float32, device=cfg.device)
    pred_normal_global = torch.zeros((cam_poses.shape[0], 3), dtype=torch.float32, device=cfg.device)
    pred_colors_global = torch.zeros((cam_poses.shape[0], 3), dtype=torch.float32, device=cfg.device)

    x_outer_enter_mask, x_outer_enter_t, _, _, _ = outer_mesh.ray_tracer.trace(
        cam_poses, 
        ds,
        allow_negative=False,
        allow_backward=False,
        allow_forward=True,        
    )
    x_outer_enter = cam_poses + ds * x_outer_enter_t[:, None]
    x_outer_enter = x_outer_enter[x_outer_enter_mask]

    mask_for_remaining_rays_global = x_outer_enter_mask.clone()

    # directions for rays left to trace
    ds_left = ds[x_outer_enter_mask]

    # distance already accumulated along traced segments
    accum_t = x_outer_enter_t[x_outer_enter_mask]

    i = 0
    while len(x_outer_enter) > 0:
        x_outer_enter = x_outer_enter + ds_left * eps

        # intersect orig mesh, inner shell and outer shell again 
        x_outer_exit_mask, x_outer_exit_t, _, _, _ = outer_mesh.ray_tracer.trace(
            x_outer_enter,
            ds_left,
            allow_negative=False,
            allow_backward=True,
            allow_forward=False,            
        )
        x_outer_exit = x_outer_enter + ds_left * x_outer_exit_t[:, None]

        if (~x_outer_exit_mask).sum() > 0:
            # print(f"Warning no escape from outer shell! ({(~x_outer_exit_mask).sum().item()} rays)")
            x_outer_exit_t[~x_outer_exit_mask] = 1e-8
            x_outer_exit[~x_outer_exit_mask] = x_outer_enter[~x_outer_exit_mask] + ds_left[~x_outer_exit_mask] * x_outer_exit_t[~x_outer_exit_mask][:, None]

        x_inner_mask, x_inner_t, _, _, _ = inner_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=True)
        x_inner_enter = x_outer_enter + ds_left * x_inner_t[:, None]
        x_inner_enter[~x_inner_mask] = 0

        # a = (x_inner_t < 0).sum().item()
        # if a > 0:
        #     print(f"Warning {a} rays starting inside inner shell during eval!")

        input_enter_points = x_outer_enter
        input_exit_points = torch.zeros_like(input_enter_points)
        inner_apply = x_inner_mask & (x_inner_t < x_outer_exit_t)
        input_exit_points[inner_apply] = x_inner_enter[inner_apply]
        input_exit_points[~inner_apply] = x_outer_exit[~inner_apply]
        input_directions = ds_left

        pred_intersection, pred_t, pred_normal, pred_colors = model(
            input_enter_points,
            input_exit_points,
            input_directions,
        )
        pred_intersection_mask = (pred_intersection >= 0)
        pred_intersection_mask[x_inner_mask & (x_inner_t < x_outer_exit_t)] = True
        # pred_intersection_mask = pred_intersection_mask | True
        # pred_normal[x_inner_t < 0] = x_orig_normals[mask_for_remaining_rays_global][x_inner_t < 0]

        pred_intersection_global[mask_for_remaining_rays_global] = pred_intersection_mask
        pred_t_global[mask_for_remaining_rays_global] = pred_t + accum_t
        pred_normal_global[mask_for_remaining_rays_global] = pred_normal
        pred_colors_global[mask_for_remaining_rays_global] = pred_colors

        x_outer_exit = x_outer_exit + ds_left * eps
        x_outer_enter_mask_new, x_outer_enter_t_new, _, _, _ = outer_mesh.ray_tracer.trace(
            x_outer_exit,
            ds_left,
            allow_negative=False,
            allow_backward=False,
            allow_forward=True,
        )
        x_outer_enter_new = x_outer_exit + ds_left * x_outer_enter_t_new[:, None]        

        # prepare for next iteration
        mask_for_remaining_rays = ~pred_intersection_mask & (x_outer_enter_mask_new | x_inner_mask)
        mask_for_remaining_rays_global[mask_for_remaining_rays_global.clone()] = mask_for_remaining_rays
        x_outer_enter = x_outer_enter_new[mask_for_remaining_rays]
        ds_left = ds_left[mask_for_remaining_rays]
        accum_t = accum_t[mask_for_remaining_rays] + x_outer_exit_t[mask_for_remaining_rays] + x_outer_enter_t_new[mask_for_remaining_rays]

        # print("Remaining rays to trace:", x_outer_enter.shape[0])

        i += 1
        if i > 10:
            # print(f"Breaking after {i - 1} iterations, remaining rays: {x_outer_enter.shape[0]}")
            break

    # x_inner_mask, x_inner_t, _, _ = inner_mesh.ray_tracer.trace(cam_poses, ds, allow_negative=True)
    # pred_intersection_global[x_inner_mask] = 1.0
    # pred_normal_global[x_inner_mask] = x_orig_normals[x_inner_mask]

    # prepare inputs for rendering

    true_mask = x_orig_mask.clone()
    points = cam_poses[true_mask] + ds[true_mask] * x_orig_t[true_mask][:, None]
    normals_traced = x_orig_normals.clone()

    whole_intersected_mask = pred_intersection_global.clone()

    predicted_points = (cam_poses + ds * pred_t_global[:, None])[whole_intersected_mask]

    whole_predicted_normal = pred_normal_global.clone()

    mse, psnr, accuracy = render_predictions(
        cfg, points, predicted_points, cam_poses, 
        normals_traced, whole_predicted_normal, 
        whole_intersected_mask, true_mask,
        pred_colors_global, colors,
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

        gt_intersection_mask_all = []
        gt_normals_all = []
        gt_distance_all = []
        gt_colors_all = []

        input_enter_points_list = []
        input_exit_points_list = []
        input_directions = []

        # current point on the outer shell (enering another segment)
        x_outer_enter_mask, x_outer_enter_t, _, _, _ = outer_mesh.ray_tracer.trace(x, ds, allow_negative=False, allow_backward=False)
        x_outer_enter = x + ds * x_outer_enter_t[:, None]
        x_outer_enter = x_outer_enter[x_outer_enter_mask]

        # directions for rays left to trace
        ds_left = ds[x_outer_enter_mask]

        i = 0
        while len(x_outer_enter) > 10:
            # shift a bit into the outer shell
            x_outer_enter = x_outer_enter + ds_left * eps

            # intersect orig mesh, inner shell and outer shell again 
            x_outer_exit_mask, x_outer_exit_t, x_outer_exit_normals, _, _ = outer_mesh.ray_tracer.trace(
                x_outer_enter,
                ds_left,
                allow_negative=False,
                allow_backward=True,
                allow_forward=False,
            )
            x_outer_exit = x_outer_enter + ds_left * x_outer_exit_t[:, None]

            # dot = (ds_left * x_outer_exit_normals).sum(dim=1)
            # print(f"{(dot < 0).sum().item()} rays exiting outer shell have incorrect normal direction")

            if (~x_outer_exit_mask).sum() > 0:
                # print(f"Warning no escape from outer shell! ({(~x_outer_exit_mask).sum().item()} rays)")
                x_outer_exit_t[~x_outer_exit_mask] = 1e-3
                x_outer_exit[~x_outer_exit_mask] = x_outer_enter[~x_outer_exit_mask] + ds_left[~x_outer_exit_mask] * x_outer_exit_t[~x_outer_exit_mask][:, None]

            x_inner_mask, x_inner_t, _, _, _ = inner_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
            x_inner_enter = x_outer_enter + ds_left * x_inner_t[:, None]
            x_inner_enter[~x_inner_mask] = 0

            x_orig_mask, x_orig_t, x_orig_normals, _, colors = fine_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
            x_orig = x_outer_enter + ds_left * x_orig_t[:, None]
            x_orig[~x_orig_mask] = 0

            # ground truth for training
            gt_intersection_mask = x_orig_mask & (x_orig_t < x_outer_exit_t)
            gt_normals = x_orig_normals
            gt_distance = x_orig_t
            gt_colors = colors

            # input for model
            input_enter_points = x_outer_enter
            input_exit_points = torch.zeros_like(input_enter_points)
            inner_apply = x_inner_mask & (x_inner_t < x_outer_exit_t)
            input_exit_points[inner_apply] = x_inner_enter[inner_apply]
            input_exit_points[~inner_apply] = x_outer_exit[~inner_apply]

            # store results
            gt_intersection_mask_all.append(gt_intersection_mask)
            gt_normals_all.append(gt_normals)
            gt_distance_all.append(gt_distance)
            gt_colors_all.append(gt_colors)
            input_enter_points_list.append(input_enter_points)
            input_exit_points_list.append(input_exit_points)
            input_directions.append(ds_left)

            x_outer_exit = x_outer_exit + ds_left * eps
            x_outer_enter_mask_new, x_outer_enter_t_new, _, _, _ = outer_mesh.ray_tracer.trace(
                x_outer_exit,
                ds_left,
                allow_negative=False,
                allow_backward=False,
                allow_forward=True,                
            )
            x_outer_enter_new = x_outer_exit + ds_left * x_outer_enter_t_new[:, None]

            # prepare for next iteration
            mask_for_remaining_rays = ~gt_intersection_mask & x_orig_mask & x_outer_exit_mask & x_outer_enter_mask_new
            x_outer_enter = x_outer_enter_new[mask_for_remaining_rays]
            ds_left = ds_left[mask_for_remaining_rays]

            i += 1
            if i > 10:
                break
            # print('Remaining rays:', x_outer_enter.shape[0])
        
        gt_intersection_mask = torch.cat(gt_intersection_mask_all, dim=0)
        gt_normals = torch.cat(gt_normals_all, dim=0)
        gt_distance = torch.cat(gt_distance_all, dim=0)
        gt_colors = torch.cat(gt_colors_all, dim=0)
        input_enter_points = torch.cat(input_enter_points_list, dim=0)
        input_exit_points = torch.cat(input_exit_points_list, dim=0)
        input_directions = torch.cat(input_directions, dim=0)

        predicted_intersection_mask, predicted_t, predicted_normals, predicted_colors = model(
            input_enter_points,
            input_exit_points,
            input_directions,
        )

        color_loss = F.mse_loss(
            predicted_colors[gt_intersection_mask],
            gt_colors[gt_intersection_mask],
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

        loss = cls_loss + normal_loss + color_loss * 100 #+ distance_loss

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
        flip = 0
        info = ""
        if epoch % cfg.train.evaluation_interval == cfg.train.evaluation_interval - 1:
            save_checkpoint(cfg, model_config, model, averaged_model, optimizer, scheduler, run_name, step)
            if cfg.visualization.use_neural_renderer:
                psnr, flip = render_predictions_with_neural_renderer(cfg, run_name)
                info += f"PSNR={psnr:.4f}, FLIP={flip:.4f}, "
            else:
                mse, psnr, accuracy = eval_model(cfg, fine_mesh, outer_mesh, inner_mesh, averaged_model)
                info += f"MSE={mse:.6f}, PSNR={psnr:.4f}, "
            info += f"dist={distance_loss.item():.4f}, entr={cls_loss.item():.4f}, norm={normal_loss.item():.4f}, total={loss.item():.4f}"
        if info != "":
            progress.set_postfix_str(info)
        if cfg.train.tensorboard:
            writer.add_scalar("Total_loss", loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Normal_loss", normal_loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Distance_loss", distance_loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Entropy_loss", cls_loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch * cfg.train.sample_size)
            writer.add_scalar("Color_loss", color_loss.item(), epoch * cfg.train.sample_size)
            if mse > 0:
                writer.add_scalar("MSE", mse, epoch * cfg.train.sample_size)
            if psnr > 0:
                writer.add_scalar("PSNR", psnr, epoch * cfg.train.sample_size)            
            if accuracy > 0:
                writer.add_scalar("Accuracy", accuracy, epoch * cfg.train.sample_size)
            if flip > 0:
                writer.add_scalar("FLIP", flip, epoch * cfg.train.sample_size)
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
