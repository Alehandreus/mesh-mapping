from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import torch
import torch.autograd as autograd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
import copy
from mesh_utils import Mesh
from utils import sample_points, point_query, MeshWrapper, ShellType, save_mesh_previews

from models import DisplacementModel
import time

# to make tensorflow shut up
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter

from render import render_entry


@torch.no_grad()
def build_mapped_mesh(net, orig_mesh, rough_mesh, device, cfg):
    orig_vertices = orig_mesh.mesh.get_vertices()
    orig_vertices = torch.from_numpy(orig_vertices).float().to(device)
    _, sdf_closests, sdf_barycentrics, sdf_face_idxs = point_query(rough_mesh.traverser, orig_vertices, device)
    mapped_vertices = net(x=sdf_closests, barycentrics=sdf_barycentrics, face_idxs=sdf_face_idxs).detach().cpu().numpy()
    mapped_mesh = Mesh.from_data(mapped_vertices, orig_mesh.mesh.get_faces())
    return mapped_mesh


def train_single_shell_epoch(cfg, writer, step, model, aver_model, rough_mesh, orig_mesh, optimizer, lr_scheduler, swa_scheduler, shell_type):    
    model.train()
    print(f"Training one {shell_type.name} epoch...")
    last_time = time.time()
    for it in range(step, step + cfg.train.steps_per_epoch):
        x, barycentrics, face_idxs = sample_points(rough_mesh.sampler, cfg.train.batch_size, cfg.device)
        _, y, _, _ = point_query(orig_mesh.traverser, x, cfg.device)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x=x, barycentrics=barycentrics, face_idxs=face_idxs)

        loss_cosine = (1.0 - torch.nn.functional.cosine_similarity(y_pred - x, y - x, dim=1, eps=1e-6)).mean()
        loss_angle = torch.acos(
            torch.clamp(
                torch.nn.functional.cosine_similarity(y_pred - x, y - x, dim=1, eps=1e-7),
                -1.0 + 1e-7,
                1.0 - 1e-7,
            )
        ).mean()
        loss_length = (y_pred.norm(dim=1) - y.norm(dim=1)).abs().mean()

        # loss = loss_cosine + loss_length * 2
        # loss = loss_angle #+ loss_length * 2
        loss = loss_angle + loss_length * 0.001
        # loss = (y_pred - y).abs().sum(dim=1).mean()
        # loss = (y_pred - y).square().sum(dim=1).mean() * 1000

        loss.backward()
        optimizer.step()
        if cfg.train.use_averaged_model in ['EMA', 'SWA']:
            aver_model.update_parameters(model)
        if cfg.train.use_averaged_model == 'SWA':
            swa_scheduler.step()
        lr_scheduler.step()

        if (it % cfg.train.print_interval == 0):
            time_diff = time.time() - last_time
            last_time = time.time()
            print(
                f"[{shell_type.name} it {it:07d}] "
                f"loss={loss.item():.10f}; cosine={loss_cosine.item():.10f}; "
                f"length={loss_length.item():.10f}; time={time_diff:.2f}s"
            )
        
        if cfg.train.tensorboard and (it % cfg.train.tensorboard_interval == 0):
            writer.add_scalar(f"{shell_type.name}/Loss/total", loss.item(), it)
            writer.add_scalar(f"{shell_type.name}/Loss/cosine", loss_cosine.item(), it)
            writer.add_scalar(f"{shell_type.name}/Loss/length", loss_length.item(), it)
            writer.add_scalar(f"{shell_type.name}/LR/lr", lr_scheduler.get_last_lr()[0], it)
            writer.flush()
    
    mapped_mesh = build_mapped_mesh(model, orig_mesh, rough_mesh, cfg.device, cfg)
    save_mesh_previews({f"{cfg.train.previews_dir}/{shell_type.name}_mapped_mesh.png": mapped_mesh})


def train_entry(cfg):
    if cfg.train.model_checkpoint is not None:
        checkpoint = torch.load(cfg.train.model_checkpoint, weights_only=False)
        cfg_model = checkpoint["cfg_model"]
    else:
        checkpoint = None
        cfg_model = cfg.model

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

    inner_step = 0
    outer_step = 0
            
    ##### model ####
    inner_net = DisplacementModel(cfg_model, inner_mesh.mesh).to(cfg.device)
    outer_net = DisplacementModel(cfg_model, outer_mesh.mesh).to(cfg.device)

    if cfg.train.use_averaged_model == 'EMA':
        aver_inner_net = AveragedModel(inner_net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.train.ema_decay))
        aver_outer_net = AveragedModel(outer_net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.train.ema_decay))
    elif cfg.train.use_averaged_model == 'SWA':
        aver_inner_net = AveragedModel(inner_net)
        aver_outer_net = AveragedModel(outer_net)
    else:
        aver_inner_net = None
        aver_outer_net = None

    if checkpoint is not None:
        print("Loading model state...")
        inner_net.load_state_dict(checkpoint["inner_net"]["model"])
        outer_net.load_state_dict(checkpoint["outer_net"]["model"])
        inner_step = checkpoint["inner_net"]["step"]
        outer_step = checkpoint["outer_net"]["step"]
        if cfg.train.use_averaged_model in ['EMA', 'SWA']:
            aver_inner_net.load_state_dict(checkpoint["aver_inner_net"])
            aver_outer_net.load_state_dict(checkpoint["aver_outer_net"])

    #### optimizer ####
    optimizer_inner = torch.optim.Adam(inner_net.parameters(), lr=cfg.train.lr)
    optimizer_outer = torch.optim.Adam(outer_net.parameters(), lr=cfg.train.lr)
    if checkpoint is not None:
        print("Loading optimizer state...")
        optimizer_inner.load_state_dict(checkpoint["inner_net"]["optimizer"])
        optimizer_outer.load_state_dict(checkpoint["outer_net"]["optimizer"])

    #### lr scheduler ####
    lr_scheduler_inner = CosineAnnealingLR(
        optimizer_inner,
        T_max=cfg.train.steps_total,
        eta_min=cfg.train.lr * cfg.train.lr_scheduler_min,
    )
    lr_scheduler_outer = CosineAnnealingLR(
        optimizer_outer,
        T_max=cfg.train.steps_total,
        eta_min=cfg.train.lr * cfg.train.lr_scheduler_min,
    )

    if cfg.train.use_averaged_model == 'SWA':
        swa_scheduler_inner = SWALR(optimizer_inner, swa_lr=cfg.train.swa_lr)
        swa_scheduler_outer = SWALR(optimizer_outer, swa_lr=cfg.train.swa_lr)
    else:
        swa_scheduler_inner = None
        swa_scheduler_outer = None

    #### tensorboard ####
    writer = None
    if cfg.train.tensorboard:
        run_name = cfg.train.run_name
        if run_name is None:
            run_name = f"run_{int(time.time())}"
        writer = SummaryWriter(f"tensorboard/{run_name}")

    os.makedirs(cfg.train.checkpoints_dir, exist_ok=True)

    if checkpoint is not None:
        random_idx = time.time()
        cfg_for_render = copy.deepcopy(cfg)
        cfg_for_render.render.model_checkpoint = cfg.train.model_checkpoint
        cfg_for_render.render.use_inner = True
        cfg_for_render.render.use_outer = True
        render_logs = render_entry(cfg_for_render)

        mse_nogd_inner = render_logs["mse_nogd_inner"]
        mse_nogd_outer = render_logs["mse_nogd_outer"]
        psnr_nogd_inner = render_logs["psnr_nogd_inner"]
        psnr_nogd_outer = render_logs["psnr_nogd_outer"]

        print(f"[VALIDATION] Inner shell Pixel MSE (no GD): {mse_nogd_inner}")
        print(f"[VALIDATION] Inner shell Pixel PSNR (no GD): {psnr_nogd_inner}")
        print(f"[VALIDATION] Outer shell Pixel MSE (no GD): {mse_nogd_outer}")
        print(f"[VALIDATION] Outer shell Pixel PSNR (no GD): {psnr_nogd_outer}")

        if cfg.train.tensorboard:
            writer.add_scalar(f"Validation/Inner_MSE_nogd", mse_nogd_inner, inner_step + outer_step)
            writer.add_scalar(f"Validation/Outer_MSE_nogd", mse_nogd_outer, inner_step + outer_step)
            writer.add_scalar(f"Validation/Inner_PSNR_nogd", psnr_nogd_inner, inner_step + outer_step)
            writer.add_scalar(f"Validation/Outer_PSNR_nogd", psnr_nogd_outer, inner_step + outer_step)
            writer.flush()
    
    ##### outer training loop #####
    for _ in range(cfg.train.steps_total // cfg.train.steps_per_epoch):
        if cfg.train.train_inner:
            train_single_shell_epoch(
                cfg, writer,
                inner_step, inner_net, aver_inner_net,
                inner_mesh, orig_mesh,
                optimizer_inner, lr_scheduler_inner, swa_scheduler_inner,
                ShellType.INNER,
            )
            inner_step += cfg.train.steps_per_epoch

        if cfg.train.train_outer:
            train_single_shell_epoch(
                cfg, writer,
                outer_step, outer_net, aver_outer_net,
                outer_mesh, orig_mesh,
                optimizer_outer, lr_scheduler_outer, swa_scheduler_outer,
                ShellType.OUTER,
            )
            outer_step += cfg.train.steps_per_epoch

        checkpoint_data = {
            "cfg_model": cfg_model,
            "inner_net": {
                "model": inner_net.state_dict(),
                "optimizer": optimizer_inner.state_dict(),
                "step": inner_step,
            },
            "outer_net": {
                "model": outer_net.state_dict(),
                "optimizer": optimizer_outer.state_dict(),
                "step": outer_step,
            },
        }

        if cfg.train.use_averaged_model in ['EMA', 'SWA']:
            checkpoint_data["aver_inner_net"] = aver_inner_net.state_dict()
            checkpoint_data["aver_outer_net"] = aver_outer_net.state_dict()

        random_idx = time.time()
        checkpoint_name = f"/tmp/{random_idx}.pt"
        torch.save(checkpoint_data, checkpoint_name)
        cfg_for_render = copy.deepcopy(cfg)
        cfg_for_render.render.model_checkpoint = checkpoint_name
        cfg_for_render.render.use_inner = True
        cfg_for_render.render.use_outer = True
        render_logs = render_entry(cfg_for_render)

        mse_nogd_inner = render_logs["mse_nogd_inner"]
        mse_nogd_outer = render_logs["mse_nogd_outer"]
        psnr_nogd_inner = render_logs["psnr_nogd_inner"]
        psnr_nogd_outer = render_logs["psnr_nogd_outer"]

        print(f"[VALIDATION] Inner shell Pixel MSE (no GD): {mse_nogd_inner}")
        print(f"[VALIDATION] Inner shell Pixel PSNR (no GD): {psnr_nogd_inner}")
        print(f"[VALIDATION] Outer shell Pixel MSE (no GD): {mse_nogd_outer}")
        print(f"[VALIDATION] Outer shell Pixel PSNR (no GD): {psnr_nogd_outer}")

        if cfg.train.tensorboard:
            writer.add_scalar(f"Validation/Inner_MSE_nogd", mse_nogd_inner, inner_step + outer_step)
            writer.add_scalar(f"Validation/Outer_MSE_nogd", mse_nogd_outer, inner_step + outer_step)
            writer.add_scalar(f"Validation/Inner_PSNR_nogd", psnr_nogd_inner, inner_step + outer_step)
            writer.add_scalar(f"Validation/Outer_PSNR_nogd", psnr_nogd_outer, inner_step + outer_step)
            writer.flush()

        checkpoint_name = f"{cfg.train.checkpoints_dir}/{inner_step}_{outer_step}_{cfg.mesh_name}_{mse_nogd_inner * 100:.2f}_{mse_nogd_outer * 100:.2f}.pt"
        torch.save(checkpoint_data, checkpoint_name)

        inner_net.network.params.data.cpu().numpy().astype(np.float16).tofile(f"{cfg.train.checkpoints_dir}/inner_params.bin")
        outer_net.network.params.data.cpu().numpy().astype(np.float16).tofile(f"{cfg.train.checkpoints_dir}/outer_params.bin")

        print(f"Saved checkpoint to {checkpoint_name}")


if __name__ == "__main__":
    import cfg as cfg_module
    train_entry(cfg_module.cfg)
