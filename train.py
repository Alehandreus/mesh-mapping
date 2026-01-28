import os
import numpy as np
import torch
import torch.autograd as autograd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
import math
import copy
from tqdm import tqdm
from utils import MeshWrapper, sample_directions, sample_sphere, get_camera_rays
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
        #averaged_model.module.parameters.data.cpu().numpy().astype(np.float16).tofile(f"{cfg.train.checkpoints_path}/{run_name}.bin")
    else:
        copy_model = copy.deepcopy(model)
        #model.parameters.data.cpu().numpy().astype(np.float16).tofile(f"{cfg.train.checkpoints_path}/{run_name}.bin")
    copy_model.half()
    torch.save(copy_model.state_dict(), f"{cfg.train.checkpoints_path}/{run_name}_half.pt")
    
    checkpoint_name = f"{cfg.train.checkpoints_path}/{run_name}.pt"
    torch.save(checkpoint_data, checkpoint_name)

    

@torch.no_grad()
def eval_model(cfg, fine_mesh, outer_mesh, model):
    cam_poses, ds = get_camera_rays(fine_mesh.mesh, img_size=cfg.visualization.image_size, device=cfg.device)
    ds = ds / ds.norm(dim=1, keepdim=True)
    mask, t, normals = outer_mesh.ray_tracer.trace(cam_poses, ds)
    x_src = cam_poses + ds * t[:, None]

    model.eval()
    predicted_intersection, predicted_r, predicted_normal = model(x_src[mask], ds[mask])

    intersected_mask = predicted_intersection >= 0
    whole_intesected_mask = mask.clone()
    whole_intesected_mask[mask] = intersected_mask

    true_mask, true_r, normals_traced = fine_mesh.ray_tracer.trace(x_src, ds)

    points = x_src[true_mask] + ds[true_mask] * true_r[true_mask][:, None] 
    predicted_points = x_src[whole_intesected_mask] + ds[whole_intesected_mask] * predicted_r[intersected_mask][:, None]

    mse, psnr = render_predictions(
        cfg, points, predicted_points, cam_poses, 
        normals_traced, predicted_normal, 
        intersected_mask, whole_intesected_mask, true_mask
    )
    return mse, psnr



def train_model(cfg, fine_mesh, outer_mesh, model, averaged_model, optimizer, scheduler, swa_scheduler, writer, model_config, run_name, step):
    mesh_min, mesh_max = outer_mesh.mesh.get_bounds()
    center = (mesh_min + mesh_max) / 2  #mesh-utils return multiplied by 100 value and i don't know why
    radius = np.max(mesh_max - mesh_min) / 2
    print('Sample sphere radius =', radius)

    progress = tqdm(range(step, cfg.train.epochs))
    for epoch in progress:
        model.train()
        x, normals = sample_sphere(radius + 0.1, center, cfg.train.sample_size)
        x = torch.tensor(x, device=cfg.device)
        ds = sample_directions(normals, cfg.device)
       
        _, t, _ = outer_mesh.ray_tracer.trace(x, ds)
        x_src = x + ds * t[:, None]

        mask, true_r, normals_traced = fine_mesh.ray_tracer.trace(x_src, ds)
        #mask = mask.clone()
        #true_r = true_r.clone()
        mask[true_r < 0] = 0
        true_intersection = mask.to(torch.float16)

        predicted_intersection, predicted_r, predicted_normal = model(x_src, ds)
        intersection_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        intersected_mask = predicted_intersection > 0
        entropy = intersection_loss(predicted_intersection, true_intersection)
        weights = torch.ones_like(mask, dtype=torch.float32) + (intersected_mask & mask) * math.exp(epoch / 50000)
        entropy = (entropy * weights).mean()

        #distance = ((predicted_r[mask] - true_r[mask]) ** 2).mean()
        distance = (predicted_r[mask] - true_r[mask]).abs().mean() 

        normal_error = -torch.nn.functional.cosine_similarity(predicted_normal[mask], normals_traced[mask], dim=1, eps=1e-6).mean() + 1.0

        loss = 0.01 * distance + normal_error + entropy

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
        info = ""
        if epoch % cfg.visualization.render_interval == cfg.visualization.render_interval - 1:
            mse, psnr = eval_model(cfg, fine_mesh, outer_mesh, averaged_model)
            info += f"MSE={mse:.6f}, PSNR={psnr:.4f}"
        if epoch % cfg.train.checkpoints_interval == cfg.train.checkpoints_interval - 1:
            info += f" dist={distance.item():.4f}, entr={entropy.item():.4f}, norm={normal_error.item():.4f}, total={loss.item():.4f}, insc_true={mask.sum().item()}, insc_pred={intersected_mask.sum().item()}"
            save_checkpoint(cfg, model_config, model, averaged_model, optimizer, scheduler, run_name, step)
        if info != "":
            progress.set_postfix_str(info)
        if cfg.train.tensorboard:
            writer.add_scalar("Total_loss", loss.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Normal_loss", normal_error.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Distance_loss", distance.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Entropy_loss", entropy.item(), epoch * cfg.train.sample_size)
            writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch * cfg.train.sample_size)
            if mse > 0:
                writer.add_scalar("MSE", mse, epoch * cfg.train.sample_size)
            if psnr > 0:
                writer.add_scalar("PSNR", psnr, epoch * cfg.train.sample_size)
            writer.flush()

    return model


def main(cfg):
    if cfg.train.model_start_checkpoint:
        checkpoint = torch.load(cfg.train.model_start_checkpoint, weights_only=False)
        model_config = checkpoint["model_config"]
    else:
        checkpoint = None
        model_config = cfg.model

    print()
    fine_mesh = MeshWrapper.from_file(cfg.fine_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=0.01)
    outer_mesh = MeshWrapper.from_file(cfg.outer_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=0.01)

    os.makedirs(cfg.visualization.preview_path, exist_ok=True)
    os.makedirs(cfg.visualization.render_path, exist_ok=True)
    os.makedirs(cfg.train.checkpoints_path, exist_ok=True)

    save_mesh_previews({
        f"{cfg.visualization.preview_path}/{cfg.visualization.fine_mesh_preview_name}":  fine_mesh.mesh,
        f"{cfg.visualization.preview_path}/{cfg.visualization.outer_mesh_preview_name}": outer_mesh.mesh,
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
        cfg, fine_mesh, outer_mesh, 
        model, averaged_model, optimizer, 
        learing_rate_scheduler, swa_scheduler, 
        writer, model_config, run_name, step
    )

    eval_model(cfg, fine_mesh, outer_mesh, model)
    save_checkpoint(cfg, model_config, model, averaged_model, optimizer, learing_rate_scheduler, run_name, cfg.train.epochs)
    

if __name__ == "__main__":
    import config as cfg_module
    main(cfg_module.cfg)
