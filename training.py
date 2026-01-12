from dataclasses import dataclass
from typing import Optional

import torch
import torch.autograd as autograd

from mesh_utils import Mesh
from utils import sample_points, point_query

from mesh_data import PyMesh


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    # lr = 1e-4
    # epochs: int = 2_000
    epochs: int = 20_000
    # epochs: int = 100_000
    # epochs: int = 4_000_000
    # batch_size: int = 100_000
    batch_size: int = 50_000
    log_interval: int = 200
    n_sample_points: int = 10_000
    out_obj: str = "sampled_points.obj"
    weight_decay: float = 1.0
    load_optimizer: bool = False


def train_residual_map(
    net: torch.nn.Module,
    orig_mesh: PyMesh,
    rough_mesh: PyMesh,
    cfg: Optional[TrainingConfig] = None,
) -> None:
    """Supervise net to map rough_mesh surface points onto orig_mesh surface."""
    cfg = cfg or TrainingConfig()
    device = next(net.parameters()).device

    net.train()
    # optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)

    if cfg.load_optimizer:
        print("Loading optimizer state...")
        optimizer.load_state_dict(torch.load("optimizer_state.pt"))
        net.load_state_dict(torch.load("mapping.pt")["inner_net"])

    # import tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    import time
    last_time = time.time()

    print("Starting training...")
    for it in range(1, cfg.epochs + 1):
        x, barycentrics, face_idxs = sample_points(rough_mesh.sampler, cfg.batch_size, device)
        _, y, _, _ = point_query(orig_mesh.traverser, x, device)

        optimizer.zero_grad(set_to_none=True)
        y_pred = net(x=x, barycentrics=barycentrics, face_idxs=face_idxs)

        loss_cosine = (1.0 - torch.nn.functional.cosine_similarity(y_pred - x, y - x, dim=1, eps=1e-6)).mean()
        loss_angle = torch.acos(
            torch.clamp(
                torch.nn.functional.cosine_similarity(y_pred - x, y - x, dim=1, eps=1e-7),
                -1.0 + 1e-7,
                1.0 - 1e-7,
            )
        ).mean()
        loss_length = (y_pred.norm(dim=1) - y.norm(dim=1)).abs().mean()
        # loss_length_sq = (y_pred.norm(dim=1) - y.norm(dim=1)).square().mean()

        # loss = loss_cosine + loss_length * 2
        # loss = loss_angle #+ loss_length * 2
        loss = loss_angle + loss_length * 100# * 0.1

        # loss = (y_pred - y).abs().sum(dim=1).mean()
        # loss = (y_pred - y).square().sum(dim=1).mean() * 1000
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if it % cfg.log_interval == 0:
            print(f"[it {it:05d}] loss={loss.item():.10f}; cosine={loss_cosine.item():.10f}; length={loss_length.item():.10f}; time={(time.time() - last_time):.2f}s")
            last_time = time.time()
            writer.add_scalar("Loss/total", loss.item(), it)
            writer.add_scalar("Loss/cosine", loss_cosine.item(), it)
            writer.add_scalar("Loss/length", loss_length.item(), it)
            writer.flush()
            # log_training_state(
            #     net=net,
            #     orig_mesh=orig_mesh,
            #     rough_mesh=rough_mesh,
            #     device=device,
            #     cfg=cfg,
            #     iteration=it,
            #     loss=loss,
            # )

    log_training_state(
        net=net,
        orig_mesh=orig_mesh,
        rough_mesh=rough_mesh,
        device=device,
        cfg=cfg,
        iteration=it,
        loss=loss,
    )            
    
    torch.save(optimizer.state_dict(), "optimizer_state.pt")


@torch.no_grad()
def log_training_state(net, orig_mesh: PyMesh, rough_mesh: PyMesh, device, cfg: TrainingConfig, iteration: int, loss: torch.Tensor) -> None:
    """Dump intermediate OBJ previews and metrics during training."""
    points_mapped, barycentrics_mapped, face_idxs_mapped = sample_points(
        rough_mesh.sampler, cfg.n_sample_points, device
    )
    points_mapped = net(x=points_mapped, barycentrics=barycentrics_mapped, face_idxs=face_idxs_mapped)

    with open(cfg.out_obj, "w", encoding="utf-8") as f:
        for p in points_mapped:
            f.write(f"v {p[0]} {p[2]} {-p[1]}\n")

    orig_vertices = orig_mesh.mesh.get_vertices()
    orig_vertices = torch.from_numpy(orig_vertices).float().to(device)
    _, sdf_closests, sdf_barycentrics, sdf_face_idxs = point_query(rough_mesh.traverser, orig_vertices, device)

    mapped_vertices = net(x=sdf_closests, barycentrics=sdf_barycentrics, face_idxs=sdf_face_idxs).detach().cpu().numpy()

    with open("mapped_vertices.obj", "w", encoding="utf-8") as f:
        for p in mapped_vertices:
            f.write(f"v {p[0]} {p[2]} {-p[1]}\n")

    mesh_pred = Mesh.from_data(mapped_vertices, orig_mesh.mesh.get_faces())
    mesh_pred.save_to_obj("mapped_mesh.obj")
    mesh_pred.save_preview(
        "mapped_mesh_preview.png", 512, 512, orig_mesh.mesh.get_c(), orig_mesh.mesh.get_R()
    )

    # print(f"[it {iteration:05d}] loss={loss.item():.10f}")
