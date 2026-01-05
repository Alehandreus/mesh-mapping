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
    epochs: int = 200
    batch_size: int = 100_000
    log_interval: int = 100
    n_sample_points: int = 10_000
    out_obj: str = "sampled_points.obj"
    weight_decay: float = 1.0


def gradient_penalty(critic, real, fake, gp_lambda: float = 10.0) -> torch.Tensor:
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, device=real.device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    scores = critic(interp)
    grad = autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp_lambda * gp


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
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("Starting training...")
    for it in range(1, cfg.epochs + 1):
        x, barycentrics, face_idxs = sample_points(rough_mesh.sampler, cfg.batch_size, device)
        _, y, _, _ = point_query(orig_mesh.traverser, x, device)

        optimizer.zero_grad(set_to_none=True)
        y_pred = net(x=x, barycentrics=barycentrics, face_idxs=face_idxs)

        loss = (y_pred - y).abs().sum(dim=1).mean()
        loss.backward()
        optimizer.step()

        if it % cfg.log_interval == 0:
            log_training_state(
                net=net,
                orig_mesh=orig_mesh,
                rough_mesh=rough_mesh,
                device=device,
                cfg=cfg,
                iteration=it,
                loss=loss,
            )


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

    print(f"[it {iteration:05d}] loss={loss.item():.10f}")
