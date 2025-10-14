# main.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from bvh import Mesh, MeshSamplerMode, GPUMeshSampler


# ----------------------------
# Models: small MLPs (generator = residual map x -> x + f(x))
# ----------------------------
def make_mlp(in_dim, hidden, out_dim, num_layers=4, act=nn.LeakyReLU(0.2)):
    layers = [nn.Linear(in_dim, hidden), act]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden, hidden), act]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class ResidualMap(nn.Module):
    def __init__(self, hidden=128, layers=4):
        super().__init__()
        self.f = make_mlp(3, hidden, 3, num_layers=layers)

    def forward(self, x):
        return x + self.f(x)


class Critic(nn.Module):
    def __init__(self, hidden=128, layers=4):
        super().__init__()
        self.net = make_mlp(3, hidden, 1, num_layers=layers)

    def forward(self, x):
        return self.net(x).view(-1)


# ----------------------------
# Sampling helpers
# ----------------------------
@torch.no_grad()
def sample_points(sampler, batch_size, device):
    pts = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    sampler.sample(pts, batch_size)
    return pts


def gradient_penalty(critic, real, fake, gp_lambda=10.0):
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


# ----------------------------
# Main training
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rough", type=str, default="monkey_rough_3.fbx")
    p.add_argument("--fine", type=str, default="monkey_fine.fbx")
    p.add_argument("--batch_size", type=int, default=20000)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--transport_steps", type=int, default=5)
    p.add_argument("--critic_steps", type=int, default=1)
    p.add_argument("--g_lr", type=float, default=1e-4)
    p.add_argument("--d_lr", type=float, default=1e-3)
    p.add_argument("--gp_lambda", type=float, default=10.0)
    p.add_argument("--id_lambda", type=float, default=10, help="small displacement regularizer")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_points", type=int, default=50000)
    p.add_argument("--out_obj", type=str, default="sampled_points.obj")
    p.add_argument("--ckpt", type=str, default="mapping.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device (GPUMeshSampler expects CUDA; assert for clarity)
    assert torch.cuda.is_available(), "CUDA is required for GPUMeshSampler"
    device = torch.device("cuda")

    # Meshes + samplers
    rough_mesh = Mesh(args.rough)
    fine_mesh = Mesh(args.fine)
    rough_sampler = GPUMeshSampler(rough_mesh, MeshSamplerMode.SURFACE_UNIFORM, args.batch_size)
    fine_sampler = GPUMeshSampler(fine_mesh, MeshSamplerMode.SURFACE_UNIFORM, args.batch_size)

    # Models
    G = ResidualMap(hidden=args.hidden, layers=args.layers).to(device)
    D = Critic(hidden=args.hidden, layers=args.layers).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.9))

    print("Starting training...")
    for it in range(1, args.iters + 1):
        # ---------------- Critic ----------------
        for _ in range(args.critic_steps):
            x_src = sample_points(rough_sampler, args.batch_size, device)  # source samples
            y_tgt = sample_points(fine_sampler, args.batch_size, device)   # target samples
            with torch.no_grad():
                x_fake = G(x_src)  # detached for critic update

            d_opt.zero_grad(set_to_none=True)
            d_real = D(y_tgt).mean()
            d_fake = D(x_fake).mean()
            wd = d_real - d_fake  # Wasserstein-1 estimate
            gp = gradient_penalty(D, y_tgt, x_fake, gp_lambda=args.gp_lambda)
            d_loss = d_fake - d_real
            d_loss.backward()
            d_opt.step()

        # ---------------- Generator (transport map) ----------------
        for _ in range(args.transport_steps):
            x_src = sample_points(rough_sampler, args.batch_size, device)
            g_opt.zero_grad(set_to_none=True)
            x_fake = G(x_src)
            g_adv = -D(x_fake).mean()
            # tiny identity penalty to discourage wild displacements; safe to set 0
            g_id = ((x_fake - x_src) ** 2).mean()
            g_loss = g_adv + args.id_lambda * g_id
            g_loss.backward()
            g_opt.step()

        if it % args.log_interval == 0:
            print(f"[it {it:05d}] d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f} "
                  f"wd={wd.item():.4f} gp={gp.item():.4f} g_id={g_id.item():.6f}")

    # Save checkpoint
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args)}, args.ckpt)
    print(f"Saved checkpoint to {args.ckpt}")

    # Save mapped points from a larger rough sample
    with torch.no_grad():
        n = args.save_points
        rough_sampler = GPUMeshSampler(rough_mesh, MeshSamplerMode.SURFACE_UNIFORM, n)
        pts = torch.zeros((n, 3), dtype=torch.float32, device=device)
        # sample in chunks to avoid very large temporary tensors
        chunk = n
        out = []
        done = 0
        while done < n:
            bs = min(chunk, n - done)
            buf = torch.zeros((bs, 3), dtype=torch.float32, device=device)
            rough_sampler.sample(buf, bs)
            pred = G(buf)
            out.append(pred)
            done += bs
        pred_points = torch.cat(out, dim=0)

    # Write OBJ with vertices
    with open(args.out_obj, "w") as f:
        for p in pred_points.detach().cpu().numpy():
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
    print(f"Wrote {pred_points.shape[0]} vertices to {args.out_obj}")


if __name__ == "__main__":
    main()
