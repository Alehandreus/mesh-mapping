import time
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from mesh_utils import Mesh, MeshSamplerMode, GPUMeshSampler
from mesh_utils import GPUTraverser, CPUBuilder

from utils import sample_points, point_query, chamfer_distance
from models import ResidualMap, Critic

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


def main():
    fine_path = "models/queen_fine.fbx"
    rough_path = "models/queen_rough.fbx"
    batch_size = 100000
    iters = 2000
    transport_steps = 1
    critic_steps = 1
    g_lr = 3e-4
    d_lr = 1e-4
    gp_lambda = 0.0
    id_lambda = 0.1
    log_interval = 100
    chamfer_points = 10000
    save_points = 100000
    out_obj = "sampled_points.obj"
    ckpt = "mapping.pt"
    device = "cuda"

    rough_mesh = Mesh.from_file(rough_path)
    while len(rough_mesh.get_vertices()) < 1000000: # subdivide each primitive until we have enough vertices
        rough_mesh.split_faces(0.5)
    rough_sampler = GPUMeshSampler(rough_mesh, MeshSamplerMode.SURFACE_UNIFORM, max(batch_size, chamfer_points))

    fine_mesh = Mesh.from_file(fine_path)
    fine_sampler = GPUMeshSampler(fine_mesh, MeshSamplerMode.SURFACE_UNIFORM, max(batch_size, chamfer_points))

    fine_builder = CPUBuilder(fine_mesh)
    fine_bvh = fine_builder.build_bvh(25)
    fine_traverser = GPUTraverser(fine_bvh)

    G = ResidualMap(rough_mesh).to(device)
    D = Critic(rough_mesh).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=g_lr)
    d_opt = torch.optim.Adam(D.parameters(), lr=d_lr)

    start = time.time()

    print("Starting training...")
    for it in range(1, iters + 1):
        # ---------------- Critic ----------------
        # for _ in range(critic_steps):
        #     x_src = sample_points(rough_sampler, batch_size, device)  # source samples
        #     y_tgt = sample_points(fine_sampler, batch_size, device)   # target samples
        #     with torch.no_grad():
        #         x_fake = G(x_src)  # detached for critic update

        #     d_opt.zero_grad(set_to_none=True)
        #     d_real = D(y_tgt).mean()
        #     d_fake = D(x_fake).mean()
        #     wd = d_real - d_fake  # Wasserstein-1 estimate
        #     gp = gradient_penalty(D, y_tgt, x_fake, gp_lambda=gp_lambda)
        #     d_loss = d_fake - d_real
        #     d_loss.backward()
        #     d_opt.step()

        # ---------------- Generator (transport map) ----------------
        for _ in range(transport_steps):
            x_src = sample_points(rough_sampler, batch_size, device)
            g_opt.zero_grad(set_to_none=True)
            x_fake = G(x_src)

            t, closest_pts = point_query(fine_traverser, x_fake, device)
            g_adv = (closest_pts - x_fake).abs().sum(dim=1).mean()

            # g_adv = -D(x_fake).mean()
            g_id = ((x_fake - x_src) ** 2).mean()
            g_loss = g_adv + id_lambda * g_id
            g_loss.backward()
            g_opt.step()

        if it % log_interval == 0:
            points_true = sample_points(fine_sampler, chamfer_points, device)
            points_mapped = sample_points(rough_sampler, chamfer_points, device)
            points_mapped = G(points_mapped)
            cd = chamfer_distance(points_true, points_mapped).item()
            
            vertices = rough_mesh.get_vertices()
            faces = rough_mesh.get_faces()

            vertices = torch.from_numpy(vertices).to(device)
            mapped_vertices = G(vertices).detach().cpu().numpy()
            vertices = vertices.cpu().numpy()

            mesh_pred = Mesh.from_data(mapped_vertices, faces)
            mesh_pred.save_to_obj(f"mapped_mesh.obj")
            mesh_pred.save_preview(f"mapped_mesh_preview.png")

            torch.save({"G": G.state_dict()}, ckpt)

            end = time.time()
            elapsed = end - start
            start = end

            # print(f"[it {it:05d}] d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f} "
            #       f"wd={wd.item():.4f} gp={gp.item():.4f} g_id={g_id.item():.6f} "
            #       f"chamfer={cd:.6f} time={elapsed:.2f}s")

            print(f"[it {it:05d}] g_loss={g_loss.item():.4f} "
                  f"g_id={g_id.item():.6f} "
                  f"chamfer={cd:.6f} time={elapsed:.2f}s "
                  f"g_adv={g_adv.item():.5f}")

    # Save checkpoint
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt)
    print(f"Saved checkpoint to {ckpt}")

    # Save mapped points from a larger rough sample
    with torch.no_grad():
        n = save_points
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
    with open(out_obj, "w") as f:
        for p in pred_points.detach().cpu().numpy():
            f.write(f"v {p[0]} {p[2]} {-p[1]}\n")
    print(f"Wrote {pred_points.shape[0]} vertices to {out_obj}")


if __name__ == "__main__":
    main()
