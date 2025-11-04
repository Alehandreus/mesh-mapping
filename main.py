import time
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from PIL import Image

from mesh_utils import Mesh, MeshSamplerMode, GPUMeshSampler
from mesh_utils import GPUTraverser, CPUBuilder
from mesh_utils import GPURayTracer

from utils import sample_points, point_query, chamfer_distance, get_camera_rays
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
    # fine_path = "models/sphere_fine.fbx"
    # rough_path = "models/sphere_rough_shell.fbx"
    fine_path = "models/queen_fine.fbx"
    rough_path = "models/queen_rough.fbx"
    #fine_path = "models/monkey_fine.fbx"
    #rough_path = "models/monkey_rough2.fbx"
    batch_size = 100000
    iters = 5000
    transport_steps = 1
    critic_steps = 1
    g_lr = 1e-3
    d_lr = 1e-4
    gp_lambda = 0.0
    id_lambda = 0.3
    reversed_lambda = 20
    grad_lambda = 0
    log_interval = 100
    chamfer_points = 10000
    save_points = 100000
    out_obj = "sampled_points.obj"
    ckpt = "mapping.pt"
    device = "cuda"
    img_size = 800
    raytrace = False
    load_ckpt = False

    rough_mesh_split = Mesh.from_file(rough_path)
    while len(rough_mesh_split.get_vertices()) < 1000000: # subdivide each primitive until we have enough vertices
        rough_mesh_split.split_faces(0.5)

    rough_mesh = Mesh.from_file(rough_path)
    rough_sampler = GPUMeshSampler(rough_mesh, MeshSamplerMode.SURFACE_UNIFORM, max(batch_size, chamfer_points))
    rough_builder = CPUBuilder(rough_mesh)
    rough_bvh = rough_builder.build_bvh(25)
    # rough_mesh = Mesh.from_data(rough_bvh.get_vertices(), rough_bvh.get_faces())
    rough_traverser = GPUTraverser(rough_bvh)
    rough_ray_tracer = GPURayTracer(rough_bvh)

    fine_mesh = Mesh.from_file(fine_path)
    fine_sampler = GPUMeshSampler(fine_mesh, MeshSamplerMode.SURFACE_UNIFORM, max(batch_size, chamfer_points))
    fine_builder = CPUBuilder(fine_mesh)
    fine_bvh = fine_builder.build_bvh(25)
    # fine_mesh = Mesh.from_data(fine_bvh.get_vertices(), fine_bvh.get_faces())
    fine_traverser = GPUTraverser(fine_bvh)

    rough_mesh.save_preview(f"rough_mesh_preview.png", 512, 512, rough_mesh.get_c(), rough_mesh.get_R())

    G = ResidualMap(rough_mesh).to(device)
    G_reversed = ResidualMap(fine_mesh).to(device)
    D = Critic(rough_mesh).to(device)

    # Load checkpoint if exists
    if load_ckpt:
        checkpoint = torch.load(ckpt, map_location=device)
        G.load_state_dict(checkpoint["G"])
        D.load_state_dict(checkpoint["D"])
        print(f"Loaded checkpoint from {ckpt}")

    g_opt = torch.optim.AdamW(G.parameters(), lr=g_lr, weight_decay=1.0)
    g_reversed_opt = torch.optim.AdamW(G_reversed.parameters(), lr=g_lr, weight_decay=1.0)
    d_opt = torch.optim.Adam(D.parameters(), lr=d_lr)

    start = time.time()

    print("Starting training...")
    for it in range(1, iters + 1):
        # ---------------- Generator (transport map) ----------------
        for _ in range(transport_steps):
            x_src = sample_points(rough_sampler, batch_size, device)
            x_src.requires_grad_(True)
            g_opt.zero_grad(set_to_none=True)

            def loss(G, G_reversed, traverser, x_src):
                x_fake = G(x_src)
                sdf_t, sdf_closest_pts = point_query(traverser, x_fake, device)
                g_adv = (sdf_closest_pts - x_fake).abs().sum(dim=1).mean()

                # g_adv = -D(x_fake).mean()
                g_id = ((x_fake - x_src) ** 2).mean()
                inverse = ((G_reversed(x_fake) - x_src) ** 2).mean()
                g_loss = g_adv + id_lambda * g_id + reversed_lambda * inverse
                return g_loss, g_adv, g_id, inverse
            
            g_loss, g_adv, g_id, inverse = loss(G, G_reversed, fine_traverser, x_src)
            grad = 0
            if grad_lambda != 0:
                grad = autograd.grad(
                    outputs=g_loss,
                    inputs=x_src,
                    create_graph=True,
                    grad_outputs=torch.ones_like(g_loss)
                )[0]
                new_x_src = x_src + 0.01 * grad.sign()
                g_loss = g_loss + loss(G, G_reversed, fine_traverser, new_x_src)[0] * grad_lambda
            g_loss.backward()
            g_opt.step()

            if reversed_lambda != 0:
                x_src = sample_points(fine_sampler, batch_size, device)
                x_src.requires_grad_(True)
                g_reversed_loss, _, _, _ = loss(G_reversed, G, rough_traverser, x_src)
                #if grad_lambda != 0:
                #    grad = autograd.grad(
                #        outputs=g_reversed_loss,
                #        inputs=x_src,
                #        create_graph=True,
                #        grad_outputs=torch.ones_like(g_reversed_loss)
                #    )[0]
                #    new_x_src = x_src + 0.01 * grad.sign()
                #    g_reversed_loss = g_reversed_loss + loss(G_reversed, G, rough_traverser, new_x_src)[0] * grad_lambda
                g_reversed_loss.backward()
                g_reversed_opt.step()

        if it % log_interval == 0:
            points_true = sample_points(fine_sampler, chamfer_points, device)

            points_mapped = sample_points(rough_sampler, chamfer_points, device)
            # print(barycentrics_mapped[:4], face_idxs_mapped[:4])
            # t, points_mapped, barycentrics_mapped, face_idxs_mapped = point_query(rough_traverser, points_mapped, device)
            # print(barycentrics_mapped[:4], face_idxs_mapped[:4])
            points_mapped = G(x=points_mapped)
            cd = chamfer_distance(points_true, points_mapped).item()

            # Write OBJ with vertices
            with open(out_obj, "w") as f:
                for p in points_mapped:
                    f.write(f"v {p[0]} {p[2]} {-p[1]}\n")            

            fine_vertices = fine_mesh.get_vertices()
            fine_vertices = torch.from_numpy(fine_vertices).float().to(device)
            sdf_t, sdf_closests = point_query(rough_traverser, fine_vertices, device)
            
            vertices = rough_mesh_split.get_vertices()
            faces = rough_mesh_split.get_faces()

            vertices = torch.from_numpy(vertices).to(device)
            mapped_vertices = G(x=sdf_closests).detach().cpu().numpy()
            vertices = vertices.cpu().numpy()

            mesh_pred = Mesh.from_data(mapped_vertices, fine_mesh.get_faces())
            mesh_pred = Mesh.from_data(mapped_vertices, fine_mesh.get_faces())
            mesh_pred.save_to_obj(f"mapped_mesh.obj")

            fine_mesh.save_preview(f"fine_mesh_preview.png", 512, 512, fine_mesh.get_c(), fine_mesh.get_R())
            mesh_pred.save_preview(f"mapped_mesh_preview.png", 512, 512, fine_mesh.get_c(), fine_mesh.get_R())

            torch.save({"G": G.state_dict()}, ckpt)

            end = time.time()
            elapsed = end - start
            start = end

            print(f"[it {it:05d}] g_loss={g_loss.item():.4f} "
                  f"g_id={g_id.item():.6f} "
                  f"chamfer={cd:.6f} time={elapsed:.2f}s "
                  f"g_adv={g_adv.item():.5f} "
                  f"inverse={inverse.item():.6f} ")
        
    if raytrace:
        cam_poses, dirs = get_camera_rays(fine_mesh, img_size=img_size, device=device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        mask, t, normals = rough_ray_tracer.trace(cam_poses, dirs)
        pts = cam_poses + dirs * t[:, None]
        # t_sdf, sdf_pts = point_query(fine_traverser, pts, device)

        # pts = pts.double()
        # dirs = dirs.double()
        # cam_poses = cam_poses.double()
        
        epochs = 50
        pts = nn.Parameter(pts[mask], requires_grad=True)
        # optim = torch.optim.Adam([pts], lr=1e-1)
        optim = torch.optim.LBFGS([pts], lr=3, max_iter=30, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=epochs)

        pts_mapped = G(pts)
        loss = torch.cross(pts_mapped - cam_poses[mask], dirs[mask], dim=1).norm(dim=1).mean()
        print("Loss:", loss.item())

        for _ in range(epochs):
            with torch.no_grad():
                t_sdf, sdf_pts = point_query(rough_traverser, pts.data, device)
                pts.data = sdf_pts

            # t_sdf, sdf_pts = point_query(rough_traverser, pts, device)
            # distance from point to ray dir
            pts_mapped = G(pts)
            loss = torch.cross(pts_mapped - cam_poses[mask], dirs[mask], dim=1).norm(dim=1).mean()
            print("Loss:", loss.item())

            def closure():
                optim.zero_grad()
                pts_mapped = G(pts)
                loss = torch.cross(pts_mapped - cam_poses[mask], dirs[mask], dim=1).norm(dim=1).mean()
                loss.backward()
                return loss

            optim.zero_grad()
            loss.backward()
            # optim.step()
            optim.step(closure)
            # scheduler.step()

        with torch.no_grad():
            t_sdf, sdf_pts = point_query(rough_traverser, pts.data, device)
            pts.data = sdf_pts

        threshold = 0.01

        pts_mapped = G(pts)
        loss = torch.cross(pts_mapped - cam_poses[mask], dirs[mask], dim=1).norm(dim=1)

        # save heatmap of loss
        with torch.no_grad():
            heatmap = torch.zeros((img_size * img_size,), dtype=torch.float32, device=device)
            heatmap[mask] = loss
            heatmap = heatmap.cpu().numpy()
            heatmap = heatmap.reshape(img_size, img_size)
            heatmap = heatmap / heatmap.max()
            heatmap = np.sqrt(1 - np.square(1 - heatmap))
            image = Image.fromarray((heatmap * 255).astype(np.uint8))
            image.save('loss_heatmap.png')

        m = mask.clone()[mask]
        m[loss >= threshold] = False
        mask[mask.clone()] = m            

        # save distance from mapped points to camera
        with torch.no_grad():
            dist_map = torch.zeros((img_size * img_size,), dtype=torch.float32, device=device)
            dist_map[mask] = (pts_mapped[m] - cam_poses[mask]).norm(dim=1)
            mmin = dist_map[dist_map > 0].min()
            mmax = dist_map.max()
            dist_map = (dist_map - mmin) / (mmax - mmin)
            dist_map[~mask] = 1.0
            dist_map = 1 - dist_map
            dist_map = dist_map.cpu().numpy()
            dist_map = dist_map.reshape(img_size, img_size)
            image = Image.fromarray((dist_map * 255).astype(np.uint8))
            image.save('distance_map.png')

        mask_img = mask.reshape(img_size, img_size)
        mask_img = mask_img.cpu().numpy()
        normals = normals.cpu().numpy()
        t_sdf, sdf_pts = point_query(fine_traverser, pts_mapped, device)
        # t = t.cpu().numpy()
        t = torch.zeros((img_size * img_size,), dtype=torch.float32, device=device)

        print(t.min(), t.max())

        t[mask] = 1.0
        t = t.cpu().numpy()

        light_dir = np.array([1, -1, 1])
        light_dir = light_dir / np.linalg.norm(light_dir)

        normals[np.isnan(normals)] = 0
        colors = t

        img = colors.reshape(img_size, img_size)
        img[~mask_img] = 0

        img = img / img.max()

        image = Image.fromarray((img * 255).astype(np.uint8))
        image.save('output.png')


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
