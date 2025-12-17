import time
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from PIL import Image

from mesh_utils import Mesh, MeshSamplerMode, GPUMeshSampler
from mesh_utils import GPUTraverser, CPUBuilder
from mesh_utils import GPURayTracer

from utils import sample_points, point_query, chamfer_distance, get_camera_rays, write_pairs_as_obj, write_triples_as_obj
from models import ResidualMap

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
    # fine_path = "models/queen_fine.fbx"
    # rough_path = "models/queen_rough.fbx"
    # fine_path = "models/monkey_fine.fbx"
    # rough_path = "models/monkey_rough.fbx"

    fine_path = "models/petmonster_orig.fbx"
    rough_path = "models/petmonster_outer_1000.fbx"
    # rough_path = "models/petmonster_inner_1000.fbx"
    # rough_path = "models/petmonster_rough.fbx"

    # fine_path = "models/monkey_126290.fbx"
    # rough_path = "models/monkey_outer_500.fbx"

    batch_size = 100000
    iters = 2000
    transport_steps = 1
    critic_steps = 1
    g_lr = 1e-3
    d_lr = 1e-4
    gp_lambda = 0.0
    id_lambda = 0.3
    log_interval = 100
    chamfer_points = 10000
    save_points = 100000
    out_obj = "sampled_points.obj"
    ckpt = "mapping.pt"
    device = "cuda"
    img_size = 800
    raytrace = True
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

    # Load checkpoint if exists
    if load_ckpt:
        checkpoint = torch.load(ckpt, map_location=device)
        G.load_state_dict(checkpoint["G"])
        print(f"Loaded checkpoint from {ckpt}")
    else:
        g_opt = torch.optim.AdamW(G.parameters(), lr=g_lr, weight_decay=1.0)

        start = time.time()

        print("Starting training...")
        for it in range(1, iters + 1):
            # ---------------- Generator (transport map) ----------------
            for _ in range(transport_steps):
                x_src, barycentrics, face_idxs = sample_points(rough_sampler, batch_size, device)
                g_opt.zero_grad(set_to_none=True)
                x_fake = G(x=x_src, barycentrics=barycentrics, face_idxs=face_idxs)

                # sdf_t, sdf_closest_pts, _, _ = point_query(fine_traverser, x_fake, device)
                sdf_t, sdf_closest_pts, _, _ = point_query(fine_traverser, x_src, device)
                g_adv = (sdf_closest_pts - x_fake).abs().sum(dim=1).mean()

                g_id = ((x_fake - x_src) ** 2).mean()
                g_loss = g_adv #+ id_lambda * g_id
                g_loss.backward()
                g_opt.step()

            if it % log_interval == 0:
                points_true, _, _ = sample_points(fine_sampler, chamfer_points, device)

                points_mapped, barycentrics_mapped, face_idxs_mapped = sample_points(rough_sampler, chamfer_points, device)
                points_mapped = G(x=points_mapped, barycentrics=barycentrics_mapped, face_idxs=face_idxs_mapped)
                cd = chamfer_distance(points_true, points_mapped).item()

                # Write OBJ with vertices
                with open(out_obj, "w") as f:
                    for p in points_mapped:
                        f.write(f"v {p[0]} {p[2]} {-p[1]}\n")            

                fine_vertices = fine_mesh.get_vertices()
                fine_vertices = torch.from_numpy(fine_vertices).float().to(device)
                sdf_t, sdf_closests, sdf_barycentrics, sdf_face_idxs = point_query(rough_traverser, fine_vertices, device)
                
                vertices = rough_mesh_split.get_vertices()
                faces = rough_mesh_split.get_faces()

                vertices = torch.from_numpy(vertices).to(device)
                mapped_vertices = G(x=sdf_closests, barycentrics=sdf_barycentrics, face_idxs=sdf_face_idxs).detach().cpu().numpy()
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

                print(f"[it {it:05d}] g_loss={g_loss.item():.5f} "
                    f"g_id={g_id.item():.6f} "
                    f"chamfer={cd:.6f} time={elapsed:.2f}s "
                    f"g_adv={g_adv.item():.5f}")
                
    pts = sample_points(rough_sampler, 1000, device)[0]
    pts_mapped = G(pts)
    pts_gt = point_query(fine_traverser, pts, device)[1]
    # write_pairs_as_obj(pts, pts_mapped, "mapped_points.obj")
    write_triples_as_obj(pts, pts_mapped, pts_gt, "mapped_points.obj")
        
    if raytrace:
        cam_poses, dirs = get_camera_rays(fine_mesh, img_size=img_size, device=device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        mask, t, normals = rough_ray_tracer.trace(cam_poses, dirs)
        pts = cam_poses + dirs * t[:, None]
        # t_sdf, sdf_pts = point_query(fine_traverser, pts, device)

        # pts = pts.double()
        # dirs = dirs.double()
        # cam_poses = cam_poses.double()
        
        epochs = 10
        pts = nn.Parameter(pts[mask], requires_grad=True)
        # optim = torch.optim.Adam([pts], lr=1e-1)
        optim = torch.optim.LBFGS([pts], lr=3, max_iter=30, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=epochs)

        pts_mapped = G(pts)
        loss = torch.cross(pts_mapped - cam_poses[mask], dirs[mask], dim=1).norm(dim=1).mean()
        print("Loss:", loss.item())

        for _ in range(epochs):
            with torch.no_grad():
                t_sdf, sdf_pts, sdf_barycentrics, sdf_face_idxs = point_query(rough_traverser, pts.data, device)
                pts.data = sdf_pts

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
            optim.step(closure)

        with torch.no_grad():
            t_sdf, sdf_pts, sdf_barycentrics, sdf_face_idxs = point_query(rough_traverser, pts.data, device)
            pts.data = sdf_pts

        threshold = 0.01
        # threshold = 1

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

        # compute normals
        with torch.no_grad():
            normals = (pts.data[m] - pts_mapped[m])
            normals = normals / normals.norm(dim=1, keepdim=True)

            print(pts.data[m].shape, pts_mapped[m].shape, mask.sum(), m.sum())

            colors = (-dirs[mask] * normals).sum(dim=1)
            colors = torch.abs(colors)
            colors = (colors + 1.0) * 0.5

            img = torch.zeros((img_size * img_size,), dtype=torch.float32, device=device)
            img[mask] = colors

            img = img.reshape(img_size, img_size).cpu().numpy()
            image = Image.fromarray((img * 255).astype(np.uint8))
            image.save('normal_shading.png')

        mask_img = mask.reshape(img_size, img_size)
        mask_img = mask_img.cpu().numpy()
        normals = normals.cpu().numpy()
        t_sdf, sdf_pts, sdf_barycentrics, sdf_idxs = point_query(fine_traverser, pts_mapped, device)
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
    torch.save({"G": G.state_dict()}, ckpt)
    print(f"Saved checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
