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

from dataclasses import dataclass


@dataclass
class PyMesh:
    mesh: Mesh
    mesh_split: Mesh
    sampler: GPUMeshSampler
    traverser: GPUTraverser
    ray_tracer: GPURayTracer
    
    @staticmethod
    def from_file(path: str):
        mesh = Mesh.from_file(path)
        sampler = GPUMeshSampler(mesh, MeshSamplerMode.SURFACE_UNIFORM, 100000)
        builder = CPUBuilder(mesh)
        bvh = builder.build_bvh(25)
        traverser = GPUTraverser(bvh)
        ray_tracer = GPURayTracer(bvh)

        mesh_split = Mesh.from_file(path)
        while len(mesh_split.get_vertices()) < 1000000:
            mesh_split.split_faces(0.5)

        return PyMesh(mesh, mesh_split, sampler, traverser, ray_tracer)


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


def train(net, orig_mesh, rough_mesh):
    lr = 1e-3
    epochs = 2000
    device = net.parameters().__next__().device
    batch_size = 100000
    log_interval = 100
    n_sample_points = 1000
    out_obj = "sampled_points.obj"

    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1.0)
    # opt = torch.optim.AdamW(net.parameters(), lr=lr)

    print("Starting training...")
    for it in range(1, epochs + 1):
        # outer_x, outer_barycentrics, outer_face_idxs = sample_points(outer_sampler, batch_size, device)
        # inner_x, inner_barycentrics, inner_face_idxs = sample_points(inner_sampler, batch_size, device)
        # x = torch.cat([outer_x, inner_x], dim=0)
        # barycentrics = torch.cat([outer_barycentrics, inner_barycentrics], dim=0)
        # face_idxs = torch.cat([outer_face_idxs, inner_face_idxs], dim=0)

        x, barycentrics, face_idxs = sample_points(rough_mesh.sampler, batch_size, device)

        _, y, _, _ = point_query(orig_mesh.traverser, x, device)

        opt.zero_grad(set_to_none=True)
        y_pred = net(x=x, barycentrics=barycentrics, face_idxs=face_idxs)

        # loss = ((y_pred - y) ** 2).sum(dim=1).mean()
        loss = (y_pred - y).abs().sum(dim=1).mean()
        loss.backward()
        opt.step()

        if it % log_interval == 0:
            points_mapped, barycentrics_mapped, face_idxs_mapped = sample_points(rough_mesh.sampler, n_sample_points, device)
            points_mapped = net(x=points_mapped, barycentrics=barycentrics_mapped, face_idxs=face_idxs_mapped)

            # Write OBJ with vertices
            with open(out_obj, "w") as f:
                for p in points_mapped:
                    f.write(f"v {p[0]} {p[2]} {-p[1]}\n")

            orig_vertices = orig_mesh.mesh.get_vertices()
            orig_vertices = torch.from_numpy(orig_vertices).float().to(device)
            sdf_t, sdf_closests, sdf_barycentrics, sdf_face_idxs = point_query(rough_mesh.traverser, orig_vertices, device)

            mapped_vertices = net(x=sdf_closests, barycentrics=sdf_barycentrics, face_idxs=sdf_face_idxs).detach().cpu().numpy()

            mesh_pred = Mesh.from_data(mapped_vertices, orig_mesh.mesh.get_faces())
            mesh_pred = Mesh.from_data(mapped_vertices, orig_mesh.mesh.get_faces())
            mesh_pred.save_to_obj(f"mapped_mesh.obj")

            mesh_pred.save_preview(f"mapped_mesh_preview.png", 512, 512, orig_mesh.mesh.get_c(), orig_mesh.mesh.get_R())

            # torch.save({"inner_net": net.state_dict()}, ckpt)

            print(f"[it {it:05d}] loss={loss.item():.10f}")


def get_raytrace_loss(cam_poses, dirs, y, reduction='mean'):
    if reduction == 'mean':
        return torch.cross(y - cam_poses, dirs, dim=1).norm(dim=1).mean()
    return torch.cross(y - cam_poses, dirs, dim=1).norm(dim=1)    

def do_raytrace(cam_poses, dirs, traverser, G, x0, verbose=False):
    """
        x -- points on rough mesh surface (optimized)
        y = G(x) -- mapped points on fine mesh surface
    """

    device = x0.device

    epochs = 20
    threshold = 0.0005
    # threshold = np.inf

    accepted_x1 = torch.zeros_like(x0)
    accepted_y1 = torch.zeros_like(x0)
    accepted_mask = torch.zeros((x0.shape[0],), dtype=torch.bool, device=device)

    x = nn.Parameter(x0, requires_grad=True)
    optim = torch.optim.LBFGS([x], lr=1e-1, max_iter=30, line_search_fn='strong_wolfe')
    
    if verbose:
        loss = get_raytrace_loss(cam_poses, dirs, G(x))
        print("Initial loss:", loss.item())

    for _ in range(epochs):
        with torch.no_grad():
            _, sdf_closest_pts, _, _ = point_query(traverser, x.data, device)
            x.data = sdf_closest_pts

        def closure():
            optim.zero_grad()
            y = G(x)
            loss = get_raytrace_loss(cam_poses, dirs, y)
            loss.backward()
            return loss

        optim.step(closure)

        if verbose:
            loss = get_raytrace_loss(cam_poses, dirs, G(x))
            print("Loss:", loss.item())

        loss = get_raytrace_loss(cam_poses, dirs, G(x), reduction='none')
        mask = loss < threshold

        accepted_x1[mask] = x.data[mask]
        accepted_y1[mask] = G(x)[mask]
        accepted_mask[mask] = True

        print(f"Accepted {accepted_mask.sum().item()} / {x.shape[0]}")

    x1 = accepted_x1.detach()
    y1 = accepted_y1.detach()
    mask = accepted_mask
    normals = (x1 - y1)
    normals = normals / normals.norm(dim=1, keepdim=True)        

    return x1, y1, mask, normals


def main():
    ckpt_path = "mapping.pt"
    # load_ckpt = True
    load_ckpt = False

    orig_path = "models/petmonster_orig.fbx"
    inner_path = "models/petmonster_inner_1000.fbx"
    outer_path = "models/petmonster_outer_1000.fbx"

    device = "cuda"
    img_size = 800
    raytrace = True

    orig_mesh = PyMesh.from_file(orig_path)
    inner_mesh = PyMesh.from_file(inner_path)
    outer_mesh = PyMesh.from_file(outer_path)

    outer_mesh.mesh.save_preview(f"outer_mesh_preview.png", 512, 512, outer_mesh.mesh.get_c(), outer_mesh.mesh.get_R())
    orig_mesh.mesh.save_preview(f"orig_mesh_preview.png", 512, 512, orig_mesh.mesh.get_c(), orig_mesh.mesh.get_R())

    inner_net = ResidualMap(inner_mesh.mesh).to(device)
    outer_net = ResidualMap(outer_mesh.mesh).to(device)

    if load_ckpt:
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        inner_net.load_state_dict(ckpt["inner_net"])
        outer_net.load_state_dict(ckpt["outer_net"])       
    else:
        train(inner_net, orig_mesh, inner_mesh)
        train(outer_net, orig_mesh, outer_mesh)

        print(f"Saving checkpoint to {ckpt_path}...")
        torch.save({
            "inner_net": inner_net.state_dict(),
            "outer_net": outer_net.state_dict(),
        }, ckpt_path)

    if raytrace:
        cam_poses, dirs = get_camera_rays(orig_mesh.mesh, img_size=img_size, device=device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)

        def do_raytrace_wrapper(cam_poses, dirs, mesh, net):
            initial_mask, t, normals = mesh.ray_tracer.trace(cam_poses, dirs)
            x0 = cam_poses + dirs * t[:, None]

            x1, y1, accepted_mask, normals = do_raytrace(cam_poses[initial_mask], dirs[initial_mask], mesh.traverser, net, x0[initial_mask], verbose=True)
            combined_mask = initial_mask.clone()
            combined_mask[initial_mask] = accepted_mask

            return x1, y1, combined_mask, normals, accepted_mask
        
        x1 = torch.zeros((img_size * img_size, 3), dtype=torch.float32, device=device)
        y1 = torch.zeros((img_size * img_size, 3), dtype=torch.float32, device=device)
        combined_mask = torch.zeros((img_size * img_size,), dtype=torch.bool, device=device)
        normals = torch.zeros((img_size * img_size, 3), dtype=torch.float32, device=device)
        accepted_mask = torch.zeros((img_size * img_size,), dtype=torch.bool, device=device)

        inner_x1, inner_y1, inner_combined_mask, inner_normals, inner_accepted_mask = do_raytrace_wrapper(cam_poses, dirs, inner_mesh, inner_net)
        outer_x1, outer_y1, outer_combined_mask, outer_normals, outer_accepted_mask = do_raytrace_wrapper(cam_poses, dirs, outer_mesh, outer_net)

        x1[inner_combined_mask] = inner_x1[inner_accepted_mask]
        y1[inner_combined_mask] = inner_y1[inner_accepted_mask]
        normals[inner_combined_mask] = inner_normals[inner_accepted_mask]
        accepted_mask[inner_combined_mask] = True
        combined_mask = combined_mask | inner_combined_mask

        x1[outer_combined_mask] = outer_x1[outer_accepted_mask]
        y1[outer_combined_mask] = outer_y1[outer_accepted_mask]        
        normals[outer_combined_mask] = outer_normals[outer_accepted_mask]        
        accepted_mask[outer_combined_mask] = True
        combined_mask = combined_mask | outer_combined_mask
        
        # x1, y1, combined_mask, normals, accepted_mask = do_raytrace_wrapper(cam_poses, dirs, outer_mesh, outer_net)

        # save heatmap of loss
        # with torch.no_grad():
        #     heatmap = torch.zeros((img_size * img_size,), dtype=torch.float32, device=device)
        #     loss = get_raytrace_loss(cam_poses[initial_mask], dirs[initial_mask], inner_net(y1), reduction='none')
        #     heatmap[initial_mask] = loss
        #     mmin = heatmap[heatmap > 0].min()
        #     mmax = heatmap.max()
        #     heatmap = (heatmap - mmin) / (mmax - mmin)
        #     heatmap[~initial_mask] = 0.0
        #     # heatmap = 1 - heatmap
        #     heatmap = torch.sqrt(1 - torch.square(1 - heatmap))
        #     heatmap = heatmap.cpu().numpy()
        #     heatmap = heatmap.reshape(img_size, img_size)
        #     image = Image.fromarray((heatmap * 255).astype(np.uint8))
        #     image.save('loss_heatmap.png')
        
        # save distance map
        with torch.no_grad():
            dist_map = torch.zeros((img_size * img_size,), dtype=torch.float32, device=device)
            dist_map[combined_mask] = (y1[accepted_mask] - cam_poses[combined_mask]).norm(dim=1)
            mmin = dist_map[dist_map > 0].min()
            mmax = dist_map.max()
            dist_map = (dist_map - mmin) / (mmax - mmin)
            dist_map[~combined_mask] = 1.0
            dist_map = 1 - dist_map
            dist_map = dist_map.cpu().numpy()
            dist_map = dist_map.reshape(img_size, img_size)
            image = Image.fromarray((dist_map * 255).astype(np.uint8))
            image.save('distance_map.png')

        # save normal shading
        with torch.no_grad():
            colors = torch.zeros((img_size * img_size,), dtype=torch.float32, device=device)
            colors[combined_mask] = (-dirs[combined_mask] * normals[accepted_mask]).sum(dim=1)
            colors = torch.abs(colors)
            colors = (colors + 1.0) * 0.5
            colors[~combined_mask] = 0.0
            colors = colors.cpu().numpy()
            colors = colors.reshape(img_size, img_size)
            image = Image.fromarray((colors * 255).astype(np.uint8))
            image.save('normal_shading.png')


if __name__ == "__main__":
    main()
