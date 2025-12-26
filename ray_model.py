import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from PIL import Image
import math

from mesh_utils import Mesh, MeshSamplerMode, GPUMeshSampler
from mesh_utils import GPUTraverser, CPUBuilder
from mesh_utils import GPURayTracer

from utils import sample_points, point_query, get_camera_rays
from models import RayModel, RayNormalModel

import matplotlib.pyplot as plt

DEVICE = 'cuda'
LEARNING_RATE = 1e-2
EPOCHS = 5000
LOG_INTERVAL = 100

IMG_SIZE = 512

def calculate_normals(faces, vertices, face_ids):
    v0_ids = faces[face_ids, 0]
    v1_ids = faces[face_ids, 1]
    v2_ids = faces[face_ids, 2]

    v0s = vertices[v0_ids]
    v1s = vertices[v1_ids]
    v2s = vertices[v2_ids]

    normals = np.cross(v1s - v0s, v2s - v0s, axis=1)
    return normals / np.linalg.norm(normals, axis=1)[:, None]

def sample_directions(normals):
    thetas = np.random.uniform(0, np.pi, size=normals.shape[0])
    phis = np.random.uniform(0, 2 * np.pi, size=normals.shape[0])

    x_norm = np.sin(thetas) * np.cos(phis)
    y_norm = np.sin(thetas) * np.sin(phis)
    z_norm = np.cos(thetas)
    vector = np.stack([x_norm, y_norm, z_norm], axis=1)

    basis_x_norm = np.ones(normals.shape)
    mask = normals[:, 2] != 0
    basis_x_norm[mask, 2] = -(normals[mask, 0] + normals[mask, 1]) / normals[mask, 2]
    basis_x_norm[~mask, 2] = 0
    basis_x_norm = basis_x_norm / np.linalg.norm(basis_x_norm, axis=1)[:, None]
    basis_y_norm = -normals
    basis_z_norm = np.cross(basis_x_norm, basis_y_norm, axis=1)
    basis_coefs = np.stack([basis_x_norm, basis_y_norm, basis_z_norm], axis=1)

    prop_ds = np.einsum('ijk,ik->ij', basis_coefs, vector)
    return torch.tensor(prop_ds, device=DEVICE)

def main():
    fine_path = "models/monkey_fine.fbx"
    rough_path = "models/monkey_convex_hull2.fbx"
    #rough_path = "models/monkey_rough2.fbx"
    #fine_path = "models/queen_fine.fbx"
    #rough_path = "models/queen_rough.fbx"
    #rough_path = "models/queen_convex_hull.fbx"
    sphere_path = "models/sphere_big.fbx"
    #fine_path = "models/petmonster.fbx"
    #rough_path = "models/petmonster_rough.fbx"
    #rough_path = "models/sphere.fbx"


    sample_size = 100000
    chamfer_points = 10000
    point_amount = 5

    rough_mesh_split = Mesh.from_file(rough_path)
    while len(rough_mesh_split.get_vertices()) < 1000000: # subdivide each primitive until we have enough vertices
        rough_mesh_split.split_faces(0.5)

    rough_mesh = Mesh.from_file(rough_path)
    rough_sampler = GPUMeshSampler(rough_mesh, MeshSamplerMode.SURFACE_UNIFORM, sample_size)
    rough_builder = CPUBuilder(rough_mesh)
    rough_bvh = rough_builder.build_bvh(25)
    rough_traverser = GPUTraverser(rough_bvh)
    rough_ray_tracer = GPURayTracer(rough_bvh)
    rough_faces = rough_mesh.get_faces()
    rough_vertices = rough_mesh.get_vertices()

    sphere_mesh = Mesh.from_file(sphere_path)
    sphere_sampler = GPUMeshSampler(sphere_mesh, MeshSamplerMode.SURFACE_UNIFORM, sample_size)
    sphere_faces = sphere_mesh.get_faces()
    sphere_vertices = sphere_mesh.get_vertices()

    fine_mesh = Mesh.from_file(fine_path)
    fine_sampler = GPUMeshSampler(fine_mesh, MeshSamplerMode.SURFACE_UNIFORM, sample_size)
    fine_builder = CPUBuilder(fine_mesh)
    fine_bvh = fine_builder.build_bvh(25)
    fine_traverser = GPUTraverser(fine_bvh)
    fine_ray_tracer = GPURayTracer(fine_bvh)

    rough_mesh.save_preview(f"rough_mesh_preview.png", IMG_SIZE, IMG_SIZE, rough_mesh.get_c(), rough_mesh.get_R())

    model = RayModel(rough_mesh).to(DEVICE)
    model_intersection = RayModel(rough_mesh).to(DEVICE)
    model_normal = RayNormalModel(rough_mesh).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_intersection = torch.optim.Adam(model_intersection.parameters(), lr=LEARNING_RATE)
    optimizer_normal = torch.optim.Adam(model_normal.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        x, _, face_idxs_sp = sample_points(sphere_sampler, sample_size, DEVICE)
        x = x.clone()
        face_idxs_sp = face_idxs_sp.clone()

        x_src2, barycentrics2, face_idxs2 = sample_points(rough_sampler, sample_size, DEVICE)

        normals2 = calculate_normals(rough_faces, rough_vertices, face_idxs2.detach().cpu().numpy())
        normals1 = calculate_normals(sphere_faces, sphere_vertices, face_idxs_sp.detach().cpu().numpy())
        


        ds1 = sample_directions(normals1)
        ds2 = sample_directions(normals2)
       
        mask, t, normals = rough_ray_tracer.trace(x, ds1)
        mask = mask.clone()
        x_src1 = x + ds1 * t[:, None]

        _, _, barycentrics1, face_idxs1 = point_query(rough_traverser, x_src1, DEVICE)

        
        mask1, true_r1, normals_traced = fine_ray_tracer.trace(x_src1, ds1)
        mask1 = mask1.clone()
        true_r1 = true_r1.clone()
        normals_traced = normals_traced.clone()
        
        #mask1[true_r1 < 0] = 0
        mask2, true_r2, _ = fine_ray_tracer.trace(x_src2, ds2)
        #mask2 = mask2 & mask #???
        #mask2[true_r2 < 0] = 0

        #mask_above = mask #true_r > 0
        #mask1 = true_r1 > 0
        #print(true_r[true_r > 0].sum())
    
        prediction = model_intersection(torch.concatenate([x_src2, ds2], dim=1), barycentrics=barycentrics2, face_idxs=face_idxs2)
        #prediction = model_intersection(points2)
        #print(prediction.shape)
        has_intersection = prediction[:, 0]

        prediction = model(torch.concatenate([x_src1, ds1], dim=1), barycentrics=barycentrics1, face_idxs=face_idxs1)
        #prediction = model(points1)
        predicted_r = prediction[:, 0]
       
        true_has_intersection = mask2.to(torch.float16)
        #true_has_intersection[true_has_intersection == 1] -= 0.2
        #true_has_intersection[true_has_intersection == 0] += 0.2

        prediction = model_normal(torch.concatenate([x_src1, ds1], dim=1), barycentrics=barycentrics1, face_idxs=face_idxs1)
        predicted_normal = prediction[:, :3]
        #predicted_normal = torch.zeros(normalized.shape, device=DEVICE, dtype=normalized.dtype)
        #predicted_normal[normalized.norm(dim=1) > 0.01] = normalized / normalized.norm(dim=1)[:, None]
        
        
        intersection_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        intersected_mask = has_intersection > 0
        entropy = intersection_loss(has_intersection, true_has_intersection)
        weights = torch.ones_like(mask2, dtype=torch.float32) + (intersected_mask & mask2) * math.exp(epoch / 10000)
        entropy = (entropy * weights).mean()

        
        
        #distance = ((predicted_r[mask1] - true_r1[mask1]) ** 2).mean()
        distance = (predicted_r[mask1] - true_r1[mask1]).abs().mean() 
        
        #loss = distance + entropy
        #loss = (predicted_r - true_r[mask]).abs().mean()

        normal_error = ((predicted_normal[mask1] - normals_traced[mask1]) ** 2).mean()

        optimizer.zero_grad()
        optimizer_intersection.zero_grad()
        optimizer_normal.zero_grad()
        #loss.backward()
        distance.backward()
        optimizer.step()
        entropy.backward()
        optimizer_intersection.step()
        normal_error.backward()
        optimizer_normal.step()

        if (epoch + 1) % LOG_INTERVAL == 0:
            print("EPOCH:", epoch, "distance =", distance.item(), "entr =", entropy.item(), "norm =", normal_error.item())
            print((has_intersection > 0).sum().item(), mask2.sum().item())

            plt.clf()
            plt.hist(true_r1[mask1].detach().cpu().numpy(), bins=30, edgecolor="red", range=(-2, 2))
            plt.savefig('train_hist.png')
            plt.clf()
            plt.hist(predicted_r[mask1].detach().cpu().numpy(), bins=30, edgecolor="red", range=(-2, 2))
            plt.savefig('train_hist_pred.png')


    cam_poses, dirs = get_camera_rays(fine_mesh, img_size=IMG_SIZE, device=DEVICE)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    mask, t, normals = rough_ray_tracer.trace(cam_poses, dirs)
    pts = cam_poses + dirs * t[:, None]


    t_sdf, sdf_pts, barycentrics, face_idxs = point_query(rough_traverser, pts[mask], DEVICE)

    model.eval()
    prediction = model_intersection(torch.concatenate([pts[mask], dirs[mask]], dim=1), barycentrics=barycentrics, face_idxs=face_idxs)
    
    has_intersection = nn.Sigmoid()(prediction[:, 0])
    prediction = model(torch.concatenate([pts[mask], dirs[mask]], dim=1), barycentrics=barycentrics, face_idxs=face_idxs)
   
    predicted_r = prediction[:, 0]
   
    intersected_mask = has_intersection > 0.5

    pts_masked = pts[mask]
    dirs_masked = dirs[mask]

    prediction = model_normal(torch.concatenate([pts[mask], dirs[mask]], dim=1), barycentrics=barycentrics, face_idxs=face_idxs)
    predicted_normal = prediction[:, :3]
    #predicted_normal = predicted_normal / predicted_normal.norm(dim=1)[:, None]

    predicted_points = pts_masked[intersected_mask] + dirs_masked[intersected_mask] * predicted_r[intersected_mask][:, None] 
    
    mask_true, true_r, normals_traced = fine_ray_tracer.trace(pts, dirs)
    print("true", (torch.sort(true_r, dim=0, descending=True))[0])
    print("true mean", true_r1.mean())

    whole_intesected_mask = mask.clone()
    whole_intesected_mask[mask] = intersected_mask

    plt.clf()
    plt.hist((torch.sort(true_r[whole_intesected_mask], dim=0, descending=True))[0].detach().cpu().numpy(), bins=30, edgecolor="red", range=(-2, 2))
    plt.savefig('val_hist.png')
    plt.clf()
    plt.hist(predicted_r[intersected_mask].detach().cpu().numpy(), bins=30, edgecolor="red", range=(-2, 2))
    plt.savefig('val_hist_pred.png')

    loss = ((predicted_r[intersected_mask] - true_r[whole_intesected_mask]) ** 2)
    print("distance train loss", (predicted_r[intersected_mask] - true_r[whole_intesected_mask]).abs().mean())
    plt.clf()
    plt.hist((predicted_r[intersected_mask] - true_r[whole_intesected_mask]).abs().detach().cpu().numpy(), bins=30, edgecolor="red", range=(-2, 2))
    plt.savefig('val_loss.png')
    
    with torch.no_grad():
        heatmap = torch.zeros((IMG_SIZE * IMG_SIZE), dtype=torch.float32, device=DEVICE)
        heatmap[whole_intesected_mask] = torch.sqrt(loss)
        heatmap = heatmap.cpu().numpy()
        heatmap = heatmap.reshape(IMG_SIZE, IMG_SIZE)
        print("loss max(0.99) =", np.quantile(heatmap, 0.99))
        heatmap = (heatmap - heatmap.min()) / (np.quantile(heatmap, 0.99) - heatmap.min())
        heatmap = np.sqrt(1 - np.square(1 - heatmap))
        image = Image.fromarray((heatmap * 255).astype(np.uint8))
        image.save('loss_heatmap.png')
    
    
    with torch.no_grad():
        points = pts_masked[intersected_mask] + dirs_masked[intersected_mask] * true_r[whole_intesected_mask][:, None] 

        dist_map = torch.zeros((IMG_SIZE * IMG_SIZE), dtype=torch.float32, device=DEVICE)
        dist_map[whole_intesected_mask] = (points - cam_poses[whole_intesected_mask]).norm(dim=1)
        mmin = dist_map[dist_map > 0].min()
        mmax = dist_map.max()
        true_max = mmax
        true_min = mmin
        print("value max =", mmax)
        print(dist_map[143 * IMG_SIZE + 219], ' ', dist_map[138 * IMG_SIZE + 219])
        dist_map = (dist_map - mmin) / (mmax - mmin)
        dist_map[~whole_intesected_mask] = 1.0
        dist_map = 1 - dist_map
        dist_map = dist_map.cpu().numpy()
        dist_map = dist_map.reshape(IMG_SIZE, IMG_SIZE)
        image = Image.fromarray((dist_map * 255).astype(np.uint8))
        image.save('true_distance_map.png')

    with torch.no_grad():
        dist_map = torch.zeros((IMG_SIZE * IMG_SIZE), dtype=torch.float32, device=DEVICE)
        dist_map[whole_intesected_mask] = (predicted_points - cam_poses[whole_intesected_mask]).norm(dim=1)
        plt.clf()
        plt.hist(dist_map[whole_intesected_mask].detach().cpu().numpy(), bins=30, edgecolor="red")
        plt.savefig('val_hist_real_dist.png')
        mmin = dist_map[dist_map > 0].min()
        mmax = dist_map.max()
        print("value max =", mmax)
        print(dist_map[143 * IMG_SIZE + 219], ' ', dist_map[138 * IMG_SIZE + 219])
        dist_map = (dist_map - true_min) / (true_max - true_min)
        dist_map[~whole_intesected_mask] = 1.0
        dist_map = 1 - dist_map
        dist_map = dist_map.cpu().numpy()
        dist_map = dist_map.reshape(IMG_SIZE, IMG_SIZE)
        image = Image.fromarray((dist_map * 255).astype(np.uint8))
        image.save('distance_map.png')
    
    lightnormal = torch.tensor([1.0, -1.0, 1.0], device=DEVICE, dtype=predicted_normal.dtype) 
    lightnormal = lightnormal / lightnormal.norm(dim=0)
    with torch.no_grad():
        pixel_map = torch.zeros((IMG_SIZE * IMG_SIZE), dtype=torch.float32, device=DEVICE)
        pixel_map[~whole_intesected_mask] = 0
        zero = torch.zeros((IMG_SIZE * IMG_SIZE), device=DEVICE)[whole_intesected_mask]
        pixel_map[whole_intesected_mask] = torch.maximum(zero, torch.einsum("ij,j->i", predicted_normal[intersected_mask], lightnormal))
        
        pixel_map = pixel_map / pixel_map.max()
        pixel_map = pixel_map.cpu().numpy()
        pixel_map = pixel_map.reshape(IMG_SIZE, IMG_SIZE)
        image = Image.fromarray((pixel_map * 255).astype(np.uint8))
        image.save('render.png')

    torch.save(model.state_dict(), 'model_distance.pt')
    torch.save(model_intersection.state_dict(), 'model_intersection.pt')

if __name__ == "__main__":
    main()