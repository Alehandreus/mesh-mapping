import torch
import numpy as np
from PIL import Image


def save_mesh_previews(meshes, size):
    """Save simple previews for a collection of PyMesh objects keyed by name."""
    for name, mesh in meshes.items():
        mesh.save_preview(name, size, size, mesh.get_c(), mesh.get_R())

@torch.no_grad()
def render_image(data, mask, image_size, path, device):
    map = torch.zeros((image_size * image_size), dtype=torch.float32, device=device)
    map[mask] = data

    map = map.cpu().numpy()
    map = map.reshape(image_size, image_size)
    image = Image.fromarray((map * 255).astype(np.uint8))
    image.save(path)
    return map

def prepare_distance(distance):
    mmin = distance[distance > 0].min()
    mmax = distance.max()
    distance = (distance - mmin) / (mmax - mmin)
    return 1 - distance

@torch.no_grad()
def render_predictions(cfg, points, predicted_points, cam_poses, normals_traced, predicted_normals, intersected_mask, whole_intesected_mask, true_mask):
    true_distance = (points - cam_poses[true_mask]).norm(dim=1)
    true_distance = prepare_distance(true_distance)
    path = f"{cfg.visualization.render_path}/{cfg.visualization.true_distance_render_name}"
    true_map = render_image(true_distance, true_mask, cfg.visualization.image_size, path, cfg.device)

    if predicted_points.numel() > 0:
        predicted_distance = (predicted_points - cam_poses[whole_intesected_mask]).norm(dim=1)
        predicted_distance = prepare_distance(predicted_distance)
    else:
        predicted_distance = torch.tensor([], device=cfg.device)
    path = f"{cfg.visualization.render_path}/{cfg.visualization.predicted_distance_render_name}"
    predicted_map = render_image(predicted_distance, whole_intesected_mask, cfg.visualization.image_size, path, cfg.device)

    difference = np.abs(true_map - predicted_map)
    path = f"{cfg.visualization.render_path}/{cfg.visualization.distance_difference_render_name}"
    image = Image.fromarray((difference * 255).astype(np.uint8))
    image.save(path)

    lightnormal = torch.tensor(cfg.visualization.light_normal, device=cfg.device, dtype=torch.float32) 
    lightnormal = lightnormal / lightnormal.norm(dim=0)
    zero = torch.zeros((cfg.visualization.image_size * cfg.visualization.image_size), device=cfg.device)

    true_pixels = torch.maximum(zero[true_mask], torch.einsum("ij,j->i", normals_traced[true_mask], lightnormal))
    path = f"{cfg.visualization.render_path}/{cfg.visualization.true_mesh_render_name}"
    render_image(true_pixels, true_mask, cfg.visualization.image_size, path, cfg.device)

    predicted_pixels = torch.maximum(zero[whole_intesected_mask], torch.einsum("ij,j->i", predicted_normals[intersected_mask].to(torch.float32), lightnormal))
    path = f"{cfg.visualization.render_path}/{cfg.visualization.predicted_mesh_render_name}"
    render_image(predicted_pixels, whole_intesected_mask, cfg.visualization.image_size, path, cfg.device)

    predicted_pixels_full = torch.zeros((cfg.visualization.image_size * cfg.visualization.image_size), device=cfg.device)
    predicted_pixels_full[whole_intesected_mask] = predicted_pixels

    true_pixels_full = torch.zeros((cfg.visualization.image_size * cfg.visualization.image_size), device=cfg.device)
    true_pixels_full[true_mask] = true_pixels

    # mse = np.square(true_map - predicted_map).mean()
    mse = np.square(true_pixels_full.cpu().numpy() - predicted_pixels_full.cpu().numpy()).mean()
    psnr = np.log10(1 / mse) * 10

    accuracy = (predicted_pixels_full.cpu().numpy() > 0.5) == (true_pixels_full.cpu().numpy() > 0.5)
    accuracy = accuracy.sum() / accuracy.size

    return mse, psnr, accuracy