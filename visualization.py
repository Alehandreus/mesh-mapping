import torch
import numpy as np
import json
import subprocess
from pathlib import Path
from PIL import Image


def save_mesh_previews(meshes, size):
    """Save simple previews for a collection of PyMesh objects keyed by name."""
    for name, mesh in meshes.items():
        mesh.save_preview(name, size, size)

@torch.no_grad()
def render_image(data, mask, image_size, path, device):
    map = torch.zeros((image_size * image_size, 3), dtype=torch.float32, device=device)
    map[mask] = data
    # map[~mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)

    map = map.cpu().numpy()
    map = map.reshape(image_size, image_size, 3)
    image = Image.fromarray((map * 255).astype(np.uint8))
    image.save(path)
    return map

def prepare_distance(distance):
    mmin = distance[distance > 0].min()
    mmax = distance.max()
    distance = (distance - mmin) / (mmax - mmin)
    return 1 - distance

@torch.no_grad()
def render_predictions(cfg, points, predicted_points, cam_poses, normals_traced, predicted_normals, whole_intersected_mask, true_mask,
                       whole_pred_colors, colors):
    true_distance = (points - cam_poses[true_mask]).norm(dim=1)
    true_distance = prepare_distance(true_distance)
    path = f"{cfg.visualization.render_path}/{cfg.visualization.true_distance_render_name}"
    # true_map = render_image(true_distance, true_mask, cfg.visualization.image_size, path, cfg.device)

    if predicted_points.numel() > 0:
        predicted_distance = (predicted_points - cam_poses[whole_intersected_mask]).norm(dim=1)
        predicted_distance = prepare_distance(predicted_distance)
    else:
        predicted_distance = torch.tensor([], device=cfg.device)
    path = f"{cfg.visualization.render_path}/{cfg.visualization.predicted_distance_render_name}"
    # predicted_map = render_image(predicted_distance, whole_intersected_mask, cfg.visualization.image_size, path, cfg.device)

    # difference = np.abs(true_map - predicted_map)
    # path = f"{cfg.visualization.render_path}/{cfg.visualization.distance_difference_render_name}"
    # image = Image.fromarray((difference * 255).astype(np.uint8))
    # image.save(path)

    lightnormal = torch.tensor(cfg.visualization.light_normal, device=cfg.device, dtype=torch.float32) 
    lightnormal = lightnormal / lightnormal.norm(dim=0)
    zero = torch.zeros((cfg.visualization.image_size * cfg.visualization.image_size), device=cfg.device)

    true_pixels = torch.maximum(zero[true_mask], torch.einsum("ij,j->i", normals_traced[true_mask], lightnormal))[..., None] * torch.clamp(colors[true_mask], 0, 1)
    path = f"{cfg.visualization.render_path}/{cfg.visualization.true_mesh_render_name}"
    render_image(true_pixels, true_mask, cfg.visualization.image_size, path, cfg.device)

    predicted_pixels = torch.maximum(zero[whole_intersected_mask], torch.einsum("ij,j->i", predicted_normals[whole_intersected_mask].to(torch.float32), lightnormal))[..., None] * torch.clamp(whole_pred_colors[whole_intersected_mask], 0, 1)
    path = f"{cfg.visualization.render_path}/{cfg.visualization.predicted_mesh_render_name}"
    render_image(predicted_pixels, whole_intersected_mask, cfg.visualization.image_size, path, cfg.device)

    predicted_pixels_full = torch.zeros((cfg.visualization.image_size * cfg.visualization.image_size, 3), device=cfg.device)
    predicted_pixels_full[whole_intersected_mask] = predicted_pixels

    true_pixels_full = torch.zeros((cfg.visualization.image_size * cfg.visualization.image_size, 3), device=cfg.device)
    true_pixels_full[true_mask] = true_pixels

    # mse = np.square(true_map - predicted_map).mean()
    mse = np.square(true_pixels_full.cpu().numpy() - predicted_pixels_full.cpu().numpy()).mean()
    psnr = np.log10(1 / mse) * 10

    accuracy = (predicted_pixels_full.cpu().numpy() > 0.5) == (true_pixels_full.cpu().numpy() > 0.5)
    accuracy = accuracy.sum() / accuracy.size

    return mse, psnr, accuracy

def _resolve_path(base_dir, path_str):
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)

def prepare_neural_renderer(cfg, run_name):
    json_path = Path(cfg.json_config_path).resolve()
    base_dir = json_path.parent
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for key in ("original_mesh", "inner_shell", "outer_shell", "additional_mesh"):
        if key in config.get("scene", {}):
            config["scene"][key]["path"] = _resolve_path(base_dir, config["scene"][key]["path"])

    if "hdri_path" in config.get("environment", {}):
        config["environment"]["hdri_path"] = _resolve_path(base_dir, config["environment"]["hdri_path"])

    checkpoint_path = Path(f"{cfg.train.checkpoints_path}/{run_name}.bin")
    config["checkpoint_path"] = str(checkpoint_path.resolve())

    with open(cfg.visualization.tmp_config_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def render_predictions_with_neural_renderer(cfg, run_name):
    prepare_neural_renderer(cfg, run_name)

    neural_renderer_path = Path(cfg.visualization.neural_renderer_path)
    tmp_config_json_path = Path(cfg.visualization.tmp_config_json_path)
    logs = subprocess.run([str(neural_renderer_path.resolve()), str(tmp_config_json_path.resolve())], capture_output=True, text=True)
    logs = str(logs)
    try:
        psnr_index = logs.find("PSNR:") + 6
        psnr = float(logs[psnr_index:logs.find(' ', psnr_index)])
        flip_index = logs.find("FLIP:") + 6
        flip = float(logs[flip_index:logs.find(' ', flip_index)])
    except Exception as e:
        print("Failed to parse neural renderer output:", logs)
        exit()
    return psnr, flip