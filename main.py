import numpy as np
import torch
from PIL import Image
from typing import Tuple

from mesh_data import PyMesh, save_mesh_previews
from models import ResidualMap
from raytrace import RaytraceResult, raytrace_inner_outer
from training import TrainingConfig, train_residual_map
from utils import get_camera_rays


CKPT_PATH = "mapping.pt"


def build_networks(inner_mesh: PyMesh, outer_mesh: PyMesh, device: str):
    inner_net = ResidualMap(inner_mesh.mesh_split).to(device)
    outer_net = ResidualMap(outer_mesh.mesh_split).to(device)
    return inner_net, outer_net


def load_or_train_networks(orig_mesh: PyMesh, inner_mesh: PyMesh, outer_mesh: PyMesh, device: str, load_ckpt: bool, load_optimizer: bool):
    inner_net, outer_net = build_networks(inner_mesh, outer_mesh, device)

    if load_ckpt:
        print(f"Loading checkpoint from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        inner_net.load_state_dict(ckpt["inner_net"])
        outer_net.load_state_dict(ckpt["outer_net"])
        return inner_net, outer_net

    cfg = TrainingConfig()
    cfg.load_optimizer = load_optimizer
    train_residual_map(inner_net, orig_mesh, inner_mesh, cfg)
    # train_residual_map(outer_net, orig_mesh, outer_mesh, cfg)

    print(f"Saving checkpoint to {CKPT_PATH}...")
    torch.save(
        {
            "inner_net": inner_net.state_dict(),
            "outer_net": outer_net.state_dict(),
        },
        CKPT_PATH,
    )
    return inner_net, outer_net


def render_camera_angle(
    orig_mesh: PyMesh,
    inner_mesh: PyMesh,
    outer_mesh: PyMesh,
    inner_net,
    outer_net,
    img_size: int,
    device: str,
    angle: float = 0.0,
) -> Tuple[Image.Image, Image.Image]:
    cam_poses, dirs = get_camera_rays(orig_mesh.mesh, img_size=img_size, device=device, angle=angle)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    from raytrace import get_raytrace_loss

    from raytrace import RaytraceConfig
    config = RaytraceConfig()
    config.epochs = 0
    # config.epochs = 100
    config.lr = 100
    # config.lr = 1
    # config.threshold = 0.02
    # config.threshold_edges = 0.002

    config.threshold = 100
    config.threshold_edges = 100

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        result: RaytraceResult = raytrace_inner_outer(cam_poses, dirs, inner_mesh, outer_mesh, inner_net, outer_net, config=config, verbose=True)

    loss = get_raytrace_loss(cam_poses, dirs, result.y, reduction="none")
    outer_mask, _, _ = outer_mesh.ray_tracer.trace(cam_poses, dirs)
    orig_mask, orig_t, orig_normals = orig_mesh.ray_tracer.trace(cam_poses, dirs)
    inner_mask, _, _ = inner_mesh.ray_tracer.trace(cam_poses, dirs)

    mask_1 = orig_mask & (~inner_mask)
    mask_2 = outer_mask & (~orig_mask)

    edge_mask = outer_mask & (~inner_mask) & (loss > config.threshold_edges)
    result.mask &= (~edge_mask)

    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # ax1.hist(loss[mask_1].cpu().numpy(), bins=100)
    # ax1.set_title("Loss histogram for rays intersecting orig but not inner mesh")
    # ax1.set_xlabel("Loss")
    # ax1.set_ylabel("Number of rays")
    # ax2.hist(loss[mask_2].cpu().numpy(), bins=100)
    # ax2.set_title("Loss histogram for rays intersecting outer but not orig mesh")
    # ax2.set_xlabel("Loss")
    # ax2.set_ylabel("Number of rays")
    # plt.tight_layout()
    # plt.show()

    # result.normals[result.mask] = orig_normals[result.mask]

    img_acc = (result.mask == orig_mask).float().mean().item()
    print(f"Image accuracy (matching orig mesh intersections): {img_acc * 100:.2f}%")

    orig_y = cam_poses + dirs * orig_t[:, None]
    img_mse = ((result.y[orig_mask] - orig_y[orig_mask]) ** 2).sum(dim=1).mean().item() if orig_mask.any() else 0.0
    print(f"Image MSE on orig mesh intersections: {img_mse:.6f}")

    # compute mean cosine between result.normals and orig_normals where orig_mask is True
    if orig_mask.any():
        cos_sim = (result.normals[orig_mask] * orig_normals[orig_mask]).sum(dim=1)
        mean_cos_sim = cos_sim.mean().item()
    else:
        mean_cos_sim = 0.0
    print(f"Mean cosine similarity of normals on orig mesh intersections: {mean_cos_sim:.6f}")

    print(f"Mean pixel error: {(result.normals[orig_mask] - orig_normals[orig_mask]).square().mean()}")

    distance_image = _distance_map_image(result, cam_poses, img_size)
    normal_image = _normal_shading_image(result, dirs, img_size)    
    loss_image = _loss_heatmap_image(get_raytrace_loss(cam_poses, dirs, result.y, reduction="none"), img_size, result.mask)

    def total_variation(img: Image.Image) -> float:
        img_np = np.array(img).astype(np.float32) / 255.0
        tv = np.sum(np.abs(img_np[:, 1:] - img_np[:, :-1])) + np.sum(np.abs(img_np[1:, :] - img_np[:-1, :]))
        return tv
    print(f"Normal image Total Variation: {total_variation(normal_image):.6f}")

    return normal_image, distance_image, loss_image


def _distance_map_image(result: RaytraceResult, cam_poses: torch.Tensor, img_size: int) -> Image.Image:
    dist_map = torch.ones((img_size * img_size,), dtype=torch.float32, device=cam_poses.device)

    if result.mask.any():
        distances = (result.y[result.mask] - cam_poses[result.mask]).norm(dim=1)
        mmin = distances.min()
        mmax = distances.max()
        denom = mmax - mmin
        if denom > 1e-12:
            distances = (distances - mmin) / denom
        dist_map[result.mask] = 1 - distances
    else:
        dist_map.fill_(0.0)

    dist_map = dist_map.reshape(img_size, img_size).cpu().numpy()
    return Image.fromarray((dist_map * 255).astype(np.uint8))


def _normal_shading_image(result: RaytraceResult, dirs: torch.Tensor, img_size: int) -> Image.Image:
    colors = torch.zeros((img_size * img_size,), dtype=torch.float32, device=dirs.device)
    colors[result.mask] = (-dirs[result.mask] * result.normals[result.mask]).sum(dim=1)
    # colors = torch.abs(colors)
    colors = (colors + 1.0) * 0.5
    colors[~result.mask] = 0.0
    colors = colors.cpu().numpy().reshape(img_size, img_size)
    return Image.fromarray((colors * 255).astype(np.uint8))


def _loss_heatmap_image(loss_values: torch.Tensor, img_size: int, mask: torch.Tensor) -> Image.Image:
    loss_map = loss_values.cpu().numpy().reshape(img_size, img_size)
    mask = mask.reshape(img_size, img_size)
    mmin = loss_map[mask.cpu().numpy()].min() if mask.any() else 0.0
    mmax = loss_map[mask.cpu().numpy()].max() if mask.any() else 1.0
    denom = mmax - mmin
    if denom > 1e-12:
        loss_map = (loss_map - mmin) / denom
    loss_map = np.clip(loss_map, 0.0, 1.0)
    return Image.fromarray((loss_map * 255).astype(np.uint8))


def main():
    device = "cuda"
    img_size = 1024
    raytrace = True
    # load_ckpt = False

    # read load_ckpt from command line args
    import sys
    load_ckpt = "--load-ckpt" in sys.argv
    load_optimizer = "--load-optimizer" in sys.argv

    mesh_paths = {
        "orig": "models/petmonster_orig.fbx",
        "inner": "models/petmonster_inner_2000.fbx",
        "outer": "models/petmonster_outer_2000.fbx",
        # "orig": "models/dragon_orig.fbx",
        # "inner": "models/dragon_inner_3000.fbx",
        # "outer": "models/dragon_outer_3000.fbx",
        # "orig": "models/superdragon_orig.fbx",
        # "inner": "models/superdragon_inner_5000.fbx",
        # "outer": "models/superdragon_outer_5000.fbx",
        # "orig": "models/monkey_orig.fbx",
        # "inner": "models/monkey_inner_1000.fbx",
        # "outer": "models/monkey_outer_1000.fbx",
        # "orig": "models/sphere_orig.fbx",
        # "inner": "models/sphere_inner_2.fbx",
        # "outer": "models/sphere_outer.fbx",
    }

    orig_mesh = PyMesh.from_file(mesh_paths["orig"])
    inner_mesh = PyMesh.from_file(mesh_paths["inner"])
    outer_mesh = PyMesh.from_file(mesh_paths["outer"])

    save_mesh_previews({"outer": outer_mesh, "inner": inner_mesh, "orig": orig_mesh}, size=512)

    orig_mesh.mesh.save_preview("orig_mesh_preview.png", 512, 512, orig_mesh.mesh.get_c(), orig_mesh.mesh.get_R())
    inner_mesh.mesh.save_preview("inner_mesh_preview.png", 512, 512, orig_mesh.mesh.get_c(), orig_mesh.mesh.get_R())
    outer_mesh.mesh.save_preview("outer_mesh_preview.png", 512, 512, orig_mesh.mesh.get_c(), orig_mesh.mesh.get_R())

    inner_net, outer_net = load_or_train_networks(orig_mesh, inner_mesh, outer_mesh, device, load_ckpt, load_optimizer)

    params = inner_net.network.params.data.cpu().numpy()
    print(params.shape, params.dtype)
    params.astype(np.float16).tofile("inner_params.bin")

    if not raytrace:
        return

    n_frames = 1
    angles = np.linspace(0, np.pi * 2, n_frames, endpoint=False) #+ 2.7
    frames = []

    for angle in angles:
        import time
        start_time = time.time()
        normal_img, distance_img, loss_img = render_camera_angle(
            orig_mesh, inner_mesh, outer_mesh, inner_net, outer_net, img_size, device, angle=angle
        )
        end_time = time.time()
        print(f"Rendered angle {angle:.2f} in {end_time - start_time:.2f} seconds.")

        distance_img.save("distance_map.png")
        normal_img.save("normal_shading.png")
        loss_img.save("loss_heatmap.png")
        frames.append(normal_img)

    if n_frames > 1:
        frames[0].save("mapping_animation.gif", save_all=True, append_images=frames[1:], duration=n_frames * 0.5, loop=0)


if __name__ == "__main__":
    main()
