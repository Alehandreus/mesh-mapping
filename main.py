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


def load_or_train_networks(orig_mesh: PyMesh, inner_mesh: PyMesh, outer_mesh: PyMesh, device: str, load_ckpt: bool):
    inner_net, outer_net = build_networks(inner_mesh, outer_mesh, device)

    if load_ckpt:
        print(f"Loading checkpoint from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        inner_net.load_state_dict(ckpt["inner_net"])
        outer_net.load_state_dict(ckpt["outer_net"])
        return inner_net, outer_net

    cfg = TrainingConfig()
    train_residual_map(inner_net, orig_mesh, inner_mesh, cfg)
    train_residual_map(outer_net, orig_mesh, outer_mesh, cfg)

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

    result: RaytraceResult = raytrace_inner_outer(cam_poses, dirs, inner_mesh, outer_mesh, inner_net, outer_net)

    distance_image = _distance_map_image(result, cam_poses, img_size)
    normal_image = _normal_shading_image(result, dirs, img_size)
    return normal_image, distance_image


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
    colors = torch.abs(colors)
    colors = (colors + 1.0) * 0.5
    colors[~result.mask] = 0.0
    colors = colors.cpu().numpy().reshape(img_size, img_size)
    return Image.fromarray((colors * 255).astype(np.uint8))


def main():
    device = "cuda"
    img_size = 512
    raytrace = True
    load_ckpt = False

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
        # "inner": "models/sphere_inner.fbx",
        # "outer": "models/sphere_outer.fbx",
    }

    orig_mesh = PyMesh.from_file(mesh_paths["orig"])
    inner_mesh = PyMesh.from_file(mesh_paths["inner"])
    outer_mesh = PyMesh.from_file(mesh_paths["outer"])

    save_mesh_previews({"outer": outer_mesh, "inner": inner_mesh, "orig": orig_mesh}, size=512)

    inner_net, outer_net = load_or_train_networks(orig_mesh, inner_mesh, outer_mesh, device, load_ckpt)

    if not raytrace:
        return

    n_frames = 1
    angles = np.linspace(0, np.pi * 2, n_frames, endpoint=False)
    frames = []

    for angle in angles:
        normal_img, distance_img = render_camera_angle(
            orig_mesh, inner_mesh, outer_mesh, inner_net, outer_net, img_size, device, angle=angle
        )

        distance_img.save("distance_map.png")
        normal_img.save("normal_shading.png")
        frames.append(normal_img)

    if n_frames > 1:
        frames[0].save("mapping_animation.gif", save_all=True, append_images=frames[1:], duration=n_frames * 0.5, loop=0)


if __name__ == "__main__":
    main()
