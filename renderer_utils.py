import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


# ============================================================
# 1) Mesh -> initial camera pose (same as your original function)
# ============================================================
def mesh_center_and_extent(mesh) -> Tuple[np.ndarray, float]:
    mesh_min, mesh_max = mesh.get_bounds()
    mesh_min = np.asarray(mesh_min, dtype=np.float64)
    mesh_max = np.asarray(mesh_max, dtype=np.float64)
    center = (mesh_max + mesh_min) * 0.5
    max_extent = float(np.max(mesh_max - mesh_min))
    return center, max_extent


def initial_camera_from_mesh(mesh, angle: float = 0.0) -> Dict[str, Any]:
    """
    Returns the same initial camera position implied by your get_camera_rays():
      base_dx = +1.0 * max_extent
      base_dy = -1.5 * max_extent
      z offset = +0.5 * max_extent
    then rotates that XY offset around +Z by `angle`.

    Also returns initial yaw/pitch aimed at the mesh center, plus initial distance to center.
    """
    center, max_extent = mesh_center_and_extent(mesh)

    base_dx = max_extent * 1.0
    base_dy = -max_extent * 1.5
    r_xy = float(np.hypot(base_dx, base_dy))
    base_theta = float(np.arctan2(base_dy, base_dx))
    theta = base_theta + float(angle)

    cam_pos = np.array(
        [
            center[0] + r_xy * np.cos(theta),
            center[1] + r_xy * np.sin(theta),
            center[2] + max_extent * 0.5,
        ],
        dtype=np.float64,
    )

    look = (center - cam_pos)
    dist = float(np.linalg.norm(look) + 1e-12)
    fwd = look / dist

    yaw = float(np.arctan2(fwd[1], fwd[0]))
    pitch = float(np.arcsin(np.clip(fwd[2], -1.0, 1.0)))

    return {
        "center": center,
        "max_extent": max_extent,
        "pos": cam_pos,
        "yaw": yaw,
        "pitch": pitch,
        "dist0": dist,  # initial distance to center (use as a stable scale reference if you want)
    }


# ============================================================
# 2) Rays from camera position + orientation
# ============================================================
def camera_rays_from_pose(
    cam_pos: np.ndarray,
    yaw: float,
    pitch: float,
    img_size: int,
    max_extent: float,
    dist_ref: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates rays in the SAME style as your get_camera_rays():
      cam_dir length ~ 0.9 * distance
      x_dir and y_dir scaled to max_extent/2
      dirs = cam_dir + x_dir*x + y_dir*y, where x,y in [-1,1]

    Inputs:
      cam_pos: (3,) np array
      yaw/pitch: radians
      img_size: image size (square)
      max_extent: object extent scale
      dist_ref: reference distance (e.g., initial distance to center)
      device: torch device

    Returns:
      d_cam_poses: (N,3) float32
      d_dirs:      (N,3) float32
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    fwd = np.array([cy * cp, sy * cp, sp], dtype=np.float64)
    fwd /= (np.linalg.norm(fwd) + 1e-12)

    cam_dir = fwd * (0.9 * float(dist_ref))  # matches your original "keep same scaling" intent

    x_dir = np.cross(cam_dir, up)
    nx = np.linalg.norm(x_dir)
    if nx < 1e-12:
        # Degenerate (looking almost straight up/down): pick an arbitrary right vector
        x_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        nx = 1.0
    x_dir = (x_dir / nx) * (float(max_extent) / 2.0)

    y_dir = -np.cross(x_dir, cam_dir)
    ny = np.linalg.norm(y_dir)
    y_dir = (y_dir / (ny + 1e-12)) * (float(max_extent) / 2.0)

    x_coords, y_coords = np.meshgrid(
        np.linspace(-1.0, 1.0, img_size, dtype=np.float64),
        np.linspace(-1.0, 1.0, img_size, dtype=np.float64),
        indexing="xy",
    )
    x_coords = x_coords.reshape(-1)
    y_coords = y_coords.reshape(-1)

    dirs = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]
    cam_poses = np.repeat(cam_pos[None, :], dirs.shape[0], axis=0)

    d_cam_poses = torch.from_numpy(cam_poses.astype(np.float32)).to(device)
    d_dirs = torch.from_numpy(dirs.astype(np.float32)).to(device)
    return d_cam_poses, d_dirs