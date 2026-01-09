import numpy as np
import torch

from renderer_utils import camera_rays_from_pose
from raytrace import RaytraceResult, raytrace_inner_outer


class Renderer:
    def __init__(self, inner_mesh, outer_mesh, inner_net, outer_net, device: str = "cuda"):
        self.inner_mesh = inner_mesh
        self.outer_mesh = outer_mesh
        self.inner_net = inner_net
        self.outer_net = outer_net
        self.device = device

    def draw(self, cam_params, cam_delta):
        img_size = cam_params["img_size"]

        if "dist0" not in cam_params:
            cam_params["dist0"] = cam_params["dist_ref"]

        cam_poses, dirs = camera_rays_from_pose(
            cam_pos=cam_params["pos"],
            yaw=cam_params["yaw"],
            pitch=cam_params["pitch"],
            img_size=img_size,
            max_extent=cam_params["max_extent"],
            dist_ref=cam_params["dist0"],
            device=self.device,
        )
        dirs = dirs / dirs.norm(dim=1, keepdim=True)

        result: RaytraceResult = raytrace_inner_outer(
            cam_poses, dirs, self.inner_mesh, self.outer_mesh, self.inner_net, self.outer_net
        )

        colors = torch.zeros((img_size * img_size,), dtype=torch.float32, device=self.device)
        colors[result.mask] = (-dirs[result.mask] * result.normals[result.mask]).sum(dim=1)
        # colors = torch.abs(colors)
        colors = (colors + 1.0) * 0.5
        colors[~result.mask] = 0.0
        colors = colors.cpu().numpy()
        colors = colors.reshape(img_size, img_size)
        colors = colors * 255
        return np.stack([colors, colors, colors], axis=-1)
