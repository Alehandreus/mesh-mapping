import torch
import numpy as np
import pickle
import os
import json
from types import SimpleNamespace
import csv

from train import eval_model
import config as cfg_module
from utils import MeshWrapper
from models import RayModel


def get_camera_rays(img_size, camera_matrix, yfov):
    x_coords, y_coords = np.meshgrid(
        np.linspace(-1, 1, img_size),
        np.linspace(-1, 1, img_size),
    )
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    #yfov = np.radians(yfov)
    focal_length = 2 / np.tan(yfov / 2)

    dirs = np.stack([x_coords / focal_length, -y_coords / focal_length, -np.ones_like(x_coords)], axis=1)
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    dirs = np.concatenate([dirs,  np.ones_like(x_coords)[:, None]], axis=1)

    dirs = (camera_matrix @ dirs.T).T
    dirs = (dirs / dirs[:, 3][:, None])[:, :3]

    cam_poses = camera_matrix[:3, 3][None, :]
    cam_poses = np.repeat(cam_poses, dirs.shape[0], axis=0)

    return cam_poses, dirs


def main():
    cfg = cfg_module.cfg
    with open("render_config.json", "r") as file:
        config = json.load(file)

    for scene in config["scenes"]:
        scale = scene["scale"]
        fine_mesh = MeshWrapper.from_file(scene["fine_mesh_path"], n_max_samples=cfg.mesh_n_max_samples, scale=scale)
        outer_mesh = MeshWrapper.from_file(scene["outer_mesh_path"], n_max_samples=cfg.mesh_n_max_samples, scale=scale)
        inner_mesh = MeshWrapper.from_file(scene["inner_mesh_path"], n_max_samples=cfg.mesh_n_max_samples, scale=scale)

        table = []

        for model_json in scene["model_versions"]:
            model_config = SimpleNamespace()
            model_config.network_config = model_json["config"]["network_config"]
            model_config.encoding_type = model_json["config"]["encoding_type"]
            model_config.encoding = model_json["config"]["encoding"]
            model_config.uv_encoding_config = model_json["config"]["uv_encoding_config"]
            model_config.direction_encoding_config = model_json["config"]["direction_encoding_config"]
            model_config.point_encoding_config = model_json["config"]["point_encoding_config"]

            model = RayModel(model_config, outer_mesh.mesh, cfg).to(cfg.device)

            with open(model_json["weights_path"], "rb") as file:
                weights = pickle.load(file)
                for name, param in model.point_encoding.named_parameters():
                    param.data = torch.from_numpy(weights[name]).to(cfg.device)

                #model.direction_encoding.params.data = weights["other_weights"][:] Because of constant transform
                model.network.params.data = torch.from_numpy(weights["other_weights"]).to(cfg.device)

            for camera, index in zip(scene["cameras"], range(len(scene["cameras"]))):
                yfov = camera["yfov"]
                camera_matrix = camera["matrix"]
                camera_matrix = np.array(camera_matrix).reshape(4, 4).T
                camera_matrix = np.linalg.inv(camera_matrix)
                camera_matrix = camera_matrix * scale

                cam_poses, dirs = get_camera_rays(2048, camera_matrix, yfov)
                d_cam_poses = torch.from_numpy(cam_poses).float().to(cfg.device)
                d_dirs = torch.from_numpy(dirs).float().to(cfg.device)

                cfg.visualization.render_path = f"{config['path']}/{scene['name']}/{model_json['name']}"
                cfg.visualization.predicted_mesh_render_name = f"predicted_mesh_cam{index + 1}.png"
                cfg.visualization.true_mesh_render_name = f"true_mesh_cam{index + 1}.png"

                os.makedirs(cfg.visualization.render_path, exist_ok=True)

                mse, psnr, accuracy = eval_model(cfg, d_cam_poses, d_dirs, fine_mesh, outer_mesh, inner_mesh, model)
                print(f"{model_json['name']}_cam{index + 1}: mse={mse}, psnr={psnr}, accuracy={accuracy}")
                table.append([f"{model_json['name']}_cam{index + 1}", mse, psnr, accuracy])


    headers = ["Experiment", "MSE", "PSNR", "Accuracy"]
    with open(f"{config['path']}/{scene['name']}/comparison.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(table)


if __name__ == "__main__":
    main()
