from types import SimpleNamespace


cfg = SimpleNamespace()

cfg.device = "cuda"
cfg.scale = 10

# cfg.fine_mesh_path = "models/superdragon_orig.obj"
# cfg.outer_mesh_path = "models/superdragon_outer_5000_uv.obj"
# cfg.inner_mesh_path = "models/superdragon_inner_5000.obj"

# cfg.fine_mesh_path = "/home/me/Downloads/petmonster_orig_uv.obj"
# cfg.outer_mesh_path = "/home/me/Downloads/petmonster_outer_2000_uv.obj"
# cfg.inner_mesh_path = "/home/me/Downloads/petmonster_inner_2000_uv.obj"

cfg.fine_mesh_path = "models/chess_orig.obj"
cfg.outer_mesh_path = "models/chess_outer_10000_minconst.obj"
cfg.inner_mesh_path = "models/chess_inner_10000.obj"

# cfg.scale = 10
# cfg.fine_mesh_path = "/home/me/brain/scenes/Andalusian/ours/meshes/andalusian_orig.obj"
# cfg.outer_mesh_path = "/home/me/brain/scenes/Andalusian/ours/meshes/andalusian_outer_10000_voxel.obj"
# cfg.inner_mesh_path = "/home/me/brain/scenes/Andalusian/ours/meshes/andalusian_inner_10000.obj"

# cfg.scale = 0.1
# cfg.fine_mesh_path = "models/monkey_orig.fbx"
# cfg.outer_mesh_path = "models/monkey_outer_1000.fbx"
# cfg.inner_mesh_path = "models/monkey_inner_1000.fbx"

# cfg.fine_mesh_path = "/home/me/Downloads/sphere_orig.obj"
# cfg.outer_mesh_path = "/home/me/Downloads/sphere_outer.obj"
# cfg.inner_mesh_path = "/home/me/Downloads/sphere_inner.obj"

# cfg.scale = 0.1
# cfg.fine_mesh_path = "/home/me/Downloads/statuette_orig.obj"
# cfg.outer_mesh_path = "/home/me/Downloads/statuette_outer_10000_voxel.obj"
# cfg.inner_mesh_path = "/home/me/Downloads/statuette_inner_10000.obj"

# cfg.scale = 0.001
# cfg.fine_mesh_path = "/home/me/Downloads/statuette_orig.fbx"
# cfg.outer_mesh_path = "/home/me/Downloads/statuette_outer_10000.fbx"
# cfg.inner_mesh_path = "/home/me/Downloads/statuette_inner_10000.fbx"

cfg.mesh_n_max_samples = 1_000_000


# MODEL #

cfg.model = SimpleNamespace()

cfg.model.network_config = {
    "otype": "CutlassMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}

# cfg.model.encoding_type = "2d"
#cfg.model.encoding_type = "3d"
cfg.model.encoding_type = "3d+1"

# Can be 'HashGrid' or 'RBF'
cfg.model.encoding = "RBF"

# for 3d point encoding
cfg.model.point_encoding_config = {
    "otype": "HashGrid",
    "n_levels": 8,
    "n_features_per_level": 4,
    "log2_hashmap_size": 16,
    # "base_resolution": 16,
    "base_resolution": 16,
    "per_level_scale": 2,
    "fixed_point_pos": False,
}

# for 2d uv encoding
cfg.model.uv_encoding_config = {
    "otype": "HashGrid",
    "n_levels": 8,
    "n_features_per_level": 4,
    "log2_hashmap_size": 14,
    "base_resolution": 16,
    "per_level_scale": 2,
    "fixed_point_pos": False,
}

# for direction encoding
cfg.model.direction_encoding_config = {
    "otype": "SphericalHarmonics", 
    "degree": 4  
}


# TRAINING #

cfg.train = SimpleNamespace()

# each epoch contains <cfg.train.sample_size> rays
cfg.train.sample_size = 10_000
cfg.train.epochs = 100_000

cfg.train.learning_rate = 1e-3
cfg.train.learning_rate_scheduler_min = 1.0
#cfg.train.learning_rate_scheduler_min = 0.1
cfg.train.learning_rate_scheduler_total_iters = 10 * 5000

# can be 'EMA' or 'SWA', other strings mean that no averaged model used
cfg.train.use_averaged_model = None
#cfg.train.use_averaged_model = None
cfg.train.ema_decay = 0.999
cfg.train.swa_learing_rate = 1e-3

# if not None, load model and optimizer state from given checkpoint
cfg.train.model_start_checkpoint = None
# cfg.train.model_start_checkpoint = "/home/me/brain/mesh-mapping/checkpoints/sphere_debug.pt"

# path where checkpoint will be saved during training
cfg.train.model_save_checkpoint = None
cfg.train.checkpoints_path = "checkpoints"
cfg.train.evaluation_interval = 5000
cfg.train.evaluate = True
cfg.train.save_pt = True
cfg.train.save_bin = True

# tensorboard logging
cfg.train.tensorboard = True
cfg.train.tensorboard_path = "runs"
cfg.train.run_name = None

cfg.train.loss_weights = {
    "cls_loss": 0.1,
    "normal_loss": 1.0,
    "color_loss": 10.0,
    "distance_loss": 0.01,
}


# VISUALIZATION #

cfg.visualization = SimpleNamespace()

cfg.visualization.image_size = 2048

cfg.visualization.light_normal = [1.0, 1.0, 1.0]
# directory where all rendered images will be saved
cfg.visualization.render_path = "rendered"
cfg.visualization.true_distance_render_name = "true_distance_map.png"
cfg.visualization.predicted_distance_render_name = "predicted_distance_map.png"
cfg.visualization.distance_difference_render_name = "distance_difference.png"
cfg.visualization.predicted_mesh_render_name = "predicted_mesh.png"
cfg.visualization.true_mesh_render_name = "true_mesh.png"

# directory where all previews will be saved
cfg.visualization.preview_path = "preview"
cfg.visualization.fine_mesh_preview_name = "fine_preview.png"
cfg.visualization.outer_mesh_preview_name = "outer_preview.png"
cfg.visualization.inner_mesh_preview_name = "inner_preview.png"

# cfg.visualization.camera_angle = 135
cfg.visualization.camera_angle = 0

cfg.visualization.use_neural_renderer = False
cfg.visualization.neural_renderer_path = "/mnt/Programming/RenderingProjects/neural-renderer/build/evaluate"
cfg.visualization.config_json_path = "configs/chess.json"
cfg.visualization.tmp_config_json_path = "/tmp/config.json"