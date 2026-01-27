from types import SimpleNamespace


cfg = SimpleNamespace()

cfg.device = "cuda"

cfg.fine_mesh_path = "models/chess_orig.fbx"
cfg.outer_mesh_path = "models/chess_outer_10000.fbx"
cfg.mesh_n_max_samples = 1_000_000


# MODEL #

cfg.model = SimpleNamespace()

cfg.model.network_config = {
    "otype": "FullyFusedMLP",
    "activation": "LeakyReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 4,
}
cfg.model.point_encoding_config = {
    "otype": "HashGrid",
    "n_levels": 8,
    "n_features_per_level": 8,
    "log2_hashmap_size": 16,
    "base_resolution": 16,
    "per_level_scale": 2,
    "fixed_point_pos": False,
}
cfg.model.direction_encoding_config = {
    "otype": "SphericalHarmonics", 
    "degree": 4  
}


# TRAINING #

cfg.train = SimpleNamespace()

# each epoch contains <cfg.train.sample_size> rays
cfg.train.sample_size = 100_000
cfg.train.epochs = 10_000 

cfg.train.learning_rate = 1e-2
cfg.train.learning_rate_scheduler_min = 1.0

# can be 'EMA' or 'SWA', other strings mean that no averaged model used
cfg.train.use_averaged_model = 'EMA'
cfg.train.ema_decay = 0.999
cfg.train.swa_learing_rate = 1e-3

# if not None, load model and optimizer state from given checkpoint
cfg.train.model_start_checkpoint = None

# path where checkpoint will be saved during training
cfg.train.model_save_checkpoint = None
cfg.train.checkpoints_path = "checkpoints"
cfg.train.checkpoints_interval = 100

# tensorboard logging
cfg.train.tensorboard = False
cfg.train.tensorboard_path = "runs"
cfg.train.tensorboard_run_name = None


# VISUALIZATION #

cfg.visualization = SimpleNamespace()

cfg.visualization.image_size = 800

cfg.visualization.light_normal = [1.0, 1.0, 1.0]
cfg.visualization.render_interval = 100
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
