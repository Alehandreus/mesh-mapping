from types import SimpleNamespace


cfg = SimpleNamespace()

##### GENERAL #####

cfg.device = "cuda"
cfg.mesh_name = "dragon"
cfg.orig_mesh = f"models/{cfg.mesh_name}_orig.fbx"
cfg.inner_mesh = f"models/{cfg.mesh_name}_inner_3000.fbx"
cfg.outer_mesh = f"models/{cfg.mesh_name}_outer_3000.fbx"
# cfg.mesh_name = "chess"
# cfg.orig_mesh = f"/home/me/Downloads/chess_orig.fbx"
# cfg.inner_mesh = f"/home/me/Downloads/chess_inner_10000.fbx"
# cfg.outer_mesh = f"/home/me/Downloads/chess_outer_10000.fbx"
cfg.mesh_n_max_samples = 1_000_000

##### MODEL #####

cfg.model = SimpleNamespace()

cfg.model.encoding_config = {
    "otype": "HashGrid",
    "n_levels": 8,
    "n_features_per_level": 8,
    "log2_hashmap_size": 10,
    "base_resolution": 2,
    "per_level_scale": 2,
    "fixed_point_pos": False,
}

cfg.model.network_config = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 4,
}

##### TRAINING #####

cfg.train = SimpleNamespace()

cfg.train.previews_dir = "previews"
cfg.train.checkpoints_dir = "checkpoints"

cfg.train.tensorboard = True
cfg.train.run_name = None

cfg.train.train_inner = False
cfg.train.train_outer = True

# if not None, load model and optimizer state from given checkpoint
# cfg.train.model_checkpoint = None
cfg.train.model_checkpoint = "/home/me/brain/mesh-mapping/checkpoints/0_400000_chess_3.56_0.02.pt"

cfg.train.lr = 1e-3
cfg.train.lr_scheduler_min = 1.0

cfg.train.steps_total = 1_000_000
# cfg.train.steps_per_epoch = 10_000
cfg.train.steps_per_epoch = 500
cfg.train.batch_size = 50_000

cfg.train.print_interval = 500
cfg.train.tensorboard_interval = 1

##### RENDERING #####

cfg.render = SimpleNamespace()

cfg.render.use_inner = True
cfg.render.use_outer = True

cfg.render.angle = 0.0
cfg.render.img_size = 512

cfg.render.gd_steps = 0
cfg.render.gd_lr = 1
cfg.render.inner_loss_threshold = 10000
cfg.render.outer_loss_threshold = 10000
cfg.render.verbose = True

cfg.render.model_checkpoint = "/home/me/brain/mesh-mapping/checkpoints/0_20000_chess_3.56_nan.pt"
cfg.render.output_dir = "render_outputs"
