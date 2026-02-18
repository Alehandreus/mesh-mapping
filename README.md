# Neural Ray-Mesh Mapping

## Requirements

- PyTorch with CUDA
- [tinycudann](https://github.com/NVlabs/tiny-cuda-nn): `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- [Alehandreus/bvh](https://github.com/Alehandreus/bvh): `pip install git+https://github.com/Alehandreus/bvh`

## Setup

In `config.py`, set the paths to your three meshes and the scene scale:

```python
cfg.scale = 0.1
cfg.fine_mesh_path  = "models/monkey_orig.fbx"
cfg.outer_mesh_path = "models/monkey_outer_1000.fbx"
cfg.inner_mesh_path = "models/monkey_inner_1000.fbx"
```

Alternatively, point `cfg.json_config_path` at a scene JSON (e.g. from `cuda-rendering/dbrt_data/`) to load mesh paths and hash grid settings automatically.

Example monkey meshes:
```bash
gdown 1nh92awGECnb-TTJ-lcRjxdWUntbVCo6h
gdown 1oJOzRWa89QH-74vJ13793W8MkoiaPCOJ
```

## Run

```bash
python train.py
```
