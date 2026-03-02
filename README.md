# RandLA-Net Toronto3D Inference

Run semantic segmentation on a `.ply` point cloud using a pretrained
[RandLA-Net](https://github.com/QingyongHu/RandLA-Net) model and
[Open3D-ML](https://github.com/isl-org/Open3D).

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10 or 3.11 (Open3D wheels are not published for 3.12+) |
| `libGL` / `libEGL` | Required by Open3D on headless Linux: `sudo apt-get install -y libgl1` |
| NumPy | `<2` (Open3D-ML is not compatible with NumPy 2) |
| PyTorch (CPU) | `2.2.*` recommended; install the CPU-only wheel to keep image small |

## Quick start with `uv`

```bash
# 1. Install dependencies
uv sync          # or: uv pip install -r requirements.txt

# 2. Run inference
uv run python -u main.py \
    --input  your_pointcloud.ply \
    --output segmented.ply \
    --labels predicted_labels.npy \
    --device cpu
```

The pretrained weights are downloaded automatically on first run.

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--config` | `randlanet_toronto3d_config.yml` | Path to the YAML pipeline config |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--output` | `output.ply` | Colourised output point cloud |
| `--labels` | `labels.npy` | Raw predicted label array (NumPy) |

If the config file is missing you will see a clear error message that
includes the path that was searched and a suggestion to use `--config`.

## Configuration

The default config is `randlanet_toronto3d_config.yml` in the repo root.
You can point to any compatible YAML file with `--config`:

```bash
uv run python main.py --input cloud.ply --config /path/to/my_config.yml
```
