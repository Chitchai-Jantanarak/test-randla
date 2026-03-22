"""
RandLA-Net semantic segmentation inference.

Supports three input modes:
  1. 3D only      — point cloud (PLY/LAS) with RGB
  2. 3D + BEV     — adds self-computed BEV features from point cloud
  3. 3D + 2D      — adds raster-projected features (footprints, DEM, DSM)
  4. 3D + BEV + 2D — all combined (maximum context)

When 2D data is available it gets projected onto points as extra feature
channels. When not available, falls back gracefully to 3D-only.

Handles extremely large files (150-300 GB LAS) via chunked processing.

Usage examples:
  # 3D only (original behavior)
  python main.py --input scan.las --labels out.npy

  # 3D + BEV features
  python main.py --input scan.las --labels out.npy --bev

  # 3D + 2D (dual input — footprints + DEM)
  python main.py --input scan.las --labels out.npy \\
      --footprints buildings.tif --dem ground.tif

  # 3D + BEV + 2D (full stack)
  python main.py --input scan.las --labels out.npy --bev \\
      --footprints buildings.tif --dem ground.tif --dsm surface.tif

  # Large file with chunked processing
  python main.py --input huge.las --labels out.npy --chunk-size 5000000 \\
      --footprints buildings.tif --dem ground.tif
"""

import os
import sys
import argparse
import urllib.request
import numpy as np

try:
    import laspy
except ImportError:
    laspy = None

import torch
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from bev_features import compute_bev_features, compute_bev_features_chunked
from raster_features import compute_raster_features

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "randlanet_toronto3d_config.yml")
WEIGHTS_URL = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_toronto3d_202201071330utc.pth"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Pretrained checkpoint was trained with 6 input channels (xyz + rgb).
PRETRAINED_IN_CHANNELS = 6


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

def download_weights(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        print("Downloading pretrained weights ...")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, ckpt_path)
        print("Download complete.")


# ---------------------------------------------------------------------------
# Freeze + Extend: adapt pretrained weights for extra input channels
# ---------------------------------------------------------------------------
# Open3D-ML RandLANet first layer: fc0 = nn.Linear(in_channels, dim_features)
# Checkpoint has fc0.weight (dim_features, 6) and fc0.bias (dim_features,).
#
# Strategy:
#   - Keep the first 6 columns of fc0.weight (pretrained RGB knowledge)
#   - Append new columns for BEV / 2D channels, initialized with Xavier
#   - Bias stays unchanged (same output dim)
#   - All other layers are untouched (they don't depend on in_channels)

# Keys in the Open3D-ML RandLANet checkpoint that depend on in_channels.
# fc0 is the input projection layer: nn.Linear(in_channels, dim_features).
# In the saved state_dict these appear as:
#   fc0.weight  →  shape (dim_features, in_channels)
#   fc0.bias    →  shape (dim_features,)
#
# Additionally, the first encoder's local spatial encoding (fc_encoder_0)
# contains a sub-layer whose weight has shape (*, 2 * in_channels) because
# RandLA-Net concatenates [point_features, relative_position_features].
# We handle both cases by scanning for any weight whose last dimension
# equals PRETRAINED_IN_CHANNELS.

def extend_checkpoint(
    ckpt_path: str,
    target_in_channels: int,
    device: str = "cpu",
) -> dict:
    """Load pretrained checkpoint and extend fc0 weights for more channels.

    Returns the modified state_dict ready for model.load_state_dict().

    Mode 1 (6 ch): returns original weights unchanged.
    Mode 2 (11 ch): extends fc0 by 5 columns for BEV.
    Mode 3 (9 ch): extends fc0 by 3 columns for 2D raster.
    Mode 4 (14 ch): extends fc0 by 8 columns for BEV + 2D.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Open3D-ML checkpoints may wrap state_dict under various keys
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if target_in_channels == PRETRAINED_IN_CHANNELS:
        print(f"  Weights: using original checkpoint (in_channels={PRETRAINED_IN_CHANNELS})")
        return state_dict

    extra = target_in_channels - PRETRAINED_IN_CHANNELS
    print(f"  Weights: Freeze+Extend fc0 from {PRETRAINED_IN_CHANNELS} → {target_in_channels} "
          f"(+{extra} channels)")

    extended_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dim() < 2:
            # bias or 1D param — keep as-is
            extended_state_dict[key] = tensor
            continue

        # Check if the last dimension matches pretrained in_channels
        # fc0.weight has shape (dim_features, in_channels)
        last_dim = tensor.shape[-1]

        if last_dim == PRETRAINED_IN_CHANNELS:
            # This is fc0.weight or similar — extend it
            new_shape = list(tensor.shape)
            new_shape[-1] = target_in_channels
            extended = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

            # Copy pretrained weights for first 6 channels
            extended[..., :PRETRAINED_IN_CHANNELS] = tensor

            # Xavier init for new channels
            fan_in = target_in_channels
            fan_out = tensor.shape[0] if tensor.dim() == 2 else tensor.shape[-2]
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            torch.nn.init.normal_(extended[..., PRETRAINED_IN_CHANNELS:], mean=0.0, std=std)

            extended_state_dict[key] = extended
            print(f"    Extended: {key}  {list(tensor.shape)} → {list(extended.shape)}")

        elif last_dim == 2 * PRETRAINED_IN_CHANNELS:
            # RandLA-Net local spatial encoding concatenates features:
            # [d_features || relative_position] where relative_position = 2*in_channels
            new_last = 2 * target_in_channels
            new_shape = list(tensor.shape)
            new_shape[-1] = new_last
            extended = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
            extended[..., :2 * PRETRAINED_IN_CHANNELS] = tensor

            fan_in = new_last
            fan_out = tensor.shape[0] if tensor.dim() == 2 else tensor.shape[-2]
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            torch.nn.init.normal_(extended[..., 2 * PRETRAINED_IN_CHANNELS:], mean=0.0, std=std)

            extended_state_dict[key] = extended
            print(f"    Extended: {key}  {list(tensor.shape)} → {list(extended.shape)}")

        else:
            # Doesn't depend on in_channels — keep unchanged
            extended_state_dict[key] = tensor

    return extended_state_dict


def load_weights_extended(pipeline, ckpt_path: str, target_in_channels: int, device: str = "cpu"):
    """Load checkpoint into pipeline, extending weights if needed."""
    if target_in_channels == PRETRAINED_IN_CHANNELS:
        # Mode 1: original 6-channel — use standard loading
        pipeline.load_ckpt(ckpt_path=ckpt_path)
    else:
        # Mode 2/3/4: extended channels — manual weight surgery
        extended_sd = extend_checkpoint(ckpt_path, target_in_channels, device=device)
        pipeline.model.load_state_dict(extended_sd, strict=False)
        print(f"  Loaded extended weights into model (strict=False)")


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def create_pipeline(cfg_file: str | None = None, device: str = "cpu"):
    if cfg_file is None:
        cfg_file = DEFAULT_CONFIG

    if not os.path.exists(cfg_file):
        raise FileNotFoundError(
            f"Config file not found: '{cfg_file}'\n"
            "Pass the correct path with --config <path>"
        )

    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    model = ml3d.models.RandLANet(**cfg.model)

    dataset_cfg = dict(cfg.dataset)
    dataset_cfg.pop("dataset_path", None)
    dataset = ml3d.datasets.SemanticKITTI("", **dataset_cfg)

    pipeline = ml3d.pipelines.SemanticSegmentation(
        model, dataset=dataset, device=device, **cfg.pipeline
    )
    return pipeline


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------

def read_pointcloud_ply(path: str):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0 if len(pcd.colors) > 0 else np.zeros_like(points)
    return points.astype(np.float32), colors.astype(np.float32)


def iter_las_chunks(path: str, chunk_size: int = 5_000_000):
    if laspy is None:
        sys.exit("laspy is required for LAS/LAZ files.  pip install laspy")

    with laspy.open(path) as reader:
        total = reader.header.point_count
        print(f"LAS file: {total:,} points total, reading in chunks of {chunk_size:,}")

        for chunk in reader.chunk_iterator(chunk_size):
            pts = np.stack([chunk.x, chunk.y, chunk.z], axis=-1).astype(np.float32)
            try:
                colors = np.stack([
                    chunk.red / 256.0,
                    chunk.green / 256.0,
                    chunk.blue / 256.0,
                ], axis=-1).astype(np.float32)
            except Exception:
                colors = np.zeros_like(pts)
            yield pts, colors


def read_pointcloud(path: str, chunk_size: int = 0):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        pts, colors = read_pointcloud_ply(path)
        return [(pts, colors)]

    elif ext in (".las", ".laz"):
        if chunk_size > 0:
            return iter_las_chunks(path, chunk_size)
        if laspy is None:
            sys.exit("laspy is required.  pip install laspy")
        las = laspy.read(path)
        n = las.header.point_count
        print(f"LAS file: {n:,} points, loading all at once (~{n * 3 * 4 / 1e9:.1f} GB for xyz float32)")
        pts = np.empty((n, 3), dtype=np.float32)
        pts[:, 0] = las.x
        pts[:, 1] = las.y
        pts[:, 2] = las.z
        try:
            colors = np.empty((n, 3), dtype=np.float32)
            colors[:, 0] = las.red / 256.0
            colors[:, 1] = las.green / 256.0
            colors[:, 2] = las.blue / 256.0
        except Exception:
            colors = np.zeros((n, 3), dtype=np.float32)
        del las  # free laspy internal buffers (~60 GB for 2B points)
        return [(pts, colors)]

    else:
        sys.exit(f"Unsupported format: {ext}  (use .ply, .las, or .laz)")


# ---------------------------------------------------------------------------
# Feature assembly — combines RGB + BEV + 2D raster into one feat array
# ---------------------------------------------------------------------------

def build_features(
    points: np.ndarray,
    colors: np.ndarray,
    use_bev: bool = False,
    bev_cell_size: float = 1.0,
    footprint_path: str | None = None,
    dem_path: str | None = None,
    dsm_path: str | None = None,
) -> np.ndarray:
    """Assemble the full per-point feature vector.

    Channel layout (concatenated left to right):
      [RGB (3)] + [BEV (5) if --bev] + [2D raster (1-4) if --footprints/--dem/--dsm]

    Returns: (N, C) float32
    """
    parts = [colors]

    # BEV features from point cloud itself
    if use_bev:
        bev = compute_bev_features(points, cell_size=bev_cell_size)
        parts.append(bev)

    # 2D raster features (only when available)
    has_2d = any(p is not None for p in [footprint_path, dem_path, dsm_path])
    if has_2d:
        raster_feats, raster_names = compute_raster_features(
            points,
            footprint_path=footprint_path,
            dem_path=dem_path,
            dsm_path=dsm_path,
        )
        if raster_feats.shape[1] > 0:
            parts.append(raster_feats)

    feat = np.hstack(parts).astype(np.float32)
    return feat


# ---------------------------------------------------------------------------
# Inference (single chunk)
# ---------------------------------------------------------------------------

def infer_chunk(
    pipeline,
    points: np.ndarray,
    colors: np.ndarray,
    use_bev: bool = False,
    bev_cell_size: float = 1.0,
    footprint_path: str | None = None,
    dem_path: str | None = None,
    dsm_path: str | None = None,
) -> np.ndarray:
    """Run segmentation on one chunk. Returns predicted labels."""

    feat = build_features(
        points, colors,
        use_bev=use_bev,
        bev_cell_size=bev_cell_size,
        footprint_path=footprint_path,
        dem_path=dem_path,
        dsm_path=dsm_path,
    )

    data = {
        "name": "chunk",
        "point": points.astype(np.float32),
        "feat": feat,
        "label": np.zeros(len(points), dtype=np.int32),
    }

    result = pipeline.run_inference(data)
    return result["predict_labels"]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def count_feature_channels(
    use_bev: bool,
    footprint_path: str | None,
    dem_path: str | None,
    dsm_path: str | None,
) -> int:
    """Calculate expected in_channels so user knows which config to use."""
    c = 6  # xyz (3) + rgb (3) — Open3D-ML convention
    if use_bev:
        c += 7
    if footprint_path:
        c += 1  # footprint_mask
    if dem_path:
        c += 2  # ground_elevation + height_above_ground
    if dsm_path:
        c += 1  # surface_elevation
    return c


def process_pointcloud(
    input_path: str,
    output_labels: str,
    output_ply: str | None = None,
    cfg_file: str | None = None,
    device: str = "cpu",
    chunk_size: int = 0,
    use_bev: bool = False,
    bev_cell_size: float = 1.0,
    footprint_path: str | None = None,
    dem_path: str | None = None,
    dsm_path: str | None = None,
):
    # report input mode
    mode_parts = ["3D"]
    if use_bev:
        mode_parts.append("BEV")
    if any(p is not None for p in [footprint_path, dem_path, dsm_path]):
        mode_parts.append("2D")
    expected_ch = count_feature_channels(use_bev, footprint_path, dem_path, dsm_path)
    print(f"Input mode: {' + '.join(mode_parts)}  (expected in_channels={expected_ch})")

    # pipeline & weights (Freeze+Extend for extra channels)
    pipeline = create_pipeline(cfg_file=cfg_file, device=device)
    ckpt_path = os.path.join(WEIGHTS_DIR, "randlanet_toronto3d_202201071330utc.pth")
    download_weights(ckpt_path)
    load_weights_extended(pipeline, ckpt_path, expected_ch, device=device)

    # read & infer
    all_labels = []
    all_points = [] if output_ply else None
    total_points = 0

    chunks = read_pointcloud(input_path, chunk_size=chunk_size)

    for idx, (pts, colors) in enumerate(chunks):
        n = len(pts)
        total_points += n
        print(f"  chunk {idx}: {n:,} points  (cumulative {total_points:,})")

        labels = infer_chunk(
            pipeline, pts, colors,
            use_bev=use_bev,
            bev_cell_size=bev_cell_size,
            footprint_path=footprint_path,
            dem_path=dem_path,
            dsm_path=dsm_path,
        )
        all_labels.append(labels)

        if all_points is not None:
            all_points.append(pts)

    all_labels = np.concatenate(all_labels)
    print(f"Inference done: {len(all_labels):,} points total")

    # save labels
    _ensure_parent_dir(output_labels)
    np.save(output_labels, all_labels.astype(np.int32))
    print(f"Saved labels → {output_labels}")

    # optional colored PLY
    if output_ply:
        all_pts = np.concatenate(all_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)

        max_label = all_labels.max()
        norm = all_labels / max_label if max_label > 0 else all_labels.astype(float)
        pcd.colors = o3d.utility.Vector3dVector(np.stack([norm] * 3, axis=1))

        _ensure_parent_dir(output_ply)
        o3d.io.write_point_cloud(output_ply, pcd)
        print(f"Saved colored PLY → {output_ply}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RandLA-Net inference: 3D-only, 3D+BEV, 3D+2D, or 3D+BEV+2D"
    )

    # required
    parser.add_argument("--input", required=True, help="Input point cloud (.ply/.las/.laz)")
    parser.add_argument("--labels", default="labels.npy", help="Output labels .npy")

    # optional outputs
    parser.add_argument("--output", default=None, help="Optional colored PLY output")

    # model
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")

    # batching
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Points per chunk for large files (0=all). Recommended: 5000000")

    # BEV features (computed from 3D points)
    parser.add_argument("--bev", action="store_true", help="Enable BEV feature injection")
    parser.add_argument("--bev-cell-size", type=float, default=1.0, help="BEV grid cell size (meters)")

    # 2D raster inputs (optional — dual input mode)
    parser.add_argument("--footprints", default=None,
                        help="Building footprints: GeoTIFF (.tif) or vector (.geojson/.shp/.gpkg)")
    parser.add_argument("--dem", default=None,
                        help="DEM/DTM GeoTIFF for ground elevation")
    parser.add_argument("--dsm", default=None,
                        help="DSM GeoTIFF for surface elevation")

    args = parser.parse_args()

    process_pointcloud(
        input_path=args.input,
        output_labels=args.labels,
        output_ply=args.output,
        cfg_file=args.config,
        device=args.device,
        chunk_size=args.chunk_size,
        use_bev=args.bev,
        bev_cell_size=args.bev_cell_size,
        footprint_path=args.footprints,
        dem_path=args.dem,
        dsm_path=args.dsm,
    )
