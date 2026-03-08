"""
RandLA-Net semantic segmentation inference with:
  - Batch/chunked processing for extremely large LAS/PLY files (150-300 GB)
  - Optional BEV (Bird's Eye View) feature injection
  - LAS native input support alongside PLY
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

import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from bev_features import compute_bev_features, compute_bev_features_chunked

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "randlanet_toronto3d_config.yml")
WEIGHTS_URL = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_toronto3d_202201071330utc.pth"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


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
# Point cloud I/O (supports PLY and LAS/LAZ)
# ---------------------------------------------------------------------------

def read_pointcloud_ply(path: str):
    """Read a PLY file, return (points, colors) as numpy arrays."""
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0 if len(pcd.colors) > 0 else np.zeros_like(points)
    return points.astype(np.float32), colors.astype(np.float32)


def iter_las_chunks(path: str, chunk_size: int = 5_000_000):
    """Yield (points, colors) chunks from a LAS/LAZ file without loading it all.

    Each chunk is at most ``chunk_size`` points.  This keeps RAM bounded even
    for 150-300 GB files.
    """
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
    """Unified reader.  If chunk_size > 0 and the file is LAS/LAZ, yields chunks.
    Otherwise returns full arrays."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        pts, colors = read_pointcloud_ply(path)
        return [(pts, colors)]

    elif ext in (".las", ".laz"):
        if chunk_size > 0:
            return iter_las_chunks(path, chunk_size)
        # load entire file
        if laspy is None:
            sys.exit("laspy is required.  pip install laspy")
        las = laspy.read(path)
        pts = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32)
        try:
            colors = np.stack([
                las.red / 256.0, las.green / 256.0, las.blue / 256.0
            ], axis=-1).astype(np.float32)
        except Exception:
            colors = np.zeros_like(pts)
        return [(pts, colors)]

    else:
        sys.exit(f"Unsupported format: {ext}  (use .ply, .las, or .laz)")


# ---------------------------------------------------------------------------
# Inference (single chunk)
# ---------------------------------------------------------------------------

def infer_chunk(
    pipeline,
    points: np.ndarray,
    colors: np.ndarray,
    use_bev: bool = False,
    bev_cell_size: float = 1.0,
) -> np.ndarray:
    """Run segmentation on one chunk of points.  Returns predicted labels."""

    feat = colors.copy()

    if use_bev:
        bev = compute_bev_features(points, cell_size=bev_cell_size)
        feat = np.hstack([feat, bev])

    data = {
        "name": "chunk",
        "point": points.astype(np.float32),
        "feat": feat.astype(np.float32),
        "label": np.zeros(len(points), dtype=np.int32),
    }

    result = pipeline.run_inference(data)
    return result["predict_labels"]


# ---------------------------------------------------------------------------
# Main entry: batch-aware inference
# ---------------------------------------------------------------------------

def process_pointcloud(
    input_path: str,
    output_labels: str,
    output_ply: str | None = None,
    cfg_file: str | None = None,
    device: str = "cpu",
    chunk_size: int = 0,
    use_bev: bool = False,
    bev_cell_size: float = 1.0,
):
    """Run segmentation, writing labels to .npy (and optionally a colored PLY).

    For very large files set ``chunk_size`` > 0 to process in batches.
    Labels from all chunks are concatenated in file order.
    """

    # ---- pipeline & weights ----
    pipeline = create_pipeline(cfg_file=cfg_file, device=device)
    ckpt_path = os.path.join(WEIGHTS_DIR, "randlanet_toronto3d_202201071330utc.pth")
    download_weights(ckpt_path)
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    # ---- read & infer ----
    all_labels = []
    all_points = [] if output_ply else None
    total_points = 0

    chunks = read_pointcloud(input_path, chunk_size=chunk_size)

    for idx, (pts, colors) in enumerate(chunks):
        n = len(pts)
        total_points += n
        print(f"  chunk {idx}: {n:,} points  (cumulative {total_points:,})")

        labels = infer_chunk(pipeline, pts, colors, use_bev=use_bev, bev_cell_size=bev_cell_size)
        all_labels.append(labels)

        if all_points is not None:
            all_points.append(pts)

    all_labels = np.concatenate(all_labels)
    print(f"Inference done: {len(all_labels):,} points total")

    # ---- save labels ----
    np.save(output_labels, all_labels)
    print(f"Saved labels → {output_labels}")

    # ---- optional colored PLY ----
    if output_ply:
        all_pts = np.concatenate(all_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)

        max_label = all_labels.max()
        norm = all_labels / max_label if max_label > 0 else all_labels.astype(float)
        pcd.colors = o3d.utility.Vector3dVector(np.stack([norm] * 3, axis=1))

        o3d.io.write_point_cloud(output_ply, pcd)
        print(f"Saved colored PLY → {output_ply}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RandLA-Net inference with batch support for large LAS/PLY files"
    )
    parser.add_argument("--input", required=True, help="Input point cloud (.ply, .las, .laz)")
    parser.add_argument("--labels", default="labels.npy", help="Output labels .npy path")
    parser.add_argument("--output", default=None, help="Optional output colored PLY")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--chunk-size", type=int, default=0,
        help="Points per chunk for large LAS files (0 = load all at once). "
             "Recommended: 5000000 for large files."
    )
    parser.add_argument("--bev", action="store_true", help="Enable BEV feature injection")
    parser.add_argument("--bev-cell-size", type=float, default=1.0, help="BEV grid cell size (meters)")

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
    )
