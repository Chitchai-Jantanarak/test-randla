import os
import argparse
import urllib.request
import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "randlanet_toronto3d_config.yml")


# --------------------------------------------------
# Download pretrained model if not exists
# --------------------------------------------------
def download_weights(ckpt_path):
    url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_toronto3d_202201071330utc.pth"
    if not os.path.exists(ckpt_path):
        print("Downloading pretrained weights...")
        urllib.request.urlretrieve(url, ckpt_path)
        print("Download complete.")


# --------------------------------------------------
# Create Open3D-ML pipeline
# --------------------------------------------------
def create_pipeline(cfg_file=None, device="cpu"):

    if cfg_file is None:
        cfg_file = DEFAULT_CONFIG

    if not os.path.exists(cfg_file):
        raise FileNotFoundError(
            f"Config file not found: '{cfg_file}'\n"
            "Pass the correct path with --config <path/to/config.yml>"
        )

    cfg = _ml3d.utils.Config.load_from_file(cfg_file)

    model = ml3d.models.RandLANet(**cfg.model)

    # Dummy dataset (required by pipeline).
    # Pop 'dataset_path' from cfg.dataset to avoid passing it both as a
    # positional argument and as a keyword argument, which raises TypeError.
    dataset_cfg = dict(cfg.dataset)
    dataset_cfg.pop("dataset_path", None)
    dataset = ml3d.datasets.SemanticKITTI("", **dataset_cfg)

    pipeline = ml3d.pipelines.SemanticSegmentation(
        model, dataset=dataset, device=device, **cfg.pipeline
    )

    return pipeline


# --------------------------------------------------
# Process PLY
# --------------------------------------------------
def process_ply(ply_path, output_ply, output_labels, cfg_file=None, device="cpu"):

    print("Loading PLY...")
    pcd = o3d.io.read_point_cloud(ply_path)

    points = np.asarray(pcd.points)

    if len(points) == 0:
        raise ValueError("Empty point cloud.")

    # Handle colors
    if len(pcd.colors) > 0:
        colors = np.asarray(pcd.colors) * 255.0
    else:
        print("PLY has no RGB. Using dummy features.")
        colors = np.zeros_like(points)

    num_points = points.shape[0]
    print(f"Points loaded: {num_points:,}")

    data = {
        "name": "ply_cloud",
        "point": points.astype(np.float32),
        "feat": colors.astype(np.float32),
        "label": np.zeros((num_points,), dtype=np.int32),
    }

    # Create pipeline
    pipeline = create_pipeline(cfg_file=cfg_file, device=device)

    # Load weights
    ckpt_folder = "./weights"
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_folder,
        "randlanet_toronto3d_202201071330utc.pth"
    )

    download_weights(ckpt_path)
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    print("Running inference...")
    result = pipeline.run_inference(data)

    predicted_labels = result["predict_labels"]

    # Save labels separately
    np.save(output_labels, predicted_labels)
    print(f"Saved labels to {output_labels}")

    # Colorize output PLY
    max_label = predicted_labels.max()
    norm_labels = predicted_labels / max_label if max_label > 0 else predicted_labels

    colored = np.stack([norm_labels]*3, axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colored)

    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Saved colored PLY to {output_ply}")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--output", default="output.ply", help="Output PLY")
    parser.add_argument("--labels", default="labels.npy", help="Output labels file")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the YAML config file (default: randlanet_toronto3d_config.yml)",
    )

    args = parser.parse_args()

    process_ply(
        args.input,
        args.output,
        args.labels,
        cfg_file=args.config,
        device=args.device
    )