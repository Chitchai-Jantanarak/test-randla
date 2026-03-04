"""
apply_labels_to_las.py
======================
Takes an input point cloud (LAS or PLY) and the predicted label file (.npy)
from the RandLANet segmentation script, then writes a LAS 1.2 file with proper
classification codes that Potree-viewer (and other LAS viewers) can display.

Usage examples
--------------
  # Basic — PLY input, .npy labels → classified LAS
  python apply_labels_to_las.py --input cloud.ply --labels labels.npy --output classified.las

  # LAS input
  python apply_labels_to_las.py --input cloud.las --labels labels.npy --output classified.las

  # Custom label mapping via JSON
  python apply_labels_to_las.py --input cloud.ply --labels labels.npy --output classified.las \
      --mapping my_mapping.json

Dependencies
------------
  pip install laspy numpy open3d
  (open3d only needed when the input is PLY)
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import laspy
except ImportError:
    sys.exit("laspy is required.  Install with:  pip install laspy")


# Toronto3D classes produced by RandLANet:
#   0  Unclassified
#   1  Road
#   2  Road Marking
#   3  Natural (vegetation)
#   4  Building
#   5  Utility Line
#   6  Pole
#   7  Car
#   8  Fence
#
# ASPRS standard LAS classification codes:
#   0  Created / never classified
#   1  Unclassified
#   2  Ground
#   3  Low vegetation
#   4  Medium vegetation
#   5  High vegetation
#   6  Building
#   7  Low point (noise)
#   9  Water
#  11  Road surface
#  13  Wire – guard
#  14  Wire – conductor
#  15  Transmission tower
#  17  Bridge deck
#  64+ User-definable
#
# Mapping chosen for best Potree/CloudCompare visual compatibility:
DEFAULT_LABEL_MAP = {
    0: 1,    # Unclassified          → 1  (Unclassified)
    1: 11,   # Road                  → 11 (Road Surface)
    2: 11,   # Road Marking          → 11 (Road Surface)  [or use 64 for custom]
    3: 3,    # Natural / vegetation  → 3  (Low Vegetation)
    4: 6,    # Building              → 6  (Building)
    5: 14,   # Utility Line          → 14 (Wire – Conductor)
    6: 15,   # Pole                  → 15 (Transmission Tower)
    7: 64,   # Car                   → 64 (User-defined: vehicle)
    8: 65,   # Fence                 → 65 (User-defined: fence)
}


def load_label_mapping(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def read_point_cloud(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".las", ".laz"):
        las = laspy.read(path)
        points = np.stack([las.x, las.y, las.z], axis=-1)
        colors = np.empty((0,))
        try:
            colors = np.stack([las.red, las.green, las.blue], axis=-1)
        except Exception:
            pass
        return points, colors, las

    elif ext == ".ply":
        try:
            import open3d as o3d
        except ImportError:
            sys.exit("open3d is required to read PLY files.  pip install open3d")

        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        if len(pcd.colors) > 0:
            colors = (np.asarray(pcd.colors) * 65535).astype(np.uint16)
        else:
            colors = np.empty((0,))
        return points, colors, None

    else:
        sys.exit(f"Unsupported format: {ext}  (use .las, .laz, or .ply)")


def write_classified_las(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    classification: np.ndarray,
    source_las=None,
    use_las14: bool = False,
):
    max_class = int(classification.max())

    if max_class > 31 and not use_las14:
        print(f"WARNING: max classification value {max_class} > 31; "
              f"auto-upgrading to LAS 1.4 point format 6.")
        use_las14 = True

    if source_las is not None:
        src_version = f"{source_las.header.version.major}.{source_las.header.version.minor}"
        if use_las14 and src_version < "1.4":
            print(f"Source LAS is {src_version}; rebuilding as LAS 1.4 ...")
        else:
            las = source_las
            las.classification = classification.astype(np.uint8)
            las.write(output_path)
            return

    has_color = colors.size > 0

    if use_las14:
        point_format_id = 7 if has_color else 6
        version = "1.4"
    else:
        point_format_id = 2 if has_color else 0
        version = "1.2"

    header = laspy.LasHeader(point_format=point_format_id, version=version)

    mins = points.min(axis=0)
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])  # mm precision

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if has_color:
        las.red   = colors[:, 0].astype(np.uint16)
        las.green = colors[:, 1].astype(np.uint16)
        las.blue  = colors[:, 2].astype(np.uint16)

    las.classification = classification.astype(np.uint8)

    las.write(output_path)



def apply_labels(
    input_path: str,
    labels_path: str,
    output_path: str,
    mapping: dict | None = None,
    use_las14: bool = False,
):
    label_map = mapping if mapping is not None else DEFAULT_LABEL_MAP
    ext = os.path.splitext(labels_path)[1].lower()
    if ext == ".npy":
        pred_labels = np.load(labels_path)
    elif ext in (".txt", ".csv"):
        pred_labels = np.loadtxt(labels_path, dtype=np.int32)
    else:
        sys.exit(f"Unsupported label format: {ext}  (use .npy or .txt)")

    pred_labels = pred_labels.astype(np.int32).ravel()
    print(f"Loaded {len(pred_labels):,} labels  "
          f"(unique classes: {np.unique(pred_labels).tolist()})")

    points, colors, source_las = read_point_cloud(input_path)
    print(f"Loaded {len(points):,} points from {input_path}")

    if len(points) != len(pred_labels):
        sys.exit(
            f"Mismatch: point cloud has {len(points):,} points but "
            f"labels file has {len(pred_labels):,} entries."
        )

    classification = np.zeros(len(pred_labels), dtype=np.uint8)
    unmapped = set()
    for model_label in np.unique(pred_labels):
        mask = pred_labels == model_label
        if model_label in label_map:
            classification[mask] = label_map[model_label]
        else:
            unmapped.add(int(model_label))
            classification[mask] = 1  

    if unmapped:
        print(f"WARNING: labels {unmapped} had no mapping → set to 1 (Unclassified)")

    unique_cls, counts = np.unique(classification, return_counts=True)
    print("\nClassification summary:")
    for c, n in zip(unique_cls, counts):
        print(f"  LAS class {c:3d}  →  {n:>10,} points")

    write_classified_las(output_path, points, colors, classification, source_las)
    print(f"\nSaved classified LAS to: {output_path}")
    print("Open this file in Potree-viewer / CloudCompare to see classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    custom_mapping = None
    if hasattr(args, "mapping") and args.mapping:
        custom_mapping = load_label_mapping(args.mapping)

    apply_labels(
        input_path=args.input,
        labels_path=args.labels,
        output_path=args.output,
        mapping=custom_mapping,
    )