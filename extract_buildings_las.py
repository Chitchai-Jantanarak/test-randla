"""
Extract a single LAS file covering all building bounding boxes from a JSON report.

Reads the buildings JSON (from apply.py), computes one overall 3D bounding box
that envelopes all detected buildings, then crops the source LAS/LAZ to that
region and writes a single output LAS file.

Usage:
    python extract_buildings_las.py --json buildings.json --input source.las --output buildings_area.las
    python extract_buildings_las.py --json buildings.json --input source.las --output buildings_area.las --padding 2.0
    python extract_buildings_las.py --json buildings.json --input source.las --output buildings_area.las --chunk-size 5000000
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


def compute_overall_bbox(buildings, padding=0.0):
    """Compute one 3D bounding box that covers all buildings.

    Args:
        buildings: list of dicts with 'bbox_min' and 'bbox_max' keys.
        padding: extra meters around the combined envelope.

    Returns:
        (bbox_min, bbox_max) as numpy arrays of shape (3,).
    """
    all_mins = np.array([b["bbox_min"] for b in buildings])
    all_maxs = np.array([b["bbox_max"] for b in buildings])

    bbox_min = all_mins.min(axis=0) - padding
    bbox_max = all_maxs.max(axis=0) + padding

    return bbox_min, bbox_max


def extract_las(input_path, bbox_min, bbox_max, output_path):
    """Crop source LAS to 3D bbox and write output LAS."""
    las = laspy.read(input_path)
    points = np.stack([las.x, las.y, las.z], axis=-1)

    mask = (
        (points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
        (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
        (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2])
    )

    count = mask.sum()
    if count == 0:
        print("No points found within the bounding box.")
        return 0

    out_las = laspy.LasData(las.header)
    out_las.points = las.points[mask]
    out_las.write(output_path)

    return count


def extract_las_chunked(input_path, bbox_min, bbox_max, output_path, chunk_size):
    """Crop source LAS to 3D bbox using chunked reads for large files."""
    with laspy.open(input_path) as reader:
        total = reader.header.point_count
        print(f"Source: {total:,} points, reading in chunks of {chunk_size:,}")

        header = reader.header
        collected_points = []

        for idx, chunk in enumerate(reader.chunk_iterator(chunk_size)):
            pts = np.stack([chunk.x, chunk.y, chunk.z], axis=-1)

            mask = (
                (pts[:, 0] >= bbox_min[0]) & (pts[:, 0] <= bbox_max[0]) &
                (pts[:, 1] >= bbox_min[1]) & (pts[:, 1] <= bbox_max[1]) &
                (pts[:, 2] >= bbox_min[2]) & (pts[:, 2] <= bbox_max[2])
            )

            if mask.any():
                collected_points.append(chunk.array[mask])

            print(f"  chunk {idx}: {len(pts):,} pts → {mask.sum():,} inside bbox")

    if not collected_points:
        print("No points found within the bounding box.")
        return 0

    all_arrays = np.concatenate(collected_points)
    out_las = laspy.LasData(header)
    out_las.points = laspy.ScaleAwarePointRecord(all_arrays, header)
    out_las.write(output_path)

    return len(all_arrays)


def main():
    parser = argparse.ArgumentParser(
        description="Extract one LAS file covering all building bounding boxes from JSON"
    )
    parser.add_argument("--json", required=True,
                        help="Buildings JSON file (from apply.py)")
    parser.add_argument("--input", required=True,
                        help="Source point cloud (.las/.laz)")
    parser.add_argument("--output", required=True,
                        help="Output LAS file path")
    parser.add_argument("--padding", type=float, default=1.0,
                        help="Padding (meters) around overall bbox (default: 1.0)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Chunk size for large files (0 = load all)")

    args = parser.parse_args()

    # load JSON
    with open(args.json) as f:
        report = json.load(f)

    buildings = report.get("buildings", [])
    if not buildings:
        print("No buildings in JSON — nothing to extract.")
        sys.exit(0)

    # compute overall bbox
    bbox_min, bbox_max = compute_overall_bbox(buildings, padding=args.padding)

    extent = bbox_max - bbox_min
    print(f"Buildings: {len(buildings)}")
    print(f"Overall bbox min: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]")
    print(f"Overall bbox max: [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
    print(f"Extent: {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m  "
          f"(padding={args.padding}m)")

    # extract
    if args.chunk_size > 0:
        count = extract_las_chunked(
            args.input, bbox_min, bbox_max, args.output, args.chunk_size
        )
    else:
        count = extract_las(args.input, bbox_min, bbox_max, args.output)

    if count > 0:
        print(f"\nExtracted {count:,} points → {args.output}")


if __name__ == "__main__":
    main()
