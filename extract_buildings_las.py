"""
Convert building bounding boxes from JSON into a LAS file.

Input: only the buildings JSON (from apply.py).
Output: a LAS file with 8 corner points per building bbox + centroid point.

No source LAS needed — the geometry comes entirely from the JSON coordinates.

Usage:
    python extract_buildings_las.py --json buildings.json --output buildings.las
    python extract_buildings_las.py --json buildings.json --output buildings.las --class-id 6
"""

import sys
import os
import json
import argparse
import numpy as np

try:
    import laspy
except ImportError:
    sys.exit("laspy is required.  Install with:  pip install laspy")


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def bbox_corners(bbox_min, bbox_max):
    """Generate 8 corner points of a 3D bounding box."""
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x0, y1, z0],
        [x1, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],
    ])


def buildings_to_las(buildings, output_path, class_id=6):
    """Write all building bboxes as points in a single LAS file.

    Per building: 8 corner points + 1 centroid = 9 points.
    Each point gets:
      - classification = class_id (default 6 = ASPRS building)
      - intensity = building ID (so you can color by building in viewers)
    """
    all_points = []
    all_intensity = []
    all_classification = []

    for b in buildings:
        bid = b["id"]
        bmin = b["bbox_min"]
        bmax = b["bbox_max"]
        centroid = b["centroid"]

        corners = bbox_corners(bmin, bmax)
        center = np.array([centroid]).reshape(1, 3)
        pts = np.vstack([corners, center])  # 9 points

        all_points.append(pts)
        all_intensity.append(np.full(len(pts), bid, dtype=np.uint16))
        all_classification.append(np.full(len(pts), class_id, dtype=np.uint8))

    all_points = np.concatenate(all_points)
    all_intensity = np.concatenate(all_intensity)
    all_classification = np.concatenate(all_classification)

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = all_points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = all_points[:, 0]
    las.y = all_points[:, 1]
    las.z = all_points[:, 2]
    las.intensity = all_intensity
    las.classification = all_classification

    _ensure_parent_dir(output_path)
    las.write(output_path)
    return len(all_points)


def main():
    parser = argparse.ArgumentParser(
        description="Convert building bounding boxes from JSON to LAS file"
    )
    parser.add_argument("--json", required=True,
                        help="Buildings JSON file (from apply.py)")
    parser.add_argument("--output", required=True,
                        help="Output LAS file path")
    parser.add_argument("--class-id", type=int, default=6,
                        help="ASPRS classification code for building points (default: 6)")

    args = parser.parse_args()

    with open(args.json) as f:
        report = json.load(f)

    buildings = report.get("buildings", [])
    if not buildings:
        print("No buildings in JSON — nothing to write.")
        sys.exit(0)

    print(f"Buildings in JSON: {len(buildings)}")

    for b in buildings:
        w = b["bbox_max"][0] - b["bbox_min"][0]
        h = b["bbox_max"][1] - b["bbox_min"][1]
        z = b["bbox_max"][2] - b["bbox_min"][2]
        print(f"  Building {b['id']:3d}: {b['point_count']:>8,} pts  "
              f"{w:.1f}x{h:.1f}x{z:.1f}m  "
              f"center ({b['centroid'][0]:.1f}, {b['centroid'][1]:.1f}, {b['centroid'][2]:.1f})")

    count = buildings_to_las(buildings, args.output, class_id=args.class_id)
    print(f"\nWrote {count} points ({len(buildings)} buildings x 9 pts) → {args.output}")
    print(f"  9 pts per building = 8 bbox corners + 1 centroid")
    print(f"  intensity field = building ID (for coloring in viewers)")


if __name__ == "__main__":
    main()
