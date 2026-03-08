"""
Auto-label generation: project 3D point clouds onto 2D building footprints
to generate per-point training labels WITHOUT manual annotation.

Uses the shared raster_features module for 2D→3D projection.

Sources for building footprints:
  - OpenStreetMap (good for UK / Europe / urban Thailand)
  - Microsoft Global ML Building Footprints (worldwide)
  - Google Open Buildings (Asia/Africa coverage)

Usage:
  # With a building footprint raster + DEM
  python generate_labels.py \\
      --input scan.las \\
      --footprints buildings.tif \\
      --dem ground.tif \\
      --output labels_auto.npy

  # With GeoJSON vector footprints (auto-rasterized)
  python generate_labels.py \\
      --input scan.las \\
      --footprints buildings.geojson \\
      --output labels_auto.npy
"""

import os
import sys
import argparse
import numpy as np

try:
    import laspy
except ImportError:
    laspy = None

from raster_features import lookup_footprints, load_raster, raster_lookup

# Toronto3D class indices (matching the model)
CLASS_GROUND = 0
CLASS_BUILDING = 3
CLASS_VEGETATION = 2
CLASS_UNLABELED = -1


# ---------------------------------------------------------------------------
# Point cloud loading
# ---------------------------------------------------------------------------

def load_points(path: str) -> np.ndarray:
    """Load xyz from LAS/LAZ/PLY. Returns (N, 3) float64."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".las", ".laz"):
        if laspy is None:
            sys.exit("laspy required: pip install laspy")
        las = laspy.read(path)
        return np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)

    elif ext == ".ply":
        try:
            import open3d as o3d
        except ImportError:
            sys.exit("open3d required: pip install open3d")
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points).astype(np.float64)

    else:
        sys.exit(f"Unsupported: {ext}")


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_auto_labels(
    points: np.ndarray,
    footprint_path: str,
    dem_path: str | None = None,
    min_building_height: float = 2.0,
    default_class: int = CLASS_UNLABELED,
) -> np.ndarray:
    """Generate per-point labels by projecting onto 2D building footprints.

    Args:
        points: (N, 3) xyz
        footprint_path: GeoTIFF raster (.tif) or vector (.geojson/.shp/.gpkg)
        dem_path: optional DEM GeoTIFF for ground elevation
        min_building_height: min height above ground to count as building

    Returns:
        labels: (N,) int32 array with Toronto3D class indices
    """
    n = len(points)
    labels = np.full(n, default_class, dtype=np.int32)
    xy = points[:, :2]
    z = points[:, 2].astype(np.float64)

    # --- footprint lookup (uses shared module) ---
    inside_building = lookup_footprints(xy, footprint_path)
    print(f"Points inside building footprints: {inside_building.sum():,} / {n:,}")

    # --- ground elevation ---
    if dem_path is not None:
        dem_raster, dem_transform, _ = load_raster(dem_path)
        ground_z = raster_lookup(xy, dem_raster, dem_transform)
        no_coverage = ground_z == 0
        if no_coverage.any():
            ground_z[no_coverage] = np.percentile(z, 5)
    else:
        ground_z = np.full(n, np.percentile(z, 5))

    height_above_ground = z - ground_z

    # --- assign labels ---
    # building: inside footprint AND above ground
    is_building = inside_building & (height_above_ground > min_building_height)
    labels[is_building] = CLASS_BUILDING

    # ground: inside footprint but at ground level (footprint shadow)
    is_ground_in_fp = inside_building & (height_above_ground <= min_building_height)
    labels[is_ground_in_fp] = CLASS_GROUND

    # vegetation: outside footprint AND high
    is_veg = ~inside_building & (height_above_ground > min_building_height)
    labels[is_veg] = CLASS_VEGETATION

    # ground: outside footprint AND low
    is_ground = ~inside_building & (height_above_ground <= min_building_height)
    labels[is_ground] = CLASS_GROUND

    # summary
    for cls_name, cls_id in [("Ground", 0), ("Vegetation", 2), ("Building", 3), ("Unlabeled", -1)]:
        count = (labels == cls_id).sum()
        print(f"  {cls_name:12s} (class {cls_id:2d}): {count:>12,} points")

    return labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training labels from 2D building footprints + 3D point cloud"
    )
    parser.add_argument("--input", required=True, help="Input point cloud (.las/.laz/.ply)")
    parser.add_argument("--footprints", required=True,
                        help="Building footprints: GeoTIFF (.tif) or vector (.geojson/.shp/.gpkg)")
    parser.add_argument("--dem", default=None, help="Optional DEM/DTM GeoTIFF for ground elevation")
    parser.add_argument("--output", required=True, help="Output labels .npy path")
    parser.add_argument("--min-building-height", type=float, default=2.0,
                        help="Min height above ground to classify as building (meters)")

    args = parser.parse_args()

    points = load_points(args.input)
    print(f"Loaded {len(points):,} points")

    labels = generate_auto_labels(
        points,
        footprint_path=args.footprints,
        dem_path=args.dem,
        min_building_height=args.min_building_height,
    )

    np.save(args.output, labels)
    print(f"\nSaved auto-labels → {args.output}")
    print(f"Ready for training: use as ground truth with fine-tuning pipeline")
