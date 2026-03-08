"""
Auto-label generation: project 3D point clouds onto 2D building footprints
to generate per-point training labels WITHOUT manual annotation.

Sources for building footprints:
  - OpenStreetMap (good for UK / Europe / urban Thailand)
  - Microsoft Global ML Building Footprints (worldwide)
  - Google Open Buildings (Asia/Africa coverage)

Workflow:
  1. Load building footprint raster (GeoTIFF binary mask) OR vector (GeoJSON)
  2. Load DEM/DTM for ground elevation (optional, falls back to percentile)
  3. For each 3D point: project (x,y) → raster → check if inside building
  4. Refine with height: above ground + threshold → building, else ground
  5. Output: per-point labels ready for training

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
# Raster-based footprint lookup
# ---------------------------------------------------------------------------

def load_raster_footprints(tif_path: str):
    """Load a GeoTIFF building mask. Returns (data_2d, transform).

    Expected: single-band raster where pixel > 0 = building.
    """
    try:
        import rasterio
    except ImportError:
        sys.exit("rasterio required for GeoTIFF: pip install rasterio")

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
    print(f"Loaded footprint raster: {data.shape}, CRS={crs}")
    return data, transform


def load_dem(tif_path: str):
    """Load a DEM/DTM GeoTIFF. Returns (data_2d, transform)."""
    try:
        import rasterio
    except ImportError:
        sys.exit("rasterio required: pip install rasterio")

    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float64)
        transform = src.transform
    print(f"Loaded DEM: {data.shape}, elevation range [{data.min():.1f}, {data.max():.1f}]")
    return data, transform


def raster_lookup(points_xy: np.ndarray, raster: np.ndarray, transform) -> np.ndarray:
    """For each (x,y), look up the raster value. Returns (N,) array.

    Points outside raster bounds get value 0.
    """
    import rasterio.transform

    rows, cols = rasterio.transform.rowcol(transform, points_xy[:, 0], points_xy[:, 1])
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    h, w = raster.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    result = np.zeros(len(points_xy), dtype=raster.dtype)
    result[valid] = raster[rows[valid], cols[valid]]
    return result


# ---------------------------------------------------------------------------
# Vector-based footprint lookup (GeoJSON/Shapefile)
# ---------------------------------------------------------------------------

def vector_footprint_lookup(points_xy: np.ndarray, vector_path: str) -> np.ndarray:
    """Check which points fall inside building polygons from a vector file.

    Returns (N,) bool array.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.strtree import STRtree
    except ImportError:
        sys.exit("geopandas + shapely required: pip install geopandas shapely")

    gdf = gpd.read_file(vector_path)
    print(f"Loaded {len(gdf)} building polygons from {vector_path}")

    # build spatial index
    tree = STRtree(gdf.geometry.values)
    inside = np.zeros(len(points_xy), dtype=bool)

    # batch query: for each point, check if inside any polygon
    # process in chunks to manage memory
    chunk_size = 100_000
    for start in range(0, len(points_xy), chunk_size):
        end = min(start + chunk_size, len(points_xy))
        for i in range(start, end):
            pt = Point(points_xy[i, 0], points_xy[i, 1])
            candidates = tree.query(pt)
            for geom in candidates:
                if geom.contains(pt):
                    inside[i] = True
                    break

        if (start // chunk_size) % 10 == 0:
            print(f"  Vector lookup: {end:,}/{len(points_xy):,} points checked")

    return inside


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
        default_class: label for non-building points

    Returns:
        labels: (N,) int32 array with Toronto3D class indices
    """
    n = len(points)
    labels = np.full(n, default_class, dtype=np.int32)
    xy = points[:, :2]
    z = points[:, 2]

    # --- determine which points are inside building footprints ---
    ext = os.path.splitext(footprint_path)[1].lower()

    if ext in (".tif", ".tiff"):
        raster, transform = load_raster_footprints(footprint_path)
        footprint_values = raster_lookup(xy, raster, transform)
        inside_building = footprint_values > 0
    elif ext in (".geojson", ".json", ".shp", ".gpkg"):
        inside_building = vector_footprint_lookup(xy, footprint_path)
    else:
        sys.exit(f"Unsupported footprint format: {ext}")

    print(f"Points inside building footprints: {inside_building.sum():,} / {n:,}")

    # --- get ground elevation ---
    if dem_path is not None:
        dem_raster, dem_transform = load_dem(dem_path)
        ground_z = raster_lookup(xy, dem_raster, dem_transform).astype(np.float64)
        # fallback for points outside DEM coverage
        no_dem = ground_z == 0
        if no_dem.any():
            ground_z[no_dem] = np.percentile(z, 5)
    else:
        # simple fallback: ground = low percentile of z
        ground_z = np.full(n, np.percentile(z, 5))

    height_above_ground = z - ground_z

    # --- assign labels ---
    # building: inside footprint AND sufficiently above ground
    is_building = inside_building & (height_above_ground > min_building_height)
    labels[is_building] = CLASS_BUILDING

    # ground: inside footprint but AT ground level (footprint shadow)
    is_ground_in_fp = inside_building & (height_above_ground <= min_building_height)
    labels[is_ground_in_fp] = CLASS_GROUND

    # vegetation: outside footprint AND high above ground
    is_veg = ~inside_building & (height_above_ground > min_building_height)
    labels[is_veg] = CLASS_VEGETATION

    # ground: outside footprint AND at ground level
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
