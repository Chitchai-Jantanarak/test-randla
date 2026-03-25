"""
Raster-to-point feature projection.

Projects 2D geospatial rasters (GeoTIFF, DEM, building footprints, DSM, etc.)
onto 3D point clouds, producing per-point feature channels.

Used in two contexts:
  1. Inference (main.py)  — 2D priors as extra input features alongside 3D
  2. Label gen (generate_labels.py) — 2D footprints → per-point class labels

Feature channels when all sources available (4 channels):
  - footprint_mask:      1.0 if point xy falls inside building footprint, else 0.0
  - ground_elevation:    DEM ground z at point's (x,y), normalized
  - height_above_ground: point_z - ground_z (from DEM)
  - surface_elevation:   DSM/nDSM value at point's (x,y), normalized (if provided)

Gracefully degrades: only produces channels for data you actually provide.
"""

import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def _require_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        sys.exit("rasterio required for GeoTIFF: pip install rasterio")


def load_raster(tif_path: str, band: int = 1):
    """Load a single-band GeoTIFF. Returns (data_2d, transform, crs)."""
    rasterio = _require_rasterio()
    with rasterio.open(tif_path) as src:
        data = src.read(band)
        transform = src.transform
        crs = src.crs
    return data, transform, crs


def raster_lookup(points_xy: np.ndarray, raster: np.ndarray, transform) -> np.ndarray:
    """For each point (x,y), sample the raster value. Out-of-bounds → 0."""
    rasterio = _require_rasterio()
    import rasterio.transform

    rows, cols = rasterio.transform.rowcol(transform, points_xy[:, 0], points_xy[:, 1])
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    h, w = raster.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    result = np.zeros(len(points_xy), dtype=np.float64)
    result[valid] = raster[rows[valid], cols[valid]].astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# Vector footprint lookup (GeoJSON / SHP / GPKG)
# ---------------------------------------------------------------------------

def vector_footprint_lookup(points_xy: np.ndarray, vector_path: str) -> np.ndarray:
    """Check which points fall inside building polygons. Returns (N,) bool."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.strtree import STRtree
    except ImportError:
        sys.exit("geopandas + shapely required: pip install geopandas shapely")

    gdf = gpd.read_file(vector_path)
    print(f"Loaded {len(gdf)} polygons from {vector_path}")

    tree = STRtree(gdf.geometry.values)
    inside = np.zeros(len(points_xy), dtype=bool)

    batch = 100_000
    for start in range(0, len(points_xy), batch):
        end = min(start + batch, len(points_xy))
        for i in range(start, end):
            pt = Point(points_xy[i, 0], points_xy[i, 1])
            for geom in tree.query(pt):
                if geom.contains(pt):
                    inside[i] = True
                    break
        if (start // batch) % 10 == 0:
            print(f"  Vector lookup: {end:,}/{len(points_xy):,}")

    return inside


def lookup_footprints(points_xy: np.ndarray, footprint_path: str) -> np.ndarray:
    """Unified footprint lookup. Returns (N,) bool — True if inside building."""
    ext = os.path.splitext(footprint_path)[1].lower()

    if ext in (".tif", ".tiff"):
        raster, transform, _ = load_raster(footprint_path)
        values = raster_lookup(points_xy, raster, transform)
        return values > 0
    elif ext in (".geojson", ".json", ".shp", ".gpkg"):
        return vector_footprint_lookup(points_xy, footprint_path)
    else:
        sys.exit(f"Unsupported footprint format: {ext}")


# ---------------------------------------------------------------------------
# Per-point 2D feature computation
# ---------------------------------------------------------------------------

def compute_raster_features(
    points: np.ndarray,
    footprint_path: str | None = None,
    dem_path: str | None = None,
    dsm_path: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Project 2D raster data onto 3D points as per-point feature channels.

    Only produces channels for data actually provided. Returns:
      features: (N, C) float32, where C = number of available channels
      channel_names: list of channel name strings

    Channel layout (in order, only if source provided):
      footprint_mask       (from footprint_path)   — binary 0/1
      ground_elevation     (from dem_path)          — normalized
      height_above_ground  (from dem_path + z)      — raw meters
      surface_elevation    (from dsm_path)          — normalized
    """
    n = len(points)
    xy = points[:, :2]
    z = points[:, 2].astype(np.float64)

    channels = []
    names = []

    # --- footprint mask ---
    if footprint_path is not None:
        inside = lookup_footprints(xy, footprint_path)
        channels.append(inside.astype(np.float32))
        names.append("footprint_mask")
        print(f"  2D footprint: {inside.sum():,}/{n:,} points inside building footprints")

    # --- DEM ground elevation + height above ground ---
    if dem_path is not None:
        dem_raster, dem_transform, _ = load_raster(dem_path)
        ground_z = raster_lookup(xy, dem_raster, dem_transform)

        # fallback for points outside DEM coverage
        no_coverage = ground_z == 0
        if no_coverage.any():
            fallback_z = np.percentile(z, 5)
            ground_z[no_coverage] = fallback_z
            print(f"  DEM: {no_coverage.sum():,} points outside coverage, fallback ground_z={fallback_z:.1f}")

        # normalize ground_z to [0,1] range for the feature
        gz_min, gz_max = ground_z.min(), ground_z.max()
        gz_range = gz_max - gz_min if gz_max > gz_min else 1.0
        ground_z_norm = ((ground_z - gz_min) / gz_range).astype(np.float32)
        channels.append(ground_z_norm)
        names.append("ground_elevation")

        # height above ground (raw meters — not normalized, model learns scale)
        hag = (z - ground_z).astype(np.float32)
        channels.append(hag)
        names.append("height_above_ground")

        print(f"  DEM: ground range [{gz_min:.1f}, {gz_max:.1f}]m, "
              f"height above ground range [{hag.min():.1f}, {hag.max():.1f}]m")

    # --- DSM surface elevation ---
    if dsm_path is not None:
        dsm_raster, dsm_transform, _ = load_raster(dsm_path)
        surface_z = raster_lookup(xy, dsm_raster, dsm_transform)

        sz_min, sz_max = surface_z.min(), surface_z.max()
        sz_range = sz_max - sz_min if sz_max > sz_min else 1.0
        surface_z_norm = ((surface_z - sz_min) / sz_range).astype(np.float32)
        channels.append(surface_z_norm)
        names.append("surface_elevation")

        print(f"  DSM: surface range [{sz_min:.1f}, {sz_max:.1f}]m")

    if not channels:
        return np.empty((n, 0), dtype=np.float32), []

    features = np.column_stack(channels)
    print(f"  2D features: {len(names)} channels → {names}")
    return features, names
