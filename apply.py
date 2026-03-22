"""
Post-processing pipeline: apply predicted labels to point clouds, then
count and extract individual buildings via DBSCAN clustering.

Outputs:
  - Classified LAS file with ASPRS codes
  - Building count + per-building bounding boxes (JSON)
  - Optional per-building extracted point clouds

Handles very large files (150-300 GB) via chunked LAS I/O.
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

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from bev_features import compute_bev_features


# ---------------------------------------------------------------------------
# BEV channel indices (must match bev_features.py output order)
# ---------------------------------------------------------------------------
BEV_DENSITY = 0
BEV_HEIGHT_ABOVE_GROUND = 1
BEV_HEIGHT_RANGE = 2
BEV_HEIGHT_MEAN = 3
BEV_HEIGHT_STD = 4
BEV_Z_NORMALIZED = 5
BEV_HEIGHT_RANK = 6


# ---------------------------------------------------------------------------
# Label mapping: Toronto3D model class → ASPRS LAS classification
# ---------------------------------------------------------------------------

DEFAULT_LABEL_MAP = {
    0: 2,   # Road             → ground
    1: 2,   # Road Marking     → ground
    2: 5,   # Natural          → high vegetation
    3: 6,   # Building         → building
    4: 3,   # Utility Line     → low vegetation
    5: 8,   # Pole             → key-point
    6: 9,   # Car              → water (visual placeholder)
    7: 7,   # Fence            → low point (noise)
}

BUILDING_MODEL_CLASS = 3   # Toronto3D class index for "building"
BUILDING_ASPRS_CLASS = 6   # ASPRS class for "building"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_label_mapping(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def read_point_cloud(path: str):
    """Read full point cloud. Returns (points, colors, source_header_or_None).

    For LAS/LAZ: extracts arrays and frees the laspy object to avoid
    holding ~60 GB of duplicate data for large (2B+ point) files.
    Returns the header (not full las) so write_classified_las can
    preserve format info without keeping point data alive twice.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".las", ".laz"):
        las = laspy.read(path)
        n = las.header.point_count
        print(f"  Allocating {n:,} points (~{n * 3 * 8 / 1e9:.1f} GB float64 xyz)")
        points = np.empty((n, 3), dtype=np.float64)
        points[:, 0] = las.x
        points[:, 1] = las.y
        points[:, 2] = las.z
        colors = np.empty((0,))
        try:
            colors = np.empty((n, 3), dtype=np.uint16)
            colors[:, 0] = las.red
            colors[:, 1] = las.green
            colors[:, 2] = las.blue
        except Exception:
            pass
        header = las.header
        del las  # free laspy internal buffers
        return points, colors, header

    elif ext == ".ply":
        try:
            import open3d as o3d
        except ImportError:
            sys.exit("open3d is required for PLY.  pip install open3d")
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 65535).astype(np.uint16) if len(pcd.colors) > 0 else np.empty((0,))
        return points, colors, None

    else:
        sys.exit(f"Unsupported format: {ext}")


def read_point_cloud_chunked(path: str, chunk_size: int = 5_000_000):
    """Yield (points, colors, chunk_index) from a LAS/LAZ file."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".las", ".laz"):
        # fallback: load all at once and yield single chunk
        pts, colors, _ = read_point_cloud(path)
        yield pts, colors, 0
        return

    with laspy.open(path) as reader:
        total = reader.header.point_count
        print(f"Chunked read: {total:,} points, chunk_size={chunk_size:,}")
        for idx, chunk in enumerate(reader.chunk_iterator(chunk_size)):
            pts = np.stack([chunk.x, chunk.y, chunk.z], axis=-1)
            try:
                colors = np.stack([chunk.red, chunk.green, chunk.blue], axis=-1)
            except Exception:
                colors = np.empty((0,))
            yield pts, colors, idx


# ---------------------------------------------------------------------------
# Write classified LAS
# ---------------------------------------------------------------------------

def write_classified_las(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    classification: np.ndarray,
    source_header=None,
    use_las14: bool = False,
):
    _ensure_parent_dir(output_path)

    max_class = int(classification.max())
    if max_class > 31 and not use_las14:
        print(f"WARNING: max class {max_class} > 31; upgrading to LAS 1.4")
        use_las14 = True

    has_color = colors.size > 0

    # Try to reuse source header's point format/version when compatible
    if source_header is not None and not use_las14:
        src_ver = f"{source_header.version.major}.{source_header.version.minor}"
        pf = source_header.point_format.id
        ver = src_ver
    elif use_las14:
        pf, ver = (7 if has_color else 6), "1.4"
    else:
        pf, ver = (2 if has_color else 0), "1.2"

    header = laspy.LasHeader(point_format=pf, version=ver)
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    if has_color:
        las.red   = colors[:, 0].astype(np.uint16)
        las.green = colors[:, 1].astype(np.uint16)
        las.blue  = colors[:, 2].astype(np.uint16)
    las.classification = classification.astype(np.uint8)
    las.write(output_path)


def write_classified_las_inplace(
    input_path: str,
    output_path: str,
    classification: np.ndarray,
):
    """Re-read the source LAS, overwrite only the classification field, and
    write to *output_path*.  Preserves ALL original fields (intensity,
    return number, GPS time, NIR, etc.) so the output size matches the input.
    """
    _ensure_parent_dir(output_path)
    print(f"  Re-reading source LAS to preserve all fields...")
    las = laspy.read(input_path)
    if len(las.points) != len(classification):
        sys.exit(f"Mismatch: {len(las.points):,} points in LAS vs "
                 f"{len(classification):,} classification labels")
    las.classification = classification.astype(np.uint8)
    las.write(output_path)
    del las


# ---------------------------------------------------------------------------
# Building counting via DBSCAN on 2D (xy) projection
# ---------------------------------------------------------------------------

def estimate_ground_z(points_z: np.ndarray, percentile: float = 5.0) -> float:
    """Estimate ground level as a low percentile of z values."""
    return np.percentile(points_z, percentile)


def _validate_cluster_bev(
    cluster_bev: np.ndarray,
    scene_median_density: float,
    min_height_above_ground: float = 2.5,
    min_height_range: float = 2.0,
    min_height_rank: float = 0.5,
    min_footprint_area: float = 15.0,
    cluster_pts_xy: np.ndarray | None = None,
) -> tuple[bool, dict]:
    """Score a DBSCAN cluster using BEV feature statistics.

    Args:
        cluster_bev: (M, 7) BEV features for points in this cluster.
        scene_median_density: median density across all points (for comparison).
        min_height_above_ground: reject clusters whose median local height is below this.
        min_height_range: reject clusters with median cell height_range below this.
        min_height_rank: reject clusters whose median height_rank is below this.
        min_footprint_area: reject clusters whose XY footprint area is below this (m²).
        cluster_pts_xy: (M, 2) xy coordinates for footprint area calculation.

    Returns:
        (is_valid, stats_dict) — whether cluster passes, and diagnostic stats.
    """
    med_hag = float(np.median(cluster_bev[:, BEV_HEIGHT_ABOVE_GROUND]))
    med_hrange = float(np.median(cluster_bev[:, BEV_HEIGHT_RANGE]))
    med_hrank = float(np.median(cluster_bev[:, BEV_HEIGHT_RANK]))
    med_density = float(np.median(cluster_bev[:, BEV_DENSITY]))
    med_hstd = float(np.median(cluster_bev[:, BEV_HEIGHT_STD]))

    # XY footprint area (bounding box)
    footprint_area = 0.0
    if cluster_pts_xy is not None and len(cluster_pts_xy) > 1:
        xy_min = cluster_pts_xy.min(axis=0)
        xy_max = cluster_pts_xy.max(axis=0)
        footprint_area = float((xy_max[0] - xy_min[0]) * (xy_max[1] - xy_min[1]))

    stats = {
        "median_height_above_ground": round(med_hag, 2),
        "median_height_range": round(med_hrange, 2),
        "median_height_rank": round(med_hrank, 3),
        "median_density": round(med_density, 2),
        "median_height_std": round(med_hstd, 2),
        "footprint_area_m2": round(footprint_area, 1),
    }

    # Rejection criteria — any failure rejects
    is_valid = True
    if med_hag < min_height_above_ground:
        is_valid = False
    if med_hrange < min_height_range:
        is_valid = False
    if med_hrank < min_height_rank:
        is_valid = False
    if footprint_area < min_footprint_area:
        is_valid = False

    return is_valid, stats


def count_buildings(
    points: np.ndarray,
    pred_labels: np.ndarray,
    building_class: int = BUILDING_MODEL_CLASS,
    min_height_above_ground: float = 2.0,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 50,
    min_building_points: int = 200,
    bev_features: np.ndarray | None = None,
    bev_cell_size: float = 1.0,
) -> dict:
    """Cluster building points and return count + per-building bounding boxes.

    When *bev_features* is provided (or computed via --bev), uses per-cell
    local height_above_ground instead of a single global ground percentile,
    and validates each cluster using BEV statistics (height_range, height_rank,
    density, footprint area).

    Steps:
        1. Filter to building class only
        2. Height filter using BEV local ground (or global percentile fallback)
        3. Project to 2D (xy) to avoid z-axis fragmentation
        4. Adaptive DBSCAN eps if not provided
        5. DBSCAN clustering
        6. Validate each cluster with BEV statistics (or point-count-only fallback)
        7. Compute 3D bounding box per surviving cluster

    Returns dict with:
        - building_count: int
        - buildings: list of {id, point_count, bbox_min, bbox_max, centroid, bev_stats}
        - cluster_labels: np.ndarray same length as input, -1 for non-building
    """
    n = len(points)
    cluster_labels = np.full(n, -1, dtype=np.int32)

    # step 1: building mask
    building_mask = pred_labels == building_class
    building_count_raw = int(building_mask.sum())
    if building_count_raw == 0:
        return {"building_count": 0, "buildings": [], "cluster_labels": cluster_labels}

    building_points = points[building_mask]

    # ---------- BEV path vs legacy path ----------
    use_bev = bev_features is not None

    if use_bev:
        building_bev = bev_features[building_mask]  # (M, 7)

        # step 2 (BEV): use per-cell height_above_ground for filtering
        local_hag = building_bev[:, BEV_HEIGHT_ABOVE_GROUND]
        height_mask = local_hag > min_height_above_ground

        if height_mask.sum() == 0:
            return {"building_count": 0, "buildings": [], "cluster_labels": cluster_labels}

        filtered_points = building_points[height_mask]
        filtered_bev = building_bev[height_mask]

        # scene-wide median density for relative comparison
        scene_median_density = float(np.median(bev_features[:, BEV_DENSITY]))

        print(f"  Building points (BEV): {building_count_raw:,} raw → {len(filtered_points):,} "
              f"after local height filter (threshold=+{min_height_above_ground:.1f}m)")
    else:
        # legacy path: global ground estimate
        building_z = building_points[:, 2]
        ground_z = estimate_ground_z(points[:, 2])
        height_mask = building_z > (ground_z + min_height_above_ground)

        if height_mask.sum() == 0:
            return {"building_count": 0, "buildings": [], "cluster_labels": cluster_labels}

        filtered_points = building_points[height_mask]
        filtered_bev = None
        scene_median_density = 0.0

        print(f"  Building points: {building_count_raw:,} raw → {len(filtered_points):,} after height filter "
              f"(ground_z={ground_z:.1f}, threshold=+{min_height_above_ground:.1f}m)")

    # step 3: project to 2D for clustering
    xy = filtered_points[:, :2]

    # step 4: adaptive eps
    if dbscan_eps is None:
        nn = NearestNeighbors(n_neighbors=min(20, len(xy)))
        nn.fit(xy)
        distances, _ = nn.kneighbors(xy)
        dbscan_eps = float(np.percentile(distances[:, -1], 90))
        dbscan_eps = max(dbscan_eps, 0.5)  # floor at 0.5m
        print(f"  Adaptive DBSCAN eps = {dbscan_eps:.2f}m")

    # step 5: DBSCAN
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(xy)
    raw_labels = clustering.labels_

    # step 6: filter clusters and build results
    buildings = []
    rejected = 0
    unique_labels = set(raw_labels)
    unique_labels.discard(-1)

    # map back to original point indices
    building_indices = np.where(building_mask)[0]
    height_indices = np.where(height_mask)[0]

    bid = 0
    for label in sorted(unique_labels):
        cluster_mask = raw_labels == label
        count = int(cluster_mask.sum())
        if count < min_building_points:
            continue

        cluster_pts = filtered_points[cluster_mask]

        # BEV validation
        if use_bev and filtered_bev is not None:
            cluster_bev = filtered_bev[cluster_mask]
            is_valid, bev_stats = _validate_cluster_bev(
                cluster_bev,
                scene_median_density,
                min_height_above_ground=min_height_above_ground,
                cluster_pts_xy=cluster_pts[:, :2],
            )
            if not is_valid:
                rejected += 1
                continue
        else:
            bev_stats = {}

        bbox_min = cluster_pts.min(axis=0).tolist()
        bbox_max = cluster_pts.max(axis=0).tolist()
        centroid = cluster_pts.mean(axis=0).tolist()

        building_info = {
            "id": bid,
            "point_count": count,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "centroid": centroid,
        }
        if bev_stats:
            building_info["bev_stats"] = bev_stats

        buildings.append(building_info)

        # write cluster ID back to full-size label array
        original_indices = building_indices[height_indices[cluster_mask]]
        cluster_labels[original_indices] = bid
        bid += 1

    if use_bev:
        print(f"  Buildings found: {len(buildings)} "
              f"(rejected {rejected} by BEV validation, "
              f"filtered clusters < {min_building_points} pts)")
    else:
        print(f"  Buildings found: {len(buildings)} (after filtering clusters < {min_building_points} pts)")

    return {
        "building_count": len(buildings),
        "buildings": buildings,
        "cluster_labels": cluster_labels,
    }


# ---------------------------------------------------------------------------
# Main: apply labels + count buildings
# ---------------------------------------------------------------------------

def apply_labels(
    input_path: str,
    labels_path: str,
    output_path: str,
    building_json_path: str | None = None,
    mapping: dict | None = None,
    use_las14: bool = False,
    min_height: float = 2.0,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 50,
    min_building_points: int = 200,
    use_bev: bool = False,
    bev_cell_size: float = 1.0,
):
    label_map = mapping if mapping is not None else DEFAULT_LABEL_MAP

    # load predicted labels
    ext = os.path.splitext(labels_path)[1].lower()
    if ext == ".npy":
        pred_labels = np.load(labels_path)
    elif ext in (".txt", ".csv"):
        pred_labels = np.loadtxt(labels_path, dtype=np.int32)
    else:
        sys.exit(f"Unsupported label format: {ext}")

    pred_labels = pred_labels.astype(np.int32).ravel()
    print(f"Loaded {len(pred_labels):,} labels (unique: {np.unique(pred_labels).tolist()})")

    # load point cloud (las object freed internally, only header kept)
    points, colors, source_header = read_point_cloud(input_path)
    print(f"Loaded {len(points):,} points from {input_path}")

    if len(points) != len(pred_labels):
        sys.exit(f"Mismatch: {len(points):,} points vs {len(pred_labels):,} labels")

    # ---- map to ASPRS classes ----
    classification = np.zeros(len(pred_labels), dtype=np.uint8)
    unmapped = set()
    for ml in np.unique(pred_labels):
        mask = pred_labels == ml
        if ml in label_map:
            classification[mask] = label_map[ml]
        else:
            unmapped.add(int(ml))
            classification[mask] = 1
    if unmapped:
        print(f"WARNING: labels {unmapped} had no mapping → Unclassified (1)")

    unique_cls, counts = np.unique(classification, return_counts=True)
    print("\nClassification summary:")
    for c, n in zip(unique_cls, counts):
        print(f"  ASPRS {c:3d}  →  {n:>12,} points")

    # ---- write classified LAS (preserve all original fields) ----
    write_classified_las_inplace(input_path, output_path, classification)
    print(f"\nSaved classified LAS → {output_path}")

    # ---- compute BEV features for building validation ----
    bev_feats = None
    if use_bev:
        print("\nComputing BEV features for building validation...")
        bev_feats = compute_bev_features(points, cell_size=bev_cell_size)
        print(f"  BEV features: {bev_feats.shape} (7 channels per point)")

    # ---- building counting ----
    print("\n--- Building Counting ---")
    result = count_buildings(
        points, pred_labels,
        building_class=BUILDING_MODEL_CLASS,
        min_height_above_ground=min_height,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        min_building_points=min_building_points,
        bev_features=bev_feats,
        bev_cell_size=bev_cell_size,
    )

    print(f"\n  Total buildings detected: {result['building_count']}")
    for b in result["buildings"]:
        w = b["bbox_max"][0] - b["bbox_min"][0]
        h = b["bbox_max"][1] - b["bbox_min"][1]
        z = b["bbox_max"][2] - b["bbox_min"][2]
        print(f"    Building {b['id']:3d}: {b['point_count']:>8,} pts  "
              f"bbox {w:.1f}x{h:.1f}x{z:.1f}m  "
              f"center ({b['centroid'][0]:.1f}, {b['centroid'][1]:.1f}, {b['centroid'][2]:.1f})")

    # ---- save building JSON ----
    if building_json_path is None:
        building_json_path = os.path.splitext(output_path)[0] + "_buildings.json"

    _ensure_parent_dir(building_json_path)

    output_data = {
        "source_file": os.path.basename(input_path),
        "total_points": len(points),
        "building_count": result["building_count"],
        "parameters": {
            "min_height_above_ground": min_height,
            "dbscan_eps": dbscan_eps or "adaptive",
            "dbscan_min_samples": dbscan_min_samples,
            "min_building_points": min_building_points,
        },
        "buildings": result["buildings"],
    }

    with open(building_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved building report → {building_json_path}")

    return result


# ---------------------------------------------------------------------------
# Chunked apply for very large files (150-300 GB)
# ---------------------------------------------------------------------------

def apply_labels_chunked(
    input_path: str,
    labels_path: str,
    output_path: str,
    building_json_path: str | None = None,
    mapping: dict | None = None,
    chunk_size: int = 5_000_000,
    min_height: float = 2.0,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 50,
    min_building_points: int = 200,
    use_bev: bool = False,
    bev_cell_size: float = 1.0,
):
    """Process huge files: stream-classify in chunks, accumulate building points
    separately for final DBSCAN pass.

    Strategy:
      Pass 1: stream through chunks → write classified LAS, collect building points
      Pass 2: run DBSCAN on accumulated building points (2D projection)
    """
    label_map = mapping if mapping is not None else DEFAULT_LABEL_MAP

    # load all labels (fits in RAM — labels are int32, ~600MB for 150M pts)
    pred_labels = np.load(labels_path).astype(np.int32).ravel()
    total_labels = len(pred_labels)
    print(f"Loaded {total_labels:,} labels")

    # pass 1: stream and classify, accumulate building xyz
    building_xyz_chunks = []
    building_index_chunks = []
    label_offset = 0

    # we'll write chunked output to a temp file, then finalize
    # for simplicity, accumulate classified chunks and write at end
    # (for truly massive files, use laspy.open in write mode with chunked writes)
    classified_chunks = []
    point_chunks = []
    color_chunks = []

    for pts, colors, chunk_idx in read_point_cloud_chunked(input_path, chunk_size):
        n = len(pts)
        chunk_labels = pred_labels[label_offset:label_offset + n]
        label_offset += n

        # classify
        cls = np.zeros(n, dtype=np.uint8)
        for ml in np.unique(chunk_labels):
            mask = chunk_labels == ml
            cls[mask] = label_map.get(int(ml), 1)

        classified_chunks.append(cls)
        point_chunks.append(pts)
        if colors.size > 0:
            color_chunks.append(colors)

        # collect building points for DBSCAN
        bld_mask = chunk_labels == BUILDING_MODEL_CLASS
        if bld_mask.any():
            building_xyz_chunks.append(pts[bld_mask])
            # global indices
            global_indices = np.where(bld_mask)[0] + (label_offset - n)
            building_index_chunks.append(global_indices)

        print(f"  chunk {chunk_idx}: {n:,} pts classified, "
              f"{bld_mask.sum():,} building pts collected")

    # Build full classification array for in-place write
    all_classification = np.concatenate(classified_chunks)
    all_points = np.concatenate(point_chunks)

    print(f"\nTotal: {len(all_points):,} points classified")

    # write LAS — re-read source to preserve all original fields
    write_classified_las_inplace(input_path, output_path, all_classification)
    print(f"Saved classified LAS → {output_path}")

    # pass 2: DBSCAN on building points (delegate to count_buildings)
    print("\n--- Building Counting (chunked) ---")
    if building_xyz_chunks:
        all_building_xyz = np.concatenate(building_xyz_chunks)
        all_building_indices = np.concatenate(building_index_chunks)

        # Build a pseudo full-points/labels array for count_buildings
        # We only need building points + their labels for the function
        n_bld = len(all_building_xyz)
        pseudo_labels = np.full(n_bld, BUILDING_MODEL_CLASS, dtype=np.int32)

        bev_feats = None
        if use_bev:
            print("  Computing BEV features on building points...")
            bev_feats = compute_bev_features(all_building_xyz, cell_size=bev_cell_size)

        result = count_buildings(
            all_building_xyz, pseudo_labels,
            building_class=BUILDING_MODEL_CLASS,
            min_height_above_ground=min_height,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            min_building_points=min_building_points,
            bev_features=bev_feats,
            bev_cell_size=bev_cell_size,
        )
        buildings = result["buildings"]
    else:
        buildings = []

    # save report
    if building_json_path is None:
        building_json_path = os.path.splitext(output_path)[0] + "_buildings.json"

    _ensure_parent_dir(building_json_path)

    output_data = {
        "source_file": os.path.basename(input_path),
        "total_points": len(all_points),
        "building_count": len(buildings),
        "parameters": {
            "min_height_above_ground": min_height,
            "dbscan_eps": dbscan_eps or "adaptive",
            "dbscan_min_samples": dbscan_min_samples,
            "min_building_points": min_building_points,
        },
        "buildings": buildings,
    }
    with open(building_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved building report → {building_json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply labels to point cloud + count buildings with DBSCAN"
    )
    parser.add_argument("--input", required=True, help="Input point cloud (.las/.laz/.ply)")
    parser.add_argument("--labels", required=True, help="Predicted labels (.npy or .txt)")
    parser.add_argument("--output", required=True, help="Output classified LAS path")
    parser.add_argument("--buildings-json", default=None, help="Output building report JSON path")
    parser.add_argument("--mapping", default=None, help="Custom label mapping JSON file")
    parser.add_argument("--las14", action="store_true", help="Force LAS 1.4 output")

    # building counting params
    parser.add_argument("--min-height", type=float, default=2.0,
                        help="Min height above ground for building points (meters)")
    parser.add_argument("--dbscan-eps", type=float, default=None,
                        help="DBSCAN eps (meters). Auto if not set.")
    parser.add_argument("--dbscan-min-samples", type=int, default=50,
                        help="DBSCAN min_samples")
    parser.add_argument("--min-building-points", type=int, default=200,
                        help="Min points per building cluster")

    # BEV-enhanced building validation
    parser.add_argument("--bev", action="store_true",
                        help="Use BEV features for building validation (local ground, "
                             "height_range, height_rank, density checks per cluster)")
    parser.add_argument("--bev-cell-size", type=float, default=1.0,
                        help="BEV grid cell size in meters (default: 1.0)")

    # chunked mode
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Chunk size for large files (0 = load all). Recommended: 5000000")

    args = parser.parse_args()

    custom_mapping = None
    if args.mapping:
        custom_mapping = load_label_mapping(args.mapping)

    if args.chunk_size > 0:
        apply_labels_chunked(
            input_path=args.input,
            labels_path=args.labels,
            output_path=args.output,
            building_json_path=args.buildings_json,
            mapping=custom_mapping,
            chunk_size=args.chunk_size,
            min_height=args.min_height,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            min_building_points=args.min_building_points,
            use_bev=args.bev,
            bev_cell_size=args.bev_cell_size,
        )
    else:
        apply_labels(
            input_path=args.input,
            labels_path=args.labels,
            output_path=args.output,
            building_json_path=args.buildings_json,
            mapping=custom_mapping,
            use_las14=args.las14,
            min_height=args.min_height,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            min_building_points=args.min_building_points,
            use_bev=args.bev,
            bev_cell_size=args.bev_cell_size,
        )
