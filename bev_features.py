"""
Bird's Eye View (BEV) feature computation for point clouds.

Computes per-point spatial context features from a top-down 2D grid projection.
These features help the model understand xy-layout information that pure 3D local
neighborhoods miss — critical for separating adjacent buildings and filtering
vegetation-on-roof confusion.

BEV features per point (7 channels):
  - point_density:       number of points in the xy cell (log-scaled)
  - height_above_ground: z - z_min in the cell
  - height_range:        z_max - z_min in the cell
  - height_mean:         mean z in the cell
  - height_std:          std of z in the cell
  - z_normalized:        global height (z - z_min) / (z_max - z_min), 0→1
                         separates water (~0) from rooftops (~0.8+)
  - height_rank:         position within cell's vertical column (0→1)
                         walls span full range, roofs cluster near 1.0
"""

import numpy as np
from typing import Tuple


def compute_bev_features(
    points: np.ndarray,
    cell_size: float = 1.0,
) -> np.ndarray:
    """Compute BEV features for each point based on a top-down xy grid.

    Args:
        points: (N, 3+) array with at least x, y, z columns.
        cell_size: resolution of the BEV grid in the same unit as the point cloud.

    Returns:
        bev_feats: (N, 7) float32 array of BEV features per point.
    """
    xyz = points[:, :3].astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # build grid indices
    x_min, y_min = x.min(), y.min()
    col = ((x - x_min) / cell_size).astype(np.int64)
    row = ((y - y_min) / cell_size).astype(np.int64)

    n_cols = col.max() + 1
    cell_id = row * n_cols + col  # flat cell index per point

    # aggregate stats per cell using numpy bincount
    unique_cells, inverse = np.unique(cell_id, return_inverse=True)
    n_cells = len(unique_cells)

    cell_count = np.bincount(inverse, minlength=n_cells).astype(np.float64)
    cell_z_sum = np.bincount(inverse, weights=z, minlength=n_cells)
    cell_z_sq_sum = np.bincount(inverse, weights=z * z, minlength=n_cells)
    cell_z_min = np.full(n_cells, np.inf)
    cell_z_max = np.full(n_cells, -np.inf)

    # min/max per cell (vectorized via sort trick)
    order = np.argsort(inverse)
    sorted_inv = inverse[order]
    sorted_z = z[order]

    # find first and last index per cell
    boundaries = np.searchsorted(sorted_inv, np.arange(n_cells), side="left")
    boundaries_right = np.searchsorted(sorted_inv, np.arange(n_cells), side="right")

    for i in range(n_cells):
        lo, hi = boundaries[i], boundaries_right[i]
        if lo < hi:
            cell_z_min[i] = sorted_z[lo:hi].min()
            cell_z_max[i] = sorted_z[lo:hi].max()

    # per-cell stats
    cell_mean = cell_z_sum / np.maximum(cell_count, 1)
    cell_var = (cell_z_sq_sum / np.maximum(cell_count, 1)) - cell_mean ** 2
    cell_std = np.sqrt(np.maximum(cell_var, 0.0))
    cell_range = cell_z_max - cell_z_min

    # map back to per-point
    density = np.log1p(cell_count[inverse])  # log-scaled density
    height_above_ground = z - cell_z_min[inverse]
    height_range = cell_range[inverse]
    height_mean = cell_mean[inverse]
    height_std = cell_std[inverse]

    # Global normalized height: separates water (~0) from rooftops (~0.8+)
    z_global_min, z_global_max = z.min(), z.max()
    z_global_range = z_global_max - z_global_min
    if z_global_range > 0:
        z_normalized = (z - z_global_min) / z_global_range
    else:
        z_normalized = np.zeros_like(z)

    # Height rank within cell: walls span 0→1, roofs cluster near 1.0,
    # vegetation is scattered. Avoids division by zero for single-height cells.
    cell_range_per_point = cell_range[inverse]
    safe_range = np.where(cell_range_per_point > 0, cell_range_per_point, 1.0)
    height_rank = np.where(
        cell_range_per_point > 0,
        height_above_ground / safe_range,
        0.5,  # single-height cell → neutral mid-value
    )

    bev_feats = np.column_stack([
        density,
        height_above_ground,
        height_range,
        height_mean,
        height_std,
        z_normalized,
        height_rank,
    ]).astype(np.float32)

    return bev_feats


def compute_bev_features_chunked(
    points: np.ndarray,
    cell_size: float = 1.0,
    chunk_size: int = 5_000_000,
) -> np.ndarray:
    """Memory-efficient BEV computation for very large point clouds.

    First pass: compute global cell statistics over chunks.
    Second pass: assign per-point features using those statistics.

    Args:
        points: (N, 3+) array.
        cell_size: BEV grid resolution.
        chunk_size: number of points to process at a time.

    Returns:
        bev_feats: (N, 7) float32.
    """
    n = len(points)
    if n <= chunk_size:
        return compute_bev_features(points, cell_size)

    xyz = points[:, :3].astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    x_min, y_min = x.min(), y.min()
    col = ((x - x_min) / cell_size).astype(np.int64)
    row = ((y - y_min) / cell_size).astype(np.int64)
    n_cols = col.max() + 1
    cell_id = row * n_cols + col

    # pass 1: accumulate cell stats in a dict (handles sparse grids for huge areas)
    cell_stats = {}  # cell_id -> [count, z_sum, z_sq_sum, z_min, z_max]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_cells = cell_id[start:end]
        chunk_z = z[start:end]

        for cid in np.unique(chunk_cells):
            mask = chunk_cells == cid
            cz = chunk_z[mask]
            if cid not in cell_stats:
                cell_stats[cid] = [0, 0.0, 0.0, np.inf, -np.inf]
            s = cell_stats[cid]
            s[0] += len(cz)
            s[1] += cz.sum()
            s[2] += (cz * cz).sum()
            s[3] = min(s[3], cz.min())
            s[4] = max(s[4], cz.max())

    # global z range for z_normalized
    z_global_min, z_global_max = z.min(), z.max()
    z_global_range = z_global_max - z_global_min
    if z_global_range == 0:
        z_global_range = 1.0

    # pass 2: build per-point features in chunks
    bev_feats = np.empty((n, 7), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_cells = cell_id[start:end]
        chunk_z = z[start:end]

        for i, cid in enumerate(chunk_cells):
            s = cell_stats[cid]
            count, z_sum, z_sq_sum, z_min, z_max = s
            mean = z_sum / count
            var = max((z_sq_sum / count) - mean * mean, 0.0)
            h_range = z_max - z_min
            h_above = chunk_z[i] - z_min
            bev_feats[start + i, 0] = np.log1p(count)
            bev_feats[start + i, 1] = h_above
            bev_feats[start + i, 2] = h_range
            bev_feats[start + i, 3] = mean
            bev_feats[start + i, 4] = np.sqrt(var)
            bev_feats[start + i, 5] = (chunk_z[i] - z_global_min) / z_global_range
            bev_feats[start + i, 6] = (h_above / h_range) if h_range > 0 else 0.5

    return bev_feats
