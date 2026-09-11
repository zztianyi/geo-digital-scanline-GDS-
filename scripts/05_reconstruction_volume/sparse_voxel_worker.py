"""Lightweight worker implementation for sparse voxel generation.

This module intentionally imports only the geometry dependencies needed by a
spawned voxel worker.  The production module contains the orchestration and
keeps the legacy implementation available as a fallback.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon


def _empty_ids() -> np.ndarray:
    return np.empty(0, dtype=np.uint64)


def _empty_diagnostics() -> dict:
    return {
        "faces_seen": 0,
        "faces_positive_height": 0,
        "faces_degenerate": 0,
        "candidate_xy_count": 0,
        "inside_xy_count": 0,
        "candidate_z_count": 0,
        "valid_z_count": 0,
        "emitted_ids_before_local_unique": 0,
        "local_unique_ids": 0,
        "contains_calls": 0,
        "contains_points_batches": 0,
        "worker_compute_seconds": 0.0,
    }


def _merge_local_runs(runs: list[np.ndarray]) -> np.ndarray:
    if not runs:
        return _empty_ids()
    if len(runs) == 1:
        return runs[0]
    return np.unique(np.concatenate(runs))


def _voxelize_triangle(
    tri: np.ndarray,
    height: float,
    grid_shape: tuple[int, int, int],
    origin: np.ndarray,
    voxel_size: float,
    max_xy_candidates: int,
    max_emit_ids: int,
    flush_ids: int,
) -> tuple[np.ndarray, dict]:
    diagnostics = _empty_diagnostics()
    diagnostics["faces_seen"] = 1
    if height <= 1e-8:
        return _empty_ids(), diagnostics

    diagnostics["faces_positive_height"] = 1
    tri_xy = tri[:, :2]
    poly = Polygon(tri_xy)
    if poly.area < 1e-10:
        diagnostics["faces_degenerate"] = 1
        return _empty_ids(), diagnostics

    path = Path(tri_xy)
    min_xy = np.min(tri_xy, axis=0)
    max_xy = np.max(tri_xy, axis=0)
    min_z = float(np.min(tri[:, 2]))
    max_z = float(np.mean(tri[:, 2]) + height)

    min_idx = np.floor(
        (np.array([min_xy[0], min_xy[1], min_z]) - origin) / voxel_size
    ).astype(np.int64)
    max_idx = np.ceil(
        (np.array([max_xy[0], max_xy[1], max_z]) - origin) / voxel_size
    ).astype(np.int64)
    min_idx = np.maximum(min_idx, 0)
    max_idx = np.minimum(max_idx, np.asarray(grid_shape, dtype=np.int64) - 1)
    if np.any(max_idx < min_idx):
        return _empty_ids(), diagnostics

    ix_values = np.arange(min_idx[0], max_idx[0] + 1, dtype=np.int32)
    iy_values = np.arange(min_idx[1], max_idx[1] + 1, dtype=np.int32)
    iz_candidates = np.arange(min_idx[2], max_idx[2] + 1, dtype=np.int32)
    diagnostics["candidate_z_count"] = int(iz_candidates.size)

    z_centers = origin[2] + voxel_size * (
        iz_candidates.astype(np.float64) + 0.5
    )
    valid_z = iz_candidates[(z_centers >= min_z) & (z_centers <= max_z)]
    diagnostics["valid_z_count"] = int(valid_z.size)
    if valid_z.size == 0:
        return _empty_ids(), diagnostics

    ny = max(1, int(iy_values.size))
    ix_chunk_size = max(1, int(max_xy_candidates) // ny)
    nz = int(valid_z.size)
    ids_per_xy_batch = max(1, int(max_emit_ids) // nz)

    local_runs: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    pending_count = 0

    def flush_pending() -> None:
        nonlocal pending, pending_count
        if pending:
            local_runs.append(np.unique(np.concatenate(pending)))
            pending = []
            pending_count = 0

    for ix_start in range(0, ix_values.size, ix_chunk_size):
        ix_chunk = ix_values[ix_start : ix_start + ix_chunk_size]
        ix_flat = np.repeat(ix_chunk, iy_values.size)
        iy_flat = np.tile(iy_values, ix_chunk.size)
        xy_centers = np.empty((ix_flat.size, 2), dtype=np.float64)
        xy_centers[:, 0] = origin[0] + voxel_size * (
            ix_flat.astype(np.float64) + 0.5
        )
        xy_centers[:, 1] = origin[1] + voxel_size * (
            iy_flat.astype(np.float64) + 0.5
        )

        inside_mask = path.contains_points(xy_centers)
        diagnostics["contains_points_batches"] += 1
        diagnostics["candidate_xy_count"] += int(ix_flat.size)
        if not np.any(inside_mask):
            continue

        inside_ix = ix_flat[inside_mask].astype(np.uint64)
        inside_iy = iy_flat[inside_mask].astype(np.uint64)
        diagnostics["inside_xy_count"] += int(inside_ix.size)
        base_ids = (
            inside_ix * np.uint64(grid_shape[1]) + inside_iy
        ) * np.uint64(grid_shape[2])

        for xy_start in range(0, base_ids.size, ids_per_xy_batch):
            base_batch = base_ids[xy_start : xy_start + ids_per_xy_batch]
            ids = (
                base_batch[:, None]
                + valid_z.astype(np.uint64)[None, :]
            ).ravel()
            diagnostics["emitted_ids_before_local_unique"] += int(ids.size)
            pending.append(ids)
            pending_count += int(ids.size)
            if pending_count >= flush_ids:
                flush_pending()

    flush_pending()
    worker_ids = _merge_local_runs(local_runs)
    diagnostics["local_unique_ids"] = int(worker_ids.size)
    return worker_ids, diagnostics


def voxelize_face_batch_worker(payload: tuple) -> tuple[np.ndarray, dict]:
    """Voxelize one balanced face batch and return sorted unique uint64 IDs."""
    (
        triangles,
        heights,
        grid_shape,
        origin,
        voxel_size,
        max_xy_candidates,
        max_emit_ids,
        flush_ids,
    ) = payload
    started = perf_counter()
    triangles = np.asarray(triangles, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    grid_shape = tuple(int(value) for value in grid_shape)

    all_runs: list[np.ndarray] = []
    totals = _empty_diagnostics()
    pending: list[np.ndarray] = []
    pending_count = 0

    def flush_pending() -> None:
        nonlocal pending, pending_count
        if pending:
            all_runs.append(np.unique(np.concatenate(pending)))
            pending = []
            pending_count = 0

    for tri, height in zip(triangles, heights):
        ids, current = _voxelize_triangle(
            tri,
            float(height),
            grid_shape,
            origin,
            float(voxel_size),
            int(max_xy_candidates),
            int(max_emit_ids),
            int(flush_ids),
        )
        for key, value in current.items():
            if key == "worker_compute_seconds":
                continue
            totals[key] += value
        if ids.size:
            pending.append(ids)
            pending_count += int(ids.size)
            if pending_count >= flush_ids:
                flush_pending()

    flush_pending()
    worker_ids = _merge_local_runs(all_runs)
    totals["local_unique_ids"] = int(worker_ids.size)
    totals["worker_compute_seconds"] = perf_counter() - started
    return worker_ids, totals
