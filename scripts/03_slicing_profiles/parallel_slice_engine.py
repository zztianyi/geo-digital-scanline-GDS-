"""Shared, spawn-safe mesh sections with a bounded persistent process pool.

Face IDs always index the Z-filtered mesh, in the original submesh order.
Workers open the prepared arrays once and never receive mesh arrays in tasks.
No angular, branch, or opposite-half-plane cropping is performed here.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
from queue import SimpleQueue
from time import perf_counter

import numpy as np
import trimesh

from generate_scanline_slices import (
    compute_slice_plane,
    filter_mesh_by_z_range,
    slice_and_check_faces,
)


_WORKER_MESH = None
_WORKER_PLANE_ARGS = None


def _validated_arc_config(arc_config):
    """Normalize the small initializer payload, rejecting invalid planes early."""
    if not isinstance(arc_config, dict):
        raise ValueError("arc_config must be a dict")
    try:
        center = np.asarray(arc_config["center"], dtype=np.float64)
        radius = float(arc_config["radius"])
        angle_min = float(arc_config["angle_min"])
        angle_max = float(arc_config["angle_max"])
        z_range = np.asarray(arc_config["z_range"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid arc configuration") from exc
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("arc_config center must contain two finite coordinates")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("arc_config radius must be finite and positive")
    if not np.isfinite([angle_min, angle_max]).all() or angle_max < angle_min:
        raise ValueError("arc_config angles must be finite and ordered")
    if z_range.shape != (2,) or not np.isfinite(z_range).all() or z_range[1] < z_range[0]:
        raise ValueError("arc_config z_range must contain two ordered finite values")
    return {
        "center": center.tolist(),
        "radius": radius,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "z_range": z_range.tolist(),
    }


def prepare_mesh(mesh_path, arc_config_path, cache_dir):
    """Prepare a new cache directory and return its JSON-serializable metadata.

    ``cache_dir`` must not already exist, even if empty. Failed preparations are
    left in place for inspection and are never silently reused or overwritten.
    ``metadata.json`` is written last, after ``vertices.npy`` and ``faces.npy``.
    Bounds are those of the retained whole triangles; an empty mesh uses null.
    """
    mesh_path = Path(mesh_path).resolve()
    arc_config_path = Path(arc_config_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    with arc_config_path.open("r", encoding="utf-8") as stream:
        arc_config = _validated_arc_config(json.load(stream))
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    cache_dir.mkdir(parents=True, exist_ok=False)

    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh_path must load a triangle mesh or mesh Scene")
    original_vertex_count = len(mesh.vertices)
    original_face_count = len(mesh.faces)
    mesh_filtered = filter_mesh_by_z_range(mesh, *arc_config["z_range"])
    del mesh
    if mesh_filtered is None:
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int64)
        bounds = None
    else:
        vertices = np.asarray(mesh_filtered.vertices, dtype=np.float64)
        faces = np.asarray(mesh_filtered.faces, dtype=np.int64)
        bounds = mesh_filtered.bounds.tolist() if len(faces) else None

    metadata = {
        "format_version": 1,
        "cache_dir": str(cache_dir),
        "mesh_path": str(mesh_path),
        "arc_config_path": str(arc_config_path),
        "arc_config": arc_config,
        "z_range": arc_config["z_range"],
        "original_vertex_count": original_vertex_count,
        "original_face_count": original_face_count,
        "filtered_vertex_count": len(vertices),
        "filtered_face_count": len(faces),
        "bounds": bounds,
        "face_id_space": "filtered_mesh",
        "trimesh_version": trimesh.__version__,
    }
    for name, values in (("vertices", vertices), ("faces", faces)):
        with (cache_dir / f"{name}.npy").open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
    with (cache_dir / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return metadata


def _initialize_worker(cache_dir, arc_config):
    global _WORKER_MESH, _WORKER_PLANE_ARGS
    cache_dir = Path(cache_dir)
    vertices = np.load(cache_dir / "vertices.npy", mmap_mode="r", allow_pickle=False)
    faces = np.load(cache_dir / "faces.npy", mmap_mode="r", allow_pickle=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype != np.float64:
        raise ValueError("vertices.npy must have shape (N, 3) and dtype float64")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != np.int64:
        raise ValueError("faces.npy must have shape (N, 3) and dtype int64")
    _WORKER_MESH = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False, validate=False
    )
    _WORKER_MESH.vertices.flags.writeable = False
    _WORKER_MESH.faces.flags.writeable = False
    radius = arc_config["radius"]
    angle_min = arc_config["angle_min"]
    _WORKER_PLANE_ARGS = (
        np.append(np.asarray(arc_config["center"], dtype=np.float64), 0.0),
        radius,
        angle_min,
        [0, radius * (arc_config["angle_max"] - angle_min)],
    )


def _slice_worker(axis, position):
    started = perf_counter()
    if axis == "vertical":
        _, radius, angle_min, _ = _WORKER_PLANE_ARGS
        if not np.isfinite(angle_min + position / radius):
            raise ValueError("The vertical slice angle must be finite")
        plane_params = compute_slice_plane(*_WORKER_PLANE_ARGS, position)
    else:
        plane_params = {
            "origin": np.array([0.0, 0.0, position], dtype=np.float64),
            "normal": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
    if len(_WORKER_MESH.faces):
        lines_3d, face_ids = slice_and_check_faces(
            _WORKER_MESH, plane_params["origin"], plane_params["normal"]
        )
    else:
        lines_3d = np.empty((0, 2, 3), dtype=np.float64)
        face_ids = np.empty(0, dtype=np.int64)
    return {
        "axis": axis,
        "position": position,
        "lines_3d": np.asarray(lines_3d, dtype=np.float64).reshape((-1, 2, 3)),
        "face_ids": np.asarray(face_ids, dtype=np.int64).reshape(-1),
        "plane_params": plane_params,
        "compute_seconds": perf_counter() - started,
        "worker_pid": os.getpid(),
    }


def _validated_spec(spec):
    try:
        axis = spec["axis"]
        position = float(spec["position"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Each slice needs an axis and finite numeric position") from exc
    if axis not in ("vertical", "horizontal"):
        raise ValueError("Slice axis must be 'vertical' or 'horizontal'")
    if not np.isfinite(position):
        raise ValueError("Slice position must be finite")
    return axis, position


class ParallelSliceEngine:
    """Context-managed persistent pool, also used when ``workers=1``.

    ``iter_slices`` accepts a list or iterable of small axis/position specs and
    yields results in observed completion order. Use (axis, position) as the
    stable result key. Each section preserves trimesh's segment and face order.
    Only one iterator may be active at a time; complete batches reuse the pool.
    Closing an iterator early or encountering an error closes the engine too.
    """

    def __init__(self, cache_dir, arc_config: dict, workers=1):
        if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if os.name == "nt" and workers > 61:
            raise ValueError("Windows ProcessPoolExecutor supports at most 61 workers")
        self.workers = int(workers)
        self.max_pending = self.workers * 2
        self.cache_dir = str(Path(cache_dir).resolve())
        self.arc_config = _validated_arc_config(arc_config)
        self.metadata = None
        self._executor = None
        self._pending = set()
        self._iterating = False
        self._closed = False

    def __enter__(self):
        if self._closed or self._executor is not None:
            raise RuntimeError("The engine is already entered or closed")
        cache_dir = Path(self.cache_dir)
        with (cache_dir / "metadata.json").open("r", encoding="utf-8") as stream:
            self.metadata = json.load(stream)
        for name in ("vertices.npy", "faces.npy"):
            if not (cache_dir / name).is_file():
                raise FileNotFoundError(cache_dir / name)
        if self.metadata["z_range"] != self.arc_config["z_range"]:
            raise ValueError("arc_config z_range does not match the prepared mesh")
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(self.cache_dir, self.arc_config),
        )
        return self

    def iter_slices(self, specs):
        if self._executor is None:
            raise RuntimeError("Use the engine inside an open context manager")
        if self._iterating:
            raise RuntimeError("Only one iter_slices iterator may be active")
        specs = iter(specs)
        completed = SimpleQueue()
        self._iterating = True

        def submit_next():
            try:
                spec = next(specs)
            except StopIteration:
                return False
            axis, position = _validated_spec(spec)
            future = self._executor.submit(_slice_worker, axis, position)
            self._pending.add(future)
            future.add_done_callback(completed.put)
            return True

        try:
            for _ in range(self.max_pending):
                if not submit_next():
                    break
            while self._pending:
                if self._executor is None:
                    raise RuntimeError("The engine was closed during iteration")
                future = completed.get()
                self._pending.remove(future)
                result = future.result()
                del future
                # Refill before handing control to a possibly slow consumer.
                submit_next()
                yield result
                del result
        except BaseException:
            self.close()
            raise
        finally:
            self._iterating = False

    def close(self):
        """Cancel queued work and wait for workers to release mapped files."""
        executor, self._executor = self._executor, None
        self._closed = True
        for future in self._pending:
            future.cancel()
        self._pending.clear()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
