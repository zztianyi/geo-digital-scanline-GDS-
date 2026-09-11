"""Run one measured optimized sparse voxelization stage from line-face output."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import threading
import time


def _commit_memory():
    try:
        from gds_project.performance_monitor import commit_memory

        return commit_memory()
    except Exception:
        return None


def _monitor_process_tree(
    pid: int,
    stop: threading.Event,
    rows: list[dict],
    state: dict,
):
    import psutil

    started = time.perf_counter()
    previous_cpu = 0.0
    previous_t = started
    known = {}
    cumulative_cpu = {}
    peak_busy_cores = 0.0
    peak_rss = 0
    peak_private = 0
    while not stop.is_set():
        now = time.perf_counter()
        scan_complete = True
        try:
            root = psutil.Process(pid)
            children = root.children(recursive=True)
            for proc in [root, *children]:
                known[(proc.pid, proc.create_time())] = proc
        except (psutil.Error, OSError):
            scan_complete = False

        rss = 0
        private = 0
        io_seen = {}
        for identity, proc in list(known.items()):
            try:
                memory = proc.memory_info()
                rss += int(memory.rss)
                private += int(getattr(memory, "private", 0))
                cpu = proc.cpu_times()
                cumulative_cpu[identity] = max(
                    cumulative_cpu.get(identity, 0.0),
                    float(cpu.user + cpu.system),
                )
                io = proc.io_counters()
                io_seen[identity] = (int(io.read_bytes), int(io.write_bytes))
            except psutil.NoSuchProcess:
                known.pop(identity, None)
            except (psutil.Error, OSError):
                scan_complete = False

        total_cpu = sum(cumulative_cpu.values())
        interval = now - previous_t
        busy_cores = 0.0 if not rows else max(
            0.0, (total_cpu - previous_cpu) / max(interval, 1e-6)
        )
        peak_busy_cores = max(peak_busy_cores, busy_cores)
        peak_rss = max(peak_rss, rss)
        peak_private = max(peak_private, private)
        commit = _commit_memory()
        rows.append({
            "elapsed_s": round(now - started, 3),
            "tree_rss_bytes": rss,
            "tree_private_commit_bytes": private,
            "cpu_busy_cores": round(busy_cores, 3),
            "system_available_bytes": int(__import__("psutil").virtual_memory().available),
            "commit_available_bytes": (commit or {}).get("available_bytes"),
            "process_scan_complete": scan_complete,
        })
        previous_cpu = total_cpu
        previous_t = now
        stop.wait(0.5)
    state.update({
        "sampled_cpu_seconds": sum(cumulative_cpu.values()),
        "peak_sampled_busy_cores": peak_busy_cores,
        "peak_tree_rss_bytes": peak_rss,
        "peak_tree_private_commit_bytes": peak_private,
    })


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--arc-config", type=Path, required=True)
    parser.add_argument("--line-faces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[1]
    voxel_module_dir = script_root / "05_reconstruction_volume"
    project_root = script_root.parent
    if str(voxel_module_dir) not in sys.path:
        sys.path.insert(0, str(voxel_module_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import only in the parent after spawn-safe module initialization.
    import trimesh
    from reconstruct_blocks_and_voxels import (
        filter_mesh_by_z_range,
        get_voxel_grid_meta,
        load_arc_config,
        load_full_line_face_data,
        merge_face_heights,
        save_active_voxels_pickle,
        voxelize_faces_sparse,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    resources: list[dict] = []
    monitor_state = {}
    stop = threading.Event()
    stage_started = time.perf_counter()
    monitor = threading.Thread(
        target=_monitor_process_tree,
        args=(os.getpid(), stop, resources, monitor_state),
        name="voxel-resource-monitor",
        daemon=True,
    )
    monitor.start()

    timings = {}
    tick = time.perf_counter()
    all_segments, face_heights_dict = load_full_line_face_data(args.line_faces)
    timings["line_face_read_s"] = time.perf_counter() - tick
    tick = time.perf_counter()
    merged_face_heights = merge_face_heights(face_heights_dict, strategy="min")
    timings["merge_face_heights_s"] = time.perf_counter() - tick
    del all_segments, face_heights_dict

    tick = time.perf_counter()
    mesh = trimesh.load_mesh(args.mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    timings["model_reload_s"] = time.perf_counter() - tick
    *_, z_min, z_max = load_arc_config(args.arc_config)
    tick = time.perf_counter()
    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
    timings["z_filter_s"] = time.perf_counter() - tick
    if mesh_filtered is None or not merged_face_heights:
        raise ValueError("没有可体素化的网格或识别面")
    if min(merged_face_heights) < 0 or max(merged_face_heights) >= len(mesh_filtered.faces):
        raise ValueError("来源面索引越界；线面结果与模型过滤参数不一致")
    grid_shape, origin, voxel_size = get_voxel_grid_meta(mesh_filtered, voxel_size=0.05)

    diagnostics = {}
    tick = time.perf_counter()
    active_voxels = voxelize_faces_sparse(
        mesh_filtered,
        merged_face_heights,
        grid_shape,
        origin,
        voxel_size,
        workers=args.workers,
        diagnostics=diagnostics,
    )
    timings["voxel_compute_s"] = time.perf_counter() - tick
    tick = time.perf_counter()
    save_active_voxels_pickle(
        active_voxels,
        grid_shape,
        origin,
        voxel_size,
        args.output / "voxels_optimized.pkl",
    )
    timings["voxel_write_s"] = time.perf_counter() - tick
    stop.set()
    monitor.join(timeout=5)
    stage_seconds = time.perf_counter() - stage_started

    if resources:
        with (args.output / "voxel_optimized_resources.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(resources[0]))
            writer.writeheader()
            writer.writerows(resources)

    peak_rss = int(monitor_state.get(
        "peak_tree_rss_bytes",
        max((row["tree_rss_bytes"] for row in resources), default=0),
    ))
    peak_private = int(monitor_state.get(
        "peak_tree_private_commit_bytes",
        max((row["tree_private_commit_bytes"] for row in resources), default=0),
    ))
    min_available = min(
        (row["system_available_bytes"] for row in resources),
        default=0,
    )
    commit_samples = [
        row["commit_available_bytes"]
        for row in resources
        if row["commit_available_bytes"] is not None
    ]
    platform_info = __import__("psutil").virtual_memory()
    sampled_cpu_seconds = float(monitor_state.get("sampled_cpu_seconds", 0.0))
    peak_busy_cores = float(monitor_state.get("peak_sampled_busy_cores", 0.0))
    resource_summary = {
        "wall_seconds": stage_seconds,
        "peak_tree_rss_bytes": peak_rss,
        "peak_tree_private_commit_bytes": peak_private,
        "min_system_available_bytes": min_available,
        "min_commit_available_bytes": min(commit_samples) if commit_samples else None,
        "sampled_cpu_seconds": sampled_cpu_seconds,
        "average_busy_cores": sampled_cpu_seconds / max(stage_seconds, 1e-9),
        "peak_sampled_busy_cores": peak_busy_cores,
        "total_ram_bytes": int(platform_info.total),
    }
    detail = {
        "timings": timings,
        "completed_faces": len(merged_face_heights),
        "active_voxel_count": len(active_voxels),
        "grid_shape": list(grid_shape),
        "voxel_size_m": voxel_size,
        "occupied_volume_m3": len(active_voxels) * voxel_size**3,
        "resource_summary": resource_summary,
        "diagnostics": diagnostics,
    }
    _write_json(args.output / "voxel_optimized_detail.json", detail)
    _write_json(args.output / "voxel_optimized_diagnostics.json", diagnostics)
    print(json.dumps(detail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
