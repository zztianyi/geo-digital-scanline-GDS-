"""Batch scanline slicing.

Generates mesh-section line and face outputs along the digital scanline set.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import json
import trimesh
import pickle
import numpy as np

def load_arc_config(config_path):
    """
    加载配置文件，返回圆心、半径、角度范围、弧长范围及 Z 轴范围。
    配置中角度均为弧度，弧长范围计算公式：radius * (angle_max - angle_min)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    center_2d = np.array(config["center"], dtype=float)
    center_3d = np.append(center_2d, 0.0)
    radius = float(config["radius"])
    angle_min = config["angle_min"]
    angle_max = config["angle_max"]
    arc_length = radius * (angle_max - angle_min)
    arc_length_range = [0, arc_length]
    z_min, z_max = config["z_range"]
    return center_3d, radius, angle_min, angle_max, arc_length_range, z_min, z_max

def compute_slice_plane(center, radius, angle_min, arc_length_range, slice_position):
    """
    根据切剖面位置（弧长坐标）计算切剖面参数：
      slice_angle = angle_min + slice_position / radius
    返回一个字典，包括平面原点、法向量、径向方向、计算得到的切剖面角度，
    以及 radial_dir 相对于正北方向（[0,1]）顺时针的方位角（角度值，保留三位小数）。
    """
    if slice_position is None:
        slice_position = arc_length_range[1] / 2
    slice_angle = angle_min + slice_position / radius
    radial_dir = np.array([np.cos(slice_angle), np.sin(slice_angle), 0.0])
    radial_dir /= np.linalg.norm(radial_dir)
    normal = np.cross(radial_dir, [0, 0, 1])
    normal /= np.linalg.norm(normal)
    vertical_dir = np.array([0, 0, 1])
    # 计算 radial_dir 与正北方向的夹角，顺时针测量：
    # 设正北方向为 [0,1]，使用 np.arctan2(radial_dir[0], radial_dir[1])
    azimuth_rad = np.arctan2(radial_dir[0], radial_dir[1])
    azimuth_deg = np.degrees(azimuth_rad) % 360
    azimuth_deg = round(azimuth_deg, 3)
    return {
        "origin": center,
        "normal": normal,
        "radial_dir": radial_dir,
        "slice_angle": slice_angle,
        "vertical_dir": vertical_dir,
        "azimuth": azimuth_deg  # 单独保存的方位角（顺时针测量）
    }

def filter_mesh_by_z_range(mesh, z_min, z_max):
    """
    根据 Z 轴范围过滤网格，仅保留符合条件的三角面。
    """
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if len(submeshes) > 0 else None

def slice_and_check_faces(mesh, origin, normal):
    """
    使用 mesh_plane 并返回 (lines_3d, face_ids_1d)，用于后续处理
    """
    lines_3d, face_ids_1d = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=normal,
        plane_origin=origin,
        return_faces=True
    )
    return lines_3d, face_ids_1d




def main(workers=4):
    import os
    from tempfile import TemporaryDirectory
    from parallel_slice_engine import ParallelSliceEngine, prepare_mesh

    # Spawned workers inherit these limits before importing NumPy. Four workers
    # are the lower-memory near-tie in the real 1/4/8/12 sweep; CLI can override.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    config_path = str(get_path("arc_config"))
    mesh_path = str(get_path("mesh_model")) 
    # mesh_path = str(get_path("segmented_mesh"))
    # mesh_path = str(get_path("segmented_mesh"))
    mesh_path = str(get_path("segmented_mesh"))
    
    # 加载配置
    with open(config_path, 'r') as f:
        arc_config = json.load(f)
    
    # 切剖面位置序列
    slice_positions = np.arange(59.45, 72.65, 0.05)

    # 存储所有切剖面数据
    slices_data = {}

    with TemporaryDirectory(prefix="gds-slices-") as temporary_dir:
        metadata = prepare_mesh(mesh_path, config_path, _GDSPath(temporary_dir) / "mesh_cache")
        specs = ({"axis": "vertical", "position": float(pos)} for pos in slice_positions)
        with ParallelSliceEngine(metadata["cache_dir"], arc_config, workers=workers) as engine:
            for result in engine.iter_slices(specs):
                plane_params = result["plane_params"]
                lines_3d = result["lines_3d"]
                key = f"{result['position']:.2f}"
                slices_data[key] = {
                    "plane_params": plane_params,
                    "slicing": {
                        "lines_3d": lines_3d,
                        "face_ids": result["face_ids"]
                    }
                }
                print(f"切剖面位置 {key}（弧长坐标），方位角 {plane_params['azimuth']}°，线段数量 {len(lines_3d)}")
    slices_data = {f"{pos:.2f}": slices_data[f"{pos:.2f}"] for pos in slice_positions}

    # 保存数据
    output_path = str(get_path("slice_output", create_parent=True))
    with open(output_path, "wb") as f:
        pickle.dump(slices_data, f)

    print("全部切剖面数据已保存到", output_path)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Persistent slicing workers (default: 4)")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 61:
        parser.error("--workers must be between 1 and 61")
    main(workers=args.workers)
