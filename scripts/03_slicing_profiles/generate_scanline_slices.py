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




def main():
    config_path = str(get_path("arc_config"))
    mesh_path = str(get_path("mesh_model")) 
    # mesh_path = str(get_path("segmented_mesh"))
    # mesh_path = str(get_path("segmented_mesh"))
    mesh_path = str(get_path("segmented_mesh"))
    
    # 加载配置
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)
    
    # 切剖面位置序列
    slice_positions = np.arange(59.45, 72.65, 0.05)

    # 加载并过滤网格
    mesh = trimesh.load_mesh(mesh_path)
    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)

    # 存储所有切剖面数据
    slices_data = {}

    for pos in slice_positions:
        plane_params = compute_slice_plane(center, radius, angle_min, arc_length_range, pos)
        origin = plane_params["origin"]
        normal = plane_params["normal"]
        

        # 使用带面索引的切割函数
        lines_3d, face_ids_1d = slice_and_check_faces(mesh_filtered, origin, normal)

        # 保存为字符串键，同时单独保存方位角
        key = f"{pos:.2f}"
        slices_data[key] = {
            "plane_params": plane_params,
            "slicing": {
                "lines_3d": lines_3d,
                "face_ids": face_ids_1d
            }
        }

        print(f"切剖面位置 {key}（弧长坐标），方位角 {plane_params['azimuth']}°，线段数量 {len(lines_3d)}")

    # 清理内存
    del mesh, mesh_filtered

    # 保存数据
    output_path = str(get_path("slice_output", create_parent=True))
    with open(output_path, "wb") as f:
        pickle.dump(slices_data, f)

    print("全部切剖面数据已保存到", output_path)
    
if __name__ == "__main__":
    main()
