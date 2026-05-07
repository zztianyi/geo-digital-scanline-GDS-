"""Arc configuration query helper.

Computes scanline plane locations from the arc configuration.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import json
import numpy as np

def load_arc_config(config_path):
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

def compute_all_slice_params(center, radius, angle_min, arc_length_range, step=0.01):
    """
    批量计算所有测平面的法向量、径向方向和方位角
    """
    positions = np.arange(0, arc_length_range[1], step)  # shape: (N,)
    angles = angle_min + positions / radius               # shape: (N,)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    
    radial_dirs = np.stack([cos_vals, sin_vals, np.zeros_like(cos_vals)], axis=1)  # shape: (N,3)
    radial_dirs /= np.linalg.norm(radial_dirs, axis=1, keepdims=True)

    normals = np.cross(radial_dirs, np.array([[0, 0, 1]]))  # shape: (N,3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    azimuths = (np.degrees(np.arctan2(radial_dirs[:, 0], radial_dirs[:, 1])) % 360).round(3)
    origins = np.repeat(center.reshape(1, 3), len(positions), axis=0)

    return positions, origins, normals, radial_dirs, azimuths

def find_nearest_plane(query_point, origins, normals, positions, radial_dirs, azimuths):
    """
    矢量方式计算 query_point 到所有平面的距离
    """
    vectors = query_point - origins              # shape: (N,3)
    distances = np.abs(np.einsum("ij,ij->i", vectors, normals))  # 点到平面距离，shape: (N,)

    idx = np.argmin(distances)
    return {
        "index": idx,
        "arc_pos": positions[idx],
        "distance": distances[idx],
        "origin": origins[idx],
        "normal": normals[idx],
        "radial_dir": radial_dirs[idx],
        "azimuth": azimuths[idx]
    }

def main():
    config_path = str(get_path("arc_config"))
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)

    query_point = np.array([-32.69, -26.52,0])

    # 一次性计算所有测面参数
    positions, origins, normals, radial_dirs, azimuths = compute_all_slice_params(
        center, radius, angle_min, arc_length_range, step=0.05
    )

    # 查找最近测面
    result = find_nearest_plane(query_point, origins, normals, positions, radial_dirs, azimuths)

    key = f"{result['arc_pos']:.2f}"
    print(f"最近测平面位置 {key}（弧长坐标）")
    print(f"距离查询点垂直距离为：{result['distance']:.3f} m")
    print(f"平面原点：{result['origin']}")
    print(f"法向量：{result['normal']}")
    print(f"径向方向：{result['radial_dir']}")
    print(f"方位角（顺时针相对于正北）：{result['azimuth']}°")

if __name__ == "__main__":
    main()
