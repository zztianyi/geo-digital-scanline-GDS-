"""Digital scanline grid preview.

Visualizes an arc-based scanline grid over a point-cloud sample.
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
import open3d as o3d

def load_arc_from_config(file_name="arc_config.json"):
    """从配置文件中加载圆弧的参数"""
    with open(file_name, "r") as f:
        arc_data = json.load(f)
    
    # 读取圆心、半径和角度
    center = np.array(arc_data["center"])
    radius = arc_data["radius"]
    angle_min = arc_data["angle_min"]
    angle_max = arc_data["angle_max"]
    
    return center, radius, angle_min, angle_max


def load_point_cloud_from_txt(file_path):
    """加载点云数据"""
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0  # 正常化颜色值
    return points, colors


def create_3d_grid_on_arc(center, radius, angle_min, angle_max, points, colors, num_latitude_lines, num_longitude_lines):
    """在给定圆弧上生成网格，并在3D空间中绘制出来"""
    # 获取点云的z轴范围
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    
    # 生成纬线：每条纬线对应不同的z值，并平行于xy平面
    z_vals = np.linspace(z_min, z_max, num_latitude_lines)  # 纬线的不同高度
    latitudes = []
    
    for z in z_vals:
        angles = np.linspace(angle_min, angle_max, num_longitude_lines)  # 均匀分布的经线角度
        x_vals = center[0] + radius * np.cos(angles)
        y_vals = center[1] + radius * np.sin(angles)
        latitudes.append(np.column_stack([x_vals, y_vals, np.full_like(x_vals, z)]))  # 保存每个纬线的点

    # 生成经线：这些是平行于z轴的线
    longitudes = []
    for angle in np.linspace(angle_min, angle_max, num_longitude_lines):
        x_vals = center[0] + radius * np.cos(angle)
        y_vals = center[1] + radius * np.sin(angle)
        z_vals = np.linspace(z_min, z_max, num_latitude_lines)  # 高度从最小到最大z值
        longitudes.append(np.column_stack([np.full_like(z_vals, x_vals), np.full_like(z_vals, y_vals), z_vals]))  # 保存每条经线的点

    # 使用Open3D绘制3D网格和圆弧
    # 创建点云对象
    pcd_points = o3d.geometry.PointCloud()
    pcd_points.points = o3d.utility.Vector3dVector(points)
    pcd_points.colors = o3d.utility.Vector3dVector(colors)  # 使用导入的颜色

    # 创建 LineSet 对象，添加所有纬线
    lat_line_sets = []
    for latitude in latitudes:
        lat_line_lines = o3d.geometry.LineSet()
        lat_line_lines.points = o3d.utility.Vector3dVector(latitude)
        lat_line_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(latitude) - 1)])
        lat_line_lines.paint_uniform_color([1, 0, 0])  # 红色显示纬线
        lat_line_sets.append(lat_line_lines)

    # 创建 LineSet 对象，添加所有经线
    lon_line_sets = []
    for longitude in longitudes:
        lon_line_lines = o3d.geometry.LineSet()
        lon_line_lines.points = o3d.utility.Vector3dVector(longitude)
        lon_line_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(longitude) - 1)])
        lon_line_lines.paint_uniform_color([0, 1, 0])  # 绿色显示经线
        lon_line_sets.append(lon_line_lines)

    # 可视化所有对象
    o3d.visualization.draw_geometries([pcd_points] + lat_line_sets + lon_line_sets)

def save_grid_equations_to_config(center, radius, angle_min, angle_max, file_name="grid_equations_config.json"):
    """保存生成经纬线的方程到配置文件"""
    grid_data = {
        "center": center.tolist(),
        "radius": radius,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "latitude_equation": "x = center[0] + radius * np.cos(angle), y = center[1] + radius * np.sin(angle), z = z_value",
        "longitude_equation": "x = center[0] + radius * np.cos(angle), y = center[1] + radius * np.sin(angle), z = z_value"
    }
    
    with open(file_name, "w") as f:
        json.dump(grid_data, f, indent=4)

# ------------------------- 主程序 -------------------------

if __name__ == "__main__":
    # 1. 读取点云数据
    file_path = str(get_path("point_cloud_txt"))
    points, colors = load_point_cloud_from_txt(file_path)
    
    # 2. 加载圆弧配置
    center, radius, angle_min, angle_max = load_arc_from_config(file_name=str(get_path("arc_config")))
    
    # 3. 在3D空间绘制网格和圆弧
    num_latitude_lines = 10  # 纬线的数量
    num_longitude_lines = 12  # 经线的数量
    create_3d_grid_on_arc(center, radius, angle_min, angle_max, points, colors, num_latitude_lines, num_longitude_lines)

