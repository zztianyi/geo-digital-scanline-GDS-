"""Vertical profile analysis prototype.

Runs a simple vertical projection and high-elevation center analysis.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import numpy as np
import open3d as o3d
import random
import matplotlib.pyplot as plt

# 1. 加载数据
def load_point_cloud_from_txt(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0
    normals = data[:, 7:10]
    return points, colors, normals

# 2. 下采样点云
def downsample_points(points, sample_size=100000):
    """
    对点云数据进行随机抽样
    :param points: 原始点云数据
    :param sample_size: 抽样点数
    :return: 抽样后的点云数据
    """
    if len(points) > sample_size:
        sampled_indices = random.sample(range(len(points)), sample_size)
        points = points[sampled_indices]
    return points

# 3. 计算圆心
def calculate_center_by_z_quantile(points, quantile=0.95):
    """
    根据 z 值的分位数计算圆心
    :param points: 三维点云数据
    :param quantile: z 值分位数（默认为 95%）
    :return: 圆心的三维坐标, 高 z 值的点集
    """
    z_values = points[:, 2]  # 提取 z 值
    threshold = np.quantile(z_values, quantile)  # 计算 95% 分位数阈值
    high_z_points = points[z_values >= threshold]  # 筛选 z 值大于阈值的点集
    center = np.mean(high_z_points, axis=0)  # 计算高 z 值点集的中心点
    print(f"圆心位置（通过 z 值 95% 分位数确定）: {center}")
    return center, high_z_points

# 4. 可视化投影点和高 z 值点集
def plot_projected_points_with_high_z(points, high_z_points, center, padding=0.1):
    """
    绘制投影点、高 z 值点集和圆心位置
    :param points: 三维点云数据
    :param high_z_points: z 值大于 95% 分位数的点集
    :param center: 圆心的三维坐标
    :param padding: 图像范围边距比例
    """
    # 投影到 XY 平面
    xy_projection = points[:, :2]  # 所有点的投影
    high_z_projection = high_z_points[:, :2]  # 高 z 值点的投影
    center_xy = center[:2]  # 圆心的 XY 坐标

    # 设置显示范围
    x_min, x_max = np.min(xy_projection[:, 0]), np.max(xy_projection[:, 0])
    y_min, y_max = np.min(xy_projection[:, 1]), np.max(xy_projection[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    # 创建绘图
    plt.figure(figsize=(8, 8))
    plt.scatter(xy_projection[:, 0], xy_projection[:, 1], s=1, label="Sampled Points", alpha=0.5, color="gray")
    plt.scatter(high_z_projection[:, 0], high_z_projection[:, 1], s=1, label="High Z Points", alpha=0.8, color="red")
    plt.scatter(center_xy[0], center_xy[1], color="blue", label="Center", s=100)

    # 设置图像范围和标题
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("equal")
    plt.title("Projected Points with High Z Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# 主函数
if __name__ == "__main__":
    file_path = str(get_path("point_cloud_txt"))

    # 加载点云数据
    points, colors, normals = load_point_cloud_from_txt(file_path)

    # 下采样点云
    sampled_points = downsample_points(points, sample_size=100000)

    # 计算圆心和高 z 值点集
    center, high_z_points = calculate_center_by_z_quantile(sampled_points, quantile=0.15)

    # 可视化投影点和高 z 值点集
    plot_projected_points_with_high_z(sampled_points, high_z_points, center)
