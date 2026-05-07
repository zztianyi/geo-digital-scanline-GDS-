"""Arc baseline fitting and scanline setup.

Fits a curved baseline from point-cloud data and writes the arc configuration.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import numpy as np
import random
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import least_squares
import json
from matplotlib.font_manager import FontProperties

# 设置字体
font_en = FontProperties(family='Times New Roman', size=12)
font_en_13 = FontProperties(family='Times New Roman', size=13)
# 设置中文字体（宋体）
font_zh = FontProperties(fname=str(get_path("font_zh")), size=12)
font_zh_13 = FontProperties(fname=str(get_path("font_zh")), size=13)
plt.rcParams['font.size'] = 12               # 设置默认字体大小
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
# ------------------------- 工具函数 -------------------------

def uniform_downsample(points, voxel_size=0.02):
    """使用 Open3D 的体素滤波做均匀采样"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.hstack([points, np.zeros((len(points),1))])[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)[:, :2]  # 返回投影后的 2D 点


def filter_points_by_z(points, z_min=-np.inf, z_max=np.inf):
    """根据Z轴高度范围过滤点云"""
    return points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]

def fit_arc_least_squares(points_2d):
    """使用最小二乘法拟合部分圆弧"""
    def calc_residuals(params, points_2d):
        cx, cy, r = params
        distances = np.linalg.norm(points_2d - np.array([cx, cy]), axis=1)
        return distances - r
    
    # 初始猜测：圆心的初始位置是数据点的均值，半径是最大距离
    cx_init, cy_init = np.mean(points_2d, axis=0)
    r_init = np.max(np.linalg.norm(points_2d - np.array([cx_init, cy_init]), axis=1))
    
    # 使用最小二乘法拟合
    result = least_squares(calc_residuals, [cx_init, cy_init, r_init], args=(points_2d,))
    return result.x[:2], result.x[2]

def get_arc_angles(points_2d, center, radius):
    """计算给定圆心和半径下，数据点在圆上的起始和结束角度"""
    angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
    return np.min(angles), np.max(angles)

def plot_arc(center, radius, points_2d, angle_min, angle_max):
    """绘制拟合的部分圆弧和点云"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=1, alpha=0.3, label="Points")
    
    # 绘制拟合圆弧
    theta = np.linspace(angle_min, angle_max, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, '--r', label="Fitted Arc")
    
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Fitted Arc on Circle')
    ax.legend()
    plt.show()

def save_arc_to_config(center, radius, angle_min, angle_max, z_min, z_max, file_name="arc_config.json"):
    """将圆弧拟合的参数和Z轴范围保存到配置文件"""
    arc_data = {
        "center": center.tolist(),
        "radius": radius,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "z_range": [z_min, z_max]  # 添加Z轴范围
    }
    
    with open(file_name, "w") as f:
        json.dump(arc_data, f, indent=4)
    print(f"圆弧配置已保存到 {file_name}")


def load_arc_from_config(file_name="arc_config.json"):
    """从配置文件中加载圆弧参数，包括Z轴范围"""
    with open(file_name, "r") as f:
        arc_data = json.load(f)
    
    center = np.array(arc_data["center"])
    radius = arc_data["radius"]
    angle_min = arc_data["angle_min"]
    angle_max = arc_data["angle_max"]
    z_min, z_max = arc_data["z_range"]  # 读取Z轴范围

    return center, radius, angle_min, angle_max, z_min, z_max

# ------------------------- 绘图函数 -------------------------

def plot_2d_with_features(points_2d, z_range):
    """绘制包含部分圆弧拟合的2D点云"""

    
    center, radius = fit_arc_least_squares(points_2d)
    angle_min, angle_max = get_arc_angles(points_2d, center, radius)
    
    # 绘制拟合的部分圆弧
    plot_arc(center, radius, points_2d, angle_min, angle_max)
    


# ------------------------- 点云处理 -------------------------

def downsample_points(points_2d, sample_size=10000):
    """对 2D 点云数据进行下采样，随机选择一定数量的点"""
    if len(points_2d) > sample_size:
        indices = np.random.choice(len(points_2d), sample_size, replace=False)
        return points_2d[indices]
    return points_2d

def load_point_cloud_from_txt(file_path):
    """加载点云数据"""
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0
    normals = data[:, 7:10]
    return points, colors, normals

def create_point_cloud(points, colors, normals):
    """创建 Open3D 点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def statistical_outlier_removal_2d(points_2d, k=20, std_ratio=2.0):
    """统计学滤波"""
    if len(points_2d) < k:
        return points_2d, np.ones(len(points_2d), dtype=bool)

    nbrs = NearestNeighbors(n_neighbors=k).fit(points_2d)
    distances, _ = nbrs.kneighbors(points_2d)
    mean_dist_each_point = distances.mean(axis=1)
    
    global_mean = mean_dist_each_point.mean()
    global_std  = mean_dist_each_point.std()
    threshold   = global_mean + std_ratio * global_std

    mask = mean_dist_each_point < threshold
    return points_2d[mask], mask


def fit_line_least_squares(points_2d):
    """
    返回直线系数 (a, b) 使得 y = a*x + b
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    # 最小二乘解
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

def distance_to_line(points_2d, a, b):
    """
    计算点到直线 y = a*x + b 的垂直距离
    """
    x, y = points_2d[:, 0], points_2d[:, 1]
    return np.abs(a * x - y + b) / np.sqrt(a**2 + 1)

def distance_to_arc(points_2d, center, radius):
    """
    计算点到圆弧（圆）的径向距离差 |d - r|
    """
    return np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)

# ---------- 新增：对比绘图 ----------

def project_points_to_line(points_2d, a, b):
    """
    将点投影到直线 y = a*x + b 上，返回所有投影点
    """
    x0, y0 = points_2d[:, 0], points_2d[:, 1]
    denom = a**2 + 1
    x_proj = (x0 + a * (y0 - b)) / denom
    y_proj = a * x_proj + b
    return np.stack([x_proj, y_proj], axis=1)

def compare_line_vs_arc(points_2d):
    # —— 直线拟合 —— #
    a, b = fit_line_least_squares(points_2d)
    res_line = distance_to_line(points_2d, a, b)
    rmse_line = np.sqrt(np.mean(res_line**2))

    # —— 圆弧拟合 —— #
    center, r = fit_arc_least_squares(points_2d)
    res_arc = distance_to_arc(points_2d, center, r)
    rmse_arc = np.sqrt(np.mean(res_arc**2))
    ang_min, ang_max = get_arc_angles(points_2d, center, r)

    # —— 获取投影后的直线绘图范围 —— #
    proj_pts = project_points_to_line(points_2d, a, b)
    y_proj_min, y_proj_max = proj_pts[:, 1].min(), proj_pts[:, 1].max()
    x_proj_min = (y_proj_min - b) / a
    x_proj_max = (y_proj_max - b) / a

    # —— 创建并列子图（3个） —— #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)

    # —— 子图1：直线拟合 —— #
    ax1 = axes[0]
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=3, alpha=0.3)
    ax1.plot([x_proj_min, x_proj_max], [y_proj_min, y_proj_max], 'r-', lw=2, label='直线拟合')
    # ax1.set_title(f'直线拟合 (RMSE={rmse_line:.3e})', fontproperties=font_zh_13)
    ax1.set_xlabel('X(m)', fontproperties=font_en_13)
    ax1.set_ylabel('Y(m)', fontproperties=font_en_13)
    ax1.set_aspect('equal')
    ax1.legend(prop=font_zh_13)

    # —— 子图2：圆弧拟合 —— #
    ax2 = axes[1]
    ax2.scatter(points_2d[:, 0], points_2d[:, 1], s=3, alpha=0.3)
    theta = np.linspace(ang_min, ang_max, 300)
    arc_x = center[0] + r * np.cos(theta)
    arc_y = center[1] + r * np.sin(theta)
    ax2.plot(arc_x, arc_y, 'r--', lw=2, label='圆弧拟合')
    # ax2.set_title(f'圆弧拟合 (RMSE={rmse_arc:.3e})', fontproperties=font_zh_13)
    ax2.set_xlabel('X(m)', fontproperties=font_en_13)
    ax2.set_ylabel('Y(m)', fontproperties=font_en_13)
    ax2.set_aspect('equal')
    ax2.legend(prop=font_zh_13)

    # —— 子图3：残差直方图 —— #
    ax3 = axes[2]
    ax3.hist(res_line, bins=50, alpha=0.6, label=f'RMSE={rmse_line:.1f}')
    ax3.hist(res_arc, bins=50, alpha=0.6, label=f'RMSE={rmse_arc:.1f}')
    ax3.set_xlabel('残差(m)', fontproperties=font_zh_13)
    ax3.set_ylabel('频数(个)', fontproperties=font_zh_13)
    # ax3.set_title('残差分布直方图', fontproperties=font_zh_13)
    ax3.legend(prop=font_en_13)

    plt.tight_layout(w_pad=1.0)
    plt.show()
# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    file_path = str(get_path("point_cloud_txt"))
    
    # 1. 读取点云
    points, colors, normals = load_point_cloud_from_txt(file_path)
    
    # 2. 根据Z轴范围提取点云（示例：取中间1/3高度范围）
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    z_max = 2*(z_max - z_min)/5+z_min
    z_min = (z_max - z_min)/5+z_min
    filtered_points = filter_points_by_z(points, z_min, z_max)
    
    # 3. 获取 2D 投影 (X-Y)
    points_2d = filtered_points[:, :2]
    
    # 4. 下采样
    downsampled = downsample_points(points_2d, 20000)
    

    
    # 6. 拟合并可视化
    # plot_2d_with_features(cleaned_points, (z_min, z_max))

    cleaned_points = uniform_downsample(downsampled, voxel_size=1)
    compare_line_vs_arc(cleaned_points)
    # 7. 保存配置文件（添加Z轴范围）
    center, radius = fit_arc_least_squares(cleaned_points)
    angle_min, angle_max = get_arc_angles(cleaned_points, center, radius)
    save_arc_to_config(center, radius, angle_min, angle_max, z_min, z_max, file_name=str(get_path("arc_config")))