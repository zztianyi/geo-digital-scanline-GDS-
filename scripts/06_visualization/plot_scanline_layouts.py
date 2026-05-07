"""Scanline layout plotting.

Creates 2D/3D schematic figures for curved and straight scanline layouts.
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
import trimesh
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import math
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, AuxTransformBox, AnnotationBbox,AnchoredOffsetbox
from matplotlib.transforms import Affine2D
from matplotlib.patches import FancyArrowPatch

# 设置英文字体
font_en = FontProperties(family='Arial', size=12)
font_en_13 = FontProperties(family='Arial', size=13)
# 设置中文字体（宋体）
font_zh = FontProperties(fname=str(get_path("font_zh")), size=12)
font_zh_13 = FontProperties(fname=str(get_path("font_zh")), size=13)
plt.rcParams['font.size'] = 12               # 设置默认字体大小
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
import open3d as o3d
print(o3d.__version__)

def plot_2d(projected_lines,plane_params):
    """
    在二维平面上绘制切剖面数据，根据指定平面投影，x轴对应 radial_dir，y轴对应 vertical_dir。
    方位角度azimuth_deg
    """
    #
    azimuth_deg = plane_params["azimuth"]
    plt.figure()
    for proj_line in projected_lines:
        x = proj_line[:, 0]
        y = proj_line[:, 1]
        plt.plot(x, y, 'b-')
    plt.xlabel("u (沿 radial_dir 方向)")
    plt.ylabel("v (沿 vertical_dir 方向)")
    plt.title("二维切剖面投影")
    plt.axis("equal")
    plt.show()

def plot_3d(lines_3d):
    """
    在三维空间中绘制切剖面数据。
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for line in lines_3d:
        x = line[:, 0]
        y = line[:, 1]
        z = line[:, 2]
        ax.plot(x, y, z, 'r-')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("三维切剖面展示")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def generate_arc_wall(center, radius, angle_min, angle_max, num_points=500):
    """生成弧形岩壁轮廓（用于拟合后的岩脚）"""
    angles = np.linspace(angle_min, angle_max, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.vstack((x, y)).T

def generate_straight_wall(x_start, x_end, y_level, num_points=500):
    """生成近似线性岩脚"""
    x = np.linspace(x_start, x_end, num_points)
    y = np.full_like(x, y_level)
    return np.vstack((x, y)).T

def generate_radial_lines(center, radius, angle_min, angle_max, num_lines=12):
    """生成圆弧式放射测线"""
    angles = np.linspace(angle_min, angle_max, num_lines)
    lines = []
    for angle in angles:
        x = [center[0], center[0] + radius * np.cos(angle)]
        y = [center[1], center[1] + radius * np.sin(angle)]
        lines.append((x, y))
    return lines

def generate_vertical_lines(x_start, x_end, y_top, y_bottom, num_lines=12):
    """生成垂直测线"""
    xs = np.linspace(x_start, x_end, num_lines)
    lines = []
    for x in xs:
        lines.append(([x, x], [y_bottom, y_top]))
    return lines

def plot_comparison():
    center = np.array([0, 0])
    radius = 10
    angle_min = -np.pi / 3
    angle_max = np.pi / 3
    y_top = 10

    arc_wall = generate_arc_wall(center, radius, angle_min, angle_max)
    straight_wall = generate_straight_wall(-radius * np.cos(angle_min), radius * np.cos(angle_max), 0)

    arc_lines = generate_radial_lines(center, radius * 1.5, angle_min, angle_max)
    straight_lines = generate_vertical_lines(-radius * np.cos(angle_min), radius * np.cos(angle_max), y_top, 0)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左侧：直线布设
    axs[0].plot(straight_wall[:, 0], straight_wall[:, 1], 'k-', label='岩脚轮廓')
    for x, y in straight_lines:
        axs[0].plot(x, y, 'b--')
    axs[0].set_title("直线布设测线")
    axs[0].set_aspect('equal')
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()

    # 右侧：圆弧布设
    axs[1].plot(arc_wall[:, 0], arc_wall[:, 1], 'k-', label='岩脚轮廓')
    for x, y in arc_lines:
        axs[1].plot(x, y, 'r--')
    axs[1].set_title("圆弧布设测线")
    axs[1].set_aspect('equal')
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()

    plt.tight_layout()
    plt.show()






