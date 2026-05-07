"""Point-cloud projection utilities.

Projects 3D point-cloud data into local 2D profile coordinates.
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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class PointCloudProcessor:
    def __init__(self, arc_config_file, point_cloud_file):
        self.center = self.radius = self.angle_min = self.angle_max = None
        self.load_arc_from_config(arc_config_file)
        self.points = self.load_point_cloud_from_txt(point_cloud_file)
    
    def load_arc_from_config(self, file_name="arc_config.json"):
        """从配置文件中加载圆弧的参数"""
        with open(file_name, "r") as f:
            arc_data = json.load(f)
        self.center = np.array(arc_data["center"])  # 这里是二维的center
        self.radius = arc_data["radius"]
        self.angle_min = arc_data["angle_min"]  # 弧度制
        self.angle_max = arc_data["angle_max"]

    def load_point_cloud_from_txt(self, file_path):
        """加载点云数据"""
        data = np.loadtxt(file_path)
        points = data[:, :3]
        return points
    
    def get_3d_line(self, angle_percent):
        """生成三维直线参数"""
        theta = (self.angle_min + (self.angle_max - self.angle_min) * angle_percent / 100)
        dir_2d = np.array([np.cos(theta), np.sin(theta)])
        point_2d = self.center[:2] + self.radius * dir_2d
        return np.append(point_2d, 0), np.append(dir_2d, 0)  # [x, y, 0], [dx, dy, 0]

    def extract_near_points(self, points, line_point, line_dir, max_dist=0.1):
        """提取三维邻近点"""
        dist = self.distance_point_to_line_3d(points, line_point, line_dir)
        return points[dist <= max_dist]

    def distance_point_to_line_3d(self, points, line_point, line_dir):
        """三维空间投影点到直线的距离"""
        vec = points - line_point
        # 将 vec 的 z 分量置为 0,这是计算在xy平面上的投影
        vec[:, 2] = 0  # 直接将所有点的 z 分量设为 0
        cross = np.cross(vec, line_dir)
        return np.linalg.norm(cross, axis=1) / np.linalg.norm(line_dir)

    def project_to_local_coords(self, proj_points, line_point, line_dir):
        """将投影点转换到平面局部坐标系"""
        u = line_dir / np.linalg.norm(line_dir)  # 沿直线方向
        v = np.array([0, 0, 1])  # 假设垂直方向
        w = np.cross(u, v)
        return np.dot(proj_points, np.vstack([u, v]).T)

    def project_points_to_plane(self, points, line_direction_3d, line_point_3d):
        """投影到特定平面"""
        # 确保 line_direction_3d 在xy平面内
        line_direction_2d = line_direction_3d[:2]  # 只取x和y分量
        # 计算与直线正交的法向量 (xy平面)
        normal = np.array([-line_direction_2d[1], line_direction_2d[0], 0])  # 垂直于line_direction_2d的向量
        # 归一化法向量
        normal = normal / np.linalg.norm(normal)
        vec_to_point = points - line_point_3d
        proj_length = np.dot(vec_to_point, normal)
        return points - proj_length[:, None] * normal

    def generate_arc_points(self, num_points=100):
        """生成圆弧上的二维点用于可视化"""
        angles = np.linspace(self.angle_min, self.angle_max, num_points)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        return np.column_stack((x, y))

    def downsample_points(self, points_2d, sample_size=10000):
        """对 2D 点云数据进行下采样，随机选择一定数量的点"""
        if len(points_2d) > sample_size:
            indices = np.random.choice(len(points_2d), sample_size, replace=False)
            return points_2d[indices]
        return points_2d

    def process_warp(self, angle_percent, max_dist=0.01):
        """处理并获取局部二维坐标"""
        line_3d_point, line_3d_dir = self.get_3d_line(angle_percent)
        arc_points = self.generate_arc_points()
        near_points = self.extract_near_points(self.points, line_3d_point, line_3d_dir, max_dist)
        
        # 投影到法平面并转换坐标
        proj_3d = self.project_points_to_plane(near_points, line_3d_dir, line_3d_point)
        
        # 平移校正
        proj_3d_ = proj_3d - line_3d_point
        proj_3d_[:, 2] -= np.min(self.points[:, 2])

        local_2d = self.project_to_local_coords(proj_3d_, line_3d_point, line_3d_dir)

        return local_2d
    


# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    # 1. 初始化 PointCloudProcessor
    processor = PointCloudProcessor(
        arc_config_file=str(get_path("arc_config")), 
        point_cloud_file=str(get_path("point_cloud_txt"))
    )
    
    # 2. 处理点云，输出局部坐标
    local_2d = processor.process_warp(angle_percent=32)

    # 3. 可视化结果
    plt.scatter(local_2d[:, 0], local_2d[:, 1], s=1)
    plt.axis('equal')
    plt.ticklabel_format(style='plain', axis='both')
    plt.show()

    # proj_3d = downsample_points(proj_3d, sample_size=1000)

    # # 3. 绘制三维投影点
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制投影后的三维点
    # ax.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2], c='g', s=5)

    # # 设置标题和标签
    # ax.set_title("3D Projection of Points")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    # plt.show()