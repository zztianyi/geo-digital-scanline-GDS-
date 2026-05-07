"""Point-cloud loading preview.

Loads text point-cloud data and displays it with Open3D.
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

# 1. 加载数据（保持原样）
def load_point_cloud_from_txt(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0
    normals = data[:, 7:10]
    return points, colors, normals

# 2. 创建点云对象（保持原样）
def create_point_cloud(points, colors, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

# 3. 转换为体素网格（保持原样）
def point_cloud_to_voxel(pcd, voxel_size=0.5):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    print(f"体素网格已生成，体素大小: {voxel_size}")
    return voxel_grid

# 4. 修改后的可视化函数
def visualize_with_custom_view(voxel_grid):
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Voxel Grid with Coordinate Axis")
    
    # 添加体素网格
    vis.add_geometry(voxel_grid)
    
    # # 添加坐标轴（尺寸设置为1.0，可根据需要调整）
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.0, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    
    # 获取视图控制器
    view_ctl = vis.get_view_control()
    
    # 设置初始视角参数
    view_ctl.set_front([-1, 0, 0])  # 朝向Y轴负方向（正对XZ平面）
    view_ctl.set_up([0, 0, 1])      # Z轴向上

    
    # 自动调整视角以显示完整内容
    vis.get_render_option().point_size = 1.0  # 可选：调整点云显示尺寸
    vis.run()
    vis.destroy_window()

# 主程序（保持原样）
if __name__ == "__main__":
    file_path = str(get_path("point_cloud_txt"))
    points, colors, normals = load_point_cloud_from_txt(file_path)
    pcd = create_point_cloud(points, colors, normals)
    voxel_size = 0.2
    voxel_grid = point_cloud_to_voxel(pcd, voxel_size)
    visualize_with_custom_view(voxel_grid)