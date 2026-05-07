"""Block and cluster plotting.

Plots structural blocks and cluster results from voxel data.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.tri import Triangulation
from collections import Counter
import pyvista as pv

# ========== 1. 加载稀疏点云数据 ==========
voxel_pkl = str(get_path("voxel_output", create_parent=True))
with open(voxel_pkl, "rb") as f:
    voxel_data = pickle.load(f)

voxel_indices = np.array(voxel_data["active_voxels"])
origin = np.array(voxel_data["origin"])
voxel_size = voxel_data["voxel_size"]
# 所有点云坐标 (N, 3)
points = (voxel_indices + 0.5) * voxel_size + origin

print(f"[✓] 加载点云完成，共 {len(points)} 个点")

# ========== 2. 抽稀采样 & DBSCAN 聚类 ==========
sample_ratio = 0.1  # 例如取 10% 的点进行聚类
sample_indices = np.random.choice(len(points), size=int(len(points) * sample_ratio), replace=False)
sampled_points = points[sample_indices]

dbscan = DBSCAN(eps=0.2, min_samples=5)
sample_labels = dbscan.fit_predict(sampled_points)

num_clusters = len(set(sample_labels)) - (1 if -1 in sample_labels else 0)
print(f"[✓] 聚类完成，簇数: {num_clusters}")

# ========== 3. 利用 KDTree 将聚类标签映射回所有原始点 ==========
tree = cKDTree(sampled_points)
_, idx = tree.query(points, k=1)
labels_full = sample_labels[idx]  # 每个原始点对应的聚类标签

# ========== 4. 统计有效聚类信息 ==========
voxel_volume = 0.05 ** 3  # 每个体素体积（单位：m³）
# 只统计有效聚类（不含噪点，标签 != -1）
label_counter = Counter(labels_full[labels_full != -1])

# 构造聚类信息列表，格式为 (聚类编号, 点数, 体积, 质心)
cluster_info = []
for label in label_counter:
    count = label_counter[label]
    vol = count * voxel_volume
    cluster_points = points[labels_full == label]
    centroid = cluster_points.mean(axis=0)
    cluster_info.append((label, count, vol, centroid))

# 按体积降序排序
sorted_clusters = sorted(cluster_info, key=lambda x: x[2], reverse=True)

print("\n[✓] 各有效聚类信息（按体积降序排序）：")
for i, (label, count, vol, centroid) in enumerate(sorted_clusters, start=1):
    print(f"  悬空体{i}: Label={label:<3} 点数={count:<6} 体积={vol:.4f} m³")

# ========== 5. 加载网格文件 ==========
mesh_path = str(get_path("segmented_mesh"))
mesh = pv.read(mesh_path)
# 提取网格顶点以及面数据
mesh_points = mesh.points  # (M, 3)
# 假定是三角形网格，faces 数组格式：[3, i1, i2, i3, 3, ...]
faces = mesh.faces.reshape(-1, 4)[:, 1:4]  # 取每个面前三个点的索引

# ========== 6. 定义函数：绘制块体 ==========
def plot_block(block_index, sorted_clusters, points, labels, mesh_points, faces):
    """
    绘制两个子图：
      左图：显示网格文件在 x-y 平面的投影（通过三角剖分绘制网格轮廓）。
      右图：显示选定聚类（块体）的点在 x-y 投影中，并在标题中显示块体序号和体积，
             标题使用新罗马字体，白色，字号16。
    
    参数：
      block_index: 目标块体的序号（1-indexed，1 表示体积最大的块体）。
      sorted_clusters: 按体积降序排序的聚类信息列表，格式为 (label, count, vol, centroid)。
      points: 所有点云坐标 (N, 3)。
      labels: 每个点对应的聚类标签（与 points 一一对应）。
      mesh_points: 网格文件的所有顶点坐标 (M, 3)。
      faces: 网格的三角面数据，形状为 (K, 3)。
    """
    # 检查序号合法性
    if block_index < 1 or block_index > len(sorted_clusters):
        print("块体序号超出范围！")
        return

    # 选取目标聚类信息
    target_cluster = sorted_clusters[block_index - 1]
    target_label, count, vol, centroid = target_cluster
    # 过滤出该聚类的点
    block_points = points[labels == target_label]

    # 在 x-y 平面投影（忽略 z 坐标）
    # 网格投影
    mesh_x = mesh_points[:, 0]
    mesh_y = mesh_points[:, 1]
    tri = Triangulation(mesh_x, mesh_y, triangles=faces)

    # 所有点云投影（可选，若需要叠加点云背景）
    all_x = points[:, 0]
    all_y = points[:, 1]
    # 目标块体的点
    block_x = block_points[:, 0]
    block_y = block_points[:, 1]

    # 创建包含两个子图的图形窗口
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 左子图：展示网格轮廓（x-y 投影）
    axes[0].triplot(tri, color='gray', linewidth=0.8)
    axes[0].scatter(all_x, all_y, s=1, c='lightgray', alpha=0.5)
    axes[0].set_title("网格文件 (x-y 投影)", fontfamily="Times New Roman", fontsize=16)
    axes[0].set_xlabel("X", fontfamily="Times New Roman", fontsize=14)
    axes[0].set_ylabel("Y", fontfamily="Times New Roman", fontsize=14)
    
    # 右子图：仅展示目标块体点云
    axes[1].scatter(block_x, block_y, s=10, c='red')
    # 设置标题采用新罗马、白色、字号16，且背景设为黑色以突出白色文本
    axes[1].set_title(f"(悬空体{block_index}) 体积: {vol:.4f} m³", 
                      fontfamily="Times New Roman", fontsize=16, color="white", pad=20)
    axes[1].set_xlabel("X", fontfamily="Times New Roman", fontsize=14, color="white")
    axes[1].set_ylabel("Y", fontfamily="Times New Roman", fontsize=14, color="white")
    axes[1].set_facecolor("black")
    
    plt.tight_layout()
    plt.show()

# ========== 7. 示例调用：绘制目标块体 ==========
# 例如：展示排序后第1个块体（体积最大的块体）
plot_block(1, sorted_clusters, points, labels_full, mesh_points, faces)
