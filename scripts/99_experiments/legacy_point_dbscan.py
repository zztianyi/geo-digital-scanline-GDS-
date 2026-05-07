"""Legacy DBSCAN point clustering prototype.

Older compact variant of point clustering kept for reference.
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
import pyvista as pv
from matplotlib import cm

# ===== 1. 加载点云 =====
voxel_pkl = str(get_path("voxel_output", create_parent=True))
with open(voxel_pkl, "rb") as f:
    voxel_data = pickle.load(f)

points = np.asarray(voxel_data)
print(f"[✓] 加载点云完成，共 {len(points)} 个点")

# ===== 2. DBSCAN 聚类 =====
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(points)

unique_labels = np.unique(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
print(f"[✓] 聚类完成，簇数: {n_clusters}，噪声点: {(labels==-1).sum()}")

# ===== 3. 分配颜色（用 matplotlib colormap） =====
colormap = cm.get_cmap("tab20", len(unique_labels))
colors = np.array([colormap(l if l >= 0 else 0)[:3] for l in labels]) * 255  # 转为 RGB 0-255
colors[labels == -1] = [128, 128, 128]  # 噪声点为灰色

# ===== 4. 创建 PyVista 点云对象并显示 =====
point_cloud = pv.PolyData(points)
point_cloud["Colors"] = colors.astype(np.uint8)

plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars="Colors", rgb=True, render_points_as_spheres=True, point_size=5)
plotter.add_axes()
plotter.show_bounds(grid="front", location="outer", all_edges=True)
plotter.show(title="DBSCAN聚类结果 - 三维点云")
