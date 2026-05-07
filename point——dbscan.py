import pickle
import numpy as np
from sklearn.cluster import DBSCAN
import pyvista as pv
from matplotlib import cm

# ===== 1. 鍔犺浇鐐逛簯 =====
voxel_pkl = r"data/private/raw_model\voxels_sparse_buchang.pkl"
with open(voxel_pkl, "rb") as f:
    voxel_data = pickle.load(f)

points = np.asarray(voxel_data)
print(f"[鉁揮 鍔犺浇鐐逛簯瀹屾垚锛屽叡 {len(points)} 涓偣")

# ===== 2. DBSCAN 鑱氱被 =====
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(points)

unique_labels = np.unique(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
print(f"[鉁揮 鑱氱被瀹屾垚锛岀皣鏁? {n_clusters}锛屽櫔澹扮偣: {(labels==-1).sum()}")

# ===== 3. 鍒嗛厤棰滆壊锛堢敤 matplotlib colormap锛?=====
colormap = cm.get_cmap("tab20", len(unique_labels))
colors = np.array([colormap(l if l >= 0 else 0)[:3] for l in labels]) * 255  # 杞负 RGB 0-255
colors[labels == -1] = [128, 128, 128]  # 鍣０鐐逛负鐏拌壊

# ===== 4. 鍒涘缓 PyVista 鐐逛簯瀵硅薄骞舵樉绀?=====
point_cloud = pv.PolyData(points)
point_cloud["Colors"] = colors.astype(np.uint8)

plotter = pv.Plotter()
plotter.add_points(point_cloud, scalars="Colors", rgb=True, render_points_as_spheres=True, point_size=5)
plotter.add_axes()
plotter.show_bounds(grid="front", location="outer", all_edges=True)
plotter.show(title="DBSCAN鑱氱被缁撴灉 - 涓夌淮鐐逛簯")

