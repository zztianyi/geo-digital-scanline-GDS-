"""Point-cluster analysis for voxel outputs.

Clusters voxel-derived points and exports centroid summaries.
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
import pyvista as pv
from matplotlib import cm
import laspy
from collections import Counter
import vtk  # 用于创建2D比例尺相关 actor
# ===== 1. 加载稀疏点云数据 =====
voxel_pkl = str(get_path("voxel_output", create_parent=True))
mesh_path = str(get_path("segmented_mesh"))
with open(voxel_pkl, "rb") as f:
    voxel_data = pickle.load(f)

voxel_indices = np.array(voxel_data["active_voxels"])
origin = np.array(voxel_data["origin"])
voxel_size = voxel_data["voxel_size"]
points = (voxel_indices + 0.5) * voxel_size + origin  # 所有点云坐标 (N, 3)

print(f"[✓] 加载点云完成，共 {len(points)} 个点")

# ===== 2. 抽稀：随机采样一部分点进行聚类 =====
sample_ratio = 0.1  # 抽稀比例，例如取 10%
sample_indices = np.random.choice(len(points), size=int(len(points) * sample_ratio), replace=False)
sampled_points = points[sample_indices]

# ===== 3. DBSCAN 聚类（在抽稀点上）=====
dbscan = DBSCAN(eps=0.2, min_samples=5)
sample_labels = dbscan.fit_predict(sampled_points)

num_clusters = len(set(sample_labels)) - (1 if -1 in sample_labels else 0)
print(f"[✓] 聚类完成，簇数: {num_clusters}")

# ===== 4. 使用 KDTree 将标签映射回所有原始点 =====
tree = cKDTree(sampled_points)
_, idx = tree.query(points, k=1)
labels_full = sample_labels[idx]  # 为每个原始点分配聚类标签

# ===== 5. 根据聚类统计体积，并筛选前10大簇 =====
voxel_volume = 0.05 ** 3  # 每个体素体积（立方米）

# 统计每个聚类编号的体素数量（不含 -1）
label_counter = Counter(labels_full[labels_full != -1])
top10 = label_counter.most_common(10)  # [(label, count), ...]

print("\n[✓] 前10大聚类体积：")
for i, (label, count) in enumerate(top10):
    volume = count * voxel_volume
    print(f"  Top-{i+1}: Label={label:<3}  点数={count:<6}  体积={volume:.4f} m³")

# 构建颜色映射：仅前10大聚类分配颜色，其余为蓝色
top10_labels = set(l for l, _ in top10)
top10_labels_sorted = sorted(top10_labels)
colormap = cm.get_cmap("turbo", len(top10_labels))
colors = np.full((len(labels_full), 4), fill_value=[220, 220, 255, 150], dtype=np.uint8)  # 浅蓝 + 透明度150


for idx_color, label in enumerate(top10_labels_sorted):
    mask = labels_full == label
    rgba = np.append((np.array(colormap(idx_color)[:3]) * 255).astype(np.uint8), 255)  # 加上不透明
    colors[mask] = rgba


# ===== 6. 创建 PyVista 点云对象并展示 =====
cloud = pv.PolyData(points)
cloud["Colors"] = colors.astype(np.uint8)

# ===== 7. 计算所有有效聚类的质心 =====
# 遍历有效聚类（排除噪点）提取质心
all_centroids = []
for label in sorted(label_counter):
    cluster_points = points[labels_full == label]
    centroid = cluster_points.mean(axis=0)
    all_centroids.append(centroid)
all_centroids = np.array(all_centroids)
print(f"[✓] 已提取有效聚类质心数量：{len(all_centroids)}")
def update_scale_bar(plotter, rect_actor, text_actor, scale_bar_height=10):
    """
    根据当前相机视角更新比例尺显示：
      1. 在窗口底部设定底边显示坐标（例如窗口宽度的 10% 到 35%，底边在窗口高度的 5%）。
      2. 使用当前相机焦点对应的 z 坐标，将这两个显示坐标转换为世界坐标（单位与网格一致，m）。
      3. 计算这两个点之间的水平世界距离，不受矩形高度影响。
      4. 更新矩形方框与文本标签，标签采用新罗马字体、字号为16，单位为 m。
    """
    ren = plotter.renderer
    width, height = plotter.window_size

    # 获取当前相机焦点的世界坐标，并转换为显示坐标，
    # 使用焦点所在的 z 值作为转换深度，确保转换结果稳定
    camera = ren.GetActiveCamera()
    focal_point = camera.GetFocalPoint()
    ren.SetWorldPoint(focal_point[0], focal_point[1], focal_point[2], 1.0)
    ren.WorldToDisplay()
    display_fp = ren.GetDisplayPoint()
    z_depth = display_fp[2]

    # 底边固定显示坐标（单位：像素）
    start_x = 0.1 * width
    end_x = 0.35 * width
    y_base = 0.05 * height

    # 将底边的两个显示坐标转换为世界坐标，使用焦点所在的 z 值
    ren.SetDisplayPoint(start_x, y_base, z_depth)
    ren.DisplayToWorld()
    wp1_raw = ren.GetWorldPoint()  # 返回 [x, y, z, w]
    wp1 = np.array(wp1_raw[:3]) / wp1_raw[3]

    ren.SetDisplayPoint(end_x, y_base, z_depth)
    ren.DisplayToWorld()
    wp2_raw = ren.GetWorldPoint()
    wp2 = np.array(wp2_raw[:3]) / wp2_raw[3]

    # 仅计算底边两个点间的世界距离（水平距离）
    world_distance = np.linalg.norm(wp2 - wp1)

    # 更新矩形方框的点坐标（构成一个横向矩形边框）
    # 点顺序：底左、底右、顶右、顶左、再闭合到底左
    polydata = rect_actor.GetMapper().GetInput()
    pts = polydata.GetPoints()
    pts.SetPoint(0, start_x, y_base, 0)                      # 底左
    pts.SetPoint(1, end_x,   y_base, 0)                      # 底右
    pts.SetPoint(2, end_x,   y_base + scale_bar_height, 0)   # 顶右
    pts.SetPoint(3, start_x, y_base + scale_bar_height, 0)   # 顶左
    pts.SetPoint(4, start_x, y_base, 0)                      # 闭合回底左
    pts.Modified()

    # 更新文本标签，显示水平距离（单位 m），设置新罗马字体及字号为16
    label = f"{world_distance:.2f} m"
    text_actor.SetInput(label)
    text_actor.GetTextProperty().SetFontFamilyToTimes()   # 设置字体为新罗马 (Times New Roman)
    text_actor.GetTextProperty().SetFontSize(16)            # 设置字号为 16
    # 将文本放置在矩形下方，根据需要可微调位置
    text_actor.SetPosition((start_x + end_x) / 2 - 20, y_base - 30)
    
    plotter.render()

# ===== 8. 定义函数，根据体积上下限提取聚类质心 =====
def extract_centroids_by_volume(points, labels, voxel_volume, volume_min, volume_max):
    """
    输入：
      points: 所有点云坐标 (N, 3)
      labels: 每个点对应的聚类标签，噪点标签为 -1
      voxel_volume: 每个体素的体积（m³）
      volume_min, volume_max: 需要筛选的体积范围（单位 m³）
    
    功能：
      1. 统计所有有效（排除噪点）聚类的体积，并打印每个聚类的体积信息；
      2. 打印所有有效聚类的总体积；
      3. 针对体积在 [volume_min, volume_max] 范围内的聚类，打印其体积信息及它们在总体积中的占比；
      4. 返回满足条件聚类的质心（和各自体积）。
    """
    valid_labels = labels[labels != -1]
    label_counter = Counter(valid_labels)
    total_valid_volume = sum(count * voxel_volume for count in label_counter.values())
    
    # 统计噪点的体积
    noise_count = (labels == -1).sum()
    noise_volume = noise_count * voxel_volume
    # 所有点云的总体积（包括噪点）
    total_volume = total_valid_volume + noise_volume

    print("\n所有有效聚类累积总体积 (排除噪点): {:.4f} m³".format(total_valid_volume))
    print("所有点云总体积 (包括噪点): {:.4f} m³".format(total_volume))
    if total_volume > 0:
        percentage = total_valid_volume / total_volume * 100
    else:
        percentage = 0
    print("有效聚类体积占比: {:.2f}%".format(percentage))
    
    selected_centroids = []
    selected_volumes = []
    selected_total_volume = 0.0
    # print("\n满足条件（体积范围: {:.4f} m³ ~ {:.4f} m³）的聚类：".format(volume_min, volume_max))
    for label in sorted(label_counter):
        count = label_counter[label]
        vol = count * voxel_volume
        if volume_min <= vol <= volume_max:
            cluster_points = points[labels == label]
            centroid = cluster_points.mean(axis=0)
            selected_centroids.append(centroid)
            selected_volumes.append(vol)
            selected_total_volume += vol
            print(f"  Label={label:<3}  点数={count:<6}  体积={vol:.4f} m³")
    
    percentage = (selected_total_volume / total_valid_volume * 100) if total_valid_volume > 0 else 0
    print("\n满足条件的聚类总体积: {:.4f} m³，占有效聚类总体积的 {:.2f}%".format(selected_total_volume, percentage))
    
    return np.array(selected_centroids), np.array(selected_volumes)

# 例：设置体积下限为 0.002 m³，上限为 0.01 m³（可根据实际情况调整）
volume_lower_bound = 0
volume_upper_bound = 0.1
selected_centroids, selected_volumes = extract_centroids_by_volume(points, labels_full, voxel_volume, volume_lower_bound, volume_upper_bound)

# ===== 9. 保存提取的聚类质心为 LAS 文件 =====
if selected_centroids.size > 0:
    print(f"[✓] 提取满足条件的聚类数量: {len(selected_centroids)}")
    # 统一颜色设置为红色
    centroid_colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint16), (len(selected_centroids), 1))
    
    # 创建 las 文件头
    centroid_header = laspy.LasHeader(point_format=3, version="1.2")
    centroid_header.offsets = np.min(selected_centroids, axis=0)
    centroid_header.scales = np.array([0.001, 0.001, 0.001])
    
    # 创建 las 数据对象
    centroid_las = laspy.LasData(centroid_header)
    centroid_las.x = selected_centroids[:, 0]
    centroid_las.y = selected_centroids[:, 1]
    centroid_las.z = selected_centroids[:, 2]
    centroid_las.red   = centroid_colors[:, 0]
    centroid_las.green = centroid_colors[:, 1]
    centroid_las.blue  = centroid_colors[:, 2]
    
    # 写入文件
    centroid_output_path = str(get_path("centroid_las_output", create_parent=True))
    centroid_las.write(centroid_output_path)
    print(f"[✓] 满足条件的聚类质心已保存为 LAS 文件：{centroid_output_path}")
else:
    print("[!] 没有找到满足条件的聚类。")

plotter = pv.Plotter()
plotter.add_points(cloud, scalars="Colors", rgba=True, render_points_as_spheres=True, point_size=5)

# plotter.show_bounds(grid="front", location="outer", all_edges=True)
plotter.hide_axes() 
# ===== 添加聚类注释标签 =====
from collections import Counter
label_counter = Counter(labels_full[labels_full != -1])

# 计算每个聚类的质心及体积，并整理到列表中：(聚类编号, 点数, 体积, 质心)
cluster_info = []
for label in label_counter:
    count = label_counter[label]
    vol = count * voxel_volume
    cluster_points = points[labels_full == label]
    centroid = cluster_points.mean(axis=0)
    cluster_info.append((label, count, vol, centroid))

# 按体积降序排序，并取前20个聚类
sorted_cluster_info = sorted(cluster_info, key=lambda x: x[2], reverse=True)[:15]

# 构造注释文本与对应位置列表
annotation_points = []
annotation_labels = []
for i, (label, count, vol, centroid) in enumerate(sorted_cluster_info, start=1):
    annotation_points.append(centroid)
    annotation_labels.append(f"(悬空体{i}) 体积: {vol:.4f} m³")

# 添加点标签，设置字体为新罗马、白色、字号16
plotter.add_point_labels(
    annotation_points,
    annotation_labels,
    font_size=20,
    font_family="times",
    text_color="white"
)



# ==== 添加动态比例尺（横着的白色矩形方框）====
# 创建矩形框的 polydata，包含5个点（闭合）
points = vtk.vtkPoints()
for _ in range(5):
    points.InsertNextPoint(0, 0, 0)
lines = vtk.vtkCellArray()
lines.InsertNextCell(5)  # 定义包含5个点的闭合线
for i in range(5):
    lines.InsertCellPoint(i)
rect_polydata = vtk.vtkPolyData()
rect_polydata.SetPoints(points)
rect_polydata.SetLines(lines)

rect_mapper = vtk.vtkPolyDataMapper2D()
rect_mapper.SetInputData(rect_polydata)

rect_actor = vtk.vtkActor2D()
rect_actor.SetMapper(rect_mapper)
rect_actor.GetProperty().SetColor(0, 0, 0)  # 白色
rect_actor.GetProperty().SetLineWidth(2)     # 边框宽度
plotter.renderer.AddActor2D(rect_actor)

# 创建文本标签显示世界距离（单位 m，新罗马字体）
text_actor = vtk.vtkTextActor()
text_actor.SetInput("0.00 m")
text_actor.GetTextProperty().SetFontSize(12)
text_actor.GetTextProperty().SetColor(0, 0, 0)  # 白色
# 初始位置，后续由 update_scale_bar 动态更新
text_actor.SetPosition(100, 20)
plotter.renderer.AddActor2D(text_actor)

def scale_bar_callback(caller, event):
    update_scale_bar(plotter, rect_actor, text_actor, scale_bar_height=10)

# 将观察者添加到当前活动相机上（当视角修改时触发更新）
camera = plotter.renderer.GetActiveCamera()
camera.AddObserver("ModifiedEvent", scale_bar_callback)

# 初始时更新一次比例尺显示
update_scale_bar(plotter, rect_actor, text_actor, scale_bar_height=10)


plotter.show(title="稀疏抽样 DBSCAN 聚类结果 (映射到全部点云)")