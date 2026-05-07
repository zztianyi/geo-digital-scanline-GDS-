"""Slice visualization and inspection.

Displays generated scanline slices, grouped features, and profile outputs.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import json
import trimesh
import pickle
import numpy as np
import pyvista as pv
import vtk

# 允许 PyVista 加载空网格
pv.global_theme.allow_empty_mesh = True

def load_arc_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    center_2d = np.array(config["center"], dtype=float)
    center_3d = np.append(center_2d, 0.0)
    radius = float(config["radius"])
    angle_min = config["angle_min"]
    angle_max = config["angle_max"]
    arc_length = radius * (angle_max - angle_min)
    arc_length_range = [0, arc_length]
    z_min, z_max = config["z_range"]
    return center_3d, radius, angle_min, angle_max, arc_length_range, z_min, z_max

def filter_mesh_by_z_range(mesh, z_min, z_max):
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if len(submeshes) > 0 else None

def load_slices_data(pickle_file):
    with open(pickle_file, "rb") as f:
        slices_data = pickle.load(f)
    return slices_data

def add_slice_lines_to_plotter(plotter, slices_data, lower=40.0, upper=60.0, step=0.05):
    """
    每隔 step 提取一个 key（float 类型），添加对应的切剖面线段到 plotter 中。
    """
    points = []
    lines = []

    # 构造目标弧长序列（保留两位小数避免浮点误差）
    target_keys = {round(lower + i * step, 8) for i in range(int((upper - lower) / step) + 1)}

    for key, data in slices_data.items():
        try:
            pos = float(key)
        except ValueError:
            continue
        if round(pos, 8) not in target_keys:
            continue
        lines_3d = data["slicing"]["lines_3d"]
        for segment in lines_3d:
            segment = np.asarray(segment)
            if segment.shape[0] < 2:
                continue
            start = segment[0]
            end = segment[1]
            idx_start = len(points)
            idx_end = idx_start + 1
            points.extend([start, end])
            lines.append([2, idx_start, idx_end])

    if points and lines:
        pdata = pv.PolyData()
        pdata.points = np.array(points)
        pdata.lines = np.array(lines, dtype=np.int64)
        plotter.add_mesh(pdata, color="red", line_width=2, label="切剖面线段")

    return plotter
def bind_on_key_press(plotter):
    def on_key_press():
        save_path = str(get_path("figure_output_dir") / "figure.png")
        plotter.screenshot(
            filename=save_path,
            scale=2.0,
            transparent_background=True,
            return_img=False
        )
        print(f"[✓] 当前视图已保存为：{save_path}")
    return on_key_press

def multi_panel_display(mesh_pv, slices_data, step_values=(None, 0.5, 0.1, 0.05)):
    """
    创建一个 1×4 子图，依次显示：
    1. 原网格
    2. 添加 step=0.5 的剖面线
    3. 添加 step=0.1 的剖面线
    4. 添加 step=0.05 的剖面线
    """
    # titles = ["原网格", "剖面 step=0.5", "剖面 step=0.1", "剖面 step=0.05"]

    plotter = pv.Plotter(shape=(1, 4), window_size=(2400, 600))
    plotter.enable_parallel_projection()
    plotter.link_views()  # 联动视角

    for i, step in enumerate(step_values):
        if step is not None:
            plotter.subplot(0, i)
            add_slice_lines_to_plotter(plotter, slices_data, lower=80.7, upper=92.7, step=step)
        else:
            plotter.subplot(0, i)
            plotter.add_mesh(mesh_pv, color="lightgray", opacity=1, show_edges=False)
        # plotter.add_text(titles[i], position="upper_edge", font_size=12)
    # 添加按键截图
    plotter.add_key_event("s", bind_on_key_press(plotter))

    plotter.show(title="", auto_close=False)

def add_slice_lines_to_plotter(plotter, slices_data, lower, upper, step):
    points = []
    lines = []
    keys = {round(lower + i * step, 8) for i in range(int((upper - lower) / step) + 1)}

    for key, data in slices_data.items():
        try:
            pos = float(key)
        except ValueError:
            continue
        if round(pos, 8) not in keys:
            continue
        for segment in data["slicing"]["lines_3d"]:
            segment = np.asarray(segment)
            if segment.shape[0] < 2:
                continue
            idx_start = len(points)
            idx_end = idx_start + 1
            points.extend([segment[0], segment[1]])
            lines.append([2, idx_start, idx_end])

    if points and lines:
        pdata = pv.PolyData()
        pdata.points = np.array(points)
        pdata.lines = np.array(lines, dtype=np.int64)
        plotter.add_mesh(pdata, color="red", line_width=2)
    return plotter


def multi_panel_display_composite(mesh_pv, slices_data, group_pkl_path,
                                   lower=0, upper=140, step=0.05):
    plotter = pv.Plotter(shape=(1, 4), window_size=(2400, 600))
    plotter.enable_parallel_projection()
    plotter.link_views()

    # 1. 原始网格
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_pv, color="lightgray", opacity=1, show_edges=False)

    # 2. 添加 step=0.05 剖面线
    plotter.subplot(0, 1)
    add_slice_lines_to_plotter(plotter, slices_data, lower, upper, step=step)

    # 3. 显示 red_group 坐标
    plotter.subplot(0, 2)
    with open(group_pkl_path, "rb") as f:
        group_data = pickle.load(f)
    for group_segments, _, _ in group_data:
        for seg in group_segments:
            p1 = np.array(seg[4])
            p2 = np.array(seg[5])
            plotter.add_lines(np.array([p1, p2]), color="red", width=2)

    # 4. 显示 mesh_pv 中的对应三角面
    plotter.subplot(0, 3)
    group_faces = set()
    for _, _, face_indices in group_data:
        group_faces.update(face_indices)

    if group_faces:
        face_indices = np.array(sorted(group_faces), dtype=int)
        mesh_subset = mesh_pv.extract_cells(face_indices)
        plotter.add_mesh(mesh_subset, color="red", show_edges=False, opacity=1)
    plotter.add_key_event("s", bind_on_key_press(plotter))
    plotter.show(title="四图对比：原网格/剖面线/红线段/目标面", auto_close=False)

def load_full_segments(pickle_path):
    """
    从 pickle 文件中加载 all_segments。
    返回一个 N×3×2 的数组，表示 N 条线段的两个端点坐标。
    """
    with open(pickle_path, "rb") as f:
        all_segments, _ = pickle.load(f)
    # 只取前两项 p1, p2，忽略 height
    lines = np.array([[seg[0], seg[1]] for seg in all_segments], dtype=float)
    return lines  # shape = (N, 2, 3)

def add_full_segments_to_plotter(plotter, lines, color="blue", line_width=2):
    """
    将多条线段整合为一个 PolyData 对象并添加到 plotter。
    lines: shape (N,2,3)
    """
    # 构建 points 和 lines 数组
    pts = lines.reshape(-1, 3)  # (N*2, 3)
    # 每条线段对应的索引
    n = lines.shape[0]
    connectivity = np.empty((n, 3), dtype=np.int64)
    # VTK line 格式： [2, idx0, idx1]
    # 我们先构造二维 (N,2)，然后在赋给 PolyData.lines 时自动铺平
    for i in range(n):
        connectivity[i, 0] = 2
        connectivity[i, 1] = 2*i
        connectivity[i, 2] = 2*i + 1

    pdata = pv.PolyData()
    pdata.points = pts
    pdata.lines = connectivity
    plotter.add_mesh(pdata, color=color, line_width=line_width, label="full_segments")
    return plotter
def add_dynamic_scale_bar(plotter,
                          length=50.0,
                          bar_height=10,
                          display_margin_ratio=0.05,
                          font_size=12):
    """
    在屏幕左下角添加一个动态比例尺（50 m），距边缘约 display_margin_ratio 的窗口宽度，
    颜色固定为黑色。
    """
    ren = plotter.renderer
    win_w, win_h = plotter.window_size

    # ==== 构造空矩形 polydata ====
    pts = vtk.vtkPoints()
    for _ in range(5):
        pts.InsertNextPoint(0, 0, 0)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(5)
    for i in range(5):
        lines.InsertCellPoint(i)
    rect_pd = vtk.vtkPolyData()
    rect_pd.SetPoints(pts)
    rect_pd.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(rect_pd)
    bar_actor = vtk.vtkActor2D()
    bar_actor.SetMapper(mapper)
    bar_actor.GetProperty().SetColor(0, 0, 0)  # 黑色
    bar_actor.GetProperty().SetLineWidth(2)
    ren.AddActor2D(bar_actor)

    # ==== 文本 Actor ====
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(f"0 {int(length)} m")
    tp = text_actor.GetTextProperty()
    tp.SetFontSize(font_size)
    tp.SetFontFamilyToTimes()
    tp.SetColor(0, 0, 0)  # 黑色
    ren.AddActor2D(text_actor)

    # ==== 更新函数 ====
    def update_scale_bar(caller, event):
        # 1. 计算屏幕像素边距
        mx = win_w * display_margin_ratio
        my = win_h * display_margin_ratio

        # 2. 计算 world → display，得到“length 米”对应的像素宽度
        # 任意一个世界点 p0 用于测量比例尺缩放，不影响位置
        bounds = plotter.bounds
        p0 = [bounds[0], bounds[2], bounds[4]]  # world 最小点
        p1 = [p0[0] + length, p0[1], p0[2]]

        ren.SetWorldPoint(*p0, 1.0)
        ren.WorldToDisplay()
        d0 = ren.GetDisplayPoint()

        ren.SetWorldPoint(*p1, 1.0)
        ren.WorldToDisplay()
        d1 = ren.GetDisplayPoint()

        w = abs(d1[0] - d0[0])  # 像素宽度

        # 3. 更新矩形坐标（全部用 display 坐标）
        pts.SetPoint(0, mx,     my,      0)
        pts.SetPoint(1, mx + w, my,      0)
        pts.SetPoint(2, mx + w, my + bar_height, 0)
        pts.SetPoint(3, mx,     my + bar_height, 0)
        pts.SetPoint(4, mx,     my,      0)
        pts.Modified()

        # 4. 更新文字位置和内容
        text_actor.SetPosition(mx, my + bar_height + 2)
        text_actor.SetInput(f"0 {int(length)} m")

    # 绑定相机更新事件，保证视角变化时重绘
    cam = ren.GetActiveCamera()
    cam.AddObserver("ModifiedEvent", update_scale_bar)
    # 初始绘制一次
    update_scale_bar(None, None)

    return plotter

def main():
    # ===== 路径设置 =====
    config_path = str(get_path("arc_config"))
    mesh_path = str(get_path("segmented_mesh"))
    # mesh_path = str(get_path("segmented_mesh"))str(get_path("segmented_mesh"))str(get_path("segmented_mesh"))
    mesh_path = str(get_path("segmented_mesh"))
    # mesh_path = str(get_path("segmented_mesh"))
    pickle_file = str(get_path("slice_output", create_parent=True))
    pickle_file = str(get_path("group_centroid_output", create_parent=True))
    # pickle_file = str(get_path("slice_output", create_parent=True))
    # pickle_file = str(get_path("slice_output", create_parent=True))
    pickle_file = str(get_path("slice_output", create_parent=True))
    import os
    # ===== 加载数据 =====
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)
    mesh = trimesh.load_mesh(mesh_path)

    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
    slices_data = load_slices_data(pickle_file)

    # ===== 转换为 PyVista 网格 =====
    mesh_pv = pv.wrap(mesh_filtered) if mesh_filtered is not None else pv.PolyData()
    multi_panel_display(mesh_pv, slices_data, step_values=(None, 0.5, 0.1, 0.05))

    # multi_panel_display_composite(mesh_pv, slices_data,
    #                           group_pkl_path=str(get_path("group_centroid_output", create_parent=True)))
    # ===== 创建 PyVista 绘图对象 =====
    # pickle_path = str(get_path("line_face_output", create_parent=True))
    # full_lines = load_full_segments(pickle_path)

    # 4. 可视化
    # 图4-
    plotter = pv.Plotter(window_size=(1200, 600))
    plotter.enable_parallel_projection()
    plotter.add_mesh(mesh_pv, color="lightgray", opacity=1, show_edges=False)
    plotter = add_slice_lines_to_plotter(plotter, slices_data, lower=0, upper=140, step=0.05)

    # 在这里添加全部线段
    # plotter = add_full_segments_to_plotter(plotter, full_lines, color="red", line_width=3)
    # plotter.add_key_event("s", bind_on_key_press(plotter))
    # 3. 加动态比例尺
    # add_dynamic_scale_bar(plotter, length=50.0, bar_height=10, display_margin_ratio=0.05, font_size=12)
    # plotter.show(title="网格 + 剖面线段 + 全部面线段", auto_close=False)
    plotter.show()


if __name__ == "__main__":
    main()
