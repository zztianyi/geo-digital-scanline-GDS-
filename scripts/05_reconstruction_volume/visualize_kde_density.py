"""KDE mesh coloring and visualization.

Colors meshes and visualizes spatial density results.
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
import trimesh
import pyvista as pv
from plyfile import PlyData, PlyElement
import matplotlib.colors as mcolors
import matplotlib
import vtk  # 用于创建2D比例尺相关 actor

def create_custom_summer_red():
    """
    创建一个自定义 colormap，底部采用 summer 色带，
    上部从 summer(0.8) 线性过渡到红色 (1,0,0)。
    """
    summer = matplotlib.cm.get_cmap("summer")
    # 使用 from_list 来构造新的 colormap
    new_cmp = mcolors.LinearSegmentedColormap.from_list(
        'SummerRed',
        [summer(0.0), summer(0.8), (1, 0, 0, 1)],
        N=256
    )
    return new_cmp


def load_trimesh_without_texture(obj_path):
    scene = trimesh.load(obj_path, force='scene')
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("OBJ 加载失败")
    all_meshes = [geom for _, geom in scene.geometry.items()]
    mesh_combined = trimesh.util.concatenate(all_meshes)
    return mesh_combined


def trimesh_to_pyvista_colored(mesh, vertex_scalar=None, scalar_name="Density"):
    faces = np.hstack([[3, *face] for face in mesh.faces])
    mesh_pv = pv.PolyData(mesh.vertices, faces)

    # 如果给定了标量数组，则绑定为点数据
    if vertex_scalar is not None and len(vertex_scalar) == len(mesh.vertices):
        mesh_pv.point_data[scalar_name] = vertex_scalar
    elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        # 否则优先尝试可视化颜色
        vc = mesh.visual.vertex_colors[:, :3]
        mesh_pv.point_data["Colors"] = vc / 255.0

    return mesh_pv

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


def main():
    # ==== 路径设置 ====
    mesh_path = str(get_path("segmented_mesh"))
    # kde_result_path = str(get_path("kde_result_output", create_parent=True))#总体积的空间分布密度数据kde估计带宽选择为5m

    kde_result_path = str(get_path("kde_result_output", create_parent=True))
    # ==== 加载网格（不加载纹理） ====
    mesh_trimesh = load_trimesh_without_texture(mesh_path)

    # ==== 加载 KDE 估值结果 ====
    with open(kde_result_path, "rb") as f:
        data = pickle.load(f)
    vertex_density = data["vertex_density"]

    # ==== 转为 PyVista 并赋值标量 ====
    mesh_pv = trimesh_to_pyvista_colored(mesh_trimesh)
    mesh_pv.point_data["kde"] = vertex_density

    # 创建两个独立副本：基础层和热力图层
    mesh_base = mesh_pv.copy(deep=True)
    # 移除基础层的标量数据，使其只显示固定颜色
    if "kde" in mesh_base.point_data.keys():
        del mesh_base.point_data["kde"]

    mesh_overlay = mesh_pv.copy(deep=True)

   # ==== 计算非线性映射后的色彩与透明度 ====
    scalars = mesh_overlay.point_data["kde"]
    # 归一化到 [0, 1]
    scalars_min = scalars.min()
    scalars_max = scalars.max()
    scalars_norm = (scalars - scalars_min) / (scalars_max - scalars_min + 1e-8)

    gamma = 4.0  # 非线性参数，根据需要调节
    # 非线性映射后的标量，用于颜色映射
    transformed = np.power(scalars_norm, gamma)
    # 用于透明度的非线性映射
    min_opacity = 0.8
    max_opacity = 1.0
    opacities = min_opacity + (max_opacity - min_opacity) * transformed

    # 将非线性映射后的标量保存下来，用于热力图色彩（范围仍在 [0,1]）
    mesh_overlay.point_data["nonlinear_kde"] = transformed

    # ==== 可选：设定你想要的 colormap，这里以 "viridis" 为例 ====

    selected_cmap = create_custom_summer_red()
    # ==== 设置全局字体为 Times New Roman，字号14 ====
    pv.global_theme.font.family = "times"  # 修改全局字体属性
    pv.global_theme.font.size = 14

    # ==== 设置 colormap 与图例参数 ====

    scalar_bar_args = {
        "vertical": True,          # 图例竖直排列
        "title": "kde",            # 图例标题
        "position_x": 0.85,        # 调整图例X位置
        "position_y": 0.1,         # 调整图例Y位置
        "width": 0.1,             # 图例宽度
        "height": 0.8,             # 图例高度
        "font_family": "times",    # 图例中所有文字字体
        "label_font_size": 16,     # 图例中标签文字字号
        "title_font_size": 18      # 图例标题字号
    }
    # ==== 分层显示 ====
    plotter = pv.Plotter()
    # 第一层：基础网格，全不透明，使用固定颜色显示
    plotter.add_mesh(mesh_base, color="lightgray", opacity=0.9, show_edges=False)
    # 第二层：叠加热力图，应用标量和透明度
    plotter.add_mesh(
        mesh_overlay,
        scalars="kde",
        cmap=selected_cmap,
        opacity=opacities,
        show_edges=False,
        scalar_bar_args=scalar_bar_args
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

    plotter.show()

if __name__ == "__main__":
    main()
