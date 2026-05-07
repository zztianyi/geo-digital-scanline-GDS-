"""Mesh clipping and profile comparison.

Experiments with slice-plane clipping and 3D/2D profile visualization.
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
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import pyvista as pv
# 设置英文字体
font_en = FontProperties(family='Arial', size=12)
font_en_13 = FontProperties(family='Arial', size=13)
# 设置中文字体（宋体）
font_zh = FontProperties(fname=str(get_path("font_zh")), size=12)
font_zh_13 = FontProperties(fname=str(get_path("font_zh")), size=13)
plt.rcParams['font.size'] = 12               # 设置默认字体大小
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def load_arc_config(config_path):
    """
    加载配置文件，返回圆心、半径、角度范围、弧长范围及 Z 轴范围。
    配置中角度均为弧度，弧长范围计算公式：radius * (angle_max - angle_min)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    center_2d = np.array(config["center"], dtype=float)
    center_3d = np.append(center_2d, 1400)
    radius = float(config["radius"])
    angle_min = config["angle_min"]
    angle_max = config["angle_max"]
    arc_length = radius * (angle_max - angle_min)
    arc_length_range = [0, arc_length]
    z_min, z_max = config["z_range"]
    return center_3d, radius, angle_min, angle_max, arc_length_range, z_min, z_max

def compute_slice_plane(center, radius, angle_min, arc_length_range, slice_position):
    """
    根据切剖面位置（弧长坐标）计算切剖面参数：
      slice_angle = angle_min + slice_position / radius
    返回字典，包括平面原点、法向量、径向方向及切剖面角度。
    """
    if slice_position is None:
        slice_position = arc_length_range[1] / 2
    slice_angle = angle_min + slice_position / radius
    radial_dir = np.array([np.cos(slice_angle), np.sin(slice_angle), 0.0])
    radial_dir /= np.linalg.norm(radial_dir)
    normal = np.cross(radial_dir, [0, 0, 1])
    normal /= np.linalg.norm(normal)
    return {"origin": center, "normal": normal, "radial_dir": radial_dir, "slice_angle": slice_angle}

def filter_mesh_by_z_range(mesh, z_min, z_max):
    """
    根据 Z 轴范围过滤网格，仅保留符合条件的三角面。
    """
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if len(submeshes) > 0 else None

def slice_mesh_custom(mesh, origin, normal, near_tol=0.1):
    """
    自定义切面函数：返回距离平面近似的点和网格与平面的交线段。
    """
    vertices = mesh.vertices
    dist = np.dot(vertices - origin, normal)
    near_points = vertices[np.abs(dist) <= near_tol]
    intersections = trimesh.intersections.mesh_plane(mesh, normal, origin)
    return near_points, intersections

def project_to_plane_custom(points, origin, radial_dir):
    """
    将 3D 点投影到局部二维坐标系。
    定义 u 轴为 radial_dir，v 轴为全局 Z 方向。
    支持单个点或多个点。
    """
    points = np.atleast_2d(points)
    vec = points - origin
    u = np.dot(vec, radial_dir)
    v = vec[:, 2]  # 取 z 分量
    return np.column_stack((u, v))

def extract_section_paths(section):
    """
    从 trimesh 返回的 section 中提取所有线/多段线，保留拓扑结构。
    """
    import networkx as nx
    graph = nx.Graph()

    for entity in section.entities:
        points = entity.points
        # 如果是线段(2个点)
        if len(points) == 2:
            graph.add_edge(points[0], points[1])
        # 如果是多段线(多个点)，拆分为连续线段
        elif len(points) > 2:
            for i in range(len(points)-1):
                graph.add_edge(points[i], points[i+1])

    paths = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        endpoints = [node for node, degree in subgraph.degree() if degree == 1]

        if len(endpoints) == 0:
            # 闭合路径
            cycle = nx.find_cycle(subgraph)
            path = [edge[0] for edge in cycle] + [cycle[0][0]]
        elif len(endpoints) == 2:
            # 开口路径
            path = nx.shortest_path(subgraph, endpoints[0], endpoints[1])
        else:
            continue  # 复杂拓扑跳过

        coords = section.vertices[path]
        paths.append(coords)

    return paths

# ============= 以下为添加的关键功能：面邻接检查 =============
def build_face_adjacency_list(mesh):
    """
    构建 face -> set_of_neighbor_faces 的邻接表
    """
    adjacency = mesh.face_adjacency  # (M,2)，每行是一对相邻面
    adj_dict = {}
    for f_id in range(len(mesh.faces)):
        adj_dict[f_id] = set()
    for (f1, f2) in adjacency:
        adj_dict[f1].add(f2)
        adj_dict[f2].add(f1)
    return adj_dict

def is_valid_segment(face_a, face_b, adjacency_dict):
    """
    face_a == face_b 或 face_b 在 adjacency_dict[face_a] 中
    则认为线段是合法的（蓝色）
    """
    if face_a == face_b:
        return True
    if face_b in adjacency_dict[face_a]:
        return True
    return False

def slice_and_check_faces(mesh, origin, normal):
    """
    使用 mesh_plane 并返回 (lines_3d, face_ids_2d)，用于后续检查
    """
    # return_faces=True 可以让我们拿到每个端点的面 ID
    lines_3d, face_ids_2d = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=normal,
        plane_origin=origin,
        return_faces=True
    )
    return lines_3d, face_ids_2d


def plot_trimesh_on_ax(ax, mesh, face_color=(0.5, 0.5, 0.5, 0.3)):
    """
    使用 matplotlib 绘制 trimesh 模型到已有的 ax
    """
    faces = mesh.faces
    vertices = mesh.vertices

    # 构造每个三角形的 3D 顶点集合
    triangles = vertices[faces]  # shape: (N, 3, 3)

    # 创建 Poly3DCollection
    mesh_collection = Poly3DCollection(triangles, facecolors=face_color, edgecolors='k', linewidths=0.1)
    mesh_collection.set_alpha(face_color[3])  # 设置透明度
    ax.add_collection3d(mesh_collection)

    # 自动缩放
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
# =========================================================

def test_single_slice_comparison():
    config_path = str(get_path("arc_config"))
    mesh_path = str(get_path("mesh_model"))
    
    # 1. 加载配置和网格
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)
    mesh = trimesh.load_mesh(mesh_path)
    print("Vertices shape:", mesh.vertices.shape)
    print("Faces shape:", mesh.faces.shape)

    # 2. Z 过滤
    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
    
    # 3. 切面参数
    slice_position = 52
    plane_params = compute_slice_plane(center, radius, angle_min, arc_length_range, slice_position)
    origin = plane_params["origin"]
    normal = plane_params["normal"]
    radial_dir = plane_params["radial_dir"]
    
    # 4. 自定义 slice_mesh（仅做示例对比）
    near_points, intersections_custom = slice_mesh_custom(mesh_filtered, origin, normal, near_tol=0.1)
    custom_2d_segments = []
    for seg in intersections_custom:
        pt1, pt2 = seg[0], seg[1]
        pt1_2d = project_to_plane_custom(pt1, origin, radial_dir)
        pt2_2d = project_to_plane_custom(pt2, origin, radial_dir)
        custom_2d_segments.append((pt1_2d, pt2_2d))

    # 5. 使用 trimesh.section 提取路径 (不带面信息，但可做拓扑提取)
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is not None and len(section.entities) > 0:
        paths = extract_section_paths(section)
    else:
        paths = []
        print("无剖面线，请检查切面位置。")

    # 6. 面邻接检查：对 intersection 直接获取端点面 ID 并判断
    #    （这里使用 mesh_filtered 与 plane，因为我们只关心已过滤部分）
    lines_3d, face_ids = slice_and_check_faces(mesh_filtered, origin, normal)
    face_adjacency_dict = build_face_adjacency_list(mesh_filtered)

    print(face_ids.shape)


    # 3. 使用 Trimesh 创建一个子网格，只包含这些面
    highlight_mesh = mesh_filtered.submesh([face_ids], append=True)

    def trimesh_to_pv(mesh):
        faces = np.hstack([[3] + list(face) for face in mesh.faces]).astype(np.int32)
        return pv.PolyData(mesh.vertices, faces)
    
    def lines_to_pv_polyline(lines_3d):
        """
        将 lines_3d (N,2,3) 转换为 pyvista.PolyData，其中包含多条线段
        """
        points = []
        lines = []
        for line in lines_3d:
            pt1, pt2 = line
            idx1 = len(points)
            idx2 = idx1 + 1
            points.extend([pt1, pt2])
            lines.append([2, idx1, idx2])  # 每条线段两个点，2 是数量标记
        points = np.array(points)
        lines = np.array(lines, dtype=np.int64)

        # 1. 创建空的 PolyData 对象
        pdata = pv.PolyData()
        
        # 2. 设置点
        pdata.points = points
        
        # 3. 设置线单元（必须使用 set_lines）
        pdata.lines = lines

        return pdata
    def create_cut_plane(origin, normal, size=300.0):
        """
        构造一个平面用于可视化，origin 是平面上的点，normal 是法向量
        """
        return pv.Plane(center=origin, direction=normal, i_size=size, j_size=size)
    # 创建 pyvista 网格
    pv_mesh = trimesh_to_pv(mesh)
    pv_highlight = trimesh_to_pv(highlight_mesh)
    pv_lines = lines_to_pv_polyline(lines_3d)

    # 创建 plotter
    plotter = pv.Plotter()
    plotter.enable_parallel_projection()

    plotter.add_mesh(pv_mesh, color='lightgray', opacity=1, show_edges=False)
    plotter.add_mesh(pv_highlight, color='red', opacity=1.0, show_edges=False)
    plotter.add_mesh(pv_lines, color='blue', line_width=3, label='切面线段')
    plane_mesh = create_cut_plane(origin, normal)
    # plotter.add_mesh(plane_mesh, color='green', style='wireframe', opacity=0.3, label='切割平面')
    plotter.add_title("被切割三角面展示", font_size=14)
    # plotter.show_grid()

    # 显示
    plotter.show()

    # 如果想看 2D 对比图（custom_2d_segments vs. mesh.section）可解开下面注释
    # 假设你之前做了类似 section_2d 的投影，可以用这个函数对比
    # display_slice_comparison(custom_2d_segments, section_2d, slice_position, plane_params)

if __name__ == "__main__":
    test_single_slice_comparison()
