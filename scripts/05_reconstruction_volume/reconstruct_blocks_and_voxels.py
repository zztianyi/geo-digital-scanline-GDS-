"""Block reconstruction and voxel analysis.

Constructs structural blocks, voxel representations, and volume-related outputs.
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
import json
import trimesh
import open3d as o3d
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.sparse import lil_matrix
from tqdm import tqdm
from collections import defaultdict
import pyvista as pv
from scipy.spatial import cKDTree
import concurrent.futures
from shapely.geometry import Polygon
from rtree import index
from matplotlib.path import Path

# ==================== Mesh 转换与可视化函数 ==================== #

def trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(tri_mesh.vertices, dtype=np.float64)
    faces = np.asarray(tri_mesh.faces, dtype=np.int32)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def create_open3d_lineset(all_segments, color=[1.0, 0.0, 0.0]):
    points, lines, colors = [], [], []
    idx = 0
    for seg in all_segments:
        p1, p2, _ = seg  # 忽略高度
        points.extend([p1, p2])
        lines.append([idx, idx + 1])
        idx += 2
    colors = [color] * len(lines)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return line_set

def extract_faces_submesh_o3d(mesh_filtered: trimesh.Trimesh, face_indices, color=[1, 0, 0]):
    if not face_indices:
        return None
    mask = np.zeros(len(mesh_filtered.faces), dtype=bool)
    mask[list(face_indices)] = True
    submeshes = mesh_filtered.submesh([mask], only_watertight=False)
    if not submeshes:
        return None
    sub_mesh_o3d = trimesh_to_open3d(submeshes[0])
    sub_mesh_o3d.paint_uniform_color(color)
    return sub_mesh_o3d

def visualize_o3d(mesh_filtered: trimesh.Trimesh,
                  all_segments,
                  face_indices,
                  line_color=[1, 0, 0],
                  mesh_color=[0.7, 0.7, 0.7],
                  highlight_color=[1, 0, 0]):
    mesh_o3d = trimesh_to_open3d(mesh_filtered)
    mesh_o3d.paint_uniform_color(mesh_color)
    line_set_o3d = create_open3d_lineset(all_segments, color=line_color)
    highlight_submesh_o3d = extract_faces_submesh_o3d(mesh_filtered, face_indices, color=highlight_color)
    geometry_list = [mesh_o3d, line_set_o3d]
    if highlight_submesh_o3d:
        geometry_list.append(highlight_submesh_o3d)
    print("使用 Open3D 展示：网格 + 线段 + 高亮面")
    o3d.visualization.draw_geometries(geometry_list)

# ==================== 配置与网格处理函数 ==================== #

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
    return submeshes[0] if submeshes else None

# ==================== 面邻接及聚类（保留原有逻辑） ==================== #

def build_face_adjacency_graph(mesh: trimesh.Trimesh) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(mesh.face_adjacency)
    return G

def compute_partial_distance_matrix(graph: nx.Graph, target_faces: set, max_distance=4):
    target_faces = list(target_faces)
    index_map = {face: i for i, face in enumerate(target_faces)}
    N = len(target_faces)
    dist_matrix = lil_matrix((N, N), dtype=np.float32)
    for face in tqdm(target_faces, desc="计算局部邻接距离"):
        lengths = nx.single_source_shortest_path_length(graph, face, cutoff=max_distance)
        i = index_map[face]
        for neighbor_face, d in lengths.items():
            if neighbor_face in index_map and d > 0:
                j = index_map[neighbor_face]
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d  # 对称
    return dist_matrix.tocsr(), target_faces

def perform_dbscan_clustering(dist_matrix, eps=2, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    return labels

def visualize_mesh_clusters_pyvista(mesh: trimesh.Trimesh,
                                    face_indices: list,
                                    labels: np.ndarray,
                                    show_edges=True,
                                    show_noise=False):
    faces_np = mesh.faces
    verts_np = mesh.vertices
    faces_pv = np.hstack([np.insert(faces_np[i], 0, 3) for i in range(len(faces_np))])
    pv_mesh = pv.PolyData(verts_np, faces_pv)
    face_labels = np.full(len(faces_np), -1, dtype=int)
    for idx, face_id in enumerate(face_indices):
        face_labels[face_id] = labels[idx]
    if not show_noise:
        pv_mesh = pv_mesh.extract_cells(face_labels != -1)
        face_labels = face_labels[face_labels != -1]
    pv_mesh.cell_data["ClusterID"] = face_labels
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="ClusterID", show_edges=show_edges,
                     cmap="tab20", nan_color="gray")
    plotter.add_scalar_bar(title="Cluster ID")
    plotter.show()

# ==================== 辅助函数：Bounding Box 与 棱柱构造 ==================== #

def get_xy_bounding_box(verts_3d: np.ndarray):
    xs = verts_3d[:, 0]
    ys = verts_3d[:, 1]
    return (xs.min(), xs.max(), ys.min(), ys.max())

def is_bbox_overlap(bb1, bb2):
    return not (bb1[1] < bb2[0] or bb1[0] > bb2[1] or bb1[3] < bb2[2] or bb1[2] > bb2[3])

def construct_prism_mesh(base_verts_3d: np.ndarray, height_z: float) -> trimesh.Trimesh:
    centroid_z = base_verts_3d[:, 2].mean()
    top_verts_3d = base_verts_3d.copy()
    top_verts_3d[:, 2] = centroid_z + height_z
    vertices = np.vstack([base_verts_3d, top_verts_3d])
    faces = []
    faces.append([0, 1, 2])  # 底面
    faces.append([3, 4, 5])  # 顶面
    def quad_to_tri(q):
        return [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    for quad in ([0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]):
        faces.extend(quad_to_tri(quad))
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False)

# ==================== 体素化及相关函数 ==================== #

def voxel_union_o3d(all_meshes, voxel_size=0.05):
    min_corner = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_corner = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for mesh in all_meshes:
        bmin, bmax = mesh.bounds
        min_corner = np.minimum(min_corner, bmin)
        max_corner = np.maximum(max_corner, bmax)
    xs = np.arange(min_corner[0], max_corner[0], voxel_size)
    ys = np.arange(min_corner[1], max_corner[1], voxel_size)
    zs = np.arange(min_corner[2], max_corner[2], voxel_size)
    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        print("Voxel 范围太小，可能无效！")
        return None, None
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')
    all_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    del X, Y, Z
    occupancy = np.zeros(len(all_points), dtype=bool)
    for tm in all_meshes:
        occupancy |= tm.contains(all_points)
    union_points = all_points[occupancy]
    if len(union_points) == 0:
        print("体素判断后无占据点，可能网格与设置不匹配")
        return None, None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(union_points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=voxel_size,
        min_bound=o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner).get_min_bound(),
        max_bound=o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner).get_max_bound()
    )
    return voxel_grid, pcd

def voxelize_faces_sparse(mesh_filtered, face_heights, grid_shape, origin, voxel_size):
    faces = mesh_filtered.faces
    vertices = mesh_filtered.vertices
    active_voxels = set()
    for face_idx, height in tqdm(face_heights.items(), desc="稀疏体素化"):
        if height <= 1e-8:
            continue
        tri = vertices[faces[face_idx]]
        tri_xy = tri[:, :2]
        poly = Polygon(tri_xy)
        if poly.area < 1e-10:
            continue
        path = Path(tri_xy)
        min_xy = np.min(tri_xy, axis=0)
        max_xy = np.max(tri_xy, axis=0)
        min_z = np.min(tri[:, 2])
        max_z = np.mean(tri[:, 2]) + height
        min_idx = np.floor((np.array([min_xy[0], min_xy[1], min_z]) - origin) / voxel_size).astype(int)
        max_idx = np.ceil((np.array([max_xy[0], max_xy[1], max_z]) - origin) / voxel_size).astype(int)
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, np.array(grid_shape) - 1)
        for ix in range(min_idx[0], max_idx[0] + 1):
            for iy in range(min_idx[1], max_idx[1] + 1):
                for iz in range(min_idx[2], max_idx[2] + 1):
                    center = origin + voxel_size * (np.array([ix, iy, iz]) + 0.5)
                    if path.contains_point(center[:2]) and (min_z <= center[2] <= max_z):
                        active_voxels.add((ix, iy, iz))
    return list(active_voxels)

def get_voxel_grid_meta(mesh_filtered, voxel_size=0.05):
    bounds = mesh_filtered.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    grid_shape = tuple(dims.tolist())
    return grid_shape, min_bound, voxel_size

def save_active_voxels_pickle(active_voxels, grid_shape, origin, voxel_size, output_path):
    data = {
        "active_voxels": active_voxels,
        "grid_shape": grid_shape,
        "origin": origin,
        "voxel_size": voxel_size
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ 体素数据（含元信息）已保存为：{output_path}")

def create_voxel_pointcloud(active_voxel_indices, voxel_size=0.05):
    centers = np.array(active_voxel_indices, dtype=np.float32)
    centers = (centers + 0.5) * voxel_size
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    return pcd

def visualize_mesh_and_voxel_pcd(mesh_filtered, active_voxel_indices, voxel_size=0.05):
    mesh_o3d = trimesh_to_open3d(mesh_filtered)
    pcd = create_voxel_pointcloud(active_voxel_indices, voxel_size)
    o3d.visualization.draw_geometries([mesh_o3d, pcd])

# ==================== 面高度相关函数 ==================== #

def merge_face_heights(face_heights_dict, strategy='max'):
    """
    合并同一面对应的多个高度值。策略可选：
      - 'max'：取最大高度
      - 'min'：取最小高度
      - 'avg'：取平均高度
    """
    merged = {}
    for face_idx, heights in face_heights_dict.items():
        if not heights:
            continue
        if strategy == 'max':
            merged[face_idx] = max(heights)
        elif strategy == 'min':
            merged[face_idx] = min(heights)
        elif strategy == 'avg':
            merged[face_idx] = sum(heights) / len(heights)
        else:
            merged[face_idx] = max(heights)
    return merged

def load_full_line_face_data(input_path):
    """
    从 pickle 文件中加载完整的线段数据及原始面高度字典，
    数据格式为 (all_segments, face_heights_dict)
    """
    with open(input_path, "rb") as f:
        all_segments, face_heights_dict = pickle.load(f)
    print("已加载完整线段及面数据")
    return all_segments, face_heights_dict

# ==================== 主流程 ==================== #

def main():
    """
    主流程：
      1. 加载配置与数据；
      2. 提取线段及对应的面索引（含各自高度）；
      3. 对每个面合并高度，策略选择 'min'（即选择最小高度）；
      4. 读入网格并根据 z 范围过滤；
      5. 利用新面高度构造稀疏体素，并保存。
    """
    # 请根据实际情况调整路径
    mesh_path   = str(get_path("mesh_model"))
    config_path = str(get_path("arc_config"))
    data_path   = str(get_path("line_face_output", create_parent=True))

    # 加载配置：获取 z 范围
    _, _, _, _, _, z_min, z_max = load_arc_config(config_path)

    # 加载完整数据：线段数据及原始面高度字典（未合并）
    all_segments, face_heights_dict = load_full_line_face_data(data_path)
    # 获取所有涉及的面索引
    all_faces = list(face_heights_dict.keys())

    # 对每个面合并多个高度数据，策略选择 'min'
    merged_face_heights = merge_face_heights(face_heights_dict, strategy='min')

    # 读入网格，若为 Scene 则合并为单一网格；再过滤 z 范围内的面
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
    if mesh_filtered is None:
        print("mesh_filtered 为空，请检查 z_min, z_max 或原始网格")
        return

    # 采用基于线段计算的合并高度作为面高度
    face_heights = merged_face_heights

    grid_shape, origin, voxel_size = get_voxel_grid_meta(mesh_filtered, voxel_size=0.05)

    active_voxel_indices = voxelize_faces_sparse(
        mesh_filtered,
        face_heights,
        grid_shape,
        origin,
        voxel_size
    )

    voxels_output = str(get_path("voxel_output", create_parent=True))
    save_active_voxels_pickle(active_voxel_indices, grid_shape, origin, voxel_size, voxels_output)

if __name__ == "__main__":
    main()