"""Surface grouping and face clustering.

Builds adjacency relations and clusters connected structural faces.
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
import matplotlib.pyplot as plt
import pyvista as pv

def gather_line_data(single_res):
    line_segments = []
    face_indices = set()

    if not single_res:
        return line_segments, face_indices

    red_groups_corrected = single_res.get("red_groups_corrected", {})
    for parent_idx, groups in red_groups_corrected.items():
        for group in groups:
            group_segments = group.get("group_segments", [])
            for seg in group_segments:
                p1_3d = seg[4]
                p2_3d = seg[5]
                face_idx = seg[6]

                line_segments.append((p1_3d, p2_3d))
                if face_idx is not None:
                    face_indices.add(face_idx)

    return line_segments, face_indices

def serial_collect_line_data(all_results):
    """
    串行地收集所有结果的线段与面索引。
    """
    all_segments = []
    all_faces = set()

    for single_res in all_results:
        segs, faces = gather_line_data(single_res)
        all_segments.extend(segs)
        all_faces |= faces

    return all_segments, all_faces

def trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(tri_mesh.vertices, dtype=np.float64)
    faces = np.asarray(tri_mesh.faces, dtype=np.int32)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def create_open3d_lineset(all_segments, color=[1.0, 0.0, 0.0]):
    points = []
    lines = []
    colors = []

    idx = 0
    for seg in all_segments:
        p1, p2 = seg
        points.append(p1)
        points.append(p2)
        lines.append([idx, idx+1])
        idx += 2

    for _ in range(len(lines)):
        colors.append(color)

    points_np = np.array(points, dtype=np.float64)
    lines_np = np.array(lines, dtype=np.int32)
    colors_np = np.array(colors, dtype=np.float64)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_np)
    line_set.lines = o3d.utility.Vector2iVector(lines_np)
    line_set.colors = o3d.utility.Vector3dVector(colors_np)
    return line_set

def extract_faces_submesh_o3d(mesh_filtered: trimesh.Trimesh, face_indices, color=[1, 0, 0]):
    if not face_indices:
        return None

    face_indices_list = list(face_indices)
    mask = np.zeros(len(mesh_filtered.faces), dtype=bool)
    mask[face_indices_list] = True

    submeshes = mesh_filtered.submesh([mask], only_watertight=False)
    if len(submeshes) == 0:
        return None
    sub_mesh = submeshes[0]

    sub_mesh_o3d = trimesh_to_open3d(sub_mesh)
    sub_mesh_o3d.paint_uniform_color(color)
    return sub_mesh_o3d

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

def visualize_o3d(mesh_filtered: trimesh.Trimesh,
                  all_segments,
                  all_faces,
                  line_color=[1, 0, 0],
                  mesh_color=[0.7, 0.7, 0.7],
                  highlight_color=[1, 0, 0]):
    mesh_o3d = trimesh_to_open3d(mesh_filtered)
    mesh_o3d.paint_uniform_color(mesh_color)

    line_set_o3d = create_open3d_lineset(all_segments, color=line_color)

    highlight_submesh_o3d = extract_faces_submesh_o3d(mesh_filtered, all_faces, color=highlight_color)

    geometry_list = [mesh_o3d, line_set_o3d]
    if highlight_submesh_o3d:
        geometry_list.append(highlight_submesh_o3d)

    print("使用 Open3D 展示：网格 + 线段 + 高亮面")
    o3d.visualization.draw_geometries(geometry_list)
def build_face_adjacency_graph(mesh: trimesh.Trimesh) -> nx.Graph:
    """
    构建三角面邻接图，面索引为节点，面之间共享边为邻接。
    """
    G = nx.Graph()
    G.add_edges_from(mesh.face_adjacency)
    return G

def compute_distance_matrix(graph: nx.Graph, max_distance=10):
    """
    计算所有三角面之间的最短路径长度，构成距离矩阵。
    可设置 max_distance 截断，避免太稀疏。
    """
    face_indices = list(graph.nodes)
    num_faces = len(face_indices)
    index_map = {face_idx: i for i, face_idx in enumerate(face_indices)}
    
    dist_matrix = np.full((num_faces, num_faces), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    for source in tqdm(face_indices, desc="计算最短路径"):
        lengths = nx.single_source_shortest_path_length(graph, source, cutoff=max_distance)
        for target, d in lengths.items():
            i = index_map[source]
            j = index_map[target]
            dist_matrix[i, j] = d

    return dist_matrix, face_indices

def perform_dbscan_clustering(dist_matrix, eps=2, min_samples=5):
    """
    使用 DBSCAN 基于距离矩阵聚类三角面索引。
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    return labels
def compute_partial_distance_matrix(graph: nx.Graph, target_faces: set, max_distance=4):
    """
    仅在目标面之间计算距离 ≤ max_distance 的最短路径。
    返回稀疏矩阵和索引列表。
    """
    target_faces = list(target_faces)
    index_map = {face: i for i, face in enumerate(target_faces)}
    N = len(target_faces)

    # 使用稀疏矩阵
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

def save_sparse_distance_matrix(filename, dist_matrix, face_indices):
    with open(filename, 'wb') as f:
        pickle.dump({
            'dist_matrix': dist_matrix,
            'face_indices': face_indices
        }, f)

def load_sparse_distance_matrix(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['dist_matrix'], data['face_indices']

def visualize_mesh_clusters_pyvista(mesh: trimesh.Trimesh,
                                     face_indices: list,
                                     labels: np.ndarray,
                                     show_edges=True,
                                     show_noise=False):
    """
    使用 PyVista 根据 labels 可视化聚类面（按 Cell 显示）。
    """
    # 构建 PyVista Mesh
    faces_np = mesh.faces
    verts_np = mesh.vertices

    # PyVista 要求 faces 为 [n, v0, v1, v2, n, ...] 格式（含每个面点数量）
    faces_pv = np.hstack([
        np.insert(faces_np[i], 0, 3)
        for i in range(faces_np.shape[0])
    ])

    # 创建 PyVista PolyData 网格
    pv_mesh = pv.PolyData(verts_np, faces_pv)

    # 默认每个面是 -1（不参与聚类）
    face_labels = np.full(len(faces_np), -1, dtype=int)
    for idx, face_id in enumerate(face_indices):
        face_labels[face_id] = labels[idx]

    if not show_noise:
        # 去除 label == -1 的面
        pv_mesh = pv_mesh.extract_cells(face_labels != -1)
        face_labels = face_labels[face_labels != -1]

    # 添加标签作为 Cell Scalar 属性
    pv_mesh.cell_data["ClusterID"] = face_labels

    # 创建绘图器并绘制
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="ClusterID", show_edges=show_edges,
                     cmap="tab20", nan_color="gray")
    plotter.add_scalar_bar(title="Cluster ID")
    plotter.show()



def main():
    results_pkl = str(get_path("multi_profile_output", create_parent=True))
    mesh_path   = str(get_path("mesh_model"))
    config_path = str(get_path("arc_config"))

    _, _, _, _, _, z_min, z_max = load_arc_config(config_path)

    all_results = []
    with open(results_pkl, "rb") as f:
        while True:
            try:
                batch_data = pickle.load(f)
                all_results.append(batch_data)
            except EOFError:
                break

    # 串行提取
    all_segments, all_faces = serial_collect_line_data(all_results)

    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)

    # visualize_o3d(
    #     mesh_filtered,
    #     all_segments,
    #     all_faces,
    #     line_color=[1, 0, 0],
    #     mesh_color=[0.7, 0.7, 0.7],
    #     highlight_color=[1, 1, 0]
    # )
    # 1. 构建邻接图
    G = build_face_adjacency_graph(mesh_filtered)


    # # 执行保存邻接表
    # dist_matrix, face_indices = compute_partial_distance_matrix(G, all_faces, max_distance=4)
    # save_sparse_distance_matrix(str(get_path("face_distance_matrix", create_parent=True)), dist_matrix, face_indices)

    dist_matrix, face_indices = load_sparse_distance_matrix(str(get_path("face_distance_matrix", create_parent=True)))

    #dbscan邻接表聚类
    labels = perform_dbscan_clustering(dist_matrix, eps=2, min_samples=8)



    clusters = defaultdict(list)
    for idx, label in zip(face_indices, labels):
        if label != -1:
            clusters[label].append(idx)

    print(f"聚类数量: {len(clusters)}")
    visualize_mesh_clusters_pyvista(mesh_filtered, face_indices, labels)

if __name__ == "__main__":
    main()
