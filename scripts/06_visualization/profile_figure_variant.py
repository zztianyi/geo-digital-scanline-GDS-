"""Profile visualization variant.

Experimental visualization variant kept for comparison with image_.py.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D

def load_arc_config(config_path):
    """
    加载arc_config.json，返回:
      - center (3D圆心, z默认为0)
      - radius (浮点数)
      - angle_range (角度区间, 单位弧度或度看你配置)
      - z_min, z_max (Z轴范围)
    """
    with open(config_path, 'r') as f:
        arc_config = json.load(f)
    center_2d = np.array(arc_config["center"], dtype=float)
    center_3d = np.append(center_2d, 0.0)
    
    radius = float(arc_config["radius"])
    angle_range = [arc_config["angle_min"], arc_config["angle_max"]]
    
    z_min, z_max = arc_config["z_range"]
    return center_3d, radius, angle_range, z_min, z_max

def compute_plane(center, radius, angle_min, arc_length_range, slice_position):
    """
    根据外部输入的切剖面位置（弧长坐标）计算剖切平面。
    计算方法：
      - 将切剖面位置转换为角度： slice_angle = angle_min + slice_position/radius
      - 根据该角度计算径向方向及平面法向量
    返回：平面原点、法向量、径向方向，以及计算得到的切剖面角度
    """
    if slice_position is None:
        slice_position = arc_length_range[1] / 2
    slice_angle = angle_min + slice_position / radius
    radial_dir = np.array([np.cos(slice_angle), np.sin(slice_angle), 0.0])
    radial_dir /= np.linalg.norm(radial_dir)
    normal = np.cross(radial_dir, [0, 0, 1])
    normal /= np.linalg.norm(normal)
    return center, normal, radial_dir, slice_angle

def filter_mesh_by_z_range(mesh, z_min, z_max):
    """
    根据 Z 轴范围过滤网格，只保留在 [z_min, z_max] 区间内的三角面
    """
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min_z = face_vertices_z.min(axis=1)
    face_max_z = face_vertices_z.max(axis=1)
    face_mask = (face_max_z >= z_min) & (face_min_z <= z_max)
    submeshes = mesh.submesh([face_mask], only_watertight=False)
    return submeshes[0] if len(submeshes) > 0 else None

def slice_mesh(mesh, origin, normal, near_tol):
    """
    提取:
      1) near_points: 距离平面在 ±near_tol 内的顶点
      2) intersections: 网格与平面的精确交线(线段集合)
    """
    vertices = mesh.vertices
    dist = np.dot(vertices - origin, normal)
    mask = np.abs(dist) <= near_tol
    near_points = vertices[mask]
    start_time = time.time()
    intersections = trimesh.intersections.mesh_plane(mesh, normal, origin)
    end_time = time.time()
    print(f"mesh_plane 计算用时: {end_time - start_time:.6f} 秒")
    return near_points, intersections

def project_to_plane(points_3d, origin, axis_x, axis_y):
    """
    将 points_3d 投影到平面上:
      (u,v) = (dot(p-origin, axis_x), dot(p-origin, axis_y))
    """
    out_2d = []
    for p in points_3d:
        vec = p - origin
        u = np.dot(vec, axis_x)
        v = np.dot(vec, axis_y)
        out_2d.append([u, v])
    return np.array(out_2d)

def merge_intersection_nodes(intersections, tol=1e-6):
    """
    将交线段所有端点根据 tol 合并为唯一节点，返回合并后的节点数组
    """
    if intersections is None or len(intersections) == 0:
        return np.empty((0, 3))
    nodes = []
    for seg in intersections:
        nodes.append(seg[0])
        nodes.append(seg[1])
    nodes = np.array(nodes)
    nodes_rounded = np.round(nodes, decimals=6)
    unique_nodes = np.unique(nodes_rounded, axis=0)
    return unique_nodes

def compute_node_connectivity(intersections, tol=1e-6):
    """
    统计每个节点连接的线段个数，返回字典，键为节点（tuple格式），值为连接数
    """
    node_conn = {}
    for seg in intersections:
        for p in seg:
            key = tuple(np.round(p, decimals=6))
            node_conn[key] = node_conn.get(key, 0) + 1
    return node_conn

def split_segments_to_paths(intersections, tol=1e-6):
    """
    根据 tol 合并后的节点构造图，利用 DFS 分离连通分量，
    每个连通分量对应一条路径子集，返回一个路径子集列表，每个子集为线段列表。
    """
    graph = {}
    edges = []
    for seg in intersections:
        p1 = tuple(np.round(seg[0], decimals=6))
        p2 = tuple(np.round(seg[1], decimals=6))
        edges.append((p1, p2))
        graph.setdefault(p1, set()).add(p2)
        graph.setdefault(p2, set()).add(p1)
    visited = set()
    components = []
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            stack.extend(graph[cur] - visited)
        components.append(comp)
    path_subsets = []
    for comp in components:
        subset = []
        for edge in edges:
            if edge[0] in comp and edge[1] in comp:
                subset.append(edge)
        path_subsets.append(subset)
    return path_subsets

def is_closed_subset(path_subset, node_conn):
    """
    判断路径子集是否为封闭集合：若所有节点均为连接2的内点则认为是封闭的，否则为非闭合集合
    """
    nodes = set()
    for edge in path_subset:
        nodes.add(edge[0])
        nodes.add(edge[1])
    for node in nodes:
        if node_conn.get(node, 0) == 1:
            return False
    return True

def order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir):
    """
    对任意子集构造图，并按照连接关系返回有序的节点列表，
    确定其在平面投影后纵坐标(y值)最大的点和最小的点，
    路径为从y值最大的点开始到y值最小的点结束。
    返回的节点序列为3D坐标列表。
    """
    # 构造无向图
    graph = {}
    for edge in path_subset:
        for node in edge:
            graph.setdefault(node, set())
        node1, node2 = edge
        graph[node1].add(node2)
        graph[node2].add(node1)

    # 得到所有节点并投影到平面
    all_nodes = list(graph.keys())
    nodes_arr = [np.array(nd) for nd in all_nodes]
    nodes_2d = project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
    
    # 找到投影后y值最大的和y值最小的点
    y_values = [pt[1] for pt in nodes_2d]
    max_index = y_values.index(max(y_values))
    min_index = y_values.index(min(y_values))
    start_node = all_nodes[max_index]
    end_node = all_nodes[min_index]

    # 使用BFS寻找从start_node到end_node的路径
    from collections import deque
    queue = deque([[start_node]])
    visited = set([start_node])
    path_found = None
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == end_node:
            path_found = path
            break
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                
    if path_found is None:
        print("无法找到从y值最大到y值最小的路径")
        return []
    
    return [np.array(nd) for nd in path_found]

def compute_normals_with_edges_for_subset(path_subset, closed, plane_origin, radial_dir, vertical_dir):
    """
    与 compute_normals_for_subset 类似，但同时返回对应线段端点（投影后的2D坐标）。
    返回列表，每个元素为 (起点投影, 终点投影, 线段中点, 法向量2D)。
    """
    result = []
    if not closed:
        ordered_nodes = order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir)
        if len(ordered_nodes) < 2:
            return result
        for i in range(len(ordered_nodes) - 1):
            p1 = np.array(ordered_nodes[i])
            p2 = np.array(ordered_nodes[i+1])
            p1_2d = project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            direction = p2_2d - p1_2d
            norm_val = np.linalg.norm(direction)
            if norm_val == 0:
                continue
            direction /= norm_val
            normal_vec = np.array([-direction[1], direction[0]])
            midpoint = (p1_2d + p2_2d) / 2.0
            result.append((p1_2d, p2_2d, midpoint, normal_vec))
    else:
        nodes = set()
        for edge in path_subset:
            nodes.add(edge[0])
            nodes.add(edge[1])
        nodes_arr = np.array([list(node) for node in nodes])
        nodes_2d = project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
        centroid = np.mean(nodes_2d, axis=0)
        for edge in path_subset:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            p1_2d = project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            direction = p2_2d - p1_2d
            norm_val = np.linalg.norm(direction)
            if norm_val == 0:
                continue
            direction /= norm_val
            candidate = np.array([-direction[1], direction[0]])
            midpoint = (p1_2d + p2_2d) / 2.0
            vec = midpoint - centroid
            if np.dot(candidate, vec) < 0:
                candidate = -candidate
            result.append((p1_2d, p2_2d, midpoint, candidate))
    return result

def get_ordered_red_segments_for_path(path_subset, plane_origin, radial_dir, vertical_dir, threshold=-0.1):
    """
    对非闭合路径（path_subset），先利用 order_nonclosed_path 得到节点顺序，
    然后构造相邻节点间的线段（计算中点和法向量），筛选出法向量 y 分量小于 threshold 的红色线段，
    返回按原连接顺序排列的红色线段列表。
    """
    ordered_nodes = order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir)
    red_segments = []
    if len(ordered_nodes) < 2:
        return red_segments
    for i in range(len(ordered_nodes) - 1):
        p1 = np.array(ordered_nodes[i])
        p2 = np.array(ordered_nodes[i+1])
        p1_2d = project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
        p2_2d = project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
        direction = p2_2d - p1_2d
        norm_val = np.linalg.norm(direction)
        if norm_val == 0:
            continue
        direction /= norm_val
        normal_vec = np.array([-direction[1], direction[0]])
        midpoint = (p1_2d + p2_2d) / 2.0
        if normal_vec[1] < threshold:
            red_segments.append((p1_2d, p2_2d, midpoint, normal_vec))
    return red_segments

def group_red_segments_by_connection_order(ordered_red_segments, tol=0.1):
    """
    对传入的红色线段列表（已按子路径连接顺序排列）先逆序，
    然后遍历逆序后的线段：对于每个相邻线段，计算前一线段的终止点 x 值（较大值）
    与后一线段的起始点 x 值（较小值）之差 gap，若 gap < tol 则归为同一组，否则开启新组。
    返回分组后的列表，每组为一个红色线段子集。
    """
    if not ordered_red_segments:
        return []
    reversed_segments = ordered_red_segments[::-1]
    groups = []
    current_group = [reversed_segments[0]]
    for seg in reversed_segments[1:]:
        prev_seg = current_group[-1]
        prev_end_x = max(prev_seg[0][0], prev_seg[1][0])
        curr_start_x = min(seg[0][0], seg[1][0])
        gap = prev_end_x - curr_start_x
        if gap < tol:
            current_group.append(seg)
        else:
            groups.append(current_group)
            current_group = [seg]
    if current_group:
        groups.append(current_group)
    return groups

def convert_2d_points_to_3d(points_2d, plane_origin, radial_dir, vertical_dir):
    """
    根据测剖面所在平面坐标系，将二维点 (u,v) 转换为三维点:
      p_3d = plane_origin + u * radial_dir + v * vertical_dir
    """
    points_3d = []
    for p in points_2d:
        p3d = plane_origin + p[0] * radial_dir + p[1] * vertical_dir
        points_3d.append(p3d)
    return np.array(points_3d)

def convert_groups_to_3d_points(groups_by_path, plane_origin, radial_dir, vertical_dir):
    """
    对每个路径的每组红色线段，将组内所有线段的两个端点取并集（去重）后，
    利用 convert_2d_points_to_3d 转换为三维点，返回字典，键为路径编号，
    值为列表，每一组为对应的三维点数组。
    """
    groups_3d = {}
    for path_idx, groups in groups_by_path.items():
        groups_3d[path_idx] = []
        for group in groups:
            pts = []
            for seg in group:
                pts.append(seg[0])
                pts.append(seg[1])
            pts_arr = np.array(pts)
            pts_arr = np.round(pts_arr, decimals=6)
            unique_pts = np.unique(pts_arr, axis=0)
            pts_3d = convert_2d_points_to_3d(unique_pts, plane_origin, radial_dir, vertical_dir)
            groups_3d[path_idx].append(pts_3d)
    return groups_3d

# ---------------- 整理绘图函数 ----------------
def plot_original_subpaths(path_subsets, plane_origin, radial_dir, vertical_dir):
    """
    绘制原始子路径展示（每条子路径使用不同颜色），
    并对非闭合子路径按照节点顺序标记第一个节点（红色）和最后一个节点（绿色）
    """

    plt.figure(figsize=(6,6))
    path_cmap = plt.cm.get_cmap("Set1", len(path_subsets))
    for i, subset in enumerate(path_subsets):
        color = path_cmap(i)

        # 绘制每条子路径的所有线段
        for edge in subset:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            p1_2d = project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], c=color, linewidth=2)
        # 对非闭合子路径，利用 order_nonclosed_path 得到节点顺序
        ordered_nodes = order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir)
        if len(ordered_nodes) >= 2:
            first_node = ordered_nodes[0]
            last_node = ordered_nodes[-1]
            first_2d = project_to_plane([first_node], plane_origin, radial_dir, vertical_dir)[0]
            last_2d = project_to_plane([last_node], plane_origin, radial_dir, vertical_dir)[0]
            plt.scatter(first_2d[0], first_2d[1], c="red", s=50, marker="o", label="起始点" if i==0 else "")
            plt.scatter(last_2d[0], last_2d[1], c="green", s=50, marker="o", label="终止点" if i==0 else "")
    plt.title("原始子路径展示")
    plt.xlabel("径向方向 /m")
    plt.ylabel("垂直方向 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.legend()
    plt.show()

def plot_merged_paths_and_normals(merged_paths, merged_normals, path_subsets, plane_origin, radial_dir, vertical_dir):
    """
    绘制两幅子图：
      左侧为原始子路径背景（灰色显示），
      右侧为合并后路径及其法向量（红色箭头表示）。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    # 左侧：绘制原始子路径背景
    axes[0].set_title("原始子路径背景")
    axes[0].set_xlabel("径向方向 /m")
    axes[0].set_ylabel("垂直方向 /m")
    for subset in path_subsets:
        for edge in subset:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            p1_2d = project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            axes[0].plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color='gray', linewidth=1)
    axes[0].set_aspect('equal', adjustable='datalim')
    
    # 右侧：绘制合并后路径及法向量
    axes[1].set_title("合并后路径及法向量")
    axes[1].set_xlabel("径向方向 /m")
    axes[1].set_ylabel("垂直方向 /m")
    for idx, m_path in enumerate(merged_paths):
        m_path = np.array(m_path)
        m_path_2d = project_to_plane(m_path, plane_origin, radial_dir, vertical_dir)
        axes[1].plot(m_path_2d[:,0], m_path_2d[:,1], linewidth=2, label=f"合并路径 {idx+1}")
        axes[1].scatter(m_path_2d[:,0], m_path_2d[:,1], s=20)
        normals = merged_normals.get(idx, [])
        for (p1_2d, p2_2d, midpoint, normal) in normals:
            scale = 0.2
            axes[1].arrow(midpoint[0], midpoint[1], normal[0]*scale, normal[1]*scale,
                          head_width=0.05, head_length=0.1, fc='red', ec='red')
    axes[1].legend()
    axes[1].set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()

# ---------------- 合并路径函数 ----------------
def merge_open_paths_new(path_subsets, plane_origin, radial_dir, vertical_dir):
    """
    合并非闭合子路径：
    1. 对每个非闭合子路径利用 order_nonclosed_path 得到节点序列，并计算其在二维平面上的纵坐标范围，
       按起始点纵坐标从高到低排序，构造 open_paths 列表。
    2. 从 open_paths 中依次选取起始点最高的子路径作为合并起点，利用延伸逻辑（基于零点定理和线性插值）
       逐步延伸当前合并路径，直到无法再延伸。
    3. 每次合并结束后，根据当前合并路径在二维平面上的纵坐标范围，
       删除 open_paths 中完全被包含的子路径，再重新排序后继续处理。
    4. 返回一个列表，每个元素为一条合并后的3D节点序列，与原来存储开路径集的数据格式一致。
    """
    open_paths = []
    for subset in path_subsets:
        ordered_nodes = order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir)
        if len(ordered_nodes) < 2:
            continue
        proj = project_to_plane(ordered_nodes, plane_origin, radial_dir, vertical_dir)
        y_min = np.min(proj[:, 1])
        y_max = np.max(proj[:, 1])
        open_paths.append({
            "nodes": ordered_nodes,
            "proj": proj,
            "start2d": proj[0],
            "end2d": proj[-1],
            "y_range": (y_min, y_max)
        })
    open_paths.sort(key=lambda p: p["start2d"][1], reverse=True)
    
    if not open_paths:
        print("没有足够的非闭合子路径")
        return []
    
    merged_paths = []
    while open_paths:
        current = open_paths.pop(0)
        merged_path = current["nodes"][:]  # 复制节点序列
        
        while True:
            P = merged_path[-1]
            P2d = project_to_plane([P], plane_origin, radial_dir, vertical_dir)[0]
            Py = P2d[1]
            candidates = []
            for idx, path in enumerate(open_paths):
                y_min, y_max = path["y_range"]
                if y_min <= Py <= y_max:
                    candidates.append((idx, path))
            if not candidates:
                break
            if len(candidates) > 1:
                candidates.sort(key=lambda x: x[1]["end2d"][1])
                chosen_idx, chosen = candidates[0]
            else:
                chosen_idx, chosen = candidates[0]
            rel = chosen["proj"] - P2d
            pair_y_index = None
            for i in range(len(rel) - 1):
                if rel[i][1] * rel[i+1][1] < 0:
                    pair_y_index = i
                    break
            pair_x_index = None
            for i in range(len(rel) - 1):
                if rel[i][0] * rel[i+1][0] < 0:
                    pair_x_index = i
                    break
            proj_point_y = None
            proj_point_x = None
            M_y = None
            M_x = None
            if pair_y_index is not None:
                q = rel[pair_y_index]
                w = rel[pair_y_index + 1]
                d = w - q
                t_y = -np.dot(q, d) / (np.dot(d, d) + 1e-8)
                proj_rel_y = q + t_y * d
                proj_point_y = P2d + proj_rel_y
                M_y = chosen["proj"][pair_y_index + 1]
            if pair_x_index is not None:
                e = rel[pair_x_index]
                r = rel[pair_x_index + 1]
                d = r - e
                t_x = -np.dot(e, d) / (np.dot(d, d) + 1e-8)
                proj_rel_x = e + t_x * d
                proj_point_x = P2d + proj_rel_x
                M_x = chosen["proj"][pair_x_index + 1]
            if proj_point_x is not None:
                dist_y = np.linalg.norm(P2d - proj_point_y) if proj_point_y is not None else np.inf
                dist_x = np.linalg.norm(P2d - proj_point_x)
                if dist_y <= dist_x:
                    o = proj_point_y
                    M = M_y
                    index_used = pair_y_index + 1
                else:
                    o = proj_point_x
                    M = M_x
                    index_used = pair_x_index + 1
            else:
                if proj_point_y is not None:
                    o = proj_point_y
                    M = M_y
                    index_used = pair_y_index + 1
                else:
                    break
            o_3d = convert_2d_points_to_3d([o], plane_origin, radial_dir, vertical_dir)[0]
            merged_path[-1] = o_3d
            additional_nodes = chosen["nodes"][index_used:]
            if len(additional_nodes) == 0:
                open_paths.pop(chosen_idx)
            else:
                merged_path.extend(additional_nodes)
                open_paths.pop(chosen_idx)
        merged_paths.append(merged_path)
        merged_proj = project_to_plane(merged_path, plane_origin, radial_dir, vertical_dir)
        merged_y_min = np.min(merged_proj[:, 1])
        merged_y_max = np.max(merged_proj[:, 1])
        new_open_paths = []
        for path in open_paths:
            y_min, y_max = path["y_range"]
            if y_min < merged_y_min or y_max > merged_y_max:
                new_open_paths.append(path)
        open_paths = new_open_paths
        open_paths.sort(key=lambda p: p["start2d"][1], reverse=True)
    
    return merged_paths

def compute_normals_for_merged_path(merged_path, plane_origin, radial_dir, vertical_dir):
    """
    对合并后的路径（有序3D节点序列）计算二维法向量，
    返回列表，每个元素为 (起点投影, 终点投影, 线段中点, 法向量2D)。
    """
    merged_path = np.array(merged_path)
    path_2d = project_to_plane(merged_path, plane_origin, radial_dir, vertical_dir)
    normals = []
    for i in range(len(path_2d) - 1):
        p1 = path_2d[i]
        p2 = path_2d[i+1]
        direction = p2 - p1
        norm_val = np.linalg.norm(direction)
        if norm_val == 0:
            continue
        direction /= norm_val
        normal_vec = np.array([-direction[1], direction[0]])
        midpoint = (p1 + p2) / 2.0
        normals.append((p1, p2, midpoint, normal_vec))
    return normals

def load_arc_config_full(config_path):
    """
    加载配置文件，返回圆心、半径、角度范围、弧长范围及Z轴范围。
    配置文件中角度以弧度制给出，转换方法如下：
      - 原始角度范围：[angle_min, angle_max]
      - 弧长范围：以圆弧左端点为原点，右端点的弧长为 radius*(angle_max - angle_min)
    """
    with open(config_path, 'r') as f:
        arc_config = json.load(f)
    center_2d = np.array(arc_config["center"], dtype=float)
    center_3d = np.append(center_2d, 0.0)
    radius = float(arc_config["radius"])
    angle_min = arc_config["angle_min"]
    angle_max = arc_config["angle_max"]
    # 计算整个圆弧的弧长
    arc_length = radius * (angle_max - angle_min)
    arc_length_range = [0, arc_length]
    z_min, z_max = arc_config["z_range"]
    return center_3d, radius, angle_min, angle_max, arc_length_range, z_min, z_max

def main():
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config_full(str(get_path("arc_config")))

    mesh_start_time = time.time()
    mesh = trimesh.load_mesh(str(get_path("mesh_model")))
    mesh_end_time = time.time()
    print(f"网格加载时间: {mesh_end_time - mesh_start_time:.6f} 秒")
    
    # 2. 对网格进行Z轴裁剪
    mesh_in_range = filter_mesh_by_z_range(mesh, z_min, z_max)
    if mesh_in_range is None:
        print("警告：在给定Z范围内无三角面！")
        return

    # 3. 计算剖切平面
    plane_origin, plane_normal, radial_dir, slice_angle = compute_plane(center, radius, angle_min, arc_length_range, 44.2)
    near_tol = 0.1

    # 4. 切割网格
    near_points, intersections = slice_mesh(mesh_in_range, plane_origin, plane_normal, near_tol)
    print(f"近似剖面点数: {len(near_points)}")
    if intersections is not None:
        print(f"交线线段数: {len(intersections)}")
    else:
        print("无交线")
    
    # 步骤1：合并交线段端点并统计连接数
    unique_nodes = merge_intersection_nodes(intersections, tol=1e-6)
    print(f"合并后的唯一节点数: {len(unique_nodes)}")
    node_conn = compute_node_connectivity(intersections, tol=1e-6)
    conn_count = {}
    for count in node_conn.values():
        conn_count[count] = conn_count.get(count, 0) + 1
    print("节点连接线段统计结果:")
    for k in sorted(conn_count.keys()):
        print(f"连接{k}个线段的节点数量: {conn_count[k]}")
    
    # 步骤2：将交线段分为多个路径子集
    path_subsets = split_segments_to_paths(intersections, tol=1e-6)
    print(f"分离出 {len(path_subsets)} 条路径子集。")
    for i, subset in enumerate(path_subsets):
        print(f"路径子集 {i+1} 包含 {len(subset)} 个线段。")
    
    # 定义平面坐标系：x轴为径向方向，y轴为垂直方向
    vertical_dir = np.array([0, 0, 1])
    
    # 在合并子集之前，先调用绘图函数展示原始子路径，同时标记每个非闭合子路径的起始点（红色）和终止点（绿色）
    plot_original_subpaths(path_subsets, plane_origin, radial_dir, vertical_dir)
    
    # 使用合并算法得到合并后的路径（仅处理非闭合子路径），格式与原来一致
    merged_paths = merge_open_paths_new(path_subsets, plane_origin, radial_dir, vertical_dir)
    
    # 基于合并后的路径计算法向量
    merged_normals = {}
    for idx, m_path in enumerate(merged_paths):
        normals = compute_normals_for_merged_path(m_path, plane_origin, radial_dir, vertical_dir)
        merged_normals[idx] = normals
    
    # 调用函数绘制合并后路径及法向量
    plot_merged_paths_and_normals(merged_paths, merged_normals, path_subsets, plane_origin, radial_dir, vertical_dir)
    
    # 后续可以基于合并后的路径和法向量进行红色线段筛选、分组、转换为3D点和3D展示等操作

if __name__ == "__main__":
    main()
