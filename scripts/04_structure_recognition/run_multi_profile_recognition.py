"""Multi-profile structural recognition workflow.

Groups and analyzes structural features across multiple digital scanline profiles.
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
from matplotlib.font_manager import FontProperties
import math
from collections import deque
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import time
from tqdm import tqdm
import concurrent.futures
import os
import gc
from memory_profiler import profile
import tracemalloc
import psutil
# 设置字体
font_en = FontProperties(family='Arial', size=12)
font_en_13 = FontProperties(family='Arial', size=13)
font_zh = FontProperties(fname=str(get_path("font_zh")), size=12)
font_zh_13 = FontProperties(fname=str(get_path("font_zh")), size=13)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 几何与数据处理工具
# =============================================================================
class ArcUtils:
    """封装几何计算、网格切割、路径分割及节点提取等工具函数"""
    @staticmethod
    def load_arc_config(config_path):
        with open(config_path, 'r') as f:
            arc_config = json.load(f)
        center_2d = np.array(arc_config["center"], dtype=float)
        center_3d = np.append(center_2d, 0.0)
        radius = float(arc_config["radius"])
        angle_min = arc_config["angle_min"]
        angle_max = arc_config["angle_max"]
        arc_length = radius * (angle_max - angle_min)
        arc_length_range = [0, arc_length]
        z_min, z_max = arc_config["z_range"]
        return center_3d, radius, angle_min, angle_max, arc_length_range, z_min, z_max

    @staticmethod
    def compute_plane(center, radius, angle_min, arc_length_range, slice_position):
        if slice_position is None:
            slice_position = arc_length_range[1] / 2
        slice_angle = angle_min + slice_position / radius
        radial_dir = np.array([np.cos(slice_angle), np.sin(slice_angle), 0.0])
        radial_dir /= np.linalg.norm(radial_dir)
        normal = np.cross(radial_dir, [0, 0, 1])
        normal /= np.linalg.norm(normal)
        return center, normal, radial_dir, slice_angle

    @staticmethod
    def filter_mesh_by_z_range(mesh, z_min, z_max):
        face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
        face_min_z = face_vertices_z.min(axis=1)
        face_max_z = face_vertices_z.max(axis=1)
        face_mask = (face_max_z >= z_min) & (face_min_z <= z_max)
        submeshes = mesh.submesh([face_mask], only_watertight=False)
        return submeshes[0] if len(submeshes) > 0 else None

    @staticmethod
    def slice_mesh(mesh, origin, normal):
        lines_3d, face_indices = trimesh.intersections.mesh_plane(
            mesh=mesh,
            plane_normal=normal,
            plane_origin=origin,
            return_faces=True
        )
        segments = []
        for i in range(len(lines_3d)):
            seg = lines_3d[i]
            fidx = face_indices[i]
            segments.append((seg[0], seg[1], fidx))
        return segments

    @staticmethod
    def project_to_plane(points_3d, origin, axis_x, axis_y):
        out = []
        for p in points_3d:
            vec = p - origin
            u = np.dot(vec, axis_x)
            v = np.dot(vec, axis_y)
            out.append([u, v])
        return np.array(out)

    @staticmethod
    def merge_intersection_nodes(intersections, tol=1e-6):
        if intersections is None or len(intersections) == 0:
            return np.empty((0, 3))
        nodes = []
        for seg in intersections:
            p1, p2, _ = seg
            nodes.append(p1)
            nodes.append(p2)
        nodes = np.array(nodes)
        nodes_rounded = np.round(nodes, decimals=6)
        unique_nodes = np.unique(nodes_rounded, axis=0)
        return unique_nodes

    @staticmethod
    def compute_node_connectivity(intersections, tol=1e-6):
        node_conn = {}
        for seg in intersections:
            p1, p2, _ = seg
            key1 = tuple(np.round(p1, decimals=6))
            key2 = tuple(np.round(p2, decimals=6))
            node_conn[key1] = node_conn.get(key1, 0) + 1
            node_conn[key2] = node_conn.get(key2, 0) + 1
        return node_conn

    @staticmethod
    def split_segments_to_paths(intersections, tol=1e-6):
        graph = {}
        edges = []
        for seg in intersections:
            p1, p2, fidx = seg
            p1_r = tuple(np.round(p1, decimals=6))
            p2_r = tuple(np.round(p2, decimals=6))
            edges.append((p1_r, p2_r, fidx))
            graph.setdefault(p1_r, set()).add(p2_r)
            graph.setdefault(p2_r, set()).add(p1_r)
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
            subset = []
            for e in edges:
                p1_r, p2_r, fidx = e
                if p1_r in comp and p2_r in comp:
                    subset.append((p1_r, p2_r, fidx))
            components.append(subset)
        return components

    @staticmethod
    def get_edges_from_nodes(path_nodes_3d):
        edges = []
        for i in range(len(path_nodes_3d) - 1):
            p1 = tuple(path_nodes_3d[i])
            p2 = tuple(path_nodes_3d[i + 1])
            edges.append((p1, p2, None))
        return edges

    @staticmethod
    def is_closed_subset(path_subset, node_conn):
        nodes = set()
        for edge in path_subset:
            p1, p2, _ = edge
            nodes.add(p1)
            nodes.add(p2)
        for node in nodes:
            if node_conn.get(node, 0) == 1:
                return False
        return True

    @staticmethod
    def order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir):
        graph = {}
        for edge in path_subset:
            p1, p2, _ = edge
            graph.setdefault(p1, set())
            graph.setdefault(p2, set())
            graph[p1].add(p2)
            graph[p2].add(p1)
        all_nodes = list(graph.keys())
        nodes_arr = [np.array(n) for n in all_nodes]
        nodes_2d = ArcUtils.project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
        y_values = [pt[1] for pt in nodes_2d]
        start_node = all_nodes[y_values.index(max(y_values))]
        end_node = all_nodes[y_values.index(min(y_values))]
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
                    queue.append(path + [neighbor])
        if path_found is None:
            print("无法找到从y值最大到y值最小的路径")
            return []
        return path_found

    @staticmethod
    def compute_normals_with_edges_for_subset(path_subset, closed, plane_origin, radial_dir, vertical_dir):
        result = []
        if not closed:
            ordered_nodes = ArcUtils.order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir)
            if len(ordered_nodes) < 2:
                return result
            for i in range(len(ordered_nodes) - 1):
                p1 = np.array(ordered_nodes[i])
                p2 = np.array(ordered_nodes[i + 1])
                p1_2d = ArcUtils.project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
                p2_2d = ArcUtils.project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
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
                p1, p2, _ = edge
                nodes.add(p1)
                nodes.add(p2)
            nodes_arr = np.array([list(n) for n in nodes])
            nodes_2d = ArcUtils.project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
            centroid = np.mean(nodes_2d, axis=0)
            for edge in path_subset:
                p1, p2, _ = edge
                p1_2d = ArcUtils.project_to_plane([np.array(p1)], plane_origin, radial_dir, vertical_dir)[0]
                p2_2d = ArcUtils.project_to_plane([np.array(p2)], plane_origin, radial_dir, vertical_dir)[0]
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

    @staticmethod
    def merge_open_paths_new(path_subsets, plane_origin, radial_dir, vertical_dir, tol=1e-6):
        open_paths = []
        for subset in path_subsets:
            ordered_nodes = ArcUtils.order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir)
            if len(ordered_nodes) < 2:
                continue
            proj = ArcUtils.project_to_plane(ordered_nodes, plane_origin, radial_dir, vertical_dir)
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
        merged_paths_nodes = []
        while open_paths:
            current = open_paths.pop(0)
            merged_nodes = current["nodes"][:]
            while True:
                P = merged_nodes[-1]
                P2d = ArcUtils.project_to_plane([np.array(P)], plane_origin, radial_dir, vertical_dir)[0]
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
                if pair_y_index is not None:
                    q = rel[pair_y_index]
                    w = rel[pair_y_index + 1]
                    d = w - q
                    t_y = -np.dot(q, d) / (np.dot(d, d) + 1e-8)
                    proj_rel_y = q + t_y * d
                    proj_point_y = P2d + proj_rel_y
                if pair_x_index is not None:
                    e = rel[pair_x_index]
                    r = rel[pair_x_index + 1]
                    d = r - e
                    t_x = -np.dot(e, d) / (np.dot(d, d) + 1e-8)
                    proj_rel_x = e + t_x * d
                    proj_point_x = P2d + proj_rel_x
                if proj_point_x is not None:
                    dist_y = np.linalg.norm(P2d - proj_point_y) if proj_point_y is not None else np.inf
                    dist_x = np.linalg.norm(P2d - proj_point_x)
                    if dist_y <= dist_x:
                        o = proj_point_y
                        index_used = pair_y_index + 1
                    else:
                        o = proj_point_x
                        index_used = pair_x_index + 1
                else:
                    if proj_point_y is not None:
                        o = proj_point_y
                        index_used = pair_y_index + 1
                    else:
                        break
                o_3d = ArcUtils.convert_2d_points_to_3d([o], plane_origin, radial_dir, vertical_dir)[0]
                if isinstance(o_3d, np.ndarray):
                    o_3d = tuple(o_3d.tolist())
                merged_nodes.append(o_3d)
                additional_nodes = chosen["nodes"][index_used:]
                merged_nodes.extend(additional_nodes)
                open_paths.pop(chosen_idx)
            merged_paths_nodes.append(merged_nodes)
            new_open_paths = []
            for path in open_paths:
                y_min, y_max = path["y_range"]
                proj_nodes = ArcUtils.project_to_plane(path["nodes"], plane_origin, radial_dir, vertical_dir)
                if np.min(proj_nodes[:, 1]) < np.min(ArcUtils.project_to_plane(merged_nodes, plane_origin, radial_dir, vertical_dir)) \
                or np.max(proj_nodes[:, 1]) > np.max(ArcUtils.project_to_plane(merged_nodes, plane_origin, radial_dir, vertical_dir)):
                    new_open_paths.append(path)
            open_paths = new_open_paths
            open_paths.sort(key=lambda p: p["start2d"][1], reverse=True)
        def assign_face_idx(p1, p2, subsets, tol=tol):
            for subset in subsets:
                for edge in subset:
                    q1 = np.array(edge[0])
                    q2 = np.array(edge[1])
                    if (np.allclose(p1, q1, atol=tol) and np.allclose(p2, q2, atol=tol)) or \
                    (np.allclose(p1, q2, atol=tol) and np.allclose(p2, q1, atol=tol)):
                        return edge[2]
            return None
        merged_paths_edges = []
        for nodes in merged_paths_nodes:
            edges = []
            for i in range(len(nodes) - 1):
                p1 = nodes[i]
                p2 = nodes[i+1]
                fidx = assign_face_idx(p1, p2, path_subsets, tol=tol)
                edges.append((p1, p2, fidx))
            merged_paths_edges.append(edges)
        return merged_paths_edges

    @staticmethod
    def _extract_ordered_edges(ordered_nodes, subset):
        edges_in_order = []
        subset_map = {}
        for (a, b, fidx) in subset:
            subset_map.setdefault(tuple(a), {})[tuple(b)] = fidx
            subset_map.setdefault(tuple(b), {})[tuple(a)] = fidx
        for i in range(len(ordered_nodes) - 1):
            pA = tuple(ordered_nodes[i])
            pB = tuple(ordered_nodes[i + 1])
            if pA in subset_map and pB in subset_map[pA]:
                fidx = subset_map[pA][pB]
                edge = (pA, pB, fidx)
            elif pB in subset_map and pA in subset_map[pB]:
                fidx = subset_map[pB][pA]
                edge = (pA, pB, fidx)
            else:
                return None
            edges_in_order.append(edge)
        return edges_in_order

    @staticmethod
    def get_ordered_red_segments_for_path(path_subset, plane_origin, radial_dir, vertical_dir, threshold=-0.1, tol_match=1e-6):
        ordered_nodes = ArcUtils.order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir)
        red_segments = []
        if len(ordered_nodes) < 2:
            return red_segments
        for i in range(len(ordered_nodes) - 1):
            p1 = np.array(ordered_nodes[i])
            p2 = np.array(ordered_nodes[i + 1])
            p1_2d = ArcUtils.project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = ArcUtils.project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            direction = p2_2d - p1_2d
            norm_val = np.linalg.norm(direction)
            if norm_val == 0:
                continue
            direction /= norm_val
            normal_vec = np.array([-direction[1], direction[0]])
            if normal_vec[1] < threshold:
                mid_2d = (p1_2d + p2_2d) / 2.0
                face_idx_found = None
                for edge in path_subset:
                    q1 = np.array(edge[0])
                    q2 = np.array(edge[1])
                    if (np.allclose(p1, q1, atol=tol_match) and np.allclose(p2, q2, atol=tol_match)) or \
                    (np.allclose(p1, q2, atol=tol_match) and np.allclose(p2, q1, atol=tol_match)):
                        face_idx_found = edge[2]
                        break
                red_segments.append((p1_2d, p2_2d, mid_2d, normal_vec, p1, p2, face_idx_found))
        return red_segments

    @staticmethod
    def group_red_segments_by_connection_order(ordered_red_segments, tol=0.1, y_tol=0.5):
        """
        简化的分组逻辑，根据相邻线段是否相连、是否具有一定“生长空间”来分组。
        这里仅使用2D投影判断连接性，不严格处理face_idx。

        新增打断机制：在扩展路径时，
        如果当前候选线段的末端（用该线段的第一个端点表示）与种子线段的起始端（用种子线段的第二个端点表示）
        构成的向量与X轴的夹角超过45度，则终止当前路径的生长，开始下一个生长。
        """
        n = len(ordered_red_segments)
        grouped_flags = [False] * n
        reversed_indices = list(range(n - 1, -1, -1))

        def is_growth_allowed(segment):
            """
            用线段方向的法向量与X轴的夹角进行简单判断；
            示例函数，不影响主流程。
            """
            dx = segment[1][0] - segment[0][0]
            dy = segment[1][1] - segment[0][1]
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm == 0:
                return False
            angle = math.degrees(math.acos(abs(nx) / norm))
            return angle > 40

        def connected(seg1, seg2):
            """
            判断两个线段是否相连。
            这里以seg1的第一个端点和seg2的第二个端点为判断依据。
            """
            seg1_end_x = seg1[0][0]
            seg2_start_x = seg2[1][0]
            seg1_end_y = seg1[0][1]
            seg2_start_y = seg2[1][1]
            return (seg2_start_x - seg1_end_x > tol) and (abs(seg1_end_y - seg2_start_y) < y_tol)

        # 挑选允许生长的起点索引（倒序）
        growth_points = [i for i in reversed_indices if is_growth_allowed(ordered_red_segments[i])]
        groups = []
        while growth_points:
            seed_idx = growth_points[0]
            group_indices = []
            stack = [seed_idx]
            # 记录种子线段的“起始端”，这里约定使用种子线段的第二个端点作为起始端
            group_start = ordered_red_segments[seed_idx][1]
            while stack:
                idx = stack.pop()
                if grouped_flags[idx]:
                    continue
                grouped_flags[idx] = True
                group_indices.append(idx)
                # 检查能否扩展至前一个线段
                if idx - 1 >= 0 and connected(ordered_red_segments[idx], ordered_red_segments[idx - 1]):
                    candidate_end = ordered_red_segments[idx - 1][0]
                    vec = np.array(candidate_end) - np.array(group_start)
                    # 计算向量与X轴的夹角（取绝对值，保证比较正角度）
                    angle = math.degrees(math.atan2(abs(vec[1]), abs(vec[0])))
                    if angle <= 70:
                        stack.append(idx - 1)
                    else:
                        # 如果候选扩展的角度超过45度，则终止当前生长
                        break
            group_indices.sort(reverse=True)
            group = [ordered_red_segments[i] for i in group_indices]
            groups.append(group)
            growth_points = [i for i in growth_points if not grouped_flags[i]]
        return groups

    @staticmethod
    def convert_2d_points_to_3d(points_2d, plane_origin, radial_dir, vertical_dir):
        points_3d = []
        for p in points_2d:
            p3d = plane_origin + p[0] * radial_dir + p[1] * vertical_dir
            points_3d.append(p3d)
        return np.array(points_3d)

    @staticmethod
    def correct_group_for_segments(parent_index, parent_path_subset, group_segments,
                                   plane_origin, radial_dir, vertical_dir, tol=1e-6):
        parent_order = ArcUtils.order_nonclosed_path(parent_path_subset, plane_origin, radial_dir, vertical_dir)
        if not parent_order:
            return None
        parent_order_2d = [
            tuple(np.round(ArcUtils.project_to_plane([np.array(node)], plane_origin, radial_dir, vertical_dir)[0], decimals=6))
            for node in parent_order
        ]
        group_nodes = set()
        for seg in group_segments:
            node1 = tuple(np.round(seg[0], decimals=6))
            node2 = tuple(np.round(seg[1], decimals=6))
            group_nodes.add(node1)
            group_nodes.add(node2)
        indices = [i for i, node in enumerate(parent_order_2d) if node in group_nodes]
        if not indices:
            return None
        start_idx = min(indices)
        end_idx = max(indices)
        corrected_nodes = parent_order[start_idx:end_idx + 1]
        connectivity = {
            node: (1 if i == 0 or i == len(corrected_nodes) - 1 else 2)
            for i, node in enumerate(corrected_nodes)
        }
        return {
            "parent_index": parent_index,
            "node_order": corrected_nodes,
            "nodes": list(set(corrected_nodes)),
            "connectivity": connectivity,
            "group_segments": group_segments
        }

    @staticmethod
    def filter_corrected_groups_by_x_distance(results, min_distance=0.1):
        filtered = {}
        origin = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        for parent_idx, groups in results.get("red_groups_corrected", {}).items():
            filtered_groups = []
            for group in groups:
                node_order = group["node_order"]
                if len(node_order) < 2:
                    continue
                node_order_array = np.array(node_order)
                node_order_2d = ArcUtils.project_to_plane(node_order_array, origin, radial_dir, vertical_dir)
                x_distance = abs(node_order_2d[-1, 0] - node_order_2d[0, 0])
                if x_distance >= min_distance:
                    filtered_groups.append(group)
            if filtered_groups:
                filtered[parent_idx] = filtered_groups
        results["FloatingSurface"] = filtered
        return results

    @staticmethod
    def compute_2d_centroids(results):
        red_groups_corrected = results.get("red_groups_corrected", {})
        origin = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        red_centroids = {}
        for parent_idx, groups in red_groups_corrected.items():
            centroids = []
            for group in groups:
                node_order = group.get("node_order", [])
                if len(node_order) < 1:
                    continue
                points_2d = ArcUtils.project_to_plane(np.array(node_order), origin, radial_dir, vertical_dir)
                centroid = np.mean(points_2d, axis=0)
                centroids.append(centroid)
            red_centroids[parent_idx] = centroids
        results["red_centroids"] = red_centroids
        return results

# =============================================================================
# 2. 核心剖面切割器
# =============================================================================
class ArcSlicer:
    def __init__(self, res, use_merged_paths=False):
        self.results = res
        self.use_merged_paths = use_merged_paths

    def load_data(self):
        lines_3d = self.results["slicing"]["lines_3d"]
        face_ids = self.results["slicing"]["face_ids"]
        intersections = []
        for i in range(len(lines_3d)):
            seg = lines_3d[i]
            fidx = face_ids[i]
            intersections.append((seg[0], seg[1], fidx))
        self.results["slicing"] = {"intersections": intersections}

    def extract_nodes(self):
        intersections = self.results["slicing"]["intersections"]
        unique_nodes = ArcUtils.merge_intersection_nodes(intersections)
        node_conn = ArcUtils.compute_node_connectivity(intersections)
        self.results["nodes"] = {"unique_nodes": unique_nodes, "node_connectivity": node_conn}

    def segment_paths(self):
        intersections = self.results["slicing"]["intersections"]
        unmerged_subsets = ArcUtils.split_segments_to_paths(intersections)
        self.results["paths"] = {"unmerged_subsets": unmerged_subsets}

    def merge_open_paths(self):
        origin = self.results["plane_params"]["origin"]
        radial_dir = self.results["plane_params"]["radial_dir"]
        vertical_dir = self.results["plane_params"]["vertical_dir"]
        unmerged_subsets = self.results["paths"]["unmerged_subsets"]
        merged_paths = ArcUtils.merge_open_paths_new(unmerged_subsets, origin, radial_dir, vertical_dir)
        self.results["paths"]["merged_open_subsets"] = merged_paths

    def compute_normals(self):
        origin = self.results["plane_params"]["origin"]
        radial_dir = self.results["plane_params"]["radial_dir"]
        vertical_dir = self.results["plane_params"]["vertical_dir"]
        node_conn = self.results["nodes"]["node_connectivity"]
        if self.use_merged_paths:
            path_subsets = self.results["paths"]["merged_open_subsets"]
        else:
            path_subsets = self.results["paths"]["unmerged_subsets"]
        normals_info = []
        for subset in path_subsets:
            closed = ArcUtils.is_closed_subset(subset, node_conn)
            normals = ArcUtils.compute_normals_with_edges_for_subset(subset, closed, origin, radial_dir, vertical_dir)
            normals_info.append({"subset": subset, "closed": closed, "normals": normals})
        self.results["normals"] = normals_info

    def extract_and_group_red_segments(self):
        origin = self.results["plane_params"]["origin"]
        radial_dir = self.results["plane_params"]["radial_dir"]
        vertical_dir = self.results["plane_params"]["vertical_dir"]
        if self.use_merged_paths:
            path_subsets = self.results["paths"]["merged_open_subsets"]
        else:
            path_subsets = self.results["paths"]["unmerged_subsets"]
        red_groups = {}
        for idx, subset in enumerate(path_subsets):
            closed = ArcUtils.is_closed_subset(subset, self.results["nodes"]["node_connectivity"])
            if closed:
                continue
            ordered_red = ArcUtils.get_ordered_red_segments_for_path(subset, origin, radial_dir, vertical_dir, threshold=-0.1)
            if ordered_red:
                groups = ArcUtils.group_red_segments_by_connection_order(ordered_red, tol=-0.3, y_tol=1)
                red_groups[idx] = groups
        self.results["red_groups"] = red_groups

    def correct_red_groups(self, tol=1e-6):
        red_groups = self.results.get("red_groups", {})
        corrected = {}
        if self.use_merged_paths:
            path_subsets = self.results["paths"]["merged_open_subsets"]
        else:
            path_subsets = self.results["paths"]["unmerged_subsets"]
        origin = self.results["plane_params"]["origin"]
        radial_dir = self.results["plane_params"]["radial_dir"]
        vertical_dir = self.results["plane_params"]["vertical_dir"]
        for parent_index, groups in red_groups.items():
            corrected[parent_index] = []
            if parent_index >= len(path_subsets):
                continue
            for group in groups:
                correction = ArcUtils.correct_group_for_segments(
                    parent_index, path_subsets[parent_index], group,
                    origin, radial_dir, vertical_dir, tol
                )
                if correction is not None:
                    corrected[parent_index].append(correction)
        self.results["red_groups_corrected"] = corrected

    def run_all(self):
        self.load_data()
        self.extract_nodes()
        self.segment_paths()
        # 若不需要合并 open paths，则注释下一行
        # self.merge_open_paths()
        self.compute_normals()
        self.extract_and_group_red_segments()
        self.correct_red_groups()
        self.results["use_merged_paths"] = self.use_merged_paths
        return self.results

# =============================================================================
# 多剖面处理及结果保存（主函数）
# =============================================================================

def process_single_slice(res):
    slicer = ArcSlicer(res, use_merged_paths=False)
    results = slicer.run_all()
    results = ArcUtils.compute_2d_centroids(results)
    return results

def convert_to_line_segments_format(processed_batch):
    all_line_segments_data = []
    for res in processed_batch:
        slice_data = {}
        slice_data["slice_key"] = res.get("slice_key", None)
        slice_data["slice_angle"] = res["plane_params"]["slice_angle"]
        slice_data["plane_params"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in res["plane_params"].items()
        }
        if "paths" in res and "merged_open_subsets" in res["paths"]:
            merged_subsets = res["paths"]["merged_open_subsets"]
            new_merged = []
            for subset in merged_subsets:
                new_subset = []
                for edge in subset:
                    new_edge = [list(edge[0]), list(edge[1])]
                    new_subset.append(new_edge)
                new_merged.append(new_subset)
            slice_data["merged_open_subsets"] = new_merged
        if "red_centroids" in res:
            red_centroids = res["red_centroids"]
            new_centroids = {}
            for key, centroids in red_centroids.items():
                new_centroids[str(key)] = [centroid.tolist() if isinstance(centroid, np.ndarray) else centroid
                                             for centroid in centroids]
            slice_data["red_centroids"] = new_centroids
        segments = []
        for parent_idx, groups in res.get("red_groups_corrected", {}).items():
            for group in groups:
                seg = {}
                seg["node_order"] = [list(node) for node in group["node_order"]]
                new_conn = {}
                for k, v in group["connectivity"].items():
                    new_conn[str(k)] = v
                seg["connectivity"] = new_conn
                segments.append(seg)
        slice_data["line_segments"] = segments
        all_line_segments_data.append(slice_data)
    return all_line_segments_data

def process_slice_block(slice_block):
    """
    接收一小块（比如50个）切面数据，串行处理并返回结果列表。
    """
    results = []
    for slice_data in slice_block:
        res = process_single_slice(slice_data)
        results.append(res)
    return results


def main():
    file_path = str(get_path("slice_output", create_parent=True))
    file_path = str(get_path("slice_output", create_parent=True))
    # file_path = str(get_path("slice_output", create_parent=True))
    output_file = str(get_path("multi_profile_output", create_parent=True))
    output_file = str(get_path("multi_profile_output", create_parent=True))
    # output_file = str(get_path("multi_profile_output", create_parent=True))
    # 读取所有切面数据
    with open(file_path, "rb") as f:
        all_slices = pickle.load(f)

    # 准备待处理切面
    desired_positions = np.arange(0,140, 0.05)

    precomputed_results = []
    for pos in desired_positions:
        
        key = f"{pos:.2f}"
        if key in all_slices:
            slice_result = all_slices[key]
            slice_result["slice_key"] = key
            precomputed_results.append(slice_result)

    print(f"前处理完毕，共 {len(precomputed_results)} 个切面待处理")

    # 如果 output_file 存在，删除以防止本次写入和旧内容混合
    if os.path.exists(output_file):
        os.remove(output_file)

    start_time = time.time()

    # 可以根据需要灵活调整下列参数
    num_processes = 15   # 一次并行多少个进程
    sub_task_size = 40   # 每个进程要串行处理多少个切面
    batch_size = num_processes * sub_task_size  # 每批处理的切面总数

    total_slices = len(precomputed_results)

    # 用 tqdm 设置一个全局进度条，统计总共处理了多少切面
    global_progress = tqdm(total=total_slices, desc="总体进度")

    # 分批次处理
    for batch_start in range(0, total_slices, batch_size):
        # 取出本批需要处理的切面
        batch_data = precomputed_results[batch_start: batch_start + batch_size]

        # 按 sub_task_size 划分给每个进程处理
        blocks = []
        for i in range(0, len(batch_data), sub_task_size):
            block_slice = batch_data[i: i + sub_task_size]
            blocks.append(block_slice)

        # 并行处理本批数据
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            block_results_iter = executor.map(process_slice_block, blocks)
            # 收集并写入结果
            with open(output_file, "ab") as f_out:
                for block_res in block_results_iter:
                    # block_res 是这个子块里所有切面的处理结果(list)
                    # 写入文件
                    for single_res in block_res:
                        pickle.dump(single_res, f_out)
                    # 更新全局进度条
                    global_progress.update(len(block_res))

        # 释放已写入的数据
        del blocks

    global_progress.close()
    end_time = time.time()

    print(f"所有切面处理完毕，总用时: {end_time - start_time:.2f} 秒")
    print(f"结果已分段写入到: {output_file}")


    # 如果需要在最后一次性将所有结果合并成一个List再存一次，下面给出一个示例：
    # （需要先把上面分段写入的数据都读回来）
    # final_results = []
    # with open(output_file, "rb") as f_in:
    #     while True:
    #         try:
    #             item = pickle.load(f_in)
    #             final_results.append(item)
    #         except EOFError:
    #             break
    #
    # # 此时 final_results 就是全部结果的合集，如果需要可以再单独按你的格式存一次
    # with open(output_file, "wb") as f_out:
    #     pickle.dump(final_results, f_out)
    # print("已合并所有结果到一个列表，重新写入完成。")

if __name__ == "__main__":
    main()
