"""Normal-vector and path-analysis workflow.

Computes ordered profile paths, local vectors, and overhanging-structure indicators.
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 创建实际的箭头元素，注意 mutation_scale 控制箭头大小
        p = FancyArrowPatch((xdescent, ydescent + height/2),
                            (xdescent + width, ydescent + height/2),
                            transform=trans,
                            color=orig_handle.get_facecolor(),
                            arrowstyle='->',
                            mutation_scale=15,
                            lw=1)
        return [p]

# 设置字体
font_en = FontProperties(family='Times New Roman', size=12)
font_en_13 = FontProperties(family='Times New Roman', size=13)
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
        # 设正北方向为 [0,1]，使用 np.arctan2(radial_dir[0], radial_dir[1])
        azimuth_rad = np.arctan2(radial_dir[0], radial_dir[1])
        azimuth_deg = np.degrees(azimuth_rad) % 360
        azimuth_deg = round(azimuth_deg, 3)
        return center, normal, radial_dir, slice_angle, azimuth_deg

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
        """
        返回一个列表，每个元素格式为 (p1_3d, p2_3d, face_idx)。
        p1_3d 和 p2_3d 为线段两端点坐标，face_idx 为与该线段对应的面索引。
        """
        lines_3d, face_indices = trimesh.intersections.mesh_plane(
            mesh=mesh,
            plane_normal=normal,
            plane_origin=origin,
            return_faces=True
        )
        # 将 (端点, face_index) 合并存储在一起
        segments = []
        for i in range(len(lines_3d)):
            seg = lines_3d[i]
            fidx = face_indices[i]
            # seg[0] 和 seg[1] 分别是线段两端点 (x, y, z)
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
        """
        注意：这里只对线段端点做去重，并没有记录面索引。
        若需要端点级别的面索引，需要在此做自定义逻辑。
        """
        if intersections is None or len(intersections) == 0:
            return np.empty((0, 3))
        nodes = []
        for seg in intersections:
            p1, p2, face_idx = seg
            nodes.append(p1)
            nodes.append(p2)
        nodes = np.array(nodes)
        nodes_rounded = np.round(nodes, decimals=6)
        unique_nodes = np.unique(nodes_rounded, axis=0)
        return unique_nodes

    @staticmethod
    def compute_node_connectivity(intersections, tol=1e-6):
        """
        仅计算节点被连接次数；并不区分来自哪个面索引。
        """
        node_conn = {}
        for seg in intersections:
            p1, p2, face_idx = seg
            key1 = tuple(np.round(p1, decimals=6))
            key2 = tuple(np.round(p2, decimals=6))
            node_conn[key1] = node_conn.get(key1, 0) + 1
            node_conn[key2] = node_conn.get(key2, 0) + 1
        return node_conn

    @staticmethod
    def split_segments_to_paths(intersections, tol=1e-6):
        """
        将线段集拆分成若干连通子集。
        每条边的格式为 (p1, p2, face_idx)，p1、p2已经做了round(6)处理。
        """
        graph = {}
        edges = []
        for seg in intersections:
            p1, p2, face_idx = seg
            p1_r = tuple(np.round(p1, decimals=6))
            p2_r = tuple(np.round(p2, decimals=6))
            # 存储时也把面索引带上
            edges.append((p1_r, p2_r, face_idx))

            # 建立无向图
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
            # comp 是一个连通分量内的所有节点
            # 把 edges 中 属于该连通分量的拿出来
            subset = []
            for e in edges:
                p1_r, p2_r, face_idx = e
                if p1_r in comp and p2_r in comp:
                    subset.append((p1_r, p2_r, face_idx))
            components.append(subset)

        return components

    @staticmethod
    def get_edges_from_nodes(path_nodes_3d):
        """
        由于人工合并时，我们只知道节点顺序，因此此处无法对应原始 mesh 面索引。
        故这里统一将 face_idx 设置为 None。
        """
        edges = []
        for i in range(len(path_nodes_3d) - 1):
            p1 = tuple(path_nodes_3d[i])
            p2 = tuple(path_nodes_3d[i + 1])
            edges.append((p1, p2, None))  # face_idx = None
        return edges

    @staticmethod
    def is_closed_subset(path_subset, node_conn):
        """
        path_subset 中每个元素: (p1, p2, face_idx)
        """
        nodes = set()
        for edge in path_subset:
            p1, p2, fidx = edge
            nodes.add(p1)
            nodes.add(p2)
        for node in nodes:
            if node_conn.get(node, 0) == 1:
                return False
        return True

    @staticmethod
    def order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir, return_edges=False):
        """
        参数:
        path_subset: 每个元素为 (p1, p2, face_idx) 的边列表
        plane_origin, radial_dir, vertical_dir: 用于投影的参数
        return_edges: 若为 True，则返回按从 top->bottom 排序好的边列表，
                        每条边格式为 (p1, p2, face_idx)；否则返回节点列表(3D 坐标)
                        
        返回:
        如果 return_edges=False，返回节点列表；
        如果 return_edges=True，返回边列表，边是根据节点路径重构得来。
        """
        from collections import deque
        import numpy as np

        # 构建无向图，利用 p1, p2 作为图的节点
        graph = {}
        for edge in path_subset:
            p1, p2, face_idx = edge
            graph.setdefault(p1, set()).add(p2)
            graph.setdefault(p2, set()).add(p1)

        # 获取所有节点及其在 2D 投影中的 y 值
        all_nodes = list(graph.keys())
        nodes_arr = [np.array(n) for n in all_nodes]
        nodes_2d = ArcUtils.project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
        y_values = [pt[1] for pt in nodes_2d]

        # 取 2D 投影中 y 值最大的点作为起点，y 值最小的点作为终点
        start_node = all_nodes[y_values.index(max(y_values))]
        end_node = all_nodes[y_values.index(min(y_values))]

        # BFS 搜索从 start_node 到 end_node 的一条路径
        queue = deque([[start_node]])
        path_found = None
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == end_node:
                path_found = path
                break
            # 此处不使用全局 visited，而是在每条路径中检查，避免遗漏不同分支
            for neighbor in graph[current]:
                if neighbor not in path:
                    queue.append(path + [neighbor])
        if path_found is None:
            print("无法找到从 y 值最大到 y 值最小的路径")
            return []

        if not return_edges:
            return path_found
        else:
            # 按节点路径构造有序边序列
            ordered_edges = []
            for i in range(len(path_found) - 1):
                node_a = path_found[i]
                node_b = path_found[i + 1]
                selected_edge = None
                # 从原始的边列表中查找连接 node_a 与 node_b 的边（忽略方向）
                for edge in path_subset:
                    p1, p2, face_idx = edge
                    if (p1 == node_a and p2 == node_b) or (p1 == node_b and p2 == node_a):
                        selected_edge = edge
                        break
                # 若未找到对应边，则认为是 bridging edge，face_idx 设置为 None
                if selected_edge is None:
                    ordered_edges.append((node_a, node_b, None))
                else:
                    ordered_edges.append(selected_edge)
            return ordered_edges

    @staticmethod
    def compute_normals_with_edges_for_subset(path_subset, closed, plane_origin, radial_dir, vertical_dir):
        """
        返回值中，存储每条边在 2D 投影下的中点及法向信息等。
        """
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
            # 闭合情况
            nodes = set()
            for edge in path_subset:
                p1, p2, face_idx = edge
                nodes.add(p1)
                nodes.add(p2)
            nodes_arr = np.array([list(n) for n in nodes])
            nodes_2d = ArcUtils.project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
            centroid = np.mean(nodes_2d, axis=0)
            for edge in path_subset:
                p1, p2, face_idx = edge
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
        """
        优化后的版本：
        - 合并过程仅在2D平面下进行，避免重复的3D/2D转换；
        - 最终在所有合并完成后一次性将2D路径转换为3D；
        - 对于 bridging edge（衔接边），其 face_idx 设置为 None。
        
        返回值：一个列表，每个元素为一组边，每条边为 (p1, p2, face_idx)。
        """
        # 1) 预处理：对每个子路径计算有序的2D点序列，并保存对应的原始3D数据，仅供face_idx匹配时参考
        open_paths = []
        for subset in path_subsets:
            ordered_nodes = ArcUtils.order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir)
            if len(ordered_nodes) < 2:
                continue
            proj = ArcUtils.project_to_plane(ordered_nodes, plane_origin, radial_dir, vertical_dir)
            y_min = np.min(proj[:, 1])
            y_max = np.max(proj[:, 1])
            open_paths.append({
                "proj": proj,              # 用于合并计算的2D点序列
                "nodes": ordered_nodes,    # 原始3D数据，仅作 face_idx 匹配参考
                "start2d": proj[0],
                "end2d": proj[-1],
                "y_range": (y_min, y_max)
            })
        open_paths.sort(key=lambda p: p["start2d"][1], reverse=True)
        if not open_paths:
            print("没有足够的非闭合子路径")
            return []

        # 2) 在2D上进行合并操作
        # merged_paths_2d 保存每一条合并的2D路径，同时保留一份原始3D数据记录，
        # 方便后续判断是否为 bridging edge
        merged_paths_2d = []  # 每个元素为 (merged_2d, merged_3d_original)
        while open_paths:
            current = open_paths.pop(0)
            # 初始化合并序列
            merged_2d = current["proj"].tolist()           # 2D点列表
            merged_3d_original = current["nodes"][:]         # 3D点列表，记录原始值
            while True:
                P2d = merged_2d[-1]
                Py = P2d[1]
                # 选择满足 y_range 要求的候选路径
                candidates = []
                for idx, path in enumerate(open_paths):
                    y_min, y_max = path["y_range"]
                    if y_min <= Py <= y_max:
                        candidates.append((idx, path))
                if not candidates:
                    break
                # 若有多个候选，则按 end2d 的 y 值排序，选择排序最靠前的
                if len(candidates) > 1:
                    candidates.sort(key=lambda x: x[1]["end2d"][1])
                    chosen_idx, chosen = candidates[0]
                else:
                    chosen_idx, chosen = candidates[0]

                # 计算当前末端点与候选路径之间的相对向量，并检测 x 或 y 分量交叉
                rel = chosen["proj"] - np.array(P2d)
                pair_y_index = None
                pair_x_index = None
                for i in range(len(rel) - 1):
                    if rel[i][1] * rel[i + 1][1] < 0:
                        pair_y_index = i
                        break
                for i in range(len(rel) - 1):
                    if rel[i][0] * rel[i + 1][0] < 0:
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
                    proj_point_y = np.array(P2d) + proj_rel_y
                if pair_x_index is not None:
                    e = rel[pair_x_index]
                    r = rel[pair_x_index + 1]
                    d = r - e
                    t_x = -np.dot(e, d) / (np.dot(d, d) + 1e-8)
                    proj_rel_x = e + t_x * d
                    proj_point_x = np.array(P2d) + proj_rel_x

                if proj_point_x is not None:
                    dist_y = np.linalg.norm(np.array(P2d) - proj_point_y) if proj_point_y is not None else np.inf
                    dist_x = np.linalg.norm(np.array(P2d) - proj_point_x)
                    if dist_y <= dist_x:
                        o = proj_point_y.tolist()
                        index_used = pair_y_index + 1
                    else:
                        o = proj_point_x.tolist()
                        index_used = pair_x_index + 1
                else:
                    if proj_point_y is not None:
                        o = proj_point_y.tolist()
                        index_used = pair_y_index + 1
                    else:
                        break

                # 此处 o 为 bridging 的2D点，直接替换当前合并序列的末端;
                # 对应的3D数据置为 None，表示这条边为 bridging edge
                merged_2d[-1] = o
                merged_3d_original[-1] = None
                # 追加候选路径中剩余的2D及3D数据
                additional_2d = chosen["proj"][index_used:].tolist()
                additional_3d = chosen["nodes"][index_used:]
                merged_2d.extend(additional_2d)
                merged_3d_original.extend(additional_3d)
                open_paths.pop(chosen_idx)
            merged_paths_2d.append((merged_2d, merged_3d_original))
            # 更新 open_paths，移除那些已被合并的路径（通过 y 值判断）
            new_open_paths = []
            merged_arr = np.array(merged_2d)
            for path in open_paths:
                if np.min(path["proj"][:, 1]) < np.min(merged_arr[:, 1]) or \
                np.max(path["proj"][:, 1]) > np.max(merged_arr[:, 1]):
                    new_open_paths.append(path)
            open_paths = new_open_paths
            open_paths.sort(key=lambda p: p["start2d"][1], reverse=True)

        # 3) 将合并后的2D路径一次性转换为3D，然后为每个边分配 face_idx
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

        for merged_2d, merged_3d_original in merged_paths_2d:
            # 仅在所有合并结束后将2D路径转换为3D
            merged_3d = ArcUtils.convert_2d_points_to_3d(merged_2d, plane_origin, radial_dir, vertical_dir)
            edges = []
            for i in range(len(merged_3d) - 1):
                p1 = merged_3d[i]
                p2 = merged_3d[i + 1]
                # 若原始3D数据中有对应的点，则尝试匹配 face_idx；否则视作 bridging edge
                if merged_3d_original[i] is not None and merged_3d_original[i + 1] is not None:
                    fidx = assign_face_idx(p1, p2, path_subsets, tol=tol)
                else:
                    fidx = None
                edges.append((p1, p2, fidx))
            merged_paths_edges.append(edges)
        
        return merged_paths_edges

    @staticmethod
    def merge_open_paths_new(path_subsets, plane_origin, radial_dir, vertical_dir, tol=1e-6):
        """
        合并多个非闭合子路径：
        - 合并操作均在 2D 平面上完成，bridging edge 对应的 face_idx 置为 None；
        - 返回格式：列表，每个元素为一组边，每条边格式为 (p1, p2, face_idx)；
        - p1、p2 为 3D 坐标。
        其中：
        1. 对每个子路径（边列表），先调用 ArcUtils.order_nonclosed_path 获取有序边序列，
            再将边转换为点序列，每个点直接绑定其 face 信息，
            具体为：对于边 (p1, p2, face_idx)，将 p1 作为起始点（face 为第一条边的 face_idx），
            然后依次记录每条边的终点 p2 并绑定该边的 face_idx，最后一个点因无后续边，其 face 置为 None。
        2. 合并过程在 2D 平面上进行，当需要衔接时，直接将候选路径的首个点作为 bridging 点，
            并将其 face 信息置为 None，然后将候选路径剩余部分附加到当前路径后。
        3. 遍历合并后的点序列生成边时，如果相邻两点的 face 信息一致且不为 None，则认为该边保留原始 face，
            否则置为 bridging edge（face=None）。
        """
        import numpy as np

        # ------------------------------
        # Step 1: 将边列表转换为点序列，并直接绑定 face 信息
        # ------------------------------
        open_paths = []
        for subset in path_subsets:
            # 使用 ArcUtils.order_nonclosed_path 得到有序边序列
            # 这里要求支持 return_edges=True，返回 ordered_edges 为 [(p1, p2, face_idx), ...]
            ordered_edges = ArcUtils.order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir, return_edges=True)
            if not ordered_edges:
                continue

            points = []
            # 取第一条边的起点，记录其 3D 与 2D 坐标，face 取第一条边的 face_idx
            first_edge = ordered_edges[0]
            pt3d = first_edge[0]
            pt2d = ArcUtils.project_to_plane([pt3d], plane_origin, radial_dir, vertical_dir)[0].tolist()
            points.append({"pt3d": pt3d, "pt2d": pt2d, "face": first_edge[2]})
            
            # 对每条边依次记录终点及其 face 信息
            for edge in ordered_edges:
                pt3d = edge[1]
                pt2d = ArcUtils.project_to_plane([pt3d], plane_origin, radial_dir, vertical_dir)[0].tolist()
                points.append({"pt3d": pt3d, "pt2d": pt2d, "face": edge[2]})
            # 最后一个点无后续边，face 置为 None
            points[-1]["face"] = None

            ys = [p["pt2d"][1] for p in points]
            open_paths.append({
                "points": points,               # 点序列，每个点为 dict {pt3d, pt2d, face}
                "start_y": points[0]["pt2d"][1],
                "end_y": points[-1]["pt2d"][1],
                "y_range": (min(ys), max(ys))
            })
        open_paths.sort(key=lambda p: p["start_y"], reverse=True)
        if not open_paths:
            print("没有足够的非闭合子路径")
            return []

        # ------------------------------
        # Step 2: 在2D平面上合并路径
        # ------------------------------
        # 此处全部采用 points 数据，不再拆分为 proj 和 nodes。
        merged_paths = []  # 每个元素为点序列（列表，每个元素为字典：{pt3d, pt2d, face}）
        while open_paths:
            current = open_paths.pop(0)
            # 初始化合并序列：直接复制当前路径的点列表
            merged_points = current["points"][:]
            while True:
                # 取当前合并路径末尾点的 2D 坐标
                P2d = merged_points[-1]["pt2d"]
                Py = P2d[1]
                # 选择候选路径：其中的 y_range 包含当前末端点的 y 值
                candidates = []
                for idx, path in enumerate(open_paths):
                    y_min, y_max = path["y_range"]
                    if y_min <= Py <= y_max:
                        candidates.append((idx, path))
                if not candidates:
                    break
                # 若有多个候选，则按候选路径末端的 y 值排序，取最小者
                candidates.sort(key=lambda x: x[1]["end_y"])
                chosen_idx, chosen = candidates[0]

                # 将候选路径的点列表转换为 2D 坐标数组
                candidate_pts_2d = np.array([pt["pt2d"] for pt in chosen["points"]])
                # 计算当前末端点与候选路径所有点的相对向量
                rel = candidate_pts_2d - np.array(P2d)
                pair_y_index = None
                pair_x_index = None
                # 检测 y 分量交叉
                for i in range(len(rel) - 1):
                    if rel[i][1] * rel[i + 1][1] < 0:
                        pair_y_index = i
                        break
                # 检测 x 分量交叉
                for i in range(len(rel) - 1):
                    if rel[i][0] * rel[i + 1][0] < 0:
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
                    proj_point_y = np.array(P2d) + proj_rel_y
                if pair_x_index is not None:
                    e = rel[pair_x_index]
                    r = rel[pair_x_index + 1]
                    d = r - e
                    t_x = -np.dot(e, d) / (np.dot(d, d) + 1e-8)
                    proj_rel_x = e + t_x * d
                    proj_point_x = np.array(P2d) + proj_rel_x

                if proj_point_x is not None:
                    dist_y = np.linalg.norm(np.array(P2d) - proj_point_y) if proj_point_y is not None else np.inf
                    dist_x = np.linalg.norm(np.array(P2d) - proj_point_x)
                    if dist_y <= dist_x:
                        o = proj_point_y.tolist()
                        index_used = pair_y_index + 1
                    else:
                        o = proj_point_x.tolist()
                        index_used = pair_x_index + 1
                else:
                    if proj_point_y is not None:
                        o = proj_point_y.tolist()
                        index_used = pair_y_index + 1
                    else:
                        break

                # 将 bridging 点 o 更新到合并路径的末端，同时将对应 3D 数据置为 None 表示新衔接
                merged_points[-1]["pt2d"] = o
                pt3d_new = ArcUtils.convert_2d_points_to_3d([o], plane_origin, radial_dir, vertical_dir)[0]
                merged_points[-1]["pt3d"] = tuple(pt3d_new)
                # 将候选路径中 index_used 之后的点追加到合并路径上
                additional_points = chosen["points"][index_used:]
                merged_points.extend(additional_points)
                open_paths.pop(chosen_idx)
            merged_paths.append(merged_points)
            # 更新 open_paths：移除与当前合并路径在 y 值上重叠的路径
            new_open_paths = []
            merged_ys = [pt["pt2d"][1] for pt in merged_points]
            merged_min, merged_max = min(merged_ys), max(merged_ys)
            for path in open_paths:
                pts_y = [pt["pt2d"][1] for pt in path["points"]]
                if min(pts_y) < merged_min or max(pts_y) > merged_max:
                    new_open_paths.append(path)
            open_paths = new_open_paths
            open_paths.sort(key=lambda p: p["start_y"], reverse=True)

        # ------------------------------
        # Step 3: 根据合并后的点序列生成边
        # ------------------------------
        # 生成边时：遍历相邻两个点，
        # 若两点绑定的 face 相同且不为 None，则该边保留原始 face；否则视为 bridging edge，face 置为 None
        merged_paths_edges = []
        for points in merged_paths:
            edges = []
            for i in range(len(points) - 1):
                p1 = points[i]["pt3d"]
                p2 = points[i+1]["pt3d"]
                if points[i]["face"] is not None and points[i]["face"] == points[i+1]["face"]:
                    face_idx = points[i]["face"]
                else:
                    face_idx = None
                edges.append((p1, p2, face_idx))
            merged_paths_edges.append(edges)
        
        return merged_paths_edges



    @staticmethod
    def _extract_ordered_edges(ordered_nodes, subset):
        """
        根据 ordered_nodes 的顺序，从 subset(无序线段) 中
        找到连续的 (p1, p2, face_idx)，保证 p1->p2 正序。
        如果某一段没找到匹配，可能返回不完整。
        """
        edges_in_order = []
        # subset 也是 (p1, p2, face_idx)
        # 先做一个索引，方便查找
        subset_map = {}
        for (a, b, fidx) in subset:
            subset_map.setdefault(tuple(a), {})[tuple(b)] = fidx
            subset_map.setdefault(tuple(b), {})[tuple(a)] = fidx

        # 遍历 ordered_nodes，相邻构造线段
        for i in range(len(ordered_nodes) - 1):
            pA = tuple(ordered_nodes[i])
            pB = tuple(ordered_nodes[i + 1])
            # 看看 subset_map 是否有记录
            if pA in subset_map and pB in subset_map[pA]:
                fidx = subset_map[pA][pB]
                edge = (pA, pB, fidx)
            elif pB in subset_map and pA in subset_map[pB]:
                # 反向
                fidx = subset_map[pB][pA]
                edge = (pA, pB, fidx)
            else:
                # 没有匹配到（可能已被合并或出现其它断点）
                return None
            edges_in_order.append(edge)
        return edges_in_order

    @staticmethod
    def get_ordered_red_segments_for_path(path_subset, plane_origin, radial_dir, vertical_dir, threshold=-0.4, tol_match=1e-6):
        """
        遍历子路径(非闭合)的有序节点，选出法向在Y方向小于某阈值的片段，
        作为“红色线段”。同时返回对应的 2D 和 3D 信息以及该段原生的 face_idx。

        返回的元组格式为：
        (p1_2d, p2_2d, mid_2d, normal_vec, p1_3d, p2_3d, face_idx)
        
        其中：
        - p1_2d, p2_2d: 该线段在 2D 投影下的起始端点
        - mid_2d: 2D 中点
        - normal_vec: 2D 法向量
        - p1_3d, p2_3d: 对应的原始 3D 端点
        - face_idx: 从 path_subset 中匹配得到的面索引，若没有匹配则为 None
        """
        # 先利用已有函数得到有序的 3D 节点序列
        ordered_nodes = ArcUtils.order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir)
        red_segments = []
        if len(ordered_nodes) < 2:
            return red_segments

        for i in range(len(ordered_nodes) - 1):
            # 原始3D端点
            p1 = np.array(ordered_nodes[i])
            p2 = np.array(ordered_nodes[i + 1])
            # 投影到2D
            p1_2d = ArcUtils.project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = ArcUtils.project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            direction = p2_2d - p1_2d
            norm_val = np.linalg.norm(direction)
            if norm_val == 0:
                continue
            direction /= norm_val
            normal_vec = np.array([-direction[1], direction[0]])
            # 判断是否满足阈值条件
            if normal_vec[1] < threshold:
                mid_2d = (p1_2d + p2_2d) / 2.0
                # 尝试在原始的 path_subset 中匹配这段边，以获得 face_idx
                face_idx_found = None
                for edge in path_subset:
                    q1 = np.array(edge[0])
                    q2 = np.array(edge[1])
                    # 检查是否与 p1, p2 匹配（考虑顺序或反向）
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
                    if angle <= 40:
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
        """
        根据 parent_path_subset 的真实走向，将红段 group_segments 在其上做一次校正。
        注意：group_segments 是 ((x1,y1),(x2,y2),...) 的 2D 投影片段，而 parent_path_subset
        则是 (p1_3d, p2_3d, face_idx) 。
        这里省略 face_idx 的准确处理，仅演示 node 匹配原理。
        """
        parent_order = ArcUtils.order_nonclosed_path(parent_path_subset, plane_origin, radial_dir, vertical_dir)
        if not parent_order:
            return None

        # 将 parent_order 的 3D节点转为 2D并做round
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

        # 找到 group_nodes 在 parent_order_2d 里的最小和最大索引
        indices = [i for i, node in enumerate(parent_order_2d) if node in group_nodes]
        if not indices:
            return None

        start_idx = min(indices)
        end_idx = max(indices)
        corrected_nodes = parent_order[start_idx:end_idx + 1]

        # 连接信息只是示例
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
        """
        仅对校正后的红线段分组计算 2D 质心。
        """
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
# 2. 绘图工具
# =============================================================================

class ArcPlotter3D:
    """
    用于在 PyVista 中可视化 red_groups_corrected 中的线段及其对应的网格面。
    """
    @staticmethod
    def plot_red_groups_corrected_with_faces(results, color="red", line_width=3, face_opacity=0.6):
        """
        使用 PyVista 绘制 red_groups_corrected 中的线段以及它们对应的网格面。
        
        数据直接从 red_groups_corrected 中的 "group_segments" 提取，
        每个 group_segments 元素的格式为：
            (p1_2d, p2_2d, mid_2d, normal_vec, p1, p2, face_idx_found)
            
        其中：
        - p1_2d, p2_2d: 2D 投影坐标
        - mid_2d: 中点坐标
        - normal_vec: 2D 法向量
        - p1, p2: 对应的原始 3D 端点
        - face_idx_found: 匹配到的面索引，如果没有匹配则为 None
        """
        # 1) 获取过滤后的网格，并转换为 PyVista 对象
        filtered_mesh = results["slicing"]["filtered_mesh"]
        if filtered_mesh is None:
            print("未找到有效的过滤后网格！无法绘制面。")
            return
        mesh_pv = pv.wrap(filtered_mesh)
        
        # 2) 创建 PyVista 绘图对象，并添加背景网格
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_pv, color="lightgray", opacity=0.4, show_edges=True)
        
        # 3) 获取 red_groups_corrected 数据
        red_groups_corrected = results.get("red_groups_corrected", {})
        if not red_groups_corrected:
            print("red_groups_corrected 为空，没有可绘制的红线段。")
            return

        # 4) 遍历 red_groups_corrected，从 group_segments 中提取线段及面索引
        all_face_indices = set()
        for parent_idx, groups in red_groups_corrected.items():
            for group in groups:
                group_segments = group.get("group_segments", [])
                for seg in group_segments:
                    # seg 格式: (p1_2d, p2_2d, mid_2d, normal_vec, p1, p2, face_idx_found)
                    p1_2d, p2_2d, mid_2d, normal_vec, p1, p2, face_idx_found = seg
                    # 绘制线段：使用原始的 3D 端点 p1 和 p2
                    segment_points = np.array([p1, p2])
                    lineset = pv.lines_from_points(segment_points)
                    plotter.add_mesh(lineset, color=color, line_width=line_width)
                    
                    # 如果 face_idx_found 有值，则收集
                    if face_idx_found is not None:
                        all_face_indices.add(face_idx_found)
        
        # 5) 如果匹配到面索引，则提取对应的面，并高亮显示
        if all_face_indices:
            face_list = list(all_face_indices)
            face_selection = mesh_pv.extract_cells(face_list)
            if face_selection.n_cells > 0:
                plotter.add_mesh(face_selection, color=color, show_edges=False, opacity=face_opacity)
        
        plotter.add_title("red_groups_corrected线段 + 相应Face展示")
        plotter.show()

class ArcPlotter:
    """封装各种二维和三维结果展示函数"""

    @staticmethod
    def polt_merged_open_subsets(results):
        red_groups_corrected = results.get("red_groups_corrected", {})
        red_centroids = results.get("red_centroids", {})
        origin = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        azimuth = results["plane_params"]['azimuth']

        k = 5
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(k)]
        default_color = cmap(6)
        fig, ax = plt.subplots(figsize=(8, 8))

        merged_subsets = results["paths"].get("merged_open_subsets", [])
        best_subset_nodes = []
        max_nodes = 0
        for subset in merged_subsets:
            nodes = []
            for edge in subset:
                p1, p2, face_idx = edge
                if not nodes:
                    nodes.append(np.array(p1))
                nodes.append(np.array(p2))
            if len(nodes) > max_nodes:
                max_nodes = len(nodes)
                best_subset_nodes = nodes

        if best_subset_nodes:
            best_subset_nodes = np.array(best_subset_nodes)
            points_2d = ArcUtils.project_to_plane(best_subset_nodes, origin, radial_dir, vertical_dir)
            new_origin = np.array([np.min(points_2d[:, 0]) - 5, np.min(points_2d[:, 1]) - 5])
            new_points_2d = points_2d - new_origin


            ax.plot(new_points_2d[:, 0], new_points_2d[:, 1], color="red", linewidth=2, label="完整路径")


            x_min, x_max = np.min(new_points_2d[:, 0]), np.max(new_points_2d[:, 0])
            y_min, y_max = np.min(new_points_2d[:, 1]), np.max(new_points_2d[:, 1])
            ax.set_xticks(np.arange(x_min, x_max + 10, 10))
            ax.set_yticks(np.arange(y_min, y_max + 10, 10))
            ax.grid(True, linestyle="--", color="grey")

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            max_range = max(x_max - x_min, y_max - y_min)
            ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax.set_ylim(y_center - max_range/2 - 5, y_center + max_range/2 + 5)


            # 标注 AB 点，加粗
            A = new_points_2d[0]
            ax.text(A[0]+4, A[1]+2, "崖顶", fontproperties=font_zh_13, ha="center", va="top",
                    color="black", weight="bold")
            B = new_points_2d[-1]
            ax.text(B[0]-4, B[1]-3, "崖脚", fontproperties=font_zh_13, ha="center", va="bottom",
                    color="black", weight="bold")

            # 构造箭头图例句柄（注意 facecolor 设置为黑色）
            arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
            # 添加图例，包含标题
            legend = ax.legend(
                [arrow_legend],
                [f"{azimuth:.1f}°"],
                loc='upper right',
                bbox_to_anchor=(1.0, 1.0),
                handler_map={FancyArrowPatch: HandlerArrow()},
                prop=font_en,
                frameon=True,
                title="剖面方位角"
            )
            # 设置标题字体为中文字体
            legend.get_title().set_fontproperties(font_zh)
            ax.set_xlabel("(m)", fontproperties=font_en)
            ax.set_ylabel("(m)", fontproperties=font_en)
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_en)

            for label in ax.get_yticklabels():
                
                label.set_fontproperties(font_en)
        ax.set_aspect("equal", adjustable="datalim")
        plt.show()

    @staticmethod
    def plot_zoomed_path(points_2d,
                        x_center: float,
                        y_center: float,
                        width: float = 5.0,
                        height: float = 5.0,
                        azimuth: float = None,
                        font_en=font_en,
                        font_zh=font_zh,
                        ax=None):
        """
        绘制一个放大的路径子图，带等比例坐标和刻度网格。
        
        参数：
            points_2d : (N,2) 的 ndarray，二维投影点序列
            x_center, y_center : 显示区域中心点坐标
            width, height : 显示区域的宽度和高度（单位：米）
            azimuth : 若提供，将显示方向箭头图例（可选）
            font_en : 英文字体设置
            font_zh : 中文字体设置（用于图例标题）
            ax : 可选的 matplotlib 子图对象，若不传入自动创建
        """
        #处理points_2d的重新投影

        new_points_2d = points_2d

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # 坐标范围
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        # 绘制完整路径
        ax.plot(new_points_2d[:, 0], new_points_2d[:, 1], color="red", linewidth=2, label="完整路径")
        '''
        # # 正交向量场示意
        # for i in range(len(new_points_2d) - 1):
        #     p_start = new_points_2d[i]
        #     p_end = new_points_2d[i + 1]
        #     vec = p_end - p_start
        #     norm = np.linalg.norm(vec)
        #     if norm == 0:
        #         continue

        #     # 单位方向向量
        #     dir_unit = vec / norm
        #     # 逆时针旋转90度的法向量（未缩放）
        #     normal_vec = np.array([-dir_unit[1], dir_unit[0]])

        #     # 以线段中点为起点
        #     mid_point = (p_start + p_end) / 2
        #     length = 0.2  # 可调：法向量箭头长度
        #     p1 = mid_point
        #     p2 = mid_point + normal_vec * length

        #     arrow = FancyArrowPatch(
        #         posA=p1,
        #         posB=p2,
        #         arrowstyle='->',
        #         mutation_scale=10,
        #         color='blue',
        #         linewidth=1.5
        #     )
        #     ax.add_patch(arrow)
        # 绘制方向向量（不画完整路径线）
        # for i in range(len(new_points_2d) - 1):
        #     p_start = new_points_2d[i]
        #     p_end = new_points_2d[i + 1]
        #     direction = p_end - p_start
        #     norm = np.linalg.norm(direction)
        #     if norm == 0:
        #         continue

        #     # 箭头只在原线段范围内，略微缩短避免重叠
        #     shrink_ratio = 0.15
        #     shrink_len = norm * shrink_ratio
        #     new_start = p_start + direction / norm * shrink_len
        #     new_end = p_end - direction / norm * shrink_len

        #     arrow = FancyArrowPatch(
        #         posA=new_start,
        #         posB=new_end,
        #         arrowstyle='->',
        #         mutation_scale=10,
        #         color='red',
        #         linewidth=1.5
        #     )
        #     ax.add_patch(arrow)

        # # 若提供方向角，添加箭头图例
        # if azimuth is not None:
        #     arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        #     legend = ax.legend(
        #         [arrow_legend],
        #         [f"{azimuth:.1f}°"],
        #         loc='upper right',
        #         bbox_to_anchor=(1.0, 1.0),
        #         handler_map={FancyArrowPatch: HandlerArrow()},
        #         prop=font_en,
        #         frameon=True,
        #         title="剖面方位角"
        #     )
        #     if font_zh:
        #         legend.get_title().set_fontproperties(font_zh)
        '''
        # 设置坐标轴与样式
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(x_min, x_max, 4))
        ax.set_yticks(np.linspace(y_min, y_max, 4))
        # 坐标刻度格式化：保留一位小数
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.grid(True, linestyle="--", color="grey")
        ax.set_xlabel("(m)", fontproperties=font_en)
        ax.set_ylabel("(m)", fontproperties=font_en)

        return ax

    # @staticmethod
    # def polt_merged_open_subsets(results):
    #     red_groups_corrected = results.get("red_groups_corrected", {})
    #     red_centroids = results.get("red_centroids", {})
    #     origin = results["plane_params"]["origin"]
    #     radial_dir = results["plane_params"]["radial_dir"]
    #     vertical_dir = results["plane_params"]["vertical_dir"]
    #     azimuth = results["plane_params"]['azimuth']



    #     merged_subsets = results["paths"].get("merged_open_subsets", [])
    #     best_subset_nodes = []
    #     max_nodes = 0
    #     for subset in merged_subsets:
    #         nodes = []
    #         for edge in subset:
    #             p1, p2, face_idx = edge
    #             if not nodes:
    #                 nodes.append(np.array(p1))
    #             nodes.append(np.array(p2))
    #         if len(nodes) > max_nodes:
    #             max_nodes = len(nodes)
    #             best_subset_nodes = nodes

    #     if best_subset_nodes:
    #         best_subset_nodes = np.array(best_subset_nodes)
    #         points_2d = ArcUtils.project_to_plane(best_subset_nodes, origin, radial_dir, vertical_dir)
    #         new_origin = np.array([np.min(points_2d[:, 0]) - 5, np.min(points_2d[:, 1]) - 5])
    #         points_2d = points_2d - new_origin
    #         # 设定三个放大区域中心点

    #         # ========= 布局 =========
    #         fig = plt.figure(figsize=(10, 8))
    #         gs  = GridSpec(3, 2, width_ratios=[1.5, 1], wspace=0.15, hspace=0.3)
    #         centers = [(26.956, 107.573), (50.04, 61.30), (70.43, 20.57)]
    #         # centers = [(47.6, 69.8), (63.25, 36.06), (78.10, 11.38)]
    #         ax_full = fig.add_subplot(gs[:, 0])
    #         ax_zoom = [fig.add_subplot(gs[i, 1]) for i in range(len(centers))]

    #         # ---------- 通用绘制 ----------
    #         def draw_all(ax):
    #             if points_2d.size:
    #                 ax.plot(points_2d[:,0], points_2d[:,1],
    #                         color="red", lw=2)
    #                         # # 正交向量场示意


    #         # ---------- 左列完整 ----------
    #         draw_all(ax_full)
    #         ax_full.grid(True, ls="--", color="grey")
    #         # 方向箭头图例
    #         x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
    #         y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
    #         ax_full.set_xticks(np.arange(x_min, x_max + 10, 10))
    #         ax_full.set_yticks(np.arange(y_min, y_max + 10, 10))
    #         # === 新的颜色分组图例 ===
    #         x_center = (x_min + x_max) / 2
    #         y_center = (y_min + y_max) / 2
    #         max_range = max(x_max - x_min, y_max - y_min)
    #         ax_full.set_xlim(x_center - max_range/2, x_center + max_range/2)
    #         ax_full.set_ylim(y_center - max_range/2 - 5, y_center + max_range/2 + 5)


    #         # 标注 AB 点，加粗
    #         A = points_2d[0]
    #         ax_full.text(A[0]+4, A[1]+2, "崖顶", fontproperties=font_zh_13, ha="center", va="top",
    #                 color="black", weight="bold")
    #         B = points_2d[-1]
    #         ax_full.text(B[0]-4, B[1]-3, "崖脚", fontproperties=font_zh_13, ha="center", va="bottom",
    #                 color="black", weight="bold")
    #         # 构造箭头图例句柄（注意 facecolor 设置为黑色）
    #         arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
    #         # 添加图例，包含标题
    #         legend = ax_full.legend(
    #             [arrow_legend],
    #             [f"{azimuth:.1f}°"],
    #             loc='upper right',
    #             bbox_to_anchor=(1.0, 1.0),
    #             handler_map={FancyArrowPatch: HandlerArrow()},
    #             prop=font_en,
    #             frameon=True,
    #             title="剖面方位角"
    #         )
    #         # 设置标题字体为中文字体
    #         legend.get_title().set_fontproperties(font_zh)
    #         ax_full.set_aspect("equal", adjustable="datalim")
    #         ax_full.set_xlabel("(m)", fontproperties=font_en_13)
    #         ax_full.set_ylabel("(m)", fontproperties=font_en_13)

    #         # ---------- 右列放大 ----------
    #         for idx, cen in enumerate(centers):
    #             ax = ax_zoom[idx]
    #             draw_all(ax)
    #             for i in range(len(points_2d) - 1):
    #                 p_start = points_2d[i]
    #                 p_end = points_2d[i + 1]
    #                 vec = p_end - p_start
    #                 norm = np.linalg.norm(vec)
    #                 if norm == 0:
    #                     continue

    #                 # 单位方向向量
    #                 dir_unit = vec / norm
    #                 # 逆时针旋转90度的法向量（未缩放）
    #                 normal_vec = np.array([-dir_unit[1], dir_unit[0]])

    #                 # 以线段中点为起点
    #                 mid_point = (p_start + p_end) / 2
    #                 length = 0.3  # 可调：法向量箭头长度
    #                 p1 = mid_point
    #                 p2 = mid_point + normal_vec * length

    #                 # arrow = FancyArrowPatch(
    #                 #     posA=p1,
    #                 #     posB=p2,
    #                 #     arrowstyle='->',
    #                 #     mutation_scale=10,
    #                 #     color='blue',
    #                 #     linewidth=1.5
    #                 # )
    #                 # ax.add_patch(arrow)

    #                 # # 箭头只在原线段范围内，略微缩短避免重叠
    #                 # shrink_ratio = 0.05
    #                 # shrink_len = norm * shrink_ratio
    #                 # new_start = p_start + vec / norm * shrink_len
    #                 # new_end = p_end - vec / norm * shrink_len

    #                 # arrow = FancyArrowPatch(
    #                 #     posA=new_start,
    #                 #     posB=new_end,
    #                 #     arrowstyle='->',
    #                 #     mutation_scale=10,
    #                 #     color='red',
    #                 #     linewidth=1.5
    #                 # )
    #                 # ax.add_patch(arrow)
    #             ArcPlotter.setup_zoomed_axes(points_2d,
    #                                         x_center=cen[0], y_center=cen[1],
    #                                         width=3, height=3,
    #                                         font_en=font_en_13, ax=ax)

    #         plt.tight_layout()
    #         plt.show()    

    # @staticmethod
    # def polt_merged_open_subsets(results):
    #     """
    #     展示 points_2d 在不同旋转角度下的 2×2 子图视图
    #     - 第1幅：原始方向
    #     - 第2幅：旋转90度
    #     - 第3幅：旋转180度
    #     - 第4幅：旋转270度
    #     """
    #     def rotate_points(points, angle_deg, center):
    #         """绕 center 逆时针旋转 angle_deg 角度"""
    #         angle_rad = np.deg2rad(angle_deg)
    #         rot_matrix = np.array([
    #             [np.cos(angle_rad), -np.sin(angle_rad)],
    #             [np.sin(angle_rad),  np.cos(angle_rad)]
    #         ])
    #         return (points - center) @ rot_matrix.T + center
    #     red_groups_corrected = results.get("red_groups_corrected", {})
    #     red_centroids = results.get("red_centroids", {})
    #     origin = results["plane_params"]["origin"]
    #     radial_dir = results["plane_params"]["radial_dir"]
    #     vertical_dir = results["plane_params"]["vertical_dir"]
    #     azimuth = results["plane_params"]['azimuth']

    #     k = 5
    #     cmap = plt.cm.get_cmap("tab10")
    #     colors = [cmap(i) for i in range(k)]
    #     default_color = cmap(6)


    #     merged_subsets = results["paths"].get("merged_open_subsets", [])
    #     best_subset_nodes = []
    #     max_nodes = 0
    #     for subset in merged_subsets:
    #         nodes = []
    #         for edge in subset:
    #             p1, p2, face_idx = edge
    #             if not nodes:
    #                 nodes.append(np.array(p1))
    #             nodes.append(np.array(p2))
    #         if len(nodes) > max_nodes:
    #             max_nodes = len(nodes)
    #             best_subset_nodes = nodes

    #     if best_subset_nodes:
    #         best_subset_nodes = np.array(best_subset_nodes)
    #         points_2d = ArcUtils.project_to_plane(best_subset_nodes, origin, radial_dir, vertical_dir)
    #         new_origin = np.array([np.min(points_2d[:, 0]) - 5, np.min(points_2d[:, 1]) - 5])
    #         points_2d = points_2d - new_origin
    #         center = (62.638, 34.873)

    #         # 设定旋转角度列表
    #         angles = [0, 90, 180, 270]
    #         titles = ["0°", "90°", "180°", "270°"]

    #         fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    #         axs = axs.flatten()

    #         for i in range(4):
    #             rotated = rotate_points(points_2d, angles[i], np.array(center))
    #             ArcPlotter.plot_zoomed_path(rotated,
    #                                         x_center=center[0],
    #                                         y_center=center[1],
    #                                         width=5,
    #                                         height=5,
    #                                         azimuth=azimuth,
    #                                         font_en=font_en_13,
    #                                         font_zh=font_zh_13,
    #                                         ax=axs[i])

    #             axs[i].set_title(titles[i], fontproperties=font_en_13)

    #         plt.tight_layout()
    #         plt.show()

    @staticmethod
    def plot_red_groups_only(results):
        red_groups = results.get("red_groups", {})
        origin = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]

        fig, ax = plt.subplots(figsize=(8, 8))
        all_points = []

        # ======= 第一步：提取 merged_open_subsets（绘制灰色完整路径）=======
        merged_subsets = results["paths"].get("merged_open_subsets", [])
        best_subset_nodes = []
        max_nodes = 0
        for subset in merged_subsets:
            nodes = []
            for edge in subset:
                p1, p2, _ = edge
                if not nodes:
                    nodes.append(np.array(p1))
                nodes.append(np.array(p2))
            if len(nodes) > max_nodes:
                max_nodes = len(nodes)
                best_subset_nodes = nodes

        if best_subset_nodes:
            best_subset_nodes = np.array(best_subset_nodes)
            points_2d = ArcUtils.project_to_plane(best_subset_nodes, origin, radial_dir, vertical_dir)
            all_points.append(points_2d)

        # ======= 第二步：提取 red_groups 中所有线段（p1, p2 是投影后二维坐标）=======
        red_segments = []
        for segment in red_groups:
            p1, p2 = np.array(segment[0]), np.array(segment[1])
            red_segments.append((p1, p2))
            all_points.append(np.vstack([p1, p2]))

        # ======= 第三步：计算统一平移原点 =======
        all_concat = np.vstack(all_points)
        new_origin = np.array([
            np.min(all_concat[:, 0]) - 5,
            np.min(all_concat[:, 1]) - 5
        ])

        # ======= 第四步：绘图 =======

        shifted = points_2d - new_origin
        ax.plot(shifted[:, 0], shifted[:, 1], color="lightgray", linewidth=2, label="完整路径")
        x_min, x_max = np.min(shifted[:, 0]), np.max(shifted[:, 0])
        y_min, y_max = np.min(shifted[:, 1]), np.max(shifted[:, 1])
        ax.set_xticks(np.arange(x_min, x_max + 10, 10))
        ax.set_yticks(np.arange(y_min, y_max + 10, 10))
        ax.grid(True, linestyle="--", color="grey")

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        max_range = max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2 - 5, y_center + max_range/2 + 5)
        # 标注 AB 点，加粗
        A = shifted[0]
        ax.text(A[0]+4, A[1]+2, "崖顶", fontproperties=font_zh_13, ha="center", va="top",
                color="black", weight="bold")
        B = shifted[-1]
        ax.text(B[0]-4, B[1]-3, "崖脚", fontproperties=font_zh_13, ha="center", va="bottom",
                color="black", weight="bold")
        for p1, p2 in red_segments:
            p1_shifted = p1 - new_origin
            p2_shifted = p2 - new_origin
            ax.plot([p1_shifted[0], p2_shifted[0]], [p1_shifted[1], p2_shifted[1]], color="red", linewidth=2)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("(m)", fontproperties=font_en_13)
        ax.set_ylabel("(m)", fontproperties=font_en_13)

        plt.show()
    
    @staticmethod   
    def setup_zoomed_axes(points_2d: np.ndarray,
                        x_center: float,
                        y_center: float,
                        width: float = 5.0,
                        height: float = 5.0,
                        font_en=font_en,
                        ax=None):
        """
        仅设置放大视窗与坐标格式，不主动绘制任何曲线。

        参数
        ----
        points_2d : ndarray
            (N, 2)，用于调整数据显示范围（可提前裁剪）。
        x_center, y_center : float
            视窗中心坐标。
        width, height : float
            视窗宽高（米）。
        font_en : FontProperties
            英文字体；坐标轴标签使用。
        ax : matplotlib.axes.Axes
            目标子图；必须由外部创建并传入。

        返回
        ----
        ax : matplotlib.axes.Axes
            经过设置的子图，便于链式调用。
        """
        if ax is None:
            raise ValueError("请先在外部创建 ax，并作为参数传入！")

        # 视窗范围
        x_min, x_max = x_center - width / 2,  x_center + width / 2
        y_min, y_max = y_center - height / 2, y_center + height / 2

        # 坐标轴设置
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        # 刻度
        ax.set_xticks(np.linspace(x_min, x_max, 4))
        ax.set_yticks(np.linspace(y_min, y_max, 4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # 网格 & 标签
        ax.grid(True, linestyle="--", color="grey")
        ax.set_xlabel("(m)", fontproperties=font_en)
        ax.set_ylabel("(m)", fontproperties=font_en)

        return ax
    
   
    @staticmethod
    def plot_full_with_zoomed_v2(results,
                                centers=[(26.956, 107.573),
                                        (51.2,   62.1),
                                        (72.0,   21.1)],
                                zoom_size=5):
        """
        左列一幅完整图 + 右列三幅放大图（全部包含完整路径与红线段）
        """
        # ---------- 预处理 ----------
        red_groups   = results.get("red_groups", [])
        origin       = results["plane_params"]["origin"]
        radial_dir   = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        azimuth = results["plane_params"]['azimuth']
        # 灰色完整路径
        merged_subsets   = results["paths"].get("merged_open_subsets", [])
        longest_nodes, max_nodes = [], 0
        for subset in merged_subsets:
            nodes = []
            for p1, p2, _ in subset:
                if not nodes:
                    nodes.append(np.array(p1))
                nodes.append(np.array(p2))
            if len(nodes) > max_nodes:
                max_nodes, longest_nodes = len(nodes), nodes
        points_full = ArcUtils.project_to_plane(np.array(longest_nodes),
                                                origin, radial_dir, vertical_dir) if longest_nodes else np.empty((0,2))

        # 红色线段
        red_segments = [(np.array(seg[0]), np.array(seg[1])) for seg in red_groups]

        # 统一平移
        concat_pts = [points_full] + [np.vstack(s) for s in red_segments] if points_full.size else [np.vstack(s) for s in red_segments]
        all_concat = np.vstack(concat_pts)
        new_origin = np.array([all_concat[:,0].min() - 5, all_concat[:,1].min() - 5])
        points_full_shifted = points_full - new_origin
        red_segments_shifted = [(p1-new_origin, p2-new_origin) for p1, p2 in red_segments]

        # ---------- 布局 ----------
        fig = plt.figure(figsize=(10, 8))
        gs  = GridSpec(3, 2, width_ratios=[1.5, 1], wspace=0.15, hspace=0.3)
        ax_full = fig.add_subplot(gs[:, 0])
        ax_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]

        # ---------- 一个通用绘制函数 ----------
        def draw_full_and_segments(ax):
            if points_full_shifted.size:
                ax.plot(points_full_shifted[:,0], points_full_shifted[:,1],
                        color="lightgray", lw=2)
            for p1, p2 in red_segments_shifted:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", lw=2)

        # ---------- 左列完整图 ----------
        draw_full_and_segments(ax_full)
        ax_full.grid(True, ls="--", color="grey")
        x_min, x_max = np.min(points_full_shifted[:, 0]), np.max(points_full_shifted[:, 0])
        y_min, y_max = np.min(points_full_shifted[:, 1]), np.max(points_full_shifted[:, 1])
        ax_full.set_xticks(np.arange(x_min, x_max + 10, 10))
        ax_full.set_yticks(np.arange(y_min, y_max + 10, 10))
        ax_full.grid(True, linestyle="--", color="grey")
        # 构造箭头图例句柄（注意 facecolor 设置为黑色）
        arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        # 添加图例，包含标题
        legend = ax_full.legend(
            [arrow_legend],
            [f"{azimuth:.1f}°"],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            handler_map={FancyArrowPatch: HandlerArrow()},
            prop=font_en_13,
            frameon=True,
            title="剖面方位角"
        )
        legend.get_title().set_fontproperties(font_zh_13)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        max_range = max(x_max - x_min, y_max - y_min)
        ax_full.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax_full.set_ylim(y_center - max_range/2 - 5, y_center + max_range/2 + 5)


        # 标注 AB 点，加粗
        A = points_full_shifted[0]
        ax_full.text(A[0]+4, A[1]+2, "崖顶", fontproperties=font_zh_13, ha="center", va="top",
                color="black", weight="bold")
        B = points_full_shifted[-1]
        ax_full.text(B[0]-4, B[1]-3, "崖脚", fontproperties=font_zh_13, ha="center", va="bottom",
                color="black", weight="bold")
        ax_full.set_xlabel("(m)", fontproperties=font_en_13)
        ax_full.set_ylabel("(m)", fontproperties=font_en_13)
        ax_full.set_aspect("equal", adjustable="datalim")

        # ---------- 右列放大图 ----------
        for idx, (xc, yc) in enumerate(centers):
            ax = ax_zoom[idx]
            # 绘制灰+红
            draw_full_and_segments(ax)
            # 设置放大窗口
            ArcPlotter.setup_zoomed_axes(points_full_shifted,
                            x_center=xc,
                            y_center=yc,
                            width=3,
                            height=3,
                            font_en=font_en_13,
                            ax=ax)

        plt.tight_layout()
        plt.show()
        
    def plot_centroid_groups_with_zoomed(results,
                                        zoom_size=5,
                                        top_n=3):
        """
        左列完整图 + 右列 top_n 个质心放大图
        - 分组线段颜色与原函数一致
        - 质心用同色圆点表示
        """
        # ========= 基本数据 =========
        red_groups_corrected = results.get("red_groups_corrected", {})
        red_centroids        = results.get("red_centroids", {})
        origin, radial_dir, vertical_dir = (results["plane_params"][k]
                                            for k in ("origin", "radial_dir", "vertical_dir"))
        azimuth = results["plane_params"]["azimuth"]

        # ---------- 灰色完整路径 ----------
        merged_subsets = results["paths"].get("merged_open_subsets", [])
        longest_nodes, max_nodes = [], 0
        for subset in merged_subsets:
            nodes = []
            for p1, p2, _ in subset:
                if not nodes: nodes.append(np.array(p1))
                nodes.append(np.array(p2))
            if len(nodes) > max_nodes:
                max_nodes, longest_nodes = len(nodes), nodes
        points_full = ArcUtils.project_to_plane(np.array(longest_nodes),
                                                origin, radial_dir, vertical_dir) if longest_nodes else np.empty((0,2))

        # ---------- 彩色分组 & 质心 ----------
        k = 3
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(k)]
        default_color = cmap(6)

        group_records = []   # [(length, color, pts_2d, centroid), ...]
        for parent_idx, groups in red_groups_corrected.items():
            lengths = [len(g["node_order"]) for g in groups]
            sorted_idx = np.argsort(lengths)[::-1]
            color_map  = {g_idx: colors[r] if r < k else default_color
                        for r, g_idx in enumerate(sorted_idx)}
            centroids_parent = red_centroids.get(parent_idx, [])

            for g_idx, group in enumerate(groups):
                pts_2d = ArcUtils.project_to_plane(np.array(group["node_order"]),
                                                origin, radial_dir, vertical_dir)
                color  = color_map.get(g_idx, default_color)
                centroid = centroids_parent[g_idx] if g_idx < len(centroids_parent) else None
                group_records.append((len(group["node_order"]), color, pts_2d, centroid))

        # ---------- 统一平移 ----------
        all_pts = [points_full] + [r[2] for r in group_records]
        if any(r[3] is not None for r in group_records):
            all_pts += [np.array(r[3]).reshape(1,2) for r in group_records if r[3] is not None]
        all_concat = np.vstack(all_pts)
        new_origin = np.array([all_concat[:,0].min()-5, all_concat[:,1].min()-5])

        points_full_s = points_full - new_origin
        group_records_s = [(L, col, pts - new_origin, (np.array(c)-new_origin) if c is not None else None)
                        for L, col, pts, c in group_records]

        # ---------- 选取 top_n 质心作为放大中心 ----------
        group_records_sorted = sorted(group_records_s, key=lambda x: x[0], reverse=True)
        centers = [rec[3] for rec in group_records_sorted[:top_n] if rec[3] is not None]

        # ========= 布局 =========
        fig = plt.figure(figsize=(10, 8))
        gs  = GridSpec(3, 2, width_ratios=[1.5, 1], wspace=0.15, hspace=0.3)

        ax_full = fig.add_subplot(gs[:, 0])
        ax_zoom = [fig.add_subplot(gs[i, 1]) for i in range(len(centers))]

        # ---------- 通用绘制 ----------
        def draw_all(ax):
            if points_full_s.size:
                ax.plot(points_full_s[:,0], points_full_s[:,1],
                        color="lightgray", lw=2)
            for _, col, pts, cen in group_records_s:
                ax.plot(pts[:,0], pts[:,1], color=col, lw=2)
                # if cen is not None:
                #     ax.scatter(cen[0], cen[1], color=col, s=25, zorder=3)

        # ---------- 左列完整 ----------
        draw_all(ax_full)
        ax_full.grid(True, ls="--", color="grey")
        # 方向箭头图例
        x_min, x_max = np.min(points_full_s[:, 0]), np.max(points_full_s[:, 0])
        y_min, y_max = np.min(points_full_s[:, 1]), np.max(points_full_s[:, 1])
        ax_full.set_xticks(np.arange(x_min, x_max + 10, 10))
        ax_full.set_yticks(np.arange(y_min, y_max + 10, 10))
        # === 新的颜色分组图例 ===
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        max_range = max(x_max - x_min, y_max - y_min)
        ax_full.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax_full.set_ylim(y_center - max_range/2 - 5, y_center + max_range/2 + 5)

        legend_elements = [Line2D([0], [0], color=colors[i], lw=3, label=f'C{i+1}')
                        for i in range(k)]
        legend_elements.append(Line2D([0], [0], color=default_color, lw=3,
                                    label=f'C{k+1}-Cn'))
        legend = ax_full.legend(handles=legend_elements,
                                title="投影结构组\n(按照长度排序)",
                                loc="upper right",
                                prop=font_en_13,
                                frameon=True)
        legend.get_title().set_fontproperties(font_zh_13)
        # 标注 AB 点，加粗
        A = points_full_s[0]
        ax_full.text(A[0]+4, A[1]+2, "崖顶", fontproperties=font_zh_13, ha="center", va="top",
                color="black", weight="bold")
        B = points_full_s[-1]
        ax_full.text(B[0]-4, B[1]-3, "崖脚", fontproperties=font_zh_13, ha="center", va="bottom",
                color="black", weight="bold")

        ax_full.set_aspect("equal", adjustable="datalim")
        ax_full.set_xlabel("(m)", fontproperties=font_en_13)
        ax_full.set_ylabel("(m)", fontproperties=font_en_13)

        # ---------- 右列放大 ----------
        for idx, cen in enumerate(centers):
            ax = ax_zoom[idx]
            draw_all(ax)
            ArcPlotter.setup_zoomed_axes(points_full_s,
                                        x_center=cen[0], y_center=cen[1],
                                        width=zoom_size, height=zoom_size,
                                        font_en=font_en_13, ax=ax)

        plt.tight_layout()
        plt.show()    

    @staticmethod
    def plot_red_groups_with_centroids(results):
        from matplotlib.lines import Line2D
        red_groups_corrected = results.get("red_groups_corrected", {})
        red_groups = results.get("red_groups")
        red_centroids = results.get("red_centroids", {})
        origin = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]

        k = 5
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(k)]
        default_color = cmap(6)
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制完整路径（选取节点数最多的路径）
        merged_subsets = results["paths"].get("merged_open_subsets", [])
        best_subset_nodes = []
        max_nodes = 0
        for subset in merged_subsets:
            nodes = []
            for edge in subset:
                p1, p2, face_idx = edge
                if not nodes:
                    nodes.append(np.array(p1))
                nodes.append(np.array(p2))
            if len(nodes) > max_nodes:
                max_nodes = len(nodes)
                best_subset_nodes = nodes
        if best_subset_nodes:
            best_subset_nodes = np.array(best_subset_nodes)
            points_2d = ArcUtils.project_to_plane(best_subset_nodes, origin, radial_dir, vertical_dir)
            ax.plot(points_2d[:, 0], points_2d[:, 1], color="lightgray", linewidth=2, label="完整路径")

        for parent_idx, groups in red_groups_corrected.items():
            group_lengths = [len(group["node_order"]) for group in groups]
            sorted_indices = np.argsort(group_lengths)[::-1]
            color_map = {}
            for rank, g_idx in enumerate(sorted_indices):
                color_map[g_idx] = colors[rank] if rank < k else default_color
            centroids = red_centroids.get(parent_idx, [])
            for idx, group in enumerate(groups):
                color = color_map.get(idx, "black")
                node_order = group.get("node_order", [])
                if len(node_order) < 2:
                    continue
                points_2d = ArcUtils.project_to_plane(np.array(node_order), origin, radial_dir, vertical_dir)
                ax.plot(points_2d[:, 0], points_2d[:, 1], color=color, linewidth=2, label=f'组 {parent_idx}-{idx+1}')
                if idx < len(centroids):
                    centroid = centroids[idx]
                    ax.scatter(centroid[0], centroid[1], color=color, marker='o', s=20)
        legend_elements = [Line2D([0], [0], color=colors[i], lw=3, label=f'C{i+1}') for i in range(k)]
        legend_elements.append(Line2D([0], [0], color=default_color, lw=3, label=f'C{k+1}-Cn'))
        legend = ax.legend(handles=legend_elements, title="投影质心和线段组\n(按照长度排序)", loc="upper right", prop=font_en)
        legend.get_title().set_fontproperties(font_zh)
        legend.get_title().set_ha('center')
        ax.set_aspect("equal", adjustable="datalim")
        plt.show()

    @staticmethod
    def display_filtered_vs_unfiltered_corrected_segments(results, min_distance=0.1):
        origin = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        intersections = results["slicing"]["intersections"]
        unfiltered = results.get("red_groups_corrected", {})
        filtered = results.get("FloatingSurface", {})

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        # 在背景中绘制所有切割线段
        for ax in axs:
            if intersections is not None:
                for seg in intersections:
                    p1, p2, face_idx = seg
                    p1_2d = ArcUtils.project_to_plane([p1], origin, radial_dir, vertical_dir)[0]
                    p2_2d = ArcUtils.project_to_plane([p2], origin, radial_dir, vertical_dir)[0]
                    ax.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color="black", linewidth=1)

        cmap = plt.cm.get_cmap("tab20")
        color_mapping = {}
        # 建立色彩索引
        for parent_idx, groups in unfiltered.items():
            mapping = {}
            for group in groups:
                node_order = group["node_order"]
                if len(node_order) < 2:
                    continue
                # 仅用首尾节点2D信息做区分
                id_key = (tuple(np.round(node_order[0], 6)), tuple(np.round(node_order[-1], 6)))
                mapping[id_key] = None
            for i, key in enumerate(mapping.keys()):
                mapping[key] = cmap(i)
            color_mapping[parent_idx] = mapping

        # 左图：未筛选
        for parent_idx, groups in unfiltered.items():
            mapping = color_mapping.get(parent_idx, {})
            for group in groups:
                node_order = group["node_order"]
                if len(node_order) < 2:
                    continue
                id_key = (tuple(np.round(node_order[0], 6)), tuple(np.round(node_order[-1], 6)))
                color = mapping.get(id_key, "black")
                node_order_array = np.array(node_order)
                node_order_2d = ArcUtils.project_to_plane(node_order_array, origin, radial_dir, vertical_dir)
                axs[0].plot(node_order_2d[:, 0], node_order_2d[:, 1], color=color, linewidth=3)
        axs[0].set_title("校正后（未筛选）")
        axs[0].set_xlabel("径向 (m)")
        axs[0].set_ylabel("垂直 (m)")
        axs[0].set_aspect("equal", adjustable="datalim")

        # 右图：筛选后
        for parent_idx, groups in filtered.items():
            mapping = color_mapping.get(parent_idx, {})
            for group in groups:
                node_order = group["node_order"]
                if len(node_order) < 2:
                    continue
                id_key = (tuple(np.round(node_order[0], 6)), tuple(np.round(node_order[-1], 6)))
                color = mapping.get(id_key, "black")
                node_order_array = np.array(node_order)
                node_order_2d = ArcUtils.project_to_plane(node_order_array, origin, radial_dir, vertical_dir)
                axs[1].plot(node_order_2d[:, 0], node_order_2d[:, 1], color=color, linewidth=3)
        axs[1].set_title(f"筛选后（x投影距离>={min_distance}）")
        axs[1].set_xlabel("径向 (m)")
        axs[1].set_ylabel("垂直 (m)")
        axs[1].set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        plt.show()


    @staticmethod
    def draw_result_2d(results, show_arrows=True):
        plt.figure(figsize=(6, 6))
        origin     = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        merged_subsets = results["paths"].get("merged_open_subsets", [])
        azimuth    = results["plane_params"]['azimuth']

        # —— 找出最佳子集并提取节点 —— 
        best_subset = None
        max_nodes = 0
        for subset in merged_subsets:
            nodes = []
            for p1, p2, _ in subset:
                if not nodes:
                    nodes.append(np.array(p1))
                nodes.append(np.array(p2))
            if len(nodes) > max_nodes:
                max_nodes = len(nodes)
                best_subset = subset

        if best_subset:
            ordered_nodes = []
            for p1, p2, _ in best_subset:
                if not ordered_nodes:
                    ordered_nodes.append(np.array(p1))
                ordered_nodes.append(np.array(p2))
            ordered_nodes = np.array(ordered_nodes)
        else:
            ordered_nodes = None

        if ordered_nodes is not None:
            # —— 投影到平面并平移到 new_origin —— 
            pts2d_raw = ArcUtils.project_to_plane(ordered_nodes, origin, radial_dir, vertical_dir)
            new_origin = np.array([
                np.min(pts2d_raw[:, 0]) - 5,
                np.min(pts2d_raw[:, 1]) - 5
            ])
            pts2d = pts2d_raw - new_origin

            # 绘制主剖面线
            plt.plot(pts2d[:, 0], pts2d[:, 1], color='k', linewidth=2)

            # 填充灰色区域示例
            start_pt, end_pt = pts2d[0], pts2d[-1]
            offset_val = 10
            left_start = start_pt - np.array([offset_val, 0])
            left_end   = end_pt   - np.array([end_pt[0] - left_start[0], 0])
            poly_pts = np.vstack((pts2d, [left_end], [left_start]))
            plt.fill(poly_pts[:, 0], poly_pts[:, 1], color='gray', alpha=0.2)

            # 标注“崖顶”，位置也要减去 new_origin
            label_pos = pts2d[0]
            plt.text(label_pos[0], label_pos[1], "崖顶",
                    fontproperties=font_zh_13, color="black")

            # 可选：沿主剖面画箭头
            if show_arrows:
                for i in range(len(pts2d)-1):
                    p_start, p_end = pts2d[i], pts2d[i+1]
                    vec = p_end - p_start
                    L = np.linalg.norm(vec)
                    if L < 1e-6:
                        continue
                    dir_vec = vec / L
                    mid = 0.5*(p_start + p_end)
                    arrow_len = L * 0.3
                    plt.arrow(mid[0], mid[1],
                            dir_vec[0]*arrow_len, dir_vec[1]*arrow_len,
                            color='k', head_width=0.01, head_length=0.05, linewidth=0.5)

        # —— 绘制所有红线段组 —— 
        red_groups = results.get("red_groups_corrected", {})
        for groups in red_groups.values():
            for group in groups:
                node_order = group["node_order"]
                if len(node_order) < 2:
                    continue
                arr = np.array(node_order)
                pts2d_raw = ArcUtils.project_to_plane(arr, origin, radial_dir, vertical_dir)
                pts2d_red = pts2d_raw - new_origin
                # plt.plot(pts2d_red[:, 0], pts2d_red[:, 1], color='r', linewidth=1)

        # —— 构造箭头图例并添加标题 —— 
        arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        legend = plt.legend(
            [arrow_legend],
            [f"{azimuth:.1f}°"],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            handler_map={FancyArrowPatch: HandlerArrow()},
            prop=font_en_13,
            frameon=True,
            title="剖面方位角"
        )
        legend.get_title().set_fontproperties(font_zh_13)

        # —— 坐标轴设置 —— 
        plt.xlabel("(m)", fontproperties=font_en_13)
        plt.ylabel("(m)", fontproperties=font_en_13)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.show()
    @staticmethod
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    @staticmethod
    def draw_result_3d(results, show_arrows=False):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.cm.get_cmap("tab10")

        # 基本参数
        origin     = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        merged_subsets = results["paths"].get("merged_open_subsets", [])
        azimuth    = results["plane_params"]['azimuth']

        # 1. 找到 best_subset
        best_subset = None
        max_nodes = 0
        for subset in merged_subsets:
            nodes = []
            for p1, p2, _ in subset:
                if not nodes:
                    nodes.append(np.array(p1))
                nodes.append(np.array(p2))
            if len(nodes) > max_nodes:
                max_nodes = len(nodes)
                best_subset = subset

        if best_subset is None:
            return  # 没有有效子集

        # 2. 提取有序节点列表
        ordered_nodes = []
        for p1, p2, _ in best_subset:
            if not ordered_nodes:
                ordered_nodes.append(np.array(p1))
            ordered_nodes.append(np.array(p2))
        ordered_nodes = np.array(ordered_nodes)  # (N,3)

        # 3. 投影到剖面平面 → 计算 new_origin → 平移
        pts2d_raw = ArcUtils.project_to_plane(ordered_nodes,
                                            origin, radial_dir, vertical_dir)
        new_origin = np.array([
            np.min(pts2d_raw[:, 0]) - 5,
            np.min(pts2d_raw[:, 1]) - 5
        ])
        pts2d = pts2d_raw - new_origin

        # 4. 绘制主剖面（先还原回三维）
        pts3d = ArcUtils.convert_2d_points_to_3d(pts2d,
                                                origin, radial_dir, vertical_dir)
        ax.plot(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
                color='k', linewidth=2)

        # 5. 灰色面片填充
        start_pt2d, end_pt2d = pts2d[0], pts2d[-1]
        offset = np.array([0, 0])  # 如果需要偏移可在这里调整
        left_start = start_pt2d - offset
        left_end = end_pt2d - np.array([end_pt2d[0] - left_start[0], 0])
        poly2d = np.vstack((pts2d, [left_end], [left_start]))
        poly3d = ArcUtils.convert_2d_points_to_3d(poly2d,
                                                origin, radial_dir, vertical_dir)
        poly_coll = Poly3DCollection([poly3d],
                                    facecolor='gray', alpha=0.2,
                                    edgecolor='none')
        ax.add_collection3d(poly_coll)

        # 6. “崖顶”文字
        label3d = ArcUtils.convert_2d_points_to_3d([start_pt2d],
                                                origin, radial_dir, vertical_dir)[0]
        ax.text(label3d[0], label3d[1], label3d[2],
                "崖顶", fontproperties=font_zh_13, color="black")

        # 7. 可选：在三维主线段上加箭头（略，可按2D示例移植）

        # 8. 红色线段组也同样平移绘制
        red_groups = results.get("red_groups_corrected", {})
        for groups in red_groups.values():
            for group in groups:
                no = np.array(group["node_order"])
                if len(no) < 2:
                    continue
                r2d = ArcUtils.project_to_plane(no,
                                            origin, radial_dir, vertical_dir) - new_origin
                r3d = ArcUtils.convert_2d_points_to_3d(r2d,
                                                    origin, radial_dir, vertical_dir)
                # ax.plot(r3d[:, 0], r3d[:, 1], r3d[:, 2],
                #         color='r', linewidth=1)

        # 9. 剖面方位角图例
        arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        legend = plt.legend(
            [arrow_legend],
            [f"{azimuth:.1f}°"],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            handler_map={FancyArrowPatch: HandlerArrow()},
            prop=font_en_13,
            frameon=True,
            title="剖面方位角"
        )
        legend.get_title().set_fontproperties(font_zh_13)

        # 10. 坐标轴标签 & 等比例
        ax.set_xlabel("(m)", fontproperties=font_en_13)
        ax.set_ylabel("(m)", fontproperties=font_en_13)
        ax.set_zlabel("(m)", fontproperties=font_en_13)
        ArcPlotter.set_axes_equal(ax)

        plt.show()
# =============================================================================
# 3. 核心剖面切割器
# =============================================================================

class ArcSlicer:
    def __init__(self, config_path, mesh_path, slice_position=None, use_merged_paths=False):
        self.config_path = config_path
        self.mesh_path = mesh_path
        self.slice_position = slice_position
        self.use_merged_paths = use_merged_paths
        self.results = {}

    def load_data(self):
        center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = ArcUtils.load_arc_config(self.config_path)
        mesh = trimesh.load_mesh(self.mesh_path)
        self.results["data_loading"] = {
            "center": center,
            "radius": radius,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "arc_length_range": arc_length_range,
            "z_range": (z_min, z_max),
            "mesh": mesh
        }

    def compute_plane_params(self):
        center = self.results["data_loading"]["center"]
        radius = self.results["data_loading"]["radius"]
        angle_min = self.results["data_loading"]["angle_min"]
        arc_length_range = self.results["data_loading"]["arc_length_range"]
        origin, normal, radial_dir, slice_angle, azimuth_deg = ArcUtils.compute_plane(center, radius, angle_min, arc_length_range, self.slice_position)
        vertical_dir = np.array([0, 0, 1])
        self.results["plane_params"] = {
            "origin": origin,
            "normal": normal,
            "radial_dir": radial_dir,
            "vertical_dir": vertical_dir,
            "slice_angle": slice_angle,
            "azimuth": azimuth_deg
        }

    def filter_and_slice(self):
        mesh = self.results["data_loading"]["mesh"]
        z_min, z_max = self.results["data_loading"]["z_range"]
        mesh_filtered = ArcUtils.filter_mesh_by_z_range(mesh, z_min, z_max)
        origin = self.results["plane_params"]["origin"]
        normal = self.results["plane_params"]["normal"]
        intersections = ArcUtils.slice_mesh(mesh_filtered, origin, normal)
        self.results["slicing"] = {
            "filtered_mesh": mesh_filtered,
            "intersections": intersections  # 这里是带有面索引的列表
        }

    def extract_nodes(self):
        intersections = self.results["slicing"]["intersections"]
        unique_nodes = ArcUtils.merge_intersection_nodes(intersections)
        node_conn = ArcUtils.compute_node_connectivity(intersections)
        self.results["nodes"] = {
            "unique_nodes": unique_nodes,
            "node_connectivity": node_conn
        }

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
                print(idx)
                continue
            # threshold可自行调整
            ordered_red = ArcUtils.get_ordered_red_segments_for_path(subset, origin, radial_dir, vertical_dir, threshold=-0.1)
            if ordered_red:
                # 这里 tol=-0.05 仅作示例
                groups = ArcUtils.group_red_segments_by_connection_order(ordered_red, tol=-0.1, y_tol=0.1)
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
                    parent_index,
                    path_subsets[parent_index],
                    group,
                    origin, radial_dir, vertical_dir, tol
                )
                if correction is not None:
                    corrected[parent_index].append(correction)
        self.results["red_groups_corrected"] = corrected


    def run_all(self):
        self.load_data()
        self.compute_plane_params()
        self.filter_and_slice()
        self.extract_nodes()
        self.segment_paths()
        self.merge_open_paths()
        self.compute_normals()
        self.extract_and_group_red_segments()
        self.correct_red_groups()
        self.results["use_merged_paths"] = self.use_merged_paths
        return self.results

# =============================================================================
# 主函数入口示例
# =============================================================================

def main():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    config_path = str(get_path("arc_config"))
    mesh_path = str(get_path("mesh_model"))
    # mesh_path = str(get_path("segmented_mesh"))
    # mesh_path = str(get_path("segmented_mesh"))

    # 设置切剖面位置，例如取 43.5；若为None，则默认取中点
    slicer = ArcSlicer(config_path, mesh_path, slice_position=62, use_merged_paths=True)
    results = slicer.run_all()

    # 打印部分信息
    data = results["data_loading"]
    angle_min = data["angle_min"]
    angle_max = data["angle_max"]
    arc_length_range = data["arc_length_range"]
    turning_angle = angle_max - angle_min
    slice_angle = results["plane_params"]["slice_angle"]
    print("原始角度范围（弧度）：", angle_min, "至", angle_max)
    print("弧长范围：", arc_length_range)
    print("圆弧转角（弧度）：", turning_angle)
    print("选择的切剖面角度（弧度）：", slice_angle)

    # 计算每个红段组的2D质心并可视化
    results = ArcUtils.compute_2d_centroids(results)
    ArcPlotter.polt_merged_open_subsets(results)
    # ArcPlotter.plot_centroid_groups_with_zoomed(results,
    #                                     zoom_size=3,
    #                                     top_n=3)
    # ArcPlotter.plot_red_groups_with_centroids(results)

    # ArcPlotter.plot_red_groups_only(results)

    # ArcPlotter.plot_full_with_zoomed_v2(results,
    #                          centers=[(26.956, 107.573),
    #                                   (50.04,   61.30),
    #                                   (70.43,   20.57)],
    #                          zoom_size=5)

    # ArcPlotter.display_filtered_vs_unfiltered_corrected_segments(results, min_distance=0.1)
    # ArcPlotter.draw_result_2d(results, show_arrows=False)
    # ArcPlotter.draw_result_3d(results, show_arrows=True)
    ArcPlotter3D.plot_red_groups_corrected_with_faces(results, color="red", line_width=3, face_opacity=0.5)

if __name__ == "__main__":
    main()



#results说明文档
"""
下面是对本脚本在执行后 `results` 字典中各个键的说明，以及它们所对应的数据含义，方便你了解和使用切剖面及后续处理所获得的信息。

---

## 1. `results["data_loading"]`
**内容**：
```python
{
    "center": center,              # 圆弧中心(3D坐标)，从 config 中解析
    "radius": radius,              # 圆弧半径
    "angle_min": angle_min,        # 圆弧起始角度(弧度)
    "angle_max": angle_max,        # 圆弧结束角度(弧度)
    "arc_length_range": arc_length_range,   # [0, 弧长]
    "z_range": (z_min, z_max),     # Z方向的过滤范围
    "mesh": mesh                   # 原始加载并转成 trimesh 的完整 3D 网格
}
```
**意义**：
- 主要是对用户配置和网格数据的记录：
  - `center`/`radius`/`angle_min`/`angle_max`：从 `arc_config.json` 中获取，用来描述圆弧切剖面所在的中心、半径和角度区间。
  - `arc_length_range`：根据 `radius*(angle_max - angle_min)` 获得整个弧的弧长范围。
  - `z_range`：切剖面前对网格做 Z 方向过滤（只保留 [z_min, z_max] 上的面）。
  - `mesh`：trimesh 加载后的网格对象。

---

## 2. `results["plane_params"]`
**内容**：
```python
{
    "origin": origin,       # 剖面平面经过的原点(3D坐标) ，即计算得到的切线位置
    "normal": normal,       # 剖面平面的法向量 (3D)
    "radial_dir": radial_dir,     # 切剖面的“径向”方向向量(3D)
    "vertical_dir": vertical_dir, # 切剖面在世界坐标系下的垂直方向(3D)，这里默认用Z轴[0,0,1]
    "slice_angle": slice_angle    # 最终选定的切剖面角度(弧度)，决定 radial_dir 
}
```
**意义**：
- 用于描述“切剖面”在三维空间中的具体位置和方向：
  - `origin`：平面过点
  - `normal`：平面法向量，用于对 trimesh 做 plane-slice
  - `radial_dir`：切剖面在 XY 平面投影时的基向量之一
  - `vertical_dir`：通常指向 Z 轴，用于在切剖面上做投影分析(垂直方向)
  - `slice_angle`：该剖面相对于圆心的角度，等于 `angle_min + slice_position / radius`。若不指定 `slice_position`，则默认取弧长中点

---

## 3. `results["slicing"]`
**内容**：
```python
{
    "filtered_mesh": mesh_filtered,   # 经过Z范围过滤之后的子网格(trimesh)
    "intersections": intersections    # 由 slice_mesh(...) 返回的线段及面索引
}
```
**意义**：
- `filtered_mesh` 是经过 `filter_mesh_by_z_range` 筛选后只包含 [z_min, z_max] 高度面片的网格。
- `intersections` 是“该子网格”与“切剖面”求交得到的线段列表。  
  每个元素的格式：`(p1, p2, face_idx)`：  
  - `p1` / `p2`：线段两端点(3D 坐标)  
  - `face_idx`：该切面线段来源于原网格的哪个面索引

---

## 4. `results["nodes"]`
**内容**：
```python
{
    "unique_nodes": unique_nodes,   # 所有线段端点去重后的坐标 (N,3)
    "node_connectivity": node_conn  # 每个端点被连接次数 { (x,y,z): count, ... }
}
```
**意义**：
- `unique_nodes`：将 `intersections` 中出现的所有线段端点做去重后的节点集合，用于后续分析路径是否闭合、或者可视化端点等。
- `node_connectivity`：统计每个节点(端点)出现的次数(即与多少条线段相连)，可判断此端点是否在路径中作为“头”“尾”或“中间节点”。

---

## 5. `results["paths"]`
**内容**：
```python
{
    "unmerged_subsets": [subset1, subset2, ...],   # 未进行人工合并的线段子集
    "merged_open_subsets": [subsetA, subsetB, ...] # 合并后 (若 use_merged_paths=True ) 的线段子集
}
```
其中：
- `unmerged_subsets`：`split_segments_to_paths` 得到的所有连通子集，每个连通子集是若干 `(p1, p2, face_idx)` 的集合。
- `merged_open_subsets`：针对非闭合子路径进行拼接、修正后得到的合并结果，每个元素里是若干 `(p1, p2, None)` 的线段（新的合并线段没有原始面索引，故为 `None`）。

**意义**：
- “路径分割”步骤会将所有切面线段按照连通性拆分。对于每个子集，可判断是否闭合、再视需要做合并。  
- `merged_open_subsets` 是最终的“可使用的路径”，可以在二维/三维中绘制出最完整的剖面线。

---

## 6. `results["normals"]`
**内容**：
这是一个列表，列表中每个元素格式形如：
```python
{
    "subset": subset,       # path_subsets 中原始或合并后的某条路径（线段集合）
    "closed": closed,       # bool, 是否闭合
    "normals": normals      # 计算得到的 2D 法向信息
}
```
**意义**：
- “法向信息”是指对每一段边在 2D 投影上计算方向向量、法向向量等，用于后续做坡度/朝向之类的分析。
- `closed` 用来标识这一条路径是否构成一个闭合回路。

---

## 7. `results["red_groups"]`
**内容**：
```python
{
    parent_idx0: [group0, group1, ...],
    parent_idx1: [group0, group1, ...],
    ...
}
```
其中每个 `group` 则是一组“红线段”的 2D 投影信息（`(p1_2d, p2_2d, mid_2d, normal_vec)` 等），源于 `get_ordered_red_segments_for_path` 和 `group_red_segments_by_connection_order` 的结果。

**意义**：
- 用于表示“判定朝向/坡向满足某种阈值”(例如 `normal_vec[1]<-0.4`) 的那部分线段，这些线段可视为“红段”或“高危段”等。
- 按照不同 parent_idx(即第几条路径子集)进行存储。

---

## 8. `results["red_groups_corrected"]`
**内容**：
```python
{
    parent_idx0: [correction0, correction1, ...],
    parent_idx1: [correction0, correction1, ...],
    ...
}
```
而每个 `correction` 大致格式是：
```python
{
    "parent_index": parent_index,
    "node_order": corrected_nodes,   # 修正后的 3D 节点序列
    "nodes": [...],                  # 去重后的节点
    "connectivity": {...},           # 简单记录节点连接关系(头节点=1,中间节点=2,...)
    "group_segments": group_segments # 原先(2D投影)那批红线段
}
```
**意义**：
- 对 `red_groups` 中的红线段做了一次“投影-匹配-纠正”，以确保它们能和 `parent_path_subset` 的真实几何对应。
- “node_order” 就是最终对该红线段组所对应的 3D 节点序列，可直接用来在三维中绘制红线位置等。

---

## 9. `results["3d_points"]`
**内容**：
```python
{
    parent_idx0: [arr0, arr1, ...],
    parent_idx1: [arr0, arr1, ...],
    ...
}
```
- 其中 `arrX` 是某个红线段组在三维坐标系下的点云 (N×3)，由 `convert_groups_to_3d_points` 得到。  
- 注意，这里仅针对原始 2D 投影线段做了简单 3D 映射，不一定保留原网格的面索引或路径顺序。

**意义**：
- 让使用者可以在 3D 场景中查看红段组的位置，大致对应在平面投影中的哪一部分。

---

## 10. `results["FloatingSurface"]`
**内容**：
```python
{
    parent_idx0: [groupX, groupY, ...],
    parent_idx1: [groupX, groupY, ...],
    ...
}
```
- 这是 `ArcUtils.filter_corrected_groups_by_x_distance(results, min_distance=0.1)` 之后得到的一个可选输出，代表在红段组中再进行一项 “X 方向投影距离” 筛选后剩余的子集。例如常用于判定那些水平长度足够的“悬空段/伸出段”。
  
---

## 11. 其他标记
- `results["use_merged_paths"]`：只是一个布尔值，表示在本次流程中，是否使用了合并后的路径去做进一步处理。

---

### 总结

- `results` 是在 `ArcSlicer.run_all()` 完成后形成的总输出，包含了从**网格读取**、**平面参数计算**、**截面求交**、**路径分割与合并**、**红线段分组**、**纠正与筛选**等一系列步骤的结果。  
- 在实际开发中，你可以从 `results` 中任意取出需要的部分来做可视化或后处理，如 `results["paths"]["merged_open_subsets"]` 就拿来画主剖面，`results["red_groups_corrected"]` 拿来画危险段，等等。
"""