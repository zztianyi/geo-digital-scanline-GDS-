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
# 璁剧疆瀛椾綋
font_en = FontProperties(family='Arial', size=12)
font_en_13 = FontProperties(family='Arial', size=13)
font_zh = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
font_zh_13 = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=13)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 鍑犱綍涓庢暟鎹鐞嗗伐鍏?
# =============================================================================
class ArcUtils:
    """灏佽鍑犱綍璁＄畻銆佺綉鏍煎垏鍓层€佽矾寰勫垎鍓插強鑺傜偣鎻愬彇绛夊伐鍏峰嚱鏁?""
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
            print("鏃犳硶鎵惧埌浠巠鍊兼渶澶у埌y鍊兼渶灏忕殑璺緞")
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
            print("娌℃湁瓒冲鐨勯潪闂悎瀛愯矾寰?)
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
        绠€鍖栫殑鍒嗙粍閫昏緫锛屾牴鎹浉閭荤嚎娈垫槸鍚︾浉杩炪€佹槸鍚﹀叿鏈変竴瀹氣€滅敓闀跨┖闂粹€濇潵鍒嗙粍銆?
        杩欓噷浠呬娇鐢?D鎶曞奖鍒ゆ柇杩炴帴鎬э紝涓嶄弗鏍煎鐞唂ace_idx銆?

        鏂板鎵撴柇鏈哄埗锛氬湪鎵╁睍璺緞鏃讹紝
        濡傛灉褰撳墠鍊欓€夌嚎娈电殑鏈锛堢敤璇ョ嚎娈电殑绗竴涓鐐硅〃绀猴級涓庣瀛愮嚎娈电殑璧峰绔紙鐢ㄧ瀛愮嚎娈电殑绗簩涓鐐硅〃绀猴級
        鏋勬垚鐨勫悜閲忎笌X杞寸殑澶硅瓒呰繃45搴︼紝鍒欑粓姝㈠綋鍓嶈矾寰勭殑鐢熼暱锛屽紑濮嬩笅涓€涓敓闀裤€?
        """
        n = len(ordered_red_segments)
        grouped_flags = [False] * n
        reversed_indices = list(range(n - 1, -1, -1))

        def is_growth_allowed(segment):
            """
            鐢ㄧ嚎娈垫柟鍚戠殑娉曞悜閲忎笌X杞寸殑澶硅杩涜绠€鍗曞垽鏂紱
            绀轰緥鍑芥暟锛屼笉褰卞搷涓绘祦绋嬨€?
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
            鍒ゆ柇涓や釜绾挎鏄惁鐩歌繛銆?
            杩欓噷浠eg1鐨勭涓€涓鐐瑰拰seg2鐨勭浜屼釜绔偣涓哄垽鏂緷鎹€?
            """
            seg1_end_x = seg1[0][0]
            seg2_start_x = seg2[1][0]
            seg1_end_y = seg1[0][1]
            seg2_start_y = seg2[1][1]
            return (seg2_start_x - seg1_end_x > tol) and (abs(seg1_end_y - seg2_start_y) < y_tol)

        # 鎸戦€夊厑璁哥敓闀跨殑璧风偣绱㈠紩锛堝€掑簭锛?
        growth_points = [i for i in reversed_indices if is_growth_allowed(ordered_red_segments[i])]
        groups = []
        while growth_points:
            seed_idx = growth_points[0]
            group_indices = []
            stack = [seed_idx]
            # 璁板綍绉嶅瓙绾挎鐨勨€滆捣濮嬬鈥濓紝杩欓噷绾﹀畾浣跨敤绉嶅瓙绾挎鐨勭浜屼釜绔偣浣滀负璧峰绔?
            group_start = ordered_red_segments[seed_idx][1]
            while stack:
                idx = stack.pop()
                if grouped_flags[idx]:
                    continue
                grouped_flags[idx] = True
                group_indices.append(idx)
                # 妫€鏌ヨ兘鍚︽墿灞曡嚦鍓嶄竴涓嚎娈?
                if idx - 1 >= 0 and connected(ordered_red_segments[idx], ordered_red_segments[idx - 1]):
                    candidate_end = ordered_red_segments[idx - 1][0]
                    vec = np.array(candidate_end) - np.array(group_start)
                    # 璁＄畻鍚戦噺涓嶺杞寸殑澶硅锛堝彇缁濆鍊硷紝淇濊瘉姣旇緝姝ｈ搴︼級
                    angle = math.degrees(math.atan2(abs(vec[1]), abs(vec[0])))
                    if angle <= 70:
                        stack.append(idx - 1)
                    else:
                        # 濡傛灉鍊欓€夋墿灞曠殑瑙掑害瓒呰繃45搴︼紝鍒欑粓姝㈠綋鍓嶇敓闀?
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
# 2. 鏍稿績鍓栭潰鍒囧壊鍣?
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
        # 鑻ヤ笉闇€瑕佸悎骞?open paths锛屽垯娉ㄩ噴涓嬩竴琛?
        # self.merge_open_paths()
        self.compute_normals()
        self.extract_and_group_red_segments()
        self.correct_red_groups()
        self.results["use_merged_paths"] = self.use_merged_paths
        return self.results

# =============================================================================
# 澶氬墫闈㈠鐞嗗強缁撴灉淇濆瓨锛堜富鍑芥暟锛?
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
    鎺ユ敹涓€灏忓潡锛堟瘮濡?0涓級鍒囬潰鏁版嵁锛屼覆琛屽鐞嗗苟杩斿洖缁撴灉鍒楄〃銆?
    """
    results = []
    for slice_data in slice_block:
        res = process_single_slice(slice_data)
        results.append(res)
    return results


def main():
    file_path = r"data/private/raw_model\all_slices_output_faces_zong.pkl"
    file_path = r"data/private/raw_model\all_slices_output_faces_seg_84-92.pkl"
    # file_path = r"outputs/intermediate\all_slices_output_faces_seg_5_5.pkl"
    output_file = r"data/private/raw_model\multi_slice_results_0.05_40_0-140_buchang.pkl"
    output_file = r"data/private/raw_model\multi_slice_results_0.05_40_84-92_buchang.pkl"
    # output_file = r"data/private/raw_model\\polt_5_\multi_slice_results_0.05_5_5_buchang.pkl"
    # 璇诲彇鎵€鏈夊垏闈㈡暟鎹?
    with open(file_path, "rb") as f:
        all_slices = pickle.load(f)

    # 鍑嗗寰呭鐞嗗垏闈?
    desired_positions = np.arange(0,140, 0.05)

    precomputed_results = []
    for pos in desired_positions:
        
        key = f"{pos:.2f}"
        if key in all_slices:
            slice_result = all_slices[key]
            slice_result["slice_key"] = key
            precomputed_results.append(slice_result)

    print(f"鍓嶅鐞嗗畬姣曪紝鍏?{len(precomputed_results)} 涓垏闈㈠緟澶勭悊")

    # 濡傛灉 output_file 瀛樺湪锛屽垹闄や互闃叉鏈鍐欏叆鍜屾棫鍐呭娣峰悎
    if os.path.exists(output_file):
        os.remove(output_file)

    start_time = time.time()

    # 鍙互鏍规嵁闇€瑕佺伒娲昏皟鏁翠笅鍒楀弬鏁?
    num_processes = 15   # 涓€娆″苟琛屽灏戜釜杩涚▼
    sub_task_size = 40   # 姣忎釜杩涚▼瑕佷覆琛屽鐞嗗灏戜釜鍒囬潰
    batch_size = num_processes * sub_task_size  # 姣忔壒澶勭悊鐨勫垏闈㈡€绘暟

    total_slices = len(precomputed_results)

    # 鐢?tqdm 璁剧疆涓€涓叏灞€杩涘害鏉★紝缁熻鎬诲叡澶勭悊浜嗗灏戝垏闈?
    global_progress = tqdm(total=total_slices, desc="鎬讳綋杩涘害")

    # 鍒嗘壒娆″鐞?
    for batch_start in range(0, total_slices, batch_size):
        # 鍙栧嚭鏈壒闇€瑕佸鐞嗙殑鍒囬潰
        batch_data = precomputed_results[batch_start: batch_start + batch_size]

        # 鎸?sub_task_size 鍒掑垎缁欐瘡涓繘绋嬪鐞?
        blocks = []
        for i in range(0, len(batch_data), sub_task_size):
            block_slice = batch_data[i: i + sub_task_size]
            blocks.append(block_slice)

        # 骞惰澶勭悊鏈壒鏁版嵁
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            block_results_iter = executor.map(process_slice_block, blocks)
            # 鏀堕泦骞跺啓鍏ョ粨鏋?
            with open(output_file, "ab") as f_out:
                for block_res in block_results_iter:
                    # block_res 鏄繖涓瓙鍧楅噷鎵€鏈夊垏闈㈢殑澶勭悊缁撴灉(list)
                    # 鍐欏叆鏂囦欢
                    for single_res in block_res:
                        pickle.dump(single_res, f_out)
                    # 鏇存柊鍏ㄥ眬杩涘害鏉?
                    global_progress.update(len(block_res))

        # 閲婃斁宸插啓鍏ョ殑鏁版嵁
        del blocks

    global_progress.close()
    end_time = time.time()

    print(f"鎵€鏈夊垏闈㈠鐞嗗畬姣曪紝鎬荤敤鏃? {end_time - start_time:.2f} 绉?)
    print(f"缁撴灉宸插垎娈靛啓鍏ュ埌: {output_file}")


    # 濡傛灉闇€瑕佸湪鏈€鍚庝竴娆℃€у皢鎵€鏈夌粨鏋滃悎骞舵垚涓€涓狶ist鍐嶅瓨涓€娆★紝涓嬮潰缁欏嚭涓€涓ず渚嬶細
    # 锛堥渶瑕佸厛鎶婁笂闈㈠垎娈靛啓鍏ョ殑鏁版嵁閮借鍥炴潵锛?
    # final_results = []
    # with open(output_file, "rb") as f_in:
    #     while True:
    #         try:
    #             item = pickle.load(f_in)
    #             final_results.append(item)
    #         except EOFError:
    #             break
    #
    # # 姝ゆ椂 final_results 灏辨槸鍏ㄩ儴缁撴灉鐨勫悎闆嗭紝濡傛灉闇€瑕佸彲浠ュ啀鍗曠嫭鎸変綘鐨勬牸寮忓瓨涓€娆?
    # with open(output_file, "wb") as f_out:
    #     pickle.dump(final_results, f_out)
    # print("宸插悎骞舵墍鏈夌粨鏋滃埌涓€涓垪琛紝閲嶆柊鍐欏叆瀹屾垚銆?)

if __name__ == "__main__":
    main()

