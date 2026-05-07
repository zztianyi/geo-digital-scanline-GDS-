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
        # 鍒涘缓瀹為檯鐨勭澶村厓绱狅紝娉ㄦ剰 mutation_scale 鎺у埗绠ご澶у皬
        p = FancyArrowPatch((xdescent, ydescent + height/2),
                            (xdescent + width, ydescent + height/2),
                            transform=trans,
                            color=orig_handle.get_facecolor(),
                            arrowstyle='->',
                            mutation_scale=15,
                            lw=1)
        return [p]

# 璁剧疆瀛椾綋
font_en = FontProperties(family='Times New Roman', size=12)
font_en_13 = FontProperties(family='Times New Roman', size=13)
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
        # 璁炬鍖楁柟鍚戜负 [0,1]锛屼娇鐢?np.arctan2(radial_dir[0], radial_dir[1])
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
        杩斿洖涓€涓垪琛紝姣忎釜鍏冪礌鏍煎紡涓?(p1_3d, p2_3d, face_idx)銆?
        p1_3d 鍜?p2_3d 涓虹嚎娈典袱绔偣鍧愭爣锛宖ace_idx 涓轰笌璇ョ嚎娈靛搴旂殑闈㈢储寮曘€?
        """
        lines_3d, face_indices = trimesh.intersections.mesh_plane(
            mesh=mesh,
            plane_normal=normal,
            plane_origin=origin,
            return_faces=True
        )
        # 灏?(绔偣, face_index) 鍚堝苟瀛樺偍鍦ㄤ竴璧?
        segments = []
        for i in range(len(lines_3d)):
            seg = lines_3d[i]
            fidx = face_indices[i]
            # seg[0] 鍜?seg[1] 鍒嗗埆鏄嚎娈典袱绔偣 (x, y, z)
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
        娉ㄦ剰锛氳繖閲屽彧瀵圭嚎娈电鐐瑰仛鍘婚噸锛屽苟娌℃湁璁板綍闈㈢储寮曘€?
        鑻ラ渶瑕佺鐐圭骇鍒殑闈㈢储寮曪紝闇€瑕佸湪姝ゅ仛鑷畾涔夐€昏緫銆?
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
        浠呰绠楄妭鐐硅杩炴帴娆℃暟锛涘苟涓嶅尯鍒嗘潵鑷摢涓潰绱㈠紩銆?
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
        灏嗙嚎娈甸泦鎷嗗垎鎴愯嫢骞茶繛閫氬瓙闆嗐€?
        姣忔潯杈圭殑鏍煎紡涓?(p1, p2, face_idx)锛宲1銆乸2宸茬粡鍋氫簡round(6)澶勭悊銆?
        """
        graph = {}
        edges = []
        for seg in intersections:
            p1, p2, face_idx = seg
            p1_r = tuple(np.round(p1, decimals=6))
            p2_r = tuple(np.round(p2, decimals=6))
            # 瀛樺偍鏃朵篃鎶婇潰绱㈠紩甯︿笂
            edges.append((p1_r, p2_r, face_idx))

            # 寤虹珛鏃犲悜鍥?
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
            # comp 鏄竴涓繛閫氬垎閲忓唴鐨勬墍鏈夎妭鐐?
            # 鎶?edges 涓?灞炰簬璇ヨ繛閫氬垎閲忕殑鎷垮嚭鏉?
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
        鐢变簬浜哄伐鍚堝苟鏃讹紝鎴戜滑鍙煡閬撹妭鐐归『搴忥紝鍥犳姝ゅ鏃犳硶瀵瑰簲鍘熷 mesh 闈㈢储寮曘€?
        鏁呰繖閲岀粺涓€灏?face_idx 璁剧疆涓?None銆?
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
        path_subset 涓瘡涓厓绱? (p1, p2, face_idx)
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
        鍙傛暟:
        path_subset: 姣忎釜鍏冪礌涓?(p1, p2, face_idx) 鐨勮竟鍒楄〃
        plane_origin, radial_dir, vertical_dir: 鐢ㄤ簬鎶曞奖鐨勫弬鏁?
        return_edges: 鑻ヤ负 True锛屽垯杩斿洖鎸変粠 top->bottom 鎺掑簭濂界殑杈瑰垪琛紝
                        姣忔潯杈规牸寮忎负 (p1, p2, face_idx)锛涘惁鍒欒繑鍥炶妭鐐瑰垪琛?3D 鍧愭爣)
                        
        杩斿洖:
        濡傛灉 return_edges=False锛岃繑鍥炶妭鐐瑰垪琛紱
        濡傛灉 return_edges=True锛岃繑鍥炶竟鍒楄〃锛岃竟鏄牴鎹妭鐐硅矾寰勯噸鏋勫緱鏉ャ€?
        """
        from collections import deque
        import numpy as np

        # 鏋勫缓鏃犲悜鍥撅紝鍒╃敤 p1, p2 浣滀负鍥剧殑鑺傜偣
        graph = {}
        for edge in path_subset:
            p1, p2, face_idx = edge
            graph.setdefault(p1, set()).add(p2)
            graph.setdefault(p2, set()).add(p1)

        # 鑾峰彇鎵€鏈夎妭鐐瑰強鍏跺湪 2D 鎶曞奖涓殑 y 鍊?
        all_nodes = list(graph.keys())
        nodes_arr = [np.array(n) for n in all_nodes]
        nodes_2d = ArcUtils.project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
        y_values = [pt[1] for pt in nodes_2d]

        # 鍙?2D 鎶曞奖涓?y 鍊兼渶澶х殑鐐逛綔涓鸿捣鐐癸紝y 鍊兼渶灏忕殑鐐逛綔涓虹粓鐐?
        start_node = all_nodes[y_values.index(max(y_values))]
        end_node = all_nodes[y_values.index(min(y_values))]

        # BFS 鎼滅储浠?start_node 鍒?end_node 鐨勪竴鏉¤矾寰?
        queue = deque([[start_node]])
        path_found = None
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == end_node:
                path_found = path
                break
            # 姝ゅ涓嶄娇鐢ㄥ叏灞€ visited锛岃€屾槸鍦ㄦ瘡鏉¤矾寰勪腑妫€鏌ワ紝閬垮厤閬楁紡涓嶅悓鍒嗘敮
            for neighbor in graph[current]:
                if neighbor not in path:
                    queue.append(path + [neighbor])
        if path_found is None:
            print("鏃犳硶鎵惧埌浠?y 鍊兼渶澶у埌 y 鍊兼渶灏忕殑璺緞")
            return []

        if not return_edges:
            return path_found
        else:
            # 鎸夎妭鐐硅矾寰勬瀯閫犳湁搴忚竟搴忓垪
            ordered_edges = []
            for i in range(len(path_found) - 1):
                node_a = path_found[i]
                node_b = path_found[i + 1]
                selected_edge = None
                # 浠庡師濮嬬殑杈瑰垪琛ㄤ腑鏌ユ壘杩炴帴 node_a 涓?node_b 鐨勮竟锛堝拷鐣ユ柟鍚戯級
                for edge in path_subset:
                    p1, p2, face_idx = edge
                    if (p1 == node_a and p2 == node_b) or (p1 == node_b and p2 == node_a):
                        selected_edge = edge
                        break
                # 鑻ユ湭鎵惧埌瀵瑰簲杈癸紝鍒欒涓烘槸 bridging edge锛宖ace_idx 璁剧疆涓?None
                if selected_edge is None:
                    ordered_edges.append((node_a, node_b, None))
                else:
                    ordered_edges.append(selected_edge)
            return ordered_edges

    @staticmethod
    def compute_normals_with_edges_for_subset(path_subset, closed, plane_origin, radial_dir, vertical_dir):
        """
        杩斿洖鍊间腑锛屽瓨鍌ㄦ瘡鏉¤竟鍦?2D 鎶曞奖涓嬬殑涓偣鍙婃硶鍚戜俊鎭瓑銆?
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
            # 闂悎鎯呭喌
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
        浼樺寲鍚庣殑鐗堟湰锛?
        - 鍚堝苟杩囩▼浠呭湪2D骞抽潰涓嬭繘琛岋紝閬垮厤閲嶅鐨?D/2D杞崲锛?
        - 鏈€缁堝湪鎵€鏈夊悎骞跺畬鎴愬悗涓€娆℃€у皢2D璺緞杞崲涓?D锛?
        - 瀵逛簬 bridging edge锛堣鎺ヨ竟锛夛紝鍏?face_idx 璁剧疆涓?None銆?
        
        杩斿洖鍊硷細涓€涓垪琛紝姣忎釜鍏冪礌涓轰竴缁勮竟锛屾瘡鏉¤竟涓?(p1, p2, face_idx)銆?
        """
        # 1) 棰勫鐞嗭細瀵规瘡涓瓙璺緞璁＄畻鏈夊簭鐨?D鐐瑰簭鍒楋紝骞朵繚瀛樺搴旂殑鍘熷3D鏁版嵁锛屼粎渚沠ace_idx鍖归厤鏃跺弬鑰?
        open_paths = []
        for subset in path_subsets:
            ordered_nodes = ArcUtils.order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir)
            if len(ordered_nodes) < 2:
                continue
            proj = ArcUtils.project_to_plane(ordered_nodes, plane_origin, radial_dir, vertical_dir)
            y_min = np.min(proj[:, 1])
            y_max = np.max(proj[:, 1])
            open_paths.append({
                "proj": proj,              # 鐢ㄤ簬鍚堝苟璁＄畻鐨?D鐐瑰簭鍒?
                "nodes": ordered_nodes,    # 鍘熷3D鏁版嵁锛屼粎浣?face_idx 鍖归厤鍙傝€?
                "start2d": proj[0],
                "end2d": proj[-1],
                "y_range": (y_min, y_max)
            })
        open_paths.sort(key=lambda p: p["start2d"][1], reverse=True)
        if not open_paths:
            print("娌℃湁瓒冲鐨勯潪闂悎瀛愯矾寰?)
            return []

        # 2) 鍦?D涓婅繘琛屽悎骞舵搷浣?
        # merged_paths_2d 淇濆瓨姣忎竴鏉″悎骞剁殑2D璺緞锛屽悓鏃朵繚鐣欎竴浠藉師濮?D鏁版嵁璁板綍锛?
        # 鏂逛究鍚庣画鍒ゆ柇鏄惁涓?bridging edge
        merged_paths_2d = []  # 姣忎釜鍏冪礌涓?(merged_2d, merged_3d_original)
        while open_paths:
            current = open_paths.pop(0)
            # 鍒濆鍖栧悎骞跺簭鍒?
            merged_2d = current["proj"].tolist()           # 2D鐐瑰垪琛?
            merged_3d_original = current["nodes"][:]         # 3D鐐瑰垪琛紝璁板綍鍘熷鍊?
            while True:
                P2d = merged_2d[-1]
                Py = P2d[1]
                # 閫夋嫨婊¤冻 y_range 瑕佹眰鐨勫€欓€夎矾寰?
                candidates = []
                for idx, path in enumerate(open_paths):
                    y_min, y_max = path["y_range"]
                    if y_min <= Py <= y_max:
                        candidates.append((idx, path))
                if not candidates:
                    break
                # 鑻ユ湁澶氫釜鍊欓€夛紝鍒欐寜 end2d 鐨?y 鍊兼帓搴忥紝閫夋嫨鎺掑簭鏈€闈犲墠鐨?
                if len(candidates) > 1:
                    candidates.sort(key=lambda x: x[1]["end2d"][1])
                    chosen_idx, chosen = candidates[0]
                else:
                    chosen_idx, chosen = candidates[0]

                # 璁＄畻褰撳墠鏈鐐逛笌鍊欓€夎矾寰勪箣闂寸殑鐩稿鍚戦噺锛屽苟妫€娴?x 鎴?y 鍒嗛噺浜ゅ弶
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

                # 姝ゅ o 涓?bridging 鐨?D鐐癸紝鐩存帴鏇挎崲褰撳墠鍚堝苟搴忓垪鐨勬湯绔?
                # 瀵瑰簲鐨?D鏁版嵁缃负 None锛岃〃绀鸿繖鏉¤竟涓?bridging edge
                merged_2d[-1] = o
                merged_3d_original[-1] = None
                # 杩藉姞鍊欓€夎矾寰勪腑鍓╀綑鐨?D鍙?D鏁版嵁
                additional_2d = chosen["proj"][index_used:].tolist()
                additional_3d = chosen["nodes"][index_used:]
                merged_2d.extend(additional_2d)
                merged_3d_original.extend(additional_3d)
                open_paths.pop(chosen_idx)
            merged_paths_2d.append((merged_2d, merged_3d_original))
            # 鏇存柊 open_paths锛岀Щ闄ら偅浜涘凡琚悎骞剁殑璺緞锛堥€氳繃 y 鍊煎垽鏂級
            new_open_paths = []
            merged_arr = np.array(merged_2d)
            for path in open_paths:
                if np.min(path["proj"][:, 1]) < np.min(merged_arr[:, 1]) or \
                np.max(path["proj"][:, 1]) > np.max(merged_arr[:, 1]):
                    new_open_paths.append(path)
            open_paths = new_open_paths
            open_paths.sort(key=lambda p: p["start2d"][1], reverse=True)

        # 3) 灏嗗悎骞跺悗鐨?D璺緞涓€娆℃€ц浆鎹负3D锛岀劧鍚庝负姣忎釜杈瑰垎閰?face_idx
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
            # 浠呭湪鎵€鏈夊悎骞剁粨鏉熷悗灏?D璺緞杞崲涓?D
            merged_3d = ArcUtils.convert_2d_points_to_3d(merged_2d, plane_origin, radial_dir, vertical_dir)
            edges = []
            for i in range(len(merged_3d) - 1):
                p1 = merged_3d[i]
                p2 = merged_3d[i + 1]
                # 鑻ュ師濮?D鏁版嵁涓湁瀵瑰簲鐨勭偣锛屽垯灏濊瘯鍖归厤 face_idx锛涘惁鍒欒浣?bridging edge
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
        鍚堝苟澶氫釜闈為棴鍚堝瓙璺緞锛?
        - 鍚堝苟鎿嶄綔鍧囧湪 2D 骞抽潰涓婂畬鎴愶紝bridging edge 瀵瑰簲鐨?face_idx 缃负 None锛?
        - 杩斿洖鏍煎紡锛氬垪琛紝姣忎釜鍏冪礌涓轰竴缁勮竟锛屾瘡鏉¤竟鏍煎紡涓?(p1, p2, face_idx)锛?
        - p1銆乸2 涓?3D 鍧愭爣銆?
        鍏朵腑锛?
        1. 瀵规瘡涓瓙璺緞锛堣竟鍒楄〃锛夛紝鍏堣皟鐢?ArcUtils.order_nonclosed_path 鑾峰彇鏈夊簭杈瑰簭鍒楋紝
            鍐嶅皢杈硅浆鎹负鐐瑰簭鍒楋紝姣忎釜鐐圭洿鎺ョ粦瀹氬叾 face 淇℃伅锛?
            鍏蜂綋涓猴細瀵逛簬杈?(p1, p2, face_idx)锛屽皢 p1 浣滀负璧峰鐐癸紙face 涓虹涓€鏉¤竟鐨?face_idx锛夛紝
            鐒跺悗渚濇璁板綍姣忔潯杈圭殑缁堢偣 p2 骞剁粦瀹氳杈圭殑 face_idx锛屾渶鍚庝竴涓偣鍥犳棤鍚庣画杈癸紝鍏?face 缃负 None銆?
        2. 鍚堝苟杩囩▼鍦?2D 骞抽潰涓婅繘琛岋紝褰撻渶瑕佽鎺ユ椂锛岀洿鎺ュ皢鍊欓€夎矾寰勭殑棣栦釜鐐逛綔涓?bridging 鐐癸紝
            骞跺皢鍏?face 淇℃伅缃负 None锛岀劧鍚庡皢鍊欓€夎矾寰勫墿浣欓儴鍒嗛檮鍔犲埌褰撳墠璺緞鍚庛€?
        3. 閬嶅巻鍚堝苟鍚庣殑鐐瑰簭鍒楃敓鎴愯竟鏃讹紝濡傛灉鐩搁偦涓ょ偣鐨?face 淇℃伅涓€鑷翠笖涓嶄负 None锛屽垯璁や负璇ヨ竟淇濈暀鍘熷 face锛?
            鍚﹀垯缃负 bridging edge锛坒ace=None锛夈€?
        """
        import numpy as np

        # ------------------------------
        # Step 1: 灏嗚竟鍒楄〃杞崲涓虹偣搴忓垪锛屽苟鐩存帴缁戝畾 face 淇℃伅
        # ------------------------------
        open_paths = []
        for subset in path_subsets:
            # 浣跨敤 ArcUtils.order_nonclosed_path 寰楀埌鏈夊簭杈瑰簭鍒?
            # 杩欓噷瑕佹眰鏀寔 return_edges=True锛岃繑鍥?ordered_edges 涓?[(p1, p2, face_idx), ...]
            ordered_edges = ArcUtils.order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir, return_edges=True)
            if not ordered_edges:
                continue

            points = []
            # 鍙栫涓€鏉¤竟鐨勮捣鐐癸紝璁板綍鍏?3D 涓?2D 鍧愭爣锛宖ace 鍙栫涓€鏉¤竟鐨?face_idx
            first_edge = ordered_edges[0]
            pt3d = first_edge[0]
            pt2d = ArcUtils.project_to_plane([pt3d], plane_origin, radial_dir, vertical_dir)[0].tolist()
            points.append({"pt3d": pt3d, "pt2d": pt2d, "face": first_edge[2]})
            
            # 瀵规瘡鏉¤竟渚濇璁板綍缁堢偣鍙婂叾 face 淇℃伅
            for edge in ordered_edges:
                pt3d = edge[1]
                pt2d = ArcUtils.project_to_plane([pt3d], plane_origin, radial_dir, vertical_dir)[0].tolist()
                points.append({"pt3d": pt3d, "pt2d": pt2d, "face": edge[2]})
            # 鏈€鍚庝竴涓偣鏃犲悗缁竟锛宖ace 缃负 None
            points[-1]["face"] = None

            ys = [p["pt2d"][1] for p in points]
            open_paths.append({
                "points": points,               # 鐐瑰簭鍒楋紝姣忎釜鐐逛负 dict {pt3d, pt2d, face}
                "start_y": points[0]["pt2d"][1],
                "end_y": points[-1]["pt2d"][1],
                "y_range": (min(ys), max(ys))
            })
        open_paths.sort(key=lambda p: p["start_y"], reverse=True)
        if not open_paths:
            print("娌℃湁瓒冲鐨勯潪闂悎瀛愯矾寰?)
            return []

        # ------------------------------
        # Step 2: 鍦?D骞抽潰涓婂悎骞惰矾寰?
        # ------------------------------
        # 姝ゅ鍏ㄩ儴閲囩敤 points 鏁版嵁锛屼笉鍐嶆媶鍒嗕负 proj 鍜?nodes銆?
        merged_paths = []  # 姣忎釜鍏冪礌涓虹偣搴忓垪锛堝垪琛紝姣忎釜鍏冪礌涓哄瓧鍏革細{pt3d, pt2d, face}锛?
        while open_paths:
            current = open_paths.pop(0)
            # 鍒濆鍖栧悎骞跺簭鍒楋細鐩存帴澶嶅埗褰撳墠璺緞鐨勭偣鍒楄〃
            merged_points = current["points"][:]
            while True:
                # 鍙栧綋鍓嶅悎骞惰矾寰勬湯灏剧偣鐨?2D 鍧愭爣
                P2d = merged_points[-1]["pt2d"]
                Py = P2d[1]
                # 閫夋嫨鍊欓€夎矾寰勶細鍏朵腑鐨?y_range 鍖呭惈褰撳墠鏈鐐圭殑 y 鍊?
                candidates = []
                for idx, path in enumerate(open_paths):
                    y_min, y_max = path["y_range"]
                    if y_min <= Py <= y_max:
                        candidates.append((idx, path))
                if not candidates:
                    break
                # 鑻ユ湁澶氫釜鍊欓€夛紝鍒欐寜鍊欓€夎矾寰勬湯绔殑 y 鍊兼帓搴忥紝鍙栨渶灏忚€?
                candidates.sort(key=lambda x: x[1]["end_y"])
                chosen_idx, chosen = candidates[0]

                # 灏嗗€欓€夎矾寰勭殑鐐瑰垪琛ㄨ浆鎹负 2D 鍧愭爣鏁扮粍
                candidate_pts_2d = np.array([pt["pt2d"] for pt in chosen["points"]])
                # 璁＄畻褰撳墠鏈鐐逛笌鍊欓€夎矾寰勬墍鏈夌偣鐨勭浉瀵瑰悜閲?
                rel = candidate_pts_2d - np.array(P2d)
                pair_y_index = None
                pair_x_index = None
                # 妫€娴?y 鍒嗛噺浜ゅ弶
                for i in range(len(rel) - 1):
                    if rel[i][1] * rel[i + 1][1] < 0:
                        pair_y_index = i
                        break
                # 妫€娴?x 鍒嗛噺浜ゅ弶
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

                # 灏?bridging 鐐?o 鏇存柊鍒板悎骞惰矾寰勭殑鏈锛屽悓鏃跺皢瀵瑰簲 3D 鏁版嵁缃负 None 琛ㄧず鏂拌鎺?
                merged_points[-1]["pt2d"] = o
                pt3d_new = ArcUtils.convert_2d_points_to_3d([o], plane_origin, radial_dir, vertical_dir)[0]
                merged_points[-1]["pt3d"] = tuple(pt3d_new)
                # 灏嗗€欓€夎矾寰勪腑 index_used 涔嬪悗鐨勭偣杩藉姞鍒板悎骞惰矾寰勪笂
                additional_points = chosen["points"][index_used:]
                merged_points.extend(additional_points)
                open_paths.pop(chosen_idx)
            merged_paths.append(merged_points)
            # 鏇存柊 open_paths锛氱Щ闄や笌褰撳墠鍚堝苟璺緞鍦?y 鍊间笂閲嶅彔鐨勮矾寰?
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
        # Step 3: 鏍规嵁鍚堝苟鍚庣殑鐐瑰簭鍒楃敓鎴愯竟
        # ------------------------------
        # 鐢熸垚杈规椂锛氶亶鍘嗙浉閭讳袱涓偣锛?
        # 鑻ヤ袱鐐圭粦瀹氱殑 face 鐩稿悓涓斾笉涓?None锛屽垯璇ヨ竟淇濈暀鍘熷 face锛涘惁鍒欒涓?bridging edge锛宖ace 缃负 None
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
        鏍规嵁 ordered_nodes 鐨勯『搴忥紝浠?subset(鏃犲簭绾挎) 涓?
        鎵惧埌杩炵画鐨?(p1, p2, face_idx)锛屼繚璇?p1->p2 姝ｅ簭銆?
        濡傛灉鏌愪竴娈垫病鎵惧埌鍖归厤锛屽彲鑳借繑鍥炰笉瀹屾暣銆?
        """
        edges_in_order = []
        # subset 涔熸槸 (p1, p2, face_idx)
        # 鍏堝仛涓€涓储寮曪紝鏂逛究鏌ユ壘
        subset_map = {}
        for (a, b, fidx) in subset:
            subset_map.setdefault(tuple(a), {})[tuple(b)] = fidx
            subset_map.setdefault(tuple(b), {})[tuple(a)] = fidx

        # 閬嶅巻 ordered_nodes锛岀浉閭绘瀯閫犵嚎娈?
        for i in range(len(ordered_nodes) - 1):
            pA = tuple(ordered_nodes[i])
            pB = tuple(ordered_nodes[i + 1])
            # 鐪嬬湅 subset_map 鏄惁鏈夎褰?
            if pA in subset_map and pB in subset_map[pA]:
                fidx = subset_map[pA][pB]
                edge = (pA, pB, fidx)
            elif pB in subset_map and pA in subset_map[pB]:
                # 鍙嶅悜
                fidx = subset_map[pB][pA]
                edge = (pA, pB, fidx)
            else:
                # 娌℃湁鍖归厤鍒帮紙鍙兘宸茶鍚堝苟鎴栧嚭鐜板叾瀹冩柇鐐癸級
                return None
            edges_in_order.append(edge)
        return edges_in_order

    @staticmethod
    def get_ordered_red_segments_for_path(path_subset, plane_origin, radial_dir, vertical_dir, threshold=-0.4, tol_match=1e-6):
        """
        閬嶅巻瀛愯矾寰?闈為棴鍚?鐨勬湁搴忚妭鐐癸紝閫夊嚭娉曞悜鍦╕鏂瑰悜灏忎簬鏌愰槇鍊肩殑鐗囨锛?
        浣滀负鈥滅孩鑹茬嚎娈碘€濄€傚悓鏃惰繑鍥炲搴旂殑 2D 鍜?3D 淇℃伅浠ュ強璇ユ鍘熺敓鐨?face_idx銆?

        杩斿洖鐨勫厓缁勬牸寮忎负锛?
        (p1_2d, p2_2d, mid_2d, normal_vec, p1_3d, p2_3d, face_idx)
        
        鍏朵腑锛?
        - p1_2d, p2_2d: 璇ョ嚎娈靛湪 2D 鎶曞奖涓嬬殑璧峰绔偣
        - mid_2d: 2D 涓偣
        - normal_vec: 2D 娉曞悜閲?
        - p1_3d, p2_3d: 瀵瑰簲鐨勫師濮?3D 绔偣
        - face_idx: 浠?path_subset 涓尮閰嶅緱鍒扮殑闈㈢储寮曪紝鑻ユ病鏈夊尮閰嶅垯涓?None
        """
        # 鍏堝埄鐢ㄥ凡鏈夊嚱鏁板緱鍒版湁搴忕殑 3D 鑺傜偣搴忓垪
        ordered_nodes = ArcUtils.order_nonclosed_path(path_subset, plane_origin, radial_dir, vertical_dir)
        red_segments = []
        if len(ordered_nodes) < 2:
            return red_segments

        for i in range(len(ordered_nodes) - 1):
            # 鍘熷3D绔偣
            p1 = np.array(ordered_nodes[i])
            p2 = np.array(ordered_nodes[i + 1])
            # 鎶曞奖鍒?D
            p1_2d = ArcUtils.project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = ArcUtils.project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            direction = p2_2d - p1_2d
            norm_val = np.linalg.norm(direction)
            if norm_val == 0:
                continue
            direction /= norm_val
            normal_vec = np.array([-direction[1], direction[0]])
            # 鍒ゆ柇鏄惁婊¤冻闃堝€兼潯浠?
            if normal_vec[1] < threshold:
                mid_2d = (p1_2d + p2_2d) / 2.0
                # 灏濊瘯鍦ㄥ師濮嬬殑 path_subset 涓尮閰嶈繖娈佃竟锛屼互鑾峰緱 face_idx
                face_idx_found = None
                for edge in path_subset:
                    q1 = np.array(edge[0])
                    q2 = np.array(edge[1])
                    # 妫€鏌ユ槸鍚︿笌 p1, p2 鍖归厤锛堣€冭檻椤哄簭鎴栧弽鍚戯級
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
                    if angle <= 40:
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
        """
        鏍规嵁 parent_path_subset 鐨勭湡瀹炶蛋鍚戯紝灏嗙孩娈?group_segments 鍦ㄥ叾涓婂仛涓€娆℃牎姝ｃ€?
        娉ㄦ剰锛歡roup_segments 鏄?((x1,y1),(x2,y2),...) 鐨?2D 鎶曞奖鐗囨锛岃€?parent_path_subset
        鍒欐槸 (p1_3d, p2_3d, face_idx) 銆?
        杩欓噷鐪佺暐 face_idx 鐨勫噯纭鐞嗭紝浠呮紨绀?node 鍖归厤鍘熺悊銆?
        """
        parent_order = ArcUtils.order_nonclosed_path(parent_path_subset, plane_origin, radial_dir, vertical_dir)
        if not parent_order:
            return None

        # 灏?parent_order 鐨?3D鑺傜偣杞负 2D骞跺仛round
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

        # 鎵惧埌 group_nodes 鍦?parent_order_2d 閲岀殑鏈€灏忓拰鏈€澶х储寮?
        indices = [i for i, node in enumerate(parent_order_2d) if node in group_nodes]
        if not indices:
            return None

        start_idx = min(indices)
        end_idx = max(indices)
        corrected_nodes = parent_order[start_idx:end_idx + 1]

        # 杩炴帴淇℃伅鍙槸绀轰緥
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
        浠呭鏍℃鍚庣殑绾㈢嚎娈靛垎缁勮绠?2D 璐ㄥ績銆?
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
# 2. 缁樺浘宸ュ叿
# =============================================================================

class ArcPlotter3D:
    """
    鐢ㄤ簬鍦?PyVista 涓彲瑙嗗寲 red_groups_corrected 涓殑绾挎鍙婂叾瀵瑰簲鐨勭綉鏍奸潰銆?
    """
    @staticmethod
    def plot_red_groups_corrected_with_faces(results, color="red", line_width=3, face_opacity=0.6):
        """
        浣跨敤 PyVista 缁樺埗 red_groups_corrected 涓殑绾挎浠ュ強瀹冧滑瀵瑰簲鐨勭綉鏍奸潰銆?
        
        鏁版嵁鐩存帴浠?red_groups_corrected 涓殑 "group_segments" 鎻愬彇锛?
        姣忎釜 group_segments 鍏冪礌鐨勬牸寮忎负锛?
            (p1_2d, p2_2d, mid_2d, normal_vec, p1, p2, face_idx_found)
            
        鍏朵腑锛?
        - p1_2d, p2_2d: 2D 鎶曞奖鍧愭爣
        - mid_2d: 涓偣鍧愭爣
        - normal_vec: 2D 娉曞悜閲?
        - p1, p2: 瀵瑰簲鐨勫師濮?3D 绔偣
        - face_idx_found: 鍖归厤鍒扮殑闈㈢储寮曪紝濡傛灉娌℃湁鍖归厤鍒欎负 None
        """
        # 1) 鑾峰彇杩囨护鍚庣殑缃戞牸锛屽苟杞崲涓?PyVista 瀵硅薄
        filtered_mesh = results["slicing"]["filtered_mesh"]
        if filtered_mesh is None:
            print("鏈壘鍒版湁鏁堢殑杩囨护鍚庣綉鏍硷紒鏃犳硶缁樺埗闈€?)
            return
        mesh_pv = pv.wrap(filtered_mesh)
        
        # 2) 鍒涘缓 PyVista 缁樺浘瀵硅薄锛屽苟娣诲姞鑳屾櫙缃戞牸
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_pv, color="lightgray", opacity=0.4, show_edges=True)
        
        # 3) 鑾峰彇 red_groups_corrected 鏁版嵁
        red_groups_corrected = results.get("red_groups_corrected", {})
        if not red_groups_corrected:
            print("red_groups_corrected 涓虹┖锛屾病鏈夊彲缁樺埗鐨勭孩绾挎銆?)
            return

        # 4) 閬嶅巻 red_groups_corrected锛屼粠 group_segments 涓彁鍙栫嚎娈靛強闈㈢储寮?
        all_face_indices = set()
        for parent_idx, groups in red_groups_corrected.items():
            for group in groups:
                group_segments = group.get("group_segments", [])
                for seg in group_segments:
                    # seg 鏍煎紡: (p1_2d, p2_2d, mid_2d, normal_vec, p1, p2, face_idx_found)
                    p1_2d, p2_2d, mid_2d, normal_vec, p1, p2, face_idx_found = seg
                    # 缁樺埗绾挎锛氫娇鐢ㄥ師濮嬬殑 3D 绔偣 p1 鍜?p2
                    segment_points = np.array([p1, p2])
                    lineset = pv.lines_from_points(segment_points)
                    plotter.add_mesh(lineset, color=color, line_width=line_width)
                    
                    # 濡傛灉 face_idx_found 鏈夊€硷紝鍒欐敹闆?
                    if face_idx_found is not None:
                        all_face_indices.add(face_idx_found)
        
        # 5) 濡傛灉鍖归厤鍒伴潰绱㈠紩锛屽垯鎻愬彇瀵瑰簲鐨勯潰锛屽苟楂樹寒鏄剧ず
        if all_face_indices:
            face_list = list(all_face_indices)
            face_selection = mesh_pv.extract_cells(face_list)
            if face_selection.n_cells > 0:
                plotter.add_mesh(face_selection, color=color, show_edges=False, opacity=face_opacity)
        
        plotter.add_title("red_groups_corrected绾挎 + 鐩稿簲Face灞曠ず")
        plotter.show()

class ArcPlotter:
    """灏佽鍚勭浜岀淮鍜屼笁缁寸粨鏋滃睍绀哄嚱鏁?""

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


            ax.plot(new_points_2d[:, 0], new_points_2d[:, 1], color="red", linewidth=2, label="瀹屾暣璺緞")


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


            # 鏍囨敞 AB 鐐癸紝鍔犵矖
            A = new_points_2d[0]
            ax.text(A[0]+4, A[1]+2, "宕栭《", fontproperties=font_zh_13, ha="center", va="top",
                    color="black", weight="bold")
            B = new_points_2d[-1]
            ax.text(B[0]-4, B[1]-3, "宕栬剼", fontproperties=font_zh_13, ha="center", va="bottom",
                    color="black", weight="bold")

            # 鏋勯€犵澶村浘渚嬪彞鏌勶紙娉ㄦ剰 facecolor 璁剧疆涓洪粦鑹诧級
            arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
            # 娣诲姞鍥句緥锛屽寘鍚爣棰?
            legend = ax.legend(
                [arrow_legend],
                [f"{azimuth:.1f}掳"],
                loc='upper right',
                bbox_to_anchor=(1.0, 1.0),
                handler_map={FancyArrowPatch: HandlerArrow()},
                prop=font_en,
                frameon=True,
                title="鍓栭潰鏂逛綅瑙?
            )
            # 璁剧疆鏍囬瀛椾綋涓轰腑鏂囧瓧浣?
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
        缁樺埗涓€涓斁澶х殑璺緞瀛愬浘锛屽甫绛夋瘮渚嬪潗鏍囧拰鍒诲害缃戞牸銆?
        
        鍙傛暟锛?
            points_2d : (N,2) 鐨?ndarray锛屼簩缁存姇褰辩偣搴忓垪
            x_center, y_center : 鏄剧ず鍖哄煙涓績鐐瑰潗鏍?
            width, height : 鏄剧ず鍖哄煙鐨勫搴﹀拰楂樺害锛堝崟浣嶏細绫筹級
            azimuth : 鑻ユ彁渚涳紝灏嗘樉绀烘柟鍚戠澶村浘渚嬶紙鍙€夛級
            font_en : 鑻辨枃瀛椾綋璁剧疆
            font_zh : 涓枃瀛椾綋璁剧疆锛堢敤浜庡浘渚嬫爣棰橈級
            ax : 鍙€夌殑 matplotlib 瀛愬浘瀵硅薄锛岃嫢涓嶄紶鍏ヨ嚜鍔ㄥ垱寤?
        """
        #澶勭悊points_2d鐨勯噸鏂版姇褰?

        new_points_2d = points_2d

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # 鍧愭爣鑼冨洿
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        # 缁樺埗瀹屾暣璺緞
        ax.plot(new_points_2d[:, 0], new_points_2d[:, 1], color="red", linewidth=2, label="瀹屾暣璺緞")
        '''
        # # 姝ｄ氦鍚戦噺鍦虹ず鎰?
        # for i in range(len(new_points_2d) - 1):
        #     p_start = new_points_2d[i]
        #     p_end = new_points_2d[i + 1]
        #     vec = p_end - p_start
        #     norm = np.linalg.norm(vec)
        #     if norm == 0:
        #         continue

        #     # 鍗曚綅鏂瑰悜鍚戦噺
        #     dir_unit = vec / norm
        #     # 閫嗘椂閽堟棆杞?0搴︾殑娉曞悜閲忥紙鏈缉鏀撅級
        #     normal_vec = np.array([-dir_unit[1], dir_unit[0]])

        #     # 浠ョ嚎娈典腑鐐逛负璧风偣
        #     mid_point = (p_start + p_end) / 2
        #     length = 0.2  # 鍙皟锛氭硶鍚戦噺绠ご闀垮害
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
        # 缁樺埗鏂瑰悜鍚戦噺锛堜笉鐢诲畬鏁磋矾寰勭嚎锛?
        # for i in range(len(new_points_2d) - 1):
        #     p_start = new_points_2d[i]
        #     p_end = new_points_2d[i + 1]
        #     direction = p_end - p_start
        #     norm = np.linalg.norm(direction)
        #     if norm == 0:
        #         continue

        #     # 绠ご鍙湪鍘熺嚎娈佃寖鍥村唴锛岀暐寰缉鐭伩鍏嶉噸鍙?
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

        # # 鑻ユ彁渚涙柟鍚戣锛屾坊鍔犵澶村浘渚?
        # if azimuth is not None:
        #     arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        #     legend = ax.legend(
        #         [arrow_legend],
        #         [f"{azimuth:.1f}掳"],
        #         loc='upper right',
        #         bbox_to_anchor=(1.0, 1.0),
        #         handler_map={FancyArrowPatch: HandlerArrow()},
        #         prop=font_en,
        #         frameon=True,
        #         title="鍓栭潰鏂逛綅瑙?
        #     )
        #     if font_zh:
        #         legend.get_title().set_fontproperties(font_zh)
        '''
        # 璁剧疆鍧愭爣杞翠笌鏍峰紡
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(x_min, x_max, 4))
        ax.set_yticks(np.linspace(y_min, y_max, 4))
        # 鍧愭爣鍒诲害鏍煎紡鍖栵細淇濈暀涓€浣嶅皬鏁?
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
    #         # 璁惧畾涓変釜鏀惧ぇ鍖哄煙涓績鐐?

    #         # ========= 甯冨眬 =========
    #         fig = plt.figure(figsize=(10, 8))
    #         gs  = GridSpec(3, 2, width_ratios=[1.5, 1], wspace=0.15, hspace=0.3)
    #         centers = [(26.956, 107.573), (50.04, 61.30), (70.43, 20.57)]
    #         # centers = [(47.6, 69.8), (63.25, 36.06), (78.10, 11.38)]
    #         ax_full = fig.add_subplot(gs[:, 0])
    #         ax_zoom = [fig.add_subplot(gs[i, 1]) for i in range(len(centers))]

    #         # ---------- 閫氱敤缁樺埗 ----------
    #         def draw_all(ax):
    #             if points_2d.size:
    #                 ax.plot(points_2d[:,0], points_2d[:,1],
    #                         color="red", lw=2)
    #                         # # 姝ｄ氦鍚戦噺鍦虹ず鎰?


    #         # ---------- 宸﹀垪瀹屾暣 ----------
    #         draw_all(ax_full)
    #         ax_full.grid(True, ls="--", color="grey")
    #         # 鏂瑰悜绠ご鍥句緥
    #         x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
    #         y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
    #         ax_full.set_xticks(np.arange(x_min, x_max + 10, 10))
    #         ax_full.set_yticks(np.arange(y_min, y_max + 10, 10))
    #         # === 鏂扮殑棰滆壊鍒嗙粍鍥句緥 ===
    #         x_center = (x_min + x_max) / 2
    #         y_center = (y_min + y_max) / 2
    #         max_range = max(x_max - x_min, y_max - y_min)
    #         ax_full.set_xlim(x_center - max_range/2, x_center + max_range/2)
    #         ax_full.set_ylim(y_center - max_range/2 - 5, y_center + max_range/2 + 5)


    #         # 鏍囨敞 AB 鐐癸紝鍔犵矖
    #         A = points_2d[0]
    #         ax_full.text(A[0]+4, A[1]+2, "宕栭《", fontproperties=font_zh_13, ha="center", va="top",
    #                 color="black", weight="bold")
    #         B = points_2d[-1]
    #         ax_full.text(B[0]-4, B[1]-3, "宕栬剼", fontproperties=font_zh_13, ha="center", va="bottom",
    #                 color="black", weight="bold")
    #         # 鏋勯€犵澶村浘渚嬪彞鏌勶紙娉ㄦ剰 facecolor 璁剧疆涓洪粦鑹诧級
    #         arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
    #         # 娣诲姞鍥句緥锛屽寘鍚爣棰?
    #         legend = ax_full.legend(
    #             [arrow_legend],
    #             [f"{azimuth:.1f}掳"],
    #             loc='upper right',
    #             bbox_to_anchor=(1.0, 1.0),
    #             handler_map={FancyArrowPatch: HandlerArrow()},
    #             prop=font_en,
    #             frameon=True,
    #             title="鍓栭潰鏂逛綅瑙?
    #         )
    #         # 璁剧疆鏍囬瀛椾綋涓轰腑鏂囧瓧浣?
    #         legend.get_title().set_fontproperties(font_zh)
    #         ax_full.set_aspect("equal", adjustable="datalim")
    #         ax_full.set_xlabel("(m)", fontproperties=font_en_13)
    #         ax_full.set_ylabel("(m)", fontproperties=font_en_13)

    #         # ---------- 鍙冲垪鏀惧ぇ ----------
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

    #                 # 鍗曚綅鏂瑰悜鍚戦噺
    #                 dir_unit = vec / norm
    #                 # 閫嗘椂閽堟棆杞?0搴︾殑娉曞悜閲忥紙鏈缉鏀撅級
    #                 normal_vec = np.array([-dir_unit[1], dir_unit[0]])

    #                 # 浠ョ嚎娈典腑鐐逛负璧风偣
    #                 mid_point = (p_start + p_end) / 2
    #                 length = 0.3  # 鍙皟锛氭硶鍚戦噺绠ご闀垮害
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

    #                 # # 绠ご鍙湪鍘熺嚎娈佃寖鍥村唴锛岀暐寰缉鐭伩鍏嶉噸鍙?
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
    #     灞曠ず points_2d 鍦ㄤ笉鍚屾棆杞搴︿笅鐨?2脳2 瀛愬浘瑙嗗浘
    #     - 绗?骞咃細鍘熷鏂瑰悜
    #     - 绗?骞咃細鏃嬭浆90搴?
    #     - 绗?骞咃細鏃嬭浆180搴?
    #     - 绗?骞咃細鏃嬭浆270搴?
    #     """
    #     def rotate_points(points, angle_deg, center):
    #         """缁?center 閫嗘椂閽堟棆杞?angle_deg 瑙掑害"""
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

    #         # 璁惧畾鏃嬭浆瑙掑害鍒楄〃
    #         angles = [0, 90, 180, 270]
    #         titles = ["0掳", "90掳", "180掳", "270掳"]

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

        # ======= 绗竴姝ワ細鎻愬彇 merged_open_subsets锛堢粯鍒剁伆鑹插畬鏁磋矾寰勶級=======
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

        # ======= 绗簩姝ワ細鎻愬彇 red_groups 涓墍鏈夌嚎娈碉紙p1, p2 鏄姇褰卞悗浜岀淮鍧愭爣锛?======
        red_segments = []
        for segment in red_groups:
            p1, p2 = np.array(segment[0]), np.array(segment[1])
            red_segments.append((p1, p2))
            all_points.append(np.vstack([p1, p2]))

        # ======= 绗笁姝ワ細璁＄畻缁熶竴骞崇Щ鍘熺偣 =======
        all_concat = np.vstack(all_points)
        new_origin = np.array([
            np.min(all_concat[:, 0]) - 5,
            np.min(all_concat[:, 1]) - 5
        ])

        # ======= 绗洓姝ワ細缁樺浘 =======

        shifted = points_2d - new_origin
        ax.plot(shifted[:, 0], shifted[:, 1], color="lightgray", linewidth=2, label="瀹屾暣璺緞")
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
        # 鏍囨敞 AB 鐐癸紝鍔犵矖
        A = shifted[0]
        ax.text(A[0]+4, A[1]+2, "宕栭《", fontproperties=font_zh_13, ha="center", va="top",
                color="black", weight="bold")
        B = shifted[-1]
        ax.text(B[0]-4, B[1]-3, "宕栬剼", fontproperties=font_zh_13, ha="center", va="bottom",
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
        浠呰缃斁澶ц绐椾笌鍧愭爣鏍煎紡锛屼笉涓诲姩缁樺埗浠讳綍鏇茬嚎銆?

        鍙傛暟
        ----
        points_2d : ndarray
            (N, 2)锛岀敤浜庤皟鏁存暟鎹樉绀鸿寖鍥达紙鍙彁鍓嶈鍓級銆?
        x_center, y_center : float
            瑙嗙獥涓績鍧愭爣銆?
        width, height : float
            瑙嗙獥瀹介珮锛堢背锛夈€?
        font_en : FontProperties
            鑻辨枃瀛椾綋锛涘潗鏍囪酱鏍囩浣跨敤銆?
        ax : matplotlib.axes.Axes
            鐩爣瀛愬浘锛涘繀椤荤敱澶栭儴鍒涘缓骞朵紶鍏ャ€?

        杩斿洖
        ----
        ax : matplotlib.axes.Axes
            缁忚繃璁剧疆鐨勫瓙鍥撅紝渚夸簬閾惧紡璋冪敤銆?
        """
        if ax is None:
            raise ValueError("璇峰厛鍦ㄥ閮ㄥ垱寤?ax锛屽苟浣滀负鍙傛暟浼犲叆锛?)

        # 瑙嗙獥鑼冨洿
        x_min, x_max = x_center - width / 2,  x_center + width / 2
        y_min, y_max = y_center - height / 2, y_center + height / 2

        # 鍧愭爣杞磋缃?
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        # 鍒诲害
        ax.set_xticks(np.linspace(x_min, x_max, 4))
        ax.set_yticks(np.linspace(y_min, y_max, 4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # 缃戞牸 & 鏍囩
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
        宸﹀垪涓€骞呭畬鏁村浘 + 鍙冲垪涓夊箙鏀惧ぇ鍥撅紙鍏ㄩ儴鍖呭惈瀹屾暣璺緞涓庣孩绾挎锛?
        """
        # ---------- 棰勫鐞?----------
        red_groups   = results.get("red_groups", [])
        origin       = results["plane_params"]["origin"]
        radial_dir   = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        azimuth = results["plane_params"]['azimuth']
        # 鐏拌壊瀹屾暣璺緞
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

        # 绾㈣壊绾挎
        red_segments = [(np.array(seg[0]), np.array(seg[1])) for seg in red_groups]

        # 缁熶竴骞崇Щ
        concat_pts = [points_full] + [np.vstack(s) for s in red_segments] if points_full.size else [np.vstack(s) for s in red_segments]
        all_concat = np.vstack(concat_pts)
        new_origin = np.array([all_concat[:,0].min() - 5, all_concat[:,1].min() - 5])
        points_full_shifted = points_full - new_origin
        red_segments_shifted = [(p1-new_origin, p2-new_origin) for p1, p2 in red_segments]

        # ---------- 甯冨眬 ----------
        fig = plt.figure(figsize=(10, 8))
        gs  = GridSpec(3, 2, width_ratios=[1.5, 1], wspace=0.15, hspace=0.3)
        ax_full = fig.add_subplot(gs[:, 0])
        ax_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]

        # ---------- 涓€涓€氱敤缁樺埗鍑芥暟 ----------
        def draw_full_and_segments(ax):
            if points_full_shifted.size:
                ax.plot(points_full_shifted[:,0], points_full_shifted[:,1],
                        color="lightgray", lw=2)
            for p1, p2 in red_segments_shifted:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", lw=2)

        # ---------- 宸﹀垪瀹屾暣鍥?----------
        draw_full_and_segments(ax_full)
        ax_full.grid(True, ls="--", color="grey")
        x_min, x_max = np.min(points_full_shifted[:, 0]), np.max(points_full_shifted[:, 0])
        y_min, y_max = np.min(points_full_shifted[:, 1]), np.max(points_full_shifted[:, 1])
        ax_full.set_xticks(np.arange(x_min, x_max + 10, 10))
        ax_full.set_yticks(np.arange(y_min, y_max + 10, 10))
        ax_full.grid(True, linestyle="--", color="grey")
        # 鏋勯€犵澶村浘渚嬪彞鏌勶紙娉ㄦ剰 facecolor 璁剧疆涓洪粦鑹诧級
        arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        # 娣诲姞鍥句緥锛屽寘鍚爣棰?
        legend = ax_full.legend(
            [arrow_legend],
            [f"{azimuth:.1f}掳"],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            handler_map={FancyArrowPatch: HandlerArrow()},
            prop=font_en_13,
            frameon=True,
            title="鍓栭潰鏂逛綅瑙?
        )
        legend.get_title().set_fontproperties(font_zh_13)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        max_range = max(x_max - x_min, y_max - y_min)
        ax_full.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax_full.set_ylim(y_center - max_range/2 - 5, y_center + max_range/2 + 5)


        # 鏍囨敞 AB 鐐癸紝鍔犵矖
        A = points_full_shifted[0]
        ax_full.text(A[0]+4, A[1]+2, "宕栭《", fontproperties=font_zh_13, ha="center", va="top",
                color="black", weight="bold")
        B = points_full_shifted[-1]
        ax_full.text(B[0]-4, B[1]-3, "宕栬剼", fontproperties=font_zh_13, ha="center", va="bottom",
                color="black", weight="bold")
        ax_full.set_xlabel("(m)", fontproperties=font_en_13)
        ax_full.set_ylabel("(m)", fontproperties=font_en_13)
        ax_full.set_aspect("equal", adjustable="datalim")

        # ---------- 鍙冲垪鏀惧ぇ鍥?----------
        for idx, (xc, yc) in enumerate(centers):
            ax = ax_zoom[idx]
            # 缁樺埗鐏?绾?
            draw_full_and_segments(ax)
            # 璁剧疆鏀惧ぇ绐楀彛
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
        宸﹀垪瀹屾暣鍥?+ 鍙冲垪 top_n 涓川蹇冩斁澶у浘
        - 鍒嗙粍绾挎棰滆壊涓庡師鍑芥暟涓€鑷?
        - 璐ㄥ績鐢ㄥ悓鑹插渾鐐硅〃绀?
        """
        # ========= 鍩烘湰鏁版嵁 =========
        red_groups_corrected = results.get("red_groups_corrected", {})
        red_centroids        = results.get("red_centroids", {})
        origin, radial_dir, vertical_dir = (results["plane_params"][k]
                                            for k in ("origin", "radial_dir", "vertical_dir"))
        azimuth = results["plane_params"]["azimuth"]

        # ---------- 鐏拌壊瀹屾暣璺緞 ----------
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

        # ---------- 褰╄壊鍒嗙粍 & 璐ㄥ績 ----------
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

        # ---------- 缁熶竴骞崇Щ ----------
        all_pts = [points_full] + [r[2] for r in group_records]
        if any(r[3] is not None for r in group_records):
            all_pts += [np.array(r[3]).reshape(1,2) for r in group_records if r[3] is not None]
        all_concat = np.vstack(all_pts)
        new_origin = np.array([all_concat[:,0].min()-5, all_concat[:,1].min()-5])

        points_full_s = points_full - new_origin
        group_records_s = [(L, col, pts - new_origin, (np.array(c)-new_origin) if c is not None else None)
                        for L, col, pts, c in group_records]

        # ---------- 閫夊彇 top_n 璐ㄥ績浣滀负鏀惧ぇ涓績 ----------
        group_records_sorted = sorted(group_records_s, key=lambda x: x[0], reverse=True)
        centers = [rec[3] for rec in group_records_sorted[:top_n] if rec[3] is not None]

        # ========= 甯冨眬 =========
        fig = plt.figure(figsize=(10, 8))
        gs  = GridSpec(3, 2, width_ratios=[1.5, 1], wspace=0.15, hspace=0.3)

        ax_full = fig.add_subplot(gs[:, 0])
        ax_zoom = [fig.add_subplot(gs[i, 1]) for i in range(len(centers))]

        # ---------- 閫氱敤缁樺埗 ----------
        def draw_all(ax):
            if points_full_s.size:
                ax.plot(points_full_s[:,0], points_full_s[:,1],
                        color="lightgray", lw=2)
            for _, col, pts, cen in group_records_s:
                ax.plot(pts[:,0], pts[:,1], color=col, lw=2)
                # if cen is not None:
                #     ax.scatter(cen[0], cen[1], color=col, s=25, zorder=3)

        # ---------- 宸﹀垪瀹屾暣 ----------
        draw_all(ax_full)
        ax_full.grid(True, ls="--", color="grey")
        # 鏂瑰悜绠ご鍥句緥
        x_min, x_max = np.min(points_full_s[:, 0]), np.max(points_full_s[:, 0])
        y_min, y_max = np.min(points_full_s[:, 1]), np.max(points_full_s[:, 1])
        ax_full.set_xticks(np.arange(x_min, x_max + 10, 10))
        ax_full.set_yticks(np.arange(y_min, y_max + 10, 10))
        # === 鏂扮殑棰滆壊鍒嗙粍鍥句緥 ===
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
                                title="鎶曞奖缁撴瀯缁刓n(鎸夌収闀垮害鎺掑簭)",
                                loc="upper right",
                                prop=font_en_13,
                                frameon=True)
        legend.get_title().set_fontproperties(font_zh_13)
        # 鏍囨敞 AB 鐐癸紝鍔犵矖
        A = points_full_s[0]
        ax_full.text(A[0]+4, A[1]+2, "宕栭《", fontproperties=font_zh_13, ha="center", va="top",
                color="black", weight="bold")
        B = points_full_s[-1]
        ax_full.text(B[0]-4, B[1]-3, "宕栬剼", fontproperties=font_zh_13, ha="center", va="bottom",
                color="black", weight="bold")

        ax_full.set_aspect("equal", adjustable="datalim")
        ax_full.set_xlabel("(m)", fontproperties=font_en_13)
        ax_full.set_ylabel("(m)", fontproperties=font_en_13)

        # ---------- 鍙冲垪鏀惧ぇ ----------
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

        # 缁樺埗瀹屾暣璺緞锛堥€夊彇鑺傜偣鏁版渶澶氱殑璺緞锛?
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
            ax.plot(points_2d[:, 0], points_2d[:, 1], color="lightgray", linewidth=2, label="瀹屾暣璺緞")

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
                ax.plot(points_2d[:, 0], points_2d[:, 1], color=color, linewidth=2, label=f'缁?{parent_idx}-{idx+1}')
                if idx < len(centroids):
                    centroid = centroids[idx]
                    ax.scatter(centroid[0], centroid[1], color=color, marker='o', s=20)
        legend_elements = [Line2D([0], [0], color=colors[i], lw=3, label=f'C{i+1}') for i in range(k)]
        legend_elements.append(Line2D([0], [0], color=default_color, lw=3, label=f'C{k+1}-Cn'))
        legend = ax.legend(handles=legend_elements, title="鎶曞奖璐ㄥ績鍜岀嚎娈电粍\n(鎸夌収闀垮害鎺掑簭)", loc="upper right", prop=font_en)
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
        # 鍦ㄨ儗鏅腑缁樺埗鎵€鏈夊垏鍓茬嚎娈?
        for ax in axs:
            if intersections is not None:
                for seg in intersections:
                    p1, p2, face_idx = seg
                    p1_2d = ArcUtils.project_to_plane([p1], origin, radial_dir, vertical_dir)[0]
                    p2_2d = ArcUtils.project_to_plane([p2], origin, radial_dir, vertical_dir)[0]
                    ax.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color="black", linewidth=1)

        cmap = plt.cm.get_cmap("tab20")
        color_mapping = {}
        # 寤虹珛鑹插僵绱㈠紩
        for parent_idx, groups in unfiltered.items():
            mapping = {}
            for group in groups:
                node_order = group["node_order"]
                if len(node_order) < 2:
                    continue
                # 浠呯敤棣栧熬鑺傜偣2D淇℃伅鍋氬尯鍒?
                id_key = (tuple(np.round(node_order[0], 6)), tuple(np.round(node_order[-1], 6)))
                mapping[id_key] = None
            for i, key in enumerate(mapping.keys()):
                mapping[key] = cmap(i)
            color_mapping[parent_idx] = mapping

        # 宸﹀浘锛氭湭绛涢€?
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
        axs[0].set_title("鏍℃鍚庯紙鏈瓫閫夛級")
        axs[0].set_xlabel("寰勫悜 (m)")
        axs[0].set_ylabel("鍨傜洿 (m)")
        axs[0].set_aspect("equal", adjustable="datalim")

        # 鍙冲浘锛氱瓫閫夊悗
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
        axs[1].set_title(f"绛涢€夊悗锛坸鎶曞奖璺濈>={min_distance}锛?)
        axs[1].set_xlabel("寰勫悜 (m)")
        axs[1].set_ylabel("鍨傜洿 (m)")
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

        # 鈥斺€?鎵惧嚭鏈€浣冲瓙闆嗗苟鎻愬彇鑺傜偣 鈥斺€?
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
            # 鈥斺€?鎶曞奖鍒板钩闈㈠苟骞崇Щ鍒?new_origin 鈥斺€?
            pts2d_raw = ArcUtils.project_to_plane(ordered_nodes, origin, radial_dir, vertical_dir)
            new_origin = np.array([
                np.min(pts2d_raw[:, 0]) - 5,
                np.min(pts2d_raw[:, 1]) - 5
            ])
            pts2d = pts2d_raw - new_origin

            # 缁樺埗涓诲墫闈㈢嚎
            plt.plot(pts2d[:, 0], pts2d[:, 1], color='k', linewidth=2)

            # 濉厖鐏拌壊鍖哄煙绀轰緥
            start_pt, end_pt = pts2d[0], pts2d[-1]
            offset_val = 10
            left_start = start_pt - np.array([offset_val, 0])
            left_end   = end_pt   - np.array([end_pt[0] - left_start[0], 0])
            poly_pts = np.vstack((pts2d, [left_end], [left_start]))
            plt.fill(poly_pts[:, 0], poly_pts[:, 1], color='gray', alpha=0.2)

            # 鏍囨敞鈥滃礀椤垛€濓紝浣嶇疆涔熻鍑忓幓 new_origin
            label_pos = pts2d[0]
            plt.text(label_pos[0], label_pos[1], "宕栭《",
                    fontproperties=font_zh_13, color="black")

            # 鍙€夛細娌夸富鍓栭潰鐢荤澶?
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

        # 鈥斺€?缁樺埗鎵€鏈夌孩绾挎缁?鈥斺€?
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

        # 鈥斺€?鏋勯€犵澶村浘渚嬪苟娣诲姞鏍囬 鈥斺€?
        arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        legend = plt.legend(
            [arrow_legend],
            [f"{azimuth:.1f}掳"],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            handler_map={FancyArrowPatch: HandlerArrow()},
            prop=font_en_13,
            frameon=True,
            title="鍓栭潰鏂逛綅瑙?
        )
        legend.get_title().set_fontproperties(font_zh_13)

        # 鈥斺€?鍧愭爣杞磋缃?鈥斺€?
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

        # 鍩烘湰鍙傛暟
        origin     = results["plane_params"]["origin"]
        radial_dir = results["plane_params"]["radial_dir"]
        vertical_dir = results["plane_params"]["vertical_dir"]
        merged_subsets = results["paths"].get("merged_open_subsets", [])
        azimuth    = results["plane_params"]['azimuth']

        # 1. 鎵惧埌 best_subset
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
            return  # 娌℃湁鏈夋晥瀛愰泦

        # 2. 鎻愬彇鏈夊簭鑺傜偣鍒楄〃
        ordered_nodes = []
        for p1, p2, _ in best_subset:
            if not ordered_nodes:
                ordered_nodes.append(np.array(p1))
            ordered_nodes.append(np.array(p2))
        ordered_nodes = np.array(ordered_nodes)  # (N,3)

        # 3. 鎶曞奖鍒板墫闈㈠钩闈?鈫?璁＄畻 new_origin 鈫?骞崇Щ
        pts2d_raw = ArcUtils.project_to_plane(ordered_nodes,
                                            origin, radial_dir, vertical_dir)
        new_origin = np.array([
            np.min(pts2d_raw[:, 0]) - 5,
            np.min(pts2d_raw[:, 1]) - 5
        ])
        pts2d = pts2d_raw - new_origin

        # 4. 缁樺埗涓诲墫闈紙鍏堣繕鍘熷洖涓夌淮锛?
        pts3d = ArcUtils.convert_2d_points_to_3d(pts2d,
                                                origin, radial_dir, vertical_dir)
        ax.plot(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
                color='k', linewidth=2)

        # 5. 鐏拌壊闈㈢墖濉厖
        start_pt2d, end_pt2d = pts2d[0], pts2d[-1]
        offset = np.array([0, 0])  # 濡傛灉闇€瑕佸亸绉诲彲鍦ㄨ繖閲岃皟鏁?
        left_start = start_pt2d - offset
        left_end = end_pt2d - np.array([end_pt2d[0] - left_start[0], 0])
        poly2d = np.vstack((pts2d, [left_end], [left_start]))
        poly3d = ArcUtils.convert_2d_points_to_3d(poly2d,
                                                origin, radial_dir, vertical_dir)
        poly_coll = Poly3DCollection([poly3d],
                                    facecolor='gray', alpha=0.2,
                                    edgecolor='none')
        ax.add_collection3d(poly_coll)

        # 6. 鈥滃礀椤垛€濇枃瀛?
        label3d = ArcUtils.convert_2d_points_to_3d([start_pt2d],
                                                origin, radial_dir, vertical_dir)[0]
        ax.text(label3d[0], label3d[1], label3d[2],
                "宕栭《", fontproperties=font_zh_13, color="black")

        # 7. 鍙€夛細鍦ㄤ笁缁翠富绾挎涓婂姞绠ご锛堢暐锛屽彲鎸?D绀轰緥绉绘锛?

        # 8. 绾㈣壊绾挎缁勪篃鍚屾牱骞崇Щ缁樺埗
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

        # 9. 鍓栭潰鏂逛綅瑙掑浘渚?
        arrow_legend = FancyArrowPatch((0, 0), (1, 0), facecolor='k')
        legend = plt.legend(
            [arrow_legend],
            [f"{azimuth:.1f}掳"],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            handler_map={FancyArrowPatch: HandlerArrow()},
            prop=font_en_13,
            frameon=True,
            title="鍓栭潰鏂逛綅瑙?
        )
        legend.get_title().set_fontproperties(font_zh_13)

        # 10. 鍧愭爣杞存爣绛?& 绛夋瘮渚?
        ax.set_xlabel("(m)", fontproperties=font_en_13)
        ax.set_ylabel("(m)", fontproperties=font_en_13)
        ax.set_zlabel("(m)", fontproperties=font_en_13)
        ArcPlotter.set_axes_equal(ax)

        plt.show()
# =============================================================================
# 3. 鏍稿績鍓栭潰鍒囧壊鍣?
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
            "intersections": intersections  # 杩欓噷鏄甫鏈夐潰绱㈠紩鐨勫垪琛?
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
            # threshold鍙嚜琛岃皟鏁?
            ordered_red = ArcUtils.get_ordered_red_segments_for_path(subset, origin, radial_dir, vertical_dir, threshold=-0.1)
            if ordered_red:
                # 杩欓噷 tol=-0.05 浠呬綔绀轰緥
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
# 涓诲嚱鏁板叆鍙ｇず渚?
# =============================================================================

def main():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    config_path = r"configs/arc_config.example.json"
    mesh_path = r"data/private/raw_model\combined_model.glb"
    # mesh_path = r"data/examples/segmented_mesh\Merged mesh_seg_3.ply"
    # mesh_path = r"data/examples/sample_outcrop.ply"

    # 璁剧疆鍒囧墫闈綅缃紝渚嬪鍙?43.5锛涜嫢涓篘one锛屽垯榛樿鍙栦腑鐐?
    slicer = ArcSlicer(config_path, mesh_path, slice_position=62, use_merged_paths=True)
    results = slicer.run_all()

    # 鎵撳嵃閮ㄥ垎淇℃伅
    data = results["data_loading"]
    angle_min = data["angle_min"]
    angle_max = data["angle_max"]
    arc_length_range = data["arc_length_range"]
    turning_angle = angle_max - angle_min
    slice_angle = results["plane_params"]["slice_angle"]
    print("鍘熷瑙掑害鑼冨洿锛堝姬搴︼級锛?, angle_min, "鑷?, angle_max)
    print("寮ч暱鑼冨洿锛?, arc_length_range)
    print("鍦嗗姬杞锛堝姬搴︼級锛?, turning_angle)
    print("閫夋嫨鐨勫垏鍓栭潰瑙掑害锛堝姬搴︼級锛?, slice_angle)

    # 璁＄畻姣忎釜绾㈡缁勭殑2D璐ㄥ績骞跺彲瑙嗗寲
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



#results璇存槑鏂囨。
"""
涓嬮潰鏄鏈剼鏈湪鎵ц鍚?`results` 瀛楀吀涓悇涓敭鐨勮鏄庯紝浠ュ強瀹冧滑鎵€瀵瑰簲鐨勬暟鎹惈涔夛紝鏂逛究浣犱簡瑙ｅ拰浣跨敤鍒囧墫闈㈠強鍚庣画澶勭悊鎵€鑾峰緱鐨勪俊鎭€?

---

## 1. `results["data_loading"]`
**鍐呭**锛?
```python
{
    "center": center,              # 鍦嗗姬涓績(3D鍧愭爣)锛屼粠 config 涓В鏋?
    "radius": radius,              # 鍦嗗姬鍗婂緞
    "angle_min": angle_min,        # 鍦嗗姬璧峰瑙掑害(寮у害)
    "angle_max": angle_max,        # 鍦嗗姬缁撴潫瑙掑害(寮у害)
    "arc_length_range": arc_length_range,   # [0, 寮ч暱]
    "z_range": (z_min, z_max),     # Z鏂瑰悜鐨勮繃婊よ寖鍥?
    "mesh": mesh                   # 鍘熷鍔犺浇骞惰浆鎴?trimesh 鐨勫畬鏁?3D 缃戞牸
}
```
**鎰忎箟**锛?
- 涓昏鏄鐢ㄦ埛閰嶇疆鍜岀綉鏍兼暟鎹殑璁板綍锛?
  - `center`/`radius`/`angle_min`/`angle_max`锛氫粠 `arc_config.json` 涓幏鍙栵紝鐢ㄦ潵鎻忚堪鍦嗗姬鍒囧墫闈㈡墍鍦ㄧ殑涓績銆佸崐寰勫拰瑙掑害鍖洪棿銆?
  - `arc_length_range`锛氭牴鎹?`radius*(angle_max - angle_min)` 鑾峰緱鏁翠釜寮х殑寮ч暱鑼冨洿銆?
  - `z_range`锛氬垏鍓栭潰鍓嶅缃戞牸鍋?Z 鏂瑰悜杩囨护锛堝彧淇濈暀 [z_min, z_max] 涓婄殑闈級銆?
  - `mesh`锛歵rimesh 鍔犺浇鍚庣殑缃戞牸瀵硅薄銆?

---

## 2. `results["plane_params"]`
**鍐呭**锛?
```python
{
    "origin": origin,       # 鍓栭潰骞抽潰缁忚繃鐨勫師鐐?3D鍧愭爣) 锛屽嵆璁＄畻寰楀埌鐨勫垏绾夸綅缃?
    "normal": normal,       # 鍓栭潰骞抽潰鐨勬硶鍚戦噺 (3D)
    "radial_dir": radial_dir,     # 鍒囧墫闈㈢殑鈥滃緞鍚戔€濇柟鍚戝悜閲?3D)
    "vertical_dir": vertical_dir, # 鍒囧墫闈㈠湪涓栫晫鍧愭爣绯讳笅鐨勫瀭鐩存柟鍚?3D)锛岃繖閲岄粯璁ょ敤Z杞碵0,0,1]
    "slice_angle": slice_angle    # 鏈€缁堥€夊畾鐨勫垏鍓栭潰瑙掑害(寮у害)锛屽喅瀹?radial_dir 
}
```
**鎰忎箟**锛?
- 鐢ㄤ簬鎻忚堪鈥滃垏鍓栭潰鈥濆湪涓夌淮绌洪棿涓殑鍏蜂綋浣嶇疆鍜屾柟鍚戯細
  - `origin`锛氬钩闈㈣繃鐐?
  - `normal`锛氬钩闈㈡硶鍚戦噺锛岀敤浜庡 trimesh 鍋?plane-slice
  - `radial_dir`锛氬垏鍓栭潰鍦?XY 骞抽潰鎶曞奖鏃剁殑鍩哄悜閲忎箣涓€
  - `vertical_dir`锛氶€氬父鎸囧悜 Z 杞达紝鐢ㄤ簬鍦ㄥ垏鍓栭潰涓婂仛鎶曞奖鍒嗘瀽(鍨傜洿鏂瑰悜)
  - `slice_angle`锛氳鍓栭潰鐩稿浜庡渾蹇冪殑瑙掑害锛岀瓑浜?`angle_min + slice_position / radius`銆傝嫢涓嶆寚瀹?`slice_position`锛屽垯榛樿鍙栧姬闀夸腑鐐?

---

## 3. `results["slicing"]`
**鍐呭**锛?
```python
{
    "filtered_mesh": mesh_filtered,   # 缁忚繃Z鑼冨洿杩囨护涔嬪悗鐨勫瓙缃戞牸(trimesh)
    "intersections": intersections    # 鐢?slice_mesh(...) 杩斿洖鐨勭嚎娈靛強闈㈢储寮?
}
```
**鎰忎箟**锛?
- `filtered_mesh` 鏄粡杩?`filter_mesh_by_z_range` 绛涢€夊悗鍙寘鍚?[z_min, z_max] 楂樺害闈㈢墖鐨勭綉鏍笺€?
- `intersections` 鏄€滆瀛愮綉鏍尖€濅笌鈥滃垏鍓栭潰鈥濇眰浜ゅ緱鍒扮殑绾挎鍒楄〃銆? 
  姣忎釜鍏冪礌鐨勬牸寮忥細`(p1, p2, face_idx)`锛? 
  - `p1` / `p2`锛氱嚎娈典袱绔偣(3D 鍧愭爣)  
  - `face_idx`锛氳鍒囬潰绾挎鏉ユ簮浜庡師缃戞牸鐨勫摢涓潰绱㈠紩

---

## 4. `results["nodes"]`
**鍐呭**锛?
```python
{
    "unique_nodes": unique_nodes,   # 鎵€鏈夌嚎娈电鐐瑰幓閲嶅悗鐨勫潗鏍?(N,3)
    "node_connectivity": node_conn  # 姣忎釜绔偣琚繛鎺ユ鏁?{ (x,y,z): count, ... }
}
```
**鎰忎箟**锛?
- `unique_nodes`锛氬皢 `intersections` 涓嚭鐜扮殑鎵€鏈夌嚎娈电鐐瑰仛鍘婚噸鍚庣殑鑺傜偣闆嗗悎锛岀敤浜庡悗缁垎鏋愯矾寰勬槸鍚﹂棴鍚堛€佹垨鑰呭彲瑙嗗寲绔偣绛夈€?
- `node_connectivity`锛氱粺璁℃瘡涓妭鐐?绔偣)鍑虹幇鐨勬鏁?鍗充笌澶氬皯鏉＄嚎娈电浉杩?锛屽彲鍒ゆ柇姝ょ鐐规槸鍚﹀湪璺緞涓綔涓衡€滃ご鈥濃€滃熬鈥濇垨鈥滀腑闂磋妭鐐光€濄€?

---

## 5. `results["paths"]`
**鍐呭**锛?
```python
{
    "unmerged_subsets": [subset1, subset2, ...],   # 鏈繘琛屼汉宸ュ悎骞剁殑绾挎瀛愰泦
    "merged_open_subsets": [subsetA, subsetB, ...] # 鍚堝苟鍚?(鑻?use_merged_paths=True ) 鐨勭嚎娈靛瓙闆?
}
```
鍏朵腑锛?
- `unmerged_subsets`锛歚split_segments_to_paths` 寰楀埌鐨勬墍鏈夎繛閫氬瓙闆嗭紝姣忎釜杩為€氬瓙闆嗘槸鑻ュ共 `(p1, p2, face_idx)` 鐨勯泦鍚堛€?
- `merged_open_subsets`锛氶拡瀵归潪闂悎瀛愯矾寰勮繘琛屾嫾鎺ャ€佷慨姝ｅ悗寰楀埌鐨勫悎骞剁粨鏋滐紝姣忎釜鍏冪礌閲屾槸鑻ュ共 `(p1, p2, None)` 鐨勭嚎娈碉紙鏂扮殑鍚堝苟绾挎娌℃湁鍘熷闈㈢储寮曪紝鏁呬负 `None`锛夈€?

**鎰忎箟**锛?
- 鈥滆矾寰勫垎鍓测€濇楠や細灏嗘墍鏈夊垏闈㈢嚎娈垫寜鐓ц繛閫氭€ф媶鍒嗐€傚浜庢瘡涓瓙闆嗭紝鍙垽鏂槸鍚﹂棴鍚堛€佸啀瑙嗛渶瑕佸仛鍚堝苟銆? 
- `merged_open_subsets` 鏄渶缁堢殑鈥滃彲浣跨敤鐨勮矾寰勨€濓紝鍙互鍦ㄤ簩缁?涓夌淮涓粯鍒跺嚭鏈€瀹屾暣鐨勫墫闈㈢嚎銆?

---

## 6. `results["normals"]`
**鍐呭**锛?
杩欐槸涓€涓垪琛紝鍒楄〃涓瘡涓厓绱犳牸寮忓舰濡傦細
```python
{
    "subset": subset,       # path_subsets 涓師濮嬫垨鍚堝苟鍚庣殑鏌愭潯璺緞锛堢嚎娈甸泦鍚堬級
    "closed": closed,       # bool, 鏄惁闂悎
    "normals": normals      # 璁＄畻寰楀埌鐨?2D 娉曞悜淇℃伅
}
```
**鎰忎箟**锛?
- 鈥滄硶鍚戜俊鎭€濇槸鎸囧姣忎竴娈佃竟鍦?2D 鎶曞奖涓婅绠楁柟鍚戝悜閲忋€佹硶鍚戝悜閲忕瓑锛岀敤浜庡悗缁仛鍧″害/鏈濆悜涔嬬被鐨勫垎鏋愩€?
- `closed` 鐢ㄦ潵鏍囪瘑杩欎竴鏉¤矾寰勬槸鍚︽瀯鎴愪竴涓棴鍚堝洖璺€?

---

## 7. `results["red_groups"]`
**鍐呭**锛?
```python
{
    parent_idx0: [group0, group1, ...],
    parent_idx1: [group0, group1, ...],
    ...
}
```
鍏朵腑姣忎釜 `group` 鍒欐槸涓€缁勨€滅孩绾挎鈥濈殑 2D 鎶曞奖淇℃伅锛坄(p1_2d, p2_2d, mid_2d, normal_vec)` 绛夛級锛屾簮浜?`get_ordered_red_segments_for_path` 鍜?`group_red_segments_by_connection_order` 鐨勭粨鏋溿€?

**鎰忎箟**锛?
- 鐢ㄤ簬琛ㄧず鈥滃垽瀹氭湞鍚?鍧″悜婊¤冻鏌愮闃堝€尖€?渚嬪 `normal_vec[1]<-0.4`) 鐨勯偅閮ㄥ垎绾挎锛岃繖浜涚嚎娈靛彲瑙嗕负鈥滅孩娈碘€濇垨鈥滈珮鍗辨鈥濈瓑銆?
- 鎸夌収涓嶅悓 parent_idx(鍗崇鍑犳潯璺緞瀛愰泦)杩涜瀛樺偍銆?

---

## 8. `results["red_groups_corrected"]`
**鍐呭**锛?
```python
{
    parent_idx0: [correction0, correction1, ...],
    parent_idx1: [correction0, correction1, ...],
    ...
}
```
鑰屾瘡涓?`correction` 澶ц嚧鏍煎紡鏄細
```python
{
    "parent_index": parent_index,
    "node_order": corrected_nodes,   # 淇鍚庣殑 3D 鑺傜偣搴忓垪
    "nodes": [...],                  # 鍘婚噸鍚庣殑鑺傜偣
    "connectivity": {...},           # 绠€鍗曡褰曡妭鐐硅繛鎺ュ叧绯?澶磋妭鐐?1,涓棿鑺傜偣=2,...)
    "group_segments": group_segments # 鍘熷厛(2D鎶曞奖)閭ｆ壒绾㈢嚎娈?
}
```
**鎰忎箟**锛?
- 瀵?`red_groups` 涓殑绾㈢嚎娈靛仛浜嗕竴娆♀€滄姇褰?鍖归厤-绾犳鈥濓紝浠ョ‘淇濆畠浠兘鍜?`parent_path_subset` 鐨勭湡瀹炲嚑浣曞搴斻€?
- 鈥渘ode_order鈥?灏辨槸鏈€缁堝璇ョ孩绾挎缁勬墍瀵瑰簲鐨?3D 鑺傜偣搴忓垪锛屽彲鐩存帴鐢ㄦ潵鍦ㄤ笁缁翠腑缁樺埗绾㈢嚎浣嶇疆绛夈€?

---

## 9. `results["3d_points"]`
**鍐呭**锛?
```python
{
    parent_idx0: [arr0, arr1, ...],
    parent_idx1: [arr0, arr1, ...],
    ...
}
```
- 鍏朵腑 `arrX` 鏄煇涓孩绾挎缁勫湪涓夌淮鍧愭爣绯讳笅鐨勭偣浜?(N脳3)锛岀敱 `convert_groups_to_3d_points` 寰楀埌銆? 
- 娉ㄦ剰锛岃繖閲屼粎閽堝鍘熷 2D 鎶曞奖绾挎鍋氫簡绠€鍗?3D 鏄犲皠锛屼笉涓€瀹氫繚鐣欏師缃戞牸鐨勯潰绱㈠紩鎴栬矾寰勯『搴忋€?

**鎰忎箟**锛?
- 璁╀娇鐢ㄨ€呭彲浠ュ湪 3D 鍦烘櫙涓煡鐪嬬孩娈电粍鐨勪綅缃紝澶ц嚧瀵瑰簲鍦ㄥ钩闈㈡姇褰变腑鐨勫摢涓€閮ㄥ垎銆?

---

## 10. `results["FloatingSurface"]`
**鍐呭**锛?
```python
{
    parent_idx0: [groupX, groupY, ...],
    parent_idx1: [groupX, groupY, ...],
    ...
}
```
- 杩欐槸 `ArcUtils.filter_corrected_groups_by_x_distance(results, min_distance=0.1)` 涔嬪悗寰楀埌鐨勪竴涓彲閫夎緭鍑猴紝浠ｈ〃鍦ㄧ孩娈电粍涓啀杩涜涓€椤?鈥淴 鏂瑰悜鎶曞奖璺濈鈥?绛涢€夊悗鍓╀綑鐨勫瓙闆嗐€備緥濡傚父鐢ㄤ簬鍒ゅ畾閭ｄ簺姘村钩闀垮害瓒冲鐨勨€滄偓绌烘/浼稿嚭娈碘€濄€?
  
---

## 11. 鍏朵粬鏍囪
- `results["use_merged_paths"]`锛氬彧鏄竴涓竷灏斿€硷紝琛ㄧず鍦ㄦ湰娆℃祦绋嬩腑锛屾槸鍚︿娇鐢ㄤ簡鍚堝苟鍚庣殑璺緞鍘诲仛杩涗竴姝ュ鐞嗐€?

---

### 鎬荤粨

- `results` 鏄湪 `ArcSlicer.run_all()` 瀹屾垚鍚庡舰鎴愮殑鎬昏緭鍑猴紝鍖呭惈浜嗕粠**缃戞牸璇诲彇**銆?*骞抽潰鍙傛暟璁＄畻**銆?*鎴潰姹備氦**銆?*璺緞鍒嗗壊涓庡悎骞?*銆?*绾㈢嚎娈靛垎缁?*銆?*绾犳涓庣瓫閫?*绛変竴绯诲垪姝ラ鐨勭粨鏋溿€? 
- 鍦ㄥ疄闄呭紑鍙戜腑锛屼綘鍙互浠?`results` 涓换鎰忓彇鍑洪渶瑕佺殑閮ㄥ垎鏉ュ仛鍙鍖栨垨鍚庡鐞嗭紝濡?`results["paths"]["merged_open_subsets"]` 灏辨嬁鏉ョ敾涓诲墫闈紝`results["red_groups_corrected"]` 鎷挎潵鐢诲嵄闄╂锛岀瓑绛夈€?
"""
