import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D

def load_arc_config(config_path):
    """
    鍔犺浇arc_config.json锛岃繑鍥?
      - center (3D鍦嗗績, z榛樿涓?)
      - radius (娴偣鏁?
      - angle_range (瑙掑害鍖洪棿, 鍗曚綅寮у害鎴栧害鐪嬩綘閰嶇疆)
      - z_min, z_max (Z杞磋寖鍥?
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
    鏍规嵁澶栭儴杈撳叆鐨勫垏鍓栭潰浣嶇疆锛堝姬闀垮潗鏍囷級璁＄畻鍓栧垏骞抽潰銆?
    璁＄畻鏂规硶锛?
      - 灏嗗垏鍓栭潰浣嶇疆杞崲涓鸿搴︼細 slice_angle = angle_min + slice_position/radius
      - 鏍规嵁璇ヨ搴﹁绠楀緞鍚戞柟鍚戝強骞抽潰娉曞悜閲?
    杩斿洖锛氬钩闈㈠師鐐广€佹硶鍚戦噺銆佸緞鍚戞柟鍚戯紝浠ュ強璁＄畻寰楀埌鐨勫垏鍓栭潰瑙掑害
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
    鏍规嵁 Z 杞磋寖鍥磋繃婊ょ綉鏍硷紝鍙繚鐣欏湪 [z_min, z_max] 鍖洪棿鍐呯殑涓夎闈?
    """
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min_z = face_vertices_z.min(axis=1)
    face_max_z = face_vertices_z.max(axis=1)
    face_mask = (face_max_z >= z_min) & (face_min_z <= z_max)
    submeshes = mesh.submesh([face_mask], only_watertight=False)
    return submeshes[0] if len(submeshes) > 0 else None

def slice_mesh(mesh, origin, normal, near_tol):
    """
    鎻愬彇:
      1) near_points: 璺濈骞抽潰鍦?卤near_tol 鍐呯殑椤剁偣
      2) intersections: 缃戞牸涓庡钩闈㈢殑绮剧‘浜ょ嚎(绾挎闆嗗悎)
    """
    vertices = mesh.vertices
    dist = np.dot(vertices - origin, normal)
    mask = np.abs(dist) <= near_tol
    near_points = vertices[mask]
    start_time = time.time()
    intersections = trimesh.intersections.mesh_plane(mesh, normal, origin)
    end_time = time.time()
    print(f"mesh_plane 璁＄畻鐢ㄦ椂: {end_time - start_time:.6f} 绉?)
    return near_points, intersections

def project_to_plane(points_3d, origin, axis_x, axis_y):
    """
    灏?points_3d 鎶曞奖鍒板钩闈笂:
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
    灏嗕氦绾挎鎵€鏈夌鐐规牴鎹?tol 鍚堝苟涓哄敮涓€鑺傜偣锛岃繑鍥炲悎骞跺悗鐨勮妭鐐规暟缁?
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
    缁熻姣忎釜鑺傜偣杩炴帴鐨勭嚎娈典釜鏁帮紝杩斿洖瀛楀吀锛岄敭涓鸿妭鐐癸紙tuple鏍煎紡锛夛紝鍊间负杩炴帴鏁?
    """
    node_conn = {}
    for seg in intersections:
        for p in seg:
            key = tuple(np.round(p, decimals=6))
            node_conn[key] = node_conn.get(key, 0) + 1
    return node_conn

def split_segments_to_paths(intersections, tol=1e-6):
    """
    鏍规嵁 tol 鍚堝苟鍚庣殑鑺傜偣鏋勯€犲浘锛屽埄鐢?DFS 鍒嗙杩為€氬垎閲忥紝
    姣忎釜杩為€氬垎閲忓搴斾竴鏉¤矾寰勫瓙闆嗭紝杩斿洖涓€涓矾寰勫瓙闆嗗垪琛紝姣忎釜瀛愰泦涓虹嚎娈靛垪琛ㄣ€?
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
    鍒ゆ柇璺緞瀛愰泦鏄惁涓哄皝闂泦鍚堬細鑻ユ墍鏈夎妭鐐瑰潎涓鸿繛鎺?鐨勫唴鐐瑰垯璁や负鏄皝闂殑锛屽惁鍒欎负闈為棴鍚堥泦鍚?
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
    瀵逛换鎰忓瓙闆嗘瀯閫犲浘锛屽苟鎸夌収杩炴帴鍏崇郴杩斿洖鏈夊簭鐨勮妭鐐瑰垪琛紝
    纭畾鍏跺湪骞抽潰鎶曞奖鍚庣旱鍧愭爣(y鍊?鏈€澶х殑鐐瑰拰鏈€灏忕殑鐐癸紝
    璺緞涓轰粠y鍊兼渶澶х殑鐐瑰紑濮嬪埌y鍊兼渶灏忕殑鐐圭粨鏉熴€?
    杩斿洖鐨勮妭鐐瑰簭鍒椾负3D鍧愭爣鍒楄〃銆?
    """
    # 鏋勯€犳棤鍚戝浘
    graph = {}
    for edge in path_subset:
        for node in edge:
            graph.setdefault(node, set())
        node1, node2 = edge
        graph[node1].add(node2)
        graph[node2].add(node1)

    # 寰楀埌鎵€鏈夎妭鐐瑰苟鎶曞奖鍒板钩闈?
    all_nodes = list(graph.keys())
    nodes_arr = [np.array(nd) for nd in all_nodes]
    nodes_2d = project_to_plane(nodes_arr, plane_origin, radial_dir, vertical_dir)
    
    # 鎵惧埌鎶曞奖鍚巠鍊兼渶澶х殑鍜寉鍊兼渶灏忕殑鐐?
    y_values = [pt[1] for pt in nodes_2d]
    max_index = y_values.index(max(y_values))
    min_index = y_values.index(min(y_values))
    start_node = all_nodes[max_index]
    end_node = all_nodes[min_index]

    # 浣跨敤BFS瀵绘壘浠巗tart_node鍒癳nd_node鐨勮矾寰?
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
        print("鏃犳硶鎵惧埌浠巠鍊兼渶澶у埌y鍊兼渶灏忕殑璺緞")
        return []
    
    return [np.array(nd) for nd in path_found]

def compute_normals_with_edges_for_subset(path_subset, closed, plane_origin, radial_dir, vertical_dir):
    """
    涓?compute_normals_for_subset 绫讳技锛屼絾鍚屾椂杩斿洖瀵瑰簲绾挎绔偣锛堟姇褰卞悗鐨?D鍧愭爣锛夈€?
    杩斿洖鍒楄〃锛屾瘡涓厓绱犱负 (璧风偣鎶曞奖, 缁堢偣鎶曞奖, 绾挎涓偣, 娉曞悜閲?D)銆?
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
    瀵归潪闂悎璺緞锛坧ath_subset锛夛紝鍏堝埄鐢?order_nonclosed_path 寰楀埌鑺傜偣椤哄簭锛?
    鐒跺悗鏋勯€犵浉閭昏妭鐐归棿鐨勭嚎娈碉紙璁＄畻涓偣鍜屾硶鍚戦噺锛夛紝绛涢€夊嚭娉曞悜閲?y 鍒嗛噺灏忎簬 threshold 鐨勭孩鑹茬嚎娈碉紝
    杩斿洖鎸夊師杩炴帴椤哄簭鎺掑垪鐨勭孩鑹茬嚎娈靛垪琛ㄣ€?
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
    瀵逛紶鍏ョ殑绾㈣壊绾挎鍒楄〃锛堝凡鎸夊瓙璺緞杩炴帴椤哄簭鎺掑垪锛夊厛閫嗗簭锛?
    鐒跺悗閬嶅巻閫嗗簭鍚庣殑绾挎锛氬浜庢瘡涓浉閭荤嚎娈碉紝璁＄畻鍓嶄竴绾挎鐨勭粓姝㈢偣 x 鍊硷紙杈冨ぇ鍊硷級
    涓庡悗涓€绾挎鐨勮捣濮嬬偣 x 鍊硷紙杈冨皬鍊硷級涔嬪樊 gap锛岃嫢 gap < tol 鍒欏綊涓哄悓涓€缁勶紝鍚﹀垯寮€鍚柊缁勩€?
    杩斿洖鍒嗙粍鍚庣殑鍒楄〃锛屾瘡缁勪负涓€涓孩鑹茬嚎娈靛瓙闆嗐€?
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
    鏍规嵁娴嬪墫闈㈡墍鍦ㄥ钩闈㈠潗鏍囩郴锛屽皢浜岀淮鐐?(u,v) 杞崲涓轰笁缁寸偣:
      p_3d = plane_origin + u * radial_dir + v * vertical_dir
    """
    points_3d = []
    for p in points_2d:
        p3d = plane_origin + p[0] * radial_dir + p[1] * vertical_dir
        points_3d.append(p3d)
    return np.array(points_3d)

def convert_groups_to_3d_points(groups_by_path, plane_origin, radial_dir, vertical_dir):
    """
    瀵规瘡涓矾寰勭殑姣忕粍绾㈣壊绾挎锛屽皢缁勫唴鎵€鏈夌嚎娈电殑涓や釜绔偣鍙栧苟闆嗭紙鍘婚噸锛夊悗锛?
    鍒╃敤 convert_2d_points_to_3d 杞崲涓轰笁缁寸偣锛岃繑鍥炲瓧鍏革紝閿负璺緞缂栧彿锛?
    鍊间负鍒楄〃锛屾瘡涓€缁勪负瀵瑰簲鐨勪笁缁寸偣鏁扮粍銆?
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

# ---------------- 鏁寸悊缁樺浘鍑芥暟 ----------------
def plot_original_subpaths(path_subsets, plane_origin, radial_dir, vertical_dir):
    """
    缁樺埗鍘熷瀛愯矾寰勫睍绀猴紙姣忔潯瀛愯矾寰勪娇鐢ㄤ笉鍚岄鑹诧級锛?
    骞跺闈為棴鍚堝瓙璺緞鎸夌収鑺傜偣椤哄簭鏍囪绗竴涓妭鐐癸紙绾㈣壊锛夊拰鏈€鍚庝竴涓妭鐐癸紙缁胯壊锛?
    """

    plt.figure(figsize=(6,6))
    path_cmap = plt.cm.get_cmap("Set1", len(path_subsets))
    for i, subset in enumerate(path_subsets):
        color = path_cmap(i)

        # 缁樺埗姣忔潯瀛愯矾寰勭殑鎵€鏈夌嚎娈?
        for edge in subset:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            p1_2d = project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], c=color, linewidth=2)
        # 瀵归潪闂悎瀛愯矾寰勶紝鍒╃敤 order_nonclosed_path 寰楀埌鑺傜偣椤哄簭
        ordered_nodes = order_nonclosed_path(subset, plane_origin, radial_dir, vertical_dir)
        if len(ordered_nodes) >= 2:
            first_node = ordered_nodes[0]
            last_node = ordered_nodes[-1]
            first_2d = project_to_plane([first_node], plane_origin, radial_dir, vertical_dir)[0]
            last_2d = project_to_plane([last_node], plane_origin, radial_dir, vertical_dir)[0]
            plt.scatter(first_2d[0], first_2d[1], c="red", s=50, marker="o", label="璧峰鐐? if i==0 else "")
            plt.scatter(last_2d[0], last_2d[1], c="green", s=50, marker="o", label="缁堟鐐? if i==0 else "")
    plt.title("鍘熷瀛愯矾寰勫睍绀?)
    plt.xlabel("寰勫悜鏂瑰悜 /m")
    plt.ylabel("鍨傜洿鏂瑰悜 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.legend()
    plt.show()

def plot_merged_paths_and_normals(merged_paths, merged_normals, path_subsets, plane_origin, radial_dir, vertical_dir):
    """
    缁樺埗涓ゅ箙瀛愬浘锛?
      宸︿晶涓哄師濮嬪瓙璺緞鑳屾櫙锛堢伆鑹叉樉绀猴級锛?
      鍙充晶涓哄悎骞跺悗璺緞鍙婂叾娉曞悜閲忥紙绾㈣壊绠ご琛ㄧず锛夈€?
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    # 宸︿晶锛氱粯鍒跺師濮嬪瓙璺緞鑳屾櫙
    axes[0].set_title("鍘熷瀛愯矾寰勮儗鏅?)
    axes[0].set_xlabel("寰勫悜鏂瑰悜 /m")
    axes[0].set_ylabel("鍨傜洿鏂瑰悜 /m")
    for subset in path_subsets:
        for edge in subset:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            p1_2d = project_to_plane([p1], plane_origin, radial_dir, vertical_dir)[0]
            p2_2d = project_to_plane([p2], plane_origin, radial_dir, vertical_dir)[0]
            axes[0].plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color='gray', linewidth=1)
    axes[0].set_aspect('equal', adjustable='datalim')
    
    # 鍙充晶锛氱粯鍒跺悎骞跺悗璺緞鍙婃硶鍚戦噺
    axes[1].set_title("鍚堝苟鍚庤矾寰勫強娉曞悜閲?)
    axes[1].set_xlabel("寰勫悜鏂瑰悜 /m")
    axes[1].set_ylabel("鍨傜洿鏂瑰悜 /m")
    for idx, m_path in enumerate(merged_paths):
        m_path = np.array(m_path)
        m_path_2d = project_to_plane(m_path, plane_origin, radial_dir, vertical_dir)
        axes[1].plot(m_path_2d[:,0], m_path_2d[:,1], linewidth=2, label=f"鍚堝苟璺緞 {idx+1}")
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

# ---------------- 鍚堝苟璺緞鍑芥暟 ----------------
def merge_open_paths_new(path_subsets, plane_origin, radial_dir, vertical_dir):
    """
    鍚堝苟闈為棴鍚堝瓙璺緞锛?
    1. 瀵规瘡涓潪闂悎瀛愯矾寰勫埄鐢?order_nonclosed_path 寰楀埌鑺傜偣搴忓垪锛屽苟璁＄畻鍏跺湪浜岀淮骞抽潰涓婄殑绾靛潗鏍囪寖鍥达紝
       鎸夎捣濮嬬偣绾靛潗鏍囦粠楂樺埌浣庢帓搴忥紝鏋勯€?open_paths 鍒楄〃銆?
    2. 浠?open_paths 涓緷娆￠€夊彇璧峰鐐规渶楂樼殑瀛愯矾寰勪綔涓哄悎骞惰捣鐐癸紝鍒╃敤寤朵几閫昏緫锛堝熀浜庨浂鐐瑰畾鐞嗗拰绾挎€ф彃鍊硷級
       閫愭寤朵几褰撳墠鍚堝苟璺緞锛岀洿鍒版棤娉曞啀寤朵几銆?
    3. 姣忔鍚堝苟缁撴潫鍚庯紝鏍规嵁褰撳墠鍚堝苟璺緞鍦ㄤ簩缁村钩闈笂鐨勭旱鍧愭爣鑼冨洿锛?
       鍒犻櫎 open_paths 涓畬鍏ㄨ鍖呭惈鐨勫瓙璺緞锛屽啀閲嶆柊鎺掑簭鍚庣户缁鐞嗐€?
    4. 杩斿洖涓€涓垪琛紝姣忎釜鍏冪礌涓轰竴鏉″悎骞跺悗鐨?D鑺傜偣搴忓垪锛屼笌鍘熸潵瀛樺偍寮€璺緞闆嗙殑鏁版嵁鏍煎紡涓€鑷淬€?
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
        print("娌℃湁瓒冲鐨勯潪闂悎瀛愯矾寰?)
        return []
    
    merged_paths = []
    while open_paths:
        current = open_paths.pop(0)
        merged_path = current["nodes"][:]  # 澶嶅埗鑺傜偣搴忓垪
        
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
    瀵瑰悎骞跺悗鐨勮矾寰勶紙鏈夊簭3D鑺傜偣搴忓垪锛夎绠椾簩缁存硶鍚戦噺锛?
    杩斿洖鍒楄〃锛屾瘡涓厓绱犱负 (璧风偣鎶曞奖, 缁堢偣鎶曞奖, 绾挎涓偣, 娉曞悜閲?D)銆?
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
    鍔犺浇閰嶇疆鏂囦欢锛岃繑鍥炲渾蹇冦€佸崐寰勩€佽搴﹁寖鍥淬€佸姬闀胯寖鍥村強Z杞磋寖鍥淬€?
    閰嶇疆鏂囦欢涓搴︿互寮у害鍒剁粰鍑猴紝杞崲鏂规硶濡備笅锛?
      - 鍘熷瑙掑害鑼冨洿锛歔angle_min, angle_max]
      - 寮ч暱鑼冨洿锛氫互鍦嗗姬宸︾鐐逛负鍘熺偣锛屽彸绔偣鐨勫姬闀夸负 radius*(angle_max - angle_min)
    """
    with open(config_path, 'r') as f:
        arc_config = json.load(f)
    center_2d = np.array(arc_config["center"], dtype=float)
    center_3d = np.append(center_2d, 0.0)
    radius = float(arc_config["radius"])
    angle_min = arc_config["angle_min"]
    angle_max = arc_config["angle_max"]
    # 璁＄畻鏁翠釜鍦嗗姬鐨勫姬闀?
    arc_length = radius * (angle_max - angle_min)
    arc_length_range = [0, arc_length]
    z_min, z_max = arc_config["z_range"]
    return center_3d, radius, angle_min, angle_max, arc_length_range, z_min, z_max

def main():
    # 璁剧疆涓枃瀛椾綋
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config_full(r"configs/arc_config.example.json")

    mesh_start_time = time.time()
    mesh = trimesh.load_mesh(r"data/private/raw_model\combined_model.glb")
    mesh_end_time = time.time()
    print(f"缃戞牸鍔犺浇鏃堕棿: {mesh_end_time - mesh_start_time:.6f} 绉?)
    
    # 2. 瀵圭綉鏍艰繘琛孼杞磋鍓?
    mesh_in_range = filter_mesh_by_z_range(mesh, z_min, z_max)
    if mesh_in_range is None:
        print("璀﹀憡锛氬湪缁欏畾Z鑼冨洿鍐呮棤涓夎闈紒")
        return

    # 3. 璁＄畻鍓栧垏骞抽潰
    plane_origin, plane_normal, radial_dir, slice_angle = compute_plane(center, radius, angle_min, arc_length_range, 44.2)
    near_tol = 0.1

    # 4. 鍒囧壊缃戞牸
    near_points, intersections = slice_mesh(mesh_in_range, plane_origin, plane_normal, near_tol)
    print(f"杩戜技鍓栭潰鐐规暟: {len(near_points)}")
    if intersections is not None:
        print(f"浜ょ嚎绾挎鏁? {len(intersections)}")
    else:
        print("鏃犱氦绾?)
    
    # 姝ラ1锛氬悎骞朵氦绾挎绔偣骞剁粺璁¤繛鎺ユ暟
    unique_nodes = merge_intersection_nodes(intersections, tol=1e-6)
    print(f"鍚堝苟鍚庣殑鍞竴鑺傜偣鏁? {len(unique_nodes)}")
    node_conn = compute_node_connectivity(intersections, tol=1e-6)
    conn_count = {}
    for count in node_conn.values():
        conn_count[count] = conn_count.get(count, 0) + 1
    print("鑺傜偣杩炴帴绾挎缁熻缁撴灉:")
    for k in sorted(conn_count.keys()):
        print(f"杩炴帴{k}涓嚎娈电殑鑺傜偣鏁伴噺: {conn_count[k]}")
    
    # 姝ラ2锛氬皢浜ょ嚎娈靛垎涓哄涓矾寰勫瓙闆?
    path_subsets = split_segments_to_paths(intersections, tol=1e-6)
    print(f"鍒嗙鍑?{len(path_subsets)} 鏉¤矾寰勫瓙闆嗐€?)
    for i, subset in enumerate(path_subsets):
        print(f"璺緞瀛愰泦 {i+1} 鍖呭惈 {len(subset)} 涓嚎娈点€?)
    
    # 瀹氫箟骞抽潰鍧愭爣绯伙細x杞翠负寰勫悜鏂瑰悜锛寉杞翠负鍨傜洿鏂瑰悜
    vertical_dir = np.array([0, 0, 1])
    
    # 鍦ㄥ悎骞跺瓙闆嗕箣鍓嶏紝鍏堣皟鐢ㄧ粯鍥惧嚱鏁板睍绀哄師濮嬪瓙璺緞锛屽悓鏃舵爣璁版瘡涓潪闂悎瀛愯矾寰勭殑璧峰鐐癸紙绾㈣壊锛夊拰缁堟鐐癸紙缁胯壊锛?
    plot_original_subpaths(path_subsets, plane_origin, radial_dir, vertical_dir)
    
    # 浣跨敤鍚堝苟绠楁硶寰楀埌鍚堝苟鍚庣殑璺緞锛堜粎澶勭悊闈為棴鍚堝瓙璺緞锛夛紝鏍煎紡涓庡師鏉ヤ竴鑷?
    merged_paths = merge_open_paths_new(path_subsets, plane_origin, radial_dir, vertical_dir)
    
    # 鍩轰簬鍚堝苟鍚庣殑璺緞璁＄畻娉曞悜閲?
    merged_normals = {}
    for idx, m_path in enumerate(merged_paths):
        normals = compute_normals_for_merged_path(m_path, plane_origin, radial_dir, vertical_dir)
        merged_normals[idx] = normals
    
    # 璋冪敤鍑芥暟缁樺埗鍚堝苟鍚庤矾寰勫強娉曞悜閲?
    plot_merged_paths_and_normals(merged_paths, merged_normals, path_subsets, plane_origin, radial_dir, vertical_dir)
    
    # 鍚庣画鍙互鍩轰簬鍚堝苟鍚庣殑璺緞鍜屾硶鍚戦噺杩涜绾㈣壊绾挎绛涢€夈€佸垎缁勩€佽浆鎹负3D鐐瑰拰3D灞曠ず绛夋搷浣?

if __name__ == "__main__":
    main()

