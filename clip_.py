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
# 璁剧疆鑻辨枃瀛椾綋
font_en = FontProperties(family='Arial', size=12)
font_en_13 = FontProperties(family='Arial', size=13)
# 璁剧疆涓枃瀛椾綋锛堝畫浣擄級
font_zh = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
font_zh_13 = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=13)
plt.rcParams['font.size'] = 12               # 璁剧疆榛樿瀛椾綋澶у皬
plt.rcParams['axes.unicode_minus'] = False   # 姝ｅ父鏄剧ず璐熷彿

def load_arc_config(config_path):
    """
    鍔犺浇閰嶇疆鏂囦欢锛岃繑鍥炲渾蹇冦€佸崐寰勩€佽搴﹁寖鍥淬€佸姬闀胯寖鍥村強 Z 杞磋寖鍥淬€?
    閰嶇疆涓搴﹀潎涓哄姬搴︼紝寮ч暱鑼冨洿璁＄畻鍏紡锛歳adius * (angle_max - angle_min)
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
    鏍规嵁鍒囧墫闈綅缃紙寮ч暱鍧愭爣锛夎绠楀垏鍓栭潰鍙傛暟锛?
      slice_angle = angle_min + slice_position / radius
    杩斿洖瀛楀吀锛屽寘鎷钩闈㈠師鐐广€佹硶鍚戦噺銆佸緞鍚戞柟鍚戝強鍒囧墫闈㈣搴︺€?
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
    鏍规嵁 Z 杞磋寖鍥磋繃婊ょ綉鏍硷紝浠呬繚鐣欑鍚堟潯浠剁殑涓夎闈€?
    """
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if len(submeshes) > 0 else None

def slice_mesh_custom(mesh, origin, normal, near_tol=0.1):
    """
    鑷畾涔夊垏闈㈠嚱鏁帮細杩斿洖璺濈骞抽潰杩戜技鐨勭偣鍜岀綉鏍间笌骞抽潰鐨勪氦绾挎銆?
    """
    vertices = mesh.vertices
    dist = np.dot(vertices - origin, normal)
    near_points = vertices[np.abs(dist) <= near_tol]
    intersections = trimesh.intersections.mesh_plane(mesh, normal, origin)
    return near_points, intersections

def project_to_plane_custom(points, origin, radial_dir):
    """
    灏?3D 鐐规姇褰卞埌灞€閮ㄤ簩缁村潗鏍囩郴銆?
    瀹氫箟 u 杞翠负 radial_dir锛寁 杞翠负鍏ㄥ眬 Z 鏂瑰悜銆?
    鏀寔鍗曚釜鐐规垨澶氫釜鐐广€?
    """
    points = np.atleast_2d(points)
    vec = points - origin
    u = np.dot(vec, radial_dir)
    v = vec[:, 2]  # 鍙?z 鍒嗛噺
    return np.column_stack((u, v))

def extract_section_paths(section):
    """
    浠?trimesh 杩斿洖鐨?section 涓彁鍙栨墍鏈夌嚎/澶氭绾匡紝淇濈暀鎷撴墤缁撴瀯銆?
    """
    import networkx as nx
    graph = nx.Graph()

    for entity in section.entities:
        points = entity.points
        # 濡傛灉鏄嚎娈?2涓偣)
        if len(points) == 2:
            graph.add_edge(points[0], points[1])
        # 濡傛灉鏄娈电嚎(澶氫釜鐐?锛屾媶鍒嗕负杩炵画绾挎
        elif len(points) > 2:
            for i in range(len(points)-1):
                graph.add_edge(points[i], points[i+1])

    paths = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        endpoints = [node for node, degree in subgraph.degree() if degree == 1]

        if len(endpoints) == 0:
            # 闂悎璺緞
            cycle = nx.find_cycle(subgraph)
            path = [edge[0] for edge in cycle] + [cycle[0][0]]
        elif len(endpoints) == 2:
            # 寮€鍙ｈ矾寰?
            path = nx.shortest_path(subgraph, endpoints[0], endpoints[1])
        else:
            continue  # 澶嶆潅鎷撴墤璺宠繃

        coords = section.vertices[path]
        paths.append(coords)

    return paths

# ============= 浠ヤ笅涓烘坊鍔犵殑鍏抽敭鍔熻兘锛氶潰閭绘帴妫€鏌?=============
def build_face_adjacency_list(mesh):
    """
    鏋勫缓 face -> set_of_neighbor_faces 鐨勯偦鎺ヨ〃
    """
    adjacency = mesh.face_adjacency  # (M,2)锛屾瘡琛屾槸涓€瀵圭浉閭婚潰
    adj_dict = {}
    for f_id in range(len(mesh.faces)):
        adj_dict[f_id] = set()
    for (f1, f2) in adjacency:
        adj_dict[f1].add(f2)
        adj_dict[f2].add(f1)
    return adj_dict

def is_valid_segment(face_a, face_b, adjacency_dict):
    """
    face_a == face_b 鎴?face_b 鍦?adjacency_dict[face_a] 涓?
    鍒欒涓虹嚎娈垫槸鍚堟硶鐨勶紙钃濊壊锛?
    """
    if face_a == face_b:
        return True
    if face_b in adjacency_dict[face_a]:
        return True
    return False

def slice_and_check_faces(mesh, origin, normal):
    """
    浣跨敤 mesh_plane 骞惰繑鍥?(lines_3d, face_ids_2d)锛岀敤浜庡悗缁鏌?
    """
    # return_faces=True 鍙互璁╂垜浠嬁鍒版瘡涓鐐圭殑闈?ID
    lines_3d, face_ids_2d = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=normal,
        plane_origin=origin,
        return_faces=True
    )
    return lines_3d, face_ids_2d


def plot_trimesh_on_ax(ax, mesh, face_color=(0.5, 0.5, 0.5, 0.3)):
    """
    浣跨敤 matplotlib 缁樺埗 trimesh 妯″瀷鍒板凡鏈夌殑 ax
    """
    faces = mesh.faces
    vertices = mesh.vertices

    # 鏋勯€犳瘡涓笁瑙掑舰鐨?3D 椤剁偣闆嗗悎
    triangles = vertices[faces]  # shape: (N, 3, 3)

    # 鍒涘缓 Poly3DCollection
    mesh_collection = Poly3DCollection(triangles, facecolors=face_color, edgecolors='k', linewidths=0.1)
    mesh_collection.set_alpha(face_color[3])  # 璁剧疆閫忔槑搴?
    ax.add_collection3d(mesh_collection)

    # 鑷姩缂╂斁
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
# =========================================================

def test_single_slice_comparison():
    config_path = r"configs/arc_config.example.json"
    mesh_path = r"data/private/raw_model\sample_site\Merged mesh_seg.obj"
    # 1. 鍔犺浇閰嶇疆鍜岀綉鏍?
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)
    mesh = trimesh.load_mesh(mesh_path)
    print("Vertices shape:", mesh.vertices.shape)
    print("Faces shape:", mesh.faces.shape)

    # 2. Z 杩囨护
    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
    
    # 3. 鍒囬潰鍙傛暟
    slice_position = 43.5
    plane_params = compute_slice_plane(center, radius, angle_min, arc_length_range, slice_position)
    origin = plane_params["origin"]
    normal = plane_params["normal"]
    radial_dir = plane_params["radial_dir"]
    
    # 4. 鑷畾涔?slice_mesh锛堜粎鍋氱ず渚嬪姣旓級
    near_points, intersections_custom = slice_mesh_custom(mesh_filtered, origin, normal, near_tol=0.1)
    custom_2d_segments = []
    for seg in intersections_custom:
        pt1, pt2 = seg[0], seg[1]
        pt1_2d = project_to_plane_custom(pt1, origin, radial_dir)
        pt2_2d = project_to_plane_custom(pt2, origin, radial_dir)
        custom_2d_segments.append((pt1_2d, pt2_2d))

    # 5. 浣跨敤 trimesh.section 鎻愬彇璺緞 (涓嶅甫闈俊鎭紝浣嗗彲鍋氭嫇鎵戞彁鍙?
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is not None and len(section.entities) > 0:
        paths = extract_section_paths(section)
    else:
        paths = []
        print("鏃犲墫闈㈢嚎锛岃妫€鏌ュ垏闈綅缃€?)

    # 6. 闈㈤偦鎺ユ鏌ワ細瀵?intersection 鐩存帴鑾峰彇绔偣闈?ID 骞跺垽鏂?
    #    锛堣繖閲屼娇鐢?mesh_filtered 涓?plane锛屽洜涓烘垜浠彧鍏冲績宸茶繃婊ら儴鍒嗭級
    lines_3d, face_ids = slice_and_check_faces(mesh_filtered, origin, normal)
    face_adjacency_dict = build_face_adjacency_list(mesh_filtered)

    print(face_ids.shape)


    # 3. 浣跨敤 Trimesh 鍒涘缓涓€涓瓙缃戞牸锛屽彧鍖呭惈杩欎簺闈?
    highlight_mesh = mesh_filtered.submesh([face_ids], append=True)

    def trimesh_to_pv(mesh):
        faces = np.hstack([[3] + list(face) for face in mesh.faces]).astype(np.int32)
        return pv.PolyData(mesh.vertices, faces)
    
    def lines_to_pv_polyline(lines_3d):
        """
        灏?lines_3d (N,2,3) 杞崲涓?pyvista.PolyData锛屽叾涓寘鍚鏉＄嚎娈?
        """
        points = []
        lines = []
        for line in lines_3d:
            pt1, pt2 = line
            idx1 = len(points)
            idx2 = idx1 + 1
            points.extend([pt1, pt2])
            lines.append([2, idx1, idx2])  # 姣忔潯绾挎涓や釜鐐癸紝2 鏄暟閲忔爣璁?
        points = np.array(points)
        lines = np.array(lines, dtype=np.int64)

        # 1. 鍒涘缓绌虹殑 PolyData 瀵硅薄
        pdata = pv.PolyData()
        
        # 2. 璁剧疆鐐?
        pdata.points = points
        
        # 3. 璁剧疆绾垮崟鍏冿紙蹇呴』浣跨敤 set_lines锛?
        pdata.lines = lines

        return pdata
    def create_cut_plane(origin, normal, size=300.0):
        """
        鏋勯€犱竴涓钩闈㈢敤浜庡彲瑙嗗寲锛宱rigin 鏄钩闈笂鐨勭偣锛宯ormal 鏄硶鍚戦噺
        """
        return pv.Plane(center=origin, direction=normal, i_size=size, j_size=size)
    # 鍒涘缓 pyvista 缃戞牸
    pv_mesh = trimesh_to_pv(mesh)
    pv_highlight = trimesh_to_pv(highlight_mesh)
    pv_lines = lines_to_pv_polyline(lines_3d)

    # 鍒涘缓 plotter
    plotter = pv.Plotter()
    plotter.enable_parallel_projection()

    plotter.add_mesh(pv_mesh, color='lightgray', opacity=1, show_edges=False)
    plotter.add_mesh(pv_highlight, color='red', opacity=1.0, show_edges=False)
    plotter.add_mesh(pv_lines, color='blue', line_width=3, label='鍒囬潰绾挎')
    plane_mesh = create_cut_plane(origin, normal)
    # plotter.add_mesh(plane_mesh, color='green', style='wireframe', opacity=0.3, label='鍒囧壊骞抽潰')
    plotter.add_title("琚垏鍓蹭笁瑙掗潰灞曠ず", font_size=14)
    # plotter.show_grid()

    # 鏄剧ず
    plotter.show()

    # 濡傛灉鎯崇湅 2D 瀵规瘮鍥撅紙custom_2d_segments vs. mesh.section锛夊彲瑙ｅ紑涓嬮潰娉ㄩ噴
    # 鍋囪浣犱箣鍓嶅仛浜嗙被浼?section_2d 鐨勬姇褰憋紝鍙互鐢ㄨ繖涓嚱鏁板姣?
    # display_slice_comparison(custom_2d_segments, section_2d, slice_position, plane_params)

if __name__ == "__main__":
    test_single_slice_comparison()


