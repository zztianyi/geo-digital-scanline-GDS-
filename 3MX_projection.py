import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib
import time
from matplotlib.path import Path

def load_config(config_path):
    """鍔犺浇閰嶇疆鏂囦欢锛岃繑鍥炲渾蹇冦€佸崐寰勩€佽搴﹀尯闂翠互鍙奪杞磋寖鍥?""
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    center = np.append(np.array(cfg["center"], dtype=float), 0.0)
    radius = float(cfg["radius"])
    angle_range = [cfg["angle_min"], cfg["angle_max"]]
    z_min, z_max = cfg["z_range"]
    return center, radius, angle_range, z_min, z_max

def compute_plane(center, angle_range):
    """鏍规嵁瑙掑害鍖洪棿璁＄畻鍓栧垏骞抽潰鐨勫師鐐瑰拰娉曞悜閲?""
    mid_angle = 0.5 * sum(angle_range)
    radial_dir = np.array([np.cos(mid_angle), np.sin(mid_angle), 0.0])
    normal = np.cross(radial_dir, [0, 0, 1])
    normal /= np.linalg.norm(normal)
    return center, normal

def filter_mesh(mesh, z_min, z_max):
    """杩囨护鍑篫杞磋寖鍥村唴锛堟湁閮ㄥ垎椤剁偣钀藉湪鍖洪棿鍐咃級鐨勪笁瑙掗潰"""
    face_z = mesh.vertices[mesh.faces][:, :, 2]
    mask = (face_z.min(axis=1) <= z_max) & (face_z.max(axis=1) >= z_min)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if submeshes else None

def slice_mesh(mesh, origin, normal, tol=0.1):
    """璁＄畻椤剁偣璺濈鍜屼氦绾匡紝杩斿洖婊¤冻tol鏉′欢鐨勯《鐐瑰拰浜ょ嚎闆嗗悎"""
    dist = np.dot(mesh.vertices - origin, normal)
    near_points = mesh.vertices[np.abs(dist) <= tol]
    start = time.time()
    intersections = trimesh.intersections.mesh_plane(mesh, normal, origin)
    print(f"mesh_plane 璁＄畻鐢ㄦ椂: {time.time() - start:.6f} 绉?)
    return near_points, intersections

def project(points, origin, axis_x, axis_y):
    """灏?D鐐规姇褰卞埌鐢眔rigin銆乤xis_x銆乤xis_y瀹氫箟鐨勫钩闈㈠潗鏍囩郴涓?""
    return np.array([[np.dot(p - origin, axis_x), np.dot(p - origin, axis_y)] for p in points])

def order_segments(segments, tol=1e-6):
    """
    鏋勫缓鍥剧粨鏋勶紝浣跨敤DFS鎵惧埌浠巣鏈€澶у埌z鏈€灏忕偣鐨勬湁搴忚矾寰?
    杩斿洖鏈夊簭璺緞锛坥rdered_path锛夛紝鍗充竴绯诲垪杩炵画鐨?D鐐?
    """
    graph, pts = {}, {}
    def add_pt(p):
        key = tuple(np.round(p, 6))
        pts.setdefault(key, p)
        return key
    for seg in segments:
        k1, k2 = add_pt(seg[0]), add_pt(seg[1])
        graph.setdefault(k1, []).append(k2)
        graph.setdefault(k2, []).append(k1)
    start = max(pts.keys(), key=lambda k: pts[k][2])
    end = min(pts.keys(), key=lambda k: pts[k][2])
    path, found = [], False
    def dfs(cur, target, cur_path):
        nonlocal path, found
        if found:
            return
        cur_path.append(cur)
        if cur == target:
            path, found = cur_path.copy(), True
            return
        for nb in graph.get(cur, []):
            if nb not in cur_path:
                dfs(nb, target, cur_path)
        cur_path.pop()
    dfs(start, end, [])
    return [pts[k] for k in path] if path else None

def compute_special_normal(seg_2d):
    """
    鏍规嵁缁欏畾浜岀淮绾挎 seg_2d锛堝寘鍚袱涓鐐圭殑鍧愭爣锛夎绠楁硶鍚戦噺锛?
    閫昏緫锛氬厛璁＄畻绾挎鏂瑰悜锛屽啀閫嗘椂閽堟棆杞?0掳寰楀埌娉曞悜閲?
    杩斿洖鐨?normal_2d 涓哄崟浣嶅悜閲?
    """
    p1, p2 = seg_2d
    seg_vec = p2 - p1
    norm = np.linalg.norm(seg_vec)
    if norm < 1e-6:
        return None
    seg_dir = seg_vec / norm
    normal_2d = np.array([-seg_dir[1], seg_dir[0]])
    return normal_2d

def draw_result(intersections, ordered_points, plane_origin, radial_dir, vertical_dir,
                z_min, z_max, show_arrows=True):
    """
    鍘熸湁鐨勭粯鍥惧嚱鏁帮細
      - 缁樺埗鎵€鏈変氦绾挎锛堝僵鑹叉樉绀猴級
      - 缁樺埗鏈夊簭璺緞鍙婂皝闂杈瑰舰锛屽苟濉厖鈥?鈥?
      - 鏍规嵁show_arrows鍙傛暟鍐冲畾鏄惁鏄剧ず璺緞绠ご
    """
    plt.figure(figsize=(6,6))
    cmap = plt.cm.get_cmap("tab10")
    for i, seg in enumerate(intersections):
        p1, p2 = seg[0], seg[1]
        p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
        plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], c=cmap(i % 10), linewidth=1)
    
    if ordered_points is not None:
        pts2d = project(ordered_points, plane_origin, radial_dir, vertical_dir)
        plt.plot(pts2d[:,0], pts2d[:,1], color='k', linewidth=2, label="鏈夊簭璺緞")
        fill_offset = 3
        start_pt = pts2d[0]
        end_pt = pts2d[-1]
        start_left = start_pt - np.array([fill_offset, 0])
        end_left   = end_pt - np.array([(end_pt[0]-start_left[0]), 0])
        poly_points = np.vstack((pts2d, [end_left], [start_left]))
        plt.plot([end_pt[0], start_left[0]], [end_pt[1], start_left[1]], 'r--', linewidth=1)
    
    plt.title(f"Z鑼冨洿[{z_min:.2f}, {z_max:.2f}]渚у墫闈㈡姇褰?)
    plt.xlabel("寰勫悜鏂瑰悜 /m")
    plt.ylabel("鍨傜洿鏂瑰悜 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.legend()
    if show_arrows and ordered_points is not None:
        for i in range(len(pts2d) - 1):
            pt_start = pts2d[i]
            pt_end = pts2d[i+1]
            seg_vec = pt_end - pt_start
            seg_length = np.linalg.norm(seg_vec)
            if seg_length < 1e-6:
                continue
            seg_dir = seg_vec / seg_length
            mid = 0.5 * (pt_start + pt_end)
            arrow_len = seg_length * 0.3
            plt.arrow(mid[0], mid[1],
                      seg_dir[0] * arrow_len, seg_dir[1] * arrow_len,
                      color='k', head_width=0.01, head_length=0.05, linewidth=1)
    plt.show()

def draw_result_with_normal(intersections, ordered_points, plane_origin, radial_dir, vertical_dir,
                              z_min, z_max, show_arrows=True):
    """
    鏂板鐨勭粯鍥惧嚱鏁帮紙鍩轰簬浜ょ嚎娈碉級锛?
      - 瀵规瘡鏉′氦绾挎璋冪敤 compute_special_normal 寰楀埌娉曞悜閲?
      - 鍒ゆ柇璇ユ硶鍚戦噺涓庤礋y杞寸殑澶硅锛岃嫢灏忎簬30掳鍒欏皢瀵瑰簲绾挎鏍囩孩鏄剧ず
      - 鍏朵粬閮ㄥ垎涓庡師鍑芥暟绫讳技
    """
    plt.figure(figsize=(6,6))
    cmap = plt.cm.get_cmap("tab10")
    neg_y = np.array([0, -1])
    
    for i, seg in enumerate(intersections):
        p1, p2 = seg[0], seg[1]
        p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
        seg_2d = np.array([p1_2d, p2_2d])
        special_normal = compute_special_normal(seg_2d)
        if special_normal is None:
            continue
        angle = np.arccos(np.clip(np.dot(special_normal, neg_y), -1.0, 1.0))
        seg_color = 'red' if angle < np.deg2rad(30) else cmap(i % 10)
        plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], c=seg_color, linewidth=1)
    
    if ordered_points is not None:
        pts2d = project(ordered_points, plane_origin, radial_dir, vertical_dir)
        plt.plot(pts2d[:,0], pts2d[:,1], color='k', linewidth=2, label="鏈夊簭璺緞")
        fill_offset = 3
        start_pt = pts2d[0]
        end_pt = pts2d[-1]
        start_left = start_pt - np.array([fill_offset, 0])
        end_left   = end_pt - np.array([(end_pt[0]-start_left[0]), 0])
        poly_points = np.vstack((pts2d, [end_left], [start_left]))
        plt.plot([end_pt[0], start_left[0]], [end_pt[1], start_left[1]], 'r--', linewidth=1)
    
    plt.title(f"Z鑼冨洿[{z_min:.2f}, {z_max:.2f}]渚у墫闈㈡姇褰憋紙绾㈣壊鏍囪澶硅<30掳锛?)
    plt.xlabel("寰勫悜鏂瑰悜 /m")
    plt.ylabel("鍨傜洿鏂瑰悜 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.legend()
    if show_arrows and ordered_points is not None:
        for i in range(len(pts2d) - 1):
            pt_start = pts2d[i]
            pt_end = pts2d[i+1]
            seg_vec = pt_end - pt_start
            seg_length = np.linalg.norm(seg_vec)
            if seg_length < 1e-6:
                continue
            seg_dir = seg_vec / seg_length
            mid = 0.5 * (pt_start + pt_end)
            arrow_len = seg_length * 0.3
            plt.arrow(mid[0], mid[1],
                      seg_dir[0] * arrow_len, seg_dir[1] * arrow_len,
                      color='k', head_width=0.01, head_length=0.05, linewidth=1)
    plt.show()

def draw_ordered_path_with_normal(ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max):
    """
    鏂板鐨勭粯鍥惧嚱鏁帮紙鍩轰簬鏈夊簭璺緞 ordered_path锛夛細
      - 閬嶅巻鏈夊簭璺緞涓繛缁殑姣忎釜绾挎锛屽厛鎶曞奖鍒颁簩缁村钩闈紝
      - 浣跨敤鎮ㄦ彁渚涚殑閫昏緫锛氱嚎娈垫柟鍚戝綊涓€鍖栧悗閫嗘椂閽堟棆杞?0掳寰楀埌娉曞悜閲忥紱
      - 璁＄畻娉曞悜閲忎笌璐?y 杞寸殑澶硅锛岃嫢澶硅灏忎簬30掳锛屽垯璇ョ嚎娈垫爣绾紝鍚﹀垯浣跨敤涓嶅悓棰滆壊缁樺埗锛?
      - 鍚屾椂鍦ㄦ瘡鏉＄嚎娈典腑缁樺埗娉曞悜閲忕澶淬€?
    """
    plt.figure(figsize=(6,6))
    if ordered_path is not None and len(ordered_path) > 0:
        cmap = plt.cm.get_cmap("tab10")
        neg_y = np.array([0, -1])
        for i in range(len(ordered_path) - 1):
            p1, p2 = ordered_path[i], ordered_path[i+1]
            # 鎶曞奖鍒颁簩缁村钩闈?
            p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
            # 缁樺埗绾挎锛堝厛璁＄畻娉曞悜閲忥級
            seg_vec = p2_2d - p1_2d
            norm_seg = np.linalg.norm(seg_vec)
            if norm_seg < 1e-6:
                continue
            seg_dir = seg_vec / norm_seg
            # 閫嗘椂閽堟棆杞?0掳寰楀埌娉曞悜閲?
            normal_2d = np.array([-seg_dir[1], seg_dir[0]])
            # 鍒ゆ柇涓庤礋y杞寸殑澶硅
            dot_val = np.clip(np.dot(normal_2d, neg_y), -1.0, 1.0)
            angle = np.arccos(dot_val)
            seg_color = 'red' if angle < np.deg2rad(30) else cmap(i % 10)
            plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color=seg_color, linewidth=2)
            
            # 缁樺埗娉曞悜閲忕澶?
            mid_2d = 0.5 * (p1_2d + p2_2d)
            normal_tip = mid_2d + normal_2d * 0.5
            plt.arrow(mid_2d[0], mid_2d[1],
                      normal_tip[0]-mid_2d[0], normal_tip[1]-mid_2d[1],
                      color='k', head_width=0.05, head_length=0.1, linewidth=1)
    
    plt.title(f"Z鑼冨洿[{z_min:.2f}, {z_max:.2f}]渚у墫闈㈡姇褰?)
    plt.xlabel("寰勫悜鏂瑰悜 /m")
    plt.ylabel("鍨傜洿鏂瑰悜 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()

def draw_ordered_path_with_normal(ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max):
    """
    鏂板鐨勭粯鍥惧嚱鏁帮紙鍩轰簬鏈夊簭璺緞 ordered_path锛夛細
      - 閬嶅巻鏈夊簭璺緞涓繛缁殑姣忎釜绾挎锛屽厛鎶曞奖鍒颁簩缁村钩闈紝
      - 浣跨敤鎮ㄧ粰瀹氱殑閫昏緫锛氱嚎娈垫柟鍚戝綊涓€鍖栧悗閫嗘椂閽堟棆杞?0掳寰楀埌娉曞悜閲忥紱
      - 璁＄畻娉曞悜閲忎笌璐?y 杞寸殑澶硅锛岃嫢澶硅灏忎簬30掳锛屽垯璇ョ嚎娈垫爣绾紝鍚﹀垯缁熶竴浣跨敤榛戣壊锛?
      - 鍚屾椂鍦ㄦ瘡鏉＄嚎娈典腑缁樺埗娉曞悜閲忕澶淬€?
    """
    plt.figure(figsize=(6,6))
    if ordered_path is not None and len(ordered_path) > 0:
        neg_y = np.array([0, -1])
        for i in range(len(ordered_path) - 1):
            p1, p2 = ordered_path[i], ordered_path[i+1]
            # 鎶曞奖鍒颁簩缁村钩闈?
            p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
            # 璁＄畻绾挎鍚戦噺
            seg_vec = p2_2d - p1_2d
            norm_seg = np.linalg.norm(seg_vec)
            if norm_seg < 1e-6:
                continue
            seg_dir = seg_vec / norm_seg
            # 閫嗘椂閽堟棆杞?0掳寰楀埌娉曞悜閲?
            normal_2d = np.array([-seg_dir[1], seg_dir[0]])
            # 鍒ゆ柇涓庤礋 y 杞寸殑澶硅
            dot_val = np.clip(np.dot(normal_2d, neg_y), -1.0, 1.0)
            angle = np.arccos(dot_val)
            seg_color = 'red' if angle < np.deg2rad(30) else 'black'
            plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color=seg_color, linewidth=2)
            
            # 缁樺埗娉曞悜閲忕澶?
            mid_2d = 0.5 * (p1_2d + p2_2d)
            normal_tip = mid_2d + normal_2d * 0.5
            # plt.arrow(mid_2d[0], mid_2d[1],
            #           normal_tip[0]-mid_2d[0], normal_tip[1]-mid_2d[1],
            #           color='k', head_width=0.05, head_length=0.1, linewidth=1)
    
    plt.title(f"娴嬪尯鍐呮煇宀╃┐鍒嗗竷澶勪晶鍓栭潰鎶曞奖")
    plt.xlabel("寰勫悜鏂瑰悜 /m")
    plt.ylabel("鍨傜洿鏂瑰悜 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()



def main():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 鍚庡彴閰嶇疆锛堣鏍规嵁瀹為檯鎯呭喌淇敼鏂囦欢璺緞锛?
    config_path = r"configs/arc_config.example.json"
    mesh_path = r"data/private/raw_model\combined_model.glb"
    show_arrows = True  # 淇敼涓篎alse鍒欎笉鏄剧ず绠ご

    # 鍔犺浇閰嶇疆銆佺綉鏍间笌瑁佸壀
    center, radius, angle_range, z_min, z_max = load_config(config_path)
    mesh = trimesh.load_mesh(mesh_path)
    print("缃戞牸鍔犺浇瀹屾垚")
    mesh = filter_mesh(mesh, z_min, z_max)
    if mesh is None:
        print("璀﹀憡锛氱粰瀹歓鑼冨洿鍐呮棤涓夎闈紒")
        return

    # 鍓栧垏骞抽潰鍙婁氦绾胯绠?
    plane_origin, plane_normal = compute_plane(center, angle_range)
    near_points, intersections = slice_mesh(mesh, plane_origin, plane_normal)
    print(f"杩戜技鍓栭潰鐐规暟: {len(near_points)}, 浜ょ嚎绾挎鏁? {len(intersections)}")

    ordered_path = order_segments(intersections)
    if ordered_path is None:
        print("鏃犳硶鏋勬垚杩炵画璺緞锛?)

    # 瀹氫箟骞抽潰鍧愭爣绯伙細x杞存部寰勫悜锛寉杞翠负绔栫洿鏂瑰悜
    mid_rad = 0.5 * sum(angle_range)
    radial_dir = np.array([np.cos(mid_rad), np.sin(mid_rad), 0])
    radial_dir /= np.linalg.norm(radial_dir)
    vertical_dir = np.array([0, 0, 1])

    # 鍙垎鍒皟鐢ㄤ笉鍚岀粯鍥惧嚱鏁帮紙鍘熸湁鍑芥暟淇濇寔涓嶅彉锛?
    # draw_result(intersections, ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max, show_arrows)
    # draw_result_with_normal(intersections, ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max, show_arrows)
    
    # 鏂板鐨勫熀浜庢湁搴忚矾寰勭殑缁樺浘鍑芥暟锛屾寜鐓ф寚瀹氶€昏緫璁＄畻娉曞悜閲忓苟鍒ゆ柇棰滆壊
    draw_ordered_path_with_normal(ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max)

if __name__ == "__main__":
    main()

