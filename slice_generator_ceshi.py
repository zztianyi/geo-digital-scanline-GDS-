import json
import trimesh
import pickle
import numpy as np
import pyvista as pv
import vtk

# 鍏佽 PyVista 鍔犺浇绌虹綉鏍?
pv.global_theme.allow_empty_mesh = True

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

def load_slices_data(pickle_file):
    with open(pickle_file, "rb") as f:
        slices_data = pickle.load(f)
    return slices_data

def add_slice_lines_to_plotter(plotter, slices_data, lower=40.0, upper=60.0, step=0.05):
    """
    姣忛殧 step 鎻愬彇涓€涓?key锛坒loat 绫诲瀷锛夛紝娣诲姞瀵瑰簲鐨勫垏鍓栭潰绾挎鍒?plotter 涓€?
    """
    points = []
    lines = []

    # 鏋勯€犵洰鏍囧姬闀垮簭鍒楋紙淇濈暀涓や綅灏忔暟閬垮厤娴偣璇樊锛?
    target_keys = {round(lower + i * step, 8) for i in range(int((upper - lower) / step) + 1)}

    for key, data in slices_data.items():
        try:
            pos = float(key)
        except ValueError:
            continue
        if round(pos, 8) not in target_keys:
            continue
        lines_3d = data["slicing"]["lines_3d"]
        for segment in lines_3d:
            segment = np.asarray(segment)
            if segment.shape[0] < 2:
                continue
            start = segment[0]
            end = segment[1]
            idx_start = len(points)
            idx_end = idx_start + 1
            points.extend([start, end])
            lines.append([2, idx_start, idx_end])

    if points and lines:
        pdata = pv.PolyData()
        pdata.points = np.array(points)
        pdata.lines = np.array(lines, dtype=np.int64)
        plotter.add_mesh(pdata, color="red", line_width=2, label="鍒囧墫闈㈢嚎娈?)

    return plotter
def bind_on_key_press(plotter):
    def on_key_press():
        save_path = r"data/examples\data_seg_5_2.png"
        plotter.screenshot(
            filename=save_path,
            scale=2.0,
            transparent_background=True,
            return_img=False
        )
        print(f"[鉁揮 褰撳墠瑙嗗浘宸蹭繚瀛樹负锛歿save_path}")
    return on_key_press

def multi_panel_display(mesh_pv, slices_data, step_values=(None, 0.5, 0.1, 0.05)):
    """
    鍒涘缓涓€涓?1脳4 瀛愬浘锛屼緷娆℃樉绀猴細
    1. 鍘熺綉鏍?
    2. 娣诲姞 step=0.5 鐨勫墫闈㈢嚎
    3. 娣诲姞 step=0.1 鐨勫墫闈㈢嚎
    4. 娣诲姞 step=0.05 鐨勫墫闈㈢嚎
    """
    # titles = ["鍘熺綉鏍?, "鍓栭潰 step=0.5", "鍓栭潰 step=0.1", "鍓栭潰 step=0.05"]

    plotter = pv.Plotter(shape=(1, 4), window_size=(2400, 600))
    plotter.enable_parallel_projection()
    plotter.link_views()  # 鑱斿姩瑙嗚

    for i, step in enumerate(step_values):
        if step is not None:
            plotter.subplot(0, i)
            add_slice_lines_to_plotter(plotter, slices_data, lower=80.7, upper=92.7, step=step)
        else:
            plotter.subplot(0, i)
            plotter.add_mesh(mesh_pv, color="lightgray", opacity=1, show_edges=False)
        # plotter.add_text(titles[i], position="upper_edge", font_size=12)
    # 娣诲姞鎸夐敭鎴浘
    plotter.add_key_event("s", bind_on_key_press(plotter))

    plotter.show(title="", auto_close=False)

def add_slice_lines_to_plotter(plotter, slices_data, lower, upper, step):
    points = []
    lines = []
    keys = {round(lower + i * step, 8) for i in range(int((upper - lower) / step) + 1)}

    for key, data in slices_data.items():
        try:
            pos = float(key)
        except ValueError:
            continue
        if round(pos, 8) not in keys:
            continue
        for segment in data["slicing"]["lines_3d"]:
            segment = np.asarray(segment)
            if segment.shape[0] < 2:
                continue
            idx_start = len(points)
            idx_end = idx_start + 1
            points.extend([segment[0], segment[1]])
            lines.append([2, idx_start, idx_end])

    if points and lines:
        pdata = pv.PolyData()
        pdata.points = np.array(points)
        pdata.lines = np.array(lines, dtype=np.int64)
        plotter.add_mesh(pdata, color="red", line_width=2)
    return plotter


def multi_panel_display_composite(mesh_pv, slices_data, group_pkl_path,
                                   lower=0, upper=140, step=0.05):
    plotter = pv.Plotter(shape=(1, 4), window_size=(2400, 600))
    plotter.enable_parallel_projection()
    plotter.link_views()

    # 1. 鍘熷缃戞牸
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_pv, color="lightgray", opacity=1, show_edges=False)

    # 2. 娣诲姞 step=0.05 鍓栭潰绾?
    plotter.subplot(0, 1)
    add_slice_lines_to_plotter(plotter, slices_data, lower, upper, step=step)

    # 3. 鏄剧ず red_group 鍧愭爣
    plotter.subplot(0, 2)
    with open(group_pkl_path, "rb") as f:
        group_data = pickle.load(f)
    for group_segments, _, _ in group_data:
        for seg in group_segments:
            p1 = np.array(seg[4])
            p2 = np.array(seg[5])
            plotter.add_lines(np.array([p1, p2]), color="red", width=2)

    # 4. 鏄剧ず mesh_pv 涓殑瀵瑰簲涓夎闈?
    plotter.subplot(0, 3)
    group_faces = set()
    for _, _, face_indices in group_data:
        group_faces.update(face_indices)

    if group_faces:
        face_indices = np.array(sorted(group_faces), dtype=int)
        mesh_subset = mesh_pv.extract_cells(face_indices)
        plotter.add_mesh(mesh_subset, color="red", show_edges=False, opacity=1)
    plotter.add_key_event("s", bind_on_key_press(plotter))
    plotter.show(title="鍥涘浘瀵规瘮锛氬師缃戞牸/鍓栭潰绾?绾㈢嚎娈?鐩爣闈?, auto_close=False)

def load_full_segments(pickle_path):
    """
    浠?pickle 鏂囦欢涓姞杞?all_segments銆?
    杩斿洖涓€涓?N脳3脳2 鐨勬暟缁勶紝琛ㄧず N 鏉＄嚎娈电殑涓や釜绔偣鍧愭爣銆?
    """
    with open(pickle_path, "rb") as f:
        all_segments, _ = pickle.load(f)
    # 鍙彇鍓嶄袱椤?p1, p2锛屽拷鐣?height
    lines = np.array([[seg[0], seg[1]] for seg in all_segments], dtype=float)
    return lines  # shape = (N, 2, 3)

def add_full_segments_to_plotter(plotter, lines, color="blue", line_width=2):
    """
    灏嗗鏉＄嚎娈垫暣鍚堜负涓€涓?PolyData 瀵硅薄骞舵坊鍔犲埌 plotter銆?
    lines: shape (N,2,3)
    """
    # 鏋勫缓 points 鍜?lines 鏁扮粍
    pts = lines.reshape(-1, 3)  # (N*2, 3)
    # 姣忔潯绾挎瀵瑰簲鐨勭储寮?
    n = lines.shape[0]
    connectivity = np.empty((n, 3), dtype=np.int64)
    # VTK line 鏍煎紡锛?[2, idx0, idx1]
    # 鎴戜滑鍏堟瀯閫犱簩缁?(N,2)锛岀劧鍚庡湪璧嬬粰 PolyData.lines 鏃惰嚜鍔ㄩ摵骞?
    for i in range(n):
        connectivity[i, 0] = 2
        connectivity[i, 1] = 2*i
        connectivity[i, 2] = 2*i + 1

    pdata = pv.PolyData()
    pdata.points = pts
    pdata.lines = connectivity
    plotter.add_mesh(pdata, color=color, line_width=line_width, label="full_segments")
    return plotter
def add_dynamic_scale_bar(plotter,
                          length=50.0,
                          bar_height=10,
                          display_margin_ratio=0.05,
                          font_size=12):
    """
    鍦ㄥ睆骞曞乏涓嬭娣诲姞涓€涓姩鎬佹瘮渚嬪昂锛?0 m锛夛紝璺濊竟缂樼害 display_margin_ratio 鐨勭獥鍙ｅ搴︼紝
    棰滆壊鍥哄畾涓洪粦鑹层€?
    """
    ren = plotter.renderer
    win_w, win_h = plotter.window_size

    # ==== 鏋勯€犵┖鐭╁舰 polydata ====
    pts = vtk.vtkPoints()
    for _ in range(5):
        pts.InsertNextPoint(0, 0, 0)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(5)
    for i in range(5):
        lines.InsertCellPoint(i)
    rect_pd = vtk.vtkPolyData()
    rect_pd.SetPoints(pts)
    rect_pd.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(rect_pd)
    bar_actor = vtk.vtkActor2D()
    bar_actor.SetMapper(mapper)
    bar_actor.GetProperty().SetColor(0, 0, 0)  # 榛戣壊
    bar_actor.GetProperty().SetLineWidth(2)
    ren.AddActor2D(bar_actor)

    # ==== 鏂囨湰 Actor ====
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(f"0鈥儃int(length)} m")
    tp = text_actor.GetTextProperty()
    tp.SetFontSize(font_size)
    tp.SetFontFamilyToTimes()
    tp.SetColor(0, 0, 0)  # 榛戣壊
    ren.AddActor2D(text_actor)

    # ==== 鏇存柊鍑芥暟 ====
    def update_scale_bar(caller, event):
        # 1. 璁＄畻灞忓箷鍍忕礌杈硅窛
        mx = win_w * display_margin_ratio
        my = win_h * display_margin_ratio

        # 2. 璁＄畻 world 鈫?display锛屽緱鍒扳€渓ength 绫斥€濆搴旂殑鍍忕礌瀹藉害
        # 浠绘剰涓€涓笘鐣岀偣 p0 鐢ㄤ簬娴嬮噺姣斾緥灏虹缉鏀撅紝涓嶅奖鍝嶄綅缃?
        bounds = plotter.bounds
        p0 = [bounds[0], bounds[2], bounds[4]]  # world 鏈€灏忕偣
        p1 = [p0[0] + length, p0[1], p0[2]]

        ren.SetWorldPoint(*p0, 1.0)
        ren.WorldToDisplay()
        d0 = ren.GetDisplayPoint()

        ren.SetWorldPoint(*p1, 1.0)
        ren.WorldToDisplay()
        d1 = ren.GetDisplayPoint()

        w = abs(d1[0] - d0[0])  # 鍍忕礌瀹藉害

        # 3. 鏇存柊鐭╁舰鍧愭爣锛堝叏閮ㄧ敤 display 鍧愭爣锛?
        pts.SetPoint(0, mx,     my,      0)
        pts.SetPoint(1, mx + w, my,      0)
        pts.SetPoint(2, mx + w, my + bar_height, 0)
        pts.SetPoint(3, mx,     my + bar_height, 0)
        pts.SetPoint(4, mx,     my,      0)
        pts.Modified()

        # 4. 鏇存柊鏂囧瓧浣嶇疆鍜屽唴瀹?
        text_actor.SetPosition(mx, my + bar_height + 2)
        text_actor.SetInput(f"0鈥儃int(length)} m")

    # 缁戝畾鐩告満鏇存柊浜嬩欢锛屼繚璇佽瑙掑彉鍖栨椂閲嶇粯
    cam = ren.GetActiveCamera()
    cam.AddObserver("ModifiedEvent", update_scale_bar)
    # 鍒濆缁樺埗涓€娆?
    update_scale_bar(None, None)

    return plotter

def main():
    # ===== 璺緞璁剧疆 =====
    config_path = r"configs/arc_config.example.json"
    mesh_path = r"data/examples/segmented_mesh\Merged mesh_seg_3.ply"
    # mesh_path = r"data/examples/sample_outcrop.ply""data/examples/segmented_mesh\Merged mesh_seg.ply"r"outputs/intermediate\Merged_seg_5_2.ply"
    mesh_path = r"data/examples/sample_outcrop.ply"
    # mesh_path = r"data/examples/segmented_mesh\Merged mesh_seg.ply"
    pickle_file = r"data/private/raw_model\all_slices_output_faces_zong.pkl"#"data/private/raw_model\line_face_data_40_buchang.pkl"
    pickle_file = r"data/private/raw_model\group_centroid_only.pkl"
    # pickle_file = r"data/private/raw_model\all_slices_output_faces_seg_60-70.pkl"
    # pickle_file = r"outputs/intermediate\all_slices_output_faces_seg_5_2.pkl"
    pickle_file = r"data/private/raw_model\all_slices_output_faces_seg_84-92.pkl"
    import os
    # ===== 鍔犺浇鏁版嵁 =====
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)
    mesh = trimesh.load_mesh(mesh_path)

    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
    slices_data = load_slices_data(pickle_file)

    # ===== 杞崲涓?PyVista 缃戞牸 =====
    mesh_pv = pv.wrap(mesh_filtered) if mesh_filtered is not None else pv.PolyData()
    multi_panel_display(mesh_pv, slices_data, step_values=(None, 0.5, 0.1, 0.05))

    # multi_panel_display_composite(mesh_pv, slices_data,
    #                           group_pkl_path=r"data/private/raw_model\polt_5_\group_centroid_only_5_2.pkl")
    # ===== 鍒涘缓 PyVista 缁樺浘瀵硅薄 =====
    # pickle_path = r"data/private/raw_model\line_face_data_40_buchang.pkl"
    # full_lines = load_full_segments(pickle_path)

    # 4. 鍙鍖?
    # 鍥?-
    plotter = pv.Plotter(window_size=(1200, 600))
    plotter.enable_parallel_projection()
    plotter.add_mesh(mesh_pv, color="lightgray", opacity=1, show_edges=False)
    plotter = add_slice_lines_to_plotter(plotter, slices_data, lower=0, upper=140, step=0.05)

    # 鍦ㄨ繖閲屾坊鍔犲叏閮ㄧ嚎娈?
    # plotter = add_full_segments_to_plotter(plotter, full_lines, color="red", line_width=3)
    # plotter.add_key_event("s", bind_on_key_press(plotter))
    # 3. 鍔犲姩鎬佹瘮渚嬪昂
    # add_dynamic_scale_bar(plotter, length=50.0, bar_height=10, display_margin_ratio=0.05, font_size=12)
    # plotter.show(title="缃戞牸 + 鍓栭潰绾挎 + 鍏ㄩ儴闈㈢嚎娈?, auto_close=False)
    plotter.show()


if __name__ == "__main__":
    main()


