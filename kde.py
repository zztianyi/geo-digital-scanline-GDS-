import pickle
import numpy as np
import trimesh
import pyvista as pv
from plyfile import PlyData, PlyElement
import matplotlib.colors as mcolors
import matplotlib
import vtk  # 鐢ㄤ簬鍒涘缓2D姣斾緥灏虹浉鍏?actor

def create_custom_summer_red():
    """
    鍒涘缓涓€涓嚜瀹氫箟 colormap锛屽簳閮ㄩ噰鐢?summer 鑹插甫锛?
    涓婇儴浠?summer(0.8) 绾挎€ц繃娓″埌绾㈣壊 (1,0,0)銆?
    """
    summer = matplotlib.cm.get_cmap("summer")
    # 浣跨敤 from_list 鏉ユ瀯閫犳柊鐨?colormap
    new_cmp = mcolors.LinearSegmentedColormap.from_list(
        'SummerRed',
        [summer(0.0), summer(0.8), (1, 0, 0, 1)],
        N=256
    )
    return new_cmp


def load_trimesh_without_texture(obj_path):
    scene = trimesh.load(obj_path, force='scene')
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("OBJ 鍔犺浇澶辫触")
    all_meshes = [geom for _, geom in scene.geometry.items()]
    mesh_combined = trimesh.util.concatenate(all_meshes)
    return mesh_combined


def trimesh_to_pyvista_colored(mesh, vertex_scalar=None, scalar_name="Density"):
    faces = np.hstack([[3, *face] for face in mesh.faces])
    mesh_pv = pv.PolyData(mesh.vertices, faces)

    # 濡傛灉缁欏畾浜嗘爣閲忔暟缁勶紝鍒欑粦瀹氫负鐐规暟鎹?
    if vertex_scalar is not None and len(vertex_scalar) == len(mesh.vertices):
        mesh_pv.point_data[scalar_name] = vertex_scalar
    elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        # 鍚﹀垯浼樺厛灏濊瘯鍙鍖栭鑹?
        vc = mesh.visual.vertex_colors[:, :3]
        mesh_pv.point_data["Colors"] = vc / 255.0

    return mesh_pv

def update_scale_bar(plotter, rect_actor, text_actor, scale_bar_height=10):
    """
    鏍规嵁褰撳墠鐩告満瑙嗚鏇存柊姣斾緥灏烘樉绀猴細
      1. 鍦ㄧ獥鍙ｅ簳閮ㄨ瀹氬簳杈规樉绀哄潗鏍囷紙渚嬪绐楀彛瀹藉害鐨?10% 鍒?35%锛屽簳杈瑰湪绐楀彛楂樺害鐨?5%锛夈€?
      2. 浣跨敤褰撳墠鐩告満鐒︾偣瀵瑰簲鐨?z 鍧愭爣锛屽皢杩欎袱涓樉绀哄潗鏍囪浆鎹负涓栫晫鍧愭爣锛堝崟浣嶄笌缃戞牸涓€鑷达紝m锛夈€?
      3. 璁＄畻杩欎袱涓偣涔嬮棿鐨勬按骞充笘鐣岃窛绂伙紝涓嶅彈鐭╁舰楂樺害褰卞搷銆?
      4. 鏇存柊鐭╁舰鏂规涓庢枃鏈爣绛撅紝鏍囩閲囩敤鏂扮綏椹瓧浣撱€佸瓧鍙蜂负16锛屽崟浣嶄负 m銆?
    """
    ren = plotter.renderer
    width, height = plotter.window_size

    # 鑾峰彇褰撳墠鐩告満鐒︾偣鐨勪笘鐣屽潗鏍囷紝骞惰浆鎹负鏄剧ず鍧愭爣锛?
    # 浣跨敤鐒︾偣鎵€鍦ㄧ殑 z 鍊间綔涓鸿浆鎹㈡繁搴︼紝纭繚杞崲缁撴灉绋冲畾
    camera = ren.GetActiveCamera()
    focal_point = camera.GetFocalPoint()
    ren.SetWorldPoint(focal_point[0], focal_point[1], focal_point[2], 1.0)
    ren.WorldToDisplay()
    display_fp = ren.GetDisplayPoint()
    z_depth = display_fp[2]

    # 搴曡竟鍥哄畾鏄剧ず鍧愭爣锛堝崟浣嶏細鍍忕礌锛?
    start_x = 0.1 * width
    end_x = 0.35 * width
    y_base = 0.05 * height

    # 灏嗗簳杈圭殑涓や釜鏄剧ず鍧愭爣杞崲涓轰笘鐣屽潗鏍囷紝浣跨敤鐒︾偣鎵€鍦ㄧ殑 z 鍊?
    ren.SetDisplayPoint(start_x, y_base, z_depth)
    ren.DisplayToWorld()
    wp1_raw = ren.GetWorldPoint()  # 杩斿洖 [x, y, z, w]
    wp1 = np.array(wp1_raw[:3]) / wp1_raw[3]

    ren.SetDisplayPoint(end_x, y_base, z_depth)
    ren.DisplayToWorld()
    wp2_raw = ren.GetWorldPoint()
    wp2 = np.array(wp2_raw[:3]) / wp2_raw[3]

    # 浠呰绠楀簳杈逛袱涓偣闂寸殑涓栫晫璺濈锛堟按骞宠窛绂伙級
    world_distance = np.linalg.norm(wp2 - wp1)

    # 鏇存柊鐭╁舰鏂规鐨勭偣鍧愭爣锛堟瀯鎴愪竴涓í鍚戠煩褰㈣竟妗嗭級
    # 鐐归『搴忥細搴曞乏銆佸簳鍙炽€侀《鍙炽€侀《宸︺€佸啀闂悎鍒板簳宸?
    polydata = rect_actor.GetMapper().GetInput()
    pts = polydata.GetPoints()
    pts.SetPoint(0, start_x, y_base, 0)                      # 搴曞乏
    pts.SetPoint(1, end_x,   y_base, 0)                      # 搴曞彸
    pts.SetPoint(2, end_x,   y_base + scale_bar_height, 0)   # 椤跺彸
    pts.SetPoint(3, start_x, y_base + scale_bar_height, 0)   # 椤跺乏
    pts.SetPoint(4, start_x, y_base, 0)                      # 闂悎鍥炲簳宸?
    pts.Modified()

    # 鏇存柊鏂囨湰鏍囩锛屾樉绀烘按骞宠窛绂伙紙鍗曚綅 m锛夛紝璁剧疆鏂扮綏椹瓧浣撳強瀛楀彿涓?6
    label = f"{world_distance:.2f} m"
    text_actor.SetInput(label)
    text_actor.GetTextProperty().SetFontFamilyToTimes()   # 璁剧疆瀛椾綋涓烘柊缃楅┈ (Times New Roman)
    text_actor.GetTextProperty().SetFontSize(16)            # 璁剧疆瀛楀彿涓?16
    # 灏嗘枃鏈斁缃湪鐭╁舰涓嬫柟锛屾牴鎹渶瑕佸彲寰皟浣嶇疆
    text_actor.SetPosition((start_x + end_x) / 2 - 20, y_base - 30)
    
    plotter.render()


def main():
    # ==== 璺緞璁剧疆 ====
    mesh_path = r"data/private/raw_model\Merged mesh_seg.obj"
    # kde_result_path = r"data/private/raw_model\kde_face_density_5.0.pkl"#鎬讳綋绉殑绌洪棿鍒嗗竷瀵嗗害鏁版嵁kde浼拌甯﹀閫夋嫨涓?m

    kde_result_path = r"data/private/raw_model\kde_batch_density_5.0_0_0.1.pkl"
    # ==== 鍔犺浇缃戞牸锛堜笉鍔犺浇绾圭悊锛?====
    mesh_trimesh = load_trimesh_without_texture(mesh_path)

    # ==== 鍔犺浇 KDE 浼板€肩粨鏋?====
    with open(kde_result_path, "rb") as f:
        data = pickle.load(f)
    vertex_density = data["vertex_density"]

    # ==== 杞负 PyVista 骞惰祴鍊兼爣閲?====
    mesh_pv = trimesh_to_pyvista_colored(mesh_trimesh)
    mesh_pv.point_data["kde"] = vertex_density

    # 鍒涘缓涓や釜鐙珛鍓湰锛氬熀纭€灞傚拰鐑姏鍥惧眰
    mesh_base = mesh_pv.copy(deep=True)
    # 绉婚櫎鍩虹灞傜殑鏍囬噺鏁版嵁锛屼娇鍏跺彧鏄剧ず鍥哄畾棰滆壊
    if "kde" in mesh_base.point_data.keys():
        del mesh_base.point_data["kde"]

    mesh_overlay = mesh_pv.copy(deep=True)

   # ==== 璁＄畻闈炵嚎鎬ф槧灏勫悗鐨勮壊褰╀笌閫忔槑搴?====
    scalars = mesh_overlay.point_data["kde"]
    # 褰掍竴鍖栧埌 [0, 1]
    scalars_min = scalars.min()
    scalars_max = scalars.max()
    scalars_norm = (scalars - scalars_min) / (scalars_max - scalars_min + 1e-8)

    gamma = 4.0  # 闈炵嚎鎬у弬鏁帮紝鏍规嵁闇€瑕佽皟鑺?
    # 闈炵嚎鎬ф槧灏勫悗鐨勬爣閲忥紝鐢ㄤ簬棰滆壊鏄犲皠
    transformed = np.power(scalars_norm, gamma)
    # 鐢ㄤ簬閫忔槑搴︾殑闈炵嚎鎬ф槧灏?
    min_opacity = 0.8
    max_opacity = 1.0
    opacities = min_opacity + (max_opacity - min_opacity) * transformed

    # 灏嗛潪绾挎€ф槧灏勫悗鐨勬爣閲忎繚瀛樹笅鏉ワ紝鐢ㄤ簬鐑姏鍥捐壊褰╋紙鑼冨洿浠嶅湪 [0,1]锛?
    mesh_overlay.point_data["nonlinear_kde"] = transformed

    # ==== 鍙€夛細璁惧畾浣犳兂瑕佺殑 colormap锛岃繖閲屼互 "viridis" 涓轰緥 ====

    selected_cmap = create_custom_summer_red()
    # ==== 璁剧疆鍏ㄥ眬瀛椾綋涓?Times New Roman锛屽瓧鍙?4 ====
    pv.global_theme.font.family = "times"  # 淇敼鍏ㄥ眬瀛椾綋灞炴€?
    pv.global_theme.font.size = 14

    # ==== 璁剧疆 colormap 涓庡浘渚嬪弬鏁?====

    scalar_bar_args = {
        "vertical": True,          # 鍥句緥绔栫洿鎺掑垪
        "title": "kde",            # 鍥句緥鏍囬
        "position_x": 0.85,        # 璋冩暣鍥句緥X浣嶇疆
        "position_y": 0.1,         # 璋冩暣鍥句緥Y浣嶇疆
        "width": 0.1,             # 鍥句緥瀹藉害
        "height": 0.8,             # 鍥句緥楂樺害
        "font_family": "times",    # 鍥句緥涓墍鏈夋枃瀛楀瓧浣?
        "label_font_size": 16,     # 鍥句緥涓爣绛炬枃瀛楀瓧鍙?
        "title_font_size": 18      # 鍥句緥鏍囬瀛楀彿
    }
    # ==== 鍒嗗眰鏄剧ず ====
    plotter = pv.Plotter()
    # 绗竴灞傦細鍩虹缃戞牸锛屽叏涓嶉€忔槑锛屼娇鐢ㄥ浐瀹氶鑹叉樉绀?
    plotter.add_mesh(mesh_base, color="lightgray", opacity=0.9, show_edges=False)
    # 绗簩灞傦細鍙犲姞鐑姏鍥撅紝搴旂敤鏍囬噺鍜岄€忔槑搴?
    plotter.add_mesh(
        mesh_overlay,
        scalars="kde",
        cmap=selected_cmap,
        opacity=opacities,
        show_edges=False,
        scalar_bar_args=scalar_bar_args
    )
    # ==== 娣诲姞鍔ㄦ€佹瘮渚嬪昂锛堟í鐫€鐨勭櫧鑹茬煩褰㈡柟妗嗭級====
    # 鍒涘缓鐭╁舰妗嗙殑 polydata锛屽寘鍚?涓偣锛堥棴鍚堬級
    points = vtk.vtkPoints()
    for _ in range(5):
        points.InsertNextPoint(0, 0, 0)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(5)  # 瀹氫箟鍖呭惈5涓偣鐨勯棴鍚堢嚎
    for i in range(5):
        lines.InsertCellPoint(i)
    rect_polydata = vtk.vtkPolyData()
    rect_polydata.SetPoints(points)
    rect_polydata.SetLines(lines)

    rect_mapper = vtk.vtkPolyDataMapper2D()
    rect_mapper.SetInputData(rect_polydata)

    rect_actor = vtk.vtkActor2D()
    rect_actor.SetMapper(rect_mapper)
    rect_actor.GetProperty().SetColor(0, 0, 0)  # 鐧借壊
    rect_actor.GetProperty().SetLineWidth(2)     # 杈规瀹藉害
    plotter.renderer.AddActor2D(rect_actor)

    # 鍒涘缓鏂囨湰鏍囩鏄剧ず涓栫晫璺濈锛堝崟浣?m锛屾柊缃楅┈瀛椾綋锛?
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput("0.00 m")
    text_actor.GetTextProperty().SetFontSize(12)
    text_actor.GetTextProperty().SetColor(0, 0, 0)  # 鐧借壊
    # 鍒濆浣嶇疆锛屽悗缁敱 update_scale_bar 鍔ㄦ€佹洿鏂?
    text_actor.SetPosition(100, 20)
    plotter.renderer.AddActor2D(text_actor)

    def scale_bar_callback(caller, event):
        update_scale_bar(plotter, rect_actor, text_actor, scale_bar_height=10)
    
    # 灏嗚瀵熻€呮坊鍔犲埌褰撳墠娲诲姩鐩告満涓婏紙褰撹瑙掍慨鏀规椂瑙﹀彂鏇存柊锛?
    camera = plotter.renderer.GetActiveCamera()
    camera.AddObserver("ModifiedEvent", scale_bar_callback)
    
    # 鍒濆鏃舵洿鏂颁竴娆℃瘮渚嬪昂鏄剧ず
    update_scale_bar(plotter, rect_actor, text_actor, scale_bar_height=10)

    plotter.show()

if __name__ == "__main__":
    main()

