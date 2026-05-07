import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import pyvista as pv
from matplotlib import cm
import laspy
from collections import Counter
import vtk  # 鐢ㄤ簬鍒涘缓2D姣斾緥灏虹浉鍏?actor
# ===== 1. 鍔犺浇绋€鐤忕偣浜戞暟鎹?=====
voxel_pkl = r"data/private/raw_model\voxels_sparse_buchang.pkl"
mesh_path = r"data/private/raw_model\Merged mesh_seg.obj"
with open(voxel_pkl, "rb") as f:
    voxel_data = pickle.load(f)

voxel_indices = np.array(voxel_data["active_voxels"])
origin = np.array(voxel_data["origin"])
voxel_size = voxel_data["voxel_size"]
points = (voxel_indices + 0.5) * voxel_size + origin  # 鎵€鏈夌偣浜戝潗鏍?(N, 3)

print(f"[鉁揮 鍔犺浇鐐逛簯瀹屾垚锛屽叡 {len(points)} 涓偣")

# ===== 2. 鎶界█锛氶殢鏈洪噰鏍蜂竴閮ㄥ垎鐐硅繘琛岃仛绫?=====
sample_ratio = 0.1  # 鎶界█姣斾緥锛屼緥濡傚彇 10%
sample_indices = np.random.choice(len(points), size=int(len(points) * sample_ratio), replace=False)
sampled_points = points[sample_indices]

# ===== 3. DBSCAN 鑱氱被锛堝湪鎶界█鐐逛笂锛?====
dbscan = DBSCAN(eps=0.2, min_samples=5)
sample_labels = dbscan.fit_predict(sampled_points)

num_clusters = len(set(sample_labels)) - (1 if -1 in sample_labels else 0)
print(f"[鉁揮 鑱氱被瀹屾垚锛岀皣鏁? {num_clusters}")

# ===== 4. 浣跨敤 KDTree 灏嗘爣绛炬槧灏勫洖鎵€鏈夊師濮嬬偣 =====
tree = cKDTree(sampled_points)
_, idx = tree.query(points, k=1)
labels_full = sample_labels[idx]  # 涓烘瘡涓師濮嬬偣鍒嗛厤鑱氱被鏍囩

# ===== 5. 鏍规嵁鑱氱被缁熻浣撶Н锛屽苟绛涢€夊墠10澶х皣 =====
voxel_volume = 0.05 ** 3  # 姣忎釜浣撶礌浣撶Н锛堢珛鏂圭背锛?

# 缁熻姣忎釜鑱氱被缂栧彿鐨勪綋绱犳暟閲忥紙涓嶅惈 -1锛?
label_counter = Counter(labels_full[labels_full != -1])
top10 = label_counter.most_common(10)  # [(label, count), ...]

print("\n[鉁揮 鍓?0澶ц仛绫讳綋绉細")
for i, (label, count) in enumerate(top10):
    volume = count * voxel_volume
    print(f"  Top-{i+1}: Label={label:<3}  鐐规暟={count:<6}  浣撶Н={volume:.4f} m鲁")

# 鏋勫缓棰滆壊鏄犲皠锛氫粎鍓?0澶ц仛绫诲垎閰嶉鑹诧紝鍏朵綑涓鸿摑鑹?
top10_labels = set(l for l, _ in top10)
top10_labels_sorted = sorted(top10_labels)
colormap = cm.get_cmap("turbo", len(top10_labels))
colors = np.full((len(labels_full), 4), fill_value=[220, 220, 255, 150], dtype=np.uint8)  # 娴呰摑 + 閫忔槑搴?50


for idx_color, label in enumerate(top10_labels_sorted):
    mask = labels_full == label
    rgba = np.append((np.array(colormap(idx_color)[:3]) * 255).astype(np.uint8), 255)  # 鍔犱笂涓嶉€忔槑
    colors[mask] = rgba


# ===== 6. 鍒涘缓 PyVista 鐐逛簯瀵硅薄骞跺睍绀?=====
cloud = pv.PolyData(points)
cloud["Colors"] = colors.astype(np.uint8)

# ===== 7. 璁＄畻鎵€鏈夋湁鏁堣仛绫荤殑璐ㄥ績 =====
# 閬嶅巻鏈夋晥鑱氱被锛堟帓闄ゅ櫔鐐癸級鎻愬彇璐ㄥ績
all_centroids = []
for label in sorted(label_counter):
    cluster_points = points[labels_full == label]
    centroid = cluster_points.mean(axis=0)
    all_centroids.append(centroid)
all_centroids = np.array(all_centroids)
print(f"[鉁揮 宸叉彁鍙栨湁鏁堣仛绫昏川蹇冩暟閲忥細{len(all_centroids)}")
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

# ===== 8. 瀹氫箟鍑芥暟锛屾牴鎹綋绉笂涓嬮檺鎻愬彇鑱氱被璐ㄥ績 =====
def extract_centroids_by_volume(points, labels, voxel_volume, volume_min, volume_max):
    """
    杈撳叆锛?
      points: 鎵€鏈夌偣浜戝潗鏍?(N, 3)
      labels: 姣忎釜鐐瑰搴旂殑鑱氱被鏍囩锛屽櫔鐐规爣绛句负 -1
      voxel_volume: 姣忎釜浣撶礌鐨勪綋绉紙m鲁锛?
      volume_min, volume_max: 闇€瑕佺瓫閫夌殑浣撶Н鑼冨洿锛堝崟浣?m鲁锛?
    
    鍔熻兘锛?
      1. 缁熻鎵€鏈夋湁鏁堬紙鎺掗櫎鍣偣锛夎仛绫荤殑浣撶Н锛屽苟鎵撳嵃姣忎釜鑱氱被鐨勪綋绉俊鎭紱
      2. 鎵撳嵃鎵€鏈夋湁鏁堣仛绫荤殑鎬讳綋绉紱
      3. 閽堝浣撶Н鍦?[volume_min, volume_max] 鑼冨洿鍐呯殑鑱氱被锛屾墦鍗板叾浣撶Н淇℃伅鍙婂畠浠湪鎬讳綋绉腑鐨勫崰姣旓紱
      4. 杩斿洖婊¤冻鏉′欢鑱氱被鐨勮川蹇冿紙鍜屽悇鑷綋绉級銆?
    """
    valid_labels = labels[labels != -1]
    label_counter = Counter(valid_labels)
    total_valid_volume = sum(count * voxel_volume for count in label_counter.values())
    
    # 缁熻鍣偣鐨勪綋绉?
    noise_count = (labels == -1).sum()
    noise_volume = noise_count * voxel_volume
    # 鎵€鏈夌偣浜戠殑鎬讳綋绉紙鍖呮嫭鍣偣锛?
    total_volume = total_valid_volume + noise_volume

    print("\n鎵€鏈夋湁鏁堣仛绫荤疮绉€讳綋绉?(鎺掗櫎鍣偣): {:.4f} m鲁".format(total_valid_volume))
    print("鎵€鏈夌偣浜戞€讳綋绉?(鍖呮嫭鍣偣): {:.4f} m鲁".format(total_volume))
    if total_volume > 0:
        percentage = total_valid_volume / total_volume * 100
    else:
        percentage = 0
    print("鏈夋晥鑱氱被浣撶Н鍗犳瘮: {:.2f}%".format(percentage))
    
    selected_centroids = []
    selected_volumes = []
    selected_total_volume = 0.0
    # print("\n婊¤冻鏉′欢锛堜綋绉寖鍥? {:.4f} m鲁 ~ {:.4f} m鲁锛夌殑鑱氱被锛?.format(volume_min, volume_max))
    for label in sorted(label_counter):
        count = label_counter[label]
        vol = count * voxel_volume
        if volume_min <= vol <= volume_max:
            cluster_points = points[labels == label]
            centroid = cluster_points.mean(axis=0)
            selected_centroids.append(centroid)
            selected_volumes.append(vol)
            selected_total_volume += vol
            print(f"  Label={label:<3}  鐐规暟={count:<6}  浣撶Н={vol:.4f} m鲁")
    
    percentage = (selected_total_volume / total_valid_volume * 100) if total_valid_volume > 0 else 0
    print("\n婊¤冻鏉′欢鐨勮仛绫绘€讳綋绉? {:.4f} m鲁锛屽崰鏈夋晥鑱氱被鎬讳綋绉殑 {:.2f}%".format(selected_total_volume, percentage))
    
    return np.array(selected_centroids), np.array(selected_volumes)

# 渚嬶細璁剧疆浣撶Н涓嬮檺涓?0.002 m鲁锛屼笂闄愪负 0.01 m鲁锛堝彲鏍规嵁瀹為檯鎯呭喌璋冩暣锛?
volume_lower_bound = 0
volume_upper_bound = 0.1
selected_centroids, selected_volumes = extract_centroids_by_volume(points, labels_full, voxel_volume, volume_lower_bound, volume_upper_bound)

# ===== 9. 淇濆瓨鎻愬彇鐨勮仛绫昏川蹇冧负 LAS 鏂囦欢 =====
if selected_centroids.size > 0:
    print(f"[鉁揮 鎻愬彇婊¤冻鏉′欢鐨勮仛绫绘暟閲? {len(selected_centroids)}")
    # 缁熶竴棰滆壊璁剧疆涓虹孩鑹?
    centroid_colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint16), (len(selected_centroids), 1))
    
    # 鍒涘缓 las 鏂囦欢澶?
    centroid_header = laspy.LasHeader(point_format=3, version="1.2")
    centroid_header.offsets = np.min(selected_centroids, axis=0)
    centroid_header.scales = np.array([0.001, 0.001, 0.001])
    
    # 鍒涘缓 las 鏁版嵁瀵硅薄
    centroid_las = laspy.LasData(centroid_header)
    centroid_las.x = selected_centroids[:, 0]
    centroid_las.y = selected_centroids[:, 1]
    centroid_las.z = selected_centroids[:, 2]
    centroid_las.red   = centroid_colors[:, 0]
    centroid_las.green = centroid_colors[:, 1]
    centroid_las.blue  = centroid_colors[:, 2]
    
    # 鍐欏叆鏂囦欢
    centroid_output_path = r"data/private/raw_model\cluster_centroids_filtered_1_10.las"
    centroid_las.write(centroid_output_path)
    print(f"[鉁揮 婊¤冻鏉′欢鐨勮仛绫昏川蹇冨凡淇濆瓨涓?LAS 鏂囦欢锛歿centroid_output_path}")
else:
    print("[!] 娌℃湁鎵惧埌婊¤冻鏉′欢鐨勮仛绫汇€?)

plotter = pv.Plotter()
plotter.add_points(cloud, scalars="Colors", rgba=True, render_points_as_spheres=True, point_size=5)

# plotter.show_bounds(grid="front", location="outer", all_edges=True)
plotter.hide_axes() 
# ===== 娣诲姞鑱氱被娉ㄩ噴鏍囩 =====
from collections import Counter
label_counter = Counter(labels_full[labels_full != -1])

# 璁＄畻姣忎釜鑱氱被鐨勮川蹇冨強浣撶Н锛屽苟鏁寸悊鍒板垪琛ㄤ腑锛?鑱氱被缂栧彿, 鐐规暟, 浣撶Н, 璐ㄥ績)
cluster_info = []
for label in label_counter:
    count = label_counter[label]
    vol = count * voxel_volume
    cluster_points = points[labels_full == label]
    centroid = cluster_points.mean(axis=0)
    cluster_info.append((label, count, vol, centroid))

# 鎸変綋绉檷搴忔帓搴忥紝骞跺彇鍓?0涓仛绫?
sorted_cluster_info = sorted(cluster_info, key=lambda x: x[2], reverse=True)[:15]

# 鏋勯€犳敞閲婃枃鏈笌瀵瑰簲浣嶇疆鍒楄〃
annotation_points = []
annotation_labels = []
for i, (label, count, vol, centroid) in enumerate(sorted_cluster_info, start=1):
    annotation_points.append(centroid)
    annotation_labels.append(f"(鎮┖浣搟i}) 浣撶Н: {vol:.4f} m鲁")

# 娣诲姞鐐规爣绛撅紝璁剧疆瀛椾綋涓烘柊缃楅┈銆佺櫧鑹层€佸瓧鍙?6
plotter.add_point_labels(
    annotation_points,
    annotation_labels,
    font_size=20,
    font_family="times",
    text_color="white"
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


plotter.show(title="绋€鐤忔娊鏍?DBSCAN 鑱氱被缁撴灉 (鏄犲皠鍒板叏閮ㄧ偣浜?")
