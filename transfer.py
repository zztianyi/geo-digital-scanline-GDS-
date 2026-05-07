import pickle
import numpy as np
import trimesh
import laspy
import os
import json
# ===== 鍙傛暟璁剧疆 =====
results_pkl = r"data/private/raw_model\line_face_data_40_buchang.pkl"

mesh_path   = r"data/private/raw_model\combined_model.glb"
las_output  = r"data/private/raw_model\segments_lines.las"
obj_output  = r"data/private/raw_model\extracted_faces.obj"
sample_rate = 0.05

def load_arc_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config["z_range"]
def load_full_line_face_data(input_path):
    """
    浠?pickle 鏂囦欢涓姞杞藉畬鏁寸殑绾挎鏁版嵁鍙婂師濮嬮潰楂樺害瀛楀吀锛?
    鏁版嵁鏍煎紡涓?(all_segments, face_heights_dict)
    """
    with open(input_path, "rb") as f:
        all_segments, face_heights_dict = pickle.load(f)
    print("宸插姞杞藉畬鏁寸嚎娈靛強闈㈡暟鎹?)
    return all_segments, face_heights_dict

# 鍔犺浇瀹屾暣鏁版嵁锛氱嚎娈垫暟鎹強鍘熷闈㈤珮搴﹀瓧鍏革紙鏈悎骞讹級
segments, face_heights_dict = load_full_line_face_data(results_pkl)
# 鑾峰彇鎵€鏈夋秹鍙婄殑闈㈢储寮?
all_faces = list(face_heights_dict.keys())
def filter_mesh_by_z_range(mesh, z_min, z_max):
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if submeshes else None

# ===== 2. 绾挎閲囨牱涓虹偣浜戯紙绾㈣壊锛?=====
def sample_segment_points(segments, step=0.05):
    points = []
    for seg in segments:
        p1 = np.array(seg[0])
        p2 = np.array(seg[1])
        length = np.linalg.norm(p2 - p1)
        num_points = max(2, int(length / step))
        for t in np.linspace(0, 1, num_points):
            pt = (1 - t) * p1 + t * p2
            points.append(pt)
    return np.array(points)

sampled_points = sample_segment_points(segments, sample_rate)

# 鍒涘缓 las 鏂囦欢骞惰缃负绾㈣壊
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(sampled_points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])

las = laspy.LasData(header)
las.x, las.y, las.z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
las.red[:]   = 255
las.green[:] = 0
las.blue[:]  = 0

las.write(las_output)
print(f"[鉁揮 鐐逛簯宸蹭繚瀛樹负 LAS: {las_output}")

# ===== 3. 鍔犺浇缃戞牸骞舵彁鍙栭潰锛堢豢鑹诧級 =====
mesh = trimesh.load(mesh_path, force='mesh')
config_path = r"configs/arc_config.example.json"
z_min, z_max = load_arc_config(config_path)
mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
submesh = mesh_filtered.submesh([all_faces], only_watertight=False, append=True)

# 璁剧疆涓虹豢鑹诧紙椤剁偣棰滆壊鎴栭潰棰滆壊锛?
green = [0.0, 1.0, 0.0]
if hasattr(submesh, 'visual') and submesh.visual.kind != 'face':
    submesh.visual.vertex_colors = np.tile(np.array(green + [1.0]) * 255, (len(submesh.vertices), 1))

submesh.export(obj_output)
print(f"[鉁揮 缃戞牸宸蹭繚瀛樹负 OBJ: {obj_output}")
