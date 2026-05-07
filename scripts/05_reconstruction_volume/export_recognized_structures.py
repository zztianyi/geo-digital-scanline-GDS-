"""Export recognized structures.

Exports extracted faces and sampled structural line segments.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import pickle
import numpy as np
import trimesh
import laspy
import os
import json
# ===== 参数设置 =====
results_pkl = str(get_path("line_face_output", create_parent=True))

mesh_path   = str(get_path("mesh_model"))
las_output  = str(get_path("segment_lines_output", create_parent=True))
obj_output  = str(get_path("extracted_faces_output", create_parent=True))
sample_rate = 0.05

def load_arc_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config["z_range"]
def load_full_line_face_data(input_path):
    """
    从 pickle 文件中加载完整的线段数据及原始面高度字典，
    数据格式为 (all_segments, face_heights_dict)
    """
    with open(input_path, "rb") as f:
        all_segments, face_heights_dict = pickle.load(f)
    print("已加载完整线段及面数据")
    return all_segments, face_heights_dict

# 加载完整数据：线段数据及原始面高度字典（未合并）
segments, face_heights_dict = load_full_line_face_data(results_pkl)
# 获取所有涉及的面索引
all_faces = list(face_heights_dict.keys())
def filter_mesh_by_z_range(mesh, z_min, z_max):
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if submeshes else None

# ===== 2. 线段采样为点云（红色） =====
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

# 创建 las 文件并设置为红色
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(sampled_points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])

las = laspy.LasData(header)
las.x, las.y, las.z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
las.red[:]   = 255
las.green[:] = 0
las.blue[:]  = 0

las.write(las_output)
print(f"[✓] 点云已保存为 LAS: {las_output}")

# ===== 3. 加载网格并提取面（绿色） =====
mesh = trimesh.load(mesh_path, force='mesh')
config_path = str(get_path("arc_config"))
z_min, z_max = load_arc_config(config_path)
mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)
submesh = mesh_filtered.submesh([all_faces], only_watertight=False, append=True)

# 设置为绿色（顶点颜色或面颜色）
green = [0.0, 1.0, 0.0]
if hasattr(submesh, 'visual') and submesh.visual.kind != 'face':
    submesh.visual.vertex_colors = np.tile(np.array(green + [1.0]) * 255, (len(submesh.vertices), 1))

submesh.export(obj_output)
print(f"[✓] 网格已保存为 OBJ: {obj_output}")