"""Mesh verification helper.

Quickly loads and inspects a configured mesh file.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import trimesh
import numpy as np

def verify_mesh(mesh):
    # 检查每个 face 的顶点索引是否在合法范围内
    num_vertices = mesh.vertices.shape[0]
    invalid_face_indices = []
    for i, face in enumerate(mesh.faces):
        # 如果某个索引超出范围，则记录
        if np.any(face >= num_vertices) or np.any(face < 0):
            invalid_face_indices.append(i)
    
    if invalid_face_indices:
        print(f"发现 {len(invalid_face_indices)} 个面存在无效索引，示例：{invalid_face_indices[:10]}")
    else:
        print("所有面的索引均在有效范围内。")
    
    # 检查网格是否封闭（watertight）
    if mesh.is_watertight:
        print("网格是封闭的（watertight）。")
    else:
        print("网格不是封闭的（not watertight）。")
    
    # 检查面法向量一致性（winding consistent）
    if mesh.is_winding_consistent:
        print("网格的面法向量方向一致。")
    else:
        print("网格的面法向量方向不一致。")
    
    # 打印一些其他基本信息
    print("顶点数：", num_vertices)
    print("面数：", mesh.faces.shape[0])
    
    # 如果 trimesh 版本支持，可以调用 validate 方法
    try:
        validation = mesh.validate()
        if validation:
            print("网格验证信息：", validation)
        else:
            print("网格未发现明显验证错误。")
    except Exception as e:
        print("调用 mesh.validate() 时出错：", e)

# 示例：加载网格并验证
mesh_path = str(get_path("mesh_model"))
mesh = trimesh.load_mesh(mesh_path)
verify_mesh(mesh)
