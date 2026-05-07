"""3MX/OBJ resource loading utility.

Inspects or combines model resources exported from 3MX-style scenes.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import trimesh
import os

# 数据目录
data_dir = get_path("raw_model_dir")

# 合并后的模型路径
combined_model_path = os.path.join(data_dir, "combined_model.obj")

# 检查文件是否存在
if os.path.exists(combined_model_path):
    print(f"加载 {combined_model_path}")
    
    # 加载模型时保留原始的颜色和纹理信息
    # process=False 防止自动处理导致丢失部分材质数据
    # use_embedded_textures=True 开启嵌入纹理的加载（注意：对于 .obj 文件，纹理通常由关联的 .mtl 文件提供，请确保 .mtl 文件与 .obj 同目录）
    mesh = trimesh.load(combined_model_path, process=False, use_embedded_textures=True)
    
    # 显示加载的网格
    mesh.show()
else:
    print(f"错误：{combined_model_path} 不存在，请检查文件路径！")