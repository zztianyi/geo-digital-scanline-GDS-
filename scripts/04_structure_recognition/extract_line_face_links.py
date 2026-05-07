"""Line-to-face extraction.

Converts profile-line recognition results into mesh face correspondence data.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import pickle
from collections import defaultdict

def gather_line_data(single_res):
    """
    从单个结果中提取线段与面索引数据，并计算高度值。
    对于每个线段组，先计算该组内所有线段端点的最高 z 值，
    然后对每条线段计算高度（定义为：组内最高 z 值减去该线段端点 z 值的平均）。
    同时，对于存在面索引的线段，将计算得到的高度记录到字典中。
    
    返回：
        - line_segments: List[ (p1_3d, p2_3d, height) ]
        - face_heights_dict: dict { face_idx : list of heights }
    """
    line_segments = []
    face_heights_dict = defaultdict(list)
    if not single_res:
        return line_segments, face_heights_dict
    
    red_groups_corrected = single_res.get("red_groups_corrected", {})
    for parent_idx, groups in red_groups_corrected.items():
        for group in groups:
            group_segments = group.get("group_segments", [])
            if not group_segments:
                continue
            # 先收集当前组中所有线段端点的 z 值
            zs = []
            for seg in group_segments:
                p1 = seg[4]
                p2 = seg[5]
                zs.append(p1[2])
                zs.append(p2[2])
            group_max_z = max(zs)
            # 遍历组内每条线段，计算高度并记录
            for seg in group_segments:
                p1 = seg[4]
                p2 = seg[5]
                face_idx = seg[6]
                avg_z = (p1[2] + p2[2]) / 2.0
                height = group_max_z - avg_z
                line_segments.append((p1, p2, height))
                if face_idx is not None:
                    face_heights_dict[face_idx].append(height)
    return line_segments, face_heights_dict

def serial_collect_line_data(all_results):
    """
    串行处理所有结果，收集所有线段数据及面对应的高度列表。
    
    返回：
        - all_segments: List[ (p1, p2, height) ]
        - all_face_heights_dict: dict { face_idx : list of heights }
    """
    all_segments = []
    all_face_heights_dict = defaultdict(list)
    for single_res in all_results:
        segments, face_heights = gather_line_data(single_res)
        all_segments.extend(segments)
        for face, heights in face_heights.items():
            all_face_heights_dict[face].extend(heights)
    return all_segments, all_face_heights_dict

def extract_line_face_from_pkl(results_pkl):
    """
    从指定的 pickle 文件中加载数据，
    并利用新的逻辑提取线段数据（含高度）以及面对应的高度列表。
    
    返回：
        - all_segments, face_heights_dict
    """
    all_results = []
    with open(results_pkl, "rb") as f:
        while True:
            try:
                batch_data = pickle.load(f)
                all_results.append(batch_data)
            except EOFError:
                break
    all_segments, face_heights_dict = serial_collect_line_data(all_results)
    return all_segments, face_heights_dict

def merge_face_heights(face_heights_dict, strategy='max'):
    """
    将同一面对应的多个高度值合并为一个代表值。
    
    参数：
        face_heights_dict: dict { face_idx: list of heights }
        strategy: 合并策略，可选 'max'（最大值）、'min'（最小值）或 'avg'（平均值）
    
    返回：
        merged: dict { face_idx: merged height }
    """
    merged = {}
    for face, heights in face_heights_dict.items():
        if not heights:
            continue
        if strategy == 'max':
            merged[face] = max(heights)
        elif strategy == 'min':
            merged[face] = min(heights)
        elif strategy == 'avg':
            merged[face] = sum(heights) / len(heights)
        else:
            merged[face] = max(heights)
    return merged

def save_full_line_face_data(all_segments, face_heights_dict, output_path):
    """
    保存提取的线段数据和原始面高度字典。
    保存的数据格式为 (all_segments, face_heights_dict)。
    
    参数：
        all_segments: 所有线段数据（含高度），列表形式，例如 [(p1, p2, height), ...]
        face_heights_dict: dict，每个面对应的多个高度数据，未合并
        output_path: str，保存的文件路径，例如 'line_face_full_data.pkl'
    """
    with open(output_path, "wb") as f:
        pickle.dump((all_segments, face_heights_dict), f)
    print("完整数据已保存到：", output_path)

if __name__ == "__main__":
    # 示例：使用新的提取逻辑和保存逻辑
    results_pkl = str(get_path("multi_profile_output", create_parent=True))
    output_path = str(get_path("line_face_output", create_parent=True))

    # results_pkl = str(get_path("multi_profile_output", create_parent=True))
    
    print("开始提取线段和面数据（含高度）...")
    all_segments, face_heights_dict = extract_line_face_from_pkl(results_pkl)
    print("提取完成，线段数：", len(all_segments), "涉及目标面数：", len(face_heights_dict))
    
    save_full_line_face_data(all_segments, face_heights_dict, output_path)

    

