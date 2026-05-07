"""Structural group centroid extraction.

Extracts group centroids from multi-profile recognition outputs.
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


def extract_group_centroids(results_pkl, output_pkl):
    """
    提取每个 red_groups_corrected 中的组、质心以及面索引列表，并保存。
    保存格式为 List[ (group_segments, centroid, face_indices) ]
    """
    all_results = []
    with open(results_pkl, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                all_results.append(batch)
            except EOFError:
                break

    group_with_centroid_faces = []
    for res in all_results:
        red_groups_corrected = res.get("red_groups_corrected", {})
        red_centroids = res.get("red_centroids", {})

        for parent_idx, groups in red_groups_corrected.items():
            centroids = red_centroids.get(parent_idx, [])
            for i, group in enumerate(groups):
                group_segments = group.get("group_segments", [])
                if not group_segments:
                    continue
                if i < len(centroids):
                    centroid = centroids[i]
                    face_indices = [seg[6] for seg in group_segments if seg[6] is not None]
                    group_with_centroid_faces.append((group_segments, centroid, face_indices))

    # 保存
    with open(output_pkl, "wb") as f:
        pickle.dump(group_with_centroid_faces, f)

    print(f"[✓] 已提取并保存 {len(group_with_centroid_faces)} 组 red_group + 质心 + 面索引 数据至：{output_pkl}")


if __name__ == "__main__":
    results_pkl = str(get_path("multi_profile_output", create_parent=True))
    output_pkl = str(get_path("group_centroid_output", create_parent=True))
    extract_group_centroids(results_pkl, output_pkl)
