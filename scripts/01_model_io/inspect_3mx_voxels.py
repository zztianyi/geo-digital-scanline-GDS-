"""3MX scene node inspection and lightweight point visualization.

This utility reads a 3MX scene folder, lists referenced 3MXB resources,
and tries to visualize simple point buffers when the resource layout is
compatible with the assumptions below. It is an inspection helper, not a
complete 3MX parser.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path

import json
import os
import struct
import numpy as np
import pyvista as pv


def load_3mx_files(data_dir):
    """Load the main scene.3mx JSON and collect nested .3mxb files."""
    data_dir = os.fspath(data_dir)
    main_file_path = os.path.join(data_dir, "scene.3mx")
    if not os.path.exists(main_file_path):
        raise FileNotFoundError(f"Main 3MX file not found: {main_file_path}")
    with open(main_file_path, "r", encoding="utf-8") as file:
        main_data = json.load(file)
    binary_dir = os.path.join(data_dir, "Data")
    if not os.path.exists(binary_dir):
        raise FileNotFoundError(f"3MXB data directory not found: {binary_dir}")
    binary_files = []
    for root, _, files in os.walk(binary_dir):
        for file in files:
            if file.lower().endswith(".3mxb"):
                binary_files.append(os.path.join(root, file))
    if not binary_files:
        raise ValueError("No .3mxb files were found in the 3MX scene folder.")
    return main_data, binary_files


def read_3mxb(file_path):
    """Read the 3MXB header and return the remaining buffer bytes."""
    with open(file_path, "rb") as f:
        magic = f.read(5).decode("utf-8", errors="replace")
        if magic != "3MXBO":
            raise ValueError(f"Invalid 3MXB magic value: {magic}")
        header_size = struct.unpack("<I", f.read(4))[0]
        json_header = f.read(header_size - 9).decode("utf-8")
        header_data = json.loads(json_header)
        buffer_data = f.read()
    return header_data, buffer_data


def parse_nodes(header_data):
    """Split nodes into parent and child lists when parent metadata exists."""
    nodes = header_data.get("nodes", [])
    parent_nodes = [node for node in nodes if "parent" not in node]
    child_nodes = [node for node in nodes if "parent" in node]
    print(f"Nodes: parents={len(parent_nodes)}, children={len(child_nodes)}")
    return parent_nodes, child_nodes


def visualize_nodes(nodes, buffer_data, title="3MX nodes"):
    """Visualize simple float32 XYZ buffers for quick inspection."""
    plotter = pv.Plotter(title=title)
    for node in nodes:
        start = int(node.get("resource_start", 0))
        end = int(node.get("resource_end", 0))
        resource_data = buffer_data[start:end]
        if len(resource_data) < 12:
            continue
        num_points = len(resource_data) // 12
        points = struct.unpack(f"<{num_points * 3}f", resource_data[:num_points * 12])
        points = np.array(points).reshape(-1, 3)
        plotter.add_mesh(pv.PolyData(points), point_size=5, render_points_as_spheres=True)
    plotter.show()


def main():
    data_dir = get_path("three_mx_scene_dir")
    main_data, binary_files = load_3mx_files(data_dir)
    print(json.dumps(main_data, indent=2, ensure_ascii=False)[:2000])
    for binary_file in binary_files:
        print(f"Parsing: {binary_file}")
        header_data, buffer_data = read_3mxb(binary_file)
        parent_nodes, child_nodes = parse_nodes(header_data)
        visualize_nodes(parent_nodes, buffer_data, title="Parent nodes")
        visualize_nodes(child_nodes, buffer_data, title="Child nodes")


if __name__ == "__main__":
    main()
