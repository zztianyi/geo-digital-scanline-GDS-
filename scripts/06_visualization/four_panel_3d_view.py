"""Four-panel 3D visualization helper.

The script displays a textured mesh, its point cloud, a plain mesh, and a DSM
mesh in linked PyVista views. Press ``s`` in the viewer to save a screenshot.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path

import os
import numpy as np
import pyvista as pv


def main():
    figure_dir = get_path("figure_output_dir")
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_path = figure_dir / "four_panel_view.png"
    mesh_ply = get_path("sample_outcrop_ply")
    tex_jpg = get_path("sample_texture")
    point_txt = get_path("sample_outcrop_txt")
    dsm_obj = get_path("sample_dsm_mesh")
    plotter = pv.Plotter(shape=(1, 4), window_size=(2400, 600))
    plotter.link_views()
    font_opts = {"font_size": 20, "font": "times"}

    def on_key_press():
        plotter.screenshot(str(save_path), scale=1)
        print(f"Saved screenshot: {save_path}")

    plotter.subplot(0, 0)
    mesh_tex = pv.read(mesh_ply)
    if os.path.isfile(tex_jpg):
        plotter.add_mesh(mesh_tex, texture=pv.read_texture(tex_jpg))
    else:
        plotter.add_mesh(mesh_tex, color="white")
    plotter.add_text("a.", position="upper_left", **font_opts)
    plotter.subplot(0, 1)
    points = np.loadtxt(point_txt, usecols=[0, 1, 2])
    plotter.add_mesh(pv.PolyData(points), render_points_as_spheres=True, point_size=1, color="lightgray")
    plotter.add_text("b.", position="upper_left", **font_opts)
    plotter.subplot(0, 2)
    plotter.add_mesh(pv.read(mesh_ply), show_edges=False, color="lightgray", opacity=1)
    plotter.add_text("c.", position="upper_left", **font_opts)
    plotter.subplot(0, 3)
    plotter.add_mesh(pv.read(dsm_obj), color="lightgray", show_edges=False, opacity=1)
    plotter.add_text("d.", position="upper_left", **font_opts)
    plotter.add_key_event("s", on_key_press)
    plotter.show()


if __name__ == "__main__":
    main()
