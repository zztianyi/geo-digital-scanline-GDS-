"""DSM mesh generation.

Converts point-cloud samples into DSM-like triangulated meshes.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import os
import math
import numpy as np
import pyvista as pv
from scipy.ndimage import generic_filter


# ------------------------------------------------------------
def load_pointcloud(txt_path: str):
    """读入点云 TXT，只取前三列 x y z"""
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(txt_path)
    return np.loadtxt(txt_path, usecols=[0, 1, 2])


def points_to_dsm(points: np.ndarray,
                  resolution: float,
                  keep: str = "max",
                  fill_gaps: bool = True):
    """
    点云 → DSM 单值栅格
    keep : 'max' | 'min' | 'mean'  → 同像素聚合方式
    fill_gaps : True=用 3×3 最近邻补 NaN
    返回：dsm (ny,nx), xmin, ymin
    """
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    ncols = math.floor((xmax - xmin) / resolution) + 1
    nrows = math.floor((ymax - ymin) / resolution) + 1
    dsm = np.full((nrows, ncols), np.nan, dtype=np.float32)

    ix = ((xs - xmin) / resolution).astype(int)
    iy = ((ys - ymin) / resolution).astype(int)

    for cx, cy, cz in zip(ix, iy, zs):
        cur = dsm[cy, cx]
        if np.isnan(cur):
            dsm[cy, cx] = cz
        else:
            if keep == "max" and cz > cur:
                dsm[cy, cx] = cz
            elif keep == "min" and cz < cur:
                dsm[cy, cx] = cz
            elif keep == "mean":
                dsm[cy, cx] = (cur + cz) / 2.0

    if fill_gaps and np.isnan(dsm).any():
        # 简单 3×3 最近邻均值填洞
        def nn(vals):
            c = vals[len(vals) // 2]
            if not np.isnan(c):
                return c
            neigh = vals[~np.isnan(vals)]
            return neigh.mean() if neigh.size else np.nan
        dsm = generic_filter(dsm, nn, size=3, mode="nearest")

    return dsm, xmin, ymin


def raster_to_tri_mesh(dsm: np.ndarray,
                       xmin: float, ymin: float,
                       res: float,
                       out_obj: str,
                       preview: bool = True):
    """
    DSM 栅格 → 连通三角网 → OBJ
    """
    ny, nx = dsm.shape

    # 构造顶点
    xs = xmin + np.arange(nx) * res
    ys = ymin + np.arange(ny) * res
    xx, yy = np.meshgrid(xs, ys)
    verts = np.c_[xx.ravel(order='F'),
                  yy.ravel(order='F'),
                  dsm.ravel(order='F')]

    # 三角面索引
    faces = np.empty(((nx - 1) * (ny - 1) * 2, 4), dtype=np.int32)
    idx = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = j + i * ny
            v10 = j + (i + 1) * ny
            v01 = j + 1 + i * ny
            v11 = j + 1 + (i + 1) * ny
            faces[idx] = [3, v00, v10, v01]  # △1
            faces[idx + 1] = [3, v10, v11, v01]  # △2
            idx += 2
    faces = faces.ravel()

    mesh = pv.PolyData(verts, faces)
    mesh.clean(inplace=True)          # 焊接顶点，保证连通
    mesh.save(out_obj)
    print(f"[✓] OBJ 已保存：{out_obj}")
    print(f"    顶点数={mesh.n_points} | 三角面数={mesh.n_cells}")

    if preview:
        mesh = pv.read(out_obj)           # ② 重新加载 OBJ
        mesh.point_data["z"] = mesh.points[:, 2]

        p = pv.Plotter(window_size=[1200, 800])
        p.add_mesh(mesh, scalars="z", cmap="terrain", show_edges=False)
        p.add_scalar_bar("Elevation")
        p.show_bounds(grid="back", location="outer")
        p.camera_position = "iso"
        p.enable_anti_aliasing()
        p.show()


# ------------------------------------------------------------
if __name__ == "__main__":
    # === 修改为你的 TXT 路径 ===
    txt_file = str(get_path("sample_outcrop_txt"))
    out_obj  = str(get_path("segmented_mesh"))

    RESOLUTION = 0.1   # 栅格分辨率（单位与点云一致）

    pts = load_pointcloud(txt_file)
    dsm, xmin, ymin = points_to_dsm(pts, RESOLUTION,
                                    keep="max", fill_gaps=True)
    raster_to_tri_mesh(dsm, xmin, ymin, RESOLUTION, out_obj, preview=True)
