"""
pointcloud_to_dsm_mesh.py
=========================
鍔熻兘锛?
1. 璇诲彇 (x y z) 鐐逛簯 TXT锛堝墠涓夊垪鍗冲潗鏍囷級
2. 鐢熸垚鍗曞€?DSM锛氬悓涓€鍍忕礌鍐呭彧淇濈暀 "鏈€楂?Z"锛堝彲鏀规渶浣?骞冲潎锛?
3. 鎶?DSM 鏍呮牸椤剁偣鎷嗘垚杩炵画涓夎缃?
4. 淇濆瓨涓?OBJ锛屽苟鍙敤 PyVista 棰勮
"""

import os
import math
import numpy as np
import pyvista as pv
from scipy.ndimage import generic_filter


# ------------------------------------------------------------
def load_pointcloud(txt_path: str):
    """璇诲叆鐐逛簯 TXT锛屽彧鍙栧墠涓夊垪 x y z"""
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(txt_path)
    return np.loadtxt(txt_path, usecols=[0, 1, 2])


def points_to_dsm(points: np.ndarray,
                  resolution: float,
                  keep: str = "max",
                  fill_gaps: bool = True):
    """
    鐐逛簯 鈫?DSM 鍗曞€兼爡鏍?
    keep : 'max' | 'min' | 'mean'  鈫?鍚屽儚绱犺仛鍚堟柟寮?
    fill_gaps : True=鐢?3脳3 鏈€杩戦偦琛?NaN
    杩斿洖锛歞sm (ny,nx), xmin, ymin
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
        # 绠€鍗?3脳3 鏈€杩戦偦鍧囧€煎～娲?
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
    DSM 鏍呮牸 鈫?杩為€氫笁瑙掔綉 鈫?OBJ
    """
    ny, nx = dsm.shape

    # 鏋勯€犻《鐐?
    xs = xmin + np.arange(nx) * res
    ys = ymin + np.arange(ny) * res
    xx, yy = np.meshgrid(xs, ys)
    verts = np.c_[xx.ravel(order='F'),
                  yy.ravel(order='F'),
                  dsm.ravel(order='F')]

    # 涓夎闈㈢储寮?
    faces = np.empty(((nx - 1) * (ny - 1) * 2, 4), dtype=np.int32)
    idx = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = j + i * ny
            v10 = j + (i + 1) * ny
            v01 = j + 1 + i * ny
            v11 = j + 1 + (i + 1) * ny
            faces[idx] = [3, v00, v10, v01]  # 鈻?
            faces[idx + 1] = [3, v10, v11, v01]  # 鈻?
            idx += 2
    faces = faces.ravel()

    mesh = pv.PolyData(verts, faces)
    mesh.clean(inplace=True)          # 鐒婃帴椤剁偣锛屼繚璇佽繛閫?
    mesh.save(out_obj)
    print(f"[鉁揮 OBJ 宸蹭繚瀛橈細{out_obj}")
    print(f"    椤剁偣鏁?{mesh.n_points} | 涓夎闈㈡暟={mesh.n_cells}")

    if preview:
        mesh = pv.read(out_obj)           # 鈶?閲嶆柊鍔犺浇 OBJ
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
    # === 淇敼涓轰綘鐨?TXT 璺緞 ===
    txt_file = r"data/examples/sample_outcrop.txt"
    out_obj  = r"data/examples/sample_outcrop_dsm_mesh.obj"

    RESOLUTION = 0.1   # 鏍呮牸鍒嗚鲸鐜囷紙鍗曚綅涓庣偣浜戜竴鑷达級

    pts = load_pointcloud(txt_file)
    dsm, xmin, ymin = points_to_dsm(pts, RESOLUTION,
                                    keep="max", fill_gaps=True)
    raster_to_tri_mesh(dsm, xmin, ymin, RESOLUTION, out_obj, preview=True)

