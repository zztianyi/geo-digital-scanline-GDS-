"""Single-profile mesh slicing.

Cuts a mesh by one scanline plane and projects the intersection to 2D.
This script is part of the Digital Scanline Framework for Complex Rock-Wall Structure Recognition.
"""

from pathlib import Path as _GDSPath
import sys as _gds_sys
_GDS_ROOT = _GDSPath(__file__).resolve().parents[2]
if str(_GDS_ROOT) not in _gds_sys.path:
    _gds_sys.path.insert(0, str(_GDS_ROOT))
from gds_project.config import get_path, get_font_properties

import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib
import time
from matplotlib.path import Path

def load_config(config_path):
    """加载配置文件，返回圆心、半径、角度区间以及Z轴范围"""
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    center = np.append(np.array(cfg["center"], dtype=float), 0.0)
    radius = float(cfg["radius"])
    angle_range = [cfg["angle_min"], cfg["angle_max"]]
    z_min, z_max = cfg["z_range"]
    return center, radius, angle_range, z_min, z_max

def compute_plane(center, angle_range):
    """根据角度区间计算剖切平面的原点和法向量"""
    mid_angle = 0.5 * sum(angle_range)
    radial_dir = np.array([np.cos(mid_angle), np.sin(mid_angle), 0.0])
    normal = np.cross(radial_dir, [0, 0, 1])
    normal /= np.linalg.norm(normal)
    return center, normal

def filter_mesh(mesh, z_min, z_max):
    """过滤出Z轴范围内（有部分顶点落在区间内）的三角面"""
    face_z = mesh.vertices[mesh.faces][:, :, 2]
    mask = (face_z.min(axis=1) <= z_max) & (face_z.max(axis=1) >= z_min)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if submeshes else None

def slice_mesh(mesh, origin, normal, tol=0.1):
    """计算顶点距离和交线，返回满足tol条件的顶点和交线集合"""
    dist = np.dot(mesh.vertices - origin, normal)
    near_points = mesh.vertices[np.abs(dist) <= tol]
    start = time.time()
    intersections = trimesh.intersections.mesh_plane(mesh, normal, origin)
    print(f"mesh_plane 计算用时: {time.time() - start:.6f} 秒")
    return near_points, intersections

def project(points, origin, axis_x, axis_y):
    """将3D点投影到由origin、axis_x、axis_y定义的平面坐标系上"""
    return np.array([[np.dot(p - origin, axis_x), np.dot(p - origin, axis_y)] for p in points])

def order_segments(segments, tol=1e-6):
    """
    构建图结构，使用DFS找到从z最大到z最小点的有序路径
    返回有序路径（ordered_path），即一系列连续的3D点
    """
    graph, pts = {}, {}
    def add_pt(p):
        key = tuple(np.round(p, 6))
        pts.setdefault(key, p)
        return key
    for seg in segments:
        k1, k2 = add_pt(seg[0]), add_pt(seg[1])
        graph.setdefault(k1, []).append(k2)
        graph.setdefault(k2, []).append(k1)
    start = max(pts.keys(), key=lambda k: pts[k][2])
    end = min(pts.keys(), key=lambda k: pts[k][2])
    path, found = [], False
    def dfs(cur, target, cur_path):
        nonlocal path, found
        if found:
            return
        cur_path.append(cur)
        if cur == target:
            path, found = cur_path.copy(), True
            return
        for nb in graph.get(cur, []):
            if nb not in cur_path:
                dfs(nb, target, cur_path)
        cur_path.pop()
    dfs(start, end, [])
    return [pts[k] for k in path] if path else None

def compute_special_normal(seg_2d):
    """
    根据给定二维线段 seg_2d（包含两个端点的坐标）计算法向量，
    逻辑：先计算线段方向，再逆时针旋转90°得到法向量
    返回的 normal_2d 为单位向量
    """
    p1, p2 = seg_2d
    seg_vec = p2 - p1
    norm = np.linalg.norm(seg_vec)
    if norm < 1e-6:
        return None
    seg_dir = seg_vec / norm
    normal_2d = np.array([-seg_dir[1], seg_dir[0]])
    return normal_2d

def draw_result(intersections, ordered_points, plane_origin, radial_dir, vertical_dir,
                z_min, z_max, show_arrows=True):
    """
    原有的绘图函数：
      - 绘制所有交线段（彩色显示）
      - 绘制有序路径及封闭多边形，并填充“*”
      - 根据show_arrows参数决定是否显示路径箭头
    """
    plt.figure(figsize=(6,6))
    cmap = plt.cm.get_cmap("tab10")
    for i, seg in enumerate(intersections):
        p1, p2 = seg[0], seg[1]
        p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
        plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], c=cmap(i % 10), linewidth=1)
    
    if ordered_points is not None:
        pts2d = project(ordered_points, plane_origin, radial_dir, vertical_dir)
        plt.plot(pts2d[:,0], pts2d[:,1], color='k', linewidth=2, label="有序路径")
        fill_offset = 3
        start_pt = pts2d[0]
        end_pt = pts2d[-1]
        start_left = start_pt - np.array([fill_offset, 0])
        end_left   = end_pt - np.array([(end_pt[0]-start_left[0]), 0])
        poly_points = np.vstack((pts2d, [end_left], [start_left]))
        plt.plot([end_pt[0], start_left[0]], [end_pt[1], start_left[1]], 'r--', linewidth=1)
    
    plt.title(f"Z范围[{z_min:.2f}, {z_max:.2f}]侧剖面投影")
    plt.xlabel("径向方向 /m")
    plt.ylabel("垂直方向 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.legend()
    if show_arrows and ordered_points is not None:
        for i in range(len(pts2d) - 1):
            pt_start = pts2d[i]
            pt_end = pts2d[i+1]
            seg_vec = pt_end - pt_start
            seg_length = np.linalg.norm(seg_vec)
            if seg_length < 1e-6:
                continue
            seg_dir = seg_vec / seg_length
            mid = 0.5 * (pt_start + pt_end)
            arrow_len = seg_length * 0.3
            plt.arrow(mid[0], mid[1],
                      seg_dir[0] * arrow_len, seg_dir[1] * arrow_len,
                      color='k', head_width=0.01, head_length=0.05, linewidth=1)
    plt.show()

def draw_result_with_normal(intersections, ordered_points, plane_origin, radial_dir, vertical_dir,
                              z_min, z_max, show_arrows=True):
    """
    新增的绘图函数（基于交线段）：
      - 对每条交线段调用 compute_special_normal 得到法向量
      - 判断该法向量与负y轴的夹角，若小于30°则将对应线段标红显示
      - 其他部分与原函数类似
    """
    plt.figure(figsize=(6,6))
    cmap = plt.cm.get_cmap("tab10")
    neg_y = np.array([0, -1])
    
    for i, seg in enumerate(intersections):
        p1, p2 = seg[0], seg[1]
        p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
        seg_2d = np.array([p1_2d, p2_2d])
        special_normal = compute_special_normal(seg_2d)
        if special_normal is None:
            continue
        angle = np.arccos(np.clip(np.dot(special_normal, neg_y), -1.0, 1.0))
        seg_color = 'red' if angle < np.deg2rad(30) else cmap(i % 10)
        plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], c=seg_color, linewidth=1)
    
    if ordered_points is not None:
        pts2d = project(ordered_points, plane_origin, radial_dir, vertical_dir)
        plt.plot(pts2d[:,0], pts2d[:,1], color='k', linewidth=2, label="有序路径")
        fill_offset = 3
        start_pt = pts2d[0]
        end_pt = pts2d[-1]
        start_left = start_pt - np.array([fill_offset, 0])
        end_left   = end_pt - np.array([(end_pt[0]-start_left[0]), 0])
        poly_points = np.vstack((pts2d, [end_left], [start_left]))
        plt.plot([end_pt[0], start_left[0]], [end_pt[1], start_left[1]], 'r--', linewidth=1)
    
    plt.title(f"Z范围[{z_min:.2f}, {z_max:.2f}]侧剖面投影（红色标记夹角<30°）")
    plt.xlabel("径向方向 /m")
    plt.ylabel("垂直方向 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.legend()
    if show_arrows and ordered_points is not None:
        for i in range(len(pts2d) - 1):
            pt_start = pts2d[i]
            pt_end = pts2d[i+1]
            seg_vec = pt_end - pt_start
            seg_length = np.linalg.norm(seg_vec)
            if seg_length < 1e-6:
                continue
            seg_dir = seg_vec / seg_length
            mid = 0.5 * (pt_start + pt_end)
            arrow_len = seg_length * 0.3
            plt.arrow(mid[0], mid[1],
                      seg_dir[0] * arrow_len, seg_dir[1] * arrow_len,
                      color='k', head_width=0.01, head_length=0.05, linewidth=1)
    plt.show()

def draw_ordered_path_with_normal(ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max):
    """
    新增的绘图函数（基于有序路径 ordered_path）：
      - 遍历有序路径中连续的每个线段，先投影到二维平面，
      - 使用您提供的逻辑：线段方向归一化后逆时针旋转90°得到法向量；
      - 计算法向量与负 y 轴的夹角，若夹角小于30°，则该线段标红，否则使用不同颜色绘制；
      - 同时在每条线段中绘制法向量箭头。
    """
    plt.figure(figsize=(6,6))
    if ordered_path is not None and len(ordered_path) > 0:
        cmap = plt.cm.get_cmap("tab10")
        neg_y = np.array([0, -1])
        for i in range(len(ordered_path) - 1):
            p1, p2 = ordered_path[i], ordered_path[i+1]
            # 投影到二维平面
            p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
            # 绘制线段（先计算法向量）
            seg_vec = p2_2d - p1_2d
            norm_seg = np.linalg.norm(seg_vec)
            if norm_seg < 1e-6:
                continue
            seg_dir = seg_vec / norm_seg
            # 逆时针旋转90°得到法向量
            normal_2d = np.array([-seg_dir[1], seg_dir[0]])
            # 判断与负y轴的夹角
            dot_val = np.clip(np.dot(normal_2d, neg_y), -1.0, 1.0)
            angle = np.arccos(dot_val)
            seg_color = 'red' if angle < np.deg2rad(30) else cmap(i % 10)
            plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color=seg_color, linewidth=2)
            
            # 绘制法向量箭头
            mid_2d = 0.5 * (p1_2d + p2_2d)
            normal_tip = mid_2d + normal_2d * 0.5
            plt.arrow(mid_2d[0], mid_2d[1],
                      normal_tip[0]-mid_2d[0], normal_tip[1]-mid_2d[1],
                      color='k', head_width=0.05, head_length=0.1, linewidth=1)
    
    plt.title(f"Z范围[{z_min:.2f}, {z_max:.2f}]侧剖面投影")
    plt.xlabel("径向方向 /m")
    plt.ylabel("垂直方向 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()

def draw_ordered_path_with_normal(ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max):
    """
    新增的绘图函数（基于有序路径 ordered_path）：
      - 遍历有序路径中连续的每个线段，先投影到二维平面，
      - 使用您给定的逻辑：线段方向归一化后逆时针旋转90°得到法向量；
      - 计算法向量与负 y 轴的夹角，若夹角小于30°，则该线段标红，否则统一使用黑色；
      - 同时在每条线段中绘制法向量箭头。
    """
    plt.figure(figsize=(6,6))
    if ordered_path is not None and len(ordered_path) > 0:
        neg_y = np.array([0, -1])
        for i in range(len(ordered_path) - 1):
            p1, p2 = ordered_path[i], ordered_path[i+1]
            # 投影到二维平面
            p1_2d, p2_2d = project([p1, p2], plane_origin, radial_dir, vertical_dir)
            # 计算线段向量
            seg_vec = p2_2d - p1_2d
            norm_seg = np.linalg.norm(seg_vec)
            if norm_seg < 1e-6:
                continue
            seg_dir = seg_vec / norm_seg
            # 逆时针旋转90°得到法向量
            normal_2d = np.array([-seg_dir[1], seg_dir[0]])
            # 判断与负 y 轴的夹角
            dot_val = np.clip(np.dot(normal_2d, neg_y), -1.0, 1.0)
            angle = np.arccos(dot_val)
            seg_color = 'red' if angle < np.deg2rad(30) else 'black'
            plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], color=seg_color, linewidth=2)
            
            # 绘制法向量箭头
            mid_2d = 0.5 * (p1_2d + p2_2d)
            normal_tip = mid_2d + normal_2d * 0.5
            # plt.arrow(mid_2d[0], mid_2d[1],
            #           normal_tip[0]-mid_2d[0], normal_tip[1]-mid_2d[1],
            #           color='k', head_width=0.05, head_length=0.1, linewidth=1)
    
    plt.title(f"测区内某岩穴分布处侧剖面投影")
    plt.xlabel("径向方向 /m")
    plt.ylabel("垂直方向 /m")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()



def main():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 后台配置（请根据实际情况修改文件路径）
    config_path = str(get_path("arc_config"))
    mesh_path = r"E:\Production_1 (2)\Data\combined_model.glb"
    show_arrows = True  # 修改为False则不显示箭头

    # 加载配置、网格与裁剪
    center, radius, angle_range, z_min, z_max = load_config(config_path)
    mesh = trimesh.load_mesh(mesh_path)
    print("网格加载完成")
    mesh = filter_mesh(mesh, z_min, z_max)
    if mesh is None:
        print("警告：给定Z范围内无三角面！")
        return

    # 剖切平面及交线计算
    plane_origin, plane_normal = compute_plane(center, angle_range)
    near_points, intersections = slice_mesh(mesh, plane_origin, plane_normal)
    print(f"近似剖面点数: {len(near_points)}, 交线线段数: {len(intersections)}")

    ordered_path = order_segments(intersections)
    if ordered_path is None:
        print("无法构成连续路径！")

    # 定义平面坐标系：x轴沿径向，y轴为竖直方向
    mid_rad = 0.5 * sum(angle_range)
    radial_dir = np.array([np.cos(mid_rad), np.sin(mid_rad), 0])
    radial_dir /= np.linalg.norm(radial_dir)
    vertical_dir = np.array([0, 0, 1])

    # 可分别调用不同绘图函数（原有函数保持不变）
    # draw_result(intersections, ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max, show_arrows)
    # draw_result_with_normal(intersections, ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max, show_arrows)
    
    # 新增的基于有序路径的绘图函数，按照指定逻辑计算法向量并判断颜色
    draw_ordered_path_with_normal(ordered_path, plane_origin, radial_dir, vertical_dir, z_min, z_max)

if __name__ == "__main__":
    main()
