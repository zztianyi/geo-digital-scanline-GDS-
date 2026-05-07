import json
import numpy as np
import open3d as o3d

def load_arc_from_config(file_name="arc_config.json"):
    """浠庨厤缃枃浠朵腑鍔犺浇鍦嗗姬鐨勫弬鏁?""
    with open(file_name, "r") as f:
        arc_data = json.load(f)
    
    # 璇诲彇鍦嗗績銆佸崐寰勫拰瑙掑害
    center = np.array(arc_data["center"])
    radius = arc_data["radius"]
    angle_min = arc_data["angle_min"]
    angle_max = arc_data["angle_max"]
    
    return center, radius, angle_min, angle_max


def load_point_cloud_from_txt(file_path):
    """鍔犺浇鐐逛簯鏁版嵁"""
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0  # 姝ｅ父鍖栭鑹插€?
    return points, colors


def create_3d_grid_on_arc(center, radius, angle_min, angle_max, points, colors, num_latitude_lines, num_longitude_lines):
    """鍦ㄧ粰瀹氬渾寮т笂鐢熸垚缃戞牸锛屽苟鍦?D绌洪棿涓粯鍒跺嚭鏉?""
    # 鑾峰彇鐐逛簯鐨剒杞磋寖鍥?
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    
    # 鐢熸垚绾嚎锛氭瘡鏉＄含绾垮搴斾笉鍚岀殑z鍊硷紝骞跺钩琛屼簬xy骞抽潰
    z_vals = np.linspace(z_min, z_max, num_latitude_lines)  # 绾嚎鐨勪笉鍚岄珮搴?
    latitudes = []
    
    for z in z_vals:
        angles = np.linspace(angle_min, angle_max, num_longitude_lines)  # 鍧囧寑鍒嗗竷鐨勭粡绾胯搴?
        x_vals = center[0] + radius * np.cos(angles)
        y_vals = center[1] + radius * np.sin(angles)
        latitudes.append(np.column_stack([x_vals, y_vals, np.full_like(x_vals, z)]))  # 淇濆瓨姣忎釜绾嚎鐨勭偣

    # 鐢熸垚缁忕嚎锛氳繖浜涙槸骞宠浜巣杞寸殑绾?
    longitudes = []
    for angle in np.linspace(angle_min, angle_max, num_longitude_lines):
        x_vals = center[0] + radius * np.cos(angle)
        y_vals = center[1] + radius * np.sin(angle)
        z_vals = np.linspace(z_min, z_max, num_latitude_lines)  # 楂樺害浠庢渶灏忓埌鏈€澶鍊?
        longitudes.append(np.column_stack([np.full_like(z_vals, x_vals), np.full_like(z_vals, y_vals), z_vals]))  # 淇濆瓨姣忔潯缁忕嚎鐨勭偣

    # 浣跨敤Open3D缁樺埗3D缃戞牸鍜屽渾寮?
    # 鍒涘缓鐐逛簯瀵硅薄
    pcd_points = o3d.geometry.PointCloud()
    pcd_points.points = o3d.utility.Vector3dVector(points)
    pcd_points.colors = o3d.utility.Vector3dVector(colors)  # 浣跨敤瀵煎叆鐨勯鑹?

    # 鍒涘缓 LineSet 瀵硅薄锛屾坊鍔犳墍鏈夌含绾?
    lat_line_sets = []
    for latitude in latitudes:
        lat_line_lines = o3d.geometry.LineSet()
        lat_line_lines.points = o3d.utility.Vector3dVector(latitude)
        lat_line_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(latitude) - 1)])
        lat_line_lines.paint_uniform_color([1, 0, 0])  # 绾㈣壊鏄剧ず绾嚎
        lat_line_sets.append(lat_line_lines)

    # 鍒涘缓 LineSet 瀵硅薄锛屾坊鍔犳墍鏈夌粡绾?
    lon_line_sets = []
    for longitude in longitudes:
        lon_line_lines = o3d.geometry.LineSet()
        lon_line_lines.points = o3d.utility.Vector3dVector(longitude)
        lon_line_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(longitude) - 1)])
        lon_line_lines.paint_uniform_color([0, 1, 0])  # 缁胯壊鏄剧ず缁忕嚎
        lon_line_sets.append(lon_line_lines)

    # 鍙鍖栨墍鏈夊璞?
    o3d.visualization.draw_geometries([pcd_points] + lat_line_sets + lon_line_sets)

def save_grid_equations_to_config(center, radius, angle_min, angle_max, file_name="grid_equations_config.json"):
    """淇濆瓨鐢熸垚缁忕含绾跨殑鏂圭▼鍒伴厤缃枃浠?""
    grid_data = {
        "center": center.tolist(),
        "radius": radius,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "latitude_equation": "x = center[0] + radius * np.cos(angle), y = center[1] + radius * np.sin(angle), z = z_value",
        "longitude_equation": "x = center[0] + radius * np.cos(angle), y = center[1] + radius * np.sin(angle), z = z_value"
    }
    
    with open(file_name, "w") as f:
        json.dump(grid_data, f, indent=4)

# ------------------------- 涓荤▼搴?-------------------------

if __name__ == "__main__":
    # 1. 璇诲彇鐐逛簯鏁版嵁
    file_path = r"data/examples/sample_points.txt"
    points, colors = load_point_cloud_from_txt(file_path)
    
    # 2. 鍔犺浇鍦嗗姬閰嶇疆
    center, radius, angle_min, angle_max = load_arc_from_config(file_name=r"configs/arc_config.example.json")
    
    # 3. 鍦?D绌洪棿缁樺埗缃戞牸鍜屽渾寮?
    num_latitude_lines = 10  # 绾嚎鐨勬暟閲?
    num_longitude_lines = 12  # 缁忕嚎鐨勬暟閲?
    create_3d_grid_on_arc(center, radius, angle_min, angle_max, points, colors, num_latitude_lines, num_longitude_lines)


