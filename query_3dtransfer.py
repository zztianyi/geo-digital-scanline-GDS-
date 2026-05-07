import json
import numpy as np

def load_arc_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    center_2d = np.array(config["center"], dtype=float)
    center_3d = np.append(center_2d, 0.0)
    radius = float(config["radius"])
    angle_min = config["angle_min"]
    angle_max = config["angle_max"]
    arc_length = radius * (angle_max - angle_min)
    arc_length_range = [0, arc_length]
    z_min, z_max = config["z_range"]
    return center_3d, radius, angle_min, angle_max, arc_length_range, z_min, z_max

def compute_all_slice_params(center, radius, angle_min, arc_length_range, step=0.01):
    """
    鎵归噺璁＄畻鎵€鏈夋祴骞抽潰鐨勬硶鍚戦噺銆佸緞鍚戞柟鍚戝拰鏂逛綅瑙?
    """
    positions = np.arange(0, arc_length_range[1], step)  # shape: (N,)
    angles = angle_min + positions / radius               # shape: (N,)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    
    radial_dirs = np.stack([cos_vals, sin_vals, np.zeros_like(cos_vals)], axis=1)  # shape: (N,3)
    radial_dirs /= np.linalg.norm(radial_dirs, axis=1, keepdims=True)

    normals = np.cross(radial_dirs, np.array([[0, 0, 1]]))  # shape: (N,3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    azimuths = (np.degrees(np.arctan2(radial_dirs[:, 0], radial_dirs[:, 1])) % 360).round(3)
    origins = np.repeat(center.reshape(1, 3), len(positions), axis=0)

    return positions, origins, normals, radial_dirs, azimuths

def find_nearest_plane(query_point, origins, normals, positions, radial_dirs, azimuths):
    """
    鐭㈤噺鏂瑰紡璁＄畻 query_point 鍒版墍鏈夊钩闈㈢殑璺濈
    """
    vectors = query_point - origins              # shape: (N,3)
    distances = np.abs(np.einsum("ij,ij->i", vectors, normals))  # 鐐瑰埌骞抽潰璺濈锛宻hape: (N,)

    idx = np.argmin(distances)
    return {
        "index": idx,
        "arc_pos": positions[idx],
        "distance": distances[idx],
        "origin": origins[idx],
        "normal": normals[idx],
        "radial_dir": radial_dirs[idx],
        "azimuth": azimuths[idx]
    }

def main():
    config_path = r"configs/arc_config.example.json"
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)

    query_point = np.array([-32.69, -26.52,0])

    # 涓€娆℃€ц绠楁墍鏈夋祴闈㈠弬鏁?
    positions, origins, normals, radial_dirs, azimuths = compute_all_slice_params(
        center, radius, angle_min, arc_length_range, step=0.05
    )

    # 鏌ユ壘鏈€杩戞祴闈?
    result = find_nearest_plane(query_point, origins, normals, positions, radial_dirs, azimuths)

    key = f"{result['arc_pos']:.2f}"
    print(f"鏈€杩戞祴骞抽潰浣嶇疆 {key}锛堝姬闀垮潗鏍囷級")
    print(f"璺濈鏌ヨ鐐瑰瀭鐩磋窛绂讳负锛歿result['distance']:.3f} m")
    print(f"骞抽潰鍘熺偣锛歿result['origin']}")
    print(f"娉曞悜閲忥細{result['normal']}")
    print(f"寰勫悜鏂瑰悜锛歿result['radial_dir']}")
    print(f"鏂逛綅瑙掞紙椤烘椂閽堢浉瀵逛簬姝ｅ寳锛夛細{result['azimuth']}掳")

if __name__ == "__main__":
    main()

