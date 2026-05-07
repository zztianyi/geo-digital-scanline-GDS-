import numpy as np
import open3d as o3d
import random
import matplotlib.pyplot as plt

# 1. 鍔犺浇鏁版嵁
def load_point_cloud_from_txt(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0
    normals = data[:, 7:10]
    return points, colors, normals

# 2. 涓嬮噰鏍风偣浜?
def downsample_points(points, sample_size=100000):
    """
    瀵圭偣浜戞暟鎹繘琛岄殢鏈烘娊鏍?
    :param points: 鍘熷鐐逛簯鏁版嵁
    :param sample_size: 鎶芥牱鐐规暟
    :return: 鎶芥牱鍚庣殑鐐逛簯鏁版嵁
    """
    if len(points) > sample_size:
        sampled_indices = random.sample(range(len(points)), sample_size)
        points = points[sampled_indices]
    return points

# 3. 璁＄畻鍦嗗績
def calculate_center_by_z_quantile(points, quantile=0.95):
    """
    鏍规嵁 z 鍊肩殑鍒嗕綅鏁拌绠楀渾蹇?
    :param points: 涓夌淮鐐逛簯鏁版嵁
    :param quantile: z 鍊煎垎浣嶆暟锛堥粯璁や负 95%锛?
    :return: 鍦嗗績鐨勪笁缁村潗鏍? 楂?z 鍊肩殑鐐归泦
    """
    z_values = points[:, 2]  # 鎻愬彇 z 鍊?
    threshold = np.quantile(z_values, quantile)  # 璁＄畻 95% 鍒嗕綅鏁伴槇鍊?
    high_z_points = points[z_values >= threshold]  # 绛涢€?z 鍊煎ぇ浜庨槇鍊肩殑鐐归泦
    center = np.mean(high_z_points, axis=0)  # 璁＄畻楂?z 鍊肩偣闆嗙殑涓績鐐?
    print(f"鍦嗗績浣嶇疆锛堥€氳繃 z 鍊?95% 鍒嗕綅鏁扮‘瀹氾級: {center}")
    return center, high_z_points

# 4. 鍙鍖栨姇褰辩偣鍜岄珮 z 鍊肩偣闆?
def plot_projected_points_with_high_z(points, high_z_points, center, padding=0.1):
    """
    缁樺埗鎶曞奖鐐广€侀珮 z 鍊肩偣闆嗗拰鍦嗗績浣嶇疆
    :param points: 涓夌淮鐐逛簯鏁版嵁
    :param high_z_points: z 鍊煎ぇ浜?95% 鍒嗕綅鏁扮殑鐐归泦
    :param center: 鍦嗗績鐨勪笁缁村潗鏍?
    :param padding: 鍥惧儚鑼冨洿杈硅窛姣斾緥
    """
    # 鎶曞奖鍒?XY 骞抽潰
    xy_projection = points[:, :2]  # 鎵€鏈夌偣鐨勬姇褰?
    high_z_projection = high_z_points[:, :2]  # 楂?z 鍊肩偣鐨勬姇褰?
    center_xy = center[:2]  # 鍦嗗績鐨?XY 鍧愭爣

    # 璁剧疆鏄剧ず鑼冨洿
    x_min, x_max = np.min(xy_projection[:, 0]), np.max(xy_projection[:, 0])
    y_min, y_max = np.min(xy_projection[:, 1]), np.max(xy_projection[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    # 鍒涘缓缁樺浘
    plt.figure(figsize=(8, 8))
    plt.scatter(xy_projection[:, 0], xy_projection[:, 1], s=1, label="Sampled Points", alpha=0.5, color="gray")
    plt.scatter(high_z_projection[:, 0], high_z_projection[:, 1], s=1, label="High Z Points", alpha=0.8, color="red")
    plt.scatter(center_xy[0], center_xy[1], color="blue", label="Center", s=100)

    # 璁剧疆鍥惧儚鑼冨洿鍜屾爣棰?
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("equal")
    plt.title("Projected Points with High Z Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# 涓诲嚱鏁?
if __name__ == "__main__":
    file_path = r"data/examples/sample_points.txt"#"data/examples/sample_points.txt"

    # 鍔犺浇鐐逛簯鏁版嵁
    points, colors, normals = load_point_cloud_from_txt(file_path)

    # 涓嬮噰鏍风偣浜?
    sampled_points = downsample_points(points, sample_size=100000)

    # 璁＄畻鍦嗗績鍜岄珮 z 鍊肩偣闆?
    center, high_z_points = calculate_center_by_z_quantile(sampled_points, quantile=0.15)

    # 鍙鍖栨姇褰辩偣鍜岄珮 z 鍊肩偣闆?
    plot_projected_points_with_high_z(sampled_points, high_z_points, center)

