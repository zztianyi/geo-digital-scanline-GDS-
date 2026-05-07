import numpy as np
import random
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import least_squares
import json
from matplotlib.font_manager import FontProperties

# 璁剧疆瀛椾綋
font_en = FontProperties(family='Times New Roman', size=12)
font_en_13 = FontProperties(family='Times New Roman', size=13)
# 璁剧疆涓枃瀛椾綋锛堝畫浣擄級
font_zh = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
font_zh_13 = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=13)
plt.rcParams['font.size'] = 12               # 璁剧疆榛樿瀛椾綋澶у皬
plt.rcParams['axes.unicode_minus'] = False   # 姝ｅ父鏄剧ず璐熷彿
# ------------------------- 宸ュ叿鍑芥暟 -------------------------

def uniform_downsample(points, voxel_size=0.02):
    """浣跨敤 Open3D 鐨勪綋绱犳护娉㈠仛鍧囧寑閲囨牱"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.hstack([points, np.zeros((len(points),1))])[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)[:, :2]  # 杩斿洖鎶曞奖鍚庣殑 2D 鐐?


def filter_points_by_z(points, z_min=-np.inf, z_max=np.inf):
    """鏍规嵁Z杞撮珮搴﹁寖鍥磋繃婊ょ偣浜?""
    return points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]

def fit_arc_least_squares(points_2d):
    """浣跨敤鏈€灏忎簩涔樻硶鎷熷悎閮ㄥ垎鍦嗗姬"""
    def calc_residuals(params, points_2d):
        cx, cy, r = params
        distances = np.linalg.norm(points_2d - np.array([cx, cy]), axis=1)
        return distances - r
    
    # 鍒濆鐚滄祴锛氬渾蹇冪殑鍒濆浣嶇疆鏄暟鎹偣鐨勫潎鍊硷紝鍗婂緞鏄渶澶ц窛绂?
    cx_init, cy_init = np.mean(points_2d, axis=0)
    r_init = np.max(np.linalg.norm(points_2d - np.array([cx_init, cy_init]), axis=1))
    
    # 浣跨敤鏈€灏忎簩涔樻硶鎷熷悎
    result = least_squares(calc_residuals, [cx_init, cy_init, r_init], args=(points_2d,))
    return result.x[:2], result.x[2]

def get_arc_angles(points_2d, center, radius):
    """璁＄畻缁欏畾鍦嗗績鍜屽崐寰勪笅锛屾暟鎹偣鍦ㄥ渾涓婄殑璧峰鍜岀粨鏉熻搴?""
    angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
    return np.min(angles), np.max(angles)

def plot_arc(center, radius, points_2d, angle_min, angle_max):
    """缁樺埗鎷熷悎鐨勯儴鍒嗗渾寮у拰鐐逛簯"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=1, alpha=0.3, label="Points")
    
    # 缁樺埗鎷熷悎鍦嗗姬
    theta = np.linspace(angle_min, angle_max, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, '--r', label="Fitted Arc")
    
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Fitted Arc on Circle')
    ax.legend()
    plt.show()

def save_arc_to_config(center, radius, angle_min, angle_max, z_min, z_max, file_name="arc_config.json"):
    """灏嗗渾寮ф嫙鍚堢殑鍙傛暟鍜孼杞磋寖鍥翠繚瀛樺埌閰嶇疆鏂囦欢"""
    arc_data = {
        "center": center.tolist(),
        "radius": radius,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "z_range": [z_min, z_max]  # 娣诲姞Z杞磋寖鍥?
    }
    
    with open(file_name, "w") as f:
        json.dump(arc_data, f, indent=4)
    print(f"鍦嗗姬閰嶇疆宸蹭繚瀛樺埌 {file_name}")


def load_arc_from_config(file_name="arc_config.json"):
    """浠庨厤缃枃浠朵腑鍔犺浇鍦嗗姬鍙傛暟锛屽寘鎷琙杞磋寖鍥?""
    with open(file_name, "r") as f:
        arc_data = json.load(f)
    
    center = np.array(arc_data["center"])
    radius = arc_data["radius"]
    angle_min = arc_data["angle_min"]
    angle_max = arc_data["angle_max"]
    z_min, z_max = arc_data["z_range"]  # 璇诲彇Z杞磋寖鍥?

    return center, radius, angle_min, angle_max, z_min, z_max

# ------------------------- 缁樺浘鍑芥暟 -------------------------

def plot_2d_with_features(points_2d, z_range):
    """缁樺埗鍖呭惈閮ㄥ垎鍦嗗姬鎷熷悎鐨?D鐐逛簯"""

    
    center, radius = fit_arc_least_squares(points_2d)
    angle_min, angle_max = get_arc_angles(points_2d, center, radius)
    
    # 缁樺埗鎷熷悎鐨勯儴鍒嗗渾寮?
    plot_arc(center, radius, points_2d, angle_min, angle_max)
    


# ------------------------- 鐐逛簯澶勭悊 -------------------------

def downsample_points(points_2d, sample_size=10000):
    """瀵?2D 鐐逛簯鏁版嵁杩涜涓嬮噰鏍凤紝闅忔満閫夋嫨涓€瀹氭暟閲忕殑鐐?""
    if len(points_2d) > sample_size:
        indices = np.random.choice(len(points_2d), sample_size, replace=False)
        return points_2d[indices]
    return points_2d

def load_point_cloud_from_txt(file_path):
    """鍔犺浇鐐逛簯鏁版嵁"""
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0
    normals = data[:, 7:10]
    return points, colors, normals

def create_point_cloud(points, colors, normals):
    """鍒涘缓 Open3D 鐐逛簯瀵硅薄"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def statistical_outlier_removal_2d(points_2d, k=20, std_ratio=2.0):
    """缁熻瀛︽护娉?""
    if len(points_2d) < k:
        return points_2d, np.ones(len(points_2d), dtype=bool)

    nbrs = NearestNeighbors(n_neighbors=k).fit(points_2d)
    distances, _ = nbrs.kneighbors(points_2d)
    mean_dist_each_point = distances.mean(axis=1)
    
    global_mean = mean_dist_each_point.mean()
    global_std  = mean_dist_each_point.std()
    threshold   = global_mean + std_ratio * global_std

    mask = mean_dist_each_point < threshold
    return points_2d[mask], mask


def fit_line_least_squares(points_2d):
    """
    杩斿洖鐩寸嚎绯绘暟 (a, b) 浣垮緱 y = a*x + b
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    # 鏈€灏忎簩涔樿В
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

def distance_to_line(points_2d, a, b):
    """
    璁＄畻鐐瑰埌鐩寸嚎 y = a*x + b 鐨勫瀭鐩磋窛绂?
    """
    x, y = points_2d[:, 0], points_2d[:, 1]
    return np.abs(a * x - y + b) / np.sqrt(a**2 + 1)

def distance_to_arc(points_2d, center, radius):
    """
    璁＄畻鐐瑰埌鍦嗗姬锛堝渾锛夌殑寰勫悜璺濈宸?|d - r|
    """
    return np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)

# ---------- 鏂板锛氬姣旂粯鍥?----------

def project_points_to_line(points_2d, a, b):
    """
    灏嗙偣鎶曞奖鍒扮洿绾?y = a*x + b 涓婏紝杩斿洖鎵€鏈夋姇褰辩偣
    """
    x0, y0 = points_2d[:, 0], points_2d[:, 1]
    denom = a**2 + 1
    x_proj = (x0 + a * (y0 - b)) / denom
    y_proj = a * x_proj + b
    return np.stack([x_proj, y_proj], axis=1)

def compare_line_vs_arc(points_2d):
    # 鈥斺€?鐩寸嚎鎷熷悎 鈥斺€?#
    a, b = fit_line_least_squares(points_2d)
    res_line = distance_to_line(points_2d, a, b)
    rmse_line = np.sqrt(np.mean(res_line**2))

    # 鈥斺€?鍦嗗姬鎷熷悎 鈥斺€?#
    center, r = fit_arc_least_squares(points_2d)
    res_arc = distance_to_arc(points_2d, center, r)
    rmse_arc = np.sqrt(np.mean(res_arc**2))
    ang_min, ang_max = get_arc_angles(points_2d, center, r)

    # 鈥斺€?鑾峰彇鎶曞奖鍚庣殑鐩寸嚎缁樺浘鑼冨洿 鈥斺€?#
    proj_pts = project_points_to_line(points_2d, a, b)
    y_proj_min, y_proj_max = proj_pts[:, 1].min(), proj_pts[:, 1].max()
    x_proj_min = (y_proj_min - b) / a
    x_proj_max = (y_proj_max - b) / a

    # 鈥斺€?鍒涘缓骞跺垪瀛愬浘锛?涓級 鈥斺€?#
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)

    # 鈥斺€?瀛愬浘1锛氱洿绾挎嫙鍚?鈥斺€?#
    ax1 = axes[0]
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=3, alpha=0.3)
    ax1.plot([x_proj_min, x_proj_max], [y_proj_min, y_proj_max], 'r-', lw=2, label='鐩寸嚎鎷熷悎')
    # ax1.set_title(f'鐩寸嚎鎷熷悎 (RMSE={rmse_line:.3e})', fontproperties=font_zh_13)
    ax1.set_xlabel('X(m)', fontproperties=font_en_13)
    ax1.set_ylabel('Y(m)', fontproperties=font_en_13)
    ax1.set_aspect('equal')
    ax1.legend(prop=font_zh_13)

    # 鈥斺€?瀛愬浘2锛氬渾寮ф嫙鍚?鈥斺€?#
    ax2 = axes[1]
    ax2.scatter(points_2d[:, 0], points_2d[:, 1], s=3, alpha=0.3)
    theta = np.linspace(ang_min, ang_max, 300)
    arc_x = center[0] + r * np.cos(theta)
    arc_y = center[1] + r * np.sin(theta)
    ax2.plot(arc_x, arc_y, 'r--', lw=2, label='鍦嗗姬鎷熷悎')
    # ax2.set_title(f'鍦嗗姬鎷熷悎 (RMSE={rmse_arc:.3e})', fontproperties=font_zh_13)
    ax2.set_xlabel('X(m)', fontproperties=font_en_13)
    ax2.set_ylabel('Y(m)', fontproperties=font_en_13)
    ax2.set_aspect('equal')
    ax2.legend(prop=font_zh_13)

    # 鈥斺€?瀛愬浘3锛氭畫宸洿鏂瑰浘 鈥斺€?#
    ax3 = axes[2]
    ax3.hist(res_line, bins=50, alpha=0.6, label=f'RMSE={rmse_line:.1f}')
    ax3.hist(res_arc, bins=50, alpha=0.6, label=f'RMSE={rmse_arc:.1f}')
    ax3.set_xlabel('娈嬪樊(m)', fontproperties=font_zh_13)
    ax3.set_ylabel('棰戞暟(涓?', fontproperties=font_zh_13)
    # ax3.set_title('娈嬪樊鍒嗗竷鐩存柟鍥?, fontproperties=font_zh_13)
    ax3.legend(prop=font_en_13)

    plt.tight_layout(w_pad=1.0)
    plt.show()
# ------------------------- 涓荤▼搴?-------------------------
if __name__ == "__main__":
    file_path = r"data/examples/sample_points.txt"
    
    # 1. 璇诲彇鐐逛簯
    points, colors, normals = load_point_cloud_from_txt(file_path)
    
    # 2. 鏍规嵁Z杞磋寖鍥存彁鍙栫偣浜戯紙绀轰緥锛氬彇涓棿1/3楂樺害鑼冨洿锛?
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    z_max = 2*(z_max - z_min)/5+z_min
    z_min = (z_max - z_min)/5+z_min
    filtered_points = filter_points_by_z(points, z_min, z_max)
    
    # 3. 鑾峰彇 2D 鎶曞奖 (X-Y)
    points_2d = filtered_points[:, :2]
    
    # 4. 涓嬮噰鏍?
    downsampled = downsample_points(points_2d, 20000)
    

    
    # 6. 鎷熷悎骞跺彲瑙嗗寲
    # plot_2d_with_features(cleaned_points, (z_min, z_max))

    cleaned_points = uniform_downsample(downsampled, voxel_size=1)
    compare_line_vs_arc(cleaned_points)
    # 7. 淇濆瓨閰嶇疆鏂囦欢锛堟坊鍔燴杞磋寖鍥达級
    center, radius = fit_arc_least_squares(cleaned_points)
    angle_min, angle_max = get_arc_angles(cleaned_points, center, radius)
    save_arc_to_config(center, radius, angle_min, angle_max, z_min, z_max, file_name=r"configs/arc_config.example.json")
