import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class PointCloudProcessor:
    def __init__(self, arc_config_file, point_cloud_file):
        self.center = self.radius = self.angle_min = self.angle_max = None
        self.load_arc_from_config(arc_config_file)
        self.points = self.load_point_cloud_from_txt(point_cloud_file)
    
    def load_arc_from_config(self, file_name="arc_config.json"):
        """浠庨厤缃枃浠朵腑鍔犺浇鍦嗗姬鐨勫弬鏁?""
        with open(file_name, "r") as f:
            arc_data = json.load(f)
        self.center = np.array(arc_data["center"])  # 杩欓噷鏄簩缁寸殑center
        self.radius = arc_data["radius"]
        self.angle_min = arc_data["angle_min"]  # 寮у害鍒?
        self.angle_max = arc_data["angle_max"]

    def load_point_cloud_from_txt(self, file_path):
        """鍔犺浇鐐逛簯鏁版嵁"""
        data = np.loadtxt(file_path)
        points = data[:, :3]
        return points
    
    def get_3d_line(self, angle_percent):
        """鐢熸垚涓夌淮鐩寸嚎鍙傛暟"""
        theta = (self.angle_min + (self.angle_max - self.angle_min) * angle_percent / 100)
        dir_2d = np.array([np.cos(theta), np.sin(theta)])
        point_2d = self.center[:2] + self.radius * dir_2d
        return np.append(point_2d, 0), np.append(dir_2d, 0)  # [x, y, 0], [dx, dy, 0]

    def extract_near_points(self, points, line_point, line_dir, max_dist=0.1):
        """鎻愬彇涓夌淮閭昏繎鐐?""
        dist = self.distance_point_to_line_3d(points, line_point, line_dir)
        return points[dist <= max_dist]

    def distance_point_to_line_3d(self, points, line_point, line_dir):
        """涓夌淮绌洪棿鎶曞奖鐐瑰埌鐩寸嚎鐨勮窛绂?""
        vec = points - line_point
        # 灏?vec 鐨?z 鍒嗛噺缃负 0,杩欐槸璁＄畻鍦▁y骞抽潰涓婄殑鎶曞奖
        vec[:, 2] = 0  # 鐩存帴灏嗘墍鏈夌偣鐨?z 鍒嗛噺璁句负 0
        cross = np.cross(vec, line_dir)
        return np.linalg.norm(cross, axis=1) / np.linalg.norm(line_dir)

    def project_to_local_coords(self, proj_points, line_point, line_dir):
        """灏嗘姇褰辩偣杞崲鍒板钩闈㈠眬閮ㄥ潗鏍囩郴"""
        u = line_dir / np.linalg.norm(line_dir)  # 娌跨洿绾挎柟鍚?
        v = np.array([0, 0, 1])  # 鍋囪鍨傜洿鏂瑰悜
        w = np.cross(u, v)
        return np.dot(proj_points, np.vstack([u, v]).T)

    def project_points_to_plane(self, points, line_direction_3d, line_point_3d):
        """鎶曞奖鍒扮壒瀹氬钩闈?""
        # 纭繚 line_direction_3d 鍦▁y骞抽潰鍐?
        line_direction_2d = line_direction_3d[:2]  # 鍙彇x鍜寉鍒嗛噺
        # 璁＄畻涓庣洿绾挎浜ょ殑娉曞悜閲?(xy骞抽潰)
        normal = np.array([-line_direction_2d[1], line_direction_2d[0], 0])  # 鍨傜洿浜巐ine_direction_2d鐨勫悜閲?
        # 褰掍竴鍖栨硶鍚戦噺
        normal = normal / np.linalg.norm(normal)
        vec_to_point = points - line_point_3d
        proj_length = np.dot(vec_to_point, normal)
        return points - proj_length[:, None] * normal

    def generate_arc_points(self, num_points=100):
        """鐢熸垚鍦嗗姬涓婄殑浜岀淮鐐圭敤浜庡彲瑙嗗寲"""
        angles = np.linspace(self.angle_min, self.angle_max, num_points)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        return np.column_stack((x, y))

    def downsample_points(self, points_2d, sample_size=10000):
        """瀵?2D 鐐逛簯鏁版嵁杩涜涓嬮噰鏍凤紝闅忔満閫夋嫨涓€瀹氭暟閲忕殑鐐?""
        if len(points_2d) > sample_size:
            indices = np.random.choice(len(points_2d), sample_size, replace=False)
            return points_2d[indices]
        return points_2d

    def process_warp(self, angle_percent, max_dist=0.01):
        """澶勭悊骞惰幏鍙栧眬閮ㄤ簩缁村潗鏍?""
        line_3d_point, line_3d_dir = self.get_3d_line(angle_percent)
        arc_points = self.generate_arc_points()
        near_points = self.extract_near_points(self.points, line_3d_point, line_3d_dir, max_dist)
        
        # 鎶曞奖鍒版硶骞抽潰骞惰浆鎹㈠潗鏍?
        proj_3d = self.project_points_to_plane(near_points, line_3d_dir, line_3d_point)
        
        # 骞崇Щ鏍℃
        proj_3d_ = proj_3d - line_3d_point
        proj_3d_[:, 2] -= np.min(self.points[:, 2])

        local_2d = self.project_to_local_coords(proj_3d_, line_3d_point, line_3d_dir)

        return local_2d
    


# ------------------------- 涓荤▼搴?-------------------------
if __name__ == "__main__":
    # 1. 鍒濆鍖?PointCloudProcessor
    processor = PointCloudProcessor(
        arc_config_file=r"configs/arc_config.example.json", 
        point_cloud_file=r"data/examples/sample_points.txt"
    )
    
    # 2. 澶勭悊鐐逛簯锛岃緭鍑哄眬閮ㄥ潗鏍?
    local_2d = processor.process_warp(angle_percent=32)

    # 3. 鍙鍖栫粨鏋?
    plt.scatter(local_2d[:, 0], local_2d[:, 1], s=1)
    plt.axis('equal')
    plt.ticklabel_format(style='plain', axis='both')
    plt.show()

    # proj_3d = downsample_points(proj_3d, sample_size=1000)

    # # 3. 缁樺埗涓夌淮鎶曞奖鐐?
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # 缁樺埗鎶曞奖鍚庣殑涓夌淮鐐?
    # ax.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2], c='g', s=5)

    # # 璁剧疆鏍囬鍜屾爣绛?
    # ax.set_title("3D Projection of Points")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    # plt.show()
