import numpy as np
import open3d as o3d

# 1. 鍔犺浇鏁版嵁锛堜繚鎸佸師鏍凤級
def load_point_cloud_from_txt(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0
    normals = data[:, 7:10]
    return points, colors, normals

# 2. 鍒涘缓鐐逛簯瀵硅薄锛堜繚鎸佸師鏍凤級
def create_point_cloud(points, colors, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

# 3. 杞崲涓轰綋绱犵綉鏍硷紙淇濇寔鍘熸牱锛?
def point_cloud_to_voxel(pcd, voxel_size=0.5):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    print(f"浣撶礌缃戞牸宸茬敓鎴愶紝浣撶礌澶у皬: {voxel_size}")
    return voxel_grid

# 4. 淇敼鍚庣殑鍙鍖栧嚱鏁?
def visualize_with_custom_view(voxel_grid):
    # 鍒涘缓鍙鍖栫獥鍙?
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Voxel Grid with Coordinate Axis")
    
    # 娣诲姞浣撶礌缃戞牸
    vis.add_geometry(voxel_grid)
    
    # # 娣诲姞鍧愭爣杞达紙灏哄璁剧疆涓?.0锛屽彲鏍规嵁闇€瑕佽皟鏁达級
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.0, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    
    # 鑾峰彇瑙嗗浘鎺у埗鍣?
    view_ctl = vis.get_view_control()
    
    # 璁剧疆鍒濆瑙嗚鍙傛暟
    view_ctl.set_front([-1, 0, 0])  # 鏈濆悜Y杞磋礋鏂瑰悜锛堟瀵筙Z骞抽潰锛?
    view_ctl.set_up([0, 0, 1])      # Z杞村悜涓?

    
    # 鑷姩璋冩暣瑙嗚浠ユ樉绀哄畬鏁村唴瀹?
    vis.get_render_option().point_size = 1.0  # 鍙€夛細璋冩暣鐐逛簯鏄剧ず灏哄
    vis.run()
    vis.destroy_window()

# 涓荤▼搴忥紙淇濇寔鍘熸牱锛?
if __name__ == "__main__":
    file_path = r"data/examples/sample_points.txt"
    points, colors, normals = load_point_cloud_from_txt(file_path)
    pcd = create_point_cloud(points, colors, normals)
    voxel_size = 0.2
    voxel_grid = point_cloud_to_voxel(pcd, voxel_size)
    visualize_with_custom_view(voxel_grid)
