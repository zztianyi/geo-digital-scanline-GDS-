import pickle
import numpy as np
import json
import trimesh
import pyvista as pv
import joblib
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import laspy

def load_trimesh_with_texture(obj_path):
    scene = trimesh.load(obj_path, force='scene')
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("OBJ 鍔犺浇澶辫触")
    texture_img = None
    all_meshes = []
    for _, geom in scene.geometry.items():
        if isinstance(geom.visual, trimesh.visual.texture.TextureVisuals):
            if hasattr(geom.visual.material, 'image') and geom.visual.material.image is not None:
                texture_img = geom.visual.material.image
        all_meshes.append(geom)
    mesh_combined = trimesh.util.concatenate(all_meshes)
    return mesh_combined, texture_img


def load_arc_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config["z_range"]


def filter_mesh_by_z_range(mesh, z_min, z_max):
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if submeshes else None


def trimesh_to_pyvista_with_visuals(mesh, texture_image=None):
    faces = np.hstack([[3, *face] for face in mesh.faces])
    mesh_pv = pv.PolyData(mesh.vertices, faces)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = mesh.visual.vertex_colors[:, :3]
        mesh_pv.point_data["Colors"] = vc / 255.0
        return mesh_pv, None
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and texture_image is not None:
        mesh.visual.uv[:, 1] = 1.0 - mesh.visual.uv[:, 1]
        mesh_pv.active_t_coords = mesh.visual.uv
        texture = pv.numpy_to_texture(texture_image[:, :, :3])
        return mesh_pv, texture
    return mesh_pv, None
def downsample_point_cloud(points, max_samples=50000):
    if len(points) <= max_samples:
        return points
    idx = np.random.choice(len(points), max_samples, replace=False)
    return points[idx]
# 骞惰 KDE 鎵ц鍑芥暟
def kde_score_batch(serialized_kde, points):
    kde = pickle.loads(serialized_kde)
    return np.exp(kde.score_samples(points))


def estimate_kde_parallel(points, kde_model, batch_size=1, num_workers=15):
    kde_serialized = pickle.dumps(kde_model)
    batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(kde_score_batch, kde_serialized, b) for b in batches]
        for f in tqdm(futures, desc="KDE 骞惰浼板€?):
            results.append(f.result())
    return np.concatenate(results)


def main():
    # === 璺緞璁剧疆 ===
    # mesh_path = r"data/private/raw_model\combined_model.obj"
    mesh_path = r"data/private/raw_model\Merged mesh_seg.obj"

    config_path = r"configs/arc_config.example.json"
    voxel_pkl = r"data/private/raw_model\voxels_sparse_buchang.pkl"
    kde_model_path = r"data/private/raw_model\kde_model.pkl"
    kde_result_path = r"data/private/raw_model\kde_batch_density_5.0_0_0.1.pkl"

    # === 鍔犺浇缃戞牸 & 杩囨护 ===
    # z_min, z_max = load_arc_config(config_path)
    mesh_trimesh, texture_img = load_trimesh_with_texture(mesh_path)
    # mesh_filtered = filter_mesh_by_z_range(mesh_trimesh, z_min, z_max)
    # if mesh_filtered is None:
    #     print("鏈彁鍙栧埌绗﹀悎 Z 鑼冨洿鐨勭綉鏍?)
    #     return

    # === 鍔犺浇鐐逛簯鏍锋湰 ===
    # with open(voxel_pkl, "rb") as f:
    #     voxel_data = pickle.load(f)
    # voxel_indices = np.array(voxel_data["active_voxels"])
    # origin = np.array(voxel_data["origin"])
    # voxel_size = voxel_data["voxel_size"]
    # point_cloud = (voxel_indices + 0.5) * voxel_size + origin

    centroid_las_path = r"data/private/raw_model\cluster_centroids.las"
    centroid_las_path = r"data/private/raw_model\cluster_centroids_filtered_0_0.1.las"
    # centroid_las_path = r"data/private/raw_model\cluster_centroids_1.las"
    # centroid_las_path = r"data/private/raw_model\cluster_centroids_0.las"
    
    # 璇诲彇 LAS 鏂囦欢
    las = laspy.read(centroid_las_path)

    # 鎻愬彇鍧愭爣 (N, 3)
    point_cloud = np.vstack((las.x, las.y, las.z)).T

    # 鎶界█鐐逛簯
    point_cloud_down = downsample_point_cloud(point_cloud, max_samples=50000)
    print(f"鍘熷鐐规暟: {len(point_cloud)}, 鎶界█鍚庣偣鏁? {len(point_cloud_down)}")

    # === KDE 鎷熷悎骞朵繚瀛樻ā鍨?===
    kde = KernelDensity(kernel='gaussian', bandwidth=5.0)
    kde.fit(point_cloud_down)
    # joblib.dump(kde, kde_model_path)

    # === KDE 骞惰浼板€?===
    vertex_coords = mesh_trimesh.vertices
    vertex_density = estimate_kde_parallel(vertex_coords, kde, batch_size=50, num_workers=12)

    # === 淇濆瓨浼板€肩粨鏋?===
    with open(kde_result_path, "wb") as f:
        pickle.dump({"face_centroids": vertex_coords, "vertex_density": vertex_density}, f)
    print(f"KDE 妯″瀷鍜岀粨鏋滀繚瀛樹簬锛歕n{kde_model_path}\n{kde_result_path}")



if __name__ == "__main__":
    main()

