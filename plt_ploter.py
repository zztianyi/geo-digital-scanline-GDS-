import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.tri import Triangulation
from collections import Counter
import pyvista as pv

# ========== 1. 鍔犺浇绋€鐤忕偣浜戞暟鎹?==========
voxel_pkl = r"data/private/raw_model\voxels_sparse_buchang.pkl"
with open(voxel_pkl, "rb") as f:
    voxel_data = pickle.load(f)

voxel_indices = np.array(voxel_data["active_voxels"])
origin = np.array(voxel_data["origin"])
voxel_size = voxel_data["voxel_size"]
# 鎵€鏈夌偣浜戝潗鏍?(N, 3)
points = (voxel_indices + 0.5) * voxel_size + origin

print(f"[鉁揮 鍔犺浇鐐逛簯瀹屾垚锛屽叡 {len(points)} 涓偣")

# ========== 2. 鎶界█閲囨牱 & DBSCAN 鑱氱被 ==========
sample_ratio = 0.1  # 渚嬪鍙?10% 鐨勭偣杩涜鑱氱被
sample_indices = np.random.choice(len(points), size=int(len(points) * sample_ratio), replace=False)
sampled_points = points[sample_indices]

dbscan = DBSCAN(eps=0.2, min_samples=5)
sample_labels = dbscan.fit_predict(sampled_points)

num_clusters = len(set(sample_labels)) - (1 if -1 in sample_labels else 0)
print(f"[鉁揮 鑱氱被瀹屾垚锛岀皣鏁? {num_clusters}")

# ========== 3. 鍒╃敤 KDTree 灏嗚仛绫绘爣绛炬槧灏勫洖鎵€鏈夊師濮嬬偣 ==========
tree = cKDTree(sampled_points)
_, idx = tree.query(points, k=1)
labels_full = sample_labels[idx]  # 姣忎釜鍘熷鐐瑰搴旂殑鑱氱被鏍囩

# ========== 4. 缁熻鏈夋晥鑱氱被淇℃伅 ==========
voxel_volume = 0.05 ** 3  # 姣忎釜浣撶礌浣撶Н锛堝崟浣嶏細m鲁锛?
# 鍙粺璁℃湁鏁堣仛绫伙紙涓嶅惈鍣偣锛屾爣绛?!= -1锛?
label_counter = Counter(labels_full[labels_full != -1])

# 鏋勯€犺仛绫讳俊鎭垪琛紝鏍煎紡涓?(鑱氱被缂栧彿, 鐐规暟, 浣撶Н, 璐ㄥ績)
cluster_info = []
for label in label_counter:
    count = label_counter[label]
    vol = count * voxel_volume
    cluster_points = points[labels_full == label]
    centroid = cluster_points.mean(axis=0)
    cluster_info.append((label, count, vol, centroid))

# 鎸変綋绉檷搴忔帓搴?
sorted_clusters = sorted(cluster_info, key=lambda x: x[2], reverse=True)

print("\n[鉁揮 鍚勬湁鏁堣仛绫讳俊鎭紙鎸変綋绉檷搴忔帓搴忥級锛?)
for i, (label, count, vol, centroid) in enumerate(sorted_clusters, start=1):
    print(f"  鎮┖浣搟i}: Label={label:<3} 鐐规暟={count:<6} 浣撶Н={vol:.4f} m鲁")

# ========== 5. 鍔犺浇缃戞牸鏂囦欢 ==========
mesh_path = r"data/private/raw_model\Merged mesh_seg.obj"
mesh = pv.read(mesh_path)
# 鎻愬彇缃戞牸椤剁偣浠ュ強闈㈡暟鎹?
mesh_points = mesh.points  # (M, 3)
# 鍋囧畾鏄笁瑙掑舰缃戞牸锛宖aces 鏁扮粍鏍煎紡锛歔3, i1, i2, i3, 3, ...]
faces = mesh.faces.reshape(-1, 4)[:, 1:4]  # 鍙栨瘡涓潰鍓嶄笁涓偣鐨勭储寮?

# ========== 6. 瀹氫箟鍑芥暟锛氱粯鍒跺潡浣?==========
def plot_block(block_index, sorted_clusters, points, labels, mesh_points, faces):
    """
    缁樺埗涓や釜瀛愬浘锛?
      宸﹀浘锛氭樉绀虹綉鏍兼枃浠跺湪 x-y 骞抽潰鐨勬姇褰憋紙閫氳繃涓夎鍓栧垎缁樺埗缃戞牸杞粨锛夈€?
      鍙冲浘锛氭樉绀洪€夊畾鑱氱被锛堝潡浣擄級鐨勭偣鍦?x-y 鎶曞奖涓紝骞跺湪鏍囬涓樉绀哄潡浣撳簭鍙峰拰浣撶Н锛?
             鏍囬浣跨敤鏂扮綏椹瓧浣擄紝鐧借壊锛屽瓧鍙?6銆?
    
    鍙傛暟锛?
      block_index: 鐩爣鍧椾綋鐨勫簭鍙凤紙1-indexed锛? 琛ㄧず浣撶Н鏈€澶х殑鍧椾綋锛夈€?
      sorted_clusters: 鎸変綋绉檷搴忔帓搴忕殑鑱氱被淇℃伅鍒楄〃锛屾牸寮忎负 (label, count, vol, centroid)銆?
      points: 鎵€鏈夌偣浜戝潗鏍?(N, 3)銆?
      labels: 姣忎釜鐐瑰搴旂殑鑱氱被鏍囩锛堜笌 points 涓€涓€瀵瑰簲锛夈€?
      mesh_points: 缃戞牸鏂囦欢鐨勬墍鏈夐《鐐瑰潗鏍?(M, 3)銆?
      faces: 缃戞牸鐨勪笁瑙掗潰鏁版嵁锛屽舰鐘朵负 (K, 3)銆?
    """
    # 妫€鏌ュ簭鍙峰悎娉曟€?
    if block_index < 1 or block_index > len(sorted_clusters):
        print("鍧椾綋搴忓彿瓒呭嚭鑼冨洿锛?)
        return

    # 閫夊彇鐩爣鑱氱被淇℃伅
    target_cluster = sorted_clusters[block_index - 1]
    target_label, count, vol, centroid = target_cluster
    # 杩囨护鍑鸿鑱氱被鐨勭偣
    block_points = points[labels == target_label]

    # 鍦?x-y 骞抽潰鎶曞奖锛堝拷鐣?z 鍧愭爣锛?
    # 缃戞牸鎶曞奖
    mesh_x = mesh_points[:, 0]
    mesh_y = mesh_points[:, 1]
    tri = Triangulation(mesh_x, mesh_y, triangles=faces)

    # 鎵€鏈夌偣浜戞姇褰憋紙鍙€夛紝鑻ラ渶瑕佸彔鍔犵偣浜戣儗鏅級
    all_x = points[:, 0]
    all_y = points[:, 1]
    # 鐩爣鍧椾綋鐨勭偣
    block_x = block_points[:, 0]
    block_y = block_points[:, 1]

    # 鍒涘缓鍖呭惈涓や釜瀛愬浘鐨勫浘褰㈢獥鍙?
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 宸﹀瓙鍥撅細灞曠ず缃戞牸杞粨锛坸-y 鎶曞奖锛?
    axes[0].triplot(tri, color='gray', linewidth=0.8)
    axes[0].scatter(all_x, all_y, s=1, c='lightgray', alpha=0.5)
    axes[0].set_title("缃戞牸鏂囦欢 (x-y 鎶曞奖)", fontfamily="Times New Roman", fontsize=16)
    axes[0].set_xlabel("X", fontfamily="Times New Roman", fontsize=14)
    axes[0].set_ylabel("Y", fontfamily="Times New Roman", fontsize=14)
    
    # 鍙冲瓙鍥撅細浠呭睍绀虹洰鏍囧潡浣撶偣浜?
    axes[1].scatter(block_x, block_y, s=10, c='red')
    # 璁剧疆鏍囬閲囩敤鏂扮綏椹€佺櫧鑹层€佸瓧鍙?6锛屼笖鑳屾櫙璁句负榛戣壊浠ョ獊鍑虹櫧鑹叉枃鏈?
    axes[1].set_title(f"(鎮┖浣搟block_index}) 浣撶Н: {vol:.4f} m鲁", 
                      fontfamily="Times New Roman", fontsize=16, color="white", pad=20)
    axes[1].set_xlabel("X", fontfamily="Times New Roman", fontsize=14, color="white")
    axes[1].set_ylabel("Y", fontfamily="Times New Roman", fontsize=14, color="white")
    axes[1].set_facecolor("black")
    
    plt.tight_layout()
    plt.show()

# ========== 7. 绀轰緥璋冪敤锛氱粯鍒剁洰鏍囧潡浣?==========
# 渚嬪锛氬睍绀烘帓搴忓悗绗?涓潡浣擄紙浣撶Н鏈€澶х殑鍧椾綋锛?
plot_block(1, sorted_clusters, points, labels_full, mesh_points, faces)

