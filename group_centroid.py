import pickle
import numpy as np


def extract_group_centroids(results_pkl, output_pkl):
    """
    鎻愬彇姣忎釜 red_groups_corrected 涓殑缁勩€佽川蹇冧互鍙婇潰绱㈠紩鍒楄〃锛屽苟淇濆瓨銆?
    淇濆瓨鏍煎紡涓?List[ (group_segments, centroid, face_indices) ]
    """
    all_results = []
    with open(results_pkl, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                all_results.append(batch)
            except EOFError:
                break

    group_with_centroid_faces = []
    for res in all_results:
        red_groups_corrected = res.get("red_groups_corrected", {})
        red_centroids = res.get("red_centroids", {})

        for parent_idx, groups in red_groups_corrected.items():
            centroids = red_centroids.get(parent_idx, [])
            for i, group in enumerate(groups):
                group_segments = group.get("group_segments", [])
                if not group_segments:
                    continue
                if i < len(centroids):
                    centroid = centroids[i]
                    face_indices = [seg[6] for seg in group_segments if seg[6] is not None]
                    group_with_centroid_faces.append((group_segments, centroid, face_indices))

    # 淇濆瓨
    with open(output_pkl, "wb") as f:
        pickle.dump(group_with_centroid_faces, f)

    print(f"[鉁揮 宸叉彁鍙栧苟淇濆瓨 {len(group_with_centroid_faces)} 缁?red_group + 璐ㄥ績 + 闈㈢储寮?鏁版嵁鑷筹細{output_pkl}")


if __name__ == "__main__":
    results_pkl = r"data/private/raw_model\\multi_slice_results_0.05_40_84-92_buchang.pkl"
    # results_pkl = r"outputs/intermediate\multi_slice_results_0.05_5_5_buchang.pkl"
    output_pkl  = r"data/private/raw_model\polt_5_\group_centroid_only.pkl"
    extract_group_centroids(results_pkl, output_pkl)


