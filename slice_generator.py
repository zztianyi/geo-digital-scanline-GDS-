import json
import trimesh
import pickle
import numpy as np

def load_arc_config(config_path):
    """
    鍔犺浇閰嶇疆鏂囦欢锛岃繑鍥炲渾蹇冦€佸崐寰勩€佽搴﹁寖鍥淬€佸姬闀胯寖鍥村強 Z 杞磋寖鍥淬€?
    閰嶇疆涓搴﹀潎涓哄姬搴︼紝寮ч暱鑼冨洿璁＄畻鍏紡锛歳adius * (angle_max - angle_min)
    """
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

def compute_slice_plane(center, radius, angle_min, arc_length_range, slice_position):
    """
    鏍规嵁鍒囧墫闈綅缃紙寮ч暱鍧愭爣锛夎绠楀垏鍓栭潰鍙傛暟锛?
      slice_angle = angle_min + slice_position / radius
    杩斿洖涓€涓瓧鍏革紝鍖呮嫭骞抽潰鍘熺偣銆佹硶鍚戦噺銆佸緞鍚戞柟鍚戙€佽绠楀緱鍒扮殑鍒囧墫闈㈣搴︼紝
    浠ュ強 radial_dir 鐩稿浜庢鍖楁柟鍚戯紙[0,1]锛夐『鏃堕拡鐨勬柟浣嶈锛堣搴﹀€硷紝淇濈暀涓変綅灏忔暟锛夈€?
    """
    if slice_position is None:
        slice_position = arc_length_range[1] / 2
    slice_angle = angle_min + slice_position / radius
    radial_dir = np.array([np.cos(slice_angle), np.sin(slice_angle), 0.0])
    radial_dir /= np.linalg.norm(radial_dir)
    normal = np.cross(radial_dir, [0, 0, 1])
    normal /= np.linalg.norm(normal)
    vertical_dir = np.array([0, 0, 1])
    # 璁＄畻 radial_dir 涓庢鍖楁柟鍚戠殑澶硅锛岄『鏃堕拡娴嬮噺锛?
    # 璁炬鍖楁柟鍚戜负 [0,1]锛屼娇鐢?np.arctan2(radial_dir[0], radial_dir[1])
    azimuth_rad = np.arctan2(radial_dir[0], radial_dir[1])
    azimuth_deg = np.degrees(azimuth_rad) % 360
    azimuth_deg = round(azimuth_deg, 3)
    return {
        "origin": center,
        "normal": normal,
        "radial_dir": radial_dir,
        "slice_angle": slice_angle,
        "vertical_dir": vertical_dir,
        "azimuth": azimuth_deg  # 鍗曠嫭淇濆瓨鐨勬柟浣嶈锛堥『鏃堕拡娴嬮噺锛?
    }

def filter_mesh_by_z_range(mesh, z_min, z_max):
    """
    鏍规嵁 Z 杞磋寖鍥磋繃婊ょ綉鏍硷紝浠呬繚鐣欑鍚堟潯浠剁殑涓夎闈€?
    """
    face_vertices_z = mesh.vertices[mesh.faces][:, :, 2]
    face_min = face_vertices_z.min(axis=1)
    face_max = face_vertices_z.max(axis=1)
    mask = (face_max >= z_min) & (face_min <= z_max)
    submeshes = mesh.submesh([mask], only_watertight=False)
    return submeshes[0] if len(submeshes) > 0 else None

def slice_and_check_faces(mesh, origin, normal):
    """
    浣跨敤 mesh_plane 骞惰繑鍥?(lines_3d, face_ids_1d)锛岀敤浜庡悗缁鐞?
    """
    lines_3d, face_ids_1d = trimesh.intersections.mesh_plane(
        mesh=mesh,
        plane_normal=normal,
        plane_origin=origin,
        return_faces=True
    )
    return lines_3d, face_ids_1d




def main():
    config_path = r"configs/arc_config.example.json"
    mesh_path = r"data/private/raw_model\combined_model.glb" 
    # mesh_path = r"data/examples/sample_outcrop.ply"
    # mesh_path = r"data/examples/segmented_mesh\Merged mesh_seg_3.ply"
    mesh_path = r"outputs/intermediate\Merged_seg_5_4.ply"
    
    # 鍔犺浇閰嶇疆
    center, radius, angle_min, angle_max, arc_length_range, z_min, z_max = load_arc_config(config_path)
    
    # 鍒囧墫闈綅缃簭鍒?
    slice_positions = np.arange(59.45, 72.65, 0.05)

    # 鍔犺浇骞惰繃婊ょ綉鏍?
    mesh = trimesh.load_mesh(mesh_path)
    mesh_filtered = filter_mesh_by_z_range(mesh, z_min, z_max)

    # 瀛樺偍鎵€鏈夊垏鍓栭潰鏁版嵁
    slices_data = {}

    for pos in slice_positions:
        plane_params = compute_slice_plane(center, radius, angle_min, arc_length_range, pos)
        origin = plane_params["origin"]
        normal = plane_params["normal"]
        

        # 浣跨敤甯﹂潰绱㈠紩鐨勫垏鍓插嚱鏁?
        lines_3d, face_ids_1d = slice_and_check_faces(mesh_filtered, origin, normal)

        # 淇濆瓨涓哄瓧绗︿覆閿紝鍚屾椂鍗曠嫭淇濆瓨鏂逛綅瑙?
        key = f"{pos:.2f}"
        slices_data[key] = {
            "plane_params": plane_params,
            "slicing": {
                "lines_3d": lines_3d,
                "face_ids": face_ids_1d
            }
        }

        print(f"鍒囧墫闈綅缃?{key}锛堝姬闀垮潗鏍囷級锛屾柟浣嶈 {plane_params['azimuth']}掳锛岀嚎娈垫暟閲?{len(lines_3d)}")

    # 娓呯悊鍐呭瓨
    del mesh, mesh_filtered

    # 淇濆瓨鏁版嵁
    output_path = r"outputs/intermediate\all_slices_output_faces_seg_5_4.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(slices_data, f)

    print("鍏ㄩ儴鍒囧墫闈㈡暟鎹凡淇濆瓨鍒?, output_path)
    
if __name__ == "__main__":
    main()

