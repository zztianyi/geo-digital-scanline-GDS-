import trimesh
import numpy as np

def verify_mesh(mesh):
    # 妫€鏌ユ瘡涓?face 鐨勯《鐐圭储寮曟槸鍚﹀湪鍚堟硶鑼冨洿鍐?
    num_vertices = mesh.vertices.shape[0]
    invalid_face_indices = []
    for i, face in enumerate(mesh.faces):
        # 濡傛灉鏌愪釜绱㈠紩瓒呭嚭鑼冨洿锛屽垯璁板綍
        if np.any(face >= num_vertices) or np.any(face < 0):
            invalid_face_indices.append(i)
    
    if invalid_face_indices:
        print(f"鍙戠幇 {len(invalid_face_indices)} 涓潰瀛樺湪鏃犳晥绱㈠紩锛岀ず渚嬶細{invalid_face_indices[:10]}")
    else:
        print("鎵€鏈夐潰鐨勭储寮曞潎鍦ㄦ湁鏁堣寖鍥村唴銆?)
    
    # 妫€鏌ョ綉鏍兼槸鍚﹀皝闂紙watertight锛?
    if mesh.is_watertight:
        print("缃戞牸鏄皝闂殑锛坵atertight锛夈€?)
    else:
        print("缃戞牸涓嶆槸灏侀棴鐨勶紙not watertight锛夈€?)
    
    # 妫€鏌ラ潰娉曞悜閲忎竴鑷存€э紙winding consistent锛?
    if mesh.is_winding_consistent:
        print("缃戞牸鐨勯潰娉曞悜閲忔柟鍚戜竴鑷淬€?)
    else:
        print("缃戞牸鐨勯潰娉曞悜閲忔柟鍚戜笉涓€鑷淬€?)
    
    # 鎵撳嵃涓€浜涘叾浠栧熀鏈俊鎭?
    print("椤剁偣鏁帮細", num_vertices)
    print("闈㈡暟锛?, mesh.faces.shape[0])
    
    # 濡傛灉 trimesh 鐗堟湰鏀寔锛屽彲浠ヨ皟鐢?validate 鏂规硶
    try:
        validation = mesh.validate()
        if validation:
            print("缃戞牸楠岃瘉淇℃伅锛?, validation)
        else:
            print("缃戞牸鏈彂鐜版槑鏄鹃獙璇侀敊璇€?)
    except Exception as e:
        print("璋冪敤 mesh.validate() 鏃跺嚭閿欙細", e)

# 绀轰緥锛氬姞杞界綉鏍煎苟楠岃瘉
mesh_path = r"data/private/raw_model\combined_model.glb"
mesh = trimesh.load_mesh(mesh_path)
verify_mesh(mesh)

