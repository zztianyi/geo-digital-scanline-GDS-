import os
import trimesh

def combine_obj_and_mtl(root_dir, output_obj="combined_model_1.obj", output_mtl="combined_model_1.mtl"):
    """
    閬嶅巻 root_dir 涓嬪悇涓瓙鏂囦欢澶癸紙濡?Tile_1銆乀ile_2 绛夛級锛?
    鍔犺浇 .obj 妯″瀷涓庡搴旂殑 .mtl 鏉愯川鏂囦欢锛屽皢鎵€鏈夋ā鍨嬪悎骞讹紝
    骞舵妸鎵€鏈?.mtl 鏂囦欢鐨勫唴瀹瑰悎骞舵垚涓€涓潗璐ㄦ枃浠讹紝
    鏈€鍚庣敓鎴愬悎骞跺悗鐨?OBJ 妯″瀷鍜屾€荤殑 MTL 鏂囦欢銆?
    """
    combined_meshes = []
    combined_mtl_content = ""
    
    # 閬嶅巻鎵€鏈夊瓙鐩綍
    for sub_name in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, sub_name)
        if os.path.isdir(sub_dir):
            # 閬嶅巻璇ュ瓙鐩綍涓嬫墍鏈夋枃浠?
            for filename in os.listdir(sub_dir):
                lower = filename.lower()
                if lower.endswith(".obj"):
                    obj_path = os.path.join(sub_dir, filename)
                    print(f"姝ｅ湪鍔犺浇妯″瀷锛歿obj_path}")
                    
                    # 鍔犺浇鏃朵繚鐣欐潗璐ㄥ拰绾圭悊淇℃伅
                    mesh = trimesh.load(obj_path, process=False, use_embedded_textures=True)
                    combined_meshes.append(mesh)
                    
                    # 灏濊瘯鍔犺浇瀵瑰簲鐨?.mtl 鏂囦欢锛屽亣瀹氫笌 .obj 鍚屽悕
                    base_name = os.path.splitext(filename)[0]
                    mtl_path = os.path.join(sub_dir, base_name + ".mtl")
                    if os.path.exists(mtl_path):
                        print(f"姝ｅ湪鍔犺浇鏉愯川锛歿mtl_path}")
                        with open(mtl_path, 'r', encoding='utf-8') as f:
                            mtl_data = f.read()
                        # 娣诲姞鍒嗛殧娉ㄩ噴锛岃〃鏄庢潵婧愪簬鍝竴涓枃浠跺す
                        combined_mtl_content += f"\n# ===== 鏉愯川鏂囦欢锛歿mtl_path} =====\n"
                        combined_mtl_content += mtl_data
                        combined_mtl_content += f"\n# ===== End of {mtl_path} =====\n"
                    else:
                        print(f"娉ㄦ剰锛歿obj_path} 瀵瑰簲鐨勬潗璐ㄦ枃浠?{mtl_path} 鏈壘鍒帮紒")
    
    if combined_meshes:
        # 鍚堝苟鎵€鏈夊姞杞界殑妯″瀷
        combined_model = trimesh.util.concatenate(combined_meshes)
        
        # 瀵煎嚭鍓嶅厛灏嗗悎骞跺悗鐨勬ā鍨嬪鍑轰负 OBJ 鏍煎紡瀛楃涓?
        exported_str = combined_model.export(file_type='obj')
        lines = exported_str.splitlines()
        # 淇敼鎴栨彃鍏?mtllib 寮曠敤锛岀‘淇?OBJ 鏂囦欢寮曠敤鍚堝苟鍚庣殑鏉愯川鏂囦欢
        found = False
        for i, line in enumerate(lines):
            if line.startswith("mtllib"):
                lines[i] = f"mtllib {output_mtl}"
                found = True
                break
        if not found:
            lines.insert(0, f"mtllib {output_mtl}")
        exported_str = "\n".join(lines)
        
        # 鍐欏嚭鍚堝苟鍚庣殑 OBJ 鏂囦欢
        output_obj_path = os.path.join(root_dir, output_obj)
        with open(output_obj_path, 'w', encoding='utf-8') as f:
            f.write(exported_str)
        print(f"\n鍚堝苟瀹屾垚锛屽鍑烘ā鍨嬫枃浠讹細{output_obj_path}")
        
        # 鍐欏嚭鍚堝苟鍚庣殑 MTL 鏉愯川鏂囦欢
        output_mtl_path = os.path.join(root_dir, output_mtl)
        with open(output_mtl_path, 'w', encoding='utf-8') as f:
            f.write(combined_mtl_content)
        print(f"鍚堝苟瀹屾垚锛屽鍑烘潗璐ㄦ枃浠讹細{output_mtl_path}")
    else:
        print("鏈彂鐜颁换浣?.obj 妯″瀷鏂囦欢锛岃妫€鏌ョ洰褰曠粨鏋勩€?)

if __name__ == "__main__":
    # 鏁版嵁鐩綍锛屽寘鍚悇涓?Tile_x 瀛愭枃浠跺す
    data_dir = r"data/private/raw_model"
    combine_obj_and_mtl(data_dir)

