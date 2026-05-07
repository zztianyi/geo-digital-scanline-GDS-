import pickle
from collections import defaultdict

def gather_line_data(single_res):
    """
    浠庡崟涓粨鏋滀腑鎻愬彇绾挎涓庨潰绱㈠紩鏁版嵁锛屽苟璁＄畻楂樺害鍊笺€?
    瀵逛簬姣忎釜绾挎缁勶紝鍏堣绠楄缁勫唴鎵€鏈夌嚎娈电鐐圭殑鏈€楂?z 鍊硷紝
    鐒跺悗瀵规瘡鏉＄嚎娈佃绠楅珮搴︼紙瀹氫箟涓猴細缁勫唴鏈€楂?z 鍊煎噺鍘昏绾挎绔偣 z 鍊肩殑骞冲潎锛夈€?
    鍚屾椂锛屽浜庡瓨鍦ㄩ潰绱㈠紩鐨勭嚎娈碉紝灏嗚绠楀緱鍒扮殑楂樺害璁板綍鍒板瓧鍏镐腑銆?
    
    杩斿洖锛?
        - line_segments: List[ (p1_3d, p2_3d, height) ]
        - face_heights_dict: dict { face_idx : list of heights }
    """
    line_segments = []
    face_heights_dict = defaultdict(list)
    if not single_res:
        return line_segments, face_heights_dict
    
    red_groups_corrected = single_res.get("red_groups_corrected", {})
    for parent_idx, groups in red_groups_corrected.items():
        for group in groups:
            group_segments = group.get("group_segments", [])
            if not group_segments:
                continue
            # 鍏堟敹闆嗗綋鍓嶇粍涓墍鏈夌嚎娈电鐐圭殑 z 鍊?
            zs = []
            for seg in group_segments:
                p1 = seg[4]
                p2 = seg[5]
                zs.append(p1[2])
                zs.append(p2[2])
            group_max_z = max(zs)
            # 閬嶅巻缁勫唴姣忔潯绾挎锛岃绠楅珮搴﹀苟璁板綍
            for seg in group_segments:
                p1 = seg[4]
                p2 = seg[5]
                face_idx = seg[6]
                avg_z = (p1[2] + p2[2]) / 2.0
                height = group_max_z - avg_z
                line_segments.append((p1, p2, height))
                if face_idx is not None:
                    face_heights_dict[face_idx].append(height)
    return line_segments, face_heights_dict

def serial_collect_line_data(all_results):
    """
    涓茶澶勭悊鎵€鏈夌粨鏋滐紝鏀堕泦鎵€鏈夌嚎娈垫暟鎹強闈㈠搴旂殑楂樺害鍒楄〃銆?
    
    杩斿洖锛?
        - all_segments: List[ (p1, p2, height) ]
        - all_face_heights_dict: dict { face_idx : list of heights }
    """
    all_segments = []
    all_face_heights_dict = defaultdict(list)
    for single_res in all_results:
        segments, face_heights = gather_line_data(single_res)
        all_segments.extend(segments)
        for face, heights in face_heights.items():
            all_face_heights_dict[face].extend(heights)
    return all_segments, all_face_heights_dict

def extract_line_face_from_pkl(results_pkl):
    """
    浠庢寚瀹氱殑 pickle 鏂囦欢涓姞杞芥暟鎹紝
    骞跺埄鐢ㄦ柊鐨勯€昏緫鎻愬彇绾挎鏁版嵁锛堝惈楂樺害锛変互鍙婇潰瀵瑰簲鐨勯珮搴﹀垪琛ㄣ€?
    
    杩斿洖锛?
        - all_segments, face_heights_dict
    """
    all_results = []
    with open(results_pkl, "rb") as f:
        while True:
            try:
                batch_data = pickle.load(f)
                all_results.append(batch_data)
            except EOFError:
                break
    all_segments, face_heights_dict = serial_collect_line_data(all_results)
    return all_segments, face_heights_dict

def merge_face_heights(face_heights_dict, strategy='max'):
    """
    灏嗗悓涓€闈㈠搴旂殑澶氫釜楂樺害鍊煎悎骞朵负涓€涓唬琛ㄥ€笺€?
    
    鍙傛暟锛?
        face_heights_dict: dict { face_idx: list of heights }
        strategy: 鍚堝苟绛栫暐锛屽彲閫?'max'锛堟渶澶у€硷級銆?min'锛堟渶灏忓€硷級鎴?'avg'锛堝钩鍧囧€硷級
    
    杩斿洖锛?
        merged: dict { face_idx: merged height }
    """
    merged = {}
    for face, heights in face_heights_dict.items():
        if not heights:
            continue
        if strategy == 'max':
            merged[face] = max(heights)
        elif strategy == 'min':
            merged[face] = min(heights)
        elif strategy == 'avg':
            merged[face] = sum(heights) / len(heights)
        else:
            merged[face] = max(heights)
    return merged

def save_full_line_face_data(all_segments, face_heights_dict, output_path):
    """
    淇濆瓨鎻愬彇鐨勭嚎娈垫暟鎹拰鍘熷闈㈤珮搴﹀瓧鍏搞€?
    淇濆瓨鐨勬暟鎹牸寮忎负 (all_segments, face_heights_dict)銆?
    
    鍙傛暟锛?
        all_segments: 鎵€鏈夌嚎娈垫暟鎹紙鍚珮搴︼級锛屽垪琛ㄥ舰寮忥紝渚嬪 [(p1, p2, height), ...]
        face_heights_dict: dict锛屾瘡涓潰瀵瑰簲鐨勫涓珮搴︽暟鎹紝鏈悎骞?
        output_path: str锛屼繚瀛樼殑鏂囦欢璺緞锛屼緥濡?'line_face_full_data.pkl'
    """
    with open(output_path, "wb") as f:
        pickle.dump((all_segments, face_heights_dict), f)
    print("瀹屾暣鏁版嵁宸蹭繚瀛樺埌锛?, output_path)

if __name__ == "__main__":
    # 绀轰緥锛氫娇鐢ㄦ柊鐨勬彁鍙栭€昏緫鍜屼繚瀛橀€昏緫
    results_pkl = r"data/private/raw_model\multi_slice_results_0.05_40_0-140_buchang.pkl"
    output_path = r"outputs/intermediate\line_face_data_5_2_buchang.pkl"

    # results_pkl = r"outputs/intermediate\multi_slice_results_0.05_5_2_buchang.pkl"
    
    print("寮€濮嬫彁鍙栫嚎娈靛拰闈㈡暟鎹紙鍚珮搴︼級...")
    all_segments, face_heights_dict = extract_line_face_from_pkl(results_pkl)
    print("鎻愬彇瀹屾垚锛岀嚎娈垫暟锛?, len(all_segments), "娑夊強鐩爣闈㈡暟锛?, len(face_heights_dict))
    
    save_full_line_face_data(all_segments, face_heights_dict, output_path)

    


