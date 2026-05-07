import os
import json
import pyvista as pv
import numpy as np
import struct
import msgpack

def load_3mx_files(data_dir):
    """
    鍔犺浇3MX涓绘枃浠跺拰3MXB瀛愭枃浠惰矾寰勩€?
    :param data_dir: 3MX鏂囦欢鐨勪富鐩綍璺緞銆?
    :return: 涓绘枃浠舵暟鎹拰3MXB鏂囦欢璺緞鍒楄〃銆?
    """
    # 鍔犺浇涓绘枃浠?
    main_file_path = os.path.join(data_dir, "scene.3mx")  # 鏇挎崲涓轰富鏂囦欢鍚?
    if not os.path.exists(main_file_path):
        raise FileNotFoundError(f"涓绘枃浠舵湭鎵惧埌: {main_file_path}")
    
    with open( main_file_path, 'rb') as file:
        binary_content = file.read()
        print(f"鏂囦欢澶у皬: {len(binary_content)} 瀛楄妭")
        print("鏂囦欢鐨勬墍鏈夊瓧鑺傚唴瀹?(鍗佸叚杩涘埗琛ㄧず):")
        print(binary_content.hex())  # 鎵撳嵃鎵€鏈夊瓧鑺傜殑鍗佸叚杩涘埗琛ㄧず

    with open(main_file_path, 'r', encoding='utf-8') as file:
        main_data = json.load(file)

    print("涓绘枃浠跺唴瀹?")
    print(json.dumps(main_data, indent=4))  # 鏍煎紡鍖栨墦鍗颁富鏂囦欢鍐呭

    # 鏀堕泦鎵€鏈夊瓙鏂囦欢璺緞
    binary_dir = os.path.join(data_dir, "Data")  # 3MXB瀛愭枃浠跺瓨鍌ㄧ洰褰?
    if not os.path.exists(binary_dir):
        raise FileNotFoundError(f"浜岃繘鍒舵枃浠剁洰褰曟湭鎵惧埌: {binary_dir}")

    binary_files = []
    for root, dirs, files in os.walk(binary_dir):
        for file in files:
            if file.endswith(".3mxb"):  # 浠呭鐞?MXB鏂囦欢
                binary_files.append(os.path.join(root, file))

    if not binary_files:
        raise ValueError("鏈壘鍒颁换浣?MXB鏂囦欢锛岃妫€鏌ユ枃浠剁粨鏋勩€?)

    return main_data, binary_files

def read_3mxb(file_path):
    """
    璇诲彇骞惰В鏋?MXB鏂囦欢銆?
    :param file_path: 3MXB鏂囦欢璺緞
    :return: 澶撮儴JSON鏁版嵁鍜岀紦鍐插尯鍐呭
    """
    with open(file_path, 'rb') as f:
        # 璇诲彇骞绘暟鍜屽ご澶у皬
        magic = f.read(5).decode('utf-8')
        if magic != "3MXBO":
            raise ValueError(f"鏃犳晥鐨勫够鏁? {magic}")

        header_size = struct.unpack('<I', f.read(4))[0]
        json_header = f.read(header_size - 9).decode('utf-8')
        header_data = json.loads(json_header)

        # 璇诲彇缂撳啿鍖?
        buffer_data = f.read()
    
    return header_data, buffer_data

def parse_nodes(header_data):
    """
    浠?MXB鏂囦欢澶翠腑瑙ｆ瀽鐖惰妭鐐瑰拰瀛愯妭鐐广€?
    :param header_data: 3MXB鏂囦欢澶碕SON鏁版嵁
    :return: 鐖惰妭鐐瑰垪琛ㄥ拰瀛愯妭鐐瑰垪琛?
    """
    nodes = header_data.get("nodes", [])
    parent_nodes = []
    child_nodes = []

    for node in nodes:
        if "parent" in node:
            child_nodes.append(node)  # 瀛愯妭鐐癸紙楂樺垎杈ㄧ巼锛?
        else:
            parent_nodes.append(node)  # 鐖惰妭鐐癸紙浣庡垎杈ㄧ巼锛?

    print(f"瑙ｆ瀽鑺傜偣淇℃伅:\n鐖惰妭鐐规暟閲? {len(parent_nodes)}\n瀛愯妭鐐规暟閲? {len(child_nodes)}")
    return parent_nodes, child_nodes

def visualize_nodes(nodes, buffer_data, title=""):
    """
    鍙鍖栨寚瀹氱殑鑺傜偣銆?
    :param nodes: 鑺傜偣鍒楄〃
    :param buffer_data: 鏂囦欢缂撳啿鍖烘暟鎹?
    :param title: 鍙鍖栨爣棰?
    """
    plotter = pv.Plotter(title=title)

    for node in nodes:
        # 瑙ｆ瀽姣忎釜鑺傜偣鐨勭偣浜戞暟鎹紙绀轰緥锛?
        resource_start = node.get("resource_start", 0)
        resource_end = node.get("resource_end", 0)
        resource_data = buffer_data[resource_start:resource_end]

        # 鍋囪姣忎釜鐐圭敱3涓猣loat琛ㄧず
        num_points = len(resource_data) // 12  # 姣忎釜鐐圭敱12瀛楄妭琛ㄧず
        points = struct.unpack(f'<{num_points * 3}f', resource_data)
        points = np.array(points).reshape(-1, 3)

        # 娣诲姞鍒扮粯鍥剧獥鍙?
        poly_data = pv.PolyData(points)
        plotter.add_mesh(poly_data, point_size=5, render_points_as_spheres=True)

    plotter.show()

# 涓荤洰褰曡矾寰勶紝鏇挎崲涓哄疄闄呰矾寰?
data_dir = r"data/private/3mx_scene"

try:
    # 鍔犺浇涓绘枃浠跺拰3MXB瀛愭枃浠惰矾寰?
    main_data, binary_files = load_3mx_files(data_dir)

    # 閬嶅巻姣忎釜3MXB鏂囦欢骞跺睍绀虹埗鑺傜偣鍜屽瓙鑺傜偣
    for binary_file in binary_files:
        print(f"姝ｅ湪瑙ｆ瀽鏂囦欢: {binary_file}")
        header_data, buffer_data = read_3mxb(binary_file)

        # 瑙ｆ瀽鑺傜偣
        parent_nodes, child_nodes = parse_nodes(header_data)

        # 鍙鍖栫埗鑺傜偣锛堜綆鍒嗚鲸鐜囷級
        print(f"灞曠ず浣庡垎杈ㄧ巼鐗堟湰: {binary_file}")
        visualize_nodes(parent_nodes, buffer_data, title="鐖惰妭鐐癸紙浣庡垎杈ㄧ巼锛?)

        # 鍙鍖栧瓙鑺傜偣锛堥珮鍒嗚鲸鐜囷級
        print(f"灞曠ず楂樺垎杈ㄧ巼鐗堟湰: {binary_file}")
        visualize_nodes(child_nodes, buffer_data, title="瀛愯妭鐐癸紙楂樺垎杈ㄧ巼锛?)

except Exception as e:
    print(f"鍙戠敓閿欒: {e}")

