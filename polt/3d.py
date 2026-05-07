import os
import numpy as np
import pyvista as pv

# -------- 蹇嵎閿簨浠讹細淇濆瓨褰撳墠瑙嗗浘鎴浘 --------
def on_key_press():
    save_path = r"data/examples\data.png"
    p.screenshot(save_path, scale=1)
    print(f"[鉁揮 褰撳墠瑙嗗浘宸蹭繚瀛樹负锛歿save_path}")

# -------- 璺緞璁剧疆 --------
data_dir = r"data/examples"
mesh_ply  = os.path.join(data_dir, "sample_outcrop.ply")
tex_jpg   = os.path.join(data_dir, "Tile_5_0.jpg")
point_txt = os.path.join(data_dir, "sample_outcrop.txt")
dsm_obj   = os.path.join(data_dir, "sample_outcrop_dsm_mesh.obj")

# -------- 鍒涘缓 Plotter 1脳4锛堟í鎺掑洓鍥撅級 --------
p = pv.Plotter(shape=(1, 4), window_size=(2400, 600))
p.link_views()
font_opts = {"font_size": 20, "font": "times"}  # 鏂扮綏椹瓧浣?

# =========== 瀛愬浘 a锛氬甫绾圭悊缃戞牸 ===========
p.subplot(0, 0)
mesh_tex = pv.read(mesh_ply)
if os.path.isfile(tex_jpg):
    texture = pv.read_texture(tex_jpg)
    p.add_mesh(mesh_tex, texture=texture)
else:
    p.add_mesh(mesh_tex, color="white")
p.add_text("a.", position='upper_left', **font_opts)

# =========== 瀛愬浘 b锛氱偣浜?===========
p.subplot(0, 1)
points = np.loadtxt(point_txt, usecols=[0, 1, 2])
point_cloud = pv.PolyData(points)
p.add_mesh(point_cloud, render_points_as_spheres=True,
           point_size=1, color="lightgray", opacity=1)
p.add_text("b.", position='upper_left', **font_opts)

# =========== 瀛愬浘 c锛氱函缃戞牸 ===========
p.subplot(0, 2)
mesh_plain = pv.read(mesh_ply)
p.add_mesh(mesh_plain, show_edges=False, color="lightgray", opacity=1)
p.add_text("c.", position='upper_left', **font_opts)

# =========== 瀛愬浘 d锛欴SM 缃戞牸 ===========
p.subplot(0, 3)
dsm_mesh = pv.read(dsm_obj)
p.add_mesh(dsm_mesh, color="lightgray", show_edges=False, opacity=1)
p.add_text("d.", position='upper_left', **font_opts)

# -------- 娣诲姞閿洏浜嬩欢锛氭寜 S 淇濆瓨褰撳墠瑙嗗浘 --------
p.add_key_event("s", on_key_press)

# -------- 鏄剧ず绐楀彛 --------
p.show()

