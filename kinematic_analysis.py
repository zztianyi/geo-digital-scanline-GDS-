import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def circle_from_3_points(A, B, C):
    """
    给定三点 A, B, C（不共线），求唯一一条过该三点的圆。
    返回 (center, R) = ((cx, cy), radius)
    若三点共线，返回 (None, None)。
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    d = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    if abs(d) < 1e-12:
        return None, None  # 三点几乎共线，无法成圆
    
    ux = ((A[0]**2 + A[1]**2)*(B[1]-C[1]) +
          (B[0]**2 + B[1]**2)*(C[1]-A[1]) +
          (C[0]**2 + C[1]**2)*(A[1]-B[1]))
    uy = ((A[0]**2 + A[1]**2)*(C[0]-B[0]) +
          (B[0]**2 + B[1]**2)*(A[0]-C[0]) +
          (C[0]**2 + C[1]**2)*(B[0]-A[0]))
    cx = ux / d
    cy = uy / d
    R = np.hypot(A[0] - cx, A[1] - cy)
    return (cx, cy), R

def draw_inclination_arc(ax, alpha_deg, beta_deg, color='red', ls='-', lw=2):
    """
    绘制“坡面倾角”弧线示意：
    1) alpha_deg：射线方位角（与 x 轴逆时针夹角，度数）
    2) beta_deg ：倾角（度）——本示例中直接将其作为弧长(弧度)使用
    """
    alpha = np.radians(alpha_deg)     # 射线方向(弧度)
    beta_rad = np.radians(beta_deg)   # 倾角转成弧度
    
    # ============= (1) 倾向射线 =============
    xA = np.cos(alpha)   # 与单位圆交点
    yA = np.sin(alpha)
    ax.plot([0, xA], [0, yA], color='red', ls='--', alpha=0.6, label='射线(倾向)')
    
    # ============= (2) 垂直直径 T1T2 =============
    alpha_perp = alpha + np.pi/2
    xT1, yT1 = np.cos(alpha_perp), np.sin(alpha_perp)
    xT2, yT2 = -xT1, -yT1
    ax.plot([xT1, xT2], [yT1, yT2], color='red', ls='--', alpha=0.6, label='垂直直径')
    
    # ============= (3) 点 M (在射线上, 距离 = beta_rad) =============
    xM = beta_rad * np.cos(alpha)
    yM = beta_rad * np.sin(alpha)
    ax.scatter(xM, yM, color='orange', zorder=5)  # 标出 M
    ax.plot([0, xM], [0, yM], color='orange', ls=':', alpha=0.7)
    
    # ============= (4) 三点共圆 -> 画弧 [T2 -> T1] =============
    A = (xT1, yT1)
    B = (xT2, yT2)
    C = (xM,  yM)
    center, R = circle_from_3_points(A, B, C)
    if center is None:
        return
    cx, cy = center
    
    # 计算圆上 T2, T1 的极角
    def angle_of_point(px, py):
        return np.degrees(np.arctan2(py - cy, px - cx))
    theta_T2 = angle_of_point(xT2, yT2)
    theta_T1 = angle_of_point(xT1, yT1)
    
    # 将它们映射到 [0,360)，并从 T2->T1 逆时针
    def wrap360(deg):
        return deg % 360
    t2_ = wrap360(theta_T2)
    t1_ = wrap360(theta_T1)
    if t1_ < t2_:
        t1_ += 360
    
    arc_patch = Arc((cx, cy),
                    2*R, 2*R,
                    angle=0,
                    theta1=t1_,
                    theta2=t2_,
                    color=color, ls=ls, lw=lw)
    ax.add_patch(arc_patch)

def plot_plane_failure_diagram(
    slope_dip=80,            # 坡面倾角（°）
    slope_dir=270,           # 坡面倾向（°）
    friction_angle=30,       # 摩擦角（°）
    offset=20                # 正常/过悬坡面区域的±容许角度（°）
):
    """
    绘制与文中示例类似的极坐标示意图（中文版本） + 倾向弧示意：
      - 左侧子图 (ax1): 极坐标表示方位角(0°~360°)和倾角(0°~90°)
      - 右侧子图 (ax2): 笛卡尔坐标下，演示“坡面倾向-倾角”弧线示意
    """
    # ========== 1) 建立画布，拆分为左右两子图 ========== 
    fig = plt.figure(figsize=(12, 6))
    
    # ========== (a) 左子图：极坐标“面滑示意” ========== 
    ax1 = fig.add_subplot(121, polar=True)
    
    # 将度数转弧度
    slope_dip_rad = np.radians(slope_dip)
    slope_dir_rad = np.radians(slope_dir)
    friction_rad = np.radians(friction_angle)
    offset_rad = np.radians(offset)
    
    # 设置极坐标参数：0° 在顶部，顺时针递增
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rlim(0, np.radians(90))  # 径向限制在 0~90°
    
    ax1.set_thetagrids([0, 90, 180, 270], labels=['北','东','南','西'])
    ax1.set_rgrids(np.radians([0, 30, 60, 90]), labels=['0°','30°','60°','90°'])
    ax1.set_title("面滑示意图（极坐标）", y=1.08, fontsize=13)
    
    # (1) 摩擦圆锥
    theta_f = np.linspace(0, 2*np.pi, 180)
    r_f = np.full_like(theta_f, friction_rad)
    ax1.plot(theta_f, r_f, 'r--', label='摩擦圆锥')
    
    # (2) 坡面倾角圆
    theta_slope = np.linspace(0, 2*np.pi, 180)
    r_slope = np.full_like(theta_slope, slope_dip_rad)
    ax1.plot(theta_slope, r_slope, 'k--', label='坡面倾角圆')
    
    # (3) 日照包络 (Daylight Envelope) 简化示意
    thetas_daylight = np.linspace(slope_dir_rad - offset_rad, slope_dir_rad + offset_rad, 100)
    ax1.fill_between(thetas_daylight, 0, slope_dip_rad, color='none', edgecolor='black', hatch='//', alpha=0.2, label='日照包络')
    
    # (4) 正常坡面潜在失稳区（蓝色）
    thetas_normal = np.linspace(slope_dir_rad - offset_rad, slope_dir_rad + offset_rad, 100)
    ax1.fill_between(thetas_normal, friction_rad, slope_dip_rad, color='blue', alpha=0.3, label='正常坡面潜在面滑区')
    
    # (5) 过悬坡面潜在失稳区（粉色）
    overhang_dir_rad = slope_dir_rad + np.pi  # slope_dir ± 180°
    thetas_overhang = np.linspace(overhang_dir_rad - offset_rad, overhang_dir_rad + offset_rad, 100)
    ax1.fill_between(thetas_overhang, friction_rad, np.radians(90), color='magenta', alpha=0.3, label='过悬坡面潜在面滑区')
    
    # (6) 红线标出“坡面倾向”及其对应倾角
    ax1.plot([slope_dir_rad, slope_dir_rad], [0, slope_dip_rad], 'r-', linewidth=2, label='坡面倾向')
    
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=10)
    
    # ========== (b) 右子图：笛卡尔坐标下“倾向-倾角”弧线示意 ========== 
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim([-2, 2])
    ax2.set_ylim([-2, 2])
    ax2.set_title("坡面倾向-倾角弧线示意", fontsize=13)
    ax2.grid(True)
    
    # 画一个单位圆做参考
    circle = plt.Circle((0,0), 1, fill=False, color='gray', ls='--')
    ax2.add_patch(circle)
    
    # 调用倾向弧线函数 —— 这里传入 slope_dir, slope_dip 作为演示
    draw_inclination_arc(ax2, alpha_deg=slope_dir, beta_deg=slope_dip, color='blue')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置中文字体（需要系统已安装黑体“SimHei”或其他中文字体）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 示例参数
    slope_dip_input = 80         # 坡面倾角（°）
    slope_dir_input = 270        # 坡面倾向（°）
    friction_angle_input = 30    # 摩擦角（°）
    offset_input = 20            # 正常/过悬坡面 ±20°

    plot_plane_failure_diagram(
        slope_dip=slope_dip_input,
        slope_dir=slope_dir_input,
        friction_angle=friction_angle_input,
        offset=offset_input
    )

