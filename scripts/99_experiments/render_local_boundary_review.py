"""Read-only A/B/C boundary atlas from frozen Phase 1 / B0 / B1 artifacts.

No mesh reader, recognition entry point, scorer, reducer or route operation.
Example: python scripts/99_experiments/render_local_boundary_review.py --out outputs/local_boundary_review/20260923_boundary
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/06_visualization'))
from locc_local_review_plot import (boundary_segments, plot_boundary_neighbors,
                                    plot_boundary_alignment)

LOCAL = ROOT / 'outputs/facetrack_local_preference/20260923_105650'
SOURCE = ROOT / 'outputs/facetrack_voting_phase1/20260923_094440'


def spec(key, region, boundary, label, box, detail, keys, count=5):
    return dict(id=key, region=region, boundary=boundary, label=label, box=box,
                detail_box=detail, key_s=keys, neighbor_count=count)


CASES = [
    spec('F01_L', 'R00806', 109.35, '3V 岛左边界', [4,19,1378.5,1385.3], [4.5,9.5,1381.6,1383.3], [109.25,109.30,109.35,109.40]),
    spec('F01_R', 'R00806', 109.50, '3V 岛右边界', [4,19,1378.5,1385.3], [4.5,9.5,1381.6,1383.3], [109.40,109.45,109.50,109.55]),
    spec('F02_L', 'R00806', 116.05, '5V 岛左边界', [-1,13,1372,1386.3], [.2,3.5,1383,1385.5], [115.95,116.00,116.05,116.15]),
    spec('F02_R', 'R00806', 116.30, '5V 岛右边界', [-1,13,1372,1386.3], [.2,3.5,1383,1385.5], [116.15,116.25,116.30,116.35]),
    spec('F_BIG', 'R00806', 131.10, '153V 大区与右邻区', [-8,5,1375,1405], [-7.3,-5.5,1402.4,1404.5], [131.00,131.05,131.10,131.15]),
    spec('F03_L', 'R00806', 138.05, '8V 岛左边界', [-10,2,1387,1404], [-9.7,-7.8,1401.4,1403.4], [137.95,138.00,138.05,138.20], 9),
    spec('F03_R', 'R00806', 138.45, '8V 岛右边界', [-10,2,1387,1404], [-9.7,-7.8,1401.4,1403.4], [138.20,138.40,138.45,138.50], 9),
    spec('D01_L', 'R00204', 92.45, '3V 原始反向票左边界', [-16,-8,1403,1414], [-13.5,-10.5,1411.5,1413.4], [92.35,92.40,92.45,92.50]),
    spec('D01_R', 'R00204', 92.60, '3V 原始反向票右边界', [-16,-8,1403,1414], [-13.5,-10.5,1411.5,1413.4], [92.50,92.55,92.60,92.65]),
    spec('D02_L', 'R00204', 135.35, '3V 原始反向票左边界', [-17,-8,1402.5,1415], [-16,-13,1411.5,1414.2], [135.25,135.30,135.35,135.40]),
    spec('D02_R', 'R00204', 135.50, '3V 原始反向票右边界', [-17,-8,1402.5,1415], [-16,-13,1411.5,1414.2], [135.40,135.45,135.50,135.55]),
    spec('E01', 'R00629', 104.90, 'PF19 → PF16 子区边界', [5,23,1381,1389], [8.7,11.2,1385.2,1386.5], [104.80,104.85,104.90,104.95]),
]

# Filled after inspecting the exported frozen-geometry figures. These comments
# are review opinions, not inputs to B0/B1 or automatic merge instructions.
OBSERVATIONS = {
    'F01_L': dict(
        purpose='检查 109.35 m 开始的 3V PF16 小岛是否伴随新结构出现。',
        observation='PF16 在 u≈5.2–9.5 m 的短上翘、平台和缓降顺序，PF19 在 u≈6–7 m 的凹部，均跨边界连续保留。'
                    '邻线有渐进的位置变化，但 109.30→109.35 没有独有的新断口或新增回钩。两个 PF 本身形状不同，差别却不只存在于小岛内。',
        evidence='109.30→109.35，PF16 的 in_CC 从否变是、CC_fraction 从 0.7903 到 1；'
                 '同侧冻结 CC 方块从约 (8.101,1382.656) 移到 (5.340,1382.279)，而真端点仅从约 (5.348,1382.325) 到 (5.309,1382.277)。'
                 '这是原始几何平顺而核心区判据发生跳变的证据；“边界判据敏感”是本轮推断，未重算或确定更深层成因。',
        recommendation='可疑，建议将 3V 岛并回 PF19 主区',
        limit='须结合 F01_R 两侧一起审核；本轮不执行合并。'),
    'F01_R': dict(
        purpose='从另一侧核对 3V 岛的独立性，避免只看岛内。',
        observation='109.40、109.45、109.50、109.55 的 PF16 都是同类台阶状缓坡，PF19 都保留浅凹后回升。'
                    '离开小岛时没有恢复一种截然不同的结构，局部走势是连续变化。PF16 浅坡的同高程交点对水平平移较敏感，右图的展开量不应当作形状误差，需优先看左图。',
        evidence='109.45→109.50，PF16 in_CC 从是变否，CC_fraction 1→0.9766，出口延展 0.2577→0 m；'
                 'PF19 的出口证据均为封顶 1 m。原始 PF16 端部仍存在，并不是在 109.50 突然消失。',
        recommendation='可疑，建议将 3V 岛并回 PF19 主区',
        limit='左右边界均缺乏小岛独有的结构变化证据；该判断不是认定 PF16 为错误曲面。'),
    'F02_L': dict(
        purpose='核对 5V 岛进入处；与右边界的端部回钩分开判断。',
        observation='115.95–116.15 的 PF16 上端在 z≈1384.6–1384.7 m 附近逐步变化，下方斜坡对齐后接近；'
                    'PF19 的缓平台转下降也连续保留。左边界未见与首选面组突然切换同等明显的拓扑变化。',
        evidence='116.00→116.05，PF16 in_CC 否→是，CC_fraction 0.9768→1；'
                 '其上侧 CC 边界约从 (1.401,1384.140) 移到 (1.247,1384.595)，真端点高程仅从 1384.648 到 1384.679 m。',
        recommendation='左边界可疑，整个 5V 岛暂不建议直接并回',
        limit='必须连同 F02_R 的真实回钩/碎段变化人工审核，不能把左边界结论推广为整岛无结构差异。'),
    'F02_R': dict(
        purpose='重点检查 116.25–116.35 m 的真实端部形态和小碎段。',
        observation='图 B 的 z≈1384.2–1385.0 m 出现明显的横向回钩。116.15 的 PF19 仍主要是平台接斜坡，'
                    '116.25 增加独立小片段，116.30/116.35 的原始分支包含向上伸出再折回的局部形态；PF16 的端部也在同步展开。'
                    '下方斜坡相似不代表端部相同，这一例不能与 3V 岛一起简单当作噪声。',
        evidence='116.25 的冻结成员 PF16=[B1,B4]、PF19=[B2,B7]，其中 B4/B7 不合格但在图 A/B 保留；'
                 '116.30 各为单条 B1/B2。与此同时 PF16 in_CC 是→否、CC_fraction 1→0.9920、出口 0.3275→0 m。'
                 '分支组成和局部形态确有变化，但这些变化不能单独证明 PF19 一定比 PF16 更真实。',
        recommendation='证据不足，保留人工审核',
        limit='建议暂保留这处待审边界，优先核对回钩与碎段的原始网格身份；不因 CC 的很小数值变化自动保留，也不按岛宽直接吞并。'),
    'F_BIG': dict(
        purpose='检验 S007/S008 是否真能充当“明显结构差异”的对照。',
        observation='图 A 给出 z=1375–1405 m 的较大上下文，图 B 放大 PF16 上端。131.00–131.15 的同组曲线依次平移，'
                    '对齐后转折位置和下降走势接近；131.05→131.10 未见明显新断口或曲线类型突变。'
                    '所以这次图像不支持预先把它当成明显结构差异的正例。',
        evidence='PF16 in_CC 是→否、CC_fraction 1→0.9861，出口 1→约 0.326 m；'
                 '真端点高程 1404.100→1404.033 m，属于逐线变化。PF19 同时保持 in_CC=否、出口=1 m。',
        recommendation='证据不足，继续人工审核大区边界',
        limit='边界附近平顺不能否定 153V 整段的选择依据。本轮只复核该交界，不建议用小岛合并规则吞掉大区。'),
    'F03_L': dict(
        purpose='检查 8V 岛进入处是否出现稳定的新细节。',
        observation='PF16 的短平头接下降段与 PF19 的连续下降折线在边界两侧都存在，邻线同组转折一致性较好。'
                    '138.00→138.05 的端点和折点没有突然分裂成另一类形状。',
        evidence='PF16 through_region 从否变是，但 in_CC 两侧均为否；端点高程约 1402.360→1402.385 m。'
                 'PF19 贯穿均为否、出口均为 1 m。投票改变与贯穿布尔值改变同时出现，不能把“贯穿”直接解释成新曲面出现。',
        recommendation='可疑，建议将 8V 岛并回 PF19 主区',
        limit='结合右边界及岛内反向票审核；建议不改写原始分支几何。'),
    'F03_R': dict(
        purpose='检查 8V 岛退出处，并照实显示岛内的不一致票。',
        observation='138.40→138.45 的 PF16 上端和下降段几乎沿同一趋势继续，PF19 的主要折点也延续。'
                    '138.20 相比后几条有局部斜率差异，但并未只在 138.45 形成一个突然的结构边界。',
        evidence='PF16 贯穿是→否，in_CC 都为否。图 A/C 还明确保留 S009 内 138.25、138.30 的 PF19 反向票：'
                 '该 8V 子区并不是每条 V 都支持 PF16。两条反向票的 PF16 贯穿也为否。',
        recommendation='可疑，建议将 8V 岛并回 PF19 主区',
        limit='左右图共同支持先做邻域一致性整理的审核方向，尚不构成人工真值。'),
    'D01_L': dict(
        purpose='核对 92.45–92.55 的 PF19 原始反向票是否值得从 PF23 主区拆出。',
        observation='PF19 在 z≈1412.8 m 有真实上端点，PF23 继续向上；这一差异在反向票之前已经存在。'
                    '92.35–92.50 各自保留近竖向剖面上的细小折转，进入反向票段时未新增独立结构。'
                    '右图横轴放大到厘米至分米量级，不能把放大的折线误读为米级台阶。',
        evidence='92.40→92.45，PF19 贯穿否→是，PF23 仍为否；两组 in_CC 均为否。'
                 'PF23 原始剖线未在该边界断开。冻结 winner 全部为 PF23，本轮没有新建 PF19 子区。',
        recommendation='支持维持 PF23 主区，不另拆 3V 段',
        limit='是对这处反向票的局部支持，不能由此证明 0.99 对所有区域都正确。'),
    'D01_R': dict(
        purpose='检查这段反向票结束时是否有对应的结构恢复。',
        observation='92.50–92.65 的 PF19 真端点高程几乎不变；局部折线细节沿邻线变化，PF23 仍向上连通。'
                    '92.55→92.60 没有与反向票结束同步的主体断裂，不能仅凭个别细节差异建立独立子区。'
                    '图 A 同时保留 92.50/92.55 在 z≈1403.46 m 的小闭合碎段，它们 MBG 不合格；'
                    '92.60 以后这处小闭环已接入 PF19 原始长分支，确有端部连通性变化，不能称为完全没有几何变化。'
                    'PF23 则仍保持同类终止形态。这处下端变化请看 A，B 专门放大上端投票敏感处。',
        evidence='PF19 贯穿是→否、CC_fraction 0.9907→0.9798；其出口仍有约 0.04–0.05 m，PF23 的出口一直为 1 m。'
                 '本线票恢复 PF23，而子区 winner 从未改变。',
        recommendation='支持维持 PF23 主区，不另拆 3V 段',
        limit='倾向保持 PF23 的邻线一致性，但请人工确认下端闭环是否是真实结构；不将该段一概认定为纯投票噪声。'),
    'D02_L': dict(
        purpose='检查第二段 135.35–135.45 的反向票是否形成新的局部形态类型。',
        observation='PF19 的上端短斜段接长缓斜段、PF23 的连续折坡都跨边界保留，对齐后重合度较高。'
                    '135.30→135.35 只有端部长度和折点位置的渐进变化，未见该 3V 段独有的结构。',
        evidence='PF19 贯穿否→是，但 CC_fraction 0.9470→0.9476，in_CC 均为否；'
                 '端点高程约 1413.321→1413.355 m。PF23 仍为原子区首选。',
        recommendation='支持维持 PF23 主区，不另拆 3V 段',
        limit='这处证据支持当前 D 的整理结果，不等于验证 0.99 为最优阈值。'),
    'D02_R': dict(
        purpose='检查反向票恢复位置，连同紧邻的再次反向票一起呈现。',
        observation='135.40–135.55 的 PF19 长缓斜段和 PF23 剖面均连续；PF19 在 z≈1412.5–1412.8 m 的小折回逐步增强。'
                    '该细节并没有整齐地只覆盖 135.35–135.45，不能据这三票划出一个独立物理区。',
        evidence='135.45→135.50，PF19 贯穿是→否，本线票 PF19→PF23；图 A/C 又保留 135.55、135.60 的 PF19 票。'
                 '所以右侧并非所有本线都一致支持 PF23，但冻结子区 winner 均为 PF23。',
        recommendation='支持维持 PF23 主区，保留端部细节供审核',
        limit='不隐藏小折回和再次反向票；当前图片不支持按这些短票段直接拆成多个子区。'),
    'E01': dict(
        purpose='同时展示 PF16、PF19、PF34，核对首选切换是否有几何/连通性证据。',
        observation='PF16 下游的凹部和下降段在四条线中相近，变化集中在 u≈9.4–9.6、z≈1385.7–1386.1 m 的端部。'
                    '104.80/104.85 上方已有一个独立短片段，并非到 104.90 才凭空长出；104.90 后该局部由一条原始分支连续表达。'
                    'PF19 在同窗仍有多个折返，PF34 的左端仍结束于约 1385.7–1385.8 m，上游覆盖范围不同。',
        evidence='PF16 冻结成员 [B1,B2]（仅一条合格）→[B1]，in_CC 否→是，CC_fraction 0.9902→1，'
                 '出口延展 0.0185→0.8118 m。合格 witness 的上端高程 1385.723→1386.028 m，'
                 '要连同此前独立短片段一起理解，不能解读为整个真实表面突变。PF19 的本区格网覆盖约 0.857，PF34 也没有相同的 CC 优势。',
        recommendation='初步建议保留该边界',
        limit='依据是缓存中可见的分支连通性与有效端部范围变化；不代表已证明三维物理面身份，建议重点人工确认这一短片段是否应视为连续表面。'),
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def table(headers, rows):
    return ('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
            ''.join('| '+' | '.join(str(x).replace('|','/') for x in row)+' |\n' for row in rows)+'\n')


def pf(value):
    return 'AMB' if value is None or pd.isna(value) else f'PF{int(value)}'


def crossings(branches, track, z, box):
    values = []
    for branch in branches.values():
        if track not in branch['physical_groups']:
            continue
        for a,b in boundary_segments(branch['points_uz'], box):
            if abs(b[1]-a[1]) < 1e-12:
                if abs(z-a[1]) < 1e-9:
                    return []  # horizontal overlap cannot be a unique anchor
                continue
            if min(a[1], b[1]) <= z <= max(a[1], b[1]):
                values.append(float(a[0]+(z-a[1])*(b[0]-a[0])/(b[1]-a[1])))
    unique = []
    for value in sorted(values):
        if not unique or abs(value-unique[-1]) > 1e-8:
            unique.append(value)
    return unique


def align(profiles, track, box):
    # Display convention only. Do not turn alignment residuals into a score.
    candidates = np.linspace(box[2]+.1*(box[3]-box[2]), box[3]-.1*(box[3]-box[2]), 81)
    candidates = sorted(candidates, key=lambda z: abs(z-(box[2]+.35*(box[3]-box[2]))))
    for z in candidates:
        all_u = [crossings(p['branches'], track, z, box) for p in profiles]
        if all(len(values)==1 for values in all_u):
            return dict(z_ref=float(z), offsets_u={str(p['s']):values[0] for p,values in zip(profiles,all_u)},
                        rule='unique within displayed local ROI; u shift only, z unchanged')
    return dict(z_ref=None, offsets_u={str(p['s']):0. for p in profiles},
                rule='no unique common crossing; original coordinates retained')


def branch_evidence(branch):
    points = branch['points_uz']
    knots = [[float(np.interp(a, branch['arc_positions'], points[:, k])) for k in (0,1)]
             for a in (branch['CC_start_arc'], branch['CC_end_arc'])]
    return dict(MBG=bool(branch['MBG']), physical_groups=branch['physical_groups'],
                endpoints_uz=points[[0,-1]], CC_boundaries_uz=knots,
                CC_start_arc=branch['CC_start_arc'], CC_end_arc=branch['CC_end_arc'])


def write_report(out, manifest):
    text = '# 局部子区边界复核报告\n\n'
    text += ('本轮只补制图和人工可读的初步判断。使用 20260923_105650 的 B0/B1 冻结结果与 '
             '20260923_094440 的 A3 评分、原始 observed 缓存；不重跑识别，不合并子区，不更改主轨或阈值。\n\n'
             '**核对编号：**R00806 的 3V、5V、8V 小岛实际为 S002、S004、S009。D 的两段是 S001 内的原始反向票，'
             '当前 winner 一直是 PF23，并不存在独立 PF19 子区。\n\n')
    text += '## 怎么看图\n\n'
    text += ('1. **先看 A 第一排，再看各 PF 分行。**同一张 A 的各格坐标范围完全相同。灰线保留所有原始剖线；'
             '固定色 PF16 蓝、PF19 橙、PF23 绿、PF34 紫。粗线是当前子区首选面组的原始候选分支，细线是竞争者。'
             '尚未生成新的 winner branch 或最终主轨，不能把粗线当作已接好的识别结果。所有进入显示窗的竞争分支都画出，'
             '并非只画 A3 的 witness。点线表示 MBG 不合格的原始分支；编号 B 只在各自测线内有效，不能跨线按 B 编号匹配。\n'
             '2. **圆点是真分支端点，空心方块是旧结果的 CC 边界。**CC 是冻结竞争核心区的弧长边界；方块移动不等于原始曲线断裂。'
             '出画框不是原始断点；各图均为显示裁剪，不会修剪输入。横纵显示比例可能不同，比较坐标值而非屏幕夹角。\n'
             '3. **B 左列看绝对位置，右列看同组邻线形状。**每组取 4 条关键测线，用实线、长虚线、短虚线、点划线区分。'
             '保持高程不变，在同一局部高程且各线都只有一个交点时把该交点平移到 u=0；不缩放、不旋转、不补线。'
             '不同 PF 各自对齐，因此不能用右列判断两个 PF 的真实间距。找不到唯一共同锚点就照实保留原坐标。'
             '锚点和每条线的平移量在 C 中列出。\n'
             '4. **C 区分本线 vote 与子区 winner。**它们可以不同，AMB 也单列。附带的贯穿/CC/延展数值直接来自冻结 A3，'
             '只解释投票变化，不在本轮重新评分。这些图没有人工真值，不能给出识别准确率。\n\n')
    text += '## 初步建议一览\n\n'
    text += table(['边界批次','位置 s (m)','初步建议'], [[c['id'],f"{c['boundary']-.05:.2f} / {c['boundary']:.2f}",
                    OBSERVATIONS.get(c['id'],{}).get('recommendation','待图像复核')] for c in manifest['cases']])
    text += ('“建议并回”只表示所展示局部没有充分的独立分区证据，不表示两个物理面组相同；'
             '它们本来就可能是不同表面。要判断的是首选面组是否有理由在这几条线内突然改变。建议均未写回算法。\n\n')
    for c in manifest['cases']:
        obs = OBSERVATIONS.get(c['id'], {})
        text += f"## {c['id']} · {c['region']} · {c['label']}\n\n"
        text += f"**边界：**{c['boundary_text']}。图 A：{c['profile_s'][0]:.2f}–{c['profile_s'][-1]:.2f} m；"
        text += f"图 B 关键测线：{', '.join(f'{s:.2f}' for s in c['key_s'])} m。\n\n"
        text += f"**复核目的：**{obs.get('purpose','比较边界两侧同面组的原始形态、端点和投票变化。')}\n\n"
        text += f"![{c['id']} A](figures_boundary_review/{c['id']}_A.png)\n\n"
        text += f"![{c['id']} B](figures_boundary_review/{c['id']}_B.png)\n\n"
        text += f"[图 C：测线、分支和冻结判据表](tables_boundary_review/{c['id']}_C.md)\n\n"
        text += f"**初步观察：**{obs.get('observation','待逐图检查。')}\n\n"
        text += f"**冻结判据的对应现象：**{obs.get('evidence','见 C 表。')}\n\n"
        text += f"**建议：{obs.get('recommendation','待图像复核')}。**{obs.get('limit','')}\n\n"
    text += '## 来源、复用和执行范围\n\n'
    text += (f"共 {len(manifest['cases'])} 组边界、24 张 A/B 图、12 份 C 表，读取 {manifest['profiles_read']} 条缓存侧线，"
             f"制图与写出用时 {manifest['elapsed_s']:.1f} 秒。此耗时是制图耗时，不是识别性能测试。\n\n"
             '扩展 `scripts/06_visualization/locc_local_review_plot.py`，实际复用其论文字体/样式 `_paper_helpers`、'
             '精确裁剪 `_clip_xyz_lines`、线段集合 `_lines` 和导出 `_save`；沿用原 FaceTrack 图册的面组分行、'
             '邻线分列、共用坐标和真端点标记。新增脚本仅编排冻结结果，未导入识别/投票入口。\n\n'
             '只保留必要的局部剖面内容；没有重复模型读取、切面、A1/A2/A3/B0/B1、换轨或 P2。'
             '详细数据和输入 SHA256 见 [manifest.json](manifest.json)。最低检查记录见 [ARTIFACT_CHECK.json](ARTIFACT_CHECK.json)。'
             '原始三维表面身份、未显示范围的连续性和边界是否最终保留，仍由人工复核。\n')
    (out/'LOCAL_BOUNDARY_REVIEW_REPORT.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/local_boundary_review'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_report(out, read_json(out/'manifest.json'))
        print(out/'LOCAL_BOUNDARY_REVIEW_REPORT.md'); return
    start = time.perf_counter()
    local_manifest = read_json(LOCAL/'manifest.json')
    protected = [Path(p) for p in local_manifest['source_hashes']]
    protected += list(LOCAL.glob('*.json'))
    protected += [ROOT/'scripts/04_structure_recognition/facetrack_local_preference.py',
                  ROOT/'scripts/99_experiments/run_facetrack_local_preference.py']
    before = {str(p):digest(p) for p in protected}
    subregions = read_json(LOCAL/'decision_subregion_index.json')['subregions']
    decisions = {d['subregion_id']:d for d in read_json(LOCAL/'subregion_track_decisions.json')['decisions']}
    member = {(s['parent_region_id'], int(si)):s for s in subregions for si in s['s_indices']}
    frame = pd.read_parquet(SOURCE/'per_slice_track_scores.parquet')
    rows = {(r['region_id'], int(r['s_index']), int(r['FaceTrack'])):r for r in frame.to_dict('records')}
    cache = Path(read_json(SOURCE/'manifest.json')['index_cache'])/'slices'
    data = {}
    def profile(region, s):
        si = round(s/.05); s = round(si*.05, 2)
        if s not in data:
            path = cache/f'{s:.2f}.pkl'
            before[str(path)] = digest(path)
            with path.open('rb') as stream:
                data[s] = pickle.load(stream)
        sub = member[(region,si)]
        winner = decisions[sub['subregion_id']]['dominant_FaceTrack']
        tracks = sorted(int(t) for t in sub['competing_FaceTracks'])
        scores = {t:rows[(region,si,t)] for t in tracks}
        one = next(iter(scores.values()))
        return dict(s=s, si=si, subregion=sub['subregion_id'], winner=winner, tracks=tracks,
                    vote=one['slice_vote'], vote_label=pf(one['slice_vote']),
                    ambiguous=one['slice_ambiguous'], scores=scores, branches=data[s]['branches'])
    records = []
    for template in CASES:
        case = dict(template)
        si = round(case['boundary']/.05); half=case['neighbor_count']//2
        profiles = [profile(case['region'], round(j*.05,2)) for j in range(si-half, si+half+1)]
        keys = [profile(case['region'], s) for s in case['key_s']]
        left, right = profile(case['region'], case['boundary']-.05), profile(case['region'], case['boundary'])
        case['boundary_text'] = (f"{left['s']:.2f} {left['subregion'].split('_')[-1]}/{pf(left['winner'])}"
                                 f" → {right['s']:.2f} {right['subregion'].split('_')[-1]}/{pf(right['winner'])}"
                                 + ('（子区内原始反向票边界）' if case['region']=='R00204' else ''))
        case['tracks'] = sorted(set(t for p in profiles+keys for t in p['tracks']))
        case['profiles'],case['key_profiles'] = profiles,keys
        case['alignment'] = {t:align(keys,t,case['detail_box']) for t in case['tracks']}
        figures = out/'figures_boundary_review'; tables = out/'tables_boundary_review'; tables.mkdir(exist_ok=True)
        plot_boundary_neighbors(case, figures/f"{case['id']}_A.png")
        plot_boundary_alignment(case, figures/f"{case['id']}_B.png")
        all_profiles = {p['s']:p for p in profiles+keys}
        ctext = f"# 图 C · {case['id']} / {case['region']} · {case['label']}\n\n{case['boundary_text']}\n\n"
        ctext += table(['s (m)','所属子区','winner','竞争 PF','slice_vote','ambiguous','出现在'],
                       [[f'{s:.2f}',p['subregion'],pf(p['winner']),', '.join(pf(t) for t in p['tracks']),p['vote_label'],
                         bool(p['ambiguous']),('A ' if s in [x['s'] for x in profiles] else '')+('B' if s in case['key_s'] else '')]
                        for s,p in sorted(all_profiles.items())])
        ctext += '## A3 冻结证据（只读）\n\n'
        ctext += table(['s','PF','成员/合格分支','witness','贯穿','全部竞争弧在CC','CC占比','格网覆盖','出口延展m'],
                       [[f'{s:.2f}',pf(t),f"{r['member_branches']} / {r['eligible_branches']}",r['witness_branch'],
                         bool(r['through_region']),bool(r['in_CC']),f"{r['CC_fraction']:.4f}",
                         f"{r['cell_coverage_fraction']:.4f}",f"{r['exit_continuation_m']:.4f}"]
                        for s,p in sorted(all_profiles.items()) for t,r in p['scores'].items()])
        visible = {}
        for s,p in sorted(all_profiles.items()):
            visible[str(s)] = {str(t):[int(bid) for bid,b in p['branches'].items()
                                      if t in b['physical_groups'] and len(boundary_segments(b['points_uz'],case['box']))]
                               for t in case['tracks']}
        ctext += '## 原始分支可见性\n\n编号属于本条测线；显示全体候选，不仅 witness。图 B 只在较小窗口内显示。\n\n'
        ctext += table(['s','PF','图 A 窗内全部 B'], [[s,pf(t),str(bids)] for s,groups in visible.items() for t,bids in groups.items()])
        ctext += '## 图 B 对齐记录\n\n'
        ctext += table(['PF','锚点 z (m)','各 s 减去的 u (m)','规则'],
                       [[pf(t),a['z_ref'],a['offsets_u'],a['rule']] for t,a in case['alignment'].items()])
        (tables/f"{case['id']}_C.md").write_text(ctext, encoding='utf-8')
        record = {k:v for k,v in case.items() if k not in ('profiles','key_profiles')}
        record['profile_s'] = [p['s'] for p in profiles]
        record['visible_branches_A'] = visible
        record['frozen_profiles'] = [{k:v for k,v in p.items() if k!='branches'} | {
            'branch_evidence':{bid:branch_evidence(b) for bid,b in p['branches'].items()
                               if set(b['physical_groups']) & set(case['tracks'])}}
            for _,p in sorted(all_profiles.items())]
        records.append(record)
        print(f"Rendered {case['id']}", flush=True)
    manifest = plain(dict(cases=records, source=str(SOURCE), local_preference=str(LOCAL), profiles_read=len(data),
                          frozen_hashes_before=before, elapsed_s=time.perf_counter()-start,
                          mesh_reads=0, A1_runs=0,A2_runs=0,A3_runs=0,B0_runs=0,B1_runs=0,route_operations=0))
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),encoding='utf-8')
    write_report(out, manifest)
    print(f"OUTPUT={out}",flush=True)


if __name__ == '__main__':
    main()
