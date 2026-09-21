"""Write evidence-based Chinese reports from the completed, frozen audit outputs."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import pickle
import re
from validate_physical_topology import output_path

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])+'\n'


def main(out):
    def js(name):return json.loads((out/name).read_text(encoding='utf-8'))
    def rows(name):
        with (out/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    def write(name,text):(out/name).write_text(text.strip()+'\n',encoding='utf-8')
    top=js('topology_summary.json');face=js('face_validation_summary.json');hr=js('horizontal_provenance_summary.json')
    routes=js('local_route_summary.json');mesh=js('face_adjacency_manifest.json');base=js('BASELINE_MANIFEST.json')
    vv=rows('neighbor_route_face_track_consistency.csv');shadow=rows('shadow_face_track_recommendations.csv')
    branches=rows('case_branch_table.csv');forks=rows('COMPONENT_WIDE_FORK_OVERREACH.csv')
    current=[r for g in routes['regions'] for r in g['rows'] if r['version']=='CURRENT_ROUTE']
    before=[r for g in routes['regions'] for r in g['rows'] if r['version']=='BEFORE_LOCAL']
    paired={(r['region_id'],r['slice_key']):r for r in before}
    tiers=Counter()
    for row in vv:
        chains=json.loads(row['witness_face_chains'])
        if chains:
            hops=min(len(c)-1 for c in chains)
            tiers['EXACT_FACE' if hops==0 else 'EDGE_ADJACENT_FACE' if hops==1 else 'LOCAL_FACE_CHAIN']+=1
    support=Counter(int(r['support_sides']) for r in shadow)
    max_connector=max(r['max_connector_m'] for r in current)
    max_r=max(r['max_combined_R'] for r in current)
    evidence={'VV_strongest_witness_tier':dict(tiers),'shadow_support_sides':dict(support),
              'shadow_all_MBG':all(r['candidate_MBG']=='True' for r in shadow),
              'CURRENT_ROUTE_count':len(current),'BEFORE_LOCAL_count':len(before),
              'maximum_current_connector_m':max_connector,'maximum_current_combined_R':max_r}
    write('face_evidence_relation_summary.json',json.dumps(evidence,ensure_ascii=False,indent=2))
    comparison=[]
    for rid,s in [('A','0.85'),('B','78.35'),('B','78.40'),('C','103.70'),('G','122.95'),('H','116.25')]:
        a=paired[(rid,s)];b=next(r for r in current if r['region_id']==rid and r['slice_key']==s)
        comparison.append([s,str(a['sequence']),f"{a['route_z_extent'][0]:.3f}–{a['route_z_extent'][1]:.3f}",str(b['sequence']),f"{b['route_z_extent'][0]:.3f}–{b['route_z_extent'][1]:.3f}"])
    comparison_table=table(['s（m）','修复前分支序列','修复前 Z 范围（m）','修复后分支序列','修复后 Z 范围（m）'],comparison)
    title='2026-09-21｜物理拓扑修复与 FaceTrack 首轮验证'
    scope="""全区完成 **2800 条纵向测线的 branch inventory 重算**。实际主轨只对 **12 个局部区域、每区 5 条邻线**做修复前后配对重算，共 120 次求解，其中 60 条为本轮 `CURRENT_ROUTE`。FaceTrack 审核覆盖这 12 个区域，没有重算全部 2800 条主轨。

前一轮全区结果保存在 `outputs/directional_full_census/20260920`，本轮不覆盖。配对重算的两版都重建同范围的 H/V 图，给每条目标 V 至少 ±1 m 的邻线证据范围；不能把本轮局部结果当成旧全区运行的逐位复现。12 区是问题导向选样，不是随机准确率样本。"""
    identity="""`CURRENT_ROUTE` 是拓扑修复后、原有主轨算法真实求出的路线。蓝色实线表示其中的原始 observed 几何，橙色虚线表示算法连接段。`SHADOW_FACE_TRACK_RECOMMENDATION` 只是从已有候选中找出的邻线来源支持，不是新增几何，不会写回路线。

FaceTrack 的身份来自同一份 **filtered mesh** 的三角面索引、真实共享边和局部面链，不按 FaceID 数值相近合并，不按几何接近焊接。每个编号只在自己的案例 ROI 内有效；例如 D_T1 与 F_T1 没有对应关系。它说明模型网格的局部来源关系，不能独自证明哪一张面是真实地质目标面。"""
    statuses=[
      ('SELF_LOOP_PHYSICAL_TOPOLOGY_FIX','SUPPORTED','32 条自环不再参与物理拓扑；全区几何与完整来源保留。'),
      ('LONG_BRANCH_RECOVERY','SUPPORTED','恢复 28 条 OPEN/MBG 候选链；原 27 条已知测线中的 52 段长核归入恢复链。'),
      ('FACE_PROVENANCE_EXACT_MATCH','SUPPORTED','6775 个已建立几何对应的 V/H 交点对均具有共同 FaceID；另有 49 个 H 交点未关联。'),
      ('FACE_ADJACENCY_TRACKING','SUPPORTED','相邻 V 同来源证据中，637 条最短见证为真实共享边，316 条为局部短链。'),
      ('LOCAL_FACE_TRACK_DISCRIMINATION','PARTIALLY_SUPPORTED','7 区能区分采样到的 MBG 竞争分支；5 区仅见单一来源，不能检验多面区分能力。'),
      ('MISSING_ROUTE_FACE_TRACK_RECOVERY_SIGNAL','SUPPORTED','G/H 等处有未选、MBG 合格、与邻线主轨同来源的实测候选；尚未执行恢复。'),
      ('NEIGHBOR_ROUTE_FACE_TRACK_CONSISTENCY','PARTIALLY_SUPPORTED','411 条仅编号变化，但仍有 161 条来源切换、38 条来源集合不一致/歧义。'),
      ('READY_FOR_LOCAL_JOINT_FACE_TRACK_LABELING','YES','仅建议下一轮开展有限局部联合标注实验；不表示已可接入生产或自动强制补齐。')]
    status_table=table(['结论项','状态','依据与适用范围'],statuses)
    counts=table(['相邻 V 的采样分类','数量','含义'],[
      ('CONSISTENT_FACE_TRACK',1910,'已选来源一致，且存在 ≤8 跳局部共享边见证。'),
      ('BRANCH_ID_CHANGE_ONLY',411,'分支编号不同，但来源集合一致且有局部见证。'),
      ('FACE_TRACK_IDENTITY_SWITCH',161,'两条邻线已选来源集合不相交。'),
      ('AMBIGUOUS_FACE_TRACK',38,'来源集合部分重叠但不相同，或来源标注不唯一。'),
      ('NO_FACE_EVIDENCE',248,'一侧主轨缺失或缺乏可用短链；不等于原始候选不存在。')])
    performance=table(['阶段','实测墙钟时间','范围'],[
      ('物理拓扑清查',f"{top['seconds']:.2f} s",'受影响切面检查及全 2800 条分支清查、写盘'),
      ('三角面邻接构建',f"{mesh['seconds']:.2f} s",'3,790,662 faces 的 shared-edge adjacency'),
      ('局部主轨配对求解',f"{routes['seconds']:.2f} s",'120 次求解、4 workers，含局部图与结果审计'),
      ('H 完整来源恢复',f"{hr['seconds']:.2f} s",'2,722,036 条 clean edges，生成 6824 个局部交点'),
      ('FaceTrack 局部审核',f"{face['seconds']:.2f} s",'12 区，已缓存邻接、分支、H 交点')])
    write('PHYSICAL_TOPOLOGY_FIX_REPORT.md',f"""# {title}：物理拓扑分报告

零长度自环确实是部分缺轨的上游原因。此前 A→A 在同一节点重复贡献两次 incidence，使正常链节点的 degree 从 2 变为 4，进而让整分量变成 FORKED、失去 OPEN/MBG 候选资格。本轮保留该记录及全部来源，只让它不参与物理图。

## 修改与保留

`vertical_profile_canonicalization.py` 新增 `topology_active_edge`、`physical_degree`、`physical_components`、`physical_branches`。原 `degree/components/branches` 与 canonical 统计保留，`canonical_degree` 可明确访问原度数。`dominant_observed_branch.py` 的开放/闭合/真实叉点判定和遍历改用物理图；有正常物理叉点的整分量分类语义本轮保留。

节点、edge ID、observed 坐标、完整 `source_face_ids`、`source_segment_indices` 没有删除或改写。端点诊断也改用物理度数。没有调整 MBG、ASC、DRS、10 mm、R_syn、future cost 参数。

## 全区清查

{table(['指标','结果'],[
('V 输入',2800),('零长 self-loop / 受影响节点','32 / 32'),('受影响切面 / component','30 / 31'),
('FORKED 分支数，修复前 → 后','109 → 19'),('恢复 OPEN / MBG 分支','28 / 28'),
('已知 27 条切面中的旧长核片段恢复',52),('正常 OPEN 丢失','0'),('错误 component merge','0'),('几何与完整来源保持','全 2800 条检查通过')])}

52 指旧分割后的长核片段，28 指修复后重新遍历形成的完整候选链，两者不能相减解释为损失。FORKED 总数还受遍历合并影响，也不等于“90 个错误已解决”。恢复映射见 [recovered_branches.csv](recovered_branches.csv)。

## 实际路线变化

{comparison_table}

分支编号可能重排，上下版本不能直接按相同整数比较身份。表中 Z 范围只是整条主轨端点范围，不能代表中间每个局部都正确。

0.85、78.40 的正常长链已经恢复资格并被实际主轨利用。78.35 自身没有 self-loop，修复前已有约 101.50 m 的 OPEN 候选；它在本轮恢复反映邻线分支清查变化影响了局部图和选择证据，不能称为“78.35 自环直接修复”。103.70 的下部在当前配对试验里也已恢复，不能再计成 FaceTrack 补点收益。

![0.85 自环及配对路线](figures/topology/0.85_before_after.png)

左栏放大自环所在节点；右两栏看统一局部坐标下的实际主轨。自环可能位于窗口之外，但整分量污染会使远处长核也失去资格，故左栏与右栏不共用 Z 范围。

![78.40 自环及配对路线](figures/topology/78.40_before_after.png)

## 尚存真实 fork 问题

{table(['s（m）','branch','ASC（m）','fork 在 ASC 内'],[(r['s'],r['branch_id'],f"{float(r['ASC_m']):.3f}",r['fork_in_core']) for r in forks])}

这 7 项仍受整分量 FORKED 类型门控。检查到的真实 fork 在分支端点，不在 ASC 内；它们是需审查的门控范围，尚不能断言是与叉点完全无关的长链，更不能直接放行。详见 [COMPONENT_WIDE_FORK_OVERREACH.csv](COMPONENT_WIDE_FORK_OVERREACH.csv)。

## 验证

拓扑新增 7 项加既有相关回归合计 44 项通过；真实 0.85、78.40 fixture 已纳入。全区断言确认坐标/来源保持、物理分量不合并、原正常 OPEN 不丢失。实际 60 条 CURRENT_ROUTE 均通过几何来源、连续性、重叠与原连接上限审计；最大连接 {max_connector*1000:.6f} mm，最大局部合成比例 {max_r:.9f}（约 {max_r*100:.5f}%）。这是本地验证路线的统计，不是全 2800 条路线的新性能或准确率。

{scope}
""")
    guide="""## 看图方法

每个案例先看相邻 5 条 V 的同坐标图，再看四栏来源图；显示约 6.2 m 的高程局部，不再用整条路径挤压细节。

1. **邻线图**：灰线是原始候选，蓝线是当前实际主轨，橙色虚线是算法连接段。每栏上方标明 s 和本窗口内已选来源。某栏只有灰线，说明窗口内有原始几何而当前主轨没有进入。
2. **四栏图①**：重点切面的原始候选几何；②：同一切面的实际主轨；③：同一位置按局部 FaceTrack 着色。黑圈只标一部分有邻线来源支持、当前未保留的候选，圈不是补出的点，也不是最终建议路线。
3. **四栏图④**：固定一个 H 高程后，比较 5 条 V 的候选来源和实际选择。彩色空圈表示该来源有 H/V 交点，圈中的蓝色实心点表示当前选中了它。纵轴是 FaceTrack **类别轴**，不是 u、不是面间距离；这样毫米级重叠也不会遮盖身份差异。同来源的多个交点在此栏合并显示，所以它也不表示细节数量。
4. ①②③使用相同 u/z 范围，横坐标 u 为径向偏移（m），纵坐标 z 为高程（m）。图④横坐标改为沿模型的 s（m），不能把它当另一个切面。若存在来源切换，④选择切换采样高程的中间一层；否则用案例中心附近 H 层。具体高程写在图题。

分支表中的 MBG/ASC 是整分支属性，FaceID 数量是当前 ROI 内的完整来源去重计数。“当前选中”表示至少一段被选，不保证整分支或整个窗口保留；“邻线来源”是左右邻线在窗口内已选来源的并集。是否在同一高程支持应查交点明细，不能只靠这张概览表。
"""
    explanations={
    'A':('0.85：自环导致的长链排除已修复','实际路线从仅有上部变为上下分支衔接。此窗有两类面来源，不是所有邻线都完全统一：1403.986 m 这一 H 层仍有两对邻线出现来源切换。优先确认已恢复的下部是目标曲面，再判断极短切换是否合理。'),
    'B':('78.35 / 78.40：候选恢复与邻线影响要区分','78.40 的自环修复恢复长链；78.35 自身无自环，但邻线清查变化后上部也被当前算法保留。本窗口五条线都属于同一局部 FaceTrack。120 条 branch-ID 差异属于编号变化，不是换了模型来源。'),
    'C':('103.50–103.70：原缺轨已在拓扑修复后恢复','103.70 当前实际主轨已覆盖此窗口，五条邻线来自同一局部来源，240 条邻线采样比较一致。本例不再是“FaceTrack 能补出缺轨”的证明；它证明修复上游候选资格后，原算法已经能选择这部分，FaceTrack 只是独立核对。但图中 103.65/103.70 在 z≈1362–1364 m 的折返比 103.50 更明显，说明来源一致仍不等于细节表达一致；本轮未判定这些折返是真实结构还是重建伪影。'),
    'D':('122.15–122.35：多数编号差异，仍有局部来源切换','171 条采样是 branch-ID 变化但同来源；18 条是真来源切换，分别在 122.15/122.20 与 122.30/122.35 两对邻线、z=1410.186–1410.986 m，各 9 层。不能仅看中心 1411.081 m 就认定全窗口一致。四栏图④特意取切换范围中的一层。'),
    'E':('104.20–104.40：三个局部来源，选择仍不完全一致','发现 3 个局部 FaceTrack；22 条采样是来源集合完全不同，34 条是集合部分重叠造成的歧义。104.25/104.30 在 z=1383.586–1384.986 m 有 15 层切换，其余见事件表。宽 u 范围来自真实展开/折返，请结合相邻图和来源表看，不能按线条复杂就删掉某个候选。'),
    'F':('130.45 / 130.50：毫米级重叠仍是不同来源','在整个采样高程 1391.286–1397.286 m，130.45/130.50 这对邻线有 61 层分别选 F_T1、F_T2。曲线外观非常接近，但在本 ROI 的真实共享边图中属于不同分量。FaceTrack 可以揭示来源跳变；单靠这项证据，还不能决定 F_T1 或 F_T2 谁是真实目标。'),
    'G':('122.95：存在同来源合格候选，实际主轨仍缺失','122.95 当前路线仍止于约 1384.385 m；在本图约 1420 m 附近，没有蓝色主轨，但灰色原始候选存在。黑圈标出与邻线已选来源一致、MBG 合格的候选。这才是下一轮局部联合选轨值得重点验证的真实缺失证据；本轮没有自动补入。'),
    'H':('116.25：另一个缺失主轨但候选存在的区域','116.25 当前主轨仍止于约 1384.783 m，1442 m 附近的候选属于邻线同一来源。来源一致能回答“它是不是同一片连续网格”，但如何保留、从现有主轨如何合法到达，还要经过联合选择、连接方向及长度约束。'),
    'I':('100.15：细节数量差异含有面来源跳变','本窗 4 个 FaceTrack；100.15/100.20 在 z=1377.086–1379.886 m 有 29 层来源切换，100.20/100.25 在 z=1380.186–1382.986 m 另有 29 层。还有 4 条来源集合歧义。来源标签能把一部分“细节复杂度不一致”转化为具体来源问题，尚未验证应保留哪一张面。'),
    'J':('108.80：同来源不保证细节表达完全相同','239 条邻线比较具有同来源证据，另有 1 条无证据。图中 108.80 在 z≈1402.2 m、u≈−8.3 m 处保留明显小回环，108.85/108.90 的相应主轨更简单，但来源仍为 J_T1。FaceTrack 能识别来源身份，却不能单独决定这个小回环应当保留还是剔除；本轮未平滑或删除原始细节。'),
    'K':('1.95：锚点平局的来源对照窗口','此窗只有单一局部 FaceTrack、240 条采样一致，暂未找到来源切换。旧锚点平局记录没有精确局部坐标，本图使用最终路线的中位高程作代理窗口，不是已经定位到原平局位置；不能据此宣称原平局已解决。'),
    'L':('67.60：续接平局的来源对照窗口','3 个局部 FaceTrack 可区分；85 条同来源、30 条仅编号变化、5 条无证据，未产生未选同来源候选建议。本图取最终路线低端边界作为代理位置，原续接平局日志没有足够坐标，结论只限当前窗口。')}
    review=[f'# {title}：局部案例分报告',scope,identity,guide,'## 案例索引',
        table(['案例','重点 s（m）','局部来源数','来源切换 / 歧义采样','未选候选支持交点'],[
        (r['region_id'],r['focus_s'],r['face_track_count'],f"{r['VV_classification_counts'].get('FACE_TRACK_IDENTITY_SWITCH',0)} / {r['VV_classification_counts'].get('AMBIGUOUS_FACE_TRACK',0)}",r['shadow_count']) for r in face['regions']])]
    fgaps=[]
    with (out/'region_data/F.pkl').open('rb') as f:fd=pickle.load(f)
    center_rows=[h for h in fd['VH'] if abs(h['z']-fd['region']['center_z'])<1e-6]
    for s in fd['region']['target_keys']:
        hs={h['H_face_tracks'][0]:h for h in center_rows if h['s']==s}
        a,b=hs['F_T1'],hs['F_T2']
        fgaps.append([s,f"{a['u']:.9f}",f"{b['u']:.9f}",f"{abs(a['u']-b['u'])*1000:.6f}",'F_T1' if a['selected_current_route'] else 'F_T2'])
    ftable=table(['s（m）','F_T1 的 u（m）','F_T2 的 u（m）','同切面间距（mm）','实际选择'],fgaps)
    for r in face['regions']:
        rid=r['region_id'];name,explain=explanations[rid]
        review.extend([f'## 案例 {rid}｜{name}',explain,
            f'![案例 {rid} 相邻实际主轨](figures/representative/{rid}_neighbors.png)',
            f'![案例 {rid} 来源对照](figures/face_tracks/{rid}_face_tracks.png)'])
        if rid=='F':review.extend(['同一 H 层 z=1394.286304 m 的实数核对。H_path 77 对应 F_T1，H_path 90 对应 F_T2；H_path 编号只在该 H 层有效。',ftable,
          '注意 s=130.55 时两候选仅相差 0.385 mm，且 u 大小关系倒转；按“更靠左/更靠右”保持来源身份并不可靠。这里“不同来源”只指该 ROI 内分开，不声称它们在全模型永不连接。'])
        br=[b for b in branches if b['region_id']==rid]
        review.append(table(['s','branch / 类型','MBG','ASC m','来源面数','FaceTrack','当前选中','邻线已选来源','shadow 建议'],[
          (b['s'],b['branch_id']+' / '+b['kind'].replace('_SURFACE_BRANCH',''),b['MBG'],f"{float(b['ASC_length_m']):.3f}",b['source_FaceID_count'],','.join(json.loads(b['FaceTrack_ID'])),b['selected_current_route'],','.join(json.loads(b['neighbor_selected_FaceTrack'])),b['shadow_recommendation']) for b in br]))
        review.append('逐边来源、选中区间与全部候选关联见 [face_track_members.csv](face_track_members.csv)；交点见 [vh_face_provenance_crossings.csv](vh_face_provenance_crossings.csv)。')
    review.extend(['## 相邻来源切换事件范围','以下一行聚合一对邻线；层数才是实际采样数，起止高程不承诺中间每层都属于该事件。'])
    groups=defaultdict(list)
    for v in vv:
        if v['classification']=='FACE_TRACK_IDENTITY_SWITCH':groups[(v['region_id'],v['left_s'],v['right_s'])].append(float(v['z']))
    review.append(table(['案例','左 s','右 s','H 层数','Z 最小–最大（m）'],[(rid,a,b,len(z),f'{min(z):.3f}–{max(z):.3f}') for (rid,a,b),z in groups.items()]))
    review.extend(['## 未选候选的来源支持','604 条是重复采样交点，来自 19 个“案例×测线”组合，不是 604 条独立缺失分支。全部候选 MBG=True；其中 136 条有左右两侧支持，468 条仅一侧支持（窗口边缘也只能有一侧）。只有来源支持，不保证连接可行。'])
    hints=defaultdict(list)
    for h in shadow:hints[(h['region_id'],h['s'],h['candidate_branch_id'],h['face_track_id'])].append(h)
    review.append(table(['案例','s','候选 branch','来源','交点数','双侧支持数','Z 最小–最大'],[
      (rid,s,bid,tid,len(rs),sum(int(x['support_sides'])==2 for x in rs),f"{min(float(x['z']) for x in rs):.3f}–{max(float(x['z']) for x in rs):.3f}") for (rid,s,bid,tid),rs in hints.items()]))
    review.append('回到 [总分析报告](FACE_PROVENANCE_VALIDATION_REPORT.md)。')
    write('FACE_TRACK_REVIEW.md','\n\n'.join(review))
    write('FACE_PROVENANCE_VALIDATION_REPORT.md',f"""# {title}：总分析报告

**本轮证据支持继续做局部 FaceTrack 联合选轨试验。**零长自环确实错误阻断过长分支资格，修复后部分实际缺轨已经恢复；FaceTrack 也能揭示外观几乎重叠的候选来自不同局部网格面。但当前算法仍有缺轨与邻线来源切换，本轮没有把 FaceTrack 建议当作最终识别结果。

## 这轮到底出了什么结果

{scope}

{identity}

源码修复仅涉及物理拓扑与分支提取两个模块；本轮基线为 backend / `{base['HEAD']}`，开始时工作区干净。输入包括 2800 条 V、1148 个 H 高程层，本次局部 FaceTrack 只抽取相应高程。完整基线见 [BASELINE_MANIFEST.json](BASELINE_MANIFEST.json)。

## 主要结果

- 32 条零长度自环涉及 30 条 V、31 个 component；恢复 28 条完整 OPEN/MBG 候选链，完整保留几何和全部来源；已知 27 条受影响 V 的 52 个旧长核片段归入恢复链。
- 0.85、78.40 的正常长分支恢复；78.35 通过邻线证据变化也恢复上部；103.70 的原缺失下部在实际 CURRENT_ROUTE 中已恢复。这些属于拓扑修复后的识别结果，不是 FaceTrack 填充。
- 在 12 个目的性选取区域中，7 区可分开采样到的 MBG 竞争来源；B/C/G/H/K 5 区只有单一局部来源，不能用来测试多面区分能力，也不应算作 FaceTrack 失败。
- 130.45/130.50 的候选仅相差约 1.855/0.734 mm，但实际路线选了不同局部面来源。122.xx、104.xx、100.15 也仍有来源切换。
- 122.95 与 116.25 的缺轨仍在，但该局部存在与邻线主轨同来源的 MBG 合格 observed 候选。下一步应研究如何合法选择它们，而不是补画长直线。

{comparison_table}

详细上游原因与全区断言见 [物理拓扑分报告](PHYSICAL_TOPOLOGY_FIX_REPORT.md)。

## FaceID、共享边与短面链提供了什么证据

使用冻结的 filtered mesh：{mesh['faces']:,} 个三角面，{mesh['metadata']['filtered_vertex_count']:,} 个顶点。构建 {mesh['shared_edge_face_pairs']:,} 对真实共享边邻接；24 个非流形边组如实保留，16 个重复顶点形成的退化边不建假邻接。不重新加载 GLB、焊点或重编号，避免 FaceID 空间改变。

H clean 缓存原来只留了部分来源，本轮从 raw H 以无向 round6 线段完全相同恢复全部 FaceID。2,722,036 条 clean edges 全部匹配。局部共 6824 个 raw H 交点，6775 个与 V 在 3e-6 m 内建立几何对应，**这 6775 对全部为 EXACT_FACE**；另外 49 个未与 V 建立对应，不能算作来源验证通过。当前样本恰无多 FaceID 的 H 交点，合并来源逻辑由专门测试验证。

对局部 V 端点与其每一个 source face 做了 29,288 次坐标核对，最大残差 {face['maximum_source_coordinate_residual_m']:.9g} m（约 0.850 微米），低于 3e-6 m 容差，支持当前 FaceID 确实使用了同一 slicing mesh 的索引空间。残差是几何来源一致性，不是识别准确率。

相邻 V 通常不会永远命中同一三角面。在 2321 条同来源采样比较中，按最短见证分类：

{table(['最短见证','条数','判据'],[('EXACT_FACE',tiers['EXACT_FACE'],'两侧完整 source_face_ids 有交集'),('EDGE_ADJACENT_FACE',tiers['EDGE_ADJACENT_FACE'],'原网格中共享真实三角边'),('LOCAL_FACE_CHAIN',tiers['LOCAL_FACE_CHAIN'],'ROI 内最多 8 跳共享边链')])}

因此，真实共享边和短面链提供了 exact FaceID 之外的证据。FaceTrack 编号用局部 ROI 内共享边分量建立；所有“邻线支持”另要求最多 8 跳的见证，不能仅凭局部分量相同就算支持。ROI 按三角面包围盒相交纳入完整三角面，并非把三角形裁碎；不沿全模型大分量任意游走。ROI 大小与 8 跳上限的敏感性未在本轮系统评估。

## 邻线是否已一致

{counts}

总计 2768 条，单位是“一对相邻 V × 一个 H 高程层”，同一个物理位置可以连续产生很多行，不能把这些数当独立缺陷数或准确率。411 条只说明当前采样中的编号差异不能当面来源差异；它不能从前轮全区 3.85% 的 H 保留差异统计中直接扣除，两者口径不同。

未选同来源候选有 604 个采样交点，来自 19 个“案例×测线”组合；136 个有双侧支持，468 个只有单侧支持。全部 MBG 合格，但本轮未检查将它们加入后能否满足完整主轨的方向与合法连接，因此只是候选证据。

## 优先人工审核的局部

优先看 **F（130.45/130.50 来源切换）、G/H（有同来源候选但主轨缺失）、D/E（局部多来源选择）**。完整 12 区、每区五条邻线、四栏来源图与分支属性表都在 [局部案例分报告](FACE_TRACK_REVIEW.md)。

![毫米级重叠的来源区别](figures/face_tracks/F_face_tracks.png)

图④的类别轴让来源差异可见：130.35–130.45 选择 F_T1，130.50–130.55 选择 F_T2。它没有人为拉开原始几何；①②③仍显示原始坐标。

{ftable}

![122.95 邻线与缺失主轨](figures/representative/G_neighbors.png)

122.95 这一栏的缺蓝线是实际缺失，不是图上没有数据。FaceTrack 审核发现已有候选能够获得邻线来源支持；正式联合选轨仍待下一轮验证。

{guide}

## 尚存实现问题与下一轮边界

1. **候选资格恢复不等于必选。**G/H 仍有合格同来源候选遗漏；下一轮可用局部 V/H 联合标签表达一致性，再由原方向、连接长度和 R_syn 约束决定是否接入。
2. **几何接近不能保持来源身份。**F 的毫米级重叠与 u 大小关系倒转说明，仅按最近距离、左右顺序或形状复杂度无法稳定选择同一来源。
3. **真实 fork 的类型门控仍较粗。**全区发现 7 项 ASC 未经过叉点、但分支在端点接触真实 fork 的门控案例，已输出审核清单；本轮不直接改变真实 fork 语义。
4. **FaceTrack 不是最终地质面标签。**真实共享边可能跨过重建拼接或非流形结构，单一来源内部也可能折返；C 的 1362–1364 m 折返、J 的约 1402.2 m 小回环仍有邻线细节差异，不能把同来源直接等同于细节完全相同或强制平滑。
5. **原平局定位信息不足。**750 条 anchor tie、14 条 continuation tie 事件没有精确局部坐标；K/L 只是路线中位/边界代理窗口，不能宣称已复核所有真实平局位置。

本轮从前轮事件、可靠范围与本轮恢复信息形成 4205 个 s/u/z 复核网格单元（0.5×2×2 m），只是聚合桶，不是 4205 个独立问题。本轮只对 12 区做 FaceTrack，尚未做全区 FaceTrack census 或新一轮全 2800 主轨运行。

## 阶段耗时与最低检查

{performance}

这些是本机各阶段的一次墙钟计时，任务有并行，不能相加当总时长，也不能当生产入口性能或优化加速比；网格读取/包围盒、绘图与写报告有额外时间。

相关测试合计 **49 项通过**（44 项拓扑与相关回归，5 项来源邻接测试）。全区分支清查通过保持性断言；60 条 CURRENT_ROUTE 通过既有路线几何审计，最大连接 {max_connector*1000:.6f} mm，小于未改动的 10 mm；最大局部合成比例 {max_r:.9f}。FaceTrack 分析前后局部路线文件哈希一致。没有运行无关全项目测试。

## 规定结论

{status_table}

进一步实现需要用户先核对局部真实目标面；本报告本身不执行下一轮联合标注。实现差异见 [IMPLEMENTATION_DEVIATIONS.md](IMPLEMENTATION_DEVIATIONS.md)，复核记录见 [FINAL_REVIEW.md](FINAL_REVIEW.md)。所有输出保存在当前目录，前轮结果保留。
""")
    deviations=[
    ('保留原始拓扑统计，分离物理图','保留 degree/components/branches 的 canonical 语义，新增 physical_*，而非替换原字段。','避免丢失原始统计及改变 H 旧编号；主轨分支提取显式采用物理图。','2800 条保持性断言及 44 项相关测试。','不影响 self-loop 修复；其他旧调用者如只读取 canonical 字段，仍看到旧度数，字段语义已明确。'),
    ('78.35 / 78.40 都应复核','78.35 作为无自环控制加入；其结果变化由邻线证据变化引起。','实际检查 78.35 没有自环，不能套用计划中的推测原因。','physical_topology_before_after.csv、local_route_summary.json。','修正因果解释，不把邻线选择变化都归为本切面自环。'),
    ('完整 H source_face_ids','从 raw H 以无向 round6 等值线段恢复来源。','clean H 缓存不能单靠 first_face_id 表示完整来源。','2,722,036 条 clean edge 全部恢复，无未匹配；保留 clean 几何和 1e-10 既有 snap。','已匹配的来源不降级；实际本批 H 交点未出现多来源，另用测试覆盖。'),
    ('FaceID 与模型一致','使用冻结 filtered mesh cache，而非重新加载原 GLB。','FaceID 属于 filtered_mesh 空间，重新载入可能重排。','mesh hashes、metadata 与 29,288 个点/源面核对。','支持索引一致，尚未恢复原模型分块身份或地质真值。'),
    ('修复前后对照当前路线','两版都重建同范围局部 H/V 图并重新求解，保留旧全区结果。','合并分支后 ID 变化，不能复用旧 graph membership；也不应将不同证据范围直接比较。','12 区各 5 条、两版共 120 次，目标邻线至少 ±1 m halo。','当前是配对局部结果，不是全 2800 路线重算或旧全区运行的精确复现。'),
    ('冲突区域 FaceTrack 首轮验证','聚合 4205 个复核桶，只验证 12 个代表区域。','本轮验证来源可行性，包含必须 A–F 和缺轨/细节/平局控制。','region_scope.json、validation_regions.json。','结论不能外推为全区已消除不一致或准确率。'),
    ('局部 FaceTrack 与短链','以 ROI 相交三角面真实共享边分量编号，邻线支持另限 8 跳。','兼顾可追溯编号与短链实证，禁止全模型连通分量直接充当轨道。','face_track_inventory.csv、邻线/建议表中的 witness_face_chains。','局部编号依赖 ROI，未做尺度敏感性评估，不能证明全局永不连接。'),
    ('锚点/续接歧义的局部案例','K/L 采用最终路线中位 Z / 低端 Z 代理。','旧 750/14 个事件缺精确局部坐标。','validation_regions.json 的 location_basis。','仅作对照，不宣称精确定位或解决原平局。'),
    ('清晰展示横向约束','图④使用 FaceTrack 类别轴，并在 F 附真实 u 与毫米间距表。','原 u 轴上的两条来源会在毫米级重叠，标记互相遮盖。','F 两候选间距 1.855/0.734 mm，图文均明确类别轴。','不修改原始几何，不伪造空间距离；该栏不表示交点数。'),
    ('真实 fixture 可随修改复现','为 tests/fixtures/physical_topology_real_slices.npz 添加精确 .gitignore 例外。','独立 review 发现通配 *.npz 会忽略真实测试 fixture。','git 状态可见该文件，fixture 为前轮两真实切面的原始几何。','只收录这个小文件；未开放 outputs 或其余缓存。')]
    dlines=['# 实现差异与执行记录','本表按“原计划 / 实际实现 / 原因 / 证据 / 影响”逐项记录。']
    for i,(plan,actual,why,proof,effect) in enumerate(deviations,1):
        dlines.append(f'## {i}. {plan}\n\n- 原计划：{plan}。\n- 实际实现：{actual}\n- 原因：{why}\n- 证据：{proof}\n- 影响：{effect}')
    dlines.append('执行中首次 inventory 在受影响集合查找 78.35 时停止，尚未进行全区阶段；改为显式无自环控制后完成全区一次。真实 78.40 测试初始断言错误沿用 z<1400，实际长链位于 1403.912–1474.143 m，核查后仅将测试探针改为 1446 m。FaceTrack 模块测试先红后绿。随后仅因类别标记遮盖调整了绘图，没有重跑算法或扩大测试。')
    dlines.append('用户要求原地最小范围执行，因此保留 backend 工作区，没有创建 worktree、重配环境、提交或推送；使用计划作为设计，不另行引入架构方案。')
    write('IMPLEMENTATION_DEVIATIONS.md','\n\n'.join(dlines))
    write('FINAL_REVIEW.md',f"""# 本轮复核记录

## 独立源码复核

按执行计划的 review 流程，由独立只读 reviewer `physical_face_final_review` 检查生产差异、真实 fixture、来源邻接及局部审核实现。未报告 Critical 项；发现 1 项 Important：`.npz` 通配忽略会使新增真实 fixture 无法正常进入版本管理。已添加仅针对 `tests/fixtures/physical_topology_real_slices.npz` 的例外，并确认 git 状态能列出该文件。没有自动提交。

reviewer 未发现其他具体拓扑、来源或有界邻接缺陷；它明确没有审核当时尚未完成的运行结果、最终报告、地质含义与图像布局，不能把源码 review 当作这些内容的验收。

## 主执行复核

主执行检查了全 2800 清查保持性、120 条配对解算摘要、来源索引坐标残差、CURRENT_ROUTE 哈希不变及图像。运行证据为 topology_checks.log（44 通过）、face_checks.log（5 通过）、topology_summary.json、local_route_summary.json、face_validation_summary.json。

图像来源分类栏曾在 u 轴上产生毫米级标记遮盖；已经改用明确标识的类别轴，原几何三栏和邻线图不移动坐标。报告另列 F 案例原始 u 数字和真实距离，避免类别轴被误认为几何距离。交付路径及资源完整性见 DELIVERY_CHECKS.json。

## 结论边界

DISCRIMINATIVE 仅表示本窗口采样到的 MBG 竞争分支可分成不同局部面来源；它不是目标曲面识别正确。局部 FaceTrack 仍可能包含重建连通、非流形边或折返。604 条 shadow 是候选交点，未承担路线合法可达的证明。49 个未关联 H 交点没有被算成来源核对通过。K/L 是代理窗口。

真实 fork 门控的 7 项、D/E/F/I 来源切换、G/H 缺轨仍开放，详见案例报告。没有把本轮认定为全区识别完成。

## 最终状态

{status_table}

人工重点复核 F 两张面谁是目标、G/H 候选是否应延续、D/E 的来源切换是否真实、J 的同来源细节是否合理。FaceTrack 写入正式联合选轨与生产 GUI 均不属于本轮。
""")
    # This validates delivery consistency; it does not rerun recognition or the test suite.
    changed=[p for p,h in base['code_sha256'].items() if sha(ROOT/p)!=h]
    assert {Path(p).name for p in changed}=={'vertical_profile_canonicalization.py','dominant_observed_branch.py'},changed
    assert len(current)==len(before)==60
    assert top['slices']==2800 and all(top[k] for k in ['affected_gate_passed','all_geometry_and_provenance_preserved','no_component_merge','no_normal_open_lost'])
    assert routes['CURRENT_ROUTE_sha256']=={p.name:sha(p) for p in (out/'local_routes').glob('*.pkl')}
    assert face['CURRENT_ROUTE_unchanged_by_FaceTrack'] and max_connector<=.010000001
    required=['BASELINE_MANIFEST.json','physical_topology_before_after.csv','self_loop_affected_branches.csv','recovered_branches.csv',
      'face_adjacency.npz','face_track_inventory.csv','face_track_members.csv','vh_face_provenance_crossings.csv',
      'conflict_region_face_tracks.csv','neighbor_route_face_track_consistency.csv','shadow_face_track_recommendations.csv',
      'PHYSICAL_TOPOLOGY_FIX_REPORT.md','FACE_TRACK_REVIEW.md','FACE_PROVENANCE_VALIDATION_REPORT.md','IMPLEMENTATION_DEVIATIONS.md','FINAL_REVIEW.md']
    assert all((out/p).is_file() and (out/p).stat().st_size for p in required)
    figures=list((out/'figures').rglob('*.png'));assert len(figures)==26
    for name in required:
        if name.endswith('.md'):
            for link in re.findall(r'\]\(([^)]+)\)',(out/name).read_text(encoding='utf-8')):
                assert (out/link).is_file(),(name,link)
    for logfile,count in [('topology_checks.log',44),('face_checks.log',5)]:
        text=(out/logfile).read_text(encoding='utf-8',errors='replace')
        assert f'Ran {count} tests' in text and text.strip().endswith('OK')
    write('DELIVERY_CHECKS.json',json.dumps({'required_artifacts_present':True,'report_links_resolve':True,'figure_count':len(figures),
       'production_source_files_changed':changed,'branch_inventory_count':top['slices'],'route_solve_count':len(current)+len(before),
       'FaceTrack_did_not_change_routes':True,'relevant_tests_passed':49,'maximum_connector_m':max_connector,
       'full_area_routes_recomputed':False,'accuracy_claimed':False},ensure_ascii=False,indent=2))
    print('REPORTS_WRITTEN_AND_DELIVERY_CHECKED',out)


if __name__=='__main__':main(output_path())
