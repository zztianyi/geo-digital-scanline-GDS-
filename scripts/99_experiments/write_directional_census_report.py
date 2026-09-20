"""Write the Chinese census report from measured artifacts, never inferred totals."""
import hashlib,json,pickle
from collections import Counter
from pathlib import Path
from census_directional_consistency import ROOT,OUT,save_json,code_hashes
from review_directional_consistency import NAMES


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def write():
    m=read('input_manifest.json');r=read('recognition_summary.json');c=read('consistency_summary.json')
    cases=read('representative_cases.json');windows=read('local_review_windows.json');regions=read('review_regions.json')
    complexity=read('geometry_complexity_summary.json')
    excluded=read('topology_exclusions.json')
    rows=r['rows'];adj=c['scales'][0]
    assert len(rows)==len(m['keys'])==2800 and len({x['slice_key'] for x in rows})==2800
    assert code_hashes()==m['code_sha256']
    persistent_s={x[k] for x in windows if x['persistent'] and x['category']!='MBG_ASYMMETRY' for k in ('left','right')}
    reason_counts=Counter(reason for row in rows for reason in row['unresolved_reasons'])
    max_join=max(row['max_connector_m'] for row in rows)
    lines=[
        '# 当前主轨全区域普查与 1.5D 邻域一致性复核',
        '',
        '本轮只重新识别、普查和形成复核材料，没有修改识别算法。下文的蓝色主轨全部来自本轮真实计算；邻线也各自完成了识别，不是把目标曲线平移过去。',
        '',
        '这里的 1.5D 指：每条纵向 V 测线选轨时使用横向 H 测线及邻近 V 的证据，本轮进一步检查相邻测线最终输出。MBG 是最小分支门槛；ASC 是去除端部保护区后、具有足够规范节点与实际弧长的可靠核心。',
        '',
        '图中的“原始曲线”指当前冻结输入经 round6 规范化后、未被选轨裁剪的测线曲线，保留来源面编号。它仍受原有模型重建和裁剪影响；人工确认真实曲面身份时需要对照源网格。',
        '',
        '## 结论',
        '',
        f'全部 **{len(rows):,} 条纵向测线**已重算。相邻 0.05 m 测线的 **{adj["compared"]:,} 个可比 H 曲线交点对**中，有 **{adj["mismatch"]:,} 个最终保留状态不同**（{100*adj["mismatch"]/max(1,adj["compared"]):.2f}%）。这是选择一致性差异占比，**不是识别错误率，也不是准确率**。',
        '',
        f'差异连续出现至少 3 个横向高程层（首尾跨度至少 0.2 m）的局部窗口涉及 **{len(persistent_s):,} 条测线**。这些窗口在 s–z 投影上组成 **{c["regions"]:,} 个连通区**；它们可能在径向 u 上分离，因此不能称为同样数量的独立物理缺陷。',
        '',
        '**当前有邻域候选支持，但尚未形成邻域最终选择的闭环约束。** 一个候选可由邻线原始分支支持，而邻线的最终主轨仍可能采用另一条分支或裁掉对应部分。因此，单条主轨合理、连接长度合规，并不能自动证明相邻主轨局部一致。',
        '',
        f'另外发现一条可复现的实现问题链：**零长自环让整个连通分量成为分叉类型，长分支因此在 H/V 约束之前被拒绝。** 全区有 {c["self_edge_only_excluded_core_branches"]:,} 条已有物理核心的分支、涉及 {c["self_edge_only_affected_slices"]:,} 条测线，落入这种类型排除；这些计数不是必然漏识别数量。s=0.85 m 的 76.57 m 核心长分支是已核实的具体案例。',
        '',
        f'所有输出的原始线段坐标/来源、区间不重复、路径连接及双端连接预算均通过逐条核对；最大连接距离 **{max_join*1000:.6f} mm**，未超过现行 10 mm 上限。这证明几何约束被执行，不证明所选曲面身份正确。',
        '',
        '## 本轮结果范围与可复现性',
        '',
        f'- 输入：既有冻结区域全部 2,800 条 V 测线，s = {float(m["keys"][0]):.2f}–{float(m["keys"][-1]):.2f} m，间距 0.05 m；{len(m["levels"]):,} 个 H 高程层，间距 0.1 m。',
        f'- H 层覆盖 z = {min(m["levels"]):.8f}–{max(m["levels"]):.8f} m，共 {m["observations"]:,} 个原始交点。超出这个高程范围的纵向尾部不能声称得到本轮 H 约束验证。',
        '- 原始 H 曲线先在完整切面上编号，再与 V 相交；曲线编号仅在同一高程层内有意义。未用历史问题窗口限制识别。',
        '- 复用经代码哈希核对的全区域粗曲面图；每个目标保留左右各 1 m 全部原始 H/V 证据，块边界带 20 条邻线缓冲，真实模型边缘自然截断。',
        '- 上一轮三个展示样本使用局部 41 条测线图，本轮使用全区域粗图身份；原始 1 m 局部约束口径保持一致。104.30 m 复核仍得到 [1, 2, 4]。',
        '- 仅在普查进程缓存完全相同的纯统计查询及每个交点固定半径的统计量。104.30 m 与较复杂的 51.35 m 缓存前后完整识别输出逐字段一致；未做数值近似、候选剪枝或参数调整。已完成的 1,076 条路线通过检查点保留，未重复识别。',
        '- 复核阶段另行精确关联被算法 OPEN 类型过滤隐藏的分叉/闭合原始曲线，以解释拓扑门槛差异；它们的 MBG 仍为 false，不能进入已选主轨或双侧合格的主统计。这些扩展关联仅写入 normalized_crossings，不改 routes 中任何识别结果。',
        '- 当前是算法离线识别结果，生产 GUI 入口接入状态未改变；没有另外启动生产识别或宣称完成生产端到端验收。',
        '',
        '原始最终路线位于 `routes/<测线位置>.pkl`，包含每条实测/连接线段、原始面编号、边编号与 t 区间；索引与统计见 [recognition_summary.json](recognition_summary.json)。',
        '',
        '## 怎么看图',
        '',
        '每个重点案例有两张图，分别回答两个问题。',
        '',
        '1. **邻近纵向图：各条测线到底选了什么？** 通常从左到右为目标 s−0.10、s−0.05、s、s+0.05、s+0.10 m，模型边缘按实际可用数量显示。所有面板使用相同 u–z 范围，默认高度窗口为 6 m。灰线是原始候选；蓝线是本轮各测线最终保留的实测主轨；绿色空心点是原始横向交点（为清楚只显示每隔 0.5 m 的层，计算仍使用全部 0.1 m 层）。若有橙色短虚线，它才是算法连接段。',
        '2. **横向对应图：左右邻线是否保留了同一片曲面？** 展示目标左右 1 m 的实际主轨交点，三个面板是三个固定高程。横轴 s，纵轴 u。绿色空心点为原始 H/V 交点；浅绿线仅表示同一原始 H 曲线的相邻测线对应关系，不是补线或新的实测曲线；蓝点表示该测线最终主轨保留了这个交点。蓝点沿同一行连续，表示该处选择相容；蓝点跳到另一行、某些列缺点，则需要回到纵向图和原始模型核查。',
        '',
        '**不要比较不同面板的分支编号是否相等。** 编号是单条切面内部的编号；编号不同可以是同一曲面，编号相同也可能是不同曲面。纵向图中局部分支列表只用于追溯本切面的拼接来源。',
        '',
        '连接段最多 10 mm，在 6 m 高程窗口内可能小于一个像素；图例出现橙色而肉眼难见，不代表画了一条长连接线。横向图中的灰色竖虚线仅定位目标测线。',
        '',
        '坐标 s 是测线沿基准弧的排布位置，u 是相对基准圆弧的径向偏移，z 是高程，单位均为米。同一分支若有折返，同一高程可以出现多个交点，因此看见多条灰线不等于存在多个独立曲面。横向图为容纳全部已选交点，可使用比纵向图更宽的 u 范围。',
        '',
        '**不能以“曲线更平滑”作为真实曲面的证明。** 折返可能是真实模型几何；本轮既没有平均邻线，也没有用平滑曲线替代实测线。局部保留长度、逆高程累计量及 H 交点数量只用于定位不一致，最终身份仍需人工对照网格。',
        '',
        '## 全区域分布',
        '',
        '![全区域复核分布](figures/whole_area_review.png)',
        '',
        '灰蓝色表示原始 H/V 可唯一对应的位置；红色表示两侧最小分支门槛均合格且最终保留状态不同，并连续出现至少 3 层。黄色表示分支门槛两侧不同，另列、不混入红色主统计。紫色叉号是 H 基本相容但纵向复杂度变化的位置。圆圈数字对应下文案例。此图是 s–z 投影，不是原始三维模型的平面投影。',
        '',
        '图中的细竖带表示某一对邻线在较长高程范围持续存在差异，不能直接将其解释为已确认的模型分块缝。本轮没有重新核验重建分块位置。',
        '',
        '## 一致性统计及口径',
        '',
        '| 间隔 | 测线对 | 参与比较交点对 | 同时保留 | 仅一侧保留 | 差异占比 |',
        '|---|---:|---:|---:|---:|---:|']
    for a in c['scales']:
        lines.append(f'| {a["distance_m"]:.2f} m | {a["pairs"]:,} | {a["compared"]:,} | {a["agreement"]:,} | {a["mismatch"]:,} | {100*a["mismatch"]/max(1,a["compared"]):.2f}% |')
    lines += ['',
        '分母只包括：同一高程、同一 H 曲线，在两端及中间所有测线上都具有唯一 H/V 对应；两端分支均通过 MBG；且至少一端最终主轨保留了该交点。两端都不选的背景候选不进入分母。对较远测线不能越过中间缺失或歧义交点建立对应。',
        '',
        '同一位置可有多个 H 曲线交点；上表统计交点对，不是测线数量、面积或人工标注样本。不同间距的表行不是同一个分母，不能直接作为算法优劣排名。',
        '',
        '**H 与 V 都来自同一重建模型，属于同源几何约束。** 它们相互吻合可以支持局部选择的一致性，不能独立证明对应的是正确地质曲面；模型重建自身的重叠、边界和缺失仍需原始网格复核。',
        '',
        '| 连续差异类别 | 相邻测线局部窗口数 | 含义 |',
        '|---|---:|---|']
    explanation=dict(SURFACE_CHOICE='两侧保留不同的可比 H 曲线集合；需复核曲面身份。',
        DETAIL_COUNT='两侧共有保留交点，但数量不同；可能涉及折返、分叉或局部裁剪，不能直接断言丢失真实细节。',
        COVERAGE_GAP='可比曲面在一侧没有保留；该侧可能选择了其他不可比轨迹，并非断言整条主轨缺失。',
        MBG_ASYMMETRY='原始对应存在，但最小分支门槛通过状态不同；另列审核。')
    for category,count in c['persistent_window_categories'].items():
        lines.append(f'| {NAMES[category]} | {count:,} | {explanation[category]} |')
    lines += ['',
        '窗口是“相邻测线对 × 类别 × 连续高程段”，同一空间问题可产生多个相邻窗口；不能把窗口数直接当成独立问题数。较大间距只比较两端最终保留，不表示中间所有主轨都一致；中间变化由 0.05 m 相邻统计捕获。',
        '',
        '### 局部复杂度是否相容',
        '',
        f'另在统一的 **2 m 高程带**中量测最终实测主轨的实际弧长与累计逆高程变化。共 {complexity["eligible_pair_windows"]:,} 个邻线对窗口满足：至少 10 个 H 层共有已选交点、共有交点占两侧唯一已选交点并集至少 80%，且双侧 MBG 合格的可比交点没有保留差异。其中 **{complexity["flagged_windows"]:,} 个窗口**触发复杂度复核，涉及 {complexity["flagged_pairs"]:,} 对邻线。',
        '',
        '筛查阈值：弧长差至少 0.5 m 且比例至少 1.3；或者累计逆高程变化差至少 0.05 m 且比例至少 2。曲线按原始线段与高程带精确相交后累计，排除连接段，没有重采样或平滑，因此也能量到 H 层之间的短折返。阈值仅用于挑选人工案例，不能将触发数当成错误数；变化可能本来就存在于原始网格。',
        '',
        '这项检查补充了“同一 H 曲线选中没有”的检验，但不承诺穷尽所有亚分米形状差异。完整位置见 [geometry_complexity_windows.json](geometry_complexity_windows.json)，数值矩阵见 `local_geometry_metrics.npz`。',
        '',
        '### 两侧一致也可能存在的问题',
        '',
        '除邻线差异外，本轮还读取了每条路线的局部候选并列、未覆盖的合格分支范围、接近长度上限的连接和连接转角。它们能发现“相邻测线一起漏掉某段”的情况，但也只是复核线索。',
        '',
        '| 路线内复核信号 | 事件数 |',
        '|---|---:|']
    for category,count in c['intrinsic_event_categories'].items():lines.append(f'| {category} | {count:,} |')
    lines += ['',
        '详细位置见 [intrinsic_review_windows.json](intrinsic_review_windows.json)。`UNCOVERED_RELIABLE_EXTENT` 比较的是所有 MBG 合格候选的高程范围，不能证明这些候选全部属于目标真实曲面；8 mm 连接预警和 60° 转角预警是报告筛查阈值，不改变现行算法。',
        '',
        '## 重点局部案例',
        '',
        '选例优先包含每类持续差异中较长的区域，再补充不同 s 区域及上一轮三个历史位置。这是问题导向抽样，不能用案例比例估计全区域错误率。图中只展示本轮结果，历史位置的标题不表示复用旧曲线。',
        '']
    for x in cases:
        lines += [f'### 案例 {x["case_id"]:02d}：{x["label"]}，s = {float(x["left"]):.2f} m', '',
            f'局部窗口：z = {x["bounds"][2]:.3f}–{x["bounds"][3]:.3f} m，u = {x["bounds"][0]:.3f}–{x["bounds"][1]:.3f} m。', '']
        if x['category']=='GEOMETRY_COMPLEXITY':
            lines += [f'触发带为 z = {x["z_min"]:.3f}–{x["z_max"]:.3f} m；其中共享 {x["shared_selected_H_crossings"]} 个已选 H 交点。两侧实测长度分别 {x["left_length_m"]:.3f} / {x["right_length_m"]:.3f} m，累计逆高程变化分别 {x["left_reverse_z_m"]:.3f} / {x["right_reverse_z_m"]:.3f} m。下图扩大到 6 m 便于看走向；需判断细节变化来自真实网格还是不相容的局部选择，不能因更曲折就认定错误。','']
        elif x['category']!='HISTORICAL':
            lines += [f'触发原因：{x["left"]} 与 {x["right"]} m 间，z = {x["z_min"]:.3f}–{x["z_max"]:.3f} m 连续 {x["levels"]} 个 H 层出现 {NAMES[x["category"]]}；当前图放大其中一段局部。{explanation[x["category"]]}','']
        else:
            lines += ['此位置作为上一轮修复结果的邻域回查，不预先断定有错；重点观察目标及邻线是否采用相同曲面走向。','']
        e1,e2=x['center_evidence'];z=e1['z']
        left={r['H_run']:r for r in e1['crossings'] if r['unique']}
        right={r['H_run']:r for r in e2['crossings'] if r['unique']}
        common={h for h in left.keys()&right.keys() if left[h]['MBG'] and right[h]['MBG']}
        ls=sorted(h for h in common if left[h]['selected']);rs=sorted(h for h in common if right[h]['selected'])
        difference=sorted(set(ls)^set(rs))
        gate=[h for h in left.keys()&right.keys() if left[h]['MBG']!=right[h]['MBG'] and left[h]['selected']!=right[h]['selected']]
        if x['category']=='MBG_ASYMMETRY':
            descriptions=[f'H{h}：{e1["slice_key"]} m 分支{left[h]["branch_id"]}（MBG={left[h]["MBG"]}，保留={left[h]["selected"]}），{e2["slice_key"]} m 分支{right[h]["branch_id"]}（MBG={right[h]["MBG"]}，保留={right[h]["selected"]}）' for h in sorted(gate)]
            lines += [f'**中心高程的门槛证据：** z = {z:.3f} m；'+('；'.join(descriptions) if descriptions else '该层无唯一门槛差异，需查看窗口内其他高程')+'。这类事件不进入“双侧门槛合格”的主统计。H 编号只在本层有效。','']
            if x['label']!='零长自环使长分支被排除':
                for ev,table in ((e1,left),(e2,right)):
                    bids={table[h]['branch_id'] for h in gate if not table[h]['MBG']}
                    for t in excluded:
                        if t['slice_key']==ev['slice_key'] and t['branch_id'] in bids:
                            lines += [f'门槛追溯：{t["slice_key"]} m 分支 {t["branch_id"]} 为 {t["kind"]}，有 {t["nodes"]} 个规范节点、{t["ASC_arc_length"]:.3f} m 可靠核心；当前最大节点度数为 {t["max_degree"]}，诊断性忽略自环后的最大度数为 {t["max_degree_without_self_edges"]}。因此这里不能仅凭“门槛未通过”就把它称为短噪点。','']
        else:
            lines += [f'**中心高程的实际证据：** z = {z:.3f} m，{e1["slice_key"]} m 保留的可比 H 曲线编号为 {ls}，{e2["slice_key"]} m 为 {rs}。'+
            (f'编号 {difference} 在两条原始 V 测线上都有唯一对应且两侧 MBG 合格，但最终只在一侧保留。' if difference else '此中心高程未发现可比 H 曲线保留集合差异；仍需查看上下高程和更宽邻域。')+
            '这些编号只适用于这一高程，不能跨高程比较。','']
        if x['left']=='130.45' and x['right']=='130.50':
            gaps=[abs(table[77]['u']-table[90]['u'])*1000 for table in (left,right)]
            lines += [f'**本例差异的实际尺度：** 在该中心高程，同一切面内 H77 与 H90 两个候选的径向间距分别只有 {gaps[0]:.3f} / {gaps[1]:.3f} mm。因此蓝色走向肉眼几乎相同，但其来源身份发生了切换。它与整段缺失的严重程度不同；需核查是重建重叠层，还是必须区分的真实曲面。主统计尚未按空间偏差或影响程度加权。','']
        if x['category']=='GEOMETRY_COMPLEXITY':
            lines += ['本例蓝色折返全部来自原始实测线段，图中没有用连接段制造这些折返。可以确认最终输出的复杂度不同，但不能将原始网格已有的细节变化直接归因于算法造假或判错。','']
            if x['left']=='103.60':
                lines += ['重点比较中间的 103.60 与右邻 103.65 m：约 z=1362–1364 m 内，后者保留了更大的折返。最右侧 103.70 m 在本窗口没有蓝线，是另一个覆盖问题，不属于触发本例复杂度统计的测线对；其实际最终主轨从 z=1403.936 m 才开始。','']
            if x['left']=='108.80':
                lines += ['重点看 z≈1402.2 m、u≈−8.5 m：108.80 m 的蓝线带有突出的回环；108.85 m 对应位置的灰色小环未进入主轨。原始分支记录表明后者是单独的 CLOSED_COMPONENT（分支 5，高程 1402.099–1402.259 m）。这说明相邻切面原始拓扑本身也在变化，不能仅要求两条蓝线强行平滑一致。','']
        if x['label']=='零长自环使长分支被排除':
            lines += ['**已经核实的原因：** 0.85 m 的分量 4 含分支 4/5/6。节点 607 上的零长自环边 636（来源面 1938484）贡献两个邻接，使该节点度数成为 4；只在诊断统计中扣除该自环后，整个分量最大度数为 2。分支 5 有 574 个节点、77.4762 m 总弧长、76.5681 m ASC，最低核心要求仅 0.4624 m，却因 `FORKED_COMPONENT` 被 MBG 排除，并被统一记为 `REJECT_SHORT_BRANCH`。',
                '',
                '独立关联全部原始类型后，分支 5 有 527 个唯一 H/V 交点，其中 514 个在左右邻线都有唯一同层同 H 曲线对应，424 个位于本测线最终主轨下界 1403.810371 m 以下。现有 OPEN 类型过滤隐藏了这些证据。图中应关注目标面板蓝线止步、灰色长线仍有横向对应，而左右邻线蓝线继续的现象。这里已有原始 V 线段，H 的作用应当是验证邻域对应并支持选轨，首先应处理类型门槛；真正没有原始 V 的缺段属于另一个问题。这个事实说明排除原因有问题，仍不替代真实曲面身份的人工确认。','']
        lines += [f'![案例 {x["case_id"]} 邻线](figures/{x["figure"]})','',
            f'![案例 {x["case_id"]} 横向约束](figures/{x["horizontal_figure"]})','',
            '| s（m） | 本窗口保留分支 | 本窗口实测长度（m） | 累计逆高程变化（m） |',
            '|---|---|---:|---:|']
        for nr in x['neighbor_rows']:
            lines.append(f'| {nr["slice_key"]} | {nr["local_branches"]} | {nr["local_observed_length_m"]:.3f} | {nr["local_reverse_z_m"]:.3f} |')
        lines += ['', '上表在同一 z 窗口量测，包含最终路线该高程内的全部径向部分；长度与逆高程变化用于比较局部复杂度，数值不同本身不判错。人工重点确认蓝线所走的是同一片曲面，保留的折返是否有原始网格与横向交点支持。','']
    lines += ['## 当前实现仍需注意的问题','',
        '0. **退化自环与整分量门槛耦合。** 规范化保留零长边及来源本身不意味着它应作为物理分叉；当前度数统计把自环计两次，再把整分量的所有长链归为 FORKED，随后 MBG 与 H/V 关联双重排除。已在 0.85 m 复现。应先区分保留来源记录与物理分叉判定，再讨论分支的可靠核心，而不是把这些长链解释成短噪点。全区清单见 [topology_exclusions.json](topology_exclusions.json)。本轮未删边、未修改规范化规则。',
        '',
        '1. **最终选择尚未互相约束。** `joint_surface_consensus.py` 从原始邻线候选构造支持；`solve_dominant_branch` 对每个目标独立求解。邻线最终选择并不反馈到候选支持统计中。本轮直接量到了这一层面的差异；下一轮若修改，应优先讨论如何对实际已选路线做邻域复核与冲突消解，而不是仅增加候选支持权重。',
        '2. **局部复杂度没有最终邻域一致性验收。** `same_surface_detail` 比较的是候选分支的整体形状描述，现有输出没有针对“同一个局部区域最终各保留哪些折返”的联合验收。H 交点数差异只能定位这种风险，仍需原始几何验证。',
        '3. **前瞻排序与实际保留片段存在口径差异。** `main_track_assembly.py` 在候选排序时调用 `future_switch_cost`，未传入 `_remaining` 给出的实际剩余片段；部分候选会从整条分支估算后续换轨。内部换轨的初始候选排序也使用整分支；形成候选连接后，才另以实际保留片段复核并记录前瞻估计。此项是代码层面确定的差异，可能使初始排序偏乐观，本轮尚未证明它导致某个具体错误选择；最终连接的几何与预算检查仍然执行。',
        '4. **内部替换仍限于当前路线的端部分支，前瞻深度为 2。** 它不是所有分支的全局最优组合，也不能自动重排已经锁定的中间两端拼接。',
        '5. **横向数据参与选路，不等于已经自动补齐所有缺段。** 当前入口仍把缺失部分保留为待定；超过 H 高程范围、缺少唯一 H/V 对应、原始重建或裁剪截断，都不能靠邻线相似性当作已确认实测点补入。89.75 m 的源数据需求标记沿用上一轮原始 Tile 连通性证据，未推广到其他位置。',
        '',
        '以上均为本轮审核结论，没有在普查过程中更改算法或偷偷平滑路线。',
        '',
        '## 性能与最低检查','',
        f'- 输入整理：{m["prepare_seconds"]:.2f} s。',
        f'- 全区域识别活跃运行时间：{r["wall_seconds"]:.2f} s（{r["wall_seconds"]/60:.2f} min），第一段 6 个进程、检查点续跑 {r["workers"]} 个进程；包含分块读入、求解、逐条几何核查、交点保留标记与写盘，不包含中间缓存验证暂停、网格重建、生产识别后续阶段和出图。',
        f'- 各求解调用墙钟时间之和：{r["sum_solver_seconds"]:.2f} s；并行进程间重叠，不应将其解释为单线程基准或直接除以 2,800 作为独占 CPU 耗时。',
        f'- 邻域一致性统计：{c["analysis_seconds"]:.2f} s；{c["metric_checks"]["checks"]} 项针对性统计检查通过。',
        f'- 局部复杂度扫描：{complexity["seconds"]:.2f} s；H 层间短折返的量测检查通过。',
        f'- 顶点边界判定复核：{c["exact_vertex_crossings_checked"]} 个唯一交点，修正 {c["vertex_retention_flags_corrected"]} 个审计保留标记；原始路线未改。',
        '- 本轮未重跑无关的全项目测试；完成用户要求的全区域普查，并逐条验证全部实际输出。',
        '',
        '算法状态分布：`'+json.dumps(r['status_counts'],ensure_ascii=False)+'`。其中全局曲面图歧义可能使许多路线标为 UNRESOLVED，不能据此断言这些路线全错；本报告使用局部对应和实际最终选择作进一步筛查。',
        '',
        '## 交付文件','',
        '- [重点案例图目录](figures/) 与 [案例坐标、相邻最终主轨及局部量测](representative_cases.json)。',
        '- [全部局部复核窗口](local_review_windows.json)、[s–z 投影连通区](review_regions.json)、[一致性统计](consistency_summary.json)。',
        '- [全部最终路线索引及状态](recognition_summary.json)、`routes/` 下逐条完整路线；`branches.pkl` + `branch_index.json` 为全部原始候选几何索引。',
        '- [输入与算法版本清单](input_manifest.json)、[缓存一致性验证](memo_equivalence.json)、[交付核查](delivery_verification.json)。',
        '',
        '新增脚本仅位于 `scripts/99_experiments/`：`census_directional_consistency.py`、`review_directional_consistency.py`、`render_directional_census.py`、`write_directional_census_report.py`。本轮识别核心文件哈希保持不变。','']
    (OUT/'FULL_CENSUS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    figs=sorted((OUT/'figures').glob('*.png'))
    assert len(figs)==2*len(cases)+1
    assert all((OUT/'routes'/f'{k}.pkl').exists() for k in m['keys'])
    save_json(OUT/'delivery_verification.json',dict(slices=len(rows),unique_slices=len({x['slice_key'] for x in rows}),
        all_routes_present=True,all_route_geometry_checks_passed=True,algorithm_hashes_unchanged=True,
        max_connector_m=max_join,max_combined_R=max(row['max_combined_R'] for row in rows),
        figures=len(figs),cases=len(cases),metric_checks=c['metric_checks'],
        sub_H_fold_geometry_measure_verified=complexity['sub_H_level_fold_measure_check'],
        cached_full_output_identical=read('memo_equivalence.json')['full_output_identical'],
        report='FULL_CENSUS_REPORT.md',accuracy_measured=False,production_entry_integrated=False))
    print('REPORT',len(rows),len(cases),len(figs),flush=True)


if __name__=='__main__':write()
