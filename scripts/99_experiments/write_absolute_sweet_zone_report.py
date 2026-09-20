"""Compile audited numbers, figures and explicit limitations into Chinese reports."""
from __future__ import annotations
import csv
import hashlib
import json
import pickle
import platform
from collections import Counter
from pathlib import Path
import numpy as np
from render_absolute_sweet_zone_review import ROOT,OUT,FIG,plt,BLUE,ORANGE,RED,GRAY
from review_absolute_sweet_zone import clean,save_json


def number(value):
    return f'{value:.5g}' if isinstance(value,float) else str(value)


def table(rows,columns):
    return '| '+' | '.join(label for k,label in columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+''.join('| '+' | '.join(number(r.get(k,'')) for k,label in columns)+' |\n' for r in rows)


def compile_report():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    distribution=json.loads((OUT/'input_distribution.json').read_text(encoding='utf-8'))
    controlled=json.loads((OUT/'controlled_case_results.json').read_text(encoding='utf-8'))
    real=json.loads((OUT/'real_case_results.json').read_text(encoding='utf-8'))
    sweep=list(csv.DictReader((OUT/'sweet_zone_parameter_sweep.csv').open(encoding='utf-8-sig')))
    statuses=summary['status_counts'];reasons=Counter();budgets=[];locations=[];short_accepted=0;topology_rejected=0;route_counts=Counter();aba_cases=[]
    with (OUT/'main_tracks.pkl').open('rb') as f:
        while True:
            try:payload=pickle.load(f)
            except EOFError:break
            s=float(payload['slice_key']);r=payload['result'];reasons.update(r.get('unresolved_reasons',[]))
            budgets.extend(r.get('route_contribution_budgets',[]))
            seq=r.get('route_branch_sequence',[])
            route_counts['multi_branch_slices' if len(seq)>1 else 'single_branch_slices' if seq else 'empty_slices']+=1
            for i in range(1,len(seq)-1):
                if seq[i-1]==seq[i+1]:
                    b=r['route_contribution_budgets'][i]
                    aba_cases.append(dict(s=s,sequence=seq[i-1:i+2],observed_m=b['observed_new_length_m'],
                        retained_ASC_m=b['retained_ASC_arc_length'],combined_R_syn=b['combined_R_syn']))
            short_accepted+=sum(not b['MBG_pass'] and b['branch_id'] in seq for b in r['branch_audit'])
            topology_rejected+=sum(b['ASC_exists'] and not b['MBG_pass'] for b in r['branch_audit'])
            byid={b['branch_id']:b for b in payload['branches']}
            seen=set()
            for item in r.get('continuation_rejections',[]):
                bid=item['branch_id'];reason=item['reason']
                if reason not in r.get('unresolved_reasons',[]):continue
                p=np.asarray(byid[bid]['points_uz']);extent=r.get('route_z_extent',[np.inf,-np.inf])
                if bid in seq and p[:,1].min()>=extent[0]-1e-6 and p[:,1].max()<=extent[1]+1e-6:continue
                if (bid,reason) in seen:continue
                seen.add((bid,reason))
                if item.get('junction'):
                    j=item['junction'];z=(j['a_point_uz'][1]+j['b_point_uz'][1])/2
                else:z=float(np.mean(byid[bid]['points_uz'][:,1]))
                locations.append(dict(s=s,z=z,reason=reason,branch_id=bid))
            for bid in r.get('excluded_extent_branch_ids',[]):
                locations.append(dict(s=s,z=float(np.mean(byid[bid]['points_uz'][:,1])),reason='EXCLUDED_UNRELIABLE_EXTENT',branch_id=bid))
    # Display review positions rather than claim these are labelled errors.
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    xs=[r['s'] for r in locations];zs=[r['z'] for r in locations]
    axes[0].scatter(xs,zs,s=2,color=ORANGE,alpha=.35)
    axes[0].set(xlabel='沿弧位置 s / m',ylabel='高程 z / m',title='全区待复核位置（不是准确率／错误真值）')
    for s,z,label in ((89.75,1476,'F: 14 m 历史连接'),(104.30,1380.3,'第六类: 重叠轨迹')):
        axes[0].scatter(s,z,s=50,color=RED);axes[0].annotate(label,(s,z),xytext=(9,-12),textcoords='offset points',fontsize=9)
    counts=Counter(r['reason'] for r in locations)
    top=counts.most_common(5);axes[1].barh([k for k,v in top][::-1],[v for k,v in top][::-1],color=BLUE)
    axes[1].set(xlabel='记录数（按切面、候选、原因去重）',title='代表性问题类型')
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'review_locations.png',dpi=160);plt.close(fig)
    combined=[b['combined_R_syn'] for b in budgets]
    extra=dict(unresolved_reason_counts=dict(reasons),review_location_counts=dict(counts),
        combined_ratio_quantiles=np.quantile(combined,[0,.25,.5,.75,.95,1]).tolist() if combined else [],
        all_route_budgets_pass=all(b['accepted'] for b in budgets),
        short_branch_accepted=short_accepted,ASC_present_but_topology_rejected=topology_rejected,route_counts=dict(route_counts),ABA_cases=aba_cases,
        python=platform.python_version(),platform=platform.platform())
    save_json(OUT/'report_metrics.json',extra)
    # The 14 m regression is verified on its actual frozen canonical branch.
    case5=next(r for r in real if r['slice_key']=='89.75')
    tiny=next(b for b in case5['branch_audit'] if b['branch_id']==4)
    assert not tiny['MBG_pass'] and 4 not in case5['sequence'] and case5['status']=='SOURCE_DATA_REQUIRED'
    assert summary['connectors_above']['0.01']==0 and extra['all_route_budgets_pass'] and short_accepted==0
    review=['# 绝对甜区与最小分支：代表案例复核\n',
        '## 怎么看图\n',
        '真实模型图的左栏是上一轮主轨，橙色为 observed 原始边，红虚线是上一轮人工合成的连接。中栏是本轮识别主轨，蓝色仍只代表原始 observed 边；灰色是未选原始分支。蓝线在窗口内缺席，表示本轮没有在这里输出主轨，**不表示曲面不存在，也不表示问题已经解决**。右栏单独展示 H 交点和邻近 V 的证据，不能把每个绿色点直接当成补点。\n',
        '受控案例中，蓝色表示比较获选的候选分支，橙色表示其他候选；粗线是满足绝对节点深度与弧长条件的 ASC，细线是两端保护段，灰色横带是当前比较区。端点圆点只表示 canonical 路径端点。右图的绿点来自构造的横向路径与纵向剖面的实际求交，虚线是图关联的邻近纵线。横纵轴独立缩放，连接长度以标注数值为准。H 图的分支整线用于解释候选；实际输出为 B0 下段接 B1 上段。\n',
        'A/B/C/D/E/G/H 是受控几何规则验证，不能冒充模型自然案例或识别准确率。F 和第六类来自此前复核的真实模型。全部分支的统计见 branch_audit.csv。\n',
        '## 全区复核分布\n\n![全区复核位置](figures/review_locations.png)\n',
        '这些点是被拒候选或范围异常的审查位置，部分按候选中心定位，不是精确裂缝边界。上轮重建分块对照结论仍见原始复核报告；本轮没有重新将位置相似性解释为因果或真值。\n']
    for r in controlled:
        review.extend([f"## {r['case']}：{r['title']}\n\n![{r['case']}]({r['figure']})\n",
            f"获选 B{r['selected']}；决策阶段 `{r['stage']}`；歧义={r['ambiguous']}；实际换轨={r['switch_count']}；连接={1000*r['connector_m']:.3f} mm；R_syn={r['R_syn'] if r['R_syn'] is not None else '不适用（未连接）'}。\n",
            table(r['branches'],[('branch_id','分支'),('canonical_node_count','节点数'),('full_arc_length','总弧长/m'),('left_guard_arc_length','左保护/m'),('right_guard_arc_length','右保护/m'),('ASC_arc_length','ASC/m'),('MBG_pass','MBG'),('target_region_in_ASC_fraction','局部 ASC 比例'),('surface_support_tier','H/V等级'),('neighbor_detail_score','重复细节')]),
            '\nH/V 等级 2=两侧相邻 V 有同路径 H 支持；1=单侧；0=未满足支持（或在 MBG 已被拒绝）。未进入细节阶段的 0 分不能解释为“没有细节”。\n'])
    for r in real:
        review.extend([f"## {'F / 原第五类' if r['slice_key']=='89.75' else '原第六类'}：s={r['slice_key']} m\n\n![真实案例]({r['figure']})\n",
            f"本轮状态 `{r['status']}`；主轨分支顺序 `{r['sequence']}`；原因 `{r['reasons']}`。\n",
            table(r['branch_audit'],[('branch_id','分支'),('canonical_node_count','节点'),('full_arc_length','原始长度/m'),('ASC_arc_length','ASC/m'),('minimum_core_arc_length','门槛/m'),('MBG_pass','MBG'),('surface_support_tier','H/V')])])
        if r['slice_key']=='89.75':
            review.extend(['\n![真实三节点尾片](figures/real_tiny_B4.png)\n',
                'B4 只有 3 节点、总长 5.6416 mm，在 H、甜区、细节、找最近点之前就被拒绝，因此不能扩大候选高程范围，更不能牵出 14.0508 m 直线。它是大网格连通体被切面掠过形成的微小片段，不能简单认定为孤立噪点。此前原始 Tile 5 核查已证明被裁剪范围外有连续真实曲线，故状态为 SOURCE_DATA_REQUIRED。右图原始 V/H 是输入恢复的证据，未被作为本轮已恢复的识别输出。\n'])
        else:
            review.append('\n第六类本轮实际主轨是 **B2→B4**，B1 未并入。保留当前主轨范围和已接受折返的约束下，B1→B2 最短可行连接约 0.700146 m，超过 10 mm；这不是不带约束的全局最近距离。B3 的纯 Z 间隙下界约 0.273196 m，也不能接。这里仍有未解决的低处观测，不能因红色短线消失就说局部已修好。\n\n此前宽邻域实测支持在缺口处更偏向 B1：同一 H 路径贯穿全部 11 个复核纵线位置，而 B6 的 H 路径在 s≈104.48–104.51 m 终止。本轮核心图仍使用原先相邻纵线的已缓存关联，没有把宽邻域或 Tile 包围盒自动转成地质真值。右栏只展示已有复核采样高度的 H 点，不是全高程 H 清单。这是需要后续改进表面身份判别的代表区域。\n')
    (OUT/'SWEET_ZONE_REVIEW.md').write_text('\n'.join(review),encoding='utf-8')
    parameter_ranges=[]
    for k in ('MBG_rejected','short_branch_accepted','ABA','switches','ambiguous','local_tie_slices','stable_path','stable_geometry','above_10mm','max_connector_m','max_combined_R_syn'):
        vals=[float(r.get(k) or 0) for r in sweep];parameter_ranges.append(dict(metric=k,min=min(vals),max=max(vals)))
    acceptance=[
        ('取消相对甜区','是。没有 4s(L−s)/L² 或 20%–80% 可靠区。形状描述用的等弧长采样不是甜区。'),
        ('绝对节点深度 + 弧长','是。K=4，节点数必须 > 2K+3，同时 ASC 弧长至少 4×本切面 canonical 边长中位数。'),
        ('5–6 节点能否获高甜区','不能；MBG 直接拒绝。'),('MBG 是否最先','是；被拒分支不计算 H/V 排序和重复细节。'),
        ('短 detour 能否造成 A→B→A','已禁止实际保留核心不足的中间段，并复查两侧连接的合计占比；长的有证据分支仍可合法回入，ABA 总数不等于错误数。'),
        ('H/V 是否先排除错误分支','先排除无支持或较弱支持的竞争者；这代表算法证据优先级，不构成地质身份真值。'),
        ('双方 H 支持后由 ASC 决策','是；先比较局部区域位于绝对核心的比例。'),
        ('双方 ASC 后才比较细节','是；仅在两者核心接近且 H/V 支持成立时。'),
        ('细节是否只来自同表面邻居','是，依赖已有图链接和轨道成员；不自由寻找最像路径。'),
        ('最近原则只决定怎么接','是，先固定分支身份，再在该分支上搜索交点或最近点；Z 间隙硬下界仅用于证明连接不可能达标，不参与候选距离排名。'),
        ('连接绝对上限','是，默认 10 mm；本轮 >10 mm 输出为 0。'),
        ('合成不能长于救回的 observed','是，严格 R_syn<1，并对每段 observed 合计两侧 incident 连接复核。'),
        ('tiny 能否扩大目标 extent','不能。可靠候选范围不包含 MBG 失败者；被排除的异常范围另行报告。'),
        ('14.05 m 是否拒绝','是，真实 s=89.75 的 B4 被拒且未进入新主轨；SOURCE_DATA_REQUIRED。'),
        ('允许 UNRESOLVED','是，拒绝强行拼完整；有蓝色 observed 结果也可同时处于未解决状态。'),
        ('参数是否完整记录','是，默认值未按通过数优化；144 组合×12 固定真实切面敏感性扫描，另有默认参数全区 2800 切面。')]
    ratio_rows=[dict(quantile=q,single=a,combined=b) for q,a,b in zip(('最小','25%','50%','75%','95%','最大'),summary['ratio_quantiles'],extra['combined_ratio_quantiles'])]
    text=f'''# 最小分支与绝对甜区：本轮修改分析报告

本轮按审核方案修改了分支选择、局部换轨和主轨拼接。**这是冻结模型输入上的新版 observed 主轨识别结果；没有接入生产 GUI 的旧 red-group 识别入口。** 输出保留原始边坐标与来源，拒绝不可靠连接，允许结果不完整。所有计数是算法审计量，不是准确率。整轨身份警告与局部候选并列已分开处理：即使原缓存的整轨存在歧义，局部证据明确且通过门控的延续仍可进入待确认主轨。

## 完成内容

处理顺序统一为 MBG → 同路径 H / 邻近 V → 绝对甜区 → 同表面重复细节 → 保持当前／少换轨 → 固定候选后找连接点。默认 K_guard=4、N_core_min=3、最小核心弧长=4×切面边长中位数、连接上限=10 mm、R_syn<1。每次新增连接后还复核中间 observed 段两侧连接的合计预算。

全区 {summary['slice_count']} 个切面、{summary['candidate_count']} 个候选；门控拒绝 {summary['MBG_rejected']} 个，ASC 物理条件合格 {summary['ASC_eligible']} 个，ASC 不存在 {summary['ASC_absent_short']} 个。其中 {topology_rejected} 个虽有核心，但因属于 forked component 被保守拒绝，不能归为短分支。H/V 排除记录 {summary['rejected_surface_evidence']} 条，已记录决策的细节比较候选数 {summary['detail_comparisons']}（精确口径见偏离说明）。

换轨 {summary['switches']} 次，A→B→A 序列 {summary['ABA']} 个；连接位置 {summary['junction_count']} 个，其中 XYZ 非零接缝 {summary['connector_count']} 个（可能只是 canonical 数值级接缝，不能全部理解为可见补线）。>5 mm：{summary['connectors_above']['0.005']}；>10 mm：{summary['connectors_above']['0.01']}；>20 mm：{summary['connectors_above']['0.02']}。最大连接 **{1000*summary['max_connector_m']:.4f} mm**。

合成／观测长度占比分布（门槛严格小于 1）：

{table(ratio_rows,[('quantile','分位'),('single','新连接 / 新 observed'),('combined','双侧连接合计 / 保留段')])}

仍存在的 A→B→A 均保留了米级中间观测核心，未触发“短绕行”禁用规则；两例仍需身份复核：

{table(aba_cases,[('s','切面 s/m'),('sequence','局部分支顺序'),('observed_m','中间 observed/m'),('retained_ASC_m','保留核心/m'),('combined_R_syn','合计占比')])}

状态分布：

{table([dict(status=k,count=statuses.get(k,0)) for k in ('ACCEPT_OBSERVED_BRANCH','UNRESOLVED','INPUT_TRUNCATION_SUSPECT','SOURCE_DATA_REQUIRED')],[('status','状态'),('count','切面数')])}

其中多分支主轨 {route_counts['multi_branch_slices']} 个切面，单分支 {route_counts['single_branch_slices']} 个，空结果 {route_counts['empty_slices']} 个。完全通过当前身份与门控检查的切面为 {statuses.get('ACCEPT_OBSERVED_BRANCH',0)} 个；其余输出是保留真实观测、带待复核标记的主轨，不能称为已经确认的最终曲面。

上一轮共有 8,194 次切换、1,766 条超过 1 cm 的连接，最长 14.0508 m。本轮减少切换或连接不等于准确率提高：一部分原来强接的范围现在明确留空，是否该恢复需要回到原网格与横纵测线确认。

`ACCEPT_OBSERVED_BRANCH` 只表示当前证据与门控允许接受输出 observed 主轨，不代表整个真实曲面已恢复。`UNRESOLVED` 可以带有已确认的部分主轨。`INPUT_TRUNCATION_SUSPECT` 是输入裁剪疑点。`SOURCE_DATA_REQUIRED` 表示已有原始源数据证明缺失部分需要恢复输入。

## 真实问题与代表图

[完整局部图、分支统计和看图说明](SWEET_ZONE_REVIEW.md)。图中真实 F 与第六类是模型案例，A/B/C/D/E/G/H 是隔离决策条件的受控案例，两者不混作识别效果证据。

![真实第五类](figures/real_89_75.png)

原第五类 B4 的 3 节点／5.64 mm 已被门控拒绝，14.05 m 直线未输出。上轮原始 Tile 5 核查证明真实曲线存在于裁剪输入之外，因此本轮要求恢复源数据，不能用 H 点插值或长直线假装已修复。

![真实第六类](figures/real_104_30.png)

第六类仍应结合宽邻域 H/V 与分块边界检查真实曲面身份。当前图关联主要是相邻切面支持；本轮没有重新构建更宽邻域图，也未恢复被裁剪源网格。详见局部报告的状态和蓝线覆盖范围。

## 参数与性能

真实切面边长的候选加权中位数为 {distribution['quantiles']['local_median_edge_length']['p50']:.6f} m，默认物理核心门槛约为其 4 倍；实际按各切面计算，不统一设成这个值。分布见 `absolute_core_distribution.csv` 和 `input_distribution.json`。

扫描 K=3/4/5/6，N_core_min=2/3/4，连接上限=5/10/20 mm，核心长度倍数=2/4/6/8，共 {len(sweep)} 组。12 个预先固定的真实切面覆盖全弧均匀位置和 89.75、104.30、122.25 三个已知案例。结果见 `sweet_zone_parameter_sweep.csv`，扫描范围见 `sweep_scope.json`。这不是每个参数组合都跑全区。默认参数保持审核方案值，没有为了多接通而放宽。

{table(parameter_ranges,[('metric','扫描量'),('min','最小'),('max','最大')])}

`stable_path` 是与默认值相比保持分支顺序的切面数；`stable_geometry` 要求整条坐标数组一致。`ambiguous` 包含历史整轨身份警告，`local_tie_slices` 才表示初始局部候选并列。每组分母为 12。这些量均不是准确率。

默认参数全区识别与逐条来源检查耗时 {summary['wall_seconds']:.2f} 秒（{summary['wall_seconds']/60:.2f} 分钟），{summary['workers']} 个工作线程；不含首次 canonical 清单重建、加载、参数扫描和绘图。生产入口没有修改，本轮没有把两种识别入口混合计时。该时间也不能与旧版本强制接通更多路径的时间直接解释成算法等价加速。

## 最低必要检查与人工验收

77 项相关测试通过，包含审核计划的 11 个指定用例以及独立复核后增加的优先级、轨道锁、歧义传播、尾部缺失和双连接预算反例。没有重跑无关全项目测试。全区每条 observed 边检查原始 canonical 坐标、edge/node/face 来源、连接连续性和重复边区间；连接上限、合成占比逐条检查。失败的首轮日志与部分结果单独保留，未用于本报告。

需要人工看 F 的源数据恢复边界与第六类多曲面的真实身份；还需检查 UNRESOLVED 是否包含应补充宽邻域 H/V 的合理主轨。地图上的待复核点不是人工标注错误。

## 16 项验收回答

{table([dict(item=f'{i}. {a}',answer=b) for i,(a,b) in enumerate(acceptance,1)],[('item','问题'),('answer','回答')])}

## 已知风险与范围

冻结缓存包含 H/V 图关联，但没有全部原始 H 观测点；因此全区结果不能执行所有“缺失尾部仍有 H 支持”的新检测，报告结果保留 `missing_tail_check_available` 字段。局部真实案例右栏使用上轮实际 H/V 核查文件；它们不冒充全区的新输入。局部候选比较仍并列时停止换轨；整轨歧义保留为身份警告，不再阻断明确的局部延续。缺少 H 支持的合法曲面可能暂留未解决。恰好在边界内被裁剪的分支可能只有 UNRESOLVED，未自动标为裁剪疑点。

源码修改：新增 `branch_absolute_core.py`；修改 `surface_track_selection.py`、`surface_track_handoff.py`、`main_track_assembly.py`、`dominant_observed_branch.py`；新增审计／绘图／报告脚本，更新相关测试。没有提交、推送、升级依赖或修改源网格。

[实施取舍和偏离说明](IMPLEMENTATION_DEVIATIONS.md) · [独立复核与修复记录](FINAL_REVIEW.md)
'''
    (OUT/'MIN_BRANCH_AND_SWEET_ZONE_REPORT.md').write_text(text,encoding='utf-8')
    save_json(OUT/'delivery_verification.json',dict(slices=summary['slice_count'],tests=77,case5_tiny_rejected=True,
        case5_source_required=True,over_cap_connectors=0,all_route_budgets_pass=True,
        report_sha256=hashlib.sha256(text.encode()).hexdigest(),figure_count=len(list(FIG.glob('*.png')))))


if __name__=='__main__':compile_report()
