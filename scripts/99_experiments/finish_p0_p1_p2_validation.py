"""Build revised-plan review reports and check delivered source intervals."""
import argparse,csv,json,pickle,re,hashlib
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from review_p0_p1_p2 import (latest,SOURCE,BASELINE,load_inventory,save_json,code_hashes,
    plt,Line2D,crop,table,colored_parts,branch_metrics,local_edge_scale)


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))


def j_report(out):
    summary=read_json(out/'P2_summary.json');rows=summary['all_10880_rows']
    selected=next(r for r in rows if (r['closure_gap_mm'],r['minimum_excursion_arc_m'],r['minimum_ratio'])==(10,.5,50))
    details=pickle.load((out/'peeling/gap_10mm_arc_0.5m_ratio_50.pkl').open('rb'))
    event=details['108.80'][2]['excursions'][0]
    report=['# J_T1：单测线 Main Spine / Side Component 审核',
        '**结论：108.80 米的右侧绕行可以仅凭本测线识别并分离；这是参数扫描中的影子结果，尚未接入全区主轨。**',
        f"图示组合为闭合距离 10 mm、最小绕行弧长 0.5 m、弧长/直距比 50。分离弧长 {selected['side_arc_m']:.6f} m，显式连接 {selected['max_bridge_m']*1000:.6f} mm，连接预算最大 R_syn={selected['max_bridge_R_syn']:.8f}<1。主脊连续。",
        '## 怎么看图',
        '从左到右依次是 108.70、108.75、108.80、108.85、108.90 米，每栏同一测线。上排灰线是同一 J_T1 的原始曲线；中排绿线是影子主脊；下排紫线是完整保留的旁支。三排共享坐标，先在中间栏上下对比，再横向看相邻测线。只有绿色虚线是新增短连接，其他线均来自原始边。图框边界是显示裁切，不是路径端点。',
        '重点看中间栏右侧约 u=-9、z=1402.2–1402.5 m 的绕行：上排存在，中排移出，下排完整保存。左侧小绕行在这个参数组合下仍留在主脊，不能说所有旁支都已分离。',
        '![五邻线局部细节](figures/J_T1_peeling_detail.png)',
        '![五邻线较大局部](figures/J_T1_peeling.png)',
        '## 108.80 米的原始证据',
        table(['量','实测值'],[
            ('沿原路径 L_path（m）',f"{event['observed_arc_m']:.9f}"),('D_xyz（mm）',f"{event['D_xyz_m']*1000:.9f}"),
            ('D_uz（mm）',f"{event['D_uz_m']*1000:.9f}"),('净高程变化 Δz（mm）',f"{event['delta_z_net_m']*1000:.9f}"),
            ('L_path / D_xyz',f"{event['arc_gap_ratio']:.3f}"),('起止原始弧长参数（m）',f"{event['start_arc']:.9f} → {event['end_arc']:.9f}")]),
        '检测只使用本条原始折线的非相邻边最近点、弧长和返回距离。没有重采样、坐标平均、平滑，也没有要求邻线先出现闭环。邻线 108.75 的 B5（约 1.234 m）和 108.85 的 B5（约 0.583 m）本来就是独立闭合分量；它们作为额外对照保存，不能当成 108.80 的剥离许可。',
        '## 36 组参数的正反结果',
        '2/5 mm 的 18 组没有检出 108.80 绕行；“连续”只表示原路径未被改动，不能算剥离成功。10 mm 的 9 组均得到同一个约 1.828 m 绕行。20 mm 的 9 组也检出右侧绕行，其中最小弧长 0.25/0.5 m 且比值 20 的 2 组还检出左侧约 0.557 m 小绕行，但留下 13.439 mm 断口，因此记录 unresolved，不强接。',
        table(['距离 mm','最小弧长 m','最小比值','分离段数','旁支弧长 m','最大实接 mm','R_syn','连续','未解决'],[
            (r['closure_gap_mm'],r['minimum_excursion_arc_m'],r['minimum_ratio'],r['near_closed_excursions'],f"{r['side_arc_m']:.6f}",f"{r['max_bridge_m']*1000:.6f}",f"{r['max_bridge_R_syn']:.6f}",r['continuous'],r['unresolved_count']) for r in rows]),
        '## 20 mm 扫描的未解决对照',
        '下图使用 20 mm / 0.5 m / 20，仅扩大检测范围；实际连接上限仍为 10 mm。108.80 中排左侧红注处保留断开，下排多出对应的小绕行。右侧仍采用合法的 8.099 mm 切口，不能因为另一个略长、但超限的切口而否认已有合法切口。',
        '![20 毫米检测的未解决对照](figures/J_T1_peeling_20mm_detail.png)',
        '## 保存与审核边界',
        '每个扫描组合分别保存 MAIN_SPINE、SIDE_COMPONENT、REJECTED_REDUNDANT。原绕行的 observed geometry、FaceID、source segment indices、edge_id、原始 t 区间保留；短连接没有伪造 FaceID。全区三层数据另存于 layers，尚未采用本参数扫描的剥离建议。',
        '数据：[36 组 × 5 测线统计](P2_parameter_scan.csv)；[原始候选事件](P2_events.json)；[10 mm 示例原始三层数据](peeling/gap_10mm_arc_0.5m_ratio_50.pkl)；[20 mm 反例三层数据](peeling/gap_20mm_arc_0.5m_ratio_20.pkl)。',
        '人工审核：右侧分离结构是否确实应作为 Side Component？左侧残余小绕行应继续保留还是等待更强证据？不能仅因图上更整齐而确认地质身份。本轮不锁定最终阈值。']
    (out/'J_T1_MAIN_SPINE_AUDIT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')


def plot_records(ax,records,bounds,color,lw=1.8):
    for e in records:
        p=crop(e['points_uz'],bounds)
        if p is not None:ax.plot(p[:,0],p[:,1],color=color,lw=lw,ls='-' if e['source'].startswith('OBSERVED') else '--')


def controls_figures(out):
    dynamic=read_json(out/'dynamic_tail_shadow.json')
    for row in dynamic:
        s=row['s'];bs=load_inventory(SOURCE,[s])[s];by={b['branch_id']:b for b in bs};scale=local_edge_scale(bs)
        old=by[row['from_branch']];candidate=by[row['to_branch']];m=branch_metrics(old,scale)
        point=np.asarray(row['junction']['a_point_uz']);z0,z1=point[1]-1.5,max(point[1]+1.5,np.max(old['points_uz'],axis=0)[1]+.2)
        parts=[crop(e['points_uz'],[-1e6,1e6,z0,z1]) for b in [old,candidate] for e in b['records']];points=np.concatenate([p for p in parts if p is not None]);pad=max(.12,np.ptp(points[:,0])*.1)
        bounds=[points[:,0].min()-pad,points[:,0].max()+pad,z0,z1]
        fig,axs=plt.subplots(1,2,figsize=(13,8),sharex=True,sharey=True)
        plot_records(axs[0],candidate['records'],bounds,'#b8bec6',1.5)
        members=[dict(e,branch_id=old['branch_id']) for e in old['records']]
        for line,tag in colored_parts(members,by,{old['branch_id']:m}):
            p=crop(line,bounds)
            if p is not None:axs[0].plot(p[:,0],p[:,1],color='#19855c' if tag=='ASC' else '#d78426',lw=2.3)
        proposal=pickle.load((out/'hard_regressions'/f'{s}_dynamic_tail_shadow.pkl').open('rb'))['proposed_geometry']
        plot_records(axs[1],old['records'],bounds,'#c9cdd3',1.3)
        plot_records(axs[1],proposal,bounds,'#147d69',2.2)
        for ax in axs:
            ax.scatter(*point,c='#9b42a3',s=35,zorder=5)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_xlabel('径向偏移 u（m）');ax.grid(alpha=.17);ax.ticklabel_format(useOffset=False,style='plain')
        axs[0].set_ylabel('高程 z（m）');axs[0].set_title('静态 ASC 对照：绿=核心，橙=端点非甜区\n灰=待接分支；紫点=更早交点')
        axs[1].set_title(f"动态尾区单接点建议（影子）\n将裁掉现有 ASC {row['ASC_removed_m']:.3f} m，未获安全认定")
        fig.suptitle(f"s={s} m｜B{row['from_branch']} → B{row['to_branch']}：能相交与可裁核心分别审核",fontsize=15)
        fig.text(.5,.025,'右图只演示这一处替换，不是重新完成的识别主轨；正式 P0 仍采用左图的静态 ASC 保护。',ha='center')
        fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.105,wspace=.12)
        fig.savefig(out/'figures'/f'{s}_static_dynamic_tail.png',dpi=160);plt.close(fig)
    for s in ['98.05','98.10']:
        current=pickle.load((out/'hard_regressions'/f'{s}.pkl').open('rb'))['result'];before=pickle.load((BASELINE/'routes'/f'{s}.pkl').open('rb'))['result']
        bs=load_inventory(SOURCE,[s])[s];junction=next(j for j in current['junctions'] if j['from_branch_id']==6 and j['to_branch_id']==3)
        u,z=junction['a_point_uz'];bounds=[u-2.2,u+2.2,z-.9,z+.9]
        fig,axs=plt.subplots(1,2,figsize=(13,8),sharex=True,sharey=True)
        for ax,result,color,title in zip(axs,[before,current],['#b7773e','#147d69'],['上一轮主轨','本轮 P0 主轨（身份仍需审核）']):
            for b in bs:plot_records(ax,b['records'],bounds,'#c2c7ce',1.)
            plot_records(ax,result['path_edges'],bounds,color,2.6)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_xlabel('径向偏移 u（m）');ax.set_title(title);ax.grid(alpha=.17);ax.ticklabel_format(useOffset=False,style='plain')
        axs[0].set_ylabel('高程 z（m）');fig.suptitle(f's={s} m｜首选不可接时，继续尝试备选分支',fontsize=16)
        fig.text(.5,.035,'灰线=原始候选；棕/绿=各轮输出。这里只比较连续性；本轮仍保留 AMBIGUOUS_SURFACE_IDENTITY 标记。',ha='center')
        fig.subplots_adjust(left=.075,right=.98,top=.89,bottom=.105,wspace=.12)
        fig.savefig(out/'figures'/f'{s}_fallback_comparison.png',dpi=160);plt.close(fig)


def verify_layers(branches,layers):
    originals={e['edge_id']:e for b in branches for e in b['records']};intervals=defaultdict(list);count=0
    for records in layers:
        for e in records:
            if not e['source'].startswith('OBSERVED'):continue
            source=originals[e['edge_id']];count+=1
            lo,hi=sorted([float(e['t0']),float(e['t1'])]);intervals[e['edge_id']].append((lo,hi))
            for field in ['face_id','source_face_ids','source_segment_indices']:
                assert np.array_equal(e.get(field),source.get(field)),(e['edge_id'],field,'provenance changed')
            t=(np.asarray([e['t0'],e['t1']])-source['t0'])/(source['t1']-source['t0'])
            for field in ['points_xyz','points_uz']:
                p=np.asarray(source[field]);expected=p[0]+t[:,None]*(p[1]-p[0])
                assert np.max(np.abs(np.asarray(e[field])-expected))<1e-8,(e['edge_id'],field,'geometry changed')
    for eid,e in originals.items():
        low,high=sorted([e['t0'],e['t1']]);pieces=sorted(intervals[eid]);assert pieces,(eid,'absent')
        cursor=low
        for a,b in pieces:
            assert abs(a-cursor)<1e-8,(eid,a,cursor,'overlap or missing interval')
            cursor=b
        assert abs(cursor-high)<1e-8,(eid,'missing end')
    return count


def delivery_check(out):
    from run_face_provenance_validation import manifest
    keys=manifest(SOURCE)['keys'];checked=0
    assert len(list((out/'routes').glob('*.pkl')))==len(keys)==2800
    assert code_hashes()==read_json(out/'manifest.json')['code_sha256']
    with (SOURCE/'branches.pkl').open('rb') as stream:
        for index,s in enumerate(keys):
            bs=pickle.load(stream);r=pickle.load((out/'layers'/f'{s}.pkl').open('rb'))
            checked+=verify_layers(bs,[r[k] for k in ['MAIN_SPINE','SIDE_COMPONENT','REJECTED_REDUNDANT']])
            if (index+1)%400==0:print('PROVENANCE_CHECK',index+1,flush=True)
    data=pickle.load((SOURCE/'region_data/J.pkl').open('rb'));inventory=load_inventory(SOURCE,data['region']['target_keys']);p2count=0
    for path in sorted((out/'peeling').glob('*.pkl')):
        for s,results in pickle.load(path.open('rb')).items():
            by={b['branch_id']:b for b in inventory[s]}
            for bid,r in results.items():
                p2count+=verify_layers([by[bid]],[r[k] for k in ['MAIN_SPINE','SIDE_COMPONENT','REJECTED_REDUNDANT']])
                assert r['max_bridge_m']<=.010+1e-12 and r['max_bridge_R_syn']<1
                assert not r['used_neighbor_permission']
                for e in r['MAIN_SPINE']:
                    if not e['source'].startswith('OBSERVED'):
                        assert np.linalg.norm(np.diff(e['points_xyz'],axis=0))<=.010+1e-12
                        assert e['face_id'] is None and not e['source_segment_indices']
    p1=read_json(out/'P1_gallery_index.json');assert len(p1)==23
    assert {'C','D','E','F','J'}<={g['region_id'] for g in p1}
    assert all(len(g['keys'])==5 and (out/'figures'/g['figure']).is_file() for g in p1)
    p0=read_json(out/'P0_summary.json');assert p0['count']==2800
    result=dict(all_2800_routes_and_layers_present=True,production_hash_matches_census=True,
        original_intervals_partition_without_gaps_or_overlap=True,geometry_FaceID_and_source_segments_preserved=True,
        global_observed_records_checked=checked,P2_observed_records_checked=p2count,P2_combinations=36,
        P2_all_bridges_le_10mm_and_R_syn_lt_1=True,FaceTrack_gallery_groups=23,mandatory_regions_present=True)
    save_json(out/'DELIVERY_CHECKS.json',result);return result


def final_report(out):
    p0=read_json(out/'P0_summary.json');p1=read_json(out/'P1_summary.json');p2=read_json(out/'P2_summary.json');census=read_json(out/'census_summary.json')
    old=read_json(BASELINE/'missing_extent_summary.json');hard=read_json(out/'hard_regressions.json');dynamic=read_json(out/'dynamic_tail_shadow.json')
    alternative=read_json(out/'alternative_junction_summary.json')
    regression=read_json(out/'static_ASC_regression_summary.json')
    regression_details=read_json(out/'static_ASC_regression_details.json')
    # Clarify metadata emitted before the read-only review. Geometry and code
    # hashes remain those of the frozen census; no result is recomputed here.
    metadata=read_json(out/'manifest.json');metadata.pop('strict_static_ASC',None)
    metadata.update(ordinary_continuation_strict_static_ASC=True,internal_handoff_policy='unchanged directional-tail policy')
    save_json(out/'manifest.json',metadata)
    for row in hard:
        row.pop('strict_static_control',None)
        row.update(ordinary_continuation_strict_static_ASC=True,internal_handoff_policy='unchanged directional-tail policy')
    save_json(out/'hard_regressions.json',hard)
    p0.update(single_nearest_junction_scope=True,alternative_finite_junction_audit=alternative,
        exhaustive_continuous_feasibility_proven=False)
    save_json(out/'P0_summary.json',p0)
    with (out/'ASC_missing_inventory.csv').open(encoding='utf-8-sig',newline='') as f:missing=[r for r in csv.DictReader(f) if r['ASC_missing']=='True']
    wide=[r for r in p2['all_10880_rows'] if r['unresolved_count']]
    status={'P0_MAIN_TRACK_CONTINUITY':'NOT_SUPPORTED','ALTERNATIVE_CANDIDATE_FALLBACK':'SUPPORTED',
        'FACETRACK_SWEET_ZONE_SELECTION':'PARTIALLY_SUPPORTED','SINGLE_PROFILE_MAIN_SPINE_PEELING':'PARTIALLY_SUPPORTED',
        'SIDE_COMPONENT_PRESERVATION':'SUPPORTED','J_T1_10880':'SUPPORTED','READY_FOR_FACETRACK_GUIDED_SELECTION':'NO'}
    save_json(out/'FINAL_STATUS.json',status)
    report=['# P0 / P1 / P2 修正版验证报告',
        f"本轮完成 2800 条 V 全区域普查、12 个局部区域 / {p1['groups']} 组 FaceTrack 五邻线图册，以及 J_T1 的 36 组参数扫描。**全区连续性验收未通过：ASC 高程缺失从 148 条增至 315 条，≥1 m 缺失从 119 条增至 300 条。备选回退局部有效，但严格保留静态 ASC 引出了更多停轨，不能把这版作为已修好的生产识别版本。**",
        '本轮实际识别结果是 `routes/*.pkl` 中的 P0 主轨，但它未通过全区连续性验收。P1 的 FaceTrack×甜区选择和 P2 的剥离均为 shadow；动态尾区也是单接点建议，没有写回本轮主轨。不能把三种影子图当成新的全区识别结果。',
        '## 建议阅读顺序',
        '1. 本报告的 P0 四个回归与未解决部分。\n2. [FaceTrack × 甜区五邻线图册](FACETRACK_SWEET_ZONE_REVIEW.md)：同来源分组，人工判断哪个分支可信。\n3. [J_T1 主脊/旁支分离图册](J_T1_MAIN_SPINE_AUDIT.md)：原始、主脊、旁支三排对照。',
        '## P0：同口径全区结果',
        table(['指标','上一轮','本轮'],[
            ('测线数',2800,p0['count']),('ASC_missing_count',old['uncovered_ASC_slices'],p0['ASC_missing_count']),
            ('ASC_missing_ge_1m_count',old['at_least_1m_uncovered_ASC_slices'],p0['ASC_missing_ge_1m_count']),
            ('停止但仍有合法续接：每候选最近接点','此前反事实探查口径不同，不作直接数值比较',p0['LEGAL_CONTINUATION_EXISTS_BUT_ROUTE_STOPPED']),
            ('停止但仍有合法续接：补查全部有限接点','本轮新增',alternative['legal_stopped_slices']),
            ('全部可靠分支 Z 范围未覆盖（含端点 guard）',old['full_branch_extent_deficit_slices'],census['slices_with_extent_deficit'])]),
        'ASC_missing 的定义是：至少一个 MBG 合格分支的 ASC 高程范围伸出了当前主轨高程范围，阈值 1e-6 m；≥1 m 使用同一测线最大缺失量。它不统计已有高程范围内所有未选竞争分支，也不等同地质真值、准确率或局部形态一致性。自然端点差异与中部缺失不能仅凭一个计数混为一谈。',
        f"第一层停止审计在最终路由上重新枚举未使用的可延展候选，共检查 {p0['candidate_pieces_audited']} 个候选片段，每片检查生产搜索返回的最近接点；保留普通续接的静态 ASC、贡献量、10 mm 和 R_syn 门槛，不以身份歧义隐藏几何可行候选。结果为 {p0['LEGAL_CONTINUATION_EXISTS_BUT_ROUTE_STOPPED']} 条测线 / {p0['legal_stopped_events']} 个事件。", 
        f"只读代码复核指出，单个最近接点不可行不代表该候选所有接点都不可行。因此又补查每个边对的真实交点与四类端点投影，在所有符合静态保护、延展、10 mm 条件的 {alternative['finite_junctions_tested']} 个有限接点上逐一复核贡献量和双侧预算，发现 {alternative['legal_stopped_slices']} 条仍有合法续接的测线。见 [补充接点审计](alternative_junction_audit.csv) 和 [反例几何](alternative_junction_counterexamples.json)。这覆盖当前算法声明的有限接点集合，未证明任意两段内部连续参数空间都无解，故不将数值零夸大为一般几何可行性的穷尽证明。",
        '数据：[全部缺失清单](ASC_missing_inventory.csv)、[最终停止候选逐项探查](stopped_continuation_audit.csv)、[当前断点日志阶段](current_missing_routes.csv)、[全部主轨清单](all_route_inventory.csv)。日志最后一条拒绝原因只是阶段线索，不能直接当成共同因果解释。',
        '### 新增缺失的共同原因：静态 ASC 保护范围与可替换重叠段冲突',
        f"逐测线对照：本轮新增 ASC 缺失 {regression['new_missing']} 条，消除 {regression['cleared_missing']} 条；新增 ≥1 m 缺失 {regression['newly_ge_1m']} 条，消除 {regression['cleared_ge_1m']} 条。不能只展示 98.05/98.10 的改善而忽略这些退步。",
        f"对新增 ≥1 m 的 {regression['newly_ge_1m']} 条，在本轮停止位置做单接点因果对照：只恢复上一轮的折返保护口径，其余候选、10 mm、最小保留核心、R_syn 和延展门槛不变。{regression['new_large_with_static_ASC_counterfactual']} 条重新出现旧规则下可行的接点，但该接点会裁掉现有静态 ASC；另 {regression['new_large_not_explained_by_this_single_junction_control']} 条没有被这个单接点对照解释。数据见 [新增缺失原因逐条表](new_ASC_missing_regression_causes.csv) 和 [接点及核心损失](static_ASC_regression_details.json)。该对照只确认局部阻断机制，不把整条反事实路由冒充已完成结果。",
        '以 112.55 米为例，上一轮主轨达到 z=1476.389 m，本轮在 z=1380.898 m 停止，可靠候选的 ASC 高程缺失达到 92.240 m。旧接点近乎相交，本轮保留静态 ASC 后的接点距离约 237.695 mm，超过 10 mm。这里不是测线上没有后续观测，而是当前保护口径把可用的早期接点排除了。',
        '这说明需要先复核“静态 ASC 都是不可替换核心”的定义，区分真正应保护的细节与重叠段/方向尾区。按照用户本轮裁定，本轮不直接批准裁剪这些核心，不放宽连接上限，也不靠几何平滑遮盖缺失。',
        table(['代表测线','旧口径连接 mm','静态口径连接 mm','若采用旧接点会裁 ASC m'],[
            (s,f"{d['previous_rule_gap_m']*1000:.9f}",f"{d['static_rule_gap_m']*1000:.6f}" if d['static_rule_gap_m'] is not None else '无接点',f"{d['removed_static_ASC_m']:.6f}")
            for s in regression['representatives'] for d in [next(x for x in regression_details if x['s']==s)]]),
        *[f'![{s} 新增停轨的局部对照](figures/{s}_static_ASC_regression.png)' for s in regression['representatives']],
        '### 四个硬回归',
        table(['s（m）','上一轮 Z 范围','本轮 Z 范围','本轮分支序列','解释'],[
            (r['s'],r['before']['route_z_extent'],r['after']['route_z_extent'],r['after']['sequence'],
             'B2 失败后继续评估并采用 B6；仍有身份歧义标签' if r['s'] in ['98.05','98.10'] else '静态 ASC 内早期交点仍不允许剪核心') for r in hard]),
        '身份排序仍由已有的身份判据产生；每个候选再接受几何可行性检查。全部候选失败才封锁该方向，接受新连接后重新核查前沿。同身份并列采用稳定的临时顺序尝试，并明确保留歧义标记，不能当成已判定真实曲面。98.05/98.10 的覆盖改善正说明连续性与身份确认是两项不同验收。',
        '![98.05 备选回退局部对照](figures/98.05_fallback_comparison.png)',
        '![98.10 备选回退局部对照](figures/98.10_fallback_comparison.png)',
        '### 保留静态 ASC，对动态尾区单独审核',
        '按照本轮用户裁定，下面两例只允许另做动态尾区验证，保留静态 ASC 对照，不直接认定可剪核心。普通 continuation 新增的保护依据是已保留的静态 ASC；原有 INTERNAL_HANDOFF 的方向尾区规则本轮没有改写。例如 98.10 仍包含既有内部换轨，不能说整条算法的所有模块都只使用静态 ASC。',
        table(['s','静态保护下最近连接 mm','早期交点距离 mm','会移除的既有 ASC m','正式结果'],[
            (r['s'],f"{r['static_connector_m']*1000:.6f}",f"{r['shadow_connector_m']*1000:.9f}",f"{r['ASC_removed_m']:.6f}",'未采用动态建议') for r in dynamic]),
        '这两例的共同矛盾已经定位：早期交点在现有静态 ASC 内，保留核心后剩余连接又超过 10 mm。仅解除“折返保护”并不能同时满足两个要求。下一步需要定义什么证据允许把这部分原有 ASC 重新归为方向尾区；本轮只展示候选几何和裁剪代价，未作这种重新分类。',
        '![122.95 静态/动态尾区对照](figures/122.95_static_dynamic_tail.png)',
        '![116.25 静态/动态尾区对照](figures/116.25_static_dynamic_tail.png)',
        '## P1：甜区的作用与限制',
        f"{p1['decisions']} 次方向性影子判定中，{p1['selected']} 次给出唯一选择，{p1['ambiguous']} 次保留 AMBIGUOUS。ASC 阶段进一步缩小候选集合 {p1['ASC_discriminative_decisions']} 次。这是筛选过程的作用统计，不是正确率。多个来源同时位于 ASC 时，甜区不能单独解决身份问题。", 
        table(['区域','s（m）','同时位于 ASC 的 FaceTrack'],[(r['region_id'],r['s'],', '.join(r['FaceTracks_in_ASC'])) for r in p1['multiple_FaceTracks_in_ASC']]),
        '固定顺序及全部字段见[图册](FACETRACK_SWEET_ZONE_REVIEW.md)。FaceTrack 连续性按五条邻线的实际出现数；竞争区是当前案例窗口中的来源边区间；最少换轨为现有深度 2 的局部可达性估计，非全局最优解。上下行分别判断，没有用曲率、复杂度、几何相似度选身份。',
        '人工对照中一个明确的限制见 C_T1：103.50 米的曲线较简单，103.55–103.70 米在约 z=1363.7 m 与 1362–1363 m 逐渐出现更突出的折返，而这些局部都被标成绿色 ASC。因而“同 FaceTrack + 位于甜区”并不自动保证邻线细节复杂度一致。这是需要你继续审核的保留问题，未用形状公式强行抹平。',
        '![C_T1 同来源且同为 ASC，细节仍有差异](figures/C_T1_sweet_detail.png)',
        '## P2：单测线绕行分离',
        f"J_T1 / 108.80 在 10 mm / 0.5 m / 50 的展示组合中，约 1.828 m 绕行进入 Side Component，8.099 mm 显式连接使主脊保持连续，最大 R_syn 约 0.000704。邻线不参与剥离许可。2/5 mm 设置未检出；宽扫描有 {len(wide)} 组保留超限断口，不强连。完整正反结果见 [J_T1 审核](J_T1_MAIN_SPINE_AUDIT.md)。", 
        '## 三层数据与来源保存',
        '全部 2800 条 V 的 `layers/*.pkl` 均有 MAIN_SPINE、SIDE_COMPONENT、REJECTED_REDUNDANT。当前全区 MAIN_SPINE 是 P0 输出，SIDE_COMPONENT 收纳物理闭合分量；其余未选原始片段完整放入 REJECTED_REDUNDANT，并带 NOT_SELECTED_REVIEW_REQUIRED 标签。**该层名是交付容器，不代表这些片段都已证明重复或低可信。** 近闭合旁支的自动归类仍只在 J 的参数化影子数据中执行。',
        '交付检查逐原始 edge_id 验证三层 t 区间无遗漏、无重叠，坐标仍等于原边插值，FaceID 与 source segment indices 不变；P2 的 36 组也执行同一检查。结果见 [DELIVERY_CHECKS.json](DELIVERY_CHECKS.json)。',
        '## 方案要求的八个问题',
        table(['问题','结论'],[
            ('1. 非甜区 fold 还会导致几十米缺失吗？','ASC 外 fold 无条件保护已解除，但静态 ASC 保留策略导致全区缺失退步，112.55 等例仍缺几十米。P0 整体未通过。'),
            ('2. 首选失败是否继续试其他候选？','是。98.05/98.10 已验证 B2 失败后尝试并接受 B6；身份排序未改成距离排序。'),
            ('3. LEGAL_CONTINUATION_EXISTS_BUT_ROUTE_STOPPED 是否归零？',str(alternative['legal_stopped_slices'])+'；已补查全部有限交点/端点投影。任意连续参数对的穷尽可行性尚未证明，硬目标按此有限范围解释。'),
            ('4. FaceTrack 后甜区是否帮助选择？',f"在 {p1['ASC_discriminative_decisions']} 次判定中缩小候选；合理性仍待人工审核。"),
            ('5. 哪些案例多 FaceTrack 同在 ASC？','见本报告 P1 表及图册完整列表。'),
            ('6. J_T1 / 108.80 能否单测线识别？','能，在 10/20 mm 扫描中检出；2/5 mm 未检出。'),
            ('7. Side provenance 是否完整？','逐原始边区间、几何、FaceID、来源 segment 交付检查；原绕行没有删除。'),
            ('8. 主脊是否连续且无 >10 mm 强连？','10 mm 示例连续；20 mm 另两组有 unresolved 断口，未强接。不能称所有参数下都连续。')]),
        '## 最终状态',table(['验收项','状态'],status.items()),
        'J_T1_10880 的 SUPPORTED 指本案例已证明存在满足约束的单测线剥离结果，不代表阈值已经定稿；SINGLE_PROFILE_MAIN_SPINE_PEELING 仍为部分支持，因为本轮只做局部参数扫描。当前不进入 FaceTrack 引导的全区主轨替换。',
        '## 检查、耗时与人工审核',
        f"最小相关检查：4 项新 P0 回归、48 项既有相关回归、6 项影子算法检查通过；另完成本方案明确要求的 2800 条普查、4 个实际硬回归、36 组参数扫描和来源保存核对。普查使用 {census['workers']} 个工作进程，墙钟 {census['seconds']/60:.2f} 分钟；这是此次固定分块上下文下的验证运行耗时，未混作 P1/P2 性能或完整生产入口基准。",
        '人工重点：C/D/E/F/J 的同来源五邻线选择、多个 FaceTrack 都处在 ASC 的位置、98.05/98.10 的身份歧义、两例动态尾区的核心身份，以及 J 的右侧/左侧旁支归属。本轮保持原图册形式与少量图例，由人工判断细节复杂度，不引入未经确认的 TC 公式。',
        '修改范围：生产文件仅 `scripts/04_structure_recognition/main_track_assembly.py`；新增验证/图册脚本与两个相关测试文件。未提交、推送、升级依赖或改动模型。复现实证与具体命令见 [WORK_LOG.md](WORK_LOG.md)。',
        '## 当前仍缺失 ASC 的测线',
        table(['s（m）','最大 ASC 高程缺失 m'],[(r['s'],f"{float(r['ASC_missing_max_m']):.6f}") for r in missing])]
    (out/'VALIDATION_REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    broken=[]
    for name in ['VALIDATION_REPORT.md','FACETRACK_SWEET_ZONE_REVIEW.md','J_T1_MAIN_SPINE_AUDIT.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(out/name).read_text(encoding='utf-8')):
            if not (out/target).exists():broken.append((name,target))
    assert not broken,broken
    print('REPORTS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['figures','jreport','finish','report']);a=p.parse_args();out=latest()
    if a.action=='figures':controls_figures(out)
    elif a.action=='jreport':j_report(out)
    elif a.action=='report':j_report(out);final_report(out)
    else:delivery_check(out);j_report(out);final_report(out)
