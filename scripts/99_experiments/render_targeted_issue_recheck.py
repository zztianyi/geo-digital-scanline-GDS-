"""Small, provenance-explicit galleries for the targeted historical recheck."""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np
import render_local_conflict_v2 as plot
from run_targeted_issue_recheck import OLD,RECENT,SOURCE,read,js,latest
from run_local_conflict_v2 import load_inventory,save_json
from review_p0_p1_p2 import table,plt,Line2D,crop
from analyze_targeted_issue_recheck import route,layers


def write(p,parts):p.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
def pathlink(p):return p.as_posix()


def gallery_case(out,name,title,ss,bounds,inventory,labels,note,after=None,before=OLD):
    after=after or out;members=[]
    for s in ss:
        for b in inventory[s]:
            for e in b['records']:
                if crop(e['points_uz'],bounds) is None:continue
                ts=sorted({labels[f] for f in e['source_face_ids'] if f in labels})
                tid=' / '.join(str(t) if isinstance(t,str) else 'T'+str(t) for t in ts) if ts else '身份未映射'
                members.append(dict(e,s=s,branch_id=b['branch_id'],face_track_ids=[tid]))
    tids=sorted({e['face_track_ids'][0] for e in members});figlinks=[];galleryrows=[]
    # Keep every visible source, splitting long galleries into readable pages.
    for index in range(0,len(tids),3):
        shown=tids[index:index+3];data=dict(region=dict(region_id=name,target_keys=ss),members=[e for e in members if e['face_track_ids'][0] in shown],tracks=[dict(face_track_id=t) for t in shown])
        tag=f'{name}_{index//3+1}';rows,_=plot.gallery(data,{s:inventory[s] for s in ss},out,bounds,tag);galleryrows.extend(rows)
        figlinks.append(f'![全部来源 第{index//3+1}页](../figures/{tag}_all_tracks.png)')
    fig,axs=plt.subplots(2,len(ss),figsize=(19,9),sharex=True,sharey=True,squeeze=False)
    for j,s in enumerate(ss):
        raw=[e for b in inventory[s] for e in b['records']]
        for i,root in enumerate([before,after]):
            ax=axs[i,j];plot.plot_records(ax,raw,bounds,'#c1c6cc',1)
            records=layers(root,s)['MAIN_SPINE'];plot.plot_records(ax,records,bounds,'#1467b1',2.2,False)
            joins=[e for e in records if not e['source'].startswith('OBSERVED')];plot.plot_records(ax,joins,bounds,'#c54138',2.2)
            for e in joins:
                p=np.mean(e['points_uz'],axis=0)
                if bounds[0]<=p[0]<=bounds[1] and bounds[2]<=p[1]<=bounds[3]:ax.scatter(*p,s=14,color='#c54138',zorder=6)
            plot.axes(ax,bounds);ax.set_title('s='+s+' m',fontsize=10)
            if j==0:ax.set_ylabel(('上一轮' if i==0 else ('当前结果（复用）' if after==RECENT else '本轮最终主轨'))+'\n高程 z（m）')
            if i==1:ax.set_xlabel('径向偏移 u（m）')
    fig.suptitle(title+'｜相同局部窗口的识别结果',fontsize=15)
    fig.legend(handles=[Line2D([],[],color='#c1c6cc',label='全部原始观测'),Line2D([],[],color='#1467b1',lw=2,label='实测主轨'),Line2D([],[],color='#c54138',ls='--',marker='o',label='算法连接')],loc='lower center',ncol=3)
    plot.savefig(fig,out,name+'_comparison.png')
    rows=[]
    for s in ss:
        r=route(after,s);ls=layers(after,s)
        rows.append((s,'→'.join('B'+str(x) for x in r['route_branch_sequence']),len(r['junctions']),dict(Counter(c['role'] for c in ls['components']))))
    write(out/'cases'/f'{name}.md',[f'# {title}',note,
        '上面的主轨对照才是最终识别结果（P1 及 P2 之后）。下面的来源图列出同一局部内所有可见 FaceTrack，分成每页最多三行；绿色不是已选主轨。B 编号只属于本测线；T 编号只属于本页引用的区域面片连通分组，不能跨不同区域直接比较。',
        f'![识别前后对照](../figures/{name}_comparison.png)',*figlinks,
        table(['s m','整条测线路径序列（并非仅图内）','接缝数','P2 层角色'],rows),
        f'局部范围：u={bounds[0]:.3f}–{bounds[1]:.3f} m，z={bounds[2]:.3f}–{bounds[3]:.3f} m。五列坐标完全一致；图框边缘不是分支真实端点。',
        f'结果来源：`{after}`；前轮：`{before}`。'])
    return dict(name=name,title=title,profiles=ss,bounds=bounds,FaceTracks=tids,rows=galleryrows,after=str(after),note=note)


def local_bounds(inventory,ss,zlo,zhi,pad=.3):
    points=[]
    for s in ss:
        for b in inventory[s]:
            for e in b['records']:
                p=crop(e['points_uz'],[-1e4,1e4,zlo,zhi])
                if p is not None:points.extend(p)
    us=np.asarray(points)[:,0];return [float(min(us)-pad),float(max(us)+pad),zlo,zhi]


def summary_report(out):
    data=js(out/'recheck_analysis.json');selection=js(out/'selection.json');frozen=js(out/'frozen_inventory.json');run=js(out/'run_summary.json')
    gallery=js(out/'gallery_inventory.json');counts=data['counts'];keys={p['s'] for p in data['profiles']}
    remaining=[p for p in data['profiles'] if p['after']['missing_extent_m']>1e-5]
    persistent=[e for e in data['old_event_comparison'] if not e['diagnostic_disappeared']]
    current=selection['reused_current_profiles'];historical=[p for p in frozen['profiles'] if p['s'] not in keys and p['s'] not in current]
    old_threshold_only=sorted({p['s'] for p in frozen['profile_flags'] if p['kind']=='OLD_HANDOFF_OVER_2MM' and p['s'] not in keys},key=float)
    band=data['bands'];applied=[r for r in band if r.get('applied')]
    audits=[js(out/'audits'/f'{g["group_id"]}.json') for g in run['groups']]
    checks=[r for a in audits for r in a['rows']]
    assert len(checks)==run['completed_count']==len(keys) and all(r['source_intervals_preserved'] for r in checks)
    assert all(r['max_coordinate_error_m']<2e-10 for r in checks)
    assert all(r['selected']['xyz_distance_m']<=.002+1e-10 and r['band_z_span_m']<=.1+1e-9 for r in applied)
    catalog=[]
    for p in frozen['profiles']:
        q=next((r for r in data['profiles'] if r['s']==p['s']),None)
        stage='THIS_RECHECK_BEFORE_PERFORMANCE_PREFILTER' if q else ('REUSED_SAME_RULES_BEFORE_PERFORMANCE_PREFILTER' if p['current_code'] else ('P1_ONLY_JOINT_INCOMPLETE' if p['s'] in run['p1_only'] else ('SELECTED_NOT_STARTED' if p['s'] in run['not_started'] else 'HISTORICAL_NOT_RERUN')))
        catalog.append(dict(s=p['s'],result_root=str(out if q else (RECENT if p['current_code'] else OLD)),execution=stage,
            provisional_P1=str(out/'preband'/f'{p["s"]}.pkl') if p['s'] in run['p1_only'] else None,
            manual_geometry_label='NOT_INDIVIDUALLY_LABELED',diagnostic=q['after'] if q else p))
    save_json(out/'consolidated_catalog.json',dict(profiles=catalog,manually_accepted_representative_cases=['E','F','89.95_NEIGHBORS'],
        note='Three accepted local representative cases do not label all 175 profiles in their parent result folders.'))
    missingrows=[]
    for p in remaining:
        distances=[c['nearest_from_path_end']['distance_m'] for d in p['missing_details'] for c in d['candidates']]
        missingrows.append((p['s'],f"{p['before']['missing_extent_m']:.6f} → {p['after']['missing_extent_m']:.6f}",
            f'{min(distances)*1000:.3f}' if distances else '无候选',','.join(p['after']['reasons'])))
    write(out/'MISSING_EXTENT_DETAILS.md',['# 候选包络缺口逐线复核',
        '本表比较所有合格候选的高程包络与主轨范围，不是“真实目标曲面缺失长度”。最近距离从当前路径对应端点到能扩展该方向的合格原始分支线段计算；它不是两整条分支任意内部点之间的全局最短距，不能排除更早换轨的可能。',
        table(['s m','候选包络未覆盖量 前→后 m','端点到可延伸候选最近距 mm','当前诊断'],missingrows),
        '逐候选的 B 编号、原始点坐标、边 ID、最近点参数和延伸拒绝原因见 [recheck_analysis.json](recheck_analysis.json)。',
        'E 外围已有当前版本结果的缺口（不重跑）：',
        table(['s m','候选包络未覆盖量 m','端点到可延伸候选最近距 mm'],[(p['s'],f"{p['info']['missing_extent_m']:.6f}",f"{min(c['nearest_from_path_end']['distance_m'] for d in p['missing_details'] for c in d['candidates'])*1000:.3f}") for p in data['reused_missing']])])
    rows=[]
    for g in selection['groups']:
        ps=[p for p in data['profiles'] if p['s'] in g['targets']];ev=[e for e in data['old_event_comparison'] if e['event_id'] in g['event_ids']]
        if not ps:
            rows.append((g['group_id'],g['targets'][0]+'–'+g['targets'][-1],'仅P1，联合未完成' if any(s in run['p1_only'] for s in g['targets']) else '未启动',len(g['seed_profiles']),'未复核','未复核'));continue
        rows.append((g['group_id'],g['targets'][0]+'–'+g['targets'][-1],len(ps),len(g['seed_profiles']),
            str(sum(p['before']['missing_extent_m']>1e-5 for p in ps))+'→'+str(sum(p['after']['missing_extent_m']>1e-5 for p in ps)),
            str(sum(e['diagnostic_disappeared'] for e in ev))+'/'+str(len(ev))))
    old_success=[('E / 104.30','同识别规则、性能改动前结果复用；用户已接受代表局部','E'),('F / 130.45','同识别规则、性能改动前结果复用；用户已接受代表局部','F'),('89.95 邻域','同识别规则、性能改动前结果复用；用户已接受；最大接缝2 mm，总高差73.071 mm','IDENTITY')]
    notes=(out/'FINDINGS.md').read_text(encoding='utf-8') if (out/'FINDINGS.md').exists() else '详细原因分类待填写。'
    write(out/'TARGETED_RECHECK_REPORT.md',['# 历史问题局部重算与成功案例汇总',
        f'**本轮未完成全部清单，已按性能限制停止。** 计划110条（65条问题线、45条必要邻线），实际完成 **{len(keys)} 条最终结果**（33条问题线、26条邻线）；另有 **18条仅完成P1、联合换轨未完成，33条尚未启动**。复用 E/F/89.95 修复规则已有的 **{len(current)} 条**结果。未运行全模型识别，也未重跑650条。',
        '已完成的59条和复用175条均产生于本轮性能预筛改动之前，保留各自原始代码指纹。性能预筛不改选轨与换轨规则，13项定向检查通过；但优化后的Q07真实大组仍未完成，不能宣称已验证其端到端提速。图中下排蓝线均来自有完整结果的生产流程 MAIN_SPINE（含P2），没有把18条未完成结果画成最终主轨。',
        f'原清单包含 {selection["event_count"]} 个历史持续异常窗口，已完成59条覆盖其中 **{counts["old_events"]} 个**：**{counts["old_events_disappeared"]} 个原诊断消失，{len(persistent)} 个仍有原诊断**；其余未完成。事件按相邻测线对及连续高程范围归并，相邻事件仍可能属于同一个物理问题，不是独立准确率样本。',
        '## 现在能否给出整体正确率',
        f'目前不能给出有真值支持的整体识别准确率。用户已确认 E、F、89.95 邻域这3个代表案例；本次新图册尚待人工判定，另有 {len(historical)} 条的最终结果仍沿用更早版本（其中18条有未完成的本轮P1）。选择的是历史疑点，且本轮清单未全部完成，不能外推到全模型。',
        table(['口径','数量 / 结果','能证明什么'],[
            ('人工已接受代表局部','3 个','仅证明这三个局部符合用户审核；不等于175条全部标为正确'),
            ('当前识别规则的完整结果来源',f'{len(keys)} 本轮 + 175 复用 = {len(keys)+175} 条','均保留性能优化前的指纹；不是正确样本数'),
            ('本轮原始坐标/FaceID/片段守恒/连续性',f'{len(keys)}/{len(keys)} 通过','未移动原始点，所有观测片段在主轨或旁支中有归属'),
            ('旧持续诊断消失',f'{counts["old_events_disappeared"]}/{counts["old_events"]}','诊断窗口变化；不是曲面身份准确率'),
            ('本轮候选包络未覆盖测线',f'{counts["missing_before"]} → {counts["missing_after"]}','详见缺口原因，不能全部计为漏识别'),
            ('重复分支序列',f'{counts["repeated_before"]} → {counts["repeated_after"]}','重复编号仅是绕行提示，需看实际几何'),
            ('相对旧结果高程范围缩小',str(counts['extent_regressions'])+' 条','不把旁支保留当作主轨覆盖')]),
        notes,
        '## 本轮代表图册',
        table(['案例','重点'],[(f'[{c["name"]} · {c["title"]}](cases/{c["name"]}.md)',c['note']) for c in gallery]),
        '## 怎样看图',
        '主轨对照：上排是上一轮，下排是本轮；每列是一条相邻纵测线，五列使用相同坐标范围。只保留三种要素：灰色原始曲线、蓝色实测主轨、红色算法接缝。沿列看前后差异，沿行看相邻测线形态是否连续变化。红点用于让毫米接缝在米级窗口中可见，不表示人工移动了节点。',
        '来源图：每行是一个 FaceTrack，每列是一条测线，列出这个局部窗口内全部可见来源，超过三行时分页。绿色表示分支的静态核心/竞核可用位置，橙色表示端部区，灰虚线表示未通过最小分支门槛。绿色不是已选主轨；没有画到某来源时明确标“本窗口没有该来源片段”。',
        'FaceTrack 表示本区域原网格面片的连通身份；它不等于“该曲线已被人工判真”。同一来源仍可能有折返，多个同高程交点也可能是真实悬空结构。T 编号不能跨不同区域直接比较，B 编号不能跨测线直接比较。图框截断不是分支物理端点。本轮没有引入新的 TC 分数代替人工看形态。',
        'C05、C06为看清走向而扩大到了原面分组区域之外，部分曲线显示在“身份未映射”汇总行。该行不是一个FaceTrack，可能包含多个原始分支，具体B编号标在各列标题中；不能根据这一行判断局部只有一个来源。C06主要验证断口距离，不对未映射片段作曲面身份结论。',
        '## 已成功案例汇总及版本边界',
        table(['既有案例','保留结论','原图册'],[(a,b,f'[{c}]({pathlink(RECENT/"cases"/(c+".md"))})') for a,b,c in old_success]),
        f'104.85 m 的 1.942 mm 原始小断口修补成功，当前175条结果中已包含该线，直接复用。98.05、98.10、116.25、122.95、112.55 m 的旧轮覆盖保持或路径简化作为**历史技术成功**保留，未写成当前版本已重新验收；见[历史 P0 案例]({pathlink(OLD/"cases/P0.md")})。本轮成功消除诊断的新案例以上面新图册为准，等待几何审核。',
        '## 清单与未重算部分',
        table(['组','s 范围 m','重算条数','问题线数','包络缺口 前→后','旧事件消失/事件数'],rows),
        f'对旧650份结果只做文件读取和诊断清点。没有仅因旧竞争接缝超过新2 mm门槛就把整片重跑；仍有 **{len(old_threshold_only)} 条**带此类记录的测线没有本轮完整最终结果（其中可能已在清单内但未完成）。没有将其认证为满足当前2 mm规则。混合版本边界与不超过100 mm的短时来源差异不单独触发重算。',
        f'仍沿用旧最终结果的 {len(historical)} 条中，{sum(p["status"]=="UNRESOLVED" for p in historical)} 条带未决提示，原因逐条保留，不归入成功；原清单之外的365条里有118条身份未决。复用的175条还有39个大于100 mm的持续事件（E：30个多来源、5个邻线来源差异；F：4个来源差异），见[完整列表](reused_current_events.json)。这些并非本轮新增的问题，图册C04–C06挑选了其中不同形态。',
        '[逐线整合台账](consolidated_catalog.json) · [明确重算清单](selection.json) · [旧窗口逐个前后结果](recheck_analysis.json) · [缺口逐线分析](MISSING_EXTENT_DETAILS.md)',
        '## 性能、修改与最低检查',
        f'两次识别运行实际累计 **{run["seconds"]:.3f} 秒（{run["seconds"]/60:.2f} 分钟）**，首次中断前 {run["interrupted_wall_seconds"]:.3f} 秒，优化后恢复 {run["resume_seconds"]:.3f} 秒。Q07的18条联合搜索仍未完成。按当前组成本对剩余51条作保守外推，总运行约 **{run["projected_seconds"]/60:.2f} 分钟**；这是假设外推，不是未启动组的实测耗时。依照超过一小时先优化、仍不可行则停止的要求，已终止识别进程。',
        '计时包含中断组耗时、原始上下文准备及缓存读取，不包含代码优化调试、历史文件清点、报告出图和人工审核；不能称为110条完成耗时，也不能作为端到端加速比。区域面标签可以读取完整原始上下文，但没有为这些上下文线执行整条识别。',
        table(['组','条数','本组耗时 s','共同换轨 s'],[(r['group_id'],r['profiles'],f"{r['seconds']:.3f}",f"{r['band_seconds']:.3f}") for r in run['groups']]),
        f'联合接缝实际应用 {len(applied)} 条记录，全部满足距离≤2 mm、所属共同带两端总高差≤100 mm；未应用原因计数：`{dict(Counter(r.get("reason") for r in band if not r.get("applied")))}`。接缝记录数不是独立测线数，也不是独立物理事件数。',
        '性能修改仅在 `terminal_junction_consensus.py` 增加共同高程窗口的几何必要条件预筛；通过条件、阈值和排名不变。已完成组和P1缓存保留，优化前后指纹与复用依据见 [performance_compatibility.json](performance_compatibility.json)。先出现预期性能检查失败（405次昂贵验证），优化后连同原端部换轨回归共13项通过，见 [perf_red.log](perf_red.log)、[perf_green.log](perf_green.log)。未跑全量测试。新增3个定向复核脚本、`test_terminal_band_prefilter.py`、计划与输出文件；运行内另检查几何/来源/连续性、联合带约束和报告引用。',
        '人工审核优先查看仍有接缝高度分散、长折返或旁支角色疑问的图册；边界散支、真实断口和缺乏真值的形态问题保留在台账，不通过放宽阈值或拉长合成线把它们伪装成解决。'])
    print('REPORT',out/'TARGETED_RECHECK_REPORT.md')


def render():
    out=latest();selection=js(out/'selection.json');cache=read(OLD/'stages/face_geometry_cache.pkl')
    for f in ['figures','cases']:(out/f).mkdir(exist_ok=True)
    groups={g['group_id']:g for g in selection['groups']};config=js(out/'case_config.json')
    previous={c['name']:c for c in js(out/'gallery_inventory.json')} if (out/'gallery_inventory.json').exists() else {}
    signatures={}
    for c in config:
        h=hashlib.sha256(json.dumps(c,sort_keys=True,ensure_ascii=False).encode())
        for root in [OLD,RECENT if c.get('reuse') else out]:
            for s in c['profiles']:h.update((root/'layers'/f'{s}.pkl').read_bytes())
        h.update(Path(__file__).read_bytes());h.update(Path(plot.__file__).read_bytes())
        signatures[c['name']]=h.hexdigest()
    pending=[c for c in config if previous.get(c['name'],{}).get('render_signature')!=signatures[c['name']]]
    ss=sorted({s for c in pending for s in c['profiles']},key=float);inventory=load_inventory(SOURCE,ss) if ss else {};gallery=[]
    for c in config:
        if c not in pending:gallery.append(previous[c['name']]);continue
        group=groups.get(c.get('group_id'));reg=group['region'] if group else next(r for r in js(OLD/'local_plan.json')['regions'] if c.get('region_tag','E') in r['case_tags'])
        labels=cache[tuple(reg['bounds'])] if reg and tuple(reg['bounds']) in cache else {}
        if c.get('source_tag'):
            data=read(SOURCE/'region_data'/f'{c["source_tag"]}.pkl')
            labels={f:t['face_track_id'] for t in data['tracks'] for f in t['face_ids']}
        bounds=c.get('bounds') or local_bounds(inventory,c['profiles'],*c['z_range'])
        item=gallery_case(out,c['name'],c['title'],c['profiles'],bounds,inventory,labels,c['note'],RECENT if c.get('reuse') else out)
        item['render_signature']=signatures[c['name']];gallery.append(item)
    save_json(out/'gallery_inventory.json',gallery)
    print('GALLERIES',len(gallery),out)


if __name__=='__main__':
    import sys
    if '--report' in sys.argv:summary_report(latest())
    else:render()
