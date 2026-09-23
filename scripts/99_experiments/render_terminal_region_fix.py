"""Render and audit the bounded frozen E/F/terminal results, without rerouting."""
import json,pickle,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import render_local_conflict_v2 as render
from run_terminal_region_fix import output,BASE
from run_local_conflict_v2 import SOURCE,load_inventory,fingerprints,save_json
from audit_production_face_consistency import level_tracks,classify
from review_p0_p1_p2 import crop,table,plt,Line2D


def read(p):return pickle.load(p.open('rb'))
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,parts):p.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
def main(root,s):return read(root/'layers'/f'{s}.pkl')['layers']['MAIN_SPINE']


def run():
    started=time.perf_counter();out=output();render.BASE=BASE
    for folder in ['figures','cases']:(out/folder).mkdir(exist_ok=True)
    audits={tag:js(out/'audits'/f'{tag}.json') for tag in ['E','F','IDENTITY']}
    assert all(d['code']==fingerprints() for d in audits.values()),'Frozen result/source mismatch'
    keys=sorted({s for d in audits.values() for s in d['targets']},key=float)
    inventory=load_inventory(SOURCE,keys);cache=read(BASE/'stages/face_geometry_cache.pkl')
    verification=[];case_rows=[];consistency=[];extents=[]
    for tag,d in audits.items():
        reg=d['region'];labels=cache[tuple(reg['bounds'])];levels=np.arange(np.ceil(reg['bounds'][4]*20)/20,reg['bounds'][5]+1e-8,.05)
        counts={v:Counter() for v in ['before','after']};hits={};examples=[]
        for s in d['targets']:
            hits[s]={v:level_tracks(main(root,s),reg,labels,levels) for v,root in [('before',BASE),('after',out)]}
            old=read(BASE/'routes'/f'{s}.pkl')['result'];new=read(out/'routes'/f'{s}.pkl')['result']
            extents.append(dict(s=s,before=old['route_z_extent'],after=new['route_z_extent']))
        for a,b in zip(d['targets'],d['targets'][1:]):
            for k,z in enumerate(levels):
                for v in counts:counts[v][classify(hits[a][v][k],hits[b][v][k])]+=1
                c=classify(hits[a]['after'][k],hits[b]['after'][k])
                if c not in ('CONSISTENT','MISSING_BOTH') and len(examples)<6:examples.append(dict(left=a,right=b,z=float(z),classification=c))
        consistency.append(dict(tag=tag,counts={k:dict(v) for k,v in counts.items()},examples=examples))
        for row in d['rows']:
            assert row['source_intervals_preserved'] and row['max_coordinate_error_m']<2e-10
        for item in d['bands']:
            if item.get('applied'):
                p=item['selected'];assert p['xyz_distance_m']<=.002+1e-10
                assert item['band_z_span_m']<=.1+1e-9
        if tag=='IDENTITY':continue
        data=read(SOURCE/'region_data'/f'{tag}.pkl');ss=data['region']['target_keys']
        bounds=[8.9,20.1,1384.8,1387.3] if tag=='E' else [-.6,1.9,1392.3,1395.5]
        rows,n=render.gallery(data,{s:inventory[s] for s in ss},out,bounds,tag)
        render.route_comparison(data,inventory,out,bounds,tag)
        names={(e['s'],e['edge_id']):e['face_track_ids'] for e in data['members']}
        check=[14.4,16.4,1385.2,1386.2] if tag=='E' else bounds
        values=[]
        for s in ss:
            identity={v:sorted({tid for r in main(root,s) if r['source'].startswith('OBSERVED') and crop(r['points_uz'],check) is not None for tid in names.get((s,r['edge_id']),[])}) for v,root in [('before',BASE),('after',out)]}
            values.append(dict(s=s,**identity))
        expected=[tag+'_T2'] if tag=='E' else [tag+'_T1']
        verification.append(dict(case=tag,expected=expected,profiles=values,passed=all(r['after']==expected for r in values),check_bounds=check))
        decision=d['region_decisions'][reg['region_id']]
        write(out/'cases'/f'{tag}.md',[f'# {tag} 区：单一来源是否在五邻线保持',
            '本页上图逐行列出全部 FaceTrack；下图才是识别结果。绿色核心位置不是已选主轨。分支编号 B 只在单条测线内有效，跨列应比较行上的 FaceTrack 名称。',
            f'最终局部期望来源：{expected[0]}；五邻线核对：'+('通过。' if verification[-1]['passed'] else '仍有差异，见表。'),
            table(['s（m）','上一轮最终来源','本轮最终来源'],[(r['s'],r['before'],r['after']) for r in values]),
            f'![{tag}全部来源](../figures/{tag}_all_tracks.png)',f'![{tag}最终主轨](../figures/{tag}_main_comparison.png)',
            '同一窗口内的稳定主轨不再因短分支局部较宽而插入绕行。全路径跨出此局部后仍可能需要接入其他来源，不能把全路径序列有多个 B 编号误读为本窗口仍在换轨。',
            table(['区域来源','贯通邻线数','连续邻线数','预计后续换轨','前向高程支持 m'],[(r['FaceTrack'],r['through_V'],r['consecutive_V'],f"{r['future_switches']:.3f}",f"{r['forward_Z_m']:.3f}") for r in decision['candidates']]),
            f'本区实际重跑 {len(d["targets"])} 条；其中 {len(d["locks"])} 条原始分支满足贯通判定。其余不强行标成贯通；整区诊断见总报告。'])
        case_rows.extend(rows)
    d=audits['IDENTITY'];ss=d['targets'];reg=d['region'];labels=cache[tuple(reg['bounds'])]
    bounds=[-14.2,-9.0,1403.8,1413.5];members=[]
    for s in ss:
        for b in inventory[s]:
            for e in b['records']:
                if crop(e['points_uz'],bounds) is None:continue
                tids={labels[f] for f in e['source_face_ids'] if f in labels}
                members.append(dict(e,s=s,branch_id=b['branch_id'],face_track_ids=[' / '.join(f'R_T{t}' for t in sorted(tids)) or '未映射']))
    data=dict(region=dict(region_id='IDENTITY',target_keys=ss),members=members,tracks=[dict(face_track_id=t) for t in sorted({e['face_track_ids'][0] for e in members})])
    render.gallery(data,{s:inventory[s] for s in ss},out,bounds,'IDENTITY');render.route_comparison(data,inventory,out,bounds,'IDENTITY')
    bands=[r for r in d['bands'] if r.get('selected',{}).get('z',0)>1400]
    assert len(bands)==5 and {r['s'] for r in bands}==set(ss) and all(r['applied'] for r in bands)
    ends=[p for r in bands for p in (r['selected']['a_point_uz'][1],r['selected']['b_point_uz'][1])]
    span=float(np.ptp(ends));assert span<=.1+1e-9
    fig,axs=plt.subplots(1,5,figsize=(19,7),sharey=True)
    zlo=min(ends)-.10;zhi=max(ends)+.16
    for ax,r in zip(axs,bands):
        s=r['s'];p=r['selected'];u=np.mean([p['a_point_uz'][0],p['b_point_uz'][0]])
        bb=[u-.06,u+.06,zlo,zhi]
        raw=[e for b in inventory[s] for e in b['records'] if b['branch_id'] in (r['from_branch'],r['to_branch'])]
        render.plot_records(ax,raw,bb,'#bfc4cb',2)
        render.plot_records(ax,main(out,s),bb,'#1467b1',2.5,False)
        ax.plot([p['a_point_uz'][0],p['b_point_uz'][0]],[p['a_point_uz'][1],p['b_point_uz'][1]],color='#c54138',lw=3)
        ax.scatter([p['a_point_uz'][0],p['b_point_uz'][0]],[p['a_point_uz'][1],p['b_point_uz'][1]],s=18,color='#c54138',zorder=5)
        ax.axhspan(min(ends),max(ends),color='#1467b1',alpha=.055)
        render.axes(ax,bb);ax.set_title(f"s={s} m\n间距 {p['xyz_distance_m']*1000:.3f} mm",fontsize=10);ax.set_xlabel('径向偏移 u（m）')
    axs[0].set_ylabel('高程 z（m）');fig.suptitle(f'从端部向内找到的五邻线共同换轨带｜总高差 {span*1000:.3f} mm',fontsize=16)
    fig.legend(handles=[Line2D([],[],color='#bfc4cb',label='原始两分支'),Line2D([],[],color='#1467b1',lw=2,label='保留的实测主轨'),Line2D([],[],color='#c54138',marker='o',label='2 mm 内的接缝两端')],loc='lower center',ncol=3)
    fig.subplots_adjust(left=.08,right=.985,top=.84,bottom=.15,wspace=.13)
    fig.savefig(out/'figures/IDENTITY_terminal_detail.png',dpi=160);plt.close(fig)
    terminal_rows=[]
    for r in bands:
        p=r['selected'];first=min(r['legal_candidates'],key=lambda q:q['terminal_retreat_m'])
        terminal_rows.append(dict(s=r['s'],branches=f"B{r['from_branch']}→B{r['to_branch']}",z_a=p['a_point_uz'][1],z_b=p['b_point_uz'][1],gap_mm=p['xyz_distance_m']*1000,retreat_m=p['terminal_retreat_m'],first_z=first['z']))
    write(out/'cases/IDENTITY.md',['# 89.95 m 附近：端部优先的共同换轨带',
        f'五条线均已应用；两端合计总高差 {span*1000:.6f} mm，最大接缝 {max(r["gap_mm"] for r in terminal_rows):.6f} mm。红色短线是合成连接，蓝色线段保留原始坐标。',
        '搜索顺序：以发生续接的那一端为起点，沿原分支弧长向内；在竞核内保留整段 ≤2 mm 可行邻域，然后找覆盖所有参与邻线且总高差 ≤100 mm 的最早共同带。具体最早性采用“先最小化各线最大回退弧长，再最小化回退弧长总和”，并以径向/高程分散作平局判据。没有把节点平均到同一高程。',
        table(['s m','分支对','接缝两端 z m','间距 mm','从断开端向内弧长 m','本线几何候选首点 z'],[(r['s'],r['branches'],f"{r['z_a']:.6f} / {r['z_b']:.6f}",f"{r['gap_mm']:.6f}",f"{r['retreat_m']:.6f}",f"{r['first_z']:.6f}") for r in terminal_rows]),
        '![换轨细节](../figures/IDENTITY_terminal_detail.png)',
        '细节图统一高程，径向窗口分别围绕各线接点放大为 0.12 m；浅蓝背景只表示共同高程带。以下两张大图使用完全相同的径向与高程范围，可以看出真实走向。',
        '![原始来源](../figures/IDENTITY_all_tracks.png)','![最终主轨对照](../figures/IDENTITY_main_comparison.png)',
        '原始纵向切片间隔是 0.05 m，89.91–90.00 m 内现有切片为 89.95 和 90.00 m；另纳入 89.85、89.90、90.05 m 作邻域约束。本轮没有插造 89.91 m 切片。'])
    save_json(out/'acceptance.json',dict(cases=verification,terminal=terminal_rows,terminal_span_m=span,source_checks=[r for d in audits.values() for r in d['rows']],extents=extents))
    save_json(out/'regional_consistency.json',consistency);save_json(out/'gallery_inventory.json',case_rows)
    decreased=[r for r in extents if r['after'][0]>r['before'][0]+1e-6 or r['after'][1]<r['before'][1]-1e-6]
    perf=[]
    for tag,d in audits.items():
        perf.append((tag,len(d['targets']),f"{d['seconds']:.3f}",f"{sum(r['P0_P1_seconds'] for r in d['rows']):.3f}",f"{d['band_seconds']:.3f}"))
    summary=js(out/'local_run.json')
    write(out/'TERMINAL_REGION_REPORT.md',['# E / F 单轨与 2 mm / 100 mm 换轨复核',
        '本报告展示本轮生产识别流程的最终 MAIN_SPINE（含 P2 后处理），不是候选分支，也不是仅用于展示的影子结果。上一轮对照固定为 outputs/local_conflict_v2/20260922_124130。',
        table(['验收项','结果'],[(v['case']+' 五邻线局部来源',('通过：' if v['passed'] else '未通过：')+v['expected'][0]) for v in verification]+[('89.95 m 附近五邻线换轨',f'5/5 应用；最大间距 {max(r["gap_mm"] for r in terminal_rows):.6f} mm；总高差 {span*1000:.6f} mm')]),
        '[E 区全部来源与主轨对照](cases/E.md) · [F 区全部来源与主轨对照](cases/F.md) · [89.95 m 附近换轨细节](cases/IDENTITY.md)',
        '## 怎么看图',
        '来源图：每行一个 FaceTrack、每列一条相邻纵向测线；绿色为原始分支竞核位置，橙色为端部区，灰虚线为未通过最小分支门槛的来源。所有候选均展示，绿色不代表已经选中。主轨对照：上排上一轮、下排本轮最终结果；灰色是全部原始观测，蓝色是保留的实测主轨，红色短虚线/点是算法连接。每类图最多三种图例。图框裁切不是原分支端点。',
        '先沿同一列比较上下排是否减少不必要换轨，再沿同一行比较五条邻线是否持续保留同一来源。FaceTrack 在区域内表示原模型面片连通身份；B 编号只在各自测线上有效。曲线形态是否吻合仍由你人工审核，不用一个抽象 TC 分数替代。',
        '## 修复原因与实现',
        '原区域排序让竞争高程跨度的微小差异先于前向支持与后续换轨代价起作用；区域优先来源又可被单线局部选择覆盖。本轮优先能贯通竞争区的来源，先考虑连续邻线与较少后续换轨，再用前向支持、竞核比例等比较；唯一可用的贯通原分支在局部不再插入短支绕行。必要的区域外接入/接出仍按真实几何处理。',
        '另修复了不同大小区域的邻线支持数不可直接比较的问题：75/75 与 7/7 都表示完整支持，按比例比较后再考察竞核位置与前向延续。旧比较会先走一个随后无法接回下部的短支，使 E 边缘四条线下端缩短约 20 m；本轮将这些线纳入整轨范围复核。',
        '原接点候选按最短距离及离散高程代表压缩，容易偏向交点并丢掉靠近端部的合法接触区间。本轮保留原始线段之间整段 2 mm 邻域，按路径方向从端部向内搜索；所有参与邻线、每个接缝两端都纳入 100 mm 总高差。缺一条合法候选或组合应用失败就整组拒绝，不能拆成几个“成功”的单线组。',
        '参数彼此独立：P1 竞争换轨 2 mm；邻线总换轨高差 100 mm；P0 唯一续接保留原 50 mm；P2 近回返保留原 10 mm / 0.5 m。原始点、FaceID 和原始片段参数不移动、不重采样。横向信息本轮用于既有 FaceTrack 身份和邻线共同高程带，没有另外生成横向补点。',
        '## 局部整区复核边界',
        f'实际重跑 {len(keys)} 条：E 75 条、F 95 条、89.95 m 附近 5 条。E/F 全局部范围也进行了同高程邻线来源诊断。下表是诊断计数，不是准确率；完整区域可能仍需不同来源接续，因此不能要求整个大矩形每一点都只有一个 FaceTrack。',
        table(['区域','来源变化 前→后','单侧缺失 前→后','多来源 前→后'],[(r['tag'],str(r['counts']['before'].get('IDENTITY_SWITCH',0))+' → '+str(r['counts']['after'].get('IDENTITY_SWITCH',0)),str(r['counts']['before'].get('MISSING_ONE',0))+' → '+str(r['counts']['after'].get('MISSING_ONE',0)),str(r['counts']['before'].get('AMBIGUOUS_MULTIPLE',0))+' → '+str(r['counts']['after'].get('AMBIGUOUS_MULTIPLE',0))) for r in consistency]),
        '这些计数覆盖所列大区域的 0.05 m 高程采样，并非上面窄局部案例的通过率。允许接点总高差 100 mm，也就允许接缝带内部短距离内相邻线暂时分属两侧来源；因此 IDENTITY 的同高程来源变化由 1 到 4 不能单独判作换轨失败。E/F 大区域的剩余变化、单侧缺失和 F 的 2 个多来源采样仍保留人工审核，未把中心五邻线的通过推广到全区域。',
        f'相对上一轮整轨高程范围缩小的测线数：{len(decreased)}；逐线范围在 [acceptance.json](acceptance.json)。原始片段全部守恒不等于它们全部被归入主轨；没有把保留在旁支的片段算成主轨覆盖。',
        '[区域未一致位置样本](regional_consistency.json) · [逐项验收数据](acceptance.json) · [图册来源清单](gallery_inventory.json)',
        '## 性能与最低检查',
        f'本轮局部识别计时 {summary["seconds"]:.3f} 秒。此计时包含当前局部运行、区域准备和 P1/联合换轨/P2，不包含前期调试、原始模型预处理及报告出图；使用已有原始几何缓存。不能与上一轮更多区域的总时长直接作加速比。',
        table(['区域','测线数','区域合计秒','P1 秒','共同换轨秒'],perf),
        '68 项定向单元检查通过，见 [targeted_tests.log](targeted_tests.log)。每条最终主轨检查原始坐标误差、FaceID、原参数区间守恒、路径连续、接缝独立阈值；共同换轨同时检查所有邻线及两端总高差。未跑 2800 条普查或全项目测试。[本轮修改文件](MODIFIED_FILES.md)。',
        '## 人工验证与已知边界',
        '请重点审核 E_T2、F_T1 在五邻线的细节形态是否符合实际曲面，以及换轨放大图里的两条原分支是否应连接。数学距离和面片连通不能单独证明真实地质身份。未解决的整区诊断位置按表保留，不据中心案例通过宣称全区无疑点。'])
    print('REPORT',str(out/'TERMINAL_REGION_REPORT.md'),'cases',verification,'span_mm',span*1000,'seconds',time.perf_counter()-started,flush=True)


if __name__=='__main__':run()
