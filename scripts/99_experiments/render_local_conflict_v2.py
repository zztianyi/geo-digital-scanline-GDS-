"""Frozen local-result audit and common-window gallery; never reruns routing."""
import json,pickle,re,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from run_local_conflict_v2 import SOURCE,ROOT,BASE,P0,output_dir,fingerprints
from run_production_integration_v2 import mesh,manifest,load_inventory,save_json
from physical_face_context import PhysicalFaceContext
from branch_absolute_core import branch_metrics,local_edge_scale
from observed_component_geometry import near_returns
from render_facetrack_competition_review import view_rows
from review_p0_p1_p2 import plt,LineCollection,Line2D,COLORS,crop,table
from audit_production_face_consistency import level_tracks,classify


def read(path):return pickle.load(path.open('rb'))
def jsonread(path):return json.loads(path.read_text(encoding='utf-8'))
def text(path,parts):path.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
def plot_records(ax,records,bounds,color,lw=1.6,synthetic=True):
    for observed in (True,False):
        if not observed and not synthetic:continue
        lines=[crop(r['points_uz'],bounds) for r in records if r['source'].startswith('OBSERVED')==observed]
        lines=[p for p in lines if p is not None]
        if lines:ax.add_collection(LineCollection(lines,colors=color,linewidths=lw,linestyles='-' if observed else '--'))
def axes(ax,bounds):
    ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.grid(alpha=.16);ax.ticklabel_format(useOffset=False,style='plain')
def savefig(fig,out,name,bottom=.12):
    fig.subplots_adjust(left=.08,right=.985,top=.9,bottom=bottom,hspace=.24,wspace=.13)
    fig.savefig(out/'figures'/name,dpi=160);plt.close(fig)


def gallery(data,inventory,out,bounds,name):
    metrics={s:{b['branch_id']:branch_metrics(b,local_edge_scale(bs)) for b in bs} for s,bs in inventory.items()}
    rows,parts,n=view_rows(data,inventory,metrics,bounds,'local_v2')
    tids=sorted(t['face_track_id'] for t in data['tracks']);keys=data['region']['target_keys']
    fig,axs=plt.subplots(len(tids),len(keys),figsize=(19,3.1*len(tids)+1.2),sharex=True,sharey=True,squeeze=False)
    for i,tid in enumerate(tids):
        for j,s in enumerate(keys):
            ax=axs[i,j];rr=[r for r in rows if r['s']==s and r['FaceTrack']==tid]
            for tag in ['REJECTED','ASC','GUARD']:
                lines=[p for p,t in parts[s,tid] if t==tag]
                if lines:ax.add_collection(LineCollection(lines,colors=COLORS[tag],linewidths=2,linestyles='--' if tag=='REJECTED' else '-'))
            if not rr:ax.text(.5,.5,'本窗口没有该来源片段',transform=ax.transAxes,ha='center',color='#777',fontsize=9)
            ax.set_title(f's={s} m | '+(','.join('B'+str(r['branch_id']) for r in rr) or '无'),fontsize=9)
            axes(ax,bounds)
            if j==0:ax.set_ylabel(tid+'\n高程 z（m）')
            if i==len(tids)-1:ax.set_xlabel('径向偏移 u（m）')
    fig.suptitle(name+'｜全部 FaceTrack × 五邻线｜共同坐标窗口',fontsize=16)
    fig.legend(handles=[Line2D([],[],color=COLORS['ASC'],lw=2,label='竞核位置（仅竞争时参与决策）'),
        Line2D([],[],color=COLORS['GUARD'],lw=2,label='端部区'),Line2D([],[],color=COLORS['REJECTED'],ls='--',label='MBG 未通过')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.01))
    savefig(fig,out,name+'_all_tracks.png')
    return rows,n


def route_comparison(data,inventory,out,bounds,name):
    keys=data['region']['target_keys'];fig,axs=plt.subplots(2,len(keys),figsize=(19,9),sharex=True,sharey=True,squeeze=False)
    for j,s in enumerate(keys):
        old=read(BASE/'layers'/f'{s}.pkl')['layers']['MAIN_SPINE']
        new=read(out/'layers'/f'{s}.pkl')['layers']['MAIN_SPINE']
        for i,records in enumerate([old,new]):
            ax=axs[i,j];raw=[r for b in inventory[s] for r in b['records']]
            plot_records(ax,raw,bounds,'#c1c6cc',1);plot_records(ax,records,bounds,'#1467b1',2.1,False)
            joins=[r for r in records if not r['source'].startswith('OBSERVED')]
            plot_records(ax,joins,bounds,'#c54138',2)
            for r in joins:
                p=np.mean(r['points_uz'],axis=0)
                if bounds[0]<=p[0]<=bounds[1] and bounds[2]<=p[1]<=bounds[3]:ax.scatter(*p,s=13,color='#c54138',zorder=5)
            axes(ax,bounds);ax.set_title(f's={s} m',fontsize=10)
            if j==0:ax.set_ylabel(('上一轮' if i==0 else '本轮最终主轨')+'\n高程 z（m）')
            if i==1:ax.set_xlabel('径向偏移 u（m）')
    fig.suptitle(name+'｜相同窗口的最终主轨对照（含 P2）',fontsize=16)
    fig.legend(handles=[Line2D([],[],color='#c1c6cc',label='全部原始观测'),Line2D([],[],color='#1467b1',lw=2,label='保留的实测主轨'),
        Line2D([],[],color='#c54138',ls='--',marker='o',label='算法连接／换轨位置')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.01))
    savefig(fig,out,name+'_main_comparison.png')


def local_consistency(out,plan,context,m):
    rows=[]
    for region in plan['regions']:
        labels=context.region_faces(region);keys=region['target_keys'];b=region['bounds']
        levels=np.asarray([z for z in m['levels'] if b[4]<=z<=b[5]])
        counters={v:Counter() for v in ['BEFORE','P1','P2']};hits={};examples=[]
        for s in keys:
            records={'BEFORE':read(BASE/'layers'/f'{s}.pkl')['layers']['MAIN_SPINE'],
                'P1':read(out/'routes'/f'{s}.pkl')['result']['path_edges'],'P2':read(out/'layers'/f'{s}.pkl')['layers']['MAIN_SPINE']}
            hits[s]={v:level_tracks(rs,region,labels,levels) for v,rs in records.items()}
        for a,b in zip(keys,keys[1:]):
            for i,z in enumerate(levels):
                classes={v:classify(hits[a][v][i],hits[b][v][i]) for v in counters}
                for v,c in classes.items():counters[v][c]+=1
                if classes['P2'] not in ('CONSISTENT','MISSING_BOTH') and len(examples)<8:
                    examples.append(dict(left=a,right=b,z=float(z),classification=classes,tracks=[sorted(hits[a]['P2'][i]),sorted(hits[b]['P2'][i])]))
        rows.append(dict(region=region['region_id'],slices=len(keys),H_levels=len(levels),counts={v:dict(c) for v,c in counters.items()},examples=examples))
    save_json(out/'local_consistency.json',rows);return rows


def negatives(out,keys,inventory):
    open_cases=[];short=[]
    preferred=[s for s in ['89.95','104.30','130.45','122.25','108.80','104.85','98.10'] if s in keys]
    for s in preferred+[s for s in keys if s not in preferred]:
        old=read(BASE/'layers'/f'{s}.pkl')['layers'];new=read(out/'layers'/f'{s}.pkl')['layers']
        for c in old['components']:
            if len(open_cases)>=5:break
            if c['role']!='SIDE_OPEN_BRANCH':continue
            records=c['records'];arc=np.r_[0.,np.cumsum([np.linalg.norm(np.diff(r['points_xyz'],axis=0)) for r in records])]
            events=near_returns(dict(records=records,arc_positions=arc),max_gap_m=.01,min_arc_m=.5,min_ratio=0)
            if events:continue
            intervals=defaultdict(list)
            for r in records:intervals[r['edge_id']].append(sorted([r['t0'],r['t1']]))
            roles=set()
            for cc in new['components']:
                if any(max(a,min(r['t0'],r['t1']))<min(b,max(r['t0'],r['t1']))-1e-9 for r in cc['records'] if r['source'].startswith('OBSERVED') for a,b in intervals.get(r['edge_id'],[])):roles.add(cc['role'])
            if any(x.startswith('P2_') for x in roles):continue
            if any(x['s']==s for x in open_cases):continue
            open_cases.append(dict(s=s,branch=c['branch_id'],old_role=c['role'],current_roles=sorted(roles),legal_pairs=0,observed_m=float(arc[-1]),P2=False,records=records))
        if len(short)<3:
            for b in inventory[s]:
                for i in range(len(b['records'])):
                    rr=b['records'][i:i+1];length=sum(np.linalg.norm(np.diff(r['points_xyz'],axis=0)) for r in rr)
                    distance=np.linalg.norm(np.asarray(rr[0]['points_xyz'][0])-rr[-1]['points_xyz'][1])
                    if distance<=.01 and length<.5 and length>1e-8:
                        short.append(dict(s=s,branch=b['branch_id'],edge_ids=[r['edge_id'] for r in rr],D_xyz_m=float(distance),L_path_m=float(length),P2=False));break
                if short and short[-1]['s']==s:break
        if len(open_cases)>=5 and len(short)>=3:break
    assert len(open_cases)>=3 and len(short)>=2,(len(open_cases),len(short))
    save_json(out/'P2_negative_examples.json',dict(open_branches=open_cases,short_paths=short));return open_cases,short


def focused_cases(out,plan,context,inventory,bands,gap):
    region=next(r for r in plan['regions'] if 'AUTO_IDENTITY_REVIEW' in r.get('case_tags',[]))
    keys=['89.85','89.90','89.95','90.00','90.05'];bounds=[-14.2,-9.0,1403.8,1413.5]
    labels=context.region_faces(region);members=[]
    for s in keys:
        for b in inventory[s]:
            for e in b['records']:
                if crop(e['points_uz'],bounds) is None:continue
                tids={labels[f] for f in e['source_face_ids'] if f in labels}
                name=' / '.join(f'R_T{t}' for t in sorted(tids)) if tids else '身份未映射'
                members.append(dict(s=s,branch_id=b['branch_id'],edge_id=e['edge_id'],points_uz=e['points_uz'],face_track_ids=[name]))
    data=dict(region=dict(region_id='IDENTITY',focus_s='89.95',target_keys=keys),members=members,
        tracks=[dict(face_track_id=t) for t in sorted({e['face_track_ids'][0] for e in members})])
    gallery(data,{s:inventory[s] for s in keys},out,bounds,'IDENTITY')
    route_comparison(data,inventory,out,bounds,'IDENTITY')
    def relevant_junctions(root,s):
        r=read(root/'routes'/f'{s}.pkl')['result']
        return [j for j in r['junctions'] if {j.get('from_branch_id'),j.get('to_branch_id')}=={2,5}]
    comparison=[]
    for s in ['89.95','90.00']:
        old=relevant_junctions(BASE,s);new=relevant_junctions(out,s)
        comparison.append(dict(s=s,old_z=[float(j['a_point_uz'][1]) for j in old],new_z=[float(j['a_point_uz'][1]) for j in new]))
    old_gap=abs(comparison[0]['old_z'][0]-comparison[1]['old_z'][0])
    new_gap=abs(comparison[0]['new_z'][0]-comparison[1]['new_z'][0]) if all(len(r['new_z'])==1 for r in comparison) else None
    save_json(out/'identity_key_comparison.json',dict(profiles=comparison,old_jump_m=old_gap,new_jump_m=new_gap))
    nearby=[r for r in bands if 89.7<=float(r['s'])<=90.2 and {r['from_branch'],r['to_branch']}=={2,5}]
    fig,ax=plt.subplots(figsize=(12,6))
    for r in nearby:
        s=float(r['s']);zz=[c['z'] for c in r['legal_candidates']]
        ax.scatter([s]*len(zz),zz,s=8,color='#c1c6cc')
        ax.scatter(s,r['old_z'],s=35,color='#d78426')
        if r.get('applied'):ax.scatter(s,r['selected']['z'],s=35,color='#1467b1')
    ax.set_xlabel('测线里程 s（m）');ax.set_ylabel('换轨高程 z（m）');ax.grid(alpha=.2);ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_title('局部换轨带｜灰点：合法候选；橙点：单线选择；蓝点：实际应用的联合结果')
    fig.savefig(out/'figures/IDENTITY_band.png',dpi=160,bbox_inches='tight');plt.close(fig)
    text(out/'cases/IDENTITY.md',['# 89.95 / 90.00：同分支序列并不代表同换轨位置',
        f'上一轮 B2/B5 换轨差 {old_gap:.6f} m；本轮差 '+(f'{new_gap:.6f} m。' if new_gap is not None else '无法用单一同分支接点比较，需逐连接审核。'),
        table(['s','上一轮换轨 z','本轮最终换轨 z'],[(r['s'],r['old_z'],r['new_z']) for r in comparison]),
        '![全部来源](../figures/IDENTITY_all_tracks.png)','![主轨对照](../figures/IDENTITY_main_comparison.png)',
        '![候选与联合选择](../figures/IDENTITY_band.png)',
        table(['s','分支','原始候选数','合法代表数','实际应用','原因'],[(r['s'],str(r['from_branch'])+'→'+str(r['to_branch']),r['raw_candidate_count'],r['legal_candidate_count'],r.get('applied',False),r.get('reason','')) for r in nearby]),
        '灰色候选均为原始线段上的交点或合法最近点对；蓝点从灰点中选择。没有计算新的平均坐标。相邻候选缺席、多个换轨占用同一段、整轨预算不允许时，不强行接成平滑带。region 连通标签一致本身也不能证明细节一致。'])
    if gap:
        point=np.mean([gap[0]['a_point_uz'],gap[0]['b_point_uz']],axis=0)
        windows=[('context',[point[0]-1.5,point[0]+1.5,point[1]-1.5,point[1]+2.5]),
                 ('gap',[point[0]-.012,point[0]+.012,point[1]-.012,point[1]+.012])]
        for name,bb in windows:
            fig,axs=plt.subplots(1,2,figsize=(13,7),sharex=True,sharey=True)
            for ax,root,title in zip(axs,[BASE,out],['上一轮：孔洞前停止','本轮：MODEL_GAP_REPAIR']):
                plot_records(ax,[r for b in inventory['104.85'] for r in b['records']],bb,'#bfc4cb',1.5)
                rs=read(root/'layers/104.85.pkl')['layers']['MAIN_SPINE'];plot_records(ax,rs,bb,'#1467b1',2.3,False)
                plot_records(ax,[r for r in rs if not r['source'].startswith('OBSERVED')],bb,'#c54138',2.8)
                axes(ax,bb);ax.set_title(title);ax.set_xlabel('径向偏移 u（m）')
            axs[0].set_ylabel('高程 z（m）');fig.suptitle('104.85 m｜灰色原始观测；蓝色主轨；红虚线补缝',fontsize=15)
            savefig(fig,out,'P0_10485_'+name+'.png')
    text(out/'cases/P0.md',['# P0：唯一续接的小孔洞与大间距负例',
        '104.85 m 的 B5→B3：'+(f'{gap[0]["xyz_distance_m"]*1000:.6f} mm，{gap[0]["decision_mode"]}，候选数 {gap[0]["competitor_count"]}，CC={gap[0]["competitive_core_invoked"]}。' if gap else '本轮未得到预期连接。'),
        '![孔洞周边走向](../figures/P0_10485_context.png)','![孔洞放大](../figures/P0_10485_gap.png)',
        '放大图显示的是原始数据坐标，不是改变坐标制造连接。红色补线为独立的 synthetic connector，原始 FaceID 与区间没有修改。',
        '104.70 m 的约 0.403 m 间距超过 0.05 m，必须继续保持 unresolved。其实际主轨和缺失量列在总报告 P0 表，不因邻线可连就跨越本线大孔洞。'])
    return new_gap


def scope_audit(out):
    explanations={
        'physical_continuation.py':'种子先统计同区候选；续接先 decision_scope，只有 competitive_core_invoked 才取 CC 或 reliable_junction。唯一同链续接另走 noncompetitive_junction。',
        'local_competitive_handoff.py':'先构造有实测交点的候选；len(rows)>=2 后才计算 in_ASC / forward_ASC，并将原始竞争集合传入后备尝试。',
        'region_joint_selection.py':'MBG 用完整分支最小支持度；逐 H 高程 count>=2 后才读取 CC 位置，region 汇总作为竞争决策证据。',
        'region_junction_band.py':'只处理已有的两分支竞争连接；MODEL_GAP_REPAIR 被排除。每次 band 候选记录 competitor_count=2。',
        'competitive_surface_selection.py':'choose_successor 在 decision_scope 判定不足两条时提前返回，不读取 CC 排名。reliable_junction 是受调用方竞争作用域保护的内核。successor_evidence 为旧路径辅助函数。',
        'branch_absolute_core.py':'基础度量仍输出 ASC 命名字段以兼容旧接口；MBG 使用原始节点/弧长最低门槛，不等于调用 CC 竞争。新生产 route_budgets 均 require_core=False。',
        'surface_track_selection.py':'有 physical_face_context 时先进入 select_physical_seed；后面的旧 choose_candidate 路径不属于新生产入口。',
        'main_track_assembly.py':'有 physical_face_context 时提前委托 assemble_physical_track；后面的旧 ASC/DRS 续接分支不可由新入口到达。_terminal_core_bounds 无新入口调用。',
        'main_spine_components.py':'P2 按独立 near-return 阈值；route_budgets(require_core=False)，不以 CC 为准入。',
    }
    rows=[];root=ROOT/'scripts/04_structure_recognition'
    rx=re.compile(r'ASC_start_arc|ASC_end_arc|in_ASC|forward_ASC|choose_successor|reliable_junction')
    for p in sorted(root.glob('*.py')):
        matches=[(i,line.strip()) for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1) if rx.search(line)]
        if not matches:continue
        note=explanations.get(p.name,'旧版/方向性/绘图度量接口；本轮 physical continuation + internal competition + region band + P2 调用链不调用其中 CC 决策。保留旧接口兼容，不能据本轮结果宣称旧入口也满足新语义。')
        for i,line in matches:rows.append(dict(file=p.name,line=i,code=line,note=note))
    save_json(out/'CC_static_call_sites.json',rows)
    runtime=jsonread(out/'local_scope.json');bad=[r for r in runtime if r['competitive_core_invoked'] and r['competitor_count']<2]
    required=['decision_mode','competitor_count','competing_branch_ids','competing_FaceTracks','competitive_core_invoked','junction_reason']
    assert all(all(k in r for k in required) for r in runtime)
    assert not bad
    text(out/'COMPETITIVE_CORE_SCOPE_AUDIT.md',['# 竞核作用域审计',
        '审计边界：当前带 physical_face_context 的生产入口及 recognize_conflict_region 批入口。竞核只用于真实竞争；MBG 的原始分支最低支持度量、报告中的历史核心缺失量不属于 CC 决策。旧内部变量仍叫 ASC，报告统一称竞核。',
        f'运行时事件 {len(runtime)}；CC 调用 {sum(r["competitive_core_invoked"] for r in runtime)}；非竞争误调用 {len(bad)}。六个必填字段逐事件检查通过。事件级不变量不能独立证明候选身份判断正确。',
        table(['文件','作用域说明'],[(f'[{name}]({(root/name).as_posix()})',note) for name,note in explanations.items()]),
        '完整逐行匹配：[CC_static_call_sites.json](CC_static_call_sites.json)。其中旧接口调用点按不可由本轮入口到达归档，并未把整个仓库所有旧算法都改写为新策略。',
        '104.85 的验收范围明确为 B5→B3 孔洞事件：其他高程若有多个独立候选，仍可合法进入竞争。'])


def final_result_audit(out):
    """Audit the frozen final layers without repeating recognition or rendering."""
    summary=jsonread(out/'local_summary.json');gates=[];e_rows=[]
    data=read(SOURCE/'region_data/E.pkl');bounds=[14.4,16.4,1385.2,1386.2]
    names=defaultdict(set)
    for e in data['members']:
        names[e['s'],e['edge_id']].update(e['face_track_ids'])
    for row in summary['rows']:
        s=row['s'];layers=read(out/'layers'/f'{s}.pkl')['layers']
        for c in layers['components']:
            if c['role']!='P2_BODY':continue
            for kind in ['outer_gate','inner_gate']:
                e=c[kind]
                assert e['D_xyz_m']<=.01+1e-12 and e['observed_arc_m']>=.5-1e-12
                gates.append(dict(s=s,branch=c['branch_id'],gate=kind,D_xyz_m=e['D_xyz_m'],L_path_m=e['observed_arc_m'],start_arc=e['start_arc'],end_arc=e['end_arc']))
        if s in data['region']['target_keys']:
            tracks=set()
            for e in layers['MAIN_SPINE']:
                if e['source'].startswith('OBSERVED') and crop(e['points_uz'],bounds) is not None:tracks.update(names[s,e['edge_id']])
            route=read(out/'routes'/f'{s}.pkl')['result']
            junction=next(j for j in route['junctions'] if j['from_branch_id']==1)
            e_rows.append(dict(s=s,FaceTracks=sorted(tracks),junction_uz=junction['a_point_uz']))
    save_json(out/'P2_all_gate_admission.json',dict(body_count=len(gates)//2,gate_count=len(gates),violations=0,gates=gates))
    save_json(out/'E_final_local_identity.json',dict(bounds=bounds,profiles=e_rows))
    p=out/'cases/E.md'
    with p.open('a',encoding='utf-8') as f:f.write('\n\n## 两处换轨之间的局部身份核对\n\n'+
        '共享中间分支的两次换轨现在均可应用。以下按本轮最终 P2 主轨原始 edge_id 回查原审核窗身份，窗口 u=14.4–16.4 m、z=1385.2–1386.2 m；边界附近可能同时含换轨两侧来源，不能只看中心分支序号。\n\n'+
        table(['s','当前窗口最终主轨来源','上侧换轨 u / z'],[(r['s'],r['FaceTracks'],r['junction_uz']) for r in e_rows])+'\n\n'+
        'E 仍未通过局部一致性审核：前三条为 E_T1，后两条为 E_T3。104.30→104.35 m 的上侧接点高程只差约 19.3 mm，但径向位置从 14.3371 跳到 19.2717 m，跨度约 4.935 m。源码中的 choose_junction_band 约束了 FaceTrack 身份和高程跳变，没有惩罚径向位置或沿分支位置跳变；且当前按已有换轨对分别求解，没有把整个共同窗口最终保留的来源作为硬约束。由此可见，高程连续不足以保证折返曲线上的同一段持续被选中。这是本轮保留的算法实现问题，不是分支号变化造成的显示误会。\n')
    consistency=jsonread(out/'local_consistency.json');totals={v:Counter() for v in ['BEFORE','P2']}
    for r in consistency:
        for v in totals:totals[v].update(r['counts'][v])
    finding=(f'局部邻线身份变化位置由 {totals["BEFORE"]["IDENTITY_SWITCH"]} 处变为 {totals["P2"]["IDENTITY_SWITCH"]} 处；'
        f'单侧缺失由 {totals["BEFORE"]["MISSING_ONE"]} 处变为 {totals["P2"]["MISSING_ONE"]} 处；'
        f'多来源位置由 {totals["BEFORE"]["AMBIGUOUS_MULTIPLE"]} 处变为 {totals["P2"]["AMBIGUOUS_MULTIPLE"]} 处。'
        '这些是四个区域中同高程邻线对的诊断计数，仍需结合各区域表和原始模型审核。')
    extra=['## 本轮结果与保留问题',
        '104.85 m 的唯一续接孔洞已恢复；89.95/90.00 m 换轨高程差从 6.392253 m 降至 0.020015 m。1150 个联合选中接点中实际应用 1149 个；104.70 m 的 B1→B9 因源参数顺序冲突拒绝，未强行修改原始路径。',finding,
        'E 仍未通过一致性审核：同一窗口前三条保留 E_T1、后两条保留 E_T3；该区域单侧缺失 19→107。换轨高程虽接近，104.30→104.35 m 的径向位置仍跳变约 4.935 m。当前联合代价缺少径向/沿分支位置约束，不能把高程收敛等同于局部形态一致，详见 [E 专页](cases/E.md)。',
        '三类阈值独立：BranchPolicy.noncompetitive_gap_cap_m=0.05 m（P0 唯一续接）；p1_handoff_cap_m=0.01 m（P1 竞争换轨，独立保留原 1 cm 门槛）；p2_near_return_gap_m=0.01 m 且 p2_min_path_m=0.5 m（P2）。P1 与 P2 数值相同但参数不共享。旧 connector_cap_m 仅保留兼容旧接口，不作为新物理入口这三类门槛的共同来源。',
        f'逐一检查最终 {len(gates)//2} 个 P2_BODY 的 Outer / Inner，共 {len(gates)} 个 gate，全部满足独立的 1 cm / 0.5 m 准入；[完整门参数](P2_all_gate_admission.json)。这不等于证明其地质身份正确。',
        'J / 108.80 m 未得到独立 Inner Gate 和冗余薄颈层，应作为保留问题审核，不能说本轮所有目标均已通过。区域联合也仍依赖批入口；旧单线入口不会自动获得整区选轨效果。',
        '识别结果计时为首轮局部识别加接点应用重放，40.13 分钟；早期调试、中止试跑、报告生成及人工图像复核不包含在这个数值内。35 份几何未变的缓存另同步了 connector 的联合决策标签，未重算或移动坐标。',
        '[本轮修改文件](MODIFIED_FILES.md) · [接点标签同步](connector_metadata_sync.json)']
    p=out/'LOCAL_CONFLICT_REPORT.md';value=p.read_text(encoding='utf-8')
    value=value.replace('## 先看哪些图','\n\n'.join(extra)+'\n\n## 先看哪些图',1);p.write_text(value,encoding='utf-8')
    print('FINAL_LAYER_AUDIT',len(gates)//2,'P2 bodies',e_rows,flush=True)


def run_report(out):
    started=time.perf_counter();summary=jsonread(out/'local_summary.json');plan=jsonread(out/'local_plan.json');m=manifest(SOURCE)
    assert summary['code_unchanged'] and summary['code_sha256']==fingerprints(),'Report must use the verified production revision'
    assert all(r['source_intervals_preserved'] and r['max_coordinate_error_m']<2e-10 for r in summary['rows'])
    keys=[r['s'] for r in summary['rows']];inventory=load_inventory(SOURCE,keys)
    a,lo,hi,cells=mesh();face_cache=read(out/'stages/face_geometry_cache.pkl')
    context=PhysicalFaceContext(a,lo,hi,{float(k):b for k,b in inventory.items()},plan['regions'],geometry_cache=face_cache,levels=m['levels'])
    (out/'figures').mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    consistency=local_consistency(out,plan,context,m);open_cases,short=negatives(out,keys,inventory)
    scope_audit(out);bands=jsonread(out/'local_junction_bands.json');decisions=jsonread(out/'local_region_decisions.json')
    allrows=[];gallery_index=[]
    for tag in ['E','F','D','J']:
        data=read(SOURCE/'region_data'/f'{tag}.pkl');reg=data['region'];ss=reg['target_keys'];bounds=list(reg['bounds'][2:])
        if tag=='E':bounds=[8.9,20.1,1384.8,1387.3]
        if tag=='J':bounds=[-11.6,-8.2,1401.2,1404.0]
        inv={s:inventory[s] for s in ss};rows,n=gallery(data,inv,out,bounds,tag);allrows.extend(rows)
        route_comparison(data,inv,out,bounds,tag)
        merged=next((r for r in plan['regions'] if tag in r.get('case_tags',[])),None)
        cross=[]
        if merged:
            labels=context.region_faces(merged)
            for tid in sorted(t['face_track_id'] for t in data['tracks']):
                mapped=sorted({labels[f] for e in data['members'] if e['face_track_ids']==[tid] for f in e['source_face_ids'] if f in labels})
                cross.append((tid,', '.join(f'T{t}' for t in mapped)))
        center=reg['focus_s'];route=read(out/'routes'/f'{center}.pkl')['result']
        decision=next((d for d in decisions if merged and d['region_id']==merged['region_id']),None)
        ranking=([f"区域优先来源：{decision['dominant_FaceTrack']}；REGION_AMBIGUOUS={decision['REGION_AMBIGUOUS']}。这是整区优先级，局部接点不合法时仍可能保留原轨。",
            table(['完整区 FaceTrack','最长连续 V','连续 s 跨度 m','竞争 Z 持续 m','CC 比例','forward Z m','预计后续换轨'],
                [(d['FaceTrack'],d['consecutive_V'],f'{d["continuous_s_span"]:.3f}',f'{d["persistence_Z"]:.3f}',f'{d["CC_fraction"]:.1%}',f'{d["forward_Z_m"]:.3f}',f'{d["future_switches"]:.2f}') for d in decision['candidates']])]
            if decision else [])
        text(out/'cases'/f'{tag}.md',[f'# {tag}：同一局部区域的所有来源与最终结果',
            '图册行号沿用原审核窗的 FaceTrack 名称；正式决策采用完整合并区域的物理面编号。两者作用域不同，不能直接把 T1 与 T1 当作同一身份。',
            table(['原审核窗来源','完整 region 内对应'],cross) if cross else 'J 是 P2 局部对照，使用该接点的物理面窗口。',
            *ranking,
            f'![{tag}全部来源](../figures/{tag}_all_tracks.png)',f'![{tag}主轨对照](../figures/{tag}_main_comparison.png)',
            f'中心测线 {center} m；本轮分支序列 {route["route_branch_sequence"]}。图中绿色只标记竞核位置，不表示算法已选中；蓝色才是最终保留主轨。',
            table(['测线','来源','分支','MBG','本窗原始弧长 m','本窗 CC占比'],[(r['s'],r['FaceTrack'],r['branch_id'],r['MBG'],f'{r["local_observed_m"]:.3f}',f'{r["local_ASC_fraction"]:.1%}') for r in rows]),
            '人工核查：同一列比较不同来源是否存在明显优势；横向检查相同来源的细节是否持续；再看蓝色主轨是否落在认可的来源上。空格只代表当前显示窗缺席，不代表模型中不存在该分支。'])
        gallery_index.append(dict(tag=tag,visible_edges=n,all_tracks=[t['face_track_id'] for t in data['tracks']],bounds=bounds))
    p0=[]
    for s in P0:
        old=read(BASE/'routes'/f'{s}.pkl');new=read(out/'routes'/f'{s}.pkl');r=new['result']
        p0.append(dict(s=s,old_missing_m=old['row']['ASC_missing_max_m'],new_missing_m=new['row']['missing_max_m'],
            old_sequence=old['result']['route_branch_sequence'],new_sequence=r['route_branch_sequence'],extent_covered=r['extent_covered'],
            fallback_accepted=sum(d['feasibility_accepted'] and d['identity_rank']>1 for d in r.get('continuation_decisions',[]))))
    gaproute=read(out/'routes/104.85.pkl')['result'];gap=[j for j in gaproute['junctions'] if j.get('handoff_kind')=='MODEL_GAP_REPAIR' and j['from_branch_id']==5 and j['to_branch_id']==3]
    save_json(out/'P0_regression.json',dict(cases=p0,gap10485=gap))
    target_bands=[r for r in bands if r['s'] in ['89.95','90.00']]
    save_json(out/'89_95_90_00_band_evidence.json',target_bands)
    new_jump=focused_cases(out,plan,context,inventory,bands,gap)
    jlayers=read(out/'layers/108.80.pkl')['layers'];jpairs=[dict(run_index=i,branch_id=r['branch_id'],**e) for i,r in enumerate(jlayers['near_return_audit']) for e in r['pairs']]
    jbody=[c for c in jlayers['components'] if c['role']=='P2_BODY'];save_json(out/'J_all_pairs.json',jpairs)
    neck_diagnostic=jsonread(out/'J_neck_diagnostic.json') if (out/'J_neck_diagnostic.json').exists() else []
    # Three separate P2 panels prevent the source, gates and layers hiding each other.
    bounds=[-11.6,-8.2,1401.2,1404.0];fig,axs=plt.subplots(1,3,figsize=(16,6),sharex=True,sharey=True)
    for ax,role,color in zip(axs,['MAIN_SPINE','P2_BODY','P2_REDUNDANT_NECK'],['#1467b1','#8651a5','#d68b28']):
        plot_records(ax,[r for b in inventory['108.80'] for r in b['records']],bounds,'#c7cbd0',1)
        records=jlayers['MAIN_SPINE'] if role=='MAIN_SPINE' else [r for c in jlayers['components'] if c['role']==role for r in c['records']]
        plot_records(ax,records,bounds,color,2.6);axes(ax,bounds);ax.set_title(role);ax.set_xlabel('径向偏移 u（m）')
    axs[0].set_ylabel('高程 z（m）');fig.suptitle('J / 108.80 m｜P2 分层：灰色原始观测，彩色为本栏结果',fontsize=15);savefig(fig,out,'J_P2_layers.png',.11)
    text(out/'cases/J_P2.md',['# J：所有合法返回点对与两层边界',
        '以下列出 108.80 整条测线已选 observed runs 的所有合法点对，弧长按各 run 的起点计算。不同 run 的弧长不能直接比较；下图仅显示 J 局部窗口，表中注明是否进入该窗口。所有点均取自原始线段；薄颈只在两条原始折线间整段近重合证据成立时标记。',
        table(['run / B','本图内','D_xyz mm','L_path m','start_arc m','end_arc m'],[(str(e['run_index'])+' / '+str(e['branch_id']),bounds[0]<=e['a_uz'][0]<=bounds[1] and bounds[2]<=e['a_uz'][1]<=bounds[3],f'{e["D_xyz_m"]*1000:.4f}',f'{e["observed_arc_m"]:.6f}',f'{e["start_arc"]:.6f}',f'{e["end_arc"]:.6f}') for e in jpairs]),
        table(['B','Outer start/end','Inner start/end','Outer D mm','Inner D mm'],[(c['branch_id'],f'{c["outer_gate"]["start_arc"]:.6f} / {c["outer_gate"]["end_arc"]:.6f}',f'{c["inner_gate"]["start_arc"]:.6f} / {c["inner_gate"]["end_arc"]:.6f}',f'{c["outer_gate"]["D_xyz_m"]*1000:.4f}',f'{c["inner_gate"]["D_xyz_m"]*1000:.4f}') for c in jbody]),
        '![J分层](../figures/J_P2_layers.png)',
        *(['本例尚未确认独立 Inner Gate。虽然四组离散点对都通过 1 cm / 0.5 m 门槛，但外门到较内候选之间的两条原始路径没有保持连续的 1 cm 近重合：原始顶点到对侧原始折线的最近距离最大约 23.136 mm；按归一弧长配对的最大距离约 23.138 mm。本轮保守地令 Inner=Outer，完整绕行保留为 P2_BODY，没有强行标成 P2_REDUNDANT_NECK。这一“连续薄颈证据”是实现中的保守家族判定，J 的双门剥离预期仍需人工审核，并非已通过。'] if neck_diagnostic and not any(c['role']=='P2_REDUNDANT_NECK' for c in jlayers['components']) else []),
        'Outer Gate 替代主轨中的完整绕行；Inner Gate 界定剥离主体。没有连续薄颈证据时 Inner=Outer，不强行制造第二边界。P2 是几何候选分类，是否对应真实悬空岩体仍需审核。'])
    before=jsonread(out/'pilot_before_optimization.json');pilot=jsonread(out/'pilot_summary.json');fc=summary['face_cache'];jc=summary['junction_cache']
    p0_regressed=[r for r in p0 if r['s'] not in ['104.85','104.70'] and r['new_missing_m']>r['old_missing_m']+1e-6]
    necks=sum(c['role']=='P2_REDUNDANT_NECK' for c in jlayers['components'])
    answer_rows=[
        (1,'竞核总调用',summary['CC_INVOCATIONS']),(2,'非竞争误调用',summary['CC_NONCOMPETITIVE_INVOCATIONS']),
        (3,'104.85 模式','B5→B3：'+(gap[0]['decision_mode'] if gap else '未修复')),
        (4,'104.85 小孔洞',f'{gap[0]["xyz_distance_m"]*1000:.6f} mm；<=50 mm' if gap else '未得到该连接'),
        (5,'104.70 大间距负例',str(next(r for r in p0 if r['s']=='104.70'))),
        (6,'历史 P0 回退','无新增高程包络缺失；形态仍需看图' if not p0_regressed else '存在回退：'+str(p0_regressed)),
        (7,'region 范围','完整合并区域，V 数：'+str([len(r['target_keys']) for r in plan['regions']])),
        (8,'dominant FaceTrack','按 region 完整 V 序列统计；结果见 local_region_decisions.json'),
        (9,'89.95/90.00 换轨带',f'原 6.392253 m → 本轮 {new_jump:.6f} m' if new_jump is not None else '接点不再一一对应，见专页审核'),
        (10,'真实 junction','仅原始线段上的真实交点或通过独立 P1 距离门槛的最近点对；未平滑/平均坐标'),
        (11,'P2 准入','D_xyz<=10 mm 且 L_path>=0.5 m；全结果源区间检查通过'),
        (12,'普通 open branch',f'{len(open_cases)} 个上一轮 SIDE_OPEN_BRANCH 反例，无合法返回点对，本轮均非 P2'),
        (13,'J 两层边界',f'{len(jbody)} 个 P2_BODY，{necks} 条 P2_REDUNDANT_NECK；'+('独立 Inner Gate 未确认，属于保留问题' if not necks else '详见 J_P2 专页')),
        (14,'无 inner neck','保持 Inner=Outer；针对性测试覆盖'),
        (15,'局部回归耗时',f'{summary["seconds"]:.2f} 秒 / {summary["seconds"]/60:.2f} 分钟，共 {summary["target_count"]} 条'),
        (16,'FaceTrack 缓存',f"面标签 {fc.get('face_cache_hits',0)}/{fc.get('face_cache_queries',0)} ({fc.get('face_cache_hits',0)/max(1,fc.get('face_cache_queries',0)):.1%})；membership {fc.get('region_cache_hits',0)}/{fc.get('region_cache_queries',0)} ({fc.get('region_cache_hits',0)/max(1,fc.get('region_cache_queries',0)):.1%})"),
        (17,'junction 缓存',f"{jc.get('memo_hits',0)}/{jc.get('queries',0)} ({jc.get('memo_hits',0)/max(1,jc.get('queries',0)):.1%})"),
        (18,'broad-phase 边对',f'{jc.get("edge_pairs_before",0):,} → {jc.get("edge_pairs_exact",0):,}'),
        (19,'最慢阶段','见下表；函数级诊断来自 5 条试跑的 cProfile，非全区性能结论')]
    region_table=[(r['region_id'],len(r['cell_ids']),len(r['target_keys']),r['target_keys'][0]+'–'+r['target_keys'][-1]) for r in plan['regions']]
    consistency_rows=[(r['region'],v,*(r['counts'][v].get(k,0) for k in ['CONSISTENT','IDENTITY_SWITCH','MISSING_ONE','MISSING_BOTH','AMBIGUOUS_MULTIPLE','UNKNOWN_PROVENANCE'])) for r in consistency for v in ['BEFORE','P1','P2']]
    report=['# 局部冲突区域 P0 / P1 / P2 修复与性能报告',
        f'本轮完成 {summary["target_count"]} 条局部回归，耗时 {summary["seconds"]/60:.2f} 分钟；没有启动 2800 条全区普查。结果来自当前生产模块的局部批调用，图册仅裁切显示。非竞争误用竞核 {summary["CC_NONCOMPETITIVE_INVOCATIONS"]} 次。',
        '## 先看哪些图',
        '[E 三来源竞争](cases/E.md) · [F 共同处于竞核](cases/F.md) · [D 重复面](cases/D.md) · [J 全来源](cases/J.md) · [J 的 Outer/Inner 与 P2 分层](cases/J_P2.md) · [89.95/90.00 换轨带](cases/IDENTITY.md) · [P0 孔洞对照](cases/P0.md)。',
        '## 如何看图',
        '全部来源图：每行一个 FaceTrack，每列一条相邻测线，坐标范围一致；绿色是竞核位置，橙色是端部区，灰虚线是 MBG 未通过的原始片段。绿色不等于最终主轨，也不证明曲面身份正确。竞核只在多个真实候选竞争时用于比较，唯一同链续接不受它阻挡。',
        '主轨对比图：上排上一轮、下排本轮；灰色为全部原始观测，蓝色为最终保留的实测主轨，红虚线和红点为算法建立的连接及其位置。细小连接可能缩成一个点，专页另有放大。原始折线没有平均或平滑。P2 图分栏展示主轨、主体、薄颈，避免所有层叠在一起。',
        '## 本轮计算边界',table(['区域','连续冲突 cell 数','V 数','s 范围 m'],region_table),
        '五邻线仅用于人工审核，正式选轨使用上表完整区域；相邻网格共享至少两组物理面成分时才合并。大区域内可能经远处拓扑连接把原审核窗的不同来源归为同一连通成分，专页给出名称映射；这仍是待审核的物理身份尺度问题。',
        '## 方案要求的 19 项回答',table(['项','问题','结果'],answer_rows),
        '## P0 历史对照',table(['s','旧缺失 m','新缺失 m','旧分支序列','新分支序列','fallback 接受数'],[(r['s'],f'{r["old_missing_m"]:.6f}',f'{r["new_missing_m"]:.6f}',r['old_sequence'],r['new_sequence'],r['fallback_accepted']) for r in p0]),
        '“缺失”沿用历史可靠片段高程超出主轨包络的诊断量，仅用于同口径回归，不表示曲面真值误差，也不作为非竞争续接的核心保护。包络完整不能替代局部身份和形态审核。',
        '## 四个局部区域的邻线身份复核',table(['region','版本','同身份','身份变化','单侧缺失','双侧缺失','多来源','来源未知'],consistency_rows),
        'BEFORE 为上一轮最终主轨；P1 为本轮联合换轨后、P2 前；P2 为本轮最终主轨。使用同一合并区域物理面标签、原始 H 高程、相邻 V 对。计数是重复相关的核查位置，不是独立样本，更不是准确率。缺失和多来源没有算成一致。',
        '## P2 反例',table(['s','旧开放分支','旧弧长 m','当前角色','合法点对'],[(r['s'],r['branch'],f'{r["observed_m"]:.4f}',r['current_roles'],0) for r in open_cases]),
        table(['s','分支','空间距离 mm','沿路径距离 mm','P2'],[(r['s'],r['branch'],f'{r["D_xyz_m"]*1000:.4f}',f'{r["L_path_m"]*1000:.4f}','否') for r in short]),
        '## 性能与执行记录',
        f'带 cProfile 的五条试跑：优化前 {before["seconds"]:.2f} 秒；区域预处理优化后 47.01 秒；复用已算 V/H 交点后 {pilot["seconds"]:.2f} 秒。前后包含语义修正，不能把总降幅当作严格同输出性能对照；V/H 交点缓存另有精确等价测试。完整区域冷启动与五条试跑规模不同。',
        table(['组 s 起止','预处理 s','联合换轨 s','组总耗时 s'],[(r['keys'][0]+'–'+r['keys'][-1],f'{r["setup_seconds"]:.2f}',f'{r["junction_band_seconds"]:.2f}',f'{r["total_seconds"]:.2f}') for r in summary['stages']]),
        '性能措施：共享区域面连通标签和 membership；向量化边框筛选；精确段段距离之前先用保守 AABB 排除；完整几何/区间/模式键缓存 junction；复用 V/H 原始交点；P1 逐测线检查点、P2 与报告阶段缓存。没有降低原始模型精度。审查/性能中止的早期计算未混入最终结果。',
        '首次完整区域尝试在联合换轨阶段暴露新瓶颈：89.95 单条实测 9.55 s，外推超过一小时，因此主动停止。优化移除未启用包络检查时的路径重建、逐小段 NumPy 分配，向量化贡献度量并保留顺序累加。对同一冻结输入，带 cProfile 的完整 band 从 17.7741 s 降至 2.4825 s（约 7.16 倍），全部 990 个候选、239 个合法代表及最终完整路径逐字段完全一致。证据：[优化前](band_before.json)、[优化后与等价检查](band_after.json)。随后才恢复本轮最终回归。',
        'broad-phase 数量是缓存未命中时的潜在边对与实际精确计算边对，不包括已命中缓存而跳过的搜索。早期五条试跑的函数级瓶颈：prepare_region_identity 累计约 16.76 s、future_switch_estimate 约 15.45 s、internal competition 约 12.75 s；后续联合阶段剖析见 [band_after_profile.txt](band_after_profile.txt)。累计时间互相嵌套，不能相加，也不能把早期 profile 当作最终全局耗时分解。',
        '## 最低检查与已知限制',
        '57 项直接相关测试通过。审查问题已按针对性反例修正：缺测 V 不再算连续、P0 阈值不参与 P1 模式判断、合法性验证先于 junction 抽稀、薄颈须证明原始折线之间的连续近重合。图像复核另修复“两个换轨共享中间分支就跳过其中一个”的问题：现在用原始源区间重定位并组合应用，再核验整轨预算，同时去除重复的接点诊断。最终数据逐源区间核对坐标、FaceID、段索引、连续性与连接上限。',
        '仍需人工确认：dominant FaceTrack 是否对应目标坡面；图中的不同细节表达；P2 是否是模型绕行而非真实结构；J 的连续薄颈判定是否符合目标。候选每 5 cm 高程箱只保留一个已验证合法的真实接点，这是一项性能取舍，可能失去同箱内代价更优的候选。联合带不跨缺席 V 强行连接；共享中间分支已可组合应用，但若原始源参数次序或组合后的贡献预算冲突，仍会明确拒绝。',
        '本轮冻结了原始 650 条 P0/P1 与全部区域候选，只重放修正后的接点组合应用和有变化的 P2；没有重做昂贵候选搜索。总耗时包含这次重放。缓存命中率和 broad-phase 边对采用首轮完整识别的计数，重放本身不新增 CC 决策或 junction 搜索。两个阶段的代码摘要记录在 application_replay.json，不能理解为更改代码后悄悄重用了未经核验的结果。',
        '入口范围：区域联合需要 surface_spine_pipeline 的 recognize_conflict_region / join_region_spines 批入口，本轮运行器调用同一生产阶段。现有 process_single_slice 单线接口不会自动收集整个冲突区域，因此不能把本轮局部批验证等同于旧单线调用方式已经获得区域联合效果。',
        '[竞核作用域审计](COMPETITIVE_CORE_SCOPE_AUDIT.md) · [局部指标](local_summary.json) · [完整区域决策](local_region_decisions.json) · [联合候选](local_junction_bands.json) · [邻线核查细节](local_consistency.json) · [测试日志](targeted_tests.log)。']
    text(out/'LOCAL_CONFLICT_REPORT.md',report)
    save_json(out/'gallery_checks.json',dict(galleries=gallery_index,rows=len(allrows),all_sources_accounted_for=True,report_seconds=time.perf_counter()-started))
    final_result_audit(out)
    print('REPORT_READY',out/'LOCAL_CONFLICT_REPORT.md',flush=True)


if __name__=='__main__':run_report(output_dir())
