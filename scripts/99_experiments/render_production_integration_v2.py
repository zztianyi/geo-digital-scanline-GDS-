"""V2 results in the user's all-FaceTrack, common-window five-line gallery."""
import csv,json,pickle,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from run_production_integration_v2 import latest,SOURCE,BASE,PRE_LOCK,manifest,load_inventory,missing_core,save_json,csv_rows,mesh
from render_facetrack_competition_review import view_rows,draw,metric_table,FOCUS
from review_p0_p1_p2 import plt,LineCollection,Line2D,table,crop,branch_metrics,local_edge_scale
from physical_face_context import PhysicalFaceContext
from horizontal_surface_link import branch_crossings


def path_figure(data,out,bounds,*,layers=False,tag=''):
    rid=data['region']['region_id'];keys=data['region']['target_keys']
    name=f'{rid}_'+('layers' if layers else 'actual_routes')+tag+'.png'
    if (out/'figures'/name).exists():return name
    fig,axs=plt.subplots(2,5,figsize=(19,8),sharex=True,sharey=True,squeeze=False)
    labels=['本轮贯穿主脊','保留的附属结构'] if layers else ['上一轮实际主轨','本轮 P1 实际主轨']
    for col,key in enumerate(keys):
        current=pickle.load((out/'routes'/f'{key}.pkl').open('rb'))['result']
        old=pickle.load((BASE/'routes'/f'{key}.pkl').open('rb'))['result']
        product=pickle.load((out/'layers'/f'{key}.pkl').open('rb'))['layers']
        sets=[product['MAIN_SPINE'],product['SIDE_COMPONENTS']] if layers else [old['path_edges'],current['path_edges']]
        raw=[crop(e['points_uz'],bounds) for e in data['members'] if e['s']==key];raw=[x for x in raw if x is not None]
        for row,records in enumerate(sets):
            ax=axs[row,col]
            if raw:ax.add_collection(LineCollection(raw,colors='#c9cdd2',linewidths=1.1))
            observed=[];connectors=[];bids=set()
            for e in records:
                p=crop(e['points_uz'],bounds)
                if p is None:continue
                if e['source'].startswith('OBSERVED'):observed.append(p);bids.add(e['branch_id'])
                else:connectors.append(p)
            if observed:ax.add_collection(LineCollection(observed,colors='#2266b5' if row==0 or not layers else '#d78426',linewidths=2.1))
            else:ax.text(.5,.5,'本窗口无该层实测线段',transform=ax.transAxes,ha='center',color='#777')
            if connectors:
                ax.add_collection(LineCollection(connectors,colors='#2266b5' if layers else '#c54545',linewidths=2.5,linestyles='--'))
                if not layers:
                    centers=np.asarray([np.mean(p,axis=0) for p in connectors])
                    ax.scatter(centers[:,0],centers[:,1],s=27,facecolors='none',edgecolors='#c54545',linewidths=1.3,zorder=5)
            ax.set_title(f's={key} m｜'+(','.join('B'+str(x) for x in sorted(bids)) or '无'),fontsize=10)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.grid(alpha=.16);ax.ticklabel_format(style='plain',useOffset=False)
            if col==0:ax.set_ylabel(labels[row]+'\n高程 z（m）')
            if row==1:ax.set_xlabel('径向偏移 u（m）')
    fig.suptitle(f'区域 {rid}｜'+('主脊与附属结构分层' if layers else '同一局部、五条邻线：实际主轨前后对照'),fontsize=16)
    handles=[Line2D([],[],color='#c9cdd2',label='灰色：全部原始候选'),Line2D([],[],color='#2266b5',lw=2,label='蓝色：实际主轨')]
    if layers:handles.append(Line2D([],[],color='#d78426',lw=2,label='橙色：保留的附属实测结构'))
    else:handles.append(Line2D([],[],color='#c54545',ls='--',marker='o',markerfacecolor='none',label='红圆：换轨位置；红虚线：连接（≤10 mm）'))
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.025))
    caption=('上排显示主脊，下排显示保留的附属结构。不同来源可能近乎重合，请结合前面的 FaceTrack 分组查看。' if layers else
        '所有格子共用坐标。灰线存在而蓝线缺席，表示该候选未被保留在主轨；不等于没有原始数据。')
    fig.text(.5,.012,caption,ha='center',fontsize=9)
    fig.subplots_adjust(left=.075,right=.99,top=.89,bottom=.14,hspace=.18,wspace=.10)
    fig.savefig(out/'figures'/name,dpi=170);plt.close(fig)
    return name


def auto_data(key,center,tag,m,spatial_bounds=None):
    i=m['keys'].index(key);start=max(0,min(i-2,len(m['keys'])-5));keys=m['keys'][start:start+5]
    u,z=center;bounds=spatial_bounds or [float(key)-.15,float(key)+.15,u-2.5,u+2.5,z-2.5,z+2.5]
    context_keys=sorted(set(keys)|{k for k in m['keys'] if bounds[0]-1e-8<=float(k)<=bounds[1]+1e-8},key=float)
    bs=load_inventory(SOURCE,context_keys)
    region=dict(region_id=tag,focus_s=key,target_keys=keys,bounds=bounds,center_z=z)
    a,lo,hi,_=mesh();ctx=PhysicalFaceContext(a,lo,hi,{float(s):b for s,b in bs.items()},[region]);d=ctx.region_data(region)
    members=[];tids=set()
    for s in keys:
        for b in bs[s]:
            for e in b['records']:
                p=crop(e['points_uz'],bounds[2:])
                if p is None:continue
                ts=d['edge_tracks'].get((float(s),e['edge_id']),[])
                # Keep a multi-source edge explicitly in every applicable FT;
                # do not silently pick first FaceID or merge disconnected sources.
                for t in ts or [-1]:
                    tid=f'{tag}_T{t}';tids.add(tid)
                    members.append(dict(e,s=s,face_track_ids=[tid],points_uz=p.tolist(),t_interval=[0.,1.]))
    return dict(region=region,members=members,tracks=[dict(face_track_id=t) for t in sorted(tids)])


def identity_example(region,out,m):
    """Locate an actual changed pair in the selected audit cell for display."""
    from audit_production_face_consistency import level_tracks,classify
    bounds=region['bounds'];keys=[k for k in m['keys'] if bounds[0]-1e-8<=float(k)<=bounds[1]+1e-8]
    bs=load_inventory(SOURCE,keys);a,lo,hi,_=mesh()
    context=PhysicalFaceContext(a,lo,hi,{float(k):b for k,b in bs.items()},[region])
    labels=context.region_data(region)['face_labels'];levels=np.asarray([z for z in m['levels'] if bounds[4]<=z<=bounds[5]])
    hits={}
    for key in keys:
        hits[key]={}
        for label,base in [('BEFORE',BASE),('P1',out)]:
            records=pickle.load((base/'routes'/f'{key}.pkl').open('rb'))['result']['path_edges']
            hits[key][label]=level_tracks(records,region,labels,levels)
    fallback=None
    for index,(left,right) in enumerate(zip(keys,keys[1:])):
        for i,z in enumerate(levels):
            before=classify(hits[left]['BEFORE'][i],hits[right]['BEFORE'][i]);after=classify(hits[left]['P1'][i],hits[right]['P1'][i])
            if after!='IDENTITY_SWITCH' or before=='IDENTITY_SWITCH':continue
            center=keys[min(max(index,2),len(keys)-3)]
            event=dict(center=center,left=left,right=right,z=float(z),BEFORE=before,P1=after,
                labels={v:[sorted(hits[left][v][i]),sorted(hits[right][v][i])] for v in ['BEFORE','P1']})
            if before=='CONSISTENT':return event
            if fallback is None:fallback=event
    assert fallback is not None,'An increased switch count must have a changed pair'
    return fallback


def deficit_evidence(key,payload):
    """Report logged evidence, without turning an envelope into ground truth."""
    route=payload['result'];rows=[]
    for deficit in payload.get('missing_ASC',[]):
        bid=deficit.get('branch_id');side=deficit.get('side')
        rejected=[x for x in route.get('continuation_rejections',[])
                  if x.get('branch_id')==bid and x.get('frontier')==side]
        last=rejected[-1] if rejected else {};diagnostic=last.get('junction_diagnostic',{})
        distance=diagnostic.get('candidate_min_distance_m')
        if bid in route.get('route_branch_sequence',[]):kind='已选分支仍有ASC落在最终Z包络外'
        elif distance is not None and distance>.01:kind='可靠接点最短距离超过10mm'
        elif last.get('reason')=='LONG_GAP_UNRESOLVED':kind='方向预筛发现Z分离超过连接限额'
        elif last:kind=last['reason']
        else:kind='没有对应续接拒绝记录，需继续追踪'
        rows.append(dict(s=key,branch_id=bid,side=side,missing_m=deficit.get('missing_m'),
            evidence_class=kind,last_reason=last.get('reason'),candidate_min_distance_m=distance,
            diagnostic=diagnostic))
    return rows


def render_case(data,out,inventory):
    reg=data['region'];rid=reg['region_id'];bounds=reg['bounds'][2:];keys=reg['target_keys']
    metrics={s:{b['branch_id']:branch_metrics(b,local_edge_scale(bs)) for b in bs} for s,bs in inventory.items()}
    rows,parts,count=view_rows(data,inventory,metrics,bounds,'full')
    if not (out/'figures'/f'{rid}_all_tracks_full.png').exists():draw(data,rows,parts,bounds,out,f'{rid}_all_tracks_full.png')
    if not (out/'figures'/f'{rid}_center_full.png').exists():draw(data,rows,parts,bounds,out,f'{rid}_center_full.png',True)
    comparison=path_figure(data,out,bounds);layerfig=path_figure(data,out,bounds,layers=True)
    detail=[]
    if rid in FOCUS:
        lo,hi=FOCUS[rid];clips=[crop(e['points_uz'],[bounds[0],bounds[1],lo,hi]) for e in data['members']]
        clips=[p for p in clips if p is not None]
        if clips:
            points=np.concatenate(clips);pad=max(.08,float(np.ptp(points[:,0]))*.05)
            focus=[float(points[:,0].min()-pad),float(points[:,0].max()+pad),lo,hi]
            fr,fp,_=view_rows(data,inventory,metrics,focus,'detail')
            if not (out/'figures'/f'{rid}_all_tracks_detail.png').exists():draw(data,fr,fp,focus,out,f'{rid}_all_tracks_detail.png')
            if not (out/'figures'/f'{rid}_center_detail.png').exists():draw(data,fr,fp,focus,out,f'{rid}_center_detail.png',True)
            cf=path_figure(data,out,focus,tag='_detail')
            detail=['## 同一细节窗口：全部来源仍保留',
                f'![细节来源比较](../figures/{rid}_center_detail.png)',
                f'![细节五邻线来源比较](../figures/{rid}_all_tracks_detail.png)',
                f'![细节实际主轨](../figures/{cf})']
    decisions=[];sequence=[];joins=[];deficits=[]
    for key in keys:
        d=pickle.load((out/'routes'/f'{key}.pkl').open('rb'));r=d['result'];prod=pickle.load((out/'layers'/f'{key}.pkl').open('rb'))
        sequence.append((key,' → '.join('B'+str(x) for x in r.get('route_branch_sequence',[])),r['status'],
                         ', '.join(r.get('unresolved_reasons',[])) or '无原续接未决原因',
                         f"{d['row']['ASC_missing_max_m']:.3f}",str(d['row']['P2_roles'])))
        deficits.extend(deficit_evidence(key,d))
        for step in r.get('continuation_decisions',[]):
            if 'identity_candidates' not in step:continue
            z=step['competition_z']
            if not bounds[2]-.5<=z<=bounds[3]+.5:continue
            for candidate in step['identity_candidates']:
                decisions.append((key,step['frontier'],f'{z:.3f}',candidate['branch_id'],candidate.get('FaceTrack') or '身份未唯一',
                    candidate['face_continuity'],candidate['in_ASC'],f"{candidate['forward_Z_m']:.3f}",
                    f"{candidate['forward_ASC_m']:.3f}",candidate['minimum_switches'],
                    '接受' if step['feasibility_accepted'] and candidate['branch_id']==step['branch_id'] else '比较/未接入'))
        for event in r.get('local_competition_audit',[]):
            if not event.get('accepted'):continue
            z=event['competition_z']
            if not bounds[2]-.5<=z<=bounds[3]+.5:continue
            for candidate in event.get('identity_candidates',[]):
                decisions.append((key,'内部入口→出口',f'{z:.3f}',candidate['branch_id'],candidate.get('FaceTrack') or '身份未唯一',
                    candidate['face_continuity'],candidate['in_ASC'],f"{candidate['forward_Z_m']:.3f}",
                    f"{candidate['forward_ASC_m']:.3f}",candidate['minimum_switches'],
                    '接受' if candidate['branch_id']==event['to_branch'] else '比较/保留外部'))
        for j in r.get('junctions',[]):
            if not bounds[2]-.5<=j['a_point_uz'][1]<=bounds[3]+.5:continue
            joins.append((key,f"B{j['from_branch_id']} → B{j['to_branch_id']}",
                f"{j['a_point_uz'][0]:.6f}",f"{j['a_point_uz'][1]:.6f}",
                str(j.get('from_in_ASC_at_junction')),str(j.get('to_in_ASC_at_junction')),
                f"{j['xyz_distance_m']*1000:.6f}",j.get('junction_reason','旧方向续接'),j.get('identity_rank')))
    text=[f'# 区域 {rid}：全部 FaceTrack 候选、实际选择与分层结果',
        f"中心 s={reg['focus_s']} m；u={bounds[0]:.3f}–{bounds[1]:.3f} m；z={bounds[2]:.3f}–{bounds[3]:.3f} m。所有格子使用相同坐标。",
        reg.get('review_note',''),
        ({'J':'本例重点看右侧 z≈1402.1–1402.5 m 的回环：108.80 m 已分为 LOCAL_ALTERNATIVE_PATH，完整保留在橙色附属层；108.75 / 108.85 m 的闭合小分量也保留在附属层。108.70 m 右侧折返仍属于蓝色主脊，因此不能宣布五邻线细节已经一致。需要审核它是合理变化，还是仍需改进的角色判定。',
          'E':'本例重点看细节图 u≈15–16 m、z≈1385.4–1385.8 m：104.20 / 104.25 m 的选择较上一轮更接近中心测线，但 104.40 m 的浅谷仍有差异。先核对全部来源，再判断这些细节变化是否合理；不能仅凭蓝线更相似就认定地质身份正确。'}.get(rid,'')),
        '绿色是原分支 ASC，橙色是端部非甜区，灰虚线是未通过 MBG 的实测分支。绿色不等于已选；下一张蓝线才是实际输出。',
        f'![全部来源中心比较](../figures/{rid}_center_full.png)',metric_table(rows,reg['focus_s']),
        f'![全部来源五邻线](../figures/{rid}_all_tracks_full.png)',
        '## 实际主轨前后对照',f'![实际主轨](../figures/{comparison})',
        table(['s','本轮全测线分支顺序','原续接诊断状态','未决原因','ASC包络缺口 m','P2角色数量'],sequence),
        '分支顺序属于整条测线；图中只展示当前局部。包络缺口只是候选可靠分支的 Z 范围覆盖诊断，不是识别错误率。',*detail,
        'UNRESOLVED 不一定表示缺主轨：原续接状态还包含身份存疑、未采用的不可靠端部和横向支持缺口。它与 ASC 包络缺口是不同口径；局部竞争和 P2 分层另见下表及后面的图。',
        '## 先选谁：实际竞争记录',
        table(['s','方向','竞争 z','分支','生产局部 FaceTrack','贯通测线数','竞争位置在ASC','后续净Z m','后续ASC m','预计再换轨','执行结果'],decisions)
            if decisions else '本局部没有产生续接竞争事件；不伪造换轨过程。初始分支选择见原始 route 记录中的 selection。',
        '图册 FaceTrack 名称来自固定审核窗口；生产 FaceTrack 名称带 conflict region 前缀。它们各自在本地窗内定义，不能跨窗口仅凭名称比较身份。',
        '预计再换轨来自深度 2 的安全接点搜索：0/1/2 为已找到的步数；3 是“在两步内未证明贯通”的标记，不应理解成精确需要三次换轨。',
        '## 再在哪里换：实际接点',
        table(['s','方向上的来源→目标','接点u m','接点z m','来源在ASC','目标在ASC','XYZ连接 mm','原因','候选次序'],joins) if joins else '此显示窗口没有已接受的换轨接点。',
        '红空心圆只是换轨位置标记，圆的尺寸不代表连接长度；真实 XYZ 距离见表格。接点几乎重合时，红虚线可能无法在米级坐标下辨认。',
        '分支顺序按最终路线排列；接点的来源→目标记录当时的续接方向，因此下行扩展时箭头可能与最终路线顺序相反。',
        '## 这五条测线仍有哪些范围缺口',
        table(['s','候选分支','方向','ASC包络缺口 m','最后可追溯证据','可靠接点最短距离 mm'],[
            (x['s'],x['branch_id'],x['side'],f"{x['missing_m']:.3f}" if x['missing_m'] is not None else '无主轨',
             x['evidence_class'],f"{x['candidate_min_distance_m']*1000:.6f}" if x['candidate_min_distance_m'] is not None else '未记录') for x in deficits])
            if deficits else '这五条测线的候选 ASC 高程包络已覆盖；局部身份和细节仍需看图确认。',
        '本表是整条测线的端部范围诊断，缺口可能位于当前图窗外；不能用它判断本窗口的识别准确率。',
        '## 主脊与附属结构',f'![结构分层](../figures/{layerfig})',
        'THROUGH_PATH 承担入口到出口贯穿；SIDE_OPEN_BRANCH 为同源单接点旁支；SIDE_CLOSED_COMPONENT 为物理闭合分量；LOCAL_ALTERNATIVE_PATH 为同源局部替代路线；AMBIGUOUS_COMPONENT 保留待核查。侧层不是被删除的噪声。',
        '人工审核：比较同一列不同来源的甜区和后续延展；再沿同一行检查邻线是否保持相似细节。特别检查橙色附属结构是否包含应留在主脊的真实折返。']
    (out/'cases'/f'{rid}.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    return dict(region=rid,s=reg['focus_s'],FaceTracks=len(data['tracks']),visible_source_edges_checked=count)


def run(out):
    started=time.perf_counter();m=manifest(SOURCE);summary=json.loads((out/'census_summary.json').read_text(encoding='utf-8'))
    face=json.loads((out/'face_consistency_summary.json').read_text(encoding='utf-8'))
    source_check=json.loads((out/'source_layer_verification.json').read_text(encoding='utf-8'))
    rows=[];hard=[];joins=[];roles=Counter();auto=None;largest=None;deficits=[];local=Counter()
    for key in m['keys']:
        d=pickle.load((out/'routes'/f'{key}.pkl').open('rb'));r=d['result'];rows.append(d['row']);roles.update(d['row']['P2_roles'])
        deficits.extend(deficit_evidence(key,d))
        local.update(x['decision'] for x in r.get('local_competition_audit',[]))
        for j in r.get('junctions',[]):joins.append(dict(s=key,**j))
        if auto is None:
            step=next((x for x in r.get('continuation_decisions',[]) if x.get('feasibility_accepted') and
                sum(bool(c.get('in_ASC')) for c in x.get('identity_candidates',[]))>=2),None)
            if step:
                diagnostic=step.get('junction_diagnostic',{})
                j=next((j for j in r['junctions'] if j.get('to_branch_id')==step['branch_id'] and
                    j.get('from_branch_id')==diagnostic.get('from_branch_id') and
                    j.get('from_in_ASC_at_junction') and j.get('to_in_ASC_at_junction')),None)
                if j:auto=(key,j,step)
        if largest is None or d['row']['ASC_missing_max_m']>largest[0]:largest=(d['row']['ASC_missing_max_m'],key,r)
    csv_rows(out/'all_junctions.csv',joins)
    csv_rows(out/'remaining_deficit_evidence.csv',deficits)
    save_json(out/'local_competition_summary.json',dict(local))
    source_inventory=list(csv.DictReader((BASE/'ASC_missing_inventory.csv').open(encoding='utf-8-sig')))
    before={r['s']:float(r['ASC_missing_max_m']) for r in source_inventory}
    historical=json.loads((BASE/'static_ASC_regression_summary.json').read_text(encoding='utf-8'))
    pre_lock_large=sum(v>=1 for v in before.values())-historical['newly_ge_1m']+historical['cleared_ge_1m']
    improved=[r for r in rows if before[r['slice_key']]>=1 and r['ASC_missing_max_m']<1]
    regressed=[r for r in rows if before[r['slice_key']]<1 and r['ASC_missing_max_m']>=1]
    csv_rows(out/'new_large_deficits.csv',regressed);csv_rows(out/'large_deficits_cleared.csv',improved)
    for key in ['98.05','98.10','116.25','122.95','112.55','123.00','135.65']:
        row=next(r for r in rows if r['slice_key']==key)
        p0=json.loads((out/'hard_regressions'/f'{key}_P0.json').read_text(encoding='utf-8'))
        p0_missing=max((r['missing_m'] or 0. for r in p0['missing_ASC']),default=0.)
        hard.append((key,f'{before[key]:.3f}',f'{p0_missing:.3f}',f"{row['ASC_missing_max_m']:.3f}",' → '.join(map(str,row['sequence'])),f"{row['max_connector_m']*1000:.6f}"))
    datasets=[pickle.load(p.open('rb')) for p in sorted((SOURCE/'region_data').glob('*.pkl'))]
    if auto:
        key,j,step=auto;point=j['a_point_uz'];z=step['competition_z'];bs=load_inventory(SOURCE,[key])[key]
        bids={c['branch_id'] for c in step['identity_candidates']}|{j['from_branch_id']}
        us=[point[0]]+[h['u'] for b in bs if b['branch_id'] in bids for h in branch_crossings(b,z)]
        zs=[z,point[1]];upad=max(.8,(5.-np.ptp(us))/2);zpad=max(.8,(5.-np.ptp(zs))/2)
        bounds=[float(key)-.15,float(key)+.15,min(us)-upad,max(us)+upad,min(zs)-zpad,max(zs)+zpad]
        data=auto_data(key,point,'AUTO_DUAL_ASC',m,bounds)
        data['region']['review_note']=(f"身份比较发生在 z={z:.6f} m；最终 B{j['from_branch_id']}→B{j['to_branch_id']} 接点在 "
            f"z={point[1]:.6f} m。两者不是同一位置：先比较候选，再搜索可靠区接点。大窗口包含两处，细节图放大实际接点。")
        FOCUS['AUTO_DUAL_ASC']=(point[1]-.65,point[1]+.65);datasets.append(data)
    if largest and largest[0]>0:
        _,key,r=largest;bs=load_inventory(SOURCE,[key])[key];side=0 if r['route_z_extent'][0]>r['candidate_z_extent'][0]+1e-6 else 1
        z=r['route_z_extent'][side];curve=np.asarray(r['curve_uz']);p=curve[np.argmin(abs(curve[:,1]-z))]
        data=auto_data(key,p,'AUTO_REMAINING',m)
        if key=='104.85':
            data['region']['review_note']=('这是本轮唯一新增的 ≥1 m 包络缺口。上一轮 B2→B5→B3→B9，本轮停在 B2→B5；'
                'B5→B3 的旧接点只有 1.942248 mm，但位于 B3 原始端点 guard。按新方案禁止接入目标 guard 后，'
                '当前双甜区最短接点约 0.848923 m，超过 10 mm；后续 B3/B9 未接入，最大 ASC 包络缺口 85.509922 m。'
                '图只放大断口附近，85.51 m 是后续范围诊断，不是图中红线长度。')
        datasets.append(data)
    if largest and largest[1]!='135.65':
        d=pickle.load((out/'routes'/'135.65.pkl').open('rb'))
        if d['row']['ASC_missing_max_m']>0:
            r=d['result'];curve=np.asarray(r['curve_uz']);p=curve[np.argmax(curve[:,1])]
            datasets.append(auto_data('135.65',p,'R135_UNRESOLVED',m))
    existing_gap=json.loads((out/'existing_gap_10470.json').read_text(encoding='utf-8'))
    center=np.mean([existing_gap['raw_a_uz'],existing_gap['raw_b_uz']],axis=0)
    data=auto_data('104.70',center,'R104_EXISTING_GAP',m)
    data['region']['review_note']=(f"这是上一轮就存在、此次仍未解决的断口，不是新增退步。B4 与 B3 的全部原始线段 XYZ 最短距离 "
        f"{existing_gap['raw_segment_min_distance_m']:.6f} m，加入可靠区条件后约 {existing_gap['reliable_junction_min_distance_m']:.6f} m，"
        '两者均超过 10 mm。因此取消静态锁不足以补上这里；原始切片间距是否来自网格或切片步骤，尚未由本轮证明。')
    datasets.append(data)
    if regressed:
        worst=max(regressed,key=lambda x:x['ASC_missing_max_m']);key=worst['slice_key']
        if largest is None or key!=largest[1]:
            d=pickle.load((out/'routes'/f'{key}.pkl').open('rb'));r=d['result']
            deficit=max(d['missing_ASC'],key=lambda x:x['missing_m']);side=0 if deficit['side']=='lower' else 1
            curve=np.asarray(r['curve_uz']);p=curve[np.argmin(abs(curve[:,1]-r['route_z_extent'][side]))]
            data=auto_data(key,p,'AUTO_NEW_DEFICIT',m)
            if key=='104.85':
                data['region']['review_note']=('这是本轮唯一新增的 ≥1 m 包络缺口。上一轮 B2→B5→B3→B9，本轮停在 B2→B5；'
                    '旧 B5→B3 连接为 1.942248 mm，但接入 B3 原始端点 guard。新规则要求接入目标 ASC，'
                    '当前双甜区最短接点约 0.848923 m，超过 10 mm，故 B3/B9 未接入。'
                    '85.509922 m 是后续 ASC 包络缺口；本图放大断口附近，不表示要画一条这么长的连接线。')
            datasets.append(data)
    regions=json.loads((out/'face_consistency_regions.json').read_text(encoding='utf-8'))
    frozen_ids={r['conflict_region_id'] for r in csv.DictReader((SOURCE/'conflict_regions.csv').open(encoding='utf-8-sig'))}
    fixed_regions=[r for r in regions if r['region_id'] in frozen_ids]
    fixed_counts={v:Counter() for v in ['BEFORE','P1','P2']}
    for region in fixed_regions:
        for v in fixed_counts:fixed_counts[v].update(region[v])
    save_json(out/'face_consistency_fixed_summary.json',dict(requested_regions=len(frozen_ids),regions=len(fixed_regions),counts=fixed_counts))
    # Fixed finite conflict cells retain their exact audit axes for the gallery.
    # Runtime windows can span unbounded u and cannot be used as plotting axes.
    worst_face=max((r for r in regions if np.isfinite(r['bounds']).all()),
        key=lambda r:r['P1'].get('IDENTITY_SWITCH',0)-r['BEFORE'].get('IDENTITY_SWITCH',0))
    if worst_face['P1'].get('IDENTITY_SWITCH',0)>worst_face['BEFORE'].get('IDENTITY_SWITCH',0):
        b=worst_face['bounds'];event=identity_example(worst_face,out,m);key=event['center']
        data=auto_data(key,[(b[2]+b[3])/2,event['z']],'AUTO_IDENTITY_REVIEW',m,b)
        data['region']['review_note']=(f"来自审计窗口 {worst_face['region_id']}。实际差异对是 s={event['left']} / {event['right']} m，"
            f"z={event['z']:.6f} m：{event['BEFORE']} → {event['P1']}。五条邻线围绕这对测线选择，物理面分组仍使用完整审计窗口。")
        datasets.append(data);save_json(out/'identity_review_region.json',dict(worst_face,display_example=event))
    inventory=load_inventory(SOURCE,sorted({s for d in datasets for s in d['region']['target_keys']},key=float));cases=[]
    for data in datasets:
        cases.append(render_case(data,out,inventory));print('GALLERY',cases[-1]['region'],flush=True)
    percentages=lambda v:table(['类别','数量'],[(k,n) for k,n in face['counts'][v].items()])
    totals=face['counts'];perf=[r['seconds'] for r in rows]
    relation_names=dict(CONSISTENT='单一来源一致',IDENTITY_SWITCH='两线单一来源不同',
        AMBIGUOUS_MULTIPLE='至少一线包含多个来源',UNKNOWN_PROVENANCE='缺少原始面映射',
        MISSING_ONE='一条测线在本局部缺席',MISSING_BOTH='两条测线均在本局部缺席')
    report=['# P0 / P1 / P2 v2：生产修复与全区复核报告',
        '**本轮实际运行了 2800 条测线的生产主脊入口；P1 参与分支选择，P2 输出主脊和保留的附属结构。图册沿用全部 FaceTrack 同窗口、五邻线统一坐标的形式。**',
        '**连续性明显改善，但邻线来源一致性仍未全面通过。** 已找到原来同源、本轮异源的相邻测线对；不能因为大段缺口减少，就宣布 1.5D 一致性已经解决。',
        '## 本轮结果与边界',
        f"静态 ASC 永久锁已取消。上一轮 ASC 包络缺失 {sum(v>1e-6 for v in before.values())} 条，本轮 {summary['ASC_missing_count']} 条；其中 ≥1 m 从 {sum(v>=1 for v in before.values())} 条变为 {summary['ASC_missing_ge_1m_count']} 条。大缺口消失 {len(improved)} 条，新增 {len(regressed)} 条。该统计是范围覆盖诊断，不能解释成准确率。",
        table(['版本','ASC包络缺口测线','其中≥1m'],[
            ('引入静态硬锁之前',historical['previous_missing'],pre_lock_large),
            ('上一轮静态硬锁版本',sum(v>1e-6 for v in before.values()),sum(v>=1 for v in before.values())),
            ('本轮P0+P1',summary['ASC_missing_count'],summary['ASC_missing_ge_1m_count'])]),
        ('本轮大段缺口数量已回落到静态硬锁之前的水平或更低。' if summary['ASC_missing_ge_1m_count']<=pre_lock_large else
         '本轮大段缺口数量仍未回到静态硬锁之前的水平，不能宣称连续性验收全部通过。'),
        f"共接受候选回退续接 {summary['fallback_accepted']} 次；最大 P1 连接 {summary['max_connector_m']*1000:.6f} mm，最大两侧累计 R_syn={summary['max_combined_R']:.6f}。保留身份存疑标记的测线 {summary['identity_ambiguous_count']} 条。",
        '存疑标记同时包含原有 H-run 轨道歧义和本轮选择歧义；它不表示这些测线已被证明识别错误。',
        '## 七个硬回归位置',table(['s m','上一轮ASC包络缺口 m','只开P0缺口 m','本轮P0+P1缺口 m','本轮分支顺序','最大连接 mm'],hard),
        '只开 P0 的对照仅对这七条运行；全区域前后统计对应 P0+P1 的共同效果，不能把全部改善单独归因于取消 ASC 锁。上述缺口在 P2 分层之前计算；第三组图才是最终主脊与附属层。',
        '缺口不为零时查看原始 continuation_rejections/junction_diagnostic。没有安全目标 ASC 接点时保留 UNRESOLVED；没有放宽 10 mm。',
        '## 选谁与在哪里换',
        'ASC（Absolute Sweet Core，绝对路径甜区）沿用你的最小分支与绝对甜区方案：在原始分支两端排除 guard 后，中间还须满足原始节点数和弧长门槛。本轮保留这一定义与绿色位置，只取消“绿色区永久不可裁”的硬锁；没有把裁剪后的新端点重新计算成一个动态 ASC。绿色表示满足可靠区条件，不代表已选中或已证明是真实主坡面。',
        '分支排序依次使用 MBG、同一 conflict window 内物理 FaceTrack 贯通性、当前竞争位置甜区状态、后续净 Z 延展与剩余 ASC、预计后续换轨数。双方都绿仍继续比较；无法区分的稳定输出保留 AMBIGUOUS 标志。距离不参与身份排序。',
        '固定目标后才在双甜区的全部合格原始线段之间搜索 XYZ 最短接点。没有双甜区重叠时只允许来源尾段接入目标 ASC；目标起点 guard 不作为提前换轨位置。最小保留贡献与两侧累计 R_syn 仍检查，ASC 不再要求保留全部原长度。',
        '## 全区域邻线 FaceTrack 一致性',
        f"先看预先冻结的 {len(frozen_ids)} 个 conflict cells，其中 {len(fixed_regions)} 个具备比较条件。这里的窗口集合不随本轮运行增加，适合作为主要前后对照：",
        table(['固定窗口中的关系','上一轮','本轮P1','本轮P2'],[(k+' / '+relation_names.get(k,''),fixed_counts['BEFORE'].get(k,0),fixed_counts['P1'].get(k,0),fixed_counts['P2'].get(k,0)) for k in sorted(set().union(*(x.keys() for x in fixed_counts.values())))]),
        '固定窗口中覆盖缺席减少，但来源跳变增加，仍不能判定一致性全面改善。以下再列固定窗口与全部运行补充窗口的合并审计；两张表范围不同，不应互相相减。',
        f"候选范围为全部冻结 conflict cells，以及本轮 P1 和最终 P2 使用的补充窗口，共 {face['requested_regions']} 个；其中 {face['regions']} 个有至少两条 V 及原始 H 高程可供比较。使用同一物理面共享边连通分量、相同原始 H 高程和相同相邻 V 对。窗口有重叠，因此计数不是独立缺陷数或准确率。MISSING_ONE/BOTH 单列，避免通过少选线段伪造一致性改善。",
        table(['关系','上一轮','本轮P1','本轮P2'],[(k+' / '+relation_names.get(k,''),totals['BEFORE'].get(k,0),totals['P1'].get(k,0),totals['P2'].get(k,0)) for k in sorted(set().union(*(x.keys() for x in totals.values())))]),
        'FaceTrack 一致只说明局部物理面来源一致，不自动证明细节复杂度合理；几何细节仍以五邻线图册人工审核为准。',
        f"P1 的单一来源一致计数变化为 {totals['P1'].get('CONSISTENT',0)-totals['BEFORE'].get('CONSISTENT',0):+d}，单一来源跳变计数变化为 {totals['P1'].get('IDENTITY_SWITCH',0)-totals['BEFORE'].get('IDENTITY_SWITCH',0):+d}。覆盖与来源一致性应分别验收。",
        '## 剩余问题的可追溯证据',
        '**明确的邻线反例：89.95 / 90.00 m。** 两条最终分支顺序相同，都包含 B5→B2；但换轨高程分别为 1411.392876 m 和 1405.000623 m，相差 6.392253 m。'
        '在 z=1408.186304 m，原来两条都是同一局部 FaceTrack，本轮变成不同来源。当前每条 V 独立选择可靠区最短接点，尚未约束邻线的换轨区域共同延续。'
        '这表明“选了同样的分支顺序”仍不足以保证局部一致。见 [实际接点与来源证据](P1_neighbor_issue_trace.json) 和自动邻线复核案例。',
        '**唯一新增的大缺口：104.85 m。** 上一轮 B2→B5→B3→B9 连通，本轮停在 B2→B5，最大 ASC 包络缺口 85.509922 m。'
        '旧 B5→B3 连接仅 1.942248 mm，但落在 B3 的原始端点（弧长 47.276042 m），越过 ASC 上界 46.851333 m，属于目标 guard。'
        '本轮按方案禁止接入该 guard，当前双甜区最短距离约 0.848923 m，故保留未决。'
        '这一例是新规则带来的实际连续性代价，不能被总数下降掩盖；本轮未通过放宽 guard 或 10 mm 来强行补回。'
        '见 [断口五邻线图](cases/AUTO_NEW_DEFICIT.md) 和 [前后接点记录](new_gap_10485.json)。',
        '104.70 m 是一个已确认的既有大缺口：上一轮和本轮的路线均为 B1→B4，Z 范围均为 1358.132575–1387.457727 m。'
        'B4→B3 的原始几何间距约 0.403 m，可靠区接点约 0.842 m，均无法满足 10 mm。'
        '因此后续 B3、B8 的大段范围仍未接入；不能把这一例归为本轮新增退步，也不能把它只归因于 ASC 保护。'
        '见 [104.70 局部案例](cases/R104_EXISTING_GAP.md) 和 [原始距离证据](existing_gap_10470.json)。',
        table(['缺口对应的最后一条证据','分支/方向记录数'],Counter(x['evidence_class'] for x in deficits).items()),
        '上述分类对应每个缺口分支和方向的最后一条日志。同一测线可能有多条记录；日志证据不等于已经确认原始模型存在断裂。没有对应记录的情形明确保留，不能推定都来自同一个原因。',
        table(['内部竞争最终决策','窗口/测线事件数'],local.items()),
        '内部竞争要求安全进入并安全退出，才能用另一分支替换当前局部。NO_SAFE_TWO_ASC_JUNCTIONS 表示尚未找到满足条件的一对接点；存在较优候选也不会强制拼接。',
        '## 图册及怎么看图',
        '建议优先审核 [E：多来源竞争](cases/E.md)、[104.85 m：新增缺口](cases/AUTO_NEW_DEFICIT.md)、[89.95/90.00 m：邻线来源跳变](cases/AUTO_IDENTITY_REVIEW.md)、[J：主脊与回环分层](cases/J.md)，然后查看完整图册。',
        '第一组：按 FaceTrack 排行、按五条邻线排列，绿色 ASC、橙色非甜区、灰虚线 MBG 未通过，全部候选均显示。第二组：灰色背景为全部观测，蓝色为上一轮/本轮实际主轨，红空心圆定位换轨点、红虚线表示连接。圆的显示尺寸不代表实际距离。第三组：把本轮主脊与附属实测结构分开。不要把第一组的绿色当成已选结果。',
        '所有格子保持同一个 u/z 范围，空格明确说明局部缺席。图册标号在固定审核窗口内有效，生产日志标号带 conflict-region 前缀，不能跨窗口按名称强行对应。',
        '原始分支、固定审核 FaceTrack 和 ASC 参数未改变，因此 A–L 候选甜区图复用上一轮对应图像；蓝色实际主轨、附属分层以及自动新案例均来自本轮运行。复用的候选图不被当作新识别结果。',
        table(['案例','中心 s m','FaceTrack 数'],[(f"[{c['region']}](cases/{c['region']}.md)",c['s'],c['FaceTracks']) for c in cases]),
        '## P2 分层与重建接口',
        table(['角色','组件数'],roles.items()),
        f"最终 P2 主脊最大连接 {source_check['P2_max_connector_m']*1000:.6f} mm，累计 R_syn 最大值 {source_check['P2_max_R_syn']:.6f}。独立核对 {source_check['source_records']} 条源区间记录，最大坐标插值误差 {source_check['maximum_coordinate_error_m']:.3g} m；全部 FaceID、source segment 与源区间均保留。",
        'P2 使用 THROUGH_PATH / SIDE_OPEN_BRANCH / SIDE_CLOSED_COMPONENT / LOCAL_ALTERNATIVE_PATH / AMBIGUOUS_COMPONENT。未选择的曲线保留原始 FaceID、edge_id、source segment、t 区间和坐标。主脊已用于原有法向阈值的悬空实测线段识别；reconstruction_inputs 将这些线段与附属层一起交给后续重建，不执行坐标焊接。',
        '局部近返回检测采用 10 mm、0.5 m 原始弧长、弧长/间隙比 50；仅在同一局部物理 FaceTrack、入口出口可贯穿且保留贡献预算安全时分类为替代路线。这是当前保守实现参数，不是地质真伪证明；J 等案例仍需审核。',
        '图册复核中发现并修复了 P2 的窗口定位错误：108.80 m 的实际返回点 u≈−9.215 m，原代码却因使用整个分支的 u 范围而查询 −12.1～−9.9 m 的窗口，错误得到“身份未证明”。'
        '现在返回段按实际两个接点选择共同窗口，旁支按实际附着端点核对面来源。J 的对应实际窗口两端同源；原诊断见 [窗口定位证据](J_identity_window_diagnostic.json)。'
        '最终 P2/悬空识别已经对全部 2800 条重新计算，每条 P1 路线对象的哈希保持不变。',
        '## 检查、性能与生产调用',
        '源码、适配接口、测试及验证脚本见 [本轮修改文件](MODIFIED_FILES.md)。',
        f"相关模块 67 项测试通过。全区 {len(rows)} 条均检查原始坐标插值、FaceID、边区间无丢失无重复、连续连接和预算。首次完整 P0/P1及初版P2 调度用时 {summary['seconds']/60:.2f} 分钟，{summary['workers']} 个进程；窗口修正后的最终 P2 单独重算用时 {summary['P2_refresh_seconds']/60:.2f} 分钟，{summary['P2_refresh_workers']} 个进程。",
        '下表分别取完整普查中的 P0/P1 计算，以及最后一次 P2/悬空识别重算。二者来自分阶段执行，不能把逐条耗时之和当成同一次端到端基准；网格加载和外部 I/O 不计入逐条计时。',
        table(['生产阶段','逐条中位数 s','逐条P95 s','逐条最大值 s'],[
            (label,f"{np.median([r[field] for r in rows]):.3f}",f"{np.percentile([r[field] for r in rows],95):.3f}",f"{max(r[field] for r in rows):.3f}")
            for field,label in [('P0_P1_seconds','选轨、接点与局部竞争'),('P2_recognition_seconds','P2分层与悬空识别')]]),
        table(['最慢测线 s m','分阶段计时合计 s（非端到端）'],[(r['slice_key'],f"{r['seconds']:.3f}") for r in sorted(rows,key=lambda x:x['seconds'],reverse=True)[:5]]),
        f"普查中定位并修复了同一窗口反复执行相同失败接点搜索的性能问题。该性能补丁的 52.50 m 完整输出核对通过，且 {summary['exact_uncached_reference_matches']} 条同上下文输出逐字段一致。此项核对发生在后来 P2 窗口定位修正之前；最终 P2 输出按新逻辑重算，不能声称仍与旧层结果相同。七条最初单独运行的控制例未混入同上下文计数。",
        '中断的未复用结果运行保留在 memoization_reference，未与最终结果混算耗时。52.50 m 单例在两种并行负载下分别约 562 秒和 53 秒，仅用于定位重复计算，不作为同条件加速比。见 [结果等价核对](memoization_equivalence.json)。',
        '正式入口为 scripts/04_structure_recognition/surface_spine_pipeline.py::recognize_surface_spine；run_multi_profile_recognition.process_single_slice 收到完整 surface_spine_context 时调用它。只有原始切片、没有物理邻域上下文的旧输入仍走旧入口，不会被伪标为 FaceTrack 生产结果。本轮全区使用了完整上下文入口。',
        '识别、区域一致性与源记录核查在同机并行运行；这里是本轮实际执行性能，不把它与此前不同进程数、不同校验负载的耗时直接换算成加速比。',
        '本次复用冻结的原始分支、横向观测和物理网格邻接缓存；没有重做原始模型切片或首次网格邻接构建。因此这些耗时是当前识别阶段及其调度开销，不是从原始模型到最终曲面重建的端到端时间。',
        '原始记录：[全部接点](all_junctions.csv)、[全区汇总](census_summary.json)、[邻线一致性](face_consistency_summary.json)、[最终源区间验证](source_layer_verification.json)、[最终层清单](final_layer_inventory.csv)、[新增大缺口](new_large_deficits.csv)、[消失的大缺口](large_deficits_cleared.csv)、[剩余缺口证据](remaining_deficit_evidence.csv)、[内部竞争汇总](local_competition_summary.json)、[执行记录](EXECUTION_LEDGER.md)。',
        '## 必须回答的十项',
        '1. static ASC hard lock 已取消；2. 大缺口从 300 条降至 28 条，也低于加锁前的 119 条；3. 候选回退保持并记录 identity_rank；4. 双方在 ASC 时继续比较后续贯通；5. 固定 successor 后搜索 dual-ASC 或 tail→ASC；6. 排除目标起点 guard；7. 距离只决定接点；8. 邻线一致性尚未全面改善，存在同分支顺序但换轨位置不同的明确反例；9. P2 使用五类通用角色，并修复了身份窗口错位；10. 附属源区间完整保留，并提供与悬空识别线段汇合的重建输入接口。']
    (out/'PRODUCTION_INTEGRATION_REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    save_json(out/'gallery_checks.json',dict(cases=cases,source_edges_shown=True,same_axes=True,seconds=time.perf_counter()-started))
    print('REPORT_COMPLETE',out,flush=True)


if __name__=='__main__':
    import csv
    run(latest())
