"""Scientific plots and review artifacts for the approved revised validation."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,pickle,itertools,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from run_p0_p1_p2_validation import latest,SOURCE,BASELINE
from validate_physical_topology import load_inventory,csv_rows,ROOT
from census_directional_consistency import save_json,code_hashes
from branch_absolute_core import branch_metrics,local_edge_scale,directional_branch_metrics,contribution_metrics,connector_gate,route_budgets
from directional_handoff import future_switch_cost
from render_facetrack_branch_gallery import crop
from spine_shadow_validation import near_returns,peel_branch,choose_sweet
import main_track_assembly as assembly

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
COLORS={'ASC':'#19855c','GUARD':'#d78426','REJECTED':'#9b9fa6'}
DISPLAY=(.010,.5,50)  # Illustration only; all 36 settings remain in the scan.


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])


def interval_union(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]+1e-9:result[-1][1]=max(result[-1][1],b)
        else:result.append([float(a),float(b)])
    return result


def member_arc(member,branch):
    i=list(branch['edge_order']).index(member['edge_id']);e=branch['records'][i]
    a,b=np.asarray(e['points_uz']);v=b-a
    t=(np.asarray(member['points_uz'])-a)@v/max(float(v@v),1e-30)
    return branch['arc_positions'][i]+t*(branch['arc_positions'][i+1]-branch['arc_positions'][i]),t,e


def p2_scan(out):
    data=pickle.load((SOURCE/'region_data/J.pkl').open('rb'));reg=data['region'];inventory=load_inventory(SOURCE,reg['target_keys'])
    bounds=[reg['bounds'][2],reg['bounds'][3],reg['bounds'][4],reg['bounds'][5]]
    selected={s:{e['branch_id'] for e in data['members'] if e['s']==s and e['face_track_ids']==['J_T1']} for s in reg['target_keys']}
    events={};raw={}
    for s,bs in inventory.items():
        raw[s]=[b for b in bs if b['branch_id'] in selected[s]]
        for b in raw[s]:
            found=[] if b['kind']=='CLOSED_COMPONENT' else near_returns(b)
            def inside(p):return bounds[0]-1e-8<=p[0]<=bounds[1]+1e-8 and bounds[2]-1e-8<=p[1]<=bounds[3]+1e-8
            events[(s,b['branch_id'])]=[e for e in found if inside(e['a_uz']) and inside(e['b_uz'])]
            print('P2_EVENTS',s,b['branch_id'],len(events[(s,b['branch_id'])]),flush=True)
    (out/'peeling').mkdir(exist_ok=True);summaries=[];shown=None;wide=None
    for gap,minimum,ratio in itertools.product([.002,.005,.010,.020],[.25,.5,1.],[20,50,100]):
        results={}
        for s,bs in raw.items():
            results[s]={b['branch_id']:peel_branch(b,events[(s,b['branch_id'])],closure_gap_m=gap,min_arc_m=minimum,min_ratio=ratio) for b in bs}
            rs=list(results[s].values());cuts=[e for r in rs for e in r['excursions']]
            summaries.append(dict(s=s,closure_gap_mm=gap*1000,minimum_excursion_arc_m=minimum,minimum_ratio=ratio,
                near_closed_excursions=len(cuts),existing_closed_components=sum(b['kind']=='CLOSED_COMPONENT' for b in bs),
                continuous=all(r['continuous'] for r in rs),unresolved_count=sum(len(r['unresolved']) for r in rs),
                side_arc_m=sum(r['side_observed_arc_m'] for r in rs),max_bridge_m=max((r['max_bridge_m'] for r in rs),default=0),
                max_bridge_R_syn=max((r['max_bridge_R_syn'] for r in rs),default=0),
                max_detected_gap_m=max((e['D_xyz_m'] for e in cuts),default=0),
                source_intervals_preserved=all(r['source_intervals_preserved'] for r in rs),used_neighbor_permission=False))
        name=f'gap_{gap*1000:g}mm_arc_{minimum:g}m_ratio_{ratio}.pkl'
        pickle.dump(results,(out/'peeling'/name).open('wb'),protocol=5)
        if (gap,minimum,ratio)==DISPLAY:shown=results
        if (gap,minimum,ratio)==(.020,.5,20):wide=results
    csv_rows(out/'P2_parameter_scan.csv',summaries)
    save_json(out/'P2_events.json',[dict(s=s,branch_id=bid,events=e) for (s,bid),e in events.items()])
    save_json(out/'P2_summary.json',dict(parameter_combinations=36,slices=reg['target_keys'],display_parameters=DISPLAY,
        display_is_final_parameter_choice=False,all_source_intervals_preserved=all(r['source_intervals_preserved'] for r in summaries),
        max_bridge_m=max(r['max_bridge_m'] for r in summaries),all_10880_rows=[r for r in summaries if r['s']=='108.80']))
    p2_figures(out,data,raw,shown,bounds)
    p2_figures(out,data,raw,wide,bounds,parameters=(.020,.5,20),suffix='_20mm')
    print('P2_SCAN_COMPLETE',flush=True)


def p2_figures(out,data,raw,shown,bounds,parameters=DISPLAY,suffix=''):
    keys=data['region']['target_keys']
    for detail in [False,True]:
        bb=list(bounds)
        if detail:
            bb[2:]=[1401.6,1403.0]
            parts=[crop(e['points_uz'],bb) for e in data['members'] if e['face_track_ids']==['J_T1']]
            pts=np.concatenate([v for v in parts if v is not None]);pad=max(.06,np.ptp(pts[:,0])*.06);bb[:2]=[pts[:,0].min()-pad,pts[:,0].max()+pad]
        fig,axs=plt.subplots(3,5,figsize=(19,12),sharex=True,sharey=True)
        for j,s in enumerate(keys):
            allowed={e['edge_id'] for e in data['members'] if e['s']==s and e['face_track_ids']==['J_T1']}
            layers=[[e for b in raw[s] for e in b['records']],
                    [e for r in shown[s].values() for e in r['MAIN_SPINE']],
                    [e for r in shown[s].values() for e in r['SIDE_COMPONENT']]]
            for i,(records,color,label) in enumerate(zip(layers,['#999fa8','#147d69','#9354ba'],['原始 FaceTrack','影子 Main Spine','保留 Side Component'])):
                ax=axs[i,j]
                for e in records:
                    observed=e['source'].startswith('OBSERVED')
                    if observed and e['edge_id'] not in allowed:continue
                    line=crop(e['points_uz'],bb)
                    if line is not None:ax.plot(line[:,0],line[:,1],color=color,lw=1.7,ls='-' if observed else '--')
                ax.set_xlim(bb[:2]);ax.set_ylim(bb[2:]);ax.grid(alpha=.17);ax.ticklabel_format(useOffset=False,style='plain')
                if j==0:ax.set_ylabel(label+'\n高程 z（m）')
                if i==0:ax.set_title(f's = {s} m')
                if i==2:ax.set_xlabel('径向偏移 u（m）')
                if i==1:
                    for r in shown[s].values():
                        for e in r['unresolved']:
                            ax.annotate(f"{e['D_xyz_m']*1000:.2f} mm：保留断开",e['a_uz'],xytext=(7,12),textcoords='offset points',
                                color='#b83238',fontsize=9,arrowprops={'arrowstyle':'-','color':'#b83238'})
        fig.suptitle('J_T1｜单测线剥离对照：原始分支、影子主脊、完整保留的旁支'+('｜细节' if detail else ''),fontsize=16)
        fig.text(.5,.035,f'图示参数：闭合距离 {parameters[0]*1000:g} mm / 绕行弧长 {parameters[1]:g} m / 比值 {parameters[2]}，仅为扫描中的一个展示组合。每条独立判定，不要求邻线先闭合。',ha='center')
        fig.text(.5,.012,'绿色虚线仅表示显式 ≤10 mm 短连接；原始区段均保留。五栏和三行共享坐标，图框截断不是分支终点。',ha='center')
        fig.subplots_adjust(left=.065,right=.99,top=.94,bottom=.085,hspace=.12,wspace=.09)
        fig.savefig(out/'figures'/('J_T1_peeling'+suffix+('_detail' if detail else '')+'.png'),dpi=160);plt.close(fig)


def colored_parts(members,branches,metrics):
    pieces=[]
    for e in members:
        b=branches[e['branch_id']];m=metrics[e['branch_id']]
        if not m['MBG_pass']:pieces.append((e['points_uz'],'REJECTED'));continue
        arcs,_,_=member_arc(e,b);a,z=sorted(arcs);cuts=[a,z]+[v for v in [m['ASC_start_arc'],m['ASC_end_arc']] if a<v<z]
        cuts=sorted(cuts);p=np.asarray(e['points_uz'])
        for lo,hi in zip(cuts,cuts[1:]):
            t=(np.array([lo,hi])-arcs[0])/(arcs[1]-arcs[0]);line=p[0]+t[:,None]*(p[1]-p[0])
            tag='ASC' if m['ASC_start_arc']-1e-9<=(lo+hi)/2<=m['ASC_end_arc']+1e-9 else 'GUARD'
            pieces.append((line,tag))
    return pieces


def p1_figure(out,region,tid,members,inventory,allmetrics,bounds,detail=False):
    fig,axs=plt.subplots(1,5,figsize=(18.5,7),sharex=True,sharey=True)
    for ax,s in zip(axs,region['target_keys']):
        mm=[e for e in members if e['s']==s];by={b['branch_id']:b for b in inventory[s]}
        for line,tag in colored_parts(mm,by,allmetrics[s]):
            clipped=crop(line,bounds)
            if clipped is not None:ax.plot(clipped[:,0],clipped[:,1],color=COLORS[tag],lw=2.15 if tag=='ASC' else 1.45,ls='--' if tag=='REJECTED' else '-')
        ax.set_title(f's = {s} m\n'+(', '.join('B'+str(i) for i in sorted({e['branch_id'] for e in mm})) or '该 FaceTrack 缺席'))
        ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.grid(alpha=.17);ax.ticklabel_format(useOffset=False,style='plain');ax.set_xlabel('径向偏移 u（m）')
    axs[0].set_ylabel('高程 z（m）');fig.suptitle(f'{tid}｜同一 FaceTrack 的甜区与非甜区'+('｜细节' if detail else ''),fontsize=17)
    fig.legend(handles=[Line2D([],[],color=COLORS['ASC'],lw=2.5,label='ASC / 可靠核心'),Line2D([],[],color=COLORS['GUARD'],label='Endpoint Guard / 非甜区'),Line2D([],[],color=COLORS['REJECTED'],ls='--',label='MBG 未通过')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.072))
    fig.text(.5,.035,'原始几何，五栏同坐标；不补线、不平滑。FaceTrack 缺席保持空白；主轨/影子选中状态只列在表格中。',ha='center')
    fig.subplots_adjust(left=.065,right=.985,top=.80,bottom=.20,wspace=.11)
    name=f'{tid}_sweet'+('_detail' if detail else '')+'.png';fig.savefig(out/'figures'/name,dpi=160);plt.close(fig);return name


def build_p1(out):
    rows=[];decisions=[];groups=[]
    zoom={'C':(1361.8,1364.15),'D':(1410.8,1413.1),'E':(1384.6,1387.),'F':(1392.8,1395.),'I':(1378.7,1381.1),'J':(1401.6,1403.)}
    report=['# FaceTrack × 甜区：五邻线影子选择审核',
        '保留上一轮图册的同来源分组、五邻线、同坐标与局部放大。仅用三种图例：绿色粗线是 ASC，橙色是端点非甜区，灰虚线是 MBG 未通过的原始分支。空白表示该测线缺少此来源，不补线。图框截断不代表原始端点。',
        '固定影子顺序：MBG → FaceTrack 连续性 → 竞争区域是否完整处于 ASC → 方向剩余支持 → 最少换轨估计 → AMBIGUOUS。没有使用曲率、细节复杂度或形状相似度。',
        '口径：竞争区就是各案例原有局部窗口内、该 FaceTrack 的原始边区间。ASC 占比按这些区间的弧长计算；完全处于 ASC 才记为“是”，部分重叠记为“混合”。前向指标从沿路径进入此窗口的位置起算，上行/下行分别输出。方向剩余 ASC 弧长与净 Z 延展采用同时不劣的比较；一长一短的权衡不自行加权。',
        'FaceTrack 连续性使用五条邻线中的实际来源出现数。最少换轨复用现有深度 2 的可达性估计，仅到当前局部窗口边界；不可达返回 3 哨兵，不能理解为全局最优换轨数。当前主轨选择只作对照标签，不作为候选过滤条件。全部候选详细字段见 [P1_candidates.csv](P1_candidates.csv)，全部判定过程见 [P1_decisions.json](P1_decisions.json)。']
    for path in sorted((SOURCE/'region_data').glob('*.pkl')):
        data=pickle.load(path.open('rb'));reg=data['region'];rid=reg['region_id'];keys=reg['target_keys'];inventory=load_inventory(SOURCE,keys)
        allmetrics={s:{b['branch_id']:branch_metrics(b,local_edge_scale(bs)) for b in bs} for s,bs in inventory.items()}
        bytid=defaultdict(list)
        for e in data['members']:
            if len(e['face_track_ids'])==1:bytid[e['face_track_ids'][0]].append(e)
        region_rows=[]
        for s in keys:
            route_file=out/'routes'/f'{s}.pkl';wait=time.monotonic()
            while not route_file.exists():
                if time.monotonic()-wait>10800:raise TimeoutError(s)
                time.sleep(5)
            route=pickle.load(route_file.open('rb'))['result'];retained=defaultdict(list)
            for e in route['path_edges']:
                if e['source'].startswith('OBSERVED'):retained[e['edge_id']].append(sorted([e['t0'],e['t1']]))
            bs=inventory[s];by={b['branch_id']:b for b in bs};scale=local_edge_scale(bs)
            for tid,members in bytid.items():
                continuity=len({e['s'] for e in members})
                for bid in sorted({e['branch_id'] for e in members if e['s']==s}):
                    b=by[bid];m=allmetrics[s][bid];em=[e for e in members if e['s']==s and e['branch_id']==bid]
                    intervals=[];selected_length=0.
                    for e in em:
                        aa,t,original=member_arc(e,b);intervals.append(sorted(aa))
                        source_t=original['t0']+t*(original['t1']-original['t0']);lo,hi=sorted(source_t)
                        selected_length+=sum(max(0.,min(hi,v)-max(lo,u)) for u,v in retained[e['edge_id']])
                    intervals=interval_union(intervals);observed=sum(hi-lo for lo,hi in intervals)
                    core=sum(max(0.,min(hi,m['ASC_end_arc'])-max(lo,m['ASC_start_arc'])) for lo,hi in intervals) if m['ASC_exists'] else 0.
                    fraction=core/observed if observed else 0.;boundary=np.asarray(intervals).ravel();n=len(b['arc_positions'])-1
                    nodes=np.interp(boundary,b['arc_positions'],np.arange(n+1));depth=min(np.minimum(nodes,n-nodes));arcdepth=min(np.minimum(boundary,b['arc_positions'][-1]-boundary))
                    for zd in [1,-1]:
                        direction=zd*(1 if b['points_uz'][-1][1]>=b['points_uz'][0][1] else -1)
                        pos=intervals[0][0] if direction==1 else intervals[-1][1]
                        forward=directional_branch_metrics(b,pos,direction,edge_scale=scale,z_direction=zd)
                        switches=future_switch_cost(bs,bid,z_direction=zd,target_z=reg['bounds'][5 if zd==1 else 4]) if m['MBG_pass'] else dict(estimated_additional_switches=3,future_frontier_reached=False)
                        row=dict(region_id=rid,FaceTrack_ID=tid,s=s,branch_id=bid,direction='UP' if zd==1 else 'DOWN',
                            candidate_id=f'{tid}/B{bid}',node_count=m['canonical_node_count'],full_arc_m=m['full_arc_length'],MBG=m['MBG_pass'],
                            ASC_length_m=m['ASC_arc_length'],competition_ASC_fraction=fraction,in_ASC=fraction>=1-1e-8,
                            competition_ASC_state='ASC' if fraction>=1-1e-8 else 'MIXED' if fraction>0 else 'NON_SWEET',
                            endpoint_node_depth=float(depth),endpoint_arc_distance_m=float(arcdepth),
                            forward_ASC_m=forward['L_forward_ASC'],forward_Z_m=forward['Z_forward_new'],
                            face_continuity=continuity,minimum_switches=switches['estimated_additional_switches'],future_frontier_reached=switches['future_frontier_reached'],
                            current_route_selected=selected_length>1e-9,shadow_sweet_selected=False)
                        region_rows.append(row)
            for direction in ['UP','DOWN']:
                candidates=[r for r in region_rows if r['s']==s and r['direction']==direction]
                choice=choose_sweet(candidates)
                for row in candidates:row['shadow_sweet_selected']=row['candidate_id']==choice['selected'];row['shadow_status']=choice['status']
                decisions.append(dict(region_id=rid,s=s,direction=direction,**choice))
        rows.extend(region_rows);report.append(f'## 案例 {rid}｜'+', '.join(keys))
        for tid,members in sorted(bytid.items()):
            bounds=[reg['bounds'][2],reg['bounds'][3],reg['bounds'][4],reg['bounds'][5]]
            figure=p1_figure(out,reg,tid,members,inventory,allmetrics,bounds);entry=dict(region_id=rid,face_track_id=tid,figure=figure,keys=keys)
            report.extend([f'### {tid}',f'![{tid} 五邻线甜区](figures/{figure})'])
            if rid in zoom:
                lo,hi=zoom[rid];parts=[crop(e['points_uz'],[bounds[0],bounds[1],lo,hi]) for e in members];parts=[p for p in parts if p is not None]
                if parts:
                    pts=np.concatenate(parts);pad=max(.06,np.ptp(pts[:,0])*.07)
                    detail=p1_figure(out,reg,tid,members,inventory,allmetrics,[pts[:,0].min()-pad,pts[:,0].max()+pad,lo,hi],True)
                    report.append(f'![{tid} 细节](figures/{detail})');entry['detail_figure']=detail
            if 'detail_figure' not in entry:
                pts=np.asarray([e['points_uz'] for e in members]).reshape(-1,2);lo,hi=pts.min(axis=0),pts.max(axis=0)
                if hi[1]-lo[1]<2.5:
                    pad=np.maximum((hi-lo)*.12,[.08,.10])
                    detail=p1_figure(out,reg,tid,members,inventory,allmetrics,[lo[0]-pad[0],hi[0]+pad[0],lo[1]-pad[1],hi[1]+pad[1]],True)
                    report.append(f'![{tid} 短来源片段放大](figures/{detail})');entry['detail_figure']=detail
            rr=[r for r in region_rows if r['FaceTrack_ID']==tid and r['direction']=='UP']
            report.append(table(['s / B','节点数','全弧长 m','MBG','ASC 长 m','竞争区 ASC','端点深度：节点 / m'],[
                (r['s']+' / B'+str(r['branch_id']),r['node_count'],f"{r['full_arc_m']:.3f}",r['MBG'],f"{r['ASC_length_m']:.3f}",f"{r['competition_ASC_state']} ({r['competition_ASC_fraction']:.0%})",f"{r['endpoint_node_depth']:.2f} / {r['endpoint_arc_distance_m']:.3f}") for r in rr]))
            report.append(table(['s / B','上行剩余 ASC m','上行净 Z m','当前主轨有保留','影子上行选择','影子下行选择'],[
                (r['s']+' / B'+str(r['branch_id']),f"{r['forward_ASC_m']:.3f}",f"{r['forward_Z_m']:.3f}",r['current_route_selected'],r['shadow_sweet_selected'],next(x['shadow_sweet_selected'] for x in region_rows if x['s']==r['s'] and x['candidate_id']==r['candidate_id'] and x['direction']=='DOWN')) for r in rr]))
            report.append('人工审核：□ 选择合理　□ 多来源仍无法区分　□ 局部细节待复核。图表只记录影子建议，不写回全区主轨。')
            groups.append(entry);print('P1_GROUP',tid,flush=True)
    csv_rows(out/'P1_candidates.csv',rows);save_json(out/'P1_decisions.json',decisions);save_json(out/'P1_gallery_index.json',groups)
    competition=[]
    for rid,s in sorted({(r['region_id'],r['s']) for r in rows}):
        tids=sorted({r['FaceTrack_ID'] for r in rows if r['region_id']==rid and r['s']==s and r['direction']=='UP' and r['MBG'] and r['in_ASC']})
        if len(tids)>1:competition.append(dict(region_id=rid,s=s,FaceTracks_in_ASC=tids))
    discriminative=sum(len(d['trace'])>1 and len(d['trace'][1]['remaining'])<len(d['trace'][0]['remaining']) for d in decisions)
    summary=dict(groups=len(groups),candidate_rows=len(rows),decisions=len(decisions),selected=sum(d['status']=='SELECTED' for d in decisions),
        ambiguous=sum(d['status']=='AMBIGUOUS' for d in decisions),ASC_discriminative_decisions=discriminative,
        multiple_FaceTracks_in_ASC=competition,production_selection_changed=False)
    save_json(out/'P1_summary.json',summary)
    report.extend(['## 多个 FaceTrack 同时位于 ASC 的位置',table(['区域','s（m）','同时位于 ASC 的 FaceTrack'],[(r['region_id'],r['s'],', '.join(r['FaceTracks_in_ASC'])) for r in competition]),
        f'甜区阶段在 {discriminative} 次上行/下行影子比较中进一步缩小了候选集合；这说明它有区分作用，不代表已经证明选中了地质真实主脊。最终仍需人工对照图册。'])
    (out/'FACETRACK_SWEET_ZONE_REVIEW.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print('P1_COMPLETE',flush=True)


def dynamic_tail(out):
    rows=[]
    for key,bid in [('122.95',4),('116.25',2)]:
        r=pickle.load((out/'hard_regressions'/f'{key}.pkl').open('rb'))['result'];bs=load_inventory(SOURCE,[key])[key]
        by={b['branch_id']:b for b in bs};scale=local_edge_scale(bs);metrics={i:branch_metrics(b,scale) for i,b in by.items()}
        terminal=assembly._oriented(r['path_edges']);old=terminal[0]['branch_id'];bounds=assembly._terminal_core_bounds(terminal,by[old],metrics[old])
        candidate=assembly._oriented(by[bid]['records']);required=r['route_z_extent'];locked=(np.inf,-np.inf)
        static=assembly._nearest_join(terminal,candidate,required,locked,'upper',terminal_core_bounds=bounds)
        proposed=assembly._nearest_join(terminal,candidate,required,locked,'upper',protect_folds=False)
        geometry=assembly._splice(terminal,candidate,proposed);contrib={i:contribution_metrics(by[i],geometry,metrics[i]) for i in [old,bid]}
        gate=connector_gate(proposed['xyz_distance_m'],contrib[bid]['observed_new_length_m']);budgets=route_budgets(geometry,by,metrics)
        removed=metrics[old]['ASC_arc_length']-contrib[old]['retained_ASC_arc_length']
        assert gate['accepted'] and all(v['accepted'] for v in budgets) and removed>0
        row=dict(s=key,from_branch=old,to_branch=bid,static_connector_m=static['xyz_distance_m'] if static else None,
            shadow_connector_m=proposed['xyz_distance_m'],ASC_removed_m=removed,
            proposed_single_junction_passes_geometry=True,static_ASC_preserved=False,core_trim_safe=None,
            production_route_changed=False,scope='USER_AUTHORIZED_DYNAMIC_TAIL_SHADOW_ONLY; not a completed recognized route',
            max_R_syn=max(v['combined_R_syn'] for v in budgets),junction=proposed)
        rows.append(row);pickle.dump(dict(metadata=row,proposed_geometry=geometry),(out/'hard_regressions'/f'{key}_dynamic_tail_shadow.pkl').open('wb'),protocol=5)
    save_json(out/'dynamic_tail_shadow.json',rows)
    print('DYNAMIC_TAIL_SHADOW_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['p1','p2','dynamic']);a=p.parse_args()
    {'p1':build_p1,'p2':p2_scan,'dynamic':dynamic_tail}[a.action](latest())
