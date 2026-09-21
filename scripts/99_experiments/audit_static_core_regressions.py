"""Explain newly missing ASC with single-junction static-core counterfactuals.

No route recomputation or production geometry update. Only the changed ASC
protection is relaxed to the previous fold rule, to isolate that mechanism.
"""
import csv,pickle,sys,json
from collections import defaultdict,Counter
import numpy as np
from run_p0_p1_p2_validation import latest,SOURCE,BASELINE,assembly,branch_metrics,local_edge_scale,contribution_metrics,connector_gate,route_budgets,save_json
from validate_physical_topology import load_inventory,csv_rows
from finish_p0_p1_p2_validation import plt,plot_records


def run(out):
    with (BASELINE/'missing_route_stage_inventory.csv').open(encoding='utf-8-sig') as f:previous=list(csv.DictReader(f))
    before=defaultdict(float)
    for r in previous:before[r['s']]=max(before[r['s']],float(r['missing_ASC_z_m']))
    with (out/'ASC_missing_inventory.csv').open(encoding='utf-8-sig') as f:after={r['s']:float(r['ASC_missing_max_m']) for r in csv.DictReader(f)}
    oldkeys={s for s,v in before.items() if v>1e-6};newkeys={s for s,v in after.items() if v>1e-6}
    newly=sorted([s for s,v in after.items() if v>=1 and before[s]<1],key=float);inventory=load_inventory(SOURCE,newly);rows=[];details=[]
    for key in newly:
        branches=inventory[key];r=pickle.load((out/'routes'/f'{key}.pkl').open('rb'))['result'];records=r['path_edges'];required=r['route_z_extent'];scale=local_edge_scale(branches)
        by={b['branch_id']:b for b in branches};metrics={i:branch_metrics(b,scale) for i,b in by.items()};pieces=assembly._remaining([b for b in branches if metrics[b['branch_id']]['MBG_pass']],records);counter=[]
        for direction in ['lower','upper']:
            terminal=[]
            for e in (records if direction=='lower' else records[::-1]):
                if not e['source'].startswith('OBSERVED') or (terminal and e['branch_id']!=terminal[0]['branch_id']):break
                terminal.append(e)
            if direction=='upper':terminal.reverse()
            if not terminal:continue
            locked=records[len(terminal):] if direction=='lower' else records[:-len(terminal)];zz=[p[1] for e in locked for p in e['points_uz']];lock=(min(zz),max(zz)) if zz else (np.inf,-np.inf)
            old=terminal[0]['branch_id'];bounds=assembly._terminal_core_bounds(terminal,by[old],metrics[old]);core_before=contribution_metrics(by[old],records,metrics[old])['retained_ASC_arc_length']
            for n,piece in enumerate(pieces):
                pp=assembly._points(piece);lo,hi=pp[:,1].min(),pp[:,1].max();bid=piece[0]['branch_id']
                if (lo>=required[0]-1e-6 if direction=='lower' else hi<=required[1]+1e-6):continue
                zgap=max(0.,required[0]-hi) if direction=='lower' else max(0.,lo-required[1])
                if zgap>.010000000001 or not contribution_metrics(by[bid],piece,metrics[bid])['contribution_pass']:continue
                target=(lo,required[0]) if direction=='lower' else (required[1],hi)
                if branch_metrics(by[bid],scale,target_z=target)['target_region_in_ASC_fraction']<=0:continue
                left,right=(piece,terminal) if direction=='lower' else (terminal,piece)
                proposed=assembly._nearest_join(left,right,required,lock,direction)
                if proposed is None:continue
                extended=assembly._splice(left,right,proposed);sep=next(i for i,e in enumerate(extended) if e['source']=='TOPOLOGY_SWITCH');c=contribution_metrics(by[bid],extended[:sep] if direction=='lower' else extended[sep+1:],metrics[bid]);new=extended+locked if direction=='lower' else locked+extended
                used={e['branch_id'] for e in new if e['source'].startswith('OBSERVED')};core_after=contribution_metrics(by[old],new,metrics[old])['retained_ASC_arc_length']
                passes=c['contribution_pass'] and connector_gate(proposed['xyz_distance_m'],c['observed_new_length_m'])['accepted'] and all(contribution_metrics(by[i],new,metrics[i])['contribution_pass'] for i in used) and all(v['accepted'] for v in route_budgets(new,by,metrics))
                if passes and core_before-core_after>1e-7:
                    static=assembly._nearest_join(left,right,required,lock,direction,terminal_core_bounds=bounds)
                    counter.append(dict(s=key,direction=direction,terminal_branch=old,candidate=bid,piece_index=n,
                        removed_static_ASC_m=core_before-core_after,previous_rule_gap_m=proposed['xyz_distance_m'],static_rule_gap_m=static['xyz_distance_m'] if static else None,
                        junction=proposed,scope='single-junction causal control only; not permission to trim ASC or a completed route'))
        rows.append(dict(s=key,before_ASC_z_deficit_m=before[key],after_ASC_z_deficit_m=after[key],
            previous_rule_feasible_but_static_ASC_blocks=bool(counter),counterfactual_junctions=len(counter)))
        details.extend(counter)
    # Pick separated locations, rather than three almost identical adjacent V.
    reps=[]
    for r in sorted(rows,key=lambda x:-x['after_ASC_z_deficit_m']):
        if r['previous_rule_feasible_but_static_ASC_blocks'] and all(abs(float(r['s'])-float(s))>=5 for s in reps):reps.append(r['s'])
        if len(reps)==3:break
    plot_regressions(out,reps,details,inventory)
    summary=dict(previous_missing=len(oldkeys),current_missing=len(newkeys),new_missing=len(newkeys-oldkeys),cleared_missing=len(oldkeys-newkeys),
        newly_ge_1m=len(newly),cleared_ge_1m=sum(v>=1 and after[s]<1 for s,v in before.items()),
        new_large_with_static_ASC_counterfactual=sum(r['previous_rule_feasible_but_static_ASC_blocks'] for r in rows),
        new_large_not_explained_by_this_single_junction_control=sum(not r['previous_rule_feasible_but_static_ASC_blocks'] for r in rows),representatives=reps,
        source_code_or_routes_changed=False)
    csv_rows(out/'new_ASC_missing_regression_causes.csv',rows);save_json(out/'static_ASC_regression_details.json',details);save_json(out/'static_ASC_regression_summary.json',summary)
    print(summary,flush=True)


def plot_regressions(out,reps,details,inventory):
    for key in reps:
        proposal=next(r for r in details if r['s']==key);old=pickle.load((BASELINE/'routes'/f'{key}.pkl').open('rb'))['result'];new=pickle.load((out/'routes'/f'{key}.pkl').open('rb'))['result'];u,z=proposal['junction']['a_point_uz']
        endpoint=np.asarray(new['path_edges'][-1]['points_uz'][1] if proposal['direction']=='upper' else new['path_edges'][0]['points_uz'][0])
        bounds=[min(u-1.5,endpoint[0]-.5),max(u+1.5,endpoint[0]+.5),min(z-1.3,endpoint[1]-.5),max(z+1.3,endpoint[1]+.5)]
        fig,axs=plt.subplots(1,2,figsize=(13,8),sharex=True,sharey=True)
        for ax,result,color,title in zip(axs,[old,new],['#b7773e','#147d69'],['上一轮主轨','本轮静态 ASC 对照：主轨新增缺失']):
            for b in inventory[key]:plot_records(ax,b['records'],bounds,'#c6cbd2',1.)
            plot_records(ax,result['path_edges'],bounds,color,2.4);ax.scatter(u,z,c='#9945aa',s=25,zorder=4)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_xlabel('径向偏移 u（m）');ax.set_title(title);ax.grid(alpha=.17);ax.ticklabel_format(useOffset=False,style='plain')
        axs[1].scatter(*endpoint,c='#147d69',s=35,zorder=5)
        axs[1].annotate(f'本轮主轨在这里停止\nz={endpoint[1]:.3f} m',endpoint,xytext=(15,-27),textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#555'},fontsize=10)
        axs[0].set_ylabel('高程 z（m）');fig.suptitle(f"s={key} m｜静态核心保护导致新增缺失的局部证据",fontsize=16)
        fig.text(.5,.045,f"紫点：旧规则可用接点；若采用会移除静态 ASC {proposal['removed_static_ASC_m']:.3f} m。本图只解释退步，不批准裁核心。",ha='center')
        fig.text(.5,.02,'灰=原始候选，棕/绿=各轮实际主轨；图框截断不是原始终点。',ha='center')
        fig.subplots_adjust(left=.075,right=.98,top=.89,bottom=.115,wspace=.12);fig.savefig(out/'figures'/f'{key}_static_ASC_regression.png',dpi=160);plt.close(fig)
if __name__=='__main__':
    out=latest()
    if '--figures-only' in sys.argv:
        summary=json.loads((out/'static_ASC_regression_summary.json').read_text());details=json.loads((out/'static_ASC_regression_details.json').read_text())
        plot_regressions(out,summary['representatives'],details,load_inventory(SOURCE,summary['representatives']))
    else:run(out)
