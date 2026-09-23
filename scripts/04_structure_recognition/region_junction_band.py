"""Apply region-wide junction DP using only legal original-edge contacts."""
from collections import defaultdict
import time
import numpy as np
from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale,route_budgets
from region_joint_selection import choose_junction_band


def legal_representatives(candidates,valid):
    bins=defaultdict(list)
    for j in candidates:bins[int(np.floor(float(j['a_point_uz'][1])/.05))].append(j)
    result=[]
    for members in bins.values():
        for j in sorted(members,key=lambda x:x['xyz_distance_m']):
            if valid(j):result.append(j);break
    return result


def _expand(run,branch,left_side):
    from main_track_assembly import _oriented
    from dominant_observed_branch import trim_record
    full=_oriented(branch['records']);edge=run[0] if left_side else run[-1]
    index=next(i for i,r in enumerate(full) if r['edge_id']==edge['edge_id']);r=full[index]
    t=(edge['t0' if left_side else 't1']-r['t0'])/(r['t1']-r['t0'])
    return [trim_record(r,t,1.)]+full[index+1:] if left_side else full[:index]+[trim_record(r,0.,t)]


def _valid(records,by,metrics,extent,policy):
    from reviewed_model_gap import is_estimated_gap,valid_estimated_gap
    for r in records:
        if is_estimated_gap(r):
            if not valid_estimated_gap(r):return False
            continue
        if not r['source'].startswith('OBSERVED') and np.linalg.norm(np.diff(r['points_xyz'],axis=0))>r.get('distance_cap_m',policy.p1_handoff_cap_m)+1e-10:return False
    observed=[r for r in records if r['source'].startswith('OBSERVED')]
    intervals=defaultdict(list)
    for r in observed:intervals[r['edge_id']].append(sorted([r['t0'],r['t1']]))
    if any(any(b[0]<a[1]-1e-8 for a,b in zip(sorted(parts),sorted(parts)[1:])) for parts in intervals.values() if len(parts)>1):return False
    zs=[p[1] for r in records for p in r['points_uz']]
    if min(zs)>extent[0]+1e-6 or max(zs)<extent[1]-1e-6:return False
    return all(b['accepted'] for b in route_budgets(records,by,metrics,policy,require_core=False))


def apply_region_junction_bands(routes,profiles,context,regions,policy=None):
    from competitive_surface_selection import reliable_junction
    from main_track_assembly import _splice
    from dominant_observed_branch import route_result
    policy=policy or BranchPolicy();groups=defaultdict(list);audit=[]
    for region in regions:
        data=context.region_data(region);bounds=region['bounds']
        for key,route in routes.items():
            if time.perf_counter()>getattr(context,'deadline',float('inf')):raise TimeoutError('Joint-band local runtime budget exceeded')
            s=float(key)
            if not bounds[0]-1e-8<=s<=bounds[1]+1e-8:continue
            branches=profiles[key];by={b['branch_id']:b for b in branches}
            metrics={i:branch_metrics(b,local_edge_scale(branches),policy) for i,b in by.items()}
            records=route['path_edges']
            for k,c in enumerate(records):
                if c['source']!='TOPOLOGY_SWITCH' or c.get('handoff_kind')=='MODEL_GAP_REPAIR' or k==0 or k==len(records)-1:continue
                a,b=records[k-1],records[k+1]
                if not a['source'].startswith('OBSERVED') or not b['source'].startswith('OBSERVED'):continue
                aid,bid=a['branch_id'],b['branch_id']
                if aid==bid:continue
                locked_ids=[bid for bid in (aid,bid) if bid in getattr(context,'region_locks',{}).get(s,{})]
                if locked_ids:
                    audit.append(dict(region_id=region['region_id'],s=key,from_branch=aid,to_branch=bid,
                        applied=False,reason='SKIPPED_TRANSITION_TOUCHES_REGION_LOCK',locked_branch_ids=locked_ids,
                        old_z=float(np.mean(c['points_uz'],axis=0)[1])))
                    continue
                # Both original branches must enter the conflict volume. A
                # previous off-center junction may lie outside its small cell.
                if (s,aid) not in data['branches'] or (s,bid) not in data['branches']:continue
                at=data['branches'][s,aid];bt=data['branches'][s,bid]
                if len(at)!=1 or len(bt)!=1:continue
                start=k-1;end=k+2
                while start and records[start-1]['source'].startswith('OBSERVED') and records[start-1]['branch_id']==aid:start-=1
                while end<len(records) and records[end]['source'].startswith('OBSERVED') and records[end]['branch_id']==bid:end+=1
                left=_expand(records[start:k],by[aid],True);right=_expand(records[k+1:end],by[bid],False)
                fixed=records[:start]+records[end:];zs=[p[1] for r in fixed for p in r['points_uz']]
                original=next((j for j in route.get('junctions',[]) if j.get('from_branch_id')==aid and j.get('to_branch_id')==bid
                    and abs(j['a_point_uz'][1]-float(np.mean(c['points_uz'],axis=0)[1]))<.003),{})
                direction=original.get('frontier','upper')
                ordered=(right,left,by[bid],by[aid],metrics[bid],metrics[aid]) if direction=='lower' else (left,right,by[aid],by[bid],metrics[aid],metrics[bid])
                candidates,_=reliable_junction(*ordered,
                    required=route['route_z_extent'],locked_range=(min(zs),max(zs)) if zs else (np.inf,-np.inf),
                    direction=direction,policy=policy,preserve_extent=False,cache=context.junction_cache,
                    all_candidates=True,junction_z_bounds=(bounds[4],bounds[5]),protected_arcs=getattr(context,'region_locks',{}).get(s,{}))
                candidates=candidates or []
                item=dict(region_id=region['region_id'],s=key,from_branch=aid,to_branch=bid,
                    from_FaceTrack=at[0],to_FaceTrack=bt[0],old_z=float(np.mean(c['points_uz'],axis=0)[1]),
                    candidates=candidates,raw_candidate_count=len(candidates))
                # Preserve continuous contact intervals for terminal-first search.
                representatives=candidates
                legal=[]
                from physical_continuation import future_switch_estimate
                future=future_switch_estimate([b for b in branches if metrics[b['branch_id']]['MBG_pass']],bid,context,s,policy)
                for j in representatives:
                    legal.append(dict(j,candidate_id=f"{key}:{j['a_edge_id']}:{j['a_t']:.17g}:{j['b_edge_id']}:{j['b_t']:.17g}",
                        FaceTrack=f"{region['region_id']}:T{bt[0]}",future_switches=future,junction=j))
                item['legal_candidates']=legal;item['legal_candidate_count']=len(legal);audit.append(item)
                context.scope_audit.append(dict(decision_mode='COMPETITIVE_SELECTION',competitor_count=2,
                    competing_branch_ids=[aid,bid],competing_FaceTracks=[f"{region['region_id']}:T{t}" for t in set(at+bt)],
                    competitive_core_invoked=True,junction_reason='REGION_BAND_CANDIDATES',s=key,region_id=region['region_id']))
                groups[region['region_id'],at[0],bt[0]].append((s,key,start,end,left,right,item))
    # No transition is silently made across an absent V or multiple occurrences
    # of the same transition on one V; such runs are split and left auditable.
    for group,items in groups.items():
        region=next(r for r in regions if r['region_id']==group[0]);data=context.region_data(region)
        present={item[1] for item in items}
        for key in routes:
            s=float(key)
            if not region['bounds'][0]-1e-8<=s<=region['bounds'][1]+1e-8 or key in present:continue
            tracks={t for (ss,bid),tt in data['branches'].items() if ss==s for t in tt}
            if group[1] not in tracks or group[2] not in tracks:continue
            missing=dict(region_id=group[0],s=key,from_branch=None,to_branch=None,from_FaceTrack=group[1],to_FaceTrack=group[2],
                candidates=[],legal_candidates=[],raw_candidate_count=0,legal_candidate_count=0,old_z=None,
                applied=False,reason='MISSING_NEIGHBOR_TRANSITION')
            items.append((s,key,0,0,[],[],missing));audit.append(missing)
        if len({item[0] for item in items})!=len(items):
            for item in items:item[-1].update(applied=False,reason='MULTIPLE_TRANSITIONS_REQUIRE_JOINT_ORDER')
            continue
        items.sort(key=lambda x:(x[0],x[2]));runs=[];current=[]
        for item in items:
            if current and item[0]-current[-1][0]>.051:runs.append(current);current=[]
            current.append(item)
        if current:runs.append(current)
        for run in runs:
            from terminal_junction_consensus import choose_terminal_band
            validation_cache={}
            def valid(index,pick):
                _,key,start,end,left,right,item=run[index]
                cache_key=(index,pick['a_edge_index'],pick['b_edge_index'],pick['a_t'],pick['b_t'])
                if cache_key not in validation_cache:
                    by={b['branch_id']:b for b in profiles[key]};scale=local_edge_scale(profiles[key])
                    metrics={bid:branch_metrics(b,scale,policy) for bid,b in by.items()}
                    rr=routes[key];records=rr['path_edges']
                    validation_cache[cache_key]=_valid(records[:start]+_splice(left,right,pick)+records[end:],by,metrics,rr['route_z_extent'],policy)
                return validation_cache[cache_key]
            def budget_breakpoints(index,pick):
                from terminal_junction_consensus import _materialize
                _,key,start,end,left,right,item=run[index]
                by={b['branch_id']:b for b in profiles[key]};scale=local_edge_scale(profiles[key])
                metrics={bid:branch_metrics(b,scale,policy) for bid,b in by.items()}
                records=routes[key]['path_edges'];lo,hi=pick['corridor']['t_range']
                samples=[];distance=[]
                for t in (lo,(lo+hi)/2,hi):
                    point=_materialize(pick,t);distance.append(point['xyz_distance_m'])
                    samples.append(route_budgets(records[:start]+_splice(left,right,point)+records[end:],by,metrics,policy,require_core=False))
                if len({len(x) for x in samples})!=1:return []
                roots=[]
                # In one original segment pair, contribution is affine and
                # squared connector distance is quadratic. Include every ratio
                # boundary, so even a narrow internal feasible interval survives.
                for index_budget in range(len(samples[0])):
                    values=[]
                    for rows,gap in zip(samples,distance):
                        b=rows[index_budget];allow=policy.synthetic_ratio_limit*b['observed_new_length_m']-(b['incident_synthetic_m']-gap)
                        values.append(gap*gap-allow*allow)
                    coeff=np.polynomial.polynomial.polyfit([0.,.5,1.],values,2)
                    for x in np.polynomial.polynomial.polyroots(coeff):
                        if abs(x.imag)<1e-8 and 0<x.real<1:roots.append(lo+float(x.real)*(hi-lo))
                return roots
            valid.breakpoints=budget_breakpoints
            selected=choose_terminal_band([item[-1]['legal_candidates'] for item in run],
                max_z_span=policy.p1_band_z_span_m,max_distance=policy.p1_handoff_cap_m,valid=valid)
            for item,pick in zip(run,selected):
                pick['junction']={k:v for k,v in pick.items() if k!='junction'};item[-1]['selected']=pick
                item[-1]['band_z_span_m']=float(np.ptp([p[side+'_point_uz'][1] for p in selected for side in ('a','b')]))
                item[-1]['band_profiles']=[r[1] for r in run]
            if not selected:
                for item in run:
                    item[-1]['applied']=False
                    item[-1]['band_failure_reason']='NO_COMMON_TERMINAL_BAND_WITHIN_100MM'
                    item[-1].setdefault('reason','NO_COMMON_TERMINAL_BAND_WITHIN_100MM')
    # Rebuild each profile once, backwards by original run positions. Overlap
    # means two proposed changes compete for the same run; keep it explicit.
    edits=defaultdict(list)
    for items in groups.values():
        for item in items:
            if 'selected' in item[-1]:edits[item[1]].append(item)
    _apply_atomic_edits(routes,profiles,edits,policy)
    return audit


def _apply_atomic_edits(routes,profiles,edits,policy):
    from copy import deepcopy
    original=deepcopy(routes);items=[row[-1] for rows in edits.values() for row in rows]
    _apply_selected_edits(routes,profiles,edits,policy)
    if any(not item.get('applied') for item in items):
        routes.clear();routes.update(original)
        for item in items:item.update(applied=False,reason=item.get('reason','ATOMIC_BAND_COMBINATION_REJECTED'))


def _apply_selected_edits(routes,profiles,edits,policy):
    from main_track_assembly import _splice
    from dominant_observed_branch import route_result
    for key,items in edits.items():
        route=routes[key];records=list(route['path_edges']);applied=[]
        by={b['branch_id']:b for b in profiles[key]};metrics={i:branch_metrics(b,local_edge_scale(profiles[key]),policy) for i,b in by.items()}
        for s,_,start,end,left,right,item in sorted(items,key=lambda x:x[2],reverse=True):
            item.pop('reason',None)
            matches=[k for k,c in enumerate(records) if c['source']=='TOPOLOGY_SWITCH' and 0<k<len(records)-1
                and records[k-1].get('branch_id')==item['from_branch'] and records[k+1].get('branch_id')==item['to_branch']
                and abs(float(np.mean(c['points_uz'],axis=0)[1])-item['old_z'])<1e-8]
            if len(matches)!=1:item.update(applied=False,reason='AMBIGUOUS_CURRENT_TRANSITION');continue
            k=matches[0];start=k-1;end=k+2;aid,bid=item['from_branch'],item['to_branch']
            while start and records[start-1]['source'].startswith('OBSERVED') and records[start-1]['branch_id']==aid:start-=1
            while end<len(records) and records[end]['source'].startswith('OBSERVED') and records[end]['branch_id']==bid:end+=1
            current_left=_expand(records[start:k],by[aid],True);current_right=_expand(records[k+1:end],by[bid],False)
            j=dict(item['selected']['junction'],handoff_kind='REGION_JOINT_HANDOFF',synthetic=True,
                decision_mode='COMPETITIVE_SELECTION',distance_cap_m=policy.p1_handoff_cap_m,
                region_id=item['region_id'],from_FaceTrack=f"{item['region_id']}:T{item['from_FaceTrack']}",
                to_FaceTrack=f"{item['region_id']}:T{item['to_FaceTrack']}")
            compatible=True
            for side,old,new in [('a',left,current_left),('b',right,current_right)]:
                original=old[j[side+'_edge_index']];t=original['t0']+j[side+'_t']*(original['t1']-original['t0'])
                candidates=[(i,r) for i,r in enumerate(new) if r['edge_id']==original['edge_id'] and
                    min(r['t0'],r['t1'])-1e-9<=t<=max(r['t0'],r['t1'])+1e-9 and abs(r['t1']-r['t0'])>1e-15]
                if len(candidates)!=1:compatible=False;break
                i,r=candidates[0];j[side+'_edge_index']=i;j[side+'_t']=float(np.clip((t-r['t0'])/(r['t1']-r['t0']),0,1))
            if not compatible:item.update(applied=False,reason='JOINT_SOURCE_PARAMETER_ORDER_CONFLICT');continue
            proposal=records[:start]+_splice(current_left,current_right,j)+records[end:]
            if not _valid(proposal,by,metrics,route['route_z_extent'],policy):
                item.update(applied=False,reason='COMBINED_BAND_CONTRIBUTION_BUDGET');continue
            records=proposal;applied.append(j);item['applied']=True
        if not applied:continue
        # Retain only diagnostics for connectors that remain in final geometry.
        positions=[np.asarray(r['points_xyz']) for r in records if r['source']=='TOPOLOGY_SWITCH']
        joins=[]
        for p in positions:
            matches=[j for j in applied+route.get('junctions',[]) if np.allclose([j['a_point_xyz'],j['b_point_xyz']],p,atol=1e-9,rtol=0)]
            if matches:joins.append(matches[0])
        sequence=[]
        for r in records:
            if r['source'].startswith('OBSERVED') and (not sequence or sequence[-1]!=r['branch_id']):sequence.append(r['branch_id'])
        route.update(route_result(records,branch_switch_count=len(sequence)-1,junction=joins[0] if joins else None))
        route.update(junctions=joins,junction_count=len(joins),route_branch_sequence=sequence,
            route_contribution_budgets=route_budgets(records,by,metrics,policy,require_core=False),
            region_joint_applied=len(applied),virtual_junction_count=sum(j['new_virtual_nodes'] for j in joins))


def replay_region_junction_application(routes,profiles,audit,policy=None):
    """Replay frozen candidate choices on their original pre-band P1 routes."""
    edits=defaultdict(list);policy=policy or BranchPolicy()
    for item in audit:
        key=item['s']
        if key not in routes or 'selected' not in item:continue
        records=routes[key]['path_edges'];aid,bid=item['from_branch'],item['to_branch']
        matches=[k for k,c in enumerate(records) if c['source']=='TOPOLOGY_SWITCH' and 0<k<len(records)-1 and
            records[k-1].get('branch_id')==aid and records[k+1].get('branch_id')==bid and
            abs(float(np.mean(c['points_uz'],axis=0)[1])-item['old_z'])<1e-8]
        if len(matches)!=1:raise ValueError('Frozen junction choice does not match its pre-band route')
        k=matches[0];start=k-1;end=k+2;by={b['branch_id']:b for b in profiles[key]}
        while start and records[start-1]['source'].startswith('OBSERVED') and records[start-1]['branch_id']==aid:start-=1
        while end<len(records) and records[end]['source'].startswith('OBSERVED') and records[end]['branch_id']==bid:end+=1
        left=_expand(records[start:k],by[aid],True);right=_expand(records[k+1:end],by[bid],False)
        edits[key].append((float(key),key,start,end,left,right,item))
    _apply_atomic_edits(routes,profiles,edits,policy)
    return audit
