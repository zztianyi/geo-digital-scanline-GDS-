"""Mode-first production continuation; CC cannot veto a unique same-chain gap."""
import numpy as np
from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale,route_budgets,connector_gate
from continuation_modes import decision_scope,noncompetitive_junction


def current_is_competitor(a,b,*,same_chain):
    overlap=min(a['z_range'][1],b['z_range'][1])-max(a['z_range'][0],b['z_range'][0])
    return a['branch_id']!=b['branch_id'] and overlap>1e-8 and not same_chain


def select_physical_seed(rows,context,s,policy=None,current_branch_id=None):
    """Choose a supported anchor, then use CC only for co-located competitors."""
    from competitive_surface_selection import choose_successor
    from region_joint_selection import prepare_region_identity
    from horizontal_surface_link import branch_crossings
    policy=policy or BranchPolicy();eligible=[r for r in rows if r['MBG_pass']]
    if not eligible:return dict(selected=None,candidates=rows,ambiguous=False,decision_stage='MBG',detail_stage_comparisons=0)
    locks=getattr(context,'region_locks',{}).get(s,{})
    committed=[r for r in eligible if r['branch_id'] in locks]
    if committed:
        chosen=dict(max(committed,key=lambda r:r['fragment']['full_arc_length']),selected=True,reason='REGION_THROUGH_TRACK')
        return dict(selected=chosen,candidates=rows,ambiguous=False,comparison_ambiguous=False,
            decision_stage='REGION_THROUGH_TRACK',detail_stage_comparisons=0,region_commitments=getattr(context,'region_lock_regions',{}).get(s,{}))
    best=max(r.get('support_s_span',0.) for r in eligible)
    tied=[r for r in eligible if r.get('support_s_span',0.)>=best-max(.025,.1*best)]
    anchor=min(tied,key=lambda r:r['branch_id']);b=anchor['fragment'];z=float(np.median(b['points_uz'][:,1]))
    region=context._region(s,b,z);competing=[]
    for row in eligible:
        hits=[h for h in branch_crossings(row['fragment'],z) if region['bounds'][2]<=h['u']<=region['bounds'][3]]
        if hits:competing.append(dict(row,**context.evidence(s,row['fragment'],z,region=region),hits=hits))
    if len(competing)<2:
        anchor=dict(anchor,selected=True,reason='OBSERVED_PERSISTENCE_ANCHOR')
        return dict(selected=anchor,candidates=rows,ambiguous=anchor.get('track_ambiguous',False),
            comparison_ambiguous=False,decision_stage='OBSERVED_PERSISTENCE_ANCHOR',detail_stage_comparisons=0,
            **decision_scope([anchor]))
    regional=prepare_region_identity(context,region,policy)
    for row in competing:
        b=row['fragment'];row.update(in_ASC=any(row['ASC_start_arc']<=h['arc_position']<=row['ASC_end_arc'] for h in row.pop('hits')),
            region_priority=int(row['FaceTrack']==regional['dominant_FaceTrack']),
            forward_Z_m=float(np.ptp(b['points_uz'][:,1])),forward_ASC_m=row['ASC_arc_length'],
            minimum_switches=sum(future_switch_estimate([x['fragment'] for x in eligible],b['branch_id'],context,s,policy,d) for d in (-1,1)))
    selected=choose_successor(competing,current_branch_id=current_branch_id)
    context.scope_audit.append({**{k:selected[k] for k in ('decision_mode','competitor_count','competing_branch_ids','competing_FaceTracks','competitive_core_invoked')},
        's':s,'region_id':region['region_id'],'junction_reason':'SEED_COMPETITION'})
    selected.update(physical_seed_candidates=[{k:v for k,v in r.items() if k!='fragment'} for r in competing],
                    seed_conflict_region=region['region_id'],candidates=rows)
    return selected


def contact_identity(context,s,a,b,j):
    point=np.mean([j['a_point_uz'],j['b_point_uz']],axis=0)
    probe=dict(u_range=[point[0],point[0]])
    region=context._region(s,probe,float(point[1]));labels=context.region_data(region)['face_labels']
    x={labels[f] for f in a[j['a_edge_index']]['source_face_ids'] if f in labels}
    y={labels[f] for f in b[j['b_edge_index']]['source_face_ids'] if f in labels}
    return len(x)==1 and x==y,region


def future_switch_estimate(branches,bid,context,s,policy,direction=1):
    """Bounded observed-connectivity search; no CC or guard filtering here."""
    from main_track_assembly import _oriented
    cache=getattr(context,'future_cache',None)
    if cache is None:cache={};context.future_cache=cache
    key=(s,bid,direction)
    if key in cache:return cache[key]
    by={b['branch_id']:b for b in branches};target=(max(b['z_range'][1] for b in branches) if direction>0 else min(b['z_range'][0] for b in branches))
    queue=[(bid,0)];seen={bid};answer=3
    for current,n in queue:
        a=by[current];front=a['z_range'][1 if direction>0 else 0]
        if direction*(front-target)>=-1e-6:answer=n;break
        if n==2:continue
        for other,b in by.items():
            if other in seen or direction*(b['z_range'][1 if direction>0 else 0]-front)<=1e-6:continue
            gap=max(0.,b['z_range'][0]-front) if direction>0 else max(0.,front-b['z_range'][1])
            if gap>policy.noncompetitive_gap_cap_m:continue
            left,right=(_oriented(a['records']),_oriented(b['records'])) if direction>0 else (_oriented(b['records']),_oriented(a['records']))
            j=context.junction_cache.search(left,right,a['z_range'],(np.inf,-np.inf),'upper' if direction>0 else 'lower',
                protect_folds=False,max_distance=policy.noncompetitive_gap_cap_m,decision_mode='OBSERVED_CONNECTIVITY_LOOKAHEAD')
            if j is None:continue
            same,_=contact_identity(context,s,left,right,j)
            if not same and j['xyz_distance_m']>policy.p1_handoff_cap_m:continue
            seen.add(other);queue.append((other,n+1))
    cache[key]=answer;return answer


def assemble_physical_track(branches,initial_route,*,anchor_branch_id,graph,target_s,policy=None):
    from main_track_assembly import _oriented,_points,_remaining,_splice
    from dominant_observed_branch import route_result
    from competitive_surface_selection import choose_successor,reliable_junction
    from region_joint_selection import prepare_region_identity
    from horizontal_surface_link import branch_crossings
    policy=policy or BranchPolicy();context=graph['physical_face_context'];s=float(target_s)
    by={b['branch_id']:b for b in branches};scale=local_edge_scale(branches)
    metrics={i:branch_metrics(b,scale,policy) for i,b in by.items()};eligible=[b for i,b in by.items() if metrics[i]['MBG_pass']]
    records=_oriented(initial_route['path_edges']);joins=[];decisions=[];rejections=[];attempts=[]
    for iteration in range(2*sum(len(b['records']) for b in eligible)+1):
        required=[float(_points(records)[:,1].min()),float(_points(records)[:,1].max())]
        pieces=_remaining(eligible,records);accepted=False
        for direction in ('lower','upper'):
            terminal=[]
            for r in (records if direction=='lower' else records[::-1]):
                if not r['source'].startswith('OBSERVED') or terminal and r['branch_id']!=terminal[0]['branch_id']:break
                terminal.append(r)
            if direction=='upper':terminal.reverse()
            if not terminal:continue
            aid=terminal[0]['branch_id'];a=by[aid]
            locked=records[len(terminal):] if direction=='lower' else records[:-len(terminal)]
            zz=[p[1] for r in locked for p in r['points_uz']];locked_range=(min(zz),max(zz)) if zz else (np.inf,-np.inf)
            rows=[];raw={};scope_current=False
            for index,piece in enumerate(pieces):
                bid=piece[0]['branch_id'];p=_points(piece);zd=-1 if direction=='lower' else 1
                frontier=required[0 if zd<0 else 1]
                if zd*((p[:,1].min() if zd<0 else p[:,1].max())-frontier)<=1e-6:continue
                z_gap=max(0.,required[0]-p[:,1].max()) if zd<0 else max(0.,p[:,1].min()-required[1])
                if z_gap>policy.noncompetitive_gap_cap_m:
                    rejections.append(dict(branch_id=bid,frontier=direction,reason='LONG_GAP_UNRESOLVED',minimum_possible_gap_m=float(z_gap)));continue
                left,right=(piece,terminal) if zd<0 else (terminal,piece)
                j=context.junction_cache.search(left,right,required,locked_range,direction,protect_folds=False,
                    max_distance=policy.noncompetitive_gap_cap_m,decision_mode='HYPOTHESIS_ELIGIBILITY')
                if j is None:
                    rejections.append(dict(branch_id=bid,frontier=direction,reason='NO_GEOMETRIC_HYPOTHESIS_WITHIN_50MM'));continue
                same,region=contact_identity(context,s,left,right,j)
                if not same and j['xyz_distance_m']>policy.p1_handoff_cap_m:
                    rejections.append(dict(branch_id=bid,frontier=direction,reason='INDEPENDENT_SURFACE_EXCEEDS_P1_CAP'));continue
                face=context.evidence(s,by[bid],frontier,region=region)
                overlap=min(a['z_range'][1],by[bid]['z_range'][1])-max(a['z_range'][0],by[bid]['z_range'][0])
                # A current track overlapping the transition is a real local
                # competing hypothesis; an endpoint seam is not a second one.
                active_current=current_is_competitor(a,by[bid],same_chain=same)
                scope_current|=active_current
                row=dict(metrics[bid],**face,branch_id=bid,piece_index=index,same_chain=same,
                         competition_z=frontier,forward_Z_m=max(0.,zd*(by[bid]['z_range'][0 if zd<0 else 1]-frontier)))
                rows.append(row);raw[index]=(j,region)
            scope_rows=list(rows)
            if scope_current:scope_rows.append(dict(branch_id=aid,MBG_pass=True))
            scope=decision_scope(scope_rows);attempts.append(dict(frontier=direction,current_z_range=required,**scope))
            if not rows:continue
            if scope['competitive_core_invoked']:
                for row in rows:
                    b=by[row['branch_id']];region=raw[row['piece_index']][1]
                    regional=prepare_region_identity(context,region,policy)
                    hits=branch_crossings(b,float(np.clip(row['competition_z'],*b['z_range'])))
                    row.update(in_ASC=any(row['ASC_start_arc']<=h['arc_position']<=row['ASC_end_arc'] for h in hits),
                        forward_ASC_m=row['ASC_arc_length'],region_priority=int(row['FaceTrack']==regional['dominant_FaceTrack']),
                        minimum_switches=future_switch_estimate(eligible,b['branch_id'],context,s,policy,zd))
            pending=list(rows);rank=0
            while pending:
                decision=choose_successor(pending,scope_rows=scope_rows);target=decision['selected'];rank+=1
                pending=[r for r in pending if r['piece_index']!=target['piece_index']]
                piece=pieces[target['piece_index']];bid=target['branch_id'];left,right=(piece,terminal) if direction=='lower' else (terminal,piece)
                if scope['competitive_core_invoked']:
                    j,diag=reliable_junction(terminal,piece,a,by[bid],metrics[aid],metrics[bid],required=required,
                        locked_range=locked_range,direction=direction,policy=policy,cache=context.junction_cache,bounded=True)
                    cap=policy.p1_handoff_cap_m
                else:
                    j,diag=noncompetitive_junction(terminal,piece,required=required,locked_range=locked_range,
                        direction=direction,same_chain=target['same_chain'],competitor_count=scope['competitor_count'],policy=policy,cache=context.junction_cache)
                    cap=policy.noncompetitive_gap_cap_m
                attempt=dict(frontier=direction,branch_id=bid,piece_index=target['piece_index'],identity_rank=rank,
                    **scope,identity_candidates=rows,identity_trace=decision['trace'],competition_z=target['competition_z'],
                    feasibility_accepted=False,junction_reason=diag['junction_reason'],junction_diagnostic=diag)
                decisions.append(attempt);context.scope_audit.append(attempt)
                if j is None:
                    rejections.append(dict(frontier=direction,branch_id=bid,reason='NO_VALID_JUNCTION',junction_diagnostic=diag));continue
                j.update(scope,handoff_kind=j.get('handoff_kind','COMPETITIVE_HANDOFF'),synthetic=True,
                    identity_rank=rank,identity_ambiguous=decision['ambiguous'],region_id=raw[target['piece_index']][1]['region_id'])
                extension=_splice(left,right,j);candidate=extension+locked if direction=='lower' else locked+extension
                observed=sum(float(np.linalg.norm(np.diff(r['points_xyz'],axis=0))) for r in extension if r['source'].startswith('OBSERVED') and r['branch_id']==bid)
                gate=connector_gate(j['xyz_distance_m'],observed,policy,cap_m=cap)
                budgets=route_budgets(candidate,by,metrics,policy,require_core=False)
                if not gate['accepted'] or not all(x['accepted'] for x in budgets):
                    rejections.append(dict(frontier=direction,branch_id=bid,reason='OBSERVED_CONTRIBUTION_BUDGET'));continue
                j.update(gate);attempt['feasibility_accepted']=True;attempt['junction_reason']=j['junction_reason']
                records=candidate;joins.insert(0,j) if direction=='lower' else joins.append(j);accepted=True;break
            if accepted:break
        if not accepted:
            from reviewed_model_gap import approved_endpoint_extension
            extension=approved_endpoint_extension(records,pieces,eligible,
                getattr(context,'reviewed_model_gaps',{}).get(s,[]))
            if extension is None:break
            records,j=extension
            joins.insert(0,j) if j['frontier']=='lower' else joins.append(j)
            decisions.append(dict(frontier=j['frontier'],branch_id=j['to_branch_id'],
                feasibility_accepted=True,junction_reason=j['junction_reason'],
                decision_mode='REVIEWED_MODEL_HOLE',structure_eligible=False))
    else:raise RuntimeError('Non-progressing physical continuation')
    sequence=[]
    for r in records:
        if r['source'].startswith('OBSERVED') and (not sequence or sequence[-1]!=r['branch_id']):sequence.append(r['branch_id'])
    result=route_result(records,branch_switch_count=max(0,len(sequence)-1),junction=joins[0] if joins else None)
    actual=[float(result['curve_uz'][:,1].min()),float(result['curve_uz'][:,1].max())]
    extent=[min(b['z_range'][0] for b in eligible),max(b['z_range'][1] for b in eligible)]
    covered=actual[0]<=extent[0]+1e-6 and actual[1]>=extent[1]-1e-6
    return dict(result,junctions=joins,junction_count=len(joins),route_branch_sequence=sequence,branch_audit=list(metrics.values()),
        continuation_attempts=attempts,continuation_decisions=decisions,continuation_rejections=rejections,
        route_contribution_budgets=route_budgets(records,by,metrics,policy,require_core=False),
        route_identity_ambiguous=any(j.get('identity_ambiguous') for j in joins),internal_handoff_audit=[],
        unresolved_reasons=sorted({r['reason'] for r in rejections}|{'UNCOVERED_RELIABLE_EXTENT'}) if not covered else [],
        candidate_z_extent=extent,route_z_extent=actual,extent_covered=covered,excluded_extent_branch_ids=[],
        virtual_junction_count=sum(j['new_virtual_nodes'] for j in joins),review_gap_m=policy.p1_handoff_cap_m,
        estimated_gap_count=sum(j.get('structure_eligible') is False for j in joins),
        large_connector_count=sum(j['xyz_distance_m']>j['distance_cap_m']+1e-12 for j in joins if 'distance_cap_m' in j),
        directional_tail_count=0,internal_handoff_count=0,internal_tail_trimmed_length=0.,
        folded_branch_handoff_evaluated_count=0,folded_branch_handoff_accepted_count=0,future_switch_avoided_count=0)
