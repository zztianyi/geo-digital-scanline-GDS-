"""P1 decisions inside the current Z envelope, with two safe junctions.

A full-envelope seed is not permission to skip a local competing FaceTrack.
Local A->B->A replacements keep both outside endpoints and all source records.
"""
import numpy as np
from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale,route_budgets,connector_gate
from competitive_surface_selection import choose_successor,reliable_junction
from horizontal_surface_link import branch_crossings


def _runs(records):
    runs=[];i=0
    while i<len(records):
        if not records[i]['source'].startswith('OBSERVED'):i+=1;continue
        j=i+1;bid=records[i]['branch_id']
        while j<len(records) and records[j]['source'].startswith('OBSERVED') and records[j]['branch_id']==bid:j+=1
        runs.append((i,j,bid));i=j
    return runs


def _at_z(records,z,bounds):
    if len(records)==0:return False
    p=records if isinstance(records,np.ndarray) else np.asarray([r['points_uz'] for r in records]);dz=p[:,1,1]-p[:,0,1]
    t=np.divide(z-p[:,0,1],dz,out=np.zeros(len(p)),where=abs(dz)>1e-12)
    u=p[:,0,0]+t*(p[:,1,0]-p[:,0,0])
    return bool(np.any((z>=p[:,:,1].min(axis=1)-1e-9)&(z<=p[:,:,1].max(axis=1)+1e-9)&
        (u>=bounds[2])&(u<=bounds[3])))


def apply_internal_competitions(branches,route,graph,target_s,*,policy=None):
    from dominant_observed_branch import route_result
    from main_track_assembly import _oriented,_remaining,_splice
    from physical_continuation import future_switch_estimate
    from region_joint_selection import prepare_region_identity
    from continuation_modes import decision_scope
    context=graph['physical_face_context'];policy=policy or BranchPolicy()
    scale=local_edge_scale(branches);by={b['branch_id']:b for b in branches}
    metrics={i:branch_metrics(b,scale,policy) for i,b in by.items()}
    reliable=[b for i,b in by.items() if metrics[i]['MBG_pass']]
    records=list(route['path_edges']);audit=[];accepted=[];future={}
    if not records:return dict(route,local_competition_audit=[])
    regions={r['region_id']:r for r in context.regions if r['bounds'][0]-1e-8<=target_s<=r['bounds'][1]+1e-8}
    regions.update({k:r for k,r in context.used.items() if r['bounds'][0]-1e-8<=target_s<=r['bounds'][1]+1e-8})
    levels=np.asarray(list(graph.get('level_z',{}).values()))
    for region in sorted(regions.values(),key=lambda r:(r['bounds'][4],r['region_id'])):
        if region['region_id'] in getattr(context,'region_lock_regions',{}).get(float(target_s),{}):
            audit.append(dict(region_id=region['region_id'],accepted=False,decision='KEEP_REGION_THROUGH_TRACK'))
            continue
        bounds=region['bounds'];data=context.region_data(region)
        ids=[bid for (s,bid) in data['branches'] if s==target_s and metrics[bid]['MBG_pass']]
        if len(ids)<2:continue
        midpoint=(bounds[4]+bounds[5])/2
        positions=set(float(z) for z in levels[(levels>=bounds[4])&(levels<=bounds[5])])
        positions.add(midpoint)
        for bid in ids:
            lo=max(bounds[4],by[bid]['z_range'][0]);hi=min(bounds[5],by[bid]['z_range'][1])
            if hi>=lo:positions.add((lo+hi)/2)
        item=dict(region_id=region['region_id'],evaluated_positions=0,decision='NO_LOCAL_COMPETITION',
                  accepted=False,failures=[],identity_trace=[])
        failed_geometry=set()
        fixed_runs=[(x,np.asarray([r['points_uz'] for r in records[x[0]:x[1]]])) for x in _runs(records)]
        for z in sorted(positions,key=lambda z:(abs(z-midpoint),z)):
            current=[x for x,points in fixed_runs if _at_z(points,z,bounds)]
            if len(current)!=1:
                item['decision']='AMBIGUOUS_CURRENT_COMPONENTS' if current else 'CURRENT_SPINE_ABSENT';continue
            start,end,aid=current[0];a=by[aid];terminal=records[start:end]
            rows=[]
            for bid in ids:
                hits=[h for h in branch_crossings(by[bid],z) if bounds[2]<=h['u']<=bounds[3]]
                if not hits:continue
                m=metrics[bid];face=context.evidence(target_s,by[bid],z,region=region)
                if bid not in future:
                    future[bid]=future_switch_estimate(reliable,bid,context,target_s,policy)
                rows.append(dict(m,**face,hits=hits,forward_Z_m=max(0.,by[bid]['z_range'][1]-z),minimum_switches=future[bid]))
            if len(rows)<2 or not any(r['branch_id']==aid for r in rows):continue
            regional=prepare_region_identity(context,region,policy)
            for row in rows:
                hits=row.pop('hits')
                row.update(in_ASC=any(row['ASC_start_arc']<=h['arc_position']<=row['ASC_end_arc'] for h in hits),
                    forward_ASC_m=max(max(0.,row['ASC_end_arc']-max(h['arc_position'],row['ASC_start_arc'])) for h in hits),
                    region_priority=int(row['FaceTrack']==regional['dominant_FaceTrack']))
            item['evaluated_positions']+=1;pending=rows;rank=0
            while pending:
                decision=choose_successor(pending,current_branch_id=aid,scope_rows=rows);winner=decision['selected'];rank+=1
                scope={k:decision[k] for k in ('decision_mode','competitor_count','competing_branch_ids','competing_FaceTracks','competitive_core_invoked')}
                context.scope_audit.append(dict(scope,region_id=region['region_id'],s=target_s,z=z,junction_reason='LOCAL_COMPETITION'))
                item.update(scope)
                item.update(identity_trace=decision['trace'],competition_z=z,selected_branch=winner['branch_id'])
                if winner['branch_id']==aid:
                    item['decision']='AMBIGUOUS_KEEP_CURRENT' if decision['ambiguous'] else 'KEEP_CURRENT';break
                bid=winner['branch_id'];pending=[r for r in pending if r['branch_id']!=bid]
                b=by[bid];bm=metrics[bid];arc=np.asarray(b['arc_positions']);zs=np.asarray(b['points_uz'])[:,1]
                core_z=np.r_[np.interp([bm['ASC_start_arc'],bm['ASC_end_arc']],arc,zs),
                    zs[(arc>bm['ASC_start_arc'])&(arc<bm['ASC_end_arc'])]]
                entry_z=max(bounds[4],float(core_z.min()));exit_z=min(bounds[5],float(core_z.max()))
                if not entry_z<=z<=exit_z:continue
                # Within this window the route and reliable entry/exit bounds
                # do not change until acceptance ends the loop. Different H
                # samples still rank identities independently, but an identical
                # failed geometric search need not be repeated at every level.
                geometry_key=(start,end,aid,bid,entry_z,exit_z)
                if geometry_key in failed_geometry:
                    if len(item['failures'])<5:item['failures'].append(dict(branch_id=bid,reason='NO_SAFE_TWO_ASC_JUNCTIONS',z=z))
                    continue
                # Do not consume an already selected interval or route through a
                # disconnected fragment. All unconsumed pieces are tried.
                pieces=_remaining([by[bid]],records);replacement=None
                for piece in pieces:
                    entry,ed=reliable_junction(terminal,piece,a,by[bid],metrics[aid],metrics[bid],
                        required=route['route_z_extent'],locked_range=(np.inf,-np.inf),direction='upper',policy=policy,
                        preserve_extent=False,junction_z_bounds=(-np.inf,entry_z),cache=context.junction_cache,bounded=True,
                        protected_arcs=getattr(context,'region_locks',{}).get(float(target_s),{}))
                    if entry is None:continue
                    first=_splice(terminal,piece,entry)
                    incoming=next(i for i,r in enumerate(first) if r['source']=='TOPOLOGY_SWITCH')
                    b_retained=first[incoming+1:]
                    remaining_a=_remaining([a],records[:start]+first+records[end:])
                    # Only rejoin the actual outside suffix of this run.
                    for suffix in remaining_a:
                        if np.linalg.norm(np.asarray(suffix[-1]['points_xyz'][1])-terminal[-1]['points_xyz'][1])>1e-8:continue
                        exitj,xd=reliable_junction(b_retained,suffix,by[bid],a,metrics[bid],metrics[aid],
                            required=route['route_z_extent'],locked_range=(np.inf,-np.inf),direction='upper',policy=policy,
                            preserve_extent=False,junction_z_bounds=(exit_z,np.inf),cache=context.junction_cache,bounded=True,
                            protected_arcs=getattr(context,'region_locks',{}).get(float(target_s),{}))
                        if exitj is None:continue
                        middle=_splice(b_retained,suffix,exitj)
                        candidate=records[:start]+first[:incoming+1]+middle+records[end:]
                        budgets=route_budgets(candidate,by,metrics,policy,require_core=False)
                        if not all(x['accepted'] for x in budgets):continue
                        zz=[p[1] for r in candidate for p in r['points_uz']]
                        if min(zz)>route['route_z_extent'][0]+1e-6 or max(zz)<route['route_z_extent'][1]-1e-6:continue
                        for j,source,target in [(entry,aid,bid),(exitj,bid,aid)]:
                            target_row=next(r for r in rows if r['branch_id']==target)
                            source_row=next(r for r in rows if r['branch_id']==source)
                            observed=sum(x['observed_new_length_m'] for x in budgets if x['branch_id']==target)
                            j.update(connector_gate(j['xyz_distance_m'],observed,policy,cap_m=policy.p1_handoff_cap_m),
                                from_FaceTrack=source_row['FaceTrack'],to_FaceTrack=target_row['FaceTrack'],
                                from_branch_id=source,to_branch_id=target,handoff_kind='LOCAL_COMPETITIVE_HANDOFF',
                                identity_rank=rank,identity_ambiguous=decision['ambiguous'],**scope,
                                to_branch_forward_Z_extent=target_row['forward_Z_m'],
                                to_branch_forward_ASC_length=target_row['forward_ASC_m'],
                                to_branch_future_switches=target_row['minimum_switches'],region_id=region['region_id'])
                        replacement=candidate,entry,exitj;break
                    if replacement:break
                if replacement:
                    records,j1,j2=replacement;accepted.extend([j1,j2])
                    item.update(accepted=True,decision='ACCEPT_LOCAL_ENTRY_EXIT',from_branch=aid,to_branch=bid,
                        junctions=[j1,j2],identity_candidates=rows);break
                failed_geometry.add(geometry_key)
                if len(item['failures'])<5:item['failures'].append(dict(branch_id=bid,reason='NO_SAFE_TWO_ASC_JUNCTIONS',z=z))
            if item['accepted']:break
        audit.append(item)
    if not accepted:return dict(route,local_competition_audit=audit)
    anchor=route['selection']['selected']['branch_id'];sequence=[]
    for r in records:
        if r['source'].startswith('OBSERVED'):
            r=dict(r)  # original source records are never edited
            if not sequence or sequence[-1]!=r['branch_id']:sequence.append(r['branch_id'])
    joins=route.get('junctions',[])+accepted
    positions=[np.asarray(r['points_xyz'][0]) for r in records if r['source']=='TOPOLOGY_SWITCH']
    joins.sort(key=lambda j:min(range(len(positions)),key=lambda i:np.linalg.norm(positions[i]-j['a_point_xyz'])))
    output=dict(route,**route_result(records,branch_switch_count=len(sequence)-1,junction=joins[0]),
        junctions=joins,junction_count=len(joins),route_branch_sequence=sequence,
        local_competition_audit=audit,route_contribution_budgets=route_budgets(records,by,metrics,policy,require_core=False),
        local_competitive_handoffs=len(accepted)//2,
        route_identity_ambiguous=route.get('route_identity_ambiguous',False) or any(j.get('identity_ambiguous') for j in accepted))
    output['virtual_junction_count']=sum(j['new_virtual_nodes'] for j in joins)
    output['route_z_extent']=[float(output['curve_uz'][:,1].min()),float(output['curve_uz'][:,1].max())]
    return output
