"""Lossless ENTRY->EXIT main spine and observed side-component roles.

All removed intervals remain reconstructable. Near-return geometry identifies
an alternative, not geological noise; topology-unknown alternatives stay on
the spine and are explicitly ambiguous.
"""
from __future__ import annotations
import numpy as np
from branch_absolute_core import BranchPolicy, branch_metrics, local_edge_scale, route_budgets
from observed_component_geometry import near_returns, _arc_records


def _length(records):
    return sum(float(np.linalg.norm(np.diff(r['points_xyz'],axis=0))) for r in records)


def _endpoint_distance(point,records):
    if not records:return float('inf')
    xyz=np.asarray([r['points_xyz'] for r in records]);v=xyz[:,1]-xyz[:,0]
    t=np.clip(np.sum((point-xyz[:,0])*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-30),0,1)
    return float(np.linalg.norm(xyz[:,0]+t[:,None]*v-point,axis=1).min())


def _endpoint_matches(point,records,limit):
    if not records:return []
    xyz=np.asarray([r['points_xyz'] for r in records]);v=xyz[:,1]-xyz[:,0]
    t=np.clip(np.sum((point-xyz[:,0])*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-30),0,1)
    distance=np.linalg.norm(xyz[:,0]+t[:,None]*v-point,axis=1)
    return [records[i] for i in np.flatnonzero(distance<=limit+1e-12)]


def classify_components(branches,route,*,context=None,target_s=None,policy=None):
    from main_track_assembly import _remaining
    policy=policy or BranchPolicy();by={b['branch_id']:b for b in branches}
    scale=local_edge_scale(branches);metrics={i:branch_metrics(b,scale,policy) for i,b in by.items()}
    main=[dict(r) for r in route['path_edges']];components=[];peel_audit=[]
    # Each current observed run has fixed ENTRY/EXIT. Replacing a local return
    # never changes those endpoints or joins unrelated measured surfaces.
    runs=[];start=0
    while start<len(main):
        if not main[start]['source'].startswith('OBSERVED'):start+=1;continue
        end=start+1;bid=main[start]['branch_id']
        while end<len(main) and main[end]['source'].startswith('OBSERVED') and main[end]['branch_id']==bid:end+=1
        runs.append((start,end,bid));start=end
    for start,end,bid in reversed(runs):
        records=main[start:end];arc=np.r_[0.,np.cumsum([_length([r]) for r in records])]
        branch=dict(records=records,arc_positions=arc)
        events=near_returns(branch,max_gap_m=policy.connector_cap_m,min_arc_m=.5,min_ratio=50)
        chosen=[]
        for e in sorted(events,key=lambda e:(-e['observed_arc_m'],e['D_xyz_m'])):
            if e['start_arc']<=1e-9 or e['end_arc']>=arc[-1]-1e-9:continue
            if any(e['start_arc']<q['end_arc'] and e['end_arc']>q['start_arc'] for q in chosen):continue
            same=False
            if context is not None:
                z=float((e['a_uz'][1]+e['b_uz'][1])/2)
                # The full branch can cross many unrelated cells at this Z.
                # Both return endpoints must use one window at the return itself.
                probe=dict(by[bid],u_range=sorted([e['a_uz'][0],e['b_uz'][0]]))
                region=context._region(float(target_s),probe,z);labels=context.region_data(region)['face_labels']
                e['identity_region']=region['region_id']
                sources=[{labels[f] for f in records[k]['source_face_ids'] if f in labels}
                         for k in (e['a_index'],e['b_index'])]
                same=len(sources[0])==1 and sources[0]==sources[1]
            if not same:
                peel_audit.append(dict(branch_id=bid,role='AMBIGUOUS_COMPONENT',reason='RETURN_IDENTITY_NOT_PROVEN',**e));continue
            chosen.append(e)
        if not chosen:continue
        candidate=[];cursor=0.;sides=[]
        for e in sorted(chosen,key=lambda e:e['start_arc']):
            candidate.extend(_arc_records(branch,cursor,e['start_arc']))
            sides.append(dict(role='LOCAL_ALTERNATIVE_PATH',branch_id=bid,
                records=_arc_records(branch,e['start_arc'],e['end_arc']),entry_xyz=e['a_xyz'],exit_xyz=e['b_xyz'],
                reason='SAME_FACETRACK_LOCAL_ENTRY_EXIT_RETURN',needs_manual_review=True))
            candidate.append(dict(source='TOPOLOGY_SWITCH',branch_id=None,face_id=None,source_face_ids=[],
                source_segment_indices=[],points_xyz=[e['a_xyz'],e['b_xyz']],points_uz=[e['a_uz'],e['b_uz']],
                handoff_kind='SIDE_COMPONENT_ENTRY_EXIT',distance_m=e['D_xyz_m']))
            cursor=e['end_arc']
        candidate.extend(_arc_records(branch,cursor,float(arc[-1])))
        proposed=main[:start]+candidate+main[end:]
        budgets=route_budgets(proposed,by,metrics,policy)
        if all(x['accepted'] for x in budgets):
            main=proposed;components.extend(sides)
        else:peel_audit.append(dict(branch_id=bid,role='AMBIGUOUS_COMPONENT',reason='ENTRY_EXIT_SYNTHETIC_BUDGET'))
    peeled=[e for c in components for e in c['records']]
    selected=[r for r in main if r['source'].startswith('OBSERVED')]
    for piece in _remaining(branches,main+peeled):
        bid=piece[0]['branch_id'];b=by[bid]
        if b['kind']=='CLOSED_COMPONENT':role='SIDE_CLOSED_COMPONENT';reason='PHYSICAL_CLOSED_COMPONENT'
        else:
            ends=[np.asarray(piece[0]['points_xyz'][0]),np.asarray(piece[-1]['points_xyz'][1])]
            matches=[_endpoint_matches(p,selected,policy.connector_cap_m) for p in ends]
            # Distance counts contacts, never proves FaceTrack membership.
            supported=[False,False]
            if context is not None:
                for i,(endpoint,point,near) in enumerate(zip([piece[0],piece[-1]],
                        [piece[0]['points_uz'][0],piece[-1]['points_uz'][1]],matches)):
                    if not near:continue
                    # Identity belongs to this actual contact, not the midpoint
                    # or the bounding box of a possibly very long side branch.
                    probe=dict(b,u_range=[point[0],point[0]])
                    region=context._region(float(target_s),probe,float(point[1]))
                    labels=context.region_data(region)['face_labels']
                    mine={labels[f] for f in endpoint['source_face_ids'] if f in labels}
                    active={labels[f] for r in near for f in r['source_face_ids'] if f in labels}
                    supported[i]=len(mine)==1 and mine.issubset(active)
            role=('LOCAL_ALTERNATIVE_PATH' if all(supported) else 'SIDE_OPEN_BRANCH') if any(supported) else 'AMBIGUOUS_COMPONENT'
            reason='ENTRY_EXIT_CONTACTS' if any(supported) else 'UNSELECTED_IDENTITY_REQUIRES_REVIEW'
        components.append(dict(role=role,branch_id=bid,records=piece,reason=reason,needs_manual_review=role!='SIDE_CLOSED_COMPONENT'))
    side=[dict(r,component_role=c['role']) for c in components for r in c['records']]
    # Compare original source intervals, not rounded coordinates or record count.
    coverage={r['edge_id']:[] for b in branches for r in b['records']}
    for r in main+side:
        if r['source'].startswith('OBSERVED'):coverage[r['edge_id']].append(sorted((r['t0'],r['t1'])))
    for eid,intervals in coverage.items():
        cursor=0.
        for a,b in sorted(intervals):
            if abs(a-cursor)>1e-7:raise AssertionError(('source loss/duplication',eid,cursor,a))
            cursor=b
        if abs(cursor-1.)>1e-7:raise AssertionError(('incomplete source interval',eid,cursor))
    gaps=[np.linalg.norm(np.asarray(a['points_xyz'][1])-b['points_xyz'][0]) for a,b in zip(main,main[1:])]
    if max(gaps,default=0.)>2e-8:raise AssertionError('Discontinuous main spine')
    return dict(MAIN_SPINE=[dict(r,component_role='THROUGH_PATH') for r in main],SIDE_COMPONENTS=side,
        components=[dict(role='THROUGH_PATH',records=main,reason='SELECTED_LOCAL_ENTRY_EXIT_CONTINUATION')]+components,
        component_audit=peel_audit,source_intervals_preserved=True,
        main_observed_arc_m=_length([r for r in main if r['source'].startswith('OBSERVED')]),side_observed_arc_m=_length(side),
        main_continuous=True,reconstruction_interface='reconstruction_inputs(layers, hanging_segments)')


def reconstruction_inputs(layers,hanging_segments):
    """Rejoin preserved observed side layers with recognized hanging segments.

The consumer receives source records, not a welded/averaged reconstruction.
"""
    return dict(main_spine=layers['MAIN_SPINE'],hanging_segments=list(hanging_segments),
        side_components=layers['SIDE_COMPONENTS'],source_intervals_preserved=layers['source_intervals_preserved'])
