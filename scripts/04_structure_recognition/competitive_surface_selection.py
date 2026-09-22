"""P1: select successor identity first, then minimize its reliable junction.

Minimum distance MUST NOT choose a surface identity. It is used only after
FaceTrack, local ASC, forward reach and future switches have fixed a successor.
"""
from __future__ import annotations
import numpy as np
from branch_absolute_core import BranchPolicy, branch_metrics, local_edge_scale


def choose_successor(rows, current_branch_id=None):
    rows=[dict(r) for r in rows]; contenders=[r for r in rows if r['MBG_pass']];trace=[]
    if not contenders:
        return dict(selected=None,candidates=rows,ambiguous=False,comparison_ambiguous=False,
                    decision_stage='MBG',detail_stage_comparisons=0,trace=[])
    # A sweet-core tie is not the end of identity selection. Net continuation
    # past the present frontier is compared before estimated future switches.
    for field,reverse,tol in [('face_continuity',True,0.),('in_ASC',True,0.),
                              ('forward_Z_m',True,1e-6),('forward_ASC_m',True,1e-6),
                              ('minimum_switches',False,0.)]:
        best=(max if reverse else min)(r.get(field,0) for r in contenders)
        contenders=[r for r in contenders if abs(r.get(field,0)-best)<=tol]
        trace.append(dict(stage=field,remaining=[r['branch_id'] for r in contenders]))
    selected=next((r for r in contenders if r['branch_id']==current_branch_id),None)
    selected=selected or min(contenders,key=lambda r:(r['branch_id'],r.get('piece_index',0)))
    ambiguous=len(contenders)>1 or selected.get('face_identity_ambiguous',False)
    selected.update(selected=True,reason='AMBIGUOUS' if ambiguous else 'FACETRACK_COMPETITIVE_ASC')
    return dict(selected=selected,candidates=rows,ambiguous=ambiguous,
        comparison_ambiguous=len(contenders)>1,decision_stage='FACETRACK_COMPETITIVE_ASC',
        detail_stage_comparisons=0,trace=trace)


def local_arc_positions(records,branch):
    lookup={int(eid):i for i,eid in enumerate(branch['edge_order'])};arc=np.asarray(branch['arc_positions'])
    return np.asarray([arc[lookup[r['edge_id']]]+np.asarray([r['t0'],r['t1']])*
        (arc[lookup[r['edge_id']]+1]-arc[lookup[r['edge_id']]]) for r in records])


def _clip(records,branch,interval):
    from dominant_observed_branch import trim_record
    clipped=[];mapping=[]
    for i,(record,(a,b)) in enumerate(zip(records,local_arc_positions(records,branch))):
        if abs(b-a)<1e-15:continue
        t0,t1=sorted((np.asarray(interval)-a)/(b-a));t0=max(0.,t0);t1=min(1.,t1)
        if t1-t0>1e-12 or (abs(interval[1]-interval[0])<=1e-12 and -1e-12<=t0<=t1+1e-12<=1.+2e-12):
            clipped.append(trim_record(record,t0,t1));mapping.append((i,t0,t1))
    return clipped,mapping


def reliable_junction(terminal,candidate,from_branch,to_branch,ma,mb,*,
                      required,locked_range,direction,policy=None,preserve_extent=True,junction_z_bounds=None):
    """Search all segments in dual ASC; only if absent use tail -> target ASC.

Static ASC may be trimmed. The retained minimum contribution still has to
exist on both sides. Guard nodes are defined on original branches, not clips.
"""
    from main_track_assembly import _nearest_join
    policy=policy or BranchPolicy();lower=direction=='lower'
    audit=dict(from_branch_id=from_branch['branch_id'],to_branch_id=to_branch['branch_id'],
        from_FaceTrack=None,to_FaceTrack=None,from_in_ASC_at_junction=None,
        to_in_ASC_at_junction=None,dual_ASC_overlap_exists=False,
        candidate_min_distance_m=None,accepted_junction_distance_m=None,junction_reason='UNRESOLVED')
    if not ma['MBG_pass'] or not mb['MBG_pass']:return None,audit
    aa=local_arc_positions(terminal,from_branch);bb=local_arc_positions(candidate,to_branch)
    adir=(1 if aa[-1,1]>=aa[0,0] else -1)*(-1 if lower else 1)
    bdir=(1 if bb[-1,1]>=bb[0,0] else -1)*(-1 if lower else 1)
    # These are minimum retained contributions, never a core_before/core_after lock.
    a_allowed=[ma['ASC_start_arc'],ma['ASC_end_arc']]
    b_allowed=[mb['ASC_start_arc'],mb['ASC_end_arc']]
    if adir==1:a_allowed[0]=max(a_allowed[0],aa.min())+ma['minimum_core_arc_length']
    else:a_allowed[1]=min(a_allowed[1],aa.max())-ma['minimum_core_arc_length']
    if bdir==1:b_allowed[1]=min(b_allowed[1],bb.max())-mb['minimum_core_arc_length']
    else:b_allowed[0]=max(b_allowed[0],bb.min())+mb['minimum_core_arc_length']
    if b_allowed[1]<b_allowed[0]:return None,audit
    bparts,bmap=_clip(candidate,to_branch,b_allowed)
    if not bparts:return None,audit
    acore,amap=_clip(terminal,from_branch,a_allowed) if a_allowed[1]>=a_allowed[0] else ([],[])
    def zrange(parts):
        z=[p[1] for r in parts for p in r['points_uz']];return min(z),max(z)
    def overlaps(a,b):
        return bool(a and b and min(zrange(a)[1],zrange(b)[1])>=max(zrange(a)[0],zrange(b)[0])-1e-9)
    dual=overlaps(acore,bparts);audit['dual_ASC_overlap_exists']=dual
    if dual:
        aparts=acore;kind='DUAL_ASC_MIN_DISTANCE'
    else:
        tail=(ma['ASC_end_arc'],float(from_branch['arc_positions'][-1])) if adir==1 else (0.,ma['ASC_start_arc'])
        aparts,amap=_clip(terminal,from_branch,tail);kind='TAIL_TO_ASC_MIN_DISTANCE'
    if not aparts:return None,audit
    left,right=(bparts,aparts) if lower else (aparts,bparts)
    # Include the immutable prefix/suffix outside the eligible search interval
    # while screening each pair, not only after taking an unsafe global minimum.
    lm,rm=(bmap,amap) if lower else (amap,bmap)
    original_left,original_right=(candidate,terminal) if lower else (terminal,candidate)
    fixed_z=[p[1] for r in original_left[:lm[0][0]]+original_right[rm[-1][0]+1:] for p in r['points_uz']]
    fixed_z.extend([original_left[0]['points_uz'][0][1],original_right[-1]['points_uz'][1][1]])
    full_locked=(min(locked_range[0],min(fixed_z)),max(locked_range[1],max(fixed_z)))
    j=_nearest_join(left,right,required,full_locked,direction,
                    protect_folds=False,preserve_extent=preserve_extent,prefer_early=True,
                    allow_clipped_boundaries=True,junction_z_bounds=junction_z_bounds)
    if j is None:return None,audit
    for side,mapping,original in [('a',bmap if lower else amap,candidate if lower else terminal),
                                 ('b',amap if lower else bmap,terminal if lower else candidate)]:
        i,t0,t1=mapping[j[side+'_edge_index']]
        j[side+'_edge_index']=i;j[side+'_t']=t0+j[side+'_t']*(t1-t0)
        j[side+'_edge_id']=original[i]['edge_id']
    audit['candidate_min_distance_m']=j['xyz_distance_m']
    if j['xyz_distance_m']>policy.connector_cap_m+1e-12:return None,audit
    from main_track_assembly import _splice
    records=_splice(candidate,terminal,j) if lower else _splice(terminal,candidate,j)
    zs=[p[1] for r in records for p in r['points_uz']]
    lo=min(min(zs),locked_range[0]);hi=max(max(zs),locked_range[1])
    if preserve_extent and (lo>required[0]+1e-6 or hi<required[1]-1e-6):return None,audit
    side='b' if lower else 'a';i=j[side+'_edge_index'];t=j[side+'_t']
    apos=aa[i,0]+t*(aa[i,1]-aa[i,0])
    audit.update(from_in_ASC_at_junction=bool(ma['ASC_start_arc']-1e-9<=apos<=ma['ASC_end_arc']+1e-9),
        to_in_ASC_at_junction=True,accepted_junction_distance_m=j['xyz_distance_m'],
        junction_reason='REAL_INTERSECTION' if j['xyz_distance_m']<1e-9 else kind)
    j.update(audit);return j,audit


def successor_evidence(row,branch,terminal,*,graph,target_s,direction,required,branches,policy=None):
    """Sample the current overlap location; do not use whole-window % green."""
    from directional_handoff import future_switch_cost
    from horizontal_surface_link import branch_crossings
    zd=-1 if direction=='lower' else 1
    frontier=required[0 if zd<0 else 1]
    a=graph['nodes'][(float(target_s),terminal[0]['branch_id'])]
    overlap=(max(a['z_range'][0],branch['z_range'][0]),min(a['z_range'][1],branch['z_range'][1]))
    z=float(np.clip(frontier,overlap[0],overlap[1])) if overlap[0]<=overlap[1] else frontier
    hits=branch_crossings(branch,z)
    pos=[h['arc_position'] for h in hits]
    in_core=any(row['ASC_start_arc']-1e-9<=p<=row['ASC_end_arc']+1e-9 for p in pos)
    # When the old frontier is a folded extremum, compare where both curves
    # actually coexist, not an invented endpoint-to-endpoint gap.
    extent=max(0.,zd*(branch['z_range'][0 if zd<0 else 1]-frontier))
    bd=zd*(1 if branch['points_uz'][-1,1]>=branch['points_uz'][0,1] else -1)
    remaining=max((max(0.,row['ASC_end_arc']-max(p,row['ASC_start_arc'])) if bd>0 else
                   max(0.,min(p,row['ASC_end_arc'])-row['ASC_start_arc']) for p in pos),default=0.)
    context=graph['physical_face_context']
    region=context._region(float(target_s),a,frontier)
    physical=context.evidence(float(target_s),branch,z,region=region)
    target=min(b['z_range'][0] for b in branches) if zd<0 else max(b['z_range'][1] for b in branches)
    future=future_switch_cost(branches,branch['branch_id'],z_direction=zd,target_z=target,policy=policy,reliable_entry=True)
    return dict(row,**physical,in_ASC=in_core,forward_Z_m=extent,forward_ASC_m=remaining,
                minimum_switches=future['estimated_additional_switches'],competition_z=z)
