"""Assemble an observed main path by continuing at both regional frontiers.

Track labels describe cross-slice evidence, not a whitelist of possible next
branches. Geometry remains on original edges. A shortest gap is an explicit
connector, never an observed edge or proof of surface identity.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-9


def _points(records):
    return np.asarray([records[0]['points_uz'][0]]+[r['points_uz'][1] for r in records])


def _oriented(records):
    from dominant_observed_branch import trim_record
    if records[0]['points_uz'][0][1] > records[-1]['points_uz'][1][1]:
        return [trim_record(r, 1., 0.) for r in records[::-1]]
    return [dict(r) for r in records]


def _remaining(branches, route):
    """Unconsumed intervals; the same branch can supply disjoint end pieces."""
    from dominant_observed_branch import trim_record
    used = {}
    for r in route:
        if r['source'].startswith('OBSERVED'):
            used.setdefault(r['edge_id'], []).append(sorted((r['t0'], r['t1'])))
    pieces = []
    for branch in branches:
        current = []
        for r in branch['records']:
            intervals, start = [], 0.
            for lo, hi in sorted(used.get(r['edge_id'], [])):
                if lo > start+EPS:
                    intervals.append((start, lo))
                start = max(start, hi)
            if start < 1.-EPS:
                intervals.append((start, 1.))
            if not intervals and current:
                pieces.append(_oriented(current)); current = []
            for lo, hi in intervals:
                edge = trim_record(r, lo, hi)
                if current and np.linalg.norm(np.asarray(current[-1]['points_uz'][1])-edge['points_uz'][0]) > EPS:
                    pieces.append(_oriented(current)); current = []
                current.append(edge)
        if current:
            pieces.append(_oriented(current))
    return pieces


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _terminal_core_bounds(records, branch, metrics):
    """Limits in terminal-local arc units that keep every retained static ASC edge.

    Return (largest lower cut, smallest upper cut). Endpoint guards are free
    to change, including their folds. Source t0/t1 also handles reversed and
    previously clipped records; no coordinates or canonical nodes are changed.
    """
    positions = {int(eid): i for i, eid in enumerate(branch['edge_order'])}
    arc = np.asarray(branch['arc_positions']); offset = 0.; intervals = []
    for record in records:
        length = float(np.linalg.norm(np.diff(record['points_uz'], axis=0)))
        i = positions[record['edge_id']]
        a, b = arc[i]+np.asarray([record['t0'], record['t1']])*(arc[i+1]-arc[i])
        low = max(min(a, b), metrics['ASC_start_arc'])
        high = min(max(a, b), metrics['ASC_end_arc'])
        if metrics['ASC_exists'] and high > low+EPS:
            intervals.append(sorted(offset+(np.asarray([low, high])-a)/(b-a)*length))
        offset += length
    return (min(x[0] for x in intervals), max(x[1] for x in intervals)) if intervals else (offset, 0.)


def _nearest_join(left, right, required, locked_range, direction, *, transition_arcs=None,
                  protect_folds=True, preserve_extent=True, terminal_core_bounds=None, prefer_early=False,
                  allow_clipped_boundaries=False, junction_z_bounds=None):
    """Exact segment intersections and four endpoint-to-segment projections.

Search all eligible edge interiors, in bounded-size arrays. Retained prefixes
and suffixes must preserve the old elevation extent and extend one frontier.
No monotonic-Z requirement is imposed on the measured curves.
"""
    pa, pb = _points(left), _points(right)
    xa_all=np.asarray([left[0]['points_xyz'][0]]+[r['points_xyz'][1] for r in left])
    xb_all=np.asarray([right[0]['points_xyz'][0]]+[r['points_xyz'][1] for r in right])
    da, db = np.diff(pa, axis=0), np.diff(pb, axis=0)
    la, lb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
    ca, cb = np.r_[0., np.cumsum(la)], np.r_[0., np.cumsum(lb)]
    # Only the current terminal is already accepted geometry. A continuation
    # may trim its tail, but cannot erase an accepted fold on the way there.
    folded_a, folded_b = np.flatnonzero(da[:, 1] < -EPS), np.flatnonzero(db[:, 1] < -EPS)
    protected_a = ca[folded_a[-1]+1] if protect_folds and direction == 'upper' and len(folded_a) else 0.
    protected_b = cb[folded_b[0]] if protect_folds and direction == 'lower' and len(folded_b) else cb[-1]
    # terminal_core_bounds is a legacy audit argument only. Static ASC is a
    # reliability descriptor, never a permanent no-trim boundary (v2 P0).
    amin, amax = np.minimum.accumulate(pa[:, 1]), np.maximum.accumulate(pa[:, 1])
    bmin = np.minimum.accumulate(pb[::-1, 1])[::-1]
    bmax = np.maximum.accumulate(pb[::-1, 1])[::-1]
    best = None
    for start in range(0, len(da), 64):
        end = min(start+64, len(da))
        p, r = pa[start:end, None, :], da[start:end, None, :]
        q, s = pb[None, :-1, :], db[None, :, :]
        xp,xq=xa_all[start:end,None,:],xb_all[None,:-1,:]
        xr,xs=np.diff(xa_all,axis=0)[start:end,None,:],np.diff(xb_all,axis=0)[None,:,:]
        rr,ss=np.sum(xr*xr,axis=2),np.sum(xs*xs,axis=2)
        rs=np.sum(xr*xs,axis=2);w=xp-xq
        rw,sw=np.sum(xr*w,axis=2),np.sum(xs*w,axis=2)
        denominator=rr*ss-rs*rs
        nonparallel=np.abs(denominator)>np.maximum(rr*ss*1e-14,1e-30)
        ta=np.divide(rs*sw-ss*rw,denominator,out=np.zeros_like(denominator),where=nonparallel)
        tb=np.divide(rr*sw-rs*rw,denominator,out=np.zeros_like(denominator),where=nonparallel)
        candidates = [(ta, tb, nonparallel & (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1))]
        shape = denominator.shape
        for fixed in (0., 1.):
            on_b = np.clip(np.sum((xp+fixed*xr-xq)*xs, axis=2)/np.maximum(ss, 1e-30), 0., 1.)
            on_a = np.clip(np.sum((xq+fixed*xs-xp)*xr, axis=2)/np.maximum(rr, 1e-30), 0., 1.)
            candidates.extend(((np.full(shape, fixed), on_b, np.ones(shape, bool)),
                               (on_a, np.full(shape, fixed), np.ones(shape, bool))))
        for ta, tb, valid in candidates:
            qa, qb = p+ta[..., None]*r, q+tb[..., None]*s
            apos = ca[start:end, None]+ta*la[start:end, None]
            bpos = cb[None, :-1]+tb*lb[None, :]
            # Both pieces must contribute observed geometry.
            if not allow_clipped_boundaries:
                valid = valid & (apos > EPS) & (cb[-1]-bpos > EPS)
            if junction_z_bounds is not None:
                valid &= (qa[...,1]>=junction_z_bounds[0]-EPS)&(qa[...,1]<=junction_z_bounds[1]+EPS)
                valid &= (qb[...,1]>=junction_z_bounds[0]-EPS)&(qb[...,1]<=junction_z_bounds[1]+EPS)
            valid &= (apos >= protected_a-EPS) & (bpos <= protected_b+EPS)
            if transition_arcs is not None:
                aa,bb=transition_arcs
                valid &= (apos>=aa[0]-EPS)&(apos<=aa[1]+EPS)&(bpos>=bb[0]-EPS)&(bpos<=bb[1]+EPS)
            lo = np.minimum(np.minimum(amin[start:end, None], qa[..., 1]),
                            np.minimum(bmin[None, 1:], qb[..., 1]))
            hi = np.maximum(np.maximum(amax[start:end, None], qa[..., 1]),
                            np.maximum(bmax[None, 1:], qb[..., 1]))
            lo, hi = np.minimum(lo, locked_range[0]), np.maximum(hi, locked_range[1])
            if preserve_extent:
                valid &= (lo <= required[0]+EPS) & (hi >= required[1]-EPS)
                valid &= (lo < required[0]-1e-6) if direction == 'lower' else (hi > required[1]+1e-6)
            if not valid.any():
                continue
            distance = np.linalg.norm(xp+ta[...,None]*xr-xq-tb[...,None]*xs, axis=2)
            # Coincident/intersecting geometry first; otherwise true minimum gap.
            distance_key = np.where(distance < EPS, 0., distance)
            removed = bpos if direction == 'lower' else ca[-1]-apos
            flat = np.flatnonzero(valid)
            timing = -removed if prefer_early else removed
            order = np.lexsort((flat, timing.ravel()[flat], distance_key.ravel()[flat]))
            k = flat[order[0]]
            i, j = np.unravel_index(k, shape)
            rank = (float(distance_key[i, j]), float(timing[i, j]))
            if best is None or rank < best[0]:
                best = (rank, start+i, j, float(ta[i, j]), float(tb[i, j]), qa[i, j], qb[i, j])
    if best is None:
        return None
    rank, i, j, ta, tb, qa, qb = best
    xa, xb = np.asarray(left[i]['points_xyz']), np.asarray(right[j]['points_xyz'])
    xa, xb = xa[0]+ta*(xa[1]-xa[0]), xb[0]+tb*(xb[1]-xb[0])
    norm = la[i]*lb[j]
    turn = float(np.degrees(np.arccos(np.clip(np.dot(da[i], db[j])/norm, -1., 1.)))) if norm > 0 else None
    virtual = sum(EPS < t < 1.-EPS for t in (ta, tb))
    return dict(junction_type='REAL_INTERSECTION' if rank[0] == 0. else 'CLOSEST_POINT_PAIR',
        a_edge_index=int(i), b_edge_index=int(j), a_edge_id=left[i]['edge_id'], b_edge_id=right[j]['edge_id'],
        a_t=ta, b_t=tb, a_point_uz=qa.tolist(), b_point_uz=qb.tolist(),
        a_point_xyz=xa.tolist(), b_point_xyz=xb.tolist(), distance_m=float(np.linalg.norm(qa-qb)),
        xyz_distance_m=float(np.linalg.norm(xa-xb)), new_virtual_nodes=virtual,
        tangent_turn_deg=turn, frontier=direction, removed_terminal_length_m=abs(rank[1]),
        from_branch_id=left[i]['branch_id'], to_branch_id=right[j]['branch_id'])


def _splice(left, right, junction):
    from dominant_observed_branch import trim_record
    i, j = junction['a_edge_index'], junction['b_edge_index']
    a = left[:i]+[trim_record(left[i], 0., junction['a_t'])]
    b = [trim_record(right[j], junction['b_t'], 1.)]+right[j+1:]
    a, b = ([r for r in records if np.linalg.norm(np.diff(r['points_uz'], axis=0)) > 1e-12] for records in (a, b))
    connector = dict(source='TOPOLOGY_SWITCH', face_id=None, source_face_ids=[], source_segment_indices=[],
        branch_id=None, points_uz=[junction['a_point_uz'], junction['b_point_uz']],
        points_xyz=[junction['a_point_xyz'], junction['b_point_xyz']])
    return a+[connector]+b


def assemble_main_track(branches, initial_route, *, anchor_branch_id, graph=None, target_s=None,
                        policy=None, review_gap_m=None):
    """Choose an evidenced branch first, then search its bounded connection.

    Rejected candidates cannot expand the requested extent. Every accepted
    piece retains a physical absolute core; even a zero-length intersection
    cannot promote an endpoint sliver or a short A-B-A excursion.
    """
    from dominant_observed_branch import route_result
    from branch_absolute_core import (BranchPolicy,local_edge_scale,branch_metrics,
                                      contribution_metrics,connector_gate,route_budgets)
    from surface_track_selection import branch_evidence,choose_candidate,same_surface_detail
    policy=policy or BranchPolicy()
    scale=local_edge_scale(branches)
    by_id={b['branch_id']:b for b in branches}
    evidence={b['branch_id']:branch_evidence(graph,(float(target_s),b['branch_id']),policy=policy,edge_scale=scale)
              if graph is not None else dict(branch_metrics(b,scale,policy),surface_support_tier=0)
              for b in branches if b['kind']!='CLOSED_COMPONENT'}
    reliable=[b for b in branches if b['branch_id'] in evidence and evidence[b['branch_id']]['MBG_pass']]
    records=[dict(r) for r in initial_route['path_edges']]
    if not records:
        return {**initial_route,'branch_audit':list(evidence.values()),'unresolved_reasons':['NO_RELIABLE_BRANCH']}
    junctions=list(initial_route.get('junctions',[]))
    if not junctions and initial_route.get('junction'):
        junctions=[initial_route['junction']]
    if not junctions:
        records=_oriented(records)
    attempts=[]; rejections=[]; decisions=[]
    internal_audit=list(initial_route.get('internal_handoff_audit',[]))
    blocked=set()
    while True:
        physical = graph is not None and graph.get('physical_face_context') is not None
        if graph is not None and graph.get('observations') and not physical:
            from directional_handoff import internal_handoff
            changed,audit=internal_handoff(reliable,records,graph=graph,target_s=target_s,
                anchor_branch_id=anchor_branch_id,policy=policy)
            internal_audit.extend(audit)
            if changed is not None:
                records=changed['path_edges'];join=changed['junction']
                if join['frontier']=='lower': junctions.insert(0,join)
                else: junctions.append(join)
                blocked.clear()
                continue
        points=_points(records)
        required=[float(points[:,1].min()),float(points[:,1].max())]
        pieces=_remaining(reliable,records)
        choices=[]
        for direction in ('lower','upper'):
            if direction in blocked:
                continue
            terminal=[]
            for r in (records if direction=='lower' else records[::-1]):
                if not r['source'].startswith('OBSERVED') or (terminal and r['branch_id']!=terminal[0]['branch_id']):
                    break
                terminal.append(r)
            if direction=='upper': terminal.reverse()
            if not terminal: continue
            locked=records[len(terminal):] if direction=='lower' else records[:-len(terminal)]
            locked_z=[p[1] for r in locked for p in r['points_uz']]
            locked_range=(min(locked_z),max(locked_z)) if locked_z else (np.inf,-np.inf)
            options=[]
            for index,piece in enumerate(pieces):
                p=_points(piece); bid=piece[0]['branch_id']
                if direction=='lower' and p[:,1].min()>=required[0]-1e-6: continue
                if direction=='upper' and p[:,1].max()<=required[1]+1e-6: continue
                z_gap=max(0.,required[0]-float(p[:,1].max())) if direction=='lower' else max(0.,float(p[:,1].min())-required[1])
                if z_gap>policy.connector_cap_m+1e-12:
                    rejections.append(dict(branch_id=bid,frontier=direction,reason='LONG_GAP_UNRESOLVED',minimum_possible_gap_m=z_gap))
                    continue
                region=(float(p[:,1].min()),required[0]) if direction=='lower' else (required[1],float(p[:,1].max()))
                row=branch_evidence(graph,(float(target_s),bid),policy=policy,edge_scale=scale,target_z=region) if graph is not None else evidence[bid].copy()
                contribution=contribution_metrics(by_id[bid],piece,evidence[bid])
                if not contribution['contribution_pass'] or row['target_region_in_ASC_fraction']<=0:
                    rejections.append(dict(branch_id=bid,frontier=direction,reason='REJECT_NO_RETAINED_CORE',**contribution))
                    continue
                options.append(dict(row,piece_index=index))
            attempts.append(dict(frontier=direction,current_z_range=required,extending_piece_count=len(options)))
            if not options: continue
            if physical:
                from competitive_surface_selection import successor_evidence
                options=[successor_evidence(row,by_id[row['branch_id']],terminal,graph=graph,
                    target_s=target_s,direction=direction,required=required,branches=reliable,policy=policy)
                    for row in options]
            elif len(options)>1:
                from directional_handoff import future_switch_cost
                zd=-1 if direction=='lower' else 1
                target_z=min(b['z_range'][0] for b in reliable) if zd==-1 else max(b['z_range'][1] for b in reliable)
                for row in options:
                    b=by_id[row['branch_id']]
                    row.update(future_switch_cost(reliable,row['branch_id'],z_direction=zd,target_z=target_z,policy=policy),
                        future_reliable_arc=row['ASC_supported_arc_length'] if row.get('raw_arc_evidence_available') else row['ASC_arc_length'],
                        future_directional_extent=max(0.,zd*(b['z_range'][0 if zd==-1 else 1]-required[0 if zd==-1 else 1])),
                        future_surface_support_span=row.get('support_s_span',0.))
            terminal_bid=terminal[0]['branch_id']
            core_before=contribution_metrics(by_id[terminal_bid],records,evidence[terminal_bid])['retained_ASC_arc_length']
            pending=list(options); rank=0
            while pending:
                if physical:
                    from competitive_surface_selection import choose_successor
                    decision=choose_successor(pending,current_branch_id=terminal_bid)
                else:
                    decision=choose_candidate(pending,policy=policy,current_branch_id=terminal_bid,
                        detail_loader=lambda r:same_surface_detail(graph,(float(target_s),r['branch_id'])))
                target=decision['selected']; rank+=1
                pending=[row for row in pending if row['piece_index']!=target['piece_index']]
                attempt=dict(frontier=direction,branch_id=target['branch_id'],piece_index=target['piece_index'],
                    identity_rank=rank,stage=decision['decision_stage'],ambiguous=decision['ambiguous'],
                    comparison_ambiguous=decision['comparison_ambiguous'],
                    detail_stage_comparisons=decision['detail_stage_comparisons'],feasibility_accepted=False)
                decisions.append(attempt)
                if physical:
                    attempt.update(identity_candidates=[{k:v for k,v in x.items() if k!='fragment'} for x in pending+[target]],
                        identity_trace=decision['trace'],competition_z=target['competition_z'],
                        FaceTrack=target['FaceTrack'],distance_used_for_identity=False)
                piece=pieces[target['piece_index']]
                left,right=(piece,terminal) if direction=='lower' else (terminal,piece)
                if physical:
                    from competitive_surface_selection import reliable_junction
                    join,diagnostic=reliable_junction(terminal,piece,by_id[terminal_bid],by_id[target['branch_id']],
                        evidence[terminal_bid],evidence[target['branch_id']],required=required,
                        locked_range=locked_range,direction=direction,policy=policy)
                    context=graph['physical_face_context']
                    region=context.cache[target['face_conflict_region']]['region']
                    from_face=context.evidence(float(target_s),by_id[terminal_bid],target['competition_z'],region=region)
                    diagnostic.update(from_FaceTrack=from_face['FaceTrack'],to_FaceTrack=target['FaceTrack'],
                        to_branch_forward_Z_extent=target['forward_Z_m'],to_branch_forward_ASC_length=target['forward_ASC_m'],
                        to_branch_future_switches=target['minimum_switches'])
                    attempt['junction_diagnostic']=diagnostic
                    if join is not None:join.update(diagnostic)
                else:
                    supported_fold=False
                    if graph is not None and graph.get('observations'):
                        from directional_handoff import _fold_protected,_arc_at
                        zd=-1 if direction=='lower' else 1
                        branch=by_id[terminal_bid]
                        begin=_arc_at(branch,terminal[0],0);end=_arc_at(branch,terminal[-1],1)
                        adir=zd*(1 if end>=begin else -1)
                        supported_fold=_fold_protected(graph,(float(target_s),terminal_bid),
                            end if zd<0 else begin,adir,zd,evidence[terminal_bid],target.get('support_s_span',0.))
                    # This protects a fold only when actual H/V evidence supports
                    # it; it does not restore the discarded static ASC lock.
                    join=_nearest_join(left,right,required,locked_range,direction,protect_folds=supported_fold)
                if join is None:
                    rejections.append(dict(frontier=direction,branch_id=target['branch_id'],
                        piece_index=target['piece_index'],identity_rank=rank,reason='NO_VALID_JUNCTION',
                        junction_diagnostic=attempt.get('junction_diagnostic')))
                    continue
                extension=_splice(left,right,join)
                separator=next(i for i,r in enumerate(extension) if r['source']=='TOPOLOGY_SWITCH')
                contribution_records=extension[:separator] if direction=='lower' else extension[separator+1:]
                contribution=contribution_metrics(by_id[target['branch_id']],contribution_records,evidence[target['branch_id']])
                gate=connector_gate(join['xyz_distance_m'],contribution['observed_new_length_m'],policy)
                new_records=extension+locked if direction=='lower' else locked+extension
                budgets=route_budgets(new_records,by_id,evidence,policy)
                budget_failure=next((b['reason'] for b in budgets if not b['accepted']),None)
                retained_ok=all(contribution_metrics(by_id[bid],new_records,evidence[bid])['contribution_pass']
                    for bid in {r['branch_id'] for r in new_records if r['source'].startswith('OBSERVED')})
                core_after=contribution_metrics(by_id[terminal_bid],new_records,evidence[terminal_bid])['retained_ASC_arc_length']
                if not contribution['contribution_pass'] or not retained_ok or not gate['accepted'] or budget_failure:
                    reason=(gate['reason'] if not gate['accepted'] else budget_failure or
                            'REJECT_NO_RETAINED_CORE')
                    rejections.append(dict(frontier=direction,branch_id=target['branch_id'],piece_index=target['piece_index'],
                        identity_rank=rank,junction=join,**contribution,**{**gate,'reason':reason}))
                    continue
                attempt['feasibility_accepted']=True
                join.update(contribution,**gate,identity_rank=rank,identity_ambiguous=decision['ambiguous'],
                    needs_manual_confirmation=decision['ambiguous'],terminal_ASC_before_m=core_before,
                    terminal_ASC_after_m=core_after)
                # A tie uses the existing deterministic identity order, remains
                # explicitly ambiguous, and is never broken by nearest distance.
                choices.append((direction,new_records,join))
                break
            else:
                blocked.add(direction)
        if not choices: break
        direction,records,join=choices[0]
        if direction=='lower': junctions.insert(0,join)
        else: junctions.append(join)
        blocked.clear()  # A changed route can make a previously failed frontier feasible.
        if len(junctions)>2*sum(len(b['records']) for b in reliable):
            raise RuntimeError('Non-progressing continuation')
    for r in records:
        if r['source'].startswith('OBSERVED'):
            r['source']='OBSERVED_DOMINANT' if r['branch_id']==anchor_branch_id else 'OBSERVED_SECONDARY'
    sequence=[]
    for r in records:
        if r['source'].startswith('OBSERVED') and (not sequence or sequence[-1]!=r['branch_id']): sequence.append(r['branch_id'])
    result=route_result(records,branch_switch_count=max(0,len(sequence)-1),junction=junctions[0] if junctions else None)
    actual=[float(result['curve_uz'][:,1].min()),float(result['curve_uz'][:,1].max())]
    extent=[min(b['z_range'][0] for b in reliable),max(b['z_range'][1] for b in reliable)] if reliable else actual
    covered=actual[0]<=extent[0]+1e-6 and actual[1]>=extent[1]-1e-6
    excluded=[bid for bid,e in evidence.items() if not e['MBG_pass'] and
              (by_id[bid]['z_range'][0]<actual[0]-1e-6 or by_id[bid]['z_range'][1]>actual[1]+1e-6)]
    return {**result,'junctions':junctions,'junction_count':len(junctions),'route_branch_sequence':sequence,
        'internal_handoff_audit':internal_audit,
        'directional_tail_count':sum(r['zone_count']>0 for r in internal_audit),
        'internal_handoff_count':sum(j.get('handoff_kind')=='INTERNAL_HANDOFF' for j in junctions),
        'internal_tail_trimmed_length':sum(j.get('internal_tail_trimmed_length',0.) for j in junctions),
        'folded_branch_handoff_evaluated_count':sum(r['folded'] for r in internal_audit),
        'folded_branch_handoff_accepted_count':sum(j.get('folded_branch_handoff',False) for j in junctions),
        'future_switch_avoided_count':sum(max(0,j.get('future_switch_alternative_estimate',0)-j.get('future_switch_estimate_after',0)) for j in junctions),
        'branch_audit':list(evidence.values()),'continuation_attempts':attempts,'continuation_decisions':decisions,
        'route_contribution_budgets':route_budgets(records,by_id,evidence,policy),
        'route_identity_ambiguous':any(evidence[bid].get('track_ambiguous',False) for bid in sequence)
            or any(j.get('identity_ambiguous',False) for j in junctions),
        'continuation_rejections':rejections,'excluded_extent_branch_ids':excluded,
        'unresolved_reasons':sorted(({r['reason'] for r in rejections} if not covered else set())|({'UNCOVERED_RELIABLE_EXTENT'} if not covered else set())|
                                  ({'EXCLUDED_UNRELIABLE_EXTENT'} if excluded else set())),
        'candidate_z_extent':extent,'route_z_extent':actual,'extent_covered':covered,
        'virtual_junction_count':sum(j['new_virtual_nodes'] for j in junctions),
        'review_gap_m':policy.connector_cap_m,'large_connector_count':sum(j['xyz_distance_m']>policy.connector_cap_m for j in junctions)}
