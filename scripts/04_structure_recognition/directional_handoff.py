"""Bounded look-ahead and arc-ordered internal replacement of directional tails."""
import numpy as np
from branch_absolute_core import (BranchPolicy, branch_metrics, local_edge_scale,
    directional_branch_metrics, contribution_metrics, connector_gate, route_budgets)
from joint_surface_consensus import arc_support_profile, joint_surface_summary
from horizontal_surface_link import branch_crossings


def _arc_at(branch, record, end):
    i = branch['edge_order'].index(record['edge_id']) if isinstance(branch['edge_order'], list) else int(np.flatnonzero(np.asarray(branch['edge_order']) == record['edge_id'])[0])
    arc = branch['arc_positions']
    return float(arc[i]+record['t0' if end == 0 else 't1']*(arc[i+1]-arc[i]))


def _forward_interval(branch, pos, direction):
    return (pos, float(branch['arc_positions'][-1])) if direction == 1 else (0., pos)


def _direction(branch, z_direction):
    return z_direction*(1 if branch['points_uz'][-1, 1] >= branch['points_uz'][0, 1] else -1)


def future_switch_cost(branches, start_branch_id, *, z_direction, target_z, policy=None, depth=2,
                       start_records=None, reliable_entry=False):
    """Depth-two search through actual bounded, core-retaining continuations.

    An unreachable frontier is reported explicitly; depth+1 is a lower-bound
    sentinel, not a claim of global optimality. Geometry never ranks identities.
    """
    from main_track_assembly import _oriented, _nearest_join, _splice
    policy = policy or BranchPolicy()
    scale = local_edge_scale(branches)
    bs = {b['branch_id']: b for b in branches if branch_metrics(b, scale, policy)['MBG_pass']}
    metrics={bid:branch_metrics(b,scale,policy) for bid,b in bs.items()}
    def search(records, bid, remaining, visited):
        zs=[p[1] for r in records for p in r['points_uz']]
        frontier=max(zs) if z_direction==1 else min(zs)
        if z_direction*(frontier-target_z) >= -1e-6:
            return 0, True
        if remaining == 0:
            return depth+1, False
        best = (depth+1, False)
        terminal=[]
        for r in (records[::-1] if z_direction==1 else records):
            if not r['source'].startswith('OBSERVED') or r['branch_id']!=bid: break
            terminal.append(r)
        if z_direction==1:terminal.reverse()
        if not terminal:return best
        locked=records[:-len(terminal)] if z_direction==1 else records[len(terminal):]
        locked_z=[p[1] for r in locked for p in r['points_uz']]
        locked_range=(min(locked_z),max(locked_z)) if locked_z else (np.inf,-np.inf)
        for other, candidate in bs.items():
            if other in visited or z_direction*(candidate['z_range'][1 if z_direction == 1 else 0]-frontier) <= 1e-6:
                continue
            gap = max(0., candidate['z_range'][0]-frontier) if z_direction == 1 else max(0., frontier-candidate['z_range'][1])
            if gap > policy.connector_cap_m:
                continue
            c = _oriented(candidate['records'])
            left, right = (terminal, c) if z_direction == 1 else (c, terminal)
            if reliable_entry:
                from competitive_surface_selection import reliable_junction
                join,_=reliable_junction(terminal,c,bs[bid],candidate,metrics[bid],metrics[other],
                    required=(min(zs),max(zs)),locked_range=locked_range,
                    direction='upper' if z_direction==1 else 'lower',policy=policy)
            else:
                join = _nearest_join(left, right, (min(zs),max(zs)), locked_range,
                                     'upper' if z_direction == 1 else 'lower', protect_folds=False)
            if join is None or join['xyz_distance_m'] > policy.connector_cap_m+1e-12:
                continue
            extension = _splice(left, right, join)
            new_records=locked+extension if z_direction==1 else extension+locked
            if not all(contribution_metrics(bs[i],new_records,metrics[i])['contribution_pass'] for i in (*visited,other)):
                continue
            if not all(x['accepted'] for x in route_budgets(new_records,bs,metrics,policy)):
                continue
            count, reached = search(new_records,other,remaining-1,visited+(other,))
            if reached and count+1 < best[0]:
                best = count+1, True
        return best
    initial=_oriented(start_records if start_records is not None else bs[start_branch_id]['records']) if start_branch_id in bs else []
    count, reached = search(initial,start_branch_id,depth,(start_branch_id,)) if initial else (depth+1, False)
    return dict(estimated_additional_switches=count, future_frontier_reached=reached, lookahead_depth=depth)


def _fold_protected(graph, node, pos, direction, z_direction, metrics, alternative_span):
    """Only measured, persistent core folds are protected from replacement."""
    b = graph['nodes'][node]; arc = np.asarray(b['arc_positions']); p = np.asarray(b['points_uz'])
    lo, hi = _forward_interval(b, pos, direction)
    mask = np.diff(p[:,1])*direction*z_direction < -1e-9
    ids = np.flatnonzero(mask)
    groups = np.split(ids, np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
    for group in groups:
        if arc[group[-1]+1]<=lo+1e-9 or arc[group[0]]>=hi-1e-9: continue
        # Cutting a protected fold's final few millimetres still cuts that fold.
        # Assess its whole original core run, not just the post-junction sliver.
        a, c = max(arc[group[0]], metrics['ASC_start_arc']), min(arc[group[-1]+1], metrics['ASC_end_arc'])
        if c-a < metrics['minimum_core_arc_length']:
            continue
        support = joint_surface_summary(graph, node, arc_interval=(a,c), metrics=metrics)
        if (support['same_H_run_level_count'] >= 1 and support['two_sided_neighbor_count'] >= 1
                and support['support_s_span'] >= max(.1, .8*alternative_span)):
            return True
    return False


def _bracket(branch, profile, pos):
    values = np.unique([0., *(r['arc_position'] for r in profile), float(branch['arc_positions'][-1])])
    i = int(np.searchsorted(values, pos))
    return (float(values[max(0,i-1)]), float(values[min(len(values)-1,i+1)]))


def _zone_join(left, right, intervals, frontier):
    """Bound geometry work to the preselected original arc intervals."""
    from main_track_assembly import _nearest_join
    clipped=[]; offsets=[]; local=[]
    for records,(low,high) in zip((left,right),intervals):
        lengths=np.array([np.linalg.norm(np.diff(r['points_uz'],axis=0)) for r in records])
        arc=np.r_[0.,np.cumsum(lengths)]
        ids=np.flatnonzero((arc[:-1]<=high+1e-9)&(arc[1:]>=low-1e-9))
        if not len(ids): return None
        start,end=int(ids[0]),int(ids[-1])+1
        clipped.append(records[start:end]);offsets.append(start)
        local.append((low-arc[start],high-arc[start]))
    join=_nearest_join(*clipped,(0.,0.),(np.inf,-np.inf),frontier,
        transition_arcs=local,protect_folds=False,preserve_extent=False)
    if join is not None:
        join['a_edge_index']+=offsets[0];join['b_edge_index']+=offsets[1]
    return join


def candidate_transition_zones(graph, a_node, b_node, *, direction, z_direction,
                               arc_interval=None, policy=None, edge_scale=None):
    """Compare hypotheses on separate original arc coordinates before geometry."""
    a, b = graph['nodes'][a_node], graph['nodes'][b_node]
    scale = edge_scale if edge_scale is not None else local_edge_scale([a,b])
    ma, mb = branch_metrics(a,scale,policy), branch_metrics(b,scale,policy)
    if not ma['MBG_pass'] or not mb['MBG_pass']:
        return []
    if min(a['z_range'][1],b['z_range'][1]) < max(a['z_range'][0],b['z_range'][0]):
        return []
    pa, pb = arc_support_profile(graph,a_node), arc_support_profile(graph,b_node)
    bdir = _direction(b,z_direction)
    zones = []; seen = set()
    for row in pa:
        pos = row['arc_position']
        if arc_interval is not None and not min(arc_interval)+1e-9 < pos < max(arc_interval)-1e-9:
            continue
        if not b['z_range'][0]-1e-9 <= row['z'] <= b['z_range'][1]+1e-9:
            continue
        crossings=branch_crossings(b,row['z'])
        if not crossings: continue
        sa = joint_surface_summary(graph,a_node,arc_interval=_forward_interval(a,pos,direction),metrics=ma)
        for cross in crossings:
            bp = cross['arc_position']
            signature = (round(pos,7),round(bp,7))
            if signature in seen: continue
            seen.add(signature)
            sb = joint_surface_summary(graph,b_node,arc_interval=_forward_interval(b,bp,bdir),metrics=mb)
            db = directional_branch_metrics(b,bp,bdir,z_direction=z_direction,edge_scale=scale,support=sb,policy=policy)
            da = directional_branch_metrics(a,pos,direction,z_direction=z_direction,edge_scale=scale,support=sa,
                alternative=dict(db,support_s_span=sb['support_s_span']),policy=policy)
            if (da['directional_state'] != 'DIRECTIONAL_TAIL' or sb['same_H_run_level_count'] < 2
                    or db['L_forward_ASC'] < mb['minimum_core_arc_length']):
                continue
            if _fold_protected(graph,a_node,pos,direction,z_direction,ma,sb['support_s_span']):
                continue
            zones.append(dict(a_arc=_bracket(a,pa,pos), b_arc=_bracket(b,pb,bp),
                a_metrics=da,b_metrics=db,a_support=sa,b_support=sb,b_direction=bdir,
                future_reliable_arc=sb['ASC_supported_arc_length'],future_directional_extent=db['Z_forward_new'],
                future_surface_support_span=sb['support_s_span']))
    return zones


def internal_handoff(branches, records, *, graph, target_s, anchor_branch_id, policy=None):
    """Replace one terminal's internal low-confidence tail, before its fold lock.

    Each call makes at most one change. The caller may repeat after progress;
    already-used branch IDs cannot be re-entered through this internal stage.
    """
    from main_track_assembly import _oriented, _nearest_join, _splice
    policy=policy or BranchPolicy(); scale=local_edge_scale(branches)
    by_id={b['branch_id']:b for b in branches if b['kind']!='CLOSED_COMPONENT'}
    metrics={bid:branch_metrics(b,scale,policy) for bid,b in by_id.items()}
    used={r['branch_id'] for r in records if r['source'].startswith('OBSERVED')}
    audit=[]; options=[]
    for frontier, zd in (('lower',-1),('upper',1)):
        terminal=[]
        for r in (records if zd==-1 else records[::-1]):
            if not r['source'].startswith('OBSERVED') or (terminal and r['branch_id']!=terminal[0]['branch_id']): break
            terminal.append(r)
        if zd==1: terminal.reverse()
        if not terminal: continue
        bid=terminal[0]['branch_id'];a=by_id[bid];an=(float(target_s),bid)
        bounds=(_arc_at(a,terminal[0],0),_arc_at(a,terminal[-1],1))
        direction=zd*(1 if bounds[1]>=bounds[0] else -1)
        target=min(b['z_range'][0] for i,b in by_id.items() if metrics[i]['MBG_pass']) if zd==-1 else max(b['z_range'][1] for i,b in by_id.items() if metrics[i]['MBG_pass'])
        for other,b in by_id.items():
            if other in used or not metrics[other]['MBG_pass']: continue
            zones=candidate_transition_zones(graph,an,(float(target_s),other),direction=direction,z_direction=zd,
                arc_interval=bounds,policy=policy,edge_scale=scale)
            folded=bool(np.any(np.diff(a['points_uz'][:,1])*direction*zd < -1e-9) or np.any(np.diff(b['points_uz'][:,1])*_direction(b,zd)*zd < -1e-9))
            item=dict(from_branch_id=bid,to_branch_id=other,frontier=frontier,zone_count=len(zones),
                folded=folded,reason='NO_DIRECTIONAL_TRANSITION' if not zones else 'CANDIDATE_TRANSITION')
            audit.append(item)
            if not zones: continue
            future=future_switch_cost(branches,other,z_direction=zd,target_z=target,policy=policy)
            before=future_switch_cost(branches,bid,z_direction=zd,target_z=target,policy=policy)
            best=max(zones,key=lambda z:(z['future_reliable_arc'],z['future_directional_extent'],z['future_surface_support_span']))
            item.update(future,estimated_additional_switches_before=before['estimated_additional_switches'],
                future_reliable_arc=best['future_reliable_arc'],future_directional_extent=best['future_directional_extent'],
                future_surface_support_span=best['future_surface_support_span'])
            options.append(dict(a=a,b=b,terminal=terminal,bounds=bounds,zones=zones,frontier=frontier,
                audit=item,rank=(best['future_reliable_arc'],-future['estimated_additional_switches'],
                                 best['future_directional_extent'],best['future_surface_support_span'])))
    if not options: return None,audit
    # Fix the target hypothesis first. Distance never selects a competing branch.
    options.sort(key=lambda x:x['rank'],reverse=True)
    chosen=options[0];a,b=chosen['a'],chosen['b'];terminal=chosen['terminal'];lower=chosen['frontier']=='lower'
    alternatives=[o['audit']['estimated_additional_switches'] for o in options[1:] if o['frontier']==chosen['frontier']]
    alternative_estimate=min(alternatives) if alternatives else chosen['audit']['estimated_additional_switches']
    candidate=_oriented(b['records']); left,right=(candidate,terminal) if lower else (terminal,candidate)
    start_a=chosen['bounds'][0]; end_a=chosen['bounds'][1]
    reversed_a=end_a<start_a
    reversed_b=b['points_uz'][0,1]>b['points_uz'][-1,1]
    def convert(interval, start, reverse):
        return tuple(sorted((start-x if reverse else x-start) for x in interval))
    joins=[]; seen=set()
    for zone in chosen['zones']:
        aa=convert(zone['a_arc'],start_a,reversed_a)
        bb=convert(zone['b_arc'],float(b['arc_positions'][-1]) if reversed_b else 0.,reversed_b)
        signature=tuple(round(v,7) for v in (*aa,*bb))
        if signature in seen: continue
        seen.add(signature)
        join=_zone_join(left,right,(bb,aa) if lower else (aa,bb),chosen['frontier'])
        if join is None or join['xyz_distance_m']>policy.connector_cap_m+1e-12: continue
        # Bracketing expands around evidence samples. It is not permission to
        # cut anywhere in that bracket: recheck at the TWO actual arc positions.
        ai,at=(join['b_edge_index'],join['b_t']) if lower else (join['a_edge_index'],join['a_t'])
        bi,bt=(join['a_edge_index'],join['a_t']) if lower else (join['b_edge_index'],join['b_t'])
        apos=_arc_at(a,terminal[ai],0)+at*(_arc_at(a,terminal[ai],1)-_arc_at(a,terminal[ai],0))
        bpos=_arc_at(b,candidate[bi],0)+bt*(_arc_at(b,candidate[bi],1)-_arc_at(b,candidate[bi],0))
        zd=-1 if lower else 1;adir=zd*(-1 if reversed_a else 1);bdir=_direction(b,zd)
        an,bn=(float(target_s),a['branch_id']),(float(target_s),b['branch_id'])
        sa=joint_surface_summary(graph,an,arc_interval=_forward_interval(a,apos,adir),metrics=metrics[a['branch_id']])
        sb=joint_surface_summary(graph,bn,arc_interval=_forward_interval(b,bpos,bdir),metrics=metrics[b['branch_id']])
        db=directional_branch_metrics(b,bpos,bdir,z_direction=zd,edge_scale=scale,support=sb,policy=policy)
        da=directional_branch_metrics(a,apos,adir,z_direction=zd,edge_scale=scale,support=sa,
            alternative=dict(db,support_s_span=sb['support_s_span']),policy=policy)
        if (da['directional_state']!='DIRECTIONAL_TAIL' or sb['same_H_run_level_count']<2
                or db['L_forward_ASC']<metrics[b['branch_id']]['minimum_core_arc_length']
                or _fold_protected(graph,an,apos,adir,zd,metrics[a['branch_id']],sb['support_s_span'])):
            continue
        extension=_splice(left,right,join)
        locked=records[len(terminal):] if lower else records[:-len(terminal)]
        assembled=extension+locked if lower else locked+extension
        contributions={i:contribution_metrics(by_id[i],assembled,metrics[i]) for i in used|{b['branch_id']}}
        gate=connector_gate(join['xyz_distance_m'],contributions[b['branch_id']]['observed_new_length_m'],policy)
        budgets=route_budgets(assembled,by_id,metrics,policy)
        if not gate['accepted'] or not all(c['contribution_pass'] for c in contributions.values()) or not all(c['accepted'] for c in budgets): continue
        trimmed=contribution_metrics(a,records,metrics[a['branch_id']])['observed_new_length_m']-contributions[a['branch_id']]['observed_new_length_m']
        retained_candidate=[r for r in extension if r['source'].startswith('OBSERVED') and r['branch_id']==b['branch_id']]
        target_z=min(x['z_range'][0] for i,x in by_id.items() if metrics[i]['MBG_pass']) if lower else max(x['z_range'][1] for i,x in by_id.items() if metrics[i]['MBG_pass'])
        actual_future=future_switch_cost(branches,b['branch_id'],z_direction=zd,target_z=target_z,
            policy=policy,start_records=retained_candidate)
        join.update(gate,**contributions[b['branch_id']],handoff_kind='INTERNAL_HANDOFF',
            internal_tail_trimmed_length=trimmed,transition_a_arc=zone['a_arc'],transition_b_arc=zone['b_arc'],
            directional_metrics=da,candidate_directional_metrics=db,
            future_switch_estimate_before=chosen['audit']['estimated_additional_switches_before'],
            future_switch_estimate_after=actual_future['estimated_additional_switches'],
            future_frontier_reached=actual_future['future_frontier_reached'],
            future_switch_alternative_estimate=alternative_estimate,
            folded_branch_handoff=chosen['audit']['folded'])
        # Within the chosen hypothesis, retain the durable observed continuation
        # before minimizing synthetic geometry. Late intersections must not lock
        # back in the low-confidence fold which triggered this replacement.
        joins.append((sb['ASC_supported_arc_length'],db['Z_forward_new'],
                      join['xyz_distance_m'],trimmed,assembled,join))
    if not joins:
        chosen['audit']['reason']='NO_SAFE_JUNCTION_IN_TRANSITION_ZONE'
        return None,audit
    _,_,_,_,assembled,join=min(joins,key=lambda x:(-x[0],-x[1],0. if x[2]<1e-6 else x[2],x[3]))
    chosen['audit']['reason']='ACCEPT_INTERNAL_HANDOFF'
    return dict(path_edges=assembled,junction=join),audit
