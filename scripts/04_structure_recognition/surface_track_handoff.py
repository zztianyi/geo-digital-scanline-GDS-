"""One local confidence handoff; whole-profile continuation is handled outside."""
from __future__ import annotations
import numpy as np
from horizontal_surface_link import branch_crossings, associate_crossing
from backtracking_junction import search_backtracking_junction, build_switch_route


def confidence_crossover(levels, support_a, support_b, *, margin=.25):
    levels, a, b = map(lambda x: np.asarray(x, dtype=float), (levels, support_a, support_b))
    if not (len(levels) == len(a) == len(b)) or len(levels) < 4:
        return None
    delta = a-b
    left = np.flatnonzero(delta >= margin)
    right = np.flatnonzero(delta <= -margin)
    # Persistent evidence on each side, with no later A->B->A reversal.
    if len(left) < 2 or len(right) < 2 or left[-1] >= right[0]:
        return None
    if not (np.all(np.diff(left) == 1) and np.all(np.diff(right) == 1)):
        return None
    return [float(levels[left[-1]]), float(levels[right[0]])]


def _oriented(branch):
    from dominant_observed_branch import trim_record
    if branch['points_uz'][0, 1] <= branch['points_uz'][-1, 1]:
        return branch
    records = [trim_record(r, 1., 0.) for r in branch['records'][::-1]]
    return {**branch, 'records': records, 'points_uz': branch['points_uz'][::-1],
            'points_xyz': branch['points_xyz'][::-1],
            'arc_positions': branch['arc_positions'][-1]-branch['arc_positions'][::-1]}


def track_confidence(graph,node,levels,*,policy=None,edge_scale=None):
    from branch_absolute_core import branch_metrics,local_edge_scale
    branch=graph['nodes'][node]
    if edge_scale is None:
        edge_scale=local_edge_scale([b for n,b in graph['nodes'].items() if n[0]==node[0]])
    metrics=branch_metrics(branch,edge_scale,policy)
    if not metrics['MBG_pass']: return np.zeros(len(levels))
    sides={}
    for link in graph['support'].get(node,[]):
        other=link['b'] if link['a']==node else link['a']
        side=int(np.sign(other[0]-node[0]))
        for li in link['levels']: sides.setdefault(li,set()).add(side)
    linked=set(sides)
    index={round(z,10):li for li,z in graph['level_z'].items()}
    result=[]
    for z in levels:
        h=index.get(round(float(z),10)) in linked
        if not h:
            result.append(0.); continue
        crossings=branch_crossings(branch,z)
        core=any(metrics['ASC_start_arc']<=c['arc_position']<=metrics['ASC_end_arc'] for c in crossings)
        h=index.get(round(float(z),10)) in linked
        tier=2 if {-1,1}.issubset(sides.get(index.get(round(float(z),10)),set())) else int(h)
        result.append(.7*tier+.3*core if crossings and h else 0.)
    return np.asarray(result)


def observed_support_length(graph, target_s, records):
    """Length on observed edges between consecutive linked H levels.

    This is an algorithmic support measure, not independently labelled truth.
    """
    intervals = {}
    total = 0.
    for record in records:
        if not record['source'].startswith('OBSERVED'):
            continue
        node = (float(target_s), record['branch_id'])
        if node not in intervals:
            levels = sorted({li for link in graph['support'].get(node, []) for li in link['levels']})
            runs=[]
            for a,b in zip(levels[:-1],levels[1:]):
                if b!=a+1: continue
                lo,hi=graph['level_z'][a],graph['level_z'][b]
                if runs and abs(runs[-1][1]-lo)<1e-9: runs[-1]=(runs[-1][0],hi)
                else: runs.append((lo,hi))
            intervals[node]=runs
        a, b = np.asarray(record['points_uz'])
        length = np.linalg.norm(b-a)
        z0, z1 = sorted((a[1], b[1]))
        if z1-z0 < 1e-12:
            if any(lo <= z0 <= hi for lo, hi in intervals[node]):
                total += length
        else:
            total += length*sum(max(0., min(z1, hi)-max(z0, lo)) for lo, hi in intervals[node])/(z1-z0)
    return float(total)


def select_handoff(graph,selected_node,*,policy=None,surface_track_id=None):
    from branch_absolute_core import BranchPolicy,local_edge_scale,contribution_metrics,connector_gate,route_budgets
    from surface_track_selection import branch_evidence,choose_candidate,same_surface_detail
    policy=policy or BranchPolicy()
    nodes=[n for n,b in graph['nodes'].items() if n[0]==selected_node[0] and b['kind']=='OPEN_SURFACE_BRANCH']
    scale=local_edge_scale([graph['nodes'][n] for n in nodes])
    evidence={n:branch_evidence(graph,n,policy=policy,edge_scale=scale) for n in nodes}
    candidates=[n for n in nodes if evidence[n]['MBG_pass'] and evidence[n]['surface_support_tier']
                and (surface_track_id is None or graph['membership'][n]==surface_track_id)]
    levels=sorted(set(graph['level_z'].values()))
    if selected_node not in candidates or len(candidates)<2 or len(levels)<4: return None
    support={n:track_confidence(graph,n,levels,policy=policy,edge_scale=scale) for n in candidates}
    options=[]
    for other in candidates:
        if other==selected_node: continue
        for a_node,b_node in ((selected_node,other),(other,selected_node)):
            interval=confidence_crossover(levels,support[a_node],support[b_node])
            if interval is None: continue
            # Compare against the incumbent over the region the other branch
            # would supply. The crossing interval alone may contain only one H
            # sample on each side of an otherwise well supported transition.
            branch=graph['nodes'][other]
            mid=float(np.mean(interval))
            region=(branch['z_range'][0],mid) if other==a_node else (mid,branch['z_range'][1])
            pair=[branch_evidence(graph,n,policy=policy,edge_scale=scale,target_z=region) for n in (selected_node,other)]
            regional=choose_candidate(pair,current_branch_id=selected_node[1],policy=policy,
                detail_loader=lambda r:same_surface_detail(graph,(selected_node[0],r['branch_id'])))
            row=regional['selected']
            crossing=branch_evidence(graph,other,policy=policy,edge_scale=scale,target_z=interval)
            if row['branch_id']!=other[1] or regional['comparison_ambiguous'] or crossing['target_region_in_ASC_fraction']<=0: continue
            options.append(dict(row,a_node=a_node,b_node=b_node,interval=interval,
                handoff_detail_comparisons=regional['detail_stage_comparisons']))
    if not options: return None
    decision=choose_candidate(options,policy=policy,
        detail_loader=lambda r:same_surface_detail(graph,(selected_node[0],r['branch_id'])))
    if decision['comparison_ambiguous']: return None
    choice=decision['selected']; a_node,b_node=choice['a_node'],choice['b_node']
    a,b=_oriented(graph['nodes'][a_node]),_oriented(graph['nodes'][b_node])
    if np.any(np.diff(a['points_uz'][:,1])<-1e-9) or np.any(np.diff(b['points_uz'][:,1])<-1e-9): return None
    # Only now, for a fixed evidenced pair, may geometry choose a junction.
    junctions=search_backtracking_junction(a,b,transition_interval=choice['interval'],maximum_distance=policy.connector_cap_m)
    routes=[]
    for junction in junctions:
        route=build_switch_route(a,b,junction,dominant_branch_id=selected_node[1])
        contributions={n:contribution_metrics(graph['nodes'][n],route['path_edges'],evidence[n]) for n in (a_node,b_node)}
        other=b_node if a_node==selected_node else a_node
        gate=connector_gate(junction['xyz_distance_m'],contributions[other]['observed_new_length_m'],policy)
        budgets=route_budgets(route['path_edges'],{n[1]:graph['nodes'][n] for n in (a_node,b_node)},
                              {n[1]:evidence[n] for n in (a_node,b_node)},policy)
        if not gate['accepted'] or not all(c['contribution_pass'] for c in contributions.values()) or not all(b['accepted'] for b in budgets): continue
        junction.update(confidence_crossover_interval=choice['interval'],in_confidence_crossover=True,
            from_branch_id=a_node[1],to_branch_id=b_node[1],**gate,**contributions[other])
        route.update(confidence_crossover_junction_count=1,junctions=[junction],
            handoff_detail_comparisons=choice['handoff_detail_comparisons']+decision['detail_stage_comparisons'])
        routes.append((junction['xyz_distance_m'],route))
    return min(routes,key=lambda r:r[0])[1] if routes else None


def linked_track_samples(graph, target_s, track_id):
    """H crossings linked to both neighboring members of an identified track.

    Works with target V missing; no target truth or closest-shape search.
    """
    order = graph['slice_order']
    if target_s not in order:
        return []
    i = order.index(target_s)
    if i == 0 or i == len(order)-1:
        return []
    sides = (order[i-1], order[i+1])
    side_nodes = {s: [(n, b) for n, b in graph['nodes'].items() if n[0] == s] for s in sides}
    by_path = {}
    for hit in graph['observations']:
        if hit['s'] in (*sides, target_s):
            by_path.setdefault((hit['level_index'], hit['h_path_id']), {}).setdefault(hit['s'], []).append(hit)
    samples = []
    for row in by_path.values():
        if any(len(row.get(s, [])) != 1 for s in (*sides, target_s)):
            continue
        verified = True
        for s in sides:
            hit = row[s][0]
            members = [n for n, b in side_nodes[s] if b['z_range'][0] <= hit['z'] <= b['z_range'][1]
                       and associate_crossing(hit, b)]
            verified &= len(members) == 1 and graph['membership'][members[0]] == track_id
        if verified:
            hit = row[target_s][0]
            samples.append((hit['z'], hit['u']))
    # Multivalued target support is ambiguous and must not be scalarized.
    values = {}
    for z, u in samples:
        values.setdefault(z, set()).add(round(u, 9))
    return sorted((z, next(iter(us))) for z, us in values.items() if len(us) == 1)


def infer_missing_interval(branches, interval, *, track_id, linked_samples=(), xyz_builder=None):
    from dominant_observed_branch import route_result
    common = dict(surface_track_id=track_id, status='UNRESOLVED_SURFACE_IDENTITY')
    if track_id is None:
        return {**route_result([]), **common}
    low, high = map(float, interval)
    if high <= low:
        raise ValueError('Missing interval must have positive height')
    # Presence of ANY identified-track edge inside the requested interval
    # blocks inference. The caller must split around observed intervals.
    for branch in branches:
        p = np.asarray(branch['points_uz'])
        if np.any((np.minimum(p[:-1, 1], p[1:, 1]) < high-1e-9) &
                  (np.maximum(p[:-1, 1], p[1:, 1]) > low+1e-9)):
            return {**route_result([]), **common, 'status': 'PRESERVED_OBSERVED_INTERVAL'}
    samples = np.asarray(sorted(set(linked_samples)), dtype=float).reshape(-1, 2)
    if (len(samples) < 2 or len(np.unique(samples[:, 0])) != len(samples)
            or samples[0, 0] > low+1e-9 or samples[-1, 0] < high-1e-9):
        return {**route_result([]), **common, 'status': 'UNRESOLVED_MISSING_SUPPORT'}
    zs = np.r_[low, samples[(samples[:, 0] > low) & (samples[:, 0] < high), 0], high]
    points = np.column_stack((np.interp(zs, samples[:, 0], samples[:, 1]), zs))
    xyz = xyz_builder(points) if xyz_builder is not None else np.column_stack((points[:, 0], np.zeros(len(points)), points[:, 1]))
    records = [dict(source='SURFACE_TRACK_INFERRED', face_id=None, source_face_ids=[], source_segment_indices=[],
                    branch_id=None, surface_track_id=track_id, points_uz=points[i:i+2].tolist(),
                    points_xyz=np.asarray(xyz)[i:i+2].tolist()) for i in range(len(points)-1)]
    return {**route_result(records), **common, 'status': 'INFERRED_IDENTIFIED_MISSING_INTERVAL'}


def recover_missing_tails(graph, node, *, xyz_builder=None):
    """Complete only provably missing tails of one monotone target member.

    Extent comes from two-sided same-path H observations, never a legacy gap.
    Multivalued/fragmented interiors remain unresolved instead of invented.
    """
    from dominant_observed_branch import route_result
    tid = graph['membership'][node]
    members = [n for n in graph['nodes'] if n[0] == node[0] and graph['membership'][n] == tid]
    if len(members) != 1:
        return None
    branch = _oriented(graph['nodes'][node])
    if np.any(np.diff(branch['points_uz'][:, 1]) < -1e-9):
        return None
    samples = linked_track_samples(graph, node[0], tid)
    if len(samples) < 2:
        return None
    low, high = branch['points_uz'][0, 1], branch['points_uz'][-1, 1]
    records, tails = list(branch['records']), []
    for lower in (True, False):
        missing = [(z, u) for z, u in samples if (z < low-1e-9 if lower else z > high+1e-9)]
        if len(missing) < 2:
            continue
        # All levels between the observed boundary and outer support must
        # carry unique same-track evidence; no unsupported extrapolation.
        bound = missing[0][0] if lower else missing[-1][0]
        interval = (bound, float(low)) if lower else (float(high), bound)
        required = {z for z in graph['level_z'].values() if interval[0] < z < interval[1]}
        if not required.issubset({z for z, _ in missing}):
            continue
        endpoint = branch['points_uz'][0 if lower else -1]
        anchored = missing+[(float(endpoint[1]), float(endpoint[0]))]
        target_observed = [b for n, b in graph['nodes'].items() if n[0] == node[0]]
        tail = infer_missing_interval(target_observed, interval, track_id=tid, linked_samples=anchored, xyz_builder=xyz_builder)
        if tail['inferred_length_m'] > 0:
            records = tail['path_edges']+records if lower else records+tail['path_edges']
            tails.append(dict(interval=interval, inferred_length_m=tail['inferred_length_m']))
    return {**route_result(records), 'missing_intervals': tails} if tails else None
