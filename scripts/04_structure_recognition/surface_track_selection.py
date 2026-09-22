"""Full-branch descriptors and endpoint-independent surface selection."""
from __future__ import annotations
import numpy as np


def local_descriptors(branch, metrics=None):
    from branch_absolute_core import branch_metrics, local_edge_scale
    metrics = metrics or branch_metrics(branch, local_edge_scale([branch]))
    points, arc = np.asarray(branch['points_uz']), np.asarray(branch['arc_positions'])
    vectors = np.diff(points, axis=0)
    angles = np.unwrap(np.arctan2(vectors[:, 0], vectors[:, 1]))
    curvature = np.r_[0., np.diff(angles)]
    return [dict(full_branch_arc_position=float(m), distance_to_branch_start=float(m),
        distance_to_branch_end=float(arc[-1]-m),
        in_absolute_sweet_core=bool(metrics['ASC_exists'] and metrics['ASC_start_arc'] <= m <= metrics['ASC_end_arc']),
        local_curvature_signature=float(curvature[i]), local_shape_signature=float(angles[i]))
        for i, m in enumerate((arc[:-1]+arc[1:])/2)]


def _shape(branch):
    points, arc = np.asarray(branch['points_uz']), np.asarray(branch['arc_positions'])
    if points[0, 1] > points[-1, 1]:
        points, arc = points[::-1], arc[-1]-arc[::-1]
    samples = np.linspace(0., max(arc[-1], 1e-12), 61)
    shape = np.column_stack([np.interp(samples, arc, points[:, k]) for k in (0, 1)])
    # Keep both U and Z detail so folded geometry is not reduced to one
    # monotone segment. Translation/trend removal is a descriptor only.
    trend = shape[0]+np.linspace(0., 1., len(shape))[:, None]*(shape[-1]-shape[0])
    return shape-trend


def same_surface_detail(graph, node):
    branch = graph['nodes'][node]
    tid = graph['membership'][node]
    shape = _shape(branch)
    matches = []
    # Adjacent graph members only, never an independent nearest-shape search.
    neighbors = {l['b'] if l['a'] == node else l['a'] for l in graph['support'].get(node, [])}
    for other in sorted(neighbors):
        if graph['membership'][other] != tid:
            continue
        reference = _shape(graph['nodes'][other])
        norm = np.linalg.norm(shape)*np.linalg.norm(reference)
        correlation = float(np.sum(shape*reference)/norm) if norm > 1e-14 else 0.
        rmse = float(np.sqrt(np.mean((shape-reference)**2)))
        rms = float(np.sqrt(np.mean(shape**2)))
        score = max(0., correlation)*np.exp(-rmse/max(rms, .0005)) if rms > .00005 else 0.
        matches.append(dict(s=other[0], branch_id=other[1], surface_track_id=tid, detail_score=float(score)))
    return dict(neighbor_detail_repeat_count=sum(m['detail_score'] >= .6 for m in matches),
        neighbor_detail_score=float(np.mean([m['detail_score'] for m in matches])) if matches else 0.,
        neighbor_matches=matches, same_surface_neighbor_detail=matches)


def branch_evidence(graph, node, *, policy=None, edge_scale=None, target_z=None, include_detail=False):
    from branch_absolute_core import branch_metrics, local_edge_scale
    branch = graph['nodes'][node]
    if edge_scale is None:
        edge_scale = local_edge_scale([b for n,b in graph['nodes'].items() if n[0]==node[0]])
    metrics = branch_metrics(branch, edge_scale, policy, target_z=target_z)
    tid = graph['membership'][node]
    track = graph.get('track_lookup', {}).get(tid)
    if track is None:
        track = next(t for t in graph['tracks'] if t['track_id'] == tid)
    links = graph['support'].get(node, []) if metrics['MBG_pass'] else []
    def active_levels(link):
        return [li for li in link['levels'] if target_z is None or min(target_z)<=graph['level_z'][li]<=max(target_z)]
    active = [(l,active_levels(l)) for l in links]
    active = [(l,ls) for l,ls in active if len(ls)>=2]
    sides = {int(np.sign((l['b'] if l['a']==node else l['a'])[0]-node[0])) for l,ls in active}
    tier = 2 if {-1,1}.issubset(sides) else int(bool(active))
    row = dict(metrics, component_id=branch['component_id'],kind=branch['kind'],fragment=branch,
        fragment_index=0,surface_track_id=tid,track_stable=track['stable'],track_ambiguous=track['ambiguous'],
        cross_slice_continuity=track['slice_count'],two_sided_support=tier==2,surface_support_tier=tier,
        observed_coverage=1.,detail_density=branch['detail_density'],
        horizontal_support=min((l['support_fraction'] for l,ls in active),default=0.),
        horizontal_supported_layers=len({li for l,ls in active for li in ls}),
        continuous_H_support_run=max((l['continuous_H_support_run'] for l,ls in active),default=0),
        detail_evaluated=False,neighbor_detail_repeat_count=0,neighbor_detail_score=0.,neighbor_matches=[],
        same_surface_neighbor_detail=[])
    from joint_surface_consensus import joint_surface_summary
    row.update(joint_surface_summary(graph, node, target_z=target_z, metrics=metrics))
    if include_detail and metrics['MBG_pass'] and tier:
        row.update(same_surface_detail(graph,node),detail_evaluated=True)
    return row


def choose_candidate(rows, *, current_branch_id=None, policy=None, detail_loader=None):
    """Lexicographic stages; geometry/nearest distance is deliberately absent."""
    from branch_absolute_core import BranchPolicy
    policy=policy or BranchPolicy()
    rows=[dict(r,selected=False) for r in rows]
    eligible=[r for r in rows if r['MBG_pass']]
    for r in rows:
        r['reason']='PASS_MINIMUM_BRANCH_GATE' if r['MBG_pass'] else 'REJECT_SHORT_BRANCH'
    if not eligible:
        return dict(selected=None,candidates=rows,ambiguous=False,decision_stage='MINIMUM_BRANCH_GATE',detail_stage_comparisons=0)
    supported=max(r['surface_support_tier'] for r in eligible)
    contenders=eligible
    stage='JOINT_SURFACE_CONSENSUS'; detail_count=0
    # H absence lowers confidence, but never removes an MBG-qualified branch.
    # Compare continuous measured persistence before coarse legacy tiers.
    if any(r.get('raw_arc_evidence_available') for r in contenders):
        best=max(r.get('support_s_span',0.) for r in contenders)
        if best>0:
            keep=[r for r in contenders if r.get('support_s_span',0.)>=best-max(.025,.1*best)]
            for r in contenders:
                if r not in keep: r['reason']='LOWER_JOINT_PERSISTENCE'
            contenders=keep
    else:
        keep=[r for r in contenders if r['surface_support_tier']==supported]
        for r in contenders:
            if r not in keep: r['reason']='LOWER_JOINT_CONFIDENCE'
        contenders=keep
    if len(contenders)>1:
        best=max(r['target_region_in_ASC_fraction'] for r in contenders)
        for r in contenders:
            if r['target_region_in_ASC_fraction']<best-policy.core_tie_tolerance:
                r['reason']='PREFER_ABSOLUTE_SWEET_CORE'
        contenders=[r for r in contenders if r['target_region_in_ASC_fraction']>=best-policy.core_tie_tolerance]
        stage='ABSOLUTE_SWEET_CORE'
    if len(contenders)>1 and all('estimated_additional_switches' in r for r in contenders):
        least=min(r['estimated_additional_switches'] for r in contenders)
        contenders=[r for r in contenders if r['estimated_additional_switches']==least]
        extent=max(r['future_directional_extent'] for r in contenders)
        contenders=[r for r in contenders if r['future_directional_extent']>=extent-policy.core_tie_tolerance]
        stage='FUTURE_SWITCH_MINIMIZATION'
    if len(contenders)>1 and supported and all(r['target_region_in_ASC_fraction']>0 for r in contenders):
        for r in contenders:
            if detail_loader is not None:
                r.update(detail_loader(r))
                r['detail_evaluated']=True
        detail_count=len(contenders)
        best=max(r.get('neighbor_detail_score',0.) for r in contenders)
        contenders=[r for r in contenders if r.get('neighbor_detail_score',0.)>=best-policy.detail_tie_tolerance]
        stage='SAME_SURFACE_DETAIL'
    comparison_ambiguous=len(contenders)>1 or not supported
    current=next((r for r in contenders if r['branch_id']==current_branch_id),None)
    selected=current or min(contenders,key=lambda r:r['branch_id'])
    ambiguous=comparison_ambiguous or bool(selected.get('track_ambiguous',False))
    selected.update(selected=True,reason='KEEP_CURRENT_NEAR_TIE' if current and ambiguous else stage)
    return dict(selected=selected,candidates=rows,ambiguous=ambiguous,
        comparison_ambiguous=comparison_ambiguous,
        decision_stage=stage,detail_stage_comparisons=detail_count,
        legacy_endpoint_influence_on_surface_identity=0)


def select_surface_track(graph,target_s,*,surface_track_id=None,policy=None,target_z=None,current_branch_id=None):
    from branch_absolute_core import local_edge_scale
    nodes=[n for n,b in graph['nodes'].items() if n[0]==float(target_s) and b['kind']!='CLOSED_COMPONENT'
           and (surface_track_id is None or graph['membership'][n]==surface_track_id)]
    scale=local_edge_scale([b for n,b in graph['nodes'].items() if n[0]==float(target_s)])
    rows=[branch_evidence(graph,n,policy=policy,edge_scale=scale,target_z=target_z) for n in nodes]
    decision=choose_candidate(rows,current_branch_id=current_branch_id,policy=policy,
        detail_loader=lambda r:same_surface_detail(graph,(float(target_s),r['branch_id'])))
    context=graph.get('physical_face_context')
    if context is None or decision['selected'] is None:return decision
    from horizontal_surface_link import branch_crossings
    from competitive_surface_selection import choose_successor
    from directional_handoff import future_switch_cost
    anchor=decision['selected']['fragment'];z=float(np.median(anchor['points_uz'][:,1]))
    region=context._region(float(target_s),anchor,z);competing=[]
    for row in rows:
        b=row['fragment'];hits=branch_crossings(b,z)
        if not hits or not row['MBG_pass']:continue
        face=context.evidence(float(target_s),b,z,region=region)
        inside=any(row['ASC_start_arc']<=h['arc_position']<=row['ASC_end_arc'] for h in hits)
        reliable=[r['fragment'] for r in rows if r['MBG_pass']]
        switches=sum(future_switch_cost(reliable,b['branch_id'],z_direction=d,
            target_z=(min(x['z_range'][0] for x in reliable) if d<0 else max(x['z_range'][1] for x in reliable)),
            policy=policy,reliable_entry=True)['estimated_additional_switches'] for d in (-1,1))
        competing.append(dict(row,**face,in_ASC=inside,forward_Z_m=float(np.ptp(b['points_uz'][:,1])),
            forward_ASC_m=row['ASC_arc_length'],minimum_switches=switches,competition_z=z))
    if len(competing)<2:return decision
    selected=choose_successor(competing,current_branch_id=current_branch_id)
    selected['candidates']=rows
    selected['physical_seed_candidates']=[{k:v for k,v in r.items() if k!='fragment'} for r in competing]
    selected['seed_conflict_region']=region['region_id']
    return selected
