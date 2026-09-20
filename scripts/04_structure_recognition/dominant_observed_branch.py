"""Select an observed identity; never blend or smooth measured coordinates.

Selection is deliberately separate from claiming that the old endpoint pair
has been repaired. All geometry records retain canonical edge provenance.
"""
from __future__ import annotations

import numpy as np


def extract_observed_branches(profile, node_uz):
    uz = np.asarray(node_uz, dtype=float)
    component_kind = {}
    for node, component in enumerate(profile.components):
        kind = component_kind.setdefault(int(component), 'CLOSED_COMPONENT')
        if profile.degree[node] > 2:
            component_kind[int(component)] = 'FORKED_COMPONENT'
        elif profile.degree[node] != 2 and kind != 'FORKED_COMPONENT':
            component_kind[int(component)] = 'OPEN_SURFACE_BRANCH'
    result = []
    for branch in profile.branches:
        order, eids = branch['nodes'], branch['edges']
        points, xyz = uz[order], profile.nodes[order]
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        valid = lengths > 1e-6
        arc = np.r_[0., np.cumsum(lengths)]
        records = []
        for i, eid in enumerate(eids):
            records.append(dict(edge_id=int(eid), branch_id=int(branch['branch_id']),
                node_ids=[int(order[i]), int(order[i+1])],
                points_uz=points[i:i+2].tolist(), points_xyz=xyz[i:i+2].tolist(),
                source='OBSERVED_DOMINANT', face_id=profile.first_face_ids[eid],
                source_face_ids=profile.source_face_ids[eid].copy(),
                source_segment_indices=profile.source_segment_indices[eid].copy(), t0=0., t1=1.))
        result.append(dict(branch_id=int(branch['branch_id']), component_id=int(branch['component_id']),
            kind=component_kind[int(branch['component_id'])], node_order=order.copy(), edge_order=eids.copy(),
            points_uz=points, points_xyz=xyz, records=records, arc_positions=arc, full_arc_length=float(arc[-1]),
            arc_length=float(arc[-1]), z_range=[float(points[:, 1].min()), float(points[:, 1].max())],
            u_range=[float(points[:, 0].min()), float(points[:, 0].max())],
            canonical_edge_count=int(valid.sum()), canonical_node_count=len(set(order)),
            detail_density=float(valid.sum()/lengths[valid].sum()) if valid.any() else 0.,
            ignored_tiny_edge_count=int((~valid).sum()),
            source_face_ids=sorted({f for eid in eids for f in profile.source_face_ids[eid]}),
            source_segment_indices=sorted({i for eid in eids for i in profile.source_segment_indices[eid]})))
    return result


def trim_record(record, t0, t1):
    out = dict(record)
    for field in ('points_uz', 'points_xyz'):
        points = np.asarray(record[field])
        out[field] = np.array([points[0] if t == 0. else points[1] if t == 1.
                              else points[0]+t*(points[1]-points[0]) for t in (t0, t1)]).tolist()
    out['t0'] = record['t0']+t0*(record['t1']-record['t0'])
    out['t1'] = record['t0']+t1*(record['t1']-record['t0'])
    return out


def clip_branch(branch, low, high):
    """Contiguous path fragments inside a Z slab, retaining folds and edge IDs."""
    groups, current, positions = [], [], []
    for i, record in enumerate(branch['records']):
        p, q = np.asarray(record['points_uz'])
        dz = q[1]-p[1]
        if abs(dz) < 1e-12:
            bounds = (0., 1.) if low-1e-10 <= p[1] <= high+1e-10 else None
        else:
            ta, tb = sorted(((low-p[1])/dz, (high-p[1])/dz))
            bounds = (max(0., ta), min(1., tb))
            if bounds[1]-bounds[0] <= 1e-12:
                bounds = None
        if bounds is None:
            if current:
                groups.append((current, positions)); current, positions = [], []
            continue
        ta, tb = bounds
        clipped = trim_record(record, ta, tb)
        if current and np.linalg.norm(np.asarray(current[-1]['points_uz'][1])-clipped['points_uz'][0]) > 1e-8:
            groups.append((current, positions)); current, positions = [], []
        current.append(clipped)
        length = branch['arc_positions'][i+1]-branch['arc_positions'][i]
        positions.append([branch['arc_positions'][i]+ta*length, branch['arc_positions'][i]+tb*length])
    if current:
        groups.append((current, positions))
    output = []
    for fragment_index, (records, positions) in enumerate(groups):
        if records[0]['points_uz'][0][1] > records[-1]['points_uz'][1][1]:
            records = [trim_record(r, 1., 0.) for r in records[::-1]]
            positions = [p[::-1] for p in positions[::-1]]
        points = np.asarray([records[0]['points_uz'][0]]+[r['points_uz'][1] for r in records])
        xyz = np.asarray([records[0]['points_xyz'][0]]+[r['points_xyz'][1] for r in records])
        output.append({**branch, 'records': records, 'points_uz': points, 'points_xyz': xyz,
                       'fragment_index': fragment_index, 'edge_arc_positions': np.asarray(positions),
                       'arc_length': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())})
    return output


def crossings(points, z):
    points = np.asarray(points)
    a, b = points[:-1], points[1:]
    dz = b[:, 1]-a[:, 1]
    keep = (np.minimum(a[:, 1], b[:, 1]) <= z+1e-9) & (np.maximum(a[:, 1], b[:, 1]) >= z-1e-9) & (abs(dz) > 1e-12)
    return a[keep, 0]+(z-a[keep, 1])*(b[keep, 0]-a[keep, 0])/dz[keep]


def _surface_context(branches, window, neighbors, surface_graph, target_s):
    from observed_surface_graph import build_surface_graph
    s = float(window.get('s', 0.) if target_s is None else target_s)
    if surface_graph is None:
        # Legacy layer hits deliberately cannot establish identity: their ROI
        # and available heights were derived from old endpoints.
        profiles = [dict(s=s, branches=branches)] + [n for n in neighbors if float(n['s']) != s]
        surface_graph = build_surface_graph(profiles)
    return surface_graph, s


def branch_reliability(branch, window=None, neighbors=(), *, surface_graph=None, target_s=None):
    from surface_track_selection import branch_evidence
    graph, s = _surface_context([branch], window or {}, neighbors, surface_graph, target_s)
    if branch['kind'] == 'CLOSED_COMPONENT' or branch['full_arc_length'] <= 1e-6:
        return []
    return [branch_evidence(graph, (s, branch['branch_id']))]


def select_dominant_branch(branches, window=None, neighbors=(), *, surface_graph=None,
                           target_s=None, surface_track_id=None):
    from surface_track_selection import select_surface_track
    graph, s = _surface_context(branches, window or {}, neighbors, surface_graph, target_s)
    return select_surface_track(graph, s, surface_track_id=surface_track_id)


def route_result(records, *, branch_switch_count=0, junction=None):
    if not records:
        curve = np.empty((0, 2))
    else:
        curve = np.asarray([records[0]['points_uz'][0]]+[r['points_uz'][1] for r in records])
    lengths = np.array([np.linalg.norm(np.diff(r['points_uz'], axis=0)) for r in records])
    observed = np.array([r['source'].startswith('OBSERVED') for r in records], dtype=bool)
    inferred = np.array([r['source'].endswith('INFERRED') or r['source'] == 'VERTICAL_PRIOR_FALLBACK' for r in records], dtype=bool)
    total = float(lengths.sum())
    return dict(curve_uz=curve, path_edges=records, branch_switch_count=branch_switch_count,
                virtual_junction_count=0 if junction is None else junction['new_virtual_nodes'],
                junction=junction, observed_length_m=float(lengths[observed].sum()),
                inferred_length_m=float(lengths[inferred].sum()),
                connector_length_m=float(lengths[~observed & ~inferred].sum()),
                observed_geometry_fraction=float(lengths[observed].sum()/total) if total else 0.)


def solve_dominant_branch(profile, node_uz, window=None, neighbors=(), *, branches=None,
                          surface_graph=None, target_s=None, surface_track_id=None, xyz_builder=None,
                          policy=None, input_z_range=None, source_data_required=False):
    """Branch-first route. ``window`` survives only as a legacy call adapter.

    Full graph inputs, target slice and an optional already-identified track
    determine identity. Neither endpoint coordinates nor window layers enter
    ranking, branch clipping, junction search or route identity.
    """
    from surface_track_selection import select_surface_track
    from surface_track_handoff import linked_track_samples, observed_support_length
    from main_track_assembly import assemble_main_track
    branches = extract_observed_branches(profile, node_uz) if branches is None else branches
    graph, s = _surface_context(branches, window or {}, neighbors, surface_graph, target_s)
    selection = select_surface_track(graph, s, surface_track_id=surface_track_id, policy=policy)
    selected = selection['selected']
    common = dict(selection=selection, original_endpoint_pair_connected=False,
        analysis_only=True, observed_coordinates_overwritten=0,
        legacy_endpoint_influence_on_surface_identity=0, route_identity=None,
        surface_track_id=surface_track_id, confidence_crossover_junction_count=0,
        observed_low_confidence_tail_removed=0., observed_high_confidence_length_preserved=0.)
    if selected is None:
        # Surface identity must be supplied/proven before missing-only inference.
        return {**route_result([]), **common, 'status': 'UNRESOLVED', 'unresolved_reasons': ['NO_RELIABLE_SURFACE_BRANCH']}
    common.update(route_identity=(selected['surface_track_id'], selected['branch_id']),
                  surface_track_id=selected['surface_track_id'],
                  observed_high_confidence_length_preserved=observed_support_length(graph, s, selected['fragment']['records']))
    # Assembly evaluates the initial internal transition and every subsequent
    # frontier once. Do not run a duplicate seed-only handoff beforehand.
    seed = route_result([dict(r) for r in selected['fragment']['records']])
    eligible = branches if surface_track_id is None else [b for b in branches
        if graph['membership'].get((s, b['branch_id'])) == surface_track_id]
    assembled = assemble_main_track(eligible, seed, anchor_branch_id=selected['branch_id'],
        graph=graph, target_s=s, policy=policy)
    reasons=list(assembled.get('unresolved_reasons',[]))
    if selection['ambiguous'] or assembled.get('route_identity_ambiguous'):
        reasons.append('AMBIGUOUS_SURFACE_IDENTITY')
    samples=linked_track_samples(graph,s,selected['surface_track_id']) if graph.get('observations') else []
    actual=assembled['route_z_extent']
    missing=[hit for hit in samples if hit[0]<actual[0]-1e-6 or hit[0]>actual[1]+1e-6]
    if missing: reasons.append('MISSING_SAME_SURFACE_OBSERVATIONS')
    truncated=bool(input_z_range is not None and any(
        b['branch_id'] in assembled.get('excluded_extent_branch_ids',[]) and
        (b['z_range'][0]<input_z_range[0]-1e-6 or b['z_range'][1]>input_z_range[1]+1e-6) for b in branches))
    status=('SOURCE_DATA_REQUIRED' if source_data_required else
            'INPUT_TRUNCATION_SUSPECT' if truncated else
            'UNRESOLVED' if reasons else 'ACCEPT_OBSERVED_BRANCH')
    tids=list(dict.fromkeys(graph['membership'][(s,bid)] for bid in assembled['route_branch_sequence']))
    # Automatic extrapolated tails used to bypass junction length and branch
    # contribution checks. Keep missing parts unresolved in this observed route.
    return {**common,**assembled,'status':status,'unresolved_reasons':sorted(set(reasons)),
        'anchor_identity':common['route_identity'],'surface_track_ids':tids,
        'input_truncation_suspect':truncated,'source_data_required':bool(source_data_required),
        'observed_high_confidence_length_preserved':observed_support_length(graph,s,assembled['path_edges']),
        'confidence_crossover_junction_count':assembled.get('internal_handoff_count',0),
        'handoff_detail_comparisons':0,
        'missing_same_surface_samples':missing,'missing_tail_check_available':bool(graph.get('observations'))}
