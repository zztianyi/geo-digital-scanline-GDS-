"""Select an observed identity; never blend or smooth measured coordinates.

Selection is deliberately separate from claiming that the old endpoint pair
has been repaired. All geometry records retain canonical edge provenance.
"""
from __future__ import annotations

import numpy as np
from local_surface_branch_tracking import _monotone_parts


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


def _detail_match(fragment, neighbors, low, high):
    # A multivalued fold is retained as geometry. Only a single monotone part
    # is used as its scalar descriptor; we never sort the actual polyline.
    parts = list(_monotone_parts(fragment['points_uz']))
    if not parts:
        return dict(neighbor_detail_score=0., neighbor_detail_repeat_count=0, neighbor_matches=[],
                    detail_status='NO_MONOTONE_DESCRIPTOR', detail_rms_m=0.)
    target = max(parts, key=lambda p: p[-1, 1]-p[0, 1])
    lo, hi = max(low, target[0, 1]), min(high, target[-1, 1])
    if hi-lo < 1e-5:
        return dict(neighbor_detail_score=0., neighbor_detail_repeat_count=0, neighbor_matches=[],
                    detail_status='INSUFFICIENT_HEIGHT', detail_rms_m=0.)
    z = np.linspace(lo, hi, 41)
    x = np.linspace(-1., 1., len(z))
    values = np.interp(z, target[:, 1], target[:, 0])
    detail = values-np.polyval(np.polyfit(x, values, 1), x)
    rms = float(np.sqrt(np.mean(detail**2)))
    matches = []
    for neighbor in neighbors:
        options = []
        for other in neighbor['branches']:
            if other['kind'] == 'CLOSED_COMPONENT' or other['z_range'][0] > lo or other['z_range'][1] < hi:
                continue
            if other['u_range'][1] < values.min()-.05 or other['u_range'][0] > values.max()+.05:
                continue
            for pi, part in enumerate(_monotone_parts(other['points_uz'])):
                if part[0, 1] > lo+1e-8 or part[-1, 1] < hi-1e-8:
                    continue
                v = np.interp(z, part[:, 1], part[:, 0])
                offset = float(np.median(abs(v-values)))
                if offset > .05:
                    continue
                d = v-np.polyval(np.polyfit(x, v, 1), x)
                other_rms = float(np.sqrt(np.mean(d*d)))
                denom = np.linalg.norm(detail)*np.linalg.norm(d)
                corr = float(np.dot(detail, d)/denom) if denom > 1e-16 else 0.
                shape_rmse = float(np.sqrt(np.mean((detail-d)**2)))
                similarity = max(0., corr)*np.exp(-shape_rmse/max(.0005, rms)) if min(rms, other_rms) >= .00005 else 0.
                options.append(dict(s=float(neighbor['s']), branch_id=other['branch_id'], monotone_part=pi,
                    correlation=corr, shape_rmse_m=shape_rmse, offset_m=offset, detail_score=float(similarity)))
        if options:
            matches.append(max(options, key=lambda r: (r['detail_score'], -r['offset_m'], -r['branch_id'])))
    return dict(neighbor_detail_score=float(np.mean([m['detail_score'] for m in matches])) if matches else 0.,
                neighbor_detail_repeat_count=sum(m['detail_score'] >= .6 for m in matches),
                neighbor_matches=matches, detail_status='DETAIL' if rms >= .00005 else 'FLAT_OR_UNDERSAMPLED',
                detail_rms_m=rms)


def crossings(points, z):
    points = np.asarray(points)
    a, b = points[:-1], points[1:]
    dz = b[:, 1]-a[:, 1]
    keep = (np.minimum(a[:, 1], b[:, 1]) <= z+1e-9) & (np.maximum(a[:, 1], b[:, 1]) >= z-1e-9) & (abs(dz) > 1e-12)
    return a[keep, 0]+(z-a[keep, 1])*(b[keep, 0]-a[keep, 0])/dz[keep]


def branch_reliability(branch, window, neighbors=()):
    low, high = window['lower_uz'][1], window['upper_uz'][1]
    span = high-low
    if span <= 0 or branch['kind'] == 'CLOSED_COMPONENT':
        return []
    endpoint_u = [window['lower_uz'][0], window['upper_uz'][0]]
    # Identity search radius only, never a snap/merge tolerance. Association
    # uses BOTH boundary positions rather than suppressing real interior
    # excursions (including folds) via their median U or a straight corridor.
    radial_pad = .25
    if branch['u_range'][1] < min(endpoint_u)-radial_pad or branch['u_range'][0] > max(endpoint_u)+radial_pad:
        return []
    result = []
    for fragment in clip_branch(branch, low, high):
        points = fragment['points_uz']
        if fragment['arc_length'] <= 1e-6:
            continue
        if (abs(points[0, 0]-endpoint_u[0]) > radial_pad or
                abs(points[-1, 0]-endpoint_u[1]) > radial_pad):
            continue
        coverage = float((points[:, 1].max()-points[:, 1].min())/span)
        positions = fragment['edge_arc_positions'].mean(axis=1)/max(branch['full_arc_length'], 1e-12)
        weights = np.linalg.norm(np.diff(points, axis=0), axis=1)
        interior = float(np.average(4*positions*(1-positions), weights=weights))
        hits, evaluated = 0, 0
        for layer in window['layers']:
            values = crossings(points, layer['z'])
            if not len(values) or not layer['horizontal']:
                continue
            evaluated += 1
            hits += any(np.min(abs(values-h['u'])) <= .003 for h in layer['horizontal'])
        detail = _detail_match(fragment, neighbors, low, high)
        endpoint_residual = float(np.linalg.norm(points[0]-window['lower_uz'])+np.linalg.norm(points[-1]-window['upper_uz']))
        result.append(dict(branch_id=branch['branch_id'], component_id=branch['component_id'],
            fragment_index=fragment['fragment_index'], kind=branch['kind'], fragment=fragment,
            observed_coverage=coverage, interior_score=interior, detail_density=branch['detail_density'],
            canonical_edge_count=branch['canonical_edge_count'], canonical_node_count=branch['canonical_node_count'],
            horizontal_support=hits/evaluated if evaluated else 0., horizontal_evaluable_layers=evaluated,
            horizontal_supported_layers=hits, endpoint_residual_m=endpoint_residual, **detail))
    return result


def _rank(row):
    # Binning the interior prior prevents insignificant differences from
    # always defeating genuinely repeated detail in the lexicographic order.
    return (round(row['observed_coverage'], 6), int(row['interior_score']/.1),
            row['neighbor_detail_repeat_count'], round(row['neighbor_detail_score'], 2),
            min(20, int(row['detail_density']/20)), round(row['horizontal_support'], 2),
            -row['endpoint_residual_m'], -row['branch_id'], -row['fragment_index'])


def select_dominant_branch(branches, window, neighbors=()):
    rows = [r for branch in branches for r in branch_reliability(branch, window, neighbors)]
    rows.sort(key=_rank, reverse=True)
    chosen = rows[0] if rows else None
    ambiguous = False
    if len(rows) > 1:
        a, b = rows[:2]
        ambiguous = (abs(a['observed_coverage']-b['observed_coverage']) < .01
            and abs(a['interior_score']-b['interior_score']) < .1
            and a['neighbor_detail_repeat_count'] == b['neighbor_detail_repeat_count']
            and abs(a['neighbor_detail_score']-b['neighbor_detail_score']) < .1
            and abs(a['horizontal_support']-b['horizontal_support']) < .1)
    for i, row in enumerate(rows):
        row['selected'] = i == 0
        row['reason'] = ('Highest lexicographic observed/interior/repeated-detail/support evidence' if i == 0
                         else 'Lower lexicographic branch reliability; no coordinate blending')
    return {'selected': chosen, 'candidates': rows, 'ambiguous': bool(ambiguous)}


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


def solve_dominant_branch(profile, node_uz, window, neighbors=(), *, branches=None):
    branches = extract_observed_branches(profile, node_uz) if branches is None else branches
    selection = select_dominant_branch(branches, window, neighbors)
    selected = selection['selected']
    common = dict(selection=selection, original_endpoint_pair_connected=False,
                  analysis_only=True, observed_coordinates_overwritten=0)
    if selected is None:
        return {**route_result([]), **common, 'status': 'NEEDS_GAP_INFERENCE'}
    low, high = window['lower_uz'][1], window['upper_uz'][1]
    spanning = [r for r in selection['candidates']
                if abs(r['fragment']['points_uz'][0, 1]-low) <= 3e-6
                and abs(r['fragment']['points_uz'][-1, 1]-high) <= 3e-6]
    if spanning and selected is not spanning[0]:
        selected = spanning[0]
        selection['selected'] = selected
        for row in selection['candidates']:
            row['selected'] = row is selected
            row['reason'] = ('Complete zero-switch observed route precedes extrema-only coverage' if row is selected
                             else 'Lower evidence among feasible spanning routes, or non-spanning fragment')
        # A different feasible ranking is itself reviewable; never silently
        # clear ambiguity inherited from similar evidence candidates.
        selection['ambiguous'] = True
    fragment = selected['fragment']
    points = fragment['points_uz']
    complete = abs(points[0, 1]-low) <= 3e-6 and abs(points[-1, 1]-high) <= 3e-6
    if complete:
        common['original_endpoint_pair_connected'] = bool(
            np.linalg.norm(points[0]-window['lower_uz']) <= 3e-6 and
            np.linalg.norm(points[-1]-window['upper_uz']) <= 3e-6)
        return {**route_result(fragment['records']), **common, 'status': 'PRESERVED_COMPLETE_OBSERVED'}
    from backtracking_junction import search_backtracking_junction, build_switch_route
    routes = []
    # At most two observed identities. No A->B->A is representable.
    for left in selection['candidates']:
        a = left['fragment']
        if abs(a['points_uz'][0, 1]-low) > 3e-6:
            continue
        for right in selection['candidates']:
            b = right['fragment']
            if left['branch_id'] == right['branch_id'] or abs(b['points_uz'][-1, 1]-high) > 3e-6:
                continue
            for junction in search_backtracking_junction(a, b):
                route = build_switch_route(a, b, junction, dominant_branch_id=selected['branch_id'])
                dominant_length = sum(float(np.linalg.norm(np.diff(e['points_uz'], axis=0))) for e in route['path_edges']
                                      if e.get('branch_id') == selected['branch_id'])
                if dominant_length > 1e-8:
                    routes.append((dominant_length, -route['connector_length_m'], route))
    if routes:
        route = max(routes, key=lambda r: r[:2])[2]
        # A sub-centimetre closest pair is a diagnostic proposal, not an
        # automatically accepted connection. Real intersections are distinguished.
        return {**route, **common, 'status': 'ONE_SWITCH_REVIEW_PROPOSAL'}
    return {**route_result(fragment['records']), **common, 'status': 'PRESERVED_PARTIAL_UNRESOLVED'}
