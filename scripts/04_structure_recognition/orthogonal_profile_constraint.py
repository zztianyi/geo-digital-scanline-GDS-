"""Conservative, multi-valued horizontal/vertical profile observations.

This module does not change the GDS recognition predicates. Coordinates are in
metres. Missing observations stay missing; disconnected branches are never
averaged into a single radial surface.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def to_suz(points, arc):
    points = np.asarray(points, dtype=float)
    delta = points[..., :2] - np.asarray(arc['center'])
    theta = np.arctan2(delta[..., 1], delta[..., 0])
    midpoint = (arc['angle_min'] + arc['angle_max']) / 2
    theta = midpoint + np.arctan2(np.sin(theta-midpoint), np.cos(theta-midpoint))
    return np.stack((arc['radius'] * (theta-arc['angle_min']),
                     np.linalg.norm(delta, axis=-1)-arc['radius'], points[..., 2]), axis=-1)


def line_graph(lines, decimals=None):
    points = np.asarray(lines, dtype=float).reshape(-1, 3)
    if decimals is not None:
        points = np.round(points, decimals)
    if not len(points):
        return np.empty((0, 3)), np.empty((0, 2), dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    nodes, inverse = np.unique(points, axis=0, return_inverse=True)
    edges = inverse.reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=len(nodes))
    graph = coo_matrix((np.ones(2*len(edges)),
                       (edges.ravel(), edges[:, ::-1].ravel())), shape=(len(nodes), len(nodes)))
    _, components = connected_components(graph, directed=False)
    return nodes, edges, degree, components


def endpoint_distances(lines, decimals=None):
    nodes, edges, degree, components = line_graph(lines, decimals)
    endpoints = np.flatnonzero(degree == 1)
    if len(endpoints) < 2:
        return np.array([])
    # Only distinct components are possible cracks, not a short intact edge.
    k = min(12, len(endpoints))
    distances, neighbors = cKDTree(nodes[endpoints]).query(nodes[endpoints], k=k)
    values = []
    for i in range(len(endpoints)):
        valid = components[endpoints[neighbors[i, 1:]]] != components[endpoints[i]]
        if np.any(valid):
            values.append(float(distances[i, 1:][valid][0]))
    return np.asarray(values)


def suggest_snap_tolerance(gaps, step):
    gaps = np.asarray(gaps)
    finite = gaps[np.isfinite(gaps) & (gaps > 0)]
    # Only the numerical near-zero population is auto-snapped. Millimetric
    # physical cracks require independent support and are not mesh repair.
    numerical = finite[finite <= min(1e-5, step*0.001)]
    tolerance = max(1e-10, float(np.quantile(numerical, .99))*2) if len(numerical) else 0.0
    return min(tolerance, 1e-5), {'gap_count': len(finite), 'numerical_gap_count': len(numerical),
                                'rule': '2*p99 of near-zero endpoint gaps, capped at 1e-5 m; no physical gap snapping'}


def _ordered_branches(nodes, edges, degree):
    adjacency = defaultdict(list)
    for eid, (a, b) in enumerate(edges):
        adjacency[int(a)].append((int(b), eid))
        adjacency[int(b)].append((int(a), eid))
    used, branches = set(), []

    def follow(start, next_node, edge):
        node_order, edge_order = [start], []
        current = start
        while edge not in used:
            used.add(edge)
            edge_order.append(edge)
            node_order.append(next_node)
            current = next_node
            if degree[current] != 2:
                break
            choices = [(n, e) for n, e in adjacency[current] if e not in used]
            if not choices:
                break
            next_node, edge = choices[0]
        branches.append({'nodes': np.asarray(node_order), 'edges': np.asarray(edge_order)})

    for node in np.flatnonzero(degree != 2):
        for next_node, eid in adjacency[int(node)]:
            if eid not in used:
                follow(int(node), next_node, eid)
    for eid, (a, b) in enumerate(edges):
        if eid not in used:
            follow(int(a), int(b), eid)
    return branches


def clean_horizontal(lines, face_ids, arc, snap_tolerance=0.0):
    lines = np.asarray(lines, dtype=float).reshape(-1, 2, 3)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    nodes, edges, degree, components = line_graph(lines)
    if not len(edges):
        return {'lines_3d': lines, 'face_ids': face_ids, 'edge_branch': np.array([], dtype=int),
                'branches': [], 'diagnostics': {'raw_segments': 0, 'clean_segments': 0, 'components': 0,
                                               'endpoints': 0, 'branches': 0, 'snapped_pairs': 0}}
    canonical = np.sort(edges, axis=1)
    _, keep = np.unique(canonical, axis=0, return_index=True)
    keep.sort()
    lines, face_ids = lines[keep].copy(), face_ids[keep]
    nodes, edges, degree, components = line_graph(lines)
    snapped = 0
    if snap_tolerance > 0:
        endpoints = np.flatnonzero(degree == 1)
        if len(endpoints) > 1:
            nearby = cKDTree(nodes[endpoints]).query_ball_point(nodes[endpoints], snap_tolerance)
            used = set()
            for i, neighbors in enumerate(nearby):
                others = [j for j in neighbors if j != i]
                if len(others) != 1:
                    continue
                j = others[0]
                if i in used or j in used or len(nearby[j]) != 2:
                    continue
                a, b = endpoints[i], endpoints[j]
                if components[a] == components[b]:
                    continue
                # Only mutual unambiguous degree-one endpoints; do not weld forks.
                nodes[b] = nodes[a]
                used.update((i, j))
                snapped += 1
            lines = nodes[edges]
    nodes, edges, degree, components = line_graph(lines)
    branches = _ordered_branches(nodes, edges, degree)
    # Split at angular turns. Same ID then proves a continuous path INSIDE
    # the adjacent s interval, not a distant connection elsewhere in a loop.
    edge_branch = np.full(len(edges), -1, dtype=np.int64)
    run_id = 0
    for branch in branches:
        s = to_suz(nodes[branch['nodes']], arc)[:, 0]
        previous_sign = 0
        for eid, ds in zip(branch['edges'], np.diff(s)):
            sign = int(np.sign(ds))
            if previous_sign and sign and previous_sign != sign:
                run_id += 1
            edge_branch[eid] = run_id
            if sign:
                previous_sign = sign
        run_id += 1
    return {'lines_3d': lines, 'face_ids': face_ids, 'edge_branch': edge_branch,
            'branches': [{'points_3d': nodes[b['nodes']], 'edge_indices': b['edges']} for b in branches],
            'diagnostics': {'raw_segments': len(face_ids)+len(canonical)-len(keep),
                            'clean_segments': len(lines), 'exact_duplicates': len(canonical)-len(keep),
                            'components': len(np.unique(components)), 'endpoints': int(np.sum(degree == 1)),
                            'fork_nodes': int(np.sum(degree > 2)), 'branches': len(branches),
                            'monotone_runs': run_id, 'snapped_pairs': snapped}}


def _expand_ranges(lo, hi, values):
    start = np.searchsorted(values, lo, side='left')
    stop = np.searchsorted(values, hi, side='right')
    counts = np.maximum(stop-start, 0)
    index = np.repeat(np.arange(len(lo)), counts)
    if not len(index):
        return index, index.copy()
    level = np.repeat(start, counts) + np.arange(len(index)) - np.repeat(np.cumsum(counts)-counts, counts)
    return index, level


def horizontal_crossings(clean, s_values, arc):
    """Exact Cartesian ray/segment crossing, returning s-index,u,branch,face."""
    lines = clean['lines_3d']
    if not len(lines):
        return np.empty((0, 4))
    suz = to_suz(lines, arc)
    ei, si = _expand_ranges(suz[:, :, 0].min(axis=1)-1e-10,
                            suz[:, :, 0].max(axis=1)+1e-10, s_values)
    theta = arc['angle_min'] + np.asarray(s_values)[si]/arc['radius']
    ray = np.column_stack((np.cos(theta), np.sin(theta)))
    p = lines[ei, 0, :2] - arc['center']
    d = lines[ei, 1, :2] - lines[ei, 0, :2]
    denominator = d[:, 0]*ray[:, 1] - d[:, 1]*ray[:, 0]
    numerator = p[:, 0]*ray[:, 1] - p[:, 1]*ray[:, 0]
    valid = np.abs(denominator) > 1e-14
    t = np.divide(-numerator, denominator, out=np.zeros_like(numerator), where=valid)
    point = p+t[:, None]*d
    radius = np.sum(point*ray, axis=1)
    valid &= (t >= -1e-9) & (t <= 1+1e-9) & (radius > 0)
    hits = np.column_stack((si[valid], radius[valid]-arc['radius'],
                           clean['edge_branch'][ei[valid]], clean['face_ids'][ei[valid]]))
    # Collinear ray edges describe an interval, not an isolated crossing.
    # Retain its endpoints, as vertical_crossings does for horizontal edges.
    flat = (np.abs(denominator) <= 1e-14) & (np.abs(numerator) <= 1e-10)
    if np.any(flat):
        endpoint_r = np.sum((lines[ei[flat], :, :2]-arc['center'])*ray[flat, None, :], axis=2)
        extra = np.column_stack((np.repeat(si[flat], 2), (endpoint_r-arc['radius']).ravel(),
                                 np.repeat(clean['edge_branch'][ei[flat]], 2),
                                 np.repeat(clean['face_ids'][ei[flat]], 2)))
        hits = np.vstack((hits, extra[endpoint_r.ravel() > 0]))
    if not len(hits):
        return hits
    _, keep = np.unique(np.column_stack((hits[:, 0], np.round(hits[:, 1], 7), hits[:, 2])),
                        axis=0, return_index=True)
    return hits[np.sort(keep)]


def vertical_crossings(lines, levels, arc, s):
    """Return level-index,u for every vertical branch, preserving multiple u."""
    lines = np.asarray(lines).reshape(-1, 2, 3)
    if not len(lines):
        return np.empty((0, 2))
    theta = arc['angle_min'] + s/arc['radius']
    radial = np.array([math.cos(theta), math.sin(theta)])
    r = np.sum((lines[:, :, :2]-arc['center'])*radial, axis=2)
    positive = (r > 0).all(axis=1)
    lines, r = lines[positive], r[positive]
    ei, zi = _expand_ranges(lines[:, :, 2].min(axis=1)-1e-10,
                            lines[:, :, 2].max(axis=1)+1e-10, levels)
    dz = lines[ei, 1, 2]-lines[ei, 0, 2]
    valid = np.abs(dz) > 1e-12
    t = np.divide(np.asarray(levels)[zi]-lines[ei, 0, 2], dz,
                  out=np.zeros_like(dz), where=valid)
    u = r[ei, 0]+t*(r[ei, 1]-r[ei, 0])-arc['radius']
    hits = np.column_stack((zi[valid], u[valid]))
    # A coplanar horizontal edge is interval-valued; retain both ends rather
    # than dividing by zero or silently dropping that observation.
    flat = ~valid
    if np.any(flat):
        hits = np.vstack((hits, np.column_stack((np.repeat(zi[flat], 2),
                                                (r[ei[flat]]-arc['radius']).ravel()))))
    return np.unique(np.round(hits, 8), axis=0)


def level_map(hits):
    result = {}
    if len(hits):
        for row in hits:
            result.setdefault(int(row[0]), []).append(row[1:])
    return {k: np.asarray(v) for k, v in result.items()}


def match_observations(vmap, hmap, level_count, tolerance):
    counts = Counter({k: 0 for k in ('VH_MATCH', 'V_ONLY', 'H_ONLY', 'VH_CONFLICT', 'EMPTY')})
    distances, states = [], []
    for zi in range(level_count):
        v = vmap.get(zi, np.empty((0, 1)))[:, 0]
        h = hmap.get(zi, np.empty((0, 3)))[:, 0]
        if not len(v) and not len(h):
            counts['EMPTY'] += 1
            continue
        if not len(v):
            counts['H_ONLY'] += len(h)
            states.append([zi, 'H_ONLY', 0, len(h), 0])
            continue
        if not len(h):
            counts['V_ONLY'] += len(v)
            states.append([zi, 'V_ONLY', len(v), 0, 0])
            continue
        # Geometry duplicate endpoints are one observation, not an extra branch.
        h = np.unique(np.round(h, 7))
        costs = np.abs(v[:, None]-h[None, :])
        distances.extend(costs.min(axis=1).tolist())
        candidates = sorted((float(costs[i, j]), i, j) for i, j in zip(*np.where(costs <= tolerance)))
        vi, hi = set(), set()
        for _, i, j in candidates:
            if i not in vi and j not in hi:
                vi.add(i); hi.add(j)
        matched = len(vi)
        conflicts = min(len(v)-matched, len(h)-matched)
        counts['VH_MATCH'] += matched
        counts['VH_CONFLICT'] += conflicts
        counts['V_ONLY'] += len(v)-matched-conflicts
        counts['H_ONLY'] += len(h)-matched-conflicts
        states_present = []
        if matched:
            states_present.append('VH_MATCH')
        if conflicts:
            states_present.append('VH_CONFLICT')
        if len(v)-matched-conflicts:
            states_present.append('V_ONLY')
        if len(h)-matched-conflicts:
            states_present.append('H_ONLY')
        state = '+'.join(states_present)
        states.append([zi, state, len(v), len(h), matched])
    return dict(counts), np.asarray(distances), states


def small_gap_candidates(lines, arc, s, max_gap_z=.25, max_gap_u=.25):
    nodes, edges, degree, components = line_graph(lines, decimals=6)
    endpoints = np.flatnonzero(degree == 1)
    if len(endpoints) < 2:
        return []
    suz = to_suz(nodes, arc)
    endpoints = endpoints[np.abs(suz[endpoints, 0]-s) < 1e-4]
    if len(endpoints) < 2:
        return []
    endpoint_edges = {}
    for eid, (a, b) in enumerate(edges):
        if degree[a] == 1:
            endpoint_edges[int(a)] = (eid, int(b))
        if degree[b] == 1:
            endpoint_edges[int(b)] = (eid, int(a))
    pairs = cKDTree(suz[endpoints, 1:]).query_pairs(math.hypot(max_gap_u, max_gap_z))
    candidates = []
    for ia, ib in sorted(pairs):
        a, b = int(endpoints[ia]), int(endpoints[ib])
        if components[a] == components[b]:
            continue
        if suz[a, 2] > suz[b, 2]:
            a, b = b, a
        dz, du = suz[b, 2]-suz[a, 2], abs(suz[b, 1]-suz[a, 1])
        if not (1e-6 < dz <= max_gap_z and du <= max_gap_u):
            continue
        na, nb = endpoint_edges[a][1], endpoint_edges[b][1]
        # A genuine break has the lower contour extending downwards and the
        # upper contour upwards; lateral fingers are not missing vertical paths.
        if not (suz[na, 2] < suz[a, 2]-1e-6 and suz[nb, 2] > suz[b, 2]+1e-6):
            continue
        candidates.append({'lower': nodes[a], 'upper': nodes[b],
                           'lower_uz': suz[a, 1:], 'upper_uz': suz[b, 1:],
                           'lower_neighbor_uz': suz[na, 1:], 'upper_neighbor_uz': suz[nb, 1:],
                           'gap_z': float(dz), 'gap_u': float(du),
                           'components': (int(components[a]), int(components[b]))})
    return candidates


def supported_gap(candidate, hmap, vmap, levels, tolerance, min_support=.75):
    lo, hi = candidate['lower_uz'], candidate['upper_uz']
    indices = np.flatnonzero((levels > lo[1]+1e-8) & (levels < hi[1]-1e-8))
    if len(indices) < 2:
        return {'supported': False, 'reason': 'insufficient_horizontal_levels', 'levels': len(indices)}
    faces, support, previous_u = set(), 0, lo[0]
    for zi in indices:
        fraction = (levels[zi]-lo[1])/(hi[1]-lo[1])
        expected = lo[0] + fraction*(hi[0]-lo[0])
        h = hmap.get(int(zi), np.empty((0, 3)))
        v = vmap.get(int(zi), np.empty((0, 1)))
        if len(v) and np.min(np.abs(v[:, 0]-expected)) <= tolerance:
            return {'supported': False, 'reason': 'direct_geometry_already_present', 'levels': len(indices)}
        near = h[np.abs(h[:, 0]-expected) <= tolerance] if len(h) else h
        unique = np.unique(np.round(near[:, 0], 7)) if len(near) else np.array([])
        if len(unique) != 1:
            continue
        u = float(unique[0])
        # Both endpoint tangents must predict the observed branch. A smooth
        # interpolant alone is not evidence that two surfaces should be joined.
        stable = True
        for endpoint, neighbor in ((lo, candidate['lower_neighbor_uz']), (hi, candidate['upper_neighbor_uz'])):
            pred = endpoint[0] + (levels[zi]-endpoint[1])*(endpoint[0]-neighbor[0])/(endpoint[1]-neighbor[1])
            stable &= abs(pred-u) <= tolerance
        if not stable or abs(u-previous_u) > max(tolerance*2, candidate['gap_u']):
            continue
        faces.update(int(f) for f in near[:, 2])
        previous_u = u
        support += 1
    return {'supported': support/len(indices) >= min_support, 'reason': 'horizontal_support',
            'levels': len(indices), 'supported_levels': support, 'fraction': support/len(indices),
            'face_ids': sorted(faces)}


def traceable_repair(candidate, recovered_lines, recovered_faces, existing_lines):
    """Accept only a unique, original-mesh chain between the two endpoints.

    The caller supplies exact plane cuts of horizontally supporting FaceIDs.
    Interpolated links, partial chains, forks and already-present segments are
    rejected. Returned FaceIDs belong to the same filtered mesh as the baseline.
    """
    lines = np.asarray(recovered_lines).reshape(-1, 2, 3)
    faces = np.asarray(recovered_faces, dtype=np.int64)
    lower, upper = np.round(candidate['lower'], 6), np.round(candidate['upper'], 6)
    inside = ((lines[:, :, 2] >= lower[2]-1e-6) & (lines[:, :, 2] <= upper[2]+1e-6)).all(axis=1)
    lines, faces = lines[inside], faces[inside]
    nodes, edges, degree, components = line_graph(lines, decimals=6)
    a = np.flatnonzero((nodes == lower).all(axis=1))
    b = np.flatnonzero((nodes == upper).all(axis=1))
    if len(a) != 1 or len(b) != 1 or components[a[0]] != components[b[0]]:
        return None, 'no_complete_mesh_chain'
    component = components[a[0]]
    members = components == component
    if degree[a[0]] != 1 or degree[b[0]] != 1 or np.any(degree[members] > 2):
        return None, 'ambiguous_mesh_chain'
    selected = members[edges[:, 0]]
    lines, faces = lines[selected], faces[selected]
    def key(line):
        return tuple(sorted(tuple(p) for p in np.round(line, 6)))
    existing = {key(line) for line in existing_lines}
    missing = np.array([key(line) not in existing for line in lines], dtype=bool)
    if not np.any(missing):
        return None, 'no_missing_mesh_segments'
    return {'lines_3d': lines[missing], 'face_ids': faces[missing]}, 'traceable_mesh_chain'


def on_scanline_ray(points, arc, s, plane_tolerance=1e-5):
    """A vertical plane contains two rays; only its positive ray is observed."""
    delta = np.asarray(points)[:, :2]-arc['center']
    theta = arc['angle_min']+s/arc['radius']
    radial = delta[:, 0]*math.cos(theta)+delta[:, 1]*math.sin(theta)
    perpendicular = delta[:, 0]*math.sin(theta)-delta[:, 1]*math.cos(theta)
    return bool(len(delta) and np.all(radial > 0) and np.all(abs(perpendicular) <= plane_tolerance))


def group_level_map(group, levels):
    if not group.get('audit_observable', True):
        return {}
    uz = np.asarray(group['uz'])
    if len(uz) < 2:
        return {}
    lo, hi = np.minimum(uz[:-1, 1], uz[1:, 1]), np.maximum(uz[:-1, 1], uz[1:, 1])
    ei, zi = _expand_ranges(lo, hi, levels)
    dz = uz[ei+1, 1]-uz[ei, 1]
    ok = np.abs(dz) > 1e-12
    t = np.divide(levels[zi]-uz[ei, 1], dz, out=np.zeros_like(dz), where=ok)
    u = uz[ei, 0]+t*(uz[ei+1, 0]-uz[ei, 0])
    return level_map(np.column_stack((zi[ok], u[ok])))


def pair_metrics(left, right, lmap, rmap, hl, hr, tolerance):
    intersection = max(0., min(left['z_max'], right['z_max'])-max(left['z_min'], right['z_min']))
    union = max(left['z_max'], right['z_max'])-min(left['z_min'], right['z_min'])
    if intersection <= 0 or union <= 0:
        return None
    common = sorted(lmap.keys() & rmap.keys())
    residuals, raw_residuals, supported = [], [], 0
    for zi in common:
        lv, rv = lmap[zi][:, 0], rmap[zi][:, 0]
        raw_residuals.append(float(np.min(np.abs(lv[:, None]-rv[None, :]))))
        lh, rh = hl.get(zi, np.empty((0, 3))), hr.get(zi, np.empty((0, 3)))
        lb = {int(row[1]) for row in lh if np.min(abs(lv-row[0])) <= tolerance}
        rb = {int(row[1]) for row in rh if np.min(abs(rv-row[0])) <= tolerance}
        common_branches = lb & rb
        supported += bool(common_branches)
        branch_residuals = []
        for branch in common_branches:
            lu = np.unique(np.concatenate([lv[abs(lv-row[0]) <= tolerance] for row in lh if int(row[1]) == branch]))
            ru = np.unique(np.concatenate([rv[abs(rv-row[0]) <= tolerance] for row in rh if int(row[1]) == branch]))
            if len(lu) and len(ru):
                branch_residuals.append(float(np.min(abs(lu[:, None]-ru[None, :]))))
        if branch_residuals:
            residuals.append(min(branch_residuals))
    # Short groups below horizontal resolution have no Ch evidence. Do not
    # inflate support by removing empty/unsupported levels from the denominator.
    return {'Cz': intersection/union, 'Ch': supported/len(common) if common else None,
            'common_levels': len(common), 'supported_levels': supported,
            'median_du': float(np.median(residuals)) if residuals else None,
            'raw_nearest_median_du': float(np.median(raw_residuals)) if raw_residuals else None}


def consistency(groups_by_slice, horizontal_maps, levels, tolerance, tau_u=None):
    pairs, candidates, unmatched = [], [], 0
    keys = sorted(groups_by_slice, key=float)
    maps = {k: [group_level_map(g, levels) for g in groups_by_slice[k]] for k in keys}
    tracks = {}
    for key in keys:
        for g in groups_by_slice[key]:
            tracks[g['id']] = g['id']

    def root(x):
        while tracks[x] != x:
            tracks[x] = tracks[tracks[x]]
            x = tracks[x]
        return x

    touched = set()
    for ka, kb in zip(keys, keys[1:]):
        left, right = groups_by_slice[ka], groups_by_slice[kb]
        pool = []
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                m = pair_metrics(a, b, maps[ka][i], maps[kb][j],
                                 horizontal_maps[ka], horizontal_maps[kb], tolerance)
                if m is not None:
                    row = {'left_id': a['id'], 'right_id': b['id'], 's_left': float(ka), 's_right': float(kb), **m}
                    candidates.append(row)
                    # Evidence-based matching allows unmatched. Ch=0 and no
                    # sampled levels never establish an adjacency relation.
                    if m['Ch'] is not None and m['Ch'] >= .5:
                        pool.append((i, j, row))
        pool.sort(key=lambda x: (-x[2]['Ch'], -x[2]['Cz'], x[2]['median_du']))
        used_l, used_r = set(), set()
        for i, j, row in pool:
            if i in used_l or j in used_r:
                continue
            used_l.add(i); used_r.add(j)
            pairs.append(row)
            touched.update((row['left_id'], row['right_id']))
            tracks[root(row['right_id'])] = root(row['left_id'])
        unmatched += len(left)+len(right)-len(used_l)-len(used_r)
    if tau_u is None:
        values = [r['median_du'] for r in pairs if r['median_du'] is not None and r['median_du'] > 0]
        tau_u = max(float(np.median(values)) if values else tolerance, 1e-6)
    for row in candidates:
        row['C'] = (.5*row['Ch'] + .3*row['Cz'] + .2*math.exp(-row['median_du']/tau_u)
                    if row['Ch'] is not None and row['median_du'] is not None else None)
    track_groups = defaultdict(list)
    byid = {g['id']: g for gs in groups_by_slice.values() for g in gs}
    for gid in tracks:
        track_groups[root(gid)].append(byid[gid])
    track_rows = []
    for tid, gs in track_groups.items():
        ss = [g['s'] for g in gs]
        track_rows.append({'track_id': tid, 'slice_count': len(set(ss)), 'group_count': len(gs),
                           'span_m': max(ss)-min(ss), 'max_height_m': max(g['z_max']-g['z_min'] for g in gs),
                           'length_sum_m': sum(g['length_2d'] for g in gs),
                           'group_ids': [g['id'] for g in gs]})
    return {'pairs': pairs, 'candidates': candidates, 'tracks': track_rows, 'tau_u': tau_u,
            'unmatched_group_adjacencies': unmatched, 'isolated_groups': len(byid)-len(touched),
            'group_count': len(byid)}
