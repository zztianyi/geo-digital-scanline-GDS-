"""Observed-first local graphs with explicit missing-interval ownership.

This module returns proposals; it never edits the source mesh, input profile or
production recognition output. A coverage diagnosis is not repair approval.
"""
from __future__ import annotations

import copy
import heapq
import itertools
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from local_orthogonal_cooperative_constraint import solve_locc


def _union(intervals, epsilon=1e-9):
    result = []
    for a, b in sorted(intervals):
        if b-a <= epsilon:
            continue
        if result and a <= result[-1][1]+epsilon:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([float(a), float(b)])
    return result


def _missing(observed, low, high):
    gaps, start = [], low
    for a, b in observed:
        if a > start+1e-9:
            gaps.append([float(start), float(a)])
        start = max(start, b)
    if high > start+1e-9:
        gaps.append([float(start), float(high)])
    return gaps


def _closest_node(uz, point, tolerance=3e-6):
    if not len(uz):
        return None
    distances = np.linalg.norm(uz-np.asarray(point), axis=1)
    index = int(np.argmin(distances))
    return index if distances[index] <= tolerance else None


def _path(records, start, stop):
    """Lexicographic costs make observed ownership hard, not a soft weight."""
    adjacency = defaultdict(list)
    for i, record in enumerate(records):
        a, b = record['nodes']
        adjacency[a].append((b, i))
        adjacency[b].append((a, i))
    if start == stop:
        return []
    sequence = itertools.count()
    best, previous = {start: (0., 0., 0.)}, {}
    queue = [((0., 0., 0.), next(sequence), start)]
    while queue:
        cost, _, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        if node == stop:
            route = []
            while node != start:
                parent, index = previous[node]
                route.append((index, records[index]['nodes'][0] == parent))
                node = parent
            return route[::-1]
        for other, index in adjacency[node]:
            record = records[index]
            length = float(np.linalg.norm(np.diff(record['points_uz'], axis=0), axis=1).sum())
            kind = record['source']
            increment = (length if kind not in ('OBSERVED_VERTICAL', 'TOPOLOGY_STITCH') else 0.,
                         length if kind == 'TOPOLOGY_STITCH' else 0., length)
            proposed = tuple(a+b for a, b in zip(cost, increment))
            if other not in best or proposed < best[other]:
                best[other], previous[other] = proposed, (node, index)
                heapq.heappush(queue, (proposed, next(sequence), other))
    return None


def _observed_records(profile, uz, edge_ids):
    return [{'nodes': (int(profile.edges[eid, 0]), int(profile.edges[eid, 1])),
             'points_uz': uz[profile.edges[eid]].tolist(), 'source': 'OBSERVED_VERTICAL',
             'edge_id': int(eid), 'face_id': profile.first_face_ids[eid],
             'source_face_ids': list(profile.source_face_ids[eid]),
             'source_segment_indices': list(profile.source_segment_indices[eid])} for eid in edge_ids]


def _route_geometry(records, route):
    pieces, curve = [], []
    for index, forward in route:
        piece = copy.deepcopy(records[index])
        if not forward:
            piece['points_uz'] = piece['points_uz'][::-1]
            piece['nodes'] = piece['nodes'][::-1]
        pieces.append(piece)
        points = piece['points_uz']
        curve.extend(points if not curve else points[1:])
    return pieces, np.asarray(curve, dtype=float).reshape(-1, 2)


def _coverage_pieces(uz, edges, guide, low, high, width):
    """Exact Z intervals of intersection with a piecewise-linear U corridor.

    Original edge vertices are not moved. Cut points describe coverage only,
    and are never used as new topology anchors unless they are original nodes.
    """
    intervals, eligible = [], set()
    for eid, (ia, ib) in enumerate(edges):
        a, b = uz[ia], uz[ib]
        if abs(b[1]-a[1]) < 1e-12:
            if low-1e-8 <= a[1] <= high+1e-8:
                center = np.interp(a[1], guide[:, 1], guide[:, 0])
                if min(a[0], b[0]) <= center+width and max(a[0], b[0]) >= center-width:
                    eligible.add(eid)
            continue
        left, right = max(low, min(a[1], b[1])), min(high, max(a[1], b[1]))
        if right < left:
            continue
        zs = np.unique(np.r_[left, guide[(guide[:, 1] > left) & (guide[:, 1] < right), 1], right])
        if len(zs) == 1:
            u = a[0]+(zs[0]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if abs(u-np.interp(zs[0], guide[:, 1], guide[:, 0])) <= width:
                eligible.add(eid)
        for z0, z1 in zip(zs, zs[1:]):
            us = a[0]+(np.array([z0, z1])-a[1])*(b[0]-a[0])/(b[1]-a[1])
            delta = us-np.interp([z0, z1], guide[:, 1], guide[:, 0])
            if abs(delta[1]-delta[0]) < 1e-14:
                if abs(delta[0]) <= width:
                    intervals.append([z0, z1]); eligible.add(eid)
                continue
            t = sorted(((-width-delta[0])/(delta[1]-delta[0]), (width-delta[0])/(delta[1]-delta[0])))
            t0, t1 = max(0., t[0]), min(1., t[1])
            if t1 >= t0:
                eligible.add(eid)
                if t1 > t0:
                    intervals.append([z0+t0*(z1-z0), z0+t1*(z1-z0)])
    return _union(intervals), sorted(eligible)


def analyze_vertical_candidate(profile, node_uz, window, corridor_half_width=.03):
    uz = np.asarray(node_uz, dtype=float).reshape(-1, 2)
    if len(uz) != len(profile.nodes) or not np.isfinite(uz).all():
        raise ValueError('Finite UZ coordinates must align with canonical nodes')
    if not np.isfinite(corridor_half_width) or corridor_half_width <= 0:
        raise ValueError('Corridor width must be positive')
    lower, upper = np.asarray(window['lower_uz']), np.asarray(window['upper_uz'])
    low, high = float(lower[1]), float(upper[1])
    if high <= low:
        raise ValueError('Candidate upper height must exceed lower height')
    start, stop = _closest_node(uz, lower), _closest_node(uz, upper)
    direct = None
    if start is not None and stop is not None and profile.components[start] == profile.components[stop]:
        observed = _observed_records(profile, uz, range(len(profile.edges)))
        direct = _path(observed, start, stop)
    if direct is not None:
        pieces, guide = _route_geometry(observed, direct)
        edge_ids = [piece['edge_id'] for piece in pieces]
        coverage, missing, source = [[low, high]], [], 'OBSERVED_VERTICAL_PATH'
    else:
        has_h = any(layer.get('horizontal') for layer in window['layers'])
        guide = solve_locc(window, mode='HORIZONTAL_ONLY' if has_h else 'LOCC')['curve_uz']
        coverage, edge_ids = _coverage_pieces(uz, profile.edges, guide, low, high, corridor_half_width)
        missing = _missing(coverage, low, high)
        source = 'HORIZONTAL_MESH_CORRIDOR' if has_h else 'NEIGHBOR_OR_ENDPOINT_CORRIDOR'
    ratio = sum(b-a for a, b in coverage)/(high-low)
    # Type I tolerates only microscopic coverage gaps; reconstruction still
    # requires a supported graph path and never invents coordinates for Type I.
    max_gap = max((b-a for a, b in missing), default=0.)
    kind = ('TYPE_I' if ratio >= 1-1e-7 or ratio >= .995 and max_gap <= 1e-4
            else 'TYPE_III' if ratio <= 1e-9 else 'TYPE_II')
    return {'gap_type': kind, 'coverage_ratio': float(ratio), 'observed_intervals': coverage,
            'missing_intervals': missing, 'observed_edge_ids': edge_ids, 'guide_curve_uz': np.asarray(guide).tolist(),
            'corridor_source': source, 'corridor_half_width_m': float(corridor_half_width),
            'start_node': start, 'stop_node': stop,
            'observed_path_connected': direct is not None,
            'boundary_candidate': not any(layer.get('horizontal') for layer in window['layers'])}


def _supported_stitches(profile, uz, eligible, window, maximum_distance):
    node_ids = sorted({int(node) for eid in eligible for node in profile.edges[eid] if profile.degree[node] == 1})
    if len(node_ids) < 2:
        return []
    incident = {}
    for eid in eligible:
        a, b = map(int, profile.edges[eid]); incident[a] = b; incident[b] = a
    guide = solve_locc(window, mode='HORIZONTAL_ONLY')['curve_uz']
    result = []
    for ia, ib in sorted(cKDTree(uz[node_ids]).query_pairs(maximum_distance)):
        a, b = node_ids[ia], node_ids[ib]
        if profile.components[a] == profile.components[b]:
            continue
        pa, pb = uz[a], uz[b]
        ta, tb = pa-uz[incident[a]], uz[incident[b]]-pb
        lengths = np.linalg.norm(ta)*np.linalg.norm(tb)
        if lengths <= 0 or np.dot(ta, tb)/lengths < .5:
            continue
        gap = pb-pa
        distance = np.linalg.norm(gap)
        if distance <= 0 or any(np.dot(t, gap)/(np.linalg.norm(t)*distance) < .5 for t in (ta, tb)):
            continue
        mid = (pa+pb)/2
        near = [layer for layer in sorted(window['layers'], key=lambda layer: abs(layer['z']-mid[1]))[:2]
                if abs(layer['z']-mid[1]) <= .05]
        h_support = (any(abs(hit['u']-mid[0]) <= .02 for layer in near for hit in layer.get('horizontal', []))
                     and abs(np.interp(mid[1], guide[:, 1], guide[:, 0])-mid[0]) <= .005)
        neighbor_sides = set()
        for layer in near:
            for observation in layer.get('neighbors', []):
                if any(abs(v-mid[0]) <= .02 for v in observation['values']):
                    neighbor_sides.add(int(np.sign(observation['s']-window['s'])))
        if not h_support or not {-1, 1}.issubset(neighbor_sides):
            continue
        result.append({'nodes': (a, b), 'points_uz': [pa.tolist(), pb.tolist()], 'source': 'TOPOLOGY_STITCH',
                       'face_id': None, 'source_face_ids': [], 'support': 'Local H corridor and both neighboring sides',
                       'coordinates_predicted': False})
    return result


def _subwindow(window, low, high, a, b, builder, tag):
    if builder is not None:
        return builder({'candidate_id': tag, 'slice_key': f"{window['s']:.2f}", 's': window['s'],
                        'lower_uz': np.asarray(a), 'upper_uz': np.asarray(b)})
    result = {k: copy.deepcopy(v) for k, v in window.items() if k != 'layers'}
    result.update(lower_uz=np.asarray(a), upper_uz=np.asarray(b))
    result['layers'] = [copy.deepcopy(layer) for layer in window['layers'] if low+1e-8 < layer['z'] < high-1e-8]
    if not result['layers']:
        for fraction in (.25, .5, .75):
            z = low+(high-low)*fraction
            neighbors = []
            for track in window.get('neighbor_surface_tracks', []):
                points = np.asarray(track['points_uz'])
                values = [float(np.interp(z, points[:, 1], points[:, 0]))] if points[0, 1] <= z <= points[-1, 1] else []
                neighbors.append({'s': track['s'], 'values': values})
            result['layers'].append({'z': z, 'horizontal': [], 'neighbors': neighbors})
    return result


def propose_topology_stitches(profile, node_uz, analysis, window, maximum_distance=1e-4):
    """Independent, cacheable topology-only stage; all coordinates observed."""
    return _supported_stitches(profile, np.asarray(node_uz), analysis['observed_edge_ids'], window, maximum_distance)


def reconstruct_observed_first(profile, node_uz, window, *, corridor_half_width=.03,
                               stitch_max_distance=1e-4, window_builder=None, candidate_id='candidate', analysis=None,
                               topology_stitches=None, allow_inference=True):
    uz = np.asarray(node_uz, dtype=float).reshape(-1, 2)
    analysis = copy.deepcopy(analysis) if analysis is not None else analyze_vertical_candidate(profile, uz, window, corridor_half_width)
    records = _observed_records(profile, uz, analysis['observed_edge_ids'])
    start = analysis['start_node'] if analysis['start_node'] is not None else ('endpoint', 'lower')
    stop = analysis['stop_node'] if analysis['stop_node'] is not None else ('endpoint', 'upper')
    stitches = (propose_topology_stitches(profile, uz, analysis, window, stitch_max_distance)
                if topology_stitches is None else copy.deepcopy(topology_stitches))
    records.extend(stitches)
    route = _path(records, start, stop)
    inferred_nodes = []
    interval_attempts = []
    if allow_inference and route is None and analysis['gap_type'] != 'TYPE_I':
        guide = np.asarray(analysis['guide_curve_uz'])
        eligible_nodes = sorted({int(n) for e in analysis['observed_edge_ids'] for n in profile.edges[e] if profile.degree[n] == 1})
        for gap_index, (low, high) in enumerate(analysis['missing_intervals']):
            def anchors(z, endpoint, anchor_key):
                if abs(z-endpoint[1]) <= 3e-6:
                    return [(anchor_key, np.asarray(endpoint))]
                expected = np.interp(z, guide[:, 1], guide[:, 0])
                return [(n, uz[n]) for n in eligible_nodes if abs(uz[n, 1]-z) <= 3e-6 and abs(uz[n, 0]-expected) <= corridor_half_width]
            lower = anchors(low, window['lower_uz'], start)
            upper = anchors(high, window['upper_uz'], stop)
            if not lower or not upper:
                interval_attempts.append({'interval': [low, high], 'status': 'UNRESOLVED_NO_OBSERVED_ENDPOINT'})
                continue
            for ai, (node_a, a) in enumerate(lower):
                for bi, (node_b, b) in enumerate(upper):
                    if b[1] <= a[1]+1e-9:
                        continue
                    # Do not bridge over an observed interval, even if that
                    # interval belongs to a disconnected competing component.
                    overlap = sum(max(0., min(b[1], y)-max(a[1], x)) for x, y in analysis['observed_intervals'])
                    if overlap > 1e-8:
                        interval_attempts.append({'interval': [low, high], 'status': 'BLOCKED_OBSERVED_OVERLAP'})
                        continue
                    tag = f'{candidate_id}:missing:{gap_index}:{ai}:{bi}'
                    sub = _subwindow(window, a[1], b[1], a, b, window_builder, tag)
                    slope = (b[0]-a[0])/(b[1]-a[1])
                    for name, node, point in (('lower_tangent', node_a, a), ('upper_tangent', node_b, b)):
                        tangent = slope
                        if isinstance(node, (int, np.integer)) and profile.degree[node] == 1:
                            edge = profile.edges[np.flatnonzero(np.any(profile.edges == node, axis=1))[0]]
                            other = uz[int(edge[1] if edge[0] == node else edge[0])]
                            if abs(point[1]-other[1]) > 1e-9:
                                tangent = float((point[0]-other[0])/(point[1]-other[1]))
                        elif abs(point[1]-window['lower_uz'][1]) <= 1e-8:
                            tangent = window.get('lower_tangent', slope)
                        elif abs(point[1]-window['upper_uz'][1]) <= 1e-8:
                            tangent = window.get('upper_tangent', slope)
                        sub[name] = tangent
                    solution = solve_locc(sub)
                    selected = solution['selected_nodes']
                    coordinates = np.asarray(solution['curve_uz'])
                    node_keys = [node_a]+[(tag, i) for i in range(1, len(coordinates)-1)]+[node_b]
                    for i, (p, q) in enumerate(zip(coordinates, coordinates[1:])):
                        nearby = [n['source'] for n in selected[max(1, i):min(len(selected)-1, i+2)]]
                        source = ('ORTHOGONAL_INFERRED' if 'HORIZONTAL_MESH' in nearby else
                                  'NEIGHBOR_INFERRED' if 'V_NEIGHBOR_PREDICTED' in nearby else 'VERTICAL_PRIOR_FALLBACK')
                        records.append({'nodes': (node_keys[i], node_keys[i+1]), 'points_uz': [p.tolist(), q.tolist()],
                                        'source': source, 'face_id': None, 'source_face_ids': [],
                                        'missing_interval': [float(a[1]), float(b[1])], 'solve_tag': tag})
                    inferred_nodes.extend({**n, 'solve_tag': tag} for n in selected[1:-1])
                    interval_attempts.append({'interval': [low, high], 'status': 'SOLVED', 'solve_tag': tag})
        route = _path(records, start, stop)
    if route is None:
        pieces, curve = [], np.empty((0, 2))
    else:
        pieces, curve = _route_geometry(records, route)
    inferred = [piece for piece in pieces if piece['source'] not in ('OBSERVED_VERTICAL', 'TOPOLOGY_STITCH')]
    topology = [piece for piece in pieces if piece['source'] == 'TOPOLOGY_STITCH']
    tags = {piece['solve_tag'] for piece in inferred}
    inferred_nodes = [n for n in inferred_nodes if n['solve_tag'] in tags]
    # Type I never generates inferred geometry, including the nearly complete
    # coverage population. Unresolved topology stays unresolved.
    if analysis['gap_type'] == 'TYPE_I' and inferred:
        raise AssertionError('Type I must not infer coordinates')
    return {**analysis, 'status': 'RESOLVED_PROPOSAL' if route is not None else 'UNRESOLVED_TOPOLOGY_OR_CORRIDOR',
            'curve_uz': curve, 'path_edges': pieces, 'observed_edge_ids': analysis['observed_edge_ids'],
            'inferred_edges': inferred, 'topology_stitches': topology, 'inferred_nodes': inferred_nodes,
            'inferred_length_m': float(sum(np.linalg.norm(np.diff(e['points_uz'], axis=0), axis=1).sum() for e in inferred)),
            'topology_stitch_length_m': float(sum(np.linalg.norm(np.diff(e['points_uz'], axis=0), axis=1).sum() for e in topology)),
            'overwritten_observed_geometry': 0, 'interval_attempts': interval_attempts,
            'analysis_only': True, 'provenance': 'Observed edges are immutable; inferred edges have no FaceID'}


def joint_endpoint_pairing(candidates, minimum_score=.5, ambiguity_margin=.03):
    """Maximum total evidence matching with explicit unmatched dummy nodes."""
    if not candidates:
        return []
    lower = list(dict.fromkeys(r['lower_key'] for r in candidates))
    upper = list(dict.fromkeys(r['upper_key'] for r in candidates))
    li, ui = {k: i for i, k in enumerate(lower)}, {k: i for i, k in enumerate(upper)}
    benefit = np.zeros((len(lower), len(upper)+len(lower)))
    benefit[:, :len(upper)] = -1e6
    best_at = {}
    for i, row in enumerate(candidates):
        score = float(row['pair_score'])
        if not np.isfinite(score):
            raise ValueError('Pair evidence must be finite')
        slot = li[row['lower_key']], ui[row['upper_key']]
        value = score-minimum_score
        if slot not in best_at or value > benefit[slot]:
            benefit[slot], best_at[slot] = value, i
    rr, cc = linear_sum_assignment(-benefit)
    chosen = {best_at[(r, c)] for r, c in zip(rr, cc) if c < len(upper) and benefit[r, c] > 0}
    result = []
    for i, row in enumerate(candidates):
        competitors = [other['pair_score'] for j, other in enumerate(candidates) if j != i and
                       (other['lower_key'] == row['lower_key'] or other['upper_key'] == row['upper_key'])]
        margin = float(row['pair_score']-max(competitors)) if competitors else None
        ambiguous = margin is not None and margin <= ambiguity_margin
        result.append({**row, 'selected': i in chosen, 'pairing_ambiguous': ambiguous,
                       'pairing_margin': margin,
                       'pairing_status': ('SELECTED_REVIEW_AMBIGUOUS' if ambiguous else 'SELECTED_ONE_TO_ONE') if i in chosen
                                         else 'UNMATCHED_OR_ALTERNATIVE'})
    return result
