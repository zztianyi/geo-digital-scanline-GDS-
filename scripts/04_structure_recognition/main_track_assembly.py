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


def _nearest_join(left, right, required, locked_range, direction):
    """Exact segment intersections and four endpoint-to-segment projections.

Search all eligible edge interiors, in bounded-size arrays. Retained prefixes
and suffixes must preserve the old elevation extent and extend one frontier.
No monotonic-Z requirement is imposed on the measured curves.
"""
    pa, pb = _points(left), _points(right)
    da, db = np.diff(pa, axis=0), np.diff(pb, axis=0)
    la, lb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
    ca, cb = np.r_[0., np.cumsum(la)], np.r_[0., np.cumsum(lb)]
    # Only the current terminal is already accepted geometry. A continuation
    # may trim its tail, but cannot erase an accepted fold on the way there.
    folded_a, folded_b = np.flatnonzero(da[:, 1] < -EPS), np.flatnonzero(db[:, 1] < -EPS)
    protected_a = ca[folded_a[-1]+1] if direction == 'upper' and len(folded_a) else 0.
    protected_b = cb[folded_b[0]] if direction == 'lower' and len(folded_b) else cb[-1]
    amin, amax = np.minimum.accumulate(pa[:, 1]), np.maximum.accumulate(pa[:, 1])
    bmin = np.minimum.accumulate(pb[::-1, 1])[::-1]
    bmax = np.maximum.accumulate(pb[::-1, 1])[::-1]
    best = None
    for start in range(0, len(da), 64):
        end = min(start+64, len(da))
        p, r = pa[start:end, None, :], da[start:end, None, :]
        q, s = pb[None, :-1, :], db[None, :, :]
        rr, ss = np.sum(r*r, axis=2), np.sum(s*s, axis=2)
        denominator = _cross(r, s)
        nonparallel = np.abs(denominator) > 1e-15
        ta = np.divide(_cross(q-p, s), denominator, out=np.zeros_like(denominator), where=nonparallel)
        tb = np.divide(_cross(q-p, r), denominator, out=np.zeros_like(denominator), where=nonparallel)
        candidates = [(ta, tb, nonparallel & (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1))]
        shape = denominator.shape
        for fixed in (0., 1.):
            on_b = np.clip(np.sum((p+fixed*r-q)*s, axis=2)/np.maximum(ss, 1e-30), 0., 1.)
            on_a = np.clip(np.sum((q+fixed*s-p)*r, axis=2)/np.maximum(rr, 1e-30), 0., 1.)
            candidates.extend(((np.full(shape, fixed), on_b, np.ones(shape, bool)),
                               (on_a, np.full(shape, fixed), np.ones(shape, bool))))
        for ta, tb, valid in candidates:
            qa, qb = p+ta[..., None]*r, q+tb[..., None]*s
            apos = ca[start:end, None]+ta*la[start:end, None]
            bpos = cb[None, :-1]+tb*lb[None, :]
            # Both pieces must contribute observed geometry.
            valid = valid & (apos > EPS) & (cb[-1]-bpos > EPS)
            valid &= (apos >= protected_a-EPS) & (bpos <= protected_b+EPS)
            lo = np.minimum(np.minimum(amin[start:end, None], qa[..., 1]),
                            np.minimum(bmin[None, 1:], qb[..., 1]))
            hi = np.maximum(np.maximum(amax[start:end, None], qa[..., 1]),
                            np.maximum(bmax[None, 1:], qb[..., 1]))
            lo, hi = np.minimum(lo, locked_range[0]), np.maximum(hi, locked_range[1])
            valid &= (lo <= required[0]+EPS) & (hi >= required[1]-EPS)
            valid &= (lo < required[0]-1e-6) if direction == 'lower' else (hi > required[1]+1e-6)
            if not valid.any():
                continue
            distance = np.linalg.norm(qa-qb, axis=2)
            # Coincident/intersecting geometry first; otherwise true minimum gap.
            distance_key = np.where(distance < EPS, 0., distance)
            removed = bpos if direction == 'lower' else ca[-1]-apos
            flat = np.flatnonzero(valid)
            order = np.lexsort((flat, removed.ravel()[flat], distance_key.ravel()[flat]))
            k = flat[order[0]]
            i, j = np.unravel_index(k, shape)
            rank = (float(distance_key[i, j]), float(removed[i, j]))
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
        tangent_turn_deg=turn, frontier=direction, removed_terminal_length_m=rank[1],
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


def assemble_main_track(branches, initial_route, *, anchor_branch_id, review_gap_m=.01):
    """Continue beyond each end; a lone local candidate does not end the scan.

Only branches with new elevation extent are continuations. Fully overlapping
alternatives cannot replace a complete measured fold just because they are
smoother. Long shortest connectors remain visible and explicitly need review.
"""
    from dominant_observed_branch import route_result
    branches = [b for b in branches if b['kind'] != 'CLOSED_COMPONENT' and b['full_arc_length'] > 1e-6]
    records = [dict(r) for r in initial_route['path_edges']]
    junctions = list(initial_route.get('junctions', []))
    if not junctions and initial_route.get('junction'):
        junctions = [initial_route['junction']]
    if not records:
        return initial_route
    # Orient a single seed once, preserving all of its internal folds.
    if not junctions:
        records = _oriented(records)
    attempts = []
    while True:
        old_points = _points(records)
        required = [float(old_points[:, 1].min()), float(old_points[:, 1].max())]
        pieces = _remaining(branches, records)
        choices = []
        for direction in ('lower', 'upper'):
            terminal = []
            for r in (records if direction == 'lower' else records[::-1]):
                if not r['source'].startswith('OBSERVED') or (terminal and r['branch_id'] != terminal[0]['branch_id']):
                    break
                terminal.append(r)
            if direction == 'upper':
                terminal.reverse()
            if not terminal:
                continue
            locked = records[len(terminal):] if direction == 'lower' else records[:-len(terminal)]
            locked_z = [p[1] for r in locked for p in r['points_uz']]
            locked_range = (min(locked_z), max(locked_z)) if locked_z else (np.inf, -np.inf)
            count = 0
            for piece in pieces:
                p = _points(piece)
                if direction == 'lower' and p[:, 1].min() >= required[0]-1e-6:
                    continue
                if direction == 'upper' and p[:, 1].max() <= required[1]+1e-6:
                    continue
                count += 1
                left, right = (piece, terminal) if direction == 'lower' else (terminal, piece)
                join = _nearest_join(left, right, required, locked_range, direction)
                if join is None:
                    continue
                rank = (0. if join['distance_m'] < EPS else join['distance_m'],
                        join['removed_terminal_length_m'], join['from_branch_id'], join['to_branch_id'])
                choices.append((rank, left, right, locked, join))
            attempts.append(dict(frontier=direction, current_z_range=required, extending_piece_count=count))
        if not choices:
            break
        _, left, right, locked, join = min(choices, key=lambda c: c[0])
        extension = _splice(left, right, join)
        records = extension+locked if join['frontier'] == 'lower' else locked+extension
        join['needs_manual_confirmation'] = join['xyz_distance_m'] > review_gap_m
        if join['frontier'] == 'lower':
            junctions.insert(0, join)
        else:
            junctions.append(join)
        # Every step must extend the observed extent; never silently stop at a
        # branch-count cap. This guard indicates an implementation error only.
        if len(junctions) > 2*sum(len(b['records']) for b in branches):
            raise RuntimeError('Main-track continuation failed to make finite progress')
    for r in records:
        if r['source'].startswith('OBSERVED'):
            r['source'] = 'OBSERVED_DOMINANT' if r['branch_id'] == anchor_branch_id else 'OBSERVED_SECONDARY'
    switches = sum(j['from_branch_id'] != j['to_branch_id'] for j in junctions)
    result = route_result(records, branch_switch_count=switches, junction=junctions[0] if junctions else None)
    extent = [min(b['z_range'][0] for b in branches), max(b['z_range'][1] for b in branches)]
    actual = [float(result['curve_uz'][:, 1].min()), float(result['curve_uz'][:, 1].max())]
    sequence = []
    for r in records:
        if r['source'].startswith('OBSERVED') and (not sequence or sequence[-1] != r['branch_id']):
            sequence.append(r['branch_id'])
    return {**result, 'junctions': junctions, 'junction_count': len(junctions), 'route_branch_sequence': sequence,
        'observed_low_confidence_tail_removed': initial_route.get('observed_low_confidence_tail_removed', 0.),
        'low_confidence_tail_metric_scope': 'initial_confidence_crossover_only',
        'virtual_junction_count': sum(j['new_virtual_nodes'] for j in junctions),
        'continuation_attempts': attempts, 'candidate_z_extent': extent, 'route_z_extent': actual,
        'extent_covered': actual[0] <= extent[0]+1e-6 and actual[1] >= extent[1]-1e-6,
        'review_gap_m': review_gap_m, 'large_connector_count': sum(j['xyz_distance_m'] > review_gap_m for j in junctions)}
