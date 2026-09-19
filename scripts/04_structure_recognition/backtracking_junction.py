"""Polyline junction proposals with exact on-edge virtual nodes.

No observed coordinates are replaced. Close points remain TWO on-edge nodes
joined by an explicitly synthetic connector, never an averaged snap point.
"""
from __future__ import annotations
import numpy as np


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def _segment_candidates(a, b, c, d):
    u, v = b-a, d-c
    denominator = _cross(u, v)
    if abs(denominator) > 1e-16:
        t, q = _cross(c-a, v)/denominator, _cross(c-a, u)/denominator
        if -1e-10 <= t <= 1+1e-10 and -1e-10 <= q <= 1+1e-10:
            yield float(np.clip(t, 0, 1)), float(np.clip(q, 0, 1))
    for t in (0., 1.):
        point = a+t*u
        q = np.clip(np.dot(point-c, v)/np.dot(v, v), 0, 1) if np.dot(v, v) > 0 else 0.
        yield t, float(q)
    for q in (0., 1.):
        point = c+q*v
        t = np.clip(np.dot(point-a, u)/np.dot(u, u), 0, 1) if np.dot(u, u) > 0 else 0.
        yield float(t), q


def _angle(a, b):
    divisor = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/divisor, -1, 1)))) if divisor > 1e-14 else None


def search_backtracking_junction(a, b, *, maximum_distance=.01, backtrack_length=.2):
    """Search last/first arc-length windows, not just endpoint-to-endpoint.

    The distance bound is a proposal gate, NOT a tolerance for merging nodes.
    REAL_INTERSECTION here means in the UZ plane; XYZ separation is also
    returned to make round6 off-plane deviations visible to the caller.
    """
    if maximum_distance < 0 or backtrack_length <= 0:
        raise ValueError('Junction search bounds must be nonnegative/positive')
    pa, pb = np.asarray(a['points_uz']), np.asarray(b['points_uz'])
    la, lb = np.linalg.norm(np.diff(pa, axis=0), axis=1), np.linalg.norm(np.diff(pb, axis=0), axis=1)
    ca, cb = np.r_[0., np.cumsum(la)], np.r_[0., np.cumsum(lb)]
    candidates, seen = [], set()
    for i in np.flatnonzero((ca[-1]-ca[:-1] <= backtrack_length+la+1e-12) & (la > 1e-12)):
        amin = max(0., (ca[-1]-backtrack_length-ca[i])/la[i])
        aa, ab = pa[i]+amin*(pa[i+1]-pa[i]), pa[i+1]
        for j in np.flatnonzero((cb[:-1] <= backtrack_length+1e-12) & (lb > 1e-12)):
            bmax = min(1., (backtrack_length-cb[j])/lb[j])
            ba, bb = pb[j], pb[j]+bmax*(pb[j+1]-pb[j])
            gap_box = np.maximum(0., np.maximum(np.minimum(aa, ab)-np.maximum(ba, bb),
                                                 np.minimum(ba, bb)-np.maximum(aa, ab)))
            if np.linalg.norm(gap_box) > maximum_distance:
                continue
            for ta, tb in _segment_candidates(aa, ab, ba, bb):
                ta, tb = amin+(1-amin)*ta, bmax*tb
                qa, qb = pa[i]+ta*(pa[i+1]-pa[i]), pb[j]+tb*(pb[j+1]-pb[j])
                distance = float(np.linalg.norm(qa-qb))
                if distance > maximum_distance+1e-12:
                    continue
                signature = (i, j, round(ta, 10), round(tb, 10))
                if signature in seen:
                    continue
                seen.add(signature)
                virtual = [1e-9 < t < 1-1e-9 for t in (ta, tb)]
                kind = ('REAL_INTERSECTION' if distance < 1e-10 else
                        'CLOSEST_POINT_PAIR' if all(virtual) else
                        'EDGE_INTERIOR_PROJECTION' if any(virtual) else 'ENDPOINT_CONNECTOR')
                xa = np.asarray(a['records'][i]['points_xyz'])
                xb = np.asarray(b['records'][j]['points_xyz'])
                xa, xb = xa[0]+ta*(xa[1]-xa[0]), xb[0]+tb*(xb[1]-xb[0])
                direct = _angle(pa[-1]-pa[-2], pb[1]-pb[0])
                turn = _angle(pa[i+1]-pa[i], pb[j+1]-pb[j])
                candidates.append(dict(junction_type=kind, a_edge_index=int(i), b_edge_index=int(j),
                    a_edge_id=a['records'][i]['edge_id'], b_edge_id=b['records'][j]['edge_id'],
                    a_t=float(ta), b_t=float(tb), a_point_uz=qa.tolist(), b_point_uz=qb.tolist(),
                    a_point_xyz=xa.tolist(), b_point_xyz=xb.tolist(), distance_m=distance,
                    xyz_distance_m=float(np.linalg.norm(xa-xb)),
                    A_backtrack_length_m=float(ca[-1]-ca[i]-ta*la[i]),
                    B_entry_position_m=float(cb[j]+tb*lb[j]), new_virtual_nodes=sum(virtual),
                    tangent_turn_deg=turn, endpoint_tangent_turn_deg=direct,
                    needs_manual_confirmation=True))
    priorities = {'REAL_INTERSECTION': 0, 'CLOSEST_POINT_PAIR': 1,
                  'EDGE_INTERIOR_PROJECTION': 2, 'ENDPOINT_CONNECTOR': 3}
    candidates.sort(key=lambda c: (priorities[c['junction_type']], c['distance_m'],
                    c['tangent_turn_deg'] if c['tangent_turn_deg'] is not None else 180., c['A_backtrack_length_m']))
    return candidates[:8]


def build_switch_route(a, b, junction, *, dominant_branch_id=None):
    from dominant_observed_branch import trim_record, route_result
    i, j = junction['a_edge_index'], junction['b_edge_index']
    left = [dict(e) for e in a['records'][:i]]+[trim_record(a['records'][i], 0., junction['a_t'])]
    right = [trim_record(b['records'][j], junction['b_t'], 1.)]+[dict(e) for e in b['records'][j+1:]]
    left = [e for e in left if np.linalg.norm(np.diff(e['points_uz'], axis=0)) > 1e-12]
    right = [e for e in right if np.linalg.norm(np.diff(e['points_uz'], axis=0)) > 1e-12]
    dominant_branch_id = a['branch_id'] if dominant_branch_id is None else dominant_branch_id
    for e in left+right:
        e['source'] = 'OBSERVED_DOMINANT' if e['branch_id'] == dominant_branch_id else 'OBSERVED_SECONDARY'
    connector = dict(source='TOPOLOGY_SWITCH', face_id=None, source_face_ids=[], source_segment_indices=[],
        branch_id=None, points_uz=[junction['a_point_uz'], junction['b_point_uz']],
        points_xyz=[junction['a_point_xyz'], junction['b_point_xyz']])
    return route_result(left+[connector]+right, branch_switch_count=1, junction=junction)
