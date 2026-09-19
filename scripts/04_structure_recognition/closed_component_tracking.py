"""Preserve canonical closed 2D components and conservatively track scanlines.

No geometry is inserted, snapped, or edited. A closed outline or even a stable
track is evidence about observed curves, not automatic rock/block identity.
Horizontal matches must be supplied by the caller from real horizontal cuts.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def analyze_closed_components(profile, node_uz, *, slice_key, s):
    """Return whole degree-two components, following canonical branch order.

    ``node_uz`` is an (N, 2) array aligned to ``profile.nodes``. Loop arrays
    repeat the original start node at the end; no closure edge is fabricated.
    Per-edge provenance arrays align with ``edge_ids``. Union fields retain
    first encounter order. All output containers are detached from the inputs.
    Degree classification and traversal cost O(nodes + edges), not O(C * E).
    """
    uz = np.asarray(node_uz, dtype=float)
    if uz.shape != (len(profile.nodes), 2) or not np.isfinite(uz).all():
        raise ValueError('node_uz must be finite with shape (N, 2)')
    if not math.isfinite(s):
        raise ValueError('s must be finite')
    nonclosed = {int(component) for component, degree
                 in zip(profile.components, profile.degree) if degree != 2}
    records = []
    for branch in profile.branches:
        component = int(branch['component_id'])
        if component in nonclosed:
            continue
        node_ids = list(branch['nodes'])
        edge_ids = list(branch['edges'])
        if not edge_ids or node_ids[0] != node_ids[-1]:
            continue
        loop = uz[node_ids]
        # Translate before products: geographic/global coordinates otherwise
        # lose small polygon areas through cancellation of large products.
        origin = loop[0]
        local = loop-origin
        cross = local[:-1, 0]*local[1:, 1] - local[1:, 0]*local[:-1, 1]
        twice_area = math.fsum(cross)
        area = abs(twice_area)*.5
        perimeter = math.fsum(np.linalg.norm(np.diff(local, axis=0), axis=1))
        extent = float(np.max(np.ptp(local, axis=0)))
        area_epsilon = 64*np.finfo(float).eps*extent*extent
        distinct = len({tuple(point) for point in loop[:-1]})
        reason = ('fewer_than_three_distinct_points' if distinct < 3 else
                  'zero_area' if area <= area_epsilon else None)
        if reason is None:
            centroid = origin + np.array([
                math.fsum((local[:-1, axis]+local[1:, axis])*cross)/(3*twice_area)
                for axis in range(2)
            ])
        else:
            centroid = origin + np.mean(local[:-1], axis=0)
        edge_faces = [list(profile.source_face_ids[eid]) for eid in edge_ids]
        edge_sources = [list(profile.source_segment_indices[eid]) for eid in edge_ids]
        records.append({
            'kind': 'CLOSED_COMPONENT', 'component_id': component,
            'component_key': f'{slice_key}:{component}', 'slice_key': slice_key,
            's': float(s), 'node_ids': node_ids, 'edge_ids': edge_ids,
            'ordered_loop_uz': loop.tolist(),
            'ordered_loop_xyz': profile.nodes[node_ids].tolist(),
            'edge_source_face_ids': edge_faces,
            'edge_source_segment_indices': edge_sources,
            'source_face_ids': list(dict.fromkeys(face for faces in edge_faces for face in faces)),
            'source_segment_indices': list(dict.fromkeys(index for indices in edge_sources
                                                        for index in indices)),
            'centroid_uz': centroid.tolist(), 'area_2d': float(area),
            'perimeter': float(perimeter),
            'u_range': [float(loop[:, 0].min()), float(loop[:, 0].max())],
            'z_range': [float(loop[:, 1].min()), float(loop[:, 1].max())],
            'degenerate': reason is not None, 'degeneracy_reason': reason,
            'no_artificial_closure': True, 'horizontal_matches': [],
        })
    return records


def _horizontal_evidence(left, right):
    """Count supported levels, not repeated crossings or globally reused IDs."""
    def levels(record):
        result = defaultdict(lambda: defaultdict(set))
        for match in record.get('horizontal_matches', ()):
            result[match['level_index']][float(match['z'])].add(match['branch_id'])
        return result

    a, b = levels(left), levels(right)
    supported, contradictory = set(), set()
    for level in a.keys() & b.keys():
        for z, branches in a[level].items():
            other = set()
            sampled = False
            for other_z, other_branches in b[level].items():
                if math.isclose(z, other_z, rel_tol=0., abs_tol=1e-6):
                    sampled = True
                    other.update(other_branches)
            if sampled:
                if branches & other:
                    supported.add(level)
                else:
                    contradictory.add(level)
    evidence = ('contradictory' if contradictory else
                'positive' if supported else 'insufficient')
    return len(supported), evidence


def _curve_samples(curve):
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    positions = np.linspace(0., cumulative[-1], 64, endpoint=False)
    indices = np.minimum(np.searchsorted(cumulative, positions, side='right')-1,
                         len(lengths)-1)
    fraction = np.divide(positions-cumulative[indices], lengths[indices],
                         out=np.zeros_like(positions), where=lengths[indices] > 0)
    uniform = curve[indices] + fraction[:, None]*(curve[indices+1]-curve[indices])
    return np.concatenate((curve[:-1], (curve[:-1]+curve[1:])*.5, uniform))


def _directed_curve_distance(samples, curve):
    """Sample-to-segment distances, chunked to bound temporary memory."""
    maximum = 0.
    for start in range(0, len(samples), 64):
        points = samples[start:start+64]
        nearest = np.full(len(points), np.inf)
        for edge in range(0, len(curve)-1, 256):
            a = curve[edge:min(edge+256, len(curve)-1)]
            b = curve[edge+1:min(edge+257, len(curve))]
            delta = b-a
            offset = points[:, None, :]-a[None, :, :]
            norm2 = np.sum(delta*delta, axis=1)
            fraction = np.divide(np.sum(offset*delta[None, :, :], axis=2), norm2,
                                 out=np.zeros(offset.shape[:2]), where=norm2 > 0)
            projected = offset-np.clip(fraction, 0., 1.)[:, :, None]*delta[None, :, :]
            nearest = np.minimum(nearest, np.sqrt(np.sum(projected*projected, axis=2)).min(axis=1))
        maximum = max(maximum, float(nearest.max()))
    return maximum


def _pair(left, right, spacing, max_centroid_distance):
    support, evidence = _horizontal_evidence(left, right)
    pair = {
        'left_component_key': left['component_key'],
        'right_component_key': right['component_key'],
        'matched': False, 'ambiguous': False, 'candidate': False,
        'horizontal_support_count': support, 'horizontal_evidence': evidence,
        'score': None, 'cost': None, 'reason': None,
    }
    difference = float(right['s'])-float(left['s'])
    if not math.isclose(difference, spacing, rel_tol=0., abs_tol=max(1e-9, spacing*1e-6)):
        pair['reason'] = 'nonadjacent_slices'
        return pair
    if left['degenerate'] or right['degenerate']:
        pair['reason'] = 'degenerate'
        return pair
    distance = float(np.linalg.norm(np.asarray(left['centroid_uz'])-right['centroid_uz']))
    pair['centroid_distance'] = distance
    if distance > max_centroid_distance:
        pair['reason'] = 'centroid_distance'
        return pair
    area_ratio = min(left['area_2d'], right['area_2d'])/max(left['area_2d'], right['area_2d'])
    perimeter_ratio = min(left['perimeter'], right['perimeter'])/max(left['perimeter'], right['perimeter'])
    pair.update(area_ratio=area_ratio, perimeter_ratio=perimeter_ratio)
    if area_ratio < .5:
        pair['reason'] = 'area_ratio'
        return pair
    if perimeter_ratio < .7:
        pair['reason'] = 'perimeter_ratio'
        return pair
    low_a, high_a = left['z_range']
    low_b, high_b = right['z_range']
    overlap = max(0., min(high_a, high_b)-max(low_a, low_b))
    height = min(high_a-low_a, high_b-low_b)
    overlap_ratio = overlap/height if height > 0 else 0.
    pair['height_overlap_ratio'] = overlap_ratio
    if overlap_ratio < .5:
        pair['reason'] = 'height_overlap'
        return pair
    # Translation is measured separately. Shape retains orientation and scale.
    curves = [np.asarray(record['ordered_loop_uz'], dtype=float)-record['centroid_uz']
              for record in (left, right)]
    shape = max(_directed_curve_distance(_curve_samples(curves[0]), curves[1]),
                _directed_curve_distance(_curve_samples(curves[1]), curves[0]))
    shape_limit = .1*min(left['perimeter'], right['perimeter'])/4
    pair.update(shape_distance=shape, shape_distance_limit=shape_limit)
    if shape > shape_limit:
        pair['reason'] = 'shape_distance'
        return pair
    cost = (distance/max_centroid_distance + (1-area_ratio) + (1-perimeter_ratio)
            + (1-overlap_ratio) + shape/shape_limit)/5
    pair.update(cost=float(cost), score=float(1-cost))
    if evidence == 'contradictory':
        pair['reason'] = 'horizontal_contradiction'
        return pair
    pair.update(candidate=True, reason='candidate')
    return pair


def track_closed_components(components, *, spacing=.05, max_centroid_distance=.15,
                            ambiguity_margin=.05):
    """Return pair diagnostics, disjoint tracks and counts without mutating inputs.

    Only consecutive observed S groups separated by ``spacing`` (+/- max(1e-9,
    spacing*1e-6)) can link; missing scanlines never get bridged. Geometry gates:
    centroid distance <= supplied limit, min/max area >= .5, min/max perimeter
    >= .7, and Z overlap / smaller height >= .5. The symmetric curve distance
    after centroid translation must be <= .1 * smaller perimeter / 4. Distance
    samples include every vertex, every edge midpoint and 64 uniform arclength
    positions, measured against actual opposite segments (not just vertices).

    Cost is the mean of normalized centroid distance, area/perimeter losses,
    height-overlap loss and normalized shape distance. Score is 1-cost.
    A candidate must be mutual best and the best-vs-runner-up cost gap at BOTH
    endpoints must exceed ``ambiguity_margin``; ties remain unmatched. Rejected
    candidates are retained, including all competitors in consecutive S groups.
    Horizontal support requires equal level_index, Z within 1e-6, and a common
    branch_id. Any shared sampled level without a common branch is contradictory.

    Stable means >=3 slices, no ambiguity and positive H support on EVERY link.
    It never establishes a confirmed independent block or creates 3D geometry.
    """
    for name, value in [('spacing', spacing), ('max_centroid_distance', max_centroid_distance)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0:
        raise ValueError('ambiguity_margin must be nonnegative and finite')
    components = list(components)
    by_key, groups = {}, defaultdict(list)
    for record in components:
        key = record['component_key']
        if key in by_key:
            raise ValueError('component_key must be unique')
        if not math.isfinite(record['s']):
            raise ValueError('component s must be finite')
        by_key[key] = record
        groups[float(record['s'])].append(record)
    positions = sorted(groups)
    pairs, ambiguous_keys = [], set()
    for left_s, right_s in zip(positions, positions[1:]):
        local = [_pair(left, right, spacing, max_centroid_distance)
                 for left in groups[left_s] for right in groups[right_s]]
        contenders = defaultdict(list)
        for pair in local:
            if pair['candidate']:
                contenders[pair['left_component_key']].append(pair)
                contenders[pair['right_component_key']].append(pair)
        best, uncertain = {}, set()
        for key, options in contenders.items():
            ranked = sorted(options, key=lambda pair: pair['cost'])
            best[key] = ranked[0]
            if len(ranked) > 1 and ranked[1]['cost']-ranked[0]['cost'] <= ambiguity_margin:
                uncertain.add(key)
        for pair in local:
            if not pair['candidate']:
                continue
            left, right = pair['left_component_key'], pair['right_component_key']
            if left in uncertain or right in uncertain:
                pair.update(ambiguous=True, reason='ambiguous_competitors')
                ambiguous_keys.update((left, right))
            elif best[left] is pair and best[right] is pair:
                pair.update(matched=True, reason=('horizontal_supported' if
                            pair['horizontal_support_count'] else 'insufficient_horizontal_evidence'))
            else:
                pair['reason'] = 'one_to_one_conflict'
        pairs.extend(local)

    outgoing, incoming = {}, set()
    for pair in pairs:
        if pair['matched']:
            outgoing[pair['left_component_key']] = pair
            incoming.add(pair['right_component_key'])
    tracks = []
    for record in sorted(components, key=lambda item: (item['s'], item['component_key'])):
        key = record['component_key']
        if key in incoming:
            continue
        keys, links = [key], []
        while key in outgoing:
            pair = outgoing[key]
            links.append(pair)
            key = pair['right_component_key']
            keys.append(key)
        supported = sum(pair['horizontal_support_count'] > 0 for pair in links)
        ambiguous = any(key in ambiguous_keys for key in keys)
        if record['degenerate']:
            status = 'degenerate'
        elif ambiguous:
            status = 'ambiguous'
        elif not links:
            status = 'unmatched'
        elif supported != len(links):
            status = 'geometric_only'
        elif len(keys) >= 3:
            status = 'stable'
        else:
            status = 'horizontal_supported'
        tracks.append({'component_keys': keys, 'slice_count': len(keys),
                       'horizontal_supported_pair_count': supported,
                       'ambiguous': ambiguous, 'status': status})
    return {
        'pairs': pairs, 'tracks': tracks,
        'summary': {
            'total_loops': len(components),
            'nondegenerate_loops': sum(not record['degenerate'] for record in components),
            'linked_tracks': sum(track['slice_count'] >= 2 for track in tracks),
            'stable_tracks': sum(track['status'] == 'stable' for track in tracks),
            'unmatched_loops': sum(track['slice_count'] == 1 for track in tracks),
            'ambiguous_pairs': sum(pair['ambiguous'] for pair in pairs),
        },
    }
