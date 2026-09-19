"""Full-branch descriptors and endpoint-independent surface selection."""
from __future__ import annotations
import numpy as np


def local_descriptors(branch):
    points = np.asarray(branch['points_uz'])
    arc = np.asarray(branch['arc_positions'])
    length = max(float(arc[-1]), 1e-12)
    mids = (arc[:-1]+arc[1:])/2
    vectors = np.diff(points, axis=0)
    angles = np.unwrap(np.arctan2(vectors[:, 0], vectors[:, 1]))
    curvature = np.r_[0., np.diff(angles)]
    return [dict(full_branch_arc_position=float(m), distance_to_branch_start=float(m),
        distance_to_branch_end=float(length-m), interior_fraction=float(4*m*(length-m)/length**2),
        local_detail_density=float(1/max(arc[i+1]-arc[i], 1e-12)),
        local_curvature_signature=float(curvature[i]), local_shape_signature=float(angles[i]))
        for i, m in enumerate(mids)]


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


def branch_evidence(graph, node):
    branch = graph['nodes'][node]
    tid = graph['membership'][node]
    track = next(t for t in graph['tracks'] if t['track_id'] == tid)
    links = graph['support'].get(node, [])
    descriptors = local_descriptors(branch)
    lengths = np.diff(branch['arc_positions'])
    interior = float(np.average([d['interior_fraction'] for d in descriptors], weights=lengths)) if lengths.sum() > 0 else 0.
    sides = {int(np.sign((l['b'] if l['a'] == node else l['a'])[0]-node[0])) for l in links}
    return dict(branch_id=branch['branch_id'], component_id=branch['component_id'], kind=branch['kind'],
        fragment=branch, fragment_index=0, surface_track_id=tid, track_stable=track['stable'],
        track_ambiguous=track['ambiguous'], cross_slice_continuity=track['slice_count'],
        two_sided_support={-1, 1}.issubset(sides), observed_coverage=1.,
        full_arc_length=branch['full_arc_length'], interior_score=interior,
        detail_density=branch['detail_density'], canonical_edge_count=branch['canonical_edge_count'],
        canonical_node_count=branch['canonical_node_count'],
        horizontal_support=min((l['support_fraction'] for l in links), default=0.),
        horizontal_supported_layers=len({li for l in links for li in l['levels']}),
        continuous_H_support_run=max((l['continuous_H_support_run'] for l in links), default=0),
        descriptors=descriptors, **same_surface_detail(graph, node))


def _rank(row):
    return (row['two_sided_support'], row['cross_slice_continuity'], row['continuous_H_support_run'],
            row['horizontal_support'], round(row['full_arc_length'], 6),
            row['neighbor_detail_repeat_count'], round(row['neighbor_detail_score'], 2),
            round(row['interior_score'], 2), row['detail_density'])


def select_surface_track(graph, target_s, *, surface_track_id=None):
    rows = [branch_evidence(graph, n) for n, b in graph['nodes'].items()
            if n[0] == float(target_s) and b['kind'] != 'CLOSED_COMPONENT'
            and b['full_arc_length'] > 1e-6
            and (surface_track_id is None or graph['membership'][n] == surface_track_id)]
    rows.sort(key=lambda r: (*_rank(r), -r['branch_id']), reverse=True)
    ambiguous = bool(rows and rows[0]['track_ambiguous']) or (len(rows) > 1 and _rank(rows[0]) == _rank(rows[1]))
    for i, row in enumerate(rows):
        row.update(selected=i == 0, reason='Full observed surface continuity / same-H-path / same-track detail')
    return dict(selected=rows[0] if rows else None, candidates=rows, ambiguous=ambiguous,
                legacy_endpoint_influence_on_surface_identity=0)
