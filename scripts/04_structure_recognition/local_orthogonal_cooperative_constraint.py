"""Analysis-only local orthogonal cooperative surface inference (LOCC).

Observed and inferred geometry remain distinguishable. This module never
modifies recognition predicates, input slices, mesh faces or source FaceIDs.
"""
from __future__ import annotations

import math
import numpy as np


def _huber(x):
    x = abs(float(x))
    return .5*x*x if x <= 1 else x-.5


def neighbor_hypotheses(observations, target_s, prior_u):
    """Branch-preserving robust local affine fits; never average distinct U.

    Each radial seed chooses one value per neighboring profile. Pairwise-line
    consensus removes at most one outlying profile, followed by one MAD pass.
    The different supported surfaces remain separate hypotheses.
    """
    observations = [o for o in observations if len(o['values']) and abs(o['s']-target_s) > 1e-9]
    if not observations:
        return []
    seeds = np.unique([prior_u]+[float(u) for o in observations for u in o['values']])
    hypotheses = []
    for seed in seeds:
        ss = np.array([o['s']-target_s for o in observations])
        uu = np.array([min(o['values'], key=lambda u: abs(u-seed)) for o in observations], dtype=float)
        active = np.ones(len(ss), dtype=bool)
        if len(ss) >= 3:
            # A four-point least-squares fit can itself be pulled by an outlier.
            # Median pairwise residual consensus is a deterministic robust start.
            candidates = []
            for i in range(len(ss)):
                for j in range(i+1, len(ss)):
                    if abs(ss[j]-ss[i]) < 1e-10:
                        continue
                    slope = (uu[j]-uu[i])/(ss[j]-ss[i])
                    intercept = uu[i]-slope*ss[i]
                    residual = abs(uu-(intercept+slope*ss))
                    core_count = max(2, len(ss)-1)
                    score = np.sort(residual)[:core_count].sum()
                    candidates.append((score, abs(intercept-seed), intercept, slope))
            _, _, intercept, slope = min(candidates)
            residual = uu-(intercept+slope*ss)
            center = np.median(residual)
            mad = np.median(abs(residual-center))
            active = abs(residual-center) <= max(.002, 3*1.4826*mad)
            if active.sum() < 2:
                active[:] = True
        x = np.column_stack((np.ones(active.sum()), ss[active]))
        weights = 1/np.maximum(abs(ss[active]), .025)
        if active.sum() >= 2:
            beta = np.linalg.lstsq(x*np.sqrt(weights[:, None]), uu[active]*np.sqrt(weights), rcond=None)[0]
        else:
            beta = np.array([uu[active][0], 0.])
        residuals = uu[active]-x@beta
        item = {'u': float(beta[0]), 'slope_s': float(beta[1]), 'support_count': int(active.sum()),
                'available_count': len(observations), 'two_sided': bool(np.any(ss[active] < 0) and np.any(ss[active] > 0)),
                'residuals': residuals.tolist(), 'rms': float(np.sqrt(np.mean(residuals**2))),
                'used_s': (ss[active]+target_s).tolist(), 'used_u': uu[active].tolist()}
        duplicate = next((i for i, h in enumerate(hypotheses) if abs(h['u']-item['u']) < 1e-6), None)
        if duplicate is None:
            hypotheses.append(item)
        elif (item['support_count'], -item['rms']) > (hypotheses[duplicate]['support_count'], -hypotheses[duplicate]['rms']):
            hypotheses[duplicate] = item
    return sorted(hypotheses, key=lambda h: h['u'])


def solve_locc(window, mode='LOCC', weights=None):
    """Second-order, two-best dynamic programming over all radial candidates.

    Inputs contain only visible fragments, neighboring observations and
    horizontal sections; a hidden truth curve is deliberately not an argument.
    Methods share endpoint anchors but only LOCC sees neighbor/tangent evidence.
    """
    if mode not in ('LOCC', 'HORIZONTAL_ONLY', 'VERTICAL_ONLY'):
        raise ValueError('Unknown reconstruction method')
    lower, upper = np.asarray(window['lower_uz'], dtype=float), np.asarray(window['upper_uz'], dtype=float)
    if lower.shape != (2,) or upper.shape != (2,) or not np.isfinite([lower, upper]).all() or upper[1] <= lower[1]:
        raise ValueError('Finite ordered gap endpoints are required')
    layers = sorted(window['layers'], key=lambda x: x['z'])
    zz = np.array([lower[1]]+[float(r['z']) for r in layers]+[upper[1]])
    if not np.all(np.diff(zz) > 1e-9):
        raise ValueError('Layers must be distinct and strictly inside the gap')
    height, du = upper[1]-lower[1], upper[0]-lower[0]
    t = (zz-lower[1])/height
    linear = lower[0]+t*du
    slope = du/height
    lower_slope = float(window.get('lower_tangent', slope))
    upper_slope = float(window.get('upper_tangent', slope))
    prior = ((2*t**3-3*t**2+1)*lower[0]+(t**3-2*t**2+t)*height*lower_slope
             +(-2*t**3+3*t**2)*upper[0]+(t**3-t**2)*height*upper_slope)
    scale = max(.005, .05*height, .1*abs(du))
    cfg = {'neighbor': 1., 'observed_neighbor_factor': .01, 'vertical': .03, 'du': .005,
           'tangent': .01, 'curvature': .01, 'virtual': .1, 'missing': .5}
    if weights:
        cfg.update(weights)
    candidates = [[{'u': float(lower[0]), 'source': 'ENDPOINT', 'face_id': None, 'branch_id': None, 'neighbor': None, 'cost': 0.}]]
    counts, hypotheses_by_level = [], []
    for j, layer in enumerate(layers, 1):
        hypotheses = neighbor_hypotheses(layer.get('neighbors', []), window['s'], prior[j]) if mode == 'LOCC' else []
        hypotheses_by_level.append(hypotheses)
        nodes = []
        if mode != 'VERTICAL_ONLY':
            for hit in layer.get('horizontal', []):
                nodes.append({'u': float(hit['u']), 'source': 'HORIZONTAL_MESH',
                              'face_id': int(hit['face_id']) if hit.get('face_id') is not None else None,
                              'branch_id': hit.get('branch_id'), 'linked_neighbors': hit.get('linked_neighbors', [])})
        if not nodes and mode == 'LOCC' and hypotheses:
            nodes = [{'u': h['u'], 'source': 'V_NEIGHBOR_PREDICTED', 'face_id': None,
                      'branch_id': None} for h in hypotheses]
        if not nodes:
            nodes = [{'u': float(linear[j]), 'source': 'VERTICAL_ONLY' if mode == 'VERTICAL_ONLY' else 'VERTICAL_PRIOR_FALLBACK',
                      'face_id': None, 'branch_id': None}]
        # Preserve branches even if their radial values coincide at a junction.
        counts.append(len(nodes))
        for node in nodes:
            linked = node.pop('linked_neighbors', [])
            linked_count = sum(bool(o['values']) for o in linked)
            node['horizontal_linked_neighbor_count'] = linked_count
            # Horizontal connectivity selects a locally coherent neighbor
            # surface when available. It is soft evidence, never a hard gate.
            local_hypotheses = (neighbor_hypotheses(linked, window['s'], node['u'])
                                if mode == 'LOCC' and linked_count >= 2 else hypotheses)
            h = min(local_hypotheses, key=lambda p: abs(p['u']-node['u'])+p['rms']) if local_hypotheses else None
            node['neighbor'] = h
            cost = 0.
            if mode == 'LOCC':
                if h is not None:
                    neighbor_cost = cfg['neighbor']*(_huber((node['u']-h['u'])/scale)+_huber(h['rms']/scale))
                    neighbor_cost += .05*(4-h['support_count']) + (0 if h['two_sided'] else .1)
                    # A local affine predictor is an approximation, whereas an
                    # H candidate is an actual observation. Calibrate only this
                    # relative influence; retain full neighbor use when H is absent.
                    factor = cfg['observed_neighbor_factor'] if node['source'] == 'HORIZONTAL_MESH' else 1.
                    cost += factor*neighbor_cost
                cost += cfg['vertical']*_huber((node['u']-prior[j])/scale)
            if node['source'] == 'V_NEIGHBOR_PREDICTED':
                cost += cfg['virtual']
            if node['source'] == 'VERTICAL_PRIOR_FALLBACK':
                cost += cfg['missing']
            node['cost'] = float(cost)
        candidates.append(nodes)
    candidates.append([{'u': float(upper[0]), 'source': 'ENDPOINT', 'face_id': None, 'branch_id': None, 'neighbor': None, 'cost': 0.}])

    def edge_cost(a, b, za, zb, previous_slope=None, boundary=None):
        current_slope = (b['u']-a['u'])/(zb-za)
        value = cfg['du']*abs(b['u']-a['u'])/scale
        if previous_slope is not None:
            value += cfg['curvature']*_huber((current_slope-previous_slope)/max(1., abs(slope)))
        if mode == 'LOCC' and boundary is not None:
            value += cfg['tangent']*_huber((current_slope-boundary)/max(1., abs(slope)))
        return value

    # Each state remembers the previous two nodes, which makes curvature a
    # genuine transition cost rather than a post-hoc reranking of one path.
    states = {}
    for k, node in enumerate(candidates[1]):
        value = node['cost']+edge_cost(candidates[0][0], node, zz[0], zz[1], boundary=lower_slope)
        states[(0, k)] = [(value, (0, k))]
    for j in range(2, len(candidates)):
        next_states = {}
        for (a, b), options in states.items():
            ua, ub = candidates[j-2][a], candidates[j-1][b]
            prev_slope = (ub['u']-ua['u'])/(zz[j-1]-zz[j-2])
            for k, node in enumerate(candidates[j]):
                extra = node['cost']+edge_cost(ub, node, zz[j-1], zz[j], prev_slope,
                                              upper_slope if j == len(candidates)-1 else None)
                pool = next_states.setdefault((b, k), [])
                pool.extend((cost+extra, path+(k,)) for cost, path in options)
                pool.sort(key=lambda item: (item[0], item[1]))
                del pool[2:]
        states = next_states
    finals = sorted((item for values in states.values() for item in values), key=lambda item: (item[0], item[1]))
    best_cost, path = finals[0]
    second = next((cost for cost, other in finals[1:] if other != path), None)
    nodes = [{**candidates[j][idx], 'z': float(zz[j])} for j, idx in enumerate(path)]
    curve = np.column_stack(([n['u'] for n in nodes], zz))
    middle = nodes[1:-1]
    n = max(1, len(middle))
    real = sum(p['source'] == 'HORIZONTAL_MESH' for p in middle)
    supported = [p for p in middle if p['neighbor'] is not None and p['neighbor']['support_count'] >= 2]
    residuals = [abs(p['u']-p['neighbor']['u']) for p in supported]
    slopes = np.diff(curve[:, 0])/np.diff(curve[:, 1])
    endpoint_tangent_error = float((abs(slopes[0]-lower_slope)+abs(slopes[-1]-upper_slope))/2)
    metrics = {'horizontal_real_coverage': real/n, 'neighbor_vertical_coverage': len(supported)/n,
               'endpoint_residual': 0., 'endpoint_tangent_error': endpoint_tangent_error,
               'neighbor_residual': float(np.median(residuals)) if residuals else None,
               'path_smoothness': float(np.sqrt(np.mean(np.diff(slopes)**2))) if len(slopes) > 1 else 0.,
               'face_support_ratio': sum(p['face_id'] is not None for p in middle)/n,
               'fallback_fraction': sum(p['source'] == 'VERTICAL_PRIOR_FALLBACK' for p in middle)/n,
               'ambiguous_level_fraction': sum(c > 1 for c in counts)/max(1, len(counts)),
               'two_sided_neighbor_fraction': sum(p['neighbor'] is not None and p['neighbor']['two_sided'] for p in middle)/n}
    metrics['horizontal_linked_neighbor_fraction'] = sum(p.get('horizontal_linked_neighbor_count', 0) >= 2 for p in middle)/n
    return {'mode': mode, 'curve_uz': curve, 'selected_nodes': nodes,
            'selected_source_per_level': [n['source'] for n in middle],
            'candidate_counts': counts, 'neighbor_hypotheses': hypotheses_by_level,
            'best_cost': float(best_cost), 'second_best_cost': float(second) if second is not None else None,
            'cost_margin': float(second-best_cost) if second is not None else None,
            'normalized_cost_margin': float((second-best_cost)/(1+abs(best_cost))) if second is not None else None,
            'metrics': metrics, 'weights': cfg, 'scale_u_m': scale,
            'provenance': 'analysis-only inferred curve; FaceID attaches to observed nodes only, not inferred edges'}
