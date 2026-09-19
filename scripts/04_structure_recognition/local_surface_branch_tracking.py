"""Read-only, cross-height neighbor tracks; no per-height surface switching."""
from __future__ import annotations

import copy
import numpy as np
from vertical_profile_canonicalization import canonicalize_vertical


def _monotone_parts(points):
    start, direction = 0, 0
    for i, dz in enumerate(np.diff(points[:, 1])):
        sign = int(np.sign(dz)) if abs(dz) > 1e-9 else 0
        if not sign or direction and sign != direction:
            part = points[start:i+1]
            if len(part) > 1:
                yield part if part[-1, 1] > part[0, 1] else part[::-1]
            start = i if sign else i+1
        direction = sign
    part = points[start:]
    if len(part) > 1:
        yield part if part[-1, 1] > part[0, 1] else part[::-1]


def build_neighbor_surface_tracks(profiles, levels, guide_curve, target_s):
    """Select one actual monotone branch per neighbor over the entire Z window.

    Missing parts of a selected branch remain missing. Guide coordinates choose
    identity only; every emitted neighbor value interpolates an observed edge.
    IDs are profile-local and never compared between horizontal cuts.
    """
    levels = np.asarray(levels, dtype=float)
    guide = np.asarray(guide_curve, dtype=float)
    expected = np.interp(levels, guide[:, 1], guide[:, 0])
    layers = [{'z': float(z), 'neighbors': []} for z in levels]
    chosen = []
    for profile in profiles:
        s = float(profile['s'])
        if abs(s-target_s) < 1e-9:
            continue
        lines = np.asarray(profile['lines_uz'], dtype=float).reshape(-1, 2, 2)
        xyz = np.zeros((len(lines), 2, 3))
        xyz[:, :, 0], xyz[:, :, 2] = lines[:, :, 0], lines[:, :, 1]
        graph = canonicalize_vertical(xyz, [None]*len(lines))
        options = []
        for branch in graph.branches:
            points = graph.nodes[branch['nodes']][:, [0, 2]]
            for part_index, part in enumerate(_monotone_parts(points)):
                valid = (levels >= part[0, 1]-1e-8) & (levels <= part[-1, 1]+1e-8)
                if not valid.any():
                    continue
                values = np.interp(levels[valid], part[:, 1], part[:, 0])
                score = float(np.median(abs(values-expected[valid]))/.05 + 2*(1-valid.mean()))
                options.append((score, int(branch['branch_id']), part_index, part, valid))
        best = min(options, key=lambda item: item[:3]) if options else None
        if best is not None:
            score, bid, pid, part, valid = best
            chosen.append({'s': s, 'branch_id': bid, 'monotone_part_id': pid, 'score': score,
                           'coverage': float(valid.mean()), 'points_uz': part.tolist()})
        for i, layer in enumerate(layers):
            values = [float(np.interp(levels[i], best[3][:, 1], best[3][:, 0]))] if best is not None and best[4][i] else []
            layer['neighbors'].append({'s': s, 'values': values,
                                       'track_id': f'{s:.6f}:{best[1]}:{best[2]}' if best is not None else None})
    return {'layers': layers, 'tracks': chosen, 'cross_height_switches': 0,
            'method': 'One observed monotone branch per neighbor; no extrapolation across missing parts'}


def apply_neighbor_surface_tracks(window, tracks):
    result = copy.deepcopy(window)
    by_z = {round(layer['z'], 8): layer for layer in tracks['layers']}
    for layer in result['layers']:
        tracked = by_z.get(round(layer['z'], 8), {'neighbors': []})
        layer['neighbors'] = copy.deepcopy(tracked['neighbors'])
        by_s = {round(o['s'], 8): o['values'] for o in layer['neighbors']}
        for hit in layer['horizontal']:
            if 'linked_neighbors' in hit:
                hit['linked_neighbors'] = [
                    {'s': o['s'], 'values': [v for v in o['values']
                      if any(abs(v-t) <= 1e-4 for t in by_s.get(round(o['s'], 8), []))]}
                    for o in hit['linked_neighbors']]
    result['neighbor_surface_tracks'] = copy.deepcopy(tracks['tracks'])
    return result
