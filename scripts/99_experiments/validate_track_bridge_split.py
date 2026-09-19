"""Analysis-only bridge diagnostics over frozen, previously matched group links.

Call ``run_track_bridge(frozen, output)`` in the existing profiled process.
No recognition, rematching, mesh slicing, plotting, or production writes occur.
Ch/Cz/median_du remain the prior CSV measurements; the new sampled support
metrics describe resolution-limited evidence, never confirmed geological splits.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from itertools import product
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
RECOGNITION = str(ROOT/'scripts/04_structure_recognition')
if RECOGNITION not in sys.path:
    sys.path.insert(0, RECOGNITION)
import orthogonal_profile_constraint as opc


L_MINIMA = (0., .05, .10, .20, .40)
BRIDGE_MINIMA = (0., .10, .25, .50)
OVERLAP_MINIMA = (0., .25, .50, .75)
ILLUSTRATIVE = (.10, .25, .50)
OUTPUT_FILES = ('track_bridge_pairs.csv', 'track_bridge_sweep.csv', 'track_split_cases.csv',
                'major_structure_preservation.csv', 'track_bridge_summary.json')
# Only a numerical comparison allowance, not an extra geological tolerance.
EPSILON = 1e-9


def load_matching_tolerance(prior):
    """Read the applied V/H tolerance, never the baseline C-score tau_u_m."""
    applied = ('applied_tau_u_m',)
    nested = ('tolerance', 'applied_tau_u_m')
    constraint = ('constraint', 'tolerance', 'applied_tau_u_m')
    sources = [('tolerance_calibration.json', (applied,)),
               ('ORTHOGONAL_SCANLINE_CONSTRAINT_REPORT.json', (constraint,)),
               ('constraint_diagnostics.json', (nested,)),
               ('manifest.json', (applied, nested, constraint, ('matching_tolerance_m',),
                                  ('tolerance_calibration', 'applied_tau_u_m')))]
    for filename, paths in sources:
        path = Path(prior)/filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        for keys in paths:
            value = payload
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f'Invalid matching tolerance in {filename}:{".".join(keys)}')
            return float(value), f'{filename}:{".".join(keys)}'
    raise ValueError('No explicit applied matching tolerance in the prior audit files')


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _save_csv(path, rows, fields):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def _safe_group_map(group, levels, arc):
    uz = np.asarray(group['uz'], dtype=float)
    if (not group.get('audit_observable', True) or uz.ndim != 2 or uz.shape[1] != 2
            or len(uz) < 2 or not np.isfinite(uz).all()
            or not np.all(uz[:, 0]+arc['radius'] > 0)):
        return {}
    if 'node_order' in group:
        nodes = np.asarray(group['node_order'], dtype=float)
        if (nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all()
                or not opc.on_scanline_ray(nodes, arc, group['s'])):
            return {}
    return opc.group_level_map(group, levels)


def build_group_evidence(frozen, needed_ids, tolerance):
    """One crossings/level_map build per needed clean level, then release it.

    ``sampled`` requires finite positive-ray H crossings at that group's s.
    Branch IDs are local to a level: only compare IDs WITHIN a horizontal cut.
    Raw crossing tables and group u observations are not retained after this pass.
    """
    levels = np.asarray(frozen.levels, dtype=float)
    by_level = defaultdict(list)
    evidence = {}
    for si, key in enumerate(frozen.keys):
        for group in frozen.groups[key]:
            gid = group['id']
            if gid not in needed_ids:
                continue
            observations = _safe_group_map(group, levels, frozen.arc)
            evidence[gid] = {'levels': set(observations), 'sampled': set(), 'branches': {}}
            for zi, values in observations.items():
                if zi in frozen.horizontal:
                    by_level[zi].append((gid, si, values[:, 0]))
    # Slice only the requested s positions without changing the clean run IDs.
    s_values = np.asarray(frozen.s, dtype=float)
    for zi in sorted(by_level):
        samples = by_level.pop(zi)
        needed_s = sorted({si for _, si, _ in samples})
        local_index = {si: i for i, si in enumerate(needed_s)}
        hits = opc.horizontal_crossings(frozen.horizontal[zi], s_values[needed_s], frozen.arc)
        if len(hits):
            hits = hits[np.isfinite(hits).all(axis=1) & (hits[:, 1]+frozen.arc['radius'] > 0)]
        hmap = opc.level_map(hits)
        for gid, si, values in samples:
            horizontal = hmap.get(local_index[si])
            if horizontal is None or not len(horizontal):
                continue
            evidence[gid]['sampled'].add(zi)
            near = np.any(np.abs(values[:, None]-horizontal[None, :, 0]) <= tolerance, axis=0)
            branches = {int(b) for b in horizontal[near, 1] if b >= 0 and float(b).is_integer()}
            if branches:
                evidence[gid]['branches'][zi] = branches
        del hits, hmap, samples
    return evidence


def bridge_metrics(left, right, levels, left_evidence, right_evidence, horizontal_levels):
    """Consecutive *grid-index* support; singleton span is exactly zero metres.

    Common vertical samples with missing H at either ray remain unobserved.
    At least two common samples with H on BOTH rays are required for a gate.
    Unsupported or missing intermediate grid levels always interrupt a run.
    """
    lo = max(left['z_min'], right['z_min'])
    hi = min(left['z_max'], right['z_max'])
    overlap = max(0., hi-lo)
    min_height = min(left['z_max']-left['z_min'], right['z_max']-right['z_min'])
    grid = set(range(int(np.searchsorted(levels, lo, side='left')),
                     int(np.searchsorted(levels, hi, side='right')))) if hi >= lo else set()
    common = left_evidence['levels'] & right_evidence['levels'] & grid
    sampled = common & left_evidence['sampled'] & right_evidence['sampled']
    supported, ambiguous = [], 0
    for zi in sorted(sampled):
        branches = left_evidence['branches'].get(zi, set()) & right_evidence['branches'].get(zi, set())
        if branches:
            supported.append(zi)
            ambiguous += len(branches) > 1
    runs = []
    for zi in supported:
        if not runs or zi != runs[-1][-1]+1:
            runs.append([zi])
        else:
            runs[-1].append(zi)
    best = max(runs, key=lambda run: (levels[run[-1]]-levels[run[0]], len(run)), default=[])
    length = float(levels[best[-1]]-levels[best[0]]) if best else 0.
    eligible = len(sampled) >= 2 and min_height > 0
    return {'L_overlap_z': float(overlap), 'min_group_height_m': float(min_height),
            'R_overlap': float(overlap/min_height) if min_height > 0 else None,
            'L_support': length, 'R_bridge': float(length/min_height) if min_height > 0 else None,
            'overlap_grid_levels': len(grid), 'sampled_levels': len(grid & horizontal_levels),
            'left_sampled_levels': len(left_evidence['levels'] & grid),
            'right_sampled_levels': len(right_evidence['levels'] & grid),
            'common_levels': len(common), 'common_sampled_levels': len(sampled),
            'supported_levels': len(supported), 'missing_horizontal_common_levels': len(common-sampled),
            'unsupported_common_sampled_levels': len(sampled)-len(supported),
            'sampled_Ch': len(supported)/len(common) if common else None,
            'undersampled': len(sampled) < 2, 'gate_eligible': eligible,
            'evidence_status': 'SAMPLED_UNREVIEWED' if eligible else 'UNRESOLVED_UNDERSAMPLED',
            'supported_level_indices': supported, 'support_run_levels': len(best),
            'support_run_z_min': float(levels[best[0]]) if best else None,
            'support_run_z_max': float(levels[best[-1]]) if best else None,
            'ambiguous_supported_levels': ambiguous}


def _group_length(group):
    for field in ('length_2d', 'length_m'):
        if field in group:
            value = float(group[field])
            if math.isfinite(value) and value >= 0:
                return value
    value = float(np.linalg.norm(np.diff(np.asarray(group['uz'], dtype=float), axis=0), axis=1).sum())
    if not math.isfinite(value):
        raise ValueError(f'No finite curve length for group {group["id"]}')
    return value


def _baseline_graph(prior, byid, slice_indices):
    """Load and verify the saved association partition, including isolated groups."""
    ids = list(byid)
    index = {gid: i for i, gid in enumerate(ids)}
    tracks = _read_csv(prior/'group_tracks.csv')
    ownership = np.full(len(ids), -1, dtype=int)
    seen_tracks = set()
    lengths = np.array([_group_length(byid[gid]) for gid in ids])
    for ti, track in enumerate(tracks):
        if track['track_id'] in seen_tracks:
            raise ValueError('Duplicate original track ID')
        seen_tracks.add(track['track_id'])
        members = json.loads(track['group_ids'])
        if not members or len(members) != len(set(members)):
            raise ValueError('Empty or duplicate group membership in group_tracks.csv')
        positions = np.array([index[gid] for gid in members], dtype=int)
        if np.any(ownership[positions] >= 0):
            raise ValueError('A group belongs to multiple original tracks')
        ownership[positions] = ti
        if int(track['group_count']) != len(members):
            raise ValueError('Prior track group count differs from frozen membership')
        length = float(track['length_sum_m'])
        if not math.isfinite(length) or not math.isclose(length, lengths[positions].sum(), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError('Prior track length differs from frozen group lengths')
        if int(track['slice_count']) != len({slice_indices[gid] for gid in members}):
            raise ValueError('Prior track slice count differs from frozen membership')
        track.update(group_ids=members, positions=positions, group_count=len(members), length_sum_m=length)
    if np.any(ownership < 0):
        raise ValueError('group_tracks.csv does not cover every frozen group')
    return ids, index, tracks, ownership, lengths


def _components(group_count, edges):
    if not group_count:
        return 0, np.array([], dtype=int)
    rows, columns = edges.T
    matrix = coo_matrix((np.ones(len(edges), dtype=np.int8), (rows, columns)),
                        shape=(group_count, group_count)).tocsr()
    return connected_components(matrix, directed=False)


def _component_stats(labels, count, ownership, lengths, slices, track_count):
    sizes = np.bincount(labels, minlength=count)
    length_sums = np.bincount(labels, weights=lengths, minlength=count)
    first = np.full(count, len(labels), dtype=int)
    np.minimum.at(first, labels, np.arange(len(labels)))
    children = np.bincount(ownership[first], minlength=track_count)
    min_slice = np.full(count, len(labels), dtype=int)
    max_slice = np.full(count, -1, dtype=int)
    np.minimum.at(min_slice, labels, slices)
    np.maximum.at(max_slice, labels, slices)
    return {'group_count': len(labels), 'components': int(count),
            'multi_slice_tracks': int(np.sum(max_slice > min_slice)),
            'isolated_groups': int(np.sum(sizes == 1)),
            'original_tracks_split': int(np.sum(children > 1))}, sizes, length_sums, children


def _threshold_fields(thresholds):
    return dict(zip(('L_support_min_m', 'R_bridge_min', 'R_overlap_min'), thresholds))


def _failed_gates(values, eligible, thresholds):
    failed = np.zeros(values.shape, dtype=bool)
    for column, minimum in enumerate(thresholds):
        if minimum > 0:
            failed[:, column] = eligible & np.isfinite(values[:, column]) & (values[:, column]+EPSILON < minimum)
    return failed


def _preservation_rows(tracks, major, labels, sizes, lengths, thresholds, status):
    result = []
    for rank, ti in enumerate(major, 1):
        track = tracks[ti]
        children = np.unique(labels[track['positions']])
        by_count = int(children[np.argmax(sizes[children])])
        by_length = int(children[np.argmax(lengths[children])])
        result.append({**_threshold_fields(thresholds), 'threshold_status': status,
                       'rank_by_original_length': rank, 'original_track_id': track['track_id'],
                       'original_group_count': track['group_count'], 'original_length_sum_m': track['length_sum_m'],
                       'child_count': len(children), 'largest_by_group_child_id': by_count,
                       'largest_by_length_child_id': by_length, 'largest_child_group_count': int(sizes[by_count]),
                       'largest_child_length_m': float(lengths[by_length]),
                       'largest_child_group_fraction': float(sizes[by_count]/track['group_count']),
                       'largest_child_length_fraction': float(lengths[by_length]/track['length_sum_m'])
                       if track['length_sum_m'] > 0 else None})
    return result


def _representatives(rows, byid, group_keys):
    removed = [r for r in rows if r['illustrative_decision'] == 'REMOVED_ILLUSTRATIVE']
    retained = [r for r in rows if r['illustrative_decision'] == 'RETAINED_SAMPLED']
    unresolved = [r for r in rows if r['illustrative_decision'] == 'RETAINED_UNRESOLVED']
    removed.sort(key=lambda r: (r['R_bridge'], r['L_support'], r['left_id'], r['right_id']))
    retained.sort(key=lambda r: (-r['R_bridge'], -r['L_support'], r['left_id'], r['right_id']))
    unresolved.sort(key=lambda r: (-r['supported_levels'], r['left_id'], r['right_id']))
    # Alternating contrasts, with an explicit unresolved example when available.
    pools = (removed[:1], retained[:1], unresolved[:1], removed[1:2], retained[1:2], unresolved[1:2])
    chosen = [row for pool in pools for row in pool]
    seen = {(r['left_id'], r['right_id']) for r in chosen}
    for row in removed+retained+unresolved:
        if len(chosen) == 6:
            break
        if (row['left_id'], row['right_id']) not in seen:
            chosen.append(row)
            seen.add((row['left_id'], row['right_id']))
    result = []
    for row in chosen:
        left, right = byid[row['left_id']], byid[row['right_id']]
        # Keep invalid geometry unresolved in tables; never emit fabricated or
        # nonfinite plot coordinates to the caller's strict JSON/plot pipeline.
        if not all(np.isfinite(np.asarray(g['uz'], dtype=float)).all() for g in (left, right)):
            continue
        if row['support_run_z_min'] is not None:
            center = (row['support_run_z_min']+row['support_run_z_max'])/2
            center_source = 'supported_run'
        else:
            center = (max(left['z_min'], right['z_min'])+min(left['z_max'], right['z_max']))/2
            center_source = 'overlap'
        result.append({'left_id': row['left_id'], 'right_id': row['right_id'],
                       'slice_key_left': group_keys[row['left_id']], 'slice_key_right': group_keys[row['right_id']],
                       's_left': float(left['s']), 's_right': float(right['s']),
                       'left_uz': np.asarray(left['uz']).tolist(), 'right_uz': np.asarray(right['uz']).tolist(),
                       'decision': row['illustrative_decision'], 'metrics': dict(row),
                       'display_z_low': float(center-.5), 'display_z_high': float(center+.5),
                       'z_window_center_source': center_source,
                       'threshold_status': 'ILLUSTRATIVE_NOT_APPROVED', 'manual_label': ''})
    return result


def run_track_bridge(frozen, output) -> dict:
    """Write five analysis artifacts and return their summary plus <=6 plot inputs.

    Original links are immutable. All 80 grids only delete sufficiently sampled
    original links; 0/0/0 preserves their exact set even with absent metrics.
    Output must be outside the prior/baseline directories and must not overwrite
    existing diagnostic files. The caller owns execution profiling and plotting.
    """
    prior, output = Path(frozen.prior).resolve(), Path(output).resolve()
    protected = [prior]
    if getattr(frozen, 'baseline', None) is not None:
        protected.append(Path(frozen.baseline).resolve())
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise ValueError('Diagnostic output must be outside the frozen prior/baseline directories')
    if any((output/name).exists() for name in OUTPUT_FILES):
        raise FileExistsError('Track diagnostic outputs already exist; use a fresh output directory')
    levels = np.asarray(frozen.levels, dtype=float)
    s_values = np.asarray(frozen.s, dtype=float)
    if (levels.ndim != 1 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0)
            or s_values.ndim != 1 or len(s_values) != len(frozen.keys)
            or not np.isfinite(s_values).all() or np.any(np.diff(s_values) <= 0)):
        raise ValueError('Frozen levels and s must be finite, strictly increasing arrays')
    tolerance, tolerance_source = load_matching_tolerance(prior)
    byid, group_keys, slice_indices = {}, {}, {}
    for si, key in enumerate(frozen.keys):
        for group in frozen.groups[key]:
            gid = group['id']
            if gid in byid:
                raise ValueError(f'Duplicate frozen group ID {gid}')
            if (not all(math.isfinite(float(group[k])) for k in ('s', 'z_min', 'z_max'))
                    or group['z_max'] < group['z_min'] or not math.isclose(group['s'], s_values[si], abs_tol=1e-8)):
                raise ValueError(f'Invalid frozen group bounds or s: {gid}')
            byid[gid], group_keys[gid], slice_indices[gid] = group, key, si
    ids, index, tracks, ownership, lengths = _baseline_graph(prior, byid, slice_indices)
    rows = _read_csv(prior/'baseline_consistency.csv')
    seen, needed_ids = set(), set()
    for row in rows:
        key = row['left_id'], row['right_id']
        if key in seen:
            raise ValueError('Duplicate candidate link in baseline_consistency.csv')
        seen.add(key)
        if any(gid not in byid for gid in key) or slice_indices[key[1]]-slice_indices[key[0]] != 1:
            raise ValueError(f'Candidate is not an adjacent frozen group pair: {key}')
        needed_ids.update(key)
        if row['matched'] not in ('True', 'False'):
            raise ValueError('Expected an explicit True/False baseline matched flag')
        row['matched'] = row['matched'] == 'True'
        for field in ('s_left', 's_right', 'Ch', 'Cz', 'median_du', 'raw_nearest_median_du', 'C'):
            if field in row:
                row[field] = float(row[field]) if row[field] else None
                if row[field] is not None and not math.isfinite(row[field]):
                    row[field] = None
        for field in ('common_levels', 'supported_levels'):
            row['baseline_'+field] = int(row[field]) if row.get(field) else None
        if row['matched'] and ownership[index[key[0]]] != ownership[index[key[1]]]:
            raise ValueError('Original matched link crosses saved track memberships')
    matched = [row for row in rows if row['matched']]
    edges = np.array([(index[r['left_id']], index[r['right_id']]) for r in matched], dtype=int).reshape(-1, 2)
    component_count, baseline_labels = _components(len(ids), edges)
    # No crossing membership plus equal component count proves partition equality.
    if component_count != len(tracks):
        raise ValueError('Saved track partition differs from the original matched link graph')
    slices = np.array([slice_indices[gid] for gid in ids])
    baseline_stats, _, _, _ = _component_stats(baseline_labels, component_count, ownership, lengths, slices, len(tracks))
    baseline = {**baseline_stats, 'matched_links': len(matched), 'candidate_pairs': len(rows),
                'prior_track_partition_exact': True}
    del baseline_labels

    evidence = build_group_evidence(frozen, needed_ids, tolerance)
    available_levels = set(frozen.horizontal)
    for row in rows:
        a, b = row['left_id'], row['right_id']
        row.update(bridge_metrics(byid[a], byid[b], levels, evidence[a], evidence[b], available_levels))
        row['original_track_id'] = tracks[ownership[index[a]]]['track_id'] if row['matched'] else None
    del evidence, needed_ids, seen
    values = np.array([[r['L_support'], r['R_bridge'], r['R_overlap']] for r in matched], dtype=float).reshape(-1, 3)
    eligible = np.array([r['gate_eligible'] for r in matched], dtype=bool) & np.isfinite(values).all(axis=1)
    undersampled = np.array([r['undersampled'] for r in matched], dtype=bool)
    major = sorted(range(len(tracks)), key=lambda ti: (-tracks[ti]['length_sum_m'], tracks[ti]['track_id']))[:20]
    sweep, preservation = [], []
    illustrative_labels = illustrative_children = illustrative_keep = None
    for thresholds in product(L_MINIMA, BRIDGE_MINIMA, OVERLAP_MINIMA):
        failed = _failed_gates(values, eligible, thresholds)
        keep = ~failed.any(axis=1)
        count, labels = _components(len(ids), edges[keep])
        stats, sizes, child_lengths, children = _component_stats(labels, count, ownership, lengths, slices, len(tracks))
        status = 'ILLUSTRATIVE_NOT_APPROVED' if thresholds == ILLUSTRATIVE else 'EXPLORATORY_NOT_APPROVED'
        row = {**_threshold_fields(thresholds), 'threshold_status': status, **stats,
               'baseline_matched_links': len(matched), 'retained_links': int(keep.sum()),
               'removed_links': int((~keep).sum()), 'eligible_sampled_links': int(eligible.sum()),
               'protected_undersampled_links': int(undersampled.sum()), 'unresolved_matched_links': int((~eligible).sum()),
               'questionable_short_sampled_bridges': int(failed[:, :2].any(axis=1).sum()),
               'overlap_only_removed_links': int((failed[:, 2] & ~failed[:, :2].any(axis=1)).sum()),
               'confirmed_overmerges': None, 'human_labels_available': False}
        if thresholds == (0., 0., 0.):
            if not keep.all() or any(stats[key] != baseline[key] for key in baseline_stats):
                raise AssertionError('Zero grid changed the baseline association')
        sweep.append(row)
        preservation.extend(_preservation_rows(tracks, major, labels, sizes, child_lengths, thresholds, status))
        if thresholds == ILLUSTRATIVE:
            illustrative_labels, illustrative_children, illustrative_keep = labels, children, keep
    illustrative = next(r for r in sweep if r['threshold_status'] == 'ILLUSTRATIVE_NOT_APPROVED')
    failed = _failed_gates(values, eligible, ILLUSTRATIVE)
    for i, row in enumerate(matched):
        row['illustrative_retained'] = bool(illustrative_keep[i])
        row['illustrative_decision'] = ('RETAINED_UNRESOLVED' if not eligible[i] else 'RETAINED_SAMPLED') if illustrative_keep[i] else 'REMOVED_ILLUSTRATIVE'
        row['questionable_short_sampled_bridge'] = bool(failed[i, :2].any())
        row['failed_illustrative_gates'] = [name for name, fail in zip(('L_support', 'R_bridge', 'R_overlap'), failed[i]) if fail]
    for row in rows:
        if not row['matched']:
            row.update(illustrative_retained=False, illustrative_decision='BASELINE_UNMATCHED',
                       questionable_short_sampled_bridge=False, failed_illustrative_gates=[])
    cases = []
    for i, row in enumerate(matched):
        if illustrative_keep[i]:
            continue
        a, b = edges[i]
        cases.append({**row, 'left_child_id': int(illustrative_labels[a]), 'right_child_id': int(illustrative_labels[b]),
                      'original_track_child_count': int(illustrative_children[ownership[a]]),
                      'endpoints_in_distinct_children': bool(illustrative_labels[a] != illustrative_labels[b]),
                      'threshold_status': 'ILLUSTRATIVE_NOT_APPROVED', 'confirmed_overmerge': None,
                      'manual_label': '', 'manual_notes': ''})
    representatives = _representatives(rows, byid, group_keys)
    summary = {'analysis_only': True, 'threshold_status': 'ILLUSTRATIVE_NOT_APPROVED',
               'production_merge_threshold': None, 'prior_directory': str(prior),
               'matching_tolerance_m': tolerance, 'matching_tolerance_source': tolerance_source,
               'baseline': baseline, 'baseline_zero_grid_exact': True, 'zero_grid_matching_symmetric_difference': 0,
               'sweep_grid_count': len(sweep), 'illustrative': illustrative,
               'illustrative_major_structure_preservation': [r for r in preservation if r['threshold_status'] == 'ILLUSTRATIVE_NOT_APPROVED'],
               'unresolved_candidate_pairs': sum(not r['gate_eligible'] for r in rows),
               'undersampled_candidate_pairs': sum(r['undersampled'] for r in rows),
               'confirmed_overmerges': None, 'human_labels_available': False,
               'metric_definitions': {
                   'Ch_Cz_median_du': 'Reused prior baseline_consistency.csv values; original matched flags never recomputed.',
                   'sampled_levels': 'Existing clean horizontal records in the closed group Z overlap.',
                   'left_right_sampled_levels': 'Finite positive-ray vertical group crossings in the overlap.',
                   'common_levels': 'Common finite positive-ray vertical group crossings, including H-missing levels.',
                   'common_sampled_levels': 'Common vertical levels with finite positive-ray H crossings at BOTH s positions.',
                   'sampled_Ch': 'New supported_levels/common_levels; missing H remains in this diagnostic denominator.',
                   'L_support': 'Longest consecutive frozen-level-index supported run, last center Z minus first; singleton=0.',
                   'R_overlap_R_bridge': 'Z intersection or L_support divided by MINIMUM of the two group heights, not union.',
                   'support': 'Same within-level clean monotone branch at both s, each within the prior radial tolerance.',
                   'undersampling': 'Fewer than two common_sampled_levels: protected in every grid, unresolved.',
                   'ambiguity': 'Multiple shared branches within a level; IDs are not comparable across levels.',
                   'preservation': 'Largest child by group count and largest child by summed length are measured separately.',
                   'child_ids': 'Local connected-component labels within each grid only.',
                   'comparison_epsilon': EPSILON},
               'limitations': ['Sampled short bridges are questionable, not confirmed overmerges; no human labels exist.',
                              'Missing H interrupts observed support but does not prove a geological discontinuity.',
                              'At least two common sampled levels permits exploratory gates, not geological validation.',
                              'No support is inferred between nonconsecutive levels or outside sampled centers.'],
               'representative_pairs': representatives,
               'artifacts': {name: str(output/name) for name in OUTPUT_FILES}}
    output.mkdir(parents=True, exist_ok=True)
    pair_fields = list(dict.fromkeys(k for row in rows for k in row)) or ['left_id', 'right_id', 'matched']
    _save_csv(output/OUTPUT_FILES[0], rows, pair_fields)
    _save_csv(output/OUTPUT_FILES[1], sweep, list(sweep[0]))
    case_fields = pair_fields+['left_child_id', 'right_child_id', 'original_track_child_count',
                              'endpoints_in_distinct_children', 'threshold_status', 'confirmed_overmerge', 'manual_label', 'manual_notes']
    _save_csv(output/OUTPUT_FILES[2], cases, case_fields)
    _save_csv(output/OUTPUT_FILES[3], preservation, list(preservation[0]) if preservation else
              ['L_support_min_m', 'R_bridge_min', 'R_overlap_min', 'original_track_id', 'child_count'])
    with (output/OUTPUT_FILES[4]).open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return summary
