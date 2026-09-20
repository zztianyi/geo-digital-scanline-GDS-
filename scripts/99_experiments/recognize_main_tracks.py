"""Export actual assembled paths for every frozen slice and local review windows.

Reuses the unchanged cross-slice evidence cache; recomputes branch geometry and
all final paths. No hidden target or historical ROI enters path assembly.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import ast
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np

from locc_data import FrozenProfiles, opc
from validate_observed_first_locc import target_canonical_profile
from validate_orthogonal_scanline_constraint import StageSampler, save_csv, save_json
from dominant_observed_branch import extract_observed_branches, solve_dominant_branch
from surface_track_local_review import clipped_segments, vertical_coverage

ROOT = Path(__file__).resolve().parents[2]


def cached_evidence(directory, frozen):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    # The cache may only be reused while its evidence-building code is unchanged.
    for name in ('observed_surface_graph.py', 'horizontal_surface_link.py', 'vertical_profile_canonicalization.py'):
        relative = 'scripts\\04_structure_recognition\\'+name
        if hashlib.sha256((ROOT/'scripts/04_structure_recognition'/name).read_bytes()).hexdigest() != manifest['code_sha256'][relative]:
            raise ValueError('Evidence cache source mismatch: '+name)
    if Path(manifest['prior']).resolve() != frozen.prior.resolve():
        raise ValueError('Evidence cache belongs to a different frozen input')
    tracks, membership, support = [], {}, defaultdict(list)
    with (directory/'surface_track_inventory.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            for k in ('members', 'branch_ids', 'slice_span', 'slice_count', 'ambiguous', 'stable', 'ambiguous_link_count'):
                row[k] = ast.literal_eval(row[k])
            row['members'] = [tuple(n) for n in row['members']]
            tracks.append(row)
            membership.update({n: row['track_id'] for n in row['members']})
    links = []
    with (directory/'surface_track_links.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            row = {k: ast.literal_eval(v) for k, v in row.items()}
            links.append(row)
            for n in (row['a'], row['b']):
                support[n].append(row)
    return dict(nodes={}, tracks=tracks, membership=membership, support=dict(support), links=links,
        observations=[], level_z=dict(enumerate(map(float, frozen.levels))), slice_order=list(frozen.s))


def verify_result(profile, uz, result):
    records = result['path_edges']
    observed = [r for r in records if r['source'].startswith('OBSERVED')]
    if observed:
        ids = np.asarray([r['node_ids'] for r in observed])
        t = np.asarray([[r['t0'], r['t1']] for r in observed])
        for field, original in (('points_uz', uz), ('points_xyz', profile.nodes)):
            p = original[ids]
            expected = p[:, :1]+t[..., None]*(p[:, 1:]-p[:, :1])
            np.testing.assert_allclose([r[field] for r in observed], expected, atol=2e-10, rtol=0)
        used = defaultdict(list)
        for r in observed:
            if set(r['node_ids']) != set(profile.edges[r['edge_id']]) or r['source_face_ids'] != profile.source_face_ids[r['edge_id']]:
                raise AssertionError('Observed provenance changed')
            used[r['edge_id']].append(sorted((r['t0'], r['t1'])))
        for intervals in used.values():
            intervals.sort()
            if any(a[1] > b[0]+1e-9 for a, b in zip(intervals, intervals[1:])):
                raise AssertionError('Repeated observed interval')
    for r in records:
        if not r['source'].startswith('OBSERVED') and (r.get('face_id') is not None or r.get('source_face_ids')):
            raise AssertionError('Connector has observed provenance')
    if len(records) > 1:
        for field in ('points_uz', 'points_xyz'):
            np.testing.assert_allclose([r[field][1] for r in records[:-1]], [r[field][0] for r in records[1:]], atol=2e-8, rtol=0)
    sequence = []
    for r in observed:
        if not sequence or sequence[-1] != r['branch_id']:
            sequence.append(r['branch_id'])
    if observed and len(sequence)-1 != result['branch_switch_count']:
        raise AssertionError('Branch sequence/count mismatch')


def _bounds(center, span=4.5):
    u, z = map(float, center)
    return [u-span*.43, u+span*.43, z-span/2, z+span/2]


def historical_bounds(plot):
    endpoints = plot['endpoints']
    center = np.mean(endpoints, axis=0)
    span = max(4.5, 2.5*float(np.ptp(endpoints[:, 1])))
    bounds = _bounds(center, span)
    bounds[0] = min(bounds[0], float(endpoints[:, 0].min())-.6)
    bounds[1] = max(bounds[1], float(endpoints[:, 0].max())+.6)
    return bounds


def region_diagnostics(key, branches, result, old_curve, historical):
    windows, local = [], []
    curve = result['curve_uz']
    for i, j in enumerate(result.get('junctions', [])):
        categories = []
        if j['xyz_distance_m'] > .01:
            categories.append('LONG_CONNECTOR')
        if (j.get('tangent_turn_deg') or 0.) > 60.:
            categories.append('SHARP_JUNCTION')
        for category in categories:
            center = (np.asarray(j['a_point_uz'])+j['b_point_uz'])/2
            span = max(4.5, j['distance_m']*2.5)
            windows.append(dict(region_id=f'S{float(key):07.2f}_J{i}_{category}', slice_key=key, category=category,
                bounds=_bounds(center, span), severity=j['xyz_distance_m'] if category == 'LONG_CONNECTOR' else j['tangent_turn_deg'],
                detail=f"{j['from_branch_id']} -> {j['to_branch_id']}", junction_index=i))
    from main_track_assembly import _remaining
    for piece_index, piece in enumerate(_remaining([b for b in branches if b['kind'] != 'CLOSED_COMPONENT'], result['path_edges'])):
        points = np.asarray([piece[0]['points_uz'][0]]+[e['points_uz'][1] for e in piece])
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        bid = piece[0]['branch_id']
        if length < .5 or len(curve) < 2:
            continue
        # An unselected nearby branch is an identity question, not known error.
        p = points[::max(1, len(points)//30)]
        q = curve[::max(1, len(curve)//500)]
        distances = np.linalg.norm(p[:, None]-q[None, :], axis=2).min(axis=1)
        near = distances < .5
        if near.mean() >= .5:
            center = np.median(p[near], axis=0)
            windows.append(dict(region_id=f'S{float(key):07.2f}_B{bid}_P{piece_index}', slice_key=key,
                category='COMPETING_BRANCH', bounds=_bounds(center), severity=length,
                detail=f'unselected portion of branch {bid}', junction_index=None))
    if result.get('extent_covered') is False:
        for side in (0, 1):
            z = result['candidate_z_extent'][side]
            if abs(z-result['route_z_extent'][side]) > 1e-6:
                b = min(branches, key=lambda b: abs(b['z_range'][side]-z))
                p = b['points_uz'][np.argmin(abs(b['points_uz'][:, 1]-z))]
                windows.append(dict(region_id=f'S{float(key):07.2f}_EXT{side}', slice_key=key,
                    category='UNCOVERED_EXTENT', bounds=_bounds(p), severity=abs(z-result['route_z_extent'][side]),
                    detail='open-branch elevation extent not reached', junction_index=None))
    for cid, plot in historical:
        bounds = historical_bounds(plot)
        # Keep the original diagnostic ROI for before/after metrics. Plot bounds
        # are deliberately wider and cannot change any recognition decision.
        roi = plot['bounds']
        zl, zh = sorted(plot['endpoints'][:, 1])
        before = clipped_segments(old_curve, roi)
        after = []
        for r in result['path_edges']:
            if r['source'].startswith('OBSERVED'):
                after.extend(clipped_segments(r['points_uz'], roi))
        after = np.asarray(after).reshape(-1, 2, 2)
        row = dict(candidate_id=cid, slice_key=key, before_coverage=vertical_coverage(before, zl, zh),
            after_coverage=vertical_coverage(after, zl, zh), before_absent=not len(before), after_absent=not len(after), bounds=bounds)
        local.append(row)
        if row['after_coverage'] < .95:
            windows.append(dict(region_id=cid, slice_key=key, category='HISTORICAL_LOCAL_MISS', bounds=bounds,
                severity=1.-row['after_coverage'], detail=f'local observed Z coverage {row["after_coverage"]:.3f}', junction_index=None))
    return windows, local


def compute_slice(key, frozen, graph, historical):
    start = time.perf_counter()
    profile = target_canonical_profile(frozen, key)
    uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
    branches = extract_observed_branches(profile, uz)
    # Isolated node mapping per worker; shared evidence and neighboring arrays
    # remain read-only. This avoids copying the large frozen input per process.
    local_graph = {**graph, 'nodes': dict(graph['nodes'])}
    local_graph['nodes'].update({(float(key), b['branch_id']): b for b in branches})
    result = solve_dominant_branch(profile, uz, branches=branches, surface_graph=local_graph, target_s=float(key))
    verify_result(profile, uz, result)
    selected = result['selection']['selected']
    old_curve = selected['fragment']['points_uz'] if selected else np.empty((0, 2))
    for _, plot in historical:
        if selected and selected['branch_id'] != plot['row']['selected_branch']:
            raise AssertionError('Single-branch baseline changed from the prior audit')
    windows, local = region_diagnostics(key, branches, result, old_curve, historical)
    selected_id = selected['branch_id'] if selected else None
    result['selection'] = dict(selected_branch=selected_id, ambiguous=result['selection']['ambiguous'])
    payload = dict(slice_key=key, result=result, old_curve=old_curve,
        branches=[dict(branch_id=b['branch_id'], kind=b['kind'], points_uz=b['points_uz']) for b in branches])
    row = dict(slice_key=key, status=result['status'], anchor_branch=selected_id,
        branch_sequence=result.get('route_branch_sequence', [selected_id] if selected else []),
        switches=result['branch_switch_count'], connectors_m=result['connector_length_m'],
        observed_m=result['observed_length_m'], inferred_m=result['inferred_length_m'],
        large_connectors=result.get('large_connector_count', 0), extent_covered=result.get('extent_covered', False),
        review_windows=len(windows), observed_provenance_verified=True, wall_seconds=time.perf_counter()-start)
    return payload, row, windows, local


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'main_tracks.pkl').exists():
        raise FileExistsError('Choose a new output directory; existing numerical results are never overwritten')
    stages, started = [], time.perf_counter()
    with StageSampler('load_frozen_inputs_and_evidence', output, stages):
        frozen = FrozenProfiles(args.prior)
        graph = cached_evidence(args.evidence, frozen)
        with (args.evidence/'local_review_checkpoint_curated.pkl').open('rb') as stream:
            review = pickle.load(stream)
        historical = defaultdict(list)
        for cid, plot in review['local_plot_cases'].items():
            historical[plot['row']['slice_key']].append((cid, plot))
    with StageSampler('rebuild_full_observed_inventory', output, stages):
        for i, key in enumerate(frozen.keys):
            profile = target_canonical_profile(frozen, key)
            uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
            for b in extract_observed_branches(profile, uz):
                b.pop('records')
                graph['nodes'][(float(key), b['branch_id'])] = b
            if (i+1) % 350 == 0:
                print(f'Inventory {i+1}/{len(frozen.keys)}', flush=True)
        if set(graph['nodes']) != set(graph['membership']):
            raise AssertionError('Cached track membership does not match current observed branches')
    rows, joins, windows, local_rows, index = [], [], [], [], {}
    with StageSampler('recognize_validate_export_all_main_tracks', output, stages):
        with (output/'main_tracks.pkl').open('xb') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            keys, pending = iter(frozen.keys), deque()
            def submit_next():
                key = next(keys, None)
                if key is not None:
                    pending.append(pool.submit(compute_slice, key, frozen, graph, historical.get(key, [])))
            for _ in range(args.workers*2):
                submit_next()
            i = 0
            while pending:
                payload, row, new_windows, local = pending.popleft().result()
                submit_next()
                key, result = payload['slice_key'], payload['result']
                windows.extend(new_windows); local_rows.extend(local)
                for ji, j in enumerate(result.get('junctions', [])):
                    joins.append(dict(slice_key=key, junction_index=ji, **j))
                index[key] = stream.tell()
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                rows.append(row)
                i += 1
                if i % 100 == 0:
                    print(f'Recognized {i}/{len(frozen.keys)} | multi-branch {sum(r["switches"] > 0 for r in rows)} | last {row["wall_seconds"]:.2f}s', flush=True)
    save_json(output/'result_index.json', index)
    save_csv(output/'slice_summary.csv', rows)
    save_csv(output/'junctions.csv', joins)
    save_csv(output/'problem_regions.csv', windows)
    save_csv(output/'historical_local_comparison.csv', local_rows)
    # Store small diagnostics for display-only rerendering.
    with (output/'review_cache.pkl').open('wb') as stream:
        pickle.dump(dict(rows=rows, joins=joins, windows=windows, local_rows=local_rows,
            historical=review['local_plot_cases']), stream)
    summary = dict(slice_count=len(rows), multi_branch_slices=sum(r['switches'] > 0 for r in rows),
        switches=sum(r['switches'] for r in rows), junction_count=len(joins), real_intersections=sum(j['distance_m'] < 1e-9 for j in joins),
        large_connectors=sum(r['large_connectors'] for r in rows), longest_connector_m=max((j['distance_m'] for j in joins), default=0.),
        full_open_z_extent_slices=sum(r['extent_covered'] for r in rows), problem_regions=dict(Counter(r['category'] for r in windows)),
        historical_cases=len(local_rows), old_local_absent=sum(r['before_absent'] for r in local_rows),
        new_local_absent=sum(r['after_absent'] for r in local_rows), old_local_below95=sum(r['before_coverage'] < .95 for r in local_rows),
        new_local_below95=sum(r['after_coverage'] < .95 for r in local_rows), all_output_coordinates_and_provenance_verified=True,
        total_wall_seconds=time.perf_counter()-started, stage_performance=stages,
        evidence_cache_reused=True, missing_tail_inference_evaluated=False, legacy_red_group_entry_changed=False)
    save_json(output/'summary.json', summary)
    save_json(output/'manifest.json', dict(prior=str(args.prior.resolve()), evidence=str(args.evidence.resolve()),
        output_kind='Actual main-track path records; sequential pickle with byte-offset index',
        baseline='Prior globally selected branch, verified against all 193 historical decisions',
        review_gap_m=.01, turn_review_degrees=60., workers=args.workers,
        inference='Disabled by absence of fresh H samples; observed assembly only',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'scripts/04_structure_recognition/main_track_assembly.py', ROOT/'scripts/04_structure_recognition/dominant_observed_branch.py', Path(__file__)]}))
    from main_track_comparison_report import render_report
    render_report(output)
    # Reuse the successfully serialized summary, including NumPy scalar values.
    print((output/'summary.json').read_text(encoding='utf-8'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ROOT/'outputs/orthogonal_scanline_constraint/20260919_081148')
    parser.add_argument('--evidence', type=Path, default=ROOT/'outputs/local_detail_full_validation/20260919_current/surface_audit')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
