"""Scoped, read-only baseline experiment for canonical observed-first LOCC.

The infer phase runs once; finish renders its checkpoint without re-running
recognition or inference. Production files and frozen inputs are never written.
"""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import numpy as np
import psutil

from locc_data import FrozenProfiles, AdaptiveMeshIndex, opc, clip_out_gap, xyz_from_uz
from validate_orthogonal_scanline_constraint import StageSampler, save_json, save_csv
from validate_locc_natural_gaps import frozen_stamp, validate_output_location, file_digest
from validate_locc_masking import REGIMES, evaluate_curve, summarize_group
from vertical_profile_canonicalization import canonicalize_vertical, endpoint_gap_diagnostics
from vertical_topology_reconstruction import (analyze_vertical_candidate, reconstruct_observed_first,
                                             propose_topology_stitches, joint_endpoint_pairing)
from local_surface_branch_tracking import build_neighbor_surface_tracks, apply_neighbor_surface_tracks
from local_orthogonal_cooperative_constraint import solve_locc

ROOT = Path(__file__).resolve().parents[2]
METHOD_D = 'OBSERVED_FIRST_LOCC'


def target_geometry(frozen, key, low=None, high=None, *, return_indices=False):
    data = frozen.slices[key]['slicing']
    lines = np.asarray(data['lines_3d'])
    suz = opc.to_suz(lines, frozen.arc)
    # An infinite cutting plane also intersects the opposite radial ray.
    keep = np.all(abs(suz[:, :, 0]-float(key)) < 1e-4, axis=1)
    if low is not None:
        keep &= (lines[:, :, 2].max(axis=1) >= low) & (lines[:, :, 2].min(axis=1) <= high)
    result = (lines[keep], np.asarray(data['face_ids'])[keep])
    return (*result, np.flatnonzero(keep)) if return_indices else result


def target_canonical_profile(frozen, key):
    lines, faces, original_indices = target_geometry(frozen, key, return_indices=True)
    profile = canonicalize_vertical(lines, faces)
    profile.source_segment_indices = [[int(original_indices[i]) for i in indices] for indices in profile.source_segment_indices]
    return profile


def edge_xyz_for_recognition(edge, profile, s, arc):
    points = xyz_from_uz(np.asarray(edge['points_uz']), s, arc)
    for i, node in enumerate(edge['nodes']):
        if isinstance(node, (int, np.integer)):
            # Canonical endpoints can be slightly off the analytic ray after
            # round6. Reprojection must not manufacture a different node key.
            points[i] = profile.nodes[node]
    return points


def neighbor_profiles(frozen, case):
    height = case['upper_uz'][1]-case['lower_uz'][1]
    pad = max(.1, height*.25)
    result = []
    for key in frozen.neighbors(case['slice_key']):
        if key == case['slice_key']:
            continue
        lines, _ = target_geometry(frozen, key, case['lower_uz'][1]-pad, case['upper_uz'][1]+pad)
        result.append({'s': float(key), 'lines_uz': opc.to_suz(lines, frozen.arc)[:, :, 1:]})
    return result


def tracked_window(frozen, case, window, guide=None):
    if guide is None:
        mode = 'HORIZONTAL_ONLY' if any(l['horizontal'] for l in window['layers']) else 'VERTICAL_ONLY'
        guide = solve_locc(window, mode)['curve_uz']
    tracks = build_neighbor_surface_tracks(neighbor_profiles(frozen, case),
                                          [l['z'] for l in window['layers']], guide, window['s'])
    return apply_neighbor_surface_tracks(window, tracks), tracks


def subwindow_builder(index, frozen, parent_window, dropout=False):
    """Reuse parent branch IDs even at adaptive subinterval heights."""
    def build(case):
        local = index.build_window(case)
        window = copy.deepcopy(local['window'])
        levels = [l['z'] for l in window['layers']]
        tracks = parent_window.get('neighbor_surface_tracks', [])
        layers = []
        for z in levels:
            observations = []
            for track in tracks:
                points = np.asarray(track['points_uz'])
                values = [float(np.interp(z, points[:, 1], points[:, 0]))] if points[0, 1] <= z <= points[-1, 1] else []
                observations.append({'s': track['s'], 'values': values,
                                     'track_id': f"{track['s']}:{track['branch_id']}:{track['monotone_part_id']}"})
            layers.append({'z': z, 'neighbors': observations})
        if dropout:
            for layer in window['layers']: layer['horizontal'] = []
        return apply_neighbor_surface_tracks(window, {'layers': layers, 'tracks': tracks})
    return build


def pair_evidence(case, analysis, window, tracks, old_row, profile, mesh_faces):
    coverage = analysis['coverage_ratio']
    h = np.mean([bool(layer['horizontal']) for layer in window['layers']]) if window['layers'] else 0.
    n = np.mean([t['coverage'] for t in tracks['tracks']]) if tracks['tracks'] else 0.
    sides = {int(np.sign(t['s']-window['s'])) for t in tracks['tracks'] if t['coverage'] >= .5}
    two_sided = {-1, 1}.issubset(sides)
    tangent = max(0., 1.-float(old_row['endpoint_turn_deg'])/180.)
    length = 1./max(1., float(old_row['length_ratio']))
    excursion = 1./(1.+float(old_row['radial_excursion_mm'])/50.)
    faces = []
    for node in (analysis['start_node'], analysis['stop_node']):
        fs = {face for eid, edge in enumerate(profile.edges) if node is not None and node in edge
              for face in profile.source_face_ids[eid]}
        faces.append(fs)
    adjacent = bool(faces[0] & faces[1])
    if not adjacent and all(faces):
        vertices = [{int(v) for face in fs if 0 <= face < len(mesh_faces) for v in mesh_faces[face]} for fs in faces]
        adjacent = bool(vertices[0] & vertices[1])
    # Fixed exploratory weights; geometry and H evidence dominate the soft prior.
    score = .35*coverage+.20*h+.15*n+.10*two_sided+.08*tangent+.05*length+.05*excursion+.02*adjacent
    endpoint_key = lambda p: (case['slice_key'], *tuple(np.round(p, 6)))
    return {'candidate_id': case['candidate_id'], 'lower_key': endpoint_key(case['lower_uz']),
            'upper_key': endpoint_key(case['upper_uz']), 'pair_score': float(score),
            'pair_evidence': {'observed_coverage': coverage, 'real_h_coverage': float(h), 'neighbor_track_coverage': float(n),
                              'both_neighbor_sides': two_sided, 'tangent_score': tangent, 'length_score': length,
                              'excursion_score': excursion, 'soft_endpoint_face_adjacency': adjacent}}


def audit_result(profile, uz, result, digest_before):
    digest_after = hashlib.sha256(profile.nodes.tobytes()+profile.edges.tobytes()).hexdigest()
    if digest_after != digest_before:
        raise AssertionError('Canonical observed geometry mutated')
    overlap_count = 0
    for edge in result['path_edges']:
        if edge['source'] == 'OBSERVED_VERTICAL':
            eid = edge['edge_id']
            actual, expected = np.asarray(edge['points_uz']), uz[profile.edges[eid]]
            if not (np.array_equal(actual, expected) or np.array_equal(actual[::-1], expected)):
                raise AssertionError('Selected observed edge coordinates changed')
            if edge['source_face_ids'] != profile.source_face_ids[eid]:
                raise AssertionError('Observed FaceID provenance changed')
        else:
            if edge['face_id'] is not None:
                raise AssertionError('Synthetic edge must not have a FaceID')
    for edge in result['inferred_edges']:
        low, high = sorted(p[1] for p in edge['points_uz'])
        overlap = sum(max(0., min(high, y)-max(low, x)) for x, y in result['observed_intervals'])
        overlap_count += overlap > 1e-8
    if overlap_count:
        raise AssertionError('Inferred geometry overlaps observed corridor intervals')
    return {'observed_input_unchanged': True, 'selected_observed_edges_exact': True,
            'inferred_observed_interval_overlaps': int(overlap_count)}


def mask_profile(frozen, case):
    low, high = case['lower_uz'][1], case['upper_uz'][1]
    lines, faces, original_indices = target_geometry(frozen, case['slice_key'], low-.1, high+.1, return_indices=True)
    clipped, provenance, source_indices = [], [], []
    for line, face, original_index in zip(lines, faces, original_indices):
        # Prevent quantization from leaking sub-micrometre fragments INTO a
        # deliberately hidden interval. Extra removed width is only 3 um/side.
        pieces = clip_out_gap(line[None], low-3e-6, high+3e-6)
        clipped.extend(pieces); provenance.extend([int(face)]*len(pieces))
        source_indices.extend([int(original_index)]*len(pieces))
    profile = canonicalize_vertical(np.asarray(clipped).reshape(-1, 2, 3), provenance)
    profile.source_segment_indices = [[source_indices[i] for i in ids] for ids in profile.source_segment_indices]
    return profile


def run_method_d(frozen, index, old, output):
    rows = [{**row, 'baseline_reused': True, 'this_run_solve_seconds': 0.} for row in old['mask_rows']]
    solutions, unresolved = {}, []
    for i, case in enumerate(old['mask_cases']):
        local = index.build_window(case)
        profile = mask_profile(frozen, case)
        uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
        stamp = hashlib.sha256(profile.nodes.tobytes()+profile.edges.tobytes()).hexdigest()
        for regime in REGIMES:
            original_window = copy.deepcopy(local['window'])
            if regime == 'TARGET_HORIZONTAL_DROPOUT':
                for layer in original_window['layers']: layer['horizontal'] = []
            start = time.perf_counter()
            window, tracks = tracked_window(frozen, case, original_window)
            result = reconstruct_observed_first(profile, uz, window, candidate_id=f"{case['candidate_id']}:{regime}",
                window_builder=subwindow_builder(index, frozen, window, regime == 'TARGET_HORIZONTAL_DROPOUT'))
            seconds = time.perf_counter()-start
            invariant = audit_result(profile, uz, result, stamp)
            row = {k: case[k] for k in ('candidate_id', 'slice_key', 's', 'gap_height', 'gap_bucket', 'spatial_block', 'split')}
            row.update(method=METHOD_D, regime=regime, solve_seconds=seconds, this_run_solve_seconds=seconds,
                       baseline_reused=False, gap_type=result['gap_type'], status=result['status'], **invariant)
            if len(result['curve_uz']) >= 2:
                curve = result['curve_uz']
                # Branch evaluation uses original H levels for comparability,
                # not the possibly denser subinterval solver's node order.
                selected = [{'u': float(curve[0, 0])}]+[{'u': float(np.interp(l['z'], curve[:, 1], curve[:, 0]))}
                           for l in original_window['layers']]+[{'u': float(curve[-1, 0])}]
                row.update(evaluate_curve({'curve_uz': curve, 'selected_nodes': selected}, case['truth_curve'], original_window))
            else:
                unresolved.append((case['candidate_id'], regime))
            rows.append(row); solutions[(case['candidate_id'], regime)] = result
        if (i+1) % 25 == 0: print(f'Method D masking {i+1}/150', flush=True)
    summary = []
    for split in ('calibration', 'holdout', 'all'):
        for regime in REGIMES:
            for method in ('VERTICAL_ONLY', 'HORIZONTAL_ONLY', 'LOCC', METHOD_D):
                subset = [r for r in rows if (split == 'all' or r['split'] == split) and r['regime'] == regime and r['method'] == method]
                resolved = [r for r in subset if 'p95_abs_u_m' in r]
                summary.append({'split': split, 'regime': regime, 'method': method, 'total': len(subset),
                                'unresolved': len(subset)-len(resolved), **(summarize_group(resolved) if resolved else {})})
    save_csv(output/'masking_four_methods.csv', rows)
    save_csv(output/'masking_four_method_summary.csv', summary)
    return {'rows': rows, 'summary': summary, 'unresolved': unresolved, 'solutions': solutions,
            'warning': 'Same 150 locations: regression, not a new blind test; A/B/C are frozen reused results.'}


def infer(args):
    output = validate_output_location(args.prior, args.output)
    old_dir = args.previous.resolve()
    if output == old_dir or output.is_relative_to(old_dir) or old_dir.is_relative_to(output):
        raise ValueError('Do not overwrite or contain the previous LOCC run')
    output.mkdir(parents=True, exist_ok=False)
    stages, started = [], time.perf_counter()
    stamp = frozen_stamp(args.prior)
    classification_path = old_dir/'review_classification_193/classification.json'
    source_hashes = {str(path): file_digest(path) for path in [old_dir/'inference_checkpoint.pkl', classification_path]}
    with StageSampler('load_frozen_inputs', output, stages):
        frozen = FrozenProfiles(args.prior)
        with (old_dir/'inference_checkpoint.pkl').open('rb') as stream: old = pickle.load(stream)
        classification = json.loads(classification_path.read_text(encoding='utf-8'))
        class_rows = {r['candidate_id']: r for r in classification['rows']}
        cases = [c for c in old['natural_cases'] if c['candidate_id'] in class_rows]
        if len(cases) != 193: raise ValueError('Expected the audited 193-case cohort')
        index = AdaptiveMeshIndex(frozen)
    keys = {c['slice_key'] for c in cases}
    full, target, stats, endpoint_rows, all_changed = {}, {}, [], [], []
    with StageSampler('vertical_canonicalization', output, stages):
        for i, key in enumerate(frozen.keys):
            data = frozen.slices[key]['slicing']
            profile = canonicalize_vertical(data['lines_3d'], data['face_ids'])
            stats.append({'slice_key': key, **profile.stats})
            if profile.stats['exact_duplicate_vertical_edges']: all_changed.append(key)
            # Source-index coverage is verified independently of edge count.
            indices = [j for group in profile.source_segment_indices for j in group]
            if sorted(indices) != list(range(len(data['lines_3d']))):
                raise AssertionError(f'Lost raw segment provenance: {key}')
            endpoint_rows.extend({'slice_key': key, **row} for row in endpoint_gap_diagnostics(profile))
            if key in keys:
                full[key] = profile
                target[key] = target_canonical_profile(frozen, key)
            if (i+1) % 700 == 0: print(f'Canonical geometry {i+1}/{len(frozen.keys)}', flush=True)
        save_csv(output/'canonical_slice_statistics.csv', stats)
        save_csv(output/'near_endpoint_diagnostics.csv', endpoint_rows)
    locals_, windows, tracks = {}, {}, {}
    with StageSampler('local_branch_graph', output, stages):
        for case in cases:
            cid = case['candidate_id']
            local = index.build_window(case)
            locals_[cid] = local
            windows[cid], tracks[cid] = tracked_window(frozen, case, local['window'])
    analyses = {}
    with StageSampler('coverage_classification', output, stages):
        for case in cases:
            cid, profile = case['candidate_id'], target[case['slice_key']]
            analyses[cid] = analyze_vertical_candidate(profile, opc.to_suz(profile.nodes, frozen.arc)[:, 1:], windows[cid])
    with StageSampler('endpoint_pairing', output, stages):
        evidence = [pair_evidence(c, analyses[c['candidate_id']], windows[c['candidate_id']], tracks[c['candidate_id']],
                                 class_rows[c['candidate_id']], target[c['slice_key']], index.faces) for c in cases]
        pairs = {r['candidate_id']: r for r in joint_endpoint_pairing(evidence)}
    results, stitches = {}, {}
    with StageSampler('topology_stitching', output, stages):
        for case in cases:
            cid, profile = case['candidate_id'], target[case['slice_key']]
            uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
            stitches[cid] = propose_topology_stitches(profile, uz, analyses[cid], windows[cid])
            results[cid] = reconstruct_observed_first(profile, uz, windows[cid], analysis=analyses[cid],
                                                     topology_stitches=stitches[cid], allow_inference=False)
    rows, extras = [], defaultdict(list)
    with StageSampler('LOCC_missing_interval_inference', output, stages):
        for case in cases:
            cid, key = case['candidate_id'], case['slice_key']
            profile, window = target[key], windows[cid]
            uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
            before = hashlib.sha256(profile.nodes.tobytes()+profile.edges.tobytes()).hexdigest()
            if not len(results[cid]['curve_uz']) and analyses[cid]['gap_type'] != 'TYPE_I':
                results[cid] = reconstruct_observed_first(profile, uz, window, candidate_id=cid,
                    analysis=analyses[cid], topology_stitches=stitches[cid], window_builder=subwindow_builder(index, frozen, window))
            result = results[cid]
            invariant = audit_result(profile, uz, result, before)
            pairing = pairs[cid]
            eligible = (pairing['selected'] and not pairing['pairing_ambiguous'] and len(result['curve_uz']) >= 2
                        and (not result['boundary_candidate'] or not result['inferred_edges']))
            # A natural inference that falls back completely to priors is never
            # inserted into the experimental re-recognition input automatically.
            if result['inferred_edges'] and not any(e['source'] == 'ORTHOGONAL_INFERRED' for e in result['inferred_edges']):
                eligible = False
            result['applied_to_experimental_recognition'] = bool(eligible)
            if eligible:
                for edge in result['topology_stitches']+result['inferred_edges']:
                    extras[key].append({'points_xyz': edge_xyz_for_recognition(edge, profile, float(key), frozen.arc),
                                        'source': edge['source'], 'face_id': None, 'candidate_id': cid})
            row = {'candidate_id': cid, 'slice_key': key, 'old_category': class_rows[cid]['primary_category'],
                   'old_existing_sample_fraction': class_rows[cid]['target_existing_sample_fraction'],
                   'gap_type': result['gap_type'], 'coverage_ratio': result['coverage_ratio'],
                   'observed_intervals': result['observed_intervals'], 'missing_intervals': result['missing_intervals'],
                   'status': result['status'], 'pair_score': pairing['pair_score'], 'selected': pairing['selected'],
                   'pairing_ambiguous': pairing['pairing_ambiguous'], 'pairing_status': pairing['pairing_status'],
                   'boundary_candidate': result['boundary_candidate'], 'inferred_length_m': result['inferred_length_m'],
                   'topology_stitches': len(result['topology_stitches']), 'topology_stitch_length_m': result['topology_stitch_length_m'],
                   'observed_path_connected': result['observed_path_connected'],
                   'applied_to_experimental_recognition': bool(eligible), **invariant}
            rows.append(row)
        save_csv(output/'candidate_classification_193.csv', rows)
        save_json(output/'candidate_proposals.json', {'pairings': pairs, 'results': results})
    # Keep expensive core results resumable without silently repeating them.
    checkpoint = {'provenance_schema': 2, 'cases': cases, 'classification': classification, 'rows': rows, 'results': results, 'windows': windows,
                  'tracks': tracks, 'locals': locals_, 'target_profiles': target, 'full_profiles': full, 'extras': dict(extras),
                  'canonical_rows': stats, 'global_changed_slice_keys': all_changed,
                  'near_endpoint_bucket_counts': dict(Counter(r['bucket'] for r in endpoint_rows)), 'pairs': pairs,
                  'old_natural_solutions': {c['candidate_id']: old['natural_solutions'][c['candidate_id']] for c in cases},
                  'arc': frozen.arc, 'source_hashes': source_hashes, 'frozen_before': stamp}
    with (output/'core_checkpoint.pkl').open('xb') as stream: pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with StageSampler('masking_method_D', output, stages):
        checkpoint['masking'] = run_method_d(frozen, index, old, output)
    from vertical_canonical_recognition_audit import run_changed_recognition, run_local_track_check
    with StageSampler('changed_slice_re_recognition', output, stages):
        checkpoint['recognition'] = run_changed_recognition(frozen, full, dict(extras), output)
    with StageSampler('track_regrouping', output, stages):
        checkpoint['track'] = run_local_track_check(frozen, checkpoint['recognition']['groups_by_slice'], output)
    unchanged = stamp == frozen_stamp(args.prior) and all(file_digest(p) == digest for p, digest in source_hashes.items())
    if not unchanged: raise AssertionError('A frozen input changed during validation')
    checkpoint['frozen_unchanged'] = unchanged
    checkpoint['inference_wall_seconds'] = time.perf_counter()-started
    checkpoint['stages'] = stages
    checkpoint['manifest'] = {'created_at': datetime.now().astimezone().isoformat(), 'prior': str(args.prior),
        'previous_locc': str(args.previous), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'commit': None, 'platform': platform.platform(), 'logical_cpus': psutil.cpu_count(),
        'physical_cores': psutil.cpu_count(logical=False), 'ram_bytes': psutil.virtual_memory().total, 'gpu_used': False,
        'processes': 1, 'threads_numeric': 1, 'corridor_half_width_m': .03, 'stitch_max_distance_m': 1e-4,
        'pair_score_min': .5, 'pair_ambiguity_margin': .03, 'mask_boundary_guard_m': 3e-6,
        'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in [Path(__file__),
             *[ROOT/'scripts/04_structure_recognition'/name for name in ('vertical_profile_canonicalization.py',
                 'vertical_topology_reconstruction.py', 'local_surface_branch_tracking.py', 'local_orthogonal_cooperative_constraint.py')]]}}
    save_json(output/'manifest.json', checkpoint['manifest'])
    with (output/'validation_checkpoint.pkl').open('xb') as stream: pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'INFERENCE COMPLETE {checkpoint["inference_wall_seconds"]:.3f}s; {dict(Counter(r["gap_type"] for r in rows))}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('infer', 'finish'), required=True)
    args = parser.parse_args()
    if args.phase == 'infer': infer(args)
    else:
        from vertical_canonical_locc_report import finish
        finish(args.output)


if __name__ == '__main__': main()
