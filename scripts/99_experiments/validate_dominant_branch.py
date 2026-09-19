"""Frozen 193-case dominant-branch audit, closed tracks and 300 masking runs.

Run infer once into a new directory; finish only renders that checkpoint.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import psutil

from locc_data import FrozenProfiles, AdaptiveMeshIndex, opc
from validate_orthogonal_scanline_constraint import StageSampler, save_csv, save_json
from validate_locc_natural_gaps import frozen_stamp, validate_output_location, file_digest
from validate_observed_first_locc import (target_canonical_profile, mask_profile, tracked_window,
                                        subwindow_builder)
from validate_locc_masking import REGIMES, evaluate_curve, summarize_group
from vertical_topology_reconstruction import reconstruct_observed_first, joint_endpoint_pairing
from dominant_observed_branch import (extract_observed_branches, solve_dominant_branch,
                                     clip_branch, crossings)
from backtracking_junction import search_backtracking_junction
from closed_component_tracking import analyze_closed_components, track_closed_components

ROOT = Path(__file__).resolve().parents[2]
METHOD_E = 'DOMINANT_BRANCH_MINIMUM_INTERVENTION'
METHOD_D = 'OBSERVED_FIRST_LOCC'


def light_branch(branch, *, inventory=False):
    omit = {'points_uz', 'points_xyz', 'records', 'arc_positions', 'edge_arc_positions', 'fragment'}
    return {k: v for k, v in branch.items() if k not in omit}


def assert_geometry_preserved(profile, uz, result):
    observed = 0
    for record in result['path_edges']:
        if not record['source'].startswith('OBSERVED'):
            if record.get('face_id') is not None:
                raise AssertionError('Synthetic edge received FaceID')
            continue
        eid = record['edge_id']
        nodes = record['node_ids']
        if set(nodes) != set(map(int, profile.edges[eid])):
            raise AssertionError('Canonical edge identity changed')
        for name, source in (('points_uz', uz), ('points_xyz', profile.nodes)):
            a, b = source[nodes]
            expected = np.array([a+t*(b-a) for t in (record['t0'], record['t1'])])
            if not np.allclose(record[name], expected, atol=2e-10, rtol=0):
                raise AssertionError('Observed geometry moved off original edge')
        if record['source_face_ids'] != profile.source_face_ids[eid]:
            raise AssertionError('FaceID provenance changed')
        if record['source_segment_indices'] != profile.source_segment_indices[eid]:
            raise AssertionError('Raw segment provenance changed')
        observed += 1
    if result['branch_switch_count'] > 1:
        raise AssertionError('More than one branch switch')
    return observed


def use_gap_inference(result, profile, uz, window, builder, cid):
    """No-observed-area fallback, numerically the same solver as method D."""
    fallback = reconstruct_observed_first(profile, uz, window, window_builder=builder, candidate_id=cid)
    curve = np.asarray(fallback['curve_uz'])
    length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()) if len(curve) else 0.
    return {**result, 'status': 'INFERRED_REVIEW_PROPOSAL' if len(curve) else 'UNRESOLVED_NO_OBSERVED_BRANCH',
        'curve_uz': curve, 'path_edges': fallback['path_edges'], 'fallback': fallback,
        'inferred_length_m': fallback['inferred_length_m'], 'connector_length_m': fallback['topology_stitch_length_m'],
        'observed_length_m': max(0., length-fallback['inferred_length_m']-fallback['topology_stitch_length_m']),
        'observed_geometry_fraction': max(0., 1-fallback['inferred_length_m']/length) if length else 0.,
        'no_horizontal_low_confidence': not any(l['horizontal'] for l in window['layers'])}


def junction_diagnostics(case, branches, decision):
    selected = decision['selection']['selected']
    if selected is None:
        return []
    dominant = next(b for b in branches if b['branch_id'] == selected['branch_id'])
    low = float(case['lower_uz'][1])
    result = []
    for branch in branches:
        if branch['branch_id'] == dominant['branch_id'] or branch['kind'] == 'CLOSED_COMPONENT':
            continue
        # Explicitly inspect the former LOWER endpoint's incident path. This
        # is diagnostic, never a second switch appended to a retained branch.
        if min(np.linalg.norm(branch['points_uz'][0]-case['lower_uz']),
               np.linalg.norm(branch['points_uz'][-1]-case['lower_uz'])) > 3e-6:
            continue
        for a in clip_branch(branch, low-.2, low+1e-8):
            if np.linalg.norm(a['points_uz'][-1]-case['lower_uz']) > 3e-6:
                continue
            for b in clip_branch(dominant, low-.2, low+.2):
                for junction in search_backtracking_junction(a, b):
                    result.append(dict(candidate_id=case['candidate_id'], from_branch_id=branch['branch_id'],
                        to_branch_id=dominant['branch_id'], applied=False, purpose='lower_endpoint_backtracking_diagnostic',
                        **junction))
    return result


def horizontal_loop_matches(frozen, loops):
    by_level = defaultdict(list)
    for loop in loops:
        low, high = loop['z_range']
        for zi in np.flatnonzero((frozen.levels >= low) & (frozen.levels <= high)):
            by_level[int(zi)].append(loop)
    for zi, local in by_level.items():
        positions = sorted({r['s'] for r in local})
        maps = opc.level_map(opc.horizontal_crossings(frozen.horizontal[zi], positions, frozen.arc))
        lookup = {s: i for i, s in enumerate(positions)}
        for loop in local:
            values = crossings(loop['ordered_loop_uz'], float(frozen.levels[zi]))
            for u, branch, face in maps.get(lookup[loop['s']], np.empty((0, 3))):
                if len(values) and np.min(abs(values-u)) <= 1e-5:
                    loop['horizontal_matches'].append(dict(level_index=zi, z=float(frozen.levels[zi]),
                                                            u=float(u), branch_id=int(branch), face_id=int(face)))


def historic_usage(row, old, previous):
    cid, regime, method = row['candidate_id'], row['regime'], row['method']
    if method == METHOD_D:
        result = previous['masking']['solutions'][(cid, regime)]
        inferred = result['inferred_length_m']
        curve = result['curve_uz']
    else:
        result = old['mask_solutions'][(cid, regime, method)]
        curve = result['curve_uz']
        inferred = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    # Full target-Z masking removes ALL V branches in the gap for all methods.
    # Old H/LOCC branch IDs are level-local, so a V-branch switch is N/A, not 0.
    return dict(observed_geometry_fraction=0., inferred_length_m=float(inferred),
                branch_switch_count=None if method != METHOD_D else 0, virtual_junction_count=0,
                curve_length_m=length, branch_switch_definition='target observed V identity; N/A when no such identity exists')


def run_masking(frozen, index, old, previous, output):
    rows = [{**r, **historic_usage(r, old, previous), 'baseline_reused': True, 'this_run_solve_seconds': 0.}
            for r in previous['masking']['rows']]
    solutions = {}
    for i, case in enumerate(old['mask_cases']):
        local = index.build_window(case)
        profile = mask_profile(frozen, case)
        uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
        for regime in REGIMES:
            original = copy.deepcopy(local['window'])
            if regime == 'TARGET_HORIZONTAL_DROPOUT':
                for layer in original['layers']:
                    layer['horizontal'] = []
            started = time.perf_counter()
            result = solve_dominant_branch(profile, uz, original)
            if result['status'] == 'NEEDS_GAP_INFERENCE':
                window, tracks = tracked_window(frozen, case, original)
                result = use_gap_inference(result, profile, uz, window,
                    subwindow_builder(index, frozen, window, regime == 'TARGET_HORIZONTAL_DROPOUT'),
                    f"{case['candidate_id']}:{regime}")
            else:
                raise AssertionError('Hidden target-V geometry leaked into a full masking interval')
            seconds = time.perf_counter()-started
            if abs(result['observed_geometry_fraction']) > 1e-8:
                raise AssertionError('Observed target V must be absent in full mask')
            curve = result['curve_uz']
            row = {k: case[k] for k in ('candidate_id', 'slice_key', 's', 'gap_height', 'gap_bucket', 'spatial_block', 'split')}
            row.update(method=METHOD_E, regime=regime, status=result['status'], solve_seconds=seconds,
                this_run_solve_seconds=seconds, baseline_reused=False,
                observed_geometry_fraction=result['observed_geometry_fraction'],
                inferred_length_m=result['inferred_length_m'], branch_switch_count=result['branch_switch_count'],
                virtual_junction_count=result['virtual_junction_count'])
            if len(curve) >= 2:
                selected = [{'u': float(curve[0, 0])}]+[{'u': float(np.interp(l['z'], curve[:, 1], curve[:, 0]))}
                             for l in original['layers']]+[{'u': float(curve[-1, 0])}]
                row.update(evaluate_curve({'curve_uz': curve, 'selected_nodes': selected}, case['truth_curve'], original))
                old_curve = previous['masking']['solutions'][(case['candidate_id'], regime)]['curve_uz']
                row['curve_equal_method_d'] = bool(np.array_equal(curve, old_curve))
            rows.append(row)
            solutions[(case['candidate_id'], regime)] = result
        if (i+1) % 25 == 0:
            print(f'Method E {i+1}/{len(old["mask_cases"])} locations', flush=True)
    summary = []
    for split in ('calibration', 'holdout', 'all'):
        for regime in REGIMES:
            for method in ('VERTICAL_ONLY', 'HORIZONTAL_ONLY', 'LOCC', METHOD_D, METHOD_E):
                subset = [r for r in rows if (split == 'all' or r['split'] == split) and r['regime'] == regime and r['method'] == method]
                resolved = [r for r in subset if 'p95_abs_u_m' in r]
                summary.append(dict(split=split, regime=regime, method=method, total=len(subset),
                    unresolved=len(subset)-len(resolved), **(summarize_group(resolved) if resolved else {}),
                    inferred_length_m=sum(r['inferred_length_m'] for r in subset),
                    mean_observed_geometry_fraction=float(np.mean([r['observed_geometry_fraction'] for r in subset])),
                    branch_switch_count=None if method not in (METHOD_D, METHOD_E) else sum(r['branch_switch_count'] for r in subset),
                    virtual_junction_count=sum(r['virtual_junction_count'] for r in subset)))
    save_csv(output/'masking_method_comparison.csv', rows)
    save_csv(output/'masking_method_summary.csv', summary)
    return dict(rows=rows, summary=summary, solutions=solutions,
                warning='Same 150 locations / 300 regimes; A-D frozen reused. Regression, not new blind validation.')


def infer(args):
    output = validate_output_location(args.prior, args.output)
    for protected in (args.previous.resolve(), args.old.resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Output must be separate from previous runs')
    output.mkdir(parents=True, exist_ok=False)
    started, stages = time.perf_counter(), []
    before = frozen_stamp(args.prior)
    sources = [args.previous/'validation_checkpoint.pkl', args.old/'inference_checkpoint.pkl']
    hashes = {str(p.resolve()): file_digest(p) for p in sources}
    with StageSampler('load_frozen_inputs', output, stages):
        frozen = FrozenProfiles(args.prior)
        with sources[0].open('rb') as f:
            previous = pickle.load(f)
        with sources[1].open('rb') as f:
            old = pickle.load(f)
        index = AdaptiveMeshIndex(frozen)
    cases = previous['cases']
    if len(cases) != 193 or len(old['mask_cases']) != 150:
        raise ValueError('Frozen cohort size changed')
    required = {k for c in cases for k in frozen.neighbors(c['slice_key'])}
    target_keys = {c['slice_key'] for c in cases}
    branch_cache, profile_cache, inventory, loops = {}, {}, [], []
    component_counts = Counter()
    with StageSampler('canonical_branch_and_closed_extraction', output, stages):
        for i, key in enumerate(frozen.keys):
            profile = target_canonical_profile(frozen, key)
            uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
            branches = extract_observed_branches(profile, uz)
            component_counts.update({(key, b['component_id'], b['kind']): 1 for b in branches})
            inventory.extend(dict(slice_key=key, s=float(key), **light_branch(b)) for b in branches)
            loops.extend(analyze_closed_components(profile, uz, slice_key=key, s=float(key)))
            if key in required:
                if key in target_keys:
                    branch_cache[key] = branches
                    profile_cache[key] = profile
                else:
                    # Neighbor descriptors never consume per-edge provenance.
                    branch_cache[key] = [{k: v for k, v in b.items() if k != 'records'} for b in branches]
            if (i+1) % 400 == 0:
                print(f'Branch inventory {i+1}/{len(frozen.keys)}; closed {len(loops)}', flush=True)
        save_csv(output/'branch_inventory.csv', inventory)
    rows, reliability, junctions, solutions, matching = [], [], [], {}, []
    old_rows = {r['candidate_id']: r for r in previous['rows']}
    with StageSampler('dominant_branch_selection_and_junctions_193', output, stages):
        for i, case in enumerate(cases):
            cid, key = case['candidate_id'], case['slice_key']
            profile = profile_cache[key]
            uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
            stamp = hashlib.sha256(profile.nodes.tobytes()+profile.edges.tobytes()).hexdigest()
            neighbors = [dict(s=float(k), branches=branch_cache[k]) for k in frozen.neighbors(key) if k != key]
            window = previous['locals'][cid]['window']
            result = solve_dominant_branch(profile, uz, window, neighbors, branches=branch_cache[key])
            if result['status'] == 'NEEDS_GAP_INFERENCE':
                tracked = previous['windows'][cid]
                result = use_gap_inference(result, profile, uz, tracked, subwindow_builder(index, frozen, tracked), cid)
                checked = 0
            else:
                checked = assert_geometry_preserved(profile, uz, result)
            if stamp != hashlib.sha256(profile.nodes.tobytes()+profile.edges.tobytes()).hexdigest():
                raise AssertionError('Input geometry mutated')
            selected = result['selection']['selected']
            reliability.extend(dict(candidate_id=cid, slice_key=key, **light_branch(r)) for r in result['selection']['candidates'])
            junctions.extend(junction_diagnostics(case, branch_cache[key], result))
            if result['junction'] is not None:
                junctions.append(dict(candidate_id=cid, purpose='one_switch_route_proposal', applied=False, **result['junction']))
            row = dict(candidate_id=cid, slice_key=key, old_category=old_rows[cid]['old_category'], status=result['status'],
                dominant_branch_id=None if selected is None else selected['branch_id'],
                observed_coverage=0. if selected is None else selected['observed_coverage'],
                interior_score=None if selected is None else selected['interior_score'],
                neighbor_detail_repeat_count=0 if selected is None else selected['neighbor_detail_repeat_count'],
                neighbor_detail_score=None if selected is None else selected['neighbor_detail_score'],
                detail_density=None if selected is None else selected['detail_density'],
                horizontal_support=None if selected is None else selected['horizontal_support'],
                branch_selection_ambiguous=result['selection']['ambiguous'],
                original_endpoint_pair_connected=result['original_endpoint_pair_connected'],
                branch_switch_count=result['branch_switch_count'], virtual_junction_count=result['virtual_junction_count'],
                inferred_length_m=result['inferred_length_m'], connector_length_m=result['connector_length_m'],
                observed_length_m=result['observed_length_m'], observed_geometry_fraction=result['observed_geometry_fraction'],
                observed_edges_checked=checked, input_geometry_unchanged=True, production_replacement_applied=False,
                old_status=old_rows[cid]['status'], old_inferred_length_m=old_rows[cid]['inferred_length_m'])
            rows.append(row)
            endpoint_key = lambda p: (key, *tuple(np.round(p, 6)))
            score = .51+.35*row['observed_coverage']+.10*(row['horizontal_support'] or 0.)
            if selected is None:
                score = .25
            matching.append(dict(candidate_id=cid, lower_key=endpoint_key(case['lower_uz']),
                upper_key=endpoint_key(case['upper_uz']), pair_score=score))
            solutions[cid] = result
            if (i+1) % 25 == 0:
                print(f'Dominant decisions {i+1}/193', flush=True)
        pair_map = {p['candidate_id']: p for p in joint_endpoint_pairing(matching)}
        for row in rows:
            pair = pair_map[row['candidate_id']]
            row.update(joint_selected=pair['selected'], pairing_ambiguous=pair['pairing_ambiguous'], pairing_status=pair['pairing_status'])
        save_csv(output/'branch_reliability.csv', reliability)
        save_csv(output/'dominant_branch_decisions.csv', rows)
        save_csv(output/'junction_candidates.csv', junctions,
                 fields=None if junctions else ['candidate_id', 'junction_type', 'applied'])
    with StageSampler('closed_horizontal_support_and_tracking', output, stages):
        horizontal_loop_matches(frozen, loops)
        closed = track_closed_components(loops)
        save_csv(output/'closed_components.csv', loops)
        save_csv(output/'closed_component_tracks.csv', closed['tracks'])
        save_csv(output/'closed_component_pairs.csv', closed['pairs'])
    with StageSampler('masking_method_e_300', output, stages):
        masking = run_masking(frozen, index, old, previous, output)
    if frozen_stamp(args.prior) != before or any(file_digest(Path(p)) != h for p, h in hashes.items()):
        raise AssertionError('Frozen baseline changed')
    numeric_seconds = time.perf_counter()-started
    categories = []
    for category in 'ABCDEF':
        group = [r for r in rows if r['old_category'] == category]
        categories.append(dict(category=category, count=len(group), complete_observed=sum(r['status'] == 'PRESERVED_COMPLETE_OBSERVED' for r in group),
            partial_unresolved=sum(r['status'] == 'PRESERVED_PARTIAL_UNRESOLVED' for r in group),
            inferred=sum(r['inferred_length_m'] > 0 for r in group),
            ambiguous=sum(r['branch_selection_ambiguous'] or r['pairing_ambiguous'] for r in group),
            total_inferred_length_m=sum(r['inferred_length_m'] for r in group)))
    summary = dict(slice_count=len(frozen.keys), branch_count=len(inventory),
        branch_kinds=dict(Counter(r['kind'] for r in inventory)),
        component_kinds=dict(Counter(k[2] for k in component_counts)),
        decisions=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
        branch_ambiguous=sum(r['branch_selection_ambiguous'] for r in rows),
        joint_selected=sum(r['joint_selected'] for r in rows), joint_ambiguous=sum(r['pairing_ambiguous'] for r in rows),
        old_endpoint_pair_connected=sum(r['original_endpoint_pair_connected'] for r in rows),
        selected_routes_switch_count=sum(r['branch_switch_count'] for r in rows),
        route_virtual_junctions=sum(r['virtual_junction_count'] for r in rows),
        diagnostic_junctions=len(junctions), diagnostic_junction_kinds=dict(Counter(r['junction_type'] for r in junctions)),
        old_inferred_length_m=sum(r['old_inferred_length_m'] for r in rows),
        inferred_length_m=sum(r['inferred_length_m'] for r in rows),
        observed_length_m=sum(r['observed_length_m'] for r in rows),
        mean_observed_geometry_fraction=float(np.mean([r['observed_geometry_fraction'] for r in rows])),
        observed_edges_checked=sum(r['observed_edges_checked'] for r in rows),
        repeated_detail_cases=sum(r['neighbor_detail_repeat_count'] > 0 for r in rows),
        closed=closed['summary'], category_summary=categories, numeric_wall_seconds=numeric_seconds,
        peak_rss_gib=max(s['peak_tree_rss_bytes'] for s in stages)/2**30,
        peak_private_gib=max(s['peak_tree_private_bytes'] for s in stages)/2**30,
        frozen_unchanged=True, production_replacement_applied=False,
        masks_equal_d=sum(r.get('curve_equal_method_d', False) for r in masking['rows'] if r['method'] == METHOD_E))
    code_paths = [Path(__file__), ROOT/'scripts/04_structure_recognition/dominant_observed_branch.py',
                  ROOT/'scripts/04_structure_recognition/backtracking_junction.py',
                  ROOT/'scripts/04_structure_recognition/closed_component_tracking.py',
                  ROOT/'scripts/04_structure_recognition/run_multi_profile_recognition.py']
    manifest = dict(created=datetime.now().isoformat(), branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        commit_created=False, source_hashes=hashes, code_hashes={str(p): file_digest(p) for p in code_paths},
        logical_cpus=psutil.cpu_count(), ram_gib=psutil.virtual_memory().total/2**30,
        process_count=1, numeric_threads=1, gpu_used=False, frozen_unchanged=True,
        inventory_scope='2800 positive scanline rays; opposite-ray intersections excluded',
        timing_scope='input load + full branch/closed inventory + 193 decisions + closed tracks + 300 method E masks; no loft/volume',
        reference_plan='C:/Users/222/Downloads/GDS_Dominant_Branch_Minimum_Intervention_Codex_Plan.md')
    checkpoint = dict(cases=cases, rows=rows, results=solutions, reliability=reliability, junctions=junctions,
        loops=loops, closed=closed, masking=masking, summary=summary, manifest=manifest, stages=stages,
        previous_representatives=previous['classification']['representatives'],
        old_natural_solutions=previous['old_natural_solutions'], windows=previous['windows'],
        neighbors={c['candidate_id']: previous['tracks'][c['candidate_id']] for c in cases},
        previous_rows=previous['rows'])
    with (output/'validation_checkpoint.pkl').open('wb') as stream:
        pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    save_json(output/'summary.json', summary)
    save_json(output/'manifest.json', manifest)
    save_csv(output/'category_summary.csv', categories)
    save_csv(output/'stage_performance.csv', stages)
    print(summary, flush=True)


def refine_boundary_association(args):
    """Recompute ONLY affected 193-case selection after diagnosed ROI bias.

    Inventory/closed tracks/300 masking outcomes are immutable reused results.
    The measured refinement includes its own data loading and canonical cache.
    """
    output = args.output.resolve()
    with (output/'validation_checkpoint.pkl').open('rb') as stream:
        data = pickle.load(stream)
    if 'boundary_association_refinement' in data:
        raise ValueError('This checkpoint already has the targeted refinement')
    stages = data['stages'].copy()
    before = frozen_stamp(args.prior)
    initial_summary = copy.deepcopy(data['summary'])
    original_results = data['results']
    with StageSampler('boundary_association_refinement_193', output, stages):
        with (args.previous/'validation_checkpoint.pkl').open('rb') as stream:
            previous = pickle.load(stream)
        prior_manifest = json.loads((args.prior/'manifest.json').read_text(encoding='utf-8'))
        with (Path(prior_manifest['baseline'])/'slices.pkl').open('rb') as stream:
            slices = pickle.load(stream)
        keys = sorted(slices, key=float)
        key_index = {k: i for i, k in enumerate(keys)}
        neighbors_of = lambda key: keys[max(0, key_index[key]-2):min(len(keys), key_index[key]+3)]
        frozen = SimpleNamespace(slices=slices, arc=previous['arc'])
        required = {k for c in data['cases'] for k in neighbors_of(c['slice_key'])}
        cache = {}
        for key in required:
            profile = previous['target_profiles'].get(key)
            if profile is None:
                profile = target_canonical_profile(frozen, key)
            cache[key] = extract_observed_branches(profile, opc.to_suz(profile.nodes, frozen.arc)[:, 1:])
        rows = copy.deepcopy(data['rows'])
        reliability, junctions, solutions, matching, changes = [], [], {}, [], []
        for row, case in zip(rows, data['cases']):
            cid, key = case['candidate_id'], case['slice_key']
            if cid != row['candidate_id']:
                raise AssertionError('Case order changed')
            profile = previous['target_profiles'][key]
            uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
            window = previous['locals'][cid]['window']
            neighbors = [dict(s=float(k), branches=cache[k]) for k in neighbors_of(key) if k != key]
            result = solve_dominant_branch(profile, uz, window, neighbors, branches=cache[key])
            if result['status'] == 'NEEDS_GAP_INFERENCE':
                # Retain an already measured same-input gap fallback, never
                # re-run the unrelated 300 masking or infer over new ambiguity.
                if original_results[cid].get('fallback') is not None:
                    result = original_results[cid]
                else:
                    result['status'] = 'UNRESOLVED_NO_ASSOCIATED_BRANCH'
                checked = 0
            else:
                checked = assert_geometry_preserved(profile, uz, result)
            selected = result['selection']['selected']
            reliability.extend(dict(candidate_id=cid, slice_key=key, **light_branch(r)) for r in result['selection']['candidates'])
            junctions.extend(junction_diagnostics(case, cache[key], result))
            if result['junction'] is not None:
                junctions.append(dict(candidate_id=cid, purpose='one_switch_route_proposal', applied=False, **result['junction']))
            if not np.array_equal(result['curve_uz'], original_results[cid]['curve_uz']):
                changes.append(dict(candidate_id=cid, before_status=row['status'], after_status=result['status']))
            row.update(status=result['status'], dominant_branch_id=None if selected is None else selected['branch_id'],
                observed_coverage=0. if selected is None else selected['observed_coverage'],
                interior_score=None if selected is None else selected['interior_score'],
                neighbor_detail_repeat_count=0 if selected is None else selected['neighbor_detail_repeat_count'],
                neighbor_detail_score=None if selected is None else selected['neighbor_detail_score'],
                detail_density=None if selected is None else selected['detail_density'],
                horizontal_support=None if selected is None else selected['horizontal_support'],
                branch_selection_ambiguous=result['selection']['ambiguous'],
                original_endpoint_pair_connected=result['original_endpoint_pair_connected'],
                branch_switch_count=result['branch_switch_count'], virtual_junction_count=result['virtual_junction_count'],
                inferred_length_m=result['inferred_length_m'], connector_length_m=result['connector_length_m'],
                observed_length_m=result['observed_length_m'], observed_geometry_fraction=result['observed_geometry_fraction'],
                observed_edges_checked=checked)
            ek = lambda p: (key, *tuple(np.round(p, 6)))
            score = .51+.35*row['observed_coverage']+.10*(row['horizontal_support'] or 0.) if selected is not None else .25
            matching.append(dict(candidate_id=cid, lower_key=ek(case['lower_uz']), upper_key=ek(case['upper_uz']), pair_score=score))
            solutions[cid] = result
        pairs = {p['candidate_id']: p for p in joint_endpoint_pairing(matching)}
        for row in rows:
            pair = pairs[row['candidate_id']]
            row.update(joint_selected=pair['selected'], pairing_ambiguous=pair['pairing_ambiguous'], pairing_status=pair['pairing_status'])
    if frozen_stamp(args.prior) != before:
        raise AssertionError('Frozen source changed during refinement')
    summary = data['summary']
    summary.update(statuses=dict(Counter(r['status'] for r in rows)),
        branch_ambiguous=sum(r['branch_selection_ambiguous'] for r in rows),
        joint_selected=sum(r['joint_selected'] for r in rows), joint_ambiguous=sum(r['pairing_ambiguous'] for r in rows),
        old_endpoint_pair_connected=sum(r['original_endpoint_pair_connected'] for r in rows),
        selected_routes_switch_count=sum(r['branch_switch_count'] for r in rows),
        route_virtual_junctions=sum(r['virtual_junction_count'] for r in rows), diagnostic_junctions=len(junctions),
        diagnostic_junction_kinds=dict(Counter(r['junction_type'] for r in junctions)),
        inferred_length_m=sum(r['inferred_length_m'] for r in rows), observed_length_m=sum(r['observed_length_m'] for r in rows),
        mean_observed_geometry_fraction=float(np.mean([r['observed_geometry_fraction'] for r in rows])),
        observed_edges_checked=sum(r['observed_edges_checked'] for r in rows),
        repeated_detail_cases=sum(r['neighbor_detail_repeat_count'] > 0 for r in rows),
        numeric_wall_seconds=initial_summary['numeric_wall_seconds']+stages[-1]['wall_seconds'],
        initial_numeric_wall_seconds=initial_summary['numeric_wall_seconds'],
        targeted_refinement_seconds=stages[-1]['wall_seconds'],
        peak_rss_gib=max(s['peak_tree_rss_bytes'] for s in stages)/2**30,
        peak_private_gib=max(s['peak_tree_private_bytes'] for s in stages)/2**30)
    summary['category_summary'] = []
    for category in 'ABCDEF':
        group = [r for r in rows if r['old_category'] == category]
        summary['category_summary'].append(dict(category=category, count=len(group),
            complete_observed=sum(r['status'] == 'PRESERVED_COMPLETE_OBSERVED' for r in group),
            partial_unresolved=sum(r['status'] == 'PRESERVED_PARTIAL_UNRESOLVED' for r in group),
            inferred=sum(r['inferred_length_m'] > 0 for r in group),
            ambiguous=sum(r['branch_selection_ambiguous'] or r['pairing_ambiguous'] for r in group),
            total_inferred_length_m=sum(r['inferred_length_m'] for r in group)))
    refinement = dict(reason='A real boundary-associated branch must not be rejected for its observed interior excursion',
        old_association='median local U inside endpoint range + height-scaled 0.05..0.25m pad',
        new_association='both local boundary U distances <= 0.25m; no interior deviation limit',
        changed_routes=changes, initial_summary=initial_summary, unchanged_masking=True, unchanged_closed_inventory=True,
        scope='193 natural decisions and dependent joint matching/junction diagnostics only')
    data.update(rows=rows, results=solutions, reliability=reliability, junctions=junctions, stages=stages,
                boundary_association_refinement=refinement)
    data['manifest']['initial_code_hashes'] = data['manifest']['code_hashes']
    data['manifest']['code_hashes'] = {p: file_digest(Path(p)) for p in data['manifest']['initial_code_hashes']}
    data['manifest']['targeted_refinement'] = refinement['scope']
    with (output/'validation_checkpoint.pkl').open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    save_json(output/'refinement_audit.json', refinement)
    save_json(output/'summary.json', summary)
    save_json(output/'manifest.json', data['manifest'])
    save_csv(output/'branch_reliability.csv', reliability)
    save_csv(output/'dominant_branch_decisions.csv', rows)
    save_csv(output/'junction_candidates.csv', junctions)
    save_csv(output/'category_summary.csv', summary['category_summary'])
    save_csv(output/'stage_performance.csv', stages)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ROOT/'outputs/orthogonal_scanline_constraint/20260919_081148')
    parser.add_argument('--previous', type=Path, default=ROOT/'outputs/vertical_canonical_locc/20260919_observed_first/run')
    parser.add_argument('--old', type=Path, default=ROOT/'outputs/locc_validation/20260919_100000/refined')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('infer', 'refine-boundary', 'finish'), default='infer')
    args = parser.parse_args()
    if args.phase == 'infer':
        infer(args)
    elif args.phase == 'refine-boundary':
        refine_boundary_association(args)
    else:
        from dominant_branch_report import finish
        finish(args.output)


if __name__ == '__main__':
    main()
