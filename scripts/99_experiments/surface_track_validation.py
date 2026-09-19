"""One scoped branch-first audit; frozen inputs and recognition remain read-only."""
from __future__ import annotations
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import unittest
import numpy as np

from locc_data import FrozenProfiles, opc, xyz_from_uz
from validate_observed_first_locc import target_canonical_profile
from validate_orthogonal_scanline_constraint import StageSampler, save_csv, save_json, json_default
from dominant_observed_branch import extract_observed_branches, solve_dominant_branch, select_dominant_branch, trim_record
from horizontal_surface_link import horizontal_mesh_observations
from observed_surface_graph import build_surface_graph
from vertical_profile_canonicalization import canonicalize_vertical

ROOT = Path(__file__).resolve().parents[2]


def verify_coordinates(profile, uz, result):
    for r in result['path_edges']:
        if not r['source'].startswith('OBSERVED'):
            if r.get('face_id') is not None or r.get('source_face_ids'):
                raise AssertionError('Synthetic edge acquired observed FaceID')
            continue
        eid = r['edge_id']
        if set(r['node_ids']) != set(profile.edges[eid]):
            raise AssertionError('Observed edge identity changed')
        for name, source in (('points_uz', uz), ('points_xyz', profile.nodes)):
            a, b = source[r['node_ids']]
            expected = [a+t*(b-a) for t in (r['t0'], r['t1'])]
            np.testing.assert_allclose(r[name], expected, atol=2e-10, rtol=0)
        if r['source_face_ids'] != profile.source_face_ids[eid]:
            raise AssertionError('Observed provenance changed')
    sequence = []
    for r in result['path_edges']:
        if r['source'].startswith('OBSERVED') and (not sequence or sequence[-1] != r['branch_id']):
            sequence.append(r['branch_id'])
    if len(sequence) > 2 or result['branch_switch_count'] > 1:
        raise AssertionError('Multiple branch handoffs')
    return len(sequence)


def run_unit_checks(output):
    sys.path.insert(0, str(ROOT/'tests'))
    names = ['test_surface_track', 'test_dominant_observed_branch',
             'test_dominant_review_fixes', 'test_dominant_boundary_association']
    suite = unittest.TestLoader().loadTestsFromNames(names)
    with (output/'focused_tests.txt').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))
    save_json(output/'focused_tests.json', summary)
    print('Focused checks:', summary, flush=True)
    if not result.wasSuccessful():
        raise RuntimeError('Focused check failed; see focused_tests.txt. Numerical audit not started.')
    return summary


def masked_tail(profile, uz, branch, fraction=.70):
    """Remove only the upper tail, preserving all other target observations."""
    from surface_track_handoff import _oriented
    branch = _oriented(branch)
    if np.any(np.diff(branch['points_uz'][:, 1]) < -1e-9):
        return None
    cutoff = fraction*branch['full_arc_length']
    retained = []
    for i, record in enumerate(branch['records']):
        start, stop = branch['arc_positions'][i:i+2]
        if start >= cutoff:
            break
        retained.append(trim_record(record, 0., min(1., (cutoff-start)/max(stop-start, 1e-12))))
    if not retained:
        return None
    removed = set(branch['edge_order'])
    lines = [profile.nodes[edge] for i, edge in enumerate(profile.edges) if i not in removed]
    faces = [profile.first_face_ids[i] for i in range(len(profile.edges)) if i not in removed]
    lines.extend(np.asarray(r['points_xyz']) for r in retained)
    faces.extend(r['face_id'] for r in retained)
    masked = canonicalize_vertical(np.asarray(lines), faces)
    return masked, retained, branch


def partial_tail_audit(data, output):
    graph, frozen = data['graph'], data['frozen']
    rows = []
    # Exact previous review cohort, one case per slice; no best-result picking.
    used = set()
    for case in data['representatives']:
        cid = case['candidate_id']
        result = data['results'][cid]
        selected = result['selection']['selected']
        key = data['case_map'][cid]['slice_key']
        if key in used or selected is None:
            continue
        used.add(key)
        s = float(key)
        profile = data['profiles'][key]
        branch = selected['fragment']
        if not selected['track_stable']:
            rows.append(dict(candidate_id=cid, status='NOT_EVALUATED_AMBIGUOUS_OR_UNSTABLE_TRACK'))
            continue
        mask = masked_tail(profile, data['uz'][key], branch)
        if mask is None:
            rows.append(dict(candidate_id=cid, status='NOT_EVALUATED_MULTIVALUED_TAIL'))
            continue
        masked, retained, original = mask
        muz = opc.to_suz(masked.nodes, frozen.arc)[:, 1:]
        mb = extract_observed_branches(masked, muz)
        keys = frozen.neighbors(key)
        inv = [dict(s=float(k), branches=mb if k == key else data['branches'][k]) for k in keys]
        positions = {float(k) for k in keys}
        obs = [h for h in graph['observations'] if h['s'] in positions]
        mg = build_surface_graph(inv, obs, slice_order=sorted(positions))
        # Retained observations identify their track before recovering the
        # hidden tail. Neither hidden target coordinates nor the old endpoints
        # are passed to the solver.
        anchor_faces = {f for r in retained for f in r['source_face_ids']}
        candidates = [b for b in mb if anchor_faces.intersection(b['source_face_ids'])]
        anchored = max(candidates, key=lambda b: len(anchor_faces.intersection(b['source_face_ids']))) if candidates else None
        tid = mg['membership'].get((s, anchored['branch_id'])) if anchored else None
        recovered = solve_dominant_branch(masked, muz, branches=mb, surface_graph=mg, target_s=s,
            surface_track_id=tid, xyz_builder=lambda p: xyz_from_uz(p, s, frozen.arc))
        verify_coordinates(masked, muz, recovered)
        chosen = recovered['selection']['selected']
        inferred = [r for r in recovered['path_edges'] if r['source'].endswith('INFERRED')]
        truth = original['points_uz']
        points = np.array([p for r in inferred for p in r['points_uz']]).reshape(-1, 2)
        error = np.abs(points[:, 0]-np.interp(points[:, 1], truth[:, 1], truth[:, 0])) if len(points) else []
        retained_identity = bool(chosen and anchored and chosen['branch_id'] == anchored['branch_id'])
        rows.append(dict(candidate_id=cid, status=recovered['status'], retained_identity=retained_identity,
            inferred_length_m=recovered['inferred_length_m'], missing_tail_recovered=bool(inferred),
            p95_abs_u_m=float(np.quantile(error, .95)) if len(error) else None,
            observed_coordinates_preserved=True, hidden_tail_used_for_selection=False,
            branch_switch_count=recovered['branch_switch_count']))
    save_csv(output/'partial_tail_masking.csv', rows)
    return rows


def run(args):
    output = args.output.resolve()
    protected = [args.prior.resolve(), args.previous.resolve(), args.canonical.resolve()]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be separate from frozen input directories')
    output.mkdir(parents=True, exist_ok=False)
    started, stages = time.perf_counter(), []
    checks = run_unit_checks(output) if args.check else None
    with StageSampler('load_frozen_inputs', output, stages):
        frozen = FrozenProfiles(args.prior)
        with (args.previous/'validation_checkpoint.pkl').open('rb') as stream:
            previous = pickle.load(stream)
        with (args.canonical/'representative_figures.csv').open(encoding='utf-8-sig', newline='') as stream:
            representatives = list(csv.DictReader(stream))
    cases = previous['cases']
    target_keys = {c['slice_key'] for c in cases}
    profiles, branches, uz_by_key, inventory = {}, {}, {}, []
    with StageSampler('full_branch_inventory', output, stages):
        for i, key in enumerate(frozen.keys):
            profile = target_canonical_profile(frozen, key)
            uz = opc.to_suz(profile.nodes, frozen.arc)[:, 1:]
            bs = extract_observed_branches(profile, uz)
            inventory.extend(dict(slice_key=key, s=float(key), **{k: b[k] for k in (
                'branch_id', 'component_id', 'kind', 'node_order', 'edge_order', 'full_arc_length',
                'z_range', 'u_range', 'canonical_edge_count', 'source_face_ids', 'source_segment_indices')}) for b in bs)
            if key in target_keys:
                profiles[key], uz_by_key[key] = profile, uz
            else:
                bs = [{k: v for k, v in b.items() if k != 'records'} for b in bs]
            branches[key] = bs
            if (i+1) % 700 == 0:
                print(f'Full branch inventory {i+1}/{len(frozen.keys)}', flush=True)
        save_csv(output/'branch_inventory.csv', inventory)
    with StageSampler('horizontal_path_build', output, stages):
        observations, horizontal_inventory = horizontal_mesh_observations(frozen.horizontal, frozen.levels, frozen.s, frozen.arc)
        save_csv(output/'horizontal_path_inventory.csv', horizontal_inventory)
    with StageSampler('surface_graph_build_and_track_extraction', output, stages):
        graph = build_surface_graph([dict(s=float(k), branches=b) for k, b in branches.items()],
                                    observations, slice_order=frozen.s)
        save_csv(output/'surface_track_inventory.csv', graph['tracks'])
        save_csv(output/'surface_track_links.csv', [{k: v for k, v in r.items() if k != 'evidence'} for r in graph['links']])
        save_csv(output/'surface_track_weak_links.csv', [{k: v for k, v in r.items() if k != 'evidence'} for r in graph['weak_links']])
        save_csv(output/'surface_track_ambiguous_links.csv', [{k: v for k, v in r.items() if k != 'evidence'} for r in graph['ambiguous_links']])
        save_csv(output/'closed_transition_evidence.csv', graph['closed_transition_evidence'])
    data = dict(graph=graph, frozen=frozen, profiles=profiles, branches=branches, uz=uz_by_key,
        cases=cases, case_map={c['candidate_id']: c for c in cases}, representatives=representatives,
        previous=previous, results={}, rows=[], stages=stages)
    old_rows = {r['candidate_id']: r for r in previous['rows']}
    perturbations, support_rows, decision_cache = [], [], {}
    with StageSampler('target_case_decision_including_junction_search', output, stages):
        for case in cases:
            cid, key = case['candidate_id'], case['slice_key']
            s = float(key)
            if key not in decision_cache:
                result = solve_dominant_branch(profiles[key], uz_by_key[key], branches=branches[key],
                    surface_graph=graph, target_s=s, xyz_builder=lambda p: xyz_from_uz(p, s, frozen.arc))
                verify_coordinates(profiles[key], uz_by_key[key], result)
                decision_cache[key] = result
            result = decision_cache[key]
            selected = result['selection']['selected']
            row = dict(candidate_id=cid, slice_key=key, old_category=old_rows[cid]['old_category'],
                surface_track_id=result['surface_track_id'], target_slice_branch_id=selected['branch_id'] if selected else None,
                status=result['status'], route_identity=result['route_identity'],
                ambiguous=result['selection']['ambiguous'], stable=bool(selected and selected['track_stable']),
                branch_switch_count=result['branch_switch_count'], virtual_junction_count=result['virtual_junction_count'],
                connector_length_m=result['connector_length_m'], inferred_length_m=result['inferred_length_m'],
                observed_length_m=result['observed_length_m'],
                low_confidence_tail_removed=result['observed_low_confidence_tail_removed'],
                confidence_crossover_junction_count=result['confidence_crossover_junction_count'],
                legacy_endpoint_influence_on_surface_identity=0,
                previous_branch_id=old_rows[cid]['dominant_branch_id'],
                previous_inferred_length_m=old_rows[cid]['inferred_length_m'])
            data['rows'].append(row)
            data['results'][cid] = result
        for key, result in decision_cache.items():
            for r in result['selection']['candidates']:
                support_rows.append(dict(slice_key=key, **{k: v for k, v in r.items() if k != 'fragment'}))
        save_csv(output/'surface_track_decisions.csv', data['rows'])
        save_csv(output/'surface_track_support.csv', support_rows)
    with StageSampler('endpoint_perturbation_and_partial_tail_masking', output, stages):
        for case in cases:
            cid, key = case['candidate_id'], case['slice_key']
            reference = data['results'][cid]['route_identity']
            for delta in (-.05, -.02, -.005, .005, .02, .05):
                for direction in ('u', 'z', 'opposite_z'):
                    w = dict(s=float(key), lower_uz=np.array(case['lower_uz']).copy(), upper_uz=np.array(case['upper_uz']).copy())
                    axis = 0 if direction == 'u' else 1
                    w['lower_uz'][axis] += delta
                    w['upper_uz'][axis] += -delta if direction == 'opposite_z' else delta
                    selection = select_dominant_branch(branches[key], w, surface_graph=graph, target_s=float(key))
                    chosen = selection['selected']
                    identity = (chosen['surface_track_id'], chosen['branch_id']) if chosen else None
                    perturbations.append(dict(candidate_id=cid, shift_m=delta, direction=direction,
                        surface_track_id=identity[0] if identity else None, branch_id=identity[1] if identity else None,
                        identity_changed=identity != reference))
        save_csv(output/'endpoint_perturbation_stability.csv', perturbations)
        data['masking'] = partial_tail_audit(data, output)
    unique = list(decision_cache.values())
    coords = all(r['observed_coordinates_overwritten'] == 0 for r in unique)
    summary = dict(scope='Full 2800-slice inventory and H graph; 193 historical cases / 168 unique target slices; previous 29 review cases',
        branch=git_value('branch', '--show-current'), head=git_value('rev-parse', 'HEAD'), commit_sha=None,
        full_branch_count=len(inventory), surface_track_count=len(graph['tracks']),
        stable_surface_tracks=sum(t['stable'] for t in graph['tracks']),
        ambiguous_surface_tracks=sum(t['ambiguous'] for t in graph['tracks']),
        raw_H_hits=graph['raw_H_hits'], surface_linked_H_hits=graph['surface_linked_H_hits'],
        raw_H_to_linked_ratio=graph['raw_H_hits']/max(1, graph['surface_linked_H_hits']),
        endpoint_perturbation_count=len(perturbations),
        endpoint_perturbation_identity_change_count=sum(r['identity_changed'] for r in perturbations),
        endpoint_perturbation_stability=1-sum(r['identity_changed'] for r in perturbations)/len(perturbations),
        unique_target_slice_count=len(unique), historical_case_count=len(cases),
        complete_observed_track_preserved_count=sum(r['status'] == 'PRESERVED_COMPLETE_OBSERVED' for r in unique),
        complete_observed_track_wrongly_replaced_count=None,
        stable_track_wrong_replacement_note='No independently labelled real-case identity truth; not equated to zero.',
        branch_switch_count=sum(r['branch_switch_count'] for r in unique), A_B_A_count=0,
        confidence_crossover_junction_count=sum(r['confidence_crossover_junction_count'] for r in unique),
        endpoint_junction_count=sum(r['junction'] is not None and r['junction']['junction_type'] == 'ENDPOINT_CONNECTOR' for r in unique),
        virtual_junction_count=sum(r['virtual_junction_count'] for r in unique),
        synthetic_connector_length=sum(r['connector_length_m'] for r in unique),
        inferred_geometry_length=sum(r['inferred_length_m'] for r in unique),
        observed_high_confidence_length_preserved=sum(r['observed_high_confidence_length_preserved'] for r in unique),
        observed_low_confidence_tail_removed=sum(r['observed_low_confidence_tail_removed'] for r in unique),
        observed_coordinate_preservation=coords, closed_tracks=sum(t['kind'] == 'CLOSED_TRACK' for t in graph['tracks']),
        closed_transition_evidence_count=len(graph['closed_transition_evidence']),
        partial_tail_evaluated=sum('retained_identity' in r for r in data['masking']),
        partial_tail_identity_retained=sum(r.get('retained_identity', False) for r in data['masking']),
        partial_tail_recovered=sum(r.get('missing_tail_recovered', False) for r in data['masking']),
        partial_tail_statuses=dict(Counter(r['status'] for r in data['masking'])),
        focused_checks=checks, production_recognition_changed=False)
    data['summary'] = summary
    with StageSampler('plotting_and_reports', output, stages):
        from surface_track_report import render_reports
        render_reports(data, output)
    summary['local_review'] = dict(
        cases=len(data['local_rows']),
        selected_absent_from_local_view=sum(r['selected_absent_from_local_view'] for r in data['local_rows']),
        local_z_coverage_below_95_percent=sum(r['local_selected_z_coverage'] < .95 for r in data['local_rows']),
        focused_cases=data['local_focus'],
        metrics_are_diagnostic_not_ground_truth=True)
    summary.update(total_wall_seconds=time.perf_counter()-started,
        peak_rss_gib=max(s['peak_tree_rss_bytes'] for s in stages)/2**30,
        peak_private_gib=max(s['peak_tree_private_bytes'] for s in stages)/2**30)
    save_csv(output/'stage_performance.csv', stages)
    save_json(output/'summary.json', summary)
    save_json(output/'manifest.json', dict(source_plan='GDS_Branch_First_Surface_Track_Codex_Plan.md',
        prior=str(args.prior.resolve()), previous=str(args.previous.resolve()),
        branch=summary['branch'], head=summary['head'], no_commit=True,
        code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [*(ROOT/'scripts/04_structure_recognition').glob('*.py'),
                               Path(__file__), Path(__file__).with_name('surface_track_report.py'),
                               Path(__file__).with_name('surface_track_local_review.py')]}))
    with (output/'FINAL_REVIEW.md').open('a', encoding='utf-8') as stream:
        stream.write(f"\n实测总时间：{summary['total_wall_seconds']:.2f}s；峰值 RSS：{summary['peak_rss_gib']:.3f} GiB。详见 stage_performance.csv。\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), flush=True)


def git_value(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ROOT/'outputs/orthogonal_scanline_constraint/20260919_081148')
    parser.add_argument('--previous', type=Path, default=ROOT/'outputs/dominant_branch_validation/20260919_143131/run')
    parser.add_argument('--canonical', type=Path, default=ROOT/'outputs/vertical_canonical_locc/20260919_observed_first/run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Run the directly related regression files once before this audit')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
