"""Read-only LOCC experiment driver; no official recognition artifact is written.

Use --phase infer followed by --phase finish to inspect calibrated results before
rendering. A checkpoint makes the finish stage independent of repeat inference.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from collections import Counter
import csv
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

from validate_orthogonal_scanline_constraint import StageSampler, save_json, save_csv, json_default
from locc_data import FrozenProfiles, AdaptiveMeshIndex, plot_payload, opc
from validate_locc_masking import run_masking, MaskingRiskCalibration, gap_bucket, BUCKETS, REGIMES
from local_orthogonal_cooperative_constraint import solve_locc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/06_visualization'))

CATEGORIES = ('CONSISTENT_VH', 'MULTIBRANCH_CLEAR', 'NEIGHBOR_STRONG_H_MISSING',
              'H_STRONG_NEIGHBOR_INCOMPLETE', 'EVIDENCE_CONFLICT', 'LOW_MARGIN_OR_WEAK')


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen_stamp(prior):
    manifest = json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
    baseline = Path(manifest['baseline'])
    paths = [baseline/name for name in ('slices.pkl', 'recognition.pkl', 'line_faces.pkl')]
    result = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            result[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    core = ROOT/'scripts/04_structure_recognition/run_multi_profile_recognition.py'
    result[str(core)] = {'sha256': file_digest(core)}
    return result


def validate_output_location(prior, output):
    """Resolve aliases before any write; never overlap frozen input trees."""
    prior, output = Path(prior).resolve(), Path(output).resolve()
    metadata = json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
    protected = (prior, Path(metadata['baseline']).resolve())
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError('Output must not equal, contain, or be inside a frozen prior/baseline directory')
    return output


def gap_signature(case):
    return (case['slice_key'], *np.round(np.r_[case['lower_uz'], case['upper_uz']], 6))


def find_natural_cases(frozen):
    with (frozen.prior/'gap_audit.jsonl').open(encoding='utf-8') as stream:
        old = [json.loads(line) for line in stream]
    old_signatures = {gap_signature(c) for c in old}
    cases = []
    for i, key in enumerate(frozen.keys):
        gaps = opc.small_gap_candidates(frozen.slices[key]['slicing']['lines_3d'], frozen.arc,
                                        float(key), max_gap_z=1.6, max_gap_u=.5)
        for gap in sorted(gaps, key=lambda g: (g['lower_uz'][1], g['lower_uz'][0], g['upper_uz'][1])):
            case = {'candidate_id': f'N{len(cases)+1:05d}', 'slice_key': key, 's': float(key), **gap,
                    'gap_height': gap['gap_z'], 'gap_bucket': gap_bucket(gap['gap_z'])}
            case['in_previous_436'] = gap_signature(case) in old_signatures
            cases.append(case)
        if (i+1) % 700 == 0:
            print(f'Natural enumeration {i+1}/{len(frozen.keys)}: {len(cases)} candidates', flush=True)
    found = {gap_signature(c) for c in cases}
    if old_signatures-found:
        raise AssertionError(f'Expanded natural enumeration lost {len(old_signatures-found)} old candidates')
    return cases


def categorize(row, calibration):
    if (calibration.conflict_residual is not None and row['neighbor_residual'] is not None
            and row['neighbor_residual'] > calibration.conflict_residual):
        return 'EVIDENCE_CONFLICT'
    if (row['normalized_cost_margin'] is not None and calibration.low_margin is not None
            and row['normalized_cost_margin'] <= calibration.low_margin):
        return 'LOW_MARGIN_OR_WEAK'
    if row['neighbor_vertical_coverage'] >= .75 and row['horizontal_real_coverage'] < .5:
        return 'NEIGHBOR_STRONG_H_MISSING'
    if row['horizontal_real_coverage'] >= .75 and row['neighbor_vertical_coverage'] < .75:
        return 'H_STRONG_NEIGHBOR_INCOMPLETE'
    if row['horizontal_real_coverage'] >= .75 and row['neighbor_vertical_coverage'] >= .75:
        return 'MULTIBRANCH_CLEAR' if row['ambiguous_level_fraction'] > 0 else 'CONSISTENT_VH'
    return 'LOW_MARGIN_OR_WEAK'


def run_natural(cases, frozen, index, calibration, output):
    rows, results = [], {}
    dp_seconds = 0.
    with (output/'natural_gap_candidates.jsonl').open('w', encoding='utf-8') as stream:
        for i, case in enumerate(cases):
            local = index.build_window(case)
            t0 = time.perf_counter()
            result = solve_locc(local['window'])
            elapsed = time.perf_counter()-t0
            dp_seconds += elapsed
            row = {k: case[k] for k in ('candidate_id', 'slice_key', 's', 'gap_height', 'gap_bucket', 'in_previous_436')}
            row.update(gap_z_low=float(case['lower_uz'][1]), gap_z_high=float(case['upper_uz'][1]),
                       neighbor_count=len(local['neighbor_keys'])-1,
                       horizontal_levels=[layer['z'] for layer in local['window']['layers']],
                       stable_branch_track=result['curve_uz'], inferred_curve=result['curve_uz'],
                       selected_source_per_level=result['selected_source_per_level'],
                       observed_node_face_ids=[n['face_id'] for n in result['selected_nodes']],
                       inferred_segment_face_ids=None,
                       neighbor_evidence=[n['neighbor'] for n in result['selected_nodes'][1:-1]],
                       best_cost=result['best_cost'], second_best_cost=result['second_best_cost'],
                       cost_margin=result['cost_margin'], normalized_cost_margin=result['normalized_cost_margin'],
                       solve_seconds=elapsed, **result['metrics'])
            row['confidence_raw_metrics'] = result['metrics']
            row.update(calibration.classify(row))
            row['review_category'] = categorize(row, calibration)
            row['analysis_only'] = True
            row['provenance'] = result['provenance']
            stream.write(json.dumps(row, ensure_ascii=False, default=json_default, allow_nan=False)+'\n')
            rows.append(row)
            results[case['candidate_id']] = result
            if (i+1) % 100 == 0:
                print(f'Natural inference {i+1}/{len(cases)}', flush=True)
    summary = {'candidate_count': len(rows), 'previous_candidates_covered': sum(c['in_previous_436'] for c in rows),
               'confidence_counts': dict(Counter(r['masking_calibrated_confidence'] for r in rows)),
               'category_counts': {k: sum(r['review_category'] == k for r in rows) for k in CATEGORIES},
               'suggested_review_repair': sum(r['suggested_action'] == 'REVIEW_REPAIR' for r in rows),
               'uncertain': sum(r['suggested_action'] == 'UNCERTAIN' for r in rows),
               'locc_solver_seconds': dp_seconds, 'official_repairs_written': 0,
               'candidate_scope': 'All opposing degree-one endpoint pairs in distinct components with 0<dz<=1.6 m and |du|<=0.5 m; 1 micrometre endpoint topology',
               'not_confirmed_gaps': 'Candidate geometry is not a diagnosis of missing rock; legitimate discontinuities and alternate branches may be included'}
    save_json(output/'natural_gap_summary.json', summary)
    return rows, results, summary


def representative_selection(rows, calibration):
    rng, chosen = np.random.default_rng(20260919), {}
    for category in CATEGORIES:
        pool = [r for r in rows if r['review_category'] == category]
        near = sorted(pool, key=lambda r: abs(r['predicted_local_p90_case_p95_error_m']-calibration.cutoffs[0]))[:10]
        ids = {r['candidate_id'] for r in near}
        rest = [r for r in pool if r['candidate_id'] not in ids]
        random_rows = [rest[i] for i in rng.choice(len(rest), min(5, len(rest)), replace=False)] if rest else []
        ids.update(r['candidate_id'] for r in random_rows)
        rest = [r for r in pool if r['candidate_id'] not in ids]
        worst = sorted(rest, key=lambda r: r['predicted_local_p90_case_p95_error_m'], reverse=True)[:5]
        chosen[category] = [(r['candidate_id'], tag) for tag, rr in (('near-pass', near), ('random', random_rows), ('worst', worst)) for r in rr]
    return chosen


def infer(prior, output, per_bucket, reference_output=None):
    validate_output_location(prior, output)
    if (output/'inference_checkpoint.pkl').exists() or (output/'masking_results.csv').exists():
        raise FileExistsError('Refusing to overwrite an existing inference run; use --phase finish or a new output directory')
    start = time.perf_counter()
    stages = []
    baseline_before = frozen_stamp(prior)
    manifest = {'created_at': datetime.now().astimezone().isoformat(), 'prior_audit': str(prior),
                'plan': 'C:/Users/222/Downloads/GDS_LOCC_Codex_Execution_Plan.txt',
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                'python': sys.executable, 'os': platform.platform(), 'physical_cores': psutil.cpu_count(logical=False),
                'logical_cpus': psutil.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
                'analysis_only': True, 'gpu_compute_used': False, 'workers': 1, 'numeric_threads': 1,
                'per_bucket': per_bucket, 'frozen_before': baseline_before,
                'initial_iteration_output': str(reference_output) if reference_output else None,
                'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in [Path(__file__),
                    ROOT/'scripts/04_structure_recognition/local_orthogonal_cooperative_constraint.py',
                    Path(__file__).with_name('locc_data.py'), Path(__file__).with_name('validate_locc_masking.py')]}}
    save_json(output/'manifest.json', manifest)
    with StageSampler('frozen_profile_loading_and_neighbor_index', output, stages):
        frozen = FrozenProfiles(prior)
    with StageSampler('local_mesh_index', output, stages):
        index = AdaptiveMeshIndex(frozen)
    with StageSampler('real_masking_validation_with_adaptive_cuts', output, stages):
        reference = None
        if reference_output:
            with (reference_output/'inference_checkpoint.pkl').open('rb') as stream:
                reference = pickle.load(stream)
        mask_cases, mask_rows, mask_solutions, mask_summary = run_masking(frozen, index, output, per_bucket, reference)
        calibration = MaskingRiskCalibration(mask_rows)
        calibration_report = calibration.report(mask_rows)
        save_json(output/'masking_calibration.json', calibration_report)
    with StageSampler('natural_gap_enumeration', output, stages):
        natural_cases = find_natural_cases(frozen)
    with StageSampler('natural_gap_inference_with_adaptive_cuts', output, stages):
        natural_rows, natural_solutions, natural_summary = run_natural(natural_cases, frozen, index, calibration, output)
    selection = representative_selection(natural_rows, calibration)
    natural_ids = {cid for selected in selection.values() for cid, _ in selected}
    # Representative masking figures: worst holdout LOCC case in each height
    # bucket/regime, plus the best actual ambiguous holdout case if available.
    mask_selection = []
    for regime in REGIMES:
        for bucket, _, _ in BUCKETS:
            rr = [r for r in mask_rows if r['method'] == 'LOCC' and r['split'] == 'holdout'
                  and r['regime'] == regime and r['gap_bucket'] == bucket]
            if rr:
                mask_selection.append((max(rr, key=lambda r: r['p95_abs_u_m'])['candidate_id'], regime, 'worst-holdout'))
    mask_ids = {cid for cid, _, _ in mask_selection}
    selected_ids = natural_ids | mask_ids
    selected_cache = {cid: local for cid, local in index.cache.items() if cid in selected_ids}
    manual = [{**r, 'representative_figure': '', 'manual_label': '', 'manual_notes': ''} for r in natural_rows]
    save_csv(output/'LOCC_manual_review.csv', manual)
    checkpoint = {'mask_cases': mask_cases, 'mask_rows': mask_rows, 'mask_solutions': mask_solutions,
                  'mask_summary': mask_summary, 'calibration_report': calibration_report,
                  'natural_cases': natural_cases, 'natural_rows': natural_rows, 'natural_solutions': natural_solutions,
                  'natural_summary': natural_summary, 'natural_selection': selection, 'mask_selection': mask_selection,
                  'local_cache': selected_cache, 'adaptive_seconds': index.adaptive_seconds,
                  'adaptive_cuts': index.adaptive_cuts, 'global_reused_cuts': index.global_reused_cuts,
                  'process_peak_working_set_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', None),
                  'inference_wall_seconds': time.perf_counter()-start}
    with StageSampler('checkpoint_output', output, stages):
        with (output/'inference_checkpoint.pkl').open('wb') as stream:
            pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
    unchanged = frozen_stamp(prior) == baseline_before
    save_json(output/'frozen_input_verification.json', {'unchanged': unchanged,
              'method': 'Size/mtime_ns of official pkl files and SHA256 of recognition core before/after; no content rewrite performed',
              'before': baseline_before, 'after': frozen_stamp(prior)})
    if not unchanged:
        raise AssertionError('Frozen input or recognition source changed during experiment')
    print(json.dumps({'inference_complete': True, 'masking_locations': len(mask_cases),
                      'natural_candidates': len(natural_cases), 'natural_summary': natural_summary,
                      'checkpoint': str(output/'inference_checkpoint.pkl')}, default=json_default), flush=True)


def render_reviews(checkpoint, frozen, index, output, track):
    from locc_local_review_plot import plot_local_review, plot_montage
    figures, modules = [], set()
    mask_by_id = {c['candidate_id']: c for c in checkpoint['mask_cases']}
    for cid, regime, tag in checkpoint['mask_selection']:
        case = mask_by_id[cid]
        local = checkpoint['local_cache'][cid]
        result = checkpoint['mask_solutions'][(cid, regime, 'LOCC')]
        payload = plot_payload(case, local, result, frozen, f'{regime}; {tag}; model truth, not geological truth',
                               truth=case['truth_curve'], masked=True)
        if regime == 'TARGET_HORIZONTAL_DROPOUT':
            # Display frozen sections as context, explicitly withheld from the
            # solver at the target; never label virtual nodes as observed H.
            for hp in payload['horizontal_profiles']:
                hp['selected_u'] = None
            payload['title_note'] += '; target H observations withheld from solver'
        path = output/'figures/masking'/f'{cid}_{regime}.png'
        rendered = plot_local_review(payload, path)
        modules.update(rendered['modules_reused'])
        figures.append({'candidate_id': cid, 'kind': 'masking', 'regime': regime, 'path': str(path), 'selection': tag})
    cases = {c['candidate_id']: c for c in checkpoint['natural_cases']}
    rows = {r['candidate_id']: r for r in checkpoint['natural_rows']}
    for category, selected in checkpoint['natural_selection'].items():
        items = []
        for cid, tag in selected:
            row = rows[cid]
            case, local, result = cases[cid], checkpoint['local_cache'][cid], checkpoint['natural_solutions'][cid]
            note = f"{category}; {row['masking_calibrated_confidence']} provisional; H={row['horizontal_real_coverage']:.2f}, N={row['neighbor_vertical_coverage']:.2f}; inferred, NOT official repair"
            payload = plot_payload(case, local, result, frozen, note)
            path = output/'figures/natural_gaps'/f'{cid}.png'
            rendered = plot_local_review(payload, path)
            modules.update(rendered['modules_reused'])
            item = {'candidate_id': cid, 'kind': 'natural_gap', 'path': str(path), 'selection': tag, 'label': category}
            figures.append(item)
            items.append(item)
        montage = output/'figures/natural_gaps'/f'montage_{category}.png'
        plot_montage(items, montage, f'{category} | {len(items)} real cases; unreviewed')
        figures.append({'candidate_id': category, 'kind': 'montage', 'path': str(montage)})
        print(f'Figures {category}: {len(items)}', flush=True)
    for i, pair in enumerate(track.get('representative_pairs', [])):
        # The track implementation provides local geometry bounds and original
        # groups, never an invented repaired group. Adapt its fields explicitly.
        s = float(pair['s_left'])
        key = min(frozen.keys, key=lambda k: abs(float(k)-s))
        uz = np.asarray(pair['left_uz'])
        uz = uz[np.argsort(uz[:, 1])]
        low, high = pair['display_z_low'], pair['display_z_high']
        case = {'candidate_id': f'T{i+1:03d}', 'slice_key': key,
                'lower_uz': [float(np.interp(low, uz[:, 1], uz[:, 0])), low],
                'upper_uz': [float(np.interp(high, uz[:, 1], uz[:, 0])), high]}
        local = index.build_window(case)
        empty = {'curve_uz': np.empty((0, 2)), 'selected_nodes': []}
        note = f"Track link {pair['left_id']} / {pair['right_id']}; illustrative split only; no gap repair or geological decision"
        payload = plot_payload(case, local, empty, frozen, note)
        payload['review_kind'] = 'track'
        path = output/'figures/track_split'/f'T{i+1:03d}.png'
        rendered = plot_local_review(payload, path)
        modules.update(rendered['modules_reused'])
        figures.append({'candidate_id': case['candidate_id'], 'kind': 'track', 'path': str(path),
                        'left_id': pair['left_id'], 'right_id': pair['right_id']})
    save_csv(output/'figure_inventory.csv', figures)
    selected_paths = {r['candidate_id']: r['path'] for r in figures if r['kind'] == 'natural_gap'}
    # A display-only replay must preserve any human annotations already made.
    manual_path = output/'LOCC_manual_review.csv'
    annotations = {}
    if manual_path.exists():
        with manual_path.open(encoding='utf-8-sig', newline='') as stream:
            annotations = {r['candidate_id']: {k: r.get(k, '') for k in ('manual_label', 'manual_notes')}
                           for r in csv.DictReader(stream)}
    save_csv(output/'LOCC_manual_review.csv', [{**r, 'representative_figure': selected_paths.get(r['candidate_id'], ''),
                                              **annotations.get(r['candidate_id'], {'manual_label': '', 'manual_notes': ''})}
                                             for r in checkpoint['natural_rows']])
    return figures, sorted(modules)


def correct_display(prior, output):
    """One display-only correction pass; reuse all inference/track results."""
    validate_output_location(prior, output)
    stages = json.loads((output/'stage_performance.json').read_text(encoding='utf-8'))
    with StageSampler('display_correction_loading', output, stages):
        with (output/'inference_checkpoint.pkl').open('rb') as stream:
            checkpoint = pickle.load(stream)
        frozen = FrozenProfiles(prior)
        index = AdaptiveMeshIndex(frozen)
        track = json.loads((output/'track_for_review.json').read_text(encoding='utf-8'))
    with StageSampler('display_clipping_correction_no_inference', output, stages):
        figures, modules = render_reviews(checkpoint, frozen, index, output, track)
    save_csv(output/'stage_performance.csv', stages)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if frozen_stamp(prior) != manifest['frozen_before']:
        raise AssertionError('Frozen inputs changed during display correction')
    from locc_validation_report import write_report
    write_report(output, manifest, checkpoint, track, stages, figures, modules,
                 finish_wall_seconds=sum(s['wall_seconds'] for s in stages[6:]),
                 extra_plot_adaptive_seconds=index.adaptive_seconds,
                 extra_plot_adaptive_cuts=index.adaptive_cuts)


def finish(prior, output):
    validate_output_location(prior, output)
    if (output/'LOCC_VALIDATION_REPORT.json').exists():
        raise FileExistsError('Completed output is immutable; select a new directory for a new run')
    start = time.perf_counter()
    stages = json.loads((output/'stage_performance.json').read_text(encoding='utf-8'))
    with StageSampler('finish_checkpoint_and_geometry_loading', output, stages):
        with (output/'inference_checkpoint.pkl').open('rb') as stream:
            checkpoint = pickle.load(stream)
        frozen = FrozenProfiles(prior)
        index = AdaptiveMeshIndex(frozen)
    from validate_track_bridge_split import run_track_bridge
    with StageSampler('track_bridge_metrics_and_threshold_sweep', output, stages):
        track = run_track_bridge(frozen, output)
    save_json(output/'track_for_review.json', track)
    with StageSampler('local_paper_figures_and_montages', output, stages):
        figures, modules = render_reviews(checkpoint, frozen, index, output, track)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    unchanged = frozen_stamp(prior) == manifest['frozen_before']
    save_json(output/'frozen_input_verification.json', {'unchanged': unchanged, 'before': manifest['frozen_before'],
                                                       'after': frozen_stamp(prior), 'official_results_written': False})
    if not unchanged:
        raise AssertionError('Frozen inputs changed')
    save_csv(output/'stage_performance.csv', stages)
    from locc_validation_report import write_report
    write_report(output, manifest, checkpoint, track, stages, figures, modules,
                 finish_wall_seconds=time.perf_counter()-start,
                 extra_plot_adaptive_seconds=index.adaptive_seconds,
                 extra_plot_adaptive_cuts=index.adaptive_cuts)
    print(f'Complete: {output / "LOCC_VALIDATION_REPORT.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('infer', 'finish', 'all', 'correct-display'), default='all')
    parser.add_argument('--per-bucket', type=int, default=25)
    parser.add_argument('--reference-output', type=Path, help='Reuse unchanged A/B results after a diagnosed LOCC-only refinement')
    args = parser.parse_args()
    validate_output_location(args.prior, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ('infer', 'all'):
        infer(args.prior, args.output, args.per_bucket, args.reference_output)
    if args.phase in ('finish', 'all'):
        finish(args.prior, args.output)
    if args.phase == 'correct-display':
        correct_display(args.prior, args.output)


if __name__ == '__main__':
    main()
