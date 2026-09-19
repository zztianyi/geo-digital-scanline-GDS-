"""Real-profile masking, paired ablations and calibration-only risk estimates."""
from __future__ import annotations

from collections import Counter
import copy
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from locc_data import opc
from local_orthogonal_cooperative_constraint import solve_locc
from validate_orthogonal_scanline_constraint import save_csv, save_json

BUCKETS = [('0.01-0.05', .01, .05), ('0.05-0.10', .05, .10),
           ('0.10-0.20', .10, .20), ('0.20-0.40', .20, .40),
           ('0.40-0.80', .40, .80), ('0.80-1.60', .80, 1.60)]
METHODS = ('VERTICAL_ONLY', 'HORIZONTAL_ONLY', 'LOCC')
REGIMES = ('INTACT_HORIZONTAL', 'TARGET_HORIZONTAL_DROPOUT')


def calibrate_observed_neighbor_weight(frozen, index, checkpoint, output):
    """One-dimensional training-only fit after diagnosed observed-branch bias.

    Full neighbor influence remains on virtual nodes. No holdout truth enters
    this parameter selection; initial holdout aggregate was already inspected,
    so a later evaluation on those cases is labeled regression, not a blind test.
    """
    grid = (.01, .03, .10, .30, 1.)
    cases = [c for c in checkpoint['mask_cases'] if c['split'] == 'calibration']
    per_factor = {factor: [] for factor in grid}
    for case in cases:
        window = index.build_window(case)['window']
        for factor in grid:
            result = solve_locc(window, weights={'observed_neighbor_factor': factor})
            metrics = evaluate_curve(result, case['truth_curve'], window)
            per_factor[factor].append({'candidate_id': case['candidate_id'], 'factor': factor, **metrics})
    rows = []
    for factor, metrics in per_factor.items():
        errors = [m['p95_abs_u_m'] for m in metrics]
        rows.append({'observed_neighbor_factor': factor, 'calibration_locations': len(cases),
                     'mean_case_p95_error_m': float(np.mean(errors)),
                     'median_case_p95_error_m': float(np.median(errors)),
                     'p95_case_p95_error_m': float(np.quantile(errors, .95)),
                     'maximum_point_error_m': max(m['max_abs_u_m'] for m in metrics)})
    selected = min(rows, key=lambda r: (r['mean_case_p95_error_m'], -r['observed_neighbor_factor']))
    report = {'selected': selected, 'grid': rows, 'selection_objective': 'Minimum mean case P95 on 82 calibration locations only',
              'diagnosis': 'Per-branch affine-neighbor residual and coverage reward can prefer a smooth but wrong H surface over the endpoint-consistent observed branch',
              'change': 'Reduce neighbor influence ONLY for actual H nodes; virtual nodes keep full weight',
              'holdout_used_for_weight_fit': False,
              'holdout_warning': 'Initial holdout aggregate was inspected before diagnosis; subsequent same-case results are regression validation, not a fresh blind test'}
    save_json(output/'observed_weight_calibration.json', report)
    save_csv(output/'observed_weight_calibration.csv', rows)
    return report


def gap_bucket(height):
    return next((name for name, low, high in BUCKETS if low <= height < high), 'outside_masking_range')


def monotone_paths(lines, arc, s):
    # Join numerical endpoint roundoff only (0.1 micrometre XYZ grid). This
    # does not bridge model holes; each truth window stays on an observed path.
    nodes, edges, degree, _ = opc.line_graph(lines, decimals=7)
    paths = []
    for branch in opc._ordered_branches(nodes, edges, degree):
        xyz = nodes[branch['nodes']]
        if not opc.on_scanline_ray(xyz, arc, s):
            continue
        uz = opc.to_suz(xyz, arc)[:, 1:]
        dz = np.diff(uz[:, 1])
        start, direction = 0, 0
        for i, delta in enumerate(dz):
            sign = int(np.sign(delta)) if abs(delta) > 1e-8 else 0
            if not sign or (direction and sign != direction):
                part = uz[start:i+1]
                if len(part) >= 2:
                    paths.append(part if part[-1, 1] > part[0, 1] else part[::-1])
                start = i if sign else i+1
            direction = sign
        part = uz[start:]
        if len(part) >= 2:
            paths.append(part if part[-1, 1] > part[0, 1] else part[::-1])
    return [p for p in paths if p[-1, 1]-p[0, 1] > .03 and np.all(np.diff(p[:, 1]) > 0)]


def select_masking_cases(frozen, per_bucket=25, seed=20260919):
    rng = np.random.default_rng(seed)
    cases, counts = [], Counter()
    keys = [k for k in frozen.keys if .15 < float(k) < frozen.s[-1]-.15
            and .2 < float(k) % 7 < 6.8]
    for key in rng.permutation(keys):
        missing = [b for b in BUCKETS if counts[b[0]] < per_bucket]
        if not missing:
            break
        # Balance height buckets without searching for low-error locations.
        bucket = min(missing, key=lambda b: (counts[b[0]], -b[1]))
        name, minimum, maximum = bucket
        height = float(rng.uniform(minimum, maximum))
        margin = max(.01, min(.05, height*.2))
        paths = monotone_paths(frozen.slices[key]['slicing']['lines_3d'], frozen.arc, float(key))
        eligible = [p for p in paths if p[-1, 1]-p[0, 1] > height+2*margin]
        if not eligible:
            continue
        widths = np.array([p[-1, 1]-p[0, 1]-height-2*margin for p in eligible])
        path = eligible[rng.choice(len(eligible), p=widths/widths.sum())]
        low = float(rng.uniform(path[0, 1]+margin, path[-1, 1]-height-margin))
        high = low+height
        interp = lambda z: float(np.interp(z, path[:, 1], path[:, 0]))
        # A fixed dense grid plus every original vertex measures geometry
        # between the adaptive quarter levels, not just fitted nodes.
        zs = np.unique(np.r_[np.linspace(low, high, 201), path[(path[:, 1] > low) & (path[:, 1] < high), 1]])
        block = int(float(key)//7)
        case = {'candidate_id': f'M{len(cases)+1:04d}', 'slice_key': str(key), 's': float(key),
                'gap_height': height, 'gap_bucket': name, 'spatial_block': block,
                'split': 'calibration' if block % 2 == 0 else 'holdout',
                'lower_uz': [interp(low), low], 'upper_uz': [interp(high), high],
                'lower_neighbor_uz': [interp(low-margin), low-margin],
                'upper_neighbor_uz': [interp(high+margin), high+margin],
                'truth_curve': np.column_stack((np.interp(zs, path[:, 1], path[:, 0]), zs))}
        cases.append(case)
        counts[name] += 1
    if not cases or not any(c['split'] == 'holdout' for c in cases):
        raise RuntimeError('No adequate real masking sample or spatial holdout')
    return cases


def evaluate_curve(result, truth, window):
    curve = np.asarray(result['curve_uz'])
    pred = np.column_stack((np.interp(truth[:, 1], curve[:, 1], curve[:, 0]), truth[:, 1]))
    errors = abs(pred[:, 0]-truth[:, 0])
    hausdorff = max(cKDTree(truth).query(pred)[0].max(), cKDTree(pred).query(truth)[0].max())
    truth_length = np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
    # Branch accuracy is defined only where a horizontal layer has multiple
    # geometrically distinct candidates AND one agrees with the hidden path.
    correct, total = 0, 0
    for node, layer in zip(result['selected_nodes'][1:-1], window['layers']):
        values = np.unique(np.round([h['u'] for h in layer['horizontal']], 6))
        actual = np.interp(layer['z'], truth[:, 1], truth[:, 0])
        if len(values) > 1 and np.min(abs(values-actual)) < 1e-4:
            total += 1
            correct += abs(node['u']-actual) < 1e-4
    return {'median_abs_u_m': float(np.median(errors)), 'p95_abs_u_m': float(np.quantile(errors, .95)),
            'max_abs_u_m': float(errors.max()), 'endpoint_continuity_m': float(max(errors[0], errors[-1])),
            'branch_correct_layers': int(correct), 'branch_evaluable_layers': total,
            'branch_accuracy': correct/total if total else None,
            'curve_length_ratio': float(np.linalg.norm(np.diff(pred, axis=0), axis=1).sum()/truth_length),
            'local_hausdorff_m': float(hausdorff), 'evaluation_points': len(truth)}


def run_masking(frozen, index, output, per_bucket=25, reference=None):
    cases = reference['mask_cases'] if reference is not None else select_masking_cases(frozen, per_bucket)
    old_rows = {(r['candidate_id'], r['regime'], r['method']): r for r in reference['mask_rows']} if reference else {}
    save_csv(output/'masking_cases.csv', [{k: v for k, v in c.items() if k != 'truth_curve'} for c in cases])
    rows, solutions, dp_seconds = [], {}, 0.
    for i, case in enumerate(cases):
        local = index.build_window(case)
        for regime in REGIMES:
            window = copy.deepcopy(local['window'])
            if regime == 'TARGET_HORIZONTAL_DROPOUT':
                # The model remains unchanged: this removes observations from
                # solver input, including FaceIDs and per-hit linked neighbors.
                for layer in window['layers']:
                    layer['horizontal'] = []
            for method in METHODS:
                lookup = (case['candidate_id'], regime, method)
                if reference is not None and method != 'LOCC':
                    rows.append({**old_rows[lookup], 'baseline_reused': True, 'this_run_solve_seconds': 0.})
                    solutions[lookup] = reference['mask_solutions'][lookup]
                    continue
                t0 = time.perf_counter()
                result = solve_locc(window, method)
                elapsed = time.perf_counter()-t0
                if method == 'LOCC':
                    dp_seconds += elapsed
                metrics = evaluate_curve(result, case['truth_curve'], window)
                row = {k: case[k] for k in ('candidate_id', 'slice_key', 's', 'gap_height', 'gap_bucket', 'spatial_block', 'split')}
                row.update(regime=regime, method=method, solve_seconds=elapsed, **metrics, **result['metrics'],
                           best_cost=result['best_cost'], second_best_cost=result['second_best_cost'],
                           normalized_cost_margin=result['normalized_cost_margin'], baseline_reused=False,
                           this_run_solve_seconds=elapsed)
                rows.append(row)
                solutions[(case['candidate_id'], regime, method)] = result
        if (i+1) % 25 == 0:
            print(f'Masking {i+1}/{len(cases)}', flush=True)
    summary = summarize_masking(rows)
    summary.update(locations=len(cases), case_regimes=len(cases)*len(REGIMES), method_runs=len(rows),
                   new_method_runs=sum(not r['baseline_reused'] for r in rows),
                   reused_method_results=sum(r['baseline_reused'] for r in rows),
                   locc_solver_seconds=dp_seconds, seed=20260919,
                   sampling='One real observed monotone-Z path per random scanline; height-stratified; no error-based selection',
                   truth='Original model direct vertical curve, not geological truth; endpoint topology quantized at 1e-7 m',
                   holdout='Alternating 7 m spatial blocks; 0.2 m guards keep the +/-0.1 m neighborhood off boundaries',
                   branch_accuracy_definition='Ambiguous horizontal layers with truth-matching branch within 0.1 mm; dropout is N/A',
                   endpoint_definition='Anchored by construction; zero endpoint distance is NOT independent evidence of accuracy')
    save_csv(output/'masking_results.csv', rows)
    save_csv(output/'masking_bucket_summary.csv', summary['by_bucket'])
    save_json(output/'masking_summary.json', summary)
    return cases, rows, solutions, summary


def summarize_group(rows):
    correct = sum(r['branch_correct_layers'] for r in rows)
    total = sum(r['branch_evaluable_layers'] for r in rows)
    return {'n': len(rows), 'median_case_median_u_m': float(np.median([r['median_abs_u_m'] for r in rows])),
            'median_case_p95_u_m': float(np.median([r['p95_abs_u_m'] for r in rows])),
            'p95_case_p95_u_m': float(np.quantile([r['p95_abs_u_m'] for r in rows], .95)),
            'max_error_m': max(r['max_abs_u_m'] for r in rows),
            'median_hausdorff_m': float(np.median([r['local_hausdorff_m'] for r in rows])),
            'median_length_ratio': float(np.median([r['curve_length_ratio'] for r in rows])),
            'branch_accuracy': correct/total if total else None, 'branch_evaluable_layers': total}


def summarize_masking(rows):
    overall, by_bucket, paired, correlations = [], [], [], []
    rng = np.random.default_rng(42)
    for split in ('calibration', 'holdout', 'all'):
        subset = [r for r in rows if split == 'all' or r['split'] == split]
        for regime in REGIMES:
            for method in METHODS:
                rr = [r for r in subset if r['regime'] == regime and r['method'] == method]
                if not rr:
                    continue
                overall.append({'split': split, 'regime': regime, 'method': method, **summarize_group(rr)})
                for name, _, _ in BUCKETS:
                    bb = [r for r in rr if r['gap_bucket'] == name]
                    if bb:
                        by_bucket.append({'split': split, 'regime': regime, 'method': method,
                                          'gap_bucket': name, **summarize_group(bb)})
            if split != 'holdout':
                continue
            lookup = {(r['candidate_id'], r['method']): r for r in subset if r['regime'] == regime}
            locc = [r for r in subset if r['regime'] == regime and r['method'] == 'LOCC']
            for comparator in METHODS[:2]:
                differences = np.array([lookup[(r['candidate_id'], comparator)]['p95_abs_u_m']-r['p95_abs_u_m'] for r in locc])
                # Cluster resampling by spatial block avoids pretending nearby
                # locations are independent observations of a new rock surface.
                blocks = sorted(set(r['spatial_block'] for r in locc))
                clusters = [differences[[r['spatial_block'] == b for r in locc]] for b in blocks]
                bootstrap = [np.median(np.concatenate([clusters[j] for j in rng.integers(len(blocks), size=len(blocks))])) for _ in range(1000)]
                base = np.array([lookup[(r['candidate_id'], comparator)]['p95_abs_u_m'] for r in locc])
                current = np.array([r['p95_abs_u_m'] for r in locc])
                paired.append({'regime': regime, 'comparator': comparator, 'n': len(locc), 'spatial_blocks': len(blocks),
                               'median_paired_p95_improvement_m': float(np.median(differences)),
                               'cluster_bootstrap_95ci_m': np.quantile(bootstrap, [.025, .975]).tolist(),
                               'ratio_of_median_p95_improvement_percent': float(100*(1-np.median(current)/max(np.median(base), 1e-12))),
                               'better_cases': int(np.sum(differences > 1e-6)), 'worse_cases': int(np.sum(differences < -1e-6)),
                               'tie_cases': int(np.sum(abs(differences) <= 1e-6))})
            for feature in ('gap_height', 'horizontal_real_coverage', 'neighbor_vertical_coverage',
                            'neighbor_residual', 'path_smoothness', 'endpoint_tangent_error',
                            'normalized_cost_margin', 'ambiguous_level_fraction'):
                valid = [r for r in locc if r.get(feature) is not None]
                if len(valid) >= 5 and np.ptp([r[feature] for r in valid]) > 1e-10:
                    corr = spearmanr([r[feature] for r in valid], [r['p95_abs_u_m'] for r in valid])
                    correlations.append({'regime': regime, 'feature': feature, 'n': len(valid),
                                         'spearman_rho': float(corr.statistic), 'pvalue_exploratory': float(corr.pvalue)})
    return {'overall': overall, 'by_bucket': by_bucket, 'paired_holdout': paired, 'evidence_correlations_holdout': correlations}


def evidence_features(row):
    return np.array([np.log2(max(row['gap_height'], .001)/.05)/4,
                     row['horizontal_real_coverage'], row['neighbor_vertical_coverage'],
                     row['two_sided_neighbor_fraction'], row['ambiguous_level_fraction'],
                     min(3., np.log1p((row.get('neighbor_residual') or 0)/.01))/3,
                     min(3., np.log1p(row['endpoint_tangent_error']))/3,
                     row.get('fallback_fraction', 0.)])


class MaskingRiskCalibration:
    """Relative risk tiers, NOT geological repair approval or an error guarantee."""
    def __init__(self, rows):
        self.train = [r for r in rows if r['method'] == 'LOCC' and r['split'] == 'calibration']
        self.features = np.array([evidence_features(r) for r in self.train])
        self.errors = np.array([r['p95_abs_u_m'] for r in self.train])
        loo = [self.estimate(r, excluded_id=r['candidate_id']) for r in self.train]
        self.cutoffs = np.quantile([r[0] for r in loo], [.5, .85])
        self.distance_cutoff = float(np.quantile([r[1] for r in loo], .95))
        residuals = [r['neighbor_residual'] for r in self.train if r['neighbor_residual'] is not None]
        margins = [r['normalized_cost_margin'] for r in self.train if r['normalized_cost_margin'] is not None]
        self.conflict_residual = float(np.quantile(residuals, .90)) if residuals else None
        self.low_margin = float(np.quantile(margins, .20)) if margins else None

    def estimate(self, row, excluded_id=None):
        distances = np.linalg.norm(self.features-evidence_features(row), axis=1)
        if excluded_id:
            distances[[r['candidate_id'] == excluded_id for r in self.train]] = np.inf
        ids = np.argsort(distances)[:8]
        ids = ids[np.isfinite(distances[ids])]
        return float(np.quantile(self.errors[ids], .9)), float(distances[ids[-1]]), len(ids)

    def classify(self, row):
        risk, distance, n = self.estimate(row)
        in_domain = (distance <= self.distance_cutoff and .01 <= row['gap_height'] <= 1.6
                     and row.get('fallback_fraction', 0) == 0)
        label = ('HIGH' if risk <= self.cutoffs[0] else 'MEDIUM' if risk <= self.cutoffs[1] else 'LOW') if in_domain else 'OUT_OF_DOMAIN'
        return {'masking_calibrated_confidence': label, 'predicted_local_p90_case_p95_error_m': risk,
                'calibration_neighbor_cases': n, 'calibration_distance': distance,
                'suggested_action': 'REVIEW_REPAIR' if label == 'HIGH' else 'UNCERTAIN',
                'confidence_status': 'PROVISIONAL_RELATIVE_RISK; manual review and engineering tolerance required'}

    def report(self, rows):
        holdout = [r for r in rows if r['method'] == 'LOCC' and r['split'] == 'holdout']
        evaluated = [{**r, **self.classify(r)} for r in holdout]
        groups = []
        for regime in REGIMES:
            for label in ('HIGH', 'MEDIUM', 'LOW', 'OUT_OF_DOMAIN'):
                rr = [r for r in evaluated if r['regime'] == regime and r['masking_calibrated_confidence'] == label]
                if rr:
                    groups.append({'regime': regime, 'confidence': label, **summarize_group(rr)})
        return {'calibration_method_runs': len(self.train), 'risk_error_cutoffs_m': self.cutoffs.tolist(),
                'distance_cutoff': self.distance_cutoff, 'holdout_by_confidence': groups,
                'calibration_only_conflict_residual_m': self.conflict_residual,
                'calibration_only_low_margin': self.low_margin,
                'meaning': 'Relative 50%/85% leave-one-location-out risk ranks, calibrated on model masking only; no absolute safe-repair threshold',
                'natural_gap_domain_shift': 'Masking an intact mesh does not recreate missing geometry or geological discontinuities'}
