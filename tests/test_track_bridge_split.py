"""Small real-geometry fixtures for the analysis-only track bridge contract."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/99_experiments'))
import validate_track_bridge_split as bridge


ARC = {'center': [0., 0.], 'radius': 10., 'angle_min': 0., 'angle_max': 1.}


def group(gid, s, lo=0., hi=1., u=0.):
    theta = s/10.
    return {'id': gid, 's': s, 'uz': [[u, lo], [u, hi]],
            'z_min': lo, 'z_max': hi, 'length_2d': hi-lo,
            'node_order': [[(10.+u)*np.cos(theta), (10.+u)*np.sin(theta), z]
                           for z in (lo, hi)], 'audit_observable': True}


def evidence(indices, supported, sampled=None):
    return {'levels': set(indices), 'sampled': set(indices if sampled is None else sampled),
            'branches': {i: {7} for i in supported}}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def fixture(prior):
    """AB is short, BC continuous, CD singleton, FG wholly unobserved; E isolated."""
    groups = {'0.00': [group('A', 0.)],
              '0.10': [group('B', .1), group('E', .1), group('F', .1, 2., 3.)],
              '0.20': [group('C', .2), group('G', .2, 2., 3.)],
              '0.30': [group('D', .3, .39, .41)]}
    byid = {g['id']: g for gs in groups.values() for g in gs}
    rows = []
    for a, b, matched in [('A', 'B', True), ('B', 'C', True), ('C', 'D', True),
                          ('F', 'G', True), ('A', 'E', False)]:
        rows.append({'left_id': a, 'right_id': b, 's_left': byid[a]['s'], 's_right': byid[b]['s'],
                     'Cz': .5, 'Ch': 1. if a != 'F' else '', 'common_levels': 1,
                     'supported_levels': 1, 'median_du': .123 if a != 'F' else '',
                     'raw_nearest_median_du': .123, 'C': .8, 'matched': matched})
    write_csv(prior/'baseline_consistency.csv', rows)
    tracks = []
    for tid, ids in [('original-ABCD', ['A', 'B', 'C', 'D']), ('original-E', ['E']),
                     ('original-FG', ['F', 'G'])]:
        ss = [byid[g]['s'] for g in ids]
        tracks.append({'track_id': tid, 'slice_count': len(set(ss)), 'group_count': len(ids),
                       'span_m': max(ss)-min(ss), 'max_height_m': 1.,
                       'length_sum_m': sum(byid[g]['length_2d'] for g in ids), 'group_ids': ids})
    write_csv(prior/'group_tracks.csv', tracks)
    (prior/'tolerance_calibration.json').write_text(json.dumps({'applied_tau_u_m': 1e-5}), encoding='utf-8')
    levels = np.arange(31, dtype=float)/10.
    horizontal = {}
    for zi in range(11):
        intervals = [(.1, .2)]
        if zi in (1, 2):
            intervals.append((0., .1))
        if zi == 4:
            intervals.append((.2, .3))
        lines = np.array([[[10.*np.cos(s/10.), 10.*np.sin(s/10.), levels[zi]]
                           for s in interval] for interval in intervals])
        horizontal[zi] = bridge.opc.clean_horizontal(lines, np.arange(len(lines)), ARC)
    return SimpleNamespace(prior=prior, keys=list(groups), s=np.array([0., .1, .2, .3]),
                           levels=levels, horizontal=horizontal, groups=groups, arc=ARC)


class BridgeMetricsTest(unittest.TestCase):
    def test_continuous_interrupted_and_singleton_center_spans(self):
        levels = np.arange(5)/10.
        left, right = group('a', 0., 0., 2.), group('b', .1, 0., 4.)
        for supported, length, count in [([0, 1, 2, 3, 4], .4, 5), ([0, 1, 3, 4], .1, 4), ([2], 0., 1)]:
            with self.subTest(supported=supported):
                result = bridge.bridge_metrics(left, right, levels, evidence(range(5), supported),
                                               evidence(range(5), supported), set(range(5)))
                self.assertAlmostEqual(result['L_support'], length)
                self.assertAlmostEqual(result['R_bridge'], length/2.)
                self.assertEqual(result['supported_levels'], count)
                self.assertEqual(result['common_sampled_levels'], 5)

    def test_overlap_and_bridge_use_minimum_height_not_union(self):
        levels = np.array([1., 1.1, 1.2, 1.3])
        result = bridge.bridge_metrics(group('a', 0., 0., 2.), group('b', .1, 1., 4.), levels,
                                       evidence(range(4), range(4)), evidence(range(4), range(4)), set(range(4)))
        self.assertAlmostEqual(result['L_overlap_z'], 1.)
        self.assertAlmostEqual(result['R_overlap'], .5)
        self.assertAlmostEqual(result['L_support'], .3)
        self.assertAlmostEqual(result['R_bridge'], .15)

    def test_missing_sample_breaks_run_and_singleton_evidence_is_unresolved(self):
        levels = np.array([0., .1, .2])
        result = bridge.bridge_metrics(group('a', 0.), group('b', .1), levels,
                                       evidence([0, 1, 2], [0, 2], [0, 2]),
                                       evidence([0, 1, 2], [0, 2], [0, 2]), {0, 2})
        self.assertEqual(result['L_support'], 0.)
        self.assertEqual(result['missing_horizontal_common_levels'], 1)
        singleton = bridge.bridge_metrics(group('a', 0.), group('b', .1), levels,
                                          evidence([0, 1, 2], [1], [1]),
                                          evidence([0, 1, 2], [1], [1]), {1})
        self.assertTrue(singleton['undersampled'])
        self.assertEqual(singleton['common_levels'], 3)
        self.assertEqual(singleton['common_sampled_levels'], 1)

    def test_real_horizontal_branch_and_radial_tolerance_required(self):
        groups = {'0.00': [group('a', 0.)], '0.10': [group('b', .1)]}
        angles = np.array([-.01, .005, .015])
        points = np.column_stack((10.*np.cos(angles), 10.*np.sin(angles), np.zeros(3)))
        # Different monotone branches cross the two rays at near-equal radius.
        clean = {'lines_3d': np.array([points[:2], points[1:]]),
                 'edge_branch': np.array([1, 2]), 'face_ids': np.array([10, 11])}
        frozen = SimpleNamespace(keys=list(groups), s=np.array([0., .1]), levels=np.array([0.]),
                                 groups=groups, arc=ARC, horizontal={0: clean})
        obs = bridge.build_group_evidence(frozen, {'a', 'b'}, .001)
        result = bridge.bridge_metrics(groups['0.00'][0], groups['0.10'][0], frozen.levels,
                                       obs['a'], obs['b'], {0})
        self.assertEqual(result['common_sampled_levels'], 1)
        self.assertEqual(result['supported_levels'], 0)
        clean['edge_branch'][:] = 1
        obs = bridge.build_group_evidence(frozen, {'a', 'b'}, .001)
        result = bridge.bridge_metrics(groups['0.00'][0], groups['0.10'][0], frozen.levels,
                                       obs['a'], obs['b'], {0})
        self.assertEqual(result['supported_levels'], 1)
        obs = bridge.build_group_evidence(frozen, {'a', 'b'}, 1e-6)
        self.assertFalse(obs['a']['branches'].get(0))
        self.assertFalse(obs['b']['branches'].get(0))

    def test_nonfinite_and_opposite_ray_groups_remain_unobserved(self):
        invalid = group('bad', 0.)
        invalid['uz'][1][0] = float('nan')
        opposite = group('opposite', .1)
        opposite['node_order'] = [[-p[0], -p[1], p[2]] for p in opposite['node_order']]
        frozen = SimpleNamespace(keys=['0.00', '0.10'], s=np.array([0., .1]), levels=np.array([0., 1.]),
                                 groups={'0.00': [invalid], '0.10': [opposite]}, horizontal={}, arc=ARC)
        obs = bridge.build_group_evidence(frozen, {'bad', 'opposite'}, 1e-5)
        self.assertFalse(obs['bad']['levels'])
        self.assertFalse(obs['opposite']['levels'])


class BridgeIntegrationTest(unittest.TestCase):
    def test_sweep_preserves_baseline_and_sparse_links_without_rematching(self):
        with tempfile.TemporaryDirectory(prefix='track-bridge-') as tmp:
            prior = Path(tmp)/'prior'
            prior.mkdir()
            frozen = fixture(prior)
            before = {p.name: p.read_bytes() for p in prior.iterdir()}
            groups_before = json.dumps(frozen.groups)
            with patch.object(bridge.opc, 'horizontal_crossings', wraps=bridge.opc.horizontal_crossings) as crossings:
                result = bridge.run_track_bridge(frozen, Path(tmp)/'output')
                self.assertEqual(crossings.call_count, 11)
            baseline = result['baseline']
            self.assertEqual((baseline['group_count'], baseline['components'], baseline['multi_slice_tracks'],
                              baseline['isolated_groups'], baseline['matched_links']), (7, 3, 2, 1, 4))
            illustrative = result['illustrative']
            self.assertEqual(illustrative['threshold_status'], 'ILLUSTRATIVE_NOT_APPROVED')
            self.assertEqual((illustrative['removed_links'], illustrative['original_tracks_split'],
                              illustrative['components'], illustrative['multi_slice_tracks'],
                              illustrative['isolated_groups'], illustrative['protected_undersampled_links']),
                             (1, 1, 4, 2, 2, 2))
            self.assertIsNone(illustrative['confirmed_overmerges'])
            self.assertTrue(result['baseline_zero_grid_exact'])
            with (Path(tmp)/'output/track_bridge_pairs.csv').open(encoding='utf-8-sig', newline='') as stream:
                pairs = {(r['left_id'], r['right_id']): r for r in csv.DictReader(stream)}
            self.assertEqual(len(pairs), 5)
            self.assertEqual(pairs['A', 'B']['illustrative_decision'], 'REMOVED_ILLUSTRATIVE')
            self.assertEqual(pairs['B', 'C']['illustrative_decision'], 'RETAINED_SAMPLED')
            self.assertEqual(pairs['C', 'D']['illustrative_decision'], 'RETAINED_UNRESOLVED')
            self.assertEqual(pairs['F', 'G']['illustrative_decision'], 'RETAINED_UNRESOLVED')
            self.assertEqual(pairs['A', 'E']['matched'], 'False')
            self.assertEqual(pairs['A', 'E']['illustrative_retained'], 'False')
            self.assertEqual(float(pairs['A', 'B']['median_du']), .123)
            self.assertEqual(pairs['F', 'G']['Ch'], '')
            self.assertEqual(int(pairs['C', 'D']['common_sampled_levels']), 1)
            self.assertEqual(float(pairs['C', 'D']['L_support']), 0.)
            with (Path(tmp)/'output/track_bridge_sweep.csv').open(encoding='utf-8-sig', newline='') as stream:
                sweep = list(csv.DictReader(stream))
            self.assertEqual(len(sweep), 80)
            origin = sweep[0]
            self.assertEqual((int(origin['retained_links']), int(origin['removed_links']),
                              int(origin['original_tracks_split']), int(origin['components'])), (4, 0, 0, 3))
            self.assertTrue(all(int(r['protected_undersampled_links']) == 2 for r in sweep))
            with (Path(tmp)/'output/major_structure_preservation.csv').open(encoding='utf-8-sig', newline='') as stream:
                preservation = list(csv.DictReader(stream))
            major = next(r for r in preservation if r['original_track_id'] == 'original-ABCD'
                         and r['threshold_status'] == 'ILLUSTRATIVE_NOT_APPROVED')
            self.assertEqual(int(major['child_count']), 2)
            self.assertAlmostEqual(float(major['largest_child_group_fraction']), .75)
            self.assertAlmostEqual(float(major['largest_child_length_fraction']), 2.02/3.02)
            representatives = result['representative_pairs']
            self.assertTrue(any(r['decision'] == 'REMOVED_ILLUSTRATIVE' for r in representatives))
            self.assertTrue(any(r['decision'].startswith('RETAINED') for r in representatives))
            self.assertLessEqual(len(representatives), 6)
            for row in representatives:
                self.assertLessEqual(row['display_z_high']-row['display_z_low'], 1.)
                self.assertTrue(row['left_uz'])
                self.assertTrue(row['right_uz'])
            self.assertEqual(json.dumps(frozen.groups), groups_before)
            self.assertEqual({p.name: p.read_bytes() for p in prior.iterdir()}, before)
            self.assertEqual(len(list((Path(tmp)/'output').iterdir())), 5)

    def test_tolerance_reads_observed_filenames_and_ignores_score_scale(self):
        sources = [('tolerance_calibration.json', {'applied_tau_u_m': .00001}),
                   ('ORTHOGONAL_SCANLINE_CONSTRAINT_REPORT.json',
                    {'constraint': {'tolerance': {'applied_tau_u_m': .00001}}, 'baseline': {'tau_u_m': 99.}}),
                   ('constraint_diagnostics.json', {'tolerance': {'applied_tau_u_m': .00001}}),
                   ('manifest.json', {'matching_tolerance_m': .00001})]
        for filename, payload in sources:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory(prefix='track-tolerance-') as tmp:
                (Path(tmp)/filename).write_text(json.dumps(payload), encoding='utf-8')
                tolerance, source = bridge.load_matching_tolerance(Path(tmp))
                self.assertEqual(tolerance, .00001)
                self.assertIn(filename, source)
        with tempfile.TemporaryDirectory(prefix='track-tolerance-') as tmp:
            (Path(tmp)/'ORTHOGONAL_SCANLINE_CONSTRAINT_REPORT.json').write_text(
                json.dumps({'baseline': {'tau_u_m': 99.}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'matching tolerance'):
                bridge.load_matching_tolerance(Path(tmp))


if __name__ == '__main__':
    unittest.main()
