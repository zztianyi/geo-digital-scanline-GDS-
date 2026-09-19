"""Focused geometry checks; no full-model or GUI execution."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))
import orthogonal_profile_constraint as opc


ARC = {'center': [0., 0.], 'radius': 10., 'angle_min': -.5, 'angle_max': .5, 'z_range': [0., 2.]}


class OrthogonalConstraintTests(unittest.TestCase):
    def test_unwrap_across_pi(self):
        arc = {**ARC, 'angle_min': 3., 'angle_max': 3.5}
        theta = np.array([3.1, 3.3])
        pts = np.column_stack((12*np.cos(theta), 12*np.sin(theta), [1, 2]))
        np.testing.assert_allclose(opc.to_suz(pts, arc), [[1, 2, 1], [3, 2, 2]], atol=1e-12)

    def test_exact_cartesian_ray_and_multiple_branches(self):
        # Chords at x=11 and x=12: s=5 means the positive X ray, exact u=1,2.
        lines = np.array([[[11, -2, 1], [11, 2, 1]], [[12, -2, 1], [12, 2, 1]]], dtype=float)
        clean = opc.clean_horizontal(lines, [42, 43], ARC)
        hits = opc.horizontal_crossings(clean, [5.], ARC)
        np.testing.assert_allclose(np.sort(hits[:, 1]), [1., 2.], atol=1e-12)
        self.assertEqual(set(hits[:, 3]), {42, 43})
        self.assertEqual(len(set(hits[:, 2])), 2)

    def test_collinear_horizontal_interval_preserves_endpoints(self):
        clean = opc.clean_horizontal(np.array([[[11., 0, 1], [12., 0, 1]]]), [7], ARC)
        hits = opc.horizontal_crossings(clean, [5.], ARC)
        np.testing.assert_allclose(np.sort(hits[:, 1]), [1., 2.])

    def test_snap_only_unique_endpoints_preserves_face_ids(self):
        lines = np.array([[[11., -1, 1], [11., 0, 1]], [[11., 1e-8, 1], [11., 1, 1]]])
        clean = opc.clean_horizontal(lines, [100, 101], ARC, 2e-8)
        self.assertEqual(clean['diagnostics']['snapped_pairs'], 1)
        self.assertEqual(clean['diagnostics']['components'], 1)
        np.testing.assert_array_equal(clean['face_ids'], [100, 101])
        # A third endpoint makes the same junction ambiguous; do not weld it.
        ambiguous = np.concatenate((lines, [[[11.+1e-8, 0, 1], [12, 0, 1]]]))
        self.assertEqual(opc.clean_horizontal(ambiguous, [100, 101, 102], ARC, 2e-8)['diagnostics']['snapped_pairs'], 0)

    def test_multi_observation_states_do_not_force_matching(self):
        v = {0: np.array([[1.], [2.]]), 1: np.array([[3.]]), 3: np.array([[8.]])}
        h = {0: np.array([[1., 0, 10], [4., 1, 11]]), 2: np.array([[5., 0, 12]])}
        counts, _, _ = opc.match_observations(v, h, 5, .01)
        self.assertEqual(counts, {'VH_MATCH': 1, 'VH_CONFLICT': 1, 'V_ONLY': 2, 'H_ONLY': 1, 'EMPTY': 1})

    def test_gap_needs_two_supported_levels_and_traceable_mesh_chain(self):
        lines = np.array([[[11., 0, 0], [11, 0, .4]], [[11, 0, .65], [11, 0, 1.]]])
        candidates = opc.small_gap_candidates(lines, ARC, 5., max_gap_z=.251)
        self.assertEqual(len(candidates), 1)
        levels = np.array([.45, .55])
        h = {0: np.array([[1., 0., 123.]]), 1: np.array([[1., 0., 124.]])}
        result = opc.supported_gap(candidates[0], h, {}, levels, 1e-5)
        self.assertTrue(result['supported'])
        repair_lines = np.array([[[11., 0, .4], [11, 0, .525]], [[11, 0, .525], [11, 0, .65]]])
        repair, reason = opc.traceable_repair(candidates[0], repair_lines, [123, 124], lines)
        self.assertEqual(reason, 'traceable_mesh_chain')
        np.testing.assert_array_equal(repair['face_ids'], [123, 124])
        missing, reason = opc.traceable_repair(candidates[0], repair_lines[:1], [123], lines)
        self.assertIsNone(missing)
        self.assertEqual(reason, 'no_complete_mesh_chain')
        self.assertFalse(opc.supported_gap(candidates[0], h, {0: np.array([[1.]])}, levels, 1e-5)['supported'])

    def test_unsupported_and_undersampled_gaps_stay_missing(self):
        lines = np.array([[[11., 0, 0], [11, 0, .4]], [[11, 0, .65], [11, 0, 1.]]])
        candidate = opc.small_gap_candidates(lines, ARC, 5., max_gap_z=.251)[0]
        self.assertFalse(opc.supported_gap(candidate, {}, {}, np.array([.45, .55]), .01)['supported'])
        self.assertEqual(opc.supported_gap(candidate, {}, {}, np.array([.5]), .01)['reason'], 'insufficient_horizontal_levels')

    def test_ch_requires_same_local_branch_and_all_common_levels(self):
        a = {'z_min': 0., 'z_max': 1.}
        b = {'z_min': 0., 'z_max': 1.}
        lm = rm = {0: np.array([[1.]]), 1: np.array([[1.]])}
        hl = {0: np.array([[1., 4, 10]]), 1: np.array([[1., 5, 11]])}
        hr = {0: np.array([[1., 4, 12]]), 1: np.array([[1., 8, 13]])}
        metrics = opc.pair_metrics(a, b, lm, rm, hl, hr, .001)
        self.assertEqual(metrics['Ch'], .5)
        self.assertEqual(metrics['Cz'], 1.)
        self.assertEqual(opc.pair_metrics(a, b, {}, {}, hl, hr, .001)['Ch'], None)

    def test_du_comes_from_the_supported_branch(self):
        group = {'z_min': 0., 'z_max': 1.}
        left = {0: np.array([[0.], [1.]])}
        right = {0: np.array([[.01], [2.]])}
        hl = {0: np.array([[1., 7., 10.]])}
        hr = {0: np.array([[2., 7., 11.]])}
        result = opc.pair_metrics(group, group, left, right, hl, hr, .001)
        self.assertEqual(result['Ch'], 1.)
        self.assertEqual(result['median_du'], 1.)
        self.assertEqual(result['raw_nearest_median_du'], .01)

    def test_opposite_half_ray_is_not_observable(self):
        nodes = np.array([[-11., 0, 0], [-11., 0, 1.]])
        self.assertFalse(opc.on_scanline_ray(nodes, ARC, 5.))
        self.assertTrue(opc.on_scanline_ray(-nodes, ARC, 5.))
        self.assertEqual(opc.group_level_map({'uz': [[1., 0.], [1., 1.]], 'audit_observable': False}, np.array([.5])), {})


if __name__ == '__main__':
    unittest.main()
