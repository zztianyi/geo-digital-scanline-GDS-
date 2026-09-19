"""Small behavioral fixtures; hidden truth is never a solver input."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))
from local_orthogonal_cooperative_constraint import solve_locc, neighbor_hypotheses


def window(horizontal=True):
    layers = []
    for z, u in ((.25, .75), (.5, 1.), (.75, .75)):
        layers.append({'z': z,
                       'horizontal': [{'u': u, 'face_id': 10, 'branch_id': 2}] if horizontal else [],
                       'neighbors': [{'s': s, 'values': [u+2*s]} for s in (-.1, -.05, .05, .1)]})
    return {'s': 0., 'lower_uz': [0., 0.], 'upper_uz': [0., 1.],
            'lower_tangent': 4., 'upper_tangent': -4., 'layers': layers}


class LOCCTests(unittest.TestCase):
    def test_neighbor_surface_restores_missing_horizontal_without_invented_face(self):
        result = solve_locc(window(False))
        np.testing.assert_allclose(result['curve_uz'][:, 0], [0, .75, 1, .75, 0], atol=1e-10)
        self.assertEqual(result['selected_source_per_level'], ['V_NEIGHBOR_PREDICTED']*3)
        self.assertTrue(all(node['face_id'] is None for node in result['selected_nodes'][1:-1]))

    def test_competing_branches_are_retained_and_neighbor_evidence_selects(self):
        data = window()
        for layer in data['layers']:
            layer['horizontal'].append({'u': -.5, 'face_id': 99, 'branch_id': 7})
        untouched = copy.deepcopy(data)
        result = solve_locc(data)
        np.testing.assert_allclose(result['curve_uz'][1:-1, 0], [.75, 1., .75])
        self.assertEqual(result['candidate_counts'], [2, 2, 2])
        self.assertIsNotNone(result['second_best_cost'])
        self.assertGreater(result['cost_margin'], 0)
        self.assertEqual(data, untouched)

    def test_horizontal_only_has_no_neighbor_leakage(self):
        data = window(False)
        result = solve_locc(data, mode='HORIZONTAL_ONLY')
        np.testing.assert_array_equal(result['curve_uz'][:, 0], np.zeros(len(result['curve_uz'])))
        self.assertEqual(result['metrics']['horizontal_real_coverage'], 0)

    def test_vertical_baseline_is_only_endpoint_interpolation(self):
        data = window()
        result = solve_locc(data, mode='VERTICAL_ONLY')
        np.testing.assert_array_equal(result['curve_uz'][:, 0], np.zeros(len(result['curve_uz'])))

    def test_neighbor_outlier_does_not_drag_interpolated_surface(self):
        observations = [{'s': -.1, 'values': [1.8]}, {'s': -.05, 'values': [1.9]},
                        {'s': .05, 'values': [2.1]}, {'s': .1, 'values': [20.]}]
        candidates = neighbor_hypotheses(observations, 0., 2.)
        robust = min(candidates, key=lambda r: abs(r['u']-2.))
        self.assertAlmostEqual(robust['u'], 2., places=8)
        self.assertEqual(robust['support_count'], 3)

    def test_no_evidence_is_explicit_fallback_not_mesh_observation(self):
        data = window(False)
        for layer in data['layers']:
            layer['neighbors'] = []
        result = solve_locc(data)
        self.assertEqual(result['metrics']['neighbor_vertical_coverage'], 0)
        self.assertEqual(result['metrics']['face_support_ratio'], 0)
        self.assertEqual(set(result['selected_source_per_level']), {'VERTICAL_PRIOR_FALLBACK'})

    def test_endpoints_are_anchored_and_invalid_layers_rejected(self):
        data = window()
        result = solve_locc(data)
        np.testing.assert_array_equal(result['curve_uz'][[0, -1]], [[0., 0.], [0., 1.]])
        data['layers'][0]['z'] = 1.1
        with self.assertRaises(ValueError):
            solve_locc(data)


if __name__ == '__main__':
    unittest.main()
