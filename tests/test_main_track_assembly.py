"""Regressions for continuing past the end of a locally selected branch."""
import unittest
import numpy as np
from test_dominant_observed_branch import fixture
from dominant_observed_branch import solve_dominant_branch
from observed_surface_graph import build_surface_graph


class MainTrackAssemblyTests(unittest.TestCase):
    def solve(self, paths):
        profile, uz, branches = fixture(paths)
        graph = build_surface_graph([dict(s=0., branches=branches)])
        before = profile.nodes.copy()
        result = solve_dominant_branch(profile, uz, branches=branches, surface_graph=graph, target_s=0.)
        np.testing.assert_array_equal(profile.nodes, before)
        return result, profile, uz

    def test_three_different_tracks_continue_across_two_gaps(self):
        result, _, _ = self.solve([[[0, 0], [0, 1]], [[0, 1.02], [0, 2.02]],
                                   [[0, 2.04], [0, 3.04]]])
        self.assertAlmostEqual(result['curve_uz'][:, 1].min(), 0.)
        self.assertAlmostEqual(result['curve_uz'][:, 1].max(), 3.04)
        self.assertEqual(result['branch_switch_count'], 2)
        self.assertAlmostEqual(result['connector_length_m'], .04)
        self.assertEqual(result['inferred_length_m'], 0.)
        self.assertEqual({e['branch_id'] for e in result['path_edges'] if e['source'].startswith('OBSERVED')}, {0, 1, 2})

    def test_intersection_is_used_and_observed_edges_keep_provenance(self):
        result, profile, uz = self.solve([[[0, 0], [.04, .8]], [[.04, .2], [0, 1]]])
        self.assertEqual(result['branch_switch_count'], 1)
        self.assertAlmostEqual(result['connector_length_m'], 0., places=9)
        self.assertAlmostEqual(result['junction']['a_point_uz'][1], .5)
        for edge in result['path_edges']:
            if not edge['source'].startswith('OBSERVED'):
                self.assertIsNone(edge['face_id'])
                continue
            a, b = uz[edge['node_ids']]
            np.testing.assert_allclose(edge['points_uz'], [a+t*(b-a) for t in (edge['t0'], edge['t1'])])
            self.assertEqual(edge['source_face_ids'], profile.source_face_ids[edge['edge_id']])

    def test_fold_survives_connection_to_next_region(self):
        fold = [[0, 0], [.1, .6], [.2, .4], [0, 1]]
        result, _, _ = self.solve([fold, [[0, 1.02], [0, 2]]])
        self.assertEqual(result['branch_switch_count'], 1)
        self.assertTrue(np.any(np.diff(result['curve_uz'][:, 1]) < 0))
        self.assertAlmostEqual(result['curve_uz'][:, 1].max(), 2.)

    def test_parallel_alternative_is_not_a_reason_to_leave_complete_branch(self):
        result, _, _ = self.solve([[[0, 0], [0, 2]], [[.02, .5], [.02, 1.5]]])
        self.assertEqual(result['branch_switch_count'], 0)
        np.testing.assert_array_equal(result['curve_uz'], [[0, 0], [0, 2]])

    def test_can_extend_both_ends_from_disjoint_portions_of_another_branch(self):
        # First branch wins the support-free ranking by its extra folded length.
        result, _, _ = self.solve([[[0, .2], [.2, .8], [-.2, .4], [0, .8]],
                                   [[0, 0], [0, 1]]])
        self.assertAlmostEqual(result['curve_uz'][:, 1].min(), 0.)
        self.assertAlmostEqual(result['curve_uz'][:, 1].max(), 1.)
        self.assertGreaterEqual(result['branch_switch_count'], 2)
        intervals = {}
        for edge in result['path_edges']:
            if edge['source'].startswith('OBSERVED'):
                intervals.setdefault(edge['edge_id'], []).append(sorted((edge['t0'], edge['t1'])))
        for pieces in intervals.values():
            pieces.sort()
            self.assertTrue(all(a[1] <= b[0]+1e-9 for a, b in zip(pieces, pieces[1:])))

    def test_earlier_intersection_cannot_cut_away_an_accepted_fold(self):
        result, _, _ = self.solve([[[0, 0], [0, .5], [2, .8], [2, .6], [1, 1]],
                                   [[-.1, .4], [.1, .4], [.1, 2]]])
        points = result['curve_uz']
        self.assertTrue(np.any(np.all(np.isclose(points, [2, .8]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(points, [2, .6]), axis=1)))
        self.assertAlmostEqual(points[:, 1].max(), 2.)

    def test_same_branch_continuation_is_not_an_extra_branch_switch(self):
        result, _, _ = self.solve([[[-10, 0], [-10, .9], [0, .9], [0, 1]],
                                   [[5, .5], [5, 3], [.1, 1.1], [.2, 2]]])
        self.assertEqual(result['route_branch_sequence'], [0, 1])
        self.assertEqual(result['branch_switch_count'], 1)
        self.assertGreaterEqual(result['junction_count'], 2)
        self.assertAlmostEqual(result['curve_uz'][:, 1].max(), 3.)


if __name__ == '__main__':
    unittest.main()
