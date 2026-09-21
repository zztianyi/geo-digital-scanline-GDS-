"""Continuations must keep accepted cores and try identity-ranked alternatives."""
import unittest
import numpy as np
from test_dominant_observed_branch import fixture, dense_path
from test_absolute_sweet_zone import line, supported_graph
from dominant_observed_branch import route_result
from branch_absolute_core import branch_metrics, local_edge_scale, contribution_metrics
from main_track_assembly import assemble_main_track


class ContinuationFeasibilityTests(unittest.TestCase):
    def solve(self, paths):
        branches = fixture(paths)[2]
        route = assemble_main_track(branches, route_result(branches[0]['records']),
            anchor_branch_id=0, graph=supported_graph(branches), target_s=0.)
        return branches, route

    def test_failed_identity_leader_does_not_hide_legal_alternative(self):
        _, route = self.solve([line(0, 0, 1), line(.3, .3, 3, 101), line(.003, .9, 2)])
        self.assertEqual(route['route_branch_sequence'], [0, 2])
        self.assertAlmostEqual(route['route_z_extent'][1], 2.)
        self.assertTrue(any(r['branch_id'] == 1 and r['reason'] == 'LONG_GAP_UNRESOLVED'
                            for r in route['continuation_rejections']))

    def test_identity_order_is_not_replaced_by_smallest_connector(self):
        _, route = self.solve([line(0, 0, 1), line(.009, .3, 3, 101), line(.001, .9, 2)])
        self.assertEqual(route['route_branch_sequence'], [0, 1])
        self.assertAlmostEqual(route['junctions'][0]['xyz_distance_m'], .009)

    def test_fold_in_endpoint_guard_can_be_replaced_without_cutting_core(self):
        anchor = np.vstack([line(0, 0, .8, 21), [[0, .85], [.2, .95], [.25, .90], [.4, 1.]]])
        branches, route = self.solve([anchor, dense_path([[-.1, .83], [.1, .83], [.1, 2.]])])
        self.assertAlmostEqual(route['route_z_extent'][1], 2.)
        m = branch_metrics(branches[0], local_edge_scale(branches))
        kept = contribution_metrics(branches[0], route['path_edges'], m)
        self.assertAlmostEqual(kept['retained_ASC_arc_length'], m['ASC_arc_length'])
        self.assertLessEqual(route['junctions'][0]['xyz_distance_m'], .010)

    def test_accepted_static_core_fold_is_not_cut_for_an_earlier_crossing(self):
        branches, route = self.solve([dense_path([[0, 0], [0, .5], [2, .8], [2, .6], [1, 1]]),
                                    dense_path([[-.1, .4], [.1, .4], [.1, 2]])])
        m = branch_metrics(branches[0], local_edge_scale(branches))
        kept = contribution_metrics(branches[0], route['path_edges'], m)
        self.assertAlmostEqual(kept['retained_ASC_arc_length'], m['ASC_arc_length'])
        self.assertEqual(route['route_branch_sequence'], [0])


if __name__ == '__main__':
    unittest.main()
