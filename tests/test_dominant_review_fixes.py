"""New, targeted fixtures for independent review findings; no bulk rerun."""
import copy
import unittest
import numpy as np
from test_dominant_observed_branch import fixture, window
from dominant_observed_branch import solve_dominant_branch, select_dominant_branch
from backtracking_junction import search_backtracking_junction, build_switch_route
from test_dominant_recognition_adapter import load_recognition_kernel


class ReviewFixTests(unittest.TestCase):
    def test_complete_alternative_precedes_folded_extrema_only_coverage(self):
        profile, uz, branches = fixture([[[.02, .2], [.025, 0], [.03, 1], [.02, .8]],
                                         np.column_stack((np.zeros(21), np.linspace(0, 1, 21)))])
        self.assertEqual(select_dominant_branch(branches, window())['selected']['branch_id'], 0)
        result = solve_dominant_branch(profile, uz, window())
        self.assertEqual(result['status'], 'PRESERVED_COMPLETE_OBSERVED')
        self.assertEqual(result['selection']['selected']['branch_id'], 1)
        self.assertEqual(result['branch_switch_count'], 0)

    def test_route_cannot_omit_its_reported_dominant_identity(self):
        profile, uz, _ = fixture([[[.04, .05], [.04, .95]], [[0, 0], [0, .55]],
                                  [[.005, .45], [.005, 1]]])
        result = solve_dominant_branch(profile, uz, window())
        self.assertEqual(result['status'], 'PRESERVED_PARTIAL_UNRESOLVED')
        self.assertEqual(result['selection']['selected']['branch_id'], 0)
        self.assertEqual(result['branch_switch_count'], 0)

    def test_right_hand_dominant_is_labelled_by_identity(self):
        _, _, branches = fixture([[[0, 0], [.04, .7]], [[.04, .3], [0, 1]]])
        j = search_backtracking_junction(*branches, backtrack_length=.5)[0]
        result = build_switch_route(*branches, j, dominant_branch_id=1)
        for edge in result['path_edges']:
            if edge['source'].startswith('OBSERVED'):
                self.assertEqual(edge['source'], 'OBSERVED_DOMINANT' if edge['branch_id'] == 1 else 'OBSERVED_SECONDARY')

    def test_empty_slice_returns_empty_closed_sidecar(self):
        slicer = load_recognition_kernel().__globals__['ArcSlicer']
        res = dict(slice_key='0.00', plane_params=dict(origin=np.zeros(3), radial_dir=np.array([1., 0., 0.]),
                    vertical_dir=np.array([0., 0., 1.])), slicing=dict(lines_3d=np.empty((0, 2, 3)), face_ids=[]))
        result = slicer(copy.deepcopy(res), preserve_closed_components=True).run_all()
        self.assertEqual(result['closed_components'], [])


if __name__ == '__main__':
    unittest.main()
