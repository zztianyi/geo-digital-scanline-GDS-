"""Real-case diagnosis: radial interior excursion is not absence of a branch."""
import unittest
import numpy as np
from test_dominant_observed_branch import fixture, window, dense_path
from dominant_observed_branch import solve_dominant_branch


class BoundaryAssociationTests(unittest.TestCase):
    def test_large_observed_interior_excursion_is_retained_without_smoothing(self):
        points = np.array([[0, 0], [-.4, .4], [-.7, .55], [-.6, .45], [-.4, .7], [.01, 1]])
        profile, uz, branches = fixture([dense_path(points)])
        result = solve_dominant_branch(profile, uz, window())
        self.assertEqual(result['status'], 'UNRESOLVED')
        np.testing.assert_array_equal(result['curve_uz'], branches[0]['points_uz'])
        self.assertEqual(result['inferred_length_m'], 0.)
        self.assertEqual(result['branch_switch_count'], 0)


if __name__ == '__main__':
    unittest.main()
