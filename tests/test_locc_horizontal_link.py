"""A real horizontal branch must be allowed to organize neighbor evidence."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))
from local_orthogonal_cooperative_constraint import solve_locc


class HorizontalLinkTest(unittest.TestCase):
    def test_same_horizontal_branch_conditions_neighbor_fit(self):
        linked = [{'s': -.05, 'values': [.1]}, {'s': .05, 'values': [-.1]}]
        window = {'s': 0., 'lower_uz': [0., 0.], 'upper_uz': [0., 1.],
                  'layers': [{'z': z, 'horizontal': [{'u': 0., 'face_id': 7,
                              'branch_id': 2, 'linked_neighbors': linked}],
                              'neighbors': [{'s': -.05, 'values': [10.]}, {'s': .05, 'values': [10.]}]}
                             for z in (.25, .5, .75)]}
        result = solve_locc(window)
        self.assertAlmostEqual(result['selected_nodes'][1]['neighbor']['u'], 0.)
        self.assertEqual(result['metrics']['horizontal_linked_neighbor_fraction'], 1.)
        self.assertIn('linked_neighbors', window['layers'][0]['horizontal'][0])


if __name__ == '__main__':
    unittest.main()
