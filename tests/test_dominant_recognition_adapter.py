"""Opt-in closed retention cannot change the open recognition predicate path."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/99_experiments'), str(ROOT/'scripts/04_structure_recognition')]
from validate_orthogonal_scanline_constraint import load_recognition_kernel, semantically_equal


class ClosedAdapterTests(unittest.TestCase):
    def test_closed_geometry_is_saved_without_new_back_wall_or_open_group_change(self):
        slicer_type = load_recognition_kernel().__globals__['ArcSlicer']
        points = np.array([[1., 0., 0.], [2., 0., 0.], [2., 0., 1.], [1., 0., 1.], [1., 0., 0.]])
        lines = np.stack((points[:-1], points[1:]), axis=1)
        res = dict(slice_key='0.00', plane_params=dict(origin=np.zeros(3), radial_dir=np.array([1., 0., 0.]),
                    vertical_dir=np.array([0., 0., 1.])), slicing=dict(lines_3d=lines, face_ids=np.arange(4)))
        original = slicer_type(copy.deepcopy(res)).run_all()
        kept = slicer_type(copy.deepcopy(res), preserve_closed_components=True).run_all()
        self.assertEqual(len(kept['closed_components']), 1)
        self.assertAlmostEqual(kept['closed_components'][0]['area_2d'], 1.)
        self.assertAlmostEqual(kept['closed_components'][0]['perimeter'], 4.)
        self.assertEqual(kept['closed_components'][0]['source_face_ids'], [0, 1, 2, 3])
        self.assertTrue(semantically_equal(original['red_groups_corrected'], kept['red_groups_corrected']))


if __name__ == '__main__':
    unittest.main()
