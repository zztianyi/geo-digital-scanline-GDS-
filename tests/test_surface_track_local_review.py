"""Local review must expose absent geometry without changing solver output."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/99_experiments'))
from surface_track_report import local_view_bounds, clipped_segments, vertical_coverage


class LocalReviewTests(unittest.TestCase):
    def test_view_uses_case_not_full_height_branch(self):
        case = dict(lower_uz=[-2., 1400.], upper_uz=[-1.9, 1400.5])
        old = np.array([[-85., 1390.], [-2., 1400.], [-1.9, 1400.5], [0., 1480.]])
        saved = old.copy()
        bounds = local_view_bounds(case, old)
        self.assertLess(bounds[1]-bounds[0], .3)
        self.assertLess(bounds[3]-bounds[2], .7)
        np.testing.assert_array_equal(old, saved)

    def test_segment_crossing_view_is_visible_without_inside_vertices(self):
        points = np.array([[0., -5.], [0., 5.]])
        segments = clipped_segments(points, (-1., 1., -1., 1.))
        np.testing.assert_allclose(segments, [[[0., -1.], [0., 1.]]])
        self.assertEqual(vertical_coverage(segments, -1., 1.), 1.)

    def test_remote_track_has_zero_local_coverage(self):
        segments = clipped_segments(np.array([[-80., 1390.], [-80., 1490.]]), (-2., -1., 1400., 1401.))
        self.assertEqual(len(segments), 0)
        self.assertEqual(vertical_coverage(segments, 1400., 1401.), 0.)

    def test_folded_overlapping_segments_not_double_counted(self):
        segments = np.array([[[0., 0.], [0., .7]], [[0., .7], [.1, .2]], [[.1, .2], [.1, 1.]]])
        self.assertEqual(vertical_coverage(segments, 0., 1.), 1.)


if __name__ == '__main__':
    unittest.main()
