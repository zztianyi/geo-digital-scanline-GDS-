"""One synthetic boundary render; no model, recovery or GUI execution."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/06_visualization'))
import locc_local_review_plot as plotting
from matplotlib.figure import Figure
from PIL import Image


class LocalReviewPlotTests(unittest.TestCase):
    def test_empty_inference_preserves_branches_xyz_and_single_item_montage(self):
        # Catches fabricated repairs, dropped competing branches, and mislabeled
        # s/u/z treated as XYZ. The save spy still performs the real Agg render.
        target = np.array([[[1., 10.], [1.2, 11.]],
                           [[1.4, 12.], [1.6, 13.]],
                           [[2.5, 10.], [2.5, 13.]]])
        neighbors = []
        for s in (4., 4.5, 5., 5.5, 6.):
            uz = target + [0.1 * (s - 5.), 0.]
            xyz = np.stack((100. + uz[:, :, 0], np.full((3, 2), 200. + s),
                            uz[:, :, 1]), axis=-1)
            neighbors.append({'s': s, 'lines_uz': uz, 'lines_xyz': xyz})
        horizontal = []
        for z, selected in ((11.25, 1.25), (11.5, 1.3),
                            (11.75, None), (10.5, None)):
            su = np.array([[[4., 1.2], [6., 1.4]],
                           [[4., 2.5], [6., 2.5]]])
            xyz = np.stack((100. + su[:, :, 1], 200. + su[:, :, 0],
                            np.full((2, 2), z)), axis=-1)
            horizontal.append({'z': z, 'selected_u': selected,
                               'lines_su': su.tolist(), 'lines_xyz': xyz})
        case = {
            'candidate_id': 'synthetic-empty', 's': 5.,
            'lower_uz': [1.2, 11.], 'upper_uz': np.array([1.4, 12.]),
            'inferred_curve': [], 'inferred_xyz': np.empty((0, 3)),
            'target_lines_uz': target, 'neighbor_profiles': neighbors,
            'horizontal_profiles': horizontal,
            'mesh_vertices': [[101., 204., 10.], [102.5, 204., 13.],
                              [102.5, 206., 13.], [101., 206., 10.]],
            'mesh_faces': [[0, 1, 2], [0, 2, 3]],
            'red_groups': [[[1., 10.], [1.2, 11.]]],
            'title_note': 'Synthetic only | empty inference / no truth',
            'bounds': {'u': [0.5, 3.], 'z': [9.5, 13.5], 's': [3.5, 6.5]},
        }
        before = deepcopy(case)
        loaded_before = set(sys.modules)
        output = Path(tempfile.mkdtemp(prefix='locc-plot-smoke-'))
        saved = []
        original_save = Figure.savefig

        def save_and_capture(fig, *args, **kwargs):
            saved.append(fig)
            return original_save(fig, *args, **kwargs)

        with patch.object(Figure, 'savefig', new=save_and_capture):
            result = plotting.plot_local_review(case, output / 'local.png', dpi=100)
            montage = plotting.plot_montage([
                {'candidate_id': case['candidate_id'], 'path': result['path'],
                 'selection': 'worst', 'label': 'Synthetic boundary case'},
            ], output / 'montage.png', 'One supplied example')

        self.assertEqual(result['path'], str(output / 'local.png'))
        self.assertTrue(result['modules_reused'])
        self.assertEqual(montage, str(output / 'montage.png'))
        np.testing.assert_equal(case, before)
        local, grid = saved
        self.assertEqual(len(local.axes), 4)
        a, b, c, d = local.axes
        # All horizontal segments (including u=2.5 competitors) reach panel C.
        plotted = [np.asarray(segment) for collection in c.collections
                   if hasattr(collection, 'get_segments')
                   for segment in collection.get_segments()]
        self.assertEqual(len(plotted), 8)
        self.assertEqual(sum(np.all(segment[:, 1] == 2.5) for segment in plotted), 4)
        points = [np.asarray(collection.get_offsets()) for collection in c.collections
                  if not hasattr(collection, 'get_segments')]
        np.testing.assert_allclose(np.concatenate(points), [[5., 1.25], [5., 1.3]])
        # Empty inference and absent truth may not produce a curve artist.
        for ax in (a, b, d):
            labels = [line.get_label().lower() for line in ax.lines]
            self.assertFalse(any('infer' in label or 'truth' in label for label in labels))
        for limits, extent in ((d.get_xlim3d(), (101., 102.5)),
                               (d.get_ylim3d(), (204., 206.)),
                               (d.get_zlim3d(), (10., 13.))):
            self.assertLessEqual(limits[0], extent[0])
            self.assertGreaterEqual(limits[1], extent[1])
        self.assertTrue(d.get_xlabel().startswith('X'))
        self.assertTrue(d.get_ylabel().startswith('Y'))
        self.assertTrue(d.get_zlabel().startswith('Z'))
        self.assertEqual(sum(len(ax.images) for ax in grid.axes), 1)
        caption = '\n'.join(grid.axes[0].get_title(loc=position)
                            for position in ('left', 'center', 'right'))
        self.assertIn('synthetic-empty', caption)
        self.assertIn('worst', caption)
        for path in (result['path'], montage):
            with Image.open(path) as image:
                self.assertGreater(min(image.size), 400)
                image.verify()
        forbidden = {'pyvista', 'open3d', 'trimesh', 'generate_profile_normals',
                     'orthogonal_profile_constraint', 'orthogonal_audit_report'}
        self.assertFalse(forbidden.intersection(set(sys.modules) - loaded_before))
        self.assertFalse(plotting.plt.get_fignums())
        print(f'Synthetic render artifacts retained: {output}', flush=True)


if __name__ == '__main__':
    unittest.main()
