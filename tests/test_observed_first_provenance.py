"""Targeted review regressions: slice indices and exact observed XYZ anchors."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts/99_experiments'),
               str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition')]
from validate_observed_first_locc import target_geometry, edge_xyz_for_recognition, target_canonical_profile
from vertical_profile_canonicalization import canonicalize_vertical
from locc_data import opc


class ProvenanceTests(unittest.TestCase):
    def test_target_graph_preserves_original_slice_indices_after_ray_filter(self):
        arc = {'center': [0., 0.], 'radius': 10., 'angle_min': 0., 'angle_max': 1.}
        line = [[10., 0., 0.], [11., 0., 1.]]
        frozen = SimpleNamespace(arc=arc, slices={'0.00': {'slicing': {
            'lines_3d': np.array([[[-10., 0., 0.], [-11., 0., 1.]], line, line[::-1]]), 'face_ids': [9, 42, 43]}}})
        _, _, indices = target_geometry(frozen, '0.00', return_indices=True)
        np.testing.assert_array_equal(indices, [1, 2])
        profile = target_canonical_profile(frozen, '0.00')
        self.assertEqual(profile.source_segment_indices, [[1, 2]])
        self.assertEqual(profile.source_face_ids, [[42, 43]])

    def test_existing_xyz_anchors_are_not_roundtripped_through_radial_plane(self):
        arc = {'center': [0., 0.], 'radius': 10., 'angle_min': 0., 'angle_max': 1.}
        xyz = np.array([[[10.234567, 3.712345, 1000.123456], [10.345678, 3.756789, 1000.223456]]])
        profile = canonicalize_vertical(xyz, [1])
        uz = opc.to_suz(profile.nodes, arc)[:, 1:]
        edge = {'nodes': (1, 0), 'points_uz': uz[::-1].tolist()}
        actual = edge_xyz_for_recognition(edge, profile, 3.5, arc)
        np.testing.assert_array_equal(actual, profile.nodes[::-1])


if __name__ == '__main__': unittest.main()
