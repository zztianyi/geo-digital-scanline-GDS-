"""Focused canonical geometry checks; runnable as a plain unittest script."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))
from vertical_profile_canonicalization import (
    CanonicalProfile, canonicalize_vertical, endpoint_gap_diagnostics,
)


class VerticalCanonicalizationTests(unittest.TestCase):
    def test_duplicate_provenance_degree_and_legacy_first_match_order(self):
        a, b, c = (11., 2., .4), (11., 2., .1), (11., 2., .8)
        lines = np.array([[a, b], [b, c], [b, a], [a, b], [a, b], [c, b]])
        faces = [100, None, 203, 100, None, 300]
        profile = canonicalize_vertical(lines, faces)
        self.assertIsInstance(profile, CanonicalProfile)
        np.testing.assert_array_equal(profile.nodes, [a, b, c])
        np.testing.assert_array_equal(profile.edges, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(profile.lines_xyz, [[a, b], [b, c]])
        self.assertEqual(profile.as_intersections(), [(a, b, 100), (b, c, None)])
        self.assertEqual(profile.first_face_ids, [100, None])
        self.assertEqual(profile.source_face_ids, [[100, 203], [300]])
        self.assertEqual(profile.source_segment_indices, [[0, 2, 3, 4], [1, 5]])
        np.testing.assert_array_equal(profile.raw_degree, [4, 6, 2])
        np.testing.assert_array_equal(profile.degree, [1, 2, 1])
        np.testing.assert_array_equal(profile.components, [0, 0, 0])
        self.assertEqual(profile.stats, {
            'raw_vertical_segments': 6, 'canonical_vertical_edges': 2,
            'exact_duplicate_vertical_edges': 4, 'duplicate_source_face_count': 1,
            'degree_changed_nodes': 3, 'collapsed_self_edges': 0,
            'component_count': 1, 'endpoint_count': 2,
        })

    def test_repeated_same_face_edge_still_has_degree_one_endpoints(self):
        lines = np.array([[[0, 0, 0], [0, 0, 1]]] * 2, dtype=float)
        profile = canonicalize_vertical(lines, [7, 7])
        np.testing.assert_array_equal(profile.degree, [1, 1])
        np.testing.assert_array_equal(profile.raw_degree, [2, 2])
        self.assertEqual(profile.source_face_ids, [[7]])
        self.assertEqual(profile.source_segment_indices, [[0, 1]])
        self.assertEqual(profile.stats['duplicate_source_face_count'], 0)

    def test_round_six_merges_only_equal_rounded_geometry(self):
        lines = np.array([
            [[11, 2, 0], [11, 2, 1]],
            [[11.0000004, 2, 0], [11.0000004, 2, 1]],
            [[11.0000012, 2, 0], [11.0000012, 2, 1]],
        ])
        profile = canonicalize_vertical(lines, [1, 2, 3])
        np.testing.assert_array_equal(profile.nodes, [
            [11, 2, 0], [11, 2, 1], [11.000001, 2, 0], [11.000001, 2, 1],
        ])
        self.assertEqual(profile.source_face_ids, [[1, 2], [3]])
        self.assertEqual(profile.stats['component_count'], 2)
        np.testing.assert_array_equal(profile.degree, [1, 1, 1, 1])

    def test_input_geometry_and_faces_are_not_mutated_or_aliased(self):
        lines = np.array([[[2.0000004, 0, 0], [3, 0, 1]]])
        original = lines.copy()
        lines.setflags(write=False)
        faces = np.array([17])
        profile = canonicalize_vertical(lines, faces)
        np.testing.assert_array_equal(lines, original)
        np.testing.assert_array_equal(faces, [17])
        self.assertFalse(np.shares_memory(profile.nodes, lines))
        profile.nodes[0, 0] = 99
        profile.first_face_ids[0] = 88
        np.testing.assert_array_equal(lines, original)
        np.testing.assert_array_equal(faces, [17])

    def test_unique_collapsed_self_edges_retain_provenance_and_two_incidences(self):
        lines = np.array([
            [[0, 0, 0], [.0000004, 0, 0]],
            [[0, 0, 0], [0, 0, 1]],
            [[5, 0, 0], [5, 0, 0]],
        ])
        profile = canonicalize_vertical(lines, [1, 2, 3])
        np.testing.assert_array_equal(profile.edges, [[0, 0], [0, 1], [2, 2]])
        np.testing.assert_array_equal(profile.raw_degree, [3, 1, 2])
        np.testing.assert_array_equal(profile.degree, [3, 1, 2])
        np.testing.assert_array_equal(profile.components, [0, 0, 1])
        self.assertEqual(profile.first_face_ids, [1, 2, 3])
        self.assertEqual(profile.source_face_ids, [[1], [2], [3]])
        self.assertEqual(profile.source_segment_indices, [[0], [1], [2]])
        self.assertEqual(profile.as_intersections(), [
            ((0., 0., 0.), (0., 0., 0.), 1),
            ((0., 0., 0.), (0., 0., 1.), 2),
            ((5., 0., 0.), (5., 0., 0.), 3),
        ])
        self.assertEqual(profile.branches, [
            {'nodes': [0, 0], 'edges': [0], 'branch_id': 0, 'component_id': 0},
            {'nodes': [0, 1], 'edges': [1], 'branch_id': 1, 'component_id': 0},
            {'nodes': [2, 2], 'edges': [2], 'branch_id': 2, 'component_id': 1},
        ])
        self.assertEqual(profile.stats['collapsed_self_edges'], 2)
        self.assertEqual(profile.stats['canonical_vertical_edges'], 3)
        self.assertEqual(profile.stats['exact_duplicate_vertical_edges'], 0)
        self.assertEqual(profile.stats['degree_changed_nodes'], 0)
        self.assertEqual(profile.stats['endpoint_count'], 1)
        self.assertEqual(profile.stats['component_count'], 2)
        self.assertEqual(endpoint_gap_diagnostics(profile), [])

    def test_duplicate_self_edges_preserve_all_sources_and_count_only_repeats(self):
        lines = np.array([
            [[0, 0, 0], [.0000004, 0, 0]],
            [[.0000004, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [.0000003, 0, 0]],
        ])
        profile = canonicalize_vertical(lines, [None, 7, 7, 9])
        np.testing.assert_array_equal(profile.edges, [[0, 0]])
        np.testing.assert_array_equal(profile.lines_xyz, np.zeros((1, 2, 3)))
        np.testing.assert_array_equal(profile.raw_degree, [8])
        np.testing.assert_array_equal(profile.degree, [2])
        self.assertEqual(profile.first_face_ids, [None])
        self.assertEqual(profile.source_face_ids, [[7, 9]])
        self.assertEqual(profile.source_segment_indices, [[0, 1, 2, 3]])
        self.assertEqual(profile.branches, [
            {'nodes': [0, 0], 'edges': [0], 'branch_id': 0, 'component_id': 0},
        ])
        self.assertEqual(profile.stats, {
            'raw_vertical_segments': 4, 'canonical_vertical_edges': 1,
            'exact_duplicate_vertical_edges': 3, 'duplicate_source_face_count': 1,
            'degree_changed_nodes': 1, 'collapsed_self_edges': 4,
            'component_count': 1, 'endpoint_count': 0,
        })
        self.assertEqual(endpoint_gap_diagnostics(profile), [])

    def test_branches_cover_forks_chains_and_closed_loops_once(self):
        points = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 2, 0],
            [10, 0, 0], [11, 0, 0], [10, 1, 0],
        ])
        lines = points[[[0, 1], [1, 2], [1, 3], [3, 4], [5, 6], [6, 7], [7, 5]]]
        profile = canonicalize_vertical(lines, list(range(7)))
        self.assertEqual(profile.branches, [
            {'nodes': [0, 1], 'edges': [0], 'branch_id': 0, 'component_id': 0},
            {'nodes': [1, 2], 'edges': [1], 'branch_id': 1, 'component_id': 0},
            {'nodes': [1, 3, 4], 'edges': [2, 3], 'branch_id': 2, 'component_id': 0},
            {'nodes': [5, 6, 7, 5], 'edges': [4, 5, 6], 'branch_id': 3, 'component_id': 1},
        ])
        np.testing.assert_array_equal(profile.components, [0, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(profile.stats['endpoint_count'], 3)

    def test_loop_attached_to_a_fork_is_a_complete_branch(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]])
        profile = canonicalize_vertical(points[[[0, 1], [1, 2], [2, 0], [0, 3]]], [1]*4)
        self.assertEqual(profile.branches, [
            {'nodes': [0, 1, 2, 0], 'edges': [0, 1, 2], 'branch_id': 0, 'component_id': 0},
            {'nodes': [0, 3], 'edges': [3], 'branch_id': 1, 'component_id': 0},
        ])

    def test_empty_and_all_collapsed_profiles(self):
        profile = canonicalize_vertical(np.empty((0, 2, 3)), [])
        self.assertEqual(profile.nodes.shape, (0, 3))
        self.assertEqual(profile.edges.shape, (0, 2))
        self.assertEqual(profile.lines_xyz.shape, (0, 2, 3))
        self.assertEqual(profile.degree.shape, (0,))
        self.assertEqual(profile.raw_degree.shape, (0,))
        self.assertEqual(profile.components.shape, (0,))
        self.assertEqual(profile.as_intersections(), [])
        self.assertEqual(profile.branches, [])
        self.assertTrue(all(value == 0 for value in profile.stats.values()))
        self.assertEqual(endpoint_gap_diagnostics(profile), [])
        collapsed = canonicalize_vertical(np.zeros((1, 2, 3)), [None])
        np.testing.assert_array_equal(collapsed.edges, [[0, 0]])
        np.testing.assert_array_equal(collapsed.degree, [2])
        self.assertEqual(collapsed.first_face_ids, [None])
        self.assertEqual(collapsed.source_face_ids, [[]])
        self.assertEqual(collapsed.source_segment_indices, [[0]])
        self.assertEqual(collapsed.branches, [
            {'nodes': [0, 0], 'edges': [0], 'branch_id': 0, 'component_id': 0},
        ])
        self.assertEqual(collapsed.stats['collapsed_self_edges'], 1)
        self.assertEqual(collapsed.stats['exact_duplicate_vertical_edges'], 0)
        self.assertEqual(collapsed.stats['component_count'], 1)

    def test_invalid_shapes_coordinates_and_face_lengths_are_rejected(self):
        for shape in [(2, 3), (1, 3, 2), (1, 2, 4), (0,), (0, 3)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'shape'):
                canonicalize_vertical(np.zeros(shape), [])
        for invalid in [np.nan, np.inf, -np.inf]:
            lines = np.zeros((1, 2, 3))
            lines[0, 1, 2] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'finite'):
                canonicalize_vertical(lines, [1])
        for faces in [[], [1, 2]]:
            with self.subTest(faces=faces), self.assertRaisesRegex(ValueError, 'face_ids.*length'):
                canonicalize_vertical(np.zeros((1, 2, 3)), faces)

    def test_nearest_foreign_component_beyond_many_same_component_endpoints(self):
        # Every leaf has over 64 nearer endpoints in its own connected star.
        leaves = np.column_stack((np.arange(80)*.001, np.ones(80), np.zeros(80)))
        star = np.stack((np.zeros_like(leaves), leaves), axis=1)
        lines = np.concatenate((star, [[[10, 1, 0], [11, 1, 0]]]))
        profile = canonicalize_vertical(lines, list(range(len(lines))))
        rows = endpoint_gap_diagnostics(profile)
        self.assertEqual(len(rows), 82)
        first = next(row for row in rows if row['node_id'] == 1)
        self.assertEqual(first['other_node_id'], 81)
        self.assertAlmostEqual(first['distance_m'], 10.)
        self.assertEqual(first['bucket'], '>=1e-2')
        self.assertEqual([row['node_id'] for row in rows], list(range(1, 83)))

    def test_nearest_foreign_component_matches_exhaustive_xyz_distances(self):
        lines = np.array([
            [[0, 0, 0], [0, 0, 1]], [[.1, 0, 0], [.1, 0, 2]],
            [[0, .2, 0], [0, .2, 3]], [[0, 0, 4], [0, 0, 5]],
            [[8, 4, 3], [9, 2, 1]],
        ])
        profile = canonicalize_vertical(lines, [1]*5)
        rows = endpoint_gap_diagnostics(profile)
        self.assertEqual(len(rows), 10)
        for row in rows:
            node = row['node_id']
            candidates = np.flatnonzero((profile.degree == 1) &
                                        (profile.components != profile.components[node]))
            distances = np.linalg.norm(profile.nodes[candidates]-profile.nodes[node], axis=1)
            self.assertAlmostEqual(row['distance_m'], float(distances.min()))
            self.assertIn(row['other_node_id'], candidates)
            self.assertAlmostEqual(row['distance_m'], float(np.linalg.norm(
                profile.nodes[node]-profile.nodes[row['other_node_id']])))

    def test_gap_bucket_boundaries_and_single_component_has_no_rows(self):
        for gap, bucket in [(1e-6, '1e-6-1e-4'), (99e-6, '1e-6-1e-4'),
                            (1e-4, '1e-4-1e-3'), (1e-3, '1e-3-1e-2'),
                            (1e-2, '>=1e-2')]:
            with self.subTest(gap=gap):
                profile = canonicalize_vertical(np.array([
                    [[0, 0, 0], [-1, 0, 0]], [[gap, 0, 0], [1, 0, 0]],
                ]), [1, 2])
                row = endpoint_gap_diagnostics(profile)[0]
                self.assertEqual(row['other_node_id'], 2)
                self.assertAlmostEqual(row['distance_m'], gap)
                self.assertEqual(row['bucket'], bucket)
        profile = canonicalize_vertical(np.array([[[0, 0, 0], [1, 0, 0]]]), [1])
        self.assertEqual(endpoint_gap_diagnostics(profile), [])
        # Sub-micrometre gaps cannot survive round6 canonicalization, but the
        # diagnostic also accepts an explicitly supplied CanonicalProfile.
        profile = canonicalize_vertical(np.array([
            [[0, 0, 0], [-1, 0, 0]], [[1e-6, 0, 0], [1, 0, 0]],
        ]), [1, 2])
        profile.nodes[2, 0] = 5e-7
        self.assertEqual(endpoint_gap_diagnostics(profile)[0]['bucket'], '<1e-6')


if __name__ == '__main__':
    unittest.main()
