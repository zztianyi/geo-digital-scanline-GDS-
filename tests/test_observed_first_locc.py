"""Behavior fixtures for hard observed-first ownership, not score tuning."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))
from vertical_profile_canonicalization import canonicalize_vertical
from vertical_topology_reconstruction import analyze_vertical_candidate, reconstruct_observed_first, joint_endpoint_pairing
from local_surface_branch_tracking import build_neighbor_surface_tracks, apply_neighbor_surface_tracks


def graph(segments):
    return canonicalize_vertical(np.array([[[u, 0., z] for u, z in edge] for edge in segments]), list(range(len(segments))))


def window():
    return {'s': 0., 'lower_uz': [0., 0.], 'upper_uz': [0., 1.], 'lower_tangent': 0., 'upper_tangent': 0.,
            'layers': [{'z': z, 'horizontal': [{'u': .2, 'face_id': 99, 'branch_id': 0}],
                        'neighbors': [{'s': s, 'values': [.2]} for s in (-.1, -.05, .05, .1)]}
                       for z in (.25, .5, .75)]}


class ObservedFirstTests(unittest.TestCase):
    def test_reverse_observed_route_nodes_match_oriented_coordinates(self):
        profile = graph([[(0., 1.), (.3, .4)], [(.3, .4), (0., 0.)]])
        result = reconstruct_observed_first(profile, profile.nodes[:, [0, 2]], window())
        for edge in result['path_edges']:
            np.testing.assert_array_equal(edge['points_uz'], profile.nodes[list(edge['nodes'])][:, [0, 2]])

    def test_connected_observed_path_wins_even_when_h_disagrees(self):
        profile = graph([[(0., 0.), (.7, .4)], [(.7, .4), (0., 1.)]])
        result = reconstruct_observed_first(profile, profile.nodes[:, [0, 2]], window())
        self.assertEqual(result['gap_type'], 'TYPE_I')
        self.assertEqual(result['inferred_edges'], [])
        self.assertEqual(set(result['observed_edge_ids']), {0, 1})
        self.assertEqual(result['overwritten_observed_geometry'], 0)

    def test_partial_missing_intervals_do_not_cover_observed_edges(self):
        profile = graph([[(0., 0.), (0., .4)], [(0., .6), (0., 1.)]])
        data = window()
        for layer in data['layers']: layer['horizontal'][0]['u'] = 0.
        result = reconstruct_observed_first(profile, profile.nodes[:, [0, 2]], data)
        self.assertEqual(result['gap_type'], 'TYPE_II')
        np.testing.assert_allclose(result['missing_intervals'], [[.4, .6]])
        self.assertTrue(result['inferred_edges'])
        for edge in result['inferred_edges']:
            self.assertIsNone(edge['face_id'])
            self.assertGreaterEqual(min(p[1] for p in edge['points_uz']), .4-1e-8)
            self.assertLessEqual(max(p[1] for p in edge['points_uz']), .6+1e-8)
        self.assertEqual(set(result['observed_edge_ids']), {0, 1})

    def test_unrelated_radial_surface_is_not_target_coverage(self):
        profile = graph([[(9., 0.), (9., 1.)]])
        result = analyze_vertical_candidate(profile, profile.nodes[:, [0, 2]], window())
        self.assertEqual(result['gap_type'], 'TYPE_III')
        self.assertEqual(result['coverage_ratio'], 0.)

    def test_full_missing_keeps_real_h_node_faces_but_not_edge_faces(self):
        profile = canonicalize_vertical(np.empty((0, 2, 3)), [])
        result = reconstruct_observed_first(profile, np.empty((0, 2)), window())
        self.assertEqual(result['gap_type'], 'TYPE_III')
        self.assertTrue(result['inferred_edges'])
        self.assertTrue(all(e['face_id'] is None for e in result['inferred_edges']))
        self.assertIn(99, [n['face_id'] for n in result['inferred_nodes']])

    def test_pairing_is_joint_one_to_one_and_allows_unmatched(self):
        rows = [{'candidate_id': 'a', 'lower_key': 'l1', 'upper_key': 'u1', 'pair_score': .9},
                {'candidate_id': 'b', 'lower_key': 'l1', 'upper_key': 'u2', 'pair_score': .8},
                {'candidate_id': 'c', 'lower_key': 'l2', 'upper_key': 'u1', 'pair_score': .85},
                {'candidate_id': 'd', 'lower_key': 'l3', 'upper_key': 'u3', 'pair_score': .1}]
        result = joint_endpoint_pairing(rows, minimum_score=.5, ambiguity_margin=0.)
        self.assertEqual({r['candidate_id'] for r in result if r['selected']}, {'b', 'c'})
        self.assertFalse(next(r for r in result if r['candidate_id']=='d')['selected'])

    def test_microscopic_stitch_requires_local_independent_evidence(self):
        profile = graph([[(0., 0.), (0., .49998)], [(0., .50002), (0., 1.)]])
        data = window()
        for layer in data['layers']:
            layer['horizontal'][0]['u'] = 0.
            for observation in layer['neighbors']: observation['values'] = [0.]
        result = reconstruct_observed_first(profile, profile.nodes[:, [0, 2]], data)
        self.assertEqual(result['gap_type'], 'TYPE_I')
        self.assertEqual(len(result['topology_stitches']), 1)
        self.assertEqual(result['inferred_edges'], [])
        self.assertIsNone(result['topology_stitches'][0]['face_id'])
        for layer in data['layers']: layer['horizontal'] = []
        unsupported = reconstruct_observed_first(profile, profile.nodes[:, [0, 2]], data)
        self.assertEqual(unsupported['topology_stitches'], [])
        self.assertEqual(unsupported['inferred_edges'], [])
        self.assertEqual(unsupported['status'], 'UNRESOLVED_TOPOLOGY_OR_CORRIDOR')

    def test_neighbor_branch_identity_does_not_switch_at_each_height(self):
        levels = np.array([.25, .5, .75])
        profiles = [{'s': -.1, 'lines_uz': np.array([[[0., 0.], [0., 1.]], [[2., 0.], [2., 1.]]])}]
        tracks = build_neighbor_surface_tracks(profiles, levels, np.array([[0., 0.], [2., .5], [0., 1.]]), 0.)
        values = [v['values'][0] for layer in tracks['layers'] for v in layer['neighbors'] if v['values']]
        self.assertEqual(len(set(values)), 1)
        adapted = apply_neighbor_surface_tracks(window(), tracks)
        self.assertEqual([l['horizontal'] for l in adapted['layers']], [l['horizontal'] for l in window()['layers']])


if __name__ == '__main__':
    unittest.main()
