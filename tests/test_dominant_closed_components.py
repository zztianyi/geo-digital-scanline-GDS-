"""Closed-component behavior with real canonicalization; parent runs this suite."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))
from vertical_profile_canonicalization import canonicalize_vertical
from closed_component_tracking import analyze_closed_components, track_closed_components


def polygon_profile(points, *, s=0., duplicate=False):
    xyz = np.array([[u, s, z] for u, z in points], dtype=float)
    lines = np.stack((xyz, np.roll(xyz, -1, axis=0)), axis=1)
    faces = list(range(10, 10+len(lines)))
    if duplicate:
        lines = np.concatenate((lines, lines[:1, ::-1]))
        faces.append(99)
    return canonicalize_vertical(lines, faces)


def square(key, s, *, u=0., z=0., width=1., height=1., matches=()):
    profile = polygon_profile([(u, z), (u+width, z),
                               (u+width, z+height), (u, z+height)], s=s)
    record, = analyze_closed_components(profile, profile.nodes[:, [0, 2]],
                                        slice_key=key, s=s)
    record['horizontal_matches'] = list(matches)
    return record


def horizontal(level=2, z=.5, branch=7, u=0.):
    # Parent supplies these records from real horizontal cuts; identity is local
    # to a sampled level, never a face attached to a vertical canonical edge.
    return {'level_index': level, 'z': z, 'branch_id': branch, 'u': u}


class ClosedComponentAnalysisTests(unittest.TestCase):
    def test_square_retains_canonical_order_xyz_and_duplicate_provenance(self):
        profile = polygon_profile([(0, 0), (1, 0), (1, 1), (0, 1)], duplicate=True)
        nodes, edges = profile.nodes.copy(), profile.edges.copy()
        uz = profile.nodes[:, [0, 2]].copy()
        record, = analyze_closed_components(profile, uz, slice_key='V0', s=0.)
        self.assertEqual(record['kind'], 'CLOSED_COMPONENT')
        self.assertEqual(record['component_id'], 0)
        self.assertEqual(record['component_key'], 'V0:0')
        self.assertEqual(record['node_ids'], [0, 1, 2, 3, 0])
        self.assertEqual(record['edge_ids'], [0, 1, 2, 3])
        self.assertEqual(record['ordered_loop_uz'], [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.assertEqual(record['ordered_loop_xyz'],
                         [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(record['edge_source_face_ids'], [[10, 99], [11], [12], [13]])
        self.assertEqual(record['edge_source_segment_indices'], [[0, 4], [1], [2], [3]])
        self.assertEqual(set(record['source_face_ids']), {10, 11, 12, 13, 99})
        self.assertEqual(set(record['source_segment_indices']), {0, 1, 2, 3, 4})
        self.assertEqual(record['centroid_uz'], [.5, .5])
        self.assertAlmostEqual(record['area_2d'], 1.)
        self.assertAlmostEqual(record['perimeter'], 4.)
        self.assertEqual(record['u_range'], [0., 1.])
        self.assertEqual(record['z_range'], [0., 1.])
        self.assertFalse(record['degenerate'])
        self.assertTrue(record['no_artificial_closure'])
        record['ordered_loop_xyz'][0][0] = 900
        record['edge_source_face_ids'][0].append(123)
        np.testing.assert_array_equal(profile.nodes, nodes)
        np.testing.assert_array_equal(profile.edges, edges)
        np.testing.assert_array_equal(uz, nodes[:, [0, 2]])
        self.assertEqual(profile.source_face_ids[0], [10, 99])

    def test_large_origin_and_reversed_winding_preserve_area_and_centroid(self):
        origin = 1e9
        profile = polygon_profile([(origin, origin), (origin, origin+1),
                                   (origin+1, origin+1), (origin+1, origin)])
        record, = analyze_closed_components(profile, profile.nodes[:, [0, 2]],
                                            slice_key='far', s=0.)
        self.assertAlmostEqual(record['area_2d'], 1.)
        self.assertEqual(record['centroid_uz'], [origin+.5, origin+.5])

    def test_collapsed_selfloop_retained_but_never_linked(self):
        profile = canonicalize_vertical(np.array([[[0., 0., 0.], [4e-7, 0., 0.]]]), [42])
        records = []
        for key, s in [('V0', 0.), ('V1', .05)]:
            record, = analyze_closed_components(profile, profile.nodes[:, [0, 2]],
                                                slice_key=key, s=s)
            self.assertTrue(record['degenerate'])
            self.assertEqual(record['degeneracy_reason'], 'fewer_than_three_distinct_points')
            self.assertEqual(record['ordered_loop_uz'], [[0., 0.], [0., 0.]])
            self.assertEqual(record['source_face_ids'], [42])
            self.assertEqual(record['area_2d'], 0.)
            self.assertEqual(record['perimeter'], 0.)
            record['horizontal_matches'] = [horizontal()]
            records.append(record)
        result = track_closed_components(records)
        self.assertFalse(result['pairs'][0]['matched'])
        self.assertEqual(result['pairs'][0]['reason'], 'degenerate')
        self.assertEqual(result['summary']['nondegenerate_loops'], 0)

    def test_loop_attached_to_fork_is_not_a_closed_component(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [-1, 0, 0]])
        profile = canonicalize_vertical(points[[[0, 1], [1, 2], [2, 0], [0, 3]]], [1]*4)
        self.assertEqual(analyze_closed_components(profile, profile.nodes[:, [0, 2]],
                                                  slice_key='fork', s=0.), [])

    def test_empty_profile_and_collinear_cycle(self):
        profile = canonicalize_vertical(np.empty((0, 2, 3)), [])
        self.assertEqual(analyze_closed_components(profile, np.empty((0, 2)),
                                                  slice_key='empty', s=0.), [])
        profile = polygon_profile([(0, 0), (1, 0), (2, 0)])
        record, = analyze_closed_components(profile, profile.nodes[:, [0, 2]],
                                            slice_key='line', s=0.)
        self.assertTrue(record['degenerate'])
        self.assertEqual(record['degeneracy_reason'], 'zero_area')


class ClosedComponentTrackingTests(unittest.TestCase):
    def test_adjacent_supported_loops_form_stable_track_without_mutation(self):
        records = [square(f'V{i}', i*.05, u=i*.01, matches=[horizontal()])
                   for i in range(3)]
        before = deepcopy(records)
        result = track_closed_components(records)
        self.assertEqual(records, before)
        self.assertEqual(len(result['pairs']), 2)
        self.assertTrue(all(pair['matched'] and not pair['ambiguous'] for pair in result['pairs']))
        self.assertEqual([pair['horizontal_support_count'] for pair in result['pairs']], [1, 1])
        track, = result['tracks']
        self.assertEqual(track['component_keys'], ['V0:0', 'V1:0', 'V2:0'])
        self.assertEqual(track['slice_count'], 3)
        self.assertEqual(track['horizontal_supported_pair_count'], 2)
        self.assertEqual(track['status'], 'stable')
        self.assertEqual(result['summary']['total_loops'], 3)
        self.assertEqual(result['summary']['linked_tracks'], 1)
        self.assertEqual(result['summary']['stable_tracks'], 1)

    def test_nonadjacent_scanlines_are_reported_but_not_linked(self):
        result = track_closed_components([square('A', 0.), square('B', .1)])
        pair, = result['pairs']
        self.assertFalse(pair['matched'])
        self.assertEqual(pair['reason'], 'nonadjacent_slices')
        self.assertEqual(result['summary']['linked_tracks'], 0)

    def test_horizontal_identity_requires_same_level_and_physical_height(self):
        for other in [horizontal(level=3), horizontal(z=.75)]:
            with self.subTest(other=other):
                result = track_closed_components([
                    square('A', 0., matches=[horizontal()]),
                    square('B', .05, matches=[other]),
                ])
                pair, = result['pairs']
                self.assertTrue(pair['matched'])
                self.assertEqual(pair['horizontal_support_count'], 0)
                self.assertEqual(pair['horizontal_evidence'], 'insufficient')
                self.assertEqual(result['tracks'][0]['status'], 'geometric_only')

    def test_shared_horizontal_level_with_different_branches_rejects_link(self):
        result = track_closed_components([
            square('A', 0., matches=[horizontal()]),
            square('B', .05, matches=[horizontal(branch=8)]),
        ])
        pair, = result['pairs']
        self.assertFalse(pair['matched'])
        self.assertEqual(pair['horizontal_evidence'], 'contradictory')
        self.assertEqual(pair['reason'], 'horizontal_contradiction')

    def test_positive_level_cannot_hide_a_contradictory_level(self):
        result = track_closed_components([
            square('A', 0., matches=[horizontal(), horizontal(3, .75, 9)]),
            square('B', .05, matches=[horizontal(), horizontal(3, .75, 10)]),
        ])
        pair, = result['pairs']
        self.assertEqual(pair['horizontal_support_count'], 1)
        self.assertFalse(pair['matched'])
        self.assertEqual(pair['horizontal_evidence'], 'contradictory')

    def test_missing_horizontal_support_does_not_confirm_three_slice_track(self):
        result = track_closed_components([square(f'V{i}', i*.05) for i in range(3)])
        self.assertEqual(result['tracks'][0]['status'], 'geometric_only')
        self.assertEqual(result['summary']['stable_tracks'], 0)

    def test_duplicate_competitors_are_ambiguous_and_remain_unmatched(self):
        # Separate XYZ components can project onto the same UZ outline.
        first = polygon_profile([(0, 0), (1, 0), (1, 1), (0, 1)], s=.05)
        second = first.lines_xyz.copy()
        second[:, :, 1] += .001
        profile = canonicalize_vertical(np.concatenate((first.lines_xyz, second)), list(range(8)))
        competitors = analyze_closed_components(profile, profile.nodes[:, [0, 2]],
                                                slice_key='B', s=.05)
        result = track_closed_components([square('A', 0.)] + competitors)
        self.assertEqual(len(result['pairs']), 2)
        self.assertTrue(all(pair['ambiguous'] for pair in result['pairs']))
        self.assertTrue(all(not pair['matched'] for pair in result['pairs']))
        self.assertEqual(result['summary']['linked_tracks'], 0)
        self.assertEqual(len(result['tracks']), 3)

    def test_one_to_one_keeps_rejected_competitor_and_unmatched_loop(self):
        result = track_closed_components([square('A', 0.), square('B', .05),
                                          square('C', .05, u=.14)], ambiguity_margin=.01)
        self.assertEqual(sum(pair['matched'] for pair in result['pairs']), 1)
        self.assertEqual(len(result['pairs']), 2)
        self.assertEqual(result['summary']['unmatched_loops'], 1)

    def test_shape_area_perimeter_height_and_centroid_rejections(self):
        cases = [
            (square('B', .05, u=.3), 'centroid_distance'),
            (square('B', .05, u=.45, z=.45, width=.1, height=.1), 'area_ratio'),
            (square('B', .05, u=-1.5, z=.4, width=4., height=.2), 'perimeter_ratio'),
            (square('B', .05, z=.8), 'height_overlap'),
            (square('B', .05, u=-.5, z=.25, width=2., height=.5), 'shape_distance'),
        ]
        for other, reason in cases:
            with self.subTest(reason=reason):
                # A relaxed centroid gate isolates the overlap check.
                distance = 1. if reason == 'height_overlap' else .15
                result = track_closed_components([square('A', 0.), other],
                                                 max_centroid_distance=distance)
                self.assertFalse(result['pairs'][0]['matched'])
                self.assertEqual(result['pairs'][0]['reason'], reason)

    def test_empty_input(self):
        result = track_closed_components([])
        self.assertEqual(result['pairs'], [])
        self.assertEqual(result['tracks'], [])
        self.assertEqual(result['summary']['total_loops'], 0)


if __name__ == '__main__':
    unittest.main()
