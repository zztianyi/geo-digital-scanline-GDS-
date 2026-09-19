"""Surface identity regressions using measured polylines, not endpoint scores."""
import copy
import unittest
import numpy as np
from test_dominant_observed_branch import fixture, window
from dominant_observed_branch import solve_dominant_branch
from horizontal_surface_link import horizontal_path_observations
from observed_surface_graph import build_surface_graph
from surface_track_handoff import confidence_crossover, select_handoff, infer_missing_interval


def scene(paths_by_s, levels=(.2, .4, .6, .8), horizontal=None):
    profiles, inventories = {}, []
    for s, paths in paths_by_s.items():
        p, uz, branches = fixture(paths)
        profiles[s] = (p, uz, branches)
        inventories.append(dict(s=s, branches=branches))
    if horizontal is None:
        horizontal = [dict(z=z, paths=[dict(h_path_id=0, points_su=[[-1., .02], [1., .02]])]) for z in levels]
    observations = horizontal_path_observations(horizontal, sorted(paths_by_s))
    return profiles, build_surface_graph(inventories, observations, slice_order=sorted(paths_by_s))


class SurfaceTrackTests(unittest.TestCase):
    def supported_scene(self):
        detailed = [[.02, 0], [.02, .7], [.02, .5], [.02, 1]]
        return scene({-1.: [detailed], 0.: [[[0, 0], [0, 1]], detailed], 1.: [detailed]})

    def test_same_path_support_preserves_fold_over_simple_spanning_branch(self):
        profiles, graph = self.supported_scene()
        p, uz, branches = profiles[0.]
        result = solve_dominant_branch(p, uz, window(), branches=branches, surface_graph=graph, target_s=0.)
        self.assertEqual(result['selection']['selected']['branch_id'], 1)
        self.assertTrue(np.any(np.diff(result['curve_uz'][:, 1]) < 0))
        self.assertEqual(result['inferred_length_m'], 0.)
        self.assertEqual(result['branch_switch_count'], 0)

    def test_endpoint_perturbation_cannot_change_identity_or_route(self):
        profiles, graph = self.supported_scene()
        p, uz, branches = profiles[0.]
        results = []
        for shift in (0., -.005, .005, -.02, .02, -.05, .05, 5.):
            w = window((shift, shift), (shift, 1+shift))
            results.append(solve_dominant_branch(p, uz, w, branches=branches, surface_graph=graph, target_s=0.))
        self.assertEqual(len({r['route_identity'] for r in results}), 1)
        for r in results:
            np.testing.assert_array_equal(r['curve_uz'], results[0]['curve_uz'])

    def test_raw_nearby_hits_do_not_link_disconnected_horizontal_paths(self):
        horizontal = [dict(z=z, paths=[dict(h_path_id=i, points_su=[[s-.1, .02], [s+.1, .02]])
                                     for i, s in enumerate((-1., 0., 1.))]) for z in (.2, .4, .6)]
        _, graph = scene({s: [[[.02, 0], [.02, 1]]] for s in (-1., 0., 1.)}, horizontal=horizontal)
        self.assertEqual(len(graph['links']), 0)
        self.assertEqual(graph['surface_linked_H_hits'], 0)
        self.assertGreater(graph['raw_H_hits'], 0)

    def test_nearby_but_nonintersecting_h_path_has_no_support(self):
        _, graph = scene({s: [[[.021, 0], [.021, 1]]] for s in (-1., 0., 1.)})
        self.assertEqual(graph['surface_linked_H_hits'], 0)

    def test_multiple_levels_on_same_path_form_one_track(self):
        profiles, graph = scene({s: [[[.02, 0], [.02, 1]]] for s in (-1., 0., 1.)})
        self.assertEqual(len(graph['tracks']), 1)
        self.assertTrue(graph['tracks'][0]['stable'])
        self.assertEqual(len(graph['links']), 2)
        p, uz, branches = profiles[0.]
        result = solve_dominant_branch(p, uz, branches=branches, surface_graph=graph, target_s=0.)
        self.assertAlmostEqual(result['observed_high_confidence_length_preserved'], .6)

    def test_global_connectivity_cannot_merge_competing_same_height_surfaces(self):
        paths = [[[.02, 0], [.02, 1]], [[.04, 0], [.04, 1]]]
        joined = [[[.02, 0], [.02, 1], [.04, 1], [.04, 0]]]
        h = [dict(z=z, paths=[dict(h_path_id=i, points_su=[[-1., u], [2., u]])
                             for i, u in enumerate((.02, .04))]) for z in (.2, .4, .6, .8)]
        _, graph = scene({-1.: joined, 0.: paths, 1.: paths, 2.: paths}, horizontal=h)
        self.assertNotEqual(graph['membership'][(0., 0)], graph['membership'][(0., 1)])
        self.assertTrue(graph['ambiguous_links'])

    def test_missing_slice_cannot_be_bridged_by_graph_adjacency(self):
        inv = [dict(s=s, branches=fixture([[[.02, 0], [.02, 1]]])[2]) for s in (-1., 1.)]
        obs = horizontal_path_observations([dict(z=z, paths=[dict(h_path_id=0, points_su=[[-1., .02], [1., .02]])])
                                           for z in (.2, .4)], [-1., 0., 1.])
        graph = build_surface_graph(inv, obs, slice_order=[-1., 0., 1.])
        self.assertEqual(len(graph['links']), 0)

    def test_shape_similarity_without_track_link_is_not_detail_support(self):
        p, uz, b = fixture([[[0, 0], [.01, .5], [0, 1]]])
        result = solve_dominant_branch(p, uz, window(), neighbors=[dict(s=.9, branches=b)])
        self.assertEqual(result['selection']['selected']['neighbor_detail_repeat_count'], 0)

    def test_locked_track_cannot_be_replaced_by_another_stable_track(self):
        paths = [[[0, 0], [0, 1]], [[.02, 0], [.02, 1]]]
        h = [dict(z=z, paths=[dict(h_path_id=i, points_su=[[-1., u], [1., u]]) for i,u in enumerate((0., .02))])
             for z in (.2, .4, .6, .8)]
        profiles, graph = scene({s: paths for s in (-1., 0., 1.)}, horizontal=h)
        target = graph['membership'][(0., 1)]
        p, uz, branches = profiles[0.]
        result = solve_dominant_branch(p, uz, window((0, 0), (0, 1)), branches=branches,
                                     surface_graph=graph, target_s=0., surface_track_id=target)
        self.assertEqual(result['route_identity'], (target, 1))

    def test_closed_loop_stays_separate_from_open_surface(self):
        loop = [[.02, 0], [.02, 1], [.05, 1], [.05, 0], [.02, 0]]
        _, graph = scene({-1.: [[[.02, 0], [.02, 1]]], 0.: [loop], 1.: [[[.02, 0], [.02, 1]]]})
        self.assertEqual(len(graph['links']), 0)
        self.assertTrue(graph['closed_transition_evidence'])


class CrossoverTests(unittest.TestCase):
    def test_partial_tail_recovery_keeps_retained_observed_coordinates(self):
        profiles, graph = scene({-1.: [[[.02, 0], [.02, 1]]],
            0.: [[[.02, 0], [.02, .55]]], 1.: [[[.02, 0], [.02, 1]]]}, levels=np.arange(.1, 1., .1))
        p, uz, branches = profiles[0.]
        before = p.nodes.copy()
        result = solve_dominant_branch(p, uz, branches=branches, surface_graph=graph, target_s=0.)
        self.assertAlmostEqual(result['inferred_length_m'], .35)
        np.testing.assert_array_equal(p.nodes, before)
        observed = [e for e in result['path_edges'] if e['source'].startswith('OBSERVED')]
        self.assertEqual(observed[0]['points_uz'], [[.02, 0.], [.02, .55]])

    def test_parallel_edges_find_constrained_interior_junction(self):
        from backtracking_junction import search_backtracking_junction
        _, _, branches = fixture([[[0, 0], [0, 1]], [[.005, 0], [.005, 1]]])
        choices = search_backtracking_junction(*branches, transition_interval=(.4, .6))
        self.assertTrue(choices)
        self.assertTrue(all(.4-1e-9 <= j['a_point_uz'][1] <= .6+1e-9 for j in choices))
        self.assertAlmostEqual(choices[0]['distance_m'], .005)

    def test_tail_inference_cannot_cover_unlinked_observed_fragment(self):
        profiles, graph = scene({-1.: [[[.02, 0], [.02, 1]]],
            0.: [[[.02, .4], [.02, .6]], [[.02, .72], [.02, .78]]],
            1.: [[[.02, 0], [.02, 1]]]}, levels=np.arange(.1, 1., .1))
        p, uz, branches = profiles[0.]
        result = solve_dominant_branch(p, uz, branches=branches, surface_graph=graph, target_s=0.)
        self.assertFalse(any(max(np.asarray(e['points_uz'])[:, 1]) > .7 for e in result['path_edges']
                             if e['source'].endswith('INFERRED')))

    def test_ambiguous_neighbor_crossing_cannot_support_missing_tail(self):
        paths = [[[.02, 0], [.02, 1]], [[.020002, .65], [.020002, 1]]]
        profiles, graph = scene({-1.: paths, 0.: [[[.02, .4], [.02, .6]]], 1.: paths},
                                 levels=np.arange(.1, 1., .1))
        p, uz, branches = profiles[0.]
        result = solve_dominant_branch(p, uz, branches=branches, surface_graph=graph, target_s=0.)
        self.assertFalse(any(max(np.asarray(e['points_uz'])[:, 1]) > .6+1e-9 for e in result['path_edges']
                             if e['source'].endswith('INFERRED')))

    def test_crossover_junction_is_interior_and_enters_final_route(self):
        p, uz, branches = fixture([[[0, 0], [.04, .8]], [[.04, .2], [0, 1]]])
        # Same-track neighboring geometry carries A below, B above the crossing.
        neighbor = [[0, 0], [.025, .5], [0, 1]]
        profiles = [dict(s=-1., branches=fixture([neighbor])[2]), dict(s=0., branches=branches),
                    dict(s=1., branches=fixture([neighbor])[2])]
        horizontal = []
        for z in (.1, .2, .3, .4, .6, .7, .8, .9):
            u = .05*z if z < .5 else .05*(1-z)
            horizontal.append(dict(z=z, paths=[dict(h_path_id=0, points_su=[[-1., u], [1., u]])]))
        graph = build_surface_graph(profiles, horizontal_path_observations(horizontal, [-1., 0., 1.]),
                                    slice_order=[-1., 0., 1.])
        result = solve_dominant_branch(p, uz, window(), branches=branches, surface_graph=graph, target_s=0.)
        self.assertEqual(result['branch_switch_count'], 1)
        self.assertEqual(result['junction']['junction_type'], 'REAL_INTERSECTION')
        self.assertAlmostEqual(result['junction']['a_point_uz'][1], .5)
        ids = [e['branch_id'] for e in result['path_edges'] if e['source'].startswith('OBSERVED')]
        self.assertEqual(len(list(dict.fromkeys(ids))), 2)
        self.assertEqual(result['inferred_length_m'], 0.)

    def test_confidence_without_reversal_cannot_switch(self):
        self.assertIsNone(confidence_crossover([0, .2, .4, .6, .8, 1], [1]*6, [.5]*6))

    def test_missing_inference_is_identity_locked_and_never_overwrites_observed(self):
        _, _, branches = fixture([[[.02, 0], [.02, 1]]])
        result = infer_missing_interval(branches, (.2, .8), track_id='T', linked_samples=[(.2, .02), (.8, .02)])
        self.assertEqual(result['inferred_length_m'], 0.)
        missing = infer_missing_interval([], (.2, .8), track_id='T', linked_samples=[(.2, .02), (.8, .02)])
        self.assertAlmostEqual(missing['inferred_length_m'], .6)
        self.assertEqual(missing['surface_track_id'], 'T')
        self.assertEqual(infer_missing_interval([], (.2, .8), track_id=None, linked_samples=[(.2, .02), (.8, .02)])['status'],
                         'UNRESOLVED_SURFACE_IDENTITY')


if __name__ == '__main__':
    unittest.main()
