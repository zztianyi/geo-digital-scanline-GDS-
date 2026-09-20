"""Behavior checks: identity selection, immutable detail and bounded switching."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))
from vertical_profile_canonicalization import canonicalize_vertical
from dominant_observed_branch import (extract_observed_branches, select_dominant_branch,
                                      solve_dominant_branch, branch_reliability)
from backtracking_junction import search_backtracking_junction, build_switch_route


def fixture(paths):
    lines = np.concatenate([np.stack((p[:-1], p[1:]), axis=1) for p in map(np.asarray, paths)])
    xyz = np.zeros((len(lines), 2, 3)); xyz[:, :, [0, 2]] = lines
    profile = canonicalize_vertical(xyz, list(range(len(lines))))
    uz = profile.nodes[:, [0, 2]]
    return profile, uz, extract_observed_branches(profile, uz)


def dense_path(path):
    """Explicit synthetic canonical nodes for tests of a reliable long branch."""
    p=np.asarray(path,dtype=float)
    t=(np.arange(28)/28.)**1.1
    return np.concatenate([a+t[:,None]*(b-a) for a,b in zip(p,p[1:])]+[p[-1:]])


def window(low=(0., 0.), high=(0., 1.)):
    return dict(s=1., lower_uz=np.array(low), upper_uz=np.array(high), layers=[],
                lower_tangent=0., upper_tangent=0.)


class DominantBranchTests(unittest.TestCase):
    def test_complete_observed_branch_is_not_averaged_or_inferred(self):
        paths = [[[0, -.2], [.01, 0], [.03, .5], [.01, 1], [0, 1.2]],
                 [[.04, 0], [.045, .5], [.04, 1]]]
        paths=[dense_path(p) for p in paths]
        profile, uz, branches = fixture(paths)
        before = profile.nodes.copy()
        result = solve_dominant_branch(profile, uz, window())
        self.assertEqual(result['branch_switch_count'], 0)
        self.assertEqual(result['inferred_length_m'], 0.)
        np.testing.assert_array_equal(result['curve_uz'], branches[0]['points_uz'])
        np.testing.assert_array_equal(before, profile.nodes)
        self.assertFalse(result['original_endpoint_pair_connected'])
        self.assertEqual(result['observed_geometry_fraction'], 1.)

    def test_duplicates_do_not_increase_detail_density(self):
        profile, uz, branches = fixture([[[0, 0], [.01, .5], [0, 1]]]*3)
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]['canonical_edge_count'], 2)
        self.assertEqual(len(branches[0]['source_segment_indices']), 6)

    def test_fold_in_selected_real_path_is_not_sorted_away(self):
        profile, uz, branches = fixture([dense_path([[0, 0], [.01, .65], [.015, .45], [0, 1]])])
        result = solve_dominant_branch(profile, uz, window())
        self.assertTrue(np.any(np.diff(result['curve_uz'][:, 1]) < 0))
        self.assertEqual(result['branch_switch_count'], 0)
        self.assertEqual(result['inferred_length_m'], 0.)

    def test_repeated_detail_beats_equal_density_unsupported_detail(self):
        zs = np.linspace(-.2, 1.2, 57)
        p = np.column_stack((.002*np.sin(zs*25), zs))
        q = np.column_stack((.02+.002*np.cos(zs*25), zs))
        profile, uz, branches = fixture([p, q])
        neighbors = []
        for s in (.90, .95, 1.05, 1.10):
            n = p.copy(); n[:, 0] += (s-1)*.01
            _, _, nb = fixture([n]); neighbors.append({'s': s, 'branches': nb})
        from horizontal_surface_link import horizontal_path_observations
        from observed_surface_graph import build_surface_graph
        positions = [.90, .95, 1., 1.05, 1.10]
        horizontal = [dict(z=float(z), paths=[dict(h_path_id=0,
                      points_su=[[s, float(np.interp(z, p[:, 1], p[:, 0])+(s-1)*.01)] for s in positions])])
                      for z in zs[1:-1]]
        graph = build_surface_graph(neighbors+[dict(s=1., branches=branches)],
                                    horizontal_path_observations(horizontal, positions), slice_order=positions)
        selected = select_dominant_branch(branches, window(), neighbors, surface_graph=graph)
        self.assertEqual(selected['selected']['branch_id'], 0)
        self.assertFalse(selected['selected']['detail_evaluated'])
        self.assertEqual(selected['decision_stage'],'H_V_SURFACE_EVIDENCE')

    def test_absent_observation_delegates_without_hidden_truth(self):
        profile = canonicalize_vertical(np.empty((0, 2, 3)), [])
        result = solve_dominant_branch(profile, np.empty((0, 2)), window())
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertEqual(len(result['curve_uz']), 0)

    def test_closed_loop_is_separate_not_an_open_dominant(self):
        profile, uz, branches = fixture([[[0, 0], [.02, .5], [0, 1], [-.02, .5], [0, 0]]])
        self.assertEqual(branches[0]['kind'], 'CLOSED_COMPONENT')
        self.assertIsNone(select_dominant_branch(branches, window())['selected'])


class JunctionTests(unittest.TestCase):
    def test_intersection_inside_edges_preserves_source_geometry(self):
        _, _, branches = fixture([[[0, 0], [.04, .7]], [[.04, .3], [0, 1]]])
        a, b = branches
        choices = search_backtracking_junction(a, b, backtrack_length=.5)
        self.assertTrue(choices)
        j = choices[0]
        self.assertEqual(j['junction_type'], 'REAL_INTERSECTION')
        np.testing.assert_allclose(j['a_point_uz'], [.0285714285714, .5], atol=1e-10)
        self.assertEqual(j['new_virtual_nodes'], 2)
        route = build_switch_route(a, b, j)
        self.assertEqual(route['branch_switch_count'], 1)
        self.assertEqual(route['inferred_length_m'], 0.)
        self.assertTrue(all(e['face_id'] is None for e in route['path_edges'] if e['source'] == 'TOPOLOGY_SWITCH'))
        for e in route['path_edges']:
            if e['source'].startswith('OBSERVED'):
                self.assertTrue(e['source_face_ids'])

    def test_distant_branches_do_not_get_connector(self):
        _, _, branches = fixture([[[0, 0], [0, .6]], [[.2, .4], [.2, 1]]])
        self.assertEqual(search_backtracking_junction(*branches, maximum_distance=.01), [])

    def test_switch_route_cannot_return_to_first_branch(self):
        _, _, branches = fixture([[[0, 0], [.02, .7]], [[.02, .4], [0, 1]]])
        result = build_switch_route(*branches, search_backtracking_junction(*branches, backtrack_length=.5)[0])
        identities = [e['branch_id'] for e in result['path_edges'] if e['source'].startswith('OBSERVED')]
        self.assertEqual(list(dict.fromkeys(identities)), [0, 1])


if __name__ == '__main__':
    unittest.main()
