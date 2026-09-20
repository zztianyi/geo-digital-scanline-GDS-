"""Directional replacement, with hand-built raw orthogonal observations."""
import importlib
import unittest
import numpy as np
from test_dominant_observed_branch import fixture, dense_path
from test_surface_track import scene
from branch_absolute_core import branch_metrics
from dominant_observed_branch import route_result
from main_track_assembly import assemble_main_track


def tail_scene(reliable_fold=False, gap=0., parallel=False):
    a = [[0, 0], [0, 5], [2, 5.1], [3, 4.9], [4, 5.05]]
    b = [[gap+.03, 4], [gap-.03, 5], [gap-.03, 10]]
    if parallel:
        a = [[-x,z] for x,z in a]
        b = [[gap,4],[gap,10]]
    ss = [-1., -.5, -.25, -.05, 0., .05, .25, .5, 1.]
    paths = {s: [a, b] for s in ss}
    levels = np.arange(.25, 9.9, .25)
    # Each crossing keeps its own H identity; no scalar u(z) averaging.
    from horizontal_surface_link import branch_crossings
    branches = fixture([dense_path(a), dense_path(b)])[2]
    horizontal = []
    for z in levels:
        hp = []
        for branch in branches:
            for ci, cross in enumerate(branch_crossings(branch, z)):
                extent = .05 if branch['branch_id'] == 0 and z >= 4.75 and not reliable_fold else 1.
                hp.append(dict(h_path_id=f"{branch['branch_id']}_{ci}",
                               points_su=[[-extent, cross['u']], [extent, cross['u']]]))
        horizontal.append(dict(z=z, paths=hp))
    return scene(paths, horizontal=horizontal)


class DirectionalConsensusTests(unittest.TestCase):
    def metrics(self, branch, **kwargs):
        mod = importlib.import_module('branch_absolute_core')
        self.assertTrue(hasattr(mod, 'directional_branch_metrics'), 'directional metrics missing')
        return mod.directional_branch_metrics(branch, edge_scale=.01, **kwargs)

    def test_arc_length_fold_does_not_fake_forward_support(self):
        b = fixture([dense_path([[0, 0], [0, 5], [20, 5.1], [40, 4.9], [60, 5.]])])[2][0]
        m = self.metrics(b, arc_position=5., direction=1, z_direction=1)
        self.assertGreater(m['L_forward_arc'], 59.)
        self.assertLess(m['Z_forward_new'], .11)
        self.assertLess(m['E_progress'], .002)

    def test_directional_tail_detected_despite_long_remaining_arc(self):
        b = fixture([dense_path([[0, 0], [0, 5], [20, 5.1], [40, 4.9], [60, 5.]])])[2][0]
        m = self.metrics(b, arc_position=5., direction=1, z_direction=1,
            support=dict(support_s_span=.1, support_slice_count=3),
            alternative=dict(Z_forward_new=10., support_s_span=2., L_forward_ASC=10.))
        self.assertEqual(m['directional_state'], 'DIRECTIONAL_TAIL')

    def test_joint_VH_surface_consensus(self):
        profiles, g = tail_scene()
        module = importlib.util.find_spec('joint_surface_consensus')
        self.assertIsNotNone(module, 'joint arc evidence missing')
        from joint_surface_consensus import arc_support_profile
        rows = arc_support_profile(g, (0., 0))
        self.assertTrue(rows)
        self.assertTrue(all('edge_id' in r and 'arc_position' in r and 'h_path_id' in r for r in rows))
        at5 = [r for r in rows if abs(r['z']-5.) < 1e-8]
        self.assertGreater(len({round(r['arc_position'], 5) for r in at5}), 1)

    def test_wide_neighbor_persistence_beats_local_tier_tie(self):
        profiles, g = tail_scene()
        from surface_track_selection import branch_evidence, choose_candidate
        a, b = [branch_evidence(g, (0., i), target_z=(4.8, 5.1)) for i in (0, 1)]
        self.assertEqual(a['surface_support_tier'], b['surface_support_tier'])
        decision = choose_candidate([a, b], current_branch_id=0)
        self.assertEqual(decision['selected']['branch_id'], 1)

    def test_H_is_not_unilateral_veto(self):
        from surface_track_selection import choose_candidate
        from test_absolute_sweet_zone import row
        decision = choose_candidate([row(0, h=0), row(1, h=2)])
        self.assertNotEqual(decision['candidates'][0]['reason'], 'REJECT_SURFACE_EVIDENCE')

    def test_internal_handoff_replaces_low_confidence_tail(self):
        profiles, graph = tail_scene()
        bs = profiles[0.][2]
        r = assemble_main_track(bs, route_result(bs[0]['records']), anchor_branch_id=0, graph=graph, target_s=0.)
        self.assertEqual(r['route_branch_sequence'], [0, 1])
        self.assertEqual(r.get('internal_handoff_count'), 1)
        self.assertGreater(r['internal_tail_trimmed_length'], 4.)
        self.assertLess(max(j['xyz_distance_m'] for j in r['junctions']), .010000001)

    def test_reliable_fold_is_protected(self):
        profiles, graph = tail_scene(reliable_fold=True)
        bs = profiles[0.][2]
        r = assemble_main_track(bs, route_result(bs[0]['records']), anchor_branch_id=0, graph=graph, target_s=0.)
        self.assertEqual(r.get('internal_handoff_count', 0), 0)
        for p in ([2, 5.1], [3, 4.9]):
            self.assertTrue(np.any(np.all(np.isclose(r['curve_uz'], p), axis=1)))

    def test_future_switch_cost_prefers_long_continuation(self):
        module = importlib.util.find_spec('directional_handoff')
        self.assertIsNotNone(module, 'bounded lookahead missing')
        from directional_handoff import future_switch_cost
        bs = fixture([dense_path([[0, 0], [0, 2]]), dense_path([[0, 2.004], [0, 4]]),
                      dense_path([[.005, 0], [.005, 4]])])[2]
        short = future_switch_cost(bs, 0, z_direction=1, target_z=4.)
        long = future_switch_cost(bs, 2, z_direction=1, target_z=4.)
        self.assertGreater(short['estimated_additional_switches'], long['estimated_additional_switches'])
        self.assertEqual(long['estimated_additional_switches'], 0)

    def test_folded_branch_can_participate_in_handoff(self):
        profiles,g=tail_scene()
        from surface_track_handoff import select_handoff
        r=select_handoff(g,(0.,0))
        self.assertIsNotNone(r)
        self.assertTrue(r['junction']['folded_branch_handoff'])

    def test_internal_replacement_searches_junction_before_tail_lock(self):
        profiles,g=tail_scene()
        bs=profiles[0.][2]
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        self.assertAlmostEqual(r['junction']['a_point_uz'][1],4.5,places=5)
        self.assertLess(r['junction']['a_point_uz'][0],.01)

    def test_10mm_cap_applies_to_true_selected_handoff(self):
        for gap,expected in ((.009,1),(.011,0)):
            with self.subTest(gap=gap):
                profiles,g=tail_scene(gap=gap,parallel=True);bs=profiles[0.][2]
                r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
                self.assertEqual(r['internal_handoff_count'],expected)
                self.assertTrue(all(j['xyz_distance_m']<=.010000001 for j in r['junctions']))

    def test_no_coordinate_averaging(self):
        profiles,g=tail_scene(gap=.009,parallel=True);p,uz,bs=profiles[0.]
        original=uz.copy()
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        np.testing.assert_array_equal(uz,original)
        for edge in r['path_edges']:
            if not edge['source'].startswith('OBSERVED'):
                self.assertIsNone(edge['face_id']);continue
            a,b=uz[edge['node_ids']]
            np.testing.assert_allclose(edge['points_uz'],[a+t*(b-a) for t in (edge['t0'],edge['t1'])],atol=2e-10,rtol=0)

    def test_14m_case_remains_rejected(self):
        from test_absolute_sweet_zone import line,supported_graph
        bs=fixture([line(0,0,5),line(14.05,5,10)])[2]
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=supported_graph(bs),target_s=0.)
        self.assertEqual(r['branch_switch_count'],0)
        self.assertIn('LONG_GAP_UNRESOLVED',r['unresolved_reasons'])

    def test_fold_inside_reliable_core_is_preserved(self):
        profiles,g=tail_scene(True);bs=profiles[0.][2]
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        self.assertEqual(r['internal_tail_trimmed_length'],0.)
        self.assertTrue(np.any(np.diff(r['curve_uz'][:,1])<0))

    def test_fold_inside_directional_tail_can_be_trimmed(self):
        profiles,g=tail_scene();bs=profiles[0.][2]
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        self.assertFalse(np.any(np.diff(r['curve_uz'][:,1])<0))
        self.assertGreater(r['internal_tail_trimmed_length'],4.)

    def test_B1_like_long_continuation_beats_short_B6_like_candidate(self):
        from test_absolute_sweet_zone import line,supported_graph
        bs=fixture([line(0,0,4),line(.001,3.5,6),line(.006,3.5,10)])[2]
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=supported_graph(bs),target_s=0.)
        self.assertEqual(r['route_branch_sequence'],[0,2])
        self.assertEqual(r['branch_switch_count'],1)

    def test_future_switch_cannot_reenter_discarded_prefix(self):
        from directional_handoff import future_switch_cost
        bs=fixture([dense_path(p) for p in (
            [[0,0],[0,4.8]],[[-3.7,1],[2.3,7]],[[-.5,0],[-.5,10]])])[2]
        r=future_switch_cost(bs,0,z_direction=1,target_z=10.)
        self.assertFalse(r['future_frontier_reached'])

    def test_actual_junction_rechecks_fold_before_sample(self):
        a=[[0,0],[0,5],[2,5.1],[3,4.9],[4,5.05]]
        b=[[2.9,4],[2.9,10]]
        ss=[-1.,-.5,-.25,-.05,0.,.05,.25,.5,1.]
        bs=fixture([dense_path(a),dense_path(b)])[2]
        from horizontal_surface_link import branch_crossings
        horizontal=[]
        for z in np.arange(.25,9.9,.25):
            hp=[]
            for branch in bs:
                for ci,c in enumerate(branch_crossings(branch,z)):
                    width=.05 if branch['branch_id']==0 and c['arc_position']>8.03 else 1.
                    hp.append(dict(h_path_id=f"{branch['branch_id']}_{ci}",
                        points_su=[[-width,c['u']],[width,c['u']]]))
            horizontal.append(dict(z=z,paths=hp))
        profiles,g=scene({s:[a,b] for s in ss},horizontal=horizontal);bs=profiles[0.][2]
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        self.assertTrue(np.any(np.all(np.isclose(r['curve_uz'],[3,4.9]),axis=1)),
            'Actual junction cut the supported fold before the sampled tail position')


if __name__ == '__main__':
    unittest.main()
