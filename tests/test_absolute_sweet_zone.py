"""Approved absolute-core policy: decision order and unsafe continuation regressions."""
import unittest
from unittest.mock import patch
import numpy as np
from test_dominant_observed_branch import fixture
from observed_surface_graph import build_surface_graph
from dominant_observed_branch import solve_dominant_branch, route_result
from branch_absolute_core import BranchPolicy, branch_metrics, local_edge_scale, connector_gate
from surface_track_selection import choose_candidate, branch_evidence
from main_track_assembly import assemble_main_track


def line(u, low, high, nodes=41):
    z = np.linspace(low, high, nodes)
    return np.column_stack((np.full(nodes, u), z))


def supported_graph(branches):
    profiles = [dict(s=s, branches=branches) for s in (-1., 0., 1.)]
    graph = build_surface_graph(profiles)
    graph['level_z'] = dict(enumerate(np.linspace(-10, 20, 301)))
    for b in branches:
        node = (0., b['branch_id'])
        levels = [i for i, z in graph['level_z'].items() if b['z_range'][0] <= z <= b['z_range'][1]]
        graph['support'][node] = [dict(a=node, b=(s,b['branch_id']), levels=levels,
            support_fraction=1., continuous_H_support_run=len(levels)) for s in (-1., 1.)]
        tid = graph['membership'][node]
        for s in (-1., 1.):
            graph['membership'][(s,b['branch_id'])] = tid
    return graph


def row(bid, *, core=.8, h=2, detail=.0):
    return dict(branch_id=bid, MBG_pass=True, surface_support_tier=h,
        target_region_in_ASC_fraction=core, neighbor_detail_score=detail,
        neighbor_detail_repeat_count=int(detail>.6), track_ambiguous=False)


class AbsoluteSweetZoneTests(unittest.TestCase):
    def test_short_branch_relative_middle_is_not_sweet(self):
        b = fixture([line(0,0,1,9)])[2][0]
        m = branch_metrics(b, .1)
        self.assertFalse(m['ASC_exists'])
        self.assertEqual(m['reason'], 'REJECT_SHORT_BRANCH')

    def test_five_node_branch_rejected_before_detail_score(self):
        b = fixture([line(0,0,1,5)])[2]
        g = supported_graph(b)
        with patch('surface_track_selection.same_surface_detail', side_effect=AssertionError('detail too early')):
            evidence = branch_evidence(g, (0., 0))
        self.assertFalse(evidence['MBG_pass'])

    def test_absolute_node_depth_sweet_core(self):
        b = fixture([line(0,0,2,21)])[2][0]
        m = branch_metrics(b,.1, target_z=(.5,1.5))
        self.assertTrue(m['ASC_exists'])
        self.assertEqual(m['left_guard_node'],4)
        self.assertEqual(m['right_guard_node'],16)
        self.assertAlmostEqual(m['ASC_arc_length'],1.2)
        self.assertAlmostEqual(m['target_region_in_ASC_fraction'],1.)
        self.assertGreaterEqual(m['target_region_distance_to_nearest_endpoint_nodes'],5)

    def test_long_branch_middle_beats_short_branch_middle(self):
        short, long = fixture([line(.1,0,1,9),line(0,0,2,41)])[2]
        rows = [dict(branch_metrics(b,.05),surface_support_tier=2,track_ambiguous=False) for b in (short,long)]
        decision = choose_candidate(rows)
        self.assertEqual(decision['selected']['branch_id'],long['branch_id'])

    def test_h_support_lowers_confidence_without_rejecting_branch(self):
        d = choose_candidate([row(0,h=0,detail=1),row(1,h=2,detail=0)])
        self.assertEqual(d['selected']['branch_id'],1)
        self.assertTrue(d['candidates'][0]['MBG_pass'])
        self.assertEqual(d['candidates'][0]['reason'],'LOWER_JOINT_CONFIDENCE')

    def test_both_supported_sweet_zone_breaks_tie(self):
        d = choose_candidate([row(0,core=.1,detail=1),row(1,core=.8,detail=0)])
        self.assertEqual(d['selected']['branch_id'],1)

    def test_detail_breaks_tie_only_after_sweet_zone(self):
        d = choose_candidate([row(0,detail=.1),row(1,detail=.9)])
        self.assertEqual(d['selected']['branch_id'],1)
        self.assertEqual(d['decision_stage'],'SAME_SURFACE_DETAIL')

    def test_close_tie_keeps_current(self):
        d = choose_candidate([row(0,detail=.78),row(1,detail=.8)],current_branch_id=0)
        self.assertEqual(d['selected']['branch_id'],0)
        self.assertTrue(d['ambiguous'])

    def test_no_short_detour_A_B_A(self):
        branches = fixture([line(0,0,2),line(.001,.7,1.3,5)])[2]
        g=supported_graph(branches)
        r=assemble_main_track(branches,route_result(branches[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        self.assertEqual(r['route_branch_sequence'],[0])
        self.assertEqual(r['branch_switch_count'],0)

    def test_connector_cannot_dominate_observed(self):
        self.assertEqual(connector_gate(.009,.008)['reason'],'REJECT_SYNTHETIC_DOMINANCE')
        self.assertFalse(connector_gate(.005,.005)['accepted'])
        self.assertTrue(connector_gate(.005,.1)['accepted'])

    def test_14m_connector_case_rejected(self):
        self.assertEqual(connector_gate(14.05,113.8)['reason'],'LONG_GAP_UNRESOLVED')
        self.assertFalse(connector_gate(.01001,100)['accepted'])

    def test_tiny_branch_does_not_expand_target_extent(self):
        profile,uz,branches=fixture([line(0,0,2),line(-13.4,6.23,6.23564,3)])
        r=solve_dominant_branch(profile,uz,branches=branches,surface_graph=supported_graph(branches),target_s=0.)
        self.assertEqual(r['candidate_z_extent'],[0.,2.])
        self.assertEqual(r['branch_switch_count'],0)
        self.assertEqual(r['inferred_length_m'],0.)

    def test_dense_tiny_branch_cannot_set_own_physical_scale(self):
        branches=fixture([line(0,0,10,101),line(1,0,.001,41)])[2]
        scale=local_edge_scale(branches)
        self.assertGreater(scale,.09)
        self.assertFalse(branch_metrics(branches[1],scale)['MBG_pass'])

    def test_supported_small_gap_continues_and_preserves_provenance(self):
        profile,uz,branches=fixture([line(0,0,1),line(0,1.005,2.005)])
        r=solve_dominant_branch(profile,uz,branches=branches,surface_graph=supported_graph(branches),target_s=0.)
        self.assertEqual(r['branch_switch_count'],1)
        self.assertAlmostEqual(r['connector_length_m'],.005)
        for edge in r['path_edges']:
            if edge['source'].startswith('OBSERVED'):
                a,b=uz[edge['node_ids']]
                np.testing.assert_allclose(edge['points_uz'],[a+t*(b-a) for t in (edge['t0'],edge['t1'])])

    def test_source_proof_is_reported_without_synthetic_replacement(self):
        profile,uz,branches=fixture([line(0,0,2),line(-13.4,6.23,6.23564,3)])
        r=solve_dominant_branch(profile,uz,branches=branches,surface_graph=supported_graph(branches),target_s=0.,
            input_z_range=(0,2),source_data_required=True)
        self.assertEqual(r['status'],'SOURCE_DATA_REQUIRED')
        self.assertEqual(r['connector_length_m'],0.)

    def test_radial_gap_over_cap_returns_rejection_without_exception(self):
        profile,uz,branches=fixture([line(0,0,1),line(.02,.9,2)])
        r=solve_dominant_branch(profile,uz,branches=branches,surface_graph=supported_graph(branches),target_s=0.)
        self.assertEqual(r['branch_switch_count'],0)
        self.assertIn('LONG_GAP_UNRESOLVED',r['unresolved_reasons'])

    def guard_scene(self):
        za=np.r_[np.linspace(0,.1,5),np.linspace(.1,.6,33)[1:],np.linspace(.6,1,5)[1:]]
        zb=np.r_[np.linspace(0,.4,5),np.linspace(.4,.9,33)[1:],np.linspace(.9,1,5)[1:]]
        p,uz,bs=fixture([np.c_[np.zeros(len(za)),za],np.c_[np.full(len(zb),.005),zb]])
        g=supported_graph(bs);g['level_z']=dict(enumerate(np.arange(.1,1,.1)))
        for links in g['support'].values():
            for link in links:link.update(levels=list(g['level_z']),continuous_H_support_run=len(g['level_z']))
        return p,uz,bs,g

    def test_handoff_cannot_trade_two_sided_H_for_one_sided_core(self):
        p,uz,bs,g=self.guard_scene();g['support'][(0.,1)]=g['support'][(0.,1)][:1]
        r=solve_dominant_branch(p,uz,branches=bs,surface_graph=g,target_s=0.)
        self.assertEqual(r['route_branch_sequence'],[0])

    def test_track_lock_applies_to_handoff(self):
        p,uz,bs,g=self.guard_scene();tid=g['membership'][(0.,0)]
        r=solve_dominant_branch(p,uz,branches=bs,surface_graph=g,target_s=0.,surface_track_id=tid)
        self.assertEqual(r['route_branch_sequence'],[0])

    def test_graph_ambiguity_survives_unique_candidate(self):
        candidate=row(0);candidate['track_ambiguous']=True
        self.assertTrue(choose_candidate([candidate])['ambiguous'])

    def test_missing_supported_tail_is_unresolved(self):
        from test_surface_track import scene
        profiles,g=scene({-1.:[[[.02,0],[.02,1]]],0.:[[[.02,0],[.02,.55]]],1.:[[[.02,0],[.02,1]]]},
            levels=(.1,.2,.3,.4,.6,.7,.8,.9))
        p,uz,bs=profiles[0.]
        r=solve_dominant_branch(p,uz,branches=bs,surface_graph=g,target_s=0.)
        self.assertEqual(r['status'],'UNRESOLVED')
        self.assertIn('MISSING_SAME_SURFACE_OBSERVATIONS',r['unresolved_reasons'])
        self.assertEqual(r['inferred_length_m'],0.)

    def test_combined_connectors_cannot_dominate_intermediate_branch(self):
        bs=fixture([line(0,0,.100,201),line(0,.109,.121,25),line(0,.130,.230,201)])[2]
        g=supported_graph(bs);g['level_z']=dict(enumerate(np.arange(0,.231,.002)))
        for b in bs:
            ls=[i for i,z in g['level_z'].items() if b['z_range'][0]<=z<=b['z_range'][1]]
            for link in g['support'][(0.,b['branch_id'])]:link.update(levels=ls,continuous_H_support_run=len(ls))
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        self.assertLessEqual(r['branch_switch_count'],1)
        self.assertIn('REJECT_SYNTHETIC_DOMINANCE',r['unresolved_reasons'])

    def test_track_lock_cannot_shrink_local_physical_scale(self):
        from surface_track_selection import select_surface_track
        bs=fixture([line(0,0,10,101),line(1,0,.001,41)])[2];g=supported_graph(bs)
        r=select_surface_track(g,0.,surface_track_id=g['membership'][(0.,1)])
        self.assertIsNone(r['selected'])

    def test_distant_track_ambiguity_does_not_block_clear_local_continuation(self):
        p,uz,bs=fixture([line(0,0,1),line(0,1.005,2.005)])
        g=supported_graph(bs)
        for track in g['tracks']:track['ambiguous']=True
        r=solve_dominant_branch(p,uz,branches=bs,surface_graph=g,target_s=0.)
        self.assertEqual(r['branch_switch_count'],1)
        self.assertTrue(r['selection']['ambiguous'])
        self.assertEqual(r['status'],'UNRESOLVED')


if __name__=='__main__': unittest.main()
