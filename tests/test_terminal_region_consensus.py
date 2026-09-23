"""Regression contracts for a through-region identity and terminal-first joints."""
import unittest
import numpy as np
from test_dominant_observed_branch import fixture
from test_absolute_sweet_zone import line


class TerminalRegionTests(unittest.TestCase):
    def test_different_region_sizes_do_not_override_full_neighbor_support(self):
        from competitive_surface_selection import choose_successor
        rows=[dict(branch_id=1,MBG_pass=True,face_continuity=7,face_context_slice_count=7,in_ASC=True,forward_Z_m=20.,minimum_switches=0),
              dict(branch_id=6,MBG_pass=True,face_continuity=75,face_context_slice_count=75,in_ASC=False,forward_Z_m=1.,minimum_switches=1)]
        self.assertEqual(choose_successor(rows)['selected']['branch_id'],1)

    def test_through_track_beats_a_slightly_wider_dead_end(self):
        from region_joint_selection import choose_region_identity
        rows=[dict(s=s,FaceTrack=t,through_region=through,MBG_pass=True,in_CC=True,
                   local_Z_m=z,forward_Z_m=forward,minimum_switches=n)
              for s in [0.,.05,.1] for t,through,z,forward,n in
              [('through',True,3.3,25.,1),('dead_end',False,3.5,2.,2)]]
        self.assertEqual(choose_region_identity(rows)['dominant_FaceTrack'],'through')

    def test_region_span_distinguishes_boundary_crossing_from_internal_endpoint(self):
        from region_joint_selection import branch_region_span
        bs=fixture([line(0,0,5,51),line(.1,2,5,31)])[2]
        region=dict(bounds=[-.1,.1,-1,1,1,4])
        self.assertTrue(branch_region_span(bs[0],region)['through_region'])
        self.assertFalse(branch_region_span(bs[1],region)['through_region'])
        folded=fixture([np.array([[2.,1.],[0.,2.],[2.,3.]])])[2][0]
        self.assertFalse(branch_region_span(folded,region)['through_region'])

    def test_terminal_search_reaches_first_close_pair_with_retained_core(self):
        from terminal_junction_consensus import terminal_candidates
        from main_track_assembly import _oriented
        from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale
        # Dense original vertices on both paths make their internal cores eligible.
        bs=fixture([line(0,0,5,101),line(.0015,0,5,101)])[2]
        metrics=[branch_metrics(b,local_edge_scale(bs)) for b in bs]
        candidates=terminal_candidates(*[_oriented(b['records']) for b in bs],*bs,*metrics,
            direction='upper',policy=BranchPolicy())
        self.assertTrue(candidates)
        first=min(candidates,key=lambda j:j['terminal_retreat_m'])
        # The right suffix must retain its minimum original core contribution;
        # the source can advance to the edge of the 2 mm tube around that point.
        last=metrics[1]['ASC_end_arc']-metrics[1]['minimum_core_arc_length']
        self.assertAlmostEqual(first['a_point_uz'][1],last+np.sqrt(.002**2-.0015**2))
        self.assertAlmostEqual(first['xyz_distance_m'],.002)

    def test_common_band_search_moves_inward_when_first_nodes_disagree(self):
        from terminal_junction_consensus import choose_terminal_band
        def c(z,r,d=.001):return dict(z=z,terminal_retreat_m=r,xyz_distance_m=d,u=0.)
        layers=[[c(10,0),c(9.80,.2)],[c(9.82,0)],[c(9.85,0)]]
        chosen=choose_terminal_band(layers)
        self.assertEqual([c['z'] for c in chosen],[9.8,9.82,9.85])
        self.assertTrue(all(any(c is x for x in layer) for c,layer in zip(chosen,layers)))

    def test_band_span_is_100mm_total_and_all_profiles_are_required(self):
        from terminal_junction_consensus import choose_terminal_band
        def c(z):return dict(z=z,terminal_retreat_m=0.,xyz_distance_m=.001,u=0.)
        self.assertEqual(choose_terminal_band([[c(0)],[c(.15)]]),[])
        self.assertEqual(choose_terminal_band([[c(0)],[]]),[])

    def test_gap_over_2mm_cannot_supply_missing_neighbor(self):
        from terminal_junction_consensus import choose_terminal_band
        a=dict(z=1.,terminal_retreat_m=0.,xyz_distance_m=.001,u=0.)
        b=dict(z=1.01,terminal_retreat_m=0.,xyz_distance_m=.0021,u=0.)
        self.assertEqual(choose_terminal_band([[a],[b]]),[])

    def test_opposite_fold_directions_find_an_interior_common_band(self):
        from terminal_junction_consensus import terminal_candidates,choose_terminal_band
        from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale
        bs=fixture([line(0,0,1,21),line(.001,0,1,21)])[2]
        mm=[branch_metrics(b,local_edge_scale(bs)) for b in bs]
        layers=[terminal_candidates(*[b['records'] for b in bs],*bs,*mm,direction=d,policy=BranchPolicy()) for d in ('upper','lower')]
        chosen=choose_terminal_band(layers)
        self.assertEqual(len(chosen),2)
        self.assertAlmostEqual(max(p['terminal_retreat_m'] for p in chosen),.45,places=6)

    def test_band_span_includes_both_ends_of_each_connector(self):
        from terminal_junction_consensus import choose_terminal_band
        a=dict(z=0.,terminal_retreat_m=0.,xyz_distance_m=.002,u=0.,a_point_uz=[0,0],b_point_uz=[0,-.002])
        b=dict(z=.1,terminal_retreat_m=0.,xyz_distance_m=.002,u=0.,a_point_uz=[0,.1],b_point_uz=[0,.102])
        self.assertEqual(choose_terminal_band([[a],[b]]),[])

    def test_internal_budget_interval_is_found_before_window_anchors(self):
        from terminal_junction_consensus import choose_terminal_band,_materialize
        g=dict(terminal_side='a',t_range=[0.,1.],a_xyz=[[0.,0.,0.],[0.,0.,1.]],b_xyz=[[.001,0.,0.],[.001,0.,1.]],
            a_uz=[[0.,0.],[0.,1.]],b_uz=[[.001,0.],[.001,1.]],a_map=[0,0.,1.],b_map=[0,0.,1.],retreat_at_0=0.,retreat_slope=1.)
        c=_materialize(dict(corridor=g,z_min=0.,z_max=1.),0.)
        def valid(i,p):return .2<p['z']<.8
        valid.breakpoints=lambda i,p:[.2,.8]
        chosen=choose_terminal_band([[c]],valid=valid)
        self.assertEqual(len(chosen),1)
        self.assertAlmostEqual(chosen[0]['z'],.2,places=6)

    def test_narrow_horizontal_budget_interval_is_not_discarded(self):
        from terminal_junction_consensus import choose_terminal_band,_materialize
        g=dict(terminal_side='a',t_range=[0.,1.],a_xyz=[[-.0005,0.,0.],[.002,0.,0.]],b_xyz=[[0.,0.,0.],[0.,0.,1.]],
            a_uz=[[-.0005,0.],[.002,0.]],b_uz=[[0.,0.],[0.,1.]],a_map=[0,0.,1.],b_map=[0,0.,1.],retreat_at_0=0.,retreat_slope=1.)
        c=_materialize(dict(corridor=g,z_min=0.,z_max=0.),0.)
        def valid(i,p):return p['xyz_distance_m']<.0002
        valid.breakpoints=lambda i,p:[.12,.28]
        chosen=choose_terminal_band([[c]],valid=valid)
        self.assertEqual(len(chosen),1)
        self.assertAlmostEqual(chosen[0]['terminal_retreat_m'],.12,places=6)


if __name__=='__main__':unittest.main()
