"""Geometry search stays exact; assembly now requires reliable evidence."""
import unittest
import numpy as np
from test_dominant_observed_branch import fixture
from test_absolute_sweet_zone import line,supported_graph
from dominant_observed_branch import solve_dominant_branch
from main_track_assembly import _nearest_join,_splice


class MainTrackAssemblyTests(unittest.TestCase):
    def solve(self,paths):
        p,uz,bs=fixture(paths)
        before=p.nodes.copy()
        result=solve_dominant_branch(p,uz,branches=bs,surface_graph=supported_graph(bs),target_s=0.)
        np.testing.assert_array_equal(p.nodes,before)
        return result,p,uz

    def test_three_tracks_continue_across_two_small_gaps(self):
        r,_,_=self.solve([line(0,0,1),line(0,1.004,2.004),line(0,2.008,3.008)])
        self.assertEqual(r['branch_switch_count'],2)
        self.assertAlmostEqual(r['connector_length_m'],.008)
        self.assertAlmostEqual(r['curve_uz'][:,1].max(),3.008)

    def test_long_gaps_remain_unresolved(self):
        r,_,_=self.solve([line(0,0,1),line(0,1.02,2.02)])
        self.assertEqual(r['branch_switch_count'],0)
        self.assertEqual(r['status'],'UNRESOLVED')
        self.assertIn('LONG_GAP_UNRESOLVED',r['unresolved_reasons'])

    def test_intersection_search_preserves_source_geometry(self):
        p,uz,b=fixture([[[0,0],[.04,.8]],[[.04,.2],[0,1]]])
        j=_nearest_join(b[0]['records'],b[1]['records'],[0,.8],(np.inf,-np.inf),'upper')
        self.assertAlmostEqual(j['distance_m'],0.)
        self.assertAlmostEqual(j['a_point_uz'][1],.5)
        for r in _splice(b[0]['records'],b[1]['records'],j):
            if r['source'].startswith('OBSERVED'):
                a,c=uz[r['node_ids']]
                np.testing.assert_allclose(r['points_uz'],[a+t*(c-a) for t in (r['t0'],r['t1'])])
                self.assertEqual(r['source_face_ids'],p.source_face_ids[r['edge_id']])
            else:self.assertIsNone(r['face_id'])

    def test_earlier_intersection_cannot_cut_accepted_fold(self):
        _,_,b=fixture([[[0,0],[0,.5],[2,.8],[2,.6],[1,1]],[[-.1,.4],[.1,.4],[.1,2]]])
        j=_nearest_join(b[0]['records'],b[1]['records'],[0,1],(np.inf,-np.inf),'upper')
        route=_splice(b[0]['records'],b[1]['records'],j)
        points=np.array([route[0]['points_uz'][0]]+[r['points_uz'][1] for r in route])
        for point in ([2,.8],[2,.6]): self.assertTrue(np.any(np.all(np.isclose(points,point),axis=1)))

    def test_parallel_alternative_keeps_current_when_evidence_tied(self):
        r,_,_=self.solve([line(0,0,2),line(.02,.5,1.5)])
        self.assertEqual(r['branch_switch_count'],0)
        self.assertTrue(r['selection']['ambiguous'])

    def test_short_fold_is_rejected_instead_of_creating_A_B_A(self):
        r,_,_=self.solve([[[0,.2],[.2,.8],[-.2,.4],[0,.8]],line(.001,0,1)])
        self.assertEqual(r['branch_switch_count'],0)
        self.assertNotIn(0,r['route_branch_sequence'])

    def test_five_node_branch_cannot_force_long_connector(self):
        r,_,_=self.solve([line(0,0,1),[[5,.5],[5,3],[.1,1.1],[.2,2]]])
        self.assertEqual(r['route_branch_sequence'],[0])
        self.assertEqual(r['candidate_z_extent'],[0.,1.])


if __name__=='__main__':unittest.main()
