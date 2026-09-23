"""Local v2 contracts: CC scope, physical thresholds, joint bands and P2 gates."""
import unittest
import numpy as np
from test_dominant_observed_branch import fixture, dense_path
from test_absolute_sweet_zone import line
from branch_absolute_core import BranchPolicy


class LocalConflictTests(unittest.TestCase):
    def test_joint_application_allows_two_switches_sharing_middle_branch(self):
        from region_junction_band import _apply_selected_edits,_expand
        from main_track_assembly import _splice
        from junction_geometry import search
        from dominant_observed_branch import route_result
        bs=fixture([line(0,0,5,51),line(.0003,1,6,51),line(.0006,2,7,51)])[2]
        a,b,c=[x['records'] for x in bs]
        def contact(left,right,z):
            return search(left,right,[0,7],(np.inf,-np.inf),'upper',protect_folds=False,
                preserve_extent=False,allow_clipped_boundaries=True,max_distance=.01,junction_z_bounds=(z,z))
        ab,bc=contact(a,b,3),contact(b,c,4)
        records=_splice(a,_splice(b,c,bc),ab)
        route=dict(route_result(records,branch_switch_count=2,junction=ab),junctions=[ab,bc],route_z_extent=[0,7])
        by={x['branch_id']:x for x in bs};edits=[]
        for k,r in enumerate(records):
            if r['source']!='TOPOLOGY_SWITCH':continue
            aid,bid=records[k-1]['branch_id'],records[k+1]['branch_id'];start=k-1;end=k+2
            while start and records[start-1]['source'].startswith('OBSERVED') and records[start-1]['branch_id']==aid:start-=1
            while end<len(records) and records[end]['source'].startswith('OBSERVED') and records[end]['branch_id']==bid:end+=1
            left=_expand(records[start:k],by[aid],True);right=_expand(records[k+1:end],by[bid],False)
            j=contact(left,right,2.8 if aid==bs[0]['branch_id'] else 4.2)
            item=dict(region_id='TEST',s='0.00',from_branch=aid,to_branch=bid,from_FaceTrack=aid,to_FaceTrack=bid,
                old_z=float(np.mean(r['points_uz'],axis=0)[1]),selected=dict(junction=j))
            edits.append((0.,'0.00',start,end,left,right,item))
        routes={'0.00':route};_apply_selected_edits(routes,{'0.00':bs},{'0.00':edits},BranchPolicy())
        self.assertTrue(all(x[-1]['applied'] for x in edits))
        self.assertEqual(len(route['junctions']),2)
        self.assertTrue(all(np.linalg.norm(np.asarray(x['points_xyz'][1])-y['points_xyz'][0])<1e-8 for x,y in zip(route['path_edges'],route['path_edges'][1:])))
        self.assertEqual(sorted(round(j['a_point_uz'][1],1) for j in route['junctions']),[2.8,4.2])

    def test_region_missing_profiles_do_not_count_as_consecutive(self):
        from region_joint_selection import choose_region_identity
        rows=[dict(s=s,FaceTrack=t) for t,ss in [('sparse',[0,.2,.4]),('continuous',[0,.05,.1])] for s in ss]
        d=choose_region_identity(rows,profile_s=np.arange(0,.401,.05))
        self.assertEqual(d['dominant_FaceTrack'],'continuous')
        self.assertEqual(next(r['consecutive_V'] for r in d['candidates'] if r['FaceTrack']=='sparse'),1)

    def test_current_competitor_is_independent_of_repair_cap(self):
        from physical_continuation import current_is_competitor
        a=dict(branch_id=0,z_range=[0,1]);b=dict(branch_id=1,z_range=[.96,2])
        self.assertTrue(current_is_competitor(a,b,same_chain=False))
        self.assertFalse(current_is_competitor(a,b,same_chain=True))

    def test_band_keeps_legal_alternative_in_same_bin(self):
        from region_junction_band import legal_representatives
        candidates=[dict(a_point_uz=[0,1.01],xyz_distance_m=d,legal=legal) for d,legal in [(0,False),(.005,True)]]
        result=legal_representatives(candidates,lambda j:j['legal'])
        self.assertEqual(result,[candidates[1]])

    def test_sparse_return_endpoints_do_not_prove_neck(self):
        from observed_component_geometry import near_return_families
        points=np.array([[0,0,0],[1,0,0],[2,0,0],[2,3,0],[2,.005,0],[1,1,0],[0,.005,0]])
        lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);arc=np.r_[0,np.cumsum(lengths)]
        branch=dict(arc_positions=arc,records=[dict(points_xyz=[a,b]) for a,b in zip(points,points[1:])])
        pairs=[dict(start_arc=arc[a],end_arc=arc[b],observed_arc_m=arc[b]-arc[a],D_xyz_m=.005,
                    a_index=a,b_index=b-1,a_xyz=points[a],b_xyz=points[b]) for a,b in [(0,6),(2,4)]]
        self.assertEqual(len(near_return_families(pairs,branch=branch)),2)

    def test_precomputed_observed_crossings_match_original_recomputation(self):
        from observed_surface_graph import build_surface_graph
        from joint_surface_consensus import _index,_matches
        branches=fixture([line(0,0,2),line(.005,0,2)])[2]
        hits=[dict(s=0.,z=z,u=0.,level_index=i,h_path_id=1) for i,z in enumerate([.5,1.,1.5])]
        g=build_surface_graph([dict(s=0.,branches=branches)],hits)
        self.assertIn('_observed_crossing_cache',g)
        original={k:v for k,v in g.items() if k!='_observed_crossing_cache'}
        for i in range(len(hits)):
            actual=_matches(g,_index(g),i);expected=_matches(original,_index(original),i)
            self.assertEqual([(n,c['arc_position']) for n,c in actual],[(n,c['arc_position']) for n,c in expected])
            for (_,a),(_,b) in zip(actual,expected):np.testing.assert_array_equal(a['point_xyz'],b['point_xyz'])

    def test_competitive_core_requires_two_or_more_competitors(self):
        from competitive_surface_selection import choose_successor
        row=dict(branch_id=1,MBG_pass=True,face_continuity=1,in_ASC=True,forward_Z_m=3.)
        one=choose_successor([row])
        self.assertFalse(one['competitive_core_invoked'])
        self.assertEqual(one['decision_mode'],'NONCOMPETITIVE_CONTINUATION')
        two=choose_successor([row,dict(row,branch_id=2)])
        self.assertTrue(two['competitive_core_invoked'])
        self.assertEqual(two['competitor_count'],2)

    def gap(self,distance):
        from continuation_modes import noncompetitive_junction
        from main_track_assembly import _oriented
        bs=fixture([line(0,0,1,31),line(0,1+distance,2+distance,31)])[2]
        a,b=[_oriented(x['records']) for x in bs]
        return noncompetitive_junction(a,b,required=[0,1],locked_range=(np.inf,-np.inf),
            direction='upper',same_chain=True,competitor_count=1)

    def test_noncompetitive_gap_bypasses_competitive_core(self):
        j,a=self.gap(.00194)
        self.assertIsNotNone(j)
        self.assertFalse(a['competitive_core_invoked'])
        self.assertEqual(j['handoff_kind'],'MODEL_GAP_REPAIR')

    def test_noncompetitive_gap_never_invokes_competitive_core(self):
        self.assertEqual(self.gap(.02)[1]['CC_NONCOMPETITIVE_INVOCATIONS'],0)

    def test_noncompetitive_gap_49mm_is_repaired(self):
        self.assertAlmostEqual(self.gap(.049)[0]['xyz_distance_m'],.049)

    def test_noncompetitive_gap_51mm_remains_unresolved(self):
        self.assertIsNone(self.gap(.051)[0])

    def test_independent_distance_thresholds(self):
        p=BranchPolicy(noncompetitive_gap_cap_m=.05,p1_handoff_cap_m=.008,p2_near_return_gap_m=.01)
        self.assertEqual((p.noncompetitive_gap_cap_m,p.p1_handoff_cap_m,p.p2_near_return_gap_m),(.05,.008,.01))

    def test_region_level_identity_shared_across_conflict_region(self):
        from region_joint_selection import choose_region_identity
        rows=[dict(s=s,FaceTrack=t,MBG_pass=True,in_CC=True,forward_Z_m=z,
                   forward_observed_m=z,minimum_switches=0) for s in range(12)
              for t,z in [('T0',1),('T1',5)]]
        d=choose_region_identity(rows)
        self.assertEqual(d['dominant_FaceTrack'],'T1')
        self.assertEqual(d['slice_count'],12)
        self.assertEqual(set(d['per_slice_identity'].values()),{'T1'})

    def test_joint_junction_band_uses_only_observed_candidates(self):
        from region_joint_selection import choose_junction_band
        layers=[[dict(candidate_id=f'{s}:{z}',z=z,xyz_distance_m=0.,FaceTrack='T',future_switches=0)
                 for z in zs] for s,zs in enumerate([[1.,7.],[1.1,6.8],[1.2,7.2]])]
        chosen=choose_junction_band(layers)
        self.assertLess(np.ptp([c['z'] for c in chosen]),.3)
        for c,layer in zip(chosen,layers):self.assertTrue(any(c is x for x in layer))

    def events(self,points):
        from observed_component_geometry import near_returns
        b=fixture([dense_path(points)])[2][0]
        return near_returns(b,max_gap_m=.01,min_arc_m=.5,min_ratio=0)

    def test_p2_requires_small_spatial_gap_and_long_path_gap(self):
        events=self.events([[0,0],[0,1],[1,1],[1,2],[.005,1.005],[0,3]])
        self.assertTrue(events)
        self.assertTrue(all(e['D_xyz_m']<=.01+1e-12 and e['observed_arc_m']>=.5 for e in events))

    def test_p2_rejects_open_branch_without_near_return(self):
        from main_spine_components import classify_components
        bs=fixture([line(0,0,5),line(.003,2,4)])[2]
        result=classify_components(bs,dict(path_edges=bs[0]['records']))
        self.assertFalse(any(c['role'].startswith('P2_') for c in result['components']))
        self.assertTrue(all(c['role'] in ('UNUSED_OPEN_BRANCH','AMBIGUOUS_UNUSED')
                            for c in result['components'] if c['role']!='THROUGH_PATH'))

    def test_p2_rejects_adjacent_near_points(self):
        self.assertEqual(self.events([[0,0],[0,.003],[.002,.005],[0,.009]]),[])

    def families(self):
        from observed_component_geometry import near_return_families
        pairs=[dict(start_arc=a,end_arc=b,observed_arc_m=b-a,D_xyz_m=.005,
                    a_index=i,b_index=20-i) for i,(a,b) in enumerate([(1.,9.),(1.2,8.8),(1.4,8.6)])]
        points=np.array([[0,0,0],[4,0,0],[4,.005,0],[0,.005,0]])
        branch=dict(arc_positions=np.array([1.,5.,5.005,9.]),records=[dict(points_xyz=[a,b]) for a,b in zip(points,points[1:])])
        return near_return_families(pairs,branch=branch)

    def test_p2_outer_gate_covers_full_detour(self):
        f=self.families()[0]
        self.assertEqual((f['outer']['start_arc'],f['outer']['end_arc']),(1.,9.))

    def test_p2_inner_gate_removes_near_coincident_neck(self):
        f=self.families()[0]
        self.assertEqual((f['inner']['start_arc'],f['inner']['end_arc']),(1.4,8.6))

    def test_p2_without_inner_neck_uses_outer_as_inner(self):
        from observed_component_geometry import near_return_families
        e=dict(start_arc=1.,end_arc=3.,observed_arc_m=2.,D_xyz_m=.005,a_index=1,b_index=8)
        f=near_return_families([e])[0]
        self.assertIs(f['inner'],f['outer'])

    def test_p2_sparse_legal_pairs_share_a_balanced_neck_family(self):
        from observed_component_geometry import near_return_families
        pairs=[dict(start_arc=a,end_arc=b,observed_arc_m=b-a,D_xyz_m=d,a_index=i,b_index=j,
                    a_xyz=[a,0,0],b_xyz=[a,d,0]) for a,b,d,i,j in
               [(40.0937,41.9212,.0081,496,518),(40.3108,41.7017,.0096,499,516),(40.4496,41.5616,.0094,501,515)]]
        points=np.array([[40.0937,0,0],[40.6,0,0],[40.6,.0081,0],[40.0937,.0081,0]])
        branch=dict(arc_positions=np.array([40.0937,40.6,41.4149,41.9212]),records=[dict(points_xyz=[a,b]) for a,b in zip(points,points[1:])])
        families=near_return_families(pairs,branch=branch)
        self.assertEqual(len(families),1)
        self.assertEqual(families[0]['inner']['start_arc'],40.4496)


if __name__=='__main__':unittest.main()
