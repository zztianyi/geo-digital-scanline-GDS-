"""Approved v2: identity before geometry; reliable entry and lossless layers."""
import unittest
import numpy as np
from test_dominant_observed_branch import fixture, dense_path
from test_absolute_sweet_zone import line
from branch_absolute_core import branch_metrics, local_edge_scale


class ProductionCompetitionTests(unittest.TestCase):
    def physical_context(self, branches, regions, extra_connections=()):
        from scipy import sparse
        from physical_face_context import PhysicalFaceContext
        count=max(e['face_id'] for b in branches for e in b['records'])+1
        adjacency=sparse.lil_matrix((count,count));low=np.zeros((count,3));high=np.zeros_like(low)
        for b in branches:
            ids=[e['face_id'] for e in b['records']]
            for a,c in zip(ids,ids[1:]):adjacency[a,c]=adjacency[c,a]=1
            for e in b['records']:
                p=np.asarray(e['points_uz']);i=e['face_id']
                low[i]=[-.1,*p.min(axis=0)];high[i]=[.1,*p.max(axis=0)]
        for a,c in extra_connections:adjacency[a,c]=adjacency[c,a]=1
        return PhysicalFaceContext(adjacency.tocsr(),low,high,{0.:branches},regions)

    def test_return_identity_uses_local_return_position(self):
        from main_spine_components import classify_components
        bs=fixture([dense_path([[-3,0],[-2,1],[0,2],[1,3],[2,3],[2.5,3.5],[2,4],[1.005,3.005],[0,5],[-2,6],[-3,7]])])[2]
        regions=[dict(region_id='A_REMOTE',bounds=[-.1,.1,-3.1,-.9,2.9,4.1]),
                 dict(region_id='B_RETURN',bounds=[-.1,.1,.9,3.1,2.9,4.1])]
        ctx=self.physical_context(bs,regions)
        result=classify_components(bs,dict(path_edges=bs[0]['records']),context=ctx,target_s=0.)
        self.assertTrue(any(c['reason']=='SAME_FACETRACK_LOCAL_ENTRY_EXIT_RETURN' for c in result['components']))
        self.assertTrue(result['source_intervals_preserved'])

    def test_side_identity_is_checked_at_attached_endpoint(self):
        from main_spine_components import classify_components
        bs=fixture([line(0,0,10,101),dense_path([[.005,2],[1,3],[2,4],[3,8]])])[2]
        anchor=min(bs[0]['records'],key=lambda e:abs(np.mean(np.asarray(e['points_uz'])[:,1])-2))['face_id']
        regions=[dict(region_id='REMOTE_MIDDLE',bounds=[-.1,.1,2.9,5.1,4.9,7.1]),
                 dict(region_id='ACTUAL_CONTACT',bounds=[-.1,.1,-.1,2.1,.9,3.1])]
        ctx=self.physical_context(bs,regions,[(anchor,bs[1]['records'][0]['face_id'])])
        result=classify_components(bs,dict(path_edges=bs[0]['records']),context=ctx,target_s=0.)
        side=[c for c in result['components'] if c.get('branch_id')==bs[1]['branch_id']]
        self.assertEqual([c['role'] for c in side],['SIDE_OPEN_BRANCH'])

    def choose(self, rows):
        from competitive_surface_selection import choose_successor
        return choose_successor(rows)

    def join(self, paths):
        from competitive_surface_selection import reliable_junction
        from main_track_assembly import _oriented
        bs = fixture(paths)[2]
        ms = [branch_metrics(b, local_edge_scale(bs)) for b in bs]
        a, b = [_oriented(x['records']) for x in bs]
        j, audit = reliable_junction(a, b, bs[0], bs[1], ms[0], ms[1],
            required=bs[0]['z_range'], locked_range=(np.inf, -np.inf), direction='upper')
        return bs, j, audit

    def test_successor_selected_by_future_continuity_when_both_in_ASC(self):
        rows = [dict(branch_id=i, MBG_pass=True, face_continuity=5, in_ASC=True,
                     forward_Z_m=z, forward_ASC_m=z, minimum_switches=k)
                for i,z,k in [(1,1.,0),(2,3.,1)]]
        self.assertEqual(self.choose(rows)['selected']['branch_id'], 2)

    def test_minimum_distance_does_not_choose_surface_identity(self):
        rows = [dict(branch_id=i, MBG_pass=True, face_continuity=n, in_ASC=True,
                     forward_Z_m=2.,forward_ASC_m=2.,minimum_switches=0,distance_m=d)
                for i,n,d in [(1,5,.009),(2,3,.0001)]]
        self.assertEqual(self.choose(rows)['selected']['branch_id'],1)

    def test_junction_search_uses_dual_ASC_overlap(self):
        _,j,audit=self.join([line(0,0,2),line(.004,.5,3)])
        self.assertIsNotNone(j)
        self.assertTrue(j['from_in_ASC_at_junction'] and j['to_in_ASC_at_junction'])
        self.assertEqual(j['junction_reason'],'DUAL_ASC_MIN_DISTANCE')
        self.assertLess(j['a_point_uz'][1],1.)

    def test_new_branch_endpoint_guard_is_not_used_for_early_switch(self):
        _,j,audit=self.join([line(0,0,1),line(.004,.95,2)])
        self.assertIsNone(j)
        self.assertEqual(audit['junction_reason'],'UNRESOLVED')

    def test_exact_intersection_in_dual_ASC_beats_later_gap(self):
        _,j,_=self.join([line(0,0,2),dense_path([[-.04,.3],[.04,1.2],[.006,3.]])])
        self.assertIsNotNone(j)
        self.assertEqual(j['junction_reason'],'REAL_INTERSECTION')
        self.assertLess(j['xyz_distance_m'],1e-9)

    def test_tail_to_candidate_ASC_handoff_allowed(self):
        a=line(0,0,1,21)
        b=dense_path([[.005,.85],[.005,2.]])
        _,j,_=self.join([a,b])
        self.assertIsNotNone(j)
        self.assertFalse(j['from_in_ASC_at_junction'])
        self.assertTrue(j['to_in_ASC_at_junction'])
        self.assertEqual(j['junction_reason'],'TAIL_TO_ASC_MIN_DISTANCE')

    def test_no_safe_ASC_junction_remains_unresolved(self):
        _,j,audit=self.join([line(0,0,2),line(.012,.5,3)])
        self.assertIsNone(j)
        self.assertGreater(audit['candidate_min_distance_m'],.010)

    def test_equal_identities_remain_ambiguous(self):
        rows=[dict(branch_id=i,MBG_pass=True,face_continuity=5,in_ASC=True,
                   forward_Z_m=2.,forward_ASC_m=2.,minimum_switches=0) for i in (1,2)]
        self.assertTrue(self.choose(rows)['ambiguous'])

    def test_safe_intersection_at_allowed_ASC_boundary_is_kept(self):
        from competitive_surface_selection import reliable_junction
        bs=fixture([line(0,0,2),dense_path([[-.4,0],[2.6,3]])])[2]
        ma=branch_metrics(bs[0],.05);mb=branch_metrics(bs[1],.05)
        ma.update(ASC_start_arc=.2,ASC_end_arc=1.8,minimum_core_arc_length=.2)
        mb.update(ASC_start_arc=.1,ASC_end_arc=3.8,minimum_core_arc_length=.1)
        j,_=reliable_junction(bs[0]['records'],bs[1]['records'],bs[0],bs[1],ma,mb,
            required=[0,2],locked_range=(np.inf,-np.inf),direction='upper')
        self.assertIsNotNone(j)
        self.assertLess(j['xyz_distance_m'],1e-9)
        self.assertAlmostEqual(j['a_point_uz'][1],.4)

    def test_internal_competitor_away_from_seed_is_evaluated_and_retained(self):
        from scipy import sparse
        from physical_face_context import PhysicalFaceContext
        from observed_surface_graph import build_surface_graph
        from surface_spine_pipeline import recognize_surface_spine
        bs=fixture([line(0,0,10,101),line(.005,2,4,81)])[2]
        profiles={s:(bs if s==0 else [bs[1]]) for s in (-.1,-.05,0.,.05,.1)}
        count=sum(len(b['records']) for b in bs);a=sparse.lil_matrix((count,count))
        low=np.zeros((count,3));high=np.zeros_like(low)
        for b in bs:
            ids=[e['face_id'] for e in b['records']]
            for x,y in zip(ids,ids[1:]):a[x,y]=a[y,x]=1
            for e in b['records']:
                xyz=np.asarray(e['points_uz']);low[e['face_id']]=[-.15,xyz[:,0].min(),xyz[:,1].min()]
                high[e['face_id']]=[.15,xyz[:,0].max(),xyz[:,1].max()]
        regions=[dict(region_id='INTERNAL',bounds=[-.15,.15,-.1,.1,2.,4.])]
        g=build_surface_graph([dict(s=s,branches=b) for s,b in profiles.items()])
        g['physical_face_context']=PhysicalFaceContext(a.tocsr(),low,high,profiles,regions)
        result=recognize_surface_spine(bs,g,0.)
        self.assertEqual(result['route']['route_branch_sequence'],[0,1,0])
        self.assertEqual(result['route']['route_z_extent'],[0.,10.])
        self.assertTrue(result['route']['local_competition_audit'])

    def test_production_hanging_output_supplies_existing_consumer_schema(self):
        from surface_spine_pipeline import hanging_group_adapter
        records=fixture([dense_path([[1,0],[0,1]])])[2][0]['records']
        groups=hanging_group_adapter(records)
        self.assertTrue(groups)
        group=groups[0][0]
        self.assertIn('group_segments',group)
        self.assertIn('node_order',group)
        self.assertEqual(len(group['group_segments']),len(records))
        import ast
        from pathlib import Path
        from collections import defaultdict
        root=Path(__file__).resolve().parents[1]
        for name in ['04_structure_recognition/extract_line_face_links.py','05_reconstruction_volume/merge_structural_surfaces.py']:
            tree=ast.parse((root/'scripts'/name).read_text(encoding='utf-8-sig'))
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gather_line_data')
            namespace={'defaultdict':defaultdict}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),name,'exec'),namespace)
            lines,faces=namespace['gather_line_data']({'red_groups_corrected':groups})
            self.assertEqual(len(lines),len(records))
            self.assertEqual(set(faces),{f for r in records for f in r['source_face_ids']})

    def test_p2_closed_and_unused_open_are_conserved(self):
        from main_spine_components import classify_components, reconstruction_inputs
        from dominant_observed_branch import route_result
        bs=fixture([line(0,0,2),line(.1,.5,1.5),[[1,0],[2,0],[2,1],[1,1],[1,0]]])[2]
        layers=classify_components(bs,route_result(bs[0]['records']))
        self.assertTrue(layers['source_intervals_preserved'])
        roles={r['role'] for r in layers['components']}
        self.assertIn('SIDE_CLOSED_COMPONENT',roles)
        self.assertIn('AMBIGUOUS_COMPONENT',roles)
        bundle=reconstruction_inputs(layers,[])
        self.assertEqual(bundle['side_components'],layers['SIDE_COMPONENTS'])


if __name__=='__main__':unittest.main()
