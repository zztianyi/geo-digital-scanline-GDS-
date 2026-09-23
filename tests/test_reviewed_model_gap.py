import unittest
from copy import deepcopy
import numpy as np
from test_dominant_observed_branch import fixture
from test_absolute_sweet_zone import line


class ReviewedModelGapTests(unittest.TestCase):
    def setUp(self):
        from main_track_assembly import _oriented
        self.bs=fixture([line(0,0,2,41),line(.3,2.4,5,53)])[2]
        self.a,self.b=[_oriented(b['records']) for b in self.bs]
        self.approval=dict(review_id='review',reason='confirmed measurement hole',direction='upper',
            current_branch_id=self.bs[0]['branch_id'],target_branch_id=self.bs[1]['branch_id'],
            endpoint_xyz=[self.a[-1]['points_xyz'][1],self.b[0]['points_xyz'][0]])

    def test_unapproved_and_changed_endpoints_are_not_joined(self):
        from reviewed_model_gap import approved_endpoint_extension
        self.assertIsNone(approved_endpoint_extension(self.a,[self.b],self.bs,[]))
        changed=deepcopy(self.approval);changed['endpoint_xyz']=np.asarray(changed['endpoint_xyz'])+.001
        self.assertIsNone(approved_endpoint_extension(self.a,[self.b],self.bs,[changed]))

    def test_exact_endpoint_bridge_is_not_a_measured_structure(self):
        from reviewed_model_gap import approved_endpoint_extension,valid_estimated_gap
        from main_spine_components import reconstruction_inputs
        from surface_spine_pipeline import hanging_group_adapter
        records,j=approved_endpoint_extension(self.a,[self.b],self.bs,[self.approval]);gap=records[len(self.a)]
        self.assertTrue(valid_estimated_gap(gap));self.assertAlmostEqual(j['xyz_distance_m'],.5)
        self.assertEqual(gap['source_face_ids'],[])
        result=reconstruction_inputs(dict(MAIN_SPINE=records,SIDE_COMPONENTS=[],source_intervals_preserved=True),[gap])
        self.assertEqual(result['hanging_segments'],[]);self.assertEqual(len(result['structural_main_spine_runs']),2)
        self.assertEqual(hanging_group_adapter([gap]),{0:[]})
        np.testing.assert_array_equal(records[len(self.a)-1]['points_xyz'],self.a[-1]['points_xyz'])
        np.testing.assert_array_equal(records[len(self.a)+1]['points_xyz'],self.b[0]['points_xyz'])

    def test_estimated_segment_breaks_hanging_groups(self):
        from reviewed_model_gap import approved_endpoint_extension
        from surface_spine_pipeline import hanging_group_adapter
        records,_=approved_endpoint_extension(self.a,[self.b],self.bs,[self.approval])
        self.assertEqual(len(hanging_group_adapter(records)[0]),2)

    def test_locked_transition_reports_the_actual_skip(self):
        from types import SimpleNamespace
        from region_junction_band import apply_region_junction_bands
        a,b=self.bs[0]['branch_id'],self.bs[1]['branch_id']
        connector=dict(source='TOPOLOGY_SWITCH',points_uz=[self.a[-1]['points_uz'][1],self.b[0]['points_uz'][0]])
        context=SimpleNamespace(region_locks={0.:{a:[0.,1.]}},scope_audit=[],
            region_data=lambda r:dict(branches={(0.,a):[0],(0.,b):[1]}))
        region=dict(region_id='test',bounds=[-.1,.1,-1,1,0,6])
        rows=apply_region_junction_bands({'0.00':dict(path_edges=self.a+[connector]+self.b)},
            {'0.00':self.bs},context,[region])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['reason'],'SKIPPED_TRANSITION_TOUCHES_REGION_LOCK')
        self.assertEqual(rows[0]['locked_branch_ids'],[a])


if __name__=='__main__':unittest.main()
