"""Behavior contracts for evidence-only physical FaceTrack voting."""
import sys,unittest
from pathlib import Path
from dataclasses import FrozenInstanceError
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))


class FaceTrackPhase1Tests(unittest.TestCase):
    def test_segment_cells_follow_segments_not_bounding_box(self):
        from facetrack_phase1_evidence import segment_cells
        parts=segment_cells(np.array([[.1,.1],[1.9,1.9]]),1.)
        self.assertEqual({p[:2] for p in parts},{(0,0),(1,1)})
        self.assertAlmostEqual(sum(p[4]-p[3] for p in parts),np.sqrt(2)*1.8)

    def test_conflicts_stop_at_missing_slice_and_changed_group_set(self):
        from facetrack_phase1_regions import build_regions
        def cell(i,t):return dict(s_index=i,u_bin=0,z_bin=0,tracks=t,branch_ids=list(range(len(t))),branch_tracks=t)
        regions=build_regions([cell(0,[1,2]),cell(1,[1,2]),cell(3,[1,2]),cell(4,[1,2,3]),cell(5,[1])],[i*.05 for i in range(6)],.5)
        self.assertEqual(sorted(len(r['cells']) for r in regions),[1,1,2])
        self.assertFalse(any(5 in r['s_indices'] for r in regions))

    def test_same_physical_group_is_not_competition(self):
        from facetrack_phase1_regions import build_regions
        rows=[dict(s_index=0,u_bin=0,z_bin=0,tracks=[7],branch_ids=[1,2],branch_tracks=[7,7])]
        self.assertEqual(build_regions(rows,[0.],.5),[])

    def test_scoring_short_terminal_cannot_claim_through_region(self):
        from facetrack_phase1_evidence import score_slice,Phase1Policy
        geometry=dict(branches={0:dict(branch_id=0,MBG=True,CC_start_arc=.1,CC_end_arc=1.9,full_arc_length=2.,forward_sign=1),
                               1:dict(branch_id=1,MBG=True,CC_start_arc=.1,CC_end_arc=.9,full_arc_length=1.,forward_sign=1)},
            cells={(0,0):{3:[(0,.5,1.5)],4:[(1,0.,1.)]}})
        region=dict(region_id='r',tracks=[3,4],cells=[[0,0,0]],s_indices=[0])
        rows=score_slice(region,0,0.,geometry,Phase1Policy())
        self.assertTrue(rows[0]['through_region']);self.assertFalse(rows[1]['through_region'])
        self.assertEqual({r['slice_vote'] for r in rows},{3})
        self.assertLessEqual(rows[0]['exit_continuation_m'],1.)

    def test_reducer_prioritizes_through_support_over_continuation(self):
        from facetrack_phase1_vote import reduce_region
        rows=self.rows();rows[0]['through_region']=True;rows[1]['exit_continuation_m']=1.
        decision=reduce_region('r',rows)
        self.assertEqual(decision.dominant_FaceTrack,1)

    def test_reentry_does_not_claim_contiguous_through(self):
        from facetrack_phase1_evidence import score_slice,Phase1Policy
        branch=dict(MBG=True,CC_start_arc=.1,CC_end_arc=9.9,full_arc_length=10.,forward_sign=1)
        geometry=dict(branches={0:branch},cells={(0,0):{3:[(0,1.,2.),(0,3.,4.)]}})
        row=score_slice(dict(region_id='r',tracks=[3],cells=[[0,0,0]]),0,0.,geometry,Phase1Policy())[0]
        self.assertFalse(row['through_region']);self.assertTrue(row['in_CC'])

    def test_partial_mask_coverage_can_still_be_entirely_in_core(self):
        from facetrack_phase1_evidence import score_slice,Phase1Policy
        branch=dict(MBG=True,CC_start_arc=.1,CC_end_arc=9.9,full_arc_length=10.,forward_sign=1)
        geometry=dict(branches={0:branch},cells={(0,0):{3:[(0,1.,2.)]}})
        row=score_slice(dict(region_id='r',tracks=[3],cells=[[0,0,0],[0,1,0]]),0,0.,geometry,Phase1Policy())[0]
        self.assertEqual(row['cell_coverage_fraction'],.5)
        self.assertFalse(row['through_region']);self.assertTrue(row['in_CC'])

    def test_tie_stays_ambiguous_and_decision_is_immutable(self):
        from facetrack_phase1_vote import reduce_region
        decision=reduce_region('r',self.rows())
        self.assertTrue(decision.ambiguous);self.assertIsNone(decision.dominant_FaceTrack)
        with self.assertRaises(FrozenInstanceError):decision.dominant_FaceTrack=1

    def test_reducer_does_not_depend_on_slice_order_or_votes_as_geometry(self):
        from facetrack_phase1_vote import reduce_region
        rows=self.rows();rows[0]['in_CC']=True
        self.assertEqual(reduce_region('r',rows),reduce_region('r',rows[::-1]))

    def test_continuous_coverage_does_not_bridge_a_missing_v(self):
        from facetrack_phase1_vote import longest_run
        self.assertEqual(longest_run([0,1,3,4,5],{i:i*.05 for i in range(6)}),(3,.1))

    @staticmethod
    def rows():
        return [dict(region_id='r',s=0.,s_index=0,FaceTrack=t,present=True,MBG=True,through_region=False,
            in_CC=False,exit_continuation_m=0.,slice_vote=None,slice_ambiguous=True) for t in (1,2)]


if __name__=='__main__':unittest.main()
