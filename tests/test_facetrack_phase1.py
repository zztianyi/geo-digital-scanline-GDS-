"""Behavior contracts for evidence-only physical FaceTrack voting."""
import sys,unittest
from pathlib import Path
from dataclasses import FrozenInstanceError
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))


def qualitative_fixture(exits=(1.,.99),coverage=(10,10),cores=(True,True),mbg=(True,True)):
    """Hand-built 10-cell mask: literal full/partial witnessed coverage."""
    branches={};cells={}
    for bid,(exit_m,n,cc,valid) in enumerate(zip(exits,coverage,cores,mbg)):
        length=1+n+exit_m
        branches[bid]=dict(branch_id=bid,MBG=valid,CC_start_arc=0.,CC_end_arc=length if cc else .5,
                           full_arc_length=length,forward_sign=1)
        for i in range(n):cells.setdefault((i,0),{})[bid+16]=[(bid,1.+i,2.+i)]
    return dict(region_id='fixture',tracks=[16,17],cells=[[0,i,0] for i in range(10)]),dict(branches=branches,cells=cells)


def score_qualitative_fixture(args):
    from facetrack_phase1_evidence import score_slice,Phase1Policy
    region,geometry=qualitative_fixture(**args)
    return score_slice(region,0,0.,geometry,Phase1Policy())


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

    def test_good_vs_good_does_not_compare_exact_exit_length(self):
        rows=score_qualitative_fixture(dict(exits=(1.,.99)))
        self.assertEqual([r['continuity_class'] for r in rows],['CONTINUITY_GOOD']*2)
        self.assertTrue(all(r['slice_ambiguous'] and r['slice_vote'] is None for r in rows))

    def test_good_vs_weak_rejects_weak(self):
        rows=score_qualitative_fixture(dict(exits=(1.,0.),coverage=(10,4)))
        self.assertEqual(rows[0]['slice_vote'],16)
        self.assertEqual(rows[1]['continuity_class'],'CONTINUITY_WEAK')
        self.assertEqual(rows[1]['elimination_reason'],'REJECT_CONTINUITY_WEAK')
        self.assertTrue(rows[1]['eliminated'])

    def test_same_continuity_same_cc_becomes_ambiguous(self):
        for cores in ((True,True),(False,False)):
            rows=score_qualitative_fixture(dict(exits=(.6,.9),cores=cores))
            self.assertIsNone(rows[0]['slice_vote'])
            self.assertTrue(all(r['elimination_reason']=='TIE_AMBIGUOUS' and not r['eliminated'] for r in rows))

    def test_through_boolean_alone_cannot_win_against_equivalent_good_track(self):
        rows=score_qualitative_fixture(dict(coverage=(10,9)))
        self.assertTrue(rows[0]['through_region']);self.assertFalse(rows[1]['through_region'])
        self.assertEqual(rows[0]['continuity_class'],rows[1]['continuity_class'])
        self.assertIsNone(rows[0]['slice_vote'])

    def test_cc_can_decide_between_continuity_equivalent_candidates(self):
        rows=score_qualitative_fixture(dict(exits=(.4,1.),cores=(True,False)))
        self.assertEqual(rows[0]['slice_vote'],16)
        self.assertEqual(rows[1]['eliminated_by'],'CC')
        self.assertEqual(rows[1]['elimination_reason'],'LOSE_CC')

    def test_micro_difference_decision_count_zero(self):
        from facetrack_phase1_evidence import micro_difference_decisions
        rows=score_qualitative_fixture(dict(exits=(1.,.99)))
        self.assertEqual(micro_difference_decisions(rows),[])
        for row in rows:row.update(slice_vote=16,final_slice_vote=16,slice_ambiguous=False)
        self.assertEqual(len(micro_difference_decisions(rows)),1,'audit must detect an injected forbidden winner')

    def test_decision_trace_records_elimination_stage(self):
        rows=score_qualitative_fixture(dict(mbg=(True,False)))
        self.assertEqual(rows[1]['stage_reached'],'MBG')
        self.assertEqual(rows[1]['eliminated_by'],'MBG')
        self.assertEqual(rows[1]['elimination_reason'],'REJECT_MBG')
        self.assertEqual({r['final_slice_vote'] for r in rows},{16})

    def test_A3_parallel_result_deterministic(self):
        fixtures=[dict(exits=(1.,.99)),dict(coverage=(10,9)),dict(cores=(True,False)),
                  dict(exits=(1.,0.),coverage=(10,4))]
        serial=[score_qualitative_fixture(f) for f in fixtures]
        with ProcessPoolExecutor(max_workers=2) as pool:
            parallel=list(pool.map(score_qualitative_fixture,fixtures[::-1]))
        self.assertEqual(serial,parallel[::-1])
        self.assertEqual([r[0]['slice_vote'] for r in serial],[None,None,16,16])

    def test_all_weak_candidates_do_not_produce_a_winner(self):
        rows=score_qualitative_fixture(dict(exits=(0.,0.),coverage=(3,4),cores=(True,False)))
        self.assertIsNone(rows[0]['slice_vote'])
        self.assertTrue(all(r['eliminated_by']=='CONTINUITY' for r in rows))

    def test_witness_choice_cannot_hide_an_exact_length_tiebreak(self):
        from facetrack_phase1_evidence import score_slice,Phase1Policy
        region,geometry=qualitative_fixture(exits=(.75,.9))
        geometry['branches'][2]=dict(geometry['branches'][0],branch_id=2,full_arc_length=11.9,CC_end_arc=11.9)
        for i in range(10):geometry['cells'][i,0][16].append((2,1.+i,2.+i))
        rows=score_slice(region,0,0.,geometry,Phase1Policy())
        self.assertEqual(rows[0]['witness_branch'],0)
        self.assertIsNone(rows[0]['slice_vote'])

    def test_tiny_mask_reentry_gap_is_not_a_continuity_failure(self):
        from facetrack_phase1_evidence import score_slice,Phase1Policy
        region,geometry=qualitative_fixture()
        geometry['cells'][4,0][17]=[(1,5.,5.4),(1,5.41,6.)]
        rows=score_slice(region,0,0.,geometry,Phase1Policy())
        self.assertFalse(rows[1]['through_region'])
        self.assertEqual(rows[1]['continuous_interval_count'],2)
        self.assertEqual(rows[1]['continuity_class'],'CONTINUITY_GOOD')
        self.assertIsNone(rows[0]['slice_vote'])

    def test_good_beats_uncertain_without_exact_comparison(self):
        rows=score_qualitative_fixture(dict(coverage=(10,7),exits=(.5,1.)))
        self.assertEqual(rows[1]['continuity_class'],'CONTINUITY_UNCERTAIN')
        self.assertEqual(rows[1]['elimination_reason'],'LOSE_CONTINUITY_CLASS')
        self.assertEqual(rows[0]['slice_vote'],16)

    @staticmethod
    def rows():
        return [dict(region_id='r',s=0.,s_index=0,FaceTrack=t,present=True,MBG=True,through_region=False,
            in_CC=False,exit_continuation_m=0.,slice_vote=None,slice_ambiguous=True) for t in (1,2)]


if __name__=='__main__':unittest.main()
