"""Table-only segmentation contracts; no mesh or geometry fixture."""
import sys,unittest
from pathlib import Path
from dataclasses import FrozenInstanceError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/04_structure_recognition'))


def scores(votes,tracks=(1,2),indices=None):
    indices=range(len(votes)) if indices is None else indices
    return [dict(region_id='R',s=round(i*.05,8),s_index=i,FaceTrack=t,present=True,MBG=True,
                 slice_vote=v,slice_ambiguous=v is None,through_region=False,in_CC=False,exit_continuation_m=0.)
            for i,v in zip(indices,votes) for t in tracks]


class LocalPreferenceTests(unittest.TestCase):
    def analyze(self,votes,policy=None,indices=None):
        from facetrack_local_preference import segment_region,reduce_subregion,LocalPreferencePolicy
        rows=scores(votes,indices=indices);policy=policy or LocalPreferencePolicy(strong_consensus_fraction=None)
        regions=segment_region('R',rows,policy)
        return regions,[reduce_subregion(r,rows,policy) for r in regions]

    def test_persistent_reversal_splits_at_start_of_new_stable_run(self):
        regions,ds=self.analyze([1]*4+[2]*4)
        self.assertEqual([d.dominant_FaceTrack for d in ds],[1,2])
        self.assertEqual([r.s_indices for r in regions],[(0,1,2,3),(4,5,6,7)])

    def test_ambiguous_buffer_is_not_assigned_or_bridged(self):
        regions,ds=self.analyze([1]*3+[None]*2+[1]*3)
        self.assertEqual([d.dominant_FaceTrack for d in ds],[1,None,1])
        self.assertEqual(regions[1].ambiguous_buffer_V,(.15,.2))

    def test_isolated_reverse_vote_is_retained_as_dissent_without_split(self):
        regions,ds=self.analyze([1,1,1,2,1,1,1])
        self.assertEqual(len(regions),1);self.assertEqual(regions[0].local_dissent_V,(.15,))
        self.assertEqual(ds[0].opposing_V,(.15,))

    def test_real_missing_v_is_hard_boundary(self):
        regions,_=self.analyze([1]*6,indices=[0,1,2,4,5,6])
        self.assertEqual(len(regions),2)

    def test_changed_competing_set_is_hard_boundary(self):
        from facetrack_local_preference import segment_region,LocalPreferencePolicy
        rows=scores([1]*3)+scores([1]*3,tracks=(1,3),indices=[3,4,5])
        regions=segment_region('R',rows,LocalPreferencePolicy())
        self.assertEqual([r.competing_FaceTracks for r in regions],[(1,2),(1,3)])

    def test_threshold_is_configurable(self):
        from facetrack_local_preference import LocalPreferencePolicy
        regions,_=self.analyze([1]*3+[2]*3+[1]*3,LocalPreferencePolicy(min_preference_run_V=4,strong_consensus_fraction=None))
        self.assertEqual(len(regions),1)

    def test_votes_beat_large_through_advantage(self):
        from facetrack_local_preference import segment_region,reduce_subregion,LocalPreferencePolicy
        rows=scores([1,1,1,2,1,1,1]);policy=LocalPreferencePolicy()
        for r in rows:r['through_region']=r['FaceTrack']==2
        decision=reduce_subregion(segment_region('R',rows,policy)[0],rows,policy)
        self.assertEqual(decision.dominant_FaceTrack,1);self.assertEqual(dict(decision.vote_counts),{1:6,2:1})

    def test_ties_stay_ambiguous_and_outputs_are_immutable(self):
        regions,ds=self.analyze([None]*4)
        self.assertIsNone(ds[0].dominant_FaceTrack)
        with self.assertRaises(FrozenInstanceError):ds[0].dominant_FaceTrack=1
        with self.assertRaises(FrozenInstanceError):regions[0].s_start=0

    def test_d_style_strong_consensus_keeps_dissent_and_one_region(self):
        from facetrack_local_preference import LocalPreferencePolicy
        votes=[1]*300+[2]*3+[1]*300
        regions,ds=self.analyze(votes,LocalPreferencePolicy(strong_consensus_fraction=.99))
        self.assertEqual(len(regions),1);self.assertEqual(len(ds[0].opposing_V),3)
        self.assertEqual(regions[0].kind,'STRONG_CONSENSUS_REGION')

    def test_strong_consensus_never_swallows_an_ambiguous_buffer(self):
        from facetrack_local_preference import LocalPreferencePolicy
        regions,ds=self.analyze([1]*200+[None]+[1]*200,LocalPreferencePolicy())
        self.assertEqual(len(regions),3);self.assertIsNone(ds[1].dominant_FaceTrack)

    def test_no_votes_are_lost_and_row_order_does_not_matter(self):
        from facetrack_local_preference import segment_region,LocalPreferencePolicy
        rows=scores([2,1,1,1,2,2,1,1,1,None,2,2,2]);p=LocalPreferencePolicy()
        regions=segment_region('R',rows,p)
        self.assertEqual(regions,segment_region('R',rows[::-1],p))
        self.assertEqual([i for r in regions for i in r.s_indices],list(range(13)))

    def test_conflicting_duplicate_slice_votes_are_rejected(self):
        from facetrack_local_preference import segment_region,LocalPreferencePolicy
        rows=scores([1]*3);rows[1]['slice_vote']=2
        with self.assertRaises(ValueError):segment_region('R',rows,LocalPreferencePolicy())

    def test_short_vote_between_buffers_does_not_become_new_frozen_winner(self):
        regions,ds=self.analyze([None,1,None])
        self.assertEqual(len(regions),3);self.assertTrue(all(d.ambiguous for d in ds))
        self.assertEqual(dict(ds[1].vote_counts),{1:1,2:0})

    def test_original_short_parent_is_distinguished_from_new_fragment(self):
        regions,ds=self.analyze([1,1])
        self.assertEqual(regions[0].origin,'UNCHANGED_INITIAL_REGION')
        self.assertEqual(regions[0].kind,'INHERITED_SHORT_REGION')
        self.assertEqual(ds[0].dominant_FaceTrack,1)


if __name__=='__main__':unittest.main()
