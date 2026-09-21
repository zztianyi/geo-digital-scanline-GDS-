"""Shadow choices remain ambiguous on ties; excursions preserve original sources."""
import sys,unittest
from pathlib import Path
import numpy as np
from test_dominant_observed_branch import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/99_experiments'))
try:
    import spine_shadow_validation as shadow
except ImportError:
    shadow=None


class SpineShadowValidationTests(unittest.TestCase):
    def require_api(self):
        self.assertIsNotNone(shadow,'shadow validation functions are not implemented')

    def test_face_continuity_precedes_sweet_position_and_exact_tie_is_ambiguous(self):
        self.require_api()
        def row(i,n,sweet=True):
            return dict(candidate_id=str(i),MBG=True,face_continuity=n,in_ASC=sweet,
                        forward_ASC_m=1.,forward_Z_m=1.,minimum_switches=0)
        r=shadow.choose_sweet([row(0,5,False),row(1,4,True)])
        self.assertEqual(r['selected'],'0')
        r=shadow.choose_sweet([row(0,5),row(1,5)])
        self.assertEqual(r['status'],'AMBIGUOUS');self.assertIsNone(r['selected'])

    def test_MBG_rejection_cannot_be_rescued_by_face_continuity(self):
        self.require_api()
        rows=[dict(candidate_id='bad',MBG=False,face_continuity=5,in_ASC=True,forward_ASC_m=9.,forward_Z_m=9.,minimum_switches=0),
              dict(candidate_id='good',MBG=True,face_continuity=1,in_ASC=True,forward_ASC_m=1.,forward_Z_m=1.,minimum_switches=1)]
        self.assertEqual(shadow.choose_sweet(rows)['selected'],'good')

    def test_single_profile_near_return_needs_no_neighbor_permission(self):
        self.require_api()
        b=fixture([[[0,0],[0,1],[.3,1],[.3,1.3],[.001,1.001],[0,2]]])[2][0]
        events=shadow.near_returns(b,max_gap_m=.002,min_arc_m=.5,min_ratio=50)
        self.assertTrue(events)
        r=shadow.peel_branch(b,events,closure_gap_m=.002,min_arc_m=.5,min_ratio=50)
        self.assertTrue(r['SIDE_COMPONENT']);self.assertTrue(r['continuous'])
        self.assertTrue(r['source_intervals_preserved'])
        self.assertLessEqual(r['max_bridge_m'],.010)
        self.assertTrue(all(e.get('source_face_ids') for e in r['SIDE_COMPONENT']))

    def test_20mm_detection_does_not_authorize_over_10mm_bridge(self):
        self.require_api()
        b=fixture([[[0,0],[0,1],[.3,1],[.3,1.3],[.015,1.015],[.015,2]]])[2][0]
        events=shadow.near_returns(b,max_gap_m=.020,min_arc_m=.5,min_ratio=20)
        r=shadow.peel_branch(b,events,closure_gap_m=.020,min_arc_m=.5,min_ratio=20)
        self.assertTrue(r['SIDE_COMPONENT'])
        self.assertFalse(r['continuous']);self.assertTrue(r['unresolved'])
        self.assertTrue(r['source_intervals_preserved'])
        self.assertTrue(all(e['source'].startswith('OBSERVED') or np.linalg.norm(np.diff(e['points_xyz'],axis=0))<=.010 for e in r['MAIN_SPINE']))

    def test_same_excursion_uses_legal_cut_when_wider_scan_also_finds_illegal_cut(self):
        self.require_api()
        b=fixture([[[0,0],[0,1],[.3,1],[.3,1.3],[.008,1.008],[.008,2]]])[2][0]
        events=shadow.near_returns(b,max_gap_m=.020,min_arc_m=.5,min_ratio=50)
        self.assertTrue(any(e['D_xyz_m']>.010 for e in events))
        r=shadow.peel_branch(b,events,closure_gap_m=.020,min_arc_m=.5,min_ratio=50)
        self.assertTrue(r['continuous'])
        self.assertTrue(r['SIDE_COMPONENT']);self.assertLessEqual(r['max_bridge_m'],.010)

    def test_short_bridge_cannot_dominate_a_retained_main_spine_stub(self):
        self.require_api()
        b=fixture([[[0,0],[.3,0],[.3,.3],[.009,.009],[.009,.010]]])[2][0]
        events=shadow.near_returns(b,max_gap_m=.010,min_arc_m=.5,min_ratio=20)
        r=shadow.peel_branch(b,events,closure_gap_m=.010,min_arc_m=.5,min_ratio=20)
        self.assertTrue(r['SIDE_COMPONENT'])
        self.assertFalse(r['continuous'])
        self.assertTrue(any(e['reason']=='REJECT_SYNTHETIC_DOMINANCE' for e in r['unresolved']))


if __name__=='__main__':unittest.main()
