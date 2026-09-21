"""Face provenance is topological identity, never a numeric-ID proximity rule."""
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/99_experiments'))


class FaceProvenanceAuditTests(unittest.TestCase):
    def index(self):
        from face_provenance_audit import FaceAdjacency
        # 0--2--3 share edges; 1 only touches vertex 0; 4 is disconnected.
        return FaceAdjacency.build(np.array([[0,1,2],[0,7,8],[1,2,3],[2,3,4],[9,10,11]]))

    def test_complete_provenance_exact_overlap_not_first_face_only(self):
        self.assertEqual(self.index().relation([1,0],[4,0],set(range(5)))['relation'],'EXACT_FACE')

    def test_true_shared_edge_not_adjacent_numeric_id(self):
        index=self.index()
        self.assertEqual(index.relation([0],[2],set(range(5)))['relation'],'EDGE_ADJACENT_FACE')
        result=index.relation([0],[1],set(range(5)))
        self.assertEqual(result['relation'],'GEOMETRY_ONLY')
        self.assertTrue(result['weak_vertex_contact'])

    def test_local_chain_is_bounded_and_cannot_leave_roi(self):
        index=self.index()
        self.assertEqual(index.relation([0],[3],{0,2,3},max_hops=2)['chain'],[0,2,3])
        self.assertEqual(index.relation([0],[3],{0,3},max_hops=8)['relation'],'GEOMETRY_ONLY')
        self.assertEqual(index.relation([0],[3],{0,2,3},max_hops=1)['relation'],'GEOMETRY_ONLY')

    def test_disconnected_and_ambiguous_provenance_stay_separate(self):
        index=self.index();tracks=index.local_labels({0,1,2,3,4})
        self.assertEqual(tracks[0],tracks[3])
        self.assertNotEqual(tracks[0],tracks[4])
        self.assertNotEqual(tracks[0],tracks[1])
        self.assertIsNone(index.unique_track([0,4],tracks))
        self.assertEqual(index.unique_track([0,2],tracks),tracks[0])

    def test_repeated_vertex_does_not_create_false_shared_edge(self):
        from face_provenance_audit import FaceAdjacency
        index=FaceAdjacency.build(np.array([[0,0,1],[0,2,3]]))
        self.assertEqual(index.relation([0],[1],{0,1})['relation'],'GEOMETRY_ONLY')


if __name__=='__main__':unittest.main()
