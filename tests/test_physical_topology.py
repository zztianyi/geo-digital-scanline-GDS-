"""Degenerate provenance must not turn a physical open path into a fork."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/04_structure_recognition'))
from vertical_profile_canonicalization import canonicalize_vertical
from dominant_observed_branch import extract_observed_branches
from branch_absolute_core import branch_metrics, local_edge_scale


def chain_with_loop():
    points = np.array([[0., 0., z] for z in range(16)])
    lines = np.concatenate((np.stack((points[:-1], points[1:]), axis=1),
                            [[points[7], points[7]]]*2))
    return canonicalize_vertical(lines, list(range(15))+[90, 91])


class PhysicalTopologyTests(unittest.TestCase):
    def test_zero_length_self_loop_keeps_provenance(self):
        p = chain_with_loop()
        self.assertEqual(p.source_face_ids[-1], [90, 91])
        self.assertEqual(p.source_segment_indices[-1], [15, 16])
        self.assertEqual(p.first_face_ids[-1], 90)
        np.testing.assert_array_equal(p.lines_xyz[-1], [[0, 0, 7], [0, 0, 7]])

    def test_zero_length_self_loop_not_in_physical_degree(self):
        p = chain_with_loop()
        self.assertEqual(p.raw_degree[7], 6)
        self.assertEqual(p.canonical_degree[7], 4)
        self.assertEqual(p.physical_degree[7], 2)
        self.assertFalse(p.topology_active_edge[-1])

    def test_zero_length_self_loop_does_not_make_open_chain_forked(self):
        p = chain_with_loop()
        bs = extract_observed_branches(p, p.nodes[:, [0, 2]])
        self.assertEqual(len(bs), 1)
        self.assertEqual(bs[0]['kind'], 'OPEN_SURFACE_BRANCH')
        self.assertTrue(branch_metrics(bs[0], local_edge_scale(bs))['MBG_pass'])
        self.assertEqual(bs[0]['edge_order'], list(range(15)))

    def test_physical_component_unchanged_except_degenerate_self_loop(self):
        p = chain_with_loop()
        np.testing.assert_array_equal(p.physical_components, p.components)
        q = canonicalize_vertical(np.array([[[4, 0, 0], [4, 0, 0]],
                                           [[8, 0, 0], [8, 0, 1]]]), [5, 6])
        np.testing.assert_array_equal(q.physical_components, [0, 1, 1])
        self.assertEqual(len(q.physical_branches), 1)
        self.assertEqual(q.physical_branches[0]['edges'], [1])

    def test_real_fork_and_nonzero_closed_cycle_keep_their_types(self):
        points = np.array([[0,0,0],[1,0,0],[2,0,0],[1,0,1],
                           [5,0,0],[6,0,0],[6,0,1]], dtype=float)
        p = canonicalize_vertical(points[[[0,1],[1,2],[1,3],[4,5],[5,6],[6,4]]], range(6))
        bs = extract_observed_branches(p, p.nodes[:, [0,2]])
        self.assertEqual([b['kind'] for b in bs], ['FORKED_COMPONENT']*3+['CLOSED_COMPONENT'])
        np.testing.assert_array_equal(p.physical_degree, p.degree)

    def _real(self, key, minimum_length, probe_z):
        with np.load(ROOT/'tests/fixtures/physical_topology_real_slices.npz') as f:
            p = canonicalize_vertical(f['lines_'+key], f['faces_'+key])
        # Radial coordinates are recovered from the frozen arc, not fitted.
        center = np.array([-107.95298795807327, -72.60829311946561])
        u = np.linalg.norm(p.nodes[:,:2]-center, axis=1)-116.46711082541985
        bs = extract_observed_branches(p, np.column_stack((u,p.nodes[:,2])))
        scale = local_edge_scale(bs)
        recovered = [b for b in bs if b['full_arc_length']>minimum_length
                     and b['z_range'][0]<probe_z<b['z_range'][1]
                     and branch_metrics(b,scale)['MBG_pass']]
        self.assertTrue(recovered, key+' long observed candidate was excluded')

    def test_real_085_long_branch_recovers_open_candidate(self):
        self._real('0.85', 77., 1400.)

    def test_real_7840_long_branch_recovers_candidate(self):
        self._real('78.40', 67., 1446.)


if __name__ == '__main__':
    unittest.main()
