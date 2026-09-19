"""Small headless recognition/local-track fixtures; no mesh or frozen bulk load."""
import copy
import csv
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/99_experiments'), str(ROOT/'scripts/04_structure_recognition')]
import vertical_canonical_recognition_audit as audit
from vertical_profile_canonicalization import canonicalize_vertical


ARC = {'center': [0., 0.], 'radius': 10., 'angle_min': 0., 'angle_max': 1.}


def frozen_slices(directory, data):
    slices = {key: {'slice_key': key, 'slicing': {'lines_3d': np.asarray(lines, dtype=float),
                                               'face_ids': np.asarray(faces)},
                    'plane_params': {'origin': np.zeros(3), 'radial_dir': np.array([1., 0., 0.]),
                                     'vertical_dir': np.array([0., 0., 1.])}}
              for key, (lines, faces) in data.items()}
    return SimpleNamespace(slices=slices, keys=list(slices), groups={key: [] for key in slices},
                           prior=directory/'prior', baseline=directory/'baseline', arc=ARC)


def read_records(path):
    records = []
    with path.open('rb') as stream:
        while True:
            try:
                records.append(pickle.load(stream))
            except EOFError:
                return records


def track_group(gid, s, high=.2):
    theta = s/10
    nodes = [[10*np.cos(theta), 10*np.sin(theta), z] for z in (0., high)]
    return {'id': gid, 'slice_key': f'{s:.2f}', 's': s, 'node_order': nodes,
            'uz': [[0., 0.], [0., high]], 'z_min': 0., 'z_max': high,
            'length_2d': high, 'audit_observable': True}


def track_fixture(directory, horizontal_indices=(0, 1, 2)):
    keys = ['0.00', '0.05', '0.10', '0.20', '0.25']
    groups = {key: [track_group(key, float(key))] for key in keys}
    levels = np.array([0., .1, .2])
    horizontal = {}
    for zi in horizontal_indices:
        points = np.array([[10*np.cos(float(key)/10), 10*np.sin(float(key)/10), levels[zi]]
                           for key in keys])
        horizontal[zi] = audit.opc.clean_horizontal(np.stack((points[:-1], points[1:]), axis=1),
                                                    list(range(4)), ARC)
    return SimpleNamespace(keys=keys, groups=groups, arc=ARC, levels=levels, horizontal=horizontal,
                           prior=directory/'prior', baseline=directory/'baseline')


class OutputProtectionTests(unittest.TestCase):
    def test_rejects_equal_descendant_and_ancestor_of_either_frozen_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = SimpleNamespace(prior=root/'prior-parent/prior', baseline=root/'baseline-parent/baseline')
            for protected in (frozen.prior, frozen.baseline):
                for output in (protected, protected/'new-audit', protected.parent, root,
                               protected/'child/../audit'):
                    with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'separate'):
                        audit._output(frozen, output, ('result.csv',))
                self.assertFalse(protected.exists())

    def test_allows_sibling_even_when_its_name_has_a_frozen_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = SimpleNamespace(prior=root/'prior', baseline=root/'baseline')
            output = root/'prior-audit'
            self.assertEqual(audit._output(frozen, output, ('result.csv',)), output.resolve())
            self.assertTrue(output.is_dir())


class ChangedRecognitionTests(unittest.TestCase):
    def test_only_changed_slices_streamed_all_rays_retained_and_clean_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            line = [[10., 0., 0.], [11., 0., 1.]]
            opposite = [[-10., 0., 0.], [-11., 0., 1.]]
            frozen = frozen_slices(root, {
                '0.00': ([line, line[::-1], opposite], [42, 43, 44]),
                '0.05': ([line], [45]), '0.10': ([line], [46]),
                '0.15': ([line], [47]), '0.20': ([line], [48]),
            })
            profiles = {key: canonicalize_vertical(row['slicing']['lines_3d'], row['slicing']['face_ids'])
                        for key, row in frozen.slices.items()}
            before = pickle.dumps(frozen.slices)
            extra = {'points_xyz': [[11., 0., 1.], [12., 0., 2.]],
                     'source': 'TOPOLOGY_STITCH', 'face_id': None, 'candidate_id': 'candidate-1'}
            result = audit.run_changed_recognition(frozen, profiles, {'0.05': [extra]}, root/'out')
            self.assertEqual(set(result['groups_by_slice']), {'0.00', '0.05'})
            self.assertEqual(result['summary']['changed_slices'], 2)
            self.assertEqual(result['summary']['source_changed_slices'], 2)
            self.assertEqual(result['summary']['unchanged_slices'], 3)
            self.assertTrue(result['summary']['clean_slice_equivalence'])
            self.assertEqual(result['summary']['clean_equivalence_samples'], 3)
            self.assertEqual(result['summary']['observed_edges_lost'], 0)
            records = read_records(root/'out/changed_vertical_recognition.pkl')
            self.assertEqual([r['slice_key'] for r in records], ['0.00', '0.05'])
            self.assertFalse(any(r['use_merged_paths'] for r in records))
            first_edges = records[0]['slicing']['intersections']
            self.assertEqual([edge[2] for edge in first_edges], [42, 44])
            np.testing.assert_array_equal(first_edges[1][:2], opposite)
            legacy_groups = audit.legacy.summarize_record(records[0], ARC)[0]
            self.assertTrue(audit.legacy.semantically_equal(result['groups_by_slice']['0.00'], legacy_groups))
            self.assertEqual(before, pickle.dumps(frozen.slices))
            self.assertTrue(all(not groups for groups in frozen.groups.values()))
            with (root/'out/changed_vertical_provenance.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            observed = next(r for r in rows if r['slice_key'] == '0.00' and r['edge_index'] == '0')
            self.assertEqual(json.loads(observed['source_face_ids']), [42, 43])
            self.assertEqual(json.loads(observed['source_segment_indices']), [0, 1])

    def test_short_synthetic_red_edge_keeps_none_despite_legacy_allclose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = [1000., 0., 1000.], [1000.000003, 0., 1000.000003], [1000.000006, 0., 1000.000006]
            frozen = frozen_slices(root, {'0.00': ([[a, b]], [42])})
            profile = canonicalize_vertical(frozen.slices['0.00']['slicing']['lines_3d'], [42])
            result = audit.run_changed_recognition(frozen, {'0.00': profile}, {'0.00': [
                {'points_xyz': [b, c], 'source': 'ORTHOGONAL_INFERRED', 'face_id': None},
            ]}, root/'out')
            record = read_records(root/'out/changed_vertical_recognition.pkl')[0]
            segments = [seg for groups in record['red_groups'].values() for group in groups for seg in group]
            faces = {audit._edge_key(seg[4:6]): seg[6] for seg in segments}
            self.assertEqual(faces[audit._edge_key([a, b])], 42)
            self.assertIsNone(faces[audit._edge_key([b, c])])
            groups = result['groups_by_slice']['0.00']
            self.assertTrue(groups)
            self.assertTrue(any(None in g['red_face_ids'] for g in groups))
            self.assertEqual({face for g in groups for face in g['face_ids']}, {42})

    def test_extra_overlapping_observed_geometry_does_not_erase_real_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            line = [[10., 0., 0.], [11., 0., 1.]]
            frozen = frozen_slices(root, {'0.00': ([line], [9])})
            profile = canonicalize_vertical(np.array([line]), [9])
            result = audit.run_changed_recognition(frozen, {'0.00': profile}, {'0.00': [
                {'points_xyz': line[::-1], 'source': 'TOPOLOGY_STITCH', 'face_id': None},
            ]}, root/'out')
            record = read_records(root/'out/changed_vertical_recognition.pkl')[0]
            self.assertEqual(len(record['slicing']['intersections']), 1)
            self.assertEqual(record['slicing']['intersections'][0][2], 9)
            self.assertEqual(result['summary']['synthetic_edges_added'], 0)

    def test_partial_slice_profile_and_fabricated_synthetic_face_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            line = [[10., 0., 0.], [11., 0., 1.]]
            opposite = [[-10., 0., 0.], [-11., 0., 1.]]
            frozen = frozen_slices(root, {'0.00': ([line, opposite], [9, 10])})
            partial = canonicalize_vertical(np.array([line]), [9])
            with self.assertRaisesRegex(ValueError, 'full|observed'):
                audit.run_changed_recognition(frozen, {'0.00': partial}, {}, root/'partial')
            full = canonicalize_vertical(np.array([line, opposite]), [9, 10])
            with self.assertRaisesRegex(ValueError, 'face_id'):
                audit.run_changed_recognition(frozen, {'0.00': full}, {'0.00': [
                    {'points_xyz': line, 'source': 'TOPOLOGY_STITCH', 'face_id': 99},
                ]}, root/'invalid')


class LocalTrackTests(unittest.TestCase):
    def test_only_true_adjacent_pairs_touching_changed_keys_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = track_fixture(root)
            before = copy.deepcopy(frozen.groups)
            changed = {'0.05': copy.deepcopy(frozen.groups['0.05'])}
            result = audit.run_local_track_check(frozen, changed, root/'out')
            self.assertEqual({(r['s_left'], r['s_right']) for r in result['rows']},
                             {(0., .05), (.05, .1)})
            self.assertEqual(result['summary']['local_slice_pairs'], 2)
            self.assertEqual(result['summary']['changed_local_pair_count'], 0)
            self.assertEqual(result['summary']['matched_pairs_after'], 2)
            self.assertEqual(result['summary']['applied_tolerance_m'], 1e-5)
            self.assertTrue(all(r['decision'] == 'RETAINED_SAMPLED' for r in result['rows']))
            self.assertTrue(all(abs(r['C']-1.) < 1e-8 for r in result['rows']))
            self.assertTrue(result['summary']['major_old_groups_geometry_preserved'])
            self.assertIsNone(result['summary']['confirmed_overmerge_reduction'])
            self.assertEqual(frozen.groups, before)

    def test_missing_horizontal_evidence_stays_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = track_fixture(root, horizontal_indices=(1,))
            result = audit.run_local_track_check(frozen, {'0.05': frozen.groups['0.05']}, root/'out')
            self.assertTrue(result['rows'])
            self.assertTrue(all(r['decision'] == 'UNRESOLVED_UNDERSAMPLED' for r in result['rows']))
            self.assertEqual(result['summary']['unresolved_pairs'], 2)
            self.assertEqual(result['summary']['illustrative_removed_pairs'], 0)

    def test_sparse_changed_keys_never_form_a_false_adjacent_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = track_fixture(root)
            changed = {key: frozen.groups[key] for key in ('0.10', '0.20')}
            result = audit.run_local_track_check(frozen, changed, root/'out')
            self.assertEqual({(r['s_left'], r['s_right']) for r in result['rows']},
                             {(.05, .1), (.2, .25)})
            self.assertEqual(result['summary']['local_slice_pairs'], 2)

    def test_sampled_short_bridge_fails_illustrative_gate_without_geological_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = track_fixture(root)
            frozen.groups = {key: [track_group(key, float(key), 1.)] for key in frozen.keys}
            result = audit.run_local_track_check(frozen, {'0.05': frozen.groups['0.05']}, root/'out')
            self.assertEqual(result['summary']['matched_pairs_after'], 2)
            self.assertEqual(result['summary']['illustrative_removed_pairs'], 2)
            self.assertTrue(all(r['decision'] == 'REMOVED_ILLUSTRATIVE' for r in result['rows']))
            self.assertTrue(all(r['failed_gates'] == ['R_bridge'] for r in result['rows']))
            self.assertIsNone(result['summary']['confirmed_overmerge_reduction'])

    def test_lost_group_changes_local_pairs_and_geometry_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = track_fixture(root)
            result = audit.run_local_track_check(frozen, {'0.05': []}, root/'out')
            self.assertEqual(result['summary']['local_slice_pairs'], 2)
            self.assertEqual(result['summary']['changed_local_pair_count'], 2)
            self.assertEqual(result['summary']['matched_pairs_before'], 2)
            self.assertEqual(result['summary']['matched_pairs_after'], 0)
            self.assertFalse(result['summary']['major_old_groups_geometry_preserved'])

    def test_undersampled_short_groups_do_not_trigger_illustrative_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = track_fixture(root)
            changed = {'0.05': [track_group('short', .05, .04)]}
            result = audit.run_local_track_check(frozen, changed, root/'out')
            self.assertTrue(all(r['decision'] == 'UNRESOLVED_UNDERSAMPLED' for r in result['rows']))
            self.assertEqual(result['summary']['illustrative_removed_pairs'], 0)

    def test_no_changes_does_not_recompute_or_emit_remote_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = track_fixture(Path(tmp))
            result = audit.run_local_track_check(frozen, {}, Path(tmp)/'out')
            self.assertEqual(result['rows'], [])
            self.assertEqual(result['summary']['local_slice_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
