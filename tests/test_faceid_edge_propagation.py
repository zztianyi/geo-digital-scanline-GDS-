"""Synthetic regression tests for FaceID propagation and BFS edge identity."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "04_structure_recognition" / "run_multi_profile_recognition.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import memory_profiler  # noqa: F401
except ModuleNotFoundError:
    memory_profiler_stub = types.ModuleType("memory_profiler")
    memory_profiler_stub.profile = lambda function: function
    sys.modules["memory_profiler"] = memory_profiler_stub

SPEC = importlib.util.spec_from_file_location("gds_faceid_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

ORIGIN = np.zeros(3)
RADIAL_DIR = np.array([1.0, 0.0, 0.0])
VERTICAL_DIR = np.array([0.0, 1.0, 0.0])


def edge(p1, p2, face_id):
    return (tuple(p1), tuple(p2), face_id)


def red_face_ids(segments):
    return [segment[6] for segment in segments]


def assert_segments_equal(test_case, left, right):
    test_case.assertEqual(len(left), len(right))
    for left_segment, right_segment in zip(left, right):
        for index in range(6):
            test_case.assertTrue(np.array_equal(left_segment[index], right_segment[index]))
        test_case.assertEqual(left_segment[6], right_segment[6])


class FaceIDPropagationTests(unittest.TestCase):
    def test_normal_single_path_and_direct_hit(self):
        subset = [
            edge((1, 1, 0), (0.5, 0.5, 0), 11),
            edge((0.5, 0.5, 0), (0, 0, 0), 12),
        ]
        nodes, indices = RUNNER.ArcUtils.order_nonclosed_path_with_edges(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        self.assertEqual(nodes, [(1, 1, 0), (0.5, 0.5, 0), (0, 0, 0)])
        self.assertEqual(indices, [0, 1])
        legacy = RUNNER.ArcUtils.get_ordered_red_segments_for_path_legacy(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        diagnostics = {}
        fast = RUNNER.ArcUtils.get_ordered_red_segments_for_path(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR, diagnostics=diagnostics
        )
        assert_segments_equal(self, legacy, fast)
        self.assertEqual(red_face_ids(fast), [11, 12])
        self.assertEqual(diagnostics["fast_direct_face_hits"], 2)

    def test_reversed_edges(self):
        subset = [
            edge((0.5, 0.5, 0), (1, 1, 0), 21),
            edge((0, 0, 0), (0.5, 0.5, 0), 22),
        ]
        nodes, indices = RUNNER.ArcUtils.order_nonclosed_path_with_edges(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        self.assertEqual(nodes[0], (1, 1, 0))
        self.assertEqual(nodes[-1], (0, 0, 0))
        self.assertEqual(indices, [0, 1])
        legacy = RUNNER.ArcUtils.get_ordered_red_segments_for_path_legacy(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        fast = RUNNER.ArcUtils.get_ordered_red_segments_for_path(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        assert_segments_equal(self, legacy, fast)

    def test_multi_node_bfs_and_edge_consistency(self):
        subset = [
            edge((1, 1, 0), (0, 0, 0), 31),
            edge((1, 1, 0), (0.5, 0.5, 0), 32),
            edge((0.5, 0.5, 0), (0, 0, 0), 33),
        ]
        nodes, indices = RUNNER.ArcUtils.order_nonclosed_path_with_edges(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        self.assertEqual(nodes, [(1, 1, 0), (0, 0, 0)])
        self.assertEqual(indices, [0])
        for index, edge_index in enumerate(indices):
            actual = subset[edge_index]
            self.assertEqual({nodes[index], nodes[index + 1]}, {actual[0], actual[1]})

    def test_exact_duplicate_same_face_id(self):
        subset = [
            edge((1, 1, 0), (0.5, 0.5, 0), 41),
            edge((1, 1, 0), (0.5, 0.5, 0), 41),
            edge((0.5, 0.5, 0), (0, 0, 0), 42),
        ]
        diagnostics = {}
        fast = RUNNER.ArcUtils.get_ordered_red_segments_for_path(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR, diagnostics=diagnostics
        )
        legacy = RUNNER.ArcUtils.get_ordered_red_segments_for_path_legacy(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        assert_segments_equal(self, legacy, fast)
        self.assertEqual(diagnostics["duplicate_exact_edge_pairs"], 1)
        self.assertEqual(diagnostics.get("duplicate_edge_pairs_different_face_ids", 0), 0)

    def test_exact_duplicate_different_face_ids(self):
        subset = [
            edge((1, 1, 0), (0.5, 0.5, 0), 51),
            edge((1, 1, 0), (0.5, 0.5, 0), 52),
            edge((0.5, 0.5, 0), (0, 0, 0), 53),
        ]
        diagnostics = {}
        fast = RUNNER.ArcUtils.get_ordered_red_segments_for_path(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR, diagnostics=diagnostics
        )
        legacy = RUNNER.ArcUtils.get_ordered_red_segments_for_path_legacy(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        assert_segments_equal(self, legacy, fast)
        self.assertEqual(diagnostics["duplicate_exact_edge_pairs"], 1)
        self.assertEqual(diagnostics["duplicate_edge_pairs_different_face_ids"], 1)
        self.assertEqual(red_face_ids(fast), [51, 53])

    def test_fuzzy_duplicate_uses_legacy_first_match_fallback(self):
        subset = [
            edge((0.999999, 0.999999, 0), (0.5, 0.5, 0), 61),
            edge((1, 1, 0), (0.5, 0.5, 0), 62),
            edge((0.5, 0.5, 0), (0, 0, 0), 63),
        ]
        diagnostics = {}
        fast = RUNNER.ArcUtils.get_ordered_red_segments_for_path(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR, diagnostics=diagnostics
        )
        legacy = RUNNER.ArcUtils.get_ordered_red_segments_for_path_legacy(
            subset, ORIGIN, RADIAL_DIR, VERTICAL_DIR
        )
        assert_segments_equal(self, legacy, fast)
        self.assertGreaterEqual(diagnostics["ambiguous_edge_count"], 1)
        self.assertGreaterEqual(diagnostics["fallback_lookup_calls"], 1)
        self.assertEqual(red_face_ids(fast), [61, 63])


if __name__ == "__main__":
    unittest.main()
