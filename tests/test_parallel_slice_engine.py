"""Tiny synthetic checks for the shared orthogonal slicing engine."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "03_slicing_profiles"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_scanline_slices import compute_slice_plane, filter_mesh_by_z_range
from generate_orthogonal_scanlines import main as orthogonal_main
from parallel_slice_engine import ParallelSliceEngine, prepare_mesh


class ParallelSliceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="gds-engine-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.config = {
            "center": [0.3, -0.2],
            "radius": 2.3,
            "angle_min": 0.17,
            "angle_max": 1.7,
            "z_range": [-0.6, 0.8],
        }
        cls.config_path = cls.root / "arc.json"
        cls.config_path.write_text(json.dumps(cls.config), encoding="utf-8")
        meshes = []
        for offset in ([2.5, 0.2, 0.2], [-2.5, -0.5, 0.1], [0, 0, 6]):
            box = trimesh.creation.box(extents=[2, 2, 2])
            box.apply_translation(offset)
            meshes.append(box)
        mesh = trimesh.util.concatenate(meshes)
        mesh.faces = mesh.faces[::-1]
        cls.mesh_path = cls.root / "boxes.ply"
        mesh.export(cls.mesh_path)
        cls.original = trimesh.load_mesh(cls.mesh_path)
        cls.filtered = filter_mesh_by_z_range(
            cls.original, *cls.config["z_range"]
        )
        cls.metadata = prepare_mesh(
            cls.mesh_path, cls.config_path, cls.root / "cache"
        )

    def serial_section(self, axis, position):
        if axis == "vertical":
            config = self.config
            params = compute_slice_plane(
                np.append(config["center"], 0.0),
                config["radius"],
                config["angle_min"],
                [0, config["radius"] * (config["angle_max"] - config["angle_min"])],
                position,
            )
        else:
            params = {
                "origin": np.array([0.0, 0.0, position]),
                "normal": np.array([0.0, 0.0, 1.0]),
            }
        lines, faces = trimesh.intersections.mesh_plane(
            self.filtered, params["normal"], params["origin"], return_faces=True
        )
        return params, lines, faces

    def test_preparation_preserves_filtered_face_order_and_refuses_overwrite(self):
        cache = Path(self.metadata["cache_dir"])
        np.testing.assert_array_equal(
            np.load(cache / "vertices.npy", allow_pickle=False), self.filtered.vertices
        )
        np.testing.assert_array_equal(
            np.load(cache / "faces.npy", allow_pickle=False), self.filtered.faces
        )
        self.assertEqual(self.metadata["filtered_face_count"], len(self.filtered.faces))
        self.assertLess(len(self.filtered.faces), len(self.original.faces))
        np.testing.assert_array_equal(self.metadata["bounds"], self.filtered.bounds)
        self.assertEqual(
            json.loads((cache / "metadata.json").read_text(encoding="utf-8")),
            self.metadata,
        )
        before = (cache / "faces.npy").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_mesh(self.mesh_path, self.config_path, cache)
        self.assertEqual((cache / "faces.npy").read_bytes(), before)

    def test_scene_preparation_applies_transforms_before_filtering(self):
        scene = trimesh.Scene()
        for offset in ([4.0, 0.0, 0.0], [-4.0, 0.0, 0.0]):
            scene.add_geometry(
                trimesh.creation.box(),
                transform=trimesh.transformations.translation_matrix(offset),
            )
        scene_path = self.root / "scene.glb"
        scene.export(scene_path)
        original = trimesh.load_mesh(scene_path)
        if isinstance(original, trimesh.Scene):
            original = original.dump(concatenate=True)
        filtered = filter_mesh_by_z_range(original, *self.config["z_range"])
        metadata = prepare_mesh(scene_path, self.config_path, self.root / "scene-cache")
        cache = Path(metadata["cache_dir"])
        np.testing.assert_array_equal(
            np.load(cache / "vertices.npy", allow_pickle=False), filtered.vertices
        )
        np.testing.assert_array_equal(
            np.load(cache / "faces.npy", allow_pickle=False), filtered.faces
        )
        np.testing.assert_array_equal(metadata["bounds"], [[-4.5, -0.5, -0.5], [4.5, 0.5, 0.5]])

    def test_one_and_two_workers_match_ordered_trimesh_sections(self):
        specs = [
            {"axis": "vertical", "position": position}
            for position in [0.0, 0.41, 1.2, -0.8, 3.519]
        ] + [
            {"axis": "horizontal", "position": position}
            for position in [float(self.filtered.vertices[:, 2].min()), -0.6, 0.0, 0.31, 0.8, 0.95, 100.0]
        ]
        runs = []
        for workers in (1, 2):
            with self.subTest(workers=workers):
                with ParallelSliceEngine(self.metadata["cache_dir"], self.config, workers) as engine:
                    self.assertEqual(list(engine.iter_slices([])), [])
                    results = list(engine.iter_slices(specs))
                    # A second batch must reuse the initialized worker pool.
                    again = list(engine.iter_slices(specs[:1]))[0]
                by_key = {(item["axis"], item["position"]): item for item in results}
                self.assertEqual(len(by_key), len(specs))
                pids = {item["worker_pid"] for item in results}
                self.assertNotIn(os.getpid(), pids)
                self.assertLessEqual(len(pids), workers)
                if workers == 1:
                    self.assertEqual(pids, {again["worker_pid"]})
                for spec in specs:
                    key = (spec["axis"], spec["position"])
                    result = by_key[key]
                    params, lines, faces = self.serial_section(*key)
                    np.testing.assert_array_equal(result["lines_3d"], lines)
                    np.testing.assert_array_equal(result["face_ids"], faces)
                    self.assertEqual(result["lines_3d"].dtype, np.dtype("float64"))
                    self.assertEqual(result["face_ids"].dtype, np.dtype("int64"))
                    self.assertGreaterEqual(result["compute_seconds"], 0.0)
                    for name, value in params.items():
                        np.testing.assert_array_equal(result["plane_params"][name], value)
                empty = by_key[("horizontal", 100.0)]
                self.assertEqual(empty["lines_3d"].shape, (0, 2, 3))
                self.assertEqual(empty["face_ids"].shape, (0,))
                vertical = by_key[("vertical", 0.0)]
                params = vertical["plane_params"]
                radial = (vertical["lines_3d"] - params["origin"]) @ params["radial_dir"]
                self.assertLess(radial.min(), 0.0)
                self.assertGreater(radial.max(), 0.0)
                # Filtering retains whole intersecting triangles, not a clipped Z slab.
                self.assertGreater(len(by_key[("horizontal", 0.95)]["face_ids"]), 0)
                runs.append(by_key)
        for key in runs[0]:
            np.testing.assert_array_equal(runs[0][key]["lines_3d"], runs[1][key]["lines_3d"])
            np.testing.assert_array_equal(runs[0][key]["face_ids"], runs[1][key]["face_ids"])

    def test_empty_filtered_mesh(self):
        config = dict(self.config, z_range=[20.0, 21.0])
        config_path = self.root / "empty-arc.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        metadata = prepare_mesh(self.mesh_path, config_path, self.root / "empty-cache")
        self.assertEqual(metadata["filtered_face_count"], 0)
        self.assertIsNone(metadata["bounds"])
        with ParallelSliceEngine(metadata["cache_dir"], config) as engine:
            results = list(engine.iter_slices([
                {"axis": "vertical", "position": 0.0},
                {"axis": "horizontal", "position": 20.5},
            ]))
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["lines_3d"].shape, (0, 2, 3))
            self.assertEqual(result["face_ids"].shape, (0,))

    def test_error_gates_worker_exception_and_closed_pool(self):
        for workers in (0, -1, 1.5, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                ParallelSliceEngine(self.metadata["cache_dir"], self.config, workers)
        for spec in (
            {"axis": "oblique", "position": 0.0},
            {"axis": "vertical", "position": float("nan")},
            {"axis": "horizontal", "position": float("inf")},
            {"axis": "vertical"},
        ):
            with self.subTest(spec=spec):
                with ParallelSliceEngine(self.metadata["cache_dir"], self.config) as engine:
                    with self.assertRaises(ValueError):
                        list(engine.iter_slices([spec]))
        engine = ParallelSliceEngine(self.metadata["cache_dir"], self.config)
        with self.assertRaises(RuntimeError):
            list(engine.iter_slices([]))
        with engine:
            self.assertEqual(list(engine.iter_slices([])), [])
        with self.assertRaises(RuntimeError):
            list(engine.iter_slices([]))
        # Finite inputs can still overflow the derived angle inside a worker.
        config = dict(self.config, radius=1e-300)
        with ParallelSliceEngine(self.metadata["cache_dir"], config) as engine:
            with self.assertRaises(ValueError):
                list(engine.iter_slices([{"axis": "vertical", "position": 1e300}]))
            with self.assertRaises(RuntimeError):
                list(engine.iter_slices([]))

    def test_bounded_submission_and_early_iterator_close(self):
        consumed = []

        def specs():
            for index in range(30):
                consumed.append(index)
                yield {"axis": "horizontal", "position": index / 100.0}

        with ParallelSliceEngine(self.metadata["cache_dir"], self.config, workers=2) as engine:
            iterator = engine.iter_slices(specs())
            first = next(iterator)
            self.assertEqual(first["axis"], "horizontal")
            self.assertLessEqual(len(consumed), 2 * engine.workers + 1)
            iterator.close()

    def test_cli_vertical_schema_horizontal_stream_and_no_overwrite(self):
        output = self.root / "cli-output"
        args = [
            "--mesh", str(self.mesh_path), "--arc-config", str(self.config_path),
            "--output-dir", str(output), "--vertical-start", "0", "--vertical-stop", "0.6",
            "--vertical-step", "0.2", "--horizontal-start", "-0.2",
            "--horizontal-stop", "0.5", "--horizontal-step", "0.25", "--workers", "2",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            orthogonal_main(args)
        with (output / "vertical_slices.pkl").open("rb") as stream:
            vertical = pickle.load(stream)
        self.assertEqual(list(vertical), ["0.00", "0.20", "0.40"])
        for key, item in vertical.items():
            self.assertEqual(set(item), {"plane_params", "slicing"})
            _, lines, faces = self.serial_section("vertical", float(key))
            np.testing.assert_array_equal(item["slicing"]["lines_3d"], lines)
            np.testing.assert_array_equal(item["slicing"]["face_ids"], faces)
        with (output / "horizontal_slices.pkl").open("rb") as stream:
            header = pickle.load(stream)
            self.assertEqual(header["format"], "gds-horizontal-slices")
            self.assertEqual(header["record_count"], 3)
            horizontal = [pickle.load(stream) for _ in range(header["record_count"])]
            with self.assertRaises(EOFError):
                pickle.load(stream)
        np.testing.assert_allclose(
            sorted(row["position"] for row in horizontal), [-0.2, 0.05, 0.3], rtol=0, atol=1e-15
        )
        for row in horizontal:
            _, lines, faces = self.serial_section("horizontal", row["position"])
            np.testing.assert_array_equal(row["lines_3d"], lines)
            np.testing.assert_array_equal(row["face_ids"], faces)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["slice_counts"], {"vertical": 3, "horizontal": 3})
        before = (output / "vertical_slices.pkl").read_bytes()
        with self.assertRaises(FileExistsError):
            orthogonal_main(args)
        self.assertEqual((output / "vertical_slices.pkl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
