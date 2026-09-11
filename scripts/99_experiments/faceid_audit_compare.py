"""Compare an existing Fast recognition run with frozen reference results.

This tool only reads the two result directories and writes new audit artifacts.
It does not rerun recognition and never overwrites either input directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import pickle
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
LINE_SCRIPT_DIR = ROOT / "scripts" / "04_structure_recognition"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LINE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LINE_SCRIPT_DIR))

from extract_line_face_links import gather_line_data


RECOGNITION_FIELDS = (
    "paths",
    "normals",
    "red_groups",
    "red_groups_corrected",
    "red_centroids",
)


def iter_results(task_dir):
    for path in sorted(Path(task_dir).glob("task_*.pkl")):
        with path.open("rb") as handle:
            while True:
                try:
                    yield pickle.load(handle)
                except EOFError:
                    break


def object_digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def result_field_digests(result):
    return {
        field: object_digest(result.get(field))
        for field in RECOGNITION_FIELDS
    }


def compare_recognition(reference_task_dir, new_task_dir):
    reference = {}
    duplicate_reference_keys = []
    for result in iter_results(reference_task_dir):
        key = str(result.get("slice_key"))
        if key in reference:
            duplicate_reference_keys.append(key)
        reference[key] = result_field_digests(result)

    new_keys = set()
    duplicate_new_keys = []
    field_mismatch_counts = {field: 0 for field in RECOGNITION_FIELDS}
    mismatch_samples = []
    for result in iter_results(new_task_dir):
        key = str(result.get("slice_key"))
        if key in new_keys:
            duplicate_new_keys.append(key)
        new_keys.add(key)
        actual = result_field_digests(result)
        expected = reference.get(key)
        if expected is None:
            mismatch_samples.append({"slice_key": key, "reason": "missing_reference"})
            continue
        for field in RECOGNITION_FIELDS:
            if actual[field] != expected[field]:
                field_mismatch_counts[field] += 1
                if len(mismatch_samples) < 20:
                    mismatch_samples.append({
                        "slice_key": key,
                        "field": field,
                        "reason": "exact_digest_mismatch",
                    })

    missing_new = sorted(set(reference) - new_keys, key=float)
    missing_reference = sorted(new_keys - set(reference), key=float)
    status = (
        not duplicate_reference_keys
        and not duplicate_new_keys
        and not missing_new
        and not missing_reference
        and not any(field_mismatch_counts.values())
    )
    return {
        "status": "PASS" if status else "FAIL",
        "reference_slice_count": len(reference),
        "new_slice_count": len(new_keys),
        "duplicate_reference_keys": duplicate_reference_keys,
        "duplicate_new_keys": duplicate_new_keys,
        "missing_new_keys": missing_new,
        "missing_reference_keys": missing_reference,
        "field_mismatch_counts": field_mismatch_counts,
        "mismatch_samples": mismatch_samples,
        "compared_fields": list(RECOGNITION_FIELDS),
    }


def canonical_segment(segment):
    p1 = tuple(float(value) for value in segment[0])
    p2 = tuple(float(value) for value in segment[1])
    return (p1, p2) if p1 <= p2 else (p2, p1)


def canonical_geometry_digest(segments):
    canonical = sorted(canonical_segment(segment) for segment in segments)
    return object_digest(canonical)


def compare_line_faces(reference_path, new_task_dir, new_output_path):
    with Path(reference_path).open("rb") as handle:
        old_segments, old_face_heights = pickle.load(handle)

    new_segments = []
    new_face_heights = defaultdict(list)
    new_slice_count = 0
    for result in iter_results(new_task_dir):
        segments, face_heights = gather_line_data(result)
        new_segments.extend(segments)
        for face, heights in face_heights.items():
            new_face_heights[int(face)].extend(float(value) for value in heights)
        new_slice_count += 1

    with Path(new_output_path).open("wb") as handle:
        pickle.dump((new_segments, new_face_heights), handle, protocol=pickle.HIGHEST_PROTOCOL)

    old_face_heights = {int(face): list(values) for face, values in old_face_heights.items()}
    old_faces = set(old_face_heights)
    new_faces = set(new_face_heights)
    height_count_mismatch = 0
    height_value_mismatch = 0
    max_abs_height_diff = 0.0
    for face in old_faces | new_faces:
        old_values = sorted(float(value) for value in old_face_heights.get(face, []))
        new_values = sorted(float(value) for value in new_face_heights.get(face, []))
        if len(old_values) != len(new_values):
            height_count_mismatch += 1
            continue
        if old_values != new_values:
            height_value_mismatch += 1
            if old_values:
                max_abs_height_diff = max(
                    max_abs_height_diff,
                    max(abs(left - right) for left, right in zip(old_values, new_values)),
                )

    geometry_equal = canonical_geometry_digest(old_segments) == canonical_geometry_digest(new_segments)
    status = (
        len(old_segments) == len(new_segments)
        and old_faces == new_faces
        and height_count_mismatch == 0
        and height_value_mismatch == 0
        and geometry_equal
    )
    return {
        "status": "PASS" if status else "FAIL",
        "new_slice_count": new_slice_count,
        "old_line_segment_count": len(old_segments),
        "new_line_segment_count": len(new_segments),
        "line_segment_count_mismatch": len(old_segments) != len(new_segments),
        "line_geometry_mismatch": not geometry_equal,
        "old_face_count": len(old_faces),
        "new_face_count": len(new_faces),
        "face_set_mismatch": old_faces != new_faces,
        "height_count_mismatch": height_count_mismatch,
        "height_value_mismatch": height_value_mismatch,
        "max_abs_height_diff": max_abs_height_diff,
        "new_line_faces_path": str(new_output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-task-dir", type=Path, required=True)
    parser.add_argument("--new-task-dir", type=Path, required=True)
    parser.add_argument("--reference-line-faces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recognition = compare_recognition(args.reference_task_dir, args.new_task_dir)
    line_faces = compare_line_faces(
        args.reference_line_faces,
        args.new_task_dir,
        args.output_dir / "line_faces.pkl",
    )
    (args.output_dir / "full_recognition_comparison.json").write_text(
        json.dumps(recognition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "line_face_comparison.json").write_text(
        json.dumps(line_faces, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"recognition": recognition, "line_faces": line_faces}, ensure_ascii=False))
    return 0 if recognition["status"] == "PASS" and line_faces["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
