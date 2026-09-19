"""Generate both orientations using the shared parallel slicing engine.

All ranges are stop-exclusive, like the original np.arange scanline range.
The output directory must be new. vertical_slices.pkl is the legacy dictionary
keyed by two-decimal arc position; colliding rounded keys are rejected.
horizontal_slices.pkl is a sequence of independent pickle records: a metadata
header followed by engine result dictionaries in completion order. Read the
header with pickle.load, then load exactly header['record_count'] records.
metadata.json is written only after both outputs complete successfully.
"""

from __future__ import annotations

import argparse
from itertools import chain
import json
import os
from pathlib import Path
import pickle

import numpy as np

from parallel_slice_engine import ParallelSliceEngine, prepare_mesh, _validated_arc_config


def _positions(start, stop, step, axis):
    if not np.isfinite([start, stop, step]).all() or step <= 0 or stop < start:
        raise ValueError(f"{axis} range needs finite start <= stop and positive step")
    return np.arange(start, stop, step, dtype=np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--arc-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vertical-start", required=True, type=float)
    parser.add_argument("--vertical-stop", required=True, type=float)
    parser.add_argument("--vertical-step", required=True, type=float)
    parser.add_argument("--horizontal-step", required=True, type=float)
    parser.add_argument("--horizontal-start", type=float)
    parser.add_argument("--horizontal-stop", type=float)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1 or (os.name == "nt" and args.workers > 61):
        parser.error("--workers must be positive (and at most 61 on Windows)")
    # Numerical-library threads are limited in spawned children before import.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with args.arc_config.open("r", encoding="utf-8") as stream:
        arc_config = _validated_arc_config(json.load(stream))
    horizontal_start = args.horizontal_start
    horizontal_stop = args.horizontal_stop
    if horizontal_start is None:
        horizontal_start = arc_config["z_range"][0]
    if horizontal_stop is None:
        horizontal_stop = arc_config["z_range"][1]
    vertical_positions = _positions(
        args.vertical_start, args.vertical_stop, args.vertical_step, "vertical"
    )
    horizontal_positions = _positions(
        horizontal_start, horizontal_stop, args.horizontal_step, "horizontal"
    )
    vertical_keys = [f"{position:.2f}" for position in vertical_positions]
    if len(set(vertical_keys)) != len(vertical_keys):
        raise ValueError("Vertical positions collide at the legacy two-decimal key precision")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    mesh_metadata = prepare_mesh(args.mesh, args.arc_config, output_dir / "mesh_cache")
    metadata = {
        "format_version": 1,
        "mesh": mesh_metadata,
        "workers": args.workers,
        "max_pending": args.workers * 2,
        "ranges": {
            "vertical": {
                "start": args.vertical_start, "stop": args.vertical_stop,
                "step": args.vertical_step,
            },
            "horizontal": {
                "start": horizontal_start, "stop": horizontal_stop,
                "step": args.horizontal_step,
            },
        },
        "stop_exclusive": True,
        "slice_counts": {"vertical": len(vertical_positions), "horizontal": len(horizontal_positions)},
        "files": {
            "vertical": "vertical_slices.pkl", "horizontal": "horizontal_slices.pkl",
        },
        "vertical_schema": "dict[two-decimal position] -> {plane_params, slicing}",
        "horizontal_schema": "pickle header then record_count independent engine result records",
        "crop": "none; all branches and both vertical half-planes are retained",
    }
    specs = chain(
        ({"axis": "vertical", "position": float(pos)} for pos in vertical_positions),
        ({"axis": "horizontal", "position": float(pos)} for pos in horizontal_positions),
    )
    # The legacy vertical dictionary is retained for compatibility. Horizontal
    # records and completed futures are consumed immediately, without a list.
    vertical_data = {}
    with (output_dir / "horizontal_slices.pkl").open("xb") as horizontal_stream:
        pickle.dump({
            "format": "gds-horizontal-slices",
            "format_version": 1,
            "record_count": len(horizontal_positions),
            "metadata": metadata,
        }, horizontal_stream, protocol=pickle.HIGHEST_PROTOCOL)
        with ParallelSliceEngine(mesh_metadata["cache_dir"], arc_config, args.workers) as engine:
            for result in engine.iter_slices(specs):
                if result["axis"] == "horizontal":
                    pickle.dump(result, horizontal_stream, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    vertical_data[f"{result['position']:.2f}"] = {
                        "plane_params": result["plane_params"],
                        "slicing": {
                            "lines_3d": result["lines_3d"],
                            "face_ids": result["face_ids"],
                        },
                    }
    # Dictionary iteration order is stable even when workers finish out of order.
    vertical_data = {key: vertical_data[key] for key in vertical_keys}
    with (output_dir / "vertical_slices.pkl").open("xb") as stream:
        pickle.dump(vertical_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata["status"] = "complete"
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved {len(vertical_positions)} vertical and {len(horizontal_positions)} horizontal slices to {output_dir}")
    return metadata


if __name__ == "__main__":
    main()
