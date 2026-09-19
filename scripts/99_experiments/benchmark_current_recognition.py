"""Measure the current production recognition kernel with its 15/40/600 scheduler.

Uses frozen slices and a fresh output directory; never invokes production main.
GUI imports are omitted, numerical function/class bodies are compiled unchanged.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import ast
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import functools
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'scripts/04_structure_recognition/run_multi_profile_recognition.py'
_kernel = None
_timings = None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def worker_init():
    import numpy as np
    import trimesh
    global _kernel, _timings
    tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name != 'main']
    namespace = dict(np=np, trimesh=trimesh, math=math, deque=deque, json=json,
                     pickle=pickle, defaultdict=defaultdict, __name__='recognition_benchmark')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    _timings = defaultdict(lambda: [0, 0.])
    # These sequential ArcSlicer phases do not call one another. The wall times
    # are worker sums, not end-to-end latency, and are not summed with parents.
    def measured(name, fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            tick = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _timings[name][0] += 1
                _timings[name][1] += time.perf_counter()-tick
        return call
    for name in ('load_data', 'extract_nodes', 'segment_paths', 'compute_normals',
                 'extract_and_group_red_segments', 'correct_red_groups'):
        cls = namespace['ArcSlicer']
        setattr(cls, name, measured('ArcSlicer.'+name, getattr(cls, name)))
    cls = namespace['ArcUtils']
    cls.compute_2d_centroids = staticmethod(measured('ArcUtils.compute_2d_centroids', cls.compute_2d_centroids))
    _kernel = namespace['process_single_slice']


def recognize_block(payload):
    task_id, block = payload
    _timings.clear()
    tick, cpu = time.perf_counter(), time.process_time()
    results, slices = [], []
    for record in block:
        # ArcSlicer intentionally consumes slicing.lines_3d in place.
        input_segments = len(record['slicing']['lines_3d'])
        start = time.perf_counter()
        results.append(_kernel(record))
        slices.append(dict(slice_key=record['slice_key'], seconds=time.perf_counter()-start,
                           input_segments=input_segments))
    return results, dict(task_id=task_id, worker_pid=os.getpid(), slice_count=len(block),
        compute_seconds=time.perf_counter()-tick, cpu_seconds=time.process_time()-cpu,
        functions=[dict(function=k, calls=v[0], worker_seconds=v[1]) for k, v in _timings.items()], slices=slices)


def run(args):
    import numpy as np
    import psutil
    from validate_orthogonal_scanline_constraint import StageSampler, save_csv, save_json
    output, source = args.output.resolve(), args.input.resolve()
    if output == source.parent or output.is_relative_to(source.parent) or source.parent.is_relative_to(output):
        raise ValueError('Output must be separate from frozen inputs')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'manifest.json', dict(created_at=datetime.now().astimezone().isoformat(),
        input=str(source), input_bytes=source.stat().st_size, input_sha256=digest(source),
        source=str(SOURCE), source_sha256=digest(SOURCE), harness_sha256=digest(__file__),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        workers=15, block_size=40, batch_size=600, scheduler='production_batch_pool_ordered_map',
        numeric_threads=1, platform=platform.platform(), python=platform.python_version(),
        logical_cpus=psutil.cpu_count(), physical_cores=psutil.cpu_count(logical=False),
        total_ram_gib=psutil.virtual_memory().total/2**30,
        scope='Current production numerical kernel; GUI imports omitted; frozen slicing input; no surface-track adapter',
        timing_note='One instrumented run; lightweight sequential phase timers; no warmup or repetitions'))
    stages, task_rows, slice_rows, phase_rows = [], [], [], []
    with StageSampler('slice_load', output, stages):
        with source.open('rb') as stream:
            all_slices = pickle.load(stream)
        keys = [f'{s:.2f}' for s in np.arange(0, 140, .05)]
        if set(keys) != set(all_slices):
            raise ValueError('Expected all 2800 frozen slices; refusing a partial run')
        records = [dict(all_slices[k], slice_key=k) for k in keys]
        input_segments = sum(len(r['slicing']['lines_3d']) for r in records)
    completed, write_seconds = set(), 0.
    with StageSampler('recognition_and_write', output, stages):
        with (output/'recognition.pkl').open('xb') as stream:
            for batch_start in range(0, len(records), 600):
                batch = records[batch_start:batch_start+600]
                blocks = [(batch_start//40+i//40, batch[i:i+40]) for i in range(0, len(batch), 40)]
                with ProcessPoolExecutor(max_workers=15, initializer=worker_init) as pool:
                    for expected, (results, row) in zip(blocks, pool.map(recognize_block, blocks)):
                        received = [r['slice_key'] for r in results]
                        if received != [r['slice_key'] for r in expected[1]] or completed.intersection(received):
                            raise AssertionError('Recognition output missing, duplicated, or out of order')
                        tick = time.perf_counter()
                        for result in results:
                            pickle.dump(result, stream)
                        row['parent_write_seconds'] = time.perf_counter()-tick
                        write_seconds += row['parent_write_seconds']
                        completed.update(received)
                        slice_rows.extend(row.pop('slices'))
                        phase_rows.extend(dict(task_id=row['task_id'], **r) for r in row.pop('functions'))
                        task_rows.append(row)
                        save_json(output/'progress.json', dict(completed=len(completed), total=len(keys)))
                        print(f'Recognized {len(completed)}/{len(keys)}', flush=True)
                        del results, result
    if completed != set(keys):
        raise AssertionError('Incomplete recognition')
    phases = defaultdict(lambda: [0, 0.])
    for row in phase_rows:
        phases[row['function']][0] += row['calls']
        phases[row['function']][1] += row['worker_seconds']
    total_compute = sum(r['compute_seconds'] for r in task_rows)
    totals = sorted([dict(function=k, calls=v[0], worker_seconds=v[1],
        fraction_of_worker_compute=v[1]/total_compute) for k, v in phases.items()], key=lambda r: -r['worker_seconds'])
    save_csv(output/'phase_totals.csv', totals)
    save_csv(output/'task_timings.csv', task_rows)
    save_csv(output/'slice_timings.csv', slice_rows)
    save_csv(output/'phase_timings.csv', phase_rows)
    stage = stages[-1]
    summary = dict(completed_slices=len(completed), expected_slices=len(keys), input_segments=input_segments,
        total_stage_seconds=sum(r['wall_seconds'] for r in stages), **stage,
        slices_per_second=len(keys)/stage['wall_seconds'],
        peak_rss_gib=stage['peak_tree_rss_bytes']/2**30,
        peak_private_gib=stage['peak_tree_private_bytes']/2**30,
        parent_write_seconds_inside_wall=write_seconds,
        worker_compute_seconds_sum=total_compute, worker_cpu_seconds_sum=sum(r['cpu_seconds'] for r in task_rows),
        per_slice_worker_seconds={f'p{p}':float(np.percentile([r['seconds'] for r in slice_rows], p)) for p in (50, 95, 99, 100)},
        pool_count=5, task_count=len(task_rows), output_bytes=(output/'recognition.pkl').stat().st_size,
        source_unchanged=digest(SOURCE)==json.loads((output/'manifest.json').read_text(encoding='utf-8'))['source_sha256'])
    save_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('D:/project_python/rock_recognize/outputs/faceid_full_performance/20260911_114911_926012/slices.pkl'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
