"""Read-only full-area census of final V routes and actual H-run correspondence.

No recognition implementation changes. Frozen canonical H runs are built before
crossing all V slices. Each solve sees the unchanged global graph identity and
every raw observation within its exact 1 m consensus radius.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, cProfile, hashlib, json, pickle, sys, time, copy, inspect
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/directional_full_census/20260920'
sys.path.insert(0, str(ROOT/'scripts/99_experiments'))
from locc_data import opc, iter_pickles
from validate_observed_first_locc import target_canonical_profile
from dominant_observed_branch import extract_observed_branches, solve_dominant_branch
from horizontal_surface_link import canonical_horizontal_paths
from recognize_main_tracks import cached_evidence, verify_result
from audit_directional_surface_consensus import verify_geometry


def save_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding='utf-8')


def code_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'scripts/04_structure_recognition').glob('*.py'))}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'routes').mkdir(exist_ok=True)
    started = time.perf_counter()
    prior = ROOT/'outputs/orthogonal_scanline_constraint/20260919_081148'
    manifest = json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
    baseline = Path(manifest['baseline'])
    with (baseline/'slices.pkl').open('rb') as f: slices = pickle.load(f)
    keys = sorted(slices, key=float)
    frozen = SimpleNamespace(prior=prior, arc=manifest['arc'], slices=slices,
        keys=keys, s=np.array(list(map(float, keys))), levels=np.array(manifest['horizontal_levels']))
    graph = cached_evidence(ROOT/'outputs/local_detail_full_validation/20260919_current/surface_audit', frozen)
    with (OUT/'graph.pkl').open('wb') as f: pickle.dump(graph, f, protocol=5)
    offsets = {}
    with (OUT/'branches.pkl').open('wb') as f:
        for i, key in enumerate(keys):
            p = target_canonical_profile(frozen, key)
            bs = extract_observed_branches(p, opc.to_suz(p.nodes, frozen.arc)[:, 1:])
            offsets[key] = f.tell()
            pickle.dump(bs, f, protocol=5)
            if i % 200 == 0: print('V', i, len(keys), round(time.perf_counter()-started, 1), flush=True)
    save_json(OUT/'branch_index.json', offsets)
    del slices, frozen.slices
    arrays, inventory = [], []
    for row in iter_pickles(prior/'horizontal_slices_clean.pkl'):
        zi = row['level_index']
        clean = canonical_horizontal_paths(row['lines_3d'], row['face_ids'], frozen.arc)
        hits = opc.horizontal_crossings(clean, frozen.s, frozen.arc)
        # si, level index, H-run id, u, source face; integers fit exactly in float64.
        a = np.asarray([[si, zi, pid, u, face] for si, u, pid, face in hits], dtype=float).reshape(-1, 5)
        arrays.append(a)
        inventory.append(dict(level_index=zi, edges=len(clean['lines_3d']), runs=clean['path_count'], crossings=len(a)))
        if zi % 100 == 0: print('H', zi, len(frozen.levels), round(time.perf_counter()-started, 1), flush=True)
    a = np.concatenate(arrays)
    a = a[np.argsort(a[:, 0], kind='stable')]
    np.save(OUT/'raw_h.npy', a)
    boundaries = np.searchsorted(a[:, 0], np.arange(len(keys)+1))
    np.save(OUT/'h_offsets.npy', boundaries)
    save_json(OUT/'input_manifest.json', dict(keys=keys, arc=frozen.arc, levels=frozen.levels,
        prior=str(prior), baseline=str(baseline), code_sha256=code_hashes(),
        observations=len(a), prepare_seconds=time.perf_counter()-started, inventory=inventory,
        scope='All frozen 2800 V slices; all frozen H levels; unchanged global track graph; exact ±1 m raw H halo.'))
    print('PREPARED', len(keys), len(a), time.perf_counter()-started, flush=True)


def load_branches(keys):
    index = json.loads((OUT/'branch_index.json').read_text(encoding='utf-8'))
    values = {}
    with (OUT/'branches.pkl').open('rb') as f:
        for key in keys:
            f.seek(index[key]); values[key] = pickle.load(f)
    return values


def block_inputs(start, stop):
    m = json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    assert code_hashes() == m['code_sha256'], 'Recognition code changed since census freeze'
    keys = m['keys']; low, high = max(0, start-20), min(len(keys), stop+20)
    bs = load_branches(keys[low:high])
    with (OUT/'graph.pkl').open('rb') as f: graph = pickle.load(f)
    graph['nodes'] = {(float(k), b['branch_id']): b for k, branches in bs.items() for b in branches}
    a = np.load(OUT/'raw_h.npy', mmap_mode='r')
    off = np.load(OUT/'h_offsets.npy')
    observations = []
    arc = m['arc']; levels = m['levels']
    for si, zi, pid, u, face in a[off[low]:off[high]]:
        s = float(keys[int(si)]); z = levels[int(zi)]
        angle = arc['angle_min']+s/arc['radius']
        observations.append(dict(s=s, z=z, level_index=int(zi), h_path_id=int(pid), u=float(u),
            point_xyz=[arc['center'][0]+(u+arc['radius'])*np.cos(angle),
                       arc['center'][1]+(u+arc['radius'])*np.sin(angle), z],
            face_id=int(face), evidence='RAW_HORIZONTAL_OBSERVATION'))
    graph['observations'] = observations
    return m, bs, graph


_MEMO_CACHES = []


def exact_memoization():
    """Audit-process only: cache identical pure queries, copying every response.

    Full source results remain authoritative. No rounding, pruning, alternate
    candidate scoring, vectorized reduction, or new recognition behavior.
    """
    if _MEMO_CACHES:
        for cache in _MEMO_CACHES: cache.clear()
        return
    import joint_surface_consensus as joint
    import directional_handoff as directional
    # Cache the unchanged per-row, per-radius statistics inside the original
    # function. Leave list order, reductions, means, unions and decision logic
    # exactly as in the frozen source. Profile rows live until block completion.
    scales={};_MEMO_CACHES.append(scales)
    def scale_stat(row,radius,target_s):
        key=(id(row),radius,target_s)
        if key not in scales:
            ss=[s for s in row['support_slices'] if abs(s-target_s)<=radius+1e-8]
            scales[key]=(max(ss)-min(ss),len(ss)-1,min(ss)<target_s-1e-8 and max(ss)>target_s+1e-8)
        return scales[key]
    source=inspect.getsource(joint.summarize_profile)
    old="""            ss = [s for s in r['support_slices'] if abs(s-target_s) <= radius+1e-8]
            spans.append(max(ss)-min(ss)); counts.append(len(ss)-1)
            two.append(min(ss) < target_s-1e-8 and max(ss) > target_s+1e-8)"""
    new="""            span, count, both = _census_scale_stat(r, radius, target_s)
            spans.append(span); counts.append(count); two.append(both)"""
    assert source.count(old)==1,'Frozen summary implementation changed'
    namespace=dict(joint.__dict__,_census_scale_stat=scale_stat)
    exec(compile(source.replace(old,new),'<audit exact row cache>','exec'),namespace)
    joint.summarize_profile=namespace['summarize_profile']
    original = joint.joint_surface_summary; cache = {}; _MEMO_CACHES.append(cache)
    def summary(graph, node, *, arc_interval=None, target_z=None, metrics=None):
        core = (metrics['ASC_start_arc'],metrics['ASC_end_arc']) if metrics and metrics['ASC_exists'] else None
        key = (id(graph),node,tuple(arc_interval) if arc_interval is not None else None,
               tuple(target_z) if target_z is not None else None,core)
        if key not in cache:cache[key]=original(graph,node,arc_interval=arc_interval,target_z=target_z,metrics=metrics)
        return copy.deepcopy(cache[key])
    joint.joint_surface_summary=summary; directional.joint_surface_summary=summary


def one_result(key, branches, graph, m):
    started = time.perf_counter()
    r = solve_dominant_branch(None, None, branches=branches, surface_graph=graph,
        target_s=float(key), input_z_range=m['arc']['z_range'], source_data_required=key=='89.75')
    elapsed = time.perf_counter()-started
    verify_geometry(branches, r)
    # compact output retains every actual observed/connector record and provenance.
    for name in ('fragments', 'excluded_fragments', 'remaining_fragments'):
        if name in r:
            r[name] = [{k:v for k,v in row.items() if k not in ('records','points_xyz','points_uz','node_ids')}
                       for row in r[name]]
    row = dict(slice_key=key, seconds=elapsed, status=r['status'], sequence=r['route_branch_sequence'],
        edges=len(r['path_edges']), switches=r['branch_switch_count'],
        max_connector_m=max((j['xyz_distance_m'] for j in r['junctions']), default=0.),
        max_combined_R=max((b['combined_R_syn'] for b in r['route_contribution_budgets']), default=0.),
        internal_handoffs=r.get('internal_handoff_count', 0),
        unresolved_reasons=r.get('unresolved_reasons', []),
        route_z_extent=r.get('route_z_extent'), extent_covered=r.get('extent_covered'))
    return r, row


def crossing_selections(key, branches, graph, result):
    """Export actual retained edge intervals; never infer choice from branch ID alone.

    Columns: level, H run, u, unique, branch, edge, t, arc, MBG,
    retained, V match count, H hit count. Unique excludes coincident H/V.
    """
    from joint_surface_consensus import _index, _matches
    from branch_absolute_core import local_edge_scale, branch_metrics
    index = _index(graph); s = float(key)
    scale = local_edge_scale(branches)
    mbg = {b['branch_id']:branch_metrics(b, scale)['MBG_pass'] for b in branches}
    retained = {}
    for r in result['path_edges']:
        if r['source'].startswith('OBSERVED'):
            retained.setdefault(r['edge_id'], []).append(sorted((r['t0'], r['t1'])))
    rows = []
    for i in index['by_s'].get(s, ()):
        hit = graph['observations'][i]; found = _matches(graph, index, i)
        nh = len(index['runs'][(hit['level_index'], hit['h_path_id'])][s])
        selected = False
        for node, c in found:
            eid = int(graph['nodes'][node]['edge_order'][c['edge_index']])
            selected |= any(lo-1e-9 <= c['t'] <= hi+1e-9 for lo, hi in retained.get(eid, ()))
        bid = eid = -1; t = arc = float('nan'); eligible = False
        if len(found)==1:
            node, c = found[0]; bid=node[1]
            eid=int(graph['nodes'][node]['edge_order'][c['edge_index']]); t=c['t']; arc=c['arc_position']
            eligible=mbg[bid]
        rows.append([hit['level_index'], hit['h_path_id'], hit['u'], int(len(found)==1 and nh==1),
                     bid, eid, t, arc, eligible, selected, len(found), nh])
    return np.asarray(rows, dtype=float).reshape(-1, 12)


def solve_block(bounds):
    start, stop = bounds
    m=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    if all((OUT/'routes'/f'{key}.pkl').exists() for key in m['keys'][start:stop]):
        return [pickle.load((OUT/'routes'/f'{key}.pkl').open('rb'))['row'] for key in m['keys'][start:stop]]
    m, bykey, graph = block_inputs(start, stop)
    exact_memoization()
    rows = []
    for key in m['keys'][start:stop]:
        target = OUT/'routes'/f'{key}.pkl'
        if target.exists():
            with target.open('rb') as f: rows.append(pickle.load(f)['row'])
            continue
        r, row = one_result(key, bykey[key], graph, m)
        crossings = crossing_selections(key, bykey[key], graph, r)
        with target.with_suffix('.pending').open('wb') as f:
            pickle.dump(dict(result=r, row=row, crossings=crossings), f, protocol=5)
        target.with_suffix('.pending').replace(target)
        rows.append(row)
        print('DONE', key, round(row['seconds'], 2), row['sequence'], flush=True)
    return rows


def run(workers=4, block=20):
    m = json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    started = time.perf_counter(); rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(solve_block, (i, min(len(m['keys']), i+block))):i
                   for i in range(0, len(m['keys']), block)}
        for future in as_completed(pending):
            rows.extend(future.result())
            save_json(OUT/'progress.json', dict(completed=len(rows), total=len(m['keys']),
                elapsed_seconds=time.perf_counter()-started, workers=workers))
            print('BLOCKS', len(rows), '/', len(m['keys']), round(time.perf_counter()-started, 1), flush=True)
    rows.sort(key=lambda r:float(r['slice_key']))
    segment=time.perf_counter()-started
    prior=json.loads((OUT/'run_checkpoint.json').read_text(encoding='utf-8')) if (OUT/'run_checkpoint.json').exists() else {}
    save_json(OUT/'recognition_summary.json', dict(rows=rows, wall_seconds=segment+prior.get('elapsed_segment_seconds',0),
        resumed_segment_seconds=segment,checkpoint=prior,
        workers=workers, completed=len(rows), status_counts=Counter(r['status'] for r in rows),
        sum_solver_seconds=sum(r['seconds'] for r in rows), production_entry_integrated=False,
        accuracy_measured=False, algorithm_code_unchanged=(code_hashes()==m['code_sha256'])))


def profile(key):
    m = json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    i = m['keys'].index(key); m, bs, g = block_inputs(i, i+1)
    p = cProfile.Profile(); p.enable()
    r, row = one_result(key, bs[key], g, m)
    p.disable(); p.dump_stats(str(OUT/'solver_profile.prof'))
    save_json(OUT/'profile_result.json', row)
    with (OUT/'profile_result.pkl').open('wb') as f: pickle.dump(dict(result=r, row=row), f, protocol=5)
    print(row, flush=True)


def verify_memo(key):
    m=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    i=m['keys'].index(key);m,bs,g=block_inputs(i,i+1)
    exact_memoization()
    r,row=one_result(key,bs[key],g,m)
    reference=OUT/'profile_result.pkl' if key=='104.30' else OUT/'routes'/f'{key}.pkl'
    with reference.open('rb') as f:old=pickle.load(f)['result']
    def canonical(value):
        return json.dumps(value,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
    assert canonical(r)==canonical(old),'Memoization changed full recognition output'
    output=OUT/('memo_equivalence.json' if key=='104.30' else f'memo_equivalence_{key}.json')
    save_json(output,dict(key=key,full_output_identical=True,cached_seconds=row['seconds'],
        scope='Exact pure joint summary and original per-row scale statistics; fresh graph; no recognition source modified.'))
    print('EXACT MEMO VERIFIED',row,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare','run','profile','verify-memo'))
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--block', type=int, default=20)
    parser.add_argument('--key', default='104.30'); args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='profile': profile(args.key)
    elif args.action=='verify-memo':verify_memo(args.key)
    else: run(args.workers,args.block)
