"""Full Phase 1 CLI: raw observed evidence -> independent votes -> frozen decision.

This entry never imports recognition/route application modules. Historical
cases are used only by the separate renderer after decisions are frozen.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys,json,pickle,hashlib,time,shutil,argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import psutil

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'scripts/04_structure_recognition'
sys.path.insert(0,str(CORE))
from facetrack_phase1_evidence import Phase1Policy,REGION_VERSION,POLICY_VERSION,segment_cells,score_slice
from facetrack_phase1_regions import build_regions
from facetrack_phase1_vote import reduce_region
from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale

SOURCE=ROOT/'outputs/face_provenance_validation/20260921_075801'
PARENT=ROOT/'outputs/facetrack_voting_phase1'
INDEX_FILES=('facetrack_phase1_regions.py','branch_absolute_core.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def latest():return Path((PARENT/'LATEST.txt').read_text(encoding='utf-8').strip())
def read_pickle(path):
    with Path(path).open('rb') as f:return pickle.load(f)
def save_pickle(path,value):
    with Path(path).open('wb') as f:pickle.dump(value,f,protocol=5)
def source_keys():return sorted(read_json(SOURCE/'branch_index.json'),key=float)


def _init_index(source,cache,offsets,policy):
    global _RAW,_GROUPS,_OFFSETS,_CACHE,_POLICY
    _RAW=(Path(source)/'branches.pkl').open('rb');_GROUPS=np.load(Path(cache)/'physical_face_groups.npy',mmap_mode='r')
    _OFFSETS=offsets;_CACHE=Path(cache);_POLICY=Phase1Policy(**policy)


def index_chunk(task):
    chunk,items=task;began=time.perf_counter();cpu=time.process_time();rows=[];occupancy=[];edges=0;unmapped=0
    for si,s in items:
        _RAW.seek(_OFFSETS[s]);branches=pickle.load(_RAW);scale=local_edge_scale(branches);geometry={};cells={};total_arc=0.;cell_arc=0.
        for b in branches:
            bid=int(b['branch_id']);metric=branch_metrics(b,scale);points=np.asarray(b['points_uz'])
            tids=sorted(set(int(x) for x in _GROUPS[np.asarray(b['source_face_ids'],dtype=np.int64)]))
            edge_tracks=[]
            for e in b['records']:
                groups=tuple(sorted(set(int(x) for x in _GROUPS[np.asarray(e['source_face_ids'],dtype=np.int64)])))
                edge_tracks.append(groups)
                if not groups:unmapped+=1
            item=dict(branch_id=bid,MBG=bool(metric['MBG_pass']),CC_start_arc=float(metric['ASC_start_arc']),CC_end_arc=float(metric['ASC_end_arc']),
                full_arc_length=float(b['full_arc_length']),forward_sign=1 if points[-1,1]>=points[0,1] else -1,
                points_uz=points,arc_positions=np.asarray(b['arc_positions']),physical_groups=tids,kind=b['kind'])
            geometry[bid]=item
            row=dict(s=float(s),s_index=si,branch_id=bid,FaceTracks=tids,source_face_ids=[int(x) for x in b['source_face_ids']],
                u_range=b['u_range'],z_range=b['z_range'],full_arc_length=float(b['full_arc_length']),node_count=b['canonical_node_count'],
                MBG=item['MBG'],CC_start_arc=item['CC_start_arc'],CC_end_arc=item['CC_end_arc'],kind=b['kind'],
                local_edge_scale_m=float(scale),unmapped_edges=sum(not t for t in edge_tracks))
            rows.append(row);edges+=len(edge_tracks);total_arc+=b['full_arc_length']
            parts=segment_cells(points,_POLICY.cell_m);cell_arc+=sum(hi-lo for _,_,_,lo,hi in parts)
            for u,z,e,lo,hi in parts:
                for track in edge_tracks[e]:cells.setdefault((u,z),{}).setdefault(track,[]).append((bid,lo,hi))
        if abs(total_arc-cell_arc)>max(1e-6,1e-8*total_arc):raise AssertionError(('cell interval conservation',s,total_arc,cell_arc))
        save_pickle(_CACHE/'slices'/f'{s}.pkl',dict(s=float(s),s_index=si,branches=geometry,cells=cells))
        for (u,z),members in cells.items():
            pairs=sorted({(t,bid) for t,parts in members.items() for bid,_,_ in parts})
            occupancy.append(dict(s_index=si,u_bin=u,z_bin=z,tracks=sorted(members),track_count=len(members),
                branch_tracks=[t for t,b in pairs],branch_ids=[b for t,b in pairs]))
    pq.write_table(pa.Table.from_pylist(rows),_CACHE/'chunks'/f'branches_{chunk:04d}.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(occupancy),_CACHE/'chunks'/f'cells_{chunk:04d}.parquet',compression='zstd')
    return dict(chunk=chunk,profiles=len(items),branches=len(rows),edges=edges,cells=len(occupancy),unmapped=unmapped,
        seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,rss=psutil.Process().memory_info().rss)


def _init_scores(cache,policy):
    global _CACHE,_POLICY
    _CACHE=Path(cache);_POLICY=Phase1Policy(**policy)


def score_task(task):
    si,s,regions=task;began=time.perf_counter();cpu=time.process_time()
    geometry=read_pickle(_CACHE/'slices'/f'{s}.pkl');rows=[]
    for r in regions:rows.extend(score_slice(r,si,float(s),geometry,_POLICY))
    return rows,dict(s=s,regions=len(regions),seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu)


def cache_contract(policy):
    manifest=read_json(SOURCE/'face_adjacency_manifest.json');mesh_cache=Path(manifest['cache'])
    hashes={name:digest(SOURCE/name) for name in ('branches.pkl','branch_index.json','face_adjacency.npz','face_bounds_min.npy','face_bounds_max.npy','BASELINE_MANIFEST.json')}
    for name in ('vertices','faces'):
        actual=digest(mesh_cache/(name+'.npy'))
        if actual!=manifest[name+'_sha256']:raise AssertionError(('source mesh hash mismatch',name))
        hashes[name+'.npy']=actual
    # Index geometry functions are kept separately from score_slice in the key;
    # changing scoring code cannot force a mesh/branch rebuild.
    import inspect
    from facetrack_phase1_evidence import segment_cells
    contract=dict(source_hashes=hashes,mesh_hash=key({n:hashes[n] for n in ('vertices.npy','faces.npy','face_adjacency.npz')}),
        branch_source_hash=hashes['branches.pkl'],region_definition_version=REGION_VERSION,
        policy_version='CANONICAL_BRANCH_METRICS_V1',metric_policy=asdict(BranchPolicy()),cell_m=policy.cell_m,
        index_code={n:digest(CORE/n) for n in INDEX_FILES},cell_geometry_code=key(inspect.getsource(segment_cells)),
        index_extraction_code=key(inspect.getsource(index_chunk)))
    return contract


def observed_rss():
    process=psutil.Process();return process.memory_info().rss+sum(p.memory_info().rss for p in process.children() if p.is_running())


def run(workers=6,policy=None,budget=3600,reuse_check=False):
    began=time.perf_counter();policy=policy or Phase1Policy();times={};peak=observed_rss();positions=[float(s) for s in source_keys()]
    tick=time.perf_counter();contract=cache_contract(policy);times['source_hash_verification_s']=time.perf_counter()-tick
    cache=PARENT/'cache'/key(contract);keys=source_keys();out=PARENT/datetime.now().strftime('%Y%m%d_%H%M%S')
    if reuse_check:
        out=latest();saved=read_json(out/'manifest.json')
        assert saved['index_key']==cache.name and (cache/'COMPLETE.json').exists()
        assert digest(Path(saved['frozen_decision_file']))==saved['frozen_decision_sha256']
        check=dict(index_reused=True,geometry_recomputed=False,score_rows_recomputed=False,source_contract_reverified=True,
            seconds=time.perf_counter()-began,profiles=len(keys),index_key=cache.name,decision_hash_verified=True)
        save_json(out/'CACHE_REUSE_CHECK.json',check);print('CACHE_REUSE',check,flush=True);return out
    for folder in ('figures','cases'):(out/folder).mkdir(parents=True,exist_ok=True)
    (PARENT/'LATEST.txt').write_text(str(out),encoding='utf-8')
    for folder in ('slices','chunks'):(cache/folder).mkdir(parents=True,exist_ok=True)
    reused=(cache/'COMPLETE.json').exists();index_stats=[]
    if not reused:
        tick=time.perf_counter();adjacency=sparse.load_npz(SOURCE/'face_adjacency.npz');n,labels=connected_components(adjacency,directed=False)
        np.save(cache/'physical_face_groups.npy',labels)
        save_json(cache/'physical_face_groups.json',dict(group_count=n,face_count=len(labels),group_sizes=np.bincount(labels).tolist()))
        del adjacency,labels
        times['physical_face_components_s']=time.perf_counter()-tick;tick=time.perf_counter()
        indexed=list(enumerate(keys));tasks=[(i//40,indexed[i:i+40]) for i in range(0,len(keys),40)]
        offsets=read_json(SOURCE/'branch_index.json')
        with ProcessPoolExecutor(max_workers=workers,initializer=_init_index,initargs=(str(SOURCE),str(cache),offsets,asdict(policy))) as pool:
            futures=[pool.submit(index_chunk,t) for t in tasks];count=0
            for future in as_completed(futures):
                row=future.result();index_stats.append(row);count+=row['profiles'];peak=max(peak,observed_rss())
                if count%200==0 or count==len(keys):print('A1_INDEX',count,'/',len(keys),round(time.perf_counter()-tick,2),flush=True)
                if time.perf_counter()-began>budget:raise TimeoutError('Phase1 A1 exceeded budget')
        times['A1_observed_branch_index_s']=time.perf_counter()-tick;tick=time.perf_counter()
        branches=pa.concat_tables([pq.read_table(p) for p in sorted((cache/'chunks').glob('branches_*.parquet'))])
        cells=pa.concat_tables([pq.read_table(p) for p in sorted((cache/'chunks').glob('cells_*.parquet'))])
        # Sorting is deterministic; worker finish order never affects identities.
        branches=branches.sort_by([('s_index','ascending'),('branch_id','ascending')]);cells=cells.sort_by([('s_index','ascending'),('u_bin','ascending'),('z_bin','ascending')])
        pq.write_table(branches,cache/'branch_index.parquet',compression='zstd');pq.write_table(cells,cache/'cell_index.parquet',compression='zstd')
        conflict=cells.filter(pa.compute.greater_equal(cells['track_count'],2)).to_pylist()
        regions=build_regions(conflict,positions,policy.cell_m)
        save_json(cache/'conflict_region_index.json',dict(region_definition_version=REGION_VERSION,cell_m=policy.cell_m,regions=regions))
        save_json(cache/'COMPLETE.json',dict(contract=contract,profiles=len(keys),branches=len(branches),cells=len(cells),
            conflict_cells=len(conflict),regions=len(regions),index_stats=index_stats))
        times['A2_conflict_index_s']=time.perf_counter()-tick
        del branches,cells,conflict
    else:
        regions=read_json(cache/'conflict_region_index.json')['regions'];times['A1_A2_cache_load_s']=time.perf_counter()-began-times['source_hash_verification_s']
        print('A1_A2_REUSED',cache.name,flush=True)
    peak=max(peak,observed_rss());tick=time.perf_counter()
    per_slice=defaultdict(list);membership=defaultdict(set);facetracks=[]
    for r in regions:
        for si in r['s_indices']:
            # Each worker sees only immutable mask cells for its own V.
            per_slice[si].append(dict(region_id=r['region_id'],tracks=r['tracks'],cells=[c for c in r['cells'] if c[0]==si]))
        for track,members in r['members'].items():
            for si,bid in members:membership[si,bid].add(r['region_id'])
            facetracks.append(dict(region_id=r['region_id'],FaceTrack=int(track),member_branches=[f'{keys[si]}/B{bid}' for si,bid in members],
                member_V=sorted({positions[si] for si,bid in members}),s_span=r['bounds'][1]-r['bounds'][0],u_range=r['bounds'][2:4],z_range=r['bounds'][4:6]))
    branch_table=pq.read_table(cache/'branch_index.parquet')
    memberships=[sorted(membership.get((si,bid),set())) for si,bid in zip(branch_table['s_index'].to_pylist(),branch_table['branch_id'].to_pylist())]
    branch_table=branch_table.append_column('conflict_regions',pa.array(memberships,pa.list_(pa.string())))
    pq.write_table(branch_table,out/'branch_facetrack_index.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(facetracks),out/'face_track_index.parquet',compression='zstd')
    shutil.copy2(cache/'conflict_region_index.json',out/'conflict_region_index.json')
    times['index_export_s']=time.perf_counter()-tick
    score_contract=dict(index_key=cache.name,policy_version=POLICY_VERSION,policy=asdict(policy),
        score_code={n:digest(CORE/n) for n in ('facetrack_phase1_evidence.py','facetrack_phase1_vote.py')})
    score_cache=cache/'scores'/key(score_contract);score_cache.mkdir(parents=True,exist_ok=True)
    score_stats=[];tick=time.perf_counter();score_reused=(score_cache/'COMPLETE.json').exists()
    if not score_reused:
        tasks=[(si,keys[si],rs) for si,rs in sorted(per_slice.items())];rows=[]
        with ProcessPoolExecutor(max_workers=workers,initializer=_init_scores,initargs=(str(cache),asdict(policy))) as pool:
            futures=[pool.submit(score_task,t) for t in tasks];done=0
            for f in as_completed(futures):
                rr,stats=f.result();rows.extend(rr);score_stats.append(stats);done+=1
                if done%400==0 or done==len(tasks):print('A3_SCORED_V',done,'/',len(tasks),'rows',len(rows),round(time.perf_counter()-tick,2),flush=True)
                # Reading every child's Windows process metadata per V can
                # dominate these small tasks. RSS is monitoring, not scoring.
                if done%100==0 or done==len(tasks):peak=max(peak,observed_rss())
                if time.perf_counter()-began>budget:raise TimeoutError('Phase1 A3 exceeded budget')
        rows.sort(key=lambda r:(r['region_id'],r['s_index'],r['FaceTrack']))
        pq.write_table(pa.Table.from_pylist(rows),score_cache/'per_slice_track_scores.parquet',compression='zstd')
        save_json(score_cache/'COMPLETE.json',dict(contract=score_contract,score_rows=len(rows),profile_tasks=len(tasks),score_stats=score_stats))
    else:rows=pq.read_table(score_cache/'per_slice_track_scores.parquet').to_pylist()
    times['A3_independent_scores_s']=time.perf_counter()-tick
    shutil.copy2(score_cache/'per_slice_track_scores.parquet',out/'per_slice_track_scores.parquet')
    through_membership=defaultdict(set)
    for row in rows:
        for witness in row['witnesses']:
            if witness['through_region']:through_membership[row['s_index'],witness['branch_id']].add(row['region_id'])
    branch_table=branch_table.append_column('through_conflict_regions',pa.array([
        sorted(through_membership.get((si,bid),set())) for si,bid in zip(branch_table['s_index'].to_pylist(),branch_table['branch_id'].to_pylist())],pa.list_(pa.string())))
    pq.write_table(branch_table,out/'branch_facetrack_index.parquet',compression='zstd')
    tick=time.perf_counter();by_region=defaultdict(list)
    for r in rows:by_region[r['region_id']].append(r)
    decisions=[reduce_region(rid,rr,policy.continuation_quantum_m) for rid,rr in sorted(by_region.items())]
    frozen=out/'region_track_decisions.json'
    save_json(frozen,dict(stage='PHASE1_FROZEN_FACE_DECISION',route_application=False,policy=asdict(policy),
        evidence_summary_columns=['FaceTrack','through_V_count','longest_eligible_V_run','run_s_span_m','CC_V_count','mean_bounded_continuation_m','eligible_V_count'],
        decisions=[d.to_dict() for d in decisions]))
    times['B_table_reduction_s']=time.perf_counter()-tick
    complete=read_json(cache/'COMPLETE.json');peak=max(peak,observed_rss())
    manifest=dict(source=str(SOURCE),profiles=len(keys),workers=workers,index_key=cache.name,index_cache=str(cache),index_contract=contract,
        score_key=score_cache.name,score_contract=score_contract,policy=asdict(policy),index_reused=reused,score_reused=score_reused,
        branch_count=complete['branches'],cell_count=complete['cells'],conflict_cell_count=complete['conflict_cells'],region_count=len(regions),
        score_row_count=len(rows),V_with_competition=len(per_slice),V_without_competition=len(keys)-len(per_slice),
        physical_groups=read_json(cache/'physical_face_groups.json'),frozen_decision_file=str(frozen),frozen_decision_sha256=digest(frozen),
        seconds=time.perf_counter()-began,stages=times,sampled_peak_parent_children_RSS_bytes=peak,
        worker_cpu_seconds=sum(r['cpu_seconds'] for r in index_stats)+sum(r['cpu_seconds'] for r in score_stats),
        logical_cpus=os.cpu_count(),physical_memory_bytes=psutil.virtual_memory().total,
        route_reads=0,route_writes=0,junction_searches=0,region_lock_uses=0,manual_review='PENDING')
    save_json(out/'manifest.json',manifest)
    save_json(out/'score_task_performance.json',score_stats)
    print('PHASE1_COMPLETE',out,dict(seconds=manifest['seconds'],branches=complete['branches'],regions=len(regions),scores=len(rows)),flush=True)
    return out


def benchmark_score_replay(workers=6):
    """One same-input full A3 replay after reducing instrumentation overhead."""
    out=latest();manifest=read_json(out/'manifest.json');cache=Path(manifest['index_cache']);policy=Phase1Policy(**manifest['policy'])
    regions=read_json(out/'conflict_region_index.json')['regions'];keys=source_keys();by_slice=defaultdict(list)
    for r in regions:
        for si in r['s_indices']:by_slice[si].append(dict(region_id=r['region_id'],tracks=r['tracks'],cells=[c for c in r['cells'] if c[0]==si]))
    began=time.perf_counter();tasks=[(si,keys[si],rs) for si,rs in sorted(by_slice.items())];rows=[];stats=[];peak=observed_rss()
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_scores,initargs=(str(cache),asdict(policy))) as pool:
        futures=[pool.submit(score_task,t) for t in tasks]
        for count,f in enumerate(as_completed(futures),1):
            rr,t=f.result();rows.extend(rr);stats.append(t)
            if count%100==0 or count==len(tasks):peak=max(peak,observed_rss())
            if count%700==0:print('A3_PERFORMANCE_REPLAY',count,round(time.perf_counter()-began,2),flush=True)
    rows.sort(key=lambda r:(r['region_id'],r['s_index'],r['FaceTrack']))
    pq.write_table(pa.Table.from_pylist(rows),out/'performance_replay_scores.parquet',compression='zstd')
    seconds=time.perf_counter()-began
    baseline=pq.read_table(out/'per_slice_track_scores.parquet').to_pylist()
    assert rows==baseline,'Performance-only change altered independent evidence'
    grouped=defaultdict(list)
    for row in rows:grouped[row['region_id']].append(row)
    decisions=[reduce_region(rid,rr,policy.continuation_quantum_m).to_dict() for rid,rr in sorted(grouped.items())]
    assert key(decisions)==key(read_json(out/'region_track_decisions.json')['decisions'])
    assert digest(out/'region_track_decisions.json')==manifest['frozen_decision_sha256']
    result=dict(profiles=len(tasks),score_rows=len(rows),original_A3_s=manifest['stages']['A3_independent_scores_s'],optimized_A3_s=seconds,
        matched_stage_speedup=manifest['stages']['A3_independent_scores_s']/seconds,all_rows_identical=True,all_decisions_identical=True,
        index_rebuilt=False,route_operations=0,worker_cpu_seconds=sum(t['cpu_seconds'] for t in stats),
        sampled_peak_RSS_bytes=peak,scope='same cached compact geometry, fresh 6-worker pool, full A3 including parquet write; comparison and reducer validation excluded',
        change='RSS process sampling every 100 V results instead of every result; no scoring/selection change')
    save_json(out/'PERFORMANCE_REPLAY.json',result);print('A3_PERFORMANCE_RESULT',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--cell-m',type=float,default=.5)
    p.add_argument('--exit-margin-m',type=float,default=1.);p.add_argument('--budget-seconds',type=float,default=3600);p.add_argument('--reuse-check',action='store_true')
    p.add_argument('--benchmark-score-replay',action='store_true')
    a=p.parse_args()
    if a.benchmark_score_replay:benchmark_score_replay(a.workers)
    else:run(a.workers,Phase1Policy(cell_m=a.cell_m,exit_margin_m=a.exit_margin_m),a.budget_seconds,a.reuse_check)
