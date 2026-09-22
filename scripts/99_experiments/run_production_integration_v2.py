"""Run the production spine entry over frozen physical inputs and audit outputs."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,csv,json,pickle,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
from scipy import sparse
from diagnose_missing_routes import SOURCE,compact,gap_rows
from validate_physical_topology import ROOT,load_inventory,csv_rows
from run_face_provenance_validation import manifest,raw_observations
from census_directional_consistency import exact_memoization,save_json,code_hashes,one_result
from observed_surface_graph import build_surface_graph
from physical_face_context import PhysicalFaceContext
from surface_spine_pipeline import recognize_surface_spine
from audit_directional_surface_consensus import verify_geometry

BASE=ROOT/'outputs/validation_p0_p1_p2/20260922_050510'
PRE_LOCK=ROOT/'outputs/face_branch_consistency/20260921_094107'
HARD=['98.05','98.10','116.25','122.95','112.55','123.00','135.65']
_MESH=None


def assert_same_tree(a,b,path='root'):
    """Exact output comparison against the completed uncached computation."""
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:assert_same_tree(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):assert_same_tree(x,y,path+f'[{i}]')
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b,equal_nan=True),path
    else:assert a==b,path


def latest():return Path((ROOT/'outputs/production_integration_v2/LATEST.txt').read_text(encoding='utf-8-sig').strip())


def mesh():
    global _MESH
    if _MESH is None:
        regions=[]
        for r in csv.DictReader((SOURCE/'conflict_regions.csv').open(encoding='utf-8-sig')):
            s,u,z=float(r['s_bin'])*.5,float(r['u_bin'])*2,float(r['z_bin'])*2
            regions.append(dict(region_id=r['conflict_region_id'],bounds=[s-.1,s+.6,u-.1,u+2.1,z-.1,z+2.1]))
        _MESH=(sparse.load_npz(SOURCE/'face_adjacency.npz'),np.load(SOURCE/'face_bounds_min.npy',mmap_mode='r'),
               np.load(SOURCE/'face_bounds_max.npy',mmap_mode='r'),regions)
    return _MESH


def missing_core(branches,r):
    from branch_absolute_core import branch_metrics,local_edge_scale
    actual=r.get('route_z_extent');rows=[]
    if actual is None:return [dict(reason='NO_ROUTE',missing_m=None)]
    scale=local_edge_scale(branches)
    for b in branches:
        m=branch_metrics(b,scale)
        if not m['MBG_pass']:continue
        arc=np.asarray(b['arc_positions']);z=np.asarray(b['points_uz'])[:,1];a,c=m['ASC_start_arc'],m['ASC_end_arc']
        zz=np.r_[np.interp([a,c],arc,z),z[(arc>a)&(arc<c)]]
        for side,gap in [('lower',actual[0]-zz.min()),('upper',zz.max()-actual[1])]:
            if gap>1e-6:rows.append(dict(branch_id=b['branch_id'],side=side,missing_m=float(gap)))
    return rows


def block(task):
    out,start,stop,controls=task;out=Path(out);m=manifest(SOURCE);keys=m['keys'];targets=keys[start:stop]
    if controls:targets=[k for k in targets if k in HARD]
    if not targets:return []
    context_keys=keys[max(0,start-20):min(len(keys),stop+20)]
    bs=load_inventory(SOURCE,context_keys);began=time.perf_counter()
    g=build_surface_graph([dict(s=float(k),branches=b) for k,b in bs.items()],raw_observations(context_keys,m))
    exact_memoization();a,lo,hi,regions=mesh()
    physical=PhysicalFaceContext(a,lo,hi,{float(k):b for k,b in bs.items()},regions)
    graph_seconds=time.perf_counter()-began;rows=[]
    for key in targets:
        file=out/'routes'/f'{key}.pkl'
        if file.exists():rows.append(pickle.load(file.open('rb'))['row']);continue
        p0=None
        if controls:
            p0,pr=one_result(key,bs[key],g,m)
            save_json(out/'hard_regressions'/f'{key}_P0.json',dict(row=pr,missing_ASC=missing_core(bs[key],p0)))
        g['physical_face_context']=physical
        result=recognize_surface_spine(bs[key],g,float(key),input_z_range=m['arc']['z_range'])
        r=result['route'];verify_geometry(bs[key],r)
        reference=out/'memoization_reference'/'routes'/f'{key}.pkl'
        if reference.exists() and key not in HARD:
            assert_same_tree(compact(r),pickle.load(reference.open('rb'))['result'])
            old=pickle.load((out/'memoization_reference'/'layers'/f'{key}.pkl').open('rb'))
            assert_same_tree({k:v for k,v in result.items() if k not in ('route','performance')},
                             {k:v for k,v in old.items() if k!='performance'})
        deficits=missing_core(bs[key],r);layers=result['layers']
        row=dict(slice_key=key,seconds=sum(result['performance'].values()),**result['performance'],
            sequence=r.get('route_branch_sequence',[]),status=r['status'],
            ASC_missing=bool(deficits),ASC_missing_max_m=max((x['missing_m'] or 0. for x in deficits),default=0.),
            route_z_extent=r.get('route_z_extent'),switches=r['branch_switch_count'],
            max_connector_m=max((j['xyz_distance_m'] for j in r.get('junctions',[])),default=0.),
            max_combined_R=max((x['combined_R_syn'] for x in r.get('route_contribution_budgets',[])),default=0.),
            fallback_accepted=sum(d['feasibility_accepted'] and d['identity_rank']>1 for d in r.get('continuation_decisions',[])),
            identity_ambiguous=r.get('route_identity_ambiguous',False) or r['selection']['ambiguous'],
            P1_competitions=len(r.get('continuation_decisions',[])),physical_seed='physical_seed_candidates' in r['selection'],
            P1_local_regions=len(r.get('local_competition_audit',[])),
            P1_local_handoffs=r.get('local_competitive_handoffs',0),
            P2_roles=dict(Counter(c['role'] for c in layers['components'])),
            P2_side_m=layers['side_observed_arc_m'],hanging_segments=len(result['hanging_segments']),
            source_intervals_preserved=layers['source_intervals_preserved'],
            context_start=context_keys[0],context_end=context_keys[-1],graph_block_seconds=graph_seconds)
        row['exact_uncached_reference_match']=reference.exists() and key not in HARD
        payload=dict(result=compact(r),row=row,missing=gap_rows(key,r),missing_ASC=deficits)
        with file.with_suffix('.pending').open('wb') as f:pickle.dump(payload,f,protocol=5)
        file.with_suffix('.pending').replace(file)
        pickle.dump({k:v for k,v in result.items() if k!='route'},(out/'layers'/f'{key}.pkl').open('wb'),protocol=5)
        rows.append(row);print('DONE',key,round(row['seconds'],2),row['sequence'],round(row['ASC_missing_max_m'],3),flush=True)
        if controls:g.pop('physical_face_context',None)
    save_json(out/'contexts'/f'{start:04d}.json',dict(regions=list(physical.used.values()),seconds=graph_seconds))
    return rows


def run(out,workers,controls=False):
    for d in ['routes','layers','hard_regressions','contexts','figures','cases']:(out/d).mkdir(exist_ok=True)
    m=manifest(SOURCE);started=time.perf_counter();rows=[]
    save_json(out/'manifest.json',dict(source=str(SOURCE),baseline=str(BASE),pre_lock_baseline=str(PRE_LOCK),
        plan='GDS_P0_P1_P2_Production_Integration_Codex_Plan_v2.md',code_sha256=code_hashes(),
        FaceTrack_used_in_recognition=True,P1_P2_shadow_only=False,static_ASC_hard_lock=False,
        census_context='Same 10-target blocks, 20-slice halos; physical local FaceTrack CSR context added.',
        P2_return_detection=dict(cap_m=.01,min_arc_m=.5,min_arc_gap_ratio=50),workers=workers))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        ids=range(0,len(m['keys']),10)
        if controls:ids=sorted({m['keys'].index(k)//10*10 for k in HARD})
        tasks=[pool.submit(block,(str(out),i,min(i+10,len(m['keys'])),controls)) for i in ids]
        for f in as_completed(tasks):
            rows.extend(f.result());save_json(out/('control_progress.json' if controls else 'progress.json'),
                dict(completed=len(rows),total=7 if controls else len(m['keys']),seconds=time.perf_counter()-started))
            print('PROGRESS',len(rows),flush=True)
    rows.sort(key=lambda x:float(x['slice_key']))
    summary=dict(count=len(rows),seconds=time.perf_counter()-started,workers=workers,
        ASC_missing_count=sum(r['ASC_missing'] for r in rows),ASC_missing_ge_1m_count=sum(r['ASC_missing_max_m']>=1 for r in rows),
        fallback_accepted=sum(r['fallback_accepted'] for r in rows),
        max_connector_m=max(r['max_connector_m'] for r in rows),max_combined_R=max(r['max_combined_R'] for r in rows),
        physical_seed_count=sum(r['physical_seed'] for r in rows),all_intervals_preserved=all(r['source_intervals_preserved'] for r in rows),
        identity_ambiguous_count=sum(r['identity_ambiguous'] for r in rows),
        exact_uncached_reference_matches=sum(r.get('exact_uncached_reference_match',False) for r in rows),
        code_unchanged=code_hashes()==json.loads((out/'manifest.json').read_text(encoding='utf-8'))['code_sha256'])
    name='hard_regression_summary' if controls else 'census_summary'
    save_json(out/(name+'.json'),summary);csv_rows(out/(name+'.csv'),rows)
    print('COMPLETE',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controls','census']);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(latest(),a.workers,a.action=='controls')
