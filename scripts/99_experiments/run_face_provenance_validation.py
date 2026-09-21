"""Prepare local validation regions, actual route controls and frozen face CSR."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,pickle,time
import numpy as np
from scipy import sparse
from validate_physical_topology import ROOT,PRIOR,output_path,load_inventory,csv_rows
from census_directional_consistency import save_json,load_branches,exact_memoization,one_result
from locc_data import opc
from observed_surface_graph import build_surface_graph
from face_provenance_audit import FaceAdjacency


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def manifest(out):return json.loads((out/'BASELINE_MANIFEST.json').read_text(encoding='utf-8'))['input_manifest']


def regions(out):
    m=manifest(out);oldcases=json.loads((PRIOR/'representative_cases.json').read_text(encoding='utf-8'))
    reference={c['left']:c for c in oldcases}
    specs=[('A','0.85','0.85','SELF_LOOP_RECOVERED'),('B','78.40','78.40','MBG_ASYMMETRY'),
           ('C','103.60','103.70','MISSING_NEIGHBOR'),('D','122.25','122.25','BRANCH_COMBINATIONS'),
           ('E','104.30','104.30','COMPETING_SURFACES'),('F','130.45','130.45','MILLIMETRE_OVERLAP'),
           ('G','122.95','122.95','COVERAGE_GAP'),('H','116.20','116.25','COVERAGE_GAP'),
           ('I','100.15','100.15','DETAIL_COUNT'),('J','108.80','108.80','GEOMETRY_COMPLEXITY')]
    selected=[]
    for name,key,focus,reason in specs:
        c=reference[key];i=m['keys'].index(key)
        selected.append(dict(region_id=name,center_s=float(key),focus_s=focus,center_z=c['center_z'],
            target_keys=m['keys'][max(0,i-2):i+3],reason=reason,
            bounds=[float(key)-.2,float(key)+.2,c['bounds'][0]-.1,c['bounds'][1]+.1,c['bounds'][2]-.1,c['bounds'][3]+.1]))
    intrinsic=json.loads((PRIOR/'intrinsic_review_windows.json').read_text(encoding='utf-8'))
    for name,category in [('K','LOCAL_ANCHOR_TIE'),('L','LOCAL_CONTINUATION_TIE')]:
        event=next(r for r in intrinsic if r['category']==category);key=event['slice_key'];i=m['keys'].index(key)
        with (PRIOR/'routes'/f'{key}.pkl').open('rb') as f:d=pickle.load(f)
        points=np.asarray(d['result']['curve_uz']);z=float(np.median(points[:,1]))
        if category=='LOCAL_CONTINUATION_TIE':z=float(points[:,1].min() if event['frontier']=='lower' else points[:,1].max())
        bs=load_branches([key])[key];us=[u for b in bs for u,zz in b['points_uz'] if abs(zz-z)<=3]
        selected.append(dict(region_id=name,center_s=float(key),focus_s=key,center_z=z,target_keys=m['keys'][max(0,i-2):i+3],
            reason=category,location_basis='FINAL_ROUTE_MEDIAN_Z' if category=='LOCAL_ANCHOR_TIE' else 'FINAL_ROUTE_FRONTIER_Z',
            bounds=[float(key)-.2,float(key)+.2,min(us)-.4,max(us)+.4,z-3.1,z+3.1]))
    # Census grouping is explicitly 3-D; a cell is a review bucket, not a surface.
    cells=defaultdict(lambda:dict(categories=set(),slices=set(),seeds=0))
    def add(s,u,z,category):
        cell=(int(np.floor(s/.5)),int(np.floor(u/2)),int(np.floor(z/2)))
        row=cells[cell];row['categories'].add(category);row['slices'].add(f'{s:.2f}');row['seeds']+=1
    windows=json.loads((PRIOR/'local_review_windows.json').read_text(encoding='utf-8'))
    for w in windows:
        if not w['persistent']:continue
        for e in w['examples']:
            z=m['levels'][e['level']]
            add(float(w['left']),e['left_u'],z,w['category']);add(float(w['right']),e['right_u'],z,w['category'])
    proxy=Counter()
    for e in intrinsic:
        if e['category'] not in ('LOCAL_ANCHOR_TIE','LOCAL_CONTINUATION_TIE','UNCOVERED_RELIABLE_EXTENT'):continue
        if 'z' in e:add(float(e['slice_key']),e['u'],e['z'],e['category'])
        else:proxy[e['category']]+=1
    recovered=__import__('csv').DictReader((out/'recovered_branches.csv').open(encoding='utf-8-sig'))
    rec_keys=sorted({r['s'] for r in recovered},key=float)
    for key,bs in load_inventory(out,rec_keys).items():
        for b in bs:
            if b['kind']=='OPEN_SURFACE_BRANCH':
                u,z=b['points_uz'][len(b['points_uz'])//2];add(float(key),float(u),float(z),'SELF_LOOP_RECOVERED_CONTEXT')
    for r in selected:
        add(r['center_s'],(r['bounds'][2]+r['bounds'][3])/2,r['center_z'],r['reason'])
    csv_rows(out/'conflict_regions.csv',[dict(conflict_region_id=f'CELL_{i:05d}',s_bin=cell[0],u_bin=cell[1],z_bin=cell[2],
        categories=sorted(d['categories']),slices=sorted(d['slices'],key=float),seed_count=d['seeds']) for i,(cell,d) in enumerate(sorted(cells.items()))])
    save_json(out/'validation_regions.json',selected)
    save_json(out/'region_scope.json',dict(conflict_region_count=len(cells),validation_region_count=len(selected),
        all_region_face_tracking=False,grouping_cell_m=[.5,2.,2.],
        intrinsic_events_without_exact_local_coordinate=dict(proxy),
        caveat='3D review buckets are not independent physical defects; 12 purpose-selected regions are not an accuracy sample.'))
    print('REGIONS',len(cells),len(selected),flush=True)


def mesh(out):
    started=time.perf_counter();m=manifest(out);cache=Path(m['prior'])/'mesh_cache'
    faces=np.load(cache/'faces.npy',mmap_mode='r');vertices=np.load(cache/'vertices.npy',mmap_mode='r')
    index=FaceAdjacency.build(faces)
    sparse.save_npz(out/'face_adjacency.npz',index.adjacency)
    suz=opc.to_suz(vertices,m['arc'])
    lower=np.full((len(faces),3),np.inf);upper=np.full_like(lower,-np.inf)
    for i in range(3):lower=np.minimum(lower,suz[faces[:,i]]);upper=np.maximum(upper,suz[faces[:,i]])
    np.save(out/'face_bounds_min.npy',lower);np.save(out/'face_bounds_max.npy',upper)
    save_json(out/'face_adjacency_manifest.json',dict(index.metadata,cache=str(cache),seconds=time.perf_counter()-started,
        vertices_sha256=digest(cache/'vertices.npy'),faces_sha256=digest(cache/'faces.npy'),
        metadata=json.loads((cache/'metadata.json').read_text(encoding='utf-8')),face_bounds_coordinates='s,u,z',
        source='Frozen filtered mesh matching slicing FaceID space; no re-load/weld/re-index of original GLB.'))
    print('MESH ADJACENCY',index.metadata,time.perf_counter()-started,flush=True)


def raw_observations(keys,m):
    allkeys=m['keys'];low=allkeys.index(keys[0]);high=allkeys.index(keys[-1])+1
    a=np.load(PRIOR/'raw_h.npy',mmap_mode='r');off=np.load(PRIOR/'h_offsets.npy');arc=m['arc'];result=[]
    for si,zi,pid,u,face in a[off[low]:off[high]]:
        s=float(allkeys[int(si)]);z=m['levels'][int(zi)];angle=arc['angle_min']+s/arc['radius']
        result.append(dict(s=s,z=z,level_index=int(zi),h_path_id=int(pid),u=float(u),face_id=int(face),
            point_xyz=[arc['center'][0]+(u+arc['radius'])*np.cos(angle),arc['center'][1]+(u+arc['radius'])*np.sin(angle),z],
            evidence='RAW_HORIZONTAL_OBSERVATION'))
    return result


def solve_region(args):
    out,r=args;out=Path(out);m=manifest(out);ids=[m['keys'].index(k) for k in r['target_keys']]
    keys=m['keys'][max(0,min(ids)-20):min(len(m['keys']),max(ids)+21)]
    observations=raw_observations(keys,m);rows=[];results={};graph_times={}
    for version,loader in [('BEFORE_LOCAL',lambda:load_branches(keys)),('CURRENT_ROUTE',lambda:load_inventory(out,keys))]:
        bs=loader();t=time.perf_counter()
        graph=build_surface_graph([dict(s=float(k),branches=b) for k,b in bs.items()],observations)
        graph_times[version]=time.perf_counter()-t;exact_memoization();results[version]={}
        for key in r['target_keys']:
            value,row=one_result(key,bs[key],graph,m)
            row.update(region_id=r['region_id'],version=version);rows.append(row)
            results[version][key]=value
            print('ROUTE',r['region_id'],version,key,round(row['seconds'],2),row['sequence'],flush=True)
    with (out/'local_routes'/f'{r["region_id"]}.pkl').open('wb') as f:pickle.dump(results,f,protocol=5)
    return dict(region_id=r['region_id'],rows=rows,graph_seconds=graph_times,context_keys=keys)


def routes(out,workers):
    assert json.loads((out/'topology_summary.json').read_text(encoding='utf-8'))['affected_gate_passed']
    selected=json.loads((out/'validation_regions.json').read_text(encoding='utf-8'))
    (out/'local_routes').mkdir(exist_ok=True);start=time.perf_counter();finished=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(solve_region,(str(out),r)) for r in selected]
        for future in as_completed(tasks):finished.append(future.result())
    save_json(out/'local_route_summary.json',dict(regions=finished,workers=workers,seconds=time.perf_counter()-start,
        face_track_used_in_recognition=False,production_route_modified=False,
        context='Paired before/after physical topology, same local H graph context with at least ±1 m halo for every target. Prior whole-area results remain separately frozen.',
        CURRENT_ROUTE_sha256={p.name:digest(p) for p in (out/'local_routes').glob('*.pkl')}))
    print('LOCAL ROUTES COMPLETE',len(finished),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('regions','mesh','routes'));p.add_argument('--output',type=Path,default=output_path());p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.action=='routes':routes(a.output,a.workers)
    else:globals()[a.action](a.output)
