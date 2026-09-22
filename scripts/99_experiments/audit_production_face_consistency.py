"""Same-window physical FaceTrack consistency, all conflict cells and H levels."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,pickle,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from run_production_integration_v2 import latest,mesh,SOURCE,BASE,manifest,load_inventory,save_json,csv_rows
from physical_face_context import PhysicalFaceContext


def clip_parameters(points,bounds):
    p=np.asarray(points);v=p[1]-p[0];lo,hi=0.,1.
    for axis,(a,b) in enumerate([(bounds[2],bounds[3]),(bounds[4],bounds[5])]):
        if abs(v[axis])<1e-14:
            if not a<=p[0,axis]<=b:return None
        else:
            t0,t1=sorted(((a-p[0,axis])/v[axis],(b-p[0,axis])/v[axis]))
            lo,hi=max(lo,t0),min(hi,t1)
            if hi<lo:return None
    return p[0]+lo*v,p[0]+hi*v


def level_tracks(records,region,labels,levels):
    hits=[set() for _ in levels]
    for e in records:
        if not e['source'].startswith('OBSERVED'):continue
        cut=clip_parameters(e['points_uz'],region['bounds'])
        if cut is None:continue
        tracks={labels[f] for f in e['source_face_ids'] if f in labels}
        if not tracks:tracks={-1}
        z=sorted(p[1] for p in cut);a=np.searchsorted(levels,z[0]-1e-8);b=np.searchsorted(levels,z[1]+1e-8,side='right')
        for i in range(a,b):hits[i].update(tracks)
    return hits


def classify(a,b):
    if not a or not b:return 'MISSING_BOTH' if not a and not b else 'MISSING_ONE'
    if -1 in a or -1 in b:return 'UNKNOWN_PROVENANCE'
    if len(a)!=1 or len(b)!=1:return 'AMBIGUOUS_MULTIPLE'
    return 'CONSISTENT' if a==b else 'IDENTITY_SWITCH'


def audit_group(task):
    out,regions=task;m=manifest(SOURCE);allkeys=m['keys'];lo=min(r['bounds'][0] for r in regions);hi=max(r['bounds'][1] for r in regions)
    keys=[k for k in allkeys if lo-1e-8<=float(k)<=hi+1e-8]
    if len(keys)<2:return []
    bs=load_inventory(SOURCE,keys);a,low,high,_=mesh()
    context=PhysicalFaceContext(a,low,high,{float(k):v for k,v in bs.items()},regions)
    from pathlib import Path
    out=Path(out);routes={}
    for k in keys:
        old=pickle.load((BASE/'routes'/f'{k}.pkl').open('rb'))['result']['path_edges']
        new=pickle.load((out/'routes'/f'{k}.pkl').open('rb'))['result']['path_edges']
        final=pickle.load((out/'layers'/f'{k}.pkl').open('rb'))['layers']['MAIN_SPINE']
        routes[k]={'BEFORE':old,'P1':new,'P2':final}
    rows=[]
    for region in regions:
        data=context.region_data(region);b=region['bounds'];ss=[k for k in keys if b[0]-1e-8<=float(k)<=b[1]+1e-8]
        levels=np.asarray([z for z in m['levels'] if b[4]<=z<=b[5]])
        if len(ss)<2 or not len(levels):continue
        table={k:{version:level_tracks(records,region,data['face_labels'],levels) for version,records in routes[k].items()} for k in ss}
        counts={v:Counter() for v in ['BEFORE','P1','P2']};examples=[]
        for left,right in zip(ss,ss[1:]):
            for i,z in enumerate(levels):
                classification={v:classify(table[left][v][i],table[right][v][i]) for v in counts}
                for v,c in classification.items():counts[v][c]+=1
                if classification['BEFORE']!=classification['P1'] and len(examples)<5:
                    examples.append(dict(left=left,right=right,z=float(z),classification=classification,
                        tracks={v:[sorted(table[left][v][i]),sorted(table[right][v][i])] for v in counts}))
        rows.append(dict(region_id=region['region_id'],bounds=b,source='ALL_FROZEN_CONFLICT_CELLS_AND_RUNTIME_WINDOWS',
            slices=len(ss),H_levels=len(levels),FaceTrack_count=len(data['members']),
            BEFORE=dict(counts['BEFORE']),P1=dict(counts['P1']),P2=dict(counts['P2']),examples=examples))
    return rows


def run(out,workers=4,stream=False):
    start=time.perf_counter();regions={r['region_id']:r for r in mesh()[3]};known=set(regions)
    groups={};results=[];keys=manifest(SOURCE)['keys'];indices={k:i for i,k in enumerate(keys)}
    def add_groups(values):
        for r in values:groups.setdefault(int(np.floor((r['bounds'][0]+.100001)/.5)),[]).append(r)
    add_groups(regions.values());runtime_added=False
    def ready(rs):
        lo=min(r['bounds'][0] for r in rs);hi=max(r['bounds'][1] for r in rs)
        needed=[k for k in keys if lo-1e-8<=float(k)<=hi+1e-8]
        for k in needed:
            layer=out/'layers'/f'{k}.pkl';marker=out/'contexts'/f'{indices[k]//10*10:04d}.json'
            if not layer.exists() or not marker.exists() or marker.stat().st_mtime<layer.stat().st_mtime:return False
        return True
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={}
        while groups or tasks or not runtime_added:
            if not runtime_added and (out/'census_summary.json').exists():
                extra=[]
                for p in (out/'contexts').glob('*.json'):
                    for r in json.loads(p.read_text(encoding='utf-8'))['regions']:
                        if r['region_id'] not in known:known.add(r['region_id']);regions[r['region_id']]=r;extra.append(r)
                add_groups(extra);runtime_added=True
            for group,rs in list(groups.items()):
                if len(tasks)>=workers*2:break
                if stream and not ready(rs):continue
                tasks[pool.submit(audit_group,(str(out),rs))]=group;del groups[group]
            changed=False
            for f in list(tasks):
                if not f.done():continue
                results.extend(f.result());del tasks[f];changed=True
            if changed:
                save_json(out/'face_audit_progress.json',dict(regions=len(results),total=len(regions),seconds=time.perf_counter()-start))
                print('FACE_AUDIT',len(results),flush=True)
            if groups or tasks or not runtime_added:time.sleep(2)
    results.sort(key=lambda r:r['region_id']);totals={v:Counter() for v in ['BEFORE','P1','P2']}
    for r in results:
        for v in totals:totals[v].update(r[v])
    save_json(out/'face_consistency_summary.json',dict(regions=len(results),requested_regions=len(regions),
        counts={v:dict(c) for v,c in totals.items()},seconds=time.perf_counter()-start,
        comparison='Same local face CSR, same measured H elevations, same adjacent V pairs; overlap windows are not independent accuracy samples.'))
    save_json(out/'face_consistency_regions.json',results);csv_rows(out/'face_consistency_regions.csv',results)
    print('FACE_AUDIT_COMPLETE',len(results),{v:dict(c) for v,c in totals.items()},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stream',action='store_true');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(latest(),a.workers,a.stream)
