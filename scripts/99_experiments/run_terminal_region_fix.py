"""Bounded E/F and 89.95-local regression; no full census route."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,pickle,time
from pathlib import Path
from collections import Counter
from run_local_conflict_v2 import (ROOT,SOURCE,mesh,manifest,load_inventory,raw_observations,
    PhysicalFaceContext,build_surface_graph,solve_dominant_branch,apply_internal_competitions,
    finish_surface_spine,join_region_spines,verify,compact,save_json,fingerprints)
from region_joint_selection import prepare_region_locks

BASE=ROOT/'outputs/local_conflict_v2/20260922_124130'


def output():return Path((ROOT/'outputs/region_single_track/LATEST.txt').read_text(encoding='utf-8-sig').strip())


def run(pilot=False,only=None,band_only=False):
    out=output();began=time.perf_counter();hashes=fingerprints();m=manifest(SOURCE)
    plan=json.loads((BASE/'local_plan.json').read_text(encoding='utf-8'));a,lo,hi,_=mesh()
    cache=pickle.load((BASE/'stages/face_geometry_cache.pkl').open('rb'))
    for folder in ['routes','layers','preband','audits']: (out/folder).mkdir(exist_ok=True)
    groups=[]
    for tag in ['E','F','IDENTITY']:
        if only and tag not in only:continue
        label='AUTO_IDENTITY_REVIEW' if tag=='IDENTITY' else tag
        region=next(r for r in plan['regions'] if label in r['case_tags'])
        targets=(['104.30'] if tag=='E' else ['130.45']) if pilot and tag!='IDENTITY' else region['target_keys']
        if tag=='IDENTITY':targets=['89.85','89.90','89.95','90.00','90.05']
        context_keys=region['target_keys'] if tag!='IDENTITY' else targets
        low_index=m['keys'].index(context_keys[0]);high_index=m['keys'].index(context_keys[-1])
        context_keys=m['keys'][max(0,low_index-2):high_index+3]
        start=time.perf_counter();profiles=load_inventory(SOURCE,context_keys)
        context=PhysicalFaceContext(a,lo,hi,{float(k):v for k,v in profiles.items()},[region],geometry_cache=cache,levels=m['levels'])
        context.deadline=began+3600
        locks=prepare_region_locks(context,[region]);print('REGION',tag,'targets',len(targets),'locks',len(locks),'dominant',context.region_decisions[region['region_id']]['dominant_FaceTrack'],flush=True)
        graph=build_surface_graph([dict(s=float(k),branches=b) for k,b in profiles.items()],raw_observations(context_keys,m))
        graph['physical_face_context']=context;routes={};times={};cached=0
        for key in targets:
            file=out/'preband'/f'{key}.pkl';payload=pickle.load(file.open('rb')) if file.exists() else None
            exempt={'region_junction_band.py'} if band_only else set()
            can_reuse=payload is not None and payload['context_keys']==context_keys and all(payload['code'].get(k)==v for k,v in hashes.items() if k not in exempt)
            if band_only and not can_reuse:raise ValueError('P1 cache invalidated; cannot use band-only replay')
            if can_reuse:
                route=payload['route'];times[key]=payload['seconds'];cached+=1
            else:
                tick=time.perf_counter();route=solve_dominant_branch(None,None,branches=profiles[key],surface_graph=graph,
                    target_s=float(key),input_z_range=m['arc']['z_range'])
                route=apply_internal_competitions(profiles[key],route,graph,float(key));times[key]=time.perf_counter()-tick
                pickle.dump(dict(route=route,seconds=times[key],code=hashes,context_keys=context_keys),file.open('wb'),protocol=5)
            verify(profiles[key],route);routes[key]=route
            print('P1',tag,key,round(times[key],3),route['route_branch_sequence'],'cache' if can_reuse else '',flush=True)
            elapsed=time.perf_counter()-began
            if elapsed>3600:raise TimeoutError('Local run exceeded one hour')
            fresh=list(times.values())
            if len(fresh)>=3 and elapsed+sum(fresh)/len(fresh)*(len(targets)-len(fresh))>3600:raise TimeoutError('Projected local run exceeds one hour; optimize before continuing')
        band_started=time.perf_counter();bands=join_region_spines(routes,profiles,context,[region]);band_seconds=time.perf_counter()-band_started
        rows=[]
        for key,route in routes.items():
            result=finish_surface_spine(profiles[key],route,context,float(key),p1_seconds=times[key]);check=verify(profiles[key],route,result['layers'])
            row=dict(s=key,sequence=route['route_branch_sequence'],**result['performance'],**check)
            # A through source stays one ordered run. Necessary terminal entry/exit
            # may lie inside the enclosing region box; the box is not a core lock.
            by={b['branch_id']:b for b in profiles[key]}
            for bid,(low,high) in context.region_locks.get(float(key),{}).items():
                from competitive_surface_selection import local_arc_positions
                rr=[r for r in route['path_edges'] if r.get('branch_id')==bid and r['source'].startswith('OBSERVED')]
                intervals=sorted(sorted(p) for p in local_arc_positions(rr,by[bid]))
                assert intervals,('REGION_SOURCE_LOST',key,bid)
                assert all(abs(x[1]-y[0])<1e-8 for x,y in zip(intervals,intervals[1:])),('REGION_SOURCE_FRAGMENTED',key,bid)
            pickle.dump(dict(result=compact(route),row=row),(out/'routes'/f'{key}.pkl').open('wb'),protocol=5)
            pickle.dump({k:v for k,v in result.items() if k!='route'},(out/'layers'/f'{key}.pkl').open('wb'),protocol=5);rows.append(row)
        assert not any(e['competitive_core_invoked'] and e['competitor_count']<2 for e in context.scope_audit)
        data=dict(tag=tag,region=region,targets=targets,context_keys=context_keys,locks=locks,rows=rows,
            region_decisions=context.region_decisions,bands=bands,scope=context.scope_audit,cached_P1=cached,
            seconds=time.perf_counter()-start,band_seconds=band_seconds,code=hashes)
        save_json(out/'audits'/f'{tag}.json',data);groups.append(dict(tag=tag,profiles=len(targets),seconds=data['seconds'],band_seconds=band_seconds))
        print('GROUP_DONE',tag,data['seconds'],'bands',Counter((r.get('applied',False),r.get('reason')) for r in bands),flush=True)
    save_json(out/('pilot_run.json' if pilot else 'local_run.json'),dict(groups=groups,seconds=time.perf_counter()-began,code=hashes,code_unchanged=hashes==fingerprints()))
    print('LOCAL_COMPLETE',time.perf_counter()-began,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--only',nargs='+',choices=['E','F','IDENTITY']);p.add_argument('--band-only',action='store_true');args=p.parse_args()
    run(args.pilot,args.only,args.band_only)
