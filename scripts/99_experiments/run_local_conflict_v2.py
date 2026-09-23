"""Local-case CLI only. Never schedules a full census implicitly."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,pickle,time,hashlib,cProfile,pstats
from pathlib import Path
from collections import Counter
import numpy as np
from run_production_integration_v2 import SOURCE,ROOT,mesh,manifest,raw_observations,load_inventory,compact,save_json,missing_core
from observed_surface_graph import build_surface_graph
from physical_face_context import PhysicalFaceContext
from dominant_observed_branch import solve_dominant_branch
from local_competitive_handoff import apply_internal_competitions
from surface_spine_pipeline import finish_surface_spine,join_region_spines

P0=['104.85','104.70','98.05','98.10','116.25','122.95','112.55','135.65']
BASE=ROOT/'outputs/production_integration_v2/20260922_075643'


def output_dir():return Path((ROOT/'outputs/local_conflict_v2/LATEST.txt').read_text(encoding='utf-8-sig').strip())


def fingerprints():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts/04_structure_recognition').glob('*.py')}


def verify(branches,route,layers=None):
    originals={r['edge_id']:r for b in branches for r in b['records']};max_error=0.;parts={}
    records=route['path_edges'] if layers is None else layers['MAIN_SPINE']+layers['SIDE_COMPONENTS']
    for r in records:
        if r['source'].startswith('OBSERVED'):
            raw=originals[r['edge_id']]
            for name in ('points_uz','points_xyz'):
                a,b=np.asarray(raw[name]);error=float(np.max(abs(np.asarray(r[name])-np.asarray([a+t*(b-a) for t in (r['t0'],r['t1'])]))))
                max_error=max(max_error,error);assert error<2e-10,(name,error)
            assert r['source_face_ids']==raw['source_face_ids'];assert r['source_segment_indices']==raw['source_segment_indices']
            parts.setdefault(r['edge_id'],[]).append(sorted([r['t0'],r['t1']]))
        else:
            assert not r['source_face_ids'];d=float(np.linalg.norm(np.diff(r['points_xyz'],axis=0)))
            from reviewed_model_gap import is_estimated_gap,valid_estimated_gap
            if is_estimated_gap(r):assert valid_estimated_gap(r)
            else:assert d<=r.get('distance_cap_m',.01)+1e-12,(d,r)
    for intervals in parts.values():
        intervals.sort();assert all(b[0]>=a[1]-1e-8 for a,b in zip(intervals,intervals[1:]))
        if layers is not None:
            assert abs(intervals[0][0])<1e-7 and abs(intervals[-1][1]-1)<1e-7
            assert all(abs(a[1]-b[0])<1e-7 for a,b in zip(intervals,intervals[1:]))
    if layers is not None:assert parts.keys()==originals.keys()
    spine=route['path_edges'] if layers is None else layers['MAIN_SPINE']
    assert all(np.linalg.norm(np.asarray(a['points_xyz'][1])-b['points_xyz'][0])<=2e-8 for a,b in zip(spine,spine[1:]))
    return dict(max_coordinate_error_m=max_error,source_records=len(parts),source_intervals_preserved=layers is not None)


def selectors(args,m,plan):
    targets=set();regions=[]
    if args.case_set=='regression':
        targets.update(P0);targets.update(f'{108.8+x*.05:.2f}' for x in range(-2,3));regions=plan['regions']
        for r in regions:targets.update(r['target_keys'])
    elif args.case_set=='pilot':targets.update(['104.85','104.70','89.95','90.00','108.80'])
    for s in args.slice:targets.add(f'{s:.2f}')
    if args.s_range:targets.update(k for k in m['keys'] if args.s_range[0]<=float(k)<=args.s_range[1])
    for rid in args.region_id:
        region=next(r for r in plan['regions'] if rid==r['region_id'] or rid in r.get('case_tags',[]))
        regions.append(region);targets.update(region['target_keys'])
    if not targets:raise ValueError('Specify --slice, --s-range, --region-id or --case-set; full census is not a default')
    if len(targets)>=len(m['keys']):raise ValueError('LOCAL_CASE_MODE forbids a 2800-profile census')
    return sorted(targets,key=float),list({r['region_id']:r for r in regions}.values())


def run(args):
    out=output_dir()
    if args.stage=='apply-band':
        from replay_local_joint_application import run as replay
        return replay()
    if args.stage=='report':
        from render_local_conflict_v2 import run_report
        return run_report(out)
    m=manifest(SOURCE);plan=json.loads((out/'local_plan.json').read_text(encoding='utf-8'))
    targets,regions=selectors(args,m,plan);started=time.perf_counter();hashes=fingerprints()
    if args.case_set=='regression' and args.stage!='p2':
        gate=json.loads((out/'RUNTIME_BUDGET.json').read_text(encoding='utf-8'))
        if gate['projected_seconds']>args.budget_seconds:raise TimeoutError('Pilot projection exceeds one hour after optimization')
        if gate['target_count']!=len(targets):raise ValueError('Budget projection does not cover this target set')
        if gate.get('code_sha256') is not None and gate['code_sha256']!=hashes:raise ValueError('Budget estimate requires recalibration after production code changes')
    all_keys=m['keys'];groups=[];used=set()
    for region in regions:
        keys=[k for k in targets if k in region['target_keys'] and k not in used]
        if keys:groups.append((keys,[region]));used.update(keys)
    for key in targets:
        if key not in used:
            near=[k for k in targets if k not in used and abs(float(k)-float(key))<.8]
            groups.append((near,[]));used.update(near)
    a,lo,hi,cells=mesh();face_cache=pickle.load((out/'stages/face_geometry_cache.pkl').open('rb'))
    rows=[];scope=[];all_bands=[];region_decisions=[];cache_stats=Counter();search_stats=Counter();stages=[]
    routes_by_key={};pilot_times=[]
    for keys,merged in groups:
        begin=time.perf_counter();indices=[all_keys.index(k) for k in keys]
        ctxkeys=all_keys[max(0,min(indices)-20):min(len(all_keys),max(indices)+21)]
        profiles=load_inventory(SOURCE,ctxkeys)
        context=PhysicalFaceContext(a,lo,hi,{float(k):v for k,v in profiles.items()},merged or cells,
            geometry_cache=face_cache,levels=m['levels'])
        context.deadline=started+args.budget_seconds
        graph=None
        if args.stage!='p2':
            graph=build_surface_graph([dict(s=float(k),branches=b) for k,b in profiles.items()],raw_observations(ctxkeys,m))
            graph['physical_face_context']=context
        setup_seconds=time.perf_counter()-begin;routes={};times={}
        for key in keys:
            began=time.perf_counter();file=out/'stages'/f'P1_{key}.pkl'
            if args.stage=='p2':
                payload=pickle.load(file.open('rb'))
                p2_only={'main_spine_components.py','observed_component_geometry.py'}
                assert all(payload['code'].get(k)==v for k,v in hashes.items() if k not in p2_only),'P1 stage cache invalidated by code changes'
                routes[key]=payload['route'];times[key]=payload['seconds'];continue
            route=solve_dominant_branch(None,None,branches=profiles[key],surface_graph=graph,
                target_s=float(key),input_z_range=m['arc']['z_range'])
            route=apply_internal_competitions(profiles[key],route,graph,float(key))
            times[key]=time.perf_counter()-began;pilot_times.append(times[key]);routes[key]=route
            if args.case_set!='pilot':
                pickle.dump(dict(route=route,seconds=times[key],code=hashes),(out/'stages'/f'P1_preband_{key}.pkl').open('wb'),protocol=5)
            print('LOCAL_P1',key,round(times[key],3),route['route_branch_sequence'],flush=True)
            elapsed=time.perf_counter()-started
            if len(pilot_times)>=10:
                projected=elapsed+float(np.median(pilot_times[-10:]))*(len(targets)-len(pilot_times))
                if projected>args.budget_seconds:raise TimeoutError(f'Updated local projection {projected:.1f}s exceeds budget; stop before expanding workload')
            save_json(out/('pilot_progress.json' if args.case_set=='pilot' else 'local_progress.json'),
                dict(completed=len(pilot_times),targets=len(targets),seconds=elapsed,last_slice=key))
            if elapsed>args.budget_seconds:raise TimeoutError('Local runtime budget exceeded; full census was not started')
        band_started=time.perf_counter()
        bands=join_region_spines(routes,profiles,context,merged) if merged and args.stage!='p2' else []
        all_bands.extend(bands);band_seconds=time.perf_counter()-band_started
        for key,route in routes.items():
            if time.perf_counter()-started>args.budget_seconds:raise TimeoutError('P2 local runtime budget exceeded')
            route['analysis_only']=False
            verify(profiles[key],route)
            result=finish_surface_spine(profiles[key],route,context,float(key),p1_seconds=times[key])
            check=verify(profiles[key],route,result['layers']);missing=missing_core(profiles[key],route)
            row=dict(s=key,**result['performance'],missing_max_m=max((r['missing_m'] or 0. for r in missing),default=0.),
                sequence=route['route_branch_sequence'],P2_roles=dict(Counter(c['role'] for c in result['layers']['components'])),
                max_connector_m=max((j['xyz_distance_m'] for j in route['junctions']),default=0.),**check)
            if args.case_set=='pilot':base=out/'stages/pilot';base.mkdir(exist_ok=True)
            else:base=out
            if base==out:
                pickle.dump(dict(result=compact(route),row=row,missing_ASC=missing),(out/'routes'/f'{key}.pkl').open('wb'),protocol=5)
                pickle.dump({k:v for k,v in result.items() if k!='route'},(out/'layers'/f'{key}.pkl').open('wb'),protocol=5)
                pickle.dump(dict(route=route,seconds=times[key],code=hashes),(out/'stages'/f'P1_{key}.pkl').open('wb'),protocol=5)
            else:pickle.dump(result,(base/f'{key}.pkl').open('wb'),protocol=5)
            rows.append(row);routes_by_key[key]=route
        scope.extend(context.scope_audit);region_decisions.extend(context.region_decisions.values())
        cache_stats.update(context.stats);search_stats.update(context.junction_cache.stats)
        stages.append(dict(keys=keys,setup_seconds=setup_seconds,junction_band_seconds=band_seconds,total_seconds=time.perf_counter()-begin))
        print('GROUP_DONE',keys[0],keys[-1],round(stages[-1]['total_seconds'],2),flush=True)
    invalid=[r for r in scope if r['competitive_core_invoked'] and r['competitor_count']<2]
    assert not invalid,invalid
    summary=dict(target_count=len(targets),local_only=True,seconds=time.perf_counter()-started,
        CC_INVOCATIONS=sum(r['competitive_core_invoked'] for r in scope),CC_NONCOMPETITIVE_INVOCATIONS=len(invalid),
        face_cache=dict(cache_stats),junction_cache=dict(search_stats),stages=stages,
        code_unchanged=hashes==fingerprints(),code_sha256=hashes,rows=rows)
    name='pilot' if args.case_set=='pilot' else 'local'
    save_json(out/f'{name}_summary.json',summary);save_json(out/f'{name}_scope.json',scope)
    save_json(out/f'{name}_region_decisions.json',region_decisions);save_json(out/f'{name}_junction_bands.json',all_bands)
    pickle.dump(face_cache,(out/'stages/face_geometry_cache.pkl').open('wb'),protocol=5)
    print('COMPLETE',name,len(targets),summary['seconds'],summary['CC_NONCOMPETITIVE_INVOCATIONS'],flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description='LOCAL_CASE_MODE; no census option')
    p.add_argument('--slice',type=float,action='append',default=[]);p.add_argument('--s-range',type=float,nargs=2)
    p.add_argument('--region-id',action='append',default=[]);p.add_argument('--case-set',choices=['pilot','regression'])
    p.add_argument('--stage',choices=['all','p2','apply-band','report'],default='all');p.add_argument('--budget-seconds',type=float,default=3600)
    p.add_argument('--profile',action='store_true');args=p.parse_args()
    profiler=cProfile.Profile() if args.profile else None
    try:
        if profiler:profiler.enable()
        run(args)
    except TimeoutError as e:
        save_json(output_dir()/'PERFORMANCE_STOP.json',dict(reason=str(e),full_census_started=False));raise
    finally:
        if profiler:
            profiler.disable();profiler.dump_stats(str(output_dir()/'local_profile.prof'))
            with (output_dir()/'local_profile.txt').open('w',encoding='utf-8') as f:pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(30)
