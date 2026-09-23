"""Apply frozen regional junction choices; no P0/P1 or candidate search rerun."""
import json,pickle,time,shutil
from collections import Counter
from run_local_conflict_v2 import (output_dir,fingerprints,SOURCE,mesh,manifest,load_inventory,
    PhysicalFaceContext,finish_surface_spine,verify,missing_core,compact,save_json)
from region_junction_band import replay_region_junction_application


def sync_connector_metadata(payload,route):
    """Keep cached P2 metadata current when a joint choice leaves geometry unchanged."""
    def signature(r):return (r['source'],tuple(x for p in r['points_xyz'] for x in p))
    kinds={'COMPETITIVE_HANDOFF','REGION_JOINT_HANDOFF'}
    current={signature(r):r for r in route['path_edges'] if r.get('handoff_kind') in kinds}
    seen=set();changed=0
    def visit(value):
        nonlocal changed
        if not isinstance(value,(dict,list,tuple)) or id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,dict):
            if value.get('handoff_kind') in kinds and 'points_xyz' in value:
                target=current.get(signature(value))
                if target is not None:
                    for field in ('handoff_kind','synthetic','decision_mode','distance_cap_m','distance_m'):
                        if value.get(field)!=target.get(field):
                            value[field]=target.get(field);changed+=1
            for child in value.values():visit(child)
        else:
            for child in value:visit(child)
    visit(payload);return changed


def run():
    out=output_dir();started=time.perf_counter();hashes=fingerprints()
    base=json.loads((out/'local_summary.json').read_text(encoding='utf-8'))
    if (out/'summary_before_joint_application.json').exists():raise ValueError('Application replay already has a baseline; do not overwrite it')
    assert all(base['code_sha256'][k]==v for k,v in hashes.items() if k!='region_junction_band.py')
    shutil.copy2(out/'local_summary.json',out/'summary_before_joint_application.json')
    shutil.copy2(out/'local_junction_bands.json',out/'bands_before_joint_application.json')
    audit=json.loads((out/'local_junction_bands.json').read_text(encoding='utf-8'))
    plan=json.loads((out/'local_plan.json').read_text(encoding='utf-8'));m=manifest(SOURCE)
    a,lo,hi,_=mesh();cache=pickle.load((out/'stages/face_geometry_cache.pkl').open('rb'))
    rows={r['s']:r for r in base['rows']};changed=[];stages=[]
    for region in plan['regions']:
        begin=time.perf_counter();keys=region['target_keys'];idx=[m['keys'].index(k) for k in keys]
        context_keys=m['keys'][max(0,min(idx)-20):min(len(m['keys']),max(idx)+21)]
        profiles=load_inventory(SOURCE,context_keys)
        context=PhysicalFaceContext(a,lo,hi,{float(k):v for k,v in profiles.items()},[region],geometry_cache=cache,levels=m['levels'])
        routes={};times={}
        for key in keys:
            p=pickle.load((out/'stages'/f'P1_preband_{key}.pkl').open('rb'))
            assert all(p['code'][k]==v for k,v in hashes.items() if k!='region_junction_band.py')
            routes[key]=p['route'];times[key]=p['seconds']
        setup=time.perf_counter()-begin;apply_start=time.perf_counter()
        replay_region_junction_application(routes,profiles,[r for r in audit if r['region_id']==region['region_id']])
        apply_seconds=time.perf_counter()-apply_start
        for key,route in routes.items():
            if base['seconds']+time.perf_counter()-started>3600:raise TimeoutError('Combined local run plus application replay exceeded one hour; stop')
            old=pickle.load((out/'routes'/f'{key}.pkl').open('rb'))['result']
            def signature(records):
                return [(r['source'],r.get('edge_id'),r.get('t0'),r.get('t1'),tuple(x for p in r['points_xyz'] for x in p)) for r in records]
            geometry_changed=signature(old['path_edges'])!=signature(route['path_edges'])
            if geometry_changed:
                changed.append(key);verify(profiles[key],route)
                result=finish_surface_spine(profiles[key],route,context,float(key),p1_seconds=times[key])
                check=verify(profiles[key],route,result['layers'])
                pickle.dump({k:v for k,v in result.items() if k!='route'},(out/'layers'/f'{key}.pkl').open('wb'),protocol=5)
                rows[key].update(result['performance'],P2_roles=dict(Counter(c['role'] for c in result['layers']['components'])),**check)
            else:
                path=out/'layers'/f'{key}.pkl';payload=pickle.load(path.open('rb'))
                if sync_connector_metadata(payload,route):pickle.dump(payload,path.open('wb'),protocol=5)
            route.update(analysis_only=False,region_joint_stage_complete=True)
            missing=missing_core(profiles[key],route)
            rows[key].update(sequence=route['route_branch_sequence'],missing_max_m=max((r['missing_m'] or 0. for r in missing),default=0.),
                max_connector_m=max((j['xyz_distance_m'] for j in route['junctions']),default=0.))
            pickle.dump(dict(result=compact(route),row=rows[key],missing_ASC=missing),(out/'routes'/f'{key}.pkl').open('wb'),protocol=5)
            pickle.dump(dict(route=route,seconds=times[key],code=hashes,frozen_P1_code=base['code_sha256']),
                (out/'stages'/f'P1_{key}.pkl').open('wb'),protocol=5)
        stages.append(dict(keys=keys,setup_seconds=setup,junction_band_seconds=apply_seconds,total_seconds=time.perf_counter()-begin,
            stage='FROZEN_JUNCTION_APPLICATION'))
        print('APPLICATION_GROUP',region['region_id'],len(changed),round(stages[-1]['total_seconds'],2),flush=True)
    elapsed=time.perf_counter()-started
    base.update(seconds=base['seconds']+elapsed,rows=list(rows.values()),code_sha256=hashes,code_unchanged=hashes==fingerprints(),
        stages=base['stages']+stages,application_replay=dict(seconds=elapsed,changed_profiles=changed,
            P1_and_candidates_reused=True,candidate_generation_code=base['code_sha256'],CC_scope_reused_without_new_CC_decisions=True))
    save_json(out/'local_summary.json',base);save_json(out/'local_junction_bands.json',audit)
    save_json(out/'application_replay.json',base['application_replay'])
    print('APPLICATION_COMPLETE',len(changed),elapsed,'COMBINED',base['seconds'],flush=True)


if __name__=='__main__':run()
