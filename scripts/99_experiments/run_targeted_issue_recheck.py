"""Recheck historical problem profiles only; frozen successes are never rerun."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,pickle,time
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
import numpy as np
from run_local_conflict_v2 import ROOT,SOURCE,manifest,save_json,fingerprints
from audit_production_face_consistency import level_tracks,classify

OLD=ROOT/'outputs/local_conflict_v2/20260922_124130'
RECENT=ROOT/'outputs/region_single_track/20260922_164301'


def read(p):return pickle.load(p.open('rb'))
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def latest():return Path((ROOT/'outputs/targeted_issue_recheck/LATEST.txt').read_text(encoding='utf-8-sig').strip())


def compatible_code(old,new,out):
    if old==new:return True
    path=out/'performance_compatibility.json'
    if not path.exists():return False
    allowed=js(path)
    return old==allowed['before'] and new==allowed['after']


def inventory():
    started=time.perf_counter();parent=ROOT/'outputs/targeted_issue_recheck';out=parent/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True);(parent/'LATEST.txt').write_text(str(out),encoding='utf-8')
    plan=js(OLD/'local_plan.json');summary=js(OLD/'local_summary.json');m=manifest(SOURCE)
    cache=read(OLD/'stages/face_geometry_cache.pkl');current=fingerprints()
    assert all(js(RECENT/'audits'/f'{tag}.json')['code']==current for tag in ['E','F','IDENTITY'])
    per_region={r['region_id']:dict(region=r,levels=np.asarray([z for z in m['levels'] if r['bounds'][4]<=z<=r['bounds'][5]]),hits={}) for r in plan['regions']}
    profiles=[];flags=[]
    for ix,row in enumerate(summary['rows']):
        s=row['s'];recent=(RECENT/'routes'/f'{s}.pkl').exists();root=RECENT if recent else OLD
        route=read(root/'routes'/f'{s}.pkl')['result'];layers=read(root/'layers'/f'{s}.pkl')['layers']
        seq=route['route_branch_sequence'];valid={r['branch_id'] for r in route['branch_audit'] if r['MBG_pass']}
        reasons=route.get('unresolved_reasons',[]);extent=route['route_z_extent'];target=route['candidate_z_extent']
        joins=[]
        for j in route.get('junctions',[]):
            joins.append({k:j.get(k) for k in ['from_branch_id','to_branch_id','a_point_uz','b_point_uz','xyz_distance_m','handoff_kind','decision_mode','region_id','from_FaceTrack','to_FaceTrack','distance_cap_m']})
        info=dict(s=s,result_root=str(root),current_code=recent,sequence=seq,status=route.get('status'),extent_covered=route.get('extent_covered'),
            route_extent=extent,candidate_extent=target,missing_extent_m=max(extent[0]-target[0],target[1]-extent[1],0.),
            unresolved=reasons,joins=joins,repeated_branch=len(seq)!=len(set(seq)),roles=dict(Counter(c['role'] for c in layers['components'])),
            max_connector_m=max([j['xyz_distance_m'] for j in joins] or [0.]))
        profiles.append(info)
        if info['missing_extent_m']>1e-5:flags.append(dict(kind='MISSING_MAIN_EXTENT',s=s,amount_m=info['missing_extent_m'],current_code=recent))
        if info['repeated_branch']:flags.append(dict(kind='REPEATED_BRANCH_DETOUR',s=s,current_code=recent))
        for j in joins:
            if j['decision_mode']=='COMPETITIVE_SELECTION' and j['xyz_distance_m']>.002+1e-10:
                flags.append(dict(kind='OLD_HANDOFF_OVER_2MM',s=s,current_code=recent,amount_m=j['xyz_distance_m'],z=j['a_point_uz'][1]))
        for reg in per_region.values():
            r=reg['region']
            if s not in r['target_keys']:continue
            labels=cache[tuple(r['bounds'])];levels=reg['levels']
            main=level_tracks(layers['MAIN_SPINE'],r,labels,levels)
            raw=level_tracks([e for e in layers['MAIN_SPINE']+layers['SIDE_COMPONENTS'] if e.get('branch_id') in valid],r,labels,levels)
            reg['hits'][s]=dict(main=main,raw=raw)
        if (ix+1)%50==0:print('READ_FROZEN',ix+1,'/',len(summary['rows']),round(time.perf_counter()-started,2),flush=True)
    events=[];regional=[]
    for rid,reg in per_region.items():
        r=reg['region'];counts=Counter();levels=reg['levels'];ss=r['target_keys']
        for a,b in zip(ss,ss[1:]):
            x,y=reg['hits'][a],reg['hits'][b];categories=[]
            for i,z in enumerate(levels):
                c=classify(x['main'][i],y['main'][i]);counts[c]+=1
                absent=[s for s,h in [(a,x),(b,y)] if not h['main'][i] and h['raw'][i]]
                kind='MAIN_ABSENT_WITH_QUALIFIED_RAW' if absent else ('NEIGHBOR_SOURCE_DIFFERENCE' if c=='IDENTITY_SWITCH' else ('MULTIPLE_SOURCE_HEIGHT' if c=='AMBIGUOUS_MULTIPLE' else None))
                categories.append((kind,tuple(absent)))
            start=0
            while start<len(levels):
                stop=start+1
                while stop<len(levels) and categories[stop]==categories[start]:stop+=1
                kind,absent=categories[start]
                if kind:
                    event=dict(region_id=rid,tag=r['case_tags'][0],left=a,right=b,kind=kind,z_min=float(levels[start]),z_max=float(levels[stop-1]),samples=stop-start,
                        span_m=float(levels[stop-1]-levels[start]),missing_profiles=list(absent),current_code=(RECENT/'routes'/f'{a}.pkl').exists() and (RECENT/'routes'/f'{b}.pkl').exists())
                    event['event_id']=f'I{len(events)+1:04d}';events.append(event)
                start=stop
        regional.append(dict(region_id=rid,counts=dict(counts),profiles=len(ss)))
    save_json(out/'frozen_inventory.json',dict(profiles=profiles,profile_flags=flags,events=events,regional=regional,seconds=time.perf_counter()-started,
        previous_count=len(profiles),reused_current_count=sum(p['current_code'] for p in profiles),code=current,source=str(SOURCE)))
    counts=Counter(e['kind'] for e in events)
    strong=[e for e in events if not e['current_code'] and e['kind']!='MULTIPLE_SOURCE_HEIGHT' and e['span_m']>.100001]
    keys={s for e in strong for s in [e['left'],e['right']]}|{f['s'] for f in flags if not f['current_code']}
    print('INVENTORY_DONE',out,'events',counts,'flags',Counter((f['kind'],f['current_code']) for f in flags),'strong_old_profiles',len(keys),sorted(keys,key=float),flush=True)


def select():
    out=latest();data=js(out/'frozen_inventory.json');profiles={r['s']:r for r in data['profiles']};seeds={};events=[]
    for e in data['events']:
        if e['current_code'] or e['span_m']<=.100001:continue
        # A border between results computed by different revisions is lineage,
        # not evidence that either revision internally selected inconsistent tracks.
        if profiles[e['left']]['current_code']!=profiles[e['right']]['current_code']:continue
        for s in [e['left'],e['right']]:seeds.setdefault(s,[]).append(e['event_id'])
        events.append(e)
    for e in data['profile_flags']:
        if e['current_code'] or e['kind']=='OLD_HANDOFF_OVER_2MM':continue
        seeds.setdefault(e['s'],[]).append(e['kind'])
    m=manifest(SOURCE);keys=m['keys'];targets=set(seeds)
    for s in seeds:
        index=keys.index(s)
        targets.update(k for k in keys[max(0,index-2):index+3] if k in profiles and not profiles[k]['current_code'])
    assert len(targets)<150 and not any(profiles[s]['current_code'] for s in targets)
    runs=[]
    for s in sorted(targets,key=float):
        if not runs or float(s)-float(runs[-1][-1])>.051:runs.append([])
        runs[-1].append(s)
    regions=js(OLD/'local_plan.json')['regions'];groups=[]
    for index,ss in enumerate(runs,1):
        region=next((r for r in regions if ss[0] in r['target_keys']),None)
        if ss[0]=='108.70':region=read(SOURCE/'region_data/J.pkl')['region']
        if region is not None:
            region=dict(region);parent=region['region_id'];context_keys=region['target_keys']
        else:parent='ISOLATED_'+ss[0];context_keys=ss
        lo=keys.index(context_keys[0]);hi=keys.index(context_keys[-1]);context_keys=keys[max(0,lo-2):hi+3]
        groups.append(dict(group_id=f'Q{index:02d}',targets=ss,seed_profiles=[s for s in ss if s in seeds],
            parent=parent,region=region,context_keys=context_keys,
            event_ids=[e['event_id'] for e in events if e['left'] in ss or e['right'] in ss]))
    save_json(out/'selection.json',dict(groups=groups,seeds=seeds,target_count=len(targets),seed_count=len(seeds),neighbor_count=len(targets)-len(seeds),
        event_count=len(events),events=events,reused_current_profiles=[s for s,r in profiles.items() if r['current_code']],
        not_rerun_policy='Only threshold>2mm, <=100mm differences, mixed-revision borders and current-code results are not standalone rerun reasons.',code=fingerprints()))
    (out/'EXECUTION_LEDGER.md').write_text('# 历史局部问题复核执行台账\n\n'
        '计划：docs/superpowers/plans/2026-09-22-targeted-issue-recheck.md\n\n'
        f'已只读核查650份冻结结果；当前代码175份直接复用。本次识别清单{len(targets)}条，其中问题线{len(seeds)}条、必要邻线{len(targets)-len(seeds)}条。\n\n'
        'Ruling: 仅旧接缝超过新2mm门槛不单独触发重算；旧门槛为10mm，这不是独立的形态错误证据。混合版本边界不当作同版本邻线不一致。\n\n'
        'Ruling: 原有完整区域面标签和原始分支参与区域身份统计；只对清单内目标执行P1/P2，不识别其余区域测线。\n\n'
        'Ruling: 沿用本地工作区和现有未提交源码，不创建另一个缺失这些改动的工作树，不提交或推送。\n',encoding='utf-8')
    print('SELECTED',len(targets),'in',len(groups),'groups',flush=True)


def run(only=None):
    from run_local_conflict_v2 import mesh,load_inventory,raw_observations,PhysicalFaceContext,build_surface_graph,solve_dominant_branch,apply_internal_competitions,finish_surface_spine,join_region_spines,verify,compact
    from region_joint_selection import prepare_region_locks
    out=latest();selection=js(out/'selection.json');groups=selection['groups'];began=time.perf_counter();hashes=fingerprints()
    if only:groups=[g for g in groups if g['group_id'] in only]
    for folder in ['preband','routes','layers','audits']:(out/folder).mkdir(exist_ok=True)
    prior_wall=js(out/'performance_compatibility.json')['interrupted_wall_seconds'] if (out/'performance_compatibility.json').exists() else 0.
    m=manifest(SOURCE);a,lo,hi,_=mesh();cache=read(OLD/'stages/face_geometry_cache.pkl');context=None;parent=None;completed=[];active_profiles=0
    for group in groups:
        tag=group['group_id'];region=group['region'];start=time.perf_counter()
        saved=js(out/'audits'/f'{tag}.json') if (out/'audits'/f'{tag}.json').exists() else None
        if saved and compatible_code(saved['code'],hashes,out) and saved['group']==group:
            completed.append(dict(group_id=tag,profiles=len(saved['rows']),seconds=saved['seconds'],band_seconds=saved['band_seconds'],reused_complete=True))
            print('REUSE_COMPLETE',tag,flush=True);continue
        if parent!=group['parent']:
            ck=group['context_keys'];profiles=load_inventory(SOURCE,ck)
            context=PhysicalFaceContext(a,lo,hi,{float(k):b for k,b in profiles.items()},[region] if region else [],geometry_cache=cache,levels=m['levels'])
            context.deadline=began+max(1.,3600-prior_wall);prepare_region_locks(context,[region] if region else [])
            graph=build_surface_graph([dict(s=float(k),branches=b) for k,b in profiles.items()],raw_observations(ck,m));graph['physical_face_context']=context
            parent=group['parent'];print('CONTEXT_READY',parent,len(ck),'raw profiles; no full recognition',round(time.perf_counter()-start,2),flush=True)
        scope_start=len(context.scope_audit);routes={};times={};reused=0
        for s in group['targets']:
            file=out/'preband'/f'{s}.pkl';payload=read(file) if file.exists() else None
            if payload and compatible_code(payload['code'],hashes,out) and payload['context_keys']==group['context_keys']:
                route=payload['route'];seconds=payload['seconds'];reused+=1
            else:
                tick=time.perf_counter();route=solve_dominant_branch(None,None,branches=profiles[s],surface_graph=graph,target_s=float(s),input_z_range=m['arc']['z_range'])
                route=apply_internal_competitions(profiles[s],route,graph,float(s));seconds=time.perf_counter()-tick
                pickle.dump(dict(route=route,code=hashes,seconds=seconds,context_keys=group['context_keys']),file.open('wb'),protocol=5)
            verify(profiles[s],route);routes[s]=route;times[s]=seconds
            print('P1',tag,s,round(seconds,3),route['route_branch_sequence'],route.get('unresolved_reasons'),flush=True)
            if prior_wall+time.perf_counter()-began>3600:raise TimeoutError('One-hour local budget exceeded')
        tick=time.perf_counter();bands=join_region_spines(routes,profiles,context,[region] if region else []);band_seconds=time.perf_counter()-tick;rows=[]
        for s,route in routes.items():
            result=finish_surface_spine(profiles[s],route,context,float(s),p1_seconds=times[s]);check=verify(profiles[s],route,result['layers'])
            old=read(OLD/'routes'/f'{s}.pkl')['result'];before=old['route_z_extent'];after=route['route_z_extent']
            row=dict(s=s,sequence=route['route_branch_sequence'],before_extent=before,after_extent=after,
                extent_regression_m=max(after[0]-before[0],before[1]-after[1],0.),**result['performance'],**check)
            pickle.dump(dict(result=compact(route),row=row),(out/'routes'/f'{s}.pkl').open('wb'),protocol=5)
            pickle.dump({k:v for k,v in result.items() if k!='route'},(out/'layers'/f'{s}.pkl').open('wb'),protocol=5);rows.append(row)
        scope=context.scope_audit[scope_start:];assert not any(r['competitive_core_invoked'] and r['competitor_count']<2 for r in scope)
        audit=[]
        for item in bands:
            row={k:v for k,v in item.items() if k not in ['candidates','legal_candidates','selected']}
            if 'selected' in item:row['selected']={k:v for k,v in item['selected'].items() if k not in ['corridor','junction']}
            audit.append(row)
        record=dict(group=group,rows=rows,bands=audit,seconds=time.perf_counter()-start,band_seconds=band_seconds,reused_P1=reused,
            scope_counts=dict(CC=sum(r['competitive_core_invoked'] for r in scope),noncompetitive_CC=0),decisions=context.region_decisions,code=hashes)
        save_json(out/'audits'/f'{tag}.json',record);completed.append(dict(group_id=tag,profiles=len(rows),seconds=record['seconds'],band_seconds=band_seconds))
        print('GROUP_DONE',tag,'seconds',round(record['seconds'],3),'bands',Counter((r.get('applied',False),r.get('reason')) for r in audit),flush=True)
        active_profiles+=len(rows);elapsed=time.perf_counter()-began;done=sum(r['profiles'] for r in completed);remaining=sum(len(g['targets']) for g in groups)-done
        if active_profiles>=5 and prior_wall+elapsed+elapsed/active_profiles*remaining>3600:raise TimeoutError('Projected runtime exceeds one hour after optimization; stop and report')
    save_json(out/'run_summary.json',dict(groups=completed,seconds=prior_wall+time.perf_counter()-began,resume_seconds=time.perf_counter()-began,interrupted_wall_seconds=prior_wall,code=hashes,unchanged=hashes==fingerprints()))
    print('TARGETED_COMPLETE',sum(g['profiles'] for g in completed),round(time.perf_counter()-began,3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','select','run']);p.add_argument('--only',nargs='+');args=p.parse_args()
    if args.stage=='inventory':inventory()
    elif args.stage=='select':select()
    else:run(args.only)
