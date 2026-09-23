"""C03-C07 review only. The only recognition rerun is six reviewed C06 holes."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,pickle,time
from pathlib import Path
from datetime import datetime
import numpy as np
from run_targeted_issue_recheck import OLD,RECENT,SOURCE,read,js,latest
from run_local_conflict_v2 import (ROOT,load_inventory,save_json,mesh,manifest,raw_observations,
    PhysicalFaceContext,build_surface_graph,solve_dominant_branch,apply_internal_competitions,
    finish_surface_spine,verify,compact,fingerprints)
from main_track_assembly import _oriented
from region_joint_selection import prepare_region_locks
from reviewed_model_gap import is_estimated_gap,valid_estimated_gap


def output():return Path((ROOT/'outputs/selected_case_review/LATEST.txt').read_text(encoding='utf-8-sig').strip())


def setup():
    parent=ROOT/'outputs/selected_case_review';out=parent/datetime.now().strftime('%Y%m%d_%H%M%S')
    for name in ('figures','cases','routes','layers','audits'):(out/name).mkdir(parents=True,exist_ok=True)
    (parent/'LATEST.txt').write_text(str(out),encoding='utf-8')
    save_json(out/'lineage.json',dict(previous=str(latest()),reused=str(RECENT),raw=str(SOURCE),code=fingerprints()))
    return out


def evidence(out):
    configs=[c for c in js(latest()/'case_config.json') if c['name'] in ('C03','C04','C05','C07')]
    inventory=load_inventory(SOURCE,sorted({s for c in configs for s in c['profiles']},key=float))
    rows=[]
    fields=('branch_id','FaceTrack','MBG_pass','region_priority','face_continuity','face_context_slice_count',
            'in_ASC','forward_Z_m','forward_ASC_m','minimum_switches','full_arc_length','support_s_span')
    for c in configs:
        root=RECENT if c.get('reuse') else latest()
        for s in c['profiles']:
            route=read(root/'routes'/f'{s}.pkl')['result'];selection=route['selection']
            seed=[{k:x.get(k) for k in fields} for x in selection.get('physical_seed_candidates',[])]
            selected=selection.get('selected') or {}
            transitions=[]
            for d in route.get('continuation_decisions',[]):
                transitions.append({k:d.get(k) for k in ('frontier','branch_id','competition_z','feasibility_accepted','junction_reason','junction_diagnostic','identity_trace')}|
                    dict(candidates=[{k:x.get(k) for k in fields} for x in d.get('identity_candidates',[])]))
            rows.append(dict(case=c['name'],s=s,root=str(root),seed=seed,selected=selected.get('branch_id'),
                seed_trace=selection.get('trace'),decision_stage=selection['decision_stage'],
                branches=[{k:b.get(k) for k in ('branch_id','z_range','u_range','full_arc_length')} for b in inventory[s]],
                junctions=route.get('junctions',[]),sequence=route['route_branch_sequence'],transitions=transitions))
    save_json(out/'selection_evidence.json',rows)
    print('EVIDENCE',len(rows),out,flush=True)


def run_gaps(out):
    began=time.perf_counter();targets=['104.55','104.60','104.65','104.70','104.75','104.80']
    target_ids=[3,4,4,3,3,4]
    audit=js(RECENT/'audits/E.json');ck=audit['context_keys'];region=audit['region'];m=manifest(SOURCE)
    profiles=load_inventory(SOURCE,ck);a,lo,hi,_=mesh();cache=read(OLD/'stages/face_geometry_cache.pkl')
    context=PhysicalFaceContext(a,lo,hi,{float(k):v for k,v in profiles.items()},[region],geometry_cache=cache,levels=m['levels'])
    context.deadline=began+600;prepare_region_locks(context,[region]);context.reviewed_model_gaps={}
    approvals=[]
    for s,bid in zip(targets,target_ids):
        previous=read(RECENT/'routes'/f'{s}.pkl')['result'];last=previous['path_edges'][-1]
        b=next(b for b in profiles[s] if b['branch_id']==bid);first=_oriented(b['records'])[0]
        item=dict(review_id='USER_C06_MODEL_HOLE_20260923',reason='User confirmed original measurement/model hole; endpoints only; exclude structural recognition',
            direction='upper',current_branch_id=last['branch_id'],target_branch_id=bid,
            endpoint_xyz=[list(last['points_xyz'][1]),list(first['points_xyz'][0])])
        context.reviewed_model_gaps[float(s)]=[item];approvals.append(dict(s=s,**item))
    save_json(out/'reviewed_gap_approvals.json',approvals)
    graph=build_surface_graph([dict(s=float(k),branches=b) for k,b in profiles.items()],raw_observations(ck,m));graph['physical_face_context']=context
    preparation=time.perf_counter()-began;rows=[]
    for s in targets:
        tick=time.perf_counter()
        route=solve_dominant_branch(None,None,branches=profiles[s],surface_graph=graph,target_s=float(s),input_z_range=m['arc']['z_range'])
        route=apply_internal_competitions(profiles[s],route,graph,float(s))
        # No joint-band replay: this run validates reviewed holes, not a new
        # interpretation of the still-under-review cross-profile path metric.
        route['region_joint_stage_complete']=False
        route['joint_stage_scope']='NOT_RERUN_REVIEWED_MODEL_HOLE_ONLY'
        p1=time.perf_counter()-tick;result=finish_surface_spine(profiles[s],route,context,float(s),p1_seconds=p1)
        check=verify(profiles[s],route,result['layers'])
        gaps=[r for r in result['layers']['MAIN_SPINE'] if is_estimated_gap(r)]
        assert len(gaps)==1 and valid_estimated_gap(gaps[0]),s
        assert len(result['reconstruction']['structural_main_spine_runs'])==2,s
        assert all(not is_estimated_gap(r) for r in result['hanging_segments']),s
        assert all(not is_estimated_gap(r) for run in result['reconstruction']['structural_main_spine_runs'] for r in run),s
        row=dict(s=s,sequence=route['route_branch_sequence'],route_extent=route['route_z_extent'],
            gap_m=gaps[0]['distance_m'],estimated_structure_count=0,structural_runs=2,
            **check,**result['performance'],seconds=time.perf_counter()-tick)
        pickle.dump(dict(result=compact(route),row=row),(out/'routes'/f'{s}.pkl').open('wb'),protocol=5)
        pickle.dump({k:v for k,v in result.items() if k!='route'},(out/'layers'/f'{s}.pkl').open('wb'),protocol=5)
        rows.append(row);print('C06',s,row,flush=True)
        if time.perf_counter()-began>600:raise TimeoutError('Six-profile hole review exceeded 10-minute bound')
    save_json(out/'audits/C06.json',dict(rows=rows,seconds=time.perf_counter()-began,preparation_seconds=preparation,
        raw_context_count=len(ck),recognized_profiles=targets,code=fingerprints(),joint_band_rerun=False))


if __name__=='__main__':
    import sys
    out=setup() if '--setup' in sys.argv else output()
    if '--evidence' in sys.argv:evidence(out)
    if '--gaps' in sys.argv:run_gaps(out)
