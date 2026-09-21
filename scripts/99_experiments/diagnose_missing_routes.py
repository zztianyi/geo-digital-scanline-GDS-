"""Current-code census and stage diagnostics; no recognition source changes."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from collections import Counter
import argparse,json,pickle,time
import numpy as np
from validate_physical_topology import ROOT,output_path,load_inventory,csv_rows
from run_face_provenance_validation import raw_observations,manifest,digest
from census_directional_consistency import one_result,exact_memoization,code_hashes,save_json
from observed_surface_graph import build_surface_graph

SOURCE=output_path()


def latest():return Path((ROOT/'outputs/face_branch_consistency/LATEST.txt').read_text().strip())


def prepare():
    out=ROOT/'outputs/face_branch_consistency'/datetime.now().strftime('%Y%m%d_%H%M%S')
    (out/'routes').mkdir(parents=True);(out/'figures').mkdir()
    (out.parent/'LATEST.txt').write_text(str(out),encoding='utf-8')
    save_json(out/'manifest.json',dict(source=str(SOURCE),code_sha256=code_hashes(),
       purpose='Current missing-route stage attribution and branch-only same-FaceTrack detail validation',
       parameters_changed=False,FaceTrack_used_in_recognition=False,
       census_context='Fixed 10-target blocks, fresh H graph plus 20 slices/1m halo on both sides; all current physical branches. Not the historical global graph.'))
    print(out)


def compact(r):
    # Original geometry is already in the immutable physical inventory.
    def clean(row):return {k:v for k,v in row.items() if k!='fragment'}
    x=dict(r);x['branch_audit']=[clean(v) for v in r.get('branch_audit',[])]
    x['selection']=dict(r['selection'],candidates=[clean(v) for v in r['selection']['candidates']],
                        selected=clean(r['selection']['selected']) if r['selection']['selected'] else None)
    return x


def gap_rows(key,r):
    if not r.get('route_z_extent') or not r.get('candidate_z_extent'):
        return [dict(s=key,side='both',missing_z_m=None,stage='NO_RELIABLE_SURFACE_BRANCH')]
    actual=r['route_z_extent'];extent=r['candidate_z_extent'];output=[]
    for side,j in [('lower',0),('upper',1)]:
        gap=(actual[0]-extent[0]) if j==0 else (extent[1]-actual[1])
        if gap<=1e-6:continue
        rejected=[v for v in r.get('continuation_rejections',[]) if v['frontier']==side]
        decisions=[v for v in r.get('continuation_decisions',[]) if v['frontier']==side]
        chosen=decisions[-1]['branch_id'] if decisions else None
        selected_rejections=[v for v in rejected if v.get('branch_id')==chosen] if chosen is not None else []
        decisive=selected_rejections[-1] if selected_rejections else (rejected[-1] if rejected else {})
        junction=decisive.get('junction',{})
        stage=('CONNECTOR_GATE' if junction and decisive.get('reason') in ('LONG_GAP_UNRESOLVED','REJECT_SYNTHETIC_DOMINANCE')
          else 'JUNCTION_SEARCH' if decisive.get('reason')=='NO_VALID_JUNCTION'
          else 'CONTINUATION_IDENTITY_TIE' if decisive.get('reason')=='AMBIGUOUS_CONTINUATION'
          else 'RETAINED_CORE_GATE' if decisive.get('reason')=='REJECT_NO_RETAINED_CORE'
          else 'PRESELECTION_Z_GAP' if decisive.get('minimum_possible_gap_m') is not None
          else 'NO_EXTENDING_OPTION')
        output.append(dict(s=key,side=side,missing_z_m=float(gap),route_boundary_z=actual[j],candidate_boundary_z=extent[j],
          severity='AT_LEAST_1M' if gap>=1. else 'AT_LEAST_10MM' if gap>=.01 else 'SUB_10MM_EXTENT',
          stage=stage,reason=decisive.get('reason'),chosen_branch_id=chosen,
          from_branch_id=junction.get('from_branch_id'),to_branch_id=junction.get('to_branch_id'),
          connector_m=junction.get('xyz_distance_m'),minimum_z_gap_m=decisive.get('minimum_possible_gap_m'),
          contribution_pass=decisive.get('contribution_pass'),retained_ASC_m=decisive.get('retained_ASC_arc_length'),
          all_rejection_reasons=sorted({v['reason'] for v in rejected}),
          internal_handoff_reasons=sorted({v['reason'] for v in r.get('internal_handoff_audit',[]) if v['frontier']==side}),
          anchor_branch_id=r['selection']['selected']['branch_id'] if r['selection']['selected'] else None,
          attribution='LAST_DECISIVE_LOG_STAGE_NOT_YET_CAUSAL_COUNTERFACTUAL'))
    return output


def block(task):
    out,start,stop=task;out=Path(out);m=manifest(SOURCE);targets=m['keys'][start:stop]
    if all((out/'routes'/f'{s}.pkl').exists() for s in targets):
        return [pickle.load((out/'routes'/f'{s}.pkl').open('rb'))['row'] for s in targets]
    keys=m['keys'][max(0,start-20):min(len(m['keys']),stop+20)]
    branches=load_inventory(SOURCE,keys);obs=raw_observations(keys,m)
    graph=build_surface_graph([dict(s=float(k),branches=b) for k,b in branches.items()],obs)
    exact_memoization();rows=[]
    for key in targets:
        file=out/'routes'/f'{key}.pkl'
        if file.exists():rows.append(pickle.load(file.open('rb'))['row']);continue
        result,row=one_result(key,branches[key],graph,m)
        missing=gap_rows(key,result)
        row.update(missing_z_max_m=max((r['missing_z_m'] or 0 for r in missing),default=0),
                   missing_sides=len(missing),context_start=keys[0],context_end=keys[-1])
        with file.with_suffix('.pending').open('wb') as f:pickle.dump(dict(result=compact(result),row=row,missing=missing),f,protocol=5)
        file.with_suffix('.pending').replace(file);rows.append(row)
        print('DONE',key,round(row['seconds'],2),round(row['missing_z_max_m'],3),flush=True)
    return rows


def run(out,workers):
    assert code_hashes()==json.loads((out/'manifest.json').read_text(encoding='utf-8'))['code_sha256']
    start=time.perf_counter();m=manifest(SOURCE);results=[]
    # 10-target blocks bound long-tail scheduling and share local evidence work.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(block,(str(out),i,min(i+10,len(m['keys'])))) for i in range(0,len(m['keys']),10)]
        for f in as_completed(futures):
            results.extend(f.result());save_json(out/'progress.json',dict(completed=len(results),total=len(m['keys']),seconds=time.perf_counter()-start))
            print('PROGRESS',len(results),flush=True)
    results.sort(key=lambda r:float(r['slice_key']));missing=[]
    for r in results:missing.extend(pickle.load((out/'routes'/f"{r['slice_key']}.pkl").open('rb'))['missing'])
    csv_rows(out/'current_missing_routes.csv',missing)
    csv_rows(out/'all_route_inventory.csv',results)
    checkpoint=json.loads((out/'census_checkpoint.json').read_text(encoding='utf-8')) if (out/'census_checkpoint.json').exists() else {}
    segment=time.perf_counter()-start
    summary=dict(count=len(results),seconds=segment+checkpoint.get('elapsed_segment_seconds',0.),workers=workers,
      resumed_segment_seconds=segment,checkpoint=checkpoint,
      slices_with_extent_deficit=len({r['s'] for r in missing}),
      slices_with_at_least_1m_deficit=len({r['s'] for r in missing if (r['missing_z_m'] or 0)>=1}),
      severity_counts=dict(Counter(r.get('severity') for r in missing)),
      stage_counts=dict(Counter(r['stage'] for r in missing)),
      large_gap_stage_counts=dict(Counter(r['stage'] for r in missing if (r['missing_z_m'] or 0)>=1)),
      rows=results,code_unchanged=code_hashes()==json.loads((out/'manifest.json').read_text(encoding='utf-8'))['code_sha256'])
    save_json(out/'census_summary.json',summary);print('CENSUS COMPLETE',summary['slices_with_at_least_1m_deficit'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(latest(),a.workers)
