"""A3-only parallel replay and trace audit against frozen A1/A2.

No mesh/index rebuild fallback; missing or incompatible cache is an error.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import time
STARTED=time.perf_counter()
import sys,json,pickle,hashlib,argparse
from pathlib import Path
from datetime import datetime
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'scripts/04_structure_recognition'
sys.path.insert(0,str(CORE))
from facetrack_phase1_evidence import Phase1Policy,POLICY_VERSION,slice_evidence,decide_slice,micro_difference_decisions
from facetrack_local_preference import LocalPreferencePolicy,segment_region,reduce_subregion

SOURCE=ROOT/'outputs/facetrack_voting_phase1/20260923_094440'
OLD_LOCAL=ROOT/'outputs/facetrack_local_preference/20260923_105650'
CASE_IDS=('R00806','R00268','R00629','R00204','R00132')


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def row_key(r):return r['region_id'],r['s_index'],r['FaceTrack']
def groups(rows):
    result=defaultdict(list)
    for row in rows:result[row['region_id'],row['s_index']].append(row)
    return result
def write_rows(path,rows):pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')


def policies():
    default=Phase1Policy()
    return {'default':default,
            'loose':replace(default,continuity_good_min_coverage=.875,continuity_good_min_exit_m=.20,
                            continuity_weak_max_coverage=.45,continuity_weak_max_exit_m=.025),
            'strict':replace(default,continuity_good_min_coverage=.925,continuity_good_min_exit_m=.30,
                             continuity_weak_max_coverage=.55,continuity_weak_max_exit_m=.075)}


def init_worker(cache,policy_dict):
    global _CACHE,_POLICIES
    _CACHE=Path(cache);_POLICIES={k:Phase1Policy(**v) for k,v in policy_dict.items()}


def score_v(task):
    si,s,regions=task;began=time.perf_counter();cpu=time.process_time()
    with (_CACHE/'slices'/f'{s:.2f}.pkl').open('rb') as f:geometry=pickle.load(f)
    rows=[];evidence_s=0.;decision_s=0.
    for region in regions:
        tick=time.perf_counter();raw=slice_evidence(region,si,s,geometry,_POLICIES['default']);evidence_s+=time.perf_counter()-tick
        tick=time.perf_counter();rows.extend(decide_slice(raw,_POLICIES['default']));decision_s+=time.perf_counter()-tick
    return rows,dict(s=s,pid=os.getpid(),seconds=time.perf_counter()-began,cpu_s=time.process_time()-cpu,
                     evidence_s=evidence_s,decision_s=decision_s)


def sensitivity_v(rows):
    # Still V-local, in the same ProcessPool. No neighboring V or geometry IO.
    by_slice=groups(rows);result={}
    for label in ('loose','strict'):
        result[label]=[r for rr in by_slice.values() for r in decide_slice(rr,_POLICIES[label])]
    return result


def distribution(rows):
    levels=[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1.]
    fields=['cell_coverage_fraction','exit_continuation_m','competition_arc_m','CC_fraction']
    return {label:dict(n=len(rr),quantiles={field:{str(q):float(v) for q,v in zip(levels,np.quantile([r[field] for r in rr],levels))}
                                           for field in fields})
            for label,rr in [('all_candidates',rows),('MBG_pass',[r for r in rows if r['MBG']])]}


def compare_votes(old,new):
    before=groups(old);after=groups(new);result=[]
    if before.keys()!=after.keys():raise AssertionError('Changed A1/A2 (region,V) population')
    for key,rr in sorted(after.items()):
        previous=before[key];old_vote=previous[0]['slice_vote'];new_vote=rr[0]['slice_vote'];changed=old_vote!=new_vote
        prior=next((r for r in previous if r['FaceTrack']==old_vote),None)
        eligible=[r for r in previous if r['present'] and r['MBG']]
        same_old_rank=[];margin=None;through_material=False
        if prior:
            same_old_rank=[r for r in eligible if r['FaceTrack']!=old_vote and r['through_region']==prior['through_region'] and r['in_CC']==prior['in_CC']]
            if same_old_rank:margin=prior['exit_continuation_m']-max(r['exit_continuation_m'] for r in same_old_rank)
            rank=lambda r:(int(r['in_CC']),round(r['exit_continuation_m']/.001))
            best=max(map(rank,eligible));without_through=[r['FaceTrack'] for r in eligible if rank(r)==best]
            through_material=prior['through_region'] and without_through!=[old_vote]
        tags=[]
        if changed:
            if through_material:tags.append('OLD_THROUGH_BOOLEAN_WIN_REMOVED')
            if margin is not None and margin>1e-9:
                tags.append('OLD_MICRO_CONTINUITY_WIN_REMOVED' if margin<=.020000001 else 'OLD_EXACT_CONTINUATION_WIN_REMOVED')
            rejected=next((r for r in rr if r['FaceTrack']==old_vote),None)
            if rejected and rejected['elimination_reason']=='REJECT_MBG':tags.append('NEW_MBG_REJECT')
            if rejected and rejected['elimination_reason']=='REJECT_CONTINUITY_WEAK':tags.append('NEW_CONTINUITY_WEAK_REJECT')
            if rr[0]['final_decision_reason']=='CC_WIN':tags.append('NEW_CC_WIN')
            if new_vote is None:tags.append('NEW_AMBIGUOUS')
            if not tags:tags.append('NEW_QUALITATIVE_CONTINUITY_WIN')
        else:tags.append('UNCHANGED_STRONG_CASE' if new_vote is not None else 'UNCHANGED_AMBIGUOUS')
        result.append(dict(region_id=key[0],s_index=key[1],s=rr[0]['s'],old_slice_vote=old_vote,new_slice_vote=new_vote,
                           changed=changed,change_reason=tags[0],change_reasons=tags,old_exit_margin_m=margin,
                           old_through_boolean_material=through_material,new_decision_reason=rr[0]['final_decision_reason']))
    return result


def reduce_tables(rows,out,policy):
    by_region=defaultdict(list)
    for row in rows:by_region[row['region_id']].append(row)
    tick=time.perf_counter();subregions=[sub for rid,rr in sorted(by_region.items()) for sub in segment_region(rid,rr,policy)]
    b0=time.perf_counter()-tick;tick=time.perf_counter()
    decisions=[reduce_subregion(sub,by_region[sub.parent_region_id],policy) for sub in subregions];b1=time.perf_counter()-tick
    save_json(out/'decision_subregion_index.json',dict(stage='B0_LOCAL_PREFERENCE',policy=asdict(policy),
               parent_stage='INITIAL_CONFLICT_REGION',spatial_extent_rule='Frozen parent mask restricted to s_indices',subregions=[s.to_dict() for s in subregions]))
    save_json(out/'subregion_track_decisions.json',dict(stage='B1_FROZEN_SUBREGION_FACE_DECISION',route_application=False,
               policy=asdict(policy),decisions=[d.to_dict() for d in decisions]))
    assignment={(d.parent_region_id,si):d.dominant_FaceTrack for s,d in zip(subregions,decisions) for si in s.s_indices}
    if set(assignment)!=set(groups(rows)):raise AssertionError('B0/B1 lost (region,V)')
    return subregions,decisions,assignment,dict(B0_s=b0,B1_s=b1)


def audit_summary(rows):
    by_slice=groups(rows);reasons=Counter(r['elimination_reason'] for r in rows)
    decisions=Counter(rr[0]['final_decision_reason'] for rr in by_slice.values())
    return dict(candidate_count=len(rows),slice_count=len(by_slice),elimination_reasons=dict(reasons),
                elimination_percent={k:100*v/len(rows) for k,v in reasons.items()},slice_decision_reasons=dict(decisions),
                slice_decision_percent={k:100*v/len(by_slice) for k,v in decisions.items()},
                ambiguous_slices=sum(rr[0]['slice_vote'] is None for rr in by_slice.values()),
                continuity_classes_MBG_pass=dict(Counter(r['continuity_class'] for r in rows if r['MBG'])),
                MICRO_DIFFERENCE_DECISION_COUNT=len(micro_difference_decisions(rows)))


def run(out,workers):
    if workers<2:raise ValueError('This audit requires ProcessPool parallelism (>=2 workers)')
    began=time.perf_counter();out.mkdir(parents=True,exist_ok=False);timing={}
    tick=time.perf_counter();manifest=load_json(SOURCE/'manifest.json');cache=Path(manifest['index_cache'])
    if not (cache/'COMPLETE.json').is_file():raise FileNotFoundError('Frozen A1/A2 cache incomplete; rebuild forbidden')
    frozen=[SOURCE/n for n in ('manifest.json','branch_facetrack_index.parquet','conflict_region_index.json','per_slice_track_scores.parquet')]
    frozen += [OLD_LOCAL/n for n in ('manifest.json','decision_subregion_index.json','subregion_track_decisions.json')]
    frozen += [CORE/n for n in ('facetrack_phase1_regions.py','facetrack_phase1_vote.py','facetrack_local_preference.py','branch_absolute_core.py')]
    frozen += [ROOT/'scripts/99_experiments/run_facetrack_voting_phase1.py',cache/'COMPLETE.json',cache/'branch_index.parquet',cache/'conflict_region_index.json']
    frozen += sorted((cache/'slices').glob('*.pkl'))
    hashes={str(p):digest(p) for p in frozen}
    old=pq.read_table(SOURCE/'per_slice_track_scores.parquet').to_pylist()
    regions=load_json(SOURCE/'conflict_region_index.json')['regions']
    if digest(SOURCE/'conflict_region_index.json')!=digest(cache/'conflict_region_index.json'):raise AssertionError('Frozen A2 copies differ')
    old_policy=manifest['policy'];policy_set=policies()
    if any(getattr(policy_set['default'],field)!=old_policy[field] for field in ('cell_m','exit_margin_m')):
        raise AssertionError('A1 scales / exit measurement cap changed')
    # Verify index extraction identities without reading any original mesh data.
    import inspect
    from facetrack_phase1_evidence import segment_cells
    cell_hash=hashlib.sha256(json.dumps(inspect.getsource(segment_cells),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if cell_hash!=manifest['index_contract']['cell_geometry_code']:raise AssertionError('A1 segment_cells changed')
    save_json(out/'old_A3_distributions.json',distribution(old))
    timing['input_hashes_and_distribution_s']=time.perf_counter()-tick
    per_v=defaultdict(list);positions={r['s_index']:r['s'] for r in old}
    for r in regions:
        by_v=defaultdict(list)
        for cell in r['cells']:by_v[cell[0]].append(cell)
        for si,cells in by_v.items():per_v[si].append(dict(region_id=r['region_id'],tracks=r['tracks'],cells=cells))
    tasks=[(si,positions[si],rs) for si,rs in sorted(per_v.items())]
    tick=time.perf_counter();rows=[];stats=[];variant_rows={'loose':[],'strict':[]};peak=psutil.Process().memory_info().rss
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(cache),{k:asdict(p) for k,p in policy_set.items()})) as pool:
        futures=[pool.submit(score_v,task) for task in tasks]
        for count,f in enumerate(as_completed(futures),1):
            rr,stat=f.result();rows.extend(rr);stats.append(stat)
            if count%400==0 or count==len(tasks):
                print('A3',count,'/',len(tasks),'seconds',round(time.perf_counter()-tick,2),flush=True)
                peak=max(peak,psutil.Process().memory_info().rss+sum(p.memory_info().rss for p in psutil.Process().children() if p.is_running()))
        rows.sort(key=row_key)
        write_rows(out/'per_slice_track_scores.parquet',rows)
        trace=[{k:v for k,v in r.items() if k not in ('witnesses','competition_intervals')} for r in rows]
        write_rows(out/'decision_trace.parquet',trace)
        timing['A3_default_parallel_and_trace_export_s']=time.perf_counter()-tick
        tick=time.perf_counter();per_v_rows=defaultdict(list)
        for row in rows:per_v_rows[row['s_index']].append(row)
        futures=[pool.submit(sensitivity_v,rr) for _,rr in sorted(per_v_rows.items())]
        for f in as_completed(futures):
            for label,rr in f.result().items():variant_rows[label].extend(rr)
        for rr in variant_rows.values():rr.sort(key=row_key)
        timing['sensitivity_parallel_s']=time.perf_counter()-tick
    timing['pool_total_s']=time.perf_counter()-began-timing['input_hashes_and_distribution_s']
    # Complete population and immutable A1/MBG membership checks.
    if [row_key(r) for r in rows]!=sorted(row_key(r) for r in old):raise AssertionError('Candidate population changed')
    old_map={row_key(r):r for r in old}
    for row in rows:
        prior=old_map[row_key(row)]
        if any(row[k]!=prior[k] for k in ('present','MBG','member_branches','eligible_branches')):raise AssertionError('A1 membership changed')
        if row['slice_vote'] is not None and row['FaceTrack']==row['slice_vote'] and row['eliminated']:raise AssertionError('Eliminated winner')
    micro=micro_difference_decisions(rows)
    save_json(out/'micro_difference_audit.json',dict(MICRO_DIFFERENCE_DECISION_COUNT=len(micro),violations=micro,
               method='Selected candidate independently checked against every MBG-eligible peer with equal continuity class and CC.'))
    if micro:raise AssertionError('Micro-difference decision audit failed')
    differences=compare_votes(old,rows);write_rows(out/'old_new_slice_votes.parquet',differences)
    local_policy=LocalPreferencePolicy(**load_json(OLD_LOCAL/'manifest.json')['policy'])
    subs,decisions,assignment,bt=reduce_tables(rows,out,local_policy);timing.update(bt)
    sensitivity=[];base_votes={k:rr[0]['slice_vote'] for k,rr in groups(rows).items()}
    for label,rr in variant_rows.items():
        folder=out/'sensitivity'/label;folder.mkdir(parents=True)
        variants=groups(rr);votes={k:r[0]['slice_vote'] for k,r in variants.items()}
        changed=sum(v!=base_votes[k] for k,v in votes.items())
        flips=sum(v is not None and base_votes[k] is not None and v!=base_votes[k] for k,v in votes.items())
        resolved=sum(v is not None for v in base_votes.values())
        resolved_changed=sum(base_votes[k] is not None and v!=base_votes[k] for k,v in votes.items())
        tick=time.perf_counter();ss,dd,aa,unused=reduce_tables(rr,folder,local_policy)
        aggregate_s=time.perf_counter()-tick
        write_rows(folder/'slice_votes.parquet',[dict(region_id=k[0],s_index=k[1],s=variants[k][0]['s'],
                    default_vote=base_votes[k],variant_vote=v,changed=v!=base_votes[k]) for k,v in sorted(votes.items())])
        variant_micro=micro_difference_decisions(rr)
        if variant_micro:raise AssertionError('Sensitivity micro-difference audit failed')
        summary=dict(policy=asdict(policy_set[label]),label=label,changed_votes=changed,changed_fraction=changed/len(votes),
                     direct_winner_identity_flips=flips,direct_flip_fraction=flips/len(votes),default_resolved=resolved,
                     default_resolved_changed=resolved_changed,default_resolved_changed_fraction=resolved_changed/max(1,resolved),
                     B1_assignment_changes=sum(aa[k]!=v for k,v in assignment.items()),subregions=len(ss),
                     acceptance=changed/len(votes)<=.05 and flips/len(votes)<=.01,aggregate_and_export_s=aggregate_s,
                     audit=audit_summary(rr),case_changes={rid:sum(v!=base_votes[k] for k,v in votes.items() if k[0]==rid) for rid in CASE_IDS})
        save_json(folder/'summary.json',summary);sensitivity.append(summary)
    summary=audit_summary(rows)
    summary.update(old_ambiguous_slices=sum(rr[0]['slice_vote'] is None for rr in groups(old).values()),
        changed_votes=sum(r['changed'] for r in differences),change_reason_counts=dict(Counter(r['change_reason'] for r in differences)),
        change_tag_counts=dict(Counter(tag for r in differences for tag in r['change_reasons'])),
        B0_subregions=len(subs),B1_resolved_subregions=sum(not d.ambiguous for d in decisions),
        B1_ambiguous_subregions=sum(d.ambiguous for d in decisions),B1_resolved_region_V=sum(v is not None for v in assignment.values()),
        B1_MAJORITY_OPPOSES_WINNER_count=sum(d.MAJORITY_OPPOSES_WINNER for d in decisions),sensitivity=sensitivity,
        sensitivity_acceptance=all(x['acceptance'] for x in sensitivity),policy=asdict(policy_set['default']),
        b0_b1_policy=asdict(local_policy))
    save_json(out/'summary.json',summary)
    unchanged=all(digest(p)==h for p,h in hashes.items())
    if not unchanged:raise AssertionError('Frozen inputs or A1/A2/B0/B1 source changed')
    timing['A3_B0_B1_default_s']=timing['A3_default_parallel_and_trace_export_s']+timing['B0_s']+timing['B1_s']
    timing['run_without_figures_s']=time.perf_counter()-began
    timing['entry_including_imports_s']=time.perf_counter()-STARTED
    save_json(out/'manifest.json',dict(source=str(SOURCE),old_local_preference=str(OLD_LOCAL),index_cache=str(cache),
        policy_version=POLICY_VERSION,policies={k:asdict(v) for k,v in policy_set.items()},b0_b1_policy=asdict(local_policy),
        workers=workers,worker_pids=sorted({s['pid'] for s in stats}),profile_tasks=len(tasks),score_rows=len(rows),
        initial_regions=len(regions),A1_runs=0,A2_runs=0,mesh_reads=0,route_operations=0,synthetic_geometry=0,
        frozen_input_hashes=hashes,frozen_inputs_unchanged=unchanged,A1_segment_cells_source_unchanged=True,
        code_hashes={str(p):digest(p) for p in (CORE/'facetrack_phase1_evidence.py',Path(__file__))},
        timing=timing,sampled_peak_parent_children_RSS_bytes=peak,
        worker_cpu_s=sum(s['cpu_s'] for s in stats),worker_evidence_s=sum(s['evidence_s'] for s in stats),
        worker_decision_s=sum(s['decision_s'] for s in stats),sensitivity_acceptance=summary['sensitivity_acceptance']))
    print('OUTPUT',str(out),'A3+B0+B1',round(timing['A3_B0_B1_default_s'],3),'MICRO',len(micro),
          'sensitivity',summary['sensitivity_acceptance'],flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/facetrack_a3_qualitative'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args();run(args.out.resolve(),args.workers)
