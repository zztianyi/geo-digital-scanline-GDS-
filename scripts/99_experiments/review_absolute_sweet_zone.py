"""Frozen-input absolute-core audit. Never reports coverage as accuracy."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import argparse
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import time
import numpy as np
from recognize_main_tracks import ROOT,cached_evidence,verify_result
from locc_data import FrozenProfiles,opc
from validate_observed_first_locc import target_canonical_profile
from dominant_observed_branch import extract_observed_branches,solve_dominant_branch
from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale

OUT=ROOT/'outputs/min_branch_sweet_zone/20260920'


def clean(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=clean),encoding='utf-8')


def save_csv(path,rows):
    rows=list(rows)
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def slim(row):
    return {k:v for k,v in row.items() if k not in ('fragment','descriptors','neighbor_matches','same_surface_neighbor_detail')}


def prepare():
    start=time.perf_counter()
    frozen=FrozenProfiles(ROOT/'outputs/orthogonal_scanline_constraint/20260919_081148')
    graph=cached_evidence(ROOT/'outputs/local_detail_full_validation/20260919_current/surface_audit',frozen)
    cache=OUT/'canonical_inventory.pkl'
    if cache.exists():
        with cache.open('rb') as f: graph['nodes']=pickle.load(f)
    else:
        for i,key in enumerate(frozen.keys):
            p=target_canonical_profile(frozen,key)
            uz=opc.to_suz(p.nodes,frozen.arc)[:,1:]
            for b in extract_observed_branches(p,uz):
                b.pop('records');graph['nodes'][(float(key),b['branch_id'])]=b
            if (i+1)%350==0: print(f'inventory {i+1}/{len(frozen.keys)}',flush=True)
        with cache.open('wb') as f: pickle.dump(graph['nodes'],f,protocol=5)
    if set(graph['nodes'])!=set(graph['membership']): raise AssertionError('Frozen branch identities changed')
    graph['track_lookup']={t['track_id']:t for t in graph['tracks']}
    proof_path=ROOT/'outputs/reconstruction_constraint_review/20260920/case5_filter_comparison.json'
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    graph['source_review_confirmations']={'89.75':dict(path=str(proof_path),sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest())} if proof.get('original_curve_connects_both_endpoints') else {}
    graph['nodes_by_s']={float(key):{} for key in frozen.keys}
    for node,b in graph['nodes'].items(): graph['nodes_by_s'][node[0]][node]=b
    rows=[]
    for s,nodes in graph['nodes_by_s'].items():
        scale=local_edge_scale(list(nodes.values()))
        for b in nodes.values():
            if b['kind']=='CLOSED_COMPONENT':continue
            rows.append(dict(s=s,**branch_metrics(b,scale)))
    save_csv(OUT/'absolute_core_distribution.csv',rows)
    quantiles={k:dict(zip(('min','p05','p25','p50','p75','p95','max'),map(float,np.quantile([r[k] for r in rows],[0,.05,.25,.5,.75,.95,1]))))
               for k in ('local_median_edge_length','full_arc_length','ASC_arc_length','canonical_node_count')}
    save_json(OUT/'input_distribution.json',dict(candidate_count=len(rows),quantiles=quantiles,prepare_seconds=time.perf_counter()-start))
    print(json.dumps(quantiles),flush=True)
    return frozen,graph


def compute(key,frozen,graph,policy):
    start=time.perf_counter()
    profile=target_canonical_profile(frozen,key)
    uz=opc.to_suz(profile.nodes,frozen.arc)[:,1:]
    branches=extract_observed_branches(profile,uz)
    nodes=dict(graph['nodes']);nodes.update({(float(key),b['branch_id']):b for b in branches})
    local={**graph,'nodes':nodes}
    result=solve_dominant_branch(profile,uz,branches=branches,surface_graph=local,target_s=float(key),
        policy=policy,input_z_range=frozen.arc.get('z_range'),source_data_required=key in graph['source_review_confirmations'])
    verify_result(profile,uz,result)
    assert all(j['xyz_distance_m']<=policy.connector_cap_m+1e-12 and j['R_syn']<policy.synthetic_ratio_limit for j in result.get('junctions',[]))
    assert all(b['accepted'] for b in result.get('route_contribution_budgets',[]))
    if key in graph['source_review_confirmations']:result['source_review_proof']=graph['source_review_confirmations'][key]
    selected=result['selection']['selected']
    result['selection']={**result['selection'],'selected':slim(selected) if selected else None,
        'candidates':[slim(r) for r in result['selection']['candidates']]}
    result['branch_audit']=[slim(r) for r in result.get('branch_audit',result['selection']['candidates'])]
    seq=result.get('route_branch_sequence',[])
    row=dict(slice_key=key,status=result['status'],anchor=selected['branch_id'] if selected else None,
        branch_sequence=seq,switches=result['branch_switch_count'],ABA=sum(a==c for a,b,c in zip(seq,seq[1:],seq[2:])),
        observed_m=result['observed_length_m'],synthetic_m=result['connector_length_m'],
        junction_count=len(result.get('junctions',[])),ambiguous=result['selection']['ambiguous'],
        candidate_count=len(result['branch_audit']),MBG_rejected=sum(not b['MBG_pass'] for b in result['branch_audit']),
        ASC_eligible=sum(b['ASC_exists'] for b in result['branch_audit']),
        rejected_surface_evidence=sum(b.get('reason')=='REJECT_SURFACE_EVIDENCE' for b in result['selection']['candidates'])+
            sum(r['reason']=='REJECT_SURFACE_EVIDENCE' for r in result.get('continuation_rejections',[])),
        detail_comparisons=result['selection'].get('detail_stage_comparisons',0)+result.get('handoff_detail_comparisons',0)+sum(r['detail_stage_comparisons'] for r in result.get('continuation_decisions',[])),
        unresolved_reasons=result.get('unresolved_reasons',[]),wall_seconds=time.perf_counter()-start)
    payload=dict(slice_key=key,result=result,branches=[dict(branch_id=b['branch_id'],kind=b['kind'],
        points_uz=b['points_uz'],arc_positions=b['arc_positions']) for b in branches])
    return payload,row


def run_all(frozen,graph,workers):
    start=time.perf_counter();rows=[];joins=[];index={};audits=[]
    policy=BranchPolicy()
    with (OUT/'main_tracks.pkl').open('wb') as f,ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(payload,row) in enumerate(pool.map(lambda key:compute(key,frozen,graph,policy),frozen.keys)):
            index[payload['slice_key']]=f.tell();pickle.dump(payload,f,protocol=5);rows.append(row)
            joins.extend(dict(slice_key=row['slice_key'],**j) for j in payload['result'].get('junctions',[]))
            audits.extend(dict(slice_key=row['slice_key'],**b) for b in payload['result']['branch_audit'])
            if (i+1)%100==0:print(f'recognition {i+1}/{len(frozen.keys)} {time.perf_counter()-start:.1f}s',flush=True)
    save_json(OUT/'main_track_index.json',index);save_csv(OUT/'slice_summary.csv',rows)
    save_csv(OUT/'branch_audit.csv',audits);save_csv(OUT/'junctions.csv',joins)
    ratios=[j['R_syn'] for j in joins]
    summary=dict(slice_count=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
        **{k:sum(r[k] for r in rows) for k in ('candidate_count','MBG_rejected','ASC_eligible','rejected_surface_evidence','detail_comparisons','switches','ABA','junction_count')},
        ASC_absent_short=sum(not b['ASC_exists'] for b in audits),
        connector_count=sum(j['xyz_distance_m']>1e-9 for j in joins),
        connectors_above={str(x):sum(j['xyz_distance_m']>x+1e-12 for j in joins) for x in (.005,.01,.02)},
        max_connector_m=max((j['xyz_distance_m'] for j in joins),default=0),
        ratio_quantiles=np.quantile(ratios,[0,.25,.5,.75,.95,1]) if ratios else [],
        ambiguous_count=sum(r['ambiguous'] for r in rows),observed_m=sum(r['observed_m'] for r in rows),
        synthetic_m=sum(r['synthetic_m'] for r in rows),wall_seconds=time.perf_counter()-start,workers=workers,
        policy=asdict(policy),accuracy_measured=False,production_entry_integrated=False)
    save_json(OUT/'summary.json',summary)
    print(json.dumps(summary,default=clean),flush=True)


def sweep(frozen,graph):
    # Fixed, predeclared coverage across the arc plus the three reviewed cases.
    keys=sorted(set([frozen.keys[i] for i in np.linspace(0,len(frozen.keys)-1,9,dtype=int)]+['89.75','104.30','122.25']),key=float)
    prepared={}
    for key in keys:
        p=target_canonical_profile(frozen,key);uz=opc.to_suz(p.nodes,frozen.arc)[:,1:];branches=extract_observed_branches(p,uz)
        nodes=dict(graph['nodes']);nodes.update({(float(key),b['branch_id']):b for b in branches})
        prepared[key]=(p,uz,branches,{**graph,'nodes':nodes})
    configurations=list(itertools.product((3,4,5,6),(2,3,4),(.005,.010,.020),(2.,4.,6.,8.)))
    rows=[];reference={};reference_curves={}
    for key,(p,uz,bs,g) in prepared.items():
        r=solve_dominant_branch(p,uz,branches=bs,surface_graph=g,target_s=float(key),policy=BranchPolicy())
        reference[key]=r.get('route_branch_sequence',[])
        reference_curves[key]=r['curve_uz'].copy()
    # Policy sweeps revisit exactly the same immutable canonical geometry.
    # Cache deterministic subcomputations, never decisions or edited outputs.
    # The uncached default above remains an equivalence check for every slice.
    import main_track_assembly as assembly
    import surface_track_selection as selection
    import surface_track_handoff as handoff
    original_join=assembly._nearest_join;original_detail=selection.same_surface_detail
    original_confidence=handoff.track_confidence
    join_cache={};detail_cache={};confidence_cache={};cache_counts=Counter()
    context=[None]
    def cached_join(left,right,required,locked_range,direction):
        digest=hashlib.sha256()
        for records in (left,right):
            digest.update(np.asarray([[r['edge_id'],r['branch_id'],r['t0'],r['t1']] for r in records],dtype=np.float64).tobytes())
            for field in ('points_uz','points_xyz'):
                digest.update(np.asarray([r[field] for r in records],dtype=np.float64).tobytes())
            digest.update(str(len(records)).encode())
        cache_key=(context[0],digest.digest(),tuple(required),tuple(locked_range),direction)
        if cache_key not in join_cache:
            join_cache[cache_key]=original_join(left,right,required,locked_range,direction);cache_counts['junction_misses']+=1
        else:cache_counts['junction_hits']+=1
        return copy.deepcopy(join_cache[cache_key])
    def cached_detail(g,node):
        cache_key=(context[0],node)
        if cache_key not in detail_cache:
            detail_cache[cache_key]=original_detail(g,node);cache_counts['detail_misses']+=1
        else:cache_counts['detail_hits']+=1
        return copy.deepcopy(detail_cache[cache_key])
    def cached_confidence(g,node,levels,*,policy=None,edge_scale=None):
        m=branch_metrics(g['nodes'][node],edge_scale,policy)
        cache_key=(context[0],node,m['MBG_pass'],m['ASC_start_arc'],m['ASC_end_arc'],tuple(levels))
        if cache_key not in confidence_cache:
            confidence_cache[cache_key]=original_confidence(g,node,levels,policy=policy,edge_scale=edge_scale);cache_counts['confidence_misses']+=1
        else:cache_counts['confidence_hits']+=1
        return confidence_cache[cache_key].copy()
    assembly._nearest_join=cached_join;selection.same_surface_detail=cached_detail;handoff.track_confidence=cached_confidence
    for i,(k,n,cap,mul) in enumerate(configurations):
        policy=BranchPolicy(K_guard=k,N_core_min=n,connector_cap_m=cap,core_length_multiplier=mul)
        counts=Counter();maximum=0.;ratios=[];combined_ratios=[]
        for key,(p,uz,bs,g) in prepared.items():
            context[0]=key
            r=solve_dominant_branch(p,uz,branches=bs,surface_graph=g,target_s=float(key),policy=policy,input_z_range=frozen.arc.get('z_range'),source_data_required=key in graph['source_review_confirmations'])
            if policy==BranchPolicy():
                np.testing.assert_array_equal(r['curve_uz'],reference_curves[key])
                assert r.get('route_branch_sequence',[])==reference[key]
            seq=r.get('route_branch_sequence',[]);audit=r.get('branch_audit',r['selection']['candidates'])
            assert all(j['xyz_distance_m']<=policy.connector_cap_m+1e-12 and j['R_syn']<policy.synthetic_ratio_limit for j in r.get('junctions',[]))
            assert all(b['accepted'] for b in r.get('route_contribution_budgets',[]))
            combined_ratios.extend(b['combined_R_syn'] for b in r.get('route_contribution_budgets',[]))
            counts.update(slices=1,MBG_rejected=sum(not b['MBG_pass'] for b in audit),
                short_branch_accepted=sum(not b['MBG_pass'] and b['branch_id'] in seq for b in audit),
                ABA=sum(a==c for a,b,c in zip(seq,seq[1:],seq[2:])),switches=r['branch_switch_count'],
                ambiguous=int(r['selection']['ambiguous']),stable_path=int(seq==reference[key]),
                stable_geometry=int(np.array_equal(r['curve_uz'],reference_curves[key])),
                local_tie_slices=int(r['selection'].get('comparison_ambiguous',False)),
                local_ambiguous_continuations=sum(d.get('comparison_ambiguous',False) for d in r.get('continuation_decisions',[])))
            counts[r['status']]+=1
            for j in r.get('junctions',[]):
                d=j['xyz_distance_m'];maximum=max(maximum,d);ratios.append(j['R_syn'])
                counts.update(connectors=int(d>1e-9),above_5mm=int(d>.005+1e-12),above_10mm=int(d>.01+1e-12),above_20mm=int(d>.02+1e-12))
        rows.append(dict(K_guard=k,N_core_min=n,connector_cap_m=cap,ASC_length_multiplier=mul,
            **counts,max_connector_m=maximum,max_R_syn=max(ratios,default=0),max_combined_R_syn=max(combined_ratios,default=0)))
        if (i+1)%12==0:print(f'sweep {i+1}/{len(configurations)}',flush=True)
    save_csv(OUT/'sweet_zone_parameter_sweep.csv',rows)
    assembly._nearest_join=original_join;selection.same_surface_detail=original_detail;handoff.track_confidence=original_confidence
    save_json(OUT/'sweep_scope.json',dict(keys=keys,configurations=len(configurations),full_factorial=True,
        default_policy=asdict(BranchPolicy()),scope='12 predeclared real slices; full 2800-slice run uses defaults',
        deterministic_subcomputation_cache=dict(cache_counts),default_routes_equal_uncached=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--sweep-only',action='store_true');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    frozen,graph=prepare()
    if not args.prepare_only:
        if not args.sweep_only:run_all(frozen,graph,args.workers)
        sweep(frozen,graph)
        save_json(OUT/'manifest.json',dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'scripts/04_structure_recognition'/n for n in ('branch_absolute_core.py','main_track_assembly.py','surface_track_selection.py','surface_track_handoff.py','dominant_observed_branch.py')]+[Path(__file__)]},
            plan_sha256=hashlib.sha256(Path('C:/Users/222/Downloads/GDS_Minimum_Branch_Absolute_Sweet_Zone_Codex_Plan.md').read_bytes()).hexdigest()))
