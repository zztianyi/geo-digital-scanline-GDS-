"""Scoped physical-topology inventory; frozen geometry/provenance are immutable."""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, json, pickle, sys, time
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/99_experiments'))
from census_directional_consistency import save_json, code_hashes
from locc_data import opc
from validate_observed_first_locc import target_canonical_profile
from dominant_observed_branch import extract_observed_branches
from branch_absolute_core import branch_metrics, local_edge_scale

PRIOR=ROOT/'outputs/directional_full_census/20260920'


def output_path():
    return Path((ROOT/'outputs/face_provenance_validation/LATEST.txt').read_text(encoding='utf-8').strip())


def csv_rows(path, rows, fields=None):
    rows=list(rows)
    names=fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def load_inventory(out, keys):
    idx=json.loads((out/'branch_index.json').read_text(encoding='utf-8'))
    result={}
    with (out/'branches.pkl').open('rb') as f:
        for key in keys:
            f.seek(idx[key]);result[key]=pickle.load(f)
    return result


def inventory(out):
    start=time.perf_counter();base=json.loads((out/'BASELINE_MANIFEST.json').read_text(encoding='utf-8'))
    m=base['input_manifest'];keys=m['keys'];oldidx=json.loads((PRIOR/'branch_index.json').read_text(encoding='utf-8'))
    affected=[]
    with (PRIOR/'branches.pkl').open('rb') as f:
        for key in keys:
            bs=pickle.load(f)
            if any(r['node_ids'][0]==r['node_ids'][1] for b in bs for r in b['records']):affected.append(key)
    known={r['slice_key'] for r in json.loads((PRIOR/'topology_exclusions.json').read_text(encoding='utf-8'))
           if r['branching_caused_only_by_self_edges']}
    assert len(known)==27 and known.issubset(affected)
    save_json(out/'affected_slices.json',dict(all_self_loop_slices=affected,known_27=sorted(known,key=float)))
    with (Path(m['baseline'])/'slices.pkl').open('rb') as f:slices=pickle.load(f)
    frozen=SimpleNamespace(slices=slices,arc=m['arc'])
    rows=[];branches_rows=[];recovered=[];overreach=[];totals=Counter();checked={};oldfile=(PRIOR/'branches.pkl').open('rb')

    def process(key):
        p=target_canonical_profile(frozen,key)
        oldfile.seek(oldidx[key]);before=pickle.load(oldfile)
        after=extract_observed_branches(p,opc.to_suz(p.nodes,m['arc'])[:,1:])
        old_records={r['edge_id']:r for b in before for r in b['records']}
        assert set(old_records)==set(range(len(p.edges)))
        for eid,r in old_records.items():
            a=np.array(r['points_xyz']);b=p.nodes[p.edges[eid]]
            assert np.array_equal(a,b) or np.array_equal(a,b[::-1]),(key,eid,'geometry')
            assert r['source_face_ids']==p.source_face_ids[eid],(key,eid,'faces')
            assert r['source_segment_indices']==p.source_segment_indices[eid],(key,eid,'segments')
        np.testing.assert_array_equal(p.components,p.physical_components)
        emitted=[eid for b in after for eid in b['edge_order']]
        assert len(emitted)==len(set(emitted))==int(p.topology_active_edge.sum())
        assert set(emitted)==set(np.flatnonzero(p.topology_active_edge))
        oldscale,newscale=local_edge_scale(before),local_edge_scale(after)
        bm={b['branch_id']:branch_metrics(b,oldscale) for b in before}
        am={b['branch_id']:branch_metrics(b,newscale) for b in after}
        edge_to_new={eid:b['branch_id'] for b in after for eid in b['edge_order']}
        for b in before:
            overlap=sorted({edge_to_new[eid] for eid in b['edge_order'] if eid in edge_to_new})
            if b['kind']=='OPEN_SURFACE_BRANCH':
                assert len(overlap)==1 and after[overlap[0]]['kind']=='OPEN_SURFACE_BRANCH'
                assert set(b['edge_order'])==set(after[overlap[0]]['edge_order'])
            branches_rows.append(dict(s=key,before_branch_id=b['branch_id'],before_kind=b['kind'],
                before_MBG=bm[b['branch_id']]['MBG_pass'],before_ASC_m=bm[b['branch_id']]['ASC_arc_length'],
                after_branch_ids=overlap,after_kinds=[after[j]['kind'] for j in overlap],
                after_MBG=[am[j]['MBG_pass'] for j in overlap],after_ASC_m=[am[j]['ASC_arc_length'] for j in overlap]))
            if bm[b['branch_id']]['ASC_exists'] and not bm[b['branch_id']]['MBG_pass'] and any(am[j]['MBG_pass'] for j in overlap):
                recovered.append(branches_rows[-1])
        old_edges_by_kind={eid:b['kind'] for b in before for eid in b['edge_order']}
        newopen=sum(b['kind']=='OPEN_SURFACE_BRANCH' and any(old_edges_by_kind[eid]!='OPEN_SURFACE_BRANCH' for eid in b['edge_order']) for b in after)
        newmbg=sum(am[b['branch_id']]['MBG_pass'] and not any(bm[o['branch_id']]['MBG_pass'] and set(o['edge_order'])&set(b['edge_order']) for o in before) for b in after)
        inactive=np.flatnonzero(~p.topology_active_edge);nodes=p.edges[inactive,0];components=set(p.components[nodes])
        row=dict(s=key,zero_length_self_loop_edge_count=len(inactive),self_loop_affected_nodes=len(set(nodes)),
            self_loop_affected_components=len(components),before_forked_branch_count=sum(b['kind']=='FORKED_COMPONENT' for b in before),
            after_physical_forked_branch_count=sum(b['kind']=='FORKED_COMPONENT' for b in after),
            recovered_OPEN_branch_count=newopen,recovered_MBG_candidate_count=newmbg,
            max_raw_degree=int(p.raw_degree.max(initial=0)),max_canonical_degree=int(p.degree.max(initial=0)),
            max_physical_degree=int(p.physical_degree.max(initial=0)),before_branches=len(before),after_branches=len(after),
            canonical_edges=len(p.edges),physical_edges=len(emitted),components_unchanged=True,
            observed_coordinates_unchanged=True,source_face_ids_unchanged=True,source_segment_indices_unchanged=True,
            old_normal_OPEN_preserved=True)
        rows.append(row);totals.update({k:int(v) for k,v in row.items() if k in ('zero_length_self_loop_edge_count','self_loop_affected_nodes',
            'self_loop_affected_components','before_forked_branch_count','after_physical_forked_branch_count','recovered_OPEN_branch_count','recovered_MBG_candidate_count')})
        for b in after:
            metric=am[b['branch_id']]
            if b['kind']=='FORKED_COMPONENT' and metric['ASC_exists']:
                forks=[i for i,n in enumerate(b['node_order']) if p.physical_degree[n]>2]
                overreach.append(dict(s=key,classification='COMPONENT_WIDE_FORK_OVERREACH',branch_id=b['branch_id'],
                    component_id=b['component_id'],ASC_m=metric['ASC_arc_length'],fork_node_positions=forks,
                    fork_in_core=any(metric['ASC_start_arc']<=b['arc_positions'][i]<=metric['ASC_end_arc'] for i in forks),
                    component_wide_type_gate=True,verdict='AUDIT_ONLY_REAL_FORK_SEMANTICS_UNCHANGED'))
        if len(inactive):
            with (out/'affected_canonical_profiles'/f'{key}.pkl').open('wb') as f:pickle.dump(p,f,protocol=5)
        return after

    (out/'affected_canonical_profiles').mkdir(exist_ok=True)
    # 78.35 is a required neighboring control, but the frozen input has NO
    # self-loop there. Its existing OPEN candidate must not be called recovery.
    for key in sorted(set(affected)|{'78.35'},key=float):checked[key]=process(key)
    for key,probe,minimum in [('0.85',1400.,77.),('78.35',1446.,60.),('78.40',1446.,67.)]:
        assert any(b['z_range'][0]<probe<b['z_range'][1] and b['full_arc_length']>minimum
                   and branch_metrics(b,local_edge_scale(checked[key]))['MBG_pass'] for b in checked[key]),key
    save_json(out/'affected_gate.json',dict(passed=True,slices=len(affected),known_27_verified=True,seconds=time.perf_counter()-start))
    print('AFFECTED GATE PASSED',len(affected),flush=True)
    offsets={}
    with (out/'branches.pkl').open('wb') as f:
        for i,key in enumerate(keys):
            after=checked.pop(key) if key in checked else process(key)
            offsets[key]=f.tell();pickle.dump(after,f,protocol=5)
            if i%400==0:print('INVENTORY',i,flush=True)
    oldfile.close();rows.sort(key=lambda r:float(r['s']))
    save_json(out/'branch_index.json',offsets)
    csv_rows(out/'physical_topology_before_after.csv',rows)
    csv_rows(out/'self_loop_affected_branches.csv',[r for r in branches_rows if r['s'] in affected])
    csv_rows(out/'recovered_branches.csv',recovered)
    csv_rows(out/'COMPONENT_WIDE_FORK_OVERREACH.csv',overreach,fields=['s','classification','branch_id','component_id','ASC_m','fork_node_positions','fork_in_core','component_wide_type_gate','verdict'])
    save_json(out/'topology_summary.json',dict(totals,slices=len(keys),affected_slices=len(affected),known_27_slices=len(known),
        recovered_old_core_branches=len(recovered),recovered_old_core_branches_in_known27=sum(r['s'] in known for r in recovered),
        component_wide_fork_overreach=len(overreach),affected_gate_passed=True,all_geometry_and_provenance_preserved=True,
        no_component_merge=True,no_normal_open_lost=True,seconds=time.perf_counter()-start,code_sha256=code_hashes()))
    print('INVENTORY COMPLETE',dict(totals),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=output_path())
    inventory(parser.parse_args().output)
