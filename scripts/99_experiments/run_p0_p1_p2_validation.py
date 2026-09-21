"""Revised-plan validation; P1/P2 remain shadow and P0 uses a frozen baseline."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
import diagnose_missing_routes as census
from diagnose_missing_routes import SOURCE,compact,gap_rows
from validate_physical_topology import ROOT,load_inventory,csv_rows
from run_face_provenance_validation import manifest,raw_observations
from census_directional_consistency import one_result,exact_memoization,save_json,code_hashes
from observed_surface_graph import build_surface_graph
from branch_absolute_core import branch_metrics,local_edge_scale,contribution_metrics,connector_gate,route_budgets
import main_track_assembly as assembly

BASELINE=ROOT/'outputs/face_branch_consistency/20260921_094107'


def latest():
    return Path((ROOT/'outputs/validation_p0_p1_p2/LATEST.txt').read_text(encoding='utf-8').strip())


def run_regressions(out):
    m=manifest(SOURCE);rows=[];(out/'hard_regressions').mkdir(exist_ok=True)
    for key in ['98.05','98.10','122.95','116.25']:
        i=m['keys'].index(key);start=i//10*10;keys=m['keys'][max(0,start-20):min(len(m['keys']),start+30)]
        bs=load_inventory(SOURCE,keys)
        g=build_surface_graph([dict(s=float(k),branches=b) for k,b in bs.items()],raw_observations(keys,m))
        exact_memoization();r,row=one_result(key,bs[key],g,m)
        before=pickle.load((BASELINE/'routes'/f'{key}.pkl').open('rb'))
        decisions=r['continuation_decisions']
        audit=dict(s=key,before=before['row'],after=row,decisions=decisions,
                   static_ASC_checks=[{k:j.get(k) for k in ['from_branch_id','to_branch_id','terminal_ASC_before_m','terminal_ASC_after_m','identity_rank','identity_ambiguous']} for j in r['junctions']],
                   ordinary_continuation_strict_static_ASC=True,
                   internal_handoff_policy='unchanged directional-tail policy')
        if key in ['98.05','98.10']:
            failed=next((n for n,d in enumerate(decisions) if d['branch_id']==2 and not d['feasibility_accepted']),None)
            audit['B2_failed_then_alternative_accepted']=failed is not None and any(d['branch_id'] in (0,6) and d['feasibility_accepted'] for d in decisions[failed+1:])
            assert audit['B2_failed_then_alternative_accepted'],audit
        assert row['max_connector_m']<=.010000000001 and row['max_combined_R']<1
        pickle.dump(dict(result=compact(r),row=row,missing=gap_rows(key,r)),(out/'hard_regressions'/f'{key}.pkl').open('wb'),protocol=5)
        rows.append(audit);print('HARD_REGRESSION',key,row['sequence'],row['route_z_extent'],flush=True)
    save_json(out/'hard_regressions.json',rows)


def run_census(out,workers):
    (out/'routes').mkdir(exist_ok=True)
    save_json(out/'manifest.json',dict(source=str(SOURCE),baseline=str(BASELINE),code_sha256=code_hashes(),
        census_context='Same fixed 10-target blocks and 20-slice/1m halos as previous census.',
        ordinary_continuation_strict_static_ASC=True,internal_handoff_policy='unchanged directional-tail policy',
        FaceTrack_used_in_recognition=False,P1_P2_shadow_only=True))
    # The four controls used exactly these blocks; keep their atomic results.
    import shutil
    for p in (out/'hard_regressions').glob('*.pkl'):shutil.copy2(p,out/'routes'/p.name)
    census.run(out,workers)


def legal_options(key,r,branches):
    """Independent final-state enumeration, including all unselected candidates.

    Identity ambiguity is not used to hide geometrically legal continuations.
    The production ordering is not called by this audit.
    """
    scale=local_edge_scale(branches);by={b['branch_id']:b for b in branches}
    metrics={i:branch_metrics(b,scale) for i,b in by.items()}
    reliable=[b for b in branches if metrics[b['branch_id']]['MBG_pass']]
    records=r['path_edges'];required=r.get('route_z_extent');rows=[]
    if not records:return rows
    pieces=assembly._remaining(reliable,records)
    for direction in ['lower','upper']:
        terminal=[]
        for e in (records if direction=='lower' else records[::-1]):
            if not e['source'].startswith('OBSERVED') or (terminal and e['branch_id']!=terminal[0]['branch_id']):break
            terminal.append(e)
        if direction=='upper':terminal.reverse()
        if not terminal:continue
        locked=records[len(terminal):] if direction=='lower' else records[:-len(terminal)]
        z=[p[1] for e in locked for p in e['points_uz']];locked_range=(min(z),max(z)) if z else (np.inf,-np.inf)
        terminal_bid=terminal[0]['branch_id'];bounds=assembly._terminal_core_bounds(terminal,by[terminal_bid],metrics[terminal_bid])
        core_before=contribution_metrics(by[terminal_bid],records,metrics[terminal_bid])['retained_ASC_arc_length']
        for index,piece in enumerate(pieces):
            p=assembly._points(piece);lo,hi=float(p[:,1].min()),float(p[:,1].max());bid=piece[0]['branch_id']
            if (lo>=required[0]-1e-6 if direction=='lower' else hi<=required[1]+1e-6):continue
            gap=max(0.,required[0]-hi) if direction=='lower' else max(0.,lo-required[1])
            if gap>.010000000001:continue
            target=(lo,required[0]) if direction=='lower' else (required[1],hi)
            if not contribution_metrics(by[bid],piece,metrics[bid])['contribution_pass']:continue
            if branch_metrics(by[bid],scale,target_z=target)['target_region_in_ASC_fraction']<=0:continue
            left,right=(piece,terminal) if direction=='lower' else (terminal,piece)
            j=assembly._nearest_join(left,right,required,locked_range,direction,terminal_core_bounds=bounds)
            passed=False;distance=None;reason='NO_VALID_JUNCTION'
            if j:
                distance=j['xyz_distance_m'];extended=assembly._splice(left,right,j)
                sep=next(n for n,e in enumerate(extended) if e['source']=='TOPOLOGY_SWITCH')
                c=contribution_metrics(by[bid],extended[:sep] if direction=='lower' else extended[sep+1:],metrics[bid])
                gate=connector_gate(distance,c['observed_new_length_m']);new=extended+locked if direction=='lower' else locked+extended
                used={e['branch_id'] for e in new if e['source'].startswith('OBSERVED')}
                core_after=contribution_metrics(by[terminal_bid],new,metrics[terminal_bid])['retained_ASC_arc_length']
                passed=c['contribution_pass'] and gate['accepted'] and core_after+1e-7>=core_before and all(contribution_metrics(by[i],new,metrics[i])['contribution_pass'] for i in used) and all(b['accepted'] for b in route_budgets(new,by,metrics))
                reason='LEGAL_CONTINUATION_EXISTS_BUT_ROUTE_STOPPED' if passed else gate['reason'] if not gate['accepted'] else 'CORE_OR_RATIO_GATE'
            rows.append(dict(s=key,direction=direction,candidate=bid,piece_index=index,xyz_distance_m=distance,legal=bool(passed),reason=reason))
    return rows


def audit(out):
    """Stream each completed result once while the census is still running."""
    m=manifest(SOURCE);inventory=[];probes=[];layer_rows=[]
    (out/'layers').mkdir(exist_ok=True)
    with (SOURCE/'branches.pkl').open('rb') as source:
        for key in m['keys']:
            branches=pickle.load(source);path=out/'routes'/f'{key}.pkl';start=time.monotonic()
            while not path.exists():
                if time.monotonic()-start>10800:raise TimeoutError(key)
                time.sleep(5)
            d=pickle.load(path.open('rb'));r=d['result'];ext=r.get('route_z_extent');deficits=[]
            metrics={b['branch_id']:b for b in r['branch_audit']}
            for b in branches:
                v=metrics.get(b['branch_id'],{})
                if not v.get('MBG_pass'):continue
                arc=np.asarray(b['arc_positions']);z=np.asarray(b['points_uz'])[:,1];lo,hi=v['ASC_start_arc'],v['ASC_end_arc']
                zz=np.r_[np.interp([lo,hi],arc,z),z[(arc>lo)&(arc<hi)]]
                for side,amount in [('lower',ext[0]-zz.min()),('upper',zz.max()-ext[1])]:
                    if amount>1e-6:deficits.append(dict(branch_id=b['branch_id'],side=side,missing_ASC_z_m=float(amount)))
            inventory.append(dict(s=key,ASC_missing=bool(deficits),ASC_missing_max_m=max((x['missing_ASC_z_m'] for x in deficits),default=0),deficits=deficits))
            if not r.get('extent_covered',True):probes.extend(legal_options(key,r,branches))
            # All V have three lossless layers. P2 near-return proposals are
            # separate parameterized shadows; unselected reliable pieces remain
            # reviewable and are never asserted to be geological noise.
            closed=[b for b in branches if b['kind']=='CLOSED_COMPONENT'];opened=[b for b in branches if b['kind']!='CLOSED_COMPONENT']
            side=[dict(e,layer_reason='PHYSICAL_CLOSED_COMPONENT') for b in closed for e in b['records']]
            rejected=[dict(e,layer_reason='NOT_SELECTED_REVIEW_REQUIRED') for piece in assembly._remaining(opened,r['path_edges']) for e in piece]
            layers=dict(s=key,MAIN_SPINE=r['path_edges'],SIDE_COMPONENT=side,REJECTED_REDUNDANT=rejected,
                scope='P0 static route + physical closed components; remaining pieces are retained, not proven redundant. P2 peeling is a separate parameter scan.')
            # Check source interval conservation independently of layer names.
            lengths={e['edge_id']:0. for b in branches for e in b['records']}
            for records in [layers['MAIN_SPINE'],side,rejected]:
                for e in records:
                    if e['source'].startswith('OBSERVED'):lengths[e['edge_id']]+=abs(e['t1']-e['t0'])
            assert all(abs(v-1)<1e-7 for v in lengths.values()),(key,'source interval loss/duplication')
            pickle.dump(layers,(out/'layers'/f'{key}.pkl').open('wb'),protocol=5)
            layer_rows.append(dict(s=key,main_edges=len(r['path_edges']),side_edges=len(side),unselected_edges=len(rejected),source_intervals_preserved=True))
    csv_rows(out/'ASC_missing_inventory.csv',inventory);csv_rows(out/'stopped_continuation_audit.csv',probes);csv_rows(out/'layer_inventory.csv',layer_rows)
    save_json(out/'P0_summary.json',dict(count=len(inventory),ASC_missing_count=sum(v['ASC_missing'] for v in inventory),
        ASC_missing_ge_1m_count=sum(v['ASC_missing_max_m']>=1 for v in inventory),
        LEGAL_CONTINUATION_EXISTS_BUT_ROUTE_STOPPED=len({v['s'] for v in probes if v['legal']}),
        legal_stopped_events=sum(v['legal'] for v in probes),candidate_pieces_audited=len(probes),
        all_three_layer_source_intervals_preserved=True))
    print('P0_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['regressions','census','audit']);p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();out=latest()
    if a.action=='regressions':run_regressions(out)
    elif a.action=='census':run_census(out,a.workers)
    else:audit(out)
