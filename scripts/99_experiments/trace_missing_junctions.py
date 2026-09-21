"""Controlled cause attribution, never a production geometry/policy change."""
import argparse,pickle,json,time
from pathlib import Path
import numpy as np
from diagnose_missing_routes import latest,SOURCE,compact
from validate_physical_topology import load_inventory,csv_rows
from run_face_provenance_validation import manifest,raw_observations
from census_directional_consistency import exact_memoization,one_result,save_json
import main_track_assembly as assembly
from branch_absolute_core import branch_metrics,local_edge_scale,contribution_metrics,connector_gate,route_budgets
from observed_surface_graph import build_surface_graph


def probe(key,r,bs):
    by={b['branch_id']:b for b in bs};scale=local_edge_scale(bs)
    metrics={bid:branch_metrics(b,scale) for bid,b in by.items()}
    reliable=[b for b in bs if metrics[b['branch_id']]['MBG_pass']]
    records=r['path_edges'];required=r['route_z_extent'];pieces=assembly._remaining(reliable,records);rows=[]
    for direction in ('lower','upper'):
        if (required[0]<=r['candidate_z_extent'][0]+1e-6 if direction=='lower' else required[1]>=r['candidate_z_extent'][1]-1e-6):continue
        terminal=[]
        for e in (records if direction=='lower' else records[::-1]):
            if not e['source'].startswith('OBSERVED') or (terminal and e['branch_id']!=terminal[0]['branch_id']):break
            terminal.append(e)
        if direction=='upper':terminal.reverse()
        if not terminal:continue
        locked=records[len(terminal):] if direction=='lower' else records[:-len(terminal)]
        locked_z=[p[1] for e in locked for p in e['points_uz']]
        locked_range=(min(locked_z),max(locked_z)) if locked_z else (np.inf,-np.inf)
        tp=assembly._points(terminal);folds=np.flatnonzero(np.diff(tp[:,1]) < -1e-9)
        lock_index=int(folds[-1] if direction=='upper' else folds[0]) if len(folds) else None
        chosen=[d['branch_id'] for d in r['continuation_decisions'] if d['frontier']==direction]
        for piece in pieces:
            pp=assembly._points(piece);bid=piece[0]['branch_id']
            if (pp[:,1].min()>=required[0]-1e-6 if direction=='lower' else pp[:,1].max()<=required[1]+1e-6):continue
            gap=max(0,required[0]-pp[:,1].max()) if direction=='lower' else max(0,pp[:,1].min()-required[1])
            if gap>.010000001:continue
            left,right=(piece,terminal) if direction=='lower' else (terminal,piece)
            variants={}
            for protect in (True,False):
                join=assembly._nearest_join(left,right,required,locked_range,direction,protect_folds=protect)
                if join is None:variants[str(protect)]=dict(join=None,passes_other_geometric_gates=False);continue
                extension=assembly._splice(left,right,join);assembled=extension+locked if direction=='lower' else locked+extension
                used={e['branch_id'] for e in assembled if e['source'].startswith('OBSERVED')}
                contributions={i:contribution_metrics(by[i],assembled,metrics[i]) for i in used}
                budgets=route_budgets(assembled,by,metrics)
                gate=connector_gate(join['xyz_distance_m'],contributions[bid]['observed_new_length_m'])
                passes=all(c['contribution_pass'] for c in contributions.values()) and all(b['accepted'] for b in budgets) and gate['accepted']
                variants[str(protect)]=dict(join=join,contributions=contributions,budgets=budgets,connector_gate=gate,passes_other_geometric_gates=passes)
            causal=(not variants['True']['passes_other_geometric_gates']) and variants['False']['passes_other_geometric_gates']
            rows.append(dict(s=key,frontier=direction,from_branch_id=terminal[0]['branch_id'],candidate_branch_id=bid,
                actually_chosen_candidate=bool(chosen and chosen[-1]==bid),variants=variants,
                blocked_only_by_ordinary_fold_lock=causal,
                lock_edge_id=terminal[lock_index]['edge_id'] if lock_index is not None else None,
                lock_edge_points_uz=terminal[lock_index]['points_uz'] if lock_index is not None else None,
                lock_edge_drop_m=float(tp[lock_index,1]-tp[lock_index+1,1]) if lock_index is not None else None,
                scope='DIAGNOSTIC_REMOVE_ONLY_PROTECT_FOLDS; all current core/10mm/ratio/extent checks retained; no route written'))
    return rows


def controls(out):
    import directional_handoff as dh
    from joint_surface_consensus import joint_surface_summary
    m=manifest(SOURCE);allrows=[];controls=[]
    for rid,key in [('G','122.95'),('H','116.25')]:
        data=pickle.load((SOURCE/'region_data'/f'{rid}.pkl').open('rb'));reg=data['region'];ids=[m['keys'].index(k) for k in reg['target_keys']]
        keys=m['keys'][max(0,min(ids)-20):min(len(m['keys']),max(ids)+21)]
        bs=load_inventory(SOURCE,keys);g=build_surface_graph([dict(s=float(k),branches=b) for k,b in bs.items()],raw_observations(keys,m))
        exact_memoization();baseline,row=one_result(key,bs[key],g,m)
        frozen=pickle.load((SOURCE/'local_routes'/f'{rid}.pkl').open('rb'))['CURRENT_ROUTE'][key]
        def sig(r):return [(e['source'],e.get('branch_id'),e.get('edge_id'),e.get('t0'),e.get('t1'),np.asarray(e['points_xyz']).tolist()) for e in r['path_edges']]
        assert sig(baseline)==sig(frozen),(key,'control differs from previous current result')
        tests=probe(key,baseline,bs[key]);allrows.extend(tests)
        by={b['branch_id']:b for b in bs[key]};scale=local_edge_scale(bs[key]);metrics={i:branch_metrics(b,scale) for i,b in by.items()}
        for t in tests:
            if not t['actually_chosen_candidate'] or not t['blocked_only_by_ordinary_fold_lock']:continue
            a=by[t['from_branch_id']];b=by[t['candidate_branch_id']];j=t['variants']['False']['join'];zd=1 if t['frontier']=='upper' else -1
            ar=assembly._oriented(a['records']);br=assembly._oriented(b['records'])
            ai,at=(j['a_edge_index'],j['a_t']) if zd==1 else (j['b_edge_index'],j['b_t'])
            bi,bt=(j['b_edge_index'],j['b_t']) if zd==1 else (j['a_edge_index'],j['a_t'])
            apos=dh._arc_at(a,ar[ai],0)+at*(dh._arc_at(a,ar[ai],1)-dh._arc_at(a,ar[ai],0))
            bpos=dh._arc_at(b,br[bi],0)+bt*(dh._arc_at(b,br[bi],1)-dh._arc_at(b,br[bi],0))
            adir=dh._direction(a,zd);bdir=dh._direction(b,zd);an=(float(key),a['branch_id']);bn=(float(key),b['branch_id'])
            sa=joint_surface_summary(g,an,arc_interval=dh._forward_interval(a,apos,adir),metrics=metrics[a['branch_id']])
            sb=joint_surface_summary(g,bn,arc_interval=dh._forward_interval(b,bpos,bdir),metrics=metrics[b['branch_id']])
            db=dh.directional_branch_metrics(b,bpos,bdir,z_direction=zd,edge_scale=scale,support=sb)
            da=dh.directional_branch_metrics(a,apos,adir,z_direction=zd,edge_scale=scale,support=sa,alternative=dict(db,support_s_span=sb['support_s_span']))
            protected=dh._fold_protected(g,an,apos,adir,zd,metrics[a['branch_id']],sb['support_s_span'])
            t['internal_at_unlocked_intersection']=dict(a_directional=da,b_directional=db,a_support=sa,b_support=sb,fold_protected=protected,
               stronger_threshold=max(sa['support_s_span']*1.25,sa['support_s_span']+.025),
               stronger=sb['support_s_span']>max(sa['support_s_span']*1.25,sa['support_s_span']+.025))
        original=assembly._nearest_join
        def unlocked(*args,**kwargs):
            kwargs.setdefault('protect_folds',False)
            return original(*args,**kwargs)
        assembly._nearest_join=unlocked
        try:
            exact_memoization();variant,vrow=one_result(key,bs[key],g,m)
        finally:assembly._nearest_join=original
        controls.append(dict(s=key,baseline_matches_frozen=True,baseline=row,unlocked_counterfactual=vrow,
            changed_only='ordinary _nearest_join default protect_folds True->False in this isolated diagnostic process',
            source_code_changed=False,production_route_written=False))
        print('CONTROL',key,row['route_z_extent'],vrow['route_z_extent'],flush=True)
    save_json(out/'junction_controls.json',controls);save_json(out/'junction_control_details.json',allrows)


def all_missing(out):
    details=[];extent_rows=[];allkeys=manifest(SOURCE)['keys']
    from collections import Counter
    # Distinguish protected endpoint tails from genuinely uncovered absolute cores.
    # This is post-processing frozen results, not a second recognition run.
    with (SOURCE/'branches.pkl').open('rb') as stream:
        for key in allkeys:
            route_file=out/'routes'/f'{key}.pkl';wait_started=time.monotonic()
            while not route_file.exists():
                if time.monotonic()-wait_started>7200:raise TimeoutError('Census result did not arrive: '+key)
                time.sleep(5)
            bs=pickle.load(stream)
            data=pickle.load(route_file.open('rb'));r=data['result'];slice_rows=[]
            metrics={v['branch_id']:v for v in r.get('branch_audit',[])};core_ranges={}
            for b in bs:
                v=metrics.get(b['branch_id'],{})
                if not v.get('MBG_pass'):continue
                arc=np.asarray(b['arc_positions']);z=np.asarray(b['points_uz'])[:,1];lo,hi=v['ASC_start_arc'],v['ASC_end_arc']
                values=np.r_[np.interp([lo,hi],arc,z),z[(arc>lo)&(arc<hi)]]
                core_ranges[b['branch_id']]=(float(values.min()),float(values.max()))
            for missing in data['missing']:
                item=dict(missing);side=missing['side'];actual=r.get('route_z_extent')
                if actual and core_ranges and side in ('lower','upper'):
                    amounts={bid:max(0.,actual[0]-bounds[0] if side=='lower' else bounds[1]-actual[1]) for bid,bounds in core_ranges.items()}
                    item['missing_ASC_z_m']=max(amounts.values(),default=0.)
                    item['branches_with_uncovered_ASC']=[bid for bid,amount in amounts.items() if amount>1e-6]
                    item['only_endpoint_guard_extent']=item['missing_ASC_z_m']<=1e-6
                else:item.update(missing_ASC_z_m=None,branches_with_uncovered_ASC=[],only_endpoint_guard_extent=False)
                extent_rows.append(item);slice_rows.append(item)
            if any((v['missing_ASC_z_m'] or 0)>=1 for v in slice_rows):
                found=probe(key,r,bs);details.extend(found)
                (out/'junction_probes').mkdir(exist_ok=True)
                save_json(out/'junction_probes'/f'{key}.json',found)
                print('PROBED',key,'fold_lock',any(v['blocked_only_by_ordinary_fold_lock'] for v in found),flush=True)
    csv_rows(out/'missing_route_stage_inventory.csv',extent_rows)
    core=[r for r in extent_rows if (r['missing_ASC_z_m'] or 0)>1e-6]
    large=[r for r in core if r['missing_ASC_z_m']>=1.]
    keys=sorted({r['s'] for r in large},key=float)
    save_json(out/'missing_extent_summary.json',dict(total_slices=len(allkeys),
      full_branch_extent_deficit_slices=len({r['s'] for r in extent_rows}),
      uncovered_ASC_slices=len({r['s'] for r in core}),
      endpoint_guard_only_slices=len({r['s'] for r in extent_rows}-{r['s'] for r in core}),
      at_least_1m_uncovered_ASC_slices=len(keys),large_missing_keys=keys,
      core_stage_counts=dict(Counter(r['stage'] for r in core)),
      large_core_stage_counts=dict(Counter(r['stage'] for r in large)),
      scope='ASC extent outside final route Z range; does not count competing unselected branches inside existing route range as missing routes'))
    save_json(out/'all_large_missing_junction_details.json',details)
    flat=[]
    for r in details:
        a,b=r['variants']['True'],r['variants']['False']
        flat.append({k:v for k,v in r.items() if k not in ('variants','scope')}|dict(
            locked_distance_m=a['join']['xyz_distance_m'] if a['join'] else None,
            unlocked_distance_m=b['join']['xyz_distance_m'] if b['join'] else None,
            unlocked_passes_other_gates=b['passes_other_geometric_gates']))
    csv_rows(out/'all_large_missing_junction_probes.csv',flat)
    save_json(out/'junction_probe_summary.json',dict(large_missing_slices=len(keys),tested_candidate_pieces=len(details),
      slices_with_fold_lock_counterexample=len({r['s'] for r in details if r['blocked_only_by_ordinary_fold_lock']}),
      slices_whose_selected_candidate_has_fold_lock_counterexample=len({r['s'] for r in details if r['actually_chosen_candidate'] and r['blocked_only_by_ordinary_fold_lock']})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controls','all']);a=p.parse_args()
    (controls if a.action=='controls' else all_missing)(latest())
