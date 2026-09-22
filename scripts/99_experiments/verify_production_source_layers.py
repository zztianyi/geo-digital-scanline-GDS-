"""Independent, streamed verification of final spine/side source records."""
import json,pickle,time
from collections import Counter
import numpy as np
from run_production_integration_v2 import latest,SOURCE,manifest,save_json,csv_rows
from branch_absolute_core import branch_metrics,local_edge_scale,route_budgets


def run(out):
    started=time.perf_counter();m=manifest(SOURCE);rows=[];maximum=0.;count=0
    with (SOURCE/'branches.pkl').open('rb') as source:
        for index,key in enumerate(m['keys']):
            branches=pickle.load(source);file=out/'layers'/f'{key}.pkl'
            marker=out/'contexts'/f'{index//10*10:04d}.json'
            deadline=time.monotonic()+14400
            while not file.exists() or not marker.exists() or marker.stat().st_mtime<file.stat().st_mtime:
                if time.monotonic()>deadline:raise TimeoutError(key)
                time.sleep(2)
            product=pickle.load(file.open('rb'));layers=product['layers'];original={e['edge_id']:e for b in branches for e in b['records']}
            actual=[e for e in layers['MAIN_SPINE']+layers['SIDE_COMPONENTS'] if e['source'].startswith('OBSERVED')]
            t=np.asarray([[e['t0'],e['t1']] for e in actual]);source_rows=[original[e['edge_id']] for e in actual]
            for name in ['points_xyz','points_uz']:
                raw=np.asarray([e[name] for e in source_rows]);expected=raw[:,:1]+t[:,:,None]*(raw[:,1:]-raw[:,:1])
                delta=float(np.max(abs(np.asarray([e[name] for e in actual])-expected),initial=0.))
                assert delta<=2e-10,(key,name,delta);maximum=max(maximum,delta)
            for a,b in zip(actual,source_rows):
                assert a['source_face_ids']==b['source_face_ids'],(key,a['edge_id'],'FaceID')
                assert a['source_segment_indices']==b['source_segment_indices'],(key,a['edge_id'],'source segment')
            intervals={eid:[] for eid in original}
            for r in actual:intervals[r['edge_id']].append(sorted((r['t0'],r['t1'])))
            for eid,values in intervals.items():
                cursor=0.
                for a,b in sorted(values):
                    assert abs(a-cursor)<=1e-7,(key,eid,'interval gap or duplication');cursor=b
                assert abs(cursor-1.)<=1e-7,(key,eid,'source interval missing')
            scale=local_edge_scale(branches);by={b['branch_id']:b for b in branches};metrics={i:branch_metrics(b,scale) for i,b in by.items()}
            budgets=route_budgets(layers['MAIN_SPINE'],by,metrics)
            assert all(b['accepted'] for b in budgets),(key,'P2 contribution budget')
            connectors=[e for e in layers['MAIN_SPINE'] if not e['source'].startswith('OBSERVED')]
            max_connector=max((float(np.linalg.norm(np.diff(e['points_xyz'],axis=0))) for e in connectors),default=0.)
            assert max_connector<=.010+1e-12,(key,'P2 cap')
            assert all(not e['source_face_ids'] and e['face_id'] is None for e in connectors),(key,'synthetic face attribution')
            p=np.asarray([x for e in layers['MAIN_SPINE'] for x in e['points_uz']])
            max_r=max((b['combined_R_syn'] for b in budgets),default=0.)
            groups=product['red_groups_corrected']
            assert sum(len(g['group_segments']) for gs in groups.values() for g in gs)==len(product['hanging_segments'])
            rows.append(dict(s=key,source_records=len(actual),P2_max_connector_m=max_connector,P2_max_R_syn=max_r,
                P2_z_extent=[float(p[:,1].min()),float(p[:,1].max())] if len(p) else None,
                P2_roles=dict(Counter(c['role'] for c in layers['components'])),
                hanging_segments=len(product['hanging_segments']),all_source_intervals_preserved=True))
            count+=len(actual)
            if (index+1)%100==0:
                save_json(out/'source_verify_progress.json',dict(slices=index+1,records=count,seconds=time.perf_counter()-started))
                print('SOURCE_VERIFIED',index+1,count,flush=True)
    summary=dict(slices=len(rows),source_records=count,maximum_coordinate_error_m=maximum,
        all_source_face_and_segment_ids_preserved=True,all_source_intervals_conserved=True,
        all_existing_consumer_groups_nonempty_when_hanging_exists=True,
        P2_max_connector_m=max(r['P2_max_connector_m'] for r in rows),
        P2_max_R_syn=max(r['P2_max_R_syn'] for r in rows),seconds=time.perf_counter()-started,
        time_includes_waiting_for_census=True)
    save_json(out/'source_layer_verification.json',summary);csv_rows(out/'final_layer_inventory.csv',rows)
    print('ALL_SOURCE_LAYERS_VERIFIED',summary,flush=True)


if __name__=='__main__':run(latest())
