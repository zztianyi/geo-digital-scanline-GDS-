"""Re-run corrected P2 on frozen P1, asserting every P1 route stays unchanged."""
from run_production_integration_v2 import (latest,SOURCE,manifest,mesh,load_inventory,
    PhysicalFaceContext,save_json,csv_rows,code_hashes)
from surface_spine_pipeline import finish_surface_spine
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,pickle,time


def refresh_block(task):
    directory,start=task;out=Path(directory);m=manifest(SOURCE);keys=m['keys']
    targets=keys[start:start+10];context_keys=keys[max(0,start-20):start+30]
    bs=load_inventory(SOURCE,context_keys);a,lo,hi,regions=mesh()
    context=PhysicalFaceContext(a,lo,hi,{float(k):v for k,v in bs.items()},regions);rows=[]
    for key in targets:
        file=out/'routes'/f'{key}.pkl';payload=pickle.load(file.open('rb'));route=payload['result'];row=payload['row']
        digest=lambda:hashlib.sha256(pickle.dumps(route,protocol=5)).hexdigest()
        before=digest()
        result=finish_surface_spine(bs[key],route,context,float(key),p1_seconds=row['P0_P1_seconds'])
        assert digest()==before,(key,'P1 route mutated by P2')
        layers=result['layers']
        row.update(P2_recognition_seconds=result['performance']['P2_recognition_seconds'],
            seconds=sum(result['performance'].values()),P2_roles=dict(Counter(c['role'] for c in layers['components'])),
            P2_side_m=layers['side_observed_arc_m'],hanging_segments=len(result['hanging_segments']),
            source_intervals_preserved=layers['source_intervals_preserved'],P2_contact_window_fix=True,
            P1_frozen_route_sha256=before)
        product={k:v for k,v in result.items() if k!='route'}
        layer=out/'layers'/f'{key}.pkl'
        with layer.with_suffix('.pending').open('wb') as f:pickle.dump(product,f,protocol=5)
        layer.with_suffix('.pending').replace(layer)
        with file.with_suffix('.pending').open('wb') as f:pickle.dump(payload,f,protocol=5)
        file.with_suffix('.pending').replace(file)
        rows.append(row)
    old=json.loads((out/'before_P2_window_fix'/'contexts'/f'{start:04d}.json').read_text(encoding='utf-8'))
    windows={r['region_id']:r for r in old['regions']}
    windows.update({rid:d['region'] for rid,d in context.cache.items()})
    save_json(out/'contexts'/f'{start:04d}.json',dict(old,regions=list(windows.values()),P2_identity_windows_included=True))
    return rows


def run(out,workers=8):
    started=time.perf_counter();before=out/'before_P2_window_fix';rows=[];hashes=code_hashes()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(refresh_block,(str(out),start)) for start in range(0,2800,10)]
        for job in as_completed(jobs):
            rows.extend(job.result())
            save_json(out/'p2_refresh_progress.json',dict(completed=len(rows),total=2800,seconds=time.perf_counter()-started))
            if len(rows)%100==0:print('P2_REFRESH',len(rows),flush=True)
    assert code_hashes()==hashes,'Code changed during final P2 refresh'
    rows.sort(key=lambda r:float(r['slice_key']));elapsed=time.perf_counter()-started
    summary=json.loads((before/'census_summary.json').read_text(encoding='utf-8'))
    summary.update(P2_refresh_seconds=elapsed,P2_refresh_workers=workers,P1_routes_reused_unchanged=len(rows),
        code_unchanged=True,code_unchanged_scope='Final P2 refresh; frozen P1 route objects independently hashed',
        all_intervals_preserved=all(r['source_intervals_preserved'] for r in rows))
    audit=dict(slices=len(rows),P1_route_objects_unchanged=True,seconds=elapsed,workers=workers,
        code_sha256=hashes,code_unchanged=True)
    save_json(out/'P2_refresh_manifest.json',audit)
    old=json.loads((before/'manifest.json').read_text(encoding='utf-8'))
    save_json(out/'manifest.json',dict(old,code_sha256=hashes,
        P1_frozen_manifest='before_P2_window_fix/manifest.json',P2_refresh_manifest='P2_refresh_manifest.json',
        final_composition='Frozen production P1 routes plus final production P2 and hanging recognition'))
    csv_rows(out/'census_summary.csv',rows)
    save_json(out/'census_summary.json',summary)
    print('P2_REFRESH_COMPLETE',audit,flush=True)


if __name__=='__main__':run(latest())
