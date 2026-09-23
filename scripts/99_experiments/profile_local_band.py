"""Exact before/after benchmark on a frozen P1 route, no full census."""
import sys,pickle,time,json,cProfile,pstats
from run_local_conflict_v2 import SOURCE,output_dir,load_inventory,mesh,manifest
from run_production_integration_v2 import assert_same_tree
from physical_face_context import PhysicalFaceContext
from region_junction_band import apply_region_junction_bands

out=output_dir();bs=load_inventory(SOURCE,['89.95']);r=pickle.load((out/'stages/P1_preband_89.95.pkl').open('rb'))['route']
reg=json.loads((out/'local_plan.json').read_text())['regions'][0];a,lo,hi,_=mesh();cache=pickle.load((out/'stages/face_geometry_cache.pkl').open('rb'))
c=PhysicalFaceContext(a,lo,hi,{float(k):v for k,v in bs.items()},[reg],geometry_cache=cache,levels=manifest(SOURCE)['levels'])
prof=cProfile.Profile();prof.enable();start=time.perf_counter()
audit=apply_region_junction_bands({'89.95':r},bs,c,[reg]);elapsed=time.perf_counter()-start;prof.disable()
mode='before' if '--reference' in sys.argv else 'after'
with (out/f'band_{mode}_profile.txt').open('w',encoding='utf-8') as f:pstats.Stats(prof,stream=f).sort_stats('cumulative').print_stats(20)
reference=out/'stages/band_equivalence_reference.pkl'
if mode=='before':pickle.dump(dict(route=r,audit=audit),reference.open('wb'),protocol=5)
else:assert_same_tree(pickle.load(reference.open('rb')),dict(route=r,audit=audit))
result=dict(seconds=elapsed,exact_reference_match=mode=='after',items=[dict(raw=x['raw_candidate_count'],legal=x['legal_candidate_count'],applied=x.get('applied')) for x in audit])
(out/f'band_{mode}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(mode,result,flush=True)
