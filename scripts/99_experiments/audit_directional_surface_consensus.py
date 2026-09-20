"""Frozen raw V/H inputs for the directional consensus review (no mesh changes)."""
from pathlib import Path
import sys, pickle, time, json, hashlib, unittest, argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/99_experiments'))
from locc_data import FrozenProfiles,opc
from validate_observed_first_locc import target_canonical_profile
from dominant_observed_branch import extract_observed_branches
from horizontal_surface_link import horizontal_mesh_observations
from observed_surface_graph import build_surface_graph
OUT=ROOT/'outputs/directional_surface_consensus/20260920'

def prepare():
    start=time.perf_counter()
    frozen=FrozenProfiles(ROOT/'outputs/orthogonal_scanline_constraint/20260919_081148')
    cases={}
    for center in (89.75,104.30,122.25):
        positions=np.round(center+np.arange(-20,21)*.05,2)
        profiles=[]
        for s in positions:
            p=target_canonical_profile(frozen,f'{s:.2f}')
            uz=opc.to_suz(p.nodes,frozen.arc)[:,1:]
            profiles.append(dict(s=float(s),branches=extract_observed_branches(p,uz)))
        cases[f'{center:.2f}']=dict(profiles=profiles)
    positions=sorted({p['s'] for c in cases.values() for p in c['profiles']})
    print('Canonical V ready',time.perf_counter()-start,flush=True)
    observations,inventory=horizontal_mesh_observations(frozen.horizontal,frozen.levels,positions,frozen.arc)
    print('Raw H ready',len(observations),time.perf_counter()-start,flush=True)
    for key,c in cases.items():
        ss={p['s'] for p in c['profiles']}
        c['graph']=build_surface_graph(c['profiles'],[h for h in observations if h['s'] in ss])
        print('Graph ready',key,time.perf_counter()-start,flush=True)
    with (OUT/'raw_cases.pkl').open('wb') as f:pickle.dump(dict(cases=cases,arc=frozen.arc),f,protocol=5)
    (OUT/'input_prepare.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,H_observations=len(observations),slices=len(positions),levels=len(inventory)),indent=2))

def save_json(name,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert),encoding='utf-8')


def verify_geometry(branches,result):
    """Check original canonical edge interpolation and both-connector budgets."""
    originals={r['edge_id']:r for b in branches for r in b['records']}
    intervals={}
    for r in result['path_edges']:
        if r['source'].startswith('OBSERVED'):
            raw=originals[r['edge_id']]
            for name in ('points_uz','points_xyz'):
                a,b=np.asarray(raw[name])
                np.testing.assert_allclose(r[name],[a+t*(b-a) for t in (r['t0'],r['t1'])],atol=2e-10,rtol=0)
            assert r['source_face_ids']==raw['source_face_ids']
            intervals.setdefault(r['edge_id'],[]).append(sorted((r['t0'],r['t1'])))
        else:
            assert r['face_id'] is None and not r['source_face_ids']
    for value in intervals.values():
        value.sort()
        assert all(a[1]<=b[0]+1e-9 for a,b in zip(value,value[1:]))
    for a,b in zip(result['path_edges'],result['path_edges'][1:]):
        for name in ('points_uz','points_xyz'):
            np.testing.assert_allclose(a[name][1],b[name][0],atol=2e-8,rtol=0)
    assert all(j['xyz_distance_m']<=.010+1e-12 and j['R_syn']<1 for j in result['junctions'])
    assert all(b['accepted'] and b['combined_R_syn']<1 for b in result['route_contribution_budgets'])


def test_real_104_30_B1_competition(branches,result):
    assert result['route_branch_sequence']==[1,2,4],result['route_branch_sequence']
    assert result['internal_handoff_count']>=1
    join=next(j for j in result['junctions'] if j.get('handoff_kind')=='INTERNAL_HANDOFF')
    assert {join['from_branch_id'],join['to_branch_id']}=={1,2}
    assert join['internal_tail_trimmed_length']>10.
    assert join['xyz_distance_m']<=.01
    assert result['future_switch_avoided_count']>=1
    verify_geometry(branches,result)


def validate():
    sys.path.insert(0,str(ROOT/'tests'))
    patterns=['test_directional_consensus.py','test_absolute_sweet_zone.py','test_main_track_assembly.py',
        'test_surface_track.py','test_dominant_observed_branch.py','test_dominant_review_fixes.py',
        'test_dominant_boundary_association.py','test_dominant_closed_components.py','test_dominant_recognition_adapter.py']
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p) for p in patterns)
    with (OUT/'targeted_tests.log').open('w',encoding='utf-8') as stream:
        tested=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    assert tested.wasSuccessful(),'See targeted_tests.log'
    from dominant_observed_branch import solve_dominant_branch,route_result
    from branch_absolute_core import branch_metrics,local_edge_scale,directional_branch_metrics
    from directional_handoff import future_switch_cost,_direction
    from joint_surface_consensus import arc_support_profile,joint_surface_summary
    data=pickle.load((OUT/'raw_cases.pkl').open('rb'));results={};audit=[];rows=[]
    for key,c in data['cases'].items():
        g=c['graph'];s=float(key);bs=[b for n,b in g['nodes'].items() if n[0]==s]
        started=time.perf_counter()
        r=solve_dominant_branch(None,None,branches=bs,surface_graph=g,target_s=s,
            input_z_range=data['arc']['z_range'],source_data_required=key=='89.75')
        seconds=time.perf_counter()-started
        verify_geometry(bs,r)
        if key=='104.30':test_real_104_30_B1_competition(bs,r)
        if key=='89.75':
            assert 4 not in r['route_branch_sequence']
            assert r['status']=='SOURCE_DATA_REQUIRED'
        rows.append(dict(slice_key=key,branch_sequence=r['route_branch_sequence'],status=r['status'],seconds=seconds,
            **{k:r[k] for k in ('internal_handoff_count','internal_tail_trimmed_length','directional_tail_count',
               'folded_branch_handoff_evaluated_count','folded_branch_handoff_accepted_count','future_switch_avoided_count')},
            switches=r['branch_switch_count'],ABA=sum(a==c for a,b,c in zip(r['route_branch_sequence'],r['route_branch_sequence'][1:],r['route_branch_sequence'][2:])),
            max_connector_m=max((j['xyz_distance_m'] for j in r['junctions']),default=0.),
            max_combined_R_syn=max((j['combined_R_syn'] for j in r['route_contribution_budgets']),default=0.),
            unresolved_reasons=r['unresolved_reasons']))
        results[key]=dict(result=r,branches=bs)
        if key=='104.30':
            scale=local_edge_scale(bs)
            for b in bs:
                if b['branch_id'] not in (1,2,3,4,6):continue
                node=(s,b['branch_id']);m=branch_metrics(b,scale);d=_direction(b,-1)
                pos=0. if d==1 else b['arc_positions'][-1]
                support=joint_surface_summary(g,node,metrics=m)
                drs=directional_branch_metrics(b,pos,d,z_direction=-1,edge_scale=scale,support=support)
                audit.append(dict(m,**support,**drs,**future_switch_cost(bs,b['branch_id'],z_direction=-1,target_z=1358.163297)))
            save_json('real104_arc_support.json',{str(bid):arc_support_profile(g,(s,bid)) for bid in (1,2,3,4,6)})
            save_json('real104_branch_audit.json',audit)
        print(key,rows[-1],flush=True)
    from test_directional_consensus import tail_scene
    from test_absolute_sweet_zone import line,supported_graph
    from test_dominant_observed_branch import fixture
    from main_track_assembly import assemble_main_track
    for name,strong in (('reliable_fold',True),('directional_tail',False)):
        p,g=tail_scene(strong);bs=p[0.][2]
        r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
        verify_geometry(bs,r);results[name]=dict(result=r,branches=bs)
    bs=fixture([line(0,0,4),line(.001,3.5,6),line(.006,3.5,10)])[2];g=supported_graph(bs)
    r=assemble_main_track(bs,route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
    verify_geometry(bs,r);results['long_continuation']=dict(result=r,branches=bs)
    first=assemble_main_track(bs[:2],route_result(bs[0]['records']),anchor_branch_id=0,graph=g,target_s=0.)
    forced=assemble_main_track(bs,first,anchor_branch_id=0,graph=g,target_s=0.)
    verify_geometry(bs,forced);results['short_counterfactual']=dict(result=forced,branches=bs)
    with (OUT/'validated_results.pkl').open('wb') as f:pickle.dump(results,f,protocol=5)
    save_json('summary.json',dict(targeted_tests=tested.testsRun,targeted_passed=tested.wasSuccessful(),
        real_104_30_regression=True,real_cases=rows,controlled_case_count=4,
        production_entry_integrated=False,accuracy_measured=False,
        benchmark_scope='3 frozen slices with raw V/H observations within ±1 m; excludes input extraction and reports'))
    sources=list((ROOT/'scripts/04_structure_recognition').glob('*.py'))+[Path(__file__).resolve()]
    save_json('manifest.json',dict(base='0a8e4bb',plan='GDS_Directional_Tail_Joint_Surface_Consensus_Codex_Plan.md',
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        raw_cases_sha256=hashlib.sha256((OUT/'raw_cases.pkl').read_bytes()).hexdigest()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','validate'),default='validate',nargs='?')
    args=parser.parse_args()
    prepare() if args.action=='prepare' else validate()
