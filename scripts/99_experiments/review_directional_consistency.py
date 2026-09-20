"""Post-hoc comparison of FINAL neighboring main tracks, not candidate support.

H-run IDs are only compared within the same elevation level. A correspondence
requires one actual V intersection and one H hit on every intervening slice.
Synthetic connector crossings never count as observed support.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse, json, pickle, time
import numpy as np
from scipy.ndimage import label
from census_directional_consistency import OUT, ROOT, save_json, load_branches, code_hashes
from horizontal_surface_link import branch_crossings,associate_crossing

NAMES = dict(SURFACE_CHOICE='可比曲面选择不同', DETAIL_COUNT='可比横向交点保留数量不同',
             COVERAGE_GAP='可比曲面一侧未保留', MBG_ASYMMETRY='最小分支门槛两侧不同',
             GEOMETRY_COMPLEXITY='横向基本相容但纵向复杂度变化')


def unique_table(crossings):
    return {(int(r[0]),int(r[1])):r for r in crossings if r[3]==1}


def audit_excluded_topology(key, crossings, branches, result, manifest):
    """Independently expose raw matches hidden by the solver's OPEN-only filter.

    Non-OPEN branches remain MBG=false and cannot become accepted main tracks.
    Their measured correspondence is useful for explaining asymmetric gaps.
    Coincident excluded geometry makes an audit correspondence ambiguous.
    """
    out=crossings.copy();levels=np.asarray(manifest['levels']);z=levels[out[:,0].astype(int)]
    arc=manifest['arc'];angle=arc['angle_min']+float(key)/arc['radius'];counts=Counter();exclusions=[]
    by_component=defaultdict(list)
    for b in branches:by_component[b['component_id']].append(b)
    metrics={b['branch_id']:b for b in result.get('branch_audit',())}
    for component,bs in by_component.items():
        if bs[0]['kind']!='FORKED_COMPONENT':continue
        full=Counter();nonself=Counter();loops=[]
        for b in bs:
            for e in b['records']:
                a,c=e['node_ids'];full[a]+=1;full[c]+=1
                if a==c:loops.append(dict(node=a,edge=e['edge_id'],faces=e['source_face_ids']))
                else:nonself[a]+=1;nonself[c]+=1
        only_self=bool(loops) and max(nonself.values(),default=0)<=2
        for b in bs:
            mm=metrics.get(b['branch_id'],{})
            if mm.get('ASC_exists'):
                exclusions.append(dict(slice_key=key,branch_id=b['branch_id'],component_id=component,
                    kind=b['kind'],nodes=b['canonical_node_count'],ASC_arc_length=mm['ASC_arc_length'],
                    z_range=b['z_range'],MBG_pass=mm['MBG_pass'],
                    max_degree=max(full.values(),default=0),max_degree_without_self_edges=max(nonself.values(),default=0),
                    self_edges=loops,branching_caused_only_by_self_edges=only_self))
    for b in branches:
        if b['kind']=='OPEN_SURFACE_BRANCH':continue
        indices=np.flatnonzero((z>=b['z_range'][0]-1e-9)&(z<=b['z_range'][1]+1e-9));cache={}
        for i in indices:
            li=int(out[i,0]);u=out[i,2]
            if li not in cache:cache[li]=branch_crossings(b,z[i])
            hit=dict(u=u,z=z[i],point_xyz=[arc['center'][0]+(u+arc['radius'])*np.cos(angle),
                         arc['center'][1]+(u+arc['radius'])*np.sin(angle),z[i]])
            found={round(c['arc_position'],8):c for c in associate_crossing(hit,b,crossings=cache[li])}
            if not found:continue
            counts[b['kind']]+=len(found)
            previous=int(out[i,10]);out[i,10]+=len(found)
            if previous==0 and len(found)==1:
                cross=next(iter(found.values()))
                out[i,4:9]=[b['branch_id'],int(b['edge_order'][cross['edge_index']]),cross['t'],cross['arc_position'],0]
            else:out[i,4:9]=[-1,-1,float('nan'),float('nan'),0]
            out[i,3]=int(out[i,10]==1 and out[i,11]==1)
    return out,counts,exclusions


def normalize_vertex_selection(key, crossings, result):
    """At a shared vertex either retained incident edge proves its retention.

    The core matching cache chooses one incident edge; this audit must not
    mistake that implementation detail for a missing final-route endpoint.
    Preserve raw route files, write a separate normalized array only if needed.
    """
    at_vertex=(crossings[:,3]==1)&((crossings[:,6]<=1e-9)|(crossings[:,6]>=1-1e-9))
    if not at_vertex.any():return crossings,0,0
    branches=load_branches([key])[key]
    edge_arcs={int(eid):(b['branch_id'],b['arc_positions'][i],b['arc_positions'][i+1])
               for b in branches for i,eid in enumerate(b['edge_order'])}
    intervals=defaultdict(list)
    for e in result['path_edges']:
        if e['source'].startswith('OBSERVED'):
            bid,a,b=edge_arcs[e['edge_id']]
            intervals[bid].append(sorted((a+e['t0']*(b-a),a+e['t1']*(b-a))))
    out=crossings.copy();changed=0
    for i in np.flatnonzero(at_vertex):
        selected=any(lo-1e-9<=out[i,7]<=hi+1e-9 for lo,hi in intervals[int(out[i,4])])
        changed+=int(bool(out[i,9])!=selected);out[i,9]=selected
    if changed:
        directory=OUT/'normalized_crossings';directory.mkdir(exist_ok=True)
        np.save(directory/f'{key}.npy',out)
    return out,int(at_vertex.sum()),changed


def compare_pair(left, right, continuous=None):
    """Symmetric, one count per H-run pair; no global/local branch-ID equality."""
    common = left.keys() & right.keys()
    if continuous is not None: common &= continuous
    eligible = {k for k in common if left[k][8] and right[k][8]}
    selected_l = defaultdict(set); selected_r = defaultdict(set)
    for k in eligible:
        r=left[k]
        if r[9]: selected_l[k[0]].add(k[1])
    for k in eligible:
        r=right[k]
        if r[9]: selected_r[k[0]].add(k[1])
    compared = {k for k in eligible if left[k][9] or right[k][9]}
    mismatches = {k for k in compared if bool(left[k][9]) != bool(right[k][9])}
    events = []
    for k in sorted(mismatches):
        a,b=selected_l[k[0]],selected_r[k[0]]
        category = ('COVERAGE_GAP' if not a or not b else
                    'DETAIL_COUNT' if a & b and len(a)!=len(b) else 'SURFACE_CHOICE')
        events.append(dict(level=k[0], run=k[1], category=category,
            left_u=float(left[k][2]), right_u=float(right[k][2]),
            left_selected=bool(left[k][9]), right_selected=bool(right[k][9]),
            left_bid=int(left[k][4]), right_bid=int(right[k][4]),
            left_count=len(a),right_count=len(b)))
    gate = {k for k in common if bool(left[k][8]) != bool(right[k][8])
            and bool(left[k][9]) != bool(right[k][9]) and (left[k][9] or right[k][9])}
    for k in sorted(gate):
        events.append(dict(level=k[0],run=k[1],category='MBG_ASYMMETRY',
            left_u=float(left[k][2]),right_u=float(right[k][2]),
            left_selected=bool(left[k][9]),right_selected=bool(right[k][9]),
            left_bid=int(left[k][4]),right_bid=int(right[k][4]),
            left_count=len(selected_l[k[0]]),right_count=len(selected_r[k[0]])))
    return dict(common_unique=len(common),eligible=len(eligible),compared=len(compared),
                agreement=len(compared)-len(mismatches),mismatch=len(mismatches),
                gate_asymmetry=len(gate)), events


def metric_checks():
    def r(level,pid,bid,selected,eligible=1,u=0):
        return np.array([level,pid,u,1,bid,17,.5,1.,eligible,selected,1,1.])
    # Different branch IDs representing the same raw H run are agreement.
    a={(1,2):r(1,2,99,1)};b={(1,2):r(1,2,1,1)}
    assert compare_pair(a,b)[0]['mismatch']==0
    # Same local branch number on different H runs is not correspondence.
    a={(1,2):r(1,2,0,1),(1,3):r(1,3,7,0)}
    b={(1,2):r(1,2,0,0),(1,3):r(1,3,7,1)}
    stats,events=compare_pair(a,b)
    assert stats['mismatch']==2 and all(e['category']=='SURFACE_CHOICE' for e in events)
    b[(1,2)][9]=1
    assert compare_pair(a,b)[1][0]['category']=='DETAIL_COUNT'
    b[(1,2)][9]=0;b[(1,3)][9]=0
    assert compare_pair(a,b)[1][0]['category']=='COVERAGE_GAP'
    b[(1,2)][8]=0
    assert compare_pair(a,b)[0]['mismatch']==0 and compare_pair(a,b)[0]['gate_asymmetry']==1
    assert compare_pair(a,b,set())[0]['common_unique']==0
    # Ambiguous and unmatched V intersections are excluded, never guessed.
    c=np.array([r(1,2,0,1),r(2,2,0,1)]);c[1,3]=0
    assert len(unique_table(c))==1
    # A non-comparable endpoint-only run must not change the classification.
    a={(1,2):r(1,2,0,1)};b={(1,2):r(1,2,0,0),(1,99):r(1,99,7,1)}
    assert compare_pair(a,b)[1][0]['category']=='COVERAGE_GAP'
    return dict(checks=8, passed=True)


def route_flags(key, result, levels):
    """Local diagnostics also catch defects shared by ALL adjacent selections."""
    flags=[];curve=np.asarray(result['curve_uz'])
    for j,join in enumerate(result.get('junctions',())):
        for category,condition in (
            ('NEAR_CONNECTOR_CAP',join['xyz_distance_m']>=.008),
            ('SHARP_JUNCTION',(join.get('tangent_turn_deg') or 0)>60)):
            if condition:
                p=np.mean([join['a_point_uz'],join['b_point_uz']],axis=0)
                flags.append(dict(slice_key=key,category=category,u=float(p[0]),z=float(p[1]),
                    junction=j,distance_m=join['xyz_distance_m'],angle_deg=join.get('tangent_turn_deg')))
    actual=result.get('route_z_extent');candidate=result.get('candidate_z_extent')
    if actual and candidate and len(curve):
        for side in (0,1):
            if abs(actual[side]-candidate[side])>1e-6:
                p=curve[np.argmin(abs(curve[:,1]-actual[side]))]
                flags.append(dict(slice_key=key,category='UNCOVERED_RELIABLE_EXTENT',u=float(p[0]),z=float(p[1]),
                    side='lower' if side==0 else 'upper',missing_z_m=abs(actual[side]-candidate[side]),
                    candidate_endpoint_z=candidate[side],reasons=result.get('unresolved_reasons',[])))
    if result['selection'].get('comparison_ambiguous'):
        flags.append(dict(slice_key=key,category='LOCAL_ANCHOR_TIE',
            stage=result['selection']['decision_stage']))
    for d in result.get('continuation_decisions',()):
        if d.get('comparison_ambiguous'):
            flags.append(dict(slice_key=key,category='LOCAL_CONTINUATION_TIE',**d))
    outside=0.;hlo,hhi=min(levels),max(levels)
    for e in result['path_edges']:
        if not e['source'].startswith('OBSERVED'):continue
        p=np.asarray(e['points_uz']);zlo,zhi=sorted(p[:,1]);length=float(np.linalg.norm(p[1]-p[0]))
        inside=max(0.,min(zhi,hhi)-max(zlo,hlo))
        fraction=inside/(zhi-zlo) if zhi-zlo>1e-12 else float(hlo<=zlo<=hhi)
        outside+=length*(1-fraction)
    return flags,outside


def audit(follow=False):
    start=time.perf_counter();waiting_seconds=0.
    m=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'));keys=m['keys']
    assert code_hashes()==m['code_sha256']
    tables=[];raw_counts=Counter(); route_rows=[];intrinsic=[];outside_lengths={};vertex_checks=vertex_fixes=0
    topology_counts=Counter();topology_exclusions=[]
    branch_stream=(OUT/'branches.pkl').open('rb')
    for i,key in enumerate(keys):
        path=OUT/'routes'/f'{key}.pkl'
        if follow and not path.exists():
            waited=time.perf_counter();print('AWAIT',key,flush=True)
            while not path.exists():
                if time.perf_counter()-waited>7200:raise TimeoutError('Missing atomic route '+key)
                time.sleep(2)
            waiting_seconds+=time.perf_counter()-waited
        with path.open('rb') as f:d=pickle.load(f)
        bs=pickle.load(branch_stream)
        a,tc,te=audit_excluded_topology(key,d['crossings'],bs,d['result'],m)
        topology_counts.update(tc);topology_exclusions.extend(te)
        a,nv,nf=normalize_vertex_selection(key,a,d['result']);vertex_checks+=nv;vertex_fixes+=nf
        if tc or nf:
            directory=OUT/'normalized_crossings';directory.mkdir(exist_ok=True);np.save(directory/f'{key}.npy',a)
        tables.append(unique_table(a));route_rows.append(d['row'])
        flags,outside=route_flags(key,d['result'],m['levels']);intrinsic.extend(flags);outside_lengths[key]=outside
        raw_counts.update(dict(observations=len(a),unique=int((a[:,3]==1).sum()),
            no_V_match=int((a[:,10]==0).sum()),multiple_V_matches=int((a[:,10]>1).sum()),
            multiple_H_hits=int((a[:,11]>1).sum()),selected_unique=int(((a[:,3]==1)&(a[:,9]==1)).sum())))
        if i%400==0:print('LOAD',i,flush=True)
    branch_stream.close()
    details=[];scale_summaries=[];regions=[]
    for distance in (1,5,10,20):
        totals=Counter();pair_rows=[]
        mask=np.zeros((len(keys)-distance,len(m['levels'])),dtype=np.uint8);gate_mask=np.zeros_like(mask)
        for i in range(len(keys)-distance):
            # Same H run cannot jump an ambiguous/missing intermediate crossing.
            common=set(tables[i])
            for j in range(i+1,i+distance+1):common.intersection_update(tables[j])
            stats,events=compare_pair(tables[i],tables[i+distance],common)
            totals.update(stats)
            pair_rows.append(dict(left=keys[i],right=keys[i+distance],**stats))
            if distance==1:
                grouped=defaultdict(list)
                for e in events:grouped[e['category']].append(e)
                for category,items in grouped.items():
                    levels=sorted({e['level'] for e in items}); runs=[];current=[]
                    for li in levels:
                        if current and li!=current[-1]+1:runs.append(current);current=[]
                        current.append(li)
                    if current:runs.append(current)
                    for run in runs:
                        subset=[e for e in items if e['level'] in run]
                        persistent=len(run)>=3
                        details.append(dict(left=keys[i],right=keys[i+1],pair_index=i,category=category,
                            start_level=run[0],end_level=run[-1],levels=len(run),persistent=persistent,
                            z_min=m['levels'][run[0]],z_max=m['levels'][run[-1]],
                            u_min=min(min(e['left_u'],e['right_u']) for e in subset),
                            u_max=max(max(e['left_u'],e['right_u']) for e in subset),
                            event_count=len(subset),examples=subset[::max(1,len(subset)//6)][:7]))
                        if persistent and category!='MBG_ASYMMETRY':mask[i,run]=1
                        if persistent and category=='MBG_ASYMMETRY':gate_mask[i,run]=1
        save_json(OUT/f'pair_metrics_{distance:02d}.json',pair_rows)
        scale_summaries.append(dict(distance_m=.05*distance,pairs=len(pair_rows),**totals,
            affected_pairs=sum(r['mismatch']>0 for r in pair_rows)))
        if distance==1:
            labels,count=label(mask,np.ones((3,3)))
            np.savez_compressed(OUT/'review_mask.npz',mask=mask,gate_mask=gate_mask,labels=labels)
            for region in range(1,count+1):
                ii,jj=np.where(labels==region)
                regions.append(dict(region_id=region,cells=len(ii),
                    s_min=float(keys[int(ii.min())]),s_max=float(keys[int(ii.max())+1]),
                    z_min=m['levels'][int(jj.min())],z_max=m['levels'][int(jj.max())],
                    pair_count=len(set(ii)),level_count=len(set(jj))))
        print('SCALE',distance,totals,flush=True)
    details.sort(key=lambda r:(-r['levels'],-r['event_count'],r['pair_index']))
    regions.sort(key=lambda r:-r['cells'])
    save_json(OUT/'local_review_windows.json',details)
    save_json(OUT/'review_regions.json',regions)
    save_json(OUT/'intrinsic_review_windows.json',intrinsic)
    save_json(OUT/'outside_H_range_lengths.json',outside_lengths)
    save_json(OUT/'topology_exclusions.json',topology_exclusions)
    save_json(OUT/'consistency_summary.json',dict(scales=scale_summaries,raw_crossings=raw_counts,
        persistent_windows=sum(d['persistent'] for d in details),
        persistent_window_categories=Counter(d['category'] for d in details if d['persistent']),
        regions=len(regions),metric_checks=metric_checks(),analysis_seconds=time.perf_counter()-start-waiting_seconds,
        waiting_for_atomic_routes_seconds=waiting_seconds,
        intrinsic_event_categories=Counter(r['category'] for r in intrinsic),
        intrinsic_affected_slices=len({r['slice_key'] for r in intrinsic}),
        observed_length_outside_H_z_range_m=sum(outside_lengths.values()),
        exact_vertex_crossings_checked=vertex_checks,vertex_retention_flags_corrected=vertex_fixes,
        additional_non_OPEN_raw_matches=topology_counts,
        core_qualified_but_topology_excluded_branches=len(topology_exclusions),
        self_edge_only_excluded_core_branches=sum(t['branching_caused_only_by_self_edges'] for t in topology_exclusions),
        self_edge_only_affected_slices=len({t['slice_key'] for t in topology_exclusions if t['branching_caused_only_by_self_edges']}),
        thresholds=dict(persistent_min_consecutive_levels=3,h_spacing_m=.1,recognition_parameters_changed=False),
        caveat='Selection consistency on matched reconstructed geometry, not geological accuracy. Regions are s-z projection clusters; different u surfaces can overlap.'))


def local_geometry(records, bins):
    """Exact line clipping by Z; no smoothing/resampling or connector length."""
    p=np.asarray([e['points_uz'] for e in records if e['source'].startswith('OBSERVED')],dtype=float).reshape(-1,2,2)
    if not len(p):return np.zeros((len(bins)-1,2))
    dz=p[:,1,1]-p[:,0,1];length=np.linalg.norm(p[:,1]-p[:,0],axis=1);flat=abs(dz)<1e-12
    rows=[]
    for low,high in zip(bins[:-1],bins[1:]):
        a=np.divide(low-p[:,0,1],dz,out=np.zeros_like(dz),where=~flat)
        b=np.divide(high-p[:,0,1],dz,out=np.ones_like(dz),where=~flat)
        fraction=np.maximum(0,np.minimum(1,np.maximum(a,b))-np.maximum(0,np.minimum(a,b)))
        fraction[flat]=((p[flat,0,1]>=low)&(p[flat,0,1]<high)).astype(float)
        rows.append([float(np.sum(length*fraction)),float(np.sum(np.maximum(0,-dz)*fraction))])
    return np.asarray(rows)


def geometry_audit():
    """Additional all-neighbor screen for detail differences BETWEEN H levels.

    Compare fixed 2 m Z bands only with dense, substantially shared selected H
    evidence and no comparable H-set mismatch. A flag is not an algorithm error:
    differences can be already present in the original reconstructed surface.
    """
    started=time.perf_counter();m=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    recognition=json.loads((OUT/'recognition_summary.json').read_text(encoding='utf-8'))
    assert (OUT/'consistency_summary.json').exists(),'Complete raw correspondence audit first'
    extents=[r['route_z_extent'] for r in recognition['rows'] if r['route_z_extent']]
    low=np.floor(min(e[0] for e in extents)/2)*2;high=np.ceil(max(e[1] for e in extents)/2)*2+2
    bins=np.arange(low,high+1,2.);levels=np.array(m['levels']);metrics=[];windows=[];previous=None;eligible_windows=0
    # A short physical fold completely between two 0.1 m H levels must survive
    # the length/backtracking measure even though its H crossing count is equal.
    fixture=[dict(source='OBSERVED',points_uz=[[0.,0.],[0.,.08]]),
             dict(source='OBSERVED',points_uz=[[0.,.08],[0.,.03]]),
             dict(source='OBSERVED',points_uz=[[0.,.03],[0.,.1]])]
    np.testing.assert_allclose(local_geometry(fixture,np.array([0.,.1])),[[.2,.05]],atol=1e-12)
    for i,key in enumerate(m['keys']):
        with (OUT/'routes'/f'{key}.pkl').open('rb') as f:d=pickle.load(f)
        normalized=OUT/'normalized_crossings'/f'{key}.npy'
        a=np.load(normalized) if normalized.exists() else d['crossings']
        table=unique_table(a);geo=local_geometry(d['result']['path_edges'],bins);metrics.append(geo)
        if previous is not None:
            oldkey,oldtable,oldgeo=previous
            common=oldtable.keys()&table.keys()
            common={h for h in common if oldtable[h][8] and table[h][8]}
            shared={h for h in common if oldtable[h][9] and table[h][9]}
            different={h for h in common if bool(oldtable[h][9])!=bool(table[h][9])}
            left_all={h for h,r in oldtable.items() if r[9]};right_all={h for h,r in table.items() if r[9]}
            def bands(hs):
                grouped=defaultdict(set)
                for h in hs:grouped[int(np.floor((levels[h[0]]-bins[0])/2))].add(h)
                return grouped
            both_by_band=bands(shared);den_by_band=bands(left_all|right_all);different_by_band=bands(different)
            for bi,(zlo,zhi) in enumerate(zip(bins[:-1],bins[1:])):
                both=both_by_band[bi];den=den_by_band[bi]
                if len({h[0] for h in both})<10 or len(both)<.8*len(den) or different_by_band[bi]:continue
                eligible_windows+=1
                lg,rg=oldgeo[bi],geo[bi]
                length_diff=abs(lg[0]-rg[0]);reverse_diff=abs(lg[1]-rg[1])
                length_flag=length_diff>=.5 and max(lg[0],rg[0])>=1.3*max(min(lg[0],rg[0]),.01)
                reverse_flag=reverse_diff>=.05 and max(lg[1],rg[1])>=2*max(min(lg[1],rg[1]),.005)
                if length_flag or reverse_flag:
                    windows.append(dict(left=oldkey,right=key,pair_index=i-1,category='GEOMETRY_COMPLEXITY',
                        z_min=float(zlo),z_max=float(zhi),levels=len({h[0] for h in both}),persistent=True,
                        event_count=len(both),shared_selected_H_crossings=len(both),
                        left_length_m=lg[0],right_length_m=rg[0],left_reverse_z_m=lg[1],right_reverse_z_m=rg[1],
                        length_difference_m=length_diff,reverse_difference_m=reverse_diff,
                        flags=[n for n,v in (('LENGTH_CHANGE',length_flag),('REVERSE_Z_CHANGE',reverse_flag)) if v]))
        previous=key,table,geo
        if i%500==0:print('GEOMETRY',i,flush=True)
    windows.sort(key=lambda w:(-w['reverse_difference_m'],-w['length_difference_m']))
    np.savez_compressed(OUT/'local_geometry_metrics.npz',bins=bins,metrics=np.array(metrics))
    save_json(OUT/'geometry_complexity_windows.json',windows)
    save_json(OUT/'geometry_complexity_summary.json',dict(eligible_pair_windows=eligible_windows,flagged_windows=len(windows),
        flagged_pairs=len({w['pair_index'] for w in windows}),seconds=time.perf_counter()-started,
        thresholds=dict(z_band_m=2.,minimum_shared_H_levels=10,minimum_shared_fraction=.8,
            length_difference_m=.5,length_ratio=1.3,reverse_z_difference_m=.05,reverse_ratio=2.),
        sub_H_level_fold_measure_check=True,accuracy_measured=False,
        caveat='Physical complexity changes can originate in raw geometry; flags require mesh review, not automatic correction.'))
    print('GEOMETRY COMPLETE',eligible_windows,len(windows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('audit','check','geometry'))
    parser.add_argument('--follow',action='store_true')
    args=parser.parse_args()
    if args.action=='check':print(metric_checks())
    elif args.action=='geometry':geometry_audit()
    else:audit(args.follow)
