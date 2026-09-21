"""Bounded local face provenance audit. Never writes recognition routes."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,pickle,time
import numpy as np
from scipy import sparse
from validate_physical_topology import output_path,load_inventory,csv_rows,PRIOR
from run_face_provenance_validation import manifest,digest
from census_directional_consistency import save_json
from locc_data import opc
from vertical_profile_canonicalization import canonicalize_vertical
from horizontal_surface_link import canonical_horizontal_paths,branch_crossings
from branch_absolute_core import branch_metrics,local_edge_scale
from face_provenance_audit import FaceAdjacency


def line_key(line):
    a,b=map(tuple,line)
    return (a,b) if a<b else (b,a)


def prepare_horizontal(out):
    start=time.perf_counter();m=manifest(out);regions=json.loads((out/'validation_regions.json').read_text(encoding='utf-8'))
    levels=np.array(m['levels']);positions=sorted({float(k) for r in regions for k in r['target_keys']})
    wanted=defaultdict(list)
    for r in regions:
        for li in np.flatnonzero(abs(levels-r['center_z'])<=3.+1e-8):
            for key in r['target_keys']:wanted[(int(li),key)].append(r['bounds'][2:4])
    raw_path=Path(m['prior'])/'horizontal_slices_raw.pkl';clean_path=Path(m['prior'])/'horizontal_slices_clean.pkl'
    offsets={};rawfile=raw_path.open('rb')
    while True:
        offset=rawfile.tell()
        try:row=pickle.load(rawfile)
        except EOFError:break
        li=int(np.argmin(abs(levels-row['position'])));offsets[li]=offset
    result={};stats=Counter()
    with clean_path.open('rb') as f:
        while True:
            try:row=pickle.load(f)
            except EOFError:break
            li=row['level_index']
            if not any((li,f'{s:.2f}') in wanted for s in positions):continue
            rawfile.seek(offsets[li]);raw=pickle.load(rawfile)
            original=canonicalize_vertical(raw['lines_3d'],raw['face_ids'])
            sources={line_key(line):ids for line,ids in zip(original.lines_xyz,original.source_face_ids)}
            clean=canonical_horizontal_paths(row['lines_3d'],row['face_ids'],m['arc'])
            full=[sources.get(line_key(line),[]) for line in clean['lines_3d']]
            stats['clean_edges_with_restored_full_provenance']+=sum(bool(x) for x in full)
            stats['clean_edges_without_raw_equivalent']+=sum(not x for x in full)
            # Keep incident-edge hits distinct before unioning all source faces.
            n=len(full);hits=opc.horizontal_crossings(dict(clean,edge_branch=np.arange(n),face_ids=np.arange(n)),positions,m['arc'])
            groups={}
            for si,u,_,eid in hits:
                key=f'{positions[int(si)]:.2f}';bounds=wanted.get((li,key))
                if not bounds or not any(lo<=u<=hi for lo,hi in bounds):continue
                eid=int(eid);pid=int(clean['edge_branch'][eid]);identity=(key,pid,round(float(u),7))
                hit=groups.setdefault(identity,dict(slice_key=key,level_index=li,z=float(levels[li]),u=float(u),h_path_id=pid,
                    source_face_ids=set(),H_edge_ids=[],provenance_complete=True))
                hit['source_face_ids'].update(full[eid]);hit['H_edge_ids'].append(eid)
                hit['provenance_complete']&=bool(full[eid])
            for hit in groups.values():
                hit['source_face_ids']=sorted(hit['source_face_ids'])
                result.setdefault((li,hit['slice_key']),[]).append(hit)
                stats['raw_H_crossings']+=1;stats['multiple_source_faces_H_crossings']+=len(hit['source_face_ids'])>1
    rawfile.close()
    with (out/'horizontal_provenance.pkl').open('wb') as f:pickle.dump(result,f,protocol=5)
    save_json(out/'horizontal_provenance_summary.json',dict(stats,seconds=time.perf_counter()-start,
        raw_source=str(raw_path),clean_geometry=str(clean_path),snap_tolerance_m=1e-10,
        policy='Full sources restored only through exact undirected round6 geometry; unmatched edges remain unknown, no first-face fallback.'))
    print('H PROVENANCE',dict(stats),flush=True)


def clip_uz(points,bounds):
    p=np.asarray(points,dtype=float);delta=p[1]-p[0];lo,hi=0.,1.
    for axis,(low,high) in enumerate(((bounds[2],bounds[3]),(bounds[4],bounds[5]))):
        if abs(delta[axis])<1e-14:
            if not low<=p[0,axis]<=high:return None
        else:
            a,b=sorted(((low-p[0,axis])/delta[axis],(high-p[0,axis])/delta[axis]))
            lo,hi=max(lo,a),min(hi,b)
            if hi<lo:return None
    return np.array([p[0]+lo*delta,p[0]+hi*delta]),lo,hi


def selected_intervals(route):
    result=defaultdict(list)
    for e in route['path_edges']:
        if e['source'].startswith('OBSERVED'):result[e['edge_id']].append(sorted((e['t0'],e['t1'])))
    return result


def audit(out):
    start=time.perf_counter();m=manifest(out);regions=json.loads((out/'validation_regions.json').read_text(encoding='utf-8'))
    routes_manifest=json.loads((out/'local_route_summary.json').read_text(encoding='utf-8'))
    with (out/'horizontal_provenance.pkl').open('rb') as f:horizontal=pickle.load(f)
    faces=np.load(Path(m['prior'])/'mesh_cache/faces.npy',mmap_mode='r')
    vertices=np.load(Path(m['prior'])/'mesh_cache/vertices.npy',mmap_mode='r')
    adjacency=FaceAdjacency(faces,sparse.load_npz(out/'face_adjacency.npz'))
    lows=np.load(out/'face_bounds_min.npy',mmap_mode='r');highs=np.load(out/'face_bounds_max.npy',mmap_mode='r')
    all_inventory=[];all_members=[];all_vh=[];all_vv=[];all_shadow=[];summaries=[];branch_tables=[]
    (out/'region_data').mkdir(exist_ok=True)
    levels=np.array(m['levels'])
    import trimesh
    for region in regions:
        rid=region['region_id'];bounds=region['bounds'];keys=region['target_keys'];bs=load_inventory(out,keys)
        with (out/'local_routes'/f'{rid}.pkl').open('rb') as f:rr=pickle.load(f)
        current=rr['CURRENT_ROUTE'];intervals={k:selected_intervals(r) for k,r in current.items()}
        metric={k:{b['branch_id']:branch_metrics(b,local_edge_scale(v)) for b in v} for k,v in bs.items()}
        member_rows=[];source_faces=set();tri_points=[];tri_ids=[];vc=defaultdict(list)
        lis=np.flatnonzero(abs(levels-region['center_z'])<=3.+1e-8)
        for key,branches in bs.items():
            for b in branches:
                for e in b['records']:
                    clipped=clip_uz(e['points_uz'],bounds)
                    if clipped is None:continue
                    p,t0,t1=clipped;is_selected=any(max(a,t0)<=min(c,t1)+1e-9 for a,c in intervals[key][e['edge_id']])
                    member_rows.append(dict(region_id=rid,s=key,branch_id=b['branch_id'],edge_id=e['edge_id'],kind=b['kind'],
                        MBG=metric[key][b['branch_id']]['MBG_pass'],source_face_ids=e['source_face_ids'],
                        selected_current_route=is_selected,points_uz=p.tolist(),t_interval=[t0,t1]))
                    source_faces.update(e['source_face_ids'])
                    for face in e['source_face_ids']:
                        tri_ids.extend([face,face]);tri_points.extend(e['points_xyz'])
                for li in lis:
                    local={}
                    for c in branch_crossings(b,levels[li]):
                        if not bounds[2]<=c['u']<=bounds[3]:continue
                        e=b['records'][c['edge_index']];identity=round(c['arc_position'],8)
                        v=local.setdefault(identity,dict(s=key,level_index=int(li),z=float(levels[li]),u=c['u'],branch_id=b['branch_id'],
                            MBG=metric[key][b['branch_id']]['MBG_pass'],edge_ids=[],source_face_ids=set(),selected=False,point_xyz=c['point_xyz'].tolist()))
                        v['edge_ids'].append(e['edge_id']);v['source_face_ids'].update(e['source_face_ids'])
                        v['selected']|=any(a-1e-9<=c['t']<=d+1e-9 for a,d in intervals[key][e['edge_id']])
                    for v in local.values():
                        v['source_face_ids']=sorted(v['source_face_ids']);source_faces.update(v['source_face_ids']);vc[(int(li),key)].append(v)
        for li in lis:
            for key in keys:
                for h in horizontal.get((int(li),key),[]):source_faces.update(h['source_face_ids'])
        lower=np.array([bounds[0],bounds[2],bounds[4]]);upper=np.array([bounds[1],bounds[3],bounds[5]])
        allowed=set(np.flatnonzero(np.all(highs>=lower,axis=1)&np.all(lows<=upper,axis=1)).tolist())
        # An observed segment can cross the ROI while its triangle spans outside.
        assert source_faces.issubset(allowed),'Observed source face does not intersect local s/u/z ROI'
        labels=adjacency.local_labels(allowed)
        active_labels=sorted({labels[f] for f in source_faces});names={lab:f'{rid}_T{i+1}' for i,lab in enumerate(active_labels)}
        def track_ids(face_ids):return sorted({names[labels[f]] for f in face_ids if f in labels and labels[f] in names})
        for e in member_rows:e['face_track_ids']=track_ids(e['source_face_ids'])
        for values in vc.values():
            for v in values:v['face_track_ids']=track_ids(v['source_face_ids'])
        # Verify that FaceID really uses the slicing mesh's indexing space.
        if tri_ids:
            actual=np.asarray(tri_points);closest=trimesh.triangles.closest_point(np.asarray(vertices[faces[np.asarray(tri_ids)]]),actual)
            residual=np.linalg.norm(actual-closest,axis=1)
            assert float(residual.max())<=3e-6,(rid,'source FaceID space mismatch',float(residual.max()))
        else:residual=np.array([0.])
        vh=[];cache={}
        def relate(a,b):
            signature=(tuple(sorted(a)),tuple(sorted(b)))
            if signature not in cache:cache[signature]=adjacency.relation(a,b,allowed,max_hops=8)
            return cache[signature]
        for li in lis:
            for key in keys:
                for hi,h in enumerate(horizontal.get((int(li),key),[])):
                    matched=[v for v in vc[(int(li),key)] if abs(v['u']-h['u'])<=3e-6]
                    for v in matched:
                        evidence=relate(v['source_face_ids'],h['source_face_ids']);ht=track_ids(h['source_face_ids'])
                        ambiguous=len(v['face_track_ids'])!=1 or len(ht)!=1 or not h['provenance_complete']
                        vh.append(dict(region_id=rid,s=key,level_index=int(li),z=float(levels[li]),u=h['u'],H_path_id=h['h_path_id'],
                            V_branch_id=v['branch_id'],V_edge_ids=v['edge_ids'],V_source_face_ids=v['source_face_ids'],H_source_face_ids=h['source_face_ids'],
                            V_face_tracks=v['face_track_ids'],H_face_tracks=ht,selected_current_route=v['selected'],
                            geometry_match_count=len(matched),relation='AMBIGUOUS_FACE_PROVENANCE' if ambiguous else evidence['relation'],
                            raw_relation=evidence['relation'],witness_face_chain=evidence['chain'],weak_vertex_contact=evidence['weak_vertex_contact']))
        vv=[];shadow=[]
        for li in lis:
            for left,right in zip(keys[:-1],keys[1:]):
                a=[v for v in vc[(int(li),left)] if v['selected']];b=[v for v in vc[(int(li),right)] if v['selected']]
                ats={t for v in a for t in v['face_track_ids']};bts={t for v in b for t in v['face_track_ids']}
                abids=sorted({v['branch_id'] for v in a});bbids=sorted({v['branch_id'] for v in b})
                witnesses=[]
                ambiguous=any(len(v['face_track_ids'])!=1 for v in a+b)
                if not a or not b:classification='NO_FACE_EVIDENCE'
                elif ambiguous:classification='AMBIGUOUS_FACE_TRACK'
                elif ats.isdisjoint(bts):classification='FACE_TRACK_IDENTITY_SWITCH'
                elif ats!=bts:classification='AMBIGUOUS_FACE_TRACK'
                else:
                    supported=set()
                    for av in a:
                        for bv in b:
                            if av['face_track_ids']!=bv['face_track_ids']:continue
                            relation=relate(av['source_face_ids'],bv['source_face_ids'])
                            if relation['relation'] in ('EXACT_FACE','EDGE_ADJACENT_FACE','LOCAL_FACE_CHAIN'):
                                supported.update(av['face_track_ids']);witnesses.append(relation['chain'])
                    classification=('BRANCH_ID_CHANGE_ONLY' if abids!=bbids else 'CONSISTENT_FACE_TRACK') if supported==ats else 'NO_FACE_EVIDENCE'
                vv.append(dict(region_id=rid,left_s=left,right_s=right,level_index=int(li),z=float(levels[li]),
                    left_selected_tracks=sorted(ats),right_selected_tracks=sorted(bts),left_branch_ids=abids,right_branch_ids=bbids,
                    classification=classification,relation=('SAME_FACE_TRACK' if classification in ('BRANCH_ID_CHANGE_ONLY','CONSISTENT_FACE_TRACK')
                        else 'DIFFERENT_FACE_TRACK' if classification=='FACE_TRACK_IDENTITY_SWITCH' else classification),
                    witness_face_chains=witnesses))
            for i,key in enumerate(keys):
                own=vc[(int(li),key)];selected={t for v in own if v['selected'] for t in v['face_track_ids']}
                for v in own:
                    if v['selected'] or len(v['face_track_ids'])!=1:continue
                    tid=v['face_track_ids'][0]
                    if tid in selected:continue
                    support=[];witnesses=[]
                    for j in (i-1,i+1):
                        if not 0<=j<len(keys):continue
                        for neighbor in vc[(int(li),keys[j])]:
                            if not neighbor['selected'] or neighbor['face_track_ids']!=[tid]:continue
                            relation=relate(v['source_face_ids'],neighbor['source_face_ids'])
                            if relation['relation'] in ('EXACT_FACE','EDGE_ADJACENT_FACE','LOCAL_FACE_CHAIN'):
                                support.append(keys[j]);witnesses.append(relation['chain'])
                    if support:
                        shadow.append(dict(region_id=rid,s=key,level_index=int(li),z=float(levels[li]),u=v['u'],
                            candidate_branch_id=v['branch_id'],candidate_edge_ids=v['edge_ids'],candidate_MBG=v['MBG'],
                            face_track_id=tid,neighbor_support_s=sorted(set(support),key=float),support_sides=len(set(support)),
                            current_selected_tracks=sorted(selected),witness_face_chains=witnesses,
                            classification='FACE_TRACK_MISSED_EXISTING_BRANCH',action='SHADOW_ONLY_NO_ROUTE_WRITE'))
        inventory=[]
        for lab,tid in names.items():
            fs=sorted(f for f,v in labels.items() if v==lab);members=[e for e in member_rows if tid in e['face_track_ids']]
            hrows=[h for h in vh if tid in h['H_face_tracks']]
            inventory.append(dict(region_id=rid,face_track_id=tid,face_ids=fs,face_count=len(fs),
                shared_edge_adjacency_count=int(adjacency.adjacency[fs][:,fs].nnz//2),
                exact_face_overlap_count=sum(h['relation']=='EXACT_FACE' for h in hrows),
                member_V_slices=sorted({e['s'] for e in members},key=float),member_V_branch_ids=sorted({(e['s'],e['branch_id']) for e in members}),
                member_V_edge_ids=sorted({(e['s'],e['edge_id']) for e in members}),
                member_H_levels=sorted({h['level_index'] for h in hrows}),member_H_path_ids=sorted({(h['level_index'],h['H_path_id']) for h in hrows}),
                s_span=[float(min((e['s'] for e in members),key=float,default=str(bounds[0]))),float(max((e['s'] for e in members),key=float,default=str(bounds[1])))],
                z_span=[float(lows[fs,2].min()),float(highs[fs,2].max())],u_range=[float(lows[fs,1].min()),float(highs[fs,1].max())],
                selected_route_members=sum(e['selected_current_route'] for e in members),candidate_route_members=len(members)))
        competition=False;merged_competition=False;ambiguous_members=any(len(e['face_track_ids'])!=1 for e in member_rows)
        for values in vc.values():
            for i,a in enumerate(values):
                for b in values[i+1:]:
                    if a['branch_id']==b['branch_id'] or not(a['MBG'] and b['MBG']):continue
                    if len(a['face_track_ids'])==len(b['face_track_ids'])==1:
                        if a['face_track_ids']==b['face_track_ids']:merged_competition=True
                        else:competition=True
        discriminative=competition and not merged_competition and not ambiguous_members
        verdict='DISCRIMINATIVE' if discriminative else 'AMBIGUOUS' if ambiguous_members or (competition and merged_competition) else 'NON_DISCRIMINATIVE'
        summary=dict(region,face_track_count=len(inventory),allowed_mesh_faces=len(allowed),verdict=verdict,
            distinct_competing_tracks_observed=competition,competing_branches_merged_in_local_track=merged_competition,
            ambiguous_member_provenance=ambiguous_members,VH_relation_counts=dict(Counter(h['relation'] for h in vh)),
            VV_classification_counts=dict(Counter(h['classification'] for h in vv)),shadow_count=len(shadow),
            source_face_endpoint_max_residual_m=float(residual.max()),source_face_endpoint_checks=len(tri_ids),max_face_chain_hops=8)
        for key,branches in bs.items():
            for b in branches:
                members=[e for e in member_rows if e['s']==key and e['branch_id']==b['branch_id']]
                if not members:continue
                tids=sorted({t for e in members for t in e['face_track_ids']})
                neighbors=sorted({t for k in keys if abs(float(k)-float(key))<=.050001 and k!=key
                    for e in member_rows if e['s']==k and e['selected_current_route'] for t in e['face_track_ids']})
                hints=[s for s in shadow if s['s']==key and s['candidate_branch_id']==b['branch_id']]
                branch_tables.append(dict(region_id=rid,s=key,branch_id=b['branch_id'],kind=b['kind'],MBG=metric[key][b['branch_id']]['MBG_pass'],
                    ASC_length_m=metric[key][b['branch_id']]['ASC_arc_length'],source_FaceID_count=len({f for e in members for f in e['source_face_ids']}),
                    FaceTrack_ID=tids,selected_current_route=any(e['selected_current_route'] for e in members),
                    neighbor_selected_FaceTrack=neighbors,shadow_recommendation=bool(hints),
                    classification='FACE_TRACK_MISSED_EXISTING_BRANCH' if hints else 'AMBIGUOUS_FACE_PROVENANCE' if any(len(e['face_track_ids'])!=1 for e in members) else 'LOCAL_FACE_TRACK_MEMBER'))
        with (out/'region_data'/f'{rid}.pkl').open('wb') as f:pickle.dump(dict(region=summary,members=member_rows,V_crossings=dict(vc),VH=vh,VV=vv,shadow=shadow,tracks=inventory),f,protocol=5)
        all_inventory.extend(inventory);all_members.extend(member_rows);all_vh.extend(vh);all_vv.extend(vv);all_shadow.extend(shadow);summaries.append(summary)
        print('FACE REGION',rid,verdict,len(inventory),dict(Counter(h['relation'] for h in vh)),len(shadow),flush=True)
    for name,rows in [('face_track_inventory',all_inventory),('face_track_members',all_members),('vh_face_provenance_crossings',all_vh),
        ('conflict_region_face_tracks',summaries),('neighbor_route_face_track_consistency',all_vv),('shadow_face_track_recommendations',all_shadow),('case_branch_table',branch_tables)]:
        csv_rows(out/(name+'.csv'),rows)
    assert routes_manifest['CURRENT_ROUTE_sha256']=={p.name:digest(p) for p in (out/'local_routes').glob('*.pkl')}
    scope=json.loads((out/'region_scope.json').read_text(encoding='utf-8'));vhc=Counter(x['relation'] for x in all_vh);vvc=Counter(x['classification'] for x in all_vv)
    save_json(out/'face_validation_summary.json',dict(scope,regions=summaries,seconds=time.perf_counter()-start,
        face_track_discriminative_regions=sum(r['verdict']=='DISCRIMINATIVE' for r in summaries),
        face_track_non_discriminative_regions=sum(r['verdict']=='NON_DISCRIMINATIVE' for r in summaries),
        face_track_ambiguous_regions=sum(r['verdict']=='AMBIGUOUS' for r in summaries),
        VH_relation_counts=dict(vhc),VV_classification_counts=dict(vvc),
        geometry_match_exact_face_count=vhc['EXACT_FACE'],geometry_match_adjacent_face_count=vhc['EDGE_ADJACENT_FACE'],
        geometry_match_local_chain_count=vhc['LOCAL_FACE_CHAIN'],geometry_only_count=vhc['GEOMETRY_ONLY'],
        route_difference_same_face_track_count=vvc['BRANCH_ID_CHANGE_ONLY'],route_difference_different_face_track_count=vvc['FACE_TRACK_IDENTITY_SWITCH'],
        missing_route_with_same_face_track_candidate_count=len(all_shadow),
        missed_candidate_slice_regions=len({(x['region_id'],x['s']) for x in all_shadow}),
        CURRENT_ROUTE_unchanged_by_FaceTrack=True,face_source_coordinate_checks=sum(r['source_face_endpoint_checks'] for r in summaries),
        maximum_source_coordinate_residual_m=max(r['source_face_endpoint_max_residual_m'] for r in summaries),
        units='VH = geometric crossing pairs; VV = adjacent V pair at one H level; shadow = candidate crossing, not whole independent missing branches.',
        geological_accuracy_measured=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('horizontal','audit'));p.add_argument('--output',type=Path,default=output_path());a=p.parse_args()
    (prepare_horizontal if a.action=='horizontal' else audit)(a.output)
