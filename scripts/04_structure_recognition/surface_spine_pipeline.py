"""Production P0/P1/P2 entry: observed branches -> spine -> hanging segments.

The caller supplies physical FaceTrack context from the exact slicing mesh.
No validation ROI, historical selected route or experiment module is imported.
"""
import time
import numpy as np
from dominant_observed_branch import solve_dominant_branch
from main_spine_components import classify_components,reconstruction_inputs


def hanging_group_adapter(records):
    """Keep the existing consumers' group_segments schema plus full FaceIDs."""
    groups=[];group=None;last=None
    for r in records:
        if r.get('structure_eligible',True) is False:
            last=None
            continue
        p=np.asarray(r['points_uz']);xyz=np.asarray(r['points_xyz']);delta=p[1]-p[0]
        length=np.linalg.norm(delta)
        if last is None or np.linalg.norm(last-xyz[0])>1e-8:
            group=dict(group_segments=[],node_order=[p[0].tolist()],connectivity={})
            groups.append(group)
        normal=np.array([-delta[1],delta[0]])/max(length,1e-30)
        group['group_segments'].append((p[0],p[1],p.mean(axis=0),normal,xyz[0],xyz[1],
            r.get('face_id'),list(r['source_face_ids'])))
        group['node_order'].append(p[1].tolist());last=xyz[1]
    return {0:groups}


def recognize_surface_spine(branches,graph,target_s,*,policy=None,input_z_range=None):
    if graph.get('physical_face_context') is None:
        raise ValueError('Production FaceTrack recognition requires physical mesh context')
    from region_joint_selection import prepare_region_locks
    context=graph['physical_face_context']
    if not hasattr(context,'region_locks'):
        prepare_region_locks(context,[r for r in context.regions if r['bounds'][0]-1e-8<=float(target_s)<=r['bounds'][1]+1e-8],policy)
    started=time.perf_counter()
    route=solve_dominant_branch(None,None,branches=branches,surface_graph=graph,
        target_s=float(target_s),policy=policy,input_z_range=input_z_range)
    from local_competitive_handoff import apply_internal_competitions
    route=apply_internal_competitions(branches,route,graph,float(target_s),policy=policy)
    route['analysis_only']=False
    return finish_surface_spine(branches,route,graph['physical_face_context'],target_s,
        policy=policy,p1_seconds=time.perf_counter()-started)


def finish_surface_spine(branches,route,context,target_s,*,policy=None,p1_seconds=0.):
    """Apply P2/recognition to an unchanged, provenance-preserving P1 route."""
    start=time.perf_counter()
    layers=classify_components(branches,route,context=context,
        target_s=float(target_s),policy=policy)
    from reviewed_model_gap import is_estimated_gap
    estimates=[r for r in layers['MAIN_SPINE'] if is_estimated_gap(r)]
    route['estimated_gap_count']=len(estimates)
    route['contains_estimated_model_hole']=bool(estimates)
    if estimates and route.get('status')=='ACCEPT_OBSERVED_BRANCH':
        route['status']='ACCEPT_WITH_REVIEWED_MODEL_GAP'
    # The assembled path already has its order. Do not run endpoint BFS again:
    # it could reorder folds or map a measured edge to another coincident face.
    hanging=[]
    for r in layers['MAIN_SPINE']:
        if not r['source'].startswith('OBSERVED') or r.get('structure_eligible',True) is False:continue
        delta=np.diff(r['points_uz'],axis=0)[0];length=np.linalg.norm(delta)
        if length>1e-12 and delta[0]/length<-.1:
            hanging.append(dict(r,recognition='HANGING_SEGMENT',normal_vertical=float(delta[0]/length)))
    return dict(route=route,layers=layers,hanging_segments=hanging,red_groups_corrected=hanging_group_adapter(hanging),
        reconstruction=reconstruction_inputs(layers,hanging),
        performance=dict(P0_P1_seconds=p1_seconds,P2_recognition_seconds=time.perf_counter()-start),
        production_stage='P0_P1_P2',FaceTrack_used_in_recognition=True)


def join_region_spines(routes,profiles,context,regions,*,policy=None):
    """Batch production gate: finish all V candidates before choosing a band."""
    from region_junction_band import apply_region_junction_bands
    audit=apply_region_junction_bands(routes,profiles,context,regions,policy)
    for route in routes.values():
        route['joint_region_decisions']=list(context.region_decisions)
        route['region_joint_stage_complete']=True
    return audit


def recognize_conflict_region(profiles,graph,regions,*,policy=None,input_z_range=None):
    """Production batch API; profiles includes every V of the conflict region."""
    from local_competitive_handoff import apply_internal_competitions
    from region_joint_selection import prepare_region_locks
    prepare_region_locks(graph['physical_face_context'],regions,policy)
    routes={};times={}
    for key,branches in profiles.items():
        began=time.perf_counter()
        route=solve_dominant_branch(None,None,branches=branches,surface_graph=graph,
            target_s=float(key),policy=policy,input_z_range=input_z_range)
        routes[key]=apply_internal_competitions(branches,route,graph,float(key),policy=policy)
        times[key]=time.perf_counter()-began
    audit=join_region_spines(routes,profiles,graph['physical_face_context'],regions,policy=policy)
    return dict(junction_bands=audit,profiles={key:finish_surface_spine(profiles[key],route,
        graph['physical_face_context'],float(key),policy=policy,p1_seconds=times[key]) for key,route in routes.items()})
