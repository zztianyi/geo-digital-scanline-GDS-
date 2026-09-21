"""FaceTrack sweet-core selection and single-profile peeling, shadow only.

No neighboring curve, selected main route, or shape score permits an excursion.
All cuts are stored as original source-edge parameter intervals.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'04_structure_recognition'))
from dominant_observed_branch import trim_record


def choose_sweet(rows):
    eligible=[r for r in rows if r['MBG']];trace=[]
    if not eligible:return dict(status='NO_MBG_CANDIDATE',selected=None,contenders=[],trace=[])
    for field in ['face_continuity','in_ASC']:
        best=max(r[field] for r in eligible);eligible=[r for r in eligible if r[field]==best]
        trace.append(dict(stage=field,remaining=[r['candidate_id'] for r in eligible]))
    # Remaining ASC and net directional reach are separate measurements. A
    # tradeoff is not silently converted into a weighted geometry/shape score.
    eligible=[r for r in eligible if not any(
        q['forward_ASC_m']>=r['forward_ASC_m']-1e-9 and q['forward_Z_m']>=r['forward_Z_m']-1e-9 and
        (q['forward_ASC_m']>r['forward_ASC_m']+1e-9 or q['forward_Z_m']>r['forward_Z_m']+1e-9)
        for q in eligible)]
    trace.append(dict(stage='directional_remaining_support',remaining=[r['candidate_id'] for r in eligible]))
    least=min(r['minimum_switches'] for r in eligible);eligible=[r for r in eligible if r['minimum_switches']==least]
    trace.append(dict(stage='minimum_switches',remaining=[r['candidate_id'] for r in eligible]))
    return dict(status='SELECTED' if len(eligible)==1 else 'AMBIGUOUS',
                selected=eligible[0]['candidate_id'] if len(eligible)==1 else None,
                contenders=[r['candidate_id'] for r in eligible],trace=trace)


def _closest_pairs(x,y):
    """Exact 3D segment minima, including interior intersections and endpoints."""
    p=x[:,0];q=y[:,0];r=x[:,1]-p;s=y[:,1]-q;w=p-q
    a=np.einsum('ij,ij->i',r,r);b=np.einsum('ij,ij->i',r,s);c=np.einsum('ij,ij->i',s,s)
    d=np.einsum('ij,ij->i',r,w);e=np.einsum('ij,ij->i',s,w);den=a*c-b*b
    ok=abs(den)>1e-24
    t=np.divide(b*e-c*d,den,out=np.zeros_like(den),where=ok)
    u=np.divide(a*e-b*d,den,out=np.zeros_like(den),where=ok)
    ts=np.column_stack([t,np.zeros_like(t),np.ones_like(t),np.clip(-d/np.maximum(a,1e-30),0,1),np.clip((b-d)/np.maximum(a,1e-30),0,1)])
    us=np.column_stack([u,np.clip(e/np.maximum(c,1e-30),0,1),np.clip((e+b)/np.maximum(c,1e-30),0,1),np.zeros_like(u),np.ones_like(u)])
    distances=np.linalg.norm(p[:,None]+ts[:,:,None]*r[:,None]-q[:,None]-us[:,:,None]*s[:,None],axis=2)
    distances[~(ok&(t>=0)&(t<=1)&(u>=0)&(u<=1)),0]=np.inf
    k=distances.argmin(axis=1);i=np.arange(len(k))
    return ts[i,k],us[i,k],distances[i,k]


def near_returns(branch,*,max_gap_m=.020,min_arc_m=.25,min_ratio=20):
    """Sweep original edge boxes, then measure nearest nonadjacent 3D segments.

    There is no resampling, coordinate snap, neighbor check, or ASC prerequisite.
    """
    records=branch['records'];uz=np.asarray([e['points_uz'] for e in records]);xyz=np.asarray([e['points_xyz'] for e in records])
    arc=np.asarray(branch['arc_positions']);length=np.diff(arc)
    low=uz.min(axis=1);high=uz.max(axis=1);active=[];events=[]
    for i in np.argsort(low[:,0],kind='stable'):
        active=[j for j in active if high[j,0]+max_gap_m>=low[i,0]]
        js=np.array([j for j in active if abs(int(i)-j)>1 and high[j,1]+max_gap_m>=low[i,1] and high[i,1]+max_gap_m>=low[j,1]],dtype=int)
        active.append(int(i))
        if not len(js):continue
        aa=np.minimum(js,i);bb=np.maximum(js,i)
        ta,tb,d=_closest_pairs(xyz[aa],xyz[bb]);start=arc[aa]+ta*length[aa];end=arc[bb]+tb*length[bb]
        along=end-start;ratio=along/np.maximum(d,1e-15)
        keep=(d<=max_gap_m+1e-12)&(along>=min_arc_m-1e-12)&(ratio>=min_ratio)
        for n in np.flatnonzero(keep):
            ia,ib=int(aa[n]),int(bb[n]);u,v=float(ta[n]),float(tb[n])
            xa=xyz[ia,0]+u*(xyz[ia,1]-xyz[ia,0]);xb=xyz[ib,0]+v*(xyz[ib,1]-xyz[ib,0])
            pa=uz[ia,0]+u*(uz[ia,1]-uz[ia,0]);pb=uz[ib,0]+v*(uz[ib,1]-uz[ib,0])
            events.append(dict(kind='SIDE_NEAR_CLOSED_EXCURSION',a_index=ia,b_index=ib,a_t=u,b_t=v,
                start_arc=float(start[n]),end_arc=float(end[n]),observed_arc_m=float(along[n]),
                D_xyz_m=float(d[n]),D_uz_m=float(np.linalg.norm(pa-pb)),delta_z_net_m=float(xb[2]-xa[2]),
                arc_gap_ratio=float(ratio[n]),a_xyz=xa.tolist(),b_xyz=xb.tolist(),a_uz=pa.tolist(),b_uz=pb.tolist()))
    # Shared vertices can describe the same interval from adjacent edge pairs.
    unique={}
    for e in events:
        k=(round(e['start_arc'],10),round(e['end_arc'],10))
        if k not in unique or e['D_xyz_m']<unique[k]['D_xyz_m']:unique[k]=e
    return sorted(unique.values(),key=lambda e:(e['start_arc'],e['end_arc']))


def _arc_records(branch,low,high):
    arc=np.asarray(branch['arc_positions']);result=[]
    for i,e in enumerate(branch['records']):
        a,b=max(low,float(arc[i])),min(high,float(arc[i+1]))
        if b-a>1e-12:result.append(trim_record(e,(a-arc[i])/(arc[i+1]-arc[i]),(b-arc[i])/(arc[i+1]-arc[i])))
    return result


def peel_branch(branch,events,*,closure_gap_m,min_arc_m,min_ratio):
    eligible=[e for e in events if e['D_xyz_m']<=closure_gap_m+1e-12 and e['observed_arc_m']>=min_arc_m-1e-12 and e['arc_gap_ratio']>=min_ratio]
    picked=[]
    # For overlapping descriptions of one excursion, keep a legal cut before
    # an over-cap cut. Wider detection must not hide a feasible 10 mm bridge.
    for e in sorted(eligible,key=lambda e:(e['D_xyz_m']>.010+1e-12,-e['observed_arc_m'],e['D_xyz_m'],e['start_arc'])):
        if not any(e['start_arc']<q['end_arc']-1e-9 and e['end_arc']>q['start_arc']+1e-9 for q in picked):picked.append(e)
    picked.sort(key=lambda e:e['start_arc']);main=[];side=[];unresolved=[];bridges=[];cursor=0.
    if branch['kind']=='CLOSED_COMPONENT':
        side=[dict(e) for e in branch['records']];picked=[];cursor=float(branch['arc_positions'][-1])
    total_arc=float(branch['arc_positions'][-1])
    chunk_bounds=list(zip([0.]+[e['end_arc'] for e in picked],
                          [e['start_arc'] for e in picked]+[total_arc]))
    chunk_lengths=[sum(float(np.linalg.norm(np.diff(r['points_xyz'],axis=0)))
                       for r in _arc_records(branch,a,b)) for a,b in chunk_bounds]
    synthetic_budgets=np.zeros(len(chunk_lengths))
    for index,e in enumerate(picked):
        main.extend(_arc_records(branch,cursor,e['start_arc']))
        side.extend(_arc_records(branch,e['start_arc'],e['end_arc']))
        if e['D_xyz_m']> .010+1e-12:unresolved.append(dict(e,reason='CLOSURE_EXCEEDS_10MM'))
        elif e['D_xyz_m']>1e-9:
            # Both incident retained chunks pay for the connector, including
            # any connector already charged at the other end of that chunk.
            incident=[index,index+1]
            if any(synthetic_budgets[j]+e['D_xyz_m']>=chunk_lengths[j] for j in incident):
                unresolved.append(dict(e,reason='REJECT_SYNTHETIC_DOMINANCE'))
            else:
                bridge=dict(source='TOPOLOGY_SWITCH',handoff_kind='SHADOW_SIDE_COMPONENT_BRIDGE',branch_id=None,
                    face_id=None,source_face_ids=[],source_segment_indices=[],points_xyz=[e['a_xyz'],e['b_xyz']],points_uz=[e['a_uz'],e['b_uz']])
                main.append(bridge);bridges.append(e['D_xyz_m'])
                synthetic_budgets[incident]+=e['D_xyz_m']
        cursor=e['end_arc']
    main.extend(_arc_records(branch,cursor,float(branch['arc_positions'][-1])))
    expected={e['edge_id']:abs(e['t1']-e['t0']) for e in branch['records']};actual={k:0. for k in expected}
    for e in main+side:
        if e['source'].startswith('OBSERVED'):actual[e['edge_id']]+=abs(e['t1']-e['t0'])
    preserved=all(abs(actual[k]-v)<1e-8 for k,v in expected.items())
    assert preserved,'Peeling lost or duplicated original source intervals'
    gaps=[float(np.linalg.norm(np.asarray(a['points_xyz'][1])-b['points_xyz'][0])) for a,b in zip(main,main[1:])]
    continuous=not unresolved and max(gaps,default=0.)<=1e-8
    return dict(MAIN_SPINE=main,SIDE_COMPONENT=side,REJECTED_REDUNDANT=[],excursions=picked,
        unresolved=unresolved,continuous=continuous,source_intervals_preserved=preserved,
        max_bridge_m=max(bridges,default=0.),bridge_count=len(bridges),
        max_bridge_R_syn=max((budget/length for budget,length in zip(synthetic_budgets,chunk_lengths) if length>0),default=0.),
        main_observed_arc_m=sum(float(np.linalg.norm(np.diff(e['points_xyz'],axis=0))) for e in main if e['source'].startswith('OBSERVED')),
        side_observed_arc_m=sum(float(np.linalg.norm(np.diff(e['points_xyz'],axis=0))) for e in side),
        used_neighbor_permission=False)
