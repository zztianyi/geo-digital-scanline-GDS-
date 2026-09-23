"""One physical identity per conflict region, then a band of real junctions."""
from collections import defaultdict
import numpy as np


def branch_region_span(branch,region):
    """Original arc envelope inside the volume; an endpoint inside is a dead end."""
    p=np.asarray(branch['points_uz']);d=np.diff(p,axis=0);low=np.zeros(len(d));high=np.ones(len(d))
    for axis,(a,b) in enumerate([(region['bounds'][2],region['bounds'][3]),(region['bounds'][4],region['bounds'][5])]):
        moving=abs(d[:,axis])>1e-14
        t0=np.divide(a-p[:-1,axis],d[:,axis],out=np.zeros(len(d)),where=moving)
        t1=np.divide(b-p[:-1,axis],d[:,axis],out=np.ones(len(d)),where=moving)
        low=np.maximum(low,np.where(moving,np.minimum(t0,t1),0.))
        high=np.minimum(high,np.where(moving,np.maximum(t0,t1),1.))
        high=np.where(~moving&((p[:-1,axis]<a)|(p[:-1,axis]>b)),-1.,high)
    valid=high>low+1e-12
    if not valid.any():return dict(through_region=False,arc_interval=None)
    arc=np.asarray(branch['arc_positions']);length=np.diff(arc)
    a=float((arc[:-1]+low*length)[valid].min());b=float((arc[:-1]+high*length)[valid].max())
    entry=np.array([np.interp(a,arc,p[:,k]) for k in (0,1)])
    exit=np.array([np.interp(b,arc,p[:,k]) for k in (0,1)])
    edges=[(0,region['bounds'][2]),(0,region['bounds'][3]),(1,region['bounds'][4]),(1,region['bounds'][5])]
    entry_faces={i for i,(k,value) in enumerate(edges) if abs(entry[k]-value)<1e-6}
    exit_faces={i for i,(k,value) in enumerate(edges) if abs(exit[k]-value)<1e-6}
    opposite=any((x in entry_faces and y in exit_faces) or (y in entry_faces and x in exit_faces) for x,y in [(0,1),(2,3)])
    # A bending through path can leave through a side wall. Covering the full
    # vertical conflict span still distinguishes it from a small corner loop.
    vertical_cover=p[:,1].min()<=region['bounds'][4] and p[:,1].max()>=region['bounds'][5]
    through=a>1e-8 and b<arc[-1]-1e-8 and (opposite or vertical_cover)
    return dict(through_region=through,arc_interval=[a,b])


def choose_region_identity(rows,profile_s=None):
    rows=[r for r in rows if r.get('MBG_pass',True) and r.get('FaceTrack')]
    tracks=defaultdict(list)
    for row in rows:tracks[row['FaceTrack']].append(row)
    summary=[]
    grid=sorted({round(float(s),8) for s in (profile_s if profile_s is not None else [r['s'] for r in rows])})
    position={s:i for i,s in enumerate(grid)}
    for tid,items in tracks.items():
        ss=sorted({float(r['s']) for r in items})
        indices=[position[round(s,8)] for s in ss]
        runs=np.split(np.asarray(ss),np.flatnonzero(np.diff(indices)>1)+1)
        longest=max(runs,key=len)
        # Reduce each V once; dense triangulation does not manufacture votes.
        per_s=[[r for r in items if float(r['s'])==s] for s in ss]
        summary.append(dict(FaceTrack=tid,consecutive_V=len(longest),continuous_s_span=float(np.ptp(longest)),
            through_V=sum(any(r.get('through_region',False) for r in part) for part in per_s),
            coverage_V=len(ss),CC_fraction=float(np.mean([np.mean([r.get('in_CC',r.get('in_ASC',False)) for r in part]) for part in per_s])),
            persistence_Z=float(np.mean([max(r.get('local_Z_m',0.) for r in part) for part in per_s])),
            forward_Z_m=float(np.mean([max(r.get('forward_Z_m',0.) for r in part) for part in per_s])),
            forward_observed_m=float(np.mean([max(r.get('forward_observed_m',0.) for r in part) for part in per_s])),
            future_switches=float(np.mean([min(r.get('minimum_switches',0) for r in part) for part in per_s]))))
    def rank(r):return (r['through_V'],r['consecutive_V'],r['continuous_s_span'],
        -r['future_switches'],r['forward_Z_m'],r['forward_observed_m'],r['CC_fraction'],r['persistence_Z'])
    summary.sort(key=lambda r:(rank(r),r['FaceTrack']),reverse=True)
    winner=summary[0]['FaceTrack'] if summary else None
    ambiguous=len(summary)>1 and rank(summary[0])==rank(summary[1])
    ss=sorted({float(r['s']) for r in rows})
    return dict(dominant_FaceTrack=winner,REGION_AMBIGUOUS=ambiguous,candidates=summary,slice_count=len(ss),
        per_slice_identity={s:winner for s in ss},competitive_core_invoked=len(summary)>=2,
        competitor_count=len(summary),decision_mode='COMPETITIVE_SELECTION' if len(summary)>=2 else 'NONCOMPETITIVE_CONTINUATION')


def choose_junction_band(layers):
    """Layered DP chooses existing candidates; coordinates are never averaged."""
    if not layers or any(not layer for layer in layers):return []
    costs=np.array([c.get('future_switches',0)*.02+c.get('xyz_distance_m',0) for c in layers[0]])
    back=[]
    for previous,current in zip(layers,layers[1:]):
        z0=np.array([c['z'] for c in previous]);z1=np.array([c['z'] for c in current])
        identity=np.array([[a.get('FaceTrack')!=b.get('FaceTrack') for b in current] for a in previous])
        transition=costs[:,None]+abs(z0[:,None]-z1[None,:])+identity*1e6
        ix=transition.argmin(axis=0);back.append(ix)
        costs=transition[ix,np.arange(len(current))]+np.array([c.get('future_switches',0)*.02+c.get('xyz_distance_m',0) for c in current])
    index=int(costs.argmin());chosen=[layers[-1][index]]
    for layer,indices in zip(layers[-2::-1],back[::-1]):index=int(indices[index]);chosen.append(layer[index])
    return chosen[::-1]


def merge_conflict_cells(cells,seed_ids,face_context):
    """Join touching cells only when two physical components continue across them.

    Shared original faces link local labels; touching boxes alone do not merge
    unrelated faults/holes into a project-wide region.
    """
    by_id={r['region_id']:r for r in cells};grid={}
    for r in cells:
        b=r['bounds'];grid[tuple(round((b[k]+.1)/step) for k,step in [(0,.5),(2,2.),(4,2.)])]=r['region_id']
    neighbors=defaultdict(list)
    for q,rid in grid.items():
        for axis in range(3):
            for delta in (-1,1):
                p=list(q);p[axis]+=delta
                if tuple(p) in grid:neighbors[rid].append(grid[tuple(p)])
    results=[];used=set()
    for seed in seed_ids:
        if seed in used:continue
        seen={seed};queue=[seed]
        for rid in queue:
            left=face_context.region_faces(by_id[rid])
            for other in neighbors[rid]:
                if other in seen:continue
                right=face_context.region_faces(by_id[other]);shared=left.keys()&right.keys()
                pairs={(left[f],right[f]) for f in shared}
                if len({a for a,b in pairs})<2 or len({b for a,b in pairs})<2:continue
                seen.add(other);queue.append(other)
        used.update(seen);bounds=np.asarray([by_id[r]['bounds'] for r in seen])
        combined=[]
        for k in (0,2,4):combined.extend([float(bounds[:,k].min()),float(bounds[:,k+1].max())])
        results.append(dict(region_id='REGION_'+min(seen),bounds=combined,cell_ids=sorted(seen)))
    return results


def prepare_region_identity(context,region,policy=None):
    from branch_absolute_core import BranchPolicy,branch_metrics,local_edge_scale
    from horizontal_surface_link import branch_crossings
    rid=region['region_id']
    if rid in context.region_decisions:return context.region_decisions[rid]
    data=context.region_data(region);b=region['bounds'];rows=[]
    for s,branches in context.profiles.items():
        if not b[0]-1e-8<=s<=b[1]+1e-8:continue
        metrics={x['branch_id']:branch_metrics(x,local_edge_scale(branches),policy) for x in branches}
        candidates=[x for x in branches if metrics[x['branch_id']]['MBG_pass'] and (s,x['branch_id']) in data['branches']]
        # Original measured H levels are used when supplied. A midpoint is a
        # geometry query only, never an invented measured point or junction.
        levels=[z for z in context.levels if b[4]<=z<=b[5]] or [(b[4]+b[5])/2]
        levels=np.asarray(levels);sampled=[]
        for branch in candidates:
            tids=data['branches'].get((s,branch['branch_id']),[])
            if len(tids)!=1:continue
            m=metrics[branch['branch_id']];points=np.asarray(branch['points_uz']);arc=np.asarray(branch['arc_positions'])
            za=points[:-1,1,None];zb=points[1:,1,None];dz=zb-za
            active=(levels[None,:]>=np.minimum(za,zb)-1e-9)&(levels[None,:]<=np.maximum(za,zb)+1e-9)
            t=np.divide(levels[None,:]-za,dz,out=np.zeros_like(active,dtype=float),where=abs(dz)>1e-12)
            t=np.clip(t,0,1);u=points[:-1,0,None]+t*np.diff(points[:,0])[:,None]
            active &= (u>=b[2])&(u<=b[3])
            positions=arc[:-1,None]+t*np.diff(arc)[:,None]
            present=active.any(axis=0)
            # Keep raw internal positions until competitor counts are known.
            sampled.append((branch,m,tids[0],present,active,positions))
        count=sum((x[3].astype(int) for x in sampled),np.zeros(len(levels),int))
        for branch,m,tid,present,active,positions in sampled:
            shared=present&(count>=2)
            if not shared.any():continue
            core=(active[:,shared]&(positions[:,shared]>=m['ASC_start_arc'])&(positions[:,shared]<=m['ASC_end_arc'])).any(axis=0)
            from physical_continuation import future_switch_estimate
            reliable=[x for x in branches if metrics[x['branch_id']]['MBG_pass']]
            future=future_switch_estimate(reliable,branch['branch_id'],context,s,policy or BranchPolicy())
            rows.append(dict(s=s,branch_id=branch['branch_id'],MBG_pass=True,FaceTrack=f'{rid}:T{tid}',
                through_region=branch_region_span(branch,region)['through_region'],
                in_CC=float(core.mean()),forward_Z_m=max(0.,branch['z_range'][1]-float(levels[shared].mean())),
                forward_observed_m=float(branch['arc_positions'][-1]),local_Z_m=float(np.ptp(levels[shared])),minimum_switches=future))
    decision=choose_region_identity(rows,profile_s=[s for s in context.profiles if b[0]-1e-8<=s<=b[1]+1e-8]);decision.update(region_id=rid,bounds=b,competing_branch_ids=sorted({r['branch_id'] for r in rows}),
        competing_FaceTracks=[r['FaceTrack'] for r in decision['candidates']],junction_reason='REGION_IDENTITY_ONLY',sample_rows=len(rows))
    context.region_decisions[rid]=decision;return decision


def prepare_region_locks(context,regions,policy=None):
    """Commit an available through-region source before any per-profile assembly."""
    context.region_locks={};context.region_lock_regions={}
    for region in regions:
        decision=prepare_region_identity(context,region,policy);winner=decision['dominant_FaceTrack']
        if winner is None or decision['REGION_AMBIGUOUS']:continue
        data=context.region_data(region)
        for s,branches in context.profiles.items():
            if not region['bounds'][0]-1e-8<=s<=region['bounds'][1]+1e-8:continue
            candidates=[]
            for branch in branches:
                tids=data['branches'].get((s,branch['branch_id']),[])
                if len(tids)!=1 or f"{region['region_id']}:T{tids[0]}"!=winner:continue
                span=branch_region_span(branch,region)
                if span['through_region']:candidates.append((branch,span))
            # Never silently turn a fragmented/ambiguous source into one branch.
            if len(candidates)!=1:continue
            branch,span=candidates[0];bid=branch['branch_id']
            lock=context.region_locks.setdefault(s,{})
            old=lock.get(bid,span['arc_interval'])
            lock[bid]=[min(old[0],span['arc_interval'][0]),max(old[1],span['arc_interval'][1])]
            context.region_lock_regions.setdefault(s,{})[region['region_id']]=dict(branch_id=bid,FaceTrack=winner,arc_interval=span['arc_interval'])
    return context.region_lock_regions
