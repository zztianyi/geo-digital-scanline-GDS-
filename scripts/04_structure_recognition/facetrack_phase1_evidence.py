"""Independent observed evidence. No route, junction or neighbor state input."""
from dataclasses import dataclass
import numpy as np

REGION_VERSION='physical-components-exact-segment-cells-v1'
POLICY_VERSION='through-contiguous-cc-bounded-exit-v2'


@dataclass(frozen=True)
class Phase1Policy:
    cell_m: float=.5
    exit_margin_m: float=1.
    continuation_quantum_m: float=.001
    def __post_init__(self):
        if min(self.cell_m,self.exit_margin_m,self.continuation_quantum_m)<=0:raise ValueError('Positive fixed scales required')


def segment_cells(points,cell_m):
    """Exact positive-length segment pieces: (u cell,z cell,edge,arc0,arc1)."""
    points=np.asarray(points);delta=np.diff(points,axis=0);length=np.linalg.norm(delta,axis=1)
    arcs=np.r_[0.,np.cumsum(length)];cells=np.floor(points/cell_m).astype(np.int64);out=[]
    for i,(a,v,L) in enumerate(zip(points[:-1],delta,length)):
        if L<=1e-12:continue
        if np.array_equal(cells[i],cells[i+1]):
            out.append((int(cells[i,0]),int(cells[i,1]),i,float(arcs[i]),float(arcs[i+1])));continue
        cuts=[0.,1.]
        for k in (0,1):
            if abs(v[k])<=1e-15:continue
            low,high=sorted((a[k],a[k]+v[k]))
            edges=np.arange(np.floor(low/cell_m)+1,np.ceil(high/cell_m))*cell_m
            cuts.extend(float(t) for t in (edges-a[k])/v[k] if 1e-12<t<1-1e-12)
        cuts=np.unique(cuts)
        for lo,hi in zip(cuts,cuts[1:]):
            if hi-lo<=1e-12:continue
            c=np.floor((a+(lo+hi)*.5*v)/cell_m).astype(np.int64)
            out.append((int(c[0]),int(c[1]),i,float(arcs[i]+lo*L),float(arcs[i]+hi*L)))
    return out


def union_intervals(values):
    result=[]
    for lo,hi in sorted(values):
        if result and lo<=result[-1][1]+1e-9:result[-1][1]=max(result[-1][1],hi)
        else:result.append([lo,hi])
    return result


def score_slice(region,s_index,s,geometry,policy):
    """Score exactly this (region,V). Geometry and region are read-only inputs."""
    cells=[(u,z) for i,u,z in region['cells'] if i==s_index];rows=[]
    for track in region['tracks']:
        members={}
        for cell in cells:
            for bid,lo,hi in geometry['cells'].get(cell,{}).get(track,[]):
                item=members.setdefault(bid,dict(intervals=[],cells=set()))
                item['intervals'].append((lo,hi));item['cells'].add(cell)
        witnesses=[]
        for bid,item in sorted(members.items()):
            b=geometry['branches'][bid];parts=union_intervals(item['intervals'])
            total=sum(hi-lo for lo,hi in parts)
            core=sum(max(0.,min(hi,b['CC_end_arc'])-max(lo,b['CC_start_arc'])) for lo,hi in parts)
            coverage=len(item['cells'])/max(1,len(cells));complete=coverage>=1.-1e-12
            lo,hi=parts[0][0],parts[-1][1]
            through=complete and len(parts)==1 and lo>1e-8 and hi<b['full_arc_length']-1e-8
            exit_length=b['full_arc_length']-hi if b['forward_sign']>0 else lo
            witnesses.append(dict(branch_id=bid,MBG=b['MBG'],through_region=through,
                competition_arc_m=total,CC_arc_m=core,CC_fraction=core/max(total,1e-30),
                cell_coverage_fraction=coverage,in_CC=total>0 and core>=total-1e-8,
                exit_continuation_m=min(policy.exit_margin_m,max(0.,exit_length))))
        eligible=[w for w in witnesses if w['MBG']]
        witness=max(eligible,key=lambda w:(w['through_region'],w['cell_coverage_fraction'],w['in_CC'],w['CC_fraction'],w['exit_continuation_m'],-w['branch_id'])) if eligible else None
        row=dict(region_id=region['region_id'],s=float(s),s_index=int(s_index),FaceTrack=int(track),present=bool(members),MBG=bool(eligible),
            member_branches=sorted(members),eligible_branches=[w['branch_id'] for w in eligible],
            witness_branch=None if witness is None else witness['branch_id'],through_region=False,in_CC=False,
            exit_continuation_m=0.,competition_arc_m=0.,CC_arc_m=0.,CC_fraction=0.,cell_coverage_fraction=0.,witnesses=witnesses)
        if witness:row.update({k:v for k,v in witness.items() if k not in ('branch_id','MBG')})
        rows.append(row)
    allowed=[r for r in rows if r['present'] and r['MBG']]
    def rank(r):return (int(r['through_region']),int(r['in_CC']),round(r['exit_continuation_m']/policy.continuation_quantum_m))
    best=max((rank(r) for r in allowed),default=None)
    tied=[r['FaceTrack'] for r in allowed if rank(r)==best]
    for r in rows:r.update(slice_vote=tied[0] if len(tied)==1 else None,slice_ambiguous=len(tied)!=1,
        slice_status='NO_ELIGIBLE_TRACK' if not tied else ('SLICE_AMBIGUOUS' if len(tied)>1 else 'UNIQUE_SLICE_VOTE'))
    return rows
