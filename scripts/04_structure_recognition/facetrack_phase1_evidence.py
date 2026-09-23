"""Independent observed evidence. No route, junction or neighbor state input."""
from dataclasses import dataclass
from collections import defaultdict
from math import isfinite,sqrt
import numpy as np

REGION_VERSION='physical-components-exact-segment-cells-v1'
POLICY_VERSION='qualitative-continuity-cc-no-exact-tiebreak-v3'


@dataclass(frozen=True)
class Phase1Policy:
    cell_m: float=.5
    exit_margin_m: float=1.
    continuation_quantum_m: float=.001
    continuity_good_min_coverage: float=.90
    continuity_good_min_exit_m: float=.25
    continuity_weak_max_coverage: float=.50
    continuity_weak_max_exit_m: float=.05
    continuity_boundary_cell_tolerance: int=1
    continuity_gap_tolerance_cell_diagonals: float=1.
    continuity_good_min_covered_arc_cells: float=2.
    continuity_good_min_interval_fraction: float=.90
    def __post_init__(self):
        values=(self.cell_m,self.exit_margin_m,self.continuation_quantum_m,self.continuity_good_min_exit_m,
                self.continuity_weak_max_exit_m,self.continuity_gap_tolerance_cell_diagonals,self.continuity_good_min_covered_arc_cells)
        if not all(isfinite(v) and v>0 for v in values):raise ValueError('Positive finite fixed scales required')
        if not 0<=self.continuity_weak_max_coverage<self.continuity_good_min_coverage<=1:raise ValueError('Separated coverage bands required')
        if not 0<self.continuity_good_min_interval_fraction<=1:raise ValueError('Invalid continuous interval fraction')
        if self.continuity_weak_max_exit_m>=self.continuity_good_min_exit_m:raise ValueError('Separated exit bands required')
        if isinstance(self.continuity_boundary_cell_tolerance,bool) or not isinstance(self.continuity_boundary_cell_tolerance,int) or self.continuity_boundary_cell_tolerance<0:
            raise ValueError('Boundary-cell tolerance must be a nonnegative integer')


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


def slice_evidence(region,s_index,s,geometry,policy):
    """Independent raw witnesses; do not select by through or exact lengths."""
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
            gaps=[right[0]-left[1] for left,right in zip(parts,parts[1:])]
            witnesses.append(dict(branch_id=bid,MBG=b['MBG'],through_region=through,
                competition_arc_m=total,CC_arc_m=core,CC_fraction=core/max(total,1e-30),
                cell_coverage_fraction=coverage,in_CC=total>0 and core>=total-1e-8,
                region_cell_count=len(cells),covered_cell_count=len(item['cells']),
                continuous_interval_count=len(parts),competition_intervals=parts,
                max_interval_gap_m=max(gaps,default=0.),competition_span_m=hi-lo,
                continuous_coverage_fraction=total/max(hi-lo,1e-30),
                exit_continuation_m=min(policy.exit_margin_m,max(0.,exit_length))))
        eligible=[w for w in witnesses if w['MBG']]
        row=dict(region_id=region['region_id'],s=float(s),s_index=int(s_index),FaceTrack=int(track),present=bool(members),MBG=bool(eligible),
            member_branches=sorted(members),eligible_branches=[w['branch_id'] for w in eligible],
            witness_branch=None,through_region=False,in_CC=False,continuous_interval_count=0,
            region_cell_count=len(cells),covered_cell_count=0,competition_intervals=[],max_interval_gap_m=0.,
            competition_span_m=0.,continuous_coverage_fraction=0.,exit_continuation_m=0.,competition_arc_m=0.,
            CC_arc_m=0.,CC_fraction=0.,cell_coverage_fraction=0.,witnesses=witnesses)
        rows.append(row)
    return rows


CONTINUITY_RANK={'CONTINUITY_WEAK':0,'CONTINUITY_UNCERTAIN':1,'CONTINUITY_GOOD':2}


def continuity_classify(evidence,policy):
    """Absolute sufficient/uncertain/clearly weak evidence, never pairwise deltas.

    Multiple mask intervals on ONE observed branch are not necessarily physical
    breaks: the branch may leave/re-enter the mask. Tolerate cell-scale holes,
    or a small uncovered fraction. Never bridge distinct observed branches.
    """
    if not evidence['MBG']:return 'CONTINUITY_WEAK','MBG_FAILED_NOT_ASSESSED'
    missing=evidence['region_cell_count']-evidence['covered_cell_count']
    boundary_only=missing<=policy.continuity_boundary_cell_tolerance
    coverage=evidence['cell_coverage_fraction'];exit_m=evidence['exit_continuation_m']
    if coverage<=policy.continuity_weak_max_coverage and not boundary_only and exit_m<=policy.continuity_weak_max_exit_m:
        return 'CONTINUITY_WEAK','LOW_MASK_COVERAGE_AND_NEGLIGIBLE_EXIT'
    covered=coverage>=policy.continuity_good_min_coverage or boundary_only
    continuous=(evidence['continuous_interval_count']==1 or
                evidence['max_interval_gap_m']<=policy.cell_m*sqrt(2)*policy.continuity_gap_tolerance_cell_diagonals or
                evidence['continuous_coverage_fraction']>=policy.continuity_good_min_interval_fraction)
    extent=(exit_m>=policy.continuity_good_min_exit_m or
            evidence['competition_arc_m']>=policy.cell_m*policy.continuity_good_min_covered_arc_cells)
    if covered and continuous and extent:
        support='MEANINGFUL_EXIT' if exit_m>=policy.continuity_good_min_exit_m else 'LONG_OBSERVED_COMPETITION_ARC'
        return 'CONTINUITY_GOOD','SUFFICIENT_COVERAGE_CONTINUITY_'+support
    reasons=[]
    if not covered:reasons.append('INTERMEDIATE_MASK_COVERAGE')
    if not continuous:reasons.append('SEPARATED_MASK_INTERVALS')
    if not extent:reasons.append('LIMITED_LOCAL_EXTENT')
    return 'CONTINUITY_UNCERTAIN',';'.join(reasons)


def decide_slice(evidence_rows,policy):
    """Only MBG -> qualitative continuity -> CC -> ambiguity. No neighbor input."""
    rows=[]
    for raw in evidence_rows:
        row=dict(raw);witnesses=[]
        for original in raw['witnesses']:
            witness=dict(original)
            cls,reason=continuity_classify(witness,policy)
            witness.update(continuity_class=cls,continuity_reason=reason);witnesses.append(witness)
        eligible=[w for w in witnesses if w['MBG']]
        # Stable ID is only a representative of equivalent evidence, not a
        # FaceTrack winner tie-break. Preserve every alternative in the trace.
        witness=min(eligible,key=lambda w:(-CONTINUITY_RANK[w['continuity_class']],-int(w['in_CC']),w['branch_id'])) if eligible else None
        row.update(witnesses=witnesses,continuity_class='CONTINUITY_WEAK',continuity_reason='MBG_FAILED_NOT_ASSESSED',
                   eliminated=False,eliminated_by=None,elimination_reason=None,stage_reached='MBG')
        if witness:
            row.update({k:v for k,v in witness.items() if k not in ('branch_id','MBG')})
            row['witness_branch']=witness['branch_id']
        rows.append(row)

    def reject(row,stage,reason):
        row.update(stage_reached=stage,eliminated=True,eliminated_by=stage,elimination_reason=reason)
    for row in rows:
        if not row['present'] or not row['MBG']:reject(row,'MBG','REJECT_MBG')
    eligible=[r for r in rows if not r['eliminated']]
    for row in eligible:
        row['stage_reached']='CONTINUITY'
        if row['continuity_class']=='CONTINUITY_WEAK':reject(row,'CONTINUITY','REJECT_CONTINUITY_WEAK')
    alive=[r for r in eligible if not r['eliminated']]
    best=max((CONTINUITY_RANK[r['continuity_class']] for r in alive),default=None)
    for row in alive:
        if CONTINUITY_RANK[row['continuity_class']]<best:reject(row,'CONTINUITY','LOSE_CONTINUITY_CLASS')
    alive=[r for r in alive if not r['eliminated']]
    reason='NO_MBG_ELIGIBLE' if not eligible else 'NO_CONTINUITY_ELIGIBLE' if not alive else 'MBG_ONLY_SURVIVOR' if len(eligible)==1 else 'CONTINUITY_WIN'
    if len(alive)>=2:
        has_core=any(r['in_CC'] for r in alive)
        for row in alive:
            row['stage_reached']='CC'
            if has_core and not row['in_CC']:reject(row,'CC','LOSE_CC')
        alive=[r for r in alive if not r['eliminated']]
        reason='CC_WIN' if len(alive)==1 else 'TIE_AMBIGUOUS'
    winner=alive[0]['FaceTrack'] if len(alive)==1 else None
    for row in alive:
        row['stage_reached']='FINAL'
        row['elimination_reason']='KEEP' if winner is not None else 'TIE_AMBIGUOUS'
    for row in rows:
        row.update(slice_vote=winner,final_slice_vote=winner,slice_ambiguous=winner is None,final_decision_reason=reason,
                   slice_status='UNIQUE_SLICE_VOTE' if winner is not None else 'NO_ELIGIBLE_TRACK' if not alive else 'SLICE_AMBIGUOUS')
    return rows


def score_slice(region,s_index,s,geometry,policy):
    return decide_slice(slice_evidence(region,s_index,s,geometry,policy),policy)


def micro_difference_decisions(rows):
    """Independent audit: a selected candidate must not have an equivalent peer."""
    groups=defaultdict(list)
    for row in rows:groups[row['region_id'],row['s_index']].append(row)
    violations=[]
    for (rid,si),candidates in sorted(groups.items()):
        winner=candidates[0]['slice_vote']
        if winner is None:continue
        chosen=next(r for r in candidates if r['FaceTrack']==winner)
        peers=[r['FaceTrack'] for r in candidates if r['FaceTrack']!=winner and r['MBG'] and r['present']
               and r['continuity_class']==chosen['continuity_class'] and r['in_CC']==chosen['in_CC']]
        if peers:violations.append(dict(region_id=rid,s_index=si,s=chosen['s'],winner=winner,equivalent_peers=peers))
    return violations
