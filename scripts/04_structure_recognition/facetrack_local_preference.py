"""B0/B1: deterministic table-only local preference decisions.

No geometry, route state, junction search or modification of A3 evidence.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict

POLICY_VERSION='local-preference-subregions-v1'


@dataclass(frozen=True)
class LocalPreferencePolicy:
    min_preference_run_V: int=3
    strong_consensus_fraction: object=.99
    continuation_quantum_m: float=.001

    def __post_init__(self):
        if isinstance(self.min_preference_run_V,bool) or not isinstance(self.min_preference_run_V,int) or self.min_preference_run_V<1:
            raise ValueError('min_preference_run_V must be a positive integer')
        if self.strong_consensus_fraction is not None and not .5<self.strong_consensus_fraction<=1:
            raise ValueError('strong_consensus_fraction must be None or in (0.5,1]')
        if not self.continuation_quantum_m>0:raise ValueError('continuation quantum must be positive')


@dataclass(frozen=True)
class PreferenceRun:
    FaceTrack: object
    s_start: float
    s_end: float
    member_V: tuple
    s_indices: tuple
    stable: bool


@dataclass(frozen=True)
class DecisionSubregion:
    parent_region_id: str
    subregion_id: str
    s_start: float
    s_end: float
    member_V: tuple
    s_indices: tuple
    competing_FaceTracks: tuple
    preference_runs: tuple
    ambiguous_buffer_V: tuple
    local_dissent_V: tuple
    anchor_preference: object
    kind: str
    boundary_start_reason: str
    origin: str

    def to_dict(self):return asdict(self)


@dataclass(frozen=True)
class SubregionDecision:
    subregion_id: str
    parent_region_id: str
    dominant_FaceTrack: object
    supporting_V: tuple
    opposing_V: tuple
    abstaining_V: tuple
    vote_counts: tuple
    longest_support_run: tuple
    through_counts: tuple
    CC_counts: tuple
    bounded_continuation: tuple
    confidence: str
    ambiguous: bool
    MAJORITY_OPPOSES_WINNER: bool

    def to_dict(self):return asdict(self)


def _observations(parent_id,rows):
    by_v=defaultdict(dict)
    for row in rows:
        if row['region_id']!=parent_id:raise ValueError('Mixed parent region input')
        si=row['s_index'];track=row['FaceTrack']
        if track in by_v[si]:raise ValueError('Duplicate (region,V,FaceTrack) row')
        by_v[si][track]=row
    result=[]
    for si,tracks in sorted(by_v.items()):
        values=list(tracks.values());first=values[0];vote=first['slice_vote'];ambiguous=first['slice_ambiguous']
        if any(r['s']!=first['s'] or r['slice_vote']!=vote or r['slice_ambiguous']!=ambiguous for r in values):
            raise ValueError('Inconsistent repeated A3 vote or position')
        if ambiguous!=(vote is None):raise ValueError('slice_ambiguous contradicts slice_vote')
        if vote is not None and (vote not in tracks or not tracks[vote]['present'] or not tracks[vote]['MBG']):
            raise ValueError('A3 vote targets an absent or ineligible candidate')
        result.append(dict(si=si,s=first['s'],vote=vote,tracks=tuple(sorted(tracks)),rows=tracks))
    if any(b['s']<=a['s'] for a,b in zip(result,result[1:])):raise ValueError('s_index and s ordering disagree')
    return result


def _runs(observations,minimum):
    groups=[]
    for v in observations:
        if groups and groups[-1][-1]['vote']==v['vote'] and v['si']==groups[-1][-1]['si']+1:groups[-1].append(v)
        else:groups.append([v])
    return [(group,PreferenceRun(group[0]['vote'],group[0]['s'],group[-1]['s'],tuple(v['s'] for v in group),
                                tuple(v['si'] for v in group),group[0]['vote'] is not None and len(group)>=minimum)) for group in groups]


def segment_region(parent_id,rows,policy=None):
    policy=policy or LocalPreferencePolicy();obs=_observations(parent_id,rows)
    blocks=[]
    for v in obs:
        reason=('PARENT_START' if not blocks else 'SOURCE_V_GAP' if v['si']!=blocks[-1][1][-1]['si']+1 else
                'COMPETING_SET_CHANGE' if v['tracks']!=blocks[-1][1][-1]['tracks'] else None)
        if reason:blocks.append((reason,[v]))
        else:blocks[-1][1].append(v)
    segments=[]

    def emit(values,kind,anchor,reason):
        segments.append((values,kind,anchor,reason))

    def explicit_block(values,reason):
        runs=_runs(values,policy.min_preference_run_V)
        stable=[(i,g,r) for i,(g,r) in enumerate(runs) if r.stable]
        count=Counter(v['vote'] for v in values);preferred,n=count.most_common(1)[0]
        if len(values)<policy.min_preference_run_V and len(values)==len(obs):
            emit(values,'INHERITED_SHORT_REGION',None,reason);return
        if policy.strong_consensus_fraction is not None and n/len(values)>=policy.strong_consensus_fraction and stable:
            emit(values,'STRONG_CONSENSUS_REGION',preferred,reason);return
        if not stable:
            emit(values,'NO_STABLE_PREFERENCE',None,reason);return
        anchor=stable[0][2].FaceTrack;current=[];current_reason=reason
        for group,run in runs:
            if run.stable and run.FaceTrack!=anchor:
                emit(current,'PREFERENCE_SUBREGION',anchor,current_reason)
                current=[];anchor=run.FaceTrack;current_reason='STABLE_PREFERENCE_REVERSAL'
            current.extend(group)
        emit(current,'PREFERENCE_SUBREGION',anchor,current_reason)

    for reason,block in blocks:
        current=[];is_ambiguous=None;current_reason=reason
        for v in block:
            ambiguous=v['vote'] is None
            if current and ambiguous!=is_ambiguous:
                if is_ambiguous:emit(current,'AMBIGUOUS_BUFFER',None,current_reason)
                else:explicit_block(current,current_reason)
                current=[];current_reason='AMBIGUOUS_BUFFER_START' if ambiguous else 'AFTER_AMBIGUOUS_BUFFER'
            current.append(v);is_ambiguous=ambiguous
        if current:
            if is_ambiguous:emit(current,'AMBIGUOUS_BUFFER',None,current_reason)
            else:explicit_block(current,current_reason)
    result=[]
    for n,(values,kind,anchor,reason) in enumerate(segments,1):
        member=tuple(v['s'] for v in values);indices=tuple(v['si'] for v in values)
        result.append(DecisionSubregion(parent_id,f'{parent_id}_S{n:03d}',member[0],member[-1],member,indices,values[0]['tracks'],
            tuple(r for _,r in _runs(values,policy.min_preference_run_V)),member if kind=='AMBIGUOUS_BUFFER' else (),
            tuple(v['s'] for v in values if anchor is not None and v['vote'] is not None and v['vote']!=anchor),anchor,kind,reason,
            'UNCHANGED_INITIAL_REGION' if len(values)==len(obs) else 'SPLIT_FROM_INITIAL_REGION'))
    assert [i for r in result for i in r.s_indices]==[v['si'] for v in obs]
    return result


def _longest_support(observations,track):
    best=(0,0.);length=0;start=None;last=None
    for v in observations:
        if v['vote']!=track:length=0;last=None;continue
        if last is None or v['si']!=last+1:start=v['s'];length=1
        else:length+=1
        best=max(best,(length,round(v['s']-start,12)));last=v['si']
    return best


def reduce_subregion(subregion,rows,policy=None):
    policy=policy or LocalPreferencePolicy();members=set(subregion.s_indices)
    obs=_observations(subregion.parent_region_id,[r for r in rows if r['s_index'] in members])
    if tuple(v['si'] for v in obs)!=subregion.s_indices:raise ValueError('Subregion evidence missing V')
    if any(v['tracks']!=subregion.competing_FaceTracks for v in obs):raise ValueError('Subregion competing set mismatch')
    votes=Counter(v['vote'] for v in obs if v['vote'] is not None);ranks={};runs=[];through=[];cores=[];exits=[];eligible=False
    for t in subregion.competing_FaceTracks:
        good=[v['rows'][t] for v in obs if v['rows'][t]['present'] and v['rows'][t]['MBG']];eligible|=bool(good)
        run,span=_longest_support(obs,t);tr=sum(r['through_region'] for r in good);cc=sum(r['in_CC'] for r in good)
        cont=sum(round(r['exit_continuation_m']/policy.continuation_quantum_m) for r in good)
        ranks[t]=(votes[t],run,span,tr,cc,cont)
        runs.append((t,run,span));through.append((t,tr));cores.append((t,cc));exits.append((t,cont*policy.continuation_quantum_m,cont*policy.continuation_quantum_m/max(1,len(obs))))
    maximum=max(ranks.values(),default=None);tied=[t for t in ranks if ranks[t]==maximum]
    winner=tied[0] if eligible and len(tied)==1 and subregion.kind not in ('AMBIGUOUS_BUFFER','NO_STABLE_PREFERENCE') else None
    support=tuple(v['s'] for v in obs if winner is not None and v['vote']==winner)
    oppose=tuple(v['s'] for v in obs if winner is not None and v['vote'] is not None and v['vote']!=winner)
    abstain=tuple(v['s'] for v in obs if v['vote'] is None)
    majority=winner is not None and len(oppose)>(len(support)+len(oppose))/2
    confidence=('REGION_NO_ELIGIBLE' if not eligible else 'NO_STABLE_PREFERENCE' if subregion.kind=='NO_STABLE_PREFERENCE' else 'REGION_AMBIGUOUS' if winner is None else
                'MAJORITY_OPPOSES_WINNER' if majority else 'LOCAL_DISSENT' if oppose else 'UNANIMOUS_SLICE_VOTES')
    return SubregionDecision(subregion.subregion_id,subregion.parent_region_id,winner,support,oppose,abstain,
        tuple((t,votes[t]) for t in subregion.competing_FaceTracks),tuple(runs),tuple(through),tuple(cores),tuple(exits),confidence,winner is None,majority)
