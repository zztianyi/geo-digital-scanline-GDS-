"""Small-table reducer. Deliberately no geometry/mesh/numpy/route imports."""
from dataclasses import dataclass,asdict


@dataclass(frozen=True)
class RegionDecision:
    region_id: str
    dominant_FaceTrack: object
    supporting_V: tuple
    opposing_V: tuple
    absent_V: tuple
    abstaining_V: tuple
    ineligible_V: tuple
    ambiguous: bool
    evidence_summary: tuple
    confidence: str
    def to_dict(self):return asdict(self)


def longest_run(indices,positions):
    best=(0,0.);start=None;last=None;count=0
    for i in sorted(set(indices)):
        if last is None or i!=last+1:start=i;count=1
        else:count+=1
        best=max(best,(count,round(positions[i]-positions[start],12)));last=i
    return best


def reduce_region(region_id,rows,quantum=.001):
    rows=sorted(rows,key=lambda r:(r['s_index'],r['FaceTrack']))
    tracks=sorted({r['FaceTrack'] for r in rows});positions={r['s_index']:r['s'] for r in rows};all_s=sorted(set(positions.values()))
    summaries=[];ranks={}
    for track in tracks:
        rr=[r for r in rows if r['FaceTrack']==track];good=[r for r in rr if r['present'] and r['MBG']]
        run,span=longest_run([r['s_index'] for r in good],positions)
        through=sum(r['through_region'] for r in good);cc=sum(r['in_CC'] for r in good)
        continuation=sum(round(r['exit_continuation_m']/quantum) for r in good)
        ranks[track]=(through,run,span,cc,continuation)
        summaries.append((track,through,run,span,cc,continuation*quantum/max(1,len(all_s)),len(good)))
    maximum=max(ranks.values(),default=None);tied=[t for t in tracks if ranks[t]==maximum]
    eligible=any(r['present'] and r['MBG'] for r in rows)
    winner=tied[0] if len(tied)==1 and eligible else None
    votes={r['s']:r['slice_vote'] for r in rows}
    support=tuple(s for s in all_s if winner is not None and votes[s]==winner)
    oppose=tuple(s for s in all_s if winner is not None and votes[s] is not None and votes[s]!=winner)
    abstain=tuple(s for s in all_s if votes[s] is None)
    mine={r['s']:r for r in rows if r['FaceTrack']==winner}
    absent=tuple(s for s in all_s if winner is not None and (s not in mine or not mine[s]['present']))
    ineligible=tuple(s for s in all_s if winner is not None and s in mine and not mine[s]['MBG'])
    confidence=('REGION_NO_ELIGIBLE' if not eligible else 'REGION_AMBIGUOUS' if winner is None else
                'UNANIMOUS_SLICE_VOTES' if len(support)==len(all_s) else 'AGGREGATED_WITH_DISSENT' if oppose else 'AGGREGATED_WITH_ABSTENTIONS')
    return RegionDecision(region_id,winner,support,oppose,absent,abstain,ineligible,winner is None,tuple(summaries),confidence)
