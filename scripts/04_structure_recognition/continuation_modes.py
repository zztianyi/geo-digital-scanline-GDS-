"""Unique semantic continuation is independent of competitive-core ranking."""
import numpy as np
from branch_absolute_core import BranchPolicy


def decision_scope(rows):
    ids=sorted({r['branch_id'] for r in rows if r.get('MBG_pass',True)})
    return dict(decision_mode='COMPETITIVE_SELECTION' if len(ids)>=2 else 'NONCOMPETITIVE_CONTINUATION',
        competitor_count=len(ids),competing_branch_ids=ids,
        competing_FaceTracks=sorted({r['FaceTrack'] for r in rows if r.get('FaceTrack')}),
        competitive_core_invoked=len(ids)>=2)


def noncompetitive_junction(terminal,candidate,*,required,locked_range,direction,
                           same_chain,competitor_count,policy=None,cache=None):
    from junction_geometry import search
    policy=policy or BranchPolicy()
    audit=dict(decision_mode='NONCOMPETITIVE_CONTINUATION',competitor_count=competitor_count,
        competitive_core_invoked=False,CC_NONCOMPETITIVE_INVOCATIONS=0,junction_reason='UNRESOLVED')
    if competitor_count!=1 or not same_chain:return None,audit
    left,right=(candidate,terminal) if direction=='lower' else (terminal,candidate)
    call=cache.search if cache is not None else search
    j=call(left,right,required,locked_range,direction,protect_folds=False,
           max_distance=policy.noncompetitive_gap_cap_m,decision_mode='NONCOMPETITIVE_CONTINUATION')
    if j is None:return None,audit
    audit.update(junction_reason='MODEL_GAP_REPAIR',distance_m=j['xyz_distance_m'])
    j.update(audit,handoff_kind='MODEL_GAP_REPAIR',source='TOPOLOGY_SWITCH',synthetic=True,
             distance_cap_m=policy.noncompetitive_gap_cap_m)
    return j,audit
