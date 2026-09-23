"""Explicitly reviewed model holes; never evidence for a rock structure.

Approvals are tied to original branch endpoints in XYZ. No distance search,
automatic nearest-branch promotion, or implicit large-gap tolerance is used.
"""
import numpy as np


def is_estimated_gap(record):
    return record.get('source') == 'MODEL_GAP_INFERRED'


def valid_estimated_gap(record):
    if not is_estimated_gap(record):return False
    approval=record.get('gap_approval',{})
    points=np.asarray(record.get('points_xyz',[]))
    expected=np.asarray(approval.get('endpoint_xyz',[]))
    return bool(approval.get('review_id') and approval.get('reason') and
        points.shape==(2,3) and expected.shape==(2,3) and
        np.allclose(points,expected,atol=1e-9,rtol=0) and
        not record.get('source_face_ids') and record.get('structure_eligible') is False)


def approved_endpoint_extension(records,pieces,branches,approvals):
    """Return one exact endpoint extension, or None without approval/matching.

    Normal measured continuation must already have failed. A branch fragment
    cut inside an original edge cannot stand in for an approved physical end.
    """
    from main_track_assembly import _oriented
    by={b['branch_id']:b for b in branches}
    for approval in approvals:
        if not approval.get('review_id') or not approval.get('reason'):
            raise ValueError('A reviewed model gap requires review_id and reason')
        direction=approval['direction']
        if direction not in ('lower','upper'):raise ValueError(direction)
        upper=direction=='upper';last=records[-1] if upper else records[0]
        current=approval['current_branch_id'];target=approval['target_branch_id']
        if not last['source'].startswith('OBSERVED') or last['branch_id']!=current:continue
        if target not in by:continue
        current_full=_oriented(by[current]['records']);target_full=_oriented(by[target]['records'])
        endpoint=current_full[-1]['points_xyz'][1] if upper else current_full[0]['points_xyz'][0]
        actual=last['points_xyz'][1 if upper else 0]
        if not np.allclose(actual,endpoint,atol=1e-9,rtol=0):continue
        for piece in pieces:
            if piece[0]['branch_id']!=target:continue
            entry=piece[0] if upper else piece[-1]
            original_entry=target_full[0] if upper else target_full[-1]
            if not np.allclose(entry['points_xyz'][0 if upper else 1],
                               original_entry['points_xyz'][0 if upper else 1],atol=1e-9,rtol=0):continue
            a,b=(last,entry) if upper else (entry,last)
            xyz=np.asarray([a['points_xyz'][1],b['points_xyz'][0]])
            expected=np.asarray(approval['endpoint_xyz'])
            if expected.shape!=(2,3) or not np.allclose(xyz,expected,atol=1e-9,rtol=0):continue
            uz=np.asarray([a['points_uz'][1],b['points_uz'][0]])
            distance=float(np.linalg.norm(xyz[1]-xyz[0]))
            connector=dict(source='MODEL_GAP_INFERRED',branch_id=None,face_id=None,
                source_face_ids=[],source_segment_indices=[],points_xyz=xyz,points_uz=uz,
                synthetic=True,structure_eligible=False,needs_manual_review=True,
                uncertainty='UNQUANTIFIED_MODEL_HOLE',gap_approval=dict(approval),
                handoff_kind='REVIEWED_MODEL_HOLE_ENDPOINTS',decision_mode='REVIEWED_MODEL_HOLE',
                distance_m=distance)
            assert valid_estimated_gap(connector)
            junction=dict(connector,from_branch_id=a['branch_id'],to_branch_id=b['branch_id'],
                a_point_xyz=xyz[0],b_point_xyz=xyz[1],a_point_uz=uz[0],b_point_uz=uz[1],
                xyz_distance_m=distance,new_virtual_nodes=0,frontier=direction,
                junction_reason='EXPLICITLY_REVIEWED_MODEL_HOLE',identity_ambiguous=False)
            joined=records+[connector]+piece if upper else piece+[connector]+records
            return joined,junction
    return None


def structural_runs(records):
    """Split at excluded estimates; consumers must not reconnect across a hole."""
    runs=[];current=[]
    for r in records:
        if r.get('structure_eligible',True) is False or is_estimated_gap(r):
            if current:runs.append(current);current=[]
        else:current.append(r)
    if current:runs.append(current)
    return runs
