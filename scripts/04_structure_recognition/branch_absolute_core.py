"""Absolute canonical-node guards shared by selection and path assembly."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BranchPolicy:
    K_guard: int = 4
    N_core_min: int = 3
    core_length_multiplier: float = 4.
    connector_cap_m: float = .010
    synthetic_ratio_limit: float = 1.
    core_tie_tolerance: float = .05
    detail_tie_tolerance: float = .05

    def __post_init__(self):
        if (self.K_guard < 1 or self.N_core_min < 2 or self.core_length_multiplier <= 0
                or self.connector_cap_m <= 0 or not 0 < self.synthetic_ratio_limit <= 1):
            raise ValueError('Invalid absolute branch policy')


def local_edge_scale(branches):
    edges = [np.diff(b['arc_positions']) for b in branches if b['kind'] != 'CLOSED_COMPONENT']
    values = np.concatenate(edges) if edges else np.array([])
    values = values[values > 1e-6]
    return float(np.median(values)) if len(values) else 1e-6


def region_arc_intervals(branch, target_z=None):
    arc = np.asarray(branch['arc_positions'])
    if target_z is None:
        return np.array([[0.,arc[-1]]])
    z = np.asarray(branch['points_uz'])[:,1]
    lo,hi = sorted(target_z)
    dz = np.diff(z)
    flat = abs(dz) < 1e-12
    ta = np.divide(lo-z[:-1],dz,out=np.zeros_like(dz),where=~flat)
    tb = np.divide(hi-z[:-1],dz,out=np.ones_like(dz),where=~flat)
    t0,t1 = np.maximum(0,np.minimum(ta,tb)),np.minimum(1,np.maximum(ta,tb))
    keep = (t1>t0) & (~flat | ((z[:-1]>=lo)&(z[:-1]<=hi)))
    length=np.diff(arc)
    return np.column_stack((arc[:-1]+t0*length,arc[:-1]+t1*length))[keep]


def branch_metrics(branch, edge_scale, policy=None, *, target_z=None):
    policy=policy or BranchPolicy()
    arc=np.asarray(branch['arc_positions'],dtype=float)
    n=len(arc)-1; length=float(arc[-1]); k=policy.K_guard
    enough=branch['canonical_node_count'] > 2*k+policy.N_core_min and n>=2*k
    left,right=(float(arc[k]),float(arc[n-k])) if n>=2*k else (length,length)
    core_length=max(0.,right-left) if enough else 0.
    minimum=policy.core_length_multiplier*edge_scale
    exists=enough and core_length+1e-12>=minimum
    intervals=region_arc_intervals(branch,target_z)
    region_length=float(np.diff(intervals,axis=1).sum()) if len(intervals) else 0.
    overlap=float(np.maximum(0,np.minimum(intervals[:,1],right)-np.maximum(intervals[:,0],left)).sum()) if exists and len(intervals) else 0.
    boundary=intervals.ravel()
    depths=np.minimum(boundary,length-boundary)
    # Fractional positions retain their true distance to canonical endpoints;
    # no resampling or duplicate source segment can create extra guard nodes.
    node_positions=np.interp(boundary,arc,np.arange(n+1))
    return dict(branch_id=branch['branch_id'],canonical_node_count=branch['canonical_node_count'],
        canonical_edge_count=branch['canonical_edge_count'],full_arc_length=length,K_guard=k,
        left_guard_node=min(k,n),right_guard_node=max(0,n-k),
        left_guard_arc_length=left,right_guard_arc_length=length-right,
        ASC_exists=bool(exists),ASC_node_count=n-2*k+1 if enough else 0,
        ASC_arc_length=core_length,ASC_start_arc=left,ASC_end_arc=right,
        minimum_core_arc_length=minimum,local_median_edge_length=edge_scale,
        target_region_in_ASC_fraction=overlap/region_length if region_length else 0.,
        target_region_ASC_arc_length=overlap,target_region_observed_arc_length=region_length,
        target_region_distance_to_nearest_endpoint_nodes=float(np.minimum(node_positions,n-node_positions).min()) if len(boundary) else None,
        target_region_distance_to_nearest_endpoint_arc=float(depths.min()) if len(depths) else None,
        MBG_pass=bool(exists and branch['kind']=='OPEN_SURFACE_BRANCH'),
        reason='PASS_MINIMUM_BRANCH_GATE' if exists and branch['kind']=='OPEN_SURFACE_BRANCH' else 'REJECT_SHORT_BRANCH')


def contribution_metrics(branch, records, metrics):
    """Measure the actually retained part in original canonical arc coordinates."""
    positions={int(eid):i for i,eid in enumerate(branch['edge_order'])}
    arc=np.asarray(branch['arc_positions']); total=core=0.
    for r in records:
        if not r['source'].startswith('OBSERVED') or r['branch_id']!=branch['branch_id']:
            continue
        i=positions[r['edge_id']]
        a,b=sorted(arc[i]+np.asarray([r['t0'],r['t1']])*(arc[i+1]-arc[i]))
        total+=b-a
        core+=max(0.,min(b,metrics['ASC_end_arc'])-max(a,metrics['ASC_start_arc']))
    return dict(observed_new_length_m=float(total),retained_ASC_arc_length=float(core),
        contribution_pass=bool(metrics['MBG_pass'] and core+1e-12>=metrics['minimum_core_arc_length']))


def directional_branch_metrics(branch, arc_position, direction, *, edge_scale,
                               z_direction=None, frontier_z=None, support=None,
                               alternative=None, policy=None):
    """Remaining support along the original polyline, never accumulated |dZ|.

    E_progress is diagnostic. A fold becomes a tail only with both depleted
    directional coverage and stronger continuing surface evidence. Static ASC
    and the branch's coordinates are untouched.
    """
    if direction not in (-1, 1) or z_direction not in (None, -1, 1):
        raise ValueError('Directions must be +1 or -1')
    arc = np.asarray(branch['arc_positions'], dtype=float)
    z = np.asarray(branch['points_uz'])[:, 1]
    pos = float(np.clip(arc_position, 0., arc[-1]))
    current_z = float(np.interp(pos, arc, z))
    forward = arc > pos+1e-9 if direction == 1 else arc < pos-1e-9
    past = arc <= pos if direction == 1 else arc >= pos
    if z_direction is None:
        z_direction = 1 if (z[-1]-z[0])*direction >= 0 else -1
    if frontier_z is None:
        frontier_z = z_direction*max(z_direction*current_z, float(np.max(z_direction*z[past]))) if past.any() else current_z
    length = float(arc[-1]-pos if direction == 1 else pos)
    new_z = max(0., float(np.max(z_direction*(z[forward]-frontier_z)))) if forward.any() else 0.
    m = branch_metrics(branch, edge_scale, policy)
    lo, hi = (pos, arc[-1]) if direction == 1 else (0., pos)
    core = max(0., min(hi, m['ASC_end_arc'])-max(lo, m['ASC_start_arc'])) if m['ASC_exists'] else 0.
    support = support or {}
    alternative = alternative or {}
    span = float(support.get('support_s_span', 0.))
    depleted = new_z < .5*alternative.get('Z_forward_new', 0.)
    stronger = alternative.get('support_s_span', 0.) > max(span*1.25, span+.025)
    state = ('DIRECTIONAL_TAIL' if depleted and stronger else
             'DIRECTIONAL_EDGE' if core < m['minimum_core_arc_length'] or new_z <= edge_scale else 'RELIABLE_CORE')
    return dict(N_forward=int(forward.sum()), L_forward_arc=length, Z_forward_new=new_z,
        E_progress=new_z/length if length else 0., L_forward_ASC=core,
        directional_state=state, arc_position=pos, direction=direction, z_direction=z_direction,
        forward_support_slice_count=support.get('support_slice_count', 0),
        forward_support_s_span=span, forward_support_z_span=support.get('support_z_span', 0.),
        forward_same_H_run_levels=support.get('same_H_run_level_count', 0))


def connector_gate(synthetic_length, observed_new_length, policy=None):
    policy=policy or BranchPolicy()
    ratio=synthetic_length/observed_new_length if observed_new_length>0 else float('inf')
    reason=('LONG_GAP_UNRESOLVED' if synthetic_length>policy.connector_cap_m+1e-12 else
            'REJECT_SYNTHETIC_DOMINANCE' if ratio>=policy.synthetic_ratio_limit else 'ACCEPT_CONNECTOR')
    return dict(accepted=reason=='ACCEPT_CONNECTOR',reason=reason,R_syn=float(ratio),
                L_syn=float(synthetic_length),L_obs_new=float(observed_new_length))


def route_budgets(records, branches, metrics, policy=None):
    """Charge both adjacent connectors to each contiguous observed contribution."""
    groups=[]; pending=0.
    for record in records:
        if not record['source'].startswith('OBSERVED'):
            gap=float(np.linalg.norm(np.diff(record['points_xyz'],axis=0)))
            if groups: groups[-1]['outgoing']+=gap
            pending+=gap
            continue
        if not groups or pending or groups[-1]['branch_id']!=record['branch_id']:
            groups.append(dict(branch_id=record['branch_id'],records=[],incoming=pending,outgoing=0.))
            pending=0.
        groups[-1]['records'].append(record)
    budgets=[]
    for i,group in enumerate(groups):
        bid=group['branch_id']
        contribution=contribution_metrics(branches[bid],group['records'],metrics[bid])
        gate=connector_gate(group['incoming']+group['outgoing'],contribution['observed_new_length_m'],policy)
        # The hard cap applies to each connector, not the sum of two legal
        # connectors. The ratio, however, must include both of them.
        ratio_ok=gate['R_syn']<(policy or BranchPolicy()).synthetic_ratio_limit
        core_ok=contribution['contribution_pass'] if 0<i<len(groups)-1 else True
        budgets.append(dict(branch_id=bid,**contribution,incident_synthetic_m=group['incoming']+group['outgoing'],
            combined_R_syn=gate['R_syn'],accepted=bool(ratio_ok and core_ok),
            reason='REJECT_SYNTHETIC_DOMINANCE' if not ratio_ok else 'REJECT_SHORT_DETOUR' if not core_ok else 'PASS_ROUTE_BUDGET'))
    return budgets
