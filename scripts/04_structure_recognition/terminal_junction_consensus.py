"""Terminal-inward search in exact close-distance corridors on original edges."""
import numpy as np


def _tube_intervals(p,v,q,w,cap):
    """Source t intervals whose closest point on the target segment is <= cap."""
    ww=float(w@w)
    if ww<1e-30:return []
    alpha=float((p-q)@w/ww);beta=float(v@w/ww)
    knots=[0.,1.]
    if abs(beta)>1e-15:knots.extend(x for x in [-alpha/beta,(1-alpha)/beta] if 0<x<1)
    knots=sorted(knots);result=[]
    for lo,hi in zip(knots,knots[1:]):
        mid=alpha+beta*(lo+hi)/2
        if mid<0:a=p-q;b=v
        elif mid>1:a=p-q-w;b=v
        else:a=p-q-alpha*w;b=v-beta*w
        aa=float(b@b);bb=2*float(a@b);cc=float(a@a)-cap*cap
        if aa<1e-24:
            if cc<=1e-16:result.append((lo,hi))
            continue
        disc=bb*bb-4*aa*cc
        if disc<0:continue
        root=np.sqrt(max(0.,disc));x=max(lo,(-bb-root)/(2*aa));y=min(hi,(-bb+root)/(2*aa))
        if y>=x-1e-12:result.append((max(lo,x),min(hi,y)))
    return result


def _materialize(template,t):
    g=template['corridor'];side=g['terminal_side'];other='b' if side=='a' else 'a'
    p=np.asarray(g[side+'_xyz']);q=np.asarray(g[other+'_xyz']);v=q[1]-q[0]
    x=p[0]+t*(p[1]-p[0]);u=float(np.clip((x-q[0])@v/max(float(v@v),1e-30),0,1))
    params={side:t,other:u};j={k:v for k,v in template.items() if k not in ('corridor',)}
    for key in ('a','b'):
        value=params[key];mapping=g[key+'_map'];j[key+'_t']=mapping[1]+value*(mapping[2]-mapping[1])
        for coords in ('xyz','uz'):
            points=np.asarray(g[key+'_'+coords]);j[key+'_point_'+coords]=(points[0]+value*(points[1]-points[0])).tolist()
    j['xyz_distance_m']=float(np.linalg.norm(np.asarray(j['a_point_xyz'])-j['b_point_xyz']))
    j['distance_m']=j['xyz_distance_m'];j['z']=float(j[side+'_point_uz'][1]);j['u']=float(j[side+'_point_uz'][0])
    j['terminal_retreat_m']=float(g['retreat_at_0']+t*g['retreat_slope'])
    j['new_virtual_nodes']=int(1e-10<j['a_t']<1-1e-10)+int(1e-10<j['b_t']<1-1e-10)
    j['junction_type']='REAL_INTERSECTION' if j['xyz_distance_m']<1e-9 else 'CLOSEST_POINT_PAIR'
    j['junction_reason']='TERMINAL_FIRST_2MM_CORRIDOR';j['corridor']=g
    return j


def terminal_candidates(left,right,from_branch,to_branch,ma,mb,*,direction,policy,protected_arcs=None):
    """Keep whole <=2 mm feasible intervals, including non-node closest pairs."""
    from competitive_surface_selection import _clip,local_arc_positions
    protected_arcs=protected_arcs or {};parts=[];maps=[]
    for records,branch,m,keep_prefix in [(left,from_branch,ma,True),(right,to_branch,mb,False)]:
        if not m['MBG_pass']:return []
        allowed=[m['ASC_start_arc'],m['ASC_end_arc']]
        positions=local_arc_positions(records,branch);increasing=positions[-1,1]>=positions[0,0]
        # Every surviving prefix/suffix contributes its own original arc. Clip
        # this linear constraint before searching continuous close corridors.
        minimum=m['minimum_core_arc_length']
        if keep_prefix:
            if increasing:allowed[0]=max(allowed[0],max(positions[0,0],m['ASC_start_arc'])+minimum)
            else:allowed[1]=min(allowed[1],min(positions[0,0],m['ASC_end_arc'])-minimum)
        else:
            if increasing:allowed[1]=min(allowed[1],min(positions[-1,1],m['ASC_end_arc'])-minimum)
            else:allowed[0]=max(allowed[0],max(positions[-1,1],m['ASC_start_arc'])+minimum)
        lock=protected_arcs.get(branch['branch_id'])
        if lock is not None:
            if increasing==keep_prefix:allowed[0]=max(allowed[0],lock[1])
            else:allowed[1]=min(allowed[1],lock[0])
        if allowed[1]<allowed[0]-1e-12:return []
        clipped,mapping=_clip(records,branch,allowed)
        if not clipped:return []
        parts.append(clipped);maps.append(mapping)
    a,b=parts;xa=np.asarray([r['points_xyz'] for r in a]);xb=np.asarray([r['points_xyz'] for r in b])
    terminal_side='a' if direction=='upper' else 'b';terminal=left if terminal_side=='a' else right
    arc=np.r_[0.,np.cumsum([np.linalg.norm(np.diff(r['points_xyz'],axis=0)) for r in terminal])]
    cap=policy.p1_handoff_cap_m;out=[]
    for start in range(0,len(a),64):
        delta=np.maximum(0.,np.maximum(xa[start:start+64].min(axis=1)[:,None]-xb.max(axis=1)[None],xb.min(axis=1)[None]-xa[start:start+64].max(axis=1)[:,None]))
        ii,jj=np.nonzero(np.sum(delta*delta,axis=2)<=(cap+1e-10)**2)
        for i,j in zip(ii+start,jj):
            p,q=(xa[i],xb[j]) if terminal_side=='a' else (xb[j],xa[i])
            for lo,hi in _tube_intervals(p[0],p[1]-p[0],q[0],q[1]-q[0],cap):
                ai,bi=maps[0][i],maps[1][j];tm=ai if terminal_side=='a' else bi
                pos0=arc[tm[0]]+tm[1]*(arc[tm[0]+1]-arc[tm[0]])
                slope=(tm[2]-tm[1])*(arc[tm[0]+1]-arc[tm[0]])
                if terminal_side=='a':pos0=arc[-1]-pos0;slope=-slope
                g=dict(terminal_side=terminal_side,t_range=[lo,hi],a_xyz=a[i]['points_xyz'],b_xyz=b[j]['points_xyz'],
                    a_uz=a[i]['points_uz'],b_uz=b[j]['points_uz'],a_map=ai,b_map=bi,retreat_at_0=pos0,retreat_slope=slope)
                zs=[g[terminal_side+'_uz'][0][1]+t*(g[terminal_side+'_uz'][1][1]-g[terminal_side+'_uz'][0][1]) for t in (lo,hi)]
                row=dict(corridor=g,a_edge_index=ai[0],b_edge_index=bi[0],a_edge_id=left[ai[0]]['edge_id'],b_edge_id=right[bi[0]]['edge_id'],
                    from_branch_id=from_branch['branch_id'],to_branch_id=to_branch['branch_id'],frontier=direction,
                    z_min=min(zs),z_max=max(zs),distance_cap_m=cap,from_in_ASC_at_junction=True,to_in_ASC_at_junction=True)
                row=_materialize(row,hi if slope<0 else lo)
                if row['xyz_distance_m']<=cap+1e-10:out.append(row)
    return sorted(out,key=lambda r:(r['terminal_retreat_m'],r['xyz_distance_m'],r['a_edge_id'],r['b_edge_id']))


def _in_band(candidate,low,high):
    a=max(low,candidate.get('z_min',candidate['z']));b=min(high,candidate.get('z_max',candidate['z']))
    if a>b+1e-10:return None
    if 'corridor' not in candidate:
        ends=[candidate.get(k,[0,candidate['z']])[1] for k in ('a_point_uz','b_point_uz')]
        return candidate if min(ends)>=low-1e-10 and max(ends)<=high+1e-10 else None
    g=dict(candidate['corridor']);lo,hi=g['t_range']
    first=_materialize(candidate,lo);last=_materialize(candidate,hi)
    for side in ('a','b'):
        z0=first[side+'_point_uz'][1];z1=last[side+'_point_uz'][1]
        if abs(z1-z0)<1e-14:
            if not low-1e-10<=z0<=high+1e-10:return None
            continue
        x,y=sorted((g['t_range'][0]+(z-z0)*(g['t_range'][1]-g['t_range'][0])/(z1-z0) for z in (low,high)))
        lo=max(lo,x);hi=min(hi,y)
    if lo>hi+1e-12:return None
    g['t_range']=[lo,max(lo,hi)];out=dict(candidate,corridor=g)
    return _materialize(out,hi if g['retreat_slope']<0 else lo)


def _retreat_limit(candidate,limit):
    if candidate['terminal_retreat_m']>limit+1e-9:return None
    if 'corridor' not in candidate:return candidate
    g=dict(candidate['corridor']);lo,hi=g['t_range'];slope=g['retreat_slope']
    if slope>1e-15:hi=min(hi,(limit-g['retreat_at_0'])/slope)
    elif slope<-1e-15:lo=max(lo,(limit-g['retreat_at_0'])/slope)
    if lo>hi+1e-12:return None
    g['t_range']=[lo,max(lo,hi)];result=dict(candidate,corridor=g)
    first=_materialize(result,lo);last=_materialize(result,hi)
    result.update(z_min=min(first['z'],last['z']),z_max=max(first['z'],last['z']))
    return _materialize(result,hi if slope<0 else lo)


def _window_bounds(candidate,width):
    if 'corridor' not in candidate:
        ends=[candidate.get(k,[0,candidate['z']])[1] for k in ('a_point_uz','b_point_uz')]
        return max(ends)-width,min(ends)
    lo,hi=candidate['corridor']['t_range'];points=[_materialize(candidate,t) for t in (lo,hi)]
    delta=[p['a_point_uz'][1]-p['b_point_uz'][1] for p in points]
    if delta[0]*delta[1]<0:points.append(_materialize(candidate,lo-delta[0]*(hi-lo)/(delta[1]-delta[0])))
    ends=[[p['a_point_uz'][1],p['b_point_uz'][1]] for p in points]
    return min(max(z) for z in ends)-width,max(min(z) for z in ends)


def _common_window_filter(layers,width):
    """Discard only contacts that cannot share any Z window across all V's."""
    windows=[[_window_bounds(c,width) for c in row] for row in layers]
    common=[(-float('inf'),float('inf'))]
    for row in windows:
        merged=[]
        for lo,hi in sorted(row):
            if merged and lo<=merged[-1][1]+1e-10:merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
            else:merged.append((lo,hi))
        intersection=[];i=j=0
        while i<len(common) and j<len(merged):
            a,b=common[i];c,d=merged[j];lo,hi=max(a,c),min(b,d)
            if lo<=hi+1e-10:intersection.append((lo,max(lo,hi)))
            if b<d:i+=1
            else:j+=1
        common=intersection
        if not common:return [[] for _ in layers]
    return [[c for c,(a,b) in zip(row,ww) if any(max(a,lo)<=min(b,hi)+1e-10 for lo,hi in common)]
            for row,ww in zip(layers,windows)]


def choose_terminal_band(layers,max_z_span=.1,max_distance=.002,valid=None):
    """Earliest inward band covering every V; total Z span is a hard bound."""
    layers=[[c for c in row if c['xyz_distance_m']<=max_distance+1e-10] for row in layers]
    if not layers or any(not row for row in layers):return []
    # Route checks reconstruct long observed paths. Geometric feasibility is a
    # necessary condition, so exclude impossible bands before those checks.
    # Retained contacts and all terminal-order / route constraints are unchanged.
    layers=_common_window_filter(layers,max_z_span)
    if any(not row for row in layers):return []
    if valid is not None and hasattr(valid,'breakpoints'):
        refined=[]
        for index,row in enumerate(layers):
            parts=[]
            for c in row:
                if 'corridor' not in c:
                    if valid(index,c):parts.append(c)
                    continue
                g=c['corridor'];lo,hi=g['t_range']
                if valid(index,_materialize(c,lo)) and valid(index,_materialize(c,hi)):
                    parts.append(c);continue
                # Refine before computing any Z-window anchors. Otherwise an
                # internal valid interval can be invisible to both end windows.
                cuts=sorted(set([lo,hi]+[t for t in valid.breakpoints(index,c) if lo<t<hi]))
                intervals=[(t,t) for t in cuts if valid(index,_materialize(c,t))]
                for a,b in zip(cuts,cuts[1:]):
                    mid=(a+b)/2
                    if not valid(index,_materialize(c,mid)):continue
                    ends=[]
                    for edge in (a,b):
                        if valid(index,_materialize(c,edge)):ends.append(edge);continue
                        rejected=edge;accepted=mid
                        for _ in range(32):
                            t=(accepted+rejected)/2
                            if valid(index,_materialize(c,t)):accepted=t
                            else:rejected=t
                        ends.append(accepted)
                    intervals.append(tuple(ends))
                for a,b in intervals:
                    part=dict(c,corridor=dict(g,t_range=[a,b]));first=_materialize(part,a);last=_materialize(part,b)
                    part.update(z_min=min(first['z'],last['z']),z_max=max(first['z'],last['z']))
                    parts.append(_materialize(part,b if g['retreat_slope']<0 else a))
            refined.append(parts)
        layers=refined
        if any(not row for row in layers):return []
    def feasible(limit,finish=False):
        restricted=[[p for c in row if (p:=_retreat_limit(c,limit)) is not None] for row in layers]
        if any(not row for row in restricted):return []
        windows=[[_window_bounds(c,max_z_span) for c in row] for row in restricted]
        anchors=sorted({x for row in windows for pair in row for x in pair});best=None
        for low in anchors:
            picks=[]
            for index,(row,ww) in enumerate(zip(restricted,windows)):
                options=[p for c,(a,b) in zip(row,ww) if a-1e-10<=low<=b+1e-10 and (p:=_in_band(c,low,low+max_z_span)) is not None]
                options.sort(key=lambda p:(p['terminal_retreat_m'],p['xyz_distance_m']))
                chosen=None
                for p in options:
                    if valid is None or valid(index,p):chosen=p;break
                    if 'corridor' not in p:continue
                    g=p['corridor'];lo,hi=g['t_range'];near,far=(hi,lo) if g['retreat_slope']<0 else (lo,hi)
                    # Keep searching inward after a route-level rejection. The
                    # core and minimum contribution constraints are preclipped;
                    # this handles remaining prefix/suffix feasibility bounds.
                    accepted=None
                    cuts=sorted(set([lo,hi]+(valid.breakpoints(index,p) if hasattr(valid,'breakpoints') else [])))
                    probes=cuts+[(a+b)/2 for a,b in zip(cuts,cuts[1:])]
                    probes.sort(reverse=g['retreat_slope']<0)
                    for t in probes:
                        q=_materialize(p,t)
                        if valid(index,q):accepted=t;break
                    if accepted is None:continue
                    rejected=near
                    for _ in range(28):
                        t=(accepted+rejected)/2;q=_materialize(p,t)
                        if valid(index,q):accepted=t
                        else:rejected=t
                    chosen=_materialize(p,accepted);break
                if chosen is None:break
                picks.append(chosen)
            if len(picks)!=len(layers):continue
            ends=[p.get(side+'_point_uz',[0,p['z']])[1] for p in picks for side in ('a','b')]
            if max(ends)-min(ends)>max_z_span+1e-9:continue
            if not finish:return picks
            retreat=[p['terminal_retreat_m'] for p in picks];u=[p.get('u',0.) for p in picks]
            score=(max(retreat),sum(retreat),float(np.ptp(u)),float(np.ptp(ends)),low)
            if best is None or score<best[0]:best=(score,picks)
        return best[1] if best else []
    initial=feasible(float('inf'))
    if not initial:return []
    low=max(min(c['terminal_retreat_m'] for c in row) for row in layers)
    high=max(c['terminal_retreat_m'] for c in initial)
    # Restrict the continuous corridors as the inward frontier advances. This
    # admits interior optima, including folds with opposite Z directions.
    while high-low>1e-7:
        mid=(low+high)/2;picks=feasible(mid)
        if picks:high=mid;initial=picks
        else:low=mid
    return feasible(high+1e-9,finish=True) or initial
