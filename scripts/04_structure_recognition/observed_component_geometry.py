"""Exact observed segment geometry shared by production component roles."""
import numpy as np
from dominant_observed_branch import trim_record

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


def _neck_corridor(outer,inner,branch,max_gap_m):
    """Prove paired original polylines stay close over their entire neck.

    Normalized arc pairing is piecewise linear. Distance is convex on each
    combined interval, so checking all original breakpoints bounds its maximum.
    """
    if branch is None:return False
    arc=np.asarray(branch['arc_positions']);xyz=np.asarray([r['points_xyz'][0] for r in branch['records']]+[branch['records'][-1]['points_xyz'][1]])
    a,b=outer['start_arc'],inner['start_arc'];c,d=outer['end_arc'],inner['end_arc']
    if b<a-1e-9 or d>c+1e-9:return False
    knots=[0.,1.]
    if b-a>1e-12:knots.extend((arc[(arc>a)&(arc<b)]-a)/(b-a))
    if c-d>1e-12:knots.extend((c-arc[(arc>d)&(arc<c)])/(c-d))
    t=np.unique(knots);left=a+t*(b-a);right=c+t*(d-c)
    x=np.column_stack([np.interp(left,arc,xyz[:,k]) for k in range(3)])
    y=np.column_stack([np.interp(right,arc,xyz[:,k]) for k in range(3)])
    return bool(np.all(np.linalg.norm(x-y,axis=1)<=max_gap_m+1e-12))


def near_return_families(events,*,branch=None,max_gap_m=.01):
    """Connected pairs along two opposing path runs; gates are actual pairs.

    Nested returns are linked only through adjacent source-edge pairs, avoiding
    merging an unrelated inner loop merely because its arc interval overlaps.
    """
    events=list(events);parent=list(range(len(events)));index={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,e in enumerate(events):
        a,b=e['a_index'],e['b_index']
        for x in range(a-1,a+2):
            for y in range(b-1,b+2):
                for j in index.get((x,y),[]):
                    q=events[j]
                    outer,inner=(e,q) if e['observed_arc_m']>=q['observed_arc_m'] else (q,e)
                    if _neck_corridor(outer,inner,branch,max_gap_m):parent[root(i)]=root(j)
        index.setdefault((a,b),[]).append(i)
    # Legal pairs may be separated by triangulation vertices whose nearest
    # pair is just outside the hard gap. Join nested pairs only when both
    # sides advance comparably along the same neck, in the same XYZ direction.
    # This family test never relaxes either gate's 10 mm / 0.5 m admission.
    for i,e in enumerate(events):
        for j,q in enumerate(events[:i]):
            outer,inner=(e,q) if e['observed_arc_m']>=q['observed_arc_m'] else (q,e)
            a=inner['start_arc']-outer['start_arc'];b=outer['end_arc']-inner['end_arc']
            if min(a,b)<-1e-9 or max(a,b)<=1e-9 or min(a,b)<.75*max(a,b):continue
            if all(k in outer and k in inner for k in ('a_xyz','b_xyz')):
                da=np.asarray(inner['a_xyz'])-outer['a_xyz'];db=np.asarray(inner['b_xyz'])-outer['b_xyz']
                den=np.linalg.norm(da)*np.linalg.norm(db)
                if den>1e-15 and np.dot(da,db)/den<.9:continue
            if _neck_corridor(outer,inner,branch,max_gap_m):parent[root(i)]=root(j)
    groups={}
    for i,e in enumerate(events):groups.setdefault(root(i),[]).append(e)
    families=[]
    for pairs in groups.values():
        outer=min(pairs,key=lambda e:(-e['observed_arc_m'],e['start_arc'],e['D_xyz_m']))
        nested=[e for e in pairs if e is outer or _neck_corridor(outer,e,branch,max_gap_m)]
        inner=min(nested,key=lambda e:(e['observed_arc_m'],e['D_xyz_m']))
        families.append(dict(outer=outer,inner=inner,pairs=pairs))
    return sorted(families,key=lambda f:(-f['outer']['observed_arc_m'],f['outer']['start_arc']))


