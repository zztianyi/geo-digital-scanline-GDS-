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


