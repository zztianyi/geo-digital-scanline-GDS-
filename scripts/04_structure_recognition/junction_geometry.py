"""Exact original segment candidates with a conservative XYZ AABB broad phase."""
from copy import deepcopy
from collections import Counter
import numpy as np


class JunctionCache:
    def __init__(self):self.values={};self.stats=Counter()

    def search(self,left,right,required,locked_range,direction,**options):
        def signature(records):
            return tuple((r['branch_id'],r['edge_id'],r.get('t0',0.),r.get('t1',1.),
                          tuple(np.asarray(r['points_xyz']).ravel())) for r in records)
        key=(signature(left),signature(right),tuple(required),tuple(locked_range),direction,
             tuple(sorted((k,tuple(v) if isinstance(v,(list,np.ndarray)) else v) for k,v in options.items())))
        self.stats['queries']+=1
        if key in self.values:
            self.stats['memo_hits']+=1;return deepcopy(self.values[key])
        result=search(left,right,required,locked_range,direction,stats=self.stats,**options)
        self.values[key]=deepcopy(result);return result


def search(left,right,required,locked_range,direction,*,transition_arcs=None,protect_folds=True,
           preserve_extent=True,terminal_core_bounds=None,prefer_early=False,
           allow_clipped_boundaries=False,junction_z_bounds=None,max_distance=None,
           all_candidates=False,decision_mode=None,stats=None):
    if not left or not right:return [] if all_candidates else None
    eps=1e-9
    pa=np.asarray([left[0]['points_uz'][0]]+[r['points_uz'][1] for r in left])
    pb=np.asarray([right[0]['points_uz'][0]]+[r['points_uz'][1] for r in right])
    xa=np.asarray([r['points_xyz'] for r in left]);xb=np.asarray([r['points_xyz'] for r in right])
    da=np.diff(pa,axis=0);db=np.diff(pb,axis=0);la=np.linalg.norm(da,axis=1);lb=np.linalg.norm(db,axis=1)
    ca=np.r_[0.,np.cumsum(la)];cb=np.r_[0.,np.cumsum(lb)]
    folds_a=np.flatnonzero(da[:,1]<-eps);folds_b=np.flatnonzero(db[:,1]<-eps)
    protected_a=ca[folds_a[-1]+1] if protect_folds and direction=='upper' and len(folds_a) else 0.
    protected_b=cb[folds_b[0]] if protect_folds and direction=='lower' and len(folds_b) else cb[-1]
    amin=np.minimum.accumulate(pa[:,1]);amax=np.maximum.accumulate(pa[:,1])
    bmin=np.minimum.accumulate(pb[::-1,1])[::-1];bmax=np.maximum.accumulate(pb[::-1,1])[::-1]
    lowa=xa.min(axis=1);higha=xa.max(axis=1);lowb=xb.min(axis=1);highb=xb.max(axis=1)
    best=None;found={}
    if stats is not None:stats['edge_pairs_before']+=len(left)*len(right)
    for start in range(0,len(left),64):
        end=min(start+64,len(left))
        bound=max_distance if max_distance is not None else (max(best[0][0],eps) if best else np.inf)
        delta=np.maximum(0.,np.maximum(lowa[start:end,None,:]-highb[None,:,:],lowb[None,:,:]-higha[start:end,None,:]))
        ii,jj=np.nonzero(np.sum(delta*delta,axis=2)<=(bound+eps)**2);ii=ii+start
        if not len(ii):continue
        if stats is not None:stats['edge_pairs_exact']+=len(ii)
        xp=xa[ii,0];xq=xb[jj,0];xr=xa[ii,1]-xp;xs=xb[jj,1]-xq;w=xp-xq
        rr=np.sum(xr*xr,axis=1);ss=np.sum(xs*xs,axis=1);rs=np.sum(xr*xs,axis=1)
        rw=np.sum(xr*w,axis=1);sw=np.sum(xs*w,axis=1);den=rr*ss-rs*rs
        ok=abs(den)>np.maximum(rr*ss*1e-14,1e-30)
        ta=np.divide(rs*sw-ss*rw,den,out=np.zeros_like(den),where=ok)
        tb=np.divide(rr*sw-rs*rw,den,out=np.zeros_like(den),where=ok)
        candidates=[(ta,tb,ok&(ta>=0)&(ta<=1)&(tb>=0)&(tb<=1))]
        for fixed in (0.,1.):
            on_b=np.clip(np.sum((xp+fixed*xr-xq)*xs,axis=1)/np.maximum(ss,1e-30),0,1)
            on_a=np.clip(np.sum((xq+fixed*xs-xp)*xr,axis=1)/np.maximum(rr,1e-30),0,1)
            candidates.extend([(np.full(len(ii),fixed),on_b,np.ones(len(ii),bool)),(on_a,np.full(len(ii),fixed),np.ones(len(ii),bool))])
        for ta,tb,valid in candidates:
            qa=pa[ii]+ta[:,None]*da[ii];qb=pb[jj]+tb[:,None]*db[jj]
            apos=ca[ii]+ta*la[ii];bpos=cb[jj]+tb*lb[jj]
            if not allow_clipped_boundaries:valid &= (apos>eps)&(cb[-1]-bpos>eps)
            valid &= (apos>=protected_a-eps)&(bpos<=protected_b+eps)
            if junction_z_bounds is not None:
                valid &= (qa[:,1]>=junction_z_bounds[0]-eps)&(qa[:,1]<=junction_z_bounds[1]+eps)&(qb[:,1]>=junction_z_bounds[0]-eps)&(qb[:,1]<=junction_z_bounds[1]+eps)
            if transition_arcs is not None:
                a,b=transition_arcs;valid &= (apos>=a[0]-eps)&(apos<=a[1]+eps)&(bpos>=b[0]-eps)&(bpos<=b[1]+eps)
            lo=np.minimum(np.minimum(amin[ii],qa[:,1]),np.minimum(bmin[jj+1],qb[:,1]))
            hi=np.maximum(np.maximum(amax[ii],qa[:,1]),np.maximum(bmax[jj+1],qb[:,1]))
            lo=np.minimum(lo,locked_range[0]);hi=np.maximum(hi,locked_range[1])
            if preserve_extent:
                valid &= (lo<=required[0]+eps)&(hi>=required[1]-eps)
                valid &= lo<required[0]-1e-6 if direction=='lower' else hi>required[1]+1e-6
            distance=np.linalg.norm(xp+ta[:,None]*xr-xq-tb[:,None]*xs,axis=1)
            if max_distance is not None:valid &= distance<=max_distance+1e-12
            ids=np.flatnonzero(valid)
            if not len(ids):continue
            dk=np.where(distance<eps,0.,distance);removed=bpos if direction=='lower' else ca[-1]-apos
            timing=-removed if prefer_early else removed
            if not all_candidates:ids=[ids[np.lexsort((jj[ids],ii[ids],timing[ids],dk[ids]))[0]]]
            for k in ids:
                rank=(float(dk[k]),float(timing[k]),int(ii[k]),int(jj[k]))
                row=(rank,int(ii[k]),int(jj[k]),float(ta[k]),float(tb[k]),qa[k],qb[k])
                if all_candidates:
                    # Identical original contact coordinates can arise from adjacent edges.
                    key=tuple(np.round(np.r_[xp[k]+ta[k]*xr[k],xq[k]+tb[k]*xs[k]],10))
                    if key not in found or rank<found[key][0]:found[key]=row
                if best is None or rank<best[0]:best=row
    def output(row):
        rank,i,j,t,u,a,b=row;x=xa[i,0]+t*(xa[i,1]-xa[i,0]);y=xb[j,0]+u*(xb[j,1]-xb[j,0])
        norm=la[i]*lb[j];turn=float(np.degrees(np.arccos(np.clip(np.dot(da[i],db[j])/norm,-1,1)))) if norm else None
        return dict(junction_type='REAL_INTERSECTION' if rank[0]==0 else 'CLOSEST_POINT_PAIR',
            a_edge_index=i,b_edge_index=j,a_edge_id=left[i]['edge_id'],b_edge_id=right[j]['edge_id'],
            a_t=t,b_t=u,a_point_uz=a.tolist(),b_point_uz=b.tolist(),a_point_xyz=x.tolist(),b_point_xyz=y.tolist(),
            distance_m=float(np.linalg.norm(a-b)),xyz_distance_m=float(np.linalg.norm(x-y)),
            new_virtual_nodes=int(eps<t<1-eps)+int(eps<u<1-eps),tangent_turn_deg=turn,frontier=direction,
            removed_terminal_length_m=abs(rank[1]),from_branch_id=left[i]['branch_id'],to_branch_id=right[j]['branch_id'])
    return [output(x) for x in sorted(found.values(),key=lambda x:x[0])] if all_candidates else (output(best) if best else None)
