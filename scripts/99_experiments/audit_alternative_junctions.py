"""Supplement final-state audit with every finite segment-junction proposal.

This checks all intersections and endpoint projections rather than the one
nearest proposal. It is not a proof over every arbitrary pair of interior
parameters on two line segments; that broader continuous feasibility remains
outside the current production junction model.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import csv,pickle,time
import numpy as np
from run_p0_p1_p2_validation import latest,SOURCE,assembly,branch_metrics,local_edge_scale,contribution_metrics,connector_gate,route_budgets,save_json
from run_face_provenance_validation import manifest
from validate_physical_topology import csv_rows


def all_junctions(left,right,required,locked,direction,bounds):
    pa,pb=assembly._points(left),assembly._points(right);da,db=np.diff(pa,axis=0),np.diff(pb,axis=0)
    la,lb=np.linalg.norm(da,axis=1),np.linalg.norm(db,axis=1);ca,cb=np.r_[0.,np.cumsum(la)],np.r_[0.,np.cumsum(lb)]
    amin,amax=np.minimum.accumulate(pa[:,1]),np.maximum.accumulate(pa[:,1]);bmin,bmax=np.minimum.accumulate(pb[::-1,1])[::-1],np.maximum.accumulate(pb[::-1,1])[::-1]
    seen=set()
    for start in range(0,len(da),64):
        end=min(start+64,len(da));p,r=pa[start:end,None],da[start:end,None];q,s=pb[None,:-1],db[None]
        den=assembly._cross(r,s);ok=abs(den)>1e-15;shape=den.shape
        ta=np.divide(assembly._cross(q-p,s),den,out=np.zeros_like(den),where=ok);tb=np.divide(assembly._cross(q-p,r),den,out=np.zeros_like(den),where=ok)
        proposals=[(ta,tb,ok&(ta>=0)&(ta<=1)&(tb>=0)&(tb<=1))]
        rr,ss=np.sum(r*r,axis=2),np.sum(s*s,axis=2)
        for fixed in [0.,1.]:
            proposals.extend([(np.full(shape,fixed),np.clip(np.sum((p+fixed*r-q)*s,axis=2)/np.maximum(ss,1e-30),0,1),np.ones(shape,bool)),
                (np.clip(np.sum((q+fixed*s-p)*r,axis=2)/np.maximum(rr,1e-30),0,1),np.full(shape,fixed),np.ones(shape,bool))])
        for ta,tb,valid in proposals:
            qa,qb=p+ta[...,None]*r,q+tb[...,None]*s;apos=ca[start:end,None]+ta*la[start:end,None];bpos=cb[None,:-1]+tb*lb[None]
            valid&=(apos>1e-9)&(cb[-1]-bpos>1e-9)&(np.linalg.norm(qa-qb,axis=2)<=.010+1e-6)
            valid&=(apos>=bounds[1]-1e-9) if direction=='upper' else (bpos<=bounds[0]+1e-9)
            lo=np.minimum(np.minimum(amin[start:end,None],qa[...,1]),np.minimum(bmin[None,1:],qb[...,1]));hi=np.maximum(np.maximum(amax[start:end,None],qa[...,1]),np.maximum(bmax[None,1:],qb[...,1]))
            lo,hi=np.minimum(lo,locked[0]),np.maximum(hi,locked[1]);valid&=(lo<=required[0]+1e-9)&(hi>=required[1]-1e-9)
            valid&=(lo<required[0]-1e-6) if direction=='lower' else (hi>required[1]+1e-6)
            for ai,bi in zip(*np.nonzero(valid)):
                i,j=int(start+ai),int(bi);a,b=float(ta[ai,bi]),float(tb[ai,bi]);key=(i,j,round(a,12),round(b,12))
                if key in seen:continue
                seen.add(key);xa,xb=np.asarray(left[i]['points_xyz']),np.asarray(right[j]['points_xyz']);xa,xb=xa[0]+a*(xa[1]-xa[0]),xb[0]+b*(xb[1]-xb[0]);gap=float(np.linalg.norm(xa-xb))
                if gap>.010+1e-12:continue
                yield dict(a_edge_index=i,b_edge_index=j,a_edge_id=left[i]['edge_id'],b_edge_id=right[j]['edge_id'],a_t=a,b_t=b,
                    a_point_uz=qa[ai,bi].tolist(),b_point_uz=qb[ai,bi].tolist(),a_point_xyz=xa.tolist(),b_point_xyz=xb.tolist(),xyz_distance_m=gap)


def run(out):
    rows=[];counterexamples=[];m=manifest(SOURCE)
    with (SOURCE/'branches.pkl').open('rb') as stream:
        for key in m['keys']:
            branches=pickle.load(stream);file=out/'routes'/f'{key}.pkl';start=time.monotonic()
            while not file.exists():
                if time.monotonic()-start>10800:raise TimeoutError(key)
                time.sleep(5)
            r=pickle.load(file.open('rb'))['result']
            if r.get('extent_covered',True):continue
            records=r['path_edges'];required=r['route_z_extent'];scale=local_edge_scale(branches);by={b['branch_id']:b for b in branches};metrics={i:branch_metrics(b,scale) for i,b in by.items()}
            pieces=assembly._remaining([b for b in branches if metrics[b['branch_id']]['MBG_pass']],records)
            for direction in ['lower','upper']:
                terminal=[]
                for e in (records if direction=='lower' else records[::-1]):
                    if not e['source'].startswith('OBSERVED') or (terminal and e['branch_id']!=terminal[0]['branch_id']):break
                    terminal.append(e)
                if direction=='upper':terminal.reverse()
                if not terminal:continue
                locked=records[len(terminal):] if direction=='lower' else records[:-len(terminal)];z=[p[1] for e in locked for p in e['points_uz']];lock=(min(z),max(z)) if z else (np.inf,-np.inf)
                old=terminal[0]['branch_id'];bounds=assembly._terminal_core_bounds(terminal,by[old],metrics[old]);core_before=contribution_metrics(by[old],records,metrics[old])['retained_ASC_arc_length']
                for n,piece in enumerate(pieces):
                    points=assembly._points(piece);lo,hi=points[:,1].min(),points[:,1].max();bid=piece[0]['branch_id']
                    if (lo>=required[0]-1e-6 if direction=='lower' else hi<=required[1]+1e-6):continue
                    gap=max(0.,required[0]-hi) if direction=='lower' else max(0.,lo-required[1])
                    if gap>.010000000001 or not contribution_metrics(by[bid],piece,metrics[bid])['contribution_pass']:continue
                    target=(lo,required[0]) if direction=='lower' else (required[1],hi)
                    if branch_metrics(by[bid],scale,target_z=target)['target_region_in_ASC_fraction']<=0:continue
                    left,right=(piece,terminal) if direction=='lower' else (terminal,piece);tested=0;legal=[]
                    for junction in all_junctions(left,right,required,lock,direction,bounds):
                        tested+=1;extended=assembly._splice(left,right,junction);sep=next(i for i,e in enumerate(extended) if e['source']=='TOPOLOGY_SWITCH');contrib=contribution_metrics(by[bid],extended[:sep] if direction=='lower' else extended[sep+1:],metrics[bid]);new=extended+locked if direction=='lower' else locked+extended
                        used={e['branch_id'] for e in new if e['source'].startswith('OBSERVED')}
                        good=contrib['contribution_pass'] and connector_gate(junction['xyz_distance_m'],contrib['observed_new_length_m'])['accepted'] and contribution_metrics(by[old],new,metrics[old])['retained_ASC_arc_length']+1e-7>=core_before and all(contribution_metrics(by[i],new,metrics[i])['contribution_pass'] for i in used) and all(v['accepted'] for v in route_budgets(new,by,metrics))
                        if good:legal.append(junction)
                    rows.append(dict(s=key,direction=direction,candidate=bid,piece_index=n,finite_junctions_tested=tested,legal_junctions=len(legal)))
                    if legal:counterexamples.append(dict(s=key,direction=direction,candidate=bid,piece_index=n,junctions=legal))
            if int(round(float(key)*100))%1000==0:print('ALL_JUNCTIONS_PROGRESS',key,flush=True)
    csv_rows(out/'alternative_junction_audit.csv',rows);save_json(out/'alternative_junction_counterexamples.json',counterexamples)
    summary=dict(candidate_pieces=len(rows),finite_junctions_tested=sum(r['finite_junctions_tested'] for r in rows),legal_stopped_slices=len({r['s'] for r in counterexamples}),legal_stopped_pieces=len(counterexamples),
        scope='All declared finite UZ segment intersections and four endpoint projections; full static terminal ASC and existing gates. Not exhaustive over arbitrary continuous interior parameter pairs.',exhaustive_continuous_feasibility_proven=False)
    save_json(out/'alternative_junction_summary.json',summary);print(summary,flush=True)


if __name__=='__main__':run(latest())
