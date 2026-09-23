"""Read-only common-path projection of existing seams, not a route selector."""
from review_selected_cases import *
from main_track_assembly import _oriented


def project(point,reference):
    p=np.asarray([r['points_uz'] for r in reference]);v=p[:,1]-p[:,0];length=np.linalg.norm(v,axis=1)
    t=np.clip(np.sum((np.asarray(point)-p[:,0])*v,axis=1)/np.maximum(length**2,1e-30),0,1)
    distance=np.linalg.norm(p[:,0]+t[:,None]*v-point,axis=1);i=int(np.argmin(distance))
    arc=np.r_[0.,np.cumsum(length)][:-1]+t*length
    alternatives=(distance<=distance[i]+.002)&(abs(arc-arc[i])>.1)
    return dict(reference_arc_m=float(arc[i]),projection_distance_m=float(distance[i]),
        ambiguous=bool(np.any(alternatives)),reference_edge=reference[i]['edge_id'])


def run():
    out=output();ev=js(out/'selection_evidence.json');result=[]
    for tag,index in [('C03',-1),('C04',0),('C05',0),('C05',1)]:
        rows=[r for r in ev if r['case']==tag];ss=[r['s'] for r in rows];mid=rows[len(rows)//2]
        inv=load_inventory(SOURCE,ss);reference_bid=mid['junctions'][index]['from_branch_id']
        reference=_oriented(next(b for b in inv[mid['s']] if b['branch_id']==reference_bid)['records']);samples=[]
        for row in rows:
            j=row['junctions'][index];ends=[project(j[side+'_point_uz'],reference) for side in ('a','b')]
            # Retreat is a different quantity: each curve has its own end.
            b=next(b for b in inv[row['s']] if b['branch_id']==j['from_branch_id'])
            full=_oriented(b['records']);own=project(j['a_point_uz'],full)
            samples.append(dict(s=row['s'],z=j['a_point_uz'][1],from_branch=j['from_branch_id'],to_branch=j['to_branch_id'],ends=ends,
                own_terminal_retreat_m=float(b['full_arc_length']-own['reference_arc_m'])))
        result.append(dict(case=tag,seam_index=index,reference_s=mid['s'],reference_branch=reference_bid,samples=samples,
            z_span_m=float(np.ptp([r['z'] for r in samples])),
            reference_arc_span_m=float(np.ptp([e['reference_arc_m'] for r in samples for e in r['ends']])),
            maximum_projection_distance_m=max(e['projection_distance_m'] for r in samples for e in r['ends']),
            ambiguous=any(e['ambiguous'] for r in samples for e in r['ends']),
            purpose='READ_ONLY_DIAGNOSTIC_NOT_ACCEPTANCE_OR_ROUTING',
            method='Project corresponding transition endpoints in local UZ onto middle-profile outgoing original branch; cumulative polyline length, not elevation or independently reset endpoint retreat.'))
    save_json(out/'common_path_arc_diagnostic.json',result)
    for r in result:print({k:v for k,v in r.items() if k not in ('samples','method')})


if __name__=='__main__':run()
