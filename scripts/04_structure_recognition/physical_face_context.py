"""Local physical FaceTracks from the slicing mesh's shared-edge CSR graph.

Triangle coordinates are never welded. IDs are scoped to a local conflict
window; separate windows' labels must not be compared as global identities.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import connected_components


class PhysicalFaceContext:
    def __init__(self,adjacency,lower,upper,profiles,regions=()):
        self.adjacency=adjacency;self.lower=lower;self.upper=upper
        self.profiles=profiles;self.regions=list(regions);self.cache={};self.used={}
        ss=list(profiles)
        self.s_faces=np.flatnonzero((upper[:,0]>=min(ss)-.2)&(lower[:,0]<=max(ss)+.2))

    def _region(self,s,branch,z):
        candidates=[r for r in self.regions if r['bounds'][0]-1e-8<=s<=r['bounds'][1]+1e-8
                    and r['bounds'][4]-1e-8<=z<=r['bounds'][5]+1e-8
                    and min(branch['u_range'][1],r['bounds'][3])>=max(branch['u_range'][0],r['bounds'][2])]
        if candidates:
            return min(candidates,key=lambda r:((r['bounds'][1]-r['bounds'][0])*
                (r['bounds'][3]-r['bounds'][2])*(r['bounds'][5]-r['bounds'][4]),r['region_id']))
        # New overlaps/tails not in the frozen conflict inventory are still
        # evaluated. Fixed spatial cells avoid a candidate-specific ROI.
        iz=int(np.floor(z/2));si=int(np.floor((s+1e-8)/.5))
        return dict(region_id=f'AUTO_S{si}_Z{iz}',bounds=[si*.5-.1,(si+1)*.5+.1,
            -np.inf,np.inf,iz*2-1.1,(iz+1)*2+1.1])

    def region_data(self,region):
        rid=region['region_id']
        if rid in self.cache:return self.cache[rid]
        b=region['bounds'];lo=np.array([b[0],b[2],b[4]]);hi=np.array([b[1],b[3],b[5]])
        ids=self.s_faces[np.all(self.upper[self.s_faces]>=lo,axis=1)&np.all(self.lower[self.s_faces]<=hi,axis=1)]
        _,labels=connected_components(self.adjacency[ids][:,ids],directed=False)
        face_labels=dict(zip(ids.tolist(),labels.tolist()));members={};branches={};edge_tracks={}
        for s,bs in self.profiles.items():
            if not b[0]-1e-8<=s<=b[1]+1e-8:continue
            for branch in bs:
                found=set()
                for e in branch['records']:
                    p=np.asarray(e['points_uz'])
                    if (p[:,0].max()<b[2] or p[:,0].min()>b[3] or p[:,1].max()<b[4] or p[:,1].min()>b[5]):continue
                    tids={face_labels[f] for f in e['source_face_ids'] if f in face_labels}
                    edge_tracks[s,e['edge_id']]=sorted(tids)
                    found.update(tids)
                    for tid in tids:members.setdefault(tid,set()).add(s)
                if found:branches[s,branch['branch_id']]=sorted(found)
        result=dict(region=region,face_labels=face_labels,members=members,branches=branches,edge_tracks=edge_tracks)
        self.cache[rid]=result;return result

    def evidence(self,s,branch,z,region=None):
        region=region or self._region(s,branch,z);data=self.region_data(region);rid=region['region_id']
        tids=data['branches'].get((s,branch['branch_id']),[])
        self.used[rid]=region
        names=[f'{rid}:T{t}' for t in tids]
        counts=[len(data['members'][t]) for t in tids]
        return dict(face_track_ids=names,FaceTrack=names[0] if len(names)==1 else None,
            face_continuity=min(counts) if counts else 0,face_identity_ambiguous=len(names)!=1,
            face_conflict_region=rid,face_evidence_kind='LOCAL_SHARED_EDGE_COMPONENT',
            face_context_slice_count=sum(region['bounds'][0]-1e-8<=x<=region['bounds'][1]+1e-8 for x in self.profiles))

    def record_identity(self,s,record):
        if not record['source'].startswith('OBSERVED'):return []
        branch=next(b for b in self.profiles[float(s)] if b['branch_id']==record['branch_id'])
        z=float(np.mean(np.asarray(record['points_uz'])[:,1]));region=self._region(float(s),branch,z)
        d=self.region_data(region)
        return [f"{region['region_id']}:T{t}" for t in d['edge_tracks'].get((float(s),record['edge_id']),[])]
