"""Local physical FaceTracks from the slicing mesh's shared-edge CSR graph.

Triangle coordinates are never welded. IDs are scoped to a local conflict
window; separate windows' labels must not be compared as global identities.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.sparse.csgraph import connected_components


class PhysicalFaceContext:
    def __init__(self,adjacency,lower,upper,profiles,regions=(),*,geometry_cache=None,levels=()):
        self.adjacency=adjacency;self.lower=lower;self.upper=upper
        self.profiles=profiles;self.regions=list(regions);self.cache={};self.used={}
        from junction_geometry import JunctionCache
        self.junction_cache=JunctionCache();self.geometry_cache={} if geometry_cache is None else geometry_cache
        self.stats=Counter();self.region_decisions={};self.levels=list(levels);self.scope_audit=[]
        self.edge_boxes={}
        ss=list(profiles)
        self.s_faces=np.flatnonzero((upper[:,0]>=min(ss)-.2)&(lower[:,0]<=max(ss)+.2)) if ss else np.arange(len(lower))

    def region_faces(self,region):
        key=tuple(region['bounds']);self.stats['face_cache_queries']+=1
        if key in self.geometry_cache:
            self.stats['face_cache_hits']+=1;return self.geometry_cache[key]
        b=region['bounds'];lo=np.array([b[0],b[2],b[4]]);hi=np.array([b[1],b[3],b[5]])
        # Mesh cache is independent of the current V batch: a halo may not clip
        # the faces of a merged region. Cache scope belongs to this mesh/run.
        ids=np.flatnonzero((self.upper[:,0]>=lo[0])&(self.lower[:,0]<=hi[0]))
        ids=ids[np.all(self.upper[ids]>=lo,axis=1)&np.all(self.lower[ids]<=hi,axis=1)]
        _,labels=connected_components(self.adjacency[ids][:,ids],directed=False)
        result=dict(zip(ids.tolist(),labels.tolist()));self.geometry_cache[key]=result
        self.stats['connected_components_calls']+=1;return result

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
        self.stats['region_cache_queries']+=1
        if rid in self.cache:self.stats['region_cache_hits']+=1;return self.cache[rid]
        b=region['bounds'];lo=np.array([b[0],b[2],b[4]]);hi=np.array([b[1],b[3],b[5]])
        face_labels=self.region_faces(region);members={};branches={};edge_tracks={}
        for s,bs in self.profiles.items():
            if not b[0]-1e-8<=s<=b[1]+1e-8:continue
            for branch in bs:
                if branch['z_range'][1]<b[4] or branch['z_range'][0]>b[5] or branch['u_range'][1]<b[2] or branch['u_range'][0]>b[3]:continue
                key=(s,branch['branch_id'])
                if key not in self.edge_boxes:
                    points=np.asarray([e['points_uz'] for e in branch['records']])
                    self.edge_boxes[key]=(points.min(axis=1),points.max(axis=1))
                low,high=self.edge_boxes[key]
                indices=np.flatnonzero((high[:,0]>=b[2])&(low[:,0]<=b[3])&(high[:,1]>=b[4])&(low[:,1]<=b[5]))
                found=set()
                for index in indices:
                    e=branch['records'][index]
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
