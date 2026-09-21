"""Shadow-only face relations on actual shared triangle edges.

No mesh welding, coordinate similarity, recognition imports or route mutation.
"""
from collections import deque
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class FaceAdjacency:
    def __init__(self,faces,adjacency,metadata=None):
        self.faces=faces
        self.adjacency=adjacency
        self.metadata=metadata or {}

    @classmethod
    def build(cls,faces):
        faces=np.asarray(faces,dtype=np.int64)
        if faces.ndim!=2 or faces.shape[1]!=3 or np.any(faces<0):
            raise ValueError('Expected nonnegative triangle vertex indices')
        edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        owner=np.repeat(np.arange(len(faces),dtype=np.int32),3)
        valid=edges[:,0]!=edges[:,1];edges=edges[valid];owner=owner[valid]
        stride=int(faces.max(initial=0))+1
        packed=edges[:,0].astype(np.uint64)*np.uint64(stride)+edges[:,1].astype(np.uint64)
        del edges
        order=np.argsort(packed,kind='stable');packed=packed[order];owner=owner[order]
        starts=np.r_[0,np.flatnonzero(packed[1:]!=packed[:-1])+1] if len(packed) else np.array([],dtype=int)
        counts=np.diff(np.r_[starts,len(packed)])
        two=starts[counts==2];a=owner[two];b=owner[two+1]
        keep=a!=b;a,b=a[keep],b[keep]
        extra=[]
        for start,count in zip(starts[counts>2],counts[counts>2]):
            extra.extend(combinations(np.unique(owner[start:start+count]).tolist(),2))
        if extra:
            e=np.asarray(extra,dtype=np.int32);a=np.r_[a,e[:,0]];b=np.r_[b,e[:,1]]
        graph=sparse.csr_matrix((np.ones(2*len(a),dtype=np.int8),(np.r_[a,b],np.r_[b,a])),shape=(len(faces),len(faces)))
        graph.data[:]=1;graph.sort_indices()
        return cls(faces,graph,dict(faces=len(faces),shared_edge_face_pairs=graph.nnz//2,
            mesh_edges_with_more_than_two_incidences=int((counts>2).sum()),
            zero_vertex_index_edges_ignored=int((~valid).sum()),identity='filtered_mesh_vertex_indices'))

    def local_labels(self,allowed):
        ids=np.array(sorted(allowed),dtype=np.int64)
        if not len(ids):return {}
        _,labels=connected_components(self.adjacency[ids][:,ids],directed=False)
        # IDs are local, deterministic by minimum source FaceID, not ID proximity.
        return dict(zip(ids.tolist(),labels.tolist()))

    @staticmethod
    def unique_track(face_ids,labels):
        values={labels[f] for f in face_ids if f in labels}
        if len(values)!=1 or any(f not in labels for f in face_ids):return None
        return next(iter(values))

    def relation(self,a,b,allowed,*,max_hops=8):
        a,b=set(map(int,a)),set(map(int,b))
        if not a or not b:
            return dict(relation='AMBIGUOUS_FACE_PROVENANCE',chain=[],weak_vertex_contact=False,reason='MISSING_SOURCE_FACE_IDS')
        if any(f<0 or f>=len(self.faces) for f in a|b):
            raise ValueError('Provenance FaceID is outside the frozen mesh ID space')
        exact=a&b
        if exact:
            return dict(relation='EXACT_FACE',chain=[min(exact)],weak_vertex_contact=False)
        parents={f:None for f in sorted(a&allowed)};queue=deque((f,0) for f in parents)
        while queue:
            f,depth=queue.popleft()
            if depth>=max_hops:continue
            for g in self.adjacency.indices[self.adjacency.indptr[f]:self.adjacency.indptr[f+1]]:
                g=int(g)
                if g not in allowed or g in parents:continue
                parents[g]=f
                if g in b:
                    chain=[g]
                    while parents[chain[-1]] is not None:chain.append(parents[chain[-1]])
                    chain.reverse()
                    return dict(relation='EDGE_ADJACENT_FACE' if len(chain)==2 else 'LOCAL_FACE_CHAIN',
                                chain=chain,weak_vertex_contact=False)
                queue.append((g,depth+1))
        weak=bool(set(self.faces[list(a)].ravel())&set(self.faces[list(b)].ravel()))
        return dict(relation='GEOMETRY_ONLY',chain=[],weak_vertex_contact=weak,
                    reason='NO_SHARED_EDGE_CHAIN_WITHIN_ROI_AND_HOP_LIMIT')
