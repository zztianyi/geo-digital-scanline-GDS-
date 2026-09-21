"""Round6 vertical geometry with stable edge order and source provenance.

No mesh access, snapping, or inferred geometry is involved. Node IDs follow
endpoint encounter order; edge IDs and orientations follow first occurrence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class CanonicalProfile:
    nodes: np.ndarray
    edges: np.ndarray
    first_face_ids: list[int | None]
    source_face_ids: list[list[int]]
    source_segment_indices: list[list[int]]
    raw_degree: np.ndarray
    degree: np.ndarray
    components: np.ndarray
    branches: list[dict]
    stats: dict
    topology_active_edge: np.ndarray = field(init=False)
    physical_degree: np.ndarray = field(init=False)
    physical_components: np.ndarray = field(init=False)
    physical_branches: list[dict] = field(init=False)

    def __post_init__(self):
        # Keep canonical incidences/chains for provenance and legacy audits.
        # A rounded A -> A record is not a physical fork or a traversable edge.
        self.topology_active_edge = self.edges[:, 0] != self.edges[:, 1]
        if np.all(self.topology_active_edge):
            self.physical_degree = self.degree.copy()
            self.physical_components = self.components.copy()
            self.physical_branches = self.branches
        else:
            (self.physical_degree, self.physical_components,
             self.physical_branches, _) = _topology(
                len(self.nodes), self.edges, active=self.topology_active_edge)

    @property
    def canonical_degree(self) -> np.ndarray:
        return self.degree

    @property
    def lines_xyz(self) -> np.ndarray:
        return self.nodes[self.edges]

    def as_intersections(self) -> list[tuple]:
        """ArcUtils input, preserving the first matching edge and FaceID."""
        return [(tuple(self.nodes[a]), tuple(self.nodes[b]), face)
                for (a, b), face in zip(self.edges, self.first_face_ids)]


def _topology(node_count, edges, *, active=None):
    """Linear-time components and maximal chains, splitting at degree != 2."""
    adjacency = [[] for _ in range(node_count)]
    for eid, (a, b) in enumerate(edges):
        if active is not None and not active[eid]:
            continue
        a, b = int(a), int(b)
        adjacency[a].append((b, eid))
        adjacency[b].append((a, eid))
    degree = np.fromiter((len(neighbors) for neighbors in adjacency),
                         dtype=np.int64, count=node_count)
    components = np.full(node_count, -1, dtype=np.int64)
    component_count = 0
    for start in range(node_count):
        if components[start] != -1:
            continue
        components[start] = component_count
        pending = [start]
        while pending:
            node = pending.pop()
            for other, _ in adjacency[node]:
                if components[other] == -1:
                    components[other] = component_count
                    pending.append(other)
        component_count += 1

    used = np.zeros(len(edges), dtype=bool)
    branches = []

    def follow(start, next_node, edge):
        node_order, edge_order = [start], []
        while not used[edge]:
            used[edge] = True
            edge_order.append(edge)
            node_order.append(next_node)
            if degree[next_node] != 2:
                break
            choices = [(other, eid) for other, eid in adjacency[next_node] if not used[eid]]
            if not choices:
                break
            next_node, edge = choices[0]
        branches.append({'nodes': node_order, 'edges': edge_order,
                         'branch_id': len(branches), 'component_id': int(components[start])})

    # Match the traversal convention in orthogonal_profile_constraint, without
    # importing unrelated recognition code or returning ndarray branch fields.
    for node in np.flatnonzero(degree != 2):
        for other, eid in adjacency[node]:
            if not used[eid]:
                follow(int(node), other, eid)
    # Remaining edges belong to closed loops, including disconnected loops.
    for eid, (a, b) in enumerate(edges):
        if not used[eid] and (active is None or active[eid]):
            follow(int(a), int(b), eid)
    return degree, components, branches, component_count


def canonicalize_vertical(
    lines_xyz: np.ndarray, face_ids: Sequence[int | None],
) -> CanonicalProfile:
    """Deduplicate undirected round6 edges, retaining ordered provenance.

    Inputs are never changed. Each unique zero-length rounded edge is retained
    with all source provenance and contributes two canonical node incidences,
    but no physical incidence. Physical chains retain original edge IDs.
    collapsed_self_edges counts raw segments whose rounded endpoints coincide;
    the raw-minus-canonical duplicate statistic counts only repeated edges.
    No tolerance beyond round6 is applied.
    """
    source = np.asarray(lines_xyz)
    if source.ndim != 3 or source.shape[1:] != (2, 3):
        raise ValueError('lines_xyz must have shape (N, 2, 3)')
    if np.iscomplexobj(source):
        raise ValueError('lines_xyz coordinates must be finite real numbers')
    try:
        lines = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('lines_xyz coordinates must be finite real numbers') from exc
    if not np.isfinite(lines).all():
        raise ValueError('lines_xyz coordinates must be finite')
    try:
        faces = list(face_ids)
    except TypeError as exc:
        raise ValueError('face_ids must be a sequence with length N') from exc
    if len(faces) != len(lines):
        raise ValueError('face_ids length must match lines_xyz segment count')
    try:
        faces = [None if face is None else operator.index(face) for face in faces]
    except TypeError as exc:
        raise ValueError('face_ids entries must be integers or None') from exc

    rounded = np.round(lines, decimals=6)
    node_lookup, edge_lookup = {}, {}
    nodes, raw_degree, edges = [], [], []
    first_faces, source_faces, source_indices, seen_faces = [], [], [], []
    collapsed = 0
    for segment_index, (segment, face) in enumerate(zip(rounded, faces)):
        endpoints = []
        for point in segment:
            key = tuple(point)
            node = node_lookup.get(key)
            if node is None:
                node = len(nodes)
                node_lookup[key] = node
                nodes.append(key)
                raw_degree.append(0)
            raw_degree[node] += 1
            endpoints.append(node)
        a, b = endpoints
        if a == b:
            collapsed += 1
        key = (min(a, b), max(a, b))
        eid = edge_lookup.get(key)
        if eid is None:
            eid = len(edges)
            edge_lookup[key] = eid
            edges.append((a, b))
            first_faces.append(face)
            source_faces.append([])
            source_indices.append([])
            seen_faces.append(set())
        source_indices[eid].append(segment_index)
        if face is not None and face not in seen_faces[eid]:
            seen_faces[eid].add(face)
            source_faces[eid].append(face)

    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    raw_degree = np.asarray(raw_degree, dtype=np.int64)
    degree, components, branches, component_count = _topology(len(nodes), edges)
    stats = {
        'raw_vertical_segments': len(lines),
        'canonical_vertical_edges': len(edges),
        'exact_duplicate_vertical_edges': len(lines)-len(edges),
        'duplicate_source_face_count': sum(max(0, len(ids)-1) for ids in source_faces),
        'degree_changed_nodes': int(np.count_nonzero(raw_degree != degree)),
        'collapsed_self_edges': collapsed,
        'component_count': component_count,
        'endpoint_count': int(np.count_nonzero(degree == 1)),
    }
    return CanonicalProfile(nodes=nodes, edges=edges, first_face_ids=first_faces,
                            source_face_ids=source_faces, source_segment_indices=source_indices,
                            raw_degree=raw_degree, degree=degree, components=components,
                            branches=branches, stats=stats)


def endpoint_gap_diagnostics(profile: CanonicalProfile) -> list[dict]:
    """Exact nearest degree-one endpoint in a DIFFERENT component, in metres.

    Split endpoint-bearing components recursively and query both sides against
    each other. Every pair of different components is separated at one split,
    so the minimum across these exact queries is the global foreign-component
    nearest neighbor. This avoids fixed-k misses and quadratic distance tables;
    each endpoint participates in O(log C) tree queries for C components.
    Rows follow node order. Equal-distance nearest endpoints may choose either
    minimizer; distances remain exact. This function never changes geometry.
    """
    endpoints = np.flatnonzero(profile.physical_degree == 1)
    if len(endpoints) < 2:
        return []
    order = np.argsort(profile.physical_components[endpoints], kind='stable')
    endpoints = endpoints[order]
    labels = profile.physical_components[endpoints]
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1])+1,
                             [len(endpoints)]))
    component_count = len(starts)-1
    if component_count < 2:
        return []
    points = profile.nodes[endpoints]
    distances = np.full(len(endpoints), np.inf)
    neighbors = np.full(len(endpoints), -1, dtype=np.int64)
    pending = [(0, component_count)]
    while pending:
        lo, hi = pending.pop()
        if hi-lo < 2:
            continue
        mid = (lo+hi)//2
        left = slice(starts[lo], starts[mid])
        right = slice(starts[mid], starts[hi])
        for query, target in ((left, right), (right, left)):
            candidate_distance, candidate_index = cKDTree(points[target]).query(points[query], k=1)
            better = candidate_distance < distances[query]
            distances[query][better] = candidate_distance[better]
            neighbors[query][better] = endpoints[target][candidate_index[better]]
        pending.extend(((lo, mid), (mid, hi)))

    buckets = ('<1e-6', '1e-6-1e-4', '1e-4-1e-3', '1e-3-1e-2', '>=1e-2')
    bucket_indices = np.searchsorted([1e-6, 1e-4, 1e-3, 1e-2], distances, side='right')
    return [{'node_id': int(endpoints[i]), 'other_node_id': int(neighbors[i]),
             'distance_m': float(distances[i]), 'bucket': buckets[bucket_indices[i]]}
            for i in np.argsort(endpoints)]
