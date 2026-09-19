"""Canonical horizontal runs and actual V/H edge crossings (metres).

Path identity is local to one Z level. Numerical round6 tolerance is not a
surface proximity score. No observed vertex is moved to create a link.
"""
from __future__ import annotations
import numpy as np
from vertical_profile_canonicalization import canonicalize_vertical
from orthogonal_profile_constraint import to_suz, horizontal_crossings

CROSSING_TOLERANCE = 3e-6


def canonical_horizontal_paths(lines, face_ids, arc):
    profile = canonicalize_vertical(np.asarray(lines).reshape(-1, 2, 3), face_ids)
    edge_path = np.full(len(profile.edges), -1, dtype=int)
    suz = to_suz(profile.nodes, arc)
    path_id = 0
    for branch in profile.branches:
        direction = 0
        for eid, ds in zip(branch['edges'], np.diff(suz[branch['nodes'], 0])):
            sign = int(np.sign(ds)) if abs(ds) > 1e-12 else 0
            if direction and sign and direction != sign:
                path_id += 1
            edge_path[eid] = path_id
            if sign:
                direction = sign
        path_id += 1
    return dict(lines_3d=profile.lines_xyz, face_ids=np.array(profile.first_face_ids),
                edge_branch=edge_path, path_count=path_id)


def horizontal_mesh_observations(horizontal, levels, positions, arc):
    """Build complete H runs before crossing any target slice; no legacy ROI."""
    rows, inventory = [], []
    positions = np.asarray(positions)
    for zi, z in enumerate(levels):
        data = horizontal[zi]
        clean = canonical_horizontal_paths(data['lines_3d'], data['face_ids'], arc)
        inventory.append(dict(level_index=zi, z=float(z), path_count=clean['path_count'],
                              edge_count=len(clean['lines_3d'])))
        for si, u, path, face in horizontal_crossings(clean, positions, arc):
            s = float(positions[int(si)])
            angle = arc['angle_min']+s/arc['radius']
            xyz = [arc['center'][0]+(u+arc['radius'])*np.cos(angle),
                   arc['center'][1]+(u+arc['radius'])*np.sin(angle), z]
            rows.append(dict(s=s, z=float(z), level_index=zi, h_path_id=int(path),
                             u=float(u), point_xyz=xyz, face_id=int(face),
                             evidence='RAW_HORIZONTAL_OBSERVATION'))
    return rows, inventory


def horizontal_path_observations(levels, positions):
    """Planar SU adapter for measured polylines / unit fixtures.

    Splits at S reversals so a distant loop connection cannot bridge slices.
    """
    rows = []
    for zi, level in enumerate(sorted(levels, key=lambda r: r['z'])):
        seen = set()
        for path in level['paths']:
            run, direction = 0, 0
            points = np.asarray(path['points_su'], dtype=float)
            for a, b in zip(points[:-1], points[1:]):
                ds = b[0]-a[0]
                sign = int(np.sign(ds))
                if direction and sign and direction != sign:
                    run += 1
                if sign:
                    direction = sign
                if abs(ds) <= 1e-12:
                    continue  # Interval-valued hits do not establish adjacency.
                for s in positions:
                    t = (s-a[0])/ds
                    if -1e-10 <= t <= 1+1e-10:
                        u = float(a[1]+t*(b[1]-a[1]))
                        pid = f"{path['h_path_id']}:{run}"
                        signature = (float(s), pid, round(u, 9))
                        if signature not in seen:
                            rows.append(dict(s=float(s), z=float(level['z']), level_index=zi,
                                h_path_id=pid, u=u, evidence='RAW_HORIZONTAL_OBSERVATION'))
                            seen.add(signature)
    return rows


def branch_crossings(branch, z):
    """All on-edge crossings, retaining multivalued folds and provenance."""
    p = np.asarray(branch['points_uz'])
    xyz = np.asarray(branch['points_xyz'])
    a, b = p[:-1], p[1:]
    dz = b[:, 1]-a[:, 1]
    indices = np.flatnonzero((np.minimum(a[:, 1], b[:, 1]) <= z+1e-9) &
                            (np.maximum(a[:, 1], b[:, 1]) >= z-1e-9) & (abs(dz) > 1e-12))
    result = []
    for i in indices:
        t = float(np.clip((z-a[i, 1])/dz[i], 0, 1))
        result.append(dict(u=float(a[i, 0]+t*(b[i, 0]-a[i, 0])), edge_index=int(i), t=t,
            point_xyz=xyz[i]+t*(xyz[i+1]-xyz[i]),
            arc_position=float(branch['arc_positions'][i]+t*(branch['arc_positions'][i+1]-branch['arc_positions'][i]))))
    return result


def associate_crossing(hit, branch, crossings=None):
    matches = []
    for cross in branch_crossings(branch, hit['z']) if crossings is None else crossings:
        if abs(cross['u']-hit['u']) > CROSSING_TOLERANCE:
            continue
        if 'point_xyz' in hit and np.linalg.norm(cross['point_xyz']-hit['point_xyz']) > CROSSING_TOLERANCE:
            continue
        matches.append(cross)
    return matches
