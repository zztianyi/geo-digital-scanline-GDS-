"""Read-only frozen data adapters and local, cached orthogonal observations."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/03_slicing_profiles'), str(ROOT/'scripts/04_structure_recognition')]
import orthogonal_profile_constraint as opc
from generate_scanline_slices import slice_and_check_faces


def iter_pickles(path):
    with Path(path).open('rb') as stream:
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                break


def clip_out_gap(lines, low, high):
    """Remove the entire target-height interval, splitting crossing edges."""
    result = []
    for line in np.asarray(lines):
        za, zb = line[:, 2]
        if max(za, zb) <= low or min(za, zb) >= high:
            result.append(line)
            continue
        if abs(zb-za) < 1e-12:
            continue
        for bound, keep_lower in ((low, True), (high, False)):
            t = (bound-za)/(zb-za)
            if 0 < t < 1:
                point = line[0]+t*(line[1]-line[0])
                for endpoint in line:
                    if (endpoint[2] < low if keep_lower else endpoint[2] > high):
                        result.append(np.array([endpoint, point]))
    return np.asarray(result, dtype=float).reshape(-1, 2, 3)


def xyz_from_uz(uz, s, arc):
    uz = np.asarray(uz)
    angle = arc['angle_min']+s/arc['radius']
    radius = uz[:, 0]+arc['radius']
    return np.column_stack((arc['center'][0]+radius*np.cos(angle),
                            arc['center'][1]+radius*np.sin(angle), uz[:, 1]))


class FrozenProfiles:
    def __init__(self, prior_dir):
        self.prior = Path(prior_dir)
        self.manifest = json.loads((self.prior/'manifest.json').read_text(encoding='utf-8'))
        self.arc = self.manifest['arc']
        self.baseline = Path(self.manifest['baseline'])
        with (self.baseline/'slices.pkl').open('rb') as stream:
            self.slices = pickle.load(stream)
        self.keys = sorted(self.slices, key=float)
        self.s = np.array([float(k) for k in self.keys])
        self.key_index = {k: i for i, k in enumerate(self.keys)}
        self.levels = np.asarray(self.manifest['horizontal_levels'])
        self.horizontal = {r['level_index']: r for r in iter_pickles(self.prior/'horizontal_slices_clean.pkl')}
        self.groups = {}
        with (self.prior/'baseline_vertical_groups.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                self.groups[row['slice_key']] = row['red_groups_corrected']
                for group in self.groups[row['slice_key']]:
                    group['audit_observable'] = opc.on_scanline_ray(group['node_order'], self.arc, group['s'])
        self.vertical_cache = {}

    def neighbors(self, key):
        i = self.key_index[key]
        return self.keys[max(0, i-2):min(len(self.keys), i+3)]

    def observations(self, key, levels):
        signature = (key, tuple(np.round(levels, 10)))
        if signature not in self.vertical_cache:
            hits = opc.vertical_crossings(self.slices[key]['slicing']['lines_3d'], levels, self.arc, float(key))
            self.vertical_cache[signature] = opc.level_map(hits)
        return self.vertical_cache[signature]

    def local_lines(self, key, low, high):
        lines = self.slices[key]['slicing']['lines_3d']
        return lines[(lines[:, :, 2].max(axis=1) >= low) & (lines[:, :, 2].min(axis=1) <= high)]


class AdaptiveMeshIndex:
    """Whole-triangle ROI index; plane cuts reuse the existing exact helper."""
    def __init__(self, frozen):
        self.frozen = frozen
        self.vertices = np.load(frozen.prior/'mesh_cache/vertices.npy', mmap_mode='r')
        self.faces = np.load(frozen.prior/'mesh_cache/faces.npy', mmap_mode='r')
        z = self.vertices[:, 2]
        self.z_min = z[self.faces].min(axis=1)
        self.z_max = z[self.faces].max(axis=1)
        s = opc.to_suz(self.vertices, frozen.arc)[:, 0]
        self.s_min = s[self.faces].min(axis=1)
        self.s_max = s[self.faces].max(axis=1)
        start = np.floor(self.z_min).astype(np.int32)
        stop = np.floor(self.z_max).astype(np.int32)
        counts = stop-start+1
        ids = np.repeat(np.arange(len(self.faces)), counts)
        bins = np.repeat(start, counts)+np.arange(len(ids))-np.repeat(np.cumsum(counts)-counts, counts)
        order = np.argsort(bins, kind='stable')
        bins, ids = bins[order], ids[order]
        values, starts = np.unique(bins, return_index=True)
        ends = np.r_[starts[1:], len(ids)]
        self.z_bins = {int(v): ids[a:b] for v, a, b in zip(values, starts, ends)}
        self.cache = {}
        self.adaptive_seconds = 0.
        self.adaptive_cuts = 0
        self.global_reused_cuts = 0

    def roi(self, s_low, s_high, z_low, z_high):
        chunks = [self.z_bins[b] for b in range(math_floor(z_low), math_floor(z_high)+1) if b in self.z_bins]
        ids = np.unique(np.concatenate(chunks)) if chunks else np.array([], dtype=int)
        ids = ids[(self.z_max[ids] >= z_low) & (self.z_min[ids] <= z_high) &
                  (self.s_max[ids] >= s_low) & (self.s_min[ids] <= s_high)]
        vertex_ids, inverse = np.unique(self.faces[ids], return_inverse=True)
        mesh = trimesh.Trimesh(vertices=np.asarray(self.vertices[vertex_ids]), faces=inverse.reshape(-1, 3),
                               process=False, validate=False)
        return mesh, ids

    def build_window(self, case):
        cid = case['candidate_id']
        if cid in self.cache:
            return self.cache[cid]
        t0 = time.perf_counter()
        frozen = self.frozen
        low, high = np.asarray(case['lower_uz']), np.asarray(case['upper_uz'])
        height = high[1]-low[1]
        zs = [low[1]+height*p for p in (.25, .5, .75)]
        zs += frozen.levels[(frozen.levels > low[1]+1e-7) & (frozen.levels < high[1]-1e-7)].tolist()
        zs = np.array(sorted(zs))
        zs = zs[np.r_[True, np.diff(zs) > 1e-6]]
        key = case['slice_key']
        neighbor_keys = frozen.neighbors(key)
        s_values = np.array([float(k) for k in neighbor_keys])
        target_index = neighbor_keys.index(key)
        z_pad = max(.10, height*.25)
        mesh, original_face_ids = self.roi(s_values.min()-.015, s_values.max()+.015,
                                          low[1]-z_pad, high[1]+z_pad)
        observations = {k: frozen.observations(k, zs) for k in neighbor_keys if k != key}
        layers, plots = [], []
        adaptive_start = time.perf_counter()
        for zi, z in enumerate(zs):
            nearest = int(np.argmin(abs(frozen.levels-z)))
            if abs(frozen.levels[nearest]-z) < 1e-7:
                full = frozen.horizontal[nearest]
                mask = np.isin(full['face_ids'], original_face_ids)
                clean = {k: full[k][mask] for k in ('lines_3d', 'face_ids', 'edge_branch')}
                self.global_reused_cuts += 1
            else:
                if len(mesh.faces):
                    lines, face_ids = slice_and_check_faces(mesh, np.array([0., 0., z]), np.array([0., 0., 1.]))
                    clean = opc.clean_horizontal(lines, original_face_ids[face_ids], frozen.arc,
                                                 frozen.manifest['snap_tolerance_m'])
                else:
                    clean = {'lines_3d': np.empty((0, 2, 3)), 'face_ids': np.array([], dtype=int), 'edge_branch': np.array([], dtype=int)}
                self.adaptive_cuts += 1
            hits = opc.horizontal_crossings(clean, s_values, frozen.arc)
            target = hits[hits[:, 0] == target_index]
            horizontal = []
            for r in target:
                linked = []
                for ni, neighbor_key in enumerate(neighbor_keys):
                    if neighbor_key == key:
                        continue
                    vertical_values = observations[neighbor_key].get(zi, np.empty((0, 1)))[:, 0]
                    hu = hits[(hits[:, 0] == ni) & (hits[:, 2] == r[2]), 1]
                    # Same horizontal path plus actual vertical coincidence,
                    # not merely a nearest radial value on another surface.
                    values = [float(u) for u in vertical_values if len(hu) and np.min(abs(hu-u)) < 1e-4]
                    linked.append({'s': float(neighbor_key), 'values': values})
                horizontal.append({'u': float(r[1]), 'branch_id': int(r[2]), 'face_id': int(r[3]),
                                   'linked_neighbors': linked})
            layers.append({'z': float(z),
                           'horizontal': horizontal,
                           'neighbors': [{'s': float(k), 'values': observations[k].get(zi, np.empty((0, 1)))[:, 0].tolist()}
                                         for k in neighbor_keys if k != key]})
            plots.append({'z': float(z), 'lines_su': opc.to_suz(clean['lines_3d'], frozen.arc)[:, :, :2],
                          'lines_xyz': clean['lines_3d'], 'selected_u': None})
        self.adaptive_seconds += time.perf_counter()-adaptive_start
        slope = (high[0]-low[0])/height
        lower_n = np.asarray(case.get('lower_neighbor_uz', low-[slope*.02, .02]))
        upper_n = np.asarray(case.get('upper_neighbor_uz', high+[slope*.02, .02]))
        lower_tangent = (low[0]-lower_n[0])/(low[1]-lower_n[1]) if abs(low[1]-lower_n[1]) > 1e-9 else slope
        upper_tangent = (upper_n[0]-high[0])/(upper_n[1]-high[1]) if abs(upper_n[1]-high[1]) > 1e-9 else slope
        window = {'s': float(key), 'lower_uz': low, 'upper_uz': high,
                  'lower_tangent': float(lower_tangent), 'upper_tangent': float(upper_tangent), 'layers': layers}
        local = {'window': window, 'mesh_vertices': np.asarray(mesh.vertices), 'mesh_faces': np.asarray(mesh.faces),
                 'mesh_face_ids': original_face_ids, 'horizontal_profiles': plots,
                 'neighbor_keys': neighbor_keys, 'z_bounds': [float(low[1]-z_pad), float(high[1]+z_pad)],
                 'build_seconds': time.perf_counter()-t0}
        self.cache[cid] = local
        return local


def math_floor(value):
    return int(np.floor(value))


def plot_payload(case, local, result, frozen, title_note='', truth=None, masked=False):
    key = case['slice_key']
    low, high = np.asarray(case['lower_uz']), np.asarray(case['upper_uz'])
    zlow, zhigh = local['z_bounds']
    target = frozen.local_lines(key, zlow, zhigh)
    if masked:
        target = clip_out_gap(target, low[1], high[1])
    neighbor_profiles = []
    for k in local['neighbor_keys']:
        lines = target if k == key else frozen.local_lines(k, zlow, zhigh)
        neighbor_profiles.append({'s': float(k), 'lines_xyz': lines,
                                  'lines_uz': opc.to_suz(lines, frozen.arc)[:, :, 1:]})
    horizontals = []
    selected = result['selected_nodes'][1:-1]
    for i, hp in enumerate(local['horizontal_profiles']):
        node = selected[i] if i < len(selected) else None
        horizontals.append({**hp, 'selected_u': node['u'] if node is not None and node['source'] == 'HORIZONTAL_MESH' else None})
    curve = np.asarray(result['curve_uz'])
    radial = np.r_[curve[:, 0], low[0], high[0]]
    pad = max(.10, np.ptp(radial)*.25)
    reds = [g['uz'] for g in frozen.groups[key] if g['z_max'] >= zlow and g['z_min'] <= zhigh]
    if masked:
        visible_red = []
        for points in reds:
            xyz = xyz_from_uz(np.asarray(points), float(key), frozen.arc)
            edges = np.stack((xyz[:-1], xyz[1:]), axis=1)
            visible_red.extend(opc.to_suz(clip_out_gap(edges, low[1], high[1]), frozen.arc)[:, :, 1:])
        reds = visible_red
    payload = {'candidate_id': case['candidate_id'], 's': float(key), 'lower_uz': low, 'upper_uz': high,
               'inferred_curve': curve, 'inferred_xyz': xyz_from_uz(curve, float(key), frozen.arc),
               'target_lines_uz': opc.to_suz(target, frozen.arc)[:, :, 1:], 'neighbor_profiles': neighbor_profiles,
               'horizontal_profiles': horizontals, 'mesh_vertices': local['mesh_vertices'], 'mesh_faces': local['mesh_faces'],
               'red_groups': reds, 'title_note': title_note,
               'bounds': {'u': [float(radial.min()-pad), float(radial.max()+pad)], 'z': [zlow, zhigh],
                          's': [float(local['neighbor_keys'][0])-.02, float(local['neighbor_keys'][-1])+.02]}}
    if truth is not None:
        payload['truth_curve'] = truth
    # All candidates remain in solver input. A local XYZ display window prevents
    # remote radial branches from shrinking the actual gap to an invisible dot.
    corners = np.vstack([xyz_from_uz(np.array([[u, z] for u in payload['bounds']['u'] for z in [zlow, zhigh]]), s, frozen.arc)
                         for s in payload['bounds']['s']])
    payload['bounds_xyz'] = np.column_stack((corners.min(axis=0), corners.max(axis=0))).tolist()
    return payload
