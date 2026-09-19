"""Changed-slice recognition and local track audit; no mesh loads or global run.

FrozenProfiles is read-only. Returned groups_by_slice is a replacement overlay,
not a complete baseline. Outputs are exclusively created inside output.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import pickle

import numpy as np

import validate_orthogonal_scanline_constraint as legacy
import validate_track_bridge_split as bridge
from vertical_profile_canonicalization import canonicalize_vertical

opc = legacy.opc
TOLERANCE = 1e-5


def _edge_key(line):
    return tuple(sorted(tuple(point) for point in np.round(line, 6)))


def _output(frozen, output, names):
    output = Path(output).resolve()
    for protected in (Path(frozen.prior).resolve(), Path(frozen.baseline).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('output must be separate from frozen prior/baseline and their ancestors/descendants')
    for name in names:
        if (output/name).exists():
            raise FileExistsError(output/name)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['slice_key']
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=legacy.json_default)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _recognize(frozen, key, intersections, namespace, synthetic=()):
    source = copy.deepcopy(frozen.slices[key])
    source['slice_key'] = key
    source['slicing'] = {'lines_3d': np.asarray([edge[:2] for edge in intersections]).reshape(-1, 2, 3),
                         'face_ids': [edge[2] for edge in intersections]}
    utils = namespace['ArcUtils']

    def red_segments(*args, **kwargs):
        segments = utils.get_ordered_red_segments_for_path(*args, **kwargs)
        # Preserve all numerical predicates, then correct only synthetic identity.
        # Legacy allclose uses relative tolerance and can match a nearby real edge.
        return [(*seg[:6], None) if _edge_key(seg[4:6]) in synthetic else seg for seg in segments]

    slicer = namespace['ArcSlicer'](source, use_merged_paths=False, red_segment_fn=red_segments)
    return utils.compute_2d_centroids(slicer.run_all())


def _groups(record, arc):
    """The frozen summarize_record schema, allowing None for synthetic FaceIDs."""
    key = record['slice_key']
    faces = {_edge_key(edge[:2]): edge[2] for edge in record['slicing']['intersections']}
    groups = []
    for pi, corrected in record['red_groups_corrected'].items():
        for gi, group in enumerate(corrected):
            nodes = np.asarray(group['node_order'], dtype=float)
            if len(nodes) < 2:
                continue
            uz = opc.to_suz(nodes, arc)[:, 1:]
            keys = [_edge_key(edge) for edge in zip(nodes, nodes[1:])]
            normals = np.asarray([seg[3] for seg in group['group_segments']])
            groups.append({'id': f'{key}:p{pi}:g{gi}', 'slice_key': key, 's': float(key),
                           'audit_observable': opc.on_scanline_ray(nodes, arc, float(key)),
                           'parent_index': int(pi), 'group_index': gi, 'node_order': nodes,
                           'uz': uz, 'z_min': float(uz[:, 1].min()), 'z_max': float(uz[:, 1].max()),
                           'u_min': float(uz[:, 0].min()), 'u_max': float(uz[:, 0].max()),
                           'endpoints_3d': nodes[[0, -1]],
                           'length_2d': float(np.linalg.norm(np.diff(uz, axis=0), axis=1).sum()),
                           'mean_red_normal_2d': normals.mean(axis=0) if len(normals) else None,
                           'face_ids': sorted({int(faces[k]) for k in keys if faces.get(k) is not None}),
                           'unmapped_corrected_edges': sum(k not in faces for k in keys),
                           'red_face_ids': [None if seg[6] is None else int(seg[6])
                                            for seg in group['group_segments']]})
    return groups


def _clean_equivalence(frozen, profiles, extras, namespace):
    rows = []
    for key in sorted(frozen.keys, key=lambda k: (len(frozen.slices[k]['slicing']['lines_3d']), float(k))):
        slicing = frozen.slices[key]['slicing']
        if not len(slicing['lines_3d']) or extras.get(key):
            continue
        profile = profiles.get(key)
        if profile is None:
            profile = canonicalize_vertical(slicing['lines_3d'], slicing['face_ids'])
        if profile.stats['exact_duplicate_vertical_edges']:
            continue
        raw = copy.deepcopy(frozen.slices[key])
        raw['slice_key'] = key
        old = namespace['process_single_slice_legacy'](raw)
        new = _recognize(frozen, key, profile.as_intersections(), namespace)
        row = {'slice_key': key}
        for field in ('nodes', 'paths', 'red_groups', 'red_groups_corrected'):
            row[field+'_equal'] = legacy.semantically_equal(old[field], new[field])
        plane = old['plane_params']
        args = (plane['origin'], plane['radial_dir'], plane['vertical_dir'])
        utils = namespace['ArcUtils']
        for name, function in [('path_nodes', utils.order_nonclosed_path),
                               ('red_segments', utils.get_ordered_red_segments_for_path_legacy)]:
            values = [[function(path, *args) for path in record['paths']['unmerged_subsets']]
                      for record in (old, new)]
            row[name+'_equal'] = legacy.semantically_equal(*values)
        row['first_face_ids_equal'] = row['paths_equal'] and row['red_segments_equal']
        rows.append(row)
        if len(rows) == 3:
            break
    return rows


def run_changed_recognition(frozen, canonical_profiles, extras_by_key, output):
    """Rerun duplicates/extras only; additionally compare up to 3 clean slices."""
    if set(extras_by_key)-set(canonical_profiles):
        raise ValueError('Every extras key must have a full canonical profile')
    for key, profile in canonical_profiles.items():
        slicing = frozen.slices[key]['slicing']
        raw_first = {}
        for line, face in zip(slicing['lines_3d'], slicing['face_ids']):
            raw_first.setdefault(_edge_key(line), face)
        canonical_first = {_edge_key(line): face for line, face in zip(profile.lines_xyz, profile.first_face_ids)}
        if raw_first != canonical_first or profile.stats['raw_vertical_segments'] != len(slicing['lines_3d']):
            raise ValueError(f'{key}: full slice observed geometry/first FaceIDs required')
        for extra in extras_by_key.get(key, []):
            line = np.asarray(extra['points_xyz'], dtype=float)
            if extra.get('face_id') is not None:
                raise ValueError('Synthetic face_id must be None')
            if line.shape != (2, 3) or not np.isfinite(line).all() or not extra['source']:
                raise ValueError('Synthetic edge requires finite (2, 3) points_xyz and source')
    names = ('changed_vertical_recognition.pkl', 'changed_vertical_provenance.csv',
             'changed_vertical_recognition.csv', 'clean_slice_equivalence.csv')
    output = _output(frozen, output, names)
    namespace = legacy.load_recognition_kernel().__globals__
    changed = [key for key in sorted(canonical_profiles, key=float)
               if canonical_profiles[key].stats['exact_duplicate_vertical_edges'] > 0 or extras_by_key.get(key)]
    groups_by_slice, rows = {}, []
    fields = ['slice_key', 'edge_index', 'source', 'face_id', 'source_face_ids',
              'source_segment_indices', 'points_xyz', 'disposition']
    with (output/names[0]).open('xb') as stream, (output/names[1]).open('x', encoding='utf-8-sig', newline='') as provenance:
        writer = csv.DictWriter(provenance, fieldnames=fields)
        writer.writeheader()
        for key in changed:
            profile = canonical_profiles[key]
            intersections = profile.as_intersections()
            indices = {_edge_key(edge[:2]): i for i, edge in enumerate(intersections)}
            synthetic = set()
            sources = [{'source': 'OBSERVED_VERTICAL', 'face_id': face,
                        'source_face_ids': profile.source_face_ids[i],
                        'source_segment_indices': profile.source_segment_indices[i], 'points_xyz': edge[:2],
                        'edge_index': i, 'disposition': 'RETAINED'}
                       for i, edge in enumerate(intersections) for face in [edge[2]]]
            for extra in extras_by_key.get(key, []):
                line = np.round(np.asarray(extra['points_xyz'], dtype=float), 6)
                edge = _edge_key(line)
                disposition = 'SKIPPED_EXISTING' if edge in indices else 'ADDED'
                if edge not in indices:
                    indices[edge] = len(intersections)
                    intersections.append((tuple(line[0]), tuple(line[1]), None))
                    synthetic.add(edge)
                sources.append({**extra, 'points_xyz': line, 'edge_index': indices[edge],
                                'source_face_ids': [], 'source_segment_indices': [], 'disposition': disposition})
            result = _recognize(frozen, key, intersections, namespace, synthetic)
            groups = _groups(result, frozen.arc)
            groups_by_slice[key] = groups
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for source in sources:
                writer.writerow({field: json.dumps(source[field], default=legacy.json_default)
                                 if field in ('source_face_ids', 'source_segment_indices', 'points_xyz')
                                 else (key if field == 'slice_key' else source.get(field)) for field in fields})
            rows.append({'slice_key': key, **profile.stats, 'synthetic_edges_added': len(synthetic),
                         'observed_edges_lost': 0, 'old_groups': len(frozen.groups[key]),
                         'new_groups': len(groups), 'old_red_length_m': sum(g['length_2d'] for g in frozen.groups[key]),
                         'new_red_length_m': sum(g['length_2d'] for g in groups)})
    equivalence = _clean_equivalence(frozen, canonical_profiles, extras_by_key, namespace)
    _csv(output/names[2], rows)
    _csv(output/names[3], equivalence)
    summary = {'changed_slices': len(changed), 'source_changed_slices': len(changed), 'changed_slice_keys': changed,
               'unchanged_slices': len(frozen.keys)-len(changed), 're_recognized_groups': sum(len(g) for g in groups_by_slice.values()),
               'synthetic_edges_added': sum(r['synthetic_edges_added'] for r in rows), 'observed_edges_lost': 0,
               'clean_equivalence_samples': len(equivalence), 'clean_slice_equivalence':
               all(value for row in equivalence for field, value in row.items() if field != 'slice_key') if len(equivalence) >= 2 else None}
    return {'groups_by_slice': groups_by_slice, 'rows': rows, 'summary': summary}


def _group_edges(group):
    nodes = group['node_order']
    return {_edge_key(edge) for edge in zip(nodes, nodes[1:])}


def _evidence(group, levels, horizontal, arc):
    observations = bridge._safe_group_map(group, levels, arc)
    result = {'levels': set(observations), 'sampled': set(), 'branches': {}}
    for zi, values in observations.items():
        hits = horizontal.get(zi)
        if hits is not None and len(hits):
            result['sampled'].add(zi)
            near = np.any(abs(values[:, 0, None]-hits[None, :, 0]) <= TOLERANCE, axis=0)
            result['branches'][zi] = {int(b) for b in hits[near, 1] if b >= 0 and float(b).is_integer()}
    return result


def run_local_track_check(frozen, groups_by_slice, output):
    """Rematch only true 0.05 m neighbors touching the changed-slice overlay.

    Bridge thresholds are illustrative, never geological confirmation. Missing
    or singleton H evidence remains UNRESOLVED; no adaptive cuts or sweeps occur.
    """
    if set(groups_by_slice)-set(frozen.keys):
        raise ValueError('Unknown changed slice key')
    output = _output(frozen, output, ('local_track_pairs.csv', 'major_old_group_preservation.csv'))
    keys = sorted(frozen.keys, key=float)
    pairs = [(a, b) for a, b in zip(keys, keys[1:])
             if (a in groups_by_slice or b in groups_by_slice) and np.isclose(float(b)-float(a), .05, atol=1e-8, rtol=0)]
    after = {**frozen.groups, **groups_by_slice}
    local = {key for pair in pairs for key in pair}
    horizontal = {key: {} for key in local}
    needed = {}
    for key in local:
        for group in frozen.groups[key]+after[key]:
            for zi in bridge._safe_group_map(group, frozen.levels, frozen.arc):
                if zi in frozen.horizontal:
                    needed.setdefault(zi, set()).add(key)
    for zi, selected in sorted(needed.items()):
        selected = sorted(selected, key=float)
        hits = opc.horizontal_crossings(frozen.horizontal[zi], np.array([float(k) for k in selected]), frozen.arc)
        hits = hits[np.isfinite(hits).all(axis=1) & (hits[:, 1]+frozen.arc['radius'] > 0)]
        for i, values in opc.level_map(hits).items():
            horizontal[selected[i]][zi] = values
    report = Path(frozen.prior)/'ORTHOGONAL_SCANLINE_CONSTRAINT_REPORT.json'
    tau = json.loads(report.read_text(encoding='utf-8-sig'))['baseline']['tau_u_m'] if report.is_file() else TOLERANCE
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('Invalid baseline consistency tau_u_m')
    rows, changed_pairs, before_count, after_count = [], 0, 0, 0
    for a, b in pairs:
        old = opc.consistency({k: frozen.groups[k] for k in (a, b)}, horizontal, frozen.levels, TOLERANCE, tau_u=tau)
        new = opc.consistency({k: after[k] for k in (a, b)}, horizontal, frozen.levels, TOLERANCE, tau_u=tau)
        byid = {g['id']: g for k in (a, b) for g in after[k]}
        evidence = {gid: _evidence(g, frozen.levels, horizontal[g['slice_key']], frozen.arc) for gid, g in byid.items()}
        old_ids = {g['id']: g for k in (a, b) for g in frozen.groups[k]}
        def signatures(result, groups):
            return {(frozenset(_group_edges(groups[r['left_id']])), frozenset(_group_edges(groups[r['right_id']])))
                    for r in result['pairs']}
        changed_pairs += signatures(old, old_ids) != signatures(new, byid)
        before_count += len(old['pairs'])
        after_count += len(new['pairs'])
        matched = {(r['left_id'], r['right_id']) for r in new['pairs']}
        for candidate in new['candidates']:
            left, right = candidate['left_id'], candidate['right_id']
            metrics = bridge.bridge_metrics(byid[left], byid[right], frozen.levels, evidence[left], evidence[right], set(frozen.horizontal))
            is_matched = (left, right) in matched
            failed = [field for field, minimum in zip(('L_support', 'R_bridge', 'R_overlap'), bridge.ILLUSTRATIVE)
                      if metrics['gate_eligible'] and metrics[field]+bridge.EPSILON < minimum]
            decision = ('UNRESOLVED_UNDERSAMPLED' if not metrics['gate_eligible'] else
                        'UNMATCHED_EVIDENCE' if not is_matched else
                        'REMOVED_ILLUSTRATIVE' if failed else 'RETAINED_SAMPLED')
            rows.append({**candidate, **metrics, 'matched': is_matched, 'decision': decision, 'failed_gates': failed})
    preservation = []
    major = sorted((g for gs in frozen.groups.values() for g in gs), key=lambda g: -g['length_2d'])[:20]
    for group in major:
        old_edges = _group_edges(group)
        new_edges = set().union(*(_group_edges(g) for g in after[group['slice_key']]))
        lengths = {edge: float(np.linalg.norm(np.asarray(edge[1])-edge[0])) for edge in old_edges}
        total = sum(lengths.values())
        preservation.append({'old_group_id': group['id'], 'slice_key': group['slice_key'],
                             'old_edges': len(old_edges), 'retained_edges': len(old_edges & new_edges),
                             'geometry_preserved': old_edges <= new_edges,
                             'retained_length_fraction': sum(lengths[e] for e in old_edges & new_edges)/total if total else None})
    _csv(output/'local_track_pairs.csv', rows)
    _csv(output/'major_old_group_preservation.csv', preservation)
    summary = {'local_slice_pairs': len(pairs), 'changed_local_pair_count': changed_pairs,
               'matched_pairs_before': before_count, 'matched_pairs_after': after_count,
               'applied_tolerance_m': TOLERANCE, 'consistency_tau_u_m': tau,
               'consistency_tau_source': 'frozen_baseline' if report.is_file() else 'fallback_matching_tolerance',
               'illustrative_gates': dict(zip(('L_support', 'R_bridge', 'R_overlap'), bridge.ILLUSTRATIVE)),
               'unresolved_pairs': sum(r['decision'] == 'UNRESOLVED_UNDERSAMPLED' for r in rows),
               'illustrative_removed_pairs': sum(r['decision'] == 'REMOVED_ILLUSTRATIVE' for r in rows),
               'major_old_groups_checked': len(preservation),
               'major_old_groups_geometry_preserved': all(r['geometry_preserved'] for r in preservation) if preservation else None,
               'confirmed_overmerge_reduction': None, 'frozen_outside_local_pairs_unchanged': True}
    return {'summary': summary, 'rows': rows}
