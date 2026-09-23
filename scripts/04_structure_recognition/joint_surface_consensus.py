"""Arc-local joint V/H hypotheses; H and V are redundant mesh observations.

Runs are identified within a level, never by a radial nearest-neighbor match.
Only a unique actual V/H intersection can propagate support across slices.
No observed coordinate is averaged, resampled or replaced here.
"""
from collections import defaultdict
import numpy as np
from horizontal_surface_link import associate_crossing, branch_crossings

SCALES = (.05, .25, .50, 1.00)


def _index(graph):
    if '_joint_index' in graph:
        return graph['_joint_index']
    runs, by_s, nodes_s = defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(list)
    for node in graph['nodes']:
        nodes_s[node[0]].append(node)
    for i, hit in enumerate(graph.get('observations', ())):
        runs[(hit['level_index'], hit['h_path_id'])][hit['s']].append(i)
        by_s[hit['s']].append(i)
    index = dict(runs=runs, by_s=by_s, nodes_s=nodes_s,
                 matches=dict(graph.get('_observed_crossing_cache',{})), profiles={})
    graph['_joint_index'] = index
    return index


def _matches(graph, index, i):
    if i not in index['matches']:
        hit = graph['observations'][i]
        result = []
        for node in index['nodes_s'][hit['s']]:
            b = graph['nodes'][node]
            if b['kind'] != 'OPEN_SURFACE_BRANCH' or not b['z_range'][0]-1e-9 <= hit['z'] <= b['z_range'][1]+1e-9:
                continue
            matches = associate_crossing(hit, b)
            # A vertex has two incident edges but one original arc position.
            unique = {round(c['arc_position'], 8): c for c in matches}
            result.extend((node, c) for c in unique.values())
        index['matches'][i] = result
    return index['matches'][i]


def arc_support_profile(graph, node):
    """Every actual crossing retains branch, edge, arc, H-run and neighbor IDs.

    Legacy link-only caches cannot locate a hit on a multivalued branch. They
    provide coarse selection evidence elsewhere, but no arc-specific consent
    to remove a fold.
    """
    index = _index(graph)
    if node in index['profiles']:
        return index['profiles'][node]
    branch = graph['nodes'][node]
    order = graph['slice_order']
    si = order.index(node[0])
    rows = []
    for hi in index['by_s'].get(node[0], ()):
        hit = graph['observations'][hi]
        matches = _matches(graph, index, hi)
        local = [cross for n, cross in matches if n == node]
        if not local:
            continue
        run = index['runs'][(hit['level_index'], hit['h_path_id'])]
        neighbors = []
        # A duplicated H hit or coincident target V remains ambiguous.
        unambiguous = len(matches) == 1 and len(run[node[0]]) == 1
        if unambiguous:
            for step in (-1, 1):
                k = si+step
                while 0 <= k < len(order) and abs(order[k]-node[0]) <= 1.+1e-8:
                    ids = run.get(order[k], ())
                    if len(ids) != 1:
                        break
                    found = _matches(graph, index, ids[0])
                    if len(found) != 1:
                        break
                    neighbors.append(found[0][0])
                    k += step
        slices = sorted({node[0], *(n[0] for n in neighbors)})
        for cross in local:
            rows.append(dict(branch_id=node[1], edge_id=int(branch['edge_order'][cross['edge_index']]),
                edge_index=cross['edge_index'], arc_position=cross['arc_position'], z=hit['z'], u=cross['u'],
                h_path_id=hit['h_path_id'], level_index=hit['level_index'], neighbor_nodes=neighbors,
                support_slices=slices, support_s_span=slices[-1]-slices[0],
                left_support_s_span=node[0]-slices[0], right_support_s_span=slices[-1]-node[0],
                ambiguous=not unambiguous))
    rows.sort(key=lambda r: (r['arc_position'], r['level_index'], str(r['h_path_id'])))
    index['profiles'][node] = rows
    return rows


def summarize_profile(rows, *, target_s, core_interval=None):
    """Continuous persistence statistics. Different H levels never share IDs."""
    supported = [r for r in rows if r['neighbor_nodes'] and not r['ambiguous']]
    levels = sorted({r['level_index'] for r in supported})
    z_by_level = {r['level_index']: r['z'] for r in supported}
    longest = current = 0; previous = None; start = None; longest_z = 0.
    for li in levels:
        if previous is None or li != previous+1:
            current, start = 1, li
        else:
            current += 1
        longest = max(longest, current)
        longest_z = max(longest_z, z_by_level[li]-z_by_level[start])
        previous = li
    slices = {target_s}; members = set()
    for r in supported:
        slices.update(r['support_slices']); members.update(r['neighbor_nodes'])
    def mean(key):
        # Include unsupported crossings: a lone wide sample cannot license a tail.
        return float(np.mean([r[key] for r in rows])) if rows else 0.
    asc = 0.
    if core_interval is not None:
        for a, b in zip(rows[:-1], rows[1:]):
            if (a['neighbor_nodes'] and b['neighbor_nodes'] and not a['ambiguous'] and not b['ambiguous']
                    and abs(a['level_index']-b['level_index']) <= 1
                    and set(a['neighbor_nodes']) & set(b['neighbor_nodes'])):
                asc += max(0., min(b['arc_position'], core_interval[1])-max(a['arc_position'], core_interval[0]))
    scales = {}
    for radius in SCALES:
        spans, counts, two = [], [], []
        for r in rows:
            ss = [s for s in r['support_slices'] if abs(s-target_s) <= radius+1e-8]
            spans.append(max(ss)-min(ss)); counts.append(len(ss)-1)
            two.append(min(ss) < target_s-1e-8 and max(ss) > target_s+1e-8)
        scales[f'{radius:.2f}'] = dict(mean_s_span=float(np.mean(spans)) if spans else 0.,
            mean_neighbor_count=float(np.mean(counts)) if counts else 0.,
            two_sided_fraction=float(np.mean(two)) if two else 0.)
    return dict(support_slice_count=len(slices)-1, support_s_span=mean('support_s_span'),
        left_support_s_span=mean('left_support_s_span'), right_support_s_span=mean('right_support_s_span'),
        support_z_span=max(z_by_level.values())-min(z_by_level.values()) if levels else 0.,
        same_H_run_level_count=len(levels), same_H_run_longest_level_count=longest,
        same_H_run_longest_z_span=longest_z,
        two_sided_neighbor_count=sum(r['left_support_s_span'] > 1e-8 and r['right_support_s_span'] > 1e-8 for r in rows),
        track_member_count=len(members), ASC_supported_arc_length=asc, wide_support=scales,
        joint_crossing_count=len(rows), raw_arc_evidence_available=bool(rows))


def joint_surface_summary(graph, node, *, arc_interval=None, target_z=None, metrics=None):
    rows = arc_support_profile(graph, node)
    if arc_interval is not None:
        lo, hi = sorted(arc_interval)
        rows = [r for r in rows if lo-1e-9 <= r['arc_position'] <= hi+1e-9]
    if target_z is not None:
        lo, hi = sorted(target_z)
        rows = [r for r in rows if lo-1e-9 <= r['z'] <= hi+1e-9]
    core = (metrics['ASC_start_arc'], metrics['ASC_end_arc']) if metrics and metrics['ASC_exists'] else None
    return summarize_profile(rows, target_s=node[0], core_interval=core)
