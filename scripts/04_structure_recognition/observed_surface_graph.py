"""Surface graph whose only adjacency is measured same-H-run continuity."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
from horizontal_surface_link import branch_crossings, associate_crossing


def _longest_run(levels):
    longest = current = 0
    previous = None
    for level in sorted(set(levels)):
        current = current+1 if previous is not None and level == previous+1 else 1
        longest = max(longest, current)
        previous = level
    return longest


def build_surface_graph(profiles, observations=(), *, slice_order=None, minimum_levels=2):
    profiles = sorted(profiles, key=lambda p: p['s'])
    nodes = {(float(p['s']), b['branch_id']): b for p in profiles for b in p['branches']}
    order = sorted(float(s) for s in (slice_order if slice_order is not None else [p['s'] for p in profiles]))
    adjacent = set(zip(order[:-1], order[1:]))
    by_s = {float(p['s']): p['branches'] for p in profiles}
    by_level_slice = defaultdict(list)
    level_z = {}
    for hit_index, h in enumerate(observations):
        level_z[h['level_index']] = h['z']
        by_level_slice[(h['level_index'], h['s'])].append((hit_index, h))
    evaluable, association, raw_paths = defaultdict(set), {}, defaultdict(dict)
    ambiguous_hits = []
    for (li, s), hits in by_level_slice.items():
        z = level_z[li]
        active = {}
        for b in by_s.get(s, ()):
            if b['z_range'][0]-1e-9 <= z <= b['z_range'][1]+1e-9:
                cross = branch_crossings(b, z)
                if cross:
                    active[b['branch_id']] = cross
                    evaluable[(s, b['branch_id'])].add(li)
        for hi, hit in hits:
            matches = [(s, bid) for bid, cross in active.items()
                       if associate_crossing(hit, nodes[(s, bid)], cross)]
            association[hi] = matches
            raw_paths[(li, hit['h_path_id'])].setdefault(s, []).append(hi)
            if len(matches) > 1:
                ambiguous_hits.append(hi)
    evidence, transitions = defaultdict(list), []
    for (li, pid), slices in raw_paths.items():
        for sa, sb in zip(sorted(slices)[:-1], sorted(slices)[1:]):
            if (sa, sb) not in adjacent:
                continue
            # A monotone H run must cross each slice uniquely. Coincident
            # vertical branches/forks remain ambiguous rather than welded.
            left, right = slices[sa], slices[sb]
            if len(left) != 1 or len(right) != 1:
                continue
            ha, hb = left[0], right[0]
            if len(association[ha]) != 1 or len(association[hb]) != 1:
                continue
            a, b = association[ha][0], association[hb][0]
            item = dict(a=a, b=b, level_index=li, z=level_z[li], h_path_id=pid, hit_indices=[ha, hb])
            ca, cb = nodes[a]['kind'] == 'CLOSED_COMPONENT', nodes[b]['kind'] == 'CLOSED_COMPONENT'
            if ca != cb:
                transitions.append(item)
            else:
                evidence[(a, b)].append(item)
    links, weak = [], []
    for (a, b), items in sorted(evidence.items()):
        levels = sorted({r['level_index'] for r in items})
        common = evaluable[a] & evaluable[b]
        fraction = len(levels)/len(common) if common else 0.
        run = _longest_run(levels)
        row = dict(a=a, b=b, common_H_levels=len(common), same_H_path_levels=len(levels),
            continuous_H_support_run=run, support_fraction=fraction,
            support_z_span=level_z[levels[-1]]-level_z[levels[0]],
            levels=levels, evidence=items, ambiguous=False)
        # Fragmented tails may share only part of a full neighboring branch.
        # Require repeated measured continuity, never shape alone.
        (links if run >= minimum_levels and fraction >= .5 else weak).append(row)
    parent = {node: node for node in nodes}
    members_by_slice = {node: {node[0]: [node]} for node in nodes}
    supported_levels = defaultdict(set)
    for link in links:
        for n in (link['a'], link['b']):
            supported_levels[n].update(link['levels'])
    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    accepted, ambiguous_links = [], []
    # A connected mesh can join different surfaces far from the target.
    # Extract tracks with a same-slice exclusion constraint instead of making
    # every graph connected component a single surface identity. Disjoint
    # support intervals remain eligible for a confidence-crossover handoff.
    links.sort(key=lambda l: (-l['support_fraction'], -l['continuous_H_support_run'],
                              -l['same_H_path_levels'], l['a'], l['b']))
    for link in links:
        a, b = root(link['a']), root(link['b'])
        if a != b:
            ma, mb = members_by_slice[a], members_by_slice[b]
            conflict = any(supported_levels[na] & supported_levels[nb]
                           for s in ma.keys() & mb.keys() for na in ma[s] for nb in mb[s])
            if conflict:
                ambiguous_links.append({**link, 'ambiguous': True,
                    'reason': 'COMPETING_SAME_SLICE_SUPPORT_INTERVALS'})
                continue
            # Union by size bounds graph extraction cost; final IDs are based
            # on sorted members and remain independent of endpoints.
            if len(ma) < len(mb):
                a, b, ma, mb = b, a, mb, ma
            parent[b] = a
            for s, group in mb.items():
                ma.setdefault(s, []).extend(group)
            del members_by_slice[b]
        accepted.append(link)
    links = accepted
    groups = defaultdict(list)
    for node in nodes:
        groups[root(node)].append(node)
    tracks, membership = [], {}
    ambiguous_degree = defaultdict(int)
    for link in ambiguous_links:
        ambiguous_degree[link['a']] += 1
        ambiguous_degree[link['b']] += 1
    for members in sorted(groups.values()):
        members = sorted(members)
        digest = hashlib.sha256(repr(members).encode()).hexdigest()[:16]
        tid = 'T_'+digest
        for n in members:
            membership[n] = tid
        slices = sorted({n[0] for n in members})
        forked = any(nodes[n]['kind'] == 'FORKED_COMPONENT' for n in members)
        ambiguous_count = sum(ambiguous_degree[n] for n in members)
        ambiguous = forked or ambiguous_count > 0
        tracks.append(dict(track_id=tid, members=members, branch_ids=[n[1] for n in members],
            slice_count=len(slices), slice_span=[slices[0], slices[-1]],
            ambiguous=ambiguous, stable=len(slices) >= 3 and not forked,
            ambiguous_link_count=ambiguous_count,
            kind='CLOSED_TRACK' if all(nodes[n]['kind'] == 'CLOSED_COMPONENT' for n in members) else 'OPEN_SURFACE_TRACK'))
    linked_indices = {hi for link in links for ev in link['evidence'] for hi in ev['hit_indices']}
    linked = [{**observations[i], 'observation_index': i, 'nodes': association[i],
               'evidence': 'SURFACE_LINKED_HORIZONTAL_SUPPORT'} for i in sorted(linked_indices)]
    support = defaultdict(list)
    for link in links:
        for n in (link['a'], link['b']):
            support[n].append(link)
    return dict(nodes=nodes, links=links, weak_links=weak, ambiguous_links=ambiguous_links,
        tracks=tracks, membership=membership,
        support=dict(support), observations=list(observations), linked_observations=linked,
        raw_H_hits=len(observations), surface_linked_H_hits=len(linked_indices),
        ambiguous_hit_count=len(ambiguous_hits), closed_transition_evidence=transitions,
        level_z=level_z, slice_order=order, legacy_endpoint_influence_on_surface_identity=0)
