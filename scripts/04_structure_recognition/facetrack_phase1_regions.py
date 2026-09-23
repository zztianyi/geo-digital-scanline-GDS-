"""Fixed observed-cell conflict masks, independent of votes and routes."""


def build_regions(cells,positions,cell_m):
    cells=[c for c in cells if len(set(c['tracks']))>=2]
    cells.sort(key=lambda c:(c['s_index'],c['u_bin'],c['z_bin']))
    parents=list(range(len(cells)));lookup={};sets=[]
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,c in enumerate(cells):
        key=(c['s_index'],c['u_bin'],c['z_bin']);tracks=tuple(sorted(set(c['tracks'])));sets.append(tracks)
        for ds in (-1,0):
            # Only actual consecutive source profiles connect across s.
            if ds and key[0]>0 and positions[key[0]]-positions[key[0]-1]>.050001:continue
            for du in (-1,0,1):
                for dz in (-1,0,1):
                    j=lookup.get((key[0]+ds,key[1]+du,key[2]+dz))
                    if j is not None and sets[j]==tracks:parents[root(i)]=root(j)
        lookup[key]=i
    groups={}
    for i in range(len(cells)):groups.setdefault(root(i),[]).append(i)
    regions=[]
    for ids in sorted(groups.values(),key=lambda ids:(cells[ids[0]]['s_index'],cells[ids[0]]['u_bin'],cells[ids[0]]['z_bin'])):
        selected=[cells[i] for i in ids];ss=sorted({c['s_index'] for c in selected});us=[c['u_bin'] for c in selected];zs=[c['z_bin'] for c in selected]
        members={t:set() for t in sets[ids[0]]}
        for c in selected:
            for bid,t in zip(c['branch_ids'],c['branch_tracks']):members[t].add((c['s_index'],bid))
        rid=f'R{len(regions)+1:05d}'
        regions.append(dict(region_id=rid,tracks=list(sets[ids[0]]),s_indices=ss,member_V=[positions[i] for i in ss],
            cells=[[c['s_index'],c['u_bin'],c['z_bin']] for c in selected],
            bounds=[positions[ss[0]],positions[ss[-1]],min(us)*cell_m,(max(us)+1)*cell_m,min(zs)*cell_m,(max(zs)+1)*cell_m],
            members={t:sorted(m) for t,m in members.items()},boundary='FIXED_OBSERVED_COOCCURRENCE_CELL_MASK'))
    return regions
