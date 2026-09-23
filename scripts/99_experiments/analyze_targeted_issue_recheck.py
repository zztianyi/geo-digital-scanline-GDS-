"""Read-only comparison of the explicitly selected historical issue profiles."""
from collections import Counter
import numpy as np
from run_targeted_issue_recheck import OLD,RECENT,SOURCE,read,js,latest
from run_local_conflict_v2 import save_json,manifest,load_inventory,fingerprints
from audit_production_face_consistency import level_tracks,classify


def route(root,s):return read(root/'routes'/f'{s}.pkl')['result']
def layers(root,s):return read(root/'layers'/f'{s}.pkl')['layers']
def info(r):
    a,b=r['route_z_extent'],r['candidate_z_extent'];seq=r['route_branch_sequence']
    return dict(sequence=seq,route_extent=a,candidate_extent=b,missing_extent_m=max(a[0]-b[0],b[1]-a[1],0.),
        repeated_branch=len(seq)!=len(set(seq)),reasons=r.get('unresolved_reasons',[]),joins=len(r['junctions']),
        competitive_over_2mm=sum(j.get('decision_mode')=='COMPETITIVE_SELECTION' and j['xyz_distance_m']>.002+1e-10 for j in r['junctions']))


def missing_details(r,branches):
    valid={v['branch_id'] for v in r['branch_audit'] if v['MBG_pass']};ans=[]
    path=r['path_edges'];endpoints=[np.asarray(path[0]['points_xyz'][0]),np.asarray(path[-1]['points_xyz'][-1])]
    for side in [0,1]:
        extent=r['route_z_extent'];target=r['candidate_z_extent']
        if (extent[0]-target[0] if side==0 else target[1]-extent[1])<1e-5:continue
        point=min(endpoints,key=lambda p:abs(p[2]-extent[side]));found=[]
        for b in branches:
            if b['branch_id'] not in valid:continue
            records=b['records'];z=[p[1] for e in records for p in e['points_uz']]
            if (min(z)>=extent[0]-1e-5 if side==0 else max(z)<=extent[1]+1e-5):continue
            best=None
            for e in records:
                a,c=np.asarray(e['points_xyz']);v=c-a;t=np.clip(np.dot(point-a,v)/max(np.dot(v,v),1e-24),0,1)
                q=a+t*v;dist=float(np.linalg.norm(q-point))
                if best is None or dist<best['distance_m']:
                    uz=np.asarray(e['points_uz']);best=dict(distance_m=dist,edge_id=e['edge_id'],point_uz=(uz[0]+t*(uz[1]-uz[0])).tolist(),t=float(t))
            found.append(dict(branch_id=b['branch_id'],branch_z=[min(z),max(z)],nearest_from_path_end=best))
        ans.append(dict(side='lower' if side==0 else 'upper',route_extent_z=extent[side],candidate_extent_z=target[side],
            path_endpoint_xyz=point.tolist(),candidates=found))
    return ans


def analyze():
    out=latest();selection=js(out/'selection.json');summary=js(out/'run_summary.json');assert summary['code']==fingerprints()
    finished={g['group_id'] for g in summary['groups']}
    completed_groups=[g for g in selection['groups'] if g['group_id'] in finished]
    keys=[s for g in completed_groups for s in g['targets']];inventory=load_inventory(SOURCE,keys)
    profiles=[]
    for s in keys:
        old,new=route(OLD,s),route(out,s)
        profiles.append(dict(s=s,before=info(old),after=info(new),seed=s in selection['seeds'],
            extent_regression_m=max(new['route_z_extent'][0]-old['route_z_extent'][0],old['route_z_extent'][1]-new['route_z_extent'][1],0.),
            missing_details=missing_details(new,inventory[s]),continuation_rejections=new['continuation_rejections']))
    cache=read(OLD/'stages/face_geometry_cache.pkl');m=manifest(SOURCE);events=[];groups=[];tables={}
    for g in completed_groups:
        reg=g['region']
        if reg is None:continue
        labels=cache.get(tuple(reg['bounds']))
        if labels is None:continue
        levels=np.asarray([z for z in m['levels'] if reg['bounds'][4]<=z<=reg['bounds'][5]])
        hits={};counts={v:Counter() for v in ['before','after']}
        for s in g['targets']:
            hits[s]={}
            valid={b['branch_id'] for b in route(out,s)['branch_audit'] if b['MBG_pass']}
            for v,root in [('before',OLD),('after',out)]:
                ls=layers(root,s);hits[s][v]=level_tracks(ls['MAIN_SPINE'],reg,labels,levels)
            ls=layers(out,s);hits[s]['raw']=level_tracks([e for e in ls['MAIN_SPINE']+ls['SIDE_COMPONENTS'] if e.get('branch_id') in valid],reg,labels,levels)
        tables[g['group_id']]=(reg,levels,hits)
        for a,b in zip(g['targets'],g['targets'][1:]):
            cats=[]
            for i,z in enumerate(levels):
                for v in counts:counts[v][classify(hits[a][v][i],hits[b][v][i])]+=1
                absent=[s for s in [a,b] if not hits[s]['after'][i] and hits[s]['raw'][i]]
                c=classify(hits[a]['after'][i],hits[b]['after'][i])
                kind='MAIN_ABSENT_WITH_QUALIFIED_RAW' if absent else ('NEIGHBOR_SOURCE_DIFFERENCE' if c=='IDENTITY_SWITCH' else ('MULTIPLE_SOURCE_HEIGHT' if c=='AMBIGUOUS_MULTIPLE' else None))
                cats.append((kind,tuple(absent)))
            start=0
            while start<len(levels):
                stop=start+1
                while stop<len(levels) and cats[stop]==cats[start]:stop+=1
                kind,absent=cats[start]
                if kind:events.append(dict(group_id=g['group_id'],left=a,right=b,kind=kind,z_min=float(levels[start]),z_max=float(levels[stop-1]),span_m=float(levels[stop-1]-levels[start]),samples=stop-start,missing_profiles=list(absent)))
                start=stop
        groups.append(dict(group_id=g['group_id'],counts={v:dict(c) for v,c in counts.items()},profiles=len(g['targets']),levels=len(levels)))
    comparisons=[]
    for e in selection['events']:
        if e['left'] not in keys or e['right'] not in keys:continue
        group=next(g for g in selection['groups'] if e['left'] in g['targets'] and e['right'] in g['targets'])
        reg,levels,hits=tables[group['group_id']];a,b=e['left'],e['right'];remaining=0;classes=Counter()
        for i,z in enumerate(levels):
            if not e['z_min']-1e-8<=z<=e['z_max']+1e-8:continue
            c=classify(hits[a]['after'][i],hits[b]['after'][i]);classes[c]+=1
            if e['kind']=='MAIN_ABSENT_WITH_QUALIFIED_RAW':present=any(not hits[s]['after'][i] and hits[s]['raw'][i] for s in e['missing_profiles'])
            else:present=c=={'NEIGHBOR_SOURCE_DIFFERENCE':'IDENTITY_SWITCH','MULTIPLE_SOURCE_HEIGHT':'AMBIGUOUS_MULTIPLE'}[e['kind']]
            remaining+=int(present)
        comparisons.append(dict(e,remaining_samples=remaining,after_classes=dict(classes),diagnostic_disappeared=remaining==0))
    frozen=js(out/'frozen_inventory.json');reused_missing=[p['s'] for p in frozen['profiles'] if p['current_code'] and p['missing_extent_m']>1e-5]
    save_json(out/'reused_current_events.json',[e for e in frozen['events'] if e['current_code'] and e['span_m']>.100001])
    reused_raw=load_inventory(SOURCE,reused_missing)
    reused_diagnostics=[dict(s=s,info=info(route(RECENT,s)),missing_details=missing_details(route(RECENT,s),reused_raw[s])) for s in reused_missing]
    audits=[js(out/'audits'/f'{g["group_id"]}.json') for g in completed_groups]
    bandrows=[dict(group_id=d['group']['group_id'],**r) for d in audits for r in d['bands']]
    save_json(out/'recheck_analysis.json',dict(profiles=profiles,events_after=events,old_event_comparison=comparisons,groups=groups,reused_missing=reused_diagnostics,
        bands=bandrows,counts=dict(profiles=len(profiles),seed_profiles=sum(p['seed'] for p in profiles),
            missing_before=sum(p['before']['missing_extent_m']>1e-5 for p in profiles),missing_after=sum(p['after']['missing_extent_m']>1e-5 for p in profiles),
            repeated_before=sum(p['before']['repeated_branch'] for p in profiles),repeated_after=sum(p['after']['repeated_branch'] for p in profiles),
            extent_regressions=sum(p['extent_regression_m']>1e-5 for p in profiles),
            old_events=len(comparisons),old_events_disappeared=sum(e['diagnostic_disappeared'] for e in comparisons)),
        code=summary['code']))
    print('COUNTS',js(out/'recheck_analysis.json')['counts'])
    print('REMAINING EVENTS',Counter((e['kind'],e['span_m']>.100001) for e in events))
    print('BANDS',Counter((r.get('applied',False),r.get('reason')) for r in bandrows))
    for p in profiles:
        if p['after']['missing_extent_m']>1e-5 or p['extent_regression_m']>1e-5:print('EXTENT',p['s'],p['before']['missing_extent_m'],p['after']['missing_extent_m'],p['extent_regression_m'],p['missing_details'])


if __name__=='__main__':analyze()
