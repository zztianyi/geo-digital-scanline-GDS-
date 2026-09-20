"""Readable final-route neighborhood figures for the 1.5D census."""
from __future__ import annotations
import json, pickle
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from census_directional_consistency import OUT, ROOT, save_json, load_branches
from review_directional_consistency import NAMES

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':11})
FIG=OUT/'figures'


def route(key):
    with (OUT/'routes'/f'{key}.pkl').open('rb') as f:d=pickle.load(f)
    normalized=OUT/'normalized_crossings'/f'{key}.npy'
    if normalized.exists():d['crossings']=np.load(normalized)
    return d


def pick_cases(windows, keys):
    chosen=[]
    def add(w,label=None):
        z=w.get('display_center_z',(w['z_min']+w['z_max'])/2)
        if any(abs(float(w['left'])-float(c['left']))<1. and abs(z-c['center_z'])<4. for c in chosen):return
        chosen.append(dict(w,center_z=z,case_id=len(chosen)+1,label=label or NAMES[w['category']]))
    special=[w for w in windows if w['left']=='0.85' and w['category']=='MBG_ASYMMETRY' and w['persistent']]
    if special:
        w=min(special,key=lambda w:max(w['z_min']-1403.386,0,1403.386-w['z_max']))
        add(dict(w,display_center_z=min(w['z_max'],max(w['z_min'],1403.386))),'零长自环使长分支被排除')
    for category in ('SURFACE_CHOICE','DETAIL_COUNT','COVERAGE_GAP','MBG_ASYMMETRY'):
        matches=[w for w in windows if w['category']==category and w['persistent']]
        for w in matches:
            before=len(chosen);add(w)
            if len(chosen)>before:break
    geometry=json.loads((OUT/'geometry_complexity_windows.json').read_text(encoding='utf-8'))
    count=0
    for w in geometry:
        if count>=2 or len(chosen)>=8:break
        before=len(chosen);add(w)
        if len(chosen)>before:count+=1
    for w in windows:
        if w['persistent'] and len(chosen)<8 and all(abs(float(w['left'])-float(c['left']))>6 for c in chosen):add(w)
    # Historical neighborhoods are explicitly rechecked, whether flagged or not.
    for s,z in ((104.30,1385.20318),(122.25,1411.08077),(89.75,1472.)):
        key=min(keys,key=lambda k:abs(float(k)-s));i=keys.index(key)
        chosen.append(dict(left=key,right=keys[min(i+1,len(keys)-1)],pair_index=i,
            category='HISTORICAL',center_z=z,z_min=z-2,z_max=z+2,case_id=len(chosen)+1,
            label='上一轮案例邻域复核',levels=None,event_count=None,persistent=None))
    return chosen


def figure_case(case, m):
    keys=m['keys'];center=keys.index(case['left']);ids=range(max(0,center-2),min(len(keys),center+3))
    localkeys=[keys[i] for i in ids];branches=load_branches(localkeys);data={k:route(k) for k in localkeys}
    zc=case['center_z'];zlo,zhi=zc-3.,zc+3.
    li=int(np.argmin(abs(np.asarray(m['levels'])-zc)))
    center_evidence=[]
    for key in (case['left'],case['right']):
        d=data.get(key) or route(key);a=d['crossings'];a=a[a[:,0]==li]
        center_evidence.append(dict(slice_key=key,level_index=li,z=m['levels'][li],
            crossings=[dict(H_run=int(row[1]),u=float(row[2]),branch_id=int(row[4]),
                unique=bool(row[3]),MBG=bool(row[8]),selected=bool(row[9])) for row in a]))
    case['center_evidence']=center_evidence
    # Include both the suspect correspondence and the actual selected alternative.
    us=[]
    for k,d in data.items():
        a=d['crossings'];in_z=(np.asarray(m['levels'])[a[:,0].astype(int)]>=zlo)&(np.asarray(m['levels'])[a[:,0].astype(int)]<=zhi)
        us.extend(a[in_z&(a[:,9]==1),2].tolist())
    # Bounds must come from THIS displayed height slab, never the possibly
    # 50 m-tall parent review window. Include unselected matched alternatives.
    if case['left'] in data and case['right'] in data:
        pair=[]
        for key in (case['left'],case['right']):
            a=data[key]['crossings'];zz=np.asarray(m['levels'])[a[:,0].astype(int)]
            pair.append({(int(row[0]),int(row[1])):row for row in a[(zz>=zlo)&(zz<=zhi)&(a[:,3]==1)]})
        for h in pair[0].keys()&pair[1].keys():
            if bool(pair[0][h][9])!=bool(pair[1][h][9]):us.extend([pair[0][h][2],pair[1][h][2]])
    if not us:
        us=[u for k in localkeys for b in branches[k] for u,z in b['points_uz'] if zlo<=z<=zhi]
    ulo,uhi=(min(us)-.5,max(us)+.5) if us else (-2.,2.)
    if uhi-ulo<2.5:
        uc=(ulo+uhi)/2;ulo,uhi=uc-1.25,uc+1.25
    fig,axes=plt.subplots(1,len(localkeys),figsize=(19,7.5),sharex=True,sharey=True)
    connector=False;case_rows=[]
    for ax,k in zip(np.atleast_1d(axes),localkeys):
        d=data[k];r=d['result'];visible=[];length=0.;backward=0.
        raw=[np.asarray(b['points_uz']) for b in branches[k]]
        ax.add_collection(LineCollection(raw,colors='#bac1c9',linewidths=1.,zorder=1))
        obs=[];syn=[]
        for e in r['path_edges']:
            p=np.asarray(e['points_uz'])
            if max(p[:,1])<zlo or min(p[:,1])>zhi:continue
            if e['source'].startswith('OBSERVED'):
                obs.append(p);visible.append(e['branch_id'])
                dz=p[1,1]-p[0,1]
                low,high=(0.,1.) if abs(dz)<1e-12 else (max(0.,min((zlo-p[0,1])/dz,(zhi-p[0,1])/dz)),min(1.,max((zlo-p[0,1])/dz,(zhi-p[0,1])/dz)))
                fraction=max(0.,high-low)
                length+=np.linalg.norm(p[1]-p[0])*fraction
                backward+=max(0.,-dz)*fraction
            else:syn.append(p);connector=True
        ax.add_collection(LineCollection(obs,colors='#1465ad',linewidths=2.6,zorder=3))
        if syn:ax.add_collection(LineCollection(syn,colors='#d77720',linewidths=2.8,linestyles='dashed',zorder=4))
        a=d['crossings'];zz=np.asarray(m['levels'])[a[:,0].astype(int)]
        keep=(zz>=zlo)&(zz<=zhi)&(a[:,2]>=ulo)&(a[:,2]<=uhi)
        # Show every fifth measured H level for readability, never alter calculation.
        keep&=(a[:,0].astype(int)%5==0)
        ax.scatter(a[keep,2],zz[keep],s=27,facecolors='none',edgecolors='#218a58',linewidths=1,zorder=5)
        seq=list(dict.fromkeys(visible))
        ax.set_title(f"s = {float(k):.2f} m"+('（目标）' if k==case['left'] else '')+f"\n局部保留分支：{','.join(map(str,seq)) or '无'}",fontsize=11)
        ax.set_xlim(ulo,uhi);ax.set_ylim(zlo,zhi);ax.grid(alpha=.2);ax.set_xlabel('径向偏移 u（m）')
        case_rows.append(dict(slice_key=k,local_branches=seq,local_observed_length_m=length,
                              local_reverse_z_m=backward,full_sequence=r['route_branch_sequence']))
    axes[0].set_ylabel('高程 z（m）')
    handles=[Line2D([],[],color='#bac1c9',label='原始纵向曲线',lw=1.3),
        Line2D([],[],color='#1465ad',label='最终主轨（实测保留部分）',lw=2.6),
        Line2D([],[],ls='',marker='o',mfc='none',mec='#218a58',label='原始横向交点（显示间隔 0.5 m）')]
    if connector:handles.append(Line2D([],[],color='#d77720',ls='--',label='算法连接段（≤10 mm）'))
    fig.legend(handles=handles,loc='lower center',ncol=len(handles),bbox_to_anchor=(.5,.025))
    fig.suptitle(f"案例 {case['case_id']:02d}｜{case['label']}｜相邻测线实际最终主轨",fontsize=18)
    fig.text(.5,.092,f'{len(localkeys)} 个面板使用相同局部坐标；分支编号仅在本测线内有效；灰线不是最终识别结果。',ha='center',fontsize=11)
    fig.subplots_adjust(left=.06,right=.985,top=.84,bottom=.2,wspace=.08)
    name=f"case_{case['case_id']:02d}_neighbors.png";fig.savefig(FIG/name,dpi=155);plt.close(fig)
    case.update(figure=name,bounds=[ulo,uhi,zlo,zhi],neighbor_rows=case_rows)
    horizontal_figure(case,m)


def horizontal_figure(case,m):
    keys=m['keys'];i=keys.index(case['left']);lo=max(0,i-20);hi=min(len(keys),i+21)
    localkeys=keys[lo:hi];data=[route(k) for k in localkeys]
    levels=np.asarray(m['levels']);base=int(np.argmin(abs(levels-case['center_z'])))
    lis=sorted({max(0,min(len(levels)-1,base+d)) for d in (-5,0,5)})
    selected_u=[r[2] for d in data for r in d['crossings'] if int(r[0]) in lis and r[9]]
    hu=(min([case['bounds'][0],*selected_u])-.1,max([case['bounds'][1],*selected_u])+.1)
    case['horizontal_u_bounds']=hu
    fig,axes=plt.subplots(1,len(lis),figsize=(18,5.7),sharex=True,sharey=True)
    for ax,li in zip(np.atleast_1d(axes),lis):
        runs={};raw=[];selected=[]
        for k,d in zip(localkeys,data):
            a=d['crossings'];a=a[a[:,0]==li]
            for r in a:
                raw.append([float(k),r[2]])
                if r[9]:selected.append([float(k),r[2]])
                if r[3]:runs.setdefault(int(r[1]),[]).append([float(k),r[2]])
        for pid,p in runs.items():
            p=np.asarray(p);breaks=np.flatnonzero(np.diff(p[:,0])>.050001)+1
            for part in np.split(p,breaks):
                if len(part)>1:ax.plot(part[:,0],part[:,1],color='#8db69d',lw=1,zorder=1)
        if raw:
            p=np.array(raw);ax.scatter(*p.T,s=16,facecolors='none',edgecolors='#218a58',zorder=2)
        if selected:
            p=np.array(selected);ax.scatter(*p.T,s=26,color='#1465ad',zorder=3)
        ax.axvline(float(case['left']),color='#60656c',ls=':',lw=1)
        ax.set_title(f'横向高程 z = {levels[li]:.3f} m');ax.set_xlabel('纵向测线位置 s（m）')
        ax.set_xlim(float(localkeys[0]),float(localkeys[-1]));ax.set_ylim(hu);ax.grid(alpha=.18)
    axes[0].set_ylabel('径向偏移 u（m）')
    fig.suptitle(f"案例 {case['case_id']:02d}｜横向对应与左右 ±1 m 主轨选择",fontsize=17)
    fig.legend(handles=[Line2D([],[],color='#8db69d',marker='o',mfc='none',mec='#218a58',label='同一原始 H 曲线交点及对应关系'),
        Line2D([],[],ls='',marker='o',color='#1465ad',label='各测线最终主轨保留的交点')],ncol=2,loc='lower center',bbox_to_anchor=(.5,.045))
    fig.text(.5,.015,'浅绿连线仅表示同一 H 曲线的测线交点对应，不是补线；蓝点连续表示选择相容，跳行或断列仅是复核线索。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.065,right=.98,top=.83,bottom=.21,wspace=.12)
    name=f"case_{case['case_id']:02d}_horizontal.png";fig.savefig(FIG/name,dpi=150);plt.close(fig)
    case['horizontal_figure']=name


def overview(cases,m):
    a=np.load(OUT/'review_mask.npz');mask=a['mask'];keys=m['keys'];levels=m['levels']
    fig,ax=plt.subplots(figsize=(16,9))
    background=np.zeros_like(mask,dtype=bool)
    for i,key in enumerate(keys[:-1]):
        d=route(key);c=d['crossings'];background[i,c[c[:,3]==1,0].astype(int)]=True
    ii,jj=np.where(background);ax.scatter(np.asarray(keys,dtype=float)[ii],np.asarray(levels)[jj],s=.08,color='#d7dfe6',rasterized=True)
    ii,jj=np.where(a['gate_mask']);ax.scatter(np.asarray(keys,dtype=float)[ii],np.asarray(levels)[jj],s=1.1,color='#d59b27',rasterized=True)
    ii,jj=np.where(mask);ax.scatter(np.asarray(keys,dtype=float)[ii],np.asarray(levels)[jj],s=1.1,color='#d05b37',rasterized=True)
    complexity=json.loads((OUT/'geometry_complexity_windows.json').read_text(encoding='utf-8'))
    if complexity:ax.scatter([float(w['left']) for w in complexity],[(w['z_min']+w['z_max'])/2 for w in complexity],
        marker='x',s=7,linewidths=.6,color='#8053b4',alpha=.8,zorder=3)
    for c in cases:
        s=float(c['left']);z=c['center_z'];ax.scatter(s,z,s=45,facecolors='white',edgecolors='#202b34',zorder=4)
        ax.annotate(str(c['case_id']),xy=(s,z),xytext=(5,5),textcoords='offset points',fontsize=11,weight='bold',zorder=5)
    ax.set_xlabel('测线位置 s（m）');ax.set_ylabel('高程 z（m）');ax.grid(alpha=.15)
    ax.set_title('全区域最终主轨一致性复核位置｜相邻 0.05 m 测线',fontsize=18,pad=15)
    ax.legend(handles=[Line2D([],[],ls='',marker='.',color='#b8c6d2',label='原始 H/V 可唯一对应的位置'),
        Line2D([],[],ls='',marker='.',color='#d05b37',label='最终选择不一致，连续 ≥3 个 H 层'),
        Line2D([],[],ls='',marker='.',color='#d59b27',label='分支门槛两侧不同，连续 ≥3 层'),
        Line2D([],[],ls='',marker='x',color='#8053b4',label='H 基本相容的局部复杂度变化'),
        Line2D([],[],ls='',marker='o',mfc='white',mec='#202b34',label='代表案例编号')],loc='lower left')
    fig.tight_layout();fig.savefig(FIG/'whole_area_review.png',dpi=170);plt.close(fig)


def render():
    FIG.mkdir(exist_ok=True)
    m=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    windows=json.loads((OUT/'local_review_windows.json').read_text(encoding='utf-8'))
    cases=pick_cases(windows,m['keys'])
    for case in cases:figure_case(case,m);print('FIG',case['case_id'],case['left'],flush=True)
    overview(cases,m);save_json(OUT/'representative_cases.json',cases)


if __name__=='__main__':render()
