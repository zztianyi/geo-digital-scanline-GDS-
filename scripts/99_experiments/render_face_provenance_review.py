"""Small local figures separating measured routes from shadow face identity."""
from pathlib import Path
from collections import Counter
import argparse,json,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from validate_physical_topology import output_path,load_inventory,PRIOR
from run_face_provenance_validation import manifest
from locc_data import opc

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
COLORS=['#238a68','#c55042','#8255a5','#d29329']
BLUE='#1567ae';GRAY='#c4c9ce'


def raw(ax,branches):
    ax.add_collection(LineCollection([b['points_uz'] for b in branches],colors=GRAY,lw=.9,zorder=1))


def route(ax,result):
    for observed,color,lw in [(True,BLUE,2.3),(False,'#df891d',2.5)]:
        rows=[r['points_uz'] for r in result['path_edges'] if r['source'].startswith('OBSERVED')==observed]
        if rows:ax.add_collection(LineCollection(rows,colors=color,lw=lw,zorder=4,linestyles='solid' if observed else 'dashed'))


def axes_setup(ax,bounds):
    ax.set_xlim(bounds[2],bounds[3]);ax.set_ylim(bounds[4],bounds[5]);ax.grid(alpha=.18);ax.set_xlabel('径向偏移 u（m）')


def selected_tracks(members,key):
    return sorted({t for e in members if e['s']==key and e['selected_current_route'] for t in e['face_track_ids']})


def render_region(out,data,m):
    r=data['region'];rid=r['region_id'];keys=r['target_keys'];focus=r['focus_s'];b=r['bounds']
    bs=load_inventory(out,keys)
    with (out/'local_routes'/f'{rid}.pkl').open('rb') as f:routes=pickle.load(f)
    current=routes['CURRENT_ROUTE'];counts=Counter(t for e in data['members'] for t in e['face_track_ids'])
    important=counts.most_common(4);colors={tid:COLORS[i] for i,(tid,_) in enumerate(important)}
    fig,axes=plt.subplots(1,4,figsize=(19,6.5),gridspec_kw={'width_ratios':[1,1,1,1.12]})
    for ax in axes[:3]:axes_setup(ax,b)
    raw(axes[0],bs[focus]);axes[0].set_title('① 原始候选\n全部为未裁剪的实测曲线')
    raw(axes[1],bs[focus]);route(axes[1],current[focus]);chosen=selected_tracks(data['members'],focus)
    axes[1].set_title('② 拓扑修复后的实际主轨\n局部来源：'+(', '.join(chosen) or '无'))
    raw(axes[2],bs[focus])
    for tid,color in colors.items():
        rows=[e['points_uz'] for e in data['members'] if e['s']==focus and e['face_track_ids']==[tid]]
        axes[2].add_collection(LineCollection(rows,colors=color,lw=2,zorder=3))
    hints=[h for h in data['shadow'] if h['s']==focus and h['level_index']%5==0]
    if hints:axes[2].scatter([h['u'] for h in hints],[h['z'] for h in hints],s=48,facecolors='none',edgecolors='#222222',lw=1,zorder=5)
    axes[2].set_title('③ 局部面来源轨道\n黑圈：有邻线支持、当前未保留')
    axes[0].set_ylabel('高程 z（m）')
    li=int(np.argmin(abs(np.array(m['levels'])-r['center_z'])))
    switch_levels=sorted({v['level_index'] for v in data['VV'] if v['classification']=='FACE_TRACK_IDENTITY_SWITCH'})
    if switch_levels:li=switch_levels[len(switch_levels)//2]
    ax=axes[3];drawn=set();track_y={tid:i for i,tid in enumerate(colors)}
    for h in data['VH']:
        if h['level_index']!=li:continue
        for tid in h['H_face_tracks']:
            signature=(h['s'],tid)
            if signature in drawn or tid not in track_y:continue
            drawn.add(signature)
            ax.scatter(float(h['s']),track_y[tid],s=85,facecolors='none',edgecolors=colors[tid],lw=2,zorder=2)
    selected={(h['s'],tid):(float(h['s']),track_y[tid]) for h in data['VH'] if h['level_index']==li and h['selected_current_route'] for tid in h['H_face_tracks'] if tid in track_y}
    if selected:
        p=np.array(list(selected.values()));ax.scatter(p[:,0],p[:,1],s=27,color=BLUE,zorder=4)
    ax.set_title(f'④ 横向来源身份（类别轴）\nz = {m["levels"][li]:.3f} m；蓝实心为已选')
    ax.set_xlabel('邻线位置 s（m）');ax.set_ylabel('局部 FaceTrack（类别，非距离）');ax.grid(alpha=.18)
    ax.set_yticks(list(track_y.values()),list(track_y));ax.set_ylim(-.55,max(track_y.values(),default=0)+.55)
    ax.set_xticks([float(k) for k in keys]);ax.tick_params(axis='x',rotation=35)
    handles=[Line2D([],[],color=color,lw=3,label=tid) for tid,color in colors.items()]
    fig.legend(handles=handles,ncol=max(1,len(handles)),loc='lower center',bbox_to_anchor=(.5,.075))
    fig.suptitle(f'案例 {rid}｜s = {float(focus):.2f} m｜几何、实际选择与面来源分开看',fontsize=17)
    fig.text(.5,.035,'灰色为原始候选，蓝色为实际主轨；颜色仅在本案例有效。第④栏圆圈表示候选来源，纵轴为类别，不表示几何间距或交点数量。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.055,right=.98,top=.81,bottom=.21,wspace=.24)
    fig.savefig(out/'figures/face_tracks'/f'{rid}_face_tracks.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(1,len(keys),figsize=(19,7),sharex=True,sharey=True)
    for ax,key in zip(np.atleast_1d(axes),keys):
        axes_setup(ax,b);raw(ax,bs[key]);route(ax,current[key])
        tids=selected_tracks(data['members'],key)
        ax.set_title(f's = {float(key):.2f} m'+('（重点）' if key==focus else '')+'\n来源：'+(', '.join(tids) or '无'),fontsize=10)
    axes[0].set_ylabel('高程 z（m）')
    fig.suptitle(f'案例 {rid}｜相邻测线各自实际主轨｜面来源编号在本组内可比较',fontsize=17)
    fig.legend(handles=[Line2D([],[],color=GRAY,lw=1,label='原始候选'),Line2D([],[],color=BLUE,lw=2.5,label='实际主轨（实测部分）'),
        Line2D([],[],color='#df891d',lw=2,label='算法连接段（≤10 mm）')],ncol=3,loc='lower center',bbox_to_anchor=(.5,.045))
    fig.text(.5,.095,'同一 FaceTrack 只证明局部网格来源连通，不代表所有折返均相同；未执行 FaceTrack 联合选轨或平滑。',ha='center')
    fig.subplots_adjust(left=.06,right=.985,top=.81,bottom=.21,wspace=.1)
    fig.savefig(out/'figures/representative'/f'{rid}_neighbors.png',dpi=155);plt.close(fig)


def render_topology(out,data,m):
    r=data['region'];key=r['focus_s'];rid=r['region_id'];bounds=r['bounds'];bs=load_inventory(out,[key])[key]
    with (out/'local_routes'/f'{rid}.pkl').open('rb') as f:rr=pickle.load(f)
    with (out/'affected_canonical_profiles'/f'{key}.pkl').open('rb') as f:p=pickle.load(f)
    node=int(p.edges[np.flatnonzero(~p.topology_active_edge)[0],0]);uz=opc.to_suz(p.nodes,m['arc'])[:,1:]
    fig,axes=plt.subplots(1,3,figsize=(16,7))
    lines=uz[p.edges];axes[0].add_collection(LineCollection(lines,colors=GRAY,lw=2))
    axes[0].scatter(*uz[node],s=60,color='#222222',zorder=4)
    axes[0].set_xlim(uz[node,0]-.3,uz[node,0]+.3);axes[0].set_ylim(uz[node,1]-.5,uz[node,1]+.5);axes[0].grid(alpha=.2)
    axes[0].set_title(f'自环节点单独放大\ncanonical degree {p.degree[node]} → physical degree {p.physical_degree[node]}')
    for ax,name,title in zip(axes[1:],['BEFORE_LOCAL','CURRENT_ROUTE'],['修复前：同条件局部重算','修复后：同条件局部重算']):
        axes_setup(ax,bounds);raw(ax,bs);route(ax,rr[name][key]);ax.set_title(title)
    for ax in axes:ax.set_xlabel('径向偏移 u（m）');ax.set_ylabel('高程 z（m）')
    fig.suptitle(f'物理拓扑修复｜s = {float(key):.2f} m｜零长记录保留，物理度数不再计入',fontsize=16)
    fig.text(.5,.055,'左栏是自环所在位置，使用独立坐标范围；右两栏统一坐标，灰色原始几何不变，蓝色为各版实际主轨。',ha='center')
    fig.subplots_adjust(left=.065,right=.98,bottom=.17,top=.81,wspace=.25)
    fig.savefig(out/'figures/topology'/f'{key}_before_after.png',dpi=160);plt.close(fig)


def render(out):
    m=manifest(out)
    for folder in ['topology','face_tracks','representative']:(out/'figures'/folder).mkdir(parents=True,exist_ok=True)
    for r in json.loads((out/'validation_regions.json').read_text(encoding='utf-8')):
        with (out/'region_data'/f'{r["region_id"]}.pkl').open('rb') as f:data=pickle.load(f)
        render_region(out,data,m)
        if r['region_id'] in ('A','B'):render_topology(out,data,m)
        print('FIGURE',r['region_id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=output_path());render(p.parse_args().output)
