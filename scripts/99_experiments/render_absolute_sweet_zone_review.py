"""Small legends, explicit evidence panels, and traceable real/controlled cases."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from review_absolute_sweet_zone import ROOT,OUT,save_json
sys.path.insert(0,str(ROOT/'tests'))
from test_dominant_observed_branch import fixture
from horizontal_surface_link import horizontal_path_observations
from horizontal_surface_link import branch_crossings
from observed_surface_graph import build_surface_graph
from surface_track_selection import select_surface_track
from surface_track_handoff import select_handoff
from branch_absolute_core import branch_metrics,local_edge_scale

plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})
BLUE='#1768ac';ORANGE='#d88724';GREEN='#16845b';GRAY='#b6bdc7';RED='#be3b3b'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)


def measured_scene(paths,*,unsupported=(),neighbor_paths=None):
    neighbor_paths=neighbor_paths or paths
    profiles=[]
    for s,curves in ((-1.,neighbor_paths),(0.,paths),(1.,neighbor_paths)):
        p,uz,bs=fixture(curves)
        profiles.append(dict(s=s,branches=bs))
        if s==0:target=(p,uz,bs)
    horizontal=[]
    for z in np.linspace(.025,.975,39):
        runs=[]
        for i,p in enumerate(paths):
            if i in unsupported or z<min(p[:,1]) or z>max(p[:,1]):continue
            q=neighbor_paths[i]
            runs.append(dict(h_path_id=i,points_su=[[-1.,float(np.interp(z,q[:,1],q[:,0]))],
                [0.,float(np.interp(z,p[:,1],p[:,0]))],[1.,float(np.interp(z,q[:,1],q[:,0]))]]))
        horizontal.append(dict(z=float(z),paths=runs))
    observations=horizontal_path_observations(horizontal,[-1.,0.,1.])
    return target,build_surface_graph(profiles,observations,slice_order=[-1.,0.,1.])


def controlled_cases():
    z=np.linspace(0,1,81)
    za=np.r_[np.linspace(0,.1,5),np.linspace(.1,.6,33)[1:],np.linspace(.6,1,5)[1:]]
    zb=np.r_[np.linspace(0,.4,5),np.linspace(.4,.9,33)[1:],np.linspace(.9,1,5)[1:]]
    guard_paths=[np.c_[np.zeros(len(za)),za],np.c_[np.full(len(zb),.005),zb]]
    cases=[('A','短分支没有绝对甜区',[np.c_[np.zeros(9),np.linspace(0,1,9)],np.c_[np.full(81,.02),z]],(.25,.75),{},None),
        ('B','同获 H/V 支持：核心优于端点保护区',guard_paths,(.72,.86),{},0),
        ('C','均在核心：只比较已关联邻面的重复细节',[np.c_[.003*np.sin(25*z),z],np.c_[.02+.006*np.sin(25*z),z]],(.3,.7),
            {'neighbor_paths':[np.c_[.003*np.cos(25*z),z],np.c_[.02+.006*np.sin(25*z),z]]},None),
        ('D','细节丰富但缺少同路径 H 关联',[np.c_[np.zeros(81),z],np.c_[.02+.006*np.sin(35*z),z]],(.3,.7),{'unsupported':(1,)},None),
        ('E','短的局部绕行不能形成 A→B→A',[np.c_[np.zeros(81),z],np.c_[np.full(7,.006),np.linspace(.35,.65,7)]],(.4,.6),{},0),
        ('G','核心与细节接近：保持当前分支',[np.c_[.004*np.sin(25*z),z],np.c_[.02+.004*np.sin(25*z),z]],(.3,.7),{},0),
        ('H','当前端点区→候选核心：再找短连接',guard_paths,(.72,.86),{},0)]
    reviews=[]
    for cid,title,paths,region,kwargs,current in cases:
        target,g=measured_scene(paths,**kwargs);p,uz,bs=target
        decision=select_surface_track(g,0.,target_z=region,current_branch_id=current)
        route=select_handoff(g,(0.,0)) if cid=='H' else None
        selected=decision['selected'];sid=selected['branch_id'] if selected else None
        fig,axes=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1.2,1]})
        ax=axes[0];ax.axhspan(*region,color='#e9edf2',zorder=0)
        for b in bs:
            m=branch_metrics(b,local_edge_scale(bs),target_z=region);pts=b['points_uz'];color=BLUE if b['branch_id']==sid else ORANGE
            ax.plot(*pts.T,color=color,alpha=.65,lw=1.2)
            if m['ASC_exists']:
                a,c=m['left_guard_node'],m['right_guard_node'];ax.plot(*pts[a:c+1].T,color=color,lw=4)
            ax.scatter(pts[[0,-1],0],pts[[0,-1],1],s=18,color=color)
            ax.annotate('B'+str(b['branch_id']),pts[len(pts)//2],xytext=(7,0),textcoords='offset points',color=color)
        if route:
            for r in route['path_edges']:
                if r['source']=='TOPOLOGY_SWITCH':ax.plot(*np.asarray(r['points_uz']).T,color=RED,lw=3)
            j=route['junction'];ax.scatter(*j['a_point_uz'],color=RED,s=40,zorder=8)
            ax.annotate(f"实际连接 {1000*j['xyz_distance_m']:.1f} mm",j['a_point_uz'],xytext=(20,14),textcoords='offset points',color=RED)
        ax.set(xlabel='径向偏移 u / m（横纵轴独立缩放）',ylabel='高程 z / m',title='候选几何：粗线为甜区，细线为端点保护段')
        ax.legend(handles=[Line2D([],[],color=BLUE,lw=3,label='选中候选'),Line2D([],[],color=ORANGE,lw=3,label='其他候选'),Line2D([],[],color='#c8d0da',lw=8,label='比较区域')],loc='upper center',ncol=3,fontsize=9)
        ax=axes[1]
        for b in bs:
            bid=b['branch_id'];pts=b['points_uz'];color=BLUE if bid==sid else ORANGE
            ax.plot(*pts.T,color=color,lw=2)
            if g['support'].get((0.,bid)):
                neighbor=g['nodes'][(-1.,bid)]['points_uz'];ax.plot(*neighbor.T,color=color,lw=1,ls='--',alpha=.7)
        for h in g['observations']:
            if h['s']==0:ax.scatter(h['u'],h['z'],s=11,color=GREEN,zorder=6)
        ax.set(xlabel='径向偏移 u / m',ylabel='高程 z / m',title='可复核证据：同一 H 路径交点 + 相邻 V 曲线')
        ax.legend(handles=[Line2D([],[],marker='o',color=GREEN,ls='',label='H 实际交点'),Line2D([],[],color=GRAY,ls='--',label='相邻 V（同表面图关联）')],loc='upper center',fontsize=9)
        for ax in axes:ax.grid(alpha=.18);ax.set_ylim(-.06,1.08)
        fig.suptitle(f'{cid}｜{title}（受控几何，不是模型准确率样本）',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.94));name=f'controlled_{cid}.png';fig.savefig(FIG/name,dpi=170);plt.close(fig)
        rows=[{k:r.get(k) for k in ('branch_id','canonical_node_count','full_arc_length','K_guard','left_guard_arc_length','right_guard_arc_length','ASC_exists','ASC_node_count','ASC_arc_length','target_region_in_ASC_fraction','MBG_pass','surface_support_tier','detail_evaluated','neighbor_detail_score','reason')} for r in decision['candidates']]
        reviews.append(dict(case=cid,title=title,figure='figures/'+name,selected=sid,stage=decision['decision_stage'],
            ambiguous=decision['ambiguous'],branches=rows,switch_count=route['branch_switch_count'] if route else 0,
            connector_m=route['connector_length_m'] if route else 0,R_syn=route['junction']['R_syn'] if route else None))
    save_json(OUT/'controlled_case_results.json',reviews)
    return reviews


def load_result(key):
    index=json.loads((OUT/'main_track_index.json').read_text(encoding='utf-8'))
    with (OUT/'main_tracks.pkl').open('rb') as f:f.seek(index[key]);return pickle.load(f)


def draw_records(ax,result,*,observed=BLUE,synthetic=RED):
    for source,color,style in ((True,observed,'-'),(False,synthetic,'--')):
        lines=[r['points_uz'] for r in result['path_edges'] if r['source'].startswith('OBSERVED')==source]
        ax.add_collection(LineCollection(lines,colors=color,linewidths=2 if source else 2.5,linestyles=style))


def real_cases():
    prior=ROOT/'outputs/reconstruction_constraint_review/20260920'
    with (prior/'local_evidence.pkl').open('rb') as f:evidence=pickle.load(f)
    with (prior/'case5_unfiltered_evidence.pkl').open('rb') as f:raw=pickle.load(f)
    reviews=[]
    for key,bounds in (('89.75',[-71,-53,1472.5,1479]),('104.30',[19.6,22.2,1377,1388])):
        case=evidence['cases'][key];new=load_result(key);old=case['payload']['result']
        fig,axes=plt.subplots(1,3,figsize=(17,7),sharey=True)
        for ax in axes:
            for b in case['branches']:ax.plot(*np.asarray(b['points_uz']).T,color=GRAY,lw=.9,zorder=1)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.grid(alpha=.18);ax.set_xlabel('径向偏移 u / m')
        axes[0].set_ylabel('高程 z / m')
        draw_records(axes[0],old,observed=ORANGE);axes[0].set_title('上一轮主轨：橙色 observed，红虚线连接')
        draw_records(axes[1],new['result']);axes[1].set_title('本轮主轨：蓝色 observed；缺口不强接')
        if key=='89.75':
            vertical=min(raw['vertical'],key=lambda r:abs(r['s']-float(key)))
            tile5=np.asarray(vertical['lines_uz'])[np.asarray(vertical['tiles'])==5]
            axes[2].add_collection(LineCollection(tile5,colors=GREEN,linewidths=1.6))
            for h in json.loads((prior/'case5_unfiltered_summary.json').read_text(encoding='utf-8')):
                axes[2].scatter(h['u'],np.full(len(h['u']),h['z']),color=GREEN,s=23)
            for ax in axes:ax.axhline(evidence['arc']['z_range'][1],color='#7d8290',ls=':',lw=1)
            axes[2].set_title('原始 Tile 5：真实 V 曲线与 H 交点')
            axes[0].text(.04,.05,'旧连接 14.0508 m\nB4 仅 3 节点 / 5.64 mm',transform=axes[0].transAxes,color=RED,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            axes[1].text(.04,.8,'B4：MBG 拒绝\n不再扩展目标范围\nSOURCE_DATA_REQUIRED',transform=axes[1].transAxes,color=BLUE,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        else:
            for neighbor in case['neighbors']:
                if abs(neighbor['s']-float(key))<=.251:
                    axes[2].add_collection(LineCollection(neighbor['lines_uz'],colors=GRAY,linewidths=.55,alpha=.35))
            target_index=int(np.argmin(abs(case['s_values']-float(key))))
            for h in case['horizontal']:
                hits=np.asarray(h['hits']);hits=hits[hits[:,0]==target_index]
                if len(hits):axes[2].scatter(hits[:,1],np.full(len(hits),h['z']),s=11,color=GREEN)
            axes[2].set_title('实际 H 交点（绿）与邻近 V（浅灰）')
            for bid,z,offset in ((1,1380.3,(-55,12)),(2,1381.2,(42,26)),(3,1379.3,(30,-22)),(6,1380.3,(34,7))):
                b=next(b for b in case['branches'] if b['branch_id']==bid)
                hits=branch_crossings(b,z)
                if hits:axes[2].annotate('B'+str(bid),(hits[0]['u'],z),xytext=offset,textcoords='offset points',
                    fontsize=10,color='#303b49',arrowprops=dict(arrowstyle='-',color='#64748b',lw=.7))
            axes[1].text(.04,.10,'B1 未并入：保留当前范围与折返时\n最短可行连接约 0.700 m，超过 10 mm',
                transform=axes[1].transAxes,color=BLUE,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        title='F｜真实案例：极小尾片不再牵出 14 米连接' if key=='89.75' else '真实第六类｜重叠轨迹仍需 H/V 证据，不能以接通代替身份确认'
        fig.suptitle(f'{title}  s={key} m',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.94));name='real_'+key.replace('.','_')+'.png';fig.savefig(FIG/name,dpi=170);plt.close(fig)
        reviews.append(dict(slice_key=key,figure='figures/'+name,status=new['result']['status'],
            sequence=new['result'].get('route_branch_sequence',[]),reasons=new['result'].get('unresolved_reasons',[]),
            branch_audit=new['result']['branch_audit'],rejections=new['result'].get('continuation_rejections',[])))
    # A separate true-scale inset exposes the three nodes hidden in the larger plot.
    b=next(b for b in evidence['cases']['89.75']['branches'] if b['branch_id']==4)
    pts=np.asarray(b['points_uz']);origin=pts[0]
    fig,ax=plt.subplots(figsize=(7,4));p=(pts-origin)*1000;ax.plot(*p.T,'o-',color=ORANGE)
    for i,q in enumerate(p):ax.annotate(f'P{i}',q,xytext=(5,5),textcoords='offset points')
    ax.set(xlabel='相对 P0 径向距离 / mm',ylabel='相对 P0 高程 / mm',title='真实 B4 放大：3 个 canonical 节点，总弧长 5.64 mm')
    ax.set_aspect('equal');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'real_tiny_B4.png',dpi=170);plt.close(fig)
    save_json(OUT/'real_case_results.json',reviews)
    return reviews


if __name__=='__main__':
    controlled_cases()
    if (OUT/'main_track_index.json').exists():real_cases()
