"""Larger context, explicit endpoints and explanatory horizontal probes."""
from review_selected_cases import *
from render_local_conflict_v2 import plot_records,axes,savefig
from review_p0_p1_p2 import plt,Line2D,table,crop
from branch_absolute_core import branch_metrics,local_edge_scale

BLUE='#1467b1';PURPLE='#8c4faf';GRAY='#c3c6cb';RED='#c74732'


def records(root,s):return read(root/'layers'/f'{s}.pkl')['layers']['MAIN_SPINE']
def raw(inv,s):return [r for b in inv[s] for r in b['records']]
def finish(fig,out,name,title,handles):
    fig.suptitle(title,fontsize=16)
    fig.legend(handles=handles,loc='lower center',ncol=len(handles))
    savefig(fig,out,name,bottom=.16 if fig.get_figheight()<=8 else .10)


def c07(out,inventory,evidence):
    rows=[r for r in evidence if r['case']=='C07'];ss=[r['s'] for r in rows];bounds=[-19,15,1368,1415]
    fig,axs=plt.subplots(3,5,figsize=(20,16),sharex=True,sharey=True)
    for j,(s,row) in enumerate(zip(ss,rows)):
        for i,t in enumerate(('T0','T4')):
            seed=next(x for x in row['seed'] if x['FaceTrack'].endswith(':'+t))
            b=next(b for b in inventory[s] if b['branch_id']==seed['branch_id']);ax=axs[i,j]
            plot_records(ax,raw(inventory,s),bounds,GRAY,.8)
            plot_records(ax,b['records'],bounds,BLUE,2.2)
            for e in (b['records'][0]['points_uz'][0],b['records'][-1]['points_uz'][1]):
                ax.scatter(*e,color=RED,s=25,zorder=5)
                right=e[0]>8
                ax.annotate(f'{e[1]:.3f}',e,xytext=(-3 if right else 3,5),ha='right' if right else 'left',textcoords='offset points',fontsize=8,color=RED)
            ax.set_title(f's={s} | {t} 原分支 B{b["branch_id"]}',fontsize=10)
        ax=axs[2,j];plot_records(ax,raw(inventory,s),bounds,GRAY,.8);rr=records(latest(),s)
        plot_records(ax,rr,bounds,BLUE,2.2,False)
        for r in rr:
            if not r['source'].startswith('OBSERVED'):
                p=np.mean(r['points_uz'],axis=0)
                if bounds[0]<p[0]<bounds[1] and bounds[2]<p[1]<bounds[3]:
                    ax.scatter(*p,color=RED,s=30,zorder=5)
        ax.set_title('当前识别：'+'→'.join('B'+str(i) for i in row['sequence']),fontsize=9)
        for i in range(3):axes(axs[i,j],bounds)
        axs[2,j].set_xlabel('径向偏移 u（m）')
    for i,label in enumerate(('T0 对应原始分支','T4 对应原始分支','最终实测主轨')):axs[i,0].set_ylabel(label+'\n高程 z（m）')
    finish(fig,out,'C07_downward_context.png','C07｜向下扩到1368 m：两条候选的物理端点都在画面内',[
        Line2D([],[],color=GRAY,label='其他原始观测'),Line2D([],[],color=BLUE,lw=2,label='本行曲线'),Line2D([],[],color=RED,marker='o',ls='',label='前两行：真实端点；末行：接缝')])
    # Detail has actual source labels on selected records, not whole-branch
    # extrapolated FaceIDs. The overview above explicitly extends raw branches.
    identity_panels(out,inventory,ss,latest(),'C07_upper_detail',[-18.5,-7.5,1402.5,1414.5],None,'C07｜上部断开与不同续接来源')


def source_labels(tag):
    if tag=='C04':reg=next(r for r in js(OLD/'local_plan.json')['regions'] if 'E' in r['case_tags'])
    else:reg=next(g['region'] for g in js(latest()/'selection.json')['groups'] if g['group_id']=='Q03')
    return read(OLD/'stages/face_geometry_cache.pkl')[tuple(reg['bounds'])],reg


def identity_panels(out,inventory,ss,root,name,bounds,z,title):
    labels,reg=source_labels('C04' if name.startswith('C04') else 'C03')
    colors={0:BLUE,1:PURPLE,3:PURPLE,4:'#25865e'}
    fig,axs=plt.subplots(1,len(ss),figsize=(20,6),sharex=True,sharey=True,squeeze=False);rows=[];visible=set()
    for j,s in enumerate(ss):
        ax=axs[0,j];rr=records(root,s);plot_records(ax,raw(inventory,s),bounds,GRAY,.9)
        count=0;arc=0.
        for r in rr:
            length=float(np.linalg.norm(np.diff(r['points_xyz'],axis=0)))
            ts=sorted({labels[f] for f in r['source_face_ids'] if f in labels})
            if r['source'].startswith('OBSERVED'):
                tid=ts[0] if len(ts)==1 else None;color=colors.get(tid,GRAY)
                plot_records(ax,[r],bounds,color,2.3);visible.update(ts if crop(r['points_uz'],bounds) is not None else [])
                p=np.asarray(r['points_uz']);dz=p[1,1]-p[0,1]
                if z is not None and abs(dz)>1e-12 and min(p[:,1])<=z<max(p[:,1]):
                    t=(z-p[0,1])/dz;u=float(p[0,0]+t*(p[1,0]-p[0,0]))
                    if bounds[0]<=u<=bounds[1]:
                        count+=1;ax.scatter(u,z,color=color,s=28,zorder=6)
                        ax.annotate('T'+str(tid)+f' / B{r["branch_id"]}',(u,z),xytext=(2,10+count%2*13),textcoords='offset points',fontsize=8,color=color)
                        rows.append(dict(s=s,z=z,u=u,FaceTracks=ts,branch_id=r['branch_id'],route_arc_m=arc+t*length,edge_id=r['edge_id']))
            else:
                p=np.mean(r['points_uz'],axis=0)
                if bounds[0]<=p[0]<=bounds[1] and bounds[2]<=p[1]<=bounds[3]:
                    ax.scatter(*p,s=18,color=RED,zorder=5)
            arc+=length
        if z is not None:ax.axhline(z,color='#222',ls=':',lw=1)
        axes(ax,bounds);ax.set_xlabel('径向偏移 u（m）');ax.set_title(f's={s} m'+(f' | 诊断线交点={count}' if z is not None else ''),fontsize=10)
    axs[0,0].set_ylabel('高程 z（m）')
    handles=[Line2D([],[],color=GRAY,label='其他原始观测')]+[Line2D([],[],color=colors.get(t,'#555'),lw=2,label=f'实测主轨 T{t}') for t in sorted(visible)]
    handles.append(Line2D([],[],color=RED,marker='o',ls='',label='接缝'))
    finish(fig,out,name+'.png',title+(f'｜水平诊断线 z={z:.3f} m' if z is not None else ''),handles)
    save_json(out/(name+'_intersections.json'),rows)
    return rows


def c05(out,inventory):
    ss=[f'{129.8+i*.05:.2f}' for i in range(5)];bounds=[-18,3,1397.0,1414.5]
    fig,axs=plt.subplots(1,5,figsize=(20,8),sharex=True,sharey=True);rows=[]
    locks=js(RECENT/'audits/F.json')['locks']
    for ax,s in zip(axs,ss):
        plot_records(ax,raw(inventory,s),bounds,GRAY,.9);rr=records(RECENT,s);plot_records(ax,rr,bounds,BLUE,2.2,False)
        route=read(RECENT/'routes'/f'{s}.pkl')['result'];locked={v['branch_id'] for v in locks.get(str(float(s)),{}).values()}
        for ix,j in enumerate(route['junctions'],1):
            p=np.mean([j['a_point_uz'],j['b_point_uz']],axis=0)
            ax.scatter(*p,color=RED,s=36,zorder=6)
            right=p[0]>-6
            ax.annotate(f'接缝{ix}\nB{j["from_branch_id"]}→B{j["to_branch_id"]}\nz={p[1]:.3f}',p,xytext=(-8 if right else 8,5),ha='right' if right else 'left',textcoords='offset points',fontsize=8,color=RED)
            rows.append(dict(s=s,seam=ix,z=float(p[1]),u=float(p[0]),from_branch=j['from_branch_id'],to_branch=j['to_branch_id'],
                blanket_lock_skip=j['from_branch_id'] in locked or j['to_branch_id'] in locked,distance_mm=j['xyz_distance_m']*1000))
        axes(ax,bounds);ax.set_title(f's={s} m | B1锁定',fontsize=10);ax.set_xlabel('径向偏移 u（m）')
    axs[0].set_ylabel('高程 z（m）')
    finish(fig,out,'C05_both_seams.png','C05｜同时展示两个接缝；结果复用，未重新选轨',[
        Line2D([],[],color=GRAY,label='全部原始观测'),Line2D([],[],color=BLUE,lw=2,label='实测主轨'),Line2D([],[],color=RED,marker='o',ls='',label='接缝及分支编号')])
    save_json(out/'C05_seams.json',rows)


def c06(out,inventory):
    ss=[f'{104.6+i*.05:.2f}' for i in range(5)];bounds=[4.6,8.6,1385.4,1389.8]
    fig,axs=plt.subplots(2,5,figsize=(20,10),sharex=True,sharey=True)
    for j,s in enumerate(ss):
        for i,root in enumerate((RECENT,out)):
            ax=axs[i,j];plot_records(ax,raw(inventory,s),bounds,GRAY,1);rr=records(root,s)
            plot_records(ax,rr,bounds,BLUE,2.2,False)
            gaps=[r for r in rr if is_estimated_gap(r)];plot_records(ax,gaps,bounds,RED,2.6)
            for r in gaps:
                p=np.mean(r['points_uz'],axis=0)
                ax.annotate(f'估计 {r["distance_m"]*1000:.1f} mm\n不参与结构识别',p,xytext=(5,8),textcoords='offset points',fontsize=8,color=RED)
                ax.scatter(*np.asarray(r['points_uz']).T,color=RED,s=20,zorder=6)
            axes(ax,bounds);ax.set_title(f's={s} m',fontsize=10)
            if j==0:ax.set_ylabel(('修复前' if i==0 else '修复后')+'\n高程 z（m）')
            if i==1:ax.set_xlabel('径向偏移 u（m）')
    finish(fig,out,'C06_estimated_hole.png','C06｜经审核的模型孔洞：原端点直线估计连接',[
        Line2D([],[],color=GRAY,label='全部原始观测'),Line2D([],[],color=BLUE,lw=2,label='实测主轨'),Line2D([],[],color=RED,ls='--',lw=2,label='孔洞估计线（误差未量化）')])


def render():
    out=output();ev=js(out/'selection_evidence.json');ss=sorted({r['s'] for r in ev}|{f'{104.6+i*.05:.2f}' for i in range(5)},key=float)
    inv=load_inventory(SOURCE,ss)
    c07(out,inv,ev);c05(out,inv);c06(out,inv)
    for tag,root,bounds,z in [('C03',latest(),[-18.4,-9.5,1408.8,1414.8],1411.5863037099523),
                             ('C04',RECENT,[7.5,22.5,1382.6,1388.8],1385.7863037099758)]:
        identity_panels(out,inv,[r['s'] for r in ev if r['case']==tag],root,tag+'_source_explanation',bounds,z,tag+'｜同高程多个来源怎样产生')
    print('FIGURES',out,flush=True)


if __name__=='__main__':render()
