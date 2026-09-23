"""Read-only atlas and audit of frozen Phase 1 evidence; never applies routes."""
import time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import run_facetrack_voting_phase1 as api

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
GREEN='#168453';ORANGE='#c87932';GREY='#a0a7ae';MASK='#e4effb'
STATUS={'UNANIMOUS_SLICE_VOTES':('邻线全部独立支持区域面组','#288b60'),
        'AGGREGATED_WITH_ABSTENTIONS':('有独立弃权，无反对票','#3c80b3'),
        'AGGREGATED_WITH_DISSENT':('存在独立反对票','#d17737'),
        'REGION_AMBIGUOUS':('区域同分待定','#a14691'),
        'REGION_NO_ELIGIBLE':('全部未过 MBG','#656970')}


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n'


def pf(track):return '待定' if track is None else f'PF{track}'
def yes(value):return '是' if value else '否'


def arc_piece(branch,lo,hi):
    arc=branch['arc_positions'];p=branch['points_uz']
    if hi<=lo:return np.empty((0,2))
    interior=p[(arc>lo)&(arc<hi)]
    ends=np.array([[np.interp(a,arc,p[:,k]) for k in (0,1)] for a in (lo,hi)])
    return np.vstack([ends[:1],interior,ends[1:]])


def intersects(points,box):
    """Liang-Barsky clipping, including segments with both vertices outside."""
    a=np.asarray(points[:-1]);v=np.diff(points,axis=0)
    if not len(a):return False
    lo=np.zeros(len(a));hi=np.ones(len(a));valid=np.ones(len(a),bool)
    for k,(lower,upper) in enumerate(((box[0],box[1]),(box[2],box[3]))):
        flat=np.abs(v[:,k])<1e-14;valid&=~flat|((a[:,k]>=lower)&(a[:,k]<=upper))
        t0=np.divide(lower-a[:,k],v[:,k],out=np.zeros(len(a)),where=~flat)
        t1=np.divide(upper-a[:,k],v[:,k],out=np.ones(len(a)),where=~flat)
        lo=np.maximum(lo,np.minimum(t0,t1));hi=np.minimum(hi,np.maximum(t0,t1))
    return bool(np.any(valid&(hi>lo+1e-12)))


class Atlas:
    def __init__(self,out):
        self.out=out;self.m=api.read_json(out/'manifest.json');self.cache=Path(self.m['index_cache']);self.cell=self.m['policy']['cell_m']
        self.regions=api.read_json(out/'conflict_region_index.json')['regions'];self.reg={r['region_id']:r for r in self.regions}
        self.decisions=api.read_json(out/'region_track_decisions.json')['decisions'];self.dec={d['region_id']:d for d in self.decisions}
        self.rows=pq.read_table(out/'per_slice_track_scores.parquet').to_pylist();self.by_region=defaultdict(list);self.score={}
        for row in self.rows:
            self.by_region[row['region_id']].append(row);self.score[row['region_id'],row['s_index'],row['FaceTrack']]=row
        self.keys=api.source_keys();self.geometries={};self.cases=[];self.crosswalk={}
        groups=np.load(self.cache/'physical_face_groups.npy',mmap_mode='r')
        for tag in ('E','F','D'):
            old=api.read_pickle(api.SOURCE/'region_data'/f'{tag}.pkl')
            for member in old['members']:
                for label in member['face_track_ids']:
                    self.crosswalk.setdefault(label,set()).update(int(g) for g in groups[member['source_face_ids']])

    def geometry(self,si):
        if si not in self.geometries:self.geometries[si]=api.read_pickle(self.cache/'slices'/f'{self.keys[si]}.pkl')
        return self.geometries[si]

    def neighbors(self,s):
        i=min(range(len(self.keys)),key=lambda i:abs(float(self.keys[i])-s))
        return list(range(max(0,i-2),min(len(self.keys),i+3)))

    def gallery(self,label,rid,s,box,suffix='local'):
        indices=self.neighbors(s);region=self.reg.get(rid);decision=self.dec.get(rid)
        tracks=set(region['tracks'] if region else [])
        for si in indices:
            for b in self.geometry(si)['branches'].values():
                if intersects(b['points_uz'],box):tracks.update(b['physical_groups'])
        tracks=sorted(tracks);height=3.5*len(tracks)+1.8
        fig,axes=plt.subplots(len(tracks),len(indices),figsize=(18,height),squeeze=False,sharex=True,sharey=True)
        visible=[]
        for i,track in enumerate(tracks):
            for j,si in enumerate(indices):
                ax=axes[i,j];geo=self.geometry(si);bids=[]
                if region:
                    patches=[Rectangle((u*self.cell,z*self.cell),self.cell,self.cell) for ii,u,z in region['cells']
                             if ii==si and u*self.cell<box[1] and (u+1)*self.cell>box[0] and z*self.cell<box[3] and (z+1)*self.cell>box[2]]
                    ax.add_collection(PatchCollection(patches,facecolor=MASK,edgecolor='none',zorder=0))
                for bid,b in geo['branches'].items():
                    if track not in b['physical_groups'] or not intersects(b['points_uz'],box):continue
                    bids.append(bid);p=b['points_uz']
                    if not b['MBG']:ax.plot(p[:,0],p[:,1],color=GREY,ls='--',lw=1.4,zorder=3)
                    else:
                        for lo,hi,color in ((0.,b['CC_start_arc'],ORANGE),(b['CC_start_arc'],b['CC_end_arc'],GREEN),(b['CC_end_arc'],b['full_arc_length'],ORANGE)):
                            part=arc_piece(b,lo,hi)
                            if len(part):ax.plot(part[:,0],part[:,1],color=color,lw=1.6,zorder=3)
                    # Same-color endpoint circles are actual branch ends, never viewport edges.
                    ax.plot(p[[0,-1],0],p[[0,-1],1],'o',color=ORANGE if b['MBG'] else GREY,ms=3,zorder=4)
                row=self.score.get((rid,si,track));vote=self.score.get((rid,si,region['tracks'][0])) if region else None
                note='本窗口无该面组片段' if not bids else '显示 '+','.join(f'B{b}' for b in bids)
                if row:
                    note+=f"\n贯穿 {yes(row['through_region'])}｜CC {yes(row['in_CC'])}｜出口 {row['exit_continuation_m']:.2f} m"
                    note+=f"\n证据 B{row['witness_branch']}｜本线票 {pf(row['slice_vote'])}"
                else:note+='\n本 region 无评分' if region else '\n单一来源，不进入区域投票'
                ax.set_title(f's={float(self.keys[si]):.2f} m\n{note}',fontsize=8.5,pad=6)
                if not bids:ax.text(.5,.5,'无片段',ha='center',va='center',transform=ax.transAxes,color=GREY)
                if j==0:ax.set_ylabel(f'{pf(track)}'+('  区域首选' if decision and track==decision['dominant_FaceTrack'] else '')+'\n高程 z (m)')
                if i==len(tracks)-1:ax.set_xlabel('横向偏移 u (m)')
                ax.set_xlim(box[:2]);ax.set_ylim(box[2:]);ax.grid(alpha=.14);ax.xaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=8)
                ax.ticklabel_format(useOffset=False,style='plain')
                visible.append(dict(s=float(self.keys[si]),FaceTrack=track,visible_branches=bids))
        title=f'{label}｜同一局部区域的全部来源 × 五邻线'
        subtitle=f'{rid} 区域首选 {pf(decision["dominant_FaceTrack"])}；每条测线的独立票保留在格内' if decision else '负对照：只有一个物理来源，无 conflict region，无投票'
        fig.suptitle(title+'\n'+subtitle,fontsize=14,y=.985)
        handles=[Line2D([0],[0],color=GREEN,lw=2,label='CC / 原分支甜区'),Line2D([0],[0],color=ORANGE,lw=2,label='原分支端部 guard'),
                 Line2D([0],[0],color=GREY,lw=2,ls='--',label='MBG 未通过的片段'),Patch(color=MASK,label='本 region 的竞争格网')]
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.006))
        fig.subplots_adjust(left=.075,right=.985,top=1-1.8/height,bottom=max(.095,.9/height),hspace=.49,wspace=.13)
        name=f'{label}_{suffix}.png';fig.savefig(self.out/'figures'/name,dpi=175);plt.close(fig)
        return name,visible

    def regional_evidence(self,rid):
        r=self.reg[rid];d=self.dec[rid];ss=r['s_indices'];tracks=r['tracks'];labels=[];data=[]
        for track in tracks:
            rr=[self.score[rid,si,track] for si in ss]
            for field,name in [('MBG','合格'),('through_region','贯穿'),('in_CC','CC'),('exit_continuation_m','出口 / 1m')]:
                labels.append(f'PF{track} {name}');data.append([float(x[field]) for x in rr])
        fig,axes=plt.subplots(2,1,figsize=(17,5.4+len(tracks)*.7),gridspec_kw={'height_ratios':[len(data),1.6]})
        x=np.array([float(self.keys[si]) for si in ss]);edges=np.r_[x-.025,x[-1]+.025]
        mesh=axes[0].pcolormesh(edges,np.arange(len(data)+1),np.array(data),vmin=0,vmax=1,cmap='Greens',shading='flat')
        axes[0].set_yticks(np.arange(len(data))+.5,labels);axes[0].invert_yaxis()
        colors=['#d5d9dd']+['#247ba0','#d07832','#8e5bb2','#4c956c'][:len(tracks)];cmap=ListedColormap(colors)
        vote=[self.score[rid,si,tracks[0]]['slice_vote'] for si in ss];values=np.array([[0 if v is None else tracks.index(v)+1 for v in vote]])
        axes[1].pcolormesh(edges,[0,1],values,vmin=-.5,vmax=len(tracks)+.5,cmap=cmap,shading='flat');axes[1].set_yticks([])
        axes[1].set_ylabel('单线票');axes[1].set_xlabel('原始测线里程 s (m)；每列 = 1 条 V，不抽样')
        axes[1].legend(handles=[Patch(color=colors[i],label='待定' if i==0 else pf(tracks[i-1])) for i in range(len(colors))],ncol=len(colors),loc='upper left',bbox_to_anchor=(0,-.5),frameon=False)
        for ax in axes:ax.set_xlim(edges[0],edges[-1]);ax.ticklabel_format(axis='x',useOffset=False,style='plain')
        fig.suptitle(f'{rid} 全部 {len(ss)} 条测线的独立证据与投票｜冻结区域首选 {pf(d["dominant_FaceTrack"])}',fontsize=15)
        fig.subplots_adjust(left=.12,right=.95,top=.92,bottom=.15,hspace=.35)
        fig.colorbar(mesh,ax=list(axes),pad=.01,label='证据：否 0 → 是 1；出口为封顶 1m 的长度')
        fig.savefig(self.out/'figures'/f'{rid}_all_V.png',dpi=190);plt.close(fig)

    def full_score_table(self,rid):
        r=self.reg[rid];rows=self.by_region[rid]
        headers=['s (m)','面组','原始分支 / 合格分支','证据分支','贯穿','全部竞争弧段在 CC','CC 占比','格网覆盖','出口 m','本线 vote']
        values=[[f"{x['s']:.2f}",pf(x['FaceTrack']),f"{x['member_branches']} / {x['eligible_branches']}",x['witness_branch'],yes(x['through_region']),yes(x['in_CC']),f"{x['CC_fraction']:.4f}",f"{x['cell_coverage_fraction']:.4f}",f"{x['exit_continuation_m']:.4f}",pf(x['slice_vote'])] for x in rows]
        text=f'# {rid} 全部测线评分\n\n{len(r["s_indices"])} 条 V，{len(rows)} 条 (V, FaceTrack) 记录。未抽样。每条 V 的 vote 在该 V 各面组行重复列出；不要重复计票。\n\n'
        text+=md_table(headers,values)
        (self.out/'cases'/f'{rid}_ALL_SCORES.md').write_text(text,encoding='utf-8')

    def case(self,label,rid,s,box,explanation,context=None):
        r=self.reg[rid];d=self.dec[rid];image,visible=self.gallery(label,rid,s,box)
        indices=self.neighbors(s);local=[self.score.get((rid,si,r['tracks'][0]),{}).get('slice_vote') for si in indices]
        nearby=[]
        for reg in self.regions:
            hits=[c for c in reg['cells'] if c[0] in indices and box[0]<(c[1]+1)*self.cell and c[1]*self.cell<box[1] and box[2]<(c[2]+1)*self.cell and c[2]*self.cell<box[3]]
            if hits:nearby.append([reg['region_id'],','.join(map(pf,reg['tracks'])),len(reg['s_indices']),pf(self.dec[reg['region_id']]['dominant_FaceTrack']),len(hits)])
        text=f'# {label}：{rid}\n\n{explanation}\n\n'
        text+=f'区域包含 **{len(r["s_indices"])} 条 V**，s={r["bounds"][0]:.2f}–{r["bounds"][1]:.2f} m。冻结首选 **{pf(d["dominant_FaceTrack"])}**；人工审核：**待用户复核**。\n\n'
        text+=f'支持 / 反对 / 本线待定 = **{len(d["supporting_V"])} / {len(d["opposing_V"])} / {len(d["abstaining_V"])}**；所选面组原始缺席 {len(d["absent_V"])} 条、存在但 MBG 不合格 {len(d["ineligible_V"])} 条。存在不是合格，也不是已识别正确。\n\n'
        text+='## 同区域面组的层级证据\n\n'+md_table(['面组','贯穿 V','最长连续合格 V','连续 s 跨度 m','CC V','平均封顶出口 m','合格 V'],[[pf(e[0]),e[1],e[2],f'{e[3]:.2f}',e[4],f'{e[5]:.4f}',e[6]] for e in d['evidence_summary']])
        text+='\n按表从左到右的证据层级裁决；首先比较贯穿数，不把后面的优势加权抵消前面的差异。平均出口的分母为本区域全部 V，未合格按零贡献。\n\n'
        text+=f'## 五邻线局部图\n\n![{label}全部来源](../figures/{image})\n\n'
        text+='绿线是原始 CC，橙线是端部 guard，灰虚线为 MBG 失败；淡蓝格子只表示本 region 的 mask。端部同色圆点是真实分支端点。所有行列共用坐标。每格显示该面组进入画面的全部分支，并非仅画证据 witness。\n\n'
        text+='格内评分针对**本 V 的整个 region mask**；图框是放大显示窗口。局部全绿但 in_CC=否时，应看下面的整区图和完整表，不能把局部绿线等同于全 region 的 CC 支持。\n\n'
        if context:
            name,_=self.gallery(label,rid,s,context,'context')
            text+=f'## 扩大观察前后延伸\n\n![{label}扩大图](../figures/{name})\n\n'
        text+=f'## 全区域一致性\n\n![全部 V](../figures/{rid}_all_V.png)\n\n[逐 V 完整评分表]({rid}_ALL_SCORES.md)。五邻线仅为局部展示；区域首选来自全部 V 的证据。\n\n'
        text+='## 当前五邻线窗口中其他竞争区\n\n以下全部列出，未将不同竞争面组强行合成一个投票区。第一张图的淡蓝格子只对应本页标题 region。\n\n'
        text+=md_table(['region','候选组','全区 V','全区首选','当前窗口命中格数'],nearby)
        text+='\n## 图内完整分支清单\n\n'+md_table(['s','面组','可见原始分支'],[[f"{x['s']:.2f}",pf(x['FaceTrack']),x['visible_branches']] for x in visible])
        (self.out/'cases'/f'{label}.md').write_text(text,encoding='utf-8')
        self.cases.append(dict(label=label,region_id=rid,s=s,box=box,local_votes=local,description=explanation,image=image))

    def negative(self):
        # Find one five-profile patch containing one observed physical source.
        indices=self.neighbors(20.);center=self.geometry(indices[2]);chosen=None
        for cell,members in sorted(center['cells'].items()):
            if len(members)!=1:continue
            u,z=cell[0]*self.cell+.25,cell[1]*self.cell+.25;box=[u-1,u+1,z-1.5,z+1.5];all_tracks=set();good=True
            for si in indices:
                present={t for b in self.geometry(si)['branches'].values() if intersects(b['points_uz'],box) for t in b['physical_groups']}
                if len(present)!=1:good=False;break
                all_tracks.update(present)
            if good and len(all_tracks)==1:
                touches=any(i in indices and box[0]<(a+1)*self.cell and a*self.cell<box[1] and box[2]<(b+1)*self.cell and b*self.cell<box[3] for r in self.regions for i,a,b in r['cells'])
                if not touches:chosen=(box,all_tracks.pop());break
        assert chosen is not None
        box,track=chosen
        hits=[r['region_id'] for r in self.regions if any(i in indices and box[0]<(u+1)*self.cell and u*self.cell<box[1] and box[2]<(z+1)*self.cell and z*self.cell<box[3] for i,u,z in r['cells'])]
        assert not hits,('negative control touched conflict cell',hits)
        image,visible=self.gallery('NEGATIVE',None,20.,box)
        text=f'# 单一 FaceTrack 负对照\n\n19.90–20.10 m 五条 V 在共同窗口 {box} 内仅有 **PF{track}**。精确线段与窗口相交检查确认只有这一来源；竞争格网命中数为零，评分行数为零，未伪造单候选投票。\n\n'
        text+=f'![负对照](../figures/{image})\n\n'+md_table(['s','面组','原始分支'],[[x['s'],pf(x['FaceTrack']),x['visible_branches']] for x in visible])
        (self.out/'cases'/'NEGATIVE.md').write_text(text,encoding='utf-8')
        return dict(s=[float(self.keys[i]) for i in indices],box=box,FaceTrack=track,conflict_regions=hits,score_rows=0)

    def overview(self,local=False):
        cells=pq.read_table(self.cache/'cell_index.parquet',columns=['s_index','u_bin','z_bin']).to_pydict()
        s=np.asarray(cells['s_index'])*.05;u=(np.asarray(cells['u_bin'])+.5)*self.cell;z=(np.asarray(cells['z_bin'])+.5)*self.cell
        fig,axes=plt.subplots(2,1,figsize=(17,11),sharex=True)
        categories=STATUS;classified=defaultdict(list)
        if local:
            categories={'support':('本线支持区域首选','#288b60'),'abstain':('本线待定','#3c80b3'),
                        'oppose':('本线选择其他面组','#d17737'),'unresolved':('区域未产生首选','#a14691')}
        for r in self.regions:
            d=self.dec[r['region_id']]
            for c in r['cells']:
                status=d['confidence']
                if local:
                    vote=self.score[r['region_id'],c[0],r['tracks'][0]]['slice_vote']
                    status='unresolved' if d['ambiguous'] else 'abstain' if vote is None else 'support' if vote==d['dominant_FaceTrack'] else 'oppose'
                classified[status].append(c)
        # Grey context can be thinned; every conflict cell is kept below.
        for ax,y,ylabel in zip(axes,(z,u),('高程 z (m)','横向偏移 u (m)')):
            ax.scatter(s[::4],y[::4],s=.14,c='#d5d9dd',rasterized=True)
            for status,(name,color) in categories.items():
                cc=np.array(classified[status])
                if not len(cc):continue
                yy=(cc[:,2 if ax is axes[0] else 1]+.5)*self.cell
                ax.scatter(cc[:,0]*.05,yy,s=1.0,c=color,rasterized=True,label=name,alpha=.75)
            ax.set_ylabel(ylabel);ax.grid(alpha=.15);ax.ticklabel_format(useOffset=False,style='plain')
        for index,c in enumerate(self.cases):
            if c['label'] in ('C07_UPPER','F'):continue
            b=c['box']
            for ax,y in ((axes[0],(b[2]+b[3])*.5),(axes[1],(b[0]+b[1])*.5)):
                ax.plot(c['s'],y,'o',ms=4,mfc='white',mec='black')
                ax.annotate(c['label'],(c['s'],y),xytext=(9,12+12*(index%2)),textcoords='offset points',fontsize=9,
                            bbox=dict(fc='white',ec='none',alpha=.9),arrowprops=dict(arrowstyle='-',color='#444',lw=.6))
        axes[0].legend(loc='upper left',ncol=2,framealpha=.95,fontsize=9);axes[1].set_xlabel('纵测线位置 s (m)')
        title='全 2800 条 V：逐测线独立票与区域首选的空间对照' if local else '全 2800 条 V：物理面组竞争位置与区域状态'
        fig.suptitle(title+'\n上图 s–z；下图 s–u。彩色为实际竞争格，不是 region 外包矩形；投影重叠不代表同一曲面。',fontsize=15)
        name='LOCAL_VOTE_CONSISTENCY.png' if local else 'ALL_CONFLICT_REGIONS.png'
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(self.out/'figures'/name,dpi=200);plt.close(fig)

    def audit(self,negative):
        rows=self.rows;status=Counter(d['confidence'] for d in self.decisions);votes={};pairs=[0,0];winner_regions=[d for d in self.decisions if not d['ambiguous']]
        for row in rows:votes[row['region_id'],row['s_index']]=row['slice_vote']
        for (rid,si),vote in votes.items():
            other=votes.get((rid,si+1))
            if vote is not None and other is not None:pairs[1]+=1;pairs[0]+=int(vote==other)
        winner_pair_count=sum(len(self.reg[d['region_id']]['s_indices']) for d in winner_regions)
        supporting=sum(len(d['supporting_V']) for d in winner_regions);opposing=sum(len(d['opposing_V']) for d in winner_regions);abstaining=sum(len(d['abstaining_V']) for d in winner_regions)
        assert supporting+opposing+abstaining==winner_pair_count
        assert len({row['s_index'] for row in pq.read_table(self.out/'branch_facetrack_index.parquet',columns=['s_index']).to_pylist()})==2800
        for r in self.regions:
            assert len(r['tracks'])>=2
            assert len(self.by_region[r['region_id']])==len(r['s_indices'])*len(r['tracks'])
        assert api.digest(self.out/'region_track_decisions.json')==self.m['frozen_decision_sha256']
        result=dict(status=dict(status),region_count=len(self.regions),winner_regions=len(winner_regions),unresolved_regions=len(self.regions)-len(winner_regions),
                    single_V_regions=sum(len(r['s_indices'])==1 for r in self.regions),region_V_pairs=len(votes),winner_region_V_pairs=winner_pair_count,
                    supporting_region_V=supporting,opposing_region_V=opposing,abstaining_winner_region_V=abstaining,
                    evaluable_neighbor_pairs=pairs[1],same_vote_neighbor_pairs=pairs[0],unique_slice_votes=sum(v is not None for v in votes.values()),
                    negative_control=negative,decision_hash_unchanged=True,all_profiles_indexed=True,all_score_cardinalities_valid=True,
                    crosswalk={k:sorted(v) for k,v in self.crosswalk.items()},cases=self.cases)
        api.save_json(self.out/'ATLAS_AUDIT.json',result)
        return result


def write_reports(atlas,audit):
    out=atlas.out;m=atlas.m
    cold=api.read_json(api.PARENT/'20260923_093358'/'manifest.json')
    replay=api.read_json(api.PARENT/'20260923_093358'/'PERFORMANCE_REPLAY.json')
    reuse=api.read_json(out/'CACHE_REUSE_CHECK.json')
    rs=atlas.regions;ds=atlas.decisions
    by_score={(r['region_id'],r['s_index'],r['FaceTrack']):r for r in atlas.rows}
    comparisons=[]
    for (rid,si,t),row in by_score.items():
        other=by_score.get((rid,si+1,t))
        if other:comparisons.append((row,other))
    eligible_pairs=[p for p in comparisons if p[0]['MBG'] and p[1]['MBG']]
    audit['adjacent_evidence']={
        'all_same_region_track_pairs':len(comparisons),
        'both_MBG_eligible_pairs':len(eligible_pairs),
        'MBG_equal_pairs':sum(a['MBG']==b['MBG'] for a,b in comparisons),
        'through_equal_eligible_pairs':sum(a['through_region']==b['through_region'] for a,b in eligible_pairs),
        'CC_equal_eligible_pairs':sum(a['in_CC']==b['in_CC'] for a,b in eligible_pairs)}
    unresolved_cells=sum(len(r['cells']) for r in rs if atlas.dec[r['region_id']]['ambiguous'])
    api.save_json(out/'ATLAS_AUDIT.json',audit)
    catalog=[]
    for r in rs:
        d=atlas.dec[r['region_id']];b=r['bounds']
        catalog.append([r['region_id'],','.join(map(pf,r['tracks'])),len(r['s_indices']),f'{b[0]:.2f}–{b[1]:.2f}',f'{b[2]:.1f}–{b[3]:.1f}',f'{b[4]:.1f}–{b[5]:.1f}',
                        pf(d['dominant_FaceTrack']),len(d['supporting_V']),len(d['opposing_V']),len(d['abstaining_V']),d['confidence']])
    (out/'REGION_CATALOG.md').write_text('# 全部 963 个固定竞争区\n\n范围列为外包范围；真正边界为 conflict_region_index.json 中 cells 列表。状态是统计结果，人工审核均待完成。\n\n'+
        md_table(['region','面组','V 数','s 范围 m','u 范围 m','z 范围 m','区域首选','支持 V','反对 V','待定 V','状态'],catalog),encoding='utf-8')
    statuses=md_table(['区域结论','区域数'],[[STATUS[k][0],audit['status'].get(k,0)] for k in STATUS])
    cases=md_table(['图册','region / V 数','五邻线独立票','区域首选','审核重点'],[
        [f"[{c['label']}](cases/{c['label']}.md)",f"{c['region_id']} / {len(atlas.reg[c['region_id']]['s_indices'])}",
         ', '.join(pf(t) for t in sorted(set(c['local_votes']),key=lambda x:-1 if x is None else x)),pf(atlas.dec[c['region_id']]['dominant_FaceTrack']),
         '局部票与区域不同' if any(t is not None and t!=atlas.dec[c['region_id']]['dominant_FaceTrack'] for t in c['local_votes']) else '待人工确认物理身份'] for c in atlas.cases])
    biggest=md_table(['region','面组','V 数','s 范围 m','u 范围 m','z 范围 m'],[
        [r['region_id'],','.join(map(pf,r['tracks'])),len(r['s_indices']),f"{r['bounds'][0]:.2f}–{r['bounds'][1]:.2f}",f"{r['bounds'][2]:.1f}–{r['bounds'][3]:.1f}",f"{r['bounds'][4]:.1f}–{r['bounds'][5]:.1f}"]
        for r in sorted(rs,key=lambda r:-len(r['cells']))[:9]])
    case_rank=[]
    for label,rid in [('E','R00629'),('C05 / F','R00806'),('C07 / 89.95','R00268'),('C07 上部','R00205'),('D','R00204')]:
        d=atlas.dec[rid]
        case_rank.append([label,rid,pf(d['dominant_FaceTrack']),'; '.join(f'PF{e[0]}={e[1]}' for e in d['evidence_summary']),
                          f"{len(d['supporting_V'])}/{len(d['opposing_V'])}/{len(d['abstaining_V'])}"])
    e=audit['adjacent_evidence']
    report=f'''# FaceTrack 投票架构第一阶段：全量分析报告

日期：2026-09-23。最终结果目录：`{out.name}`。依据：[用户提供的第一阶段方案](C:/Users/222/Downloads/GDS_Architecture_Refactor_Phase1_FaceTrack_Voting_Plan.md)。

**本轮已完成 2800 条 V 的索引、独立证据评分、区域聚合和冻结选面。计算架构已经解耦；但 E、F/C05、89.95 m 的局部独立意见会被较大区域的结论覆盖，选面正确性尚未通过人工验收，不宜据此直接进入批量换轨。**

本报告展示的是**面组评价和选面结果**。没有重新生成最终主轨、换轨点或结构识别结果，也没有改变此前认可的主轨。旧路线的成功不等于本轮区域选择已通过，图上的原始分支也不是本轮新识别的主轨。

## 1. 本次全量覆盖了什么

输入为冻结原始观测源 `outputs/face_provenance_validation/20260921_075801`，s=0.00–139.95 m，间隔 0.05 m。

| 项目 | 全量结果 |
| --- | ---: |
| V 测线 | 2800 |
| 原始分支 | {m['branch_count']:,} |
| 固定网格共享边连通物理组 | 35 |
| 有观测的 (V,u,z) 格子 | {m['cell_count']:,} |
| 多物理组共存格子 | {m['conflict_cell_count']:,} |
| 固定竞争区 | 963 |
| (region,V) 评价组合 | {audit['region_V_pairs']:,} |
| (region,V,FaceTrack) 评分行 | {m['score_row_count']:,} |
| 唯一首选 / 待定区 | 344 / 619 |

{statuses}

619 个待定区中，594 个是证据同分，25 个是全部未过 MBG。963 个区域中有 633 个仅涉及一条 V；待定区中有 468 个这样的单线小区。待定区共 991 个 (region,V) 组合、{unresolved_cells:,} 个竞争格，仅占竞争格的 **{100*unresolved_cells/m['conflict_cell_count']:.2f}%**。因此“619/963 待定”不能读成“64% 的模型识别失败”。同样，344 个唯一结果也不能读成已确认正确。

## 2. 空间上，竞争主要在哪里

![竞争区全域分布](figures/ALL_CONFLICT_REGIONS.png)

灰色是原始观测分布背景，彩色是实际线段进入的竞争格子。上、下图是 s–z 和 s–u 两种投影；合起来定位，不把二维重叠当成同一曲面。**这张图按整个 region 的状态着色：区域内只要有反对票，整个区域均为橙色；不表示橙色的每条 V 都在反对。**

主要分布为：z≈1403–1413.5 m 的长带；s≈43–104 m、z≈1369–1404.5 m 的斜向及分叉区域；E 附近 s≈99–109 m、u≈9.5–23 m；F 所在 s≈105–140 m、u≈−10.5–18.5 m；以及 s≈100–140 m、z≈1435–1477.5 m 的上部边缘。它们是竞争分布，**本轮没有读取重建分块元数据，不能把这些带状形态直接证明为分块接缝。**

{biggest}

上述范围只是方便定位的外包盒。真实区域由 cell mask 决定，内部不是整块填满；完整列表见 [全部区域目录](REGION_CATALOG.md)。

## 3. 同一个面组，邻线评价是否一致

需要区分“相邻 V 互相一致”和“每条 V 都支持区域聚合结论”。

| 口径 | 结果 | 能说明什么 |
| --- | --- | --- |
| 同 region 内相邻两条 V 都有唯一投票 | 9048 / 9186 对相同，**{100*9048/9186:.2f}%** | 局部独立投票通常平稳 |
| 已有区域首选的 (region,V) 组合 | 10,887 个 | 同一 V 在多个空间区域会分别计数 |
| 独立票支持区域首选 | 7,711，**{100*7711/10887:.2f}%** | 单线意见与区域结论一致 |
| 独立票选择其他来源 | 1,937，**{100*1937/10887:.2f}%** | 区域内存在局部偏好反转 |
| 单线证据同分 / 无合格候选 | 1,239，**{100*1239/10887:.2f}%** | 不能给单线强行归票 |

相邻票一致率只统计双方均有明确票的邻对，未包含弃权、没有下一邻线等情形；没有拿冻结后所有 V 被指定同一面组来构造“100% 一致”。这些比例均为**内部一致性，不是识别准确率或人工真实性结论**。沿用用户要求，分支细节复杂度是否相似由同 FaceTrack 五邻线原始曲线人工审核，没有另造一个 TC 公式代替形态判断。

进一步看相同 (region,FaceTrack) 的邻线证据：MBG 相同 {e['MBG_equal_pairs']}/{e['all_same_region_track_pairs']} 对；双方 MBG 均通过时，贯穿判据相同 {e['through_equal_eligible_pairs']}/{e['both_MBG_eligible_pairs']} 对，CC 判据相同 {e['CC_equal_eligible_pairs']}/{e['both_MBG_eligible_pairs']} 对。这些检查说明的是二值证据的变化，不等同于曲线几何完全一致。

![逐测线与区域结论的空间对照](figures/LOCAL_VOTE_CONSISTENCY.png)

这张图才逐条 V 着色：绿色为本线票支持区域首选，橙色为本线选了另一面组，蓝色为本线待定，紫色为区域没有首选。它定位的是需要复核的局部意见变化；原始曲线见下面各图册。

## 4. 代表案例与具体判据

{cases}

另有 [单一 FaceTrack 负对照](cases/NEGATIVE.md)：s=19.90–20.10 m 的共同窗口仅有 PF23，未建立竞争区、未产生投票。不是说这五条测线在其他位置都没有竞争。

{md_table(['案例','region','区域首选','第一层贯穿 V 数','支持/反对/待定 V'],case_rank)}

**E 是本轮最需要复核的例子。** E_T1=PF16，E_T2=PF19，E_T3=PF34，由原 source face IDs 对应。104.20–104.40 m 五线均投 E_T2，但包含 s=98.95–109.40 m 的 R00629 选 E_T1：首层贯穿票仅多 2 条（24 对 22）。这种细小首层差异就能压过局部 5/5 的另一种偏好，是当前严格层级规则和区域范围共同造成的，不是联合接点函数重新干预。此前 E_T2 的人工认可不能自动转移给现在的 E_T1。

![E 三个来源与五邻线](figures/E_local.png)

**C05/F：** 两个五邻线窗口均投 PF16，R00806 最终选 PF19，贯穿数 9 对 6 就决定结果；仅 703 条 V 中很少一部分提供贯穿差异。区域总票支持 PF19 有 527 条，但当前展示的 F 局部处于反对段。不能用总支持数替代局部检查。

**C07 与 89.95 m：** 两处同属 R00268。85.95 m 附近五线都投 PF34（原 T4），89.95 m 附近五线都投 PF19（原 T0）；区域却统一 PF34。整个区域对 PF34 的独立支持只有 288 条，反对有 362 条，仍因贯穿数 187 对 62 而选 PF34。这里“投票”实际上是**证据分层聚合，非最终 slice_vote 的简单多数表决**，必须说明清楚。

C07 上部第三张面 PF23 加入后，成为 R00205，五线票和区域首选都为 PF19。这说明同一路径上下可以有不同 region-level 来源，第一阶段并没有规定“一张面走完全路径”。下一阶段才解决这些空间区域之间怎样接续；本轮没有添加估计连接线。

**D：** 局部与区域均选 PF23（原 D_T2），1515 条 V 中有 1501 条支持、14 条反对，是本轮较强的一致性例子；仍需人工审核是否真实目标面。**R00761 同分：** 两面均通过 13 条贯穿、13 条 CC、1 m 出口，当前证据无法区分。**R00008：** 两面均不合格，不能把无合格候选写成两个稳定分支同分。

## 5. 怎么看图

1. 每行是一张物理面组 PF，每列是一条相邻 V；各格使用相同 u/z 范围。逐行横看邻线形态是否连贯，逐列竖看不同面组的真实区别。
2. 绿色仅表示原始分支的 CC（沿用既有绝对甜区边界），橙色是端部 guard，灰虚线是 MBG 未通过，淡蓝是本页 region 的真实格网 mask。仅四类图例。圆点为原始分支真实端点；图框边缘不是端点。
3. 图中画的是本窗口该面组的**全部原始分支**。格内“显示 B…”列出可见分支；“证据 B…”只说明哪条原始分支提供了评分 witness，没有删掉其他候选，更没有拼成主轨。
4. 格内“本线票”是独立评价；页首“区域首选”是全 region 聚合。它们不同正是待审核信息。绿色不代表这条曲线已经被证明正确。
5. 格内评分针对本 V 的**整个 region mask**，不是仅当前显示框。大 region 的其他位置出现 guard、断点或多个片段，会使局部看起来全绿但 `in_CC=否`。F 和 C07 附扩大图；每页还列出窗口内其他 region，避免只展示其中一组竞争来源。
6. 全区证据条带每列为一条原始 V，不抽样；绿色深浅表示 MBG/贯穿/CC/1 m 出口，最下面彩条才是独立 vote。每个代表 region 都附全部 V 的评分表，不仅给中心五条。

## 6. 实现口径与边界

**A1** 一次读取 20,017 条原始分支，计算节点门槛、固定核心弧长、source face 映射和准确线段格网相交。没有重新加载、焊接或切割 GLB。35 个 PF 由原网格共享边邻接的全局连通分量得到；旧 T 是局部标签，必须映射，不能按数字直接等同。PF 身份表示物理来源连通，不自动保证每个局部形态都属于目标坡面。

**CC / MBG** 沿用已有 `branch_absolute_core`：两端各保留 4 条原始边间隔作为 guard，节点至少 12，核心长度至少为本 V 原始边中位长度的 4 倍，并要求开放表面分支。新索引把原 ASC 的起止弧长记录为 `CC_start_arc/end_arc`；没有构造动态 ASC，也不因画图裁框重新计算核心。旧策略对象内的接点参数没有参与本轮评分。

**A2** 使用固定 0.5 m × 0.5 m 的 u/z 格网，每条 V 保留原 0.05 m 间距。只有真实线段进入同格且有至少两种物理来源才构成竞争格；不是填充分支包围盒。同一候选集合的 26 邻接格构成 region，测线缺失、竞争消失、候选集合变化即截止。边界建立后固定，无递归扩区。这个 0.5 m 是本轮明确的索引尺度，不是 2 mm 换轨阈值；同格并不能证明曲面真实相交，格网边界会产生小碎区。本轮没有做分辨率扫描。

**A3** 每条 V 只读本线原始几何与冻结 mask；同一物理组可以有多条 branch，MBG 失败的片段保留为存在证据。合格 witness 按连续贯穿、格网覆盖、CC 完整/比例、出口证据择优，末尾 branch ID 只用于等效 witness 的可复现展示，不给 FaceTrack 强行破同分。不会用多条分支的虚拟连接声称已经连续贯穿。

贯穿要求一条原始分支覆盖该 V 的全部 region cells、竞争弧段连续，且区间前后都有原始观测。多次离开再进入不计为连续贯穿。`in_CC` 只要求 witness 的实际竞争弧段全部处于固定 CC，格网覆盖率单独记录。出口按原分支从较低端点向较高端点的路径方向，最多观察 **1 m 实测弧长**；不搜后继，也不裁核心。出口比较量化至 1 mm，以免浮点尾差强行选面。

**B** 只读评分表，依次比较：贯穿 V 数 → 最长连续合格 V 数及 s 跨度 → CC 支持 V 数 → 封顶出口总证据（同区域分母一致，等价平均）。FaceTrack 精确同分保持 `REGION_AMBIGUOUS`。冻结输出为不可变 dataclass/tuple 的 JSON 快照并记录 SHA256；后续使用方必须验证和只读消费，文件本身不是操作系统禁止写入。

没有引入 junction distance、region lock、joint band、跨 V DP、future successor chain、route budget 或顺序 route 写回。也没有将 100 mm 高程带重新带回来；高程坐标只用于原始空间定位。

## 7. 本轮发现并处理的问题

初次独立代码审查发现两个实现错误：①离散竞争弧段误计为连续贯穿，原评分中 3349 个 through 行受影响；②CC 误与全格网覆盖绑定，352 个实际竞争弧段全在核心内的合格行丢失 CC 支持。均已修正，并补两项行为用例。只重算 A3/B，A1/A2 内容哈希和索引保持不变。最终 40 个区域的首选/待定状态相对初算发生变化；本报告一律使用修正后的 `20260923_094440`。

仍存在的**算法表达问题**：

- 连通竞争区域可横跨 32.55、35.10 甚至 75.70 m，原始曲面身份相同但局部偏好不同；“一个连通区只有一个目标来源”未必适合每个位置。E、F、89.95 是实证。
- 二值贯穿对格网边界、分支端部和多次进出敏感；第一层仅领先两三条 V 就排除后续证据，尚无“优势有多强”的门槛。
- 静态 CC 支持会因整片竞争区接触端部而归零；局部稳定几何与整 region 的二值 CC 并非一回事。
- 两张面都稳定且都在 CC 时，五项证据可能完全同分。区域统计不能自行证明哪张是主坡面，也没有在本阶段增加横测线或曲线复杂度判据来冒充已有约束。

这些是下一步设计的审核输入，本轮未为 E/F 或某种模型形态写专用规则。合理的后续方向是先审核区域划分与首层证据优势的解释，再讨论是否采用固定局部子区或增加通用证据；不能未经审核直接放宽规则来强选。

## 8. 验收与人工审核

| 方案验收项 | 本轮证据与结论 |
| --- | --- |
| 全 2800 V 建索引并复用 | 完成；两次评分共用相同 index key，独立缓存验证 1.61 s |
| 冲突区边界明确 | 完成固定 cell mask；没有 route 驱动扩张。0.5 m 离散尺度仍需审核 |
| 单 V 评分独立、worker 无共享路线 | 新入口不读取 route；输入只有本 V compact geometry 与 mask |
| 聚合只处理小表 | B=0.078 s，不读网格、不重算几何 |
| 代表区能否得到合理面组 | 部分有清晰证据，但 E/F/89.95 有局部与区域反向，**尚未人工验收通过** |
| 单一来源不入竞争 | 五邻线负对照通过，无评分行 |
| 第一阶段性能 | 初次全量建索引+评分 70.15 s；索引复用后的最终全量重评 9.64 s；详见性能报告 |

最低代码检查：仅运行本次专项 `test_facetrack_phase1.py`，初次 8 项通过；针对审查修复后一次复检 **10 项通过（0.009 s）**，未运行全项目测试/构建。实际分析还检查线段弧长守恒、2800 V 覆盖、评分记录基数、负对照、缓存来源哈希和出图前后冻结决策哈希；性能优化回放逐项对比全部评分和决策相同。图册已检查标题、图例、邻线坐标与主例来源完整性。

人工优先审核：①E 是否允许整个 R00629 选 E_T1；②F/C05 局部 PF16 与区域 PF19 的矛盾；③C07 和 89.95 是否应该共享一个 region 决策；④R00761 双稳定来源能否依当前证据区分。所有“人工审核结论”均为**待用户确认**，没有代填通过。

实现限制：当前入口固定使用这份完整 0.05 m 数据源；非等距测线、全空 chunk 或全模型零竞争时尚未建立通用空表 schema，需在接入其他数据前补齐。相邻分支形态一致并不证明真实面身份；没有人工全区标签，因此本轮不能给整体识别准确率。

## 9. 交付文件

- [完整区域目录](REGION_CATALOG.md)、上表各代表图册和逐 V 评分表。
- [性能报告](PERFORMANCE_REPORT.md)、[审计与图册清单](ATLAS_AUDIT.json)、[来源/参数/计时清单](manifest.json)、[缓存验证](CACHE_REUSE_CHECK.json)。
- [Branch 索引](branch_facetrack_index.parquet)、[FaceTrack 索引](face_track_index.parquet)、[固定竞争区](conflict_region_index.json)、[独立评分](per_slice_track_scores.parquet)、[冻结决策](region_track_decisions.json)。

新增源码为三个独立核心模块 `facetrack_phase1_evidence/regions/vote.py`、全量入口 `run_facetrack_voting_phase1.py`、只读图册脚本 `render_facetrack_voting_phase1.py` 及专项测试。既有主轨组装代码未改动，没有提交或推送。
'''
    (out/'ARCHITECTURE_REFACTOR_REPORT.md').write_text(report,encoding='utf-8')
    performance=f'''# 第一阶段重构性能报告

日期：2026-09-23。**完整第一阶段覆盖 2800 条 V；最终换轨与结构识别不在本轮计时范围。**

## 实测结果

| 执行 | 输入范围 | 时间 | 说明 |
| --- | --- | ---: | --- |
| 初次建立本阶段索引并评分 | 2800 V / 20017 branches | {cold['seconds']:.3f} s | 索引冷启动；已有冻结 branch/mesh cache，不含 GLB 导入和原始切片 |
| 监控优化的同输入 A3 对照 | 2800 V / 25873 rows | {replay['original_A3_s']:.3f} → {replay['optimized_A3_s']:.3f} s | 同一评分代码、全部行及决策逐项相同；匹配阶段加速 {replay['matched_stage_speedup']:.2f}× |
| 修正两项证据判据后的最终全量重评 | 复用全部 A1/A2，重算 A3/B | {m['seconds']:.3f} s | 本报告最终结果；评分语义改变，不与上一行计算同语义加速倍数 |
| 来源与缓存复用核对 | 全量源哈希 + 决策哈希 | {reuse['seconds']:.3f} s | 不重建几何、不重算评分 |

初次运行与最终运行是不同索引状态，不能用 70.15/9.64 宣称纯算法加速。初次评分判据随后修正，初次数字保留用于建索引成本和监控对照；最终选面统计均来自 `{out.name}`。

## 分阶段计时

{md_table(['阶段','初次建索引运行 s','最终复用索引运行 s'],[
    ['读取实际源文件并核 SHA256',f"{cold['stages']['source_hash_verification_s']:.3f}",f"{m['stages']['source_hash_verification_s']:.3f}"],
    ['固定物理连通分量',f"{cold['stages']['physical_face_components_s']:.3f}",'缓存复用'],
    ['A1 原始分支 / 线段格网索引',f"{cold['stages']['A1_observed_branch_index_s']:.3f}",'缓存复用'],
    ['A2 固定竞争区域索引及索引合并',f"{cold['stages']['A2_conflict_index_s']:.3f}",f"{m['stages']['A1_A2_cache_load_s']:.3f}（载入）"],
    ['输出索引 / 建立评分任务',f"{cold['stages']['index_export_s']:.3f}",f"{m['stages']['index_export_s']:.3f}"],
    ['A3 单线独立评分及评分表保存',f"{cold['stages']['A3_independent_scores_s']:.3f}",f"{m['stages']['A3_independent_scores_s']:.3f}"],
    ['B 表格聚合和冻结 JSON',f"{cold['stages']['B_table_reduction_s']:.3f}",f"{m['stages']['B_table_reduction_s']:.3f}"]])}

阶段表不严格加总为总时间：最终 branch through-membership 导出、manifest、文件复制与收尾也包含在总时间。图册只读生成另计；主要图册首次成功生成约 11.51 s，随后为标注和局部一致性图做了局部补图，均未计入算法运行时间。

环境为本机 Windows / Python 3.12，28 个逻辑 CPU，约 31.7 GiB 物理内存。实际使用 **6 个进程**，各进程 BLAS 线程设为 1。A1 每批 40 V；A3 按 V 合并该 V 的独立 (region,V) 小任务，以免每个小区域独立建进程。B 单进程只读评分表。

初次运行采样的父进程与子进程合计 RSS 峰值 **{cold['sampled_peak_parent_children_RSS_bytes']/1024**3:.3f} GiB**；最终重评 **{m['sampled_peak_parent_children_RSS_bytes']/1024**3:.3f} GiB**。这是定期采样值，不是真实瞬时峰值保证。worker CPU 累计初次 {cold['worker_cpu_seconds']:.2f} s、最终 {m['worker_cpu_seconds']:.2f} s，不包含父进程全部 CPU。

## 实际优化了什么

发现 A3 初次墙钟 31.35 s，而 worker 的有效评分时间很短。父进程每拿到一条 V 的结果都通过 Windows `psutil` 枚举所有子进程并读内存，2800 次监控反而成为主要开销。改为每 100 个结果采样一次，末次保留采样；评分、候选和聚合均未更改。

完成一次同输入全量 A3 回放：全部 25,873 行评分逐项相同，963 个区域决策逐项相同，原冻结文件 SHA256 不变。A3 从 31.348 s 到 3.862 s，**该阶段减少 {100*(1-replay['optimized_A3_s']/replay['original_A3_s']):.2f}%**。这是可比的实测优化，不能扩写为整个旧识别流程提速 8 倍。证据见 [同输入回放 JSON](../20260923_093358/PERFORMANCE_REPLAY.json)。

结构性改动是：每条 branch 的核心属性只算一次；固定物理组件只建一次；精确格网区与分支映射缓存；A3 无跨线 route 状态；B 只操作小表。新入口的 route 读写、接点搜索和 region-lock 使用计数均为零。

## 与旧联合分析怎样比较

旧 [650 条局部识别清单](../../local_conflict_v2/20260922_124130/local_summary.json) 记录 **2407.94 s（40.13 min）**，包含 349,384 次 CC 调用、889 次连通分量计算、16,260 次 junction 查询和 8,515,275 对精算边。旧 [针对问题复核记录](../../targeted_issue_recheck/20260922_181451/run_summary.json) 总耗时 **2530.59 s（42.18 min）**，计划 110 条，仅 59 条完成，因预计超时停止。

本轮第一阶段全 2800 条首次完成于 70.15 s，复用索引重新评分为 9.64 s，已把本阶段等待时间降到秒/分钟量级。**但旧流程包含主轨组装、接点和联合约束，本轮只选面；不能据此计算完整识别的公平加速倍数，也不能外推第二阶段耗时。** 为遵守本轮范围，没有重跑旧联合分析去补一个不可比的全量时间。

## 当前瓶颈与后续优化方向

1. 首次索引主要成本是 **A1 25.58 s**：pickle 解码、原始 source-face 映射和逐段格网切分。当前无需再扩大重构。若未来模型增大，可把不可变 branch 数组持久化为列式/连续数组，减少重复反序列化；须验证精确弧长与源 face 映射不变。
2. A2 固定区域合并及表合并 **7.23 s**。候选集合已分组，后续可进一步预分桶处理，但不应为了性能改变区域邻接或引入远距离连接。
3. 复用索引后，**索引导出/任务组织 4.70 s** 已高于 A3 的 2.93 s；输出中含完整 face IDs。可以复用稳定表、只增量写 through-membership 和评分表，以及预存 region→V mask。它比盲目增加 worker 数更值得优先考虑。
4. 来源哈希约 **1.58 s**，目前保留真实文件核对以防缓存串用。不能仅靠文件名跳过来源验证。
5. B 聚合 **0.078 s**，不是瓶颈；不需要再并行化这个小表。

这次没有任务接近一小时，未触发中止。下一阶段的局部 junction 性能尚未测量；必须在独立实现后另测，不把本次统计冒充换轨性能。

## 复现与边界

在项目根目录执行 `python scripts/99_experiments/run_facetrack_voting_phase1.py --workers 6 --budget-seconds 3600`；当前参数相同则复用索引/评分，评分代码哈希变化仅重新评分。`--reuse-check` 仅验证来源、索引键和冻结结果；`render_facetrack_voting_phase1.py` 只读出图。

`--benchmark-score-replay` 是监控优化时使用的同输入诊断入口，当前 latest 已是最终修正版；不要用 latest 的已优化时间再复跑来冒称第二次优化。匹配基准数据固定保存在 `20260923_093358`。

所有结果以真实 source hash、mesh hash、region definition、metric policy、scoring policy 和代码哈希区分。由于未清除操作系统磁盘缓存，这里的“冷”只表示本阶段索引尚未建立，**不表示冷磁盘或 GLB 从零启动**。
'''
    (out/'PERFORMANCE_REPORT.md').write_text(performance,encoding='utf-8')
    print('REPORTS_COMPLETE',out,flush=True)


def main():
    began=time.perf_counter();atlas=Atlas(api.latest())
    specs=[
        ('C05','R00806',129.90,[-7,7,1383,1407],
         '当前 129.80–130.00 m 五邻线均独立投 PF16（原 F_T1），但包含 703 条 V 的区域最终选 PF19（原 F_T2）。决定性证据是贯穿支持 9 比 6；区域聚合没有采用局部票数多数。这是区域内非均质性，不能宣称局部独立评价与区域结论一致。',None),
        ('F','R00806',130.45,[-3,3,1389,1400],
         'F 130.35–130.55 m 五邻线同样均投 PF16，区域首选却是 PF19。扩大图包含整个本线竞争范围，解释为何眼前两条绿线仍可能得到不同 in_CC：评分覆盖本 V 在该 region 内的全部实际竞争弧段。',[-8,7,1371,1407]),
        ('C07','R00268',85.95,[-12,2,1387,1406],
         '原 T0 对应 PF19，原 T4 对应 PF34。85.85–86.05 m 五线独立票一致为 PF34，区域贯穿支持为 PF34=187、PF19=62，因此冻结 PF34。原始完整上下延伸仍展示在扩大图中；这不是后继拼接的最终主轨。',[-19,15,1368,1415]),
        ('C07_UPPER','R00205',85.95,[-17,-7,1402,1414],
         '同一组五邻线的上部，第三个来源 PF23 加入竞争，形成另一个固定区域 R00205。这里五线都投 PF19（T0），区域也选 PF19。下部 PF34、上部 PF19 是不同空间竞争区的面组决策，不是本阶段已经求出了换轨点。',None),
        ('E','R00629',104.30,[9,23,1377,1389],
         'E 的三来源为 PF16=E_T1、PF19=E_T2、PF34=E_T3。104.20–104.40 m 五线全部独立投 E_T2，仍与 210 条 V 聚合后的 E_T1 相反。区域首层贯穿数仅 24 比 22，已经决定结果；不能称为显著优势，更不能当作此前 E 修复的成功回归。',None),
        ('D','R00204',122.25,[-13,-1,1402,1415],
         'D 的 PF19=D_T1、PF23=D_T2。五邻线都独立投 PF23，区域也选 PF23；区域贯穿 478 比 16。但该连通竞争区域横跨 75.70 m，其他位置仍有不同独立票，需要结合全区证据条带审核。',None),
        ('AUTO_IDENTITY_REVIEW','R00268',89.95,[-9,8,1382,1401],
         '历史 89.95 m 来源不一致位置：当前 89.85–90.05 m 五条 V 全部独立投 PF19，因其向上观察出口仍有 1 m；PF34 已到原分支末端，出口为零。区域 R00268 却冻结 PF34。与 C07 使用同一个区域决策，直接暴露大区域内局部偏好反转。',None),
        ('NO_ELIGIBLE','R00008',7.20,[22.5,25.5,1357.5,1361.5],
         'R00008 共 23 条 V，PF10 与 PF26 均未通过 MBG，区域状态为 NO_ELIGIBLE。原始来源存在，但不够资格参与选面；这与两个可靠候选证据同分是不同问题。',None),
        ('AMBIGUOUS','R00761',103.65,[19.5,22,1379.5,1384],
         '最长的合格候选同分案例：R00761 共 13 条 V，PF19 与 PF34 都有 13 条贯穿、13 条 CC 支持、1 m 出口，保持 REGION_AMBIGUOUS。两者均为绿色且持续存在时，当前五项证据没有区分力。',None)]
    for label,rid,s,box,explanation,context in specs:
        atlas.case(label,rid,s,box,explanation,context);print('ATLAS_CASE',label,flush=True)
    for rid in sorted({c['region_id'] for c in atlas.cases}):atlas.regional_evidence(rid);atlas.full_score_table(rid)
    negative=atlas.negative();atlas.overview();atlas.overview(local=True);result=atlas.audit(negative)
    api.save_json(atlas.out/'RENDER_PERFORMANCE.json',dict(seconds=time.perf_counter()-began,scope='read-only figures, complete score tables and artifact audit; excluded from algorithm timings'))
    write_reports(atlas,result)
    print('ATLAS_COMPLETE',atlas.out,{k:v for k,v in result.items() if k not in ('cases','crosswalk','negative_control')},flush=True)


if __name__=='__main__':main()
