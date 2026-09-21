"""Compare every local FaceTrack together, using frozen observed geometry.

Presentation/diagnostic only: no routing, ASC policy or FaceTrack changes.
"""
import json,pickle,hashlib,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from review_p0_p1_p2 import (latest,SOURCE,load_inventory,branch_metrics,local_edge_scale,
    colored_parts,member_arc,interval_union,table,crop,plt,LineCollection,Line2D,COLORS)

FOCUS={'C':(1361.8,1364.15),'D':(1410.8,1413.1),'E':(1384.6,1387.),
       'F':(1392.8,1395.),'I':(1378.7,1381.1),'J':(1401.6,1403.)}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def view_rows(data,inventory,metrics,bounds,view_name):
    buckets=defaultdict(list);all_tids=sorted(t['face_track_id'] for t in data['tracks']);keys=data['region']['target_keys']
    for e in data['members']:
        assert len(e['face_track_ids'])==1,'Ambiguous membership must not be silently hidden'
        p=crop(e['points_uz'],bounds)
        if p is not None:buckets[(e['s'],e['face_track_ids'][0],e['branch_id'])].append(dict(e,points_uz=p.tolist()))
    rows=[];parts={};seen=set()
    by={s:{b['branch_id']:b for b in bs} for s,bs in inventory.items()}
    continuity={tid:len({e['s'] for e in data['members'] if e['face_track_ids']==[tid]}) for tid in all_tids}
    for s in keys:
        for tid in all_tids:
            members=[e for (ss,tt,_),mm in buckets.items() if ss==s and tt==tid for e in mm]
            parts[s,tid]=colored_parts(members,by[s],metrics[s])
            for bid in sorted({e['branch_id'] for e in members}):
                mm=[e for e in members if e['branch_id']==bid];b=by[s][bid];m=metrics[s][bid]
                intervals=interval_union([sorted(member_arc(e,b)[0]) for e in mm]);length=sum(hi-lo for lo,hi in intervals)
                core=sum(max(0.,min(hi,m['ASC_end_arc'])-max(lo,m['ASC_start_arc'])) for lo,hi in intervals) if m['ASC_exists'] else 0.
                zz=interval_union([sorted(np.asarray(e['points_uz'])[:,1]) for e in mm]);zcoverage=sum(hi-lo for lo,hi in zz)
                boundary=np.asarray(intervals).ravel();positions=np.interp(boundary,b['arc_positions'],np.arange(len(b['arc_positions'])))
                seen.update((s,tid,e['edge_id']) for e in mm)
                rows.append(dict(region=data['region']['region_id'],view=view_name,s=s,FaceTrack=tid,branch_id=bid,
                    kind=b['kind'],node_count=m['canonical_node_count'],full_arc_m=m['full_arc_length'],MBG=m['MBG_pass'],
                    ASC_exists=m['ASC_exists'],full_ASC_m=m['ASC_arc_length'],local_observed_m=length,local_ASC_m=core,
                    local_ASC_fraction=core/length if length else 0.,local_observed_Z_m=zcoverage,
                    local_Z_window_fraction=zcoverage/(bounds[3]-bounds[2]),in_ASC=bool(m['MBG_pass'] and core/length>=1-1e-8),
                    endpoint_nodes=float(np.minimum(positions,len(b['arc_positions'])-1-positions).min()),
                    endpoint_arc_m=float(np.minimum(boundary,b['arc_positions'][-1]-boundary).min()),face_continuity=continuity[tid]))
    expected={(e['s'],e['face_track_ids'][0],e['edge_id']) for e in data['members'] if crop(e['points_uz'],bounds) is not None}
    assert seen==expected,'A visible candidate was omitted'
    return rows,parts,len(expected)


def asc_comparison(rows,s):
    raw=[r for r in rows if r['s']==s];qualified=[r for r in raw if r['MBG']]
    raw_tids=sorted({r['FaceTrack'] for r in raw});mbg_tids=sorted({r['FaceTrack'] for r in qualified})
    surviving=[];sweet=[];winner=None
    if len(raw_tids)<2:reason='本窗口只有一个来源，不能验证跨 FaceTrack 竞争'
    elif len(mbg_tids)<2:reason='MBG 后不足两个来源，不能把结果归功于甜区'
    else:
        best=max(r['face_continuity'] for r in qualified);surviving=[r for r in qualified if r['face_continuity']==best]
        tids=sorted({r['FaceTrack'] for r in surviving})
        sweet=sorted({r['FaceTrack'] for r in surviving if r['in_ASC']})
        if len(tids)<2:reason='邻线来源持续性已区分，未到甜区比较'
        elif len(sweet)==1:
            winner=sweet[0];reason='ASC 位置判据给出唯一优先来源；不等于证明地质身份'
        elif len(sweet)>1:reason='多个来源都在 ASC，甜区不能单独选出优势来源'
        else:reason='均非整段 ASC；混合比例仅展示，不另造评分选赢家'
    return dict(s=s,visible_FaceTracks=raw_tids,MBG_FaceTracks=mbg_tids,
        continuity_FaceTracks=sorted({r['FaceTrack'] for r in surviving}),ASC_FaceTracks=sweet,
        ASC_unique_preference=winner,reason=reason)


def draw(data,rows,parts,bounds,out,name,center=False):
    reg=data['region'];keys=[reg['focus_s']] if center else reg['target_keys'];tids=sorted(t['face_track_id'] for t in data['tracks'])
    if center:fig,axs=plt.subplots(1,len(tids),figsize=(max(8,4.5*len(tids)),6.6),sharex=True,sharey=True,squeeze=False)
    else:fig,axs=plt.subplots(len(tids),5,figsize=(19,max(5,3.05*len(tids)+1.25)),sharex=True,sharey=True,squeeze=False)
    for row_index,tid in enumerate(tids):
        for col_index,s in enumerate(keys):
            ax=axs[0,row_index] if center else axs[row_index,col_index]
            rr=[r for r in rows if r['s']==s and r['FaceTrack']==tid];bids=sorted(r['branch_id'] for r in rr)
            for tag in ['REJECTED','ASC','GUARD']:
                lines=[p for p,t in parts[s,tid] if t==tag]
                if lines:ax.add_collection(LineCollection(lines,colors=COLORS[tag],linewidths=2.3 if tag=='GUARD' else 1.9,linestyles='--' if tag=='REJECTED' else '-'))
            if not bids:
                original=sorted({e['branch_id'] for e in data['members'] if e['s']==s and e['face_track_ids']==[tid]})
                text='未进入此显示窗口\n大窗口存在 '+','.join('B'+str(b) for b in original) if original else '这条测线无此 FaceTrack'
                ax.text(.5,.5,text,transform=ax.transAxes,ha='center',va='center',color='#777',fontsize=9)
            label=', '.join('B'+str(b) for b in bids) or '无局部分支'
            qualified=[r['branch_id'] for r in rr if r['MBG']]
            ax.set_title((f'{tid}｜s={s} m\n' if center else f's={s} m｜')+label,fontsize=11 if center else 9.5)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.grid(alpha=.16);ax.ticklabel_format(style='plain',useOffset=False)
            if (center and row_index==0) or (not center and col_index==0):ax.set_ylabel(('' if center else tid+'\n')+'高程 z（m）')
            if center or row_index==len(tids)-1:ax.set_xlabel('径向偏移 u（m）')
    title=f"区域 {reg['region_id']}｜{len(tids)} 个 FaceTrack 同窗口比较"+('｜中心测线' if center else '｜每行一个来源，每列一条邻线')
    fig.suptitle(title,fontsize=16,y=.985)
    fig.legend(handles=[Line2D([],[],color=COLORS['ASC'],lw=2,label='绿色：ASC / 甜区'),
        Line2D([],[],color=COLORS['GUARD'],lw=2.3,label='橙色：Endpoint Guard / 非甜区'),
        Line2D([],[],color=COLORS['REJECTED'],ls='--',label='灰虚线：MBG 未通过（仍展示）')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.035),fontsize=10)
    fig.text(.5,.012,'所有格子共用同一 u/z 范围；原始分支完整按来源列出。图框裁切不是端点；绿色不是“已选主轨”。',ha='center',fontsize=9)
    fig.subplots_adjust(left=.075,right=.99,top=.88 if center else .93,bottom=.15 if center else .105,hspace=.20,wspace=.10)
    fig.savefig(out/'figures'/name,dpi=170);plt.close(fig)


def metric_table(rows,s):
    return table(['FaceTrack / 分支','MBG','全分支节点 / 弧长 m','全分支 ASC m','本窗口 ASC占比 / 长m','本窗口实测 Z跨度 m','最近端点 节点 / 弧长m'],[
        (r['FaceTrack']+'/B'+str(r['branch_id']),'通过' if r['MBG'] else '未通过：'+('闭合旁支' if r['kind']=='CLOSED_COMPONENT' else '核心不足'),
         f"{r['node_count']} / {r['full_arc_m']:.3f}",f"{r['full_ASC_m']:.3f}",
         f"{r['local_ASC_fraction']:.1%} / {r['local_ASC_m']:.3f}" if r['MBG'] else '不参与甜区选择',
         f"{r['local_observed_Z_m']:.3f}",f"{r['endpoint_nodes']:.2f} / {r['endpoint_arc_m']:.3f}") for r in rows if r['s']==s])


def asc_explanation():
    return '''## ASC 是什么，来自哪里

ASC 是 **Absolute Sweet Core（绝对路径甜区/核心区）**，来自你提供并要求执行的 [最小分支与绝对甜区方案，第 5 节](C:/Users/222/Downloads/GDS_Minimum_Branch_Absolute_Sweet_Zone_Codex_Plan.md:199)。它的目的是防止只有几个节点的短分支因为“处于自身中间”而被误判可靠。ASC 不是新增曲面、拟合线、FaceTrack，也不是识别准确率。

沿一条原始分支从一端走到另一端，两头靠近原始拓扑端点的部分叫 endpoint guard；离两端足够多原始节点、并满足实际弧长要求的中间部分叫 ASC。这里的端点是算法拆出的分支端点，不等于已经确认的真实岩面终止点。绿色只是“满足这套位置/长度条件”，不是“已经证明是真实主坡面”。局部图框边缘不会成为计算甜区的新端点。

当前实现为：两端各排除 **4 条 canonical 原始边间隔**；节点数必须 **大于 11，即至少 12 个**；剩余核心弧长至少达到 **4 × 当前测线的局部中位边长**。K_guard=4、N_core_min=3 来自方案默认建议；4 倍长度是方案允许扫描集合中的当前实现参数，不是该方案唯一指定值。见 [BranchPolicy](D:/project_python/rock_recognize_github_release/scripts/04_structure_recognition/branch_absolute_core.py:8) 与 [实际计算](D:/project_python/rock_recognize_github_release/scripts/04_structure_recognition/branch_absolute_core.py:46)。该模块最早出现在本地提交 0a8e4bb（2026-09-20），不是上一轮新建的概念。

用累计原始弧长 a_i 表示第 i 个节点位置，n 为原始边数，则候选区间为 **[a_4, a_(n−4)]**；通过节点数和实际核心长度门槛后才成立。实现没有要求两端各排除固定多少米；各端 4 条边的实际米数由原始网格决定。端点弧长距离另作记录。举例：一条 21 节点、20 条边、每条 0.1 m 的分支，总长 2 m；两端各 0.4 m 为 guard，中间 1.2 m 才是 ASC。图上的折返即使在绿色范围内，也仍可能来自重建重复面。

## 静态 ASC、方向性判断与“动态 ASC”

| 名称 | 回答的问题 | 当前实现/来源 |
| --- | --- | --- |
| 静态 ASC | 此处距离原分支两端是否足够深？核心长度是否够？ | 原分支和参数不变时区间固定。你的方向性方案明确写了“ASC = 静态核心”。 |
| DRS：Directional Remaining Support | 从当前位置按当前方向继续，还能提供多少有效延展及邻域支持？ | 统计前方节点、剩余弧长、净 Z 延展、剩余 ASC、H/V 持续支持，并可标注方向尾区。 |
| “动态 ASC” | 容易让人误以为代码会重算另一套甜区边界 | 当前没有这样一套独立、完整实现；不应当把它作为已实现算法名称。 |
| 上一轮动态尾区对照 | 如果允许更早接点，会接到哪里、裁掉多少静态 ASC？ | 单接点反事实实验，未建立新的动态核心边界，也未证明可以安全裁核心。 |

明确来源：[方向性方案第 4 节](C:/Users/222/Downloads/GDS_Directional_Tail_Joint_Surface_Consensus_Codex_Plan.md:193) 区分 ASC 与 DRS；[第 28 节](C:/Users/222/Downloads/GDS_Directional_Tail_Joint_Surface_Consensus_Codex_Plan.md:1073) 写明保留静态 ASC。[directional_branch_metrics](D:/project_python/rock_recognize_github_release/scripts/04_structure_recognition/branch_absolute_core.py:93) 是方向性计算实现。它不改写 ASC 起止位置。剩余弧长很长也不一定还有有效前向延展，例如在末端绕圈。

## 上一轮新增了什么，我的解释哪里有问题

你的最新 P0 方案要求普通续接“不得越过 ASC 边界剪掉可靠核心”。我把它实现成普通续接必须完整保留已接受的静态 ASC；这项新增保护位于 [main_track_assembly.py](D:/project_python/rock_recognize_github_release/scripts/04_structure_recognition/main_track_assembly.py:67)。但静态位置核心与方向可靠核心并不等价，方向性方案此前已指出过这一点。

因此，ASC 概念有你的方案依据；把“已保留的全部静态 ASC”落实为普通续接的硬保护、以及没有说明它与方向尾区的矛盾，是我这一轮的实现与说明责任，不能笼统归因于你要求了甜区。全区退步说明这项保护不能直接作为已验证成功的生产策略。你允许另做动态尾区验证，也不等于批准裁剪核心。

## 这次图册比较什么

同一个局部窗口内，列出全部 FaceTrack 和它们的全部原始分支；所有来源共用同一坐标范围。中心图按列比较不同 FaceTrack；五邻线图按行列出 FaceTrack、按列列出相邻测线。绿色 ASC、橙色 guard、灰虚线 MBG 未通过，只有这三类图例。

分支编号与 FaceTrack 不是一一对应：同一 FaceTrack 在一个切面内可能包含主片段、闭合旁支或短碎片。所有这些片段都列在分支清单与图中，未先选一条再画。缺席格子明确写“本细窗未进入”或“该测线无此来源”。

本轮只观察甜区阶段本身能否区分候选：先列 MBG 与邻线来源持续性，再比较竞争片段是否处于 ASC。若两个来源都是绿色，就不能宣称甜区选出了唯一优势来源。若 MBG 已经只剩一个来源，也不能把结果归功于甜区。局部 ASC 占比只反映图中片段位置；100% 不表示覆盖整个窗口，因此另列实测 Z 跨度，且不把百分比当成正确率。

**“整个窗口是否全部在 ASC”是上一轮影子代码采用的二值化口径，不是已经证实的显著优势标准。** 100% 与 99.4% 会被它分开，但这可能仅来自一端少量 guard；不能据此把整个 FaceTrack 的真实身份排定。具体重叠位置都在绿色区时，ASC 没有区分力。
'''


def main():
    parent=latest();out=parent/'competition_review';(out/'figures').mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    datasets=[pickle.load(p.open('rb')) for p in sorted((SOURCE/'region_data').glob('*.pkl'))]
    keys=sorted({s for d in datasets for s in d['region']['target_keys']},key=float);inventory=load_inventory(SOURCE,keys)
    metrics={s:{b['branch_id']:branch_metrics(b,local_edge_scale(bs)) for b in bs} for s,bs in inventory.items()}
    source_files=list((Path(__file__).parents[1]/'04_structure_recognition').glob('*.py'))
    before={str(p):digest(p) for p in source_files};rows_all=[];decisions=[];index=[];counts=[]
    for data in datasets:
        reg=data['region'];rid=reg['region_id'];tids=sorted(t['face_track_id'] for t in data['tracks']);bounds=reg['bounds'][2:]
        full_rows,full_parts,n=view_rows(data,inventory,metrics,bounds,'full');rows_all.extend(full_rows)
        parent_inventory={s:{tid:sorted({e['branch_id'] for e in data['members'] if e['s']==s and e['face_track_ids']==[tid]}) for tid in tids} for s in reg['target_keys']}
        text=[f'# 区域 {rid}：同一区域内的 {len(tids)} 个 FaceTrack 竞争对照',
            f"大窗口：u={bounds[0]:.3f}–{bounds[1]:.3f} m，z={bounds[2]:.3f}–{bounds[3]:.3f} m。中心测线 {reg['focus_s']} m。所有来源、所有分支均列出。",
            table(['测线 s']+tids,[(s,*((', '.join('B'+str(b) for b in parent_inventory[s][tid])) or '本区域无此来源' for tid in tids)) for s in reg['target_keys']]),
            '## 同窗口的五条邻线与全部竞争来源',f'![{rid} 全部来源五邻线](../figures/{rid}_all_tracks_full.png)',
            '## 中心测线：不同 FaceTrack 放在一起',f'![{rid} 中心来源对比](../figures/{rid}_center_full.png)',metric_table(full_rows,reg['focus_s'])]
        draw(data,full_rows,full_parts,bounds,out,f'{rid}_all_tracks_full.png');draw(data,full_rows,full_parts,bounds,out,f'{rid}_center_full.png',True)
        views=[('full',full_rows)]
        if rid in FOCUS:
            lo,hi=FOCUS[rid];p=[crop(e['points_uz'],[bounds[0],bounds[1],lo,hi]) for e in data['members']];p=[v for v in p if v is not None]
            if p:
                pts=np.concatenate(p);pad=max(.08,np.ptp(pts[:,0])*.05);focus=[float(pts[:,0].min()-pad),float(pts[:,0].max()+pad),lo,hi]
                fr,fp,fn=view_rows(data,inventory,metrics,focus,'detail');rows_all.extend(fr);views.append(('detail',fr))
                draw(data,fr,fp,focus,out,f'{rid}_all_tracks_detail.png');draw(data,fr,fp,focus,out,f'{rid}_center_detail.png',True)
                text.extend(['## 同一个细节窗口比较全部来源',f'u={focus[0]:.3f}–{focus[1]:.3f} m；z={lo:.3f}–{hi:.3f} m。所有来源保持相同图框，未进入的来源保留空格说明。',
                    f'![{rid} 同细窗中心比较](../figures/{rid}_center_detail.png)',f'![{rid} 同细窗五邻线比较](../figures/{rid}_all_tracks_detail.png)',metric_table(fr,reg['focus_s'])])
        for view,rr in views:
            ds=[dict(region=rid,view=view,**asc_comparison(rr,s)) for s in reg['target_keys']];decisions.extend(ds)
            text.extend([f'## {"大窗口" if view=="full" else "细节窗口"}：甜区是否独立区分了优势来源',table(['s','实际来源','MBG 后来源','ASC 位置唯一优先','理由'],[
                (d['s'],', '.join(d['visible_FaceTracks']),', '.join(d['MBG_FaceTracks']),d['ASC_unique_preference'] or '未区分',d['reason']) for d in ds])])
        text.extend(['## 人工审核',
            '先比较同一列测线内不同 FaceTrack，再横向看其优势是否在五条邻线持续。两条都绿不能强行选；一条绿、一条橙只提供位置优先证据。仍需结合真实面来源与方向支持确认目标曲面，不用形状分数替你决定。',
            '全分支 ASC 长度和本窗口 ASC 比例是不同量；不因某条完整分支很长就自动判它胜出。灰虚线仍是原始观测，只是没有通过进入主轨的 MBG；闭合旁支同样保留。数据来自上一轮冻结结果，本册没有重跑识别或改变甜区阈值。'])
        (out/'cases'/f'{rid}.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
        center=asc_comparison(full_rows,reg['focus_s']);counts.append((rid,len(tids),', '.join(tids),center['ASC_unique_preference'] or '未区分',center['reason']))
        index.append(dict(region=rid,FaceTracks=tids,keys=reg['target_keys'],parent_branch_inventory=parent_inventory,visible_source_edges_checked=n))
        print('COMPETITION_REGION',rid,len(tids),center['ASC_unique_preference'],flush=True)
    assert len(index)==12 and sum(len(x['FaceTracks']) for x in index)==23
    assert before=={str(p):digest(p) for p in source_files},'Production code changed during graphics-only task'
    from validate_physical_topology import csv_rows
    csv_rows(out/'competition_candidates.csv',rows_all)
    for filename,value in [('competition_decisions.json',decisions),('gallery_inventory.json',index),('CHECKS.json',dict(regions=12,FaceTrack_groups=23,
        all_visible_source_edges_accounted_for=True,production_code_unchanged=True,recognition_rerun=False,ASC_parameters_changed=False,
        same_bounds_across_FaceTracks_and_neighbors=True,candidate_rows=len(rows_all),source_hashes=before))]:
        (out/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# ASC 定义说明与同一区域 FaceTrack 竞争图册',
        '**这次按“同一区域里的所有竞争来源”组织图册。上一轮拆开的单来源页面适合看来源内部一致性，但不够用来判断多个 FaceTrack 中谁更有甜区优势。**',asc_explanation(),
        '## 全部区域的中心测线对照',table(['区域','来源数','全部 FaceTrack','ASC 位置唯一优先','解释'],[(f'[{rid}](cases/{rid}.md)',n,tids,w,why) for rid,n,tids,w,why in counts]),
        '这里的唯一优先只采用 MBG → 来源持续性 → ASC 位置三步，不等于上一轮把方向支持、最少换轨等也算入的最终 shadow 选择。本轮专门分清“甜区本身有没有作用”。',
        '## I：实际有四个来源，不是一条分支',
        '中心 s=100.15 m：I_T1=B0；I_T2=B3；I_T3=B4、B8；I_T4=B6、B9、B10。分支与 FaceTrack 不是一对一。下面四栏是同一测线、同一大窗口，所有来源同时列出。',
        '![I 的四个竞争 FaceTrack](figures/I_center_full.png)',
        'I_T1 的原始片段只到 z≈1378.197 m；所以在 z=1378.7–1381.1 m 的细节窗里不出现。以前给它单独缩放，容易误以为不同来源画的是同一范围。本次保留明确空格，其他三个来源在同一个细节窗比较。',
        '![I 同一细节窗口的全部来源](figures/I_center_detail.png)','[I 的五邻线全来源图、分支清单及数值比较](cases/I.md)',
        '## E：三个 FaceTrack 的直接竞争',
        '![E 三来源同窗口](figures/E_center_detail.png)','[E 的五邻线竞争对照与逐候选表](cases/E.md)',
        '## F：两个来源都在甜区，不能由甜区单独决胜',
        '![F 两来源同窗口](figures/F_center_detail.png)','[F 的五邻线竞争对照](cases/F.md)',
        '## 检查与保留问题',
        '一次图册生成检查覆盖 12 区域、23 个 FaceTrack；逐窗口核对所有原始来源边均已入图，坐标范围在来源与邻线之间一致，生产识别文件哈希保持不变。复用原始分支/FaceTrack 数据，没有再次跑 2800 条识别，也没有修改 ASC 或 DRS 算法。',
        '这次只修正解释与比较方式。静态 ASC 的硬保护造成上一轮停轨退步的问题仍存在；曲面真伪、动态尾区可否替换静态核心以及两条都在甜区时的取舍仍由后续审核决定。',
        '数据：[全部候选度量](competition_candidates.csv)、[逐测线 ASC 比较过程](competition_decisions.json)、[全部来源与分支清单](gallery_inventory.json)、[最小检查](CHECKS.json)。']
    (out/'FACETRACK_COMPETITION_REVIEW.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print('COMPETITION_GALLERY_COMPLETE',out,flush=True)


def focused_e():
    parent=latest();out=parent/'competition_review';data=pickle.load((SOURCE/'region_data/E.pkl').open('rb'));keys=data['region']['target_keys'];inventory=load_inventory(SOURCE,keys)
    metrics={s:{b['branch_id']:branch_metrics(b,local_edge_scale(bs)) for b in bs} for s,bs in inventory.items()};results=[];allrows=[]
    views=[('endpoint',[9.6,11.0,1385.03,1385.35],'端点附近'),('interior',[12.,19.,1385.35,1386.85],'三者共同的内部区域')]
    text=['## 补充核对：端点位置优势，不等于整个 FaceTrack 的明显优势',
        '以下两张均为 s=104.30 m，同时展示 E_T1/B1、E_T2/B2、E_T3/B6；只改变共同显示窗口。ASC 仍按完整原分支端点计算，没有随裁图重算。']
    for name,bounds,label in views:
        rows,parts,n=view_rows(data,inventory,metrics,bounds,name);decision=asc_comparison(rows,'104.30');allrows.extend(rows);results.append(dict(view=name,bounds=bounds,visible_source_edges_checked=n,**decision))
        draw(data,rows,parts,bounds,out,f'E_{name}_comparison.png',True)
        text.extend([f'### {label}',f'共同窗口 u={bounds[0]}–{bounds[1]} m，z={bounds[2]}–{bounds[3]} m。',
            f'![E {label}](../figures/E_{name}_comparison.png)',metric_table(rows,'104.30'),
            'ASC 位置唯一优先：'+(decision['ASC_unique_preference'] or '无')+'。'+decision['reason']])
    assert results[0]['ASC_unique_preference']=='E_T2'
    assert results[1]['ASC_unique_preference'] is None and len(results[1]['ASC_FaceTracks'])==3
    text.append('结论：在端点附近，B1/B6 有部分橙色 guard，B2 保持绿色，ASC 对该位置有区分作用；移到三者共同内部后，三条都绿，ASC 不再能选出唯一来源。不能把前一种局部优势外推成整条 E_T2 必然最好。完整曲面身份与方向性延展仍需另外审核。')
    section='\n\n'.join(text)+'\n';case=out/'cases/E.md';base=case.read_text(encoding='utf-8').split('## 补充核对：')[0];case.write_text(base+'\n'+section,encoding='utf-8')
    report=out/'FACETRACK_COMPETITION_REVIEW.md';body=report.read_text(encoding='utf-8')
    body=body.replace('沿一条原始分支从一端走到另一端，两头靠近真实拓扑端点的部分叫 endpoint guard；','沿一条原始分支从一端走到另一端，两头靠近原始拓扑端点的部分叫 endpoint guard（这些端点不等于已确认的真实岩面终止点）；')
    body=body.replace('## E：三个 FaceTrack 的直接竞争','## E：三个 FaceTrack 的直接竞争\n\n现有整窗口二值规则会把 100% 与 99.4%、98.9% 分开，但不能把这种区别当成特别显著的整体优势。下面补充的两处共同窗口更直接回答甜区能否决胜。')
    body=body.replace('## F：两个来源都在甜区，不能由甜区单独决胜',
        section.replace('../figures/','figures/')+'\n## F：两个来源都在甜区，不能由甜区单独决胜')
    report.write_text(body,encoding='utf-8')
    from validate_physical_topology import csv_rows
    csv_rows(out/'E_targeted_window_candidates.csv',allrows)
    (out/'E_targeted_window_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FOCUSED_E_COMPLETE',[(r['view'],r['ASC_unique_preference'],r['ASC_FaceTracks']) for r in results],flush=True)


if __name__=='__main__':
    if '--focused-e' in sys.argv:focused_e()
    else:main()
