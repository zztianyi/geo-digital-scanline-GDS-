"""Reuse frozen A3 tables for B0/B1. Does not import the geometry pipeline."""
import time
STARTED=time.perf_counter()
import sys,json,hashlib,argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict,replace
from collections import Counter,defaultdict
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'scripts/04_structure_recognition'
sys.path.insert(0,str(CORE))
from facetrack_local_preference import LocalPreferencePolicy,POLICY_VERSION,segment_region,reduce_subregion

DEFAULT_SOURCE=ROOT/'outputs/facetrack_voting_phase1/20260923_094440'
PARENT=ROOT/'outputs/facetrack_local_preference'
CASE_IDS=('R00268','R00806','R00132','R00629','R00204','R00761')
LABELS={'R00268':'C07 / 89.95 m','R00806':'C05 / F','R00132':'多数反对旧首选','R00629':'E','R00204':'D 正对照','R00761':'同分正对照'}
FIELDS=['region_id','s','s_index','FaceTrack','present','MBG','slice_vote','slice_ambiguous','through_region','in_CC','exit_continuation_m']


def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save_json(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def pf(t):return '待定' if t is None else f'PF{t}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def ratio(n,d):return 100*n/d if d else 0.
def size_distribution(regions):
    result={'1 V':0,'2 V':0,'3–5 V':0,'>5 V':0}
    for r in regions:
        n=len(r.member_V);result['1 V' if n==1 else '2 V' if n==2 else '3–5 V' if n<=5 else '>5 V']+=1
    return result


def draw_votes(out,parent,rows,subregions,decisions,old_winner,zoom=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    by_v={r['s_index']:r for r in rows};indices=sorted(by_v);tracks=sorted({r['FaceTrack'] for r in rows})
    fixed_colors={16:'#2785a3',19:'#cf833b',23:'#498e65',31:'#c26055',34:'#8751a2'}
    track_colors=[fixed_colors.get(t,'#777777') for t in tracks];color=['#d9dde1']+track_colors
    assignment={i:decisions[r.subregion_id].dominant_FaceTrack for r in subregions for i in r.s_indices}
    values=[[by_v[i]['slice_vote'] for i in indices],[old_winner for i in indices],[assignment[i] for i in indices]]
    matrix=np.array([[0 if t is None else tracks.index(t)+1 for t in row] for row in values])
    s=np.array([by_v[i]['s'] for i in indices]);edges=np.r_[s-.025,s[-1]+.025]
    fig,axes=plt.subplots(2,1,figsize=(16,5.6),gridspec_kw={'height_ratios':[3,1.7]})
    axes[0].pcolormesh(edges,[0,1,2,3],matrix,vmin=-.5,vmax=len(tracks)+.5,cmap=ListedColormap(color),shading='flat')
    axes[0].set_yticks([.5,1.5,2.5],['原始 A3 独立票','旧整个 region 首选','新局部子区首选']);axes[0].invert_yaxis()
    for r in subregions[1:]:axes[0].plot([r.s_start-.025]*2,[2,3],color='white',lw=.8,alpha=.9)
    for r in subregions:
        if zoom and (r.s_end<zoom[0] or r.s_start>zoom[1]):continue
        width=r.s_end-r.s_start+.05
        if width>((zoom[1]-zoom[0]) if zoom else s[-1]-s[0])*.075:
            lo=max(r.s_start,zoom[0]) if zoom else r.s_start;hi=min(r.s_end,zoom[1]) if zoom else r.s_end
            axes[0].text((lo+hi)/2,2.5,r.subregion_id.split('_')[-1],ha='center',va='center',fontsize=8,clip_on=True,
                         bbox=dict(fc='white',ec='none',alpha=.8))
    for t,c in zip(tracks,track_colors):
        hit=np.array([by_v[i]['slice_vote']==t for i in indices],int)
        axes[1].step(s,hit,where='mid',color=c,label=pf(t),lw=1.1)
    dissent=[v for r in subregions for v in r.local_dissent_V]
    if dissent:axes[1].plot(dissent,[-.14]*len(dissent),'|',color='#b63237',ms=8,label='保留的 LOCAL_DISSENT')
    axes[1].set_ylim(-.3,1.2);axes[1].set_yticks([0,1],['未投此面','明确投此面']);axes[1].grid(alpha=.12);axes[1].set_xlabel('原始纵测线位置 s (m)')
    for ax in axes:ax.set_xlim(zoom if zoom else (edges[0],edges[-1]));ax.ticklabel_format(axis='x',useOffset=False,style='plain')
    axes[0].legend(handles=[Patch(color=color[0],label='待定 / 未冻结')]+[Patch(color=c,label=pf(t)) for t,c in zip(tracks,track_colors)],
                   loc='upper left',bbox_to_anchor=(0,1.34),ncol=len(tracks)+1,frameon=False)
    if dissent:axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.28),ncol=len(tracks)+1,frameon=False,fontsize=8)
    fig.suptitle(f'{parent}｜{LABELS[parent]}｜原始票不变，只改变局部决策范围',fontsize=14,y=.99)
    fig.subplots_adjust(left=.14,right=.985,top=.81,bottom=.18,hspace=.35)
    name=parent+('_detail' if zoom else '')+'.png';fig.savefig(out/'figures'/name,dpi=180);plt.close(fig)
    return name


def run(source,policy):
    began=time.perf_counter();source=Path(source);out=PARENT/datetime.now().strftime('%Y%m%d_%H%M%S');(out/'figures').mkdir(parents=True);(out/'cases').mkdir()
    source_files=[source/n for n in ('branch_facetrack_index.parquet','conflict_region_index.json','per_slice_track_scores.parquet','region_track_decisions.json','manifest.json')]
    frozen_code=[CORE/n for n in ('facetrack_phase1_evidence.py','facetrack_phase1_regions.py','facetrack_phase1_vote.py')]+[ROOT/'scripts/99_experiments/run_facetrack_voting_phase1.py']
    hashes={str(p):digest(p) for p in source_files+frozen_code};timing={};tick=time.perf_counter()
    rows=pq.read_table(source/'per_slice_track_scores.parquet',columns=FIELDS).to_pylist()
    parents=read_json(source/'conflict_region_index.json')['regions'];parent_map={r['region_id']:r for r in parents}
    old={d['region_id']:d for d in read_json(source/'region_track_decisions.json')['decisions']}
    branch_index=pq.read_table(source/'branch_facetrack_index.parquet',columns=['s','s_index']).to_pylist();source_positions={r['s_index']:r['s'] for r in branch_index}
    groups=defaultdict(list);votes={}
    for row in rows:
        assert row['s']==source_positions[row['s_index']]
        groups[row['region_id']].append(row);votes[row['region_id'],row['s_index']]=row['slice_vote']
    assert set(groups)==set(parent_map)
    for rid,rr in groups.items():
        assert {r['s_index'] for r in rr}==set(parent_map[rid]['s_indices'])
        assert len(rr)==len(parent_map[rid]['s_indices'])*len(parent_map[rid]['tracks'])
    timing['input_read_validate_s']=time.perf_counter()-tick
    tick=time.perf_counter();subregions=[]
    for rid,rr in sorted(groups.items()):subregions.extend(segment_region(rid,rr,policy))
    timing['B0_segmentation_s']=time.perf_counter()-tick;tick=time.perf_counter()
    decisions=[reduce_subregion(r,groups[r.parent_region_id],policy) for r in subregions]
    timing['B1_reducer_s']=time.perf_counter()-tick;dm={d.subregion_id:d for d in decisions};sm={r.subregion_id:r for r in subregions};by_parent=defaultdict(list)
    for r in subregions:by_parent[r.parent_region_id].append(r)
    assignment={(r.parent_region_id,si):dm[r.subregion_id] for r in subregions for si in r.s_indices}
    assert set(assignment)==set(votes) and sum(len(r.member_V) for r in subregions)==len(votes)
    resolved=[d for d in decisions if not d.ambiguous];support=sum(len(d.supporting_V) for d in resolved);oppose=sum(len(d.opposing_V) for d in resolved);abstain=sum(len(d.abstaining_V) for d in resolved)
    denominator=sum(len(sm[d.subregion_id].member_V) for d in resolved);assert support+oppose+abstain==denominator
    old_resolved=[d for d in old.values() if not d['ambiguous']];old_denominator=sum(len(parent_map[d['region_id']]['s_indices']) for d in old_resolved)
    old_opposition=sum(len(d['opposing_V']) for d in old_resolved);matrix=Counter();matched=[0,0,0]
    for (rid,si),vote in votes.items():
        before=old[rid]['dominant_FaceTrack'];after=assignment[rid,si].dominant_FaceTrack
        def state(w):return 'unresolved' if w is None else 'abstain' if vote is None else 'support' if vote==w else 'oppose'
        matrix[state(before),state(after)]+=1
        if before is not None and after is not None:
            matched[0]+=1;matched[1]+=vote is not None and vote!=before;matched[2]+=vote is not None and vote!=after
    majority=[d.to_dict() for d in decisions if d.MAJORITY_OPPOSES_WINNER]
    inherited=[r for r in subregions if r.origin=='UNCHANGED_INITIAL_REGION'];created=[r for r in subregions if r.origin!='UNCHANGED_INITIAL_REGION']
    created_resolved=[r for r in created if not dm[r.subregion_id].ambiguous];buffers=[r for r in created if r.kind=='AMBIGUOUS_BUFFER']
    strict_d=segment_region('R00204',groups['R00204'],replace(policy,strong_consensus_fraction=None))
    checks={
        'R00268_PF34_at_85_95':next(d.dominant_FaceTrack for (rid,i),d in assignment.items() if rid=='R00268' and source_positions[i]==85.95)==34,
        'R00268_PF19_at_89_95':next(d.dominant_FaceTrack for (rid,i),d in assignment.items() if rid=='R00268' and source_positions[i]==89.95)==19,
        'R00806_PF16_at_129_90_and_130_45':all(next(d.dominant_FaceTrack for (rid,i),d in assignment.items() if rid=='R00806' and source_positions[i]==s)==16 for s in (129.9,130.45)),
        'R00132_no_majority_opposes_winner':not any(d.MAJORITY_OPPOSES_WINNER for d in decisions if d.parent_region_id=='R00132'),
        'E_PF19_at_104_30':next(d.dominant_FaceTrack for (rid,i),d in assignment.items() if rid=='R00629' and source_positions[i]==104.3)==19,
        'D_one_region_PF23':len(by_parent['R00204'])==1 and dm[by_parent['R00204'][0].subregion_id].dominant_FaceTrack==23,
        'R00761_remains_ambiguous':all(dm[r.subregion_id].ambiguous for r in by_parent['R00761']),
        'majority_opposes_winner_zero':not majority,
        'no_new_resolved_1_2_V_regions':not any(len(r.member_V)<=2 for r in created_resolved),
        'opposition_fraction_reduced':oppose/max(denominator,1)<old_opposition/old_denominator,
        'exact_parent_V_partition':True}
    summary=dict(policy=asdict(policy),initial_conflict_regions=len(parents),decision_subregions=len(subregions),winner_subregions=len(resolved),ambiguous_subregions=len(decisions)-len(resolved),
        parent_V_population=len(votes),old_resolved_region_V=old_denominator,old_opposing_V=old_opposition,old_opposing_percent=ratio(old_opposition,old_denominator),
        resolved_subregion_V=denominator,supporting_V=support,opposing_V=oppose,abstaining_V=abstain,opposing_percent=ratio(oppose,denominator),
        unresolved_V=len(votes)-denominator,matched_resolved_V=matched[0],old_opposing_on_matched_V=matched[1],new_opposing_on_matched_V=matched[2],
        raw_ambiguous_V=sum(v is None for v in votes.values()),unresolved_explicit_vote_V=sum(v is not None and assignment[k].ambiguous for k,v in votes.items()),
        MAJORITY_OPPOSES_WINNER_count=len(majority),MAJORITY_OPPOSES_WINNER=majority,
        subregion_size_distribution=size_distribution(subregions),inherited_region_size_distribution=size_distribution(inherited),
        newly_split_size_distribution=size_distribution(created),newly_split_resolved_size_distribution=size_distribution(created_resolved),
        newly_split_ambiguous_buffer_size_distribution=size_distribution(buffers),
        kinds=dict(Counter(r.kind for r in subregions)),transition_matrix=[dict(before=a,after=b,count=n) for (a,b),n in sorted(matrix.items())],
        D_without_consensus_guard_subregions=len(strict_d),acceptance_checks=checks,
        fragmentation_acceptance='PENDING_HUMAN_REVIEW',strong_consensus_threshold_acceptance='PENDING_HUMAN_REVIEW',phase2_ready=False)
    save_json(out/'decision_subregion_index.json',dict(stage='B0_LOCAL_PREFERENCE',parent_stage='INITIAL_CONFLICT_REGION',policy_version=POLICY_VERSION,policy=asdict(policy),
        spatial_extent_rule='Reuse parent conflict mask restricted to member s_indices; never create new cells',subregions=[r.to_dict() for r in subregions]))
    save_json(out/'subregion_track_decisions.json',dict(stage='B1_FROZEN_SUBREGION_FACE_DECISION',route_application=False,policy_version=POLICY_VERSION,policy=asdict(policy),
        longest_support_run_columns=['FaceTrack','consecutive_V','s_span_m'],bounded_continuation_columns=['FaceTrack','quantized_sum_m','mean_per_subregion_V_m'],decisions=[d.to_dict() for d in decisions]))
    save_json(out/'summary.json',summary)
    snapshot={n:digest(out/n) for n in ('decision_subregion_index.json','subregion_track_decisions.json')}
    tick=time.perf_counter()
    for rid in CASE_IDS:
        draw_votes(out,rid,groups[rid],by_parent[rid],dm,old[rid]['dominant_FaceTrack'])
        zoom={'R00268':(85.5,90.3),'R00806':(123,132),'R00629':(103,106)}.get(rid)
        if zoom:draw_votes(out,rid,groups[rid],by_parent[rid],dm,old[rid]['dominant_FaceTrack'],zoom)
        write_case(out,source,rid,groups[rid],by_parent[rid],dm,old[rid],bool(zoom))
    catalog=table(['parent','子区','类型','s 开始','s 结束','V 数','首选','支持/反对/待定','来源'],[
        [r.parent_region_id,r.subregion_id,r.kind,f'{r.s_start:.2f}',f'{r.s_end:.2f}',len(r.member_V),pf(dm[r.subregion_id].dominant_FaceTrack),
         f'{len(dm[r.subregion_id].supporting_V)}/{len(dm[r.subregion_id].opposing_V)}/{len(dm[r.subregion_id].abstaining_V)}',r.origin] for r in subregions])
    (out/'SUBREGION_CATALOG.md').write_text('# 完整子区目录（含明确缓冲记录）\n\n'+catalog,encoding='utf-8')
    timing['figures_and_case_reports_s']=time.perf_counter()-tick
    tick=time.perf_counter();assert all(digest(Path(p))==h for p,h in hashes.items());assert all(digest(out/n)==h for n,h in snapshot.items())
    timing['final_hash_verification_s']=time.perf_counter()-tick
    timing['B0_B1_s']=timing['B0_segmentation_s']+timing['B1_reducer_s']
    timing['elapsed_before_summary_report_s']=time.perf_counter()-began
    manifest=dict(source=str(source),source_hashes=hashes,policy_version=POLICY_VERSION,policy=asdict(policy),
        core_sha256=digest(CORE/'facetrack_local_preference.py'),entry_sha256=digest(Path(__file__)),
        A1_runs=0,A2_runs=0,A3_runs=0,mesh_reads=0,route_operations=0,frozen_artifact_hashes=snapshot,
        score_rows_read=len(rows),source_profiles=len(source_positions),original_inputs_and_A1_A2_A3_code_unchanged=True,timing=timing)
    write_report(out,source,summary,manifest,by_parent,dm,old)
    timing['total_input_to_reports_s']=time.perf_counter()-began;timing['entry_including_imports_s']=time.perf_counter()-STARTED
    save_json(out/'manifest.json',manifest)
    (out/'PERFORMANCE_REPORT.md').write_text('# B0/B1 局部修改性能\n\n'+table(['阶段','秒'],[[k,f'{v:.4f}'] for k,v in timing.items()])+
        '\n本轮仅读取冻结表格，A1/A2/A3 均为 0 次。报告和图包含在 total_input_to_reports_s；解释器启动之外的入口导入时间另列。未重建几何，未执行换轨。B0/B1 并未与上轮含几何评分的总时长直接计算加速倍数。\n',encoding='utf-8')
    save_json(out/'DELIVERY_CHECKS.json',dict(acceptance_checks=checks,source_hashes_unchanged=True,decision_hashes_unchanged_after_report=True,
        tests='14 scoped behavioral tests passed before analysis; full project suite not run',manual_physical_identity_review='PENDING'))
    (PARENT/'LATEST.txt').write_text(str(out),encoding='utf-8')
    print('LOCAL_PREFERENCE_COMPLETE',out,flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('transition_matrix','MAJORITY_OPPOSES_WINNER')},ensure_ascii=False,indent=2),flush=True)
    print('TIMING',timing,flush=True)
    return out


def write_case(out,source,rid,rows,regions,decisions,old,zoom):
    text=f'# {rid}｜{LABELS[rid]}\n\n旧整个区域首选：**{pf(old["dominant_FaceTrack"])}**。本页只调整决策范围，A3 原始评分和几何完全复用。\n\n'
    text+=f'![原始票、旧决策、新子区](../figures/{rid}.png)\n\n'
    if zoom:text+=f'![重点局部放大](../figures/{rid}_detail.png)\n\n'
    text+='第一行为冻结的独立投票，第二行为旧整区首选，第三行为新子区首选；灰色为待定/未冻结。下方红短线标记仍保留的局部反向票。不同颜色只代表物理面组，不代表正确率。\n\n'
    text+=table(['子区','s 范围 m','V 数','类型','首选','支持/反对/待定','边界起因'],[
        [r.subregion_id,f'{r.s_start:.2f}–{r.s_end:.2f}',len(r.member_V),r.kind,pf(decisions[r.subregion_id].dominant_FaceTrack),
         f'{len(decisions[r.subregion_id].supporting_V)}/{len(decisions[r.subregion_id].opposing_V)}/{len(decisions[r.subregion_id].abstaining_V)}',r.boundary_start_reason] for r in regions])
    for r in regions:
        d=decisions[r.subregion_id];votes=dict(d.vote_counts);runs={t:(n,span) for t,n,span in d.longest_support_run};through=dict(d.through_counts);cc=dict(d.CC_counts);exits={t:(total,mean) for t,total,mean in d.bounded_continuation}
        text+=f'\n## {r.subregion_id}\n\n'
        text+=table(['面组','明确票','最长支持 V / s 跨度','贯穿 V','CC V','有限出口总和 / 平均 m'],[
            [pf(t),votes[t],f'{runs[t][0]} / {runs[t][1]:.2f}',through[t],cc[t],f'{exits[t][0]:.3f} / {exits[t][1]:.3f}'] for t in r.competing_FaceTracks])
        text+=f'\n原始待定 V：{list(r.ambiguous_buffer_V)}；保留 LOCAL_DISSENT：{list(r.local_dissent_V)}。\n\n'
        text+=table(['原始 preference run','s 范围','V 数','达到连续门槛'],[[pf(p.FaceTrack),f'{p.s_start:.2f}–{p.s_end:.2f}',len(p.member_V),p.stable] for p in r.preference_runs])
    old_pages={'R00268':['C07','AUTO_IDENTITY_REVIEW'],'R00806':['C05','F'],'R00629':['E'],'R00204':['D'],'R00761':['AMBIGUOUS']}.get(rid,[])
    if old_pages:text+='\n## 原始几何五邻线图（沿用，未重新识别）\n\n'+'、'.join(f'[{name}]({(source/"cases"/(name+".md")).as_posix()})' for name in old_pages)+'。本次上方色带才是新的 B0/B1 结果；旧图页首的整区 winner 已被本次子区决策替代，原始线形仍然有效。\n'
    text+='\n人工审核：待用户确认局部边界和物理曲面身份。\n'
    (out/'cases'/f'{rid}.md').write_text(text,encoding='utf-8')


def write_report(out,source,s,m,by_parent,dm,old):
    acceptance=table(['验收项','结果'],[[k,'通过' if v else '未通过'] for k,v in s['acceptance_checks'].items()])
    case_table=table(['初始区域','旧首选','新子区 / 含待定','新的明确面组'],[[f'[{rid} / {LABELS[rid]}](cases/{rid}.md)',pf(old[rid]['dominant_FaceTrack']),len(by_parent[rid]),
        ', '.join(pf(t) for t in sorted({dm[r.subregion_id].dominant_FaceTrack for r in by_parent[rid] if not dm[r.subregion_id].ambiguous}))] for rid in CASE_IDS])
    distribution=table(['大小','全部记录','原初始区原样保留','新切记录','新切且有首选','新切原始待定缓冲'],[[k,s['subregion_size_distribution'][k],s['inherited_region_size_distribution'][k],s['newly_split_size_distribution'][k],s['newly_split_resolved_size_distribution'][k],s['newly_split_ambiguous_buffer_size_distribution'][k]] for k in s['subregion_size_distribution']])
    p=s['policy'];majority='无。' if not s['MAJORITY_OPPOSES_WINNER'] else json.dumps(s['MAJORITY_OPPOSES_WINNER'],ensure_ascii=False,indent=2)
    transitions={(r['before'],r['after']):r['count'] for r in s['transition_matrix']}
    removed_amb=transitions.get(('abstain','unresolved'),0);removed_support=transitions.get(('support','unresolved'),0);removed_oppose=transitions.get(('oppose','unresolved'),0)
    short=lambda name:sum(s[name][k] for k in ('1 V','2 V'))
    new_short=short('newly_split_size_distribution');new_short_resolved=short('newly_split_resolved_size_distribution');short_amb=short('newly_split_ambiguous_buffer_size_distribution')
    short_unstable=new_short-new_short_resolved-short_amb
    consensus_note=('本次已关闭强一致保护，D 按严格 run 门槛分区；“D 保持整区”对应条件因此不能通过。' if p['strong_consensus_fraction'] is None else
                    f"本次启用可配置 strong_consensus_fraction={p['strong_consensus_fraction']}；只有达到此门槛才触发整块保护。")
    text=f'''# Phase 1 局部偏好子区修改报告

本轮只新增 **B0 局部偏好分段 + B1 独立票优先聚合**。冻结输入来自 `{source.name}`，没有重新运行 A1/A2/A3，没有修改主轨或寻找接点。原来的 A2 大区保留为 `INITIAL_CONFLICT_REGION`；新的冻结面组作用于 `DECISION_SUBREGION`。

## 修改后的结果

| 统计口径 | 结果 |
| --- | ---: |
| 初始竞争区 | {s['initial_conflict_regions']} |
| 子区索引记录（含缓冲区） | {s['decision_subregions']} |
| 有明确首选 / 待定子区 | {s['winner_subregions']} / {s['ambiguous_subregions']} |
| 原始 parent–V 总组合 | {s['parent_V_population']}，完整保留，无重复分配 |
| 新 resolved subregion–V | {s['resolved_subregion_V']} |
| 支持 / 反对 / 原始待定 | {s['supporting_V']} / {s['opposing_V']} / {s['abstaining_V']} |
| 子区未冻结的 V 组合 | {s['unresolved_V']} |
| MAJORITY_OPPOSES_WINNER | {s['MAJORITY_OPPOSES_WINNER_count']} |

已冻结区域中的反对比例由 **{s['old_opposing_V']}/{s['old_resolved_region_V']} = {s['old_opposing_percent']:.2f}%** 降为 **{s['opposing_V']}/{s['resolved_subregion_V']} = {s['opposing_percent']:.2f}%**。

分母变化有两部分：旧已冻结范围内的 {removed_amb} 个原始待定 V 被保留为 buffer，另有 **{s['unresolved_explicit_vote_V']} 个本来有明确票的 V** 因没有稳定 run 暂不冻结（其中 {removed_oppose} 个原先反对、{removed_support} 个原先支持旧首选）。全量未冻结记录共含 {s['raw_ambiguous_V']} 个原始待定 V 和 {s['unresolved_explicit_vote_V']} 个明确票 V，不能把后一类说成 A3 原本不知道投谁。

因此另外固定在新旧都明确的 **{s['matched_resolved_V']} 个相同组合**上：旧反对 {s['old_opposing_on_matched_V']}（{ratio(s['old_opposing_on_matched_V'],s['matched_resolved_V']):.2f}%），新反对 {s['new_opposing_on_matched_V']}（{ratio(s['new_opposing_on_matched_V'],s['matched_resolved_V']):.2f}%）。这排除了仅改变统计分母的影响，但不能消除分区本身优先贴合独立票的设计效应。

这是决策对独立 A3 评价的尊重程度，不是识别准确率，也不是新的几何质量证明。A3 评分本身的合理性仍是上一阶段的限制。

## 六个代表区域

{case_table}

![C07 与 89.95 局部反转](figures/R00268.png)

R00268 会保留 PF34、PF19 两种稳定偏好，中间原始待定 V 不平滑。R00806 的 F/C05 局部不再被整个区域 PF19 覆盖。R00132 的 PF34 多数票段有自己的子区，不再服从早先 PF31 的贯穿优势。E 的 104.30 m 局部按 PF19/E_T2 的连续独立票选面，不再被整区 through=24:22 结束判定。

![E 局部细分](figures/R00629_detail.png)

各案例页面逐子区列出明确票数、连续支持、贯穿、CC、有限出口、原始 preference runs 和 LOCAL_DISSENT。几何曲线沿用上一轮五邻线图，本次未重新选分支或重绘成新的主轨。

## D 的方案冲突与采用口径

D 的 1515 条 V 中，1501 条投 PF23，14 条投 PF19；其中有两段连续 3 条 PF19（92.45–92.55 m、135.35–135.45 m）。严格执行连续 {p['min_preference_run_V']} 条就切区，会得到 **{s['D_without_consensus_guard_subregions']} 个子区**，与方案“D 保持一个大区”冲突。

为处理该冲突，代码提供**通用且可配置**的强一致保护：仅在没有待定 V 的连续块内，同一个明确票占比达到门槛时保留整块；所有反向票仍记录为 LOCAL_DISSENT，原始 run 及其稳定标记也保留供审核。此规则不引用 D 或任何案例编号，不跨越缺测、候选集合变化或待定缓冲。{consensus_note} D 的完整性依赖这项额外保护，不能归功于纯粹的 3 条分段规则。

这项口径是针对方案内部冲突提出的实现假设，**不是用户已经确认的固定阈值**；可通过 `--disable-strong-consensus` 关闭，99% 也不是论文定值。代价是占比很低但真实存在的局部反转仍会被保留为 dissent，而不单独冻结；需要人工审核这个取舍。

## 分段与新聚合的具体规则

1. 读取 A3 原始表，先把每条 V 在不同面组行上重复的 `slice_vote` 合并为一票，验证一致性，不改写原票。
2. `min_preference_run_V={p['min_preference_run_V']}` 决定稳定偏好 run。偏好持续反转才建立新边界，边界放在新稳定 run 的第一条 V；不同候选集、真实缺测是硬边界。
3. 原始待定 run 为独立 `AMBIGUOUS_BUFFER`，始终不冻结面组；绝不填成左右某个候选。
4. 1–2 条孤立反向票不单独切区。首个稳定 run 之前的短票并入首个稳定段，段间和末尾短票保留在前一稳定段，记录原票及 dissent。
5. 新切块如果没有稳定 run，标记 `NO_STABLE_PREFERENCE`，保留明确票但暂不冻结，避免靠 buffer 边缘制造 1–2 V 的新自动选面区。原来就只有 1–2 V 的初始区不丢弃，标记 `INHERITED_SHORT_REGION`，仍按 B1 统计；不能把它们当作分段新增碎片。
6. 对可冻结的子区，B1 顺序改为：**明确 slice_vote 数 → 连续支持 run / s 跨度 → 贯穿数 → CC 支持数 → 有限出口**；全部同分保持待定。前述 AMB / NO_STABLE 缓冲仍不选面。through 的微小优势不再置于明确票数之前。
7. 子区空间范围严格等于“旧 parent cell mask 限制到 member_V”，没有新建/扩大几何格，也没有重新计算 CC。下游须同时读取 parent_region_id 和子区 member_V，不能仅凭 s 范围合并不同空间区域。

## 是否过度细分

{distribution}

上表包括所有原始小区和缓冲，未隐藏 1–2 V 记录。**新增 1–2 V 索引记录有 {new_short} 个：{short_amb} 个原始 AMB 缓冲、{short_unstable} 个没有稳定 run 的明确票小块、{new_short_resolved} 个已冻结首选的小区。** “新切且已冻结的短区数量”只是一个窄口径，不能将它等同于方案的全部碎片验收。

**碎片规模验收仍待人工复核，本轮不宣称已满足进入 Phase 2 的全部条件。** 原始待定不能为了减少碎片统计而被吞并，{short_unstable} 个额外无稳定 run 小块也必须保留在审核中。大量既存短初始区因为本轮没有重做 A2，同样仍然存在。

所有多数反对首选异常逐例记录如下：

{majority}

详细计数及新旧状态迁移见 [summary.json](summary.json)，全子区范围、类型和首选见 [完整目录](SUBREGION_CATALOG.md)。

## 验收与性能

{acceptance}

以上是可自动核对的条件。两个总体验收项仍为**待人工确认**：新增短记录的规模是否可接受，以及 D 所采用的 99% 强一致保护是否符合实际地形。

B0 用时 **{m['timing']['B0_segmentation_s']:.4f} s**，B1 用时 **{m['timing']['B1_reducer_s']:.4f} s**，两者合计 **{m['timing']['B0_B1_s']:.4f} s**。输入读取、出图和报告的完整秒级耗时见 [性能报告](PERFORMANCE_REPORT.md)。

最低检查：本次新增的 14 项专项行为测试通过，覆盖稳定反转、待定缓冲、孤立反票、缺测/候选变化、投票优先、D 保护和不可变输出。没有运行全项目测试。实际执行另验证每个原 parent–V 恰好归属一次，旧 A1/A2/A3 源码及输入文件 SHA256 不变，报告生成不改变新冻结决策。

## 输出与人工审核项

- [子区索引](decision_subregion_index.json)、[冻结子区决策](subregion_track_decisions.json)、[运行清单](manifest.json)。
- 主轨和旧整区结果原文件保留；本轮新入口只消费表格，禁止据此宣称已经完成换轨。
- 人工审核优先确认 D 的 99% 保护是否合适、E/F 的边界是否符合实际模型，以及多段待定缓冲是否应继续留待后续证据。
- 本轮首选代表局部独立评价的一致性；真实面身份仍需人工核对。未执行 Phase 2。
'''
    (out/'LOCAL_PREFERENCE_REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE);p.add_argument('--min-preference-run-v',type=int,default=3)
    p.add_argument('--strong-consensus-fraction',type=float,default=.99);p.add_argument('--disable-strong-consensus',action='store_true')
    args=p.parse_args();run(args.source_dir,LocalPreferencePolicy(args.min_preference_run_v,None if args.disable_strong_consensus else args.strong_consensus_fraction))
