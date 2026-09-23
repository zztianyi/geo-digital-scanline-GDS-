"""Read-only A3 audit atlas. Raw UZ, frozen traces, no scoring/mesh/route calls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/06_visualization'))
from locc_local_review_plot import _boundary_style, _boundary_markers, _lines, _save, boundary_segments

COLORS = {16:'#2785a3', 19:'#cf833b', 23:'#498e65', 31:'#ba5267', 34:'#8751a2'}
GREY = '#ccd0d5'
CLASSES = {'CONTINUITY_GOOD':'GOOD', 'CONTINUITY_UNCERTAIN':'UNCERTAIN', 'CONTINUITY_WEAK':'WEAK'}
REASONS = {'KEEP':'保留', 'LOSE_CC':'CC 落后', 'LOSE_CONTINUITY_CLASS':'贯通类别落后',
           'REJECT_CONTINUITY_WEAK':'贯通弱淘汰', 'REJECT_MBG':'MBG 淘汰', 'TIE_AMBIGUOUS':'同档待定'}
REGIONS = ['R00806','R00268','R00629','R00204','R00132']
CASES = [
    ('F_3V','R00806',109.35,[4,19,1378.5,1385.3],'3V 岛：2 条由 CC 决胜，1 条由贯通类别决胜'),
    ('F_5V','R00806',116.25,[.2,3.5,1383,1385.5],'5V 岛右边界：CC 切换后转待定'),
    ('F_8V','R00806',138.25,[-9.7,-7.8,1401.4,1403.4],'原 8V 岛：两者 GOOD、均不在 CC，现为待定'),
    ('F_153V','R00806',129.90,[-8,7,1371,1407],'原 153V 大区：主段 CC 优势保留，少数测线例外'),
    ('C07_PF34','R00268',85.95,[-12,2,1387,1406],'PF34 稳定区：等贯通类别，由 CC 区分'),
    ('C07_8995','R00268',89.95,[-9,14,1368.5,1401],'89.95 m 改判：完整覆盖与约一半覆盖的类别差异'),
    ('C07_8995_detail','R00268',89.95,[3,7,1388,1392],'89.95 m 端部放大：局部看起来相近不等于覆盖完整竞争区'),
    ('C07_PF19','R00268',95.00,[-3,13,1373,1392],'PF19 后段：CC 优势支持保留'),
    ('E_transition','R00629',104.90,[8.7,11.2,1385.2,1386.5],'E：同为 UNCERTAIN 时，PF16 从 CC 获得优势'),
    ('D_9245','R00204',92.50,[-16,-8,1403,1414],'D 第一段 3V：原 PF19 反向票改为待定'),
    ('D_13535','R00204',135.40,[-16,-13,1411.5,1414.2],'D 第二段 3V：原 PF19 反向票改为待定'),
    ('R132_PF31','R00132',51.50,[1.5,9,1369,1376],'局部 PF31 保留：不再用整区多数压制'),
    ('R132_PF34','R00132',71.50,[5.5,12,1369,1376],'局部 PF34 仍可保留：但中间长段现为待定'),
]


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def pf(value): return 'AMB' if value is None else f'PF{value}'
def counts(values): return ' / '.join(f'{pf(k)}={v}' for k,v in sorted(Counter(values).items(),key=lambda x:-1 if x[0] is None else x[0]))
def table(headers, rows): return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def observed_line(ax, points, box, color, width, eliminated, zorder):
    # One path per connected clipped run: dashes must not restart on every tiny
    # original edge. NaN separates clipped runs; no joining across missing data.
    segments=boundary_segments(points,box)
    if not len(segments): return
    vertices=[segments[0,0],segments[0,1]]
    for a,b in segments[1:]:
        if not np.allclose(vertices[-1],a,rtol=0,atol=1e-9): vertices.extend([[np.nan,np.nan],a])
        vertices.append(b)
    vertices=np.asarray(vertices)
    ax.plot(vertices[:,0],vertices[:,1],color=color,lw=width,
            linestyle='--' if eliminated else '-',zorder=zorder)


def assignments(folder):
    subs = read_json(folder/'decision_subregion_index.json')['subregions']
    decisions = {d['subregion_id']:d for d in read_json(folder/'subregion_track_decisions.json')['decisions']}
    result = {(s['parent_region_id'],si):decisions[s['subregion_id']]['dominant_FaceTrack'] for s in subs for si in s['s_indices']}
    return subs, decisions, result


class Atlas:
    def __init__(self, folder):
        self.folder = folder
        self.manifest = read_json(folder/'manifest.json')
        self.summary = read_json(folder/'summary.json')
        self.source = Path(self.manifest['source'])
        self.old_local = Path(self.manifest['old_local_preference'])
        self.cache = Path(self.manifest['index_cache'])
        self.rows = pq.read_table(folder/'per_slice_track_scores.parquet').to_pylist()
        self.votes = pq.read_table(folder/'old_new_slice_votes.parquet').to_pylist()
        self.by = defaultdict(list)
        for r in self.rows: self.by[r['region_id'], r['s_index']].append(r)
        self.old_by = defaultdict(list)
        for r in pq.read_table(self.source/'per_slice_track_scores.parquet').to_pylist(): self.old_by[r['region_id'],r['s_index']].append(r)
        self.vmap = {(r['region_id'],r['s_index']):r for r in self.votes}
        self.old_subs,self.old_decisions,self.old_b1 = assignments(self.old_local)
        self.new_subs,self.new_decisions,self.new_b1 = assignments(folder)
        self.figures = folder/'figures'; self.figures.mkdir(exist_ok=True)
        self.tables = folder/'tables'; self.tables.mkdir(exist_ok=True)
        self.style,self.font = _boundary_style()
        self.geometry_cache = {}
        self.figure_records = []

    def rr(self, rid): return sorted((r for r in self.votes if r['region_id']==rid),key=lambda r:r['s_index'])

    def geometry(self, s):
        if s not in self.geometry_cache:
            with (self.cache/'slices'/f'{s:.2f}.pkl').open('rb') as f: self.geometry_cache[s]=pickle.load(f)
        return self.geometry_cache[s]

    def finish(self, fig, name, **metadata):
        path = self.figures/(name+'.png')
        try: _save(fig,path,180)
        finally: plt.close(fig)
        self.figure_records.append(dict(file=str(path),**metadata))

    def profile_figure(self, case):
        name,rid,center,box,title = case
        available = self.rr(rid)
        ci = min(range(len(available)),key=lambda i:abs(available[i]['s']-center))
        indices = available[max(0,ci-2):ci+3]
        tracks = sorted({r['FaceTrack'] for v in indices for r in self.by[rid,v['s_index']]})
        with plt.rc_context(self.style):
            fig,axes=plt.subplots(1+len(tracks),len(indices),figsize=(17,3.35*(1+len(tracks))),squeeze=False)
            fig.subplots_adjust(left=.07,right=.985,bottom=.13,top=.91,wspace=.12,hspace=.44)
            for col,v in enumerate(indices):
                rows = {r['FaceTrack']:r for r in self.by[rid,v['s_index']]}
                geo = self.geometry(v['s'])
                for row,track in enumerate([None,*tracks]):
                    ax=axes[row,col]
                    for branch in geo['branches'].values():
                        _lines(ax,boundary_segments(branch['points_uz'],box),GREY,'observed',width=.6,zorder=1)
                    for t in tracks if track is None else [track]:
                        r=rows[t]; bid=r['witness_branch']
                        if bid is None: continue
                        branch=geo['branches'][bid]
                        observed_line(ax,branch['points_uz'],box,COLORS[t],
                                      2.5 if r['slice_vote']==t else 1.1 if r['eliminated'] else 1.7,
                                      r['eliminated'],3+(t==r['slice_vote']))
                        _boundary_markers(ax,branch,box,COLORS[t])
                    ax.set(xlim=box[:2],ylim=box[2:])
                    ax.set_aspect('equal',adjustable='box')
                    ax.ticklabel_format(useOffset=False,style='plain')
                    ax.tick_params(labelsize=8)
                    ax.grid(True,alpha=.20,lw=.5)
                    if col==0: ax.set_ylabel('原始 z (m)',fontproperties=self.font,fontsize=10)
                    else: ax.tick_params(labelleft=False)
                    if row==len(tracks): ax.set_xlabel('原始 u (m)',fontproperties=self.font,fontsize=10)
                    if track is None:
                        heading=f"s={v['s']:.2f} m | 同一 V 叠加\n旧 {pf(v['old_slice_vote'])} → 新 {pf(v['new_slice_vote'])}"
                        ax.set_title(heading,fontproperties=self.font,fontsize=10,pad=8)
                    else:
                        r=rows[track]; bid=r['witness_branch']
                        cls=CLASSES[r['continuity_class']] if r['MBG'] else 'MBG FAIL'
                        heading=f"PF{track} / {'无合格 witness' if bid is None else 'B'+str(bid)} | {cls}\nCC={'Y' if r['in_CC'] else 'N'} | {REASONS[r['elimination_reason']]}"
                        ax.set_title(heading,color=COLORS[track],fontproperties=self.font,fontsize=9,pad=6)
            fig.suptitle(f'{rid}  |  {title}',fontproperties=self.font,fontsize=16,y=.98)
            fig.text(.5,.035,'原始坐标、无平移；各格同尺度。颜色=FaceTrack；粗实线=保留，细虚线=淘汰，普通实线=待定。\n灰线=其他原始分支；圆点=真实端点，空心方框=CC 边界。裁切边缘不是端点；判据使用完整竞争区。',
                     ha='center',fontproperties=self.font,fontsize=10)
            self.finish(fig,name,region_id=rid,profiles=[v['s'] for v in indices],box_uz=box,coordinate_transform='none',kind='raw_witness_overlay')
        return indices

    def band(self, ax, positions, values, palette, labels, title):
        codes={v:i for i,v in enumerate(labels)}
        z=np.asarray([[codes[v] for v in values]])
        # The native 0.05 m slice spacing is retained; no averaging or vote smoothing.
        edges=np.r_[np.asarray(positions)-.025,positions[-1]+.025]
        ax.pcolormesh(edges,[0,1],z,cmap=ListedColormap([palette[x] for x in labels]),vmin=-.5,vmax=len(labels)-.5,shading='flat',rasterized=True)
        ax.set_yticks([]);ax.set_ylabel(title,rotation=0,ha='right',va='center',fontproperties=self.font,fontsize=9)
        ax.set_xlim(edges[0],edges[-1]); ax.tick_params(axis='x',labelsize=8)

    def strip(self,rid):
        vv=self.rr(rid);positions=[v['s'] for v in vv];tracks=sorted({r['FaceTrack'] for v in vv for r in self.by[rid,v['s_index']]})
        vote_palette={None:'#dadde1',**{t:COLORS[t] for t in tracks}}
        rows=[('旧 A3',[v['old_slice_vote'] for v in vv],vote_palette,list(vote_palette)),
              ('新 A3',[v['new_slice_vote'] for v in vv],vote_palette,list(vote_palette)),
              ('新 B1',[self.new_b1[rid,v['s_index']] for v in vv],vote_palette,list(vote_palette))]
        for t in tracks:
            candidates=[next(r for r in self.by[rid,v['s_index']] if r['FaceTrack']==t) for v in vv]
            class_palette={'FAIL':'#aaaaaa','GOOD':'#4c9b82','UNCERTAIN':'#e7bf5c','WEAK':'#cb6771'}
            rows.append((f'PF{t} 贯通',[CLASSES[r['continuity_class']] if r['MBG'] else 'FAIL' for r in candidates],class_palette,list(class_palette)))
            rows.append((f'PF{t} CC',['Y' if r['in_CC'] else 'N' for r in candidates],{'Y':'#5577aa','N':'#eceef1'},['N','Y']))
        reason_palette={'CC_WIN':'#8751a2','CONTINUITY_WIN':'#4c9b82','MBG_ONLY_SURVIVOR':'#487baa','TIE_AMBIGUOUS':'#dadde1','NO_MBG_ELIGIBLE':'#8d9299','NO_CONTINUITY_ELIGIBLE':'#cb6771'}
        rows.append(('新 A3 原因',[v['new_decision_reason'] for v in vv],reason_palette,list(reason_palette)))
        with plt.rc_context(self.style):
            fig,axes=plt.subplots(len(rows),1,figsize=(21,1.3+.53*len(rows)),sharex=True)
            fig.subplots_adjust(left=.085,right=.99,bottom=.19,top=.88,hspace=.20)
            for ax,(title,values,palette,labels) in zip(axes,rows): self.band(ax,positions,values,palette,labels,title)
            axes[-1].set_xlabel('测线路径位置 s (m)；每条 V 独立显示，无邻线平均',fontproperties=self.font,fontsize=11)
            fig.suptitle(f'{rid} | 完整决策证据条带 | {len(vv)} 条区域内测线',fontproperties=self.font,fontsize=16,y=.98)
            tokens='  '.join(f'PF{t}' for t in tracks)
            for j,t in enumerate(tracks): fig.text(.11+j*.075,.93,f'PF{t}',color=COLORS[t],fontproperties=self.font,fontsize=11,weight='bold')
            fig.text(.5,.035,'投票灰色=AMB；贯通：绿=GOOD、黄=UNCERTAIN、红=WEAK、灰=MBG FAIL；CC：蓝=Y、浅灰=N。\n原因：紫=CC 决胜、绿=贯通类别决胜、蓝=MBG 后仅一候选、浅灰=平票、深灰=无 MBG 合格候选。B1 为独立聚合结果。',ha='center',fontproperties=self.font,fontsize=10)
            self.finish(fig,rid+'_evidence',region_id=rid,kind='all_slice_evidence_strip',slice_count=len(vv))

    def islands(self):
        rid='R00806';windows=[(109.35,109.45,'3V：PF16 全保留；2 CC + 1 贯通类别'),(116.05,116.25,'5V：PF16 全保留；CC 决胜'),
                                (138.05,138.40,'8V：全部 AMB；同为 GOOD / CC=N'),(123.45,131.05,'153V：149 PF16 / 2 PF19 / 2 AMB')]
        with plt.rc_context(self.style):
            fig=plt.figure(figsize=(19,9));outer=fig.add_gridspec(2,2,left=.09,right=.985,bottom=.10,top=.90,hspace=.6,wspace=.23)
            for cell,(lo,hi,title) in zip(outer,windows):
                vv=[v for v in self.rr(rid) if lo-.0001<=v['s']<=hi+.0001]
                grid=cell.subgridspec(4,1,hspace=.2);axes=[fig.add_subplot(grid[i,0]) for i in range(4)]
                data=[('旧 B1',[self.old_b1[rid,v['s_index']] for v in vv]),('旧 A3',[v['old_slice_vote'] for v in vv]),('新 A3',[v['new_slice_vote'] for v in vv]),('新 B1',[self.new_b1[rid,v['s_index']] for v in vv])]
                for ax,(label,values) in zip(axes,data):
                    self.band(ax,[v['s'] for v in vv],values,{None:'#dadde1',16:COLORS[16],19:COLORS[19]},[None,16,19],label)
                    if ax is not axes[-1]: ax.tick_params(labelbottom=False)
                axes[0].set_title(f'{lo:.2f}–{hi:.2f} m | {title}',fontproperties=self.font,fontsize=11,pad=12)
                axes[-1].set_xlabel('s (m)',fontsize=10)
            fig.suptitle('R00806 | 旧岛范围内的新旧逐 V 结果；四个窗口保持各自实际 s 尺度',fontproperties=self.font,fontsize=15,y=.98)
            fig.text(.5,.025,'蓝绿=PF16；橙=PF19；灰=AMB。旧 8V 是 B1 的 PF16 区，旧 A3 实为 6 票 PF16 + 2 票 PF19。',ha='center',fontproperties=self.font,fontsize=11)
            self.finish(fig,'R00806_island_comparison',region_id=rid,kind='island_comparison')

    def candidate_table(self, rid, positions):
        rr=sorted((r for r in self.rows if r['region_id']==rid and any(abs(r['s']-s)<1e-6 for s in positions)),key=lambda r:(r['s'],r['FaceTrack']))
        return table(['s (m)','FaceTrack / witness','MBG','贯通','CC','淘汰层 / 原因','最终 A3'],
                     [[f"{r['s']:.2f}",f"PF{r['FaceTrack']} / B{r['witness_branch']}" if r['witness_branch'] is not None else f"PF{r['FaceTrack']} / 无合格 witness",'PASS' if r['MBG'] else 'FAIL',CLASSES[r['continuity_class']] if r['MBG'] else '未评',
                       'Y' if r['in_CC'] else 'N',f"{r['eliminated_by'] or '—'} / {r['elimination_reason']}",pf(r['slice_vote'])] for r in rr])

    def detailed_table(self,rid,center):
        rr=min((rr for (r,si),rr in self.by.items() if r==rid),key=lambda rr:abs(rr[0]['s']-center))
        return table(['PF / witness','覆盖格 / 总格','覆盖弧长 m','interval 数 / 最大间隙 m','interval 弧长占比','出口 m','CC 占比','定性原因'],
                     [[f"PF{r['FaceTrack']} / B{r['witness_branch']}",f"{r['covered_cell_count']}/{r['region_cell_count']}",f"{r['competition_arc_m']:.3f}",f"{r['continuous_interval_count']} / {r['max_interval_gap_m']:.3f}",f"{r['continuous_coverage_fraction']:.3f}",f"{r['exit_continuation_m']:.3f}",f"{r['CC_fraction']:.6f}",r['continuity_reason']] for r in rr])

    def old_table(self,rid,center):
        rr=min((rr for (r,si),rr in self.old_by.items() if r==rid),key=lambda rr:abs(rr[0]['s']-center))
        return table(['旧 PF / witness','MBG','through','CC','出口 m','旧 A3'],
                     [[f"PF{r['FaceTrack']} / B{r['witness_branch']}",r['MBG'],r['through_region'],r['in_CC'],f"{r['exit_continuation_m']:.6f}",pf(r['slice_vote'])] for r in rr])

    def sub_table(self,rid,old=False):
        subs,decisions=(self.old_subs,self.old_decisions) if old else (self.new_subs,self.new_decisions)
        return table(['范围 s (m)','V 数','B0 偏好','B1','B0 类别'],[[f"{s['s_start']:.2f}–{s['s_end']:.2f}",len(s['s_indices']),pf(s['anchor_preference']),pf(decisions[s['subregion_id']]['dominant_FaceTrack']),s['kind']] for s in subs if s['parent_region_id']==rid])

    def export_region(self,rid,notes):
        candidates=[r for r in self.rows if r['region_id']==rid]
        fields=[k for k in candidates[0] if k not in ('witnesses','competition_intervals')]
        csv_path=self.tables/(rid+'_decision_trace.csv')
        with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(candidates)
        self.strip(rid)
        vv=self.rr(rid)
        text=[f'# {rid} 定性贯通复核',notes,
              '\n## 整区证据与汇总\n',f"新 A3：{counts(v['new_slice_vote'] for v in vv)}。新 B1：{counts(self.new_b1[rid,v['s_index']] for v in vv)}。",
              f'![完整证据条带](figures/{rid}_evidence.png)',f'[全部 {len(candidates)} 条候选淘汰记录](tables/{csv_path.name})；[读图说明与口径](A3_QUALITATIVE_AUDIT_REPORT.md#读图说明)。']
        for case in CASES:
            name,r,center,box,title=case
            if r!=rid:continue
            positions=self.profile_figure(case)
            text += [f'\n## {title}（中心 {center:.2f} m）\n',f'![{title}](figures/{name}.png)',
                     self.candidate_table(rid,[v['s'] for v in positions]),f'\n中心 V {center:.2f} m 的原始量（仅用于解释类别，不能在同档中比较大小）：\n',self.detailed_table(rid,center),
                     '\n同一中心 V 的旧 A3 证据（旧顺序为 through → CC → 精确出口，新版已停用）：\n',self.old_table(rid,center)]
        text += ['\n## 旧 B0/B1 分区\n',self.sub_table(rid,True),'\n## 新 B0/B1 分区（包含待定缓冲）\n',self.sub_table(rid)]
        (self.folder/(rid+'_REVIEW.md')).write_text('\n\n'.join(text),encoding='utf-8')


NOTES={
'R00806':'''3V（109.35–109.45）和 5V（116.05–116.25）**没有消失**。3V 中 109.35、109.45 两条都是 GOOD 对 GOOD，PF16 的 in_CC=Y、PF19=N，由 CC 决胜；109.40 则是 PF19 因 SEPARATED_MASK_INTERVALS 降为 UNCERTAIN，PF16 在贯通类别阶段即胜出。5V 的五条均为 GOOD 对 GOOD，由 PF16 的 CC 优势决胜。它们不是本轮残留的厘米延展排序；是否值得保留，下一步需要审核冻结 CC 的定义与边界位置、109.40 的 mask interval，本轮不能为了消岛改变 CC。

旧 3V、5V 均由 CC 选 PF16：through 都为 False，PF16 的 CC=True，而 PF19=False；PF16 的出口甚至比 PF19 短。旧 8V 则是 6 条由 through=True 选 PF16，另外 2 条 through/CC 都平时按精确出口选 PF19。因此不能把所有小岛统称为微小长度造成。

8V（138.05–138.40）全转 AMB：两组都是 GOOD 且 CC=N。旧 B1 把整段标为 PF16，但旧 A3 是 6 票 PF16、2 票 PF19，必须区分两个阶段。原 153V 大区（123.45–131.05）新 A3 为 149 PF16、2 PF19、2 AMB；大段优势存在，并非 153 条全部同票。新 B0 从 10 段增至 64 段，反映待定缓冲增多，不代表准确率提升。''',
'R00268':'''PF34 与 PF19 两个大尺度区域仍存在，但**不能报告为原边界完全不变**。按 B0/B1 分区口径：71.70–86.05 的 288V PF34 保留；92.90–104.25 的 228V PF19 保留；旧 86.10–89.65 的 72V PF19 变为待定，89.70–91.90 的 45V 则为 PF34，91.95–92.85 的 19V 为待定。228V 的 B1 PF19 段内，新 A3 实为 227 票 PF19、1 票 PF34；99.05 m 是贯通类别决胜产生的异议票，不能把 B1 区域归属说成逐 V 全同票。

89.95 m 是本轮最需人工审核的改判：PF19/B2 仅覆盖 42/83 格（50.6%），PF34/B6 覆盖 83/83 格。前者 UNCERTAIN 的唯一原因是覆盖不足；不是 interval 断裂。后者虽然出口为 0，但在竞争区内有 35.725 m 实测弧长，按“足够长的观测可以在自然端部结束”的初始规则判 GOOD，故胜出。两者 CC 均为 N，CC 没有参与这次决胜。这是大尺度覆盖判据造成的变化，绝非 1–2 cm 出口优势；但完整覆盖是否对应目标曲面，仍须人工确认。局部放大图与全范围图必须结合看。''',
'R00629':'''原 PF19→PF16 转换的后半段仍有 CC 依据，但前半段并非整段照旧。104.30 m 是 PF19=GOOD 对另外两组 UNCERTAIN；104.85 m 三组同为 UNCERTAIN、均 CC=N，因此 AMB；104.90 m 三组仍为 UNCERTAIN，只有 PF16 在 CC 内，因此 PF16 胜出。

104.90 m 的 PF16 覆盖全部 28 格，但同一分支进出 mask 形成 7 个 interval，覆盖弧长/首末 span 仅 0.847，最大离开 mask 弧段 0.989 m，因此是 UNCERTAIN。**mask 的间隙不等于原模型孔洞，也不等于分支物理中断。** 本轮没有补线。这是当前定性规则的解释局限，需要结合原始曲线审核。''',
'R00204':'''92.45–92.55、135.35–135.45 两段各 3V 的旧 PF19 反向票均不再保留，全部改为 AMB。这里 PF19/PF23 都是 GOOD、CC 都为 N；旧六条均是 PF19 through=True、PF23=False 直接选中 PF19，并非 PF19 的出口更长（如 92.50 m 为 0.055 m 对 1.000 m）。新版不允许 through 直接决胜。不能把“反向票消失”说成“证明 PF23 正确”，也不能说 B0/B1 已把它们并回 PF23：它们仍是未决缓冲。

旧 B0/B1 曾把 1515V 整区归 PF23，新 B0 分为 34 段。下表保留所有待定段，供判断是否需要后续局部 CC 或空间证据。''',
'R00132':'''旧大区多数压制局部的行为未回归：51.40–53.90 的 51V PF31 保留；71.20–72.30 的 23V PF34 保留；中间 53.95–71.15 的 345V 在 B1 层面全部未决，不能说原 368V PF34 大段全数保持。该中间段的新 A3 实为 338 AMB、2 PF31、5 PF34；7 条零散已决票尚不足以使对应 B1 子区稳定。新 B0 分为 15 段，B1 没有将这些未决测线改成整区 PF31。

51.50 m 两者 GOOD，PF31 CC=Y、PF34=N；71.50 m 两者 UNCERTAIN，PF34 CC=Y、PF31=N。这两处由局部 CC 决定，不依靠整区多数或细微长度。'''
}


def report(atlas, render_s):
    s=atlas.summary;m=atlas.manifest;t=m['timing'];n=s['slice_count'];cn=s['candidate_count']
    quant=read_json(atlas.folder/'old_A3_distributions.json')['MBG_pass']['quantiles']
    reason_names={'REJECT_MBG':'MBG 淘汰','REJECT_CONTINUITY_WEAK':'WEAK 淘汰','LOSE_CONTINUITY_CLASS':'贯通类别落后','LOSE_CC':'CC 落后','TIE_AMBIGUOUS':'仍在平票候选集','KEEP':'单测线保留'}
    lines=['# A3 定性贯通筛选与 FaceTrack 复核报告',
'''本轮完成 A3 改造：**MBG → 定性贯通 → CC → 待定**。同档同 CC 不再比较精确延展、弧长或覆盖率。独立微差决胜审计为 **0**，36 项指定测试通过。当前结果是 FaceTrack 候选评价及 B0/B1 聚合，**不是拼接后的生产主轨，不是识别准确率**。

最重要的人工复核项是 R00268 的 89.95 m 改判、R00806 仍保留的 3V/5V 小岛，以及大量新出现的待定区。两组阈值扰动的全局变化小于 0.6%，但不能据此认定所有局部几何正确。代码行为与微差审计通过；“原大尺度分区完全保持”不成立，几何正确性仍待人工验收。''',
'## 读图说明\n',
'''1. **候选图**：一列是一条测线，首行叠加该 V 的所有合格 FaceTrack witness，后续各行拆开看同一组。所有格使用相同原始 u–z 范围和等比例坐标；没有平移、归一化，也没有将邻线曲线拼成一条。图中只有当前审计选出的 witness 着色，其余原始分支仍以灰色呈现；全部备选 witness 保存在原始评分表中。B 编号只在该 V 内有效。
2. **颜色和线型**：PF16 蓝绿、PF19 橙、PF23 绿、PF31 红、PF34 紫；粗实线表示新 A3 保留，细虚线表示淘汰，普通实线表示候选尚未决出。灰线是其他原始观测。圆点是实际分支端点，空心方框是 CC 边界；裁切到图框不代表断轨。局部图只为读几何，分类使用整个冻结竞争区。
3. **证据条带**：每一列仍是一条原始 V；前三行分别是旧 A3、新 A3、新 B1。之后逐 PF 列出贯通类别、CC 状态，再列最终 A3 原因。因此灰色 AMB 是明确的“证据不足”，不是没有绘制候选。短岛在全区条带中可能很窄，要看小岛专图和五邻线图。
4. **淘汰表**：先看 MBG，再看 GOOD/UNCERTAIN/WEAK，再看 CC；同档同 CC 必须 AMB。数值表用于解释类别，不能据 0.01 m 差异自行推导 winner。CC=Y 沿用旧定义：完整竞争弧都在 CC 内；CC_fraction 接近 1 也可能是 N，本轮没有改变它。''',
'## 1. 运行范围与可复现输入\n',
f"复用 {m['profile_tasks']} 条缓存测线、{m['initial_regions']} 个冻结竞争区，生成 {cn:,} 条候选记录、{n:,} 个 (region,V) 比较。A1/A2 重建、mesh 读取、路线操作、synthetic geometry 均为 0。{m['workers']} 个 ProcessPool worker；全部冻结输入及 B0/B1 源码执行前后 SHA-256 一致。",
f"旧 A3：`{atlas.source}`；旧 B0/B1：`{atlas.old_local}`。本轮目录：`{atlas.folder}`。参见 [执行清单](manifest.json)、[完整 trace](decision_trace.parquet)、[含全部 witness 的评分](per_slice_track_scores.parquet)、[新旧投票](old_new_slice_votes.parquet)。",
'## 2. 分布、绝对尺度与初始门控\n',
'先读取旧 A3 25,054 条 MBG 合格候选的分布，再声明默认阈值；没有按希望得到的岛形状调参。',
table(['指标','Q01','Q05','Q25','Q50','Q95'],[[field]+[f"{qs[q]:.6f}" for q in ['0.01','0.05','0.25','0.5','0.95']] for field,qs in quant.items()]),
'''完整分位数：[旧 A3 分布](old_A3_distributions.json)。coverage 的 Q10 已为 1；exit 被旧测量上限截至 1 m，许多候选为 0 或 1。原程序在 witness 内和 FaceTrack 之间都有精确排序；本轮两层均已移除，避免先在 witness 中暗中选“多一点”的分支。

默认规则（工程初值，不是最终论文阈值）：

- GOOD：覆盖率 ≥0.90 **或缺格数不超过 1**；同时满足连续性充分；同时满足出口 ≥0.25 m **或竞争区实测弧长 ≥1 m**。这个一格容忍按缺格数量实现，不另推断该格是否在 mask 外围，仍须通过连续性门控。
- 连续性充分：仅一个 interval，**或**最大离开 mask 弧段 ≤一个 0.5 m 格子的对角线（约 0.707 m），**或**覆盖弧长/首末 span ≥0.90。interval 是同一观测分支与 mask 的相交弧段，不跨分支拼接。
- WEAK：覆盖率 ≤0.50、缺格数 >1，且出口 ≤0.05 m。所有 WEAK 淘汰；若全 WEAK，不强选 winner。其余为 UNCERTAIN。
- GOOD 可淘汰 UNCERTAIN；同类后仅比较 CC 布尔状态；同 CC 则 AMB。只有一个合格候选也仍执行 WEAK 门控。
- 同一 PF 多 witness 时按“贯通类别、CC”选择代表；二者相同则固定 branch ID，仅保证复现，不比较精确长度。所有 witness 保留在表中。不同 PF 的平票绝不用 ID 决胜。
- `through_region` 保留审计，不能直接开关 winner。本轮不引入邻线 majority；B0/B1 的原规则保留，包括其既有聚合字段，本轮微差审计结论限定于 A3。''',
table(['阈值','放宽','默认','收紧'],[['GOOD 最小覆盖',.875,.90,.925],['GOOD 最小出口 m',.20,.25,.30],['WEAK 最大覆盖',.45,.50,.55],['WEAK 最大出口 m',.025,.05,.075]]),
'## 3. 全局淘汰与决胜统计\n',
f'以下分母是 **{cn:,} 条候选**，一个 (region,V) 会有多个候选，不能当作测线准确率。',
table(['候选结果','数量','候选占比'],[[reason_names[k],s['elimination_reasons'].get(k,0),f"{s['elimination_percent'].get(k,0):.3f}%"] for k in reason_names]),
f'以下分母是 **{n:,} 个 (region,V)**，与上表分母不同。',
table(['A3 最终原因','数量','比较占比'],[[k,v,f'{100*v/n:.3f}%'] for k,v in s['slice_decision_reasons'].items()]),
f"总待定 **{s['ambiguous_slices']:,}/{n:,}={100*s['ambiguous_slices']/n:.3f}%**，包括平票 6,431 和无 MBG 候选 136；原来待定 {s['old_ambiguous_slices']:,}。新保留 5,311，并未追求更多 winner。新 B0 共 {s['B0_subregions']} 段，B1 已决 {s['B1_resolved_subregions']} 段、未决 {s['B1_ambiguous_subregions']} 段；B1 保留覆盖 {s['B1_resolved_region_V']} 个 (region,V)。多数反对 winner 计数为 {s['B1_MAJORITY_OPPOSES_WINNER_count']}。",
'## 4. 微差审计与新旧变化\n',
'''**MICRO_DIFFERENCE_DECISION_COUNT=0**（默认、放宽、收紧三组均为 0）。审计独立检查每个保留者：是否仍有 MBG 合格、同贯通类别、同 CC 的竞争者；若有则记违规。测试另行注入一个违规 winner，确认审计真的能报警，并非固定输出 0。[审计文件](micro_difference_audit.json)。''',
f"新旧 A3 共 **{s['changed_votes']:,}** 个投票变化。下表是互斥的主原因；另存的 change_reasons 是可重叠标签，不能相加。",
table(['主变化原因','个数'],list(s['change_reason_counts'].items())),
'''其中 **23** 票取消了旧 ≤0.02 m 的精确延展优势；另有 **4,230** 票取消了更大幅度的精确延展排序，不能把这 4,230 都称为“厘米级差异”。**196** 票取消了旧 through 布尔开关的胜出（按移除 through 的反事实排序识别）。变为 AMB 的变化共 **4,380**，旧 AMB 转为已决 **43**，净增待定 **4,337**。`UNCHANGED_STRONG_CASE` 只是旧新 winner 一致的工程标签，不等于人工验证正确。MBG 输入冻结，NEW_MBG_REJECT 变化标签为 0；NEW_CONTINUITY_WEAK_REJECT 变化标签为 0，不代表没有候选被 WEAK 淘汰（实际 19 条）。''',
'## 5. 五个重点区域\n',
table(['区域','结论','完整五邻线图、原始量与淘汰表'],[
['R00806 / F','3V 以 CC 为主并含 1 条类别决胜；5V 因 CC 保留；8V 全待定','[查看](R00806_REVIEW.md)'],
['R00268 / C07','两类大段仍在；89.95 m 改判 PF34，须重点审核','[查看](R00268_REVIEW.md)'],
['R00629 / E','104.85 m 待定，104.90 m PF16 由 CC 决胜','[查看](R00629_REVIEW.md)'],
['R00204 / D','两段 3V 反向票全待定，并未强并到 PF23','[查看](R00204_REVIEW.md)'],
['R00132','局部 PF31/PF34 保留，中间 345V 的 B1 未决（A3 含 7 条已决票）','[查看](R00132_REVIEW.md)']]),
'![F 小岛变化](figures/R00806_island_comparison.png)']
    for rid in REGIONS: lines += [f'### {rid}\n',NOTES[rid]]
    lines += ['## 6. 阈值敏感性\n',
    '预先声明的工程筛查线：全部 (region,V) 变化 ≤5%，两个明确 winner 直接互换 ≤1%；不是准确率阈值。只有四个表列阈值做扰动，其余 interval/格距/弧长尺度保持不变。',
    table(['变体','变化 / 全部','明确 winner 互换 / 全部','原默认已决中变化','B1 归属变化','工程筛查'],[[v['label'],f"{v['changed_votes']}/{n} ({100*v['changed_fraction']:.3f}%)",f"{v['direct_winner_identity_flips']}/{n} ({100*v['direct_flip_fraction']:.3f}%)",f"{v['default_resolved_changed']}/{v['default_resolved']} ({100*v['default_resolved_changed_fraction']:.3f}%)",v['B1_assignment_changes'],'通过' if v['acceptance'] else '不通过'] for v in s['sensitivity']]),
    table(['区域','放宽改变 V 数','收紧改变 V 数'],[[rid]+[v['case_changes'][rid] for v in s['sensitivity']] for rid in REGIONS]),
    '全局没有出现大面积翻票，但局部变化仍存在。完整变体：[放宽](sensitivity/loose/summary.json)、[收紧](sensitivity/strict/summary.json)。不能由全局低比例跳过关键边界审核。',
    '## 7. 性能与最低检查\n',
    table(['阶段','实测秒'],[['A3 并行与完整 trace 导出',f"{t['A3_default_parallel_and_trace_export_s']:.3f}"],['B0 分段',f"{t['B0_s']:.3f}"],['B1 聚合',f"{t['B1_s']:.3f}"],['默认 A3+B0+B1 合计',f"{t['A3_B0_B1_default_s']:.3f}"],['额外两组并行敏感性计算',f"{t['sensitivity_parallel_s']:.3f}"],['前置源文件指纹与旧分布',f"{t['input_hashes_and_distribution_s']:.3f}"],['全入口含导入、校验、导出，不含图册',f"{t['entry_including_imports_s']:.3f}"],['本次图册与分报告生成',f'{render_s:.3f}']]),
    f"父子进程合计 RSS 抽样峰值约 {m['sampled_peak_parent_children_RSS_bytes']/1024**2:.1f} MiB。各 worker 的取证墙钟累计 {m['worker_evidence_s']:.3f} 任务秒、定性决策墙钟累计 {m['worker_decision_s']:.3f} 任务秒（不是 CPU 时间，不可直接与并行总墙钟相加）。当前耗时主要在输入指纹/IO、进程调度与图册；A3 已是秒级。未增加跨 V 几何匹配，也没有必要做与本轮无关的性能重构。RSS 为抽样值，不是严格瞬时峰值。",
    '''最低检查：指定两份 unittest 文件共 **36 项通过**（Phase1 22，LocalPreference 14）。Python 环境未安装 pytest，pytest 入口在收集前失败，随后使用测试文件原有的标准库 unittest 运行，没有安装依赖。新增 12 项覆盖同档禁精排、GOOD/WEAK、CC、through、一格容忍、多 witness、trace、微差违规探测以及双进程一致性。[验证记录](VERIFICATION.md)。未跑其他测试或全量生产识别。''',
    '## 8. 验收问题逐项回答\n',
    table(['问题','回答'],[
    ['1. MBG 淘汰占比','819/25,873 候选，3.165%'],['2. WEAK 淘汰占比','19/25,873 候选，0.073%'],
    ['3. CC 最终决胜','4,137/11,878 比较，34.829%；CC 落后候选为 5,617 条'],['4. AMB','6,567/11,878，55.287%，含无合格候选'],
    ['5. 微差审计','默认与两变体均为 0'],['6. 旧微小贯通优势取消','严格 ≤2 cm 口径 23 票；较大精确出口排序另有 4,230 票'],
    ['7. F 的小岛','3V 保留（2 CC + 1 贯通类别）；5V 保留（5 CC）；8V 全 AMB'],['8. R00268 大段','PF34/PF19 大段仍在，边界与中间改变，不能称完全保持'],
    ['9. D 两段 3V','全部转 AMB；同类同 CC 无证据继续区分'],['10. 结构性淘汰例','89.95 m PF19 因覆盖约一半；109.35 m PF19 因 CC；全局 MBG/WEAK 见 trace'],
    ['11. 证据不足例','F 8V、D 两个 3V、E 104.85 m、R00132 中间段；候选保留为待定'],['12. 默认 A3+B0+B1',f"{t['A3_B0_B1_default_s']:.3f} 秒，不含指纹校验/敏感性/图册"]]),
    '## 9. 当前限制与人工复核\n',
    '''- 本轮证明的是 A3 决策可解释、同档不精排；没有人工真值，不能输出“整体识别正确率”。
- 冻结的 CC 布尔边界仍可能形成 3V/5V 小岛；一条弧段微量越出 CC 也可能将 in_CC 变为 N。这是保留策略的风险，不是 A3 精确延展 tie-break，本轮不越界修改。
- GOOD 使用整个竞争 mask 的足够性。89.95 m 的覆盖判据可能偏向跨更大范围的曲面；请结合上下延伸确认 PF34 是否是目标面，不能只看短局部的重叠。
- mask re-entry 与物理断裂不能混同；E 的 interval 分类需要读原始分支。GOOD 的 1 m 弧长替代出口是显式初值，尚未证明适用于所有自然端部。
- B0/B1 未改算法，新的 AMB 会产生更多未决缓冲和子区。旧全区或后续 B1 的精确字段不在本轮 A3 微差审计的保证范围。
- 原始观测线全部保留，本轮未生成估计连接、未执行联合换轨、未写入生产识别路径。下一步由人工先确认上述局部判据，再决定是否需要改 CC 或 mask，而不是用强制 winner 填满灰区。''']
    (atlas.folder/'A3_QUALITATIVE_AUDIT_REPORT.md').write_text('\n\n'.join(lines),encoding='utf-8')


def main(folder):
    start=time.perf_counter();atlas=Atlas(folder)
    frozen={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'per_slice_track_scores.parquet',folder/'decision_trace.parquet',folder/'summary.json',folder/'manifest.json']}
    for rid in REGIONS: atlas.export_region(rid,NOTES[rid]);print('REPORT',rid,flush=True)
    atlas.islands()
    report(atlas,time.perf_counter()-start)
    verification='''# 最低验证记录

- 环境：Python 3.12；pytest 未安装，pytest 命令在收集前退出，未修改环境。
- 实际验证：标准库 unittest，tests/test_facetrack_phase1.py（22）和 tests/test_facetrack_local_preference.py（14），合计 36/36 通过，测试运行时间 0.823 s。
- 新增 12 项测试，其中包含方案要求的八项，以及全 WEAK、多 witness、细小 mask 间隙、GOOD/UNCERTAIN 的补充覆盖。
- 默认全量 A3：2800 个 V 任务、963 区域、25873 候选、11878 个 (region,V)，六进程；两变体同池复用。
- 候选成员、MBG 与原缓存一致；冻结源文件与缓存执行前后指纹一致；三个策略微差审计均为 0；多数反对 B1 winner 计数 0。
- 图册只读既有评分及 compact 分支缓存，不调用 scorer/reducer/mesh。各图原始 UZ、无平移、无归一化，box 与 V 列表记录在 figure_manifest.json。

实际测试命令（未在报告生成时重复执行）：

```powershell
python -c 'import unittest; loader=unittest.TestLoader(); suite=unittest.TestSuite([loader.discover("tests",pattern=p) for p in ("test_facetrack_phase1.py","test_facetrack_local_preference.py")]); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```
'''
    (folder/'VERIFICATION.md').write_text(verification,encoding='utf-8')
    assert len(atlas.figure_records)==19
    assert all(Path(r['file']).is_file() for r in atlas.figure_records)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in frozen.items())
    write_json(folder/'figure_manifest.json',dict(figures=atlas.figure_records,render_seconds=time.perf_counter()-start,
               scores_unchanged=True,mesh_reads=0,scorer_calls=0,coordinate_transform='none',
               source=str(folder),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('OUTPUT',str(folder/'A3_QUALITATIVE_AUDIT_REPORT.md'),'FIGURES',len(atlas.figure_records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    main(parser.parse_args().input.resolve())
