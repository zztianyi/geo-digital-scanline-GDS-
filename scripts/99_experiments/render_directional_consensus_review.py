"""Sparse scientific figures and Chinese review from the validated frozen run."""
from pathlib import Path
import sys,json,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/04_structure_recognition'),str(ROOT/'scripts/03_slicing_profiles')]
from joint_surface_consensus import arc_support_profile,joint_surface_summary,SCALES
OUT=ROOT/'outputs/directional_surface_consensus/20260920'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
    'axes.titlesize':13,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#1268ad';ORANGE='#bc743a';GRAY='#b8bec5';PURPLE='#863caf';GREEN='#17794f';RED='#bf3d45'


def load_old(key):
    directory=ROOT/'outputs/min_branch_sweet_zone/20260920'
    index=json.loads((directory/'main_track_index.json').read_text(encoding="utf-8"))
    with (directory/'main_tracks.pkl').open('rb') as f:
        f.seek(index[key]);return pickle.load(f)['result']


def curves(ax,branches,bounds,*,scale=1.):
    for b in branches:
        p=np.asarray(b['points_uz']);ax.plot(p[:,0]*scale,p[:,1],color=GRAY,lw=1.2,zorder=1)
    ax.set(xlim=np.asarray(bounds[:2])*scale,ylim=bounds[2:],xlabel='径向偏移 u (mm)' if scale==1000 else '径向偏移 u (m)',ylabel='高程 z (m)')
    ax.grid(alpha=.15)


def route(ax,result,color,*,scale=1.):
    observed=[];synthetic=[]
    for r in result['path_edges']:
        p=np.asarray(r['points_uz']).copy();p[:,0]*=scale
        (observed if r['source'].startswith('OBSERVED') else synthetic).append(p)
    ax.add_collection(LineCollection(observed,colors=color,linewidths=2.5,zorder=3))
    ax.add_collection(LineCollection(synthetic,colors=RED,linewidths=1.8,linestyles='--',zorder=4))


def joins(ax,result,bounds,*,scale=1.):
    for j in result['junctions']:
        p=np.asarray(j['a_point_uz'])
        if bounds[0]<=p[0]<=bounds[1] and bounds[2]<=p[1]<=bounds[3]:
            ax.scatter([p[0]*scale],[p[1]],c=PURPLE,s=40,zorder=5)


def finish(fig,name,title,handles,footer='图中灰线均为原始候选；坐标未平滑、未平均。'):
    fig.suptitle(title,fontsize=16,y=.98)
    fig.legend(handles=handles,loc='lower center',ncol=len(handles),bbox_to_anchor=(.5,.038),frameon=False)
    fig.text(.5,.012,footer,ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=(0,.10,1,.935));fig.savefig(FIG/name,dpi=170);plt.close(fig)


def handle(color,label,style='-'):
    return Line2D([],[],color=color,lw=2.4,linestyle=style,label=label)


def compare(results,key,name,bounds,title,old=None,scale=1.):
    item=results[key];new=item['result'];bs=item['branches']
    old=load_old(key) if old is None else old
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.7),sharey=True)
    for ax in axes:curves(ax,bs,bounds,scale=scale)
    route(axes[0],old,ORANGE,scale=scale);route(axes[1],new,BLUE,scale=scale);joins(axes[1],new,bounds,scale=scale)
    axes[0].set_title('上一轮路径 / 对照路径');axes[1].set_title('本轮保留的原始观测路径')
    legend=[handle(ORANGE,'上一轮 / 对照'),handle(BLUE,'本轮观测主轨'),handle(GRAY,'其他原始候选')]
    synthetic_visible=False
    for r in old['path_edges']+new['path_edges']:
        if r['source'].startswith('OBSERVED'):continue
        p=np.asarray(r['points_uz'])
        visible=p[:,0].max()>=bounds[0] and p[:,0].min()<=bounds[1] and p[:,1].max()>=bounds[2] and p[:,1].min()<=bounds[3]
        synthetic_visible|=visible and np.linalg.norm((p[1]-p[0])/np.array([bounds[1]-bounds[0],bounds[3]-bounds[2]]))>.003
    joint_visible=any(bounds[0]<=j['a_point_uz'][0]<=bounds[1] and bounds[2]<=j['a_point_uz'][1]<=bounds[3] for j in new['junctions'])
    if synthetic_visible:legend.append(handle(RED,'显式连接段','--'))
    elif joint_visible:legend.append(Line2D([],[],marker='o',ls='',color=PURPLE,label='实际换轨点'))
    footer='图中灰线均为原始候选；坐标未平滑、未平均。'
    if name=='08_rejected_14m.png':
        axes[0].set_title('早期错误直连（非上一轮结果）')
        axes[0].text(-65.5,1476.6,'14.05 m 错误连接',color=RED)
        axes[1].annotate('B4：3 点 / 5.642 mm\n见下一幅毫米放大图',xy=(-69.599,1478.122),xytext=(-65.8,1477.4),
            arrowprops=dict(arrowstyle='->',color='#777'),fontsize=10)
    if name=='01_real104_context.png':
        axes[0].text(21.15,1380.8,'B2 原尾段',color=ORANGE,ha='right')
        axes[1].text(20.25,1380.,'B1 向下延续',color=BLUE,ha='right')
        axes[1].text(9.4,1386.35,'上方保留 B2',color=BLUE)
    if name=='07_long_continuation.png':
        axes[0].set_title('受控反事实：两次换轨');axes[1].set_title('本轮受控求解：一次换轨')
        footer='这是受控算法示例；径向轴为 mm、高程轴为 m，红线长度为毫米级，不能按屏幕角度理解坡度。'
    finish(fig,name,title,legend,footer)


def figures(results,data):
    compare(results,'104.30','01_real104_context.png',[9.,22.3,1378.7,1387.5],
        '案例 1｜真实 s=104.30 m：B2 尾段改由 B1 延续')
    actual=next(j for j in results['104.30']['result']['junctions'] if j.get('handoff_kind')=='INTERNAL_HANDOFF')
    compare(results,'104.30','01b_real104_junction.png',[10.1,11.35,1384.85,1385.50],
        f"案例 1 放大｜真正接入点位于折返重叠区，约 z={actual['a_point_uz'][1]:.3f} m")
    for i,key,title in ((2,'reliable_fold','案例 2｜受控回归：有宽邻域共识的折返保留'),
                        (3,'directional_tail','案例 3｜受控回归：内部换轨替换低置信折返尾段')):
        item=results[key];bs=item['branches'];r=item['result'];bounds=[-.3,4.4,3.8,5.6]
        fig,ax=plt.subplots(figsize=(11,6));curves(ax,bs,bounds);route(ax,r,BLUE);joins(ax,r,bounds)
        ax.annotate('原始折返段',xy=(2.5,5.),xytext=(2.6,5.4),arrowprops=dict(arrowstyle='->',color='#555'))
        legend=[handle(BLUE,'保留的观测主轨'),handle(GRAY,'原始候选')]
        if r['junctions']:legend.append(Line2D([],[],marker='o',ls='',color=PURPLE,label='换轨点'))
        finish(fig,f'0{i}_{key}.png',title,legend,footer='这是受控算法回归示例，不是模型的实测区域。')
    g=data['cases']['104.30']['graph'];node_s=104.3
    z=min(g['level_z'].values(),key=lambda x:abs(x-1380.7863))
    colors={1:BLUE,2:GRAY,6:ORANGE}
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.7))
    for bid in (1,2,6):
        profile=arc_support_profile(g,(node_s,bid));hit=min(profile,key=lambda r:abs(r['z']-z))
        same=[h for h in g['observations'] if h['level_index']==hit['level_index'] and h['h_path_id']==hit['h_path_id']]
        same.sort(key=lambda h:h['s']);xy=np.array([[h['s']-node_s,h['u']] for h in same])
        axes[0].plot(xy[:,0],xy[:,1],'.-',color=colors[bid],label=f'B{bid} 的同一 H 路径')
        b=g['nodes'][(node_s,bid)];p=b['points_uz'];axes[1].plot(p[:,0],p[:,1],color=colors[bid],lw=2)
        axes[1].scatter([hit['u']],[hit['z']],color=colors[bid],s=32)
    axes[0].axvline(0,color='#777',lw=.8,ls=':');axes[0].set(xlim=(-1.05,1.05),xlabel='邻近测线位置 s − 104.30 (m)',ylabel='H 交点径向偏移 u (m)',title=f'同一高程的横向路径：z={z:.3f} m')
    axes[1].axhline(z,color='#777',lw=.8,ls=':');axes[1].set(xlim=(20.3,21.5),ylim=(1379.,1382.),xlabel='目标 V 径向偏移 u (m)',ylabel='高程 z (m)',title='这些 H 交点分别落在三条原始 V 分支上')
    for ax in axes:ax.grid(alpha=.15)
    finish(fig,'04_competing_observations.png','案例 4｜真实数据：B1、B2、B6 都可能同时得到 H/V 观测支持',
        [handle(colors[b],f'B{b}') for b in (1,2,6)],footer='左图连接的是同一原始 H 路径在各 V 测线的交点；连接线仅辅助读图，不写入主轨。H 不代表真值。')
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.4));x=np.arange(4)
    for i,bid in enumerate((1,2,6)):
        support=joint_surface_summary(g,(node_s,bid),target_z=(1379.,1382.))
        axes[0].bar(x+(i-1)*.24,[support['wide_support'][f'{s:.2f}']['mean_s_span'] for s in SCALES],width=.23,color=colors[bid])
        axes[1].bar(x+(i-1)*.24,[support['wide_support'][f'{s:.2f}']['mean_neighbor_count'] for s in SCALES],width=.23,color=colors[bid])
    for ax in axes:
        ax.set_xticks(x,[f'±{s:.2f} m' for s in SCALES]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set(ylabel='平均连续支持跨度 (m)',title='连续跨度：不是 0 / 1 / 2 档评分')
    axes[1].set(ylabel='每个交点平均支持邻 V 数',title='统计区间 z=1379–1382 m，41 条 V 测线')
    finish(fig,'05_wide_persistence.png','案例 5｜局部都能找到支持，扩大邻域后才能区分持续程度',
        [handle(colors[b],f'B{b}') for b in (1,2,6)],footer='支持数和跨度是同一模型的冗余一致性指标，不是识别准确率。')
    item=results['short_counterfactual'];fig,ax=plt.subplots(figsize=(9,7));bounds=[-.001,.008,2.8,10.4]
    curves(ax,item['branches'],bounds,scale=1000);route(ax,item['result'],ORANGE,scale=1000);joins(ax,item['result'],bounds,scale=1000)
    ax.annotate('短分支在 z=6 结束\n仍需再换一次',xy=(1.,6),xytext=(2.5,6.7),arrowprops=dict(arrowstyle='->'))
    finish(fig,'06_short_future_switch.png','案例 6｜受控反事实：强制先选较近短支路，需要两次换轨',
        [handle(ORANGE,'强制短支路的对照路线'),handle(GRAY,'原始候选'),handle(RED,'显式连接段','--')],
        footer='对照序列 [0,1,2]，连接 1 mm + 5 mm；径向轴采用 mm，只为看清平行支路。')
    compare(results,'long_continuation','07_long_continuation.png',[-.001,.008,-.2,10.4],
        '案例 7｜受控实际求解：选择较持久支路，一次延续至目标范围',old=results['short_counterfactual']['result'],scale=1000)
    # The historical 14.05 m line is shown only as the rejected old proposal.
    old_data=pickle.load((ROOT/'outputs/reconstruction_constraint_review/20260920/local_evidence.pkl').open('rb'))
    old=old_data['cases']['89.75']['payload']['result']
    compare(results,'89.75','08_rejected_14m.png',[-72.,-51.,1472.0,1478.8],
        '案例 8｜真实 s=89.75 m：14.05 m 跨空连接继续禁止',old=old)
    b4=next(b for b in results['89.75']['branches'] if b['branch_id']==4)
    p=b4['points_uz'];fig,ax=plt.subplots(figsize=(8,4.8));ax.plot((p[:,0]-p[0,0])*1000,(p[:,1]-p[0,1])*1000,'o-',color=GRAY)
    ax.set(xlabel='相对首节点的径向偏移 (mm)',ylabel='相对首节点的高程差 (mm)',title='B4 原始断面只有 3 个节点、总长 5.642 mm');ax.grid(alpha=.2)
    finish(fig,'08b_tiny_branch.png','案例 8 细节｜B4 是短截面，不能作为跨空桥接目标',[handle(GRAY,'原始 B4（毫米坐标）')],
        footer='短截面不等于孤立噪点；前轮原始 Tile 核查已确认上部原始曲面被输入筛选截断。')
    compare(results,'122.25','09_real122.png',[-11.,-7.5,1408.5,1412.5],
        '补充真实复核｜s=122.25 m：在更早的交点进入 B10')


def write_reports(results,data):
    summary=json.loads((OUT/'summary.json').read_text(encoding="utf-8"));audit=json.loads((OUT/'real104_branch_audit.json').read_text(encoding="utf-8"))
    r=results['104.30']['result'];j=next(j for j in r['junctions'] if j.get('handoff_kind')=='INTERNAL_HANDOFF')
    rows=summary['real_cases'];maximum=max(x['max_connector_m'] for x in rows);ratio=max(x['max_combined_R_syn'] for x in rows)
    table='| 切片 s (m) | 本轮分支序列（低→高） | 内部换轨 | 裁尾 (m) | 最大连接 (mm) | 状态 | 求解 (s) |\n|---|---|---:|---:|---:|---|---:|\n'
    for row in rows:
        table+=f"| {row['slice_key']} | {row['branch_sequence']} | {row['internal_handoff_count']} | {row['internal_tail_trimmed_length']:.3f} | {row['max_connector_m']*1000:.6f} | {row['status']} | {row['seconds']:.3f} |\n"
    wide=[]
    for bid in (1,2,6):
        s=joint_surface_summary(data['cases']['104.30']['graph'],(104.3,bid),target_z=(1379.,1382.))
        wide.append(f"| B{bid} | "+' | '.join(f"{s['wide_support'][f'{k:.2f}']['mean_s_span']:.3f}" for k in SCALES)+' |')
    audit_table='| 分支 | 节点 | 总弧长 m | ASC m | MBG | 前向有效高程 m | 联合平均跨度 m | 支持 V 数 | 同 H 连续层数（最长） | 后续换轨估计 |\n|---|---:|---:|---:|---|---:|---:|---:|---:|---:|\n'
    for a in audit:
        audit_table+=f"| B{a['branch_id']} | {a['canonical_node_count']} | {a['full_arc_length']:.3f} | {a['ASC_arc_length']:.3f} | {a['MBG_pass']} | {a['Z_forward_new']:.3f} | {a['support_s_span']:.3f} | {a['support_slice_count']} | {a['same_H_run_longest_level_count']} | {a['estimated_additional_switches']} |\n"
    regional='| 分支 | ±0.05 m | ±0.25 m | ±0.50 m | ±1.00 m |\n|---|---:|---:|---:|---:|\n'+'\n'.join(wide)+'\n'
    drs_table='| 分支 / 起点 | N_forward | L_forward_arc m | Z_forward_new m | E_progress | L_forward_ASC m | DRS |\n|---|---:|---:|---:|---:|---:|---|\n'
    for a in audit:
        drs_table+=f"| B{a['branch_id']} / 向下起点 | {a['N_forward']} | {a['L_forward_arc']:.3f} | {a['Z_forward_new']:.3f} | {a['E_progress']:.3f} | {a['L_forward_ASC']:.3f} | {a['directional_state']} |\n"
    for label,a in (('B2 / 实际裁切点',j['directional_metrics']),('B1 / 实际进入点',j['candidate_directional_metrics'])):
        drs_table+=f"| {label} | {a['N_forward']} | {a['L_forward_arc']:.3f} | {a['Z_forward_new']:.3f} | {a['E_progress']:.3f} | {a['L_forward_ASC']:.3f} | {a['directional_state']} |\n"
    review=f'''# 本轮局部案例与读图说明

本轮结果目录：`directional_surface_consensus/20260920`。蓝线是这轮算法最终保留的原始 V 边；橙线是上一轮结果或明确标注的受控对照。灰线是未选中的原始候选，不是第二条最终主轨。紫点是实际换轨位置。红虚线只表示显式连接段，没有 FaceID，也不作为观测曲面。

每幅图只显示 2–4 类元素。图轴为径向偏移 u 与高程 z，通常单位 m；案例 6、7 的径向轴以及案例 8 的微小分支图明确使用 mm。所有放大范围只用于展示，不参与识别。先看蓝线与橙线在什么位置分开，再看换轨点和被留作灰线的部分。

## 1. 真实 104.30 m：B1 在锁定 B2 尾段之前参与

上一轮 `[2,4]`，本轮 `{r['route_branch_sequence']}`（按低→高列出；向下行走为 B4→B2→B1）。这里有多条重建观测重叠，算法最终只保留一条分段主轨。B2 裁尾 {j['internal_tail_trimmed_length']:.3f} m，在 z={j['a_point_uz'][1]:.6f} m 接入 B1。该数值是沿原始折线的弧长，不是缺失高程。

![上下文](figures/01_real104_context.png)

![交点放大](figures/01b_real104_junction.png)

交点三维残差 {j['xyz_distance_m']*1000:.6f} mm；这表示规范化坐标的数值差，并不代表模型有亚微米精度。没有插入长直线补出地质曲面。裁去 B2 尾段也不等于把折返变直：B1 自身的原始折返仍然保留，替换的是同一区域的竞争观测来源。

## 2. 可靠折返保留（受控回归）

此例给折返区域配置了宽邻域一致的真实折线交叉观测。蓝线保留折返；即使另有更长候选，也不能把已有共识的折返当成坏尾段删除。

![可靠折返](figures/02_reliable_fold.png)

## 3. 低置信尾段可在内部提前换轨（受控回归）

几何与案例 2 相同，但折返只在窄邻域重复，另一候选长期持续。紫点位于原曲线内部，切换后不再沿灰色折返绕行。真实观测点没有移动。

![内部替换](figures/03_directional_tail.png)

## 4. H 与 V 都可能重复（真实数据）

三条候选都能找到自己的同一 H 路径，不能把“有横向点”直接解释为正确曲面。左图看横向路径持续到多远；右图看这些点落在哪条 V 分支。只连接同一高程、同一 H-run 的实测交点，不把不同曲面的点平均。

![竞争观测](figures/04_competing_observations.png)

## 5. 宽邻域区分局部同等级候选（真实数据）

以下为 z=1379–1382 m 的平均连续支持跨度（m），不是全轨范围，也不是准确率。

{regional}
![宽邻域](figures/05_wide_persistence.png)

## 6. 较近短支路会再终止（受控反事实）

为解释前瞻代价，图中强制先进入近支路，实际组装出 `[0,1,2]`；短支路到 z=6 m 就结束，需要第二个连接。它是用于比较的受控路径，不是本轮模型识别结果。

![短支路](figures/06_short_future_switch.png)

## 7. 持久支路减少后续切换（受控求解）

同一输入中，本轮实际选择 `[0,2]`，一次 6 mm 连接即可到达目标。它没有因 1 mm 的近支路更近就优先选短支路。总连接长度相同的情况下，换轨次数也能影响决策。

![长支路](figures/07_long_continuation.png)

## 8. 14.05 m 连接仍拒绝（真实 89.75 m）

左图是早期错误的 14.05 m 直连，右图是本轮结果：不接入 B4。B4 只有 3 个规范节点和 5.642 mm 的弧长，MBG 不通过；它不能把目标范围扩大到上方再强制桥接。原始 Tile 核查表明这里还有输入截断问题，当前保持 SOURCE_DATA_REQUIRED。

![长连接拒绝](figures/08_rejected_14m.png)

![微小 B4](figures/08b_tiny_branch.png)

## 补充：122.25 m 真实交点变化

本轮序列 `{results['122.25']['result']['route_branch_sequence']}`。该位置用内部搜索找到更早交点，不能继续用旧终点间的 10.741 mm 代表本轮连接长度。完整切片最大连接仍为 {next(x['max_connector_m'] for x in rows if x['slice_key']=='122.25')*1000:.3f} mm。

![122.25](figures/09_real122.png)

## 仍需人工复核

蓝线是算法输出，不是已确认地质真值。重点检查真实 104.30 m 的 B1/B2 重叠区是否属于目标曲面，以及 89.75、122.25 m 新增内部换轨的曲面身份。本轮采用 3 个既有问题切片及其各 41 条邻近 V，未重新完成全区域 2800 切片普查。
'''
    (OUT/'REPRESENTATIVE_REVIEW.md').write_text(review,encoding='utf-8')
    real=f'''# 真实 s=104.30 m 的 B1 竞争复核

结论：B1 在本轮被选为向下持续分支。最终低→高序列 `{r['route_branch_sequence']}`，状态仍为 `{r['status']}`，原因是模型内有竞争曲面，尚无独立真值确认。

## 原始几何与新的判据

旧结果 `[2,4]` 的 0.700146 m 是先锁住 B2 所有折返后计算的剩余可接距离。本轮没有把它当作 B1 的先验拒绝理由。原始 B1/B2 在 UZ 中有 53 个相交候选；曲面假设、方向尾区和持续性先决定可换区域，再在两个独立弧长区间内寻找实际接点。

{audit_table}
表中 DRS 和前瞻用于整支路向下起点的可重复对比，并非宣称这些支路从任意位置都同样可靠。每个真实 H 交点的 edge_id、arc、u、z、H-run 和邻 V 身份保存在 `real104_arc_support.json`，完整连续指标在 `real104_branch_audit.json`。

{drs_table}
最后两行在实际接点重新计算：B2 即使还有较长原始弧长和 ASC，也可能缺少持久的新增方向覆盖。E_progress 仅解释该现象，不作为独立删除条件。

## 实际换轨点

- UZ：({j['a_point_uz'][0]:.9f}, {j['a_point_uz'][1]:.9f}) m。
- B2 裁去尾段：{j['internal_tail_trimmed_length']:.6f} m。
- 实际连接三维长度：{j['xyz_distance_m']:.12g} m，远低于 0.010 m。
- 选定前过渡区：A 原始弧长 {j['transition_a_arc']}；B 原始弧长 {j['transition_b_arc']}。A/B 指内部比较的当前 B2 / 候选 B1，最终低→高连接记录相反。
- 进入 B1 后的后续换轨估计：{j['future_switch_estimate_after']}；进入可竞争短分支的估计：{j['future_switch_alternative_estimate']}。这是深度 2 的局部估计，不是全局最短路径证明。

## 为什么 B1 优于 B6

在下方真正继续向前覆盖的区间，B1 的宽邻域共同支持更持久，也能延续到更低位置。B6 只有有限的前向范围；相同近邻 tier=2 并不能表达此差异。

{regional}
![同源竞争证据](figures/04_competing_observations.png)

![实际交点](figures/01b_real104_junction.png)

联合共识仍来自同一重建模型，故只能论证“B1 是当前证据下更好的候选”，不能等同于正确率或真实曲面认证。
'''
    (OUT/'REAL_104_30_B1_AUDIT.md').write_text(real,encoding='utf-8')
    metrics={k:sum(x[k] for x in rows) for k in ('directional_tail_count','internal_handoff_count','internal_tail_trimmed_length',
        'folded_branch_handoff_evaluated_count','folded_branch_handoff_accepted_count','future_switch_avoided_count','ABA')}
    alljoins=[j for v in results.values() for j in v['result']['junctions'] if j.get('handoff_kind')=='INTERNAL_HANDOFF']
    realjoins=[j for k in ('89.75','104.30','122.25') for j in results[k]['result']['junctions'] if j.get('handoff_kind')=='INTERNAL_HANDOFF']
    before=float(np.mean([j['future_switch_estimate_before'] for j in realjoins]));after=float(np.mean([j['future_switch_estimate_after'] for j in realjoins]))
    summary.update(metrics,mean_future_switch_estimate_before=before,mean_future_switch_estimate_after=after,
        B1_case_selected_branch_sequence=r['route_branch_sequence'],B1_case_connector_length=j['xyz_distance_m'],
        B1_case_trimmed_B2_tail_length=j['internal_tail_trimmed_length'])
    for radius in SCALES:
        summary[f'wide_support_{radius:.2f}m']={f'B{a["branch_id"]}':a['wide_support'][f'{radius:.2f}'] for a in audit}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    answers=[
        'DRS 分别计算剩余节点、弧长、有效方向高程与剩余 ASC；E_progress 仅记录，不以直线程度删支路。',
        'select_handoff 已委托弧长区域搜索，取消任何 Z 折返立即退出的逻辑。',
        '组装器在端部范围锁定之前执行 INTERNAL_HANDOFF，因此允许在已有高程覆盖内部接入。',
        '有宽邻域联合观测支持的物理 ASC 折返受保护；受控回归确认其原始节点保留。',
        'B1 在 B2 尾段锁定前竞争；真实最终结果包含 B1。',
        '0.700146 m 仅解释旧实现的结果，不参与本轮拒绝。',
        '无 H 降低证据置信度或保留歧义，不再触发 REJECT_SURFACE_EVIDENCE。',
        '联合假设绑定原始 V 边、每层 H-run、真实相交邻 V；多值交点分别保留。',
        '四尺度 ±0.05/0.25/0.50/1.00 m 全部输出；使用跨度与邻 V 数，非四个硬阈值。',
        '连续跨度参与候选竞争；局部同 tier 候选可由宽邻域分开。',
        '记录 future_reliable_arc、future_directional_extent、future_surface_support_span 及 estimated_additional_switches。',
        '深度 2 有界搜索覆盖下一次终止；不能证明整模型最少切换。',
        '受控短支路需 2 次，长支路只需 1 次；真实 B1 相比 B6 减少预计后续切换。',
        '真实 B1 胜出的原始观测依据详见 B1 专项报告与逐交点 JSON。',
        '14.05 m 连接仍被拒绝，B4 不通过 MBG。',
        '每条输出观测边按原始 edge_id 与 t0/t1 回算 UZ/XYZ，一致性检查通过。',
        f'本轮 3 真实切片最大连接 {maximum*1000:.6f} mm，保持 10 mm 硬限。',
        f'相邻两侧连接合计 R_syn 最大 {ratio:.8g}，全部严格小于 1。',
        '可靠核心内折返保留的回归通过。',
        '低置信方向尾段折返可裁，真实 B2 裁尾与受控早换轨均已验证。']
    verdicts=[('DIRECTIONAL_TAIL_DETECTION','SUPPORTED'),('FOLDED_BRANCH_HANDOFF','SUPPORTED'),
        ('INTERNAL_TRACK_REPLACEMENT','PARTIALLY_SUPPORTED'),('JOINT_ORTHOGONAL_SURFACE_CONSENSUS','SUPPORTED'),
        ('WIDE_NEIGHBOR_PERSISTENCE','SUPPORTED'),('FUTURE_SWITCH_MINIMIZATION','PARTIALLY_SUPPORTED'),
        ('REAL_104_30_B1_CASE','SUPPORTED'),('MINIMUM_GEOMETRY_INTERVENTION','SUPPORTED')]
    report=f'''# 方向性尾区与联合正交共识：本轮分析报告

**104.30 m 的识别输出已由 `[2,4]` 改为 `{r['route_branch_sequence']}`，B1 在内部交点接入，B2 低置信尾段裁去 {j['internal_tail_trimmed_length']:.3f} m。** 所有保留观测仍来自原始边，未平均、未平滑。这里的“识别输出”是 branch-first/surface-track 分析求解器的输出，尚未接入生产 GUI 的识别入口。

## 本轮改动

静态 ASC 继续判断支路是否有足够的规范节点和物理核心；新的 DRS 判断沿某个方向还能提供多少有效覆盖。每个 H 交点与各自 V 的原始弧长位置绑定，同高程多值不合并。当前分支变弱且另一候选在更宽邻域持续更好时，先决定竞争假设与过渡区域，再搜索真实交点/投影，最后执行 10 mm、实际保留核心和合计 R_syn 检查。

这轮未更改原始网格、法向筛选、40°/70°阈值、FaceID 传播、voxel union、闭合分量处理，以及 MBG/K=4/Ncore=3/4倍中位边长的绝对核心参数。

## 实际结果与阶段性能

{table}
计时仅包含各切片求解及其联合证据索引建立；原始输入准备单独记录在 `input_prepare.json`。3 个问题切片合计 {sum(x['seconds'] for x in rows):.3f} s。输入准备使用 123 条 V（3×41）与完整冻结 H 层，耗时 {json.loads((OUT/'input_prepare.json').read_text(encoding="utf-8"))['seconds']:.3f} s。没有用本次局部计时推算全区域吞吐，也没有与上一轮不同范围的 2800 切片计时作速度比。

相关测试 {summary['targeted_tests']} 项通过，另执行真实 `test_real_104_30_B1_competition` 和 3 真实切片逐边来源/连续性/连接预算核查。本轮没有重新跑全项目无关测试，也没有重跑全区域 2800 切片。

## 审核统计

```json
{json.dumps(metrics,ensure_ascii=False,indent=2)}
```

内部换轨前/后的平均后续切换估计：{before:.3f} / {after:.3f}。这些是候选评估事件和局部图搜索统计，重复评估不等于不同真实缺陷；不是准确率。UNRESOLVED 2 个，SOURCE_DATA_REQUIRED 1 个，INPUT_TRUNCATION_SUSPECT 0 个。本轮不据此宣称整个模型已解决。

## 代表案例与怎样看图

[局部案例和完整读图说明](REPRESENTATIVE_REVIEW.md) 包含方案要求的 8 类对比及 122.25 m 补充复核；[104.30 m 专项证据](REAL_104_30_B1_AUDIT.md) 给出 B1/B2/B3/B4/B6 指标与原始正交交点。

![本轮关键局部](figures/01_real104_context.png)

## 对方案 20 个问题的回答

'''+ '\n'.join(f'{i}. {text}' for i,text in enumerate(answers,1))+'''

## 判定

判定仅针对这轮局部验证和受控回归；SUPPORTED 不表示全区域正确率已验证。

| 项目 | 判定 |
|---|---|
'''+ '\n'.join(f'| {k} | {v} |' for k,v in verdicts)+'''

## 人工验收与已知风险

请重点检查 B1/B2 换轨点附近的目标曲面身份，以及真实 89.75、122.25 m 新增接点。共同 H/V 来源于同一模型，重建重复面可能在两个方向都持续存在；此时保留 UNRESOLVED。未来换轨搜索只有深度 2，方向尾区的相对证据阈值属于工程判据，尚未做全区域参数鲁棒性确认。旧的仅含分支链接缓存无法为多值交点提供弧长身份，内部换轨需要原始交点输入。

偏差与实现边界见 [IMPLEMENTATION_DEVIATIONS.md](IMPLEMENTATION_DEVIATIONS.md)，独立复核及处理结果见 [FINAL_REVIEW.md](FINAL_REVIEW.md)。
'''
    (OUT/'DIRECTIONAL_HANDOFF_REPORT.md').write_text(report,encoding='utf-8')
    deviations=f'''# 实现偏差与边界

1. 联合证据增加在 `joint_surface_consensus.py`，未修改原 `observed_surface_graph.py` 的建图规则。原链接缓存仍可用于粗置信选择；缺少原始交点的缓存不允许凭推测裁折返。本轮重新从冻结的全 H 层建立了三个目标各 ±1 m 的实际交点。
2. DRS 默认要求前向有效高程小于候选的一半，且候选联合跨度超过当前的 1.25 倍并至少增加 0.025 m，才标为可比较的方向尾区。E_progress 不作删除阈值。这是工程实现，非理论常数；尚未全域标定。
3. 可靠折返保护在原始 ASC 内检查实际 H/V 持续跨度。即使窄高程折返只有一层 H 穿过，多条左右邻 V 的同路径支持也可保护它，避免用层数不足误删窄折返。
4. 前瞻采用深度 2；不可达用 `depth+1` 哨兵并输出 `future_frontier_reached=False`，不能把它解释成精确需要 3 次换轨。没有声称全局最优。
5. 内部替换当前只替换主轨两端各自支路的内部尾段，不对两个已固定连接之间的任意中间子段执行双接点重路由。本轮 B2 位于向下前沿，符合处理范围。
6. 最终选择固定目标后，多个安全接点优先保留该候选的持久观测延续，再比较连接几何；避免仅按最少裁尾又回到原来的长折返锁定问题。
7. 本轮验证为 {summary['targeted_tests']} 项相关测试、三个真实切片和 8 类展示。未做全区域重算、生产 GUI 集成或独立地质真值标注；案例中的受控数据明确标注。
8. 89.75 m 的 SOURCE_DATA_REQUIRED 沿用前轮原始 Tile 核查结论，未重新加载并修改原网格。
9. 用户要求在本地连续修改，故直接修改既有 backend 分支，不新建工作树、不自动提交、不修改依赖环境。
10. 独立复核补强：depth-2 前瞻携带每一步实际保留的路径，不允许返回已裁掉的入口之前；实际接点重新计算 DRS、候选持续性与整段可靠折返保护，不能直接沿用邻近采样点的许可。
'''
    (OUT/'IMPLEMENTATION_DEVIATIONS.md').write_text(deviations,encoding='utf-8')


if __name__=='__main__':
    results=pickle.load((OUT/'validated_results.pkl').open('rb'))
    data=pickle.load((OUT/'raw_cases.pkl').open('rb'))
    figures(results,data);write_reports(results,data)
    print('Generated 11 focused PNGs and 4 analysis reports; final review is separate.')
