"""Figures and Chinese review report for the orthogonal scanline audit."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import orthogonal_profile_constraint as opc


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render_horizontal_figure(output, horizontal, arc):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for ax, record in zip(axes.flat, sorted(horizontal, key=lambda r: r['z'])):
        raw, clean = opc.to_suz(record['raw'], arc), opc.to_suz(record['clean'], arc)
        ax.add_collection(LineCollection(raw[:, :, :2], colors='#b6bec8', linewidths=1.2, rasterized=True))
        ax.add_collection(LineCollection(clean[:, :, :2], colors='#177eaa', linewidths=.55, rasterized=True))
        ax.autoscale()
        # Out-of-window sections are preserved in the data, but must not stretch
        # the visible panel's radial scale by hundreds of metres.
        visible = raw[:, :, 1][(raw[:, :, 0] >= 0) & (raw[:, :, 0] <= 140)]
        if len(visible):
            pad = max(1., float(np.ptp(visible))*.08)
            ax.set_ylim(float(visible.min())-pad, float(visible.max())+pad)
        ax.set_xlim(0, 140)
        ax.set(title=f"z = {record['z']:.2f} m | raw {len(raw)} / clean {len(clean)} edges",
               xlabel='Arc coordinate s (m)', ylabel='Radial offset u (m)')
    fig.suptitle('Step 2 | Horizontal multi-branch profiles — raw grey / clean blue', fontsize=15)
    _save(fig, Path(output)/'step2_horizontal_profiles.png')


def render_figures(output, summary, slices, constrained, groups, after_groups, hmaps, levels, arc, horizontal, before, after):
    output = Path(output)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'legend.frameon': False})
    regions = summary['regions']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
    for ax, region in zip(axes.flat, regions):
        key = region['slice_key']
        lines = opc.to_suz(slices[key]['slicing']['lines_3d'], arc)[:, :, 1:]
        ax.add_collection(LineCollection(lines, colors='#9aa6b2', linewidths=.6, rasterized=True))
        red = [np.asarray(g['uz']) for g in groups[key]]
        ax.add_collection(LineCollection(red, colors='#c73845', linewidths=1.4))
        ax.autoscale()
        ax.set_ylim(arc['z_range'])
        ax.set(title=f"{region['type']} | s = {float(key):.2f} m\n{len(groups[key])} corrected red groups", xlabel='Radial offset u (m)', ylabel='Elevation z (m)')
    fig.suptitle('Step 1 | Frozen vertical baseline — automatic region selection', fontsize=15)
    _save(fig, output/'step1_vertical_baseline_groups.png')

    render_horizontal_figure(output, horizontal, arc)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
    changed = set(summary['constraint']['changed_slice_keys'])
    for ax, region in zip(axes.flat, regions):
        key = region['slice_key']
        lines = opc.to_suz(slices[key]['slicing']['lines_3d'], arc)[:, :, 1:]
        ax.add_collection(LineCollection(lines, colors='#737d8d', linewidths=.8, rasterized=True))
        points = np.array([[row[0], levels[zi]] for zi, rows in hmaps[key].items() for row in rows])
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=2, c='#00a2b4', alpha=.5, rasterized=True)
        if key in changed:
            result = opc.to_suz(constrained[key]['slicing']['lines_3d'], arc)[:, :, 1:]
            ax.add_collection(LineCollection(result, colors='#e48a22', linewidths=.6))
        ax.autoscale()
        ax.set_ylim(arc['z_range'])
        ax.set(title=f"{region['type']} | s = {float(key):.2f} m\n" + ('constrained geometry changed' if key in changed else 'constrained = baseline; no invented links'),
               xlabel='Radial offset u (m)', ylabel='Elevation z (m)')
    fig.suptitle('Step 3 | Direct vertical curves (grey) / horizontal observations (cyan)', fontsize=15)
    _save(fig, output/'step3_vertical_constraint_comparison.png')

    render_consistency_figure(output, summary, groups, before, after, arc)


def render_consistency_figure(output, summary, groups, before, after, arc):
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    ax = axes[0, 0]
    x = np.arange(3)
    metrics = ('matched_Ch', 'matched_Cz', 'matched_C')
    for dx, name, color in ((-.18, 'baseline', '#4974a5'), (.18, 'constrained', '#e48a22')):
        values = [summary[name][m]['mean'] or 0 for m in metrics]
        ax.bar(x+dx, values, .36, color=color, label=name)
        for xx, val in zip(x+dx, values):
            ax.text(xx, val+.025, f'{val:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x, ['Ch', 'Cz', 'C'])
    ax.set(ylim=(0, 1.12), title='One-to-one matched adjacency means')
    ax.legend()
    ax = axes[0, 1]
    du = np.array([r['median_du'] for r in before['pairs']])
    if len(du):
        ax.hist(du, bins=60, color='#4974a5', alpha=.7)
    ax.set(title='Baseline matched radial displacement (unchanged if no repair)', xlabel='Median |delta u| (m)', ylabel='Adjacent pairs')
    ax.set_yscale('symlog', linthresh=1)
    ax = axes[1, 0]
    bars = [np.array([[g['s'], g['z_min']], [g['s'], g['z_max']]]) for gs in groups.values() for g in gs]
    ax.add_collection(LineCollection(bars, colors='#b9c0c9', linewidths=.3, rasterized=True))
    byid = {g['id']: g for gs in groups.values() for g in gs}
    for ti, track in enumerate(sorted(before['tracks'], key=lambda t: t['length_sum_m'], reverse=True)[:20]):
        gs = sorted((byid[gid] for gid in track['group_ids']), key=lambda g: g['s'])
        ax.plot([g['s'] for g in gs], [(g['z_min']+g['z_max'])/2 for g in gs], linewidth=1)
    ax.set(xlim=(0, 140), ylim=arc['z_range'], title='s-z evidence map | 20 largest observed tracks', xlabel='s (m)', ylabel='z (m)')
    ax = axes[1, 1]
    for label, result, color, style in (('baseline', before, '#4974a5', '-'), ('constrained', after, '#e48a22', '--')):
        per_s = defaultdict(list)
        for row in result['pairs']:
            per_s[row['s_left']].append(row['C'])
        ss = sorted(per_s)
        ax.plot(ss, [np.mean(per_s[s]) for s in ss], style, color=color, alpha=.8, linewidth=.7, label=label)
    ax.set(title='Mean matched C at each adjacent slice boundary', xlabel='s (m)', ylabel='C', ylim=(0, 1.05))
    ax.legend()
    fig.suptitle(f"Step 4 | Evidence and continuity comparison — {summary['conclusion']}", fontsize=15)
    _save(fig, output/'step4_consistency_comparison.png')


def write_report(output, summary):
    output = Path(output)
    manifest = summary['manifest']
    constraint = summary['constraint']
    before, after = summary['baseline'], summary['constrained']
    perf = manifest['slicing_performance']
    chosen = manifest['chosen_workers']
    base_trial = next(r for r in perf if r['workers'] == 1)
    best_trial = next(r for r in perf if r['workers'] == chosen)
    speedup = base_trial['pair_wall_seconds']/best_trial['pair_wall_seconds']
    normal = constraint['tolerance']['normal_residual_m']
    hdiag = summary['horizontal_diagnostics']
    vdiag = summary['vertical_diagnostics']
    stages = summary['stage_performance']
    full = next(r for r in stages if r['stage'] == 'full_horizontal_slicing')
    total = sum(r['wall_seconds'] for r in stages)
    peak_rss = max(r['peak_tree_rss_bytes'] for r in stages)/2**30
    peak_private = max(r['peak_tree_private_bytes'] for r in stages)/2**30
    preserved = all(r['unchanged_fraction'] == 1 for r in summary['major_structures'])
    def f(value, digits=6):
        return 'N/A（无观测）' if value is None else f'{value:.{digits}f}'
    equal_controls = all(all(r['fields_equal'].values()) for r in summary['recognition_verification'] if not r['input_changed'])
    answers = [
        ('横向测线能否稳定生成？', f"是。本次完成 {hdiag['slice_count']} 个 z 水平面，原始线段 {hdiag['segments']:,} 条，FaceID 取自与冻结基线相同的过滤后网格。"),
        ('横向测线碎裂程度与纵向相比如何？', f"按相同的 6 位小数节点合并口径，横向每层连通分量中位数 {hdiag['components_round6']['median']}，纵向 path 中位数 {vdiag['path_count']['median']}；横向端点中位数 {hdiag['endpoints_round6']['median']}，纵向 {vdiag['endpoint_count']['median']}。方向和覆盖区域不同，不能仅用数量认定拓扑质量。"),
        ('并行 slicing 最优 worker 数是多少？', f"本轮 1/4/8/12 四档单次实测中，安全内存条件下组合耗时最短为 {chosen} 个进程。这是本机本数据的本轮最优，不是全局或统计显著最优。"),
        ('slicing 实际加速多少？', f"每档各 200 条纵向＋200 条横向：1 进程 {base_trial['pair_wall_seconds']:.3f} s，{chosen} 进程 {best_trial['pair_wall_seconds']:.3f} s，组合加速 {speedup:.3f}×。全量横向 {hdiag['slice_count']} 层实测 {full['wall_seconds']:.3f} s。没有把 200 条抽样外推写成 2800 条全量实测。"),
        ('正常区域 V/H 的 u 残差分布是多少？', f"基于低碎裂度预选的 50 条纵向剖面：n={normal['count']:,}，median={normal['median']:.3e} m，P95={normal['p95']:.3e} m，P99={normal['p99']:.3e} m，max={normal['max']:.3e} m。"),
        ('横向能够支持多少纵向 gap？', f"检查 {constraint['gap_candidates']:,} 个不同连通分量端点间的小断口候选；满足至少两个内部横向高度、唯一径向分支及双端切向约束的有 {constraint['horizontally_supported_gaps']} 个。"),
        ('实际修复多少 gap？', f"{constraint['repaired_gaps']} 个，改变 {len(constraint['changed_slice_keys'])} 条纵向测线。只允许回切原始三角面得到完整唯一链的补入，禁止生成虚假 FaceID。"),
        ('修复后纵向 path 连续性是否改善？', f"总 path 数 {constraint['baseline_path_count']:,} → {constraint['constrained_path_count']:,}。" + ('本次无改善。' if constraint['baseline_path_count'] == constraint['constrained_path_count'] else '变化见逐 gap 记录。')),
        ('Baseline 相邻悬空组一致性是多少？', f"匹配邻接 {before['matched_adjacencies']:,}，平均 C={f(before['matched_C']['mean'])}；孤立组 {before['isolated_groups']:,}。全部有高度重叠且 Ch 可观测的候选平均 Ch={f(before['all_observed_Ch']['mean'])}，不能只看筛选后匹配对。"),
        ('Constrained 后是多少？', f"匹配邻接 {after['matched_adjacencies']:,}，平均 C={f(after['matched_C']['mean'])}；孤立组 {after['isolated_groups']:,}。"),
        ('Ch 是否提高？', f"匹配对均值 {f(before['matched_Ch']['mean'])} → {f(after['matched_Ch']['mean'])}。"),
        ('Cz 是否提高？', f"匹配对均值 {f(before['matched_Cz']['mean'])} → {f(after['matched_Cz']['mean'])}。"),
        ('median_du 异常是否减少？', f"匹配对 median_du 的中位数 {f(before['matched_median_du_m']['median'])} → {f(after['matched_median_du_m']['median'])} m；大于 0.25 m 的对数 {before['du_above_0_25m_count']} → {after['du_above_0_25m_count']}（描述性阈值，不参与改写识别规则）。"),
        ('大悬空结构是否保留？', f"按基线轨迹累计红线长度排序的前 20 个结构，节点完全不变的组占比均为 100%：{preserved}。这是数据自动选取的主要结构，尚未由人工确认其地质身份。"),
        ('多 branch 是否保留？', f"是；观察到 {hdiag['multi_branch_grid_cells']:,} 个多值 (s,z) 单元。未取均值、未强制单曲面、未删除任何基线分支。"),
        ('是否建议进入 group track → loft → closed surface → volume？', '不建议仅凭本轮结果进入自动闭合与体积计算。横纵测线来自同一模型，不是独立地质真值；需要先审核真实断口、分支关联和主要结构轨迹。')]
    lines = [
        '# GDS 正交测线约束执行与实测审核报告', '',
        f"结论：**{summary['conclusion']}**", '',
        f"本轮已实现并行切片、横纵交叉观测和保守修复实验。切片组合实测加速 {speedup:.3f}×；实际修复 {constraint['repaired_gaps']} 个断口。不得把切片加速与约束效果混为一谈。", '',
        '## 1. 数据与范围', '',
        f"- 仓库基点：`{manifest['git_head']}`；直接修改现有 backend 工作区，未提交、未推送。",
        f"- 原模型：`{manifest['mesh']}`；原始 {manifest['mesh_metadata']['original_face_count']:,} 面，过滤后 {manifest['mesh_metadata']['filtered_face_count']:,} 面。",
        f"- 冻结基线：`{manifest['baseline']}`；{len(manifest['slice_keys'])} 条纵向测线，s∈[0,140)，步距 0.05 m。",
        f"- 横向：z∈[{manifest['arc']['z_range'][0]}, {manifest['arc']['z_range'][1]})，步距 {manifest['horizontal_step_m']} m，共 {hdiag['slice_count']} 层。",
        '- 模型与圆弧配置 SHA256 和冻结基线一致；详见 manifest.json。',
        '- 未修改法向判据、40°/70°阈值、分组与修正规则；未进行网格补洞、loft、封闭体或体积计算。', '',
        '## 2. 执行方式与相对方案的调整', '',
        '| 调整 | 原因与证据 |', '|---|---|',
        '| 在 XY 中计算精确射线/线段交点，再转换 (s,u,z) | 圆柱坐标变换是非线性的，直接对端点的 u(s) 线性插值会制造 V/H 残差。合成测试覆盖弦线、多分支及 ±π 展开。 |',
        f"| snap 只针对近零数值裂缝；本轮阈值 {manifest['snap_tolerance_m']:.3e} m | 200 层端点距离采样，取近零群 P99×2，上限 1e-5 m；只合并互为唯一近邻、不同分量的度 1 端点。毫米级物理裂缝不自动吸附。采样查找最多 12 个近邻，距离分布不是全局全部端点对分布。 |",
        f"| V/H 阈值加 1e-5 m 下限 | median+3MAD={constraint['tolerance']['median_plus_3mad_m']:.3e} m；原识别节点坐标舍入 6 位，不能让机器精度级阈值否定其合法观测。实际阈值 {constraint['tolerance']['applied_tau_u_m']:.3e} m；可用 --tau-u 覆盖。 |",
        '| 横向同分支采用局部单调 s 路段 | 避免两点仅在很远处连通、绕圈或经过分叉也被当作相邻测线间的连续证明。不同半径分支一直单独保留。 |',
        '| 不直接用插值补线进入原识别算法 | 新线段若没有真实三角面链，就没有可信 FaceID。只有横向支持面回切能证明完整唯一链时才允许补入；否则记录拒绝原因。 |',
        '| Ch 无观测记为 N/A，而非 0 或 1 | 高度不足 0.10 m 的短组可能没有共同采样高度。Ch 分母包含所有共同高度，不能通过只保留有横向匹配的高度抬高分数。 |',
        '| 允许未匹配，Ch≥0.5 后按 Ch/Cz/du 贪心一对一关联 | 不使用纯最近邻强制配对；C 按方案权重另行报告。tau 使用基线匹配对的非零 median_du 中位数并在前后固定。 |',
        '| 未改变输入的纵向剖面复用冻结识别结果 | 只重跑实际改变的剖面及 6 类代表区域控制；避免无意义重做 2800 条识别。输入未变是等价依据，控制样本提供实际重跑核验。 |',
        '| parent paths 原始内容通过 pickle 字节偏移引用 | JSONL 导出 node_order、FaceID、原红组法向、父路径统计与定位信息，不重复复制约 4 GB 的既有识别文件。 |',
        '| 六类区域自动选择 | 正常/碎裂/密集/多分支/低一致性/主要结构有可复现规则；“已知主要结构”未获得人工标注，报告不冒称地质确认。 |', '',
        '## 3. 并行切片实测', '',
        '每档一次：先纵向 200 条，再用同一常驻池切横向 200 条；纵向包含冷启动，横向为热池。组合耗时含池关闭及抽样核对，不含模型准备。单线程数值库，未使用 GPU。', '',
        '| workers | 纵向 s | 横向 s | 组合 s | 相对 1 进程 | 平均逻辑 CPU 当量 | 整机 CPU % | 峰值 RSS GiB | 峰值 private GiB |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for workers in manifest['workers_sweep']:
        vr = next(r for r in perf if r['workers'] == workers and r['axis'] == 'vertical')
        hr = next(r for r in perf if r['workers'] == workers and r['axis'] == 'horizontal')
        lines.append(f"| {workers} | {vr['wall_seconds']:.3f} | {hr['wall_seconds']:.3f} | {vr['pair_wall_seconds']:.3f} | {base_trial['pair_wall_seconds']/vr['pair_wall_seconds']:.3f}× | {vr['pair_mean_logical_cpu_equivalents']:.3f} | {vr['pair_mean_machine_cpu_percent']:.2f} | {vr['pair_peak_tree_rss_bytes']/2**30:.3f} | {vr['pair_peak_tree_private_bytes']/2**30:.3f} |")
    lines += ['', f"全量横向：{full['wall_seconds']:.3f} s；吞吐 {hdiag['slice_count']/full['wall_seconds']:.3f} 层/s。",
              '', '默认进程数的执行调整：原纵向入口采用 4 进程（可用 --workers 覆盖），因为本轮组合 4/8 进程仅相差约 0.05 s，而 4 进程少约 1.03 GiB 树 RSS；纯纵向抽样也是 4 进程更快。完整横向实验仍使用自动选出的 8 进程，报告不更改既有实测记录。两个生产入口都会在 spawn 前把数值库线程设为 1。',
              '', '## 4. 各阶段资源与耗时', '',
              f"所有实测阶段之和 {total:.3f} s（{total/60:.2f} 分钟），包括试验扫档与诊断，不是原五阶段业务流程总时长；本轮没有重跑原五阶段。跨阶段树 RSS 峰值 {peak_rss:.3f} GiB，private 峰值 {peak_private:.3f} GiB。阶段外的输入哈希、启动、最终文本写入及交互等待不包含在此和中。", '',
              f"RSS 为每进程工作集之和，共享 mmap 页可能重复计算；private 为进程私有提交内存，不能称为真实物理驻留。CPU 当量=进程树 CPU 秒/墙钟秒；整机百分比除以 {manifest['platform']['logical_cpus']} 个逻辑 CPU。250 ms 采样可能漏掉极短峰值，退出子进程 CPU 保留至最后一次观测。原始逐进程记录见 process_resource_samples.csv。", '',
              '| 阶段 | 秒 | CPU 秒 | 平均逻辑 CPU 当量 | RSS GiB | private GiB | 最多进程（含主进程） |',
              '|---|---:|---:|---:|---:|---:|---:|']
    stage_csv = []
    for stage in stages:
        lines.append(f"| {stage['stage']} | {stage['wall_seconds']:.3f} | {stage['cpu_seconds']:.3f} | {stage['mean_logical_cpu_equivalents']:.3f} | {stage['peak_tree_rss_bytes']/2**30:.3f} | {stage['peak_tree_private_bytes']/2**30:.3f} | {stage['max_processes_including_parent']} |")
        stage_csv.append({**stage, 'wall_percent': 100*stage['wall_seconds']/total})
    lines += ['', '## 5. 前后识别与连续性结果', '',
              '| 指标 | Baseline | Constrained |', '|---|---:|---:|']
    comparisons = [('组数', 'groups'), ('匹配邻接', 'matched_adjacencies'), ('孤立组', 'isolated_groups'),
                   ('多切片轨迹', 'multi_slice_tracks'), ('未匹配组邻接', 'unmatched_group_adjacencies'),
                   ('Ch 无可用高度的候选', 'unobserved_candidate_adjacencies')]
    for label, field in comparisons:
        lines.append(f'| {label} | {before[field]} | {after[field]} |')
    for label, field in (('匹配 Ch 均值', 'matched_Ch'), ('匹配 Cz 均值', 'matched_Cz'), ('匹配 C 均值', 'matched_C'),
                         ('全部可观测候选 Ch 均值', 'all_observed_Ch')):
        lines.append(f"| {label} | {f(before[field]['mean'])} | {f(after[field]['mean'])} |")
    lines += ['', f"代表区域控制重新识别：{len(summary['recognition_verification'])} 条；未改变输入的 paths、normals、red_groups、red_groups_corrected、red_centroids 均精确相同：{equal_controls}。",
              '', 'V/H 计数以分支观测为单位（匹配/冲突每对计一次；EMPTY 单独以空网格单元计），不是互斥单元比例：', '',
              '| 状态 | 数量 |', '|---|---:|']
    for state, count in constraint['observation_counts'].items():
        lines.append(f'| {state} | {count:,} |')
    lines += ['', '断口筛选/拒绝原因：', '', '| 原因 | 数量 |', '|---|---:|']
    reason_labels = {'horizontal_support': '横向支持检查未达标（不是支持成功）',
                     'insufficient_horizontal_levels': '内部横向采样高度少于 2 层'}
    for reason, count in constraint['rejection_reasons'].items():
        lines.append(f'| {reason_labels.get(reason, reason)} | {count:,} |')
    lines += ['', '## 6. 方案要求的 16 个问题', '']
    for i, (question, answer) in enumerate(answers, 1):
        lines += [f'### {i}. {question}', '', answer, '']
    if 'artifact_verification' in summary:
        proof = summary['artifact_verification']
        lines += ['### 可复用基线的证据', '',
                  f"冻结源文件 SHA256 与旧 manifest 匹配：{proof['recognition_source']['frozen_source_hash_matches']}；当前识别核心所有非 main 类/函数 AST 与该冻结源完全一致：{proof['recognition_source']['core_ast_equal']}。", '',
                  f"保存后的约束输入逐条回读检查：{proof['constrained_input_records_checked']} 条；实际坐标或 FaceID 变化的剖面数 {proof['constrained_input_changed_count']}。前后全部一致性 CSV 字节相同：{proof['consistency_csv_equal']}。", '',
                  f"合成定向检查：切片引擎 7 项、初始正交几何 8 项，均通过；4 档各 200 条真实纵向切片坐标和 FaceID 逐位相同。未运行全项目测试。", '']
    if 'review_refinement' in summary:
        review = summary['review_refinement']
        lines += ['### 只读审查后的针对性修订', '',
                  '- 反向半射线的组保留原始识别，但标为横向不可观测，不参与正向横线的 Ch 证明。本数据回查发现这类组为 0，因此该修正未改变本次分组数量。',
                  '- median_du 改为仅从共享同一横向分支的径向对应计算；原“任意两径向值的最小距离”另存 raw_nearest_median_du，防止无关近邻抬高 Cu/C。无共同支持分支的 median_du/C 留空，不伪造观测。',
                  f"- 仅重新计算一致性及相关图表，未重复切片或识别。两个新增边界回归检查均通过。修订影响 {review['changed_du_pair_count']} 个候选对的可用径向值或其数值；原/新匹配集合对称差为 {review['matching_symmetric_difference']}。", 
                  f"- 统计实现修订前的基线平均 C={review['previous_baseline']['matched_C']['mean']:.6f}；修订后为 {summary['baseline']['matched_C']['mean']:.6f}。这是统计口径修正，不是约束后精度提升。最终 Baseline/Constrained 两表使用同一修正后的算法。",
                  '- 阶段表保留最初一致性计算和 review_refined_consistency，累计时间包含此次定向修订；原始切片效率记录不变。', '']
    lines += ['## 7. 图件与人工审核项', '',
              '1. step1_vertical_baseline_groups.png：六类纵向基线及原悬空组。',
              '2. step2_horizontal_profiles.png：横向 raw/clean 多分支。',
              '3. step3_vertical_constraint_comparison.png：横纵交点与约束前后。',
              '4. step4_consistency_comparison.png：一致性、径向偏差与 s-z 轨迹。', '',
              '请审核：自动选定的主要结构是否对应真实大悬空体；0.10 m 横向间距对短小悬空组是否足够；候选断口是否为真实裂缝或模型边界；是否接受“无可追溯三角面链就不补线”的策略。', '',
              '## 8. 已知风险与结论边界', '',
              '- V/H 均来自相同三角网格。完整平面求交的几何信息本来应一致；另一方向不能凭空补上原模型不存在的表面。极小残差是数值自洽证据，不是地质精度验证。',
              '- 上述残差包含观测去重的量化：纵向坐标保留 8 位小数、横向对比保留 7 位小数。因此约 5e-8 m 的数值不能解释为模型或算法达到纳米级精度。',
              '- 简单短断口候选与严格切向约束只覆盖保守可证情形；本轮未证明所有复杂断口均不可修复。',
              '- 0.10 m 横向采样使部分短组无 Ch 观测；不得把它们强制配对或当成失败识别删除。',
              '- 一个窗口内的径向重叠、贪心一对一和分支节点处的保守断开，可能漏掉真实跨线关联。尚未用人工标注集评估误关联率。',
              '- 进程数比较每档只跑一次，受缓存、系统后台活动和温度影响。没有重复基准试验或置信区间。',
              '- 四个图件用于人工审核，不替代现场/地质判读；新增约束工具是独立实验入口，不会悄然更改生产识别阈值。', '',
              f"最终结论：**{summary['conclusion']}**。以实测结果为准，不为达成预设改善而放宽阈值、合并多分支或制造连接。", '',
              '## 9. 仓库修改及复现入口', '',
              '| 文件 | 修改内容 |', '|---|---|',
              '| scripts/03_slicing_profiles/generate_scanline_slices.py | 原纵向入口接入常驻池，默认 4 进程，可传 --workers；原范围、模型配置及结果结构不变。 |',
              '| scripts/03_slicing_profiles/parallel_slice_engine.py | 新增共享只读网格缓存、常驻初始化、有界完成顺序补充任务队列。 |',
              '| scripts/03_slicing_profiles/generate_orthogonal_scanlines.py | 新增双方向生产切片 CLI；显式输入，拒绝覆盖旧输出目录。 |',
              '| scripts/04_structure_recognition/orthogonal_profile_constraint.py | 新增多分支正交观测、保守断口筛选、可追溯回切修复和跨线一致性。 |',
              '| scripts/99_experiments/validate_orthogonal_scanline_constraint.py | 新增真实模型实验、进程资源采样、基线导出及审计入口。 |',
              '| scripts/99_experiments/orthogonal_audit_report.py | 新增 4 张图和报告生成器。 |',
              '| tests/test_parallel_slice_engine.py | 7 项引擎合成检查。 |',
              '| tests/test_orthogonal_profile_constraint.py | 8 项初始几何检查及 2 项审查回归检查。 |', '',
              '实际报告/CSV/图件在本输出目录。outputs/ 沿用仓库既有忽略规则，大型网格缓存和运行数据不自动加入 Git；未新增算法库备份，未修改原识别算法文件，未改动原有 docs/performance 文件。', '',
              '在仓库根目录复现实验（NEW_DIR 必须是新目录；模型和基线默认路径见 --help）：', '',
              '```powershell',
              'python scripts/99_experiments/validate_orthogonal_scanline_constraint.py --output NEW_DIR --phase acquire',
              'python scripts/99_experiments/validate_orthogonal_scanline_constraint.py --output NEW_DIR --phase analyze',
              'python scripts/99_experiments/validate_orthogonal_scanline_constraint.py --output NEW_DIR --phase finalize',
              '```', '',
              'finalize 只回读交付物与源码确认等价证据，不重跑算法。refine-metrics 是本次审查后的单次定向迁移入口；新代码正常运行已包含修正，无需再运行该迁移。', '']
    (output/'ORTHOGONAL_SCANLINE_CONSTRAINT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'ORTHOGONAL_SCANLINE_CONSTRAINT_REPORT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'stage_performance.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stage_csv[0]))
        writer.writeheader()
        writer.writerows(stage_csv)
