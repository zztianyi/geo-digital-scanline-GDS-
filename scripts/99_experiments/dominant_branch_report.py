"""Render measured dominant-branch checkpoint; never rerun numerical inference."""
import csv
import json
from pathlib import Path
import pickle
import time
import numpy as np
from validate_orthogonal_scanline_constraint import StageSampler, save_csv, save_json


def _plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.grid': True, 'grid.alpha': .16, 'axes.formatter.useoffset': False})
    return plt


def render_case(data, row, output):
    plt = _plot_setup()
    cid = row['candidate_id']
    result = data['results'][cid]
    case = next(c for c in data['cases'] if c['candidate_id'] == cid)
    old = data['old_natural_solutions'][cid]
    chosen = result['selection']['selected']
    low, high = np.asarray(case['lower_uz']), np.asarray(case['upper_uz'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharex=True, sharey=True)
    curves = [np.asarray(old['curve_uz']), np.asarray([low, high])]
    if len(result['curve_uz']):
        curves.append(result['curve_uz'])
    context = []
    if '_plot_profiles' in data and row['status'] != 'PRESERVED_COMPLETE_OBSERVED':
        from dominant_observed_branch import extract_observed_branches, clip_branch
        from locc_data import opc
        profile = data['_plot_profiles'][case['slice_key']]
        uz = opc.to_suz(profile.nodes, data['_plot_arc'])[:, 1:]
        for branch in extract_observed_branches(profile, uz):
            for fragment in clip_branch(branch, low[1], high[1]):
                points = fragment['points_uz']
                if min(abs(points[0, 0]-low[0]), abs(points[-1, 0]-high[0])) <= .5:
                    context.append(points)
        curves.extend(context)
    us = np.concatenate(curves)[:, 0]
    upad = max(.02, np.ptp(us)*.15)
    zpad = max(.01, (high[1]-low[1])*.10)
    bounds = (us.min()-upad, us.max()+upad, low[1]-zpad, high[1]+zpad)
    for ax in axes:
        for points in context:
            ax.plot(points[:, 0], points[:, 1], c='#b0b5bc', lw=1., alpha=.9)
        for candidate in result['selection']['candidates']:
            points = candidate['fragment']['points_uz']
            ax.plot(points[:, 0], points[:, 1], c='#a4aab1', lw=1., alpha=.8)
        ax.scatter([low[0], high[0]], [low[1], high[1]], c='#252525', s=28, zorder=7)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='Radial offset u (m)')
        ax.ticklabel_format(style='plain', useOffset=False)
    old_curve = np.asarray(old['curve_uz'])
    axes[0].plot(old_curve[:, 0], old_curve[:, 1], c='#bd703b', lw=2., label='Old LOCC proposal')
    axes[0].set_title('Old whole-window proposal')
    axes[0].set_ylabel('Elevation z (m)')
    for edge in result['path_edges']:
        points = np.asarray(edge['points_uz'])
        observed = edge['source'].startswith('OBSERVED')
        color = '#196eb3' if observed else '#d3a413' if edge['source'] == 'TOPOLOGY_SWITCH' else '#d12e43'
        axes[1].plot(points[:, 0], points[:, 1], c=color, lw=2.6 if observed else 2.,
                     ls='-' if observed or edge['source'] == 'TOPOLOGY_SWITCH' else '--', zorder=4)
    for layer in data['windows'][cid]['layers']:
        hits = [h['u'] for h in layer['horizontal']]
        axes[1].scatter(hits, [layer['z']]*len(hits), c='#8dcda4', marker='x', s=24, zorder=6)
    junction = result['junction']
    if junction:
        pp = np.array([junction['a_point_uz'], junction['b_point_uz']])
        axes[1].scatter(pp[:, 0], pp[:, 1], c='#8846bb', s=28, zorder=8)
    axes[1].set_title(f"Branch {row['dominant_branch_id']} | Z coverage {row['observed_coverage']:.1%}\n"
                      f"Output observed fraction {row['observed_geometry_fraction']:.1%}")
    fig.suptitle(f"{row['old_category']} / {cid} / s={case['s']:.2f} m", y=.98, fontsize=14)
    fig.text(.5, .915, f"{row['status']} | switches={row['branch_switch_count']} | inferred={row['inferred_length_m']*1000:.2f} mm",
             ha='center', fontsize=10)
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color='#196eb3', lw=2.6, label='Selected observed branch'),
              Line2D([], [], color='#a4aab1', label='Secondary observed branch'),
              Line2D([], [], color='#d3a413', label='Topology connector'),
              Line2D([], [], color='#d12e43', ls='--', label='Inferred geometry'),
              Line2D([], [], color='#8dcda4', marker='x', ls='', label='Raw H observations (legacy)'),
              Line2D([], [], color='#8846bb', marker='o', ls='', label='Virtual junction')]
    fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.10, right=.98, top=.84, bottom=.18, wspace=.13)
    path = output/'figures'/row['old_category']/f'{cid}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_closed(data, output):
    plt = _plot_setup()
    by_key = {r['component_key']: r for r in data['loops']}
    tracks = sorted(data['closed']['tracks'], key=lambda t: (
        t['status'] == 'stable', t['slice_count'], max(by_key[k]['area_2d'] for k in t['component_keys'])), reverse=True)
    selected = tracks[:3]
    figures = []
    for i, track in enumerate(selected):
        fig, ax = plt.subplots(figsize=(7.2, 6.))
        members = track['component_keys'][:5]
        for key in members:
            record = by_key[key]
            points = np.asarray(record['ordered_loop_uz'])
            ax.plot(points[:, 0], points[:, 1], marker='.' if len(points) < 5 else None,
                    label=f"s={record['s']:.2f} m / C{record['component_id']}")
        ax.set(xlabel='Radial offset u (m)', ylabel='Elevation z (m)',
               title=f"Closed track {i+1}: {track['status']} / {track['slice_count']} slices")
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = output/'figures/closed_components'/f'track_{i+1}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160); plt.close(fig)
        figures.append(dict(path=path, track=track, members=members))
    return figures


def _fmt(value, digits=3):
    return 'n.a.' if value is None else f'{value:.{digits}f}'


def _table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str, row))+'|' for row in rows])


def write_deviations(data, output):
    s = data['summary']
    items = [
      ('实测反馈：不按中部偏移排除真实分支', '利用完整observed分支和最少干预保留真实折返。',
       '候选关联改为局部两端分别距离旧端点U不超过0.25m，不再限制中部median U。此半径只筛身份候选，不移动/吸附/连接节点。',
       '首轮177例完整保留。11个未选案例仍有完整开放分支，N00208与N00213有4次/3次真实Z折返；中部median筛选误排了其真实细节。',
       f"仅重算193例自然决策与依赖配对，完整保留改为{s['statuses'].get('PRESERVED_COMPLETE_OBSERVED', 0)}例。盘点和300遮挡未重复，前后记录见refinement_audit.json。",
       '固定保真目标更一致；未证明0.25m为最优身份半径。边界偏移较大者仍需人工核定，原悬空判据不变。'),
      ('区分分支保留与旧端点修复', '联合endpoint/branch matching恢复局部路径。',
       '完整主导分支可直接输出为保留产品，但不将旧端点强行吸附；原端点是否连通、联合配对、分支歧义分列。',
       '上一轮189个有观测案例仍因端点不属于同一component而未解；真实路径存在不能证明旧端点正确。',
       f"193例中完整保留{s['statuses'].get('PRESERVED_COMPLETE_OBSERVED', 0)}例，原端点真实连通{s['old_endpoint_pair_connected']}例。",
       '保留几何不改变识别法向/40°/70°；身份与端点仍需人工审核。'),
      ('甜区与细节描述', '连续甜区分数、邻纵细节重复、canonical密度联合判断。',
       '弧长4t(1-t)按0.1分档后进入词典序；41个等高采样做线性去趋势形态相关及幅差，4邻纵各固定一个单调段用于描述。原输出折返不排序、不平滑。',
       '连续浮点甜区差会在词典序中压过后续所有证据；节点密度也可能仅来自三角网细分。多值u(z)不能直接排序插值。',
       f"{s['repeated_detail_cases']}例的选中分支至少有一条邻线重复细节；保真行为测试与193例原边审计。尚无独立sweet-zone正确率真值。",
       '描述器坐标仅用于评分，不回写真实几何。需审核所选身份，不能声称甜区先验已被证明。'),
      ('回溯连接保守输出', '真实交点、最近点、内部投影、端点连接，并允许virtual junction。',
       '默认回溯0.2m，connector上限0.01m；每个局部结果最多两个观测身份、一次切换。两条近邻边各留独立投影点，不平均成一个点。',
       'UZ交点不一定是round6 XYZ中完全重合的点；近点连接也没有独立岩块归属真值。',
       f"记录{s['diagnostic_junctions']}个连接候选；输出路径中{s['selected_routes_switch_count']}次切换、{s['route_virtual_junctions']}个virtual node。候选均未自动应用生产。",
       '不改原边。仅新增无FaceID connector提案；是否应切换需要人工审核。'),
      ('残缺观测保持unresolved', 'LOCC只推断真正不存在observed的子区间。',
       '本轮对有部分观测但无法用至多一次连接贯通的窗口保留partial并明确unresolved；不依据估计走廊空洞自动补线。整个窗口无观测时使用D缺口求解器。',
       '上一轮Type II的±30mm H线性走廊会把弯曲真实纵向路径误判为缺口，不能将走廊空洞等同物理空洞。',
       f"partial unresolved {s['statuses'].get('PRESERVED_PARTIAL_UNRESOLVED', 0)}例；新自然提案推断长度{s['inferred_length_m']:.6f}m。",
       '固定保真目标不变，但部分缺口自动完成能力本轮未实现；需下一步审核端点/branch identity。'),
      ('闭合track匹配', 'one-to-one允许unmatched，H与邻纵联合判定。',
       '相邻0.05m、面积/周长/重叠/居中形态门限，双向最优且有分差才连；至少3slice且每条边有同level同Hbranch支持才标stable。退化loop保留但不正向跟踪。',
       '局部loop无H覆盖、tiny loop和并列候选不能当成独立岩块。闭环内点不存在自然端点甜区。',
       f"{s['closed']['total_loops']}个闭环，{s['closed']['nondegenerate_loops']}个非退化，{s['closed']['stable_tracks']}条stable track。",
       'stable仅指剖面轮廓连续证据，不是已确认岩块；无后壁/loft/体积。门限为保守固定先验。'),
      ('对照复用与完整遮挡', '至少A–E五方法，包含观测比例、switch、virtual junction等指标。',
       'A–D冻结复用，E独立重算300次；全高度遮挡清除了所有目标V分支，E合法回退到同一个D求解器。A–C无跨高度V分支身份，switch报n.a.而非伪造0。',
       '本样本不能同时测到目标V分支保留收益；H branch ID只在同一高度有效，不能跨高度数ID变化当branch switch。',
       f"E与D曲线逐元素相同{s['masks_equal_d']}/300；保留收益看193自然案例。相同样本是回归而非新盲测。",
       '不隐藏误差未改善；性能只报告本轮独立执行，不能把复用对照计作本轮计算。'),
      ('实验启用与源码入口', 'closed subset进入分析而非直接丢弃。',
       '实验入口对全部2800个正向剖面执行closed分析。原ArcSlicer增加preserve_closed_components=True侧路；默认开放识别输出保持兼容。主导分支不自动替换生产识别。',
       '分支身份/closest connector还需图审；直接改变默认可能把未经确认的component当岩块。',
       '最小集成测试对比闭合sidecar开启前后开放red_groups_corrected，闭环几何独立保存。',
       '原法向阈值、40°/70°等完全未变。生产默认启用留待用户确认；不提交/推送。'),
    ]
    lines = ['# 实现调整说明', '', '以下均为本轮实现选择或保守边界，未把未验证的情况隐去。', '']
    for title, original, actual, why, evidence, impact in items:
        lines += [f'## {title}', '', f'原计划：{original}', '', f'实际实现：{actual}', '',
                  f'为什么修改／遇到的验证问题：{why}', '',
                  '使用了哪些本地数据：冻结2800纵向剖面、同模型横向测线、193自然案例与150×2遮挡对照。', '',
                  f'修改后的结果：{evidence}', '', f'固定目标、原识别语义及人工审核影响：{impact}', '']
    (output/'IMPLEMENTATION_DEVIATIONS.md').write_text('\n'.join(lines), encoding='utf-8')


def finish(output):
    output = Path(output).resolve()
    with (output/'validation_checkpoint.pkl').open('rb') as stream:
        data = pickle.load(stream)
    s = data['summary']; stages = data['stages'].copy()
    row_map = {r['candidate_id']: r for r in data['rows']}
    # Reuse the previous round's exact 29 review IDs, rather than selecting
    # favorable cases after seeing the new results.
    previous_path = next(Path(p).parent for p in data['manifest']['source_hashes'] if p.endswith('validation_checkpoint.pkl'))
    with (previous_path/'validation_checkpoint.pkl').open('rb') as stream:
        plot_source = pickle.load(stream)
    data['_plot_profiles'], data['_plot_arc'] = plot_source['target_profiles'], plot_source['arc']
    with (previous_path/'representative_figures.csv').open(encoding='utf-8-sig', newline='') as stream:
        representative_rows = list(csv.DictReader(stream))
    figures, review = [], ['# 代表案例审核', '', '沿用上一轮29个案例：A/B/C/D/F各5个，E全部4个。图中几何来自本轮输出，黑点为旧候选端点。', '',
                          '粗蓝线是保留的主导纵向观测；灰线是其他候选；浅绿叉为原始横向观测（legacy，未证明同曲面）；黄线为connector；紫点为virtual junction；红虚线只表示推断。', '']
    with StageSampler('representative_figures_and_report', output, stages):
        for old in representative_rows:
            row = row_map[old['candidate_id']]
            cid = row['candidate_id']; result = data['results'][cid]
            path = render_case(data, row, output)
            figures.append(dict(category=row['old_category'], candidate_id=cid, path=path.as_posix()))
            selected = result['selection']['selected']
            diagnostic = [r for r in data['junctions'] if r['candidate_id'] == cid]
            junction = result['junction']
            if selected is not None:
                selection_text = selected['reason']
            elif row['inferred_length_m'] > 0:
                selection_text = '无满足筛选条件的目标纵向分支；沿用D的低置信度缺口推断，尚未采纳'
            else:
                selection_text = '没有满足边界关联条件的目标纵向分支；保持未解，不代表原始观测不存在'
            review += [f"## {row['old_category']} · {cid}", '', f'![{cid}]({path.as_posix()})', '',
              f"选择的dominant branch：{row['dominant_branch_id']}；observed覆盖：{row['observed_coverage']:.2%}；路径观测使用比例：{row['observed_geometry_fraction']:.2%}。", '',
              f"甜区：{_fmt(row['interior_score'])}；canonical细节密度：{_fmt(row['detail_density'])} edge/m；邻纵重复细节：{row['neighbor_detail_repeat_count']}/4，评分{_fmt(row['neighbor_detail_score'])}；H支持率：{_fmt(row['horizontal_support'])}。", '',
              f"分支切换：{row['branch_switch_count']}；junction：{junction['junction_type'] if junction else '无输出路径连接'}；回溯距离：{junction['A_backtrack_length_m'] if junction else 0:.6f}m；推断长度：{row['inferred_length_m']:.6f}m。另存诊断连接候选{len(diagnostic)}个，未自动应用。", '',
              f"选择原因：{selection_text}。其他分支可能未满足边界关联/完整性条件，或证据排序靠后；保留原始几何而不平均，详细指标见branch_reliability.csv。", '',
              f"审核状态：{row['status']}；branch歧义={row['branch_selection_ambiguous']}；联合配对歧义={row['pairing_ambiguous']}；旧端点连接已成立={row['original_endpoint_pair_connected']}。", '']
        closed_figures = render_closed(data, output)
        for item in closed_figures:
            review += ['## 闭合分量代表', '', f"![closed track]({item['path'].as_posix()})", '',
                       f"看到：{item['track']['slice_count']}个相邻切面轮廓，状态{item['track']['status']}。", '',
                       '处理：保留原loop，无人工后壁；仅用形态和同高度H分支进行track。是否为独立岩块仍需人工核定。', '']
        (output/'REPRESENTATIVE_REVIEW.md').write_text('\n'.join(review), encoding='utf-8')
        save_csv(output/'representative_figures.csv', figures)
        write_deviations(data, output)
    s['numeric_and_figures_seconds'] = s['numeric_wall_seconds']+stages[-1]['wall_seconds']
    s['peak_rss_gib'] = max(st['peak_tree_rss_bytes'] for st in stages)/2**30
    s['peak_private_gib'] = max(st['peak_tree_private_bytes'] for st in stages)/2**30
    s['statuses_review'] = dict(DOMINANT_BRANCH_SELECTION='PARTIALLY_SUPPORTED',
        BACKTRACKING_JUNCTION='PARTIALLY_SUPPORTED' if s['diagnostic_junctions'] else 'NOT_SUPPORTED',
        MINIMUM_GEOMETRY_INTERVENTION='SUPPORTED' if s['observed_edges_checked'] else 'PARTIALLY_SUPPORTED',
        CLOSED_COMPONENT_HANDLING='PARTIALLY_SUPPORTED')
    total_stages = sum(st['wall_seconds'] for st in stages)
    perf = [{**st, 'stage_share_percent': 100*st['wall_seconds']/total_stages,
             'peak_rss_gib': st['peak_tree_rss_bytes']/2**30,
             'peak_private_gib': st['peak_tree_private_bytes']/2**30} for st in stages]
    save_csv(output/'stage_performance.csv', perf)
    save_json(output/'summary.json', s)
    headline = f"本轮完成了主导观测分支保留与闭合分量跟踪实验。193例中，{s['statuses'].get('PRESERVED_COMPLETE_OBSERVED', 0)}例找到了可完整保留的纵向分支，但这不等于原端点已经修复；原端点实际连通{s['old_endpoint_pair_connected']}例。"
    lines = ['# 本轮审核报告', '', headline, '',
      f"完整遮挡测试E与D逐点相同{s['masks_equal_d']}/300，未证明遮挡误差改善。新功能仍作为可审核的实验输出，未默认替换生产路径。", '',
      '边界关联修正增加11例完整保留，同时N00014降为关联未解、N00918降为部分保留，净增9例。没有为增加通过数继续放宽边界半径。', '',
      _table(['项目', '本轮实测'], [
        ('剖面／branch', f"{s['slice_count']} / {s['branch_count']}"),
        ('闭合／非退化／stable track', f"{s['closed']['total_loops']} / {s['closed']['nondegenerate_loops']} / {s['closed']['stable_tracks']}"),
        ('主导决策／分支歧义', f"{s['decisions']} / {s['branch_ambiguous']}"),
        ('联合一对一选中／配对歧义', f"{s['joint_selected']} / {s['joint_ambiguous']}"),
        ('输出switch／virtual junction', f"{s['selected_routes_switch_count']} / {s['route_virtual_junctions']}"),
        ('诊断junction候选（非已应用）', s['diagnostic_junctions']),
        ('自然提案推断长度 D → E (m)', f"{s['old_inferred_length_m']:.6f} → {s['inferred_length_m']:.6f}"),
        ('193例路径观测占比均值', f"{s['mean_observed_geometry_fraction']:.2%}"),
        ('原边几何/来源审计', f"{s['observed_edges_checked']}条通过；输入未修改"),
        ('至少一邻纵重复细节', s['repeated_detail_cases']),
        ('数值运行／含图与报告准备 (s)', f"{s['numeric_wall_seconds']:.3f} / {s['numeric_and_figures_seconds']:.3f}"),
        ('进程树峰值RSS／private commit (GiB)', f"{s['peak_rss_gib']:.3f} / {s['peak_private_gib']:.3f}")]), '',
      '观测占比为每个提案观测弧长/总弧长后取均值；未解空路径记0，因此不是已确认岩块面积比例。自然候选存在重复窗口，总长度按候选相加，不代表全模型唯一几何长度。', '',
      '## A–F 分类', '',
      _table(['类', '例数', '完整保留', '部分未解', '有推断', '分支或配对歧义'],
             [(r['category'], r['count'], r['complete_observed'], r['partial_unresolved'], r['inferred'], r['ambiguous']) for r in s['category_summary']]), '',
      '## 五方法遮挡对照', '',
      '同150处位置、每处两种H条件；以下是68处holdout回归。A–D冻结复用，E本轮重算。P95列是各案例P95误差的中位数，尾部列是各案例P95的95分位，均为mm。', '']
    holdout = [r for r in data['masking']['summary'] if r['split'] == 'holdout']
    names = {'VERTICAL_ONLY': 'A Vertical', 'HORIZONTAL_ONLY': 'B Horizontal', 'LOCC': 'C Old LOCC',
             'OBSERVED_FIRST_LOCC': 'D Observed-first', 'DOMINANT_BRANCH_MINIMUM_INTERVENTION': 'E Dominant'}
    lines += [_table(['H条件', '方法', '中位案例P95', '尾部P95', '最大误差', 'Hausdorff中位', '分支准确率'],
        [('完整' if r['regime'] == 'INTACT_HORIZONTAL' else '缺失', names[r['method']],
          _fmt(r.get('median_case_p95_u_m', 0)*1000), _fmt(r.get('p95_case_p95_u_m', 0)*1000),
          _fmt(r.get('max_error_m', 0)*1000), _fmt(r.get('median_hausdorff_m', 0)*1000),
          'n.a.' if r.get('branch_accuracy') is None else f"{r['branch_accuracy']:.2%}") for r in holdout]), '',
      '全高度遮挡区的目标V观测比例五方法都为0；E不凭空恢复observed。A–C没有跨高度目标V branch身份，switch记n.a.；不能跨H层比较branch ID来伪造切换次数。', '',
      '## 真实资源占用', '',
      _table(['阶段', '秒', '阶段占比', '平均逻辑核', '整机CPU%', 'RSS峰值GiB', 'private GiB'],
        [(r['stage'], _fmt(r['wall_seconds']), f"{r['stage_share_percent']:.1f}%", _fmt(r['mean_logical_cpu_equivalents'], 2),
          _fmt(r['mean_machine_cpu_percent'], 2), _fmt(r['peak_rss_gib']), _fmt(r['peak_private_gib'])) for r in perf]), '',
      f"单进程、数值线程1、无GPU；本机{data['manifest']['logical_cpus']}逻辑CPU、RAM {data['manifest']['ram_gib']:.2f}GiB。0.25s采样；RSS为驻留物理页，private为私有提交量而非驻留量。峰值不应相加，整机CPU%按逻辑CPU归一化。", '',
      '范围包括输入加载、2800剖面branch/loop盘点、初轮193决策、闭合track、300次E遮挡、193定点修正重算与图。总数值时间含这次定点修正成本，不能当作最终算法单次直接运行基准；各阶段已分别计时。不是从模型切割到块体体积的全流程，也不是先前58分钟估算的替代测量。一次采样无法证明跨次稳定性。', '',
      '## 局部代表', '']
    for category in 'ABCDEF':
        figure = next(r for r in figures if r['category'] == category)
        row = row_map[figure['candidate_id']]
        if row['status'] == 'PRESERVED_COMPLETE_OBSERVED':
            action = '保留完整observed分支，原分支细节不平均、不平滑。'
            verdict = '观测几何保留符合；真实身份仍需图审'
        elif row['inferred_length_m'] > 0:
            action = '沿用D的低置信度缺口推断，仅作为待审核提案；不称为真实观测或已修复。'
            verdict = '尚未证明修复有效，不能据此认定几何保真或误差改善'
        else:
            action = '仅保留有依据的局部观测，缺失部分继续未解；不为贯通而新增几何。'
            verdict = '局部保留符合最少干预，但完整修复未达成'
        lines += [f"### {category} · {row['candidate_id']}", '', f"![{row['candidate_id']}]({figure['path']})", '',
          f"看到：{row['status']}，Z覆盖{row['observed_coverage']:.2%}，输出路径观测比例{row['observed_geometry_fraction']:.1%}，推断{row['inferred_length_m']:.6f}m。", '',
          f'处理：{action}', '',
          f"是否符合预期：{verdict}；旧端点修复={row['original_endpoint_pair_connected']}，分支歧义={row['branch_selection_ambiguous']}。", '']
    if closed_figures:
        item = closed_figures[0]
        lines += ['### 闭合分量', '', f"![closed]({item['path'].as_posix()})", '',
          f"看到：{item['track']['slice_count']}个切面，track状态{item['track']['status']}。", '',
          '处理：保留loop并关联邻slice/H，不造后壁，不算体积。', '', '是否符合预期：闭环保留符合；是否为独立岩块未确认。', '']
    lines += ['## 四项结论', '', _table(['项目', '结论'], list(s['statuses_review'].items())), '',
      'SUPPORTED仅指已执行的几何保真约束，不表示所有候选已正确修复。甜区先验、邻纵细节身份和junction改善没有独立人工真值，因此不给全面有效的结论。', '',
      '## 已解决', '',
      '- 完整observed分支不再因旧端点配错而被迫生成替代曲线；逐边保留来源。',
      '- branch选择引入甜区与邻纵重复细节；输出结构最多一次切换，原折返几何不平滑。',
      '- 闭合分量及退化闭环显式保存、跟踪，源码提供opt-in识别侧路。', '',
      '## 未解决', '',
      '- 主导身份尚无独立真值；保留另一条真实branch不等于原endpoint pair已修复。',
      '- 部分观测无法贯通者继续unresolved；最近点connector仍是待审核提案。',
      '- 遮挡误差没有改善；全遮挡样本无法衡量已有V分支保留收益。',
      '- closed track不是独立块体认证；没有运行全量识别→loft→体积。', '',
      '## 实现偏离', '',
      '详见 [IMPLEMENTATION_DEVIATIONS.md](IMPLEMENTATION_DEVIATIONS.md)：甜区分档、描述器、多值折返、回溯界限、部分未解、对照复用与实验启用范围均已说明。', '',
      '## 下一轮建议／人工验证', '',
      '先审核29个相同代表案例的dominant身份与闭合track，优先确认部分未解和shared endpoint案例。需确认最近点connector是否有真实fork/merge依据，再考虑默认启用。', '',
      f"Git branch：{data['manifest']['branch']}；HEAD：{data['manifest']['head']}。未创建新commit，未push。修改文件与最低检查记录见[EXECUTION_NOTES.md](../EXECUTION_NOTES.md)。", '',
      '[全部代表案例](REPRESENTATIVE_REVIEW.md)；[193决策CSV](dominant_branch_decisions.csv)；[五方法逐例CSV](masking_method_comparison.csv)；[资源CSV](stage_performance.csv)。', '']
    (output/'FINAL_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(output/'workbook_payload.json', dict(summary=s, stages=perf, masking=holdout,
        categories=s['category_summary'], decisions=data['rows'], manifest=data['manifest']))
    print(f'Reports rendered: {len(figures)} A-F figures, {len(closed_figures)} closed figures', flush=True)
