"""Render measured branch-first audit, keeping raw and linked H distinct."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from surface_track_local_review import (local_view_bounds, clipped_segments, vertical_coverage,
    prepare_local_review, write_local_review)


def render_case(data, cid, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    plot = data['local_plot_cases'][cid]
    row, bounds = plot['row'], plot['bounds']
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    def lines(ax, segments, color, width=1., alpha=1., style='solid', order=2):
        if len(segments):
            ax.add_collection(LineCollection(segments, colors=color, linewidths=width,
                alpha=alpha, linestyles=style, zorder=order))
    for ax in axes:
        for bid, segments in plot['background']:
            lines(ax, segments, '#aeb5bd', 1.3, .85)
        endpoints = plot['endpoints']
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=22, color='#252525', alpha=.65, zorder=7)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='Radial offset u (m)')
        ax.ticklabel_format(style='plain', useOffset=False)
        ax.grid(alpha=.15)
    for kind, segments in plot['old_paths'].items():
        lines(axes[0], segments, {'observed':'#b77541', 'switch':'#dfa814', 'inferred':'#d12e43'}[kind],
              2.8, style='dashed' if kind == 'inferred' else 'solid', order=4)
    old_title = ('Previous inference proposal (unaccepted)' if plot['old_status'] == 'INFERRED_REVIEW_PROPOSAL'
                 else f"Previous dominant branch {plot['old_branch']}")
    axes[0].set_title(f"{old_title}\nOutput / proposal Z coverage: {row['previous_local_z_coverage']:.1%}", fontsize=11)
    axes[0].set_ylabel('Elevation z (m)')
    lines(axes[1], plot['neighbors'], '#93c7e8', 1.3, .75, order=3)
    for kind, segments in plot['paths'].items():
        lines(axes[1], segments, {'observed':'#1865b1', 'switch':'#dfa814', 'inferred':'#d12e43'}[kind],
              2.8, style='dashed' if kind == 'inferred' else 'solid', order=4)
    for points, color, marker, size in ((plot['raw_H'], '#a4dcb3', 'x', 28), (plot['selected_H'], '#146b3a', 'o', 18)):
        axes[1].scatter(points[:, 0], points[:, 1], color=color, marker=marker, s=size, zorder=6)
    points = plot['junction_points']
    axes[1].scatter(points[:, 0], points[:, 1], color='#8b43ba', s=32, zorder=7)
    crossover = plot['crossover_interval']
    if crossover and max(crossover[0], bounds[2]) < min(crossover[1], bounds[3]):
        axes[1].axhspan(max(crossover[0], bounds[2]), min(crossover[1], bounds[3]), color='#e4bd57', alpha=.12)
    axes[1].set_title(f"Current {row['selected_track'] or 'unresolved'} / branch {row['selected_branch']}\nLocal Z coverage: {row['local_selected_z_coverage']:.1%} | ambiguous: {row['ambiguous']}", fontsize=11)
    note = ('SELECTED TRACK ABSENT IN THIS LOCAL VIEW' if row['selected_absent_from_local_view'] else
            'TOUCHES VIEW ONLY; NO CASE-Z COVERAGE' if row['local_selected_z_coverage'] == 0 else
            'PARTIAL LOCAL COVERAGE' if row['local_selected_z_coverage'] < .95 else
            'COVERED LOCALLY; TRACK IDENTITY AMBIGUOUS' if row['ambiguous'] else 'OBSERVED LOCAL COVERAGE; IDENTITY NEEDS REVIEW')
    axes[1].text(.02, .98, note, transform=axes[1].transAxes, va='top', fontsize=9,
                 color='#922e35', bbox=dict(facecolor='white', alpha=.92, edgecolor='#dbb8bb', pad=5), zorder=9)
    fig.suptitle(f"{row['old_category']} / {cid} / s={float(row['slice_key']):.2f} m | LOCAL DETAIL", fontsize=14, y=.98)
    fig.text(.5, .91, f"{row['status']} | raw H: {row['raw_H_in_view']} | selected-track linked H: {row['selected_track_linked_H_in_view']}",
             ha='center', fontsize=10)
    legend = [Line2D([], [], c='#1865b1', lw=2.5, label='Selected observed track'),
        Line2D([], [], c='#b77541', lw=2.5, label='Previous observed output'),
        Line2D([], [], c='#93c7e8', label='Same-track neighbor'), Line2D([], [], c='#aeb5bd', label='Other observed'),
        Line2D([], [], c='#a4dcb3', marker='x', ls='', label='Raw H observations'),
        Line2D([], [], c='#146b3a', marker='o', ls='', label='Surface-linked H'),
        Line2D([], [], c='#dfa814', label='Topology switch'),
        Line2D([], [], c='#252525', marker='o', ls='', label='Historical case endpoints'),
        Line2D([], [], c='#8b43ba', marker='o', ls='', label='Virtual junction'),
        Line2D([], [], c='#d12e43', ls='--', label='Inferred only')]
    fig.legend(handles=legend, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .105, f"Same local axes | window {bounds[1]-bounds[0]:.3f} m (u) x {bounds[3]-bounds[2]:.3f} m (z) | ROI is display-only", ha='center', fontsize=9)
    fig.subplots_adjust(top=.81, bottom=.20, left=.08, right=.98, wspace=.13)
    path = output/'figures'/f'{cid}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def render_reports(data, output):
    prepare_local_review(data, output)
    summary = data['summary']
    review = ['# 代表案例审核', '',
        '固定复用上一轮29个代表案例，左为上一轮实测结果，右为本轮结果。左右使用同尺度局部放大，窗口只按原案例及旧局部几何确定，不再被完整 branch 范围扩大。详见 LOCAL_FAILURE_REVIEW.md 的局部问题统计。不同旧案例如果属于同一 slice，将引用同一个 surface-first 决策；旧案例不是新的配对单位。', '',
        '完整观测保留不等于已证明该曲面就是人工意图中的目标。请人工核对右图 track；歧义轨道不宣称通过验收。浅绿叉仅为 raw H，深绿点才是同路径相邻剖面支持。黑色旧端点已弱化。', '']
    for item in data['representatives']:
        cid = item['candidate_id']
        path = render_case(data, cid, output)
        result = data['results'][cid]
        selected = result['selection']['selected']
        j = result['junction']
        review += [f"## {item['old_category']} · {cid}", '', f'![{cid}]({path.as_posix()})', '',
            f"Selected surface_track_id：{result['surface_track_id']}；Selected target branch_id：{selected['branch_id'] if selected else None}。", '',
            f"同一 surface 的依据：完整 H monotone connected run 在相邻 V edge 上实际相交，并有连续多层支持；raw proximity 不计入支持。",
            f"same-H-path 连续层数：{selected['continuous_H_support_run'] if selected else 0}；同 track 相邻细节重复：{selected['neighbor_detail_repeat_count'] if selected else 0}。",
            f"full-branch interior：{selected['interior_score'] if selected else None}；稳定：{selected['track_stable'] if selected else False}；歧义：{result['selection']['ambiguous']}。",
            'legacy endpoint 影响 identity：No。', '',
            f"switch：{result['branch_switch_count']}；位置：{j['a_point_uz'] if j else '无'}；junction 原因：{'同 track 置信度交叉区内的最小连接' if j else '未检测到可接受的同 track 交叉换轨'}。",
            f"低可靠尾部移除长度：{result['observed_low_confidence_tail_removed']:.6f}m；synthetic connector：{result['connector_length_m']:.6f}m；inferred：{result['inferred_length_m']:.6f}m。", '',
            '实际几何见图；是否修正上一轮人工指出的错误仍需人工逐图确认，不能用坐标未移动代替身份正确性证明。', '']
    (output/'REPRESENTATIVE_REVIEW.md').write_text('\n'.join(review), encoding='utf-8')
    write_local_review(data, output, render_case)
    statuses = dict(ENDPOINT_INDEPENDENCE='SUPPORTED', HORIZONTAL_SURFACE_IDENTITY='PARTIALLY_SUPPORTED',
        SURFACE_TRACK_SELECTION='PARTIALLY_SUPPORTED', CONFIDENCE_CROSSOVER_HANDOFF='PARTIALLY_SUPPORTED',
        OBSERVED_COORDINATE_PRESERVATION='SUPPORTED', SURFACE_IDENTITY_PRESERVATION='PARTIALLY_SUPPORTED',
        MINIMUM_GEOMETRY_INTERVENTION='PARTIALLY_SUPPORTED', CLOSED_COMPONENT_HANDLING='PARTIALLY_SUPPORTED')
    summary['conclusions'] = statuses
    final = ['# 本轮验收报告', '',
        '代码已切换为完整 branch → H surface link → surface graph → track selection → confidence crossover → missing-only inference。结论仅覆盖本轮检查，不推定未经标注的真实案例身份正确。', '',
        f"Git：{summary['branch']} / {summary['head']}；本轮未提交，commit SHA：无。", '',
        '## 固定结论', '', *[f'- {k}: {v}' for k, v in statuses.items()], '',
        '## 实测指标', '', '计数按唯一目标 slice 统计，193 个历史案例仅作对照索引，避免重复累计同一条 route。', '',
        '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
        '## 16项验收', '',
        '1. 旧 endpoint 参与 identity：否；构图及选择接口不读取端点坐标。',
        '2. old gap Z window 参与 branch identity：否；完整 branch 建档，旧 ROI 只用于历史对照。',
        '3. spanning 覆盖 selection：否；该覆盖分支已移除。',
        '4. raw H proximity 被叫作 Real Horizontal Support：否；绘图分开标注。',
        '5. H 作为 same-surface evidence：是；相邻剖面、同 run、实际 crossing、连续多层；稀疏层不足时不宣称已识别。',
        '6. 邻纵 detail：仅同 track 且直接相邻链接；保留完整折返 descriptor。',
        '7. simple-vs-detailed：针对性用例验证有支持的折返 branch 保留，真实目标正确性见图人工验收。',
        f"8. endpoint perturbation：{summary['endpoint_perturbation_identity_change_count']}/{summary['endpoint_perturbation_count']} 身份变化。",
        f"9. backtracking 进入最终 route：代码已接入；本轮自然 route switch {summary['branch_switch_count']}，合成回归另检验非零实际换轨。",
        '10. junction：只允许在 confidence crossover 区间中；折返的多值 transition 暂保留 unresolved。',
        f"11. A→B→A：{summary['A_B_A_count']}；route 结构限制最多一次换轨。",
        '12. observed 坐标平均：无；所有输出 observed edge 均检查原边参数与 provenance。',
        '13. 稳定 track 错误替换：已锁定身份时禁止换到其他 track；真实数据无独立身份真值，不填0冒充已验证。',
        '14. inference：只补有同路径双侧身份支持且目标全部 observed branches 均缺失的尾区；不明身份保持 unresolved。',
        f"15. closed components：保留为 {summary['closed_tracks']} 个独立 closed tracks；只记录与 open 的 transition，不自动合并、不造背壁。",
        '16. 上轮错误案例：29张前后图已输出；是否满足人工指定的目标仍需逐图确认。', '',
        '## 检查与边界', '',
        '一次直接相关回归文件检查及本轮 surface 数据审计；未执行全项目测试、全项目构建、旧全高遮挡实验或识别/体素重算。', '',
        '高置信度保留长度按连续同路径 H 支持层之间的 observed edge 长度统计，是算法证据量，不等于独立身份真值。graph build 与 track extraction、decision 与 junction search 共用阶段计时，避免重复执行计时。', '',
        '[逐项纠错](FAILURE_CORRECTION_REPORT.md) · [实现偏差](IMPLEMENTATION_DEVIATIONS.md) · [代表案例前后图](REPRESENTATIVE_REVIEW.md)', '']
    (output/'FINAL_REVIEW.md').write_text('\n'.join(final), encoding='utf-8')
    problems = [
        ('旧端点筛选branch', '移除radial pad与端点窗口候选筛选；surface_track_selection只读graph', 'endpoint_perturbation_cannot_change_identity_or_route', '源码修正；实际身份正确性仍待人工'),
        ('完整branch先被clip', 'reliability改为完整branch；弧长descriptor记录到两端距离与曲率', 'complete_observed_branch_is_not_averaged_or_inferred', '源码修正'),
        ('spanning推翻dominant', '删除spanning override；重写保护错误行为的旧测试', 'same_path_support_preserves_fold_over_simple_spanning_branch', '源码修正'),
        ('H proximity冒充同surface', 'canonical H run + 双端实际crossing + 多层连续；raw与linked分别计数绘图', 'raw_nearby_hits_do_not_link_disconnected_horizontal_paths', '实现；稀疏/歧义支持仍需审核'),
        ('邻纵自由挑形似branch', '只比较同track直接邻接成员的完整弧长descriptor', 'shape_similarity_without_track_link_is_not_detail_support', '源码修正'),
        ('junction只是diagnostic', '主solver调用select_handoff，候选直接组成最终route', 'crossover_junction_is_interior_and_enters_final_route', '接入；自然数据非零换轨由统计决定'),
        ('换轨越晚越好', '在crossover内先按真实交点/最小连接排序，再保留支持区', 'parallel_edges_find_constrained_interior_junction', '源码修正；多值过渡保守未决'),
        ('旧endpoint pair作为最终key', '主实验按track_id+target branch输出，193 case ID仅作历史索引', 'locked_track_cannot_be_replaced_by_another_stable_track', '源码修正'),
        ('坐标未改等同最小干预', '坐标、identity、minimum intervention三项分别报告', '缺失区与身份锁定回归；真实图审', '最小干预仅PARTIALLY_SUPPORTED')]
    lines = ['# 上一轮失败点逐项纠正', '']
    for i, (problem, change, test, state) in enumerate(problems):
        lines += [f'## 0.{i+1} {problem}', '', f'源码修改：{change}。', f'对应测试：{test}。',
            f"真实案例证据：surface_track_decisions.csv、surface_track_support.csv及29张对比图；端点扰动变化 {summary['endpoint_perturbation_identity_change_count']}。",
            f'修正状态：{state}。', '']
    (output/'FAILURE_CORRECTION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    deviations = ['# 实现调整与限制', '',
        '## 执行范围', '原设计：包含全面验证与提交回传。实际：遵守用户本轮最小检查、不自动提交规则，一次集中直接相关检查；无全项目构建、无环境改动。证据：focused_tests.txt、stage_performance.csv；固定目标不变。', '',
        '## 真实H输入', '原设计：完整原始水平测线。实际：重建冻结horizontal_slices_clean中的canonical runs，禁止新毫米级snap；crossing采用3µm数值容差及XYZ核对。原因：沿用本地冻结摄影测量数据。冻结数据上一轮可能已有数值级clean，不能声称重新验证原始mesh。证据：horizontal_path_inventory.csv、surface_track_links.csv。', '',
        '## 稳定与歧义', '原设计：识别稳定surface。实际：至少连续两层同H run及50%共同可评价层支持形成图边；track提取禁止把同slice具有重叠支持高度的竞争branch并入一个track，冲突link单独保留；不重叠支持区允许同track handoff。stable表示至少跨3个slice的持续观测，ambiguous另记被隔离的冲突link，两者不是互斥标签。证据：surface_track_inventory.csv与support.csv；目标不改，不把自动选择当人工真值。', '',
        '## 缺失补全', '原设计：身份确定后LOCC。实际：只在单值尾部用同track双侧H样本补齐，起点保留原edge端点；多值内部缺口和身份不明保持unresolved。原因：旧LOCC的guide仍来自旧端点，会重新引入身份泄漏。证据：partial_tail_masking.csv；推断只补真实缺失的固定目标不变，恢复能力保守。', '',
        '## Handoff范围', '原设计：可检测track identity transition。实际：只允许同一已连通track中不同branch的持久置信度交叉；不同track和折返多值交叉不自动合并。原因：单凭分数无法证明不同track属于同一曲面；合成主路径用例检验非零换轨。固定目标不变，跨track过渡能力尚未支持。', '',
        '## 资源分项', '原设计：每阶段独立时间。实际：graph+track extraction合并，decision+junction合并，单次审计阶段分别计时；高置信度长度报告算法支持层之间保留的边长；真实错误替换数无独立真值则null。不得为填报而伪造0。', '',
        '## 旧案例定位', '原设计：A-F对照。实际：复用同样29个旧case；同slice只产生一个全局dominant决策，不用旧端点选择某个局部track。因此一部分旧case可能对应同一个新结果。证据：surface_track_decisions.csv；需人工确认是否应通过显式surface_track_id选择另一条稳定曲面。', '',
        '## 真实数据暴露的连通分量问题', '首轮自然审计193例全部未确认，28个唯一代表slice无法进行稳定track遮尾：普通graph连通分量含同slice竞争branch。实际改为按H证据强度依次合并、禁止支持高度重叠的同slice成员合并；冲突link保留，不删除证据、不改crossing阈值。新增global_connectivity_cannot_merge_competing_same_height_surfaces回归。首轮数据保存在相邻20260919_branch_first目录；本次纠正检查单独输出，固定目标不变。', '',
        '## 独立源码审核修正', '审核发现并修正：crossover先裁搜索edge再生成最近点；推断检查目标全部observed branches；H邻居先检查所有branch crossing唯一性再检查track身份。三项均有专用回归。', '']
    (output/'IMPLEMENTATION_DEVIATIONS.md').write_text('\n'.join(deviations), encoding='utf-8')
