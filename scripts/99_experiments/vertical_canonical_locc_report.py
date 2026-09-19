"""Render a completed observed-first checkpoint, without re-running algorithms."""
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from validate_orthogonal_scanline_constraint import StageSampler, save_csv, save_json
from validate_locc_natural_gaps import file_digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/06_visualization'))


def representatives(data):
    old_rows = data['classification']['rows']
    preferred = [r['candidate_id'] for r in data['classification']['representatives']]
    result = []
    for category in 'ABCDEF':
        pool = [r for r in old_rows if r['primary_category'] == category]
        pool.sort(key=lambda r: (r['candidate_id'] not in preferred, -r['endpoint_turn_deg'], r['candidate_id']))
        chosen, groups = [], set()
        for row in pool:
            if row['spatial_group'] not in groups:
                chosen.append(row); groups.add(row['spatial_group'])
            if len(chosen) == 5: break
        for row in pool:
            if len(chosen) == min(5, len(pool)): break
            if row not in chosen: chosen.append(row)
        result.extend((category, row['candidate_id']) for row in chosen)
    return result


def render_case(data, category, cid, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from locc_local_review_plot import _paper_helpers
    from locc_data import opc
    style, _, _, _, _ = _paper_helpers()
    plt.rcParams.update(style)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})
    case = next(c for c in data['cases'] if c['candidate_id'] == cid)
    result, old = data['results'][cid], data['old_natural_solutions'][cid]
    profile = data['target_profiles'][case['slice_key']]
    nodes = opc.to_suz(profile.nodes, data['arc'])[:, 1:]
    observed = nodes[profile.edges]
    low, high = np.asarray(case['lower_uz']), np.asarray(case['upper_uz'])
    guide = np.asarray(result['guide_curve_uz'])
    radial = np.r_[old['curve_uz'][:, 0], guide[:, 0], low[0], high[0]]
    u_pad, z_pad = max(.03, np.ptp(radial)*.15), max(.015, (high[1]-low[1])*.12)
    bounds = (radial.min()-u_pad, radial.max()+u_pad, low[1]-z_pad, high[1]+z_pad)
    visible = observed[(observed[:, :, 1].max(axis=1) >= bounds[2]) & (observed[:, :, 1].min(axis=1) <= bounds[3])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax in axes:
        ax.add_collection(LineCollection(visible, colors='#9a9a9a', linewidths=1.0, alpha=.7))
        ax.scatter([low[0], high[0]], [low[1], high[1]], c='#222222', s=32, zorder=6)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='Radial offset u (m)')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.grid(alpha=.17)
    axes[0].set_ylabel('Elevation z (m)')
    curve = old['curve_uz']
    axes[0].plot(curve[:, 0], curve[:, 1], c='#dc6b26', lw=2, label='Old whole-window proposal')
    axes[0].set_title('Previous LOCC: entire endpoint interval')
    right = axes[1]
    eligible = result['observed_edge_ids']
    if eligible:
        right.add_collection(LineCollection(nodes[profile.edges[eligible]], colors='#2267a8', linewidths=2.0))
    for edge in result['path_edges']:
        if edge['source'] == 'OBSERVED_VERTICAL': continue
        xy = np.asarray(edge['points_uz'])
        color = '#b89500' if edge['source'] == 'TOPOLOGY_STITCH' else '#d22e35'
        right.plot(xy[:, 0], xy[:, 1], color=color, lw=2.4, linestyle='--', zorder=5)
    right.plot([low[0], high[0]], [low[1], high[1]], color='#444444', lw=.7, linestyle=':', zorder=1)
    for layer in data['windows'][cid]['layers']:
        hs = [h['u'] for h in layer['horizontal']]
        right.scatter(hs, [layer['z']]*len(hs), marker='x', c='#262626', s=15, zorder=6)
    for label, name, point, shift in [('Lower', 'start_node', low, (7, -15)), ('Upper', 'stop_node', high, (7, 8))]:
        node = result[name]
        component = int(profile.components[node]) if node is not None else 'external'
        right.annotate(f'{label}: C{component}', point, xytext=shift, textcoords='offset points', fontsize=9)
    for start, stop in result['missing_intervals']:
        right.axhspan(start, stop, color='#d22e35', alpha=.07)
    pairing = data['pairs'][cid]
    right.set_title(f"Observed-first: {result['gap_type']} / coverage {result['coverage_ratio']:.1%}")
    fig.suptitle(f"{category} / {cid} / s={case['s']:.2f} m", fontsize=14, y=.98)
    fig.text(.5, .905, f"{result['status']}  |  {pairing['pairing_status']}", ha='center', fontsize=10)
    handles = [Line2D([], [], c='#9a9a9a', label='Original observed geometry'),
               Line2D([], [], c='#dc6b26', lw=2, label='Old whole-window proposal'),
               Line2D([], [], c='#2267a8', lw=2, label='Canonical observed corridor'),
               Line2D([], [], c='#b89500', lw=2, ls='--', label='Topology stitch (if present)'),
               Line2D([], [], c='#d22e35', lw=2, ls='--', label='Inferred-only edge (if present)'),
               Line2D([], [], c='#444444', lw=.8, ls=':', label='Endpoint pairing, not geometry'),
               Line2D([], [], c='#262626', marker='x', ls='', label='Actual H crossing')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(.5, .015))
    fig.subplots_adjust(left=.08, right=.98, bottom=.19, top=.84, wspace=.10)
    path = output/'figures'/f'{category}_{cid}.png'
    fig.savefig(path, dpi=170); plt.close(fig)
    return {'old_category': category, 'candidate_id': cid, 'gap_type': result['gap_type'],
            'status': result['status'], 'pairing_status': pairing['pairing_status'], 'figure': str(path.resolve())}


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def reconcile_review_provenance(data, output):
    """Correct pre-review metadata only, retaining measured numerical results.

    The completed run inserted zero synthetic edges. Exact-XYZ anchoring is a
    forward correction verified by the targeted test, not a claimed new run.
    """
    from locc_data import opc
    if data.get('post_review_provenance') or data.get('provenance_schema', 1) >= 2:
        return
    if data['recognition']['summary']['synthetic_edges_added']:
        raise ValueError('Applied synthetic geometry requires a new scoped recognition audit')
    changed, checked = 0, 0
    key_for = lambda line: tuple(sorted(map(tuple, line)))
    for key, profile in data['target_profiles'].items():
        full = data['full_profiles'][key]
        lookup = {key_for(line): i for i, line in enumerate(full.lines_xyz)}
        for i, line in enumerate(profile.lines_xyz):
            j = lookup[key_for(line)]
            if profile.source_face_ids[i] != full.source_face_ids[j]:
                raise AssertionError('Target/full FaceID provenance mismatch')
            changed += profile.source_segment_indices[i] != full.source_segment_indices[j]
            profile.source_segment_indices[i] = list(full.source_segment_indices[j])
            checked += 1
    orientation_corrections = 0
    for case in data['cases']:
        result = data['results'][case['candidate_id']]
        profile = data['target_profiles'][case['slice_key']]
        uz = opc.to_suz(profile.nodes, data['arc'])[:, 1:]
        for edge in result['path_edges']:
            if edge['source'] != 'OBSERVED_VERTICAL': continue
            expected = uz[list(edge['nodes'])]
            points = np.asarray(edge['points_uz'])
            if not np.array_equal(points, expected):
                if not np.array_equal(points, expected[::-1]):
                    raise AssertionError('Unexpected measured coordinate change')
                edge['nodes'] = edge['nodes'][::-1]
                orientation_corrections += 1
            edge['source_segment_indices'] = list(profile.source_segment_indices[edge['edge_id']])
    revision = {'target_edges_checked': checked, 'target_source_index_lists_corrected': changed,
                'oriented_observed_records_corrected': orientation_corrections,
                'numerical_inference_or_recognition_repeated': False, 'geometry_changed': False,
                'measured_commit_unchanged': True, 'initial_scoped_tests_passed': 33,
                'targeted_review_regressions_passed': 3,
                'current_source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in [
                     ROOT/'scripts/04_structure_recognition/vertical_topology_reconstruction.py',
                     ROOT/'scripts/99_experiments/validate_observed_first_locc.py']}}
    data['post_review_provenance'] = revision
    data['manifest']['post_measurement_provenance_correction'] = revision
    save_json(output/'post_review_provenance.json', revision)
    save_json(output/'manifest.json', data['manifest'])
    save_json(output/'candidate_proposals.json', {'pairings': data['pairs'], 'results': data['results']})
    with (output/'validation_checkpoint.pkl').open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)


def finish(output):
    output = Path(output)
    report_path = output/'VERTICAL_CANONICAL_LOCC_REPORT.md'
    if report_path.exists(): raise FileExistsError('Do not overwrite a delivered report')
    started = time.perf_counter()
    with (output/'validation_checkpoint.pkl').open('rb') as stream: data = pickle.load(stream)
    reconcile_review_provenance(data, output)
    stages = data['stages']
    (output/'figures').mkdir(exist_ok=True)
    with StageSampler('representative_figures', output, stages):
        figures = [render_case(data, category, cid, output) for category, cid in representatives(data)]
    save_csv(output/'representative_figures.csv', figures)
    rows, counts = data['rows'], Counter(r['gap_type'] for r in data['rows'])
    totals = {name: sum(r[name] for r in data['canonical_rows']) for name in
              ('raw_vertical_segments', 'canonical_vertical_edges', 'exact_duplicate_vertical_edges',
               'duplicate_source_face_count', 'degree_changed_nodes', 'collapsed_self_edges')}
    selected = [r for r in rows if r['selected']]
    topology_only = [r for r in rows if r['status'] == 'RESOLVED_PROPOSAL' and r['inferred_length_m'] == 0]
    actual_topology = [r for r in topology_only if r['topology_stitches'] > 0]
    inferred = [r for r in rows if r['inferred_length_m'] > 0]
    apply_rows = [r for r in rows if r['applied_to_experimental_recognition']]
    existing189 = [r for r in rows if r['old_existing_sample_fraction'] == 1]
    category_rows = []
    for category in 'ABCDEF':
        sub = [r for r in rows if r['old_category'] == category]
        category_rows.append({'old_category': category, 'count': len(sub),
            **{kind: sum(r['gap_type'] == kind for r in sub) for kind in ('TYPE_I', 'TYPE_II', 'TYPE_III')},
            'observed_only_resolved': sum(r in topology_only for r in sub),
            'with_inferred_proposal': sum(r in inferred for r in sub),
            'unresolved': sum(r['status'] != 'RESOLVED_PROPOSAL' for r in sub),
            'ambiguous_pairing': sum(r['pairing_ambiguous'] for r in sub)})
    save_csv(output/'category_transition_summary.csv', category_rows)
    performance = []
    numeric_total = data['inference_wall_seconds']
    overall = numeric_total+time.perf_counter()-started
    for stage in stages:
        performance.append({'阶段': stage['stage'], '墙钟秒': stage['wall_seconds'], 'CPU秒': stage['cpu_seconds'],
            '平均逻辑核数': stage['mean_logical_cpu_equivalents'], '整机CPU百分比': stage['mean_machine_cpu_percent'],
            '峰值RSS_GiB': stage['peak_tree_rss_bytes']/2**30, '峰值Private_GiB': stage['peak_tree_private_bytes']/2**30,
            '进程数含主进程': stage['max_processes_including_parent'], '墙钟占总运行比例': stage['wall_seconds']/overall})
    save_csv(output/'stage_performance.csv', performance)
    recognition, track = data['recognition']['summary'], data['track']['summary']
    summary = {'canonical': totals, 'types': dict(counts), 'cohort': len(rows), 'selected_pairs': len(selected),
        'selected_unambiguous_pairs': sum(not r['pairing_ambiguous'] for r in selected),
        'pairing_ambiguous_all': sum(r['pairing_ambiguous'] for r in rows),
        'observed_only_resolved_proposals': len(topology_only), 'topology_stitch_resolved_proposals': len(actual_topology),
        'existing189_observed_only_resolved': sum(r in topology_only for r in existing189),
        'existing189_pairing_ambiguous': sum(r['pairing_ambiguous'] for r in existing189),
        'inferred_proposals': len(inferred), 'applied_candidates_experimental': len(apply_rows),
        'remaining_geometric_inference_candidate_count': sum(r['gap_type'] != 'TYPE_I' for r in selected),
        'partial_gap_inferred_length_m': sum(r['inferred_length_m'] for r in rows if r['gap_type']=='TYPE_II'),
        'true_gap_inferred_length_m': sum(r['inferred_length_m'] for r in rows if r['gap_type']=='TYPE_III'),
        'inferred_observed_interval_overlaps': sum(r['inferred_observed_interval_overlaps'] for r in rows),
        'global_potential_changed_slices': len(data['global_changed_slice_keys']),
        'recognition': recognition, 'track': track, 'category_summary': category_rows,
        'near_endpoint_bucket_counts': data['near_endpoint_bucket_counts'],
        'numeric_wall_seconds': numeric_total, 'numeric_and_figures_seconds': overall,
        'peak_rss_gib': max(s['peak_tree_rss_bytes'] for s in stages)/2**30,
        'peak_private_gib': max(s['peak_tree_private_bytes'] for s in stages)/2**30,
        'frozen_unchanged': data['frozen_unchanged'], 'figure_count': len(figures),
        'statuses': {'VERTICAL_CANONICALIZATION': 'PARTIALLY_SUPPORTED', 'TOPOLOGY_RECONSTRUCTION': 'NOT_SUPPORTED',
                     'OBSERVED_FIRST_LOCC': 'PARTIALLY_SUPPORTED'}}
    save_json(output/'summary.json', summary)
    save_json(output/'workbook_payload.json', {'summary': summary, 'performance': performance,
        'masking': [r for r in data['masking']['summary'] if r['split']=='holdout'], 'categories': category_rows})
    mask_rows = [r for r in data['masking']['summary'] if r['split'] == 'holdout']
    mask_table = md_table(['H条件', '方法', '已恢复/总数', '病例P95中位数(mm)', '病例P95的P95(mm)', '最大点误差(mm)'],
        [[r['regime'], r['method'], f"{r.get('n',0)}/{r['total']}",
          f"{r['median_case_p95_u_m']*1000:.3f}" if 'median_case_p95_u_m' in r else 'N/A',
          f"{r['p95_case_p95_u_m']*1000:.3f}" if 'p95_case_p95_u_m' in r else 'N/A',
          f"{r['max_error_m']*1000:.3f}" if 'max_error_m' in r else 'N/A'] for r in mask_rows])
    perf_table = md_table(['阶段','墙钟(s)','CPU(s)','平均逻辑核','整机CPU(%)','RSS峰值(GiB)','Private峰值(GiB)'],
        [[r['阶段'],*[f'{r[k]:.3f}' for k in ('墙钟秒','CPU秒','平均逻辑核数','整机CPU百分比','峰值RSS_GiB','峰值Private_GiB')]] for r in performance])
    text = f'''# Vertical Canonical / Observed-First LOCC 验证报告

## 结论与范围

本轮实现了独立的规范纵向图、连续邻线分支、连续区间覆盖分类、联合端点配对和观测优先恢复接口。
生产主流程未自动切换，原识别参数、mesh、2800条纵向测线与冻结结果未改写。
2800条测线全部做canonical统计；只对193候选涉及且实际变化的测线复跑识别。其余复用原结果。
这是同模型回归验证，不是新盲测或地质真值验证。没有把未恢复的案例算作“修复成功”。

实测结果尚未达到自然缺口自动修复目标：190/193项仍待解析，3项无H支撑的推断提案也未应用，实际短接与新增边均为0。
这表明本轮主要完成了避免错误补线的硬约束与诊断分流，不能将其表述为193项已被修复。

backend HEAD：`{data['manifest']['git_head']}`。本轮未提交，commit SHA：无，由用户自行commit/push。

## 1. Canonical与来源统计

{md_table(['指标','数量'], [[k,v] for k,v in totals.items()])}

使用与原程序相同的round6节点键，只消除唯一无向边重复。每条canonical边保留所有source_face_ids、source_segment_indices和首条FaceID。
自环不丢弃，其全部来源仍可追溯；近端点只诊断，不全局吸附。
重复边只有13/7,173,018（约0.000181%），不是193项待审的主要数量来源。duplicate_source_face_count计每条唯一边额外的不同FaceID个数。
最近异组件degree-1端点按节点计数（同一对可能双向计入）：`{data['near_endpoint_bucket_counts']}`。

## 2. 193候选的前后变化

{md_table(['类型','数量','含义'], [['I',counts['TYPE_I'],'已有几何，优先拓扑；禁止推断坐标'],['II',counts['TYPE_II'],'近似走廊内覆盖不足，尚未证实是真缺失'],['III',counts['TYPE_III'],'目标走廊无纵向覆盖，仍须排除自然边界']])}

193条旧候选记录全部保留。联合配对选中{len(selected)}条，其中明确一对一{summary['selected_unambiguous_pairs']}条，选中但竞争歧义{len(selected)-summary['selected_unambiguous_pairs']}条。
已观测路径可解析的提案{len(topology_only)}条，其中真的插入拓扑短接{len(actual_topology)}条。
有推断段的提案{len(inferred)}条；选中配对中仍涉及缺失区间{summary['remaining_geometric_inference_candidate_count']}条。
进入本轮实验识别的候选{len(apply_rows)}条，未向正式识别文件写回。
Type II推断总长{summary['partial_gap_inferred_length_m']:.6f}m，Type III推断总长{summary['true_gap_inferred_length_m']:.6f}m。
这里是所有提案长度之和，包含未获准应用的替代提案，不是去重后的新增生产几何长度。

原189项采样处有纵向几何的案例中，观测/拓扑路径可解析{summary['existing189_observed_only_resolved']}项，配对仍有竞争歧义{summary['existing189_pairing_ambiguous']}项。
“所有采样点有几何”不等于“端点之间存在同一条连续路径”；本轮按连续覆盖区间与组件连通性分别检查。
走廊本身由离散H交点间插值得到。真实纵向曲线在两层H之间弯曲并离开30mm走廊，也会表现为Type II；因此missing区间是待核定覆盖缺口，不是已确认的模型孔洞。

{md_table(list(category_rows[0]), [[r[k] for k in category_rows[0]] for r in category_rows])}

## 3. 不覆盖实测的验证

逐候选核对canonical节点/边输入哈希不变、被选observed边坐标与来源逐项相等、推断边FaceID为None，并检查推断Z区间不与目标走廊observed区间重叠。
检测到的observed overwrite / interval overlap：**{summary['inferred_observed_interval_overlaps']}**。
该值限本轮193自然候选及另存的masking不变量；不等于模型全域已证明不存在任何地质误配。
实际H交点节点可以保留真实FaceID，连接这些点的推断边不能继承FaceID。

## 4. 四方法真实遮挡回归

原150位置，82 calibration、68 holdout，原两种H条件不变。A/B/C直接复用已冻结结果，新跑D共300次。
未用隐藏truth选分支或调参。完整遮挡的边界另留3微米量化保护带，防止round6把隐藏几何残片带入求解。
表中为68个holdout位置；聚合分母、未恢复数保留，无法恢复的案例不会伪造误差0。

{mask_table}

连续邻线轨迹固定整窗branch身份，缺少覆盖不改选另一surface；这能防止逐层A/B/A切换，但不证明最初选中的分支正确。
准确度是否优于旧LOCC必须分别看完整H与H缺失条件及尾部误差，不能仅以端点误差0认定改善。
本轮holdout完整H的病例P95中位数由1.539降至1.439mm，但可评估分支准确率由77.97%降至75.71%；
H缺失时该中位数由8.221升至8.801mm（约退化7.1%），病例P95的尾部P95由45.987降至43.683mm。
因此整体是有改善也有退化，不建议自动替换旧LOCC。

## 5. 变更测线重识别与局部track

原normal<−0.1、40°/70°、grouped_flags和correct_group逻辑不改，未启用旧路径merge。
全局存在重复边、潜在会改变拓扑的测线：{summary['global_potential_changed_slices']}。
本轮只重跑193候选涉及的变更测线。统计如下：

```json
{json.dumps(recognition, ensure_ascii=False, indent=2)}
```

只重配真实相邻0.05m测线且触及变更测线的局部邻接。L_support、R_bridge、R_overlap继续保留，采样不足不判定桥接成立。
固定(.10m,.25,.50)仍是待审示例阈值，不是工程验收标准。局部结果如下：

```json
{json.dumps(track, ensure_ascii=False, indent=2)}
```

没有新增地质标注，不能把拆分数量解释为真实过度合并减少。大结构的几何保留与组标签/track连通性要分开审核。

## 6. 实测性能

平台：{data['manifest']['platform']}，{data['manifest']['logical_cpus']}逻辑CPU，RAM {data['manifest']['ram_bytes']/2**30:.2f}GiB。
本轮单进程、数值库单线程、无GPU，不是之前15进程完整识别环节的重新计时。
数值验证连同输入加载/局部重识别/track共{numeric_total:.3f}s（{numeric_total/60:.2f}分钟）；加29局部图约{overall:.3f}s。
峰值进程树RSS {summary['peak_rss_gib']:.3f}GiB，Private {summary['peak_private_gib']:.3f}GiB。
RSS是驻留物理页，Private是私有提交量，两者不是同一指标；每0.25s采样，阶段边界也采样，短瞬时峰值仍可能漏采。
CPU百分比以整机逻辑核数为分母；1个核持续满载不等于整机100%。表中各阶段峰值不能相加。

{perf_table}

## 7. 与参考方案的差异及原因

1. 不直接改旧solve_locc内部候选类型；用独立observed-first图包装，只在missing子区间调用原求解器，保留旧算法作可追溯对照。
2. 只消除round6完全重复边，真实自环和所有来源保留。未全局近端点merge、未动原mesh。
3. 真fork不自动一对多；联合匹配允许不配、竞争者留待审，防止为减少数量强制连接。
4. 30mm corridor、100微米短接上限、0.5配对分数与0.03竞争差均为明确记录的保守实验规则，未经工程容差批准。
5. 短接还需切向与间隙方向一致、50mm高度邻域内H支持和两侧邻V支持；坐标只用已有端点。
6. 本轮非全量主识别重跑；2800全局几何统计与候选测线实际重识别数单列。track局部重组不冒充全域重新评估。
7. E类仅4个真实案例，全部展示；其他五类各5图，共29图。未制造第5个E类。
8. 遮挡A/B/C复用旧结果，D新跑。不重新调原LOCC权重，不把这批回归称作新盲测。
9. 按用户最低检查要求只运行本轮专用检查，不跑全项目测试；按用户要求不提交、不推送、不另备份源算法。
10. 独立审查后的三项修正：目标射线图的原slice段索引映射、反向路径的节点顺序、已有XYZ锚点直接复用。
原33项检查通过，增加3项针对性回归并单次通过。此次自然候选实际新增边为0，修正不改变已测分类、曲线、识别结果或耗时。
目标图来源元数据已从完整slice canonical图核对还原，记录见post_review_provenance.json；manifest保留实测时源哈希并另列修正后的源哈希。
11. 短组track本轮未新增自适应横向切片，采样不足保留UNRESOLVED，不能将其直接判成断桥。

## 8. 对14个审核问题的回答

1. exact duplicate数量见第1节，原始/规范边数可在逐测线CSV核对。
2. canonical度数由唯一边计算；改变节点数见第1节。干净输入的兼容结果见第5节。
3. Type I/II/III分别为{counts['TYPE_I']}/{counts['TYPE_II']}/{counts['TYPE_III']}。
4. 原189项的可解析与配对歧义数见第2节；连续覆盖不等于组件已经接通。
5. 选中配对中{summary['remaining_geometric_inference_candidate_count']}项仍涉及缺失区间；未获人工真缺口确认。
6. 实测覆盖检测为{summary['inferred_observed_interval_overlaps']}，逐项不变量保存在明细。
7. A—F去向见转换表；分类含义保持旧轮，不重标成已确认错误。
8. 一对一选择后保留{len(selected)}个匹配，相对于193旧提案减少重复竞争；这不是修复成功数。
9. 四方法数据见第4节，结论需同时看两种H条件与尾部误差，不宣称统一提升。
10. track变化见第5节；无新增真值，不能证明过度合并已进一步减少。
11. 原observed几何全部保留；大型结构分组/track是否符合地质意义仍需图审。
12. canonical可先以显式开关接入进一步审核，不建议在本轮范围验证后无条件替换全部主流程。
13. observed-first原则得到不覆盖验证支持，但生产自动修复仍需端点身份和边界案例人工批准。
14. 优先审配对歧义、Type I仍断连、Type II区间锚点不足、E类边界、变更后大组及track拆分。

结论：VERTICAL_CANONICALIZATION = PARTIALLY_SUPPORTED；TOPOLOGY_RECONSTRUCTION = NOT_SUPPORTED（限定本轮自然候选的修复有效性）；OBSERVED_FIRST_LOCC = PARTIALLY_SUPPORTED。
这些状态表示本轮限定验证支持，不代表全域生产验收。

## 交付与复核

- `candidate_classification_193.csv`：193项完整分类、配对、应用状态及不变量。
- `canonical_slice_statistics.csv`、`near_endpoint_diagnostics.csv`：全2800测线来源和近端点统计。
- `masking_four_methods.csv`、`masking_four_method_summary.csv`：四方法逐例与汇总。
- `stage_performance.csv`、`process_resource_samples.csv`：阶段汇总与CPU/RSS/Private原始采样。
- `representative_figures.csv`、`figures/`：29局部图；蓝=实测canonical，黄虚线=短接，红虚线=推断，灰点线仅表示端点配对。
- `manifest.json`：HEAD、来源哈希和本轮阈值。冻结输入前后检查：{data['frozen_unchanged']}。
- `validation_checkpoint.pkl`为修正后的权威结果；`core_checkpoint.pkl`仅为审查前阶段中间产物，其目标图段索引未作为最终来源记录使用。
'''
    report_path.write_text(text, encoding='utf-8')
    index_lines = ['# 代表案例图审索引', '', '每类5项，E类全部4项。图中候选配对不是已批准的真实结构连接。', '']
    for item in figures:
        index_lines += [f"## {item['old_category']} · {item['candidate_id']} · {item['gap_type']}", '',
                        f"{item['status']}；{item['pairing_status']}", '', f"![{item['candidate_id']}]({item['figure']})", '']
    (output/'REPRESENTATIVE_REVIEW.md').write_text('\n'.join(index_lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
