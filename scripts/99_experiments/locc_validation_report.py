"""Chinese review report from measured LOCC artifacts, not synthetic claims."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import psutil

from validate_orthogonal_scanline_constraint import save_csv, save_json

ROOT = Path(__file__).resolve().parents[2]
STAGE_NAMES = {
    'frozen_profile_loading_and_neighbor_index': '冻结数据读取与邻线索引',
    'local_mesh_index': '局部网格空间索引',
    'real_masking_validation_with_adaptive_cuts': '真实遮挡验证（含局部横切）',
    'natural_gap_enumeration': '自然缺口候选枚举',
    'natural_gap_inference_with_adaptive_cuts': '自然缺口推断（含局部横切）',
    'checkpoint_output': '实验检查点输出',
    'finish_checkpoint_and_geometry_loading': '图审阶段数据与网格重载',
    'track_bridge_metrics_and_threshold_sweep': '短连接指标与track阈值扫描',
    'local_paper_figures_and_montages': '局部四联图与分类拼图',
    'display_correction_loading': '显示裁切纠正的数据重载（质检附加）',
    'display_clipping_correction_no_inference': '局部3D裁切纠正重绘（质检附加，无新推断）',
}


def fmt(value, precision=4):
    return '不适用/未观测' if value is None else f'{value:.{precision}f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v).replace('|', '/') for v in row)+' |' for row in rows])


def write_report(output, manifest, checkpoint, track, stages, figures, modules,
                 finish_wall_seconds, extra_plot_adaptive_seconds, extra_plot_adaptive_cuts):
    output = Path(output).resolve()
    masking, natural = checkpoint['mask_summary'], checkpoint['natural_summary']
    calibration_n = sum(c['split'] == 'calibration' for c in checkpoint['mask_cases'])
    holdout_n = sum(c['split'] == 'holdout' for c in checkpoint['mask_cases'])
    holdout = [r for r in masking['overall'] if r['split'] == 'holdout']
    comparisons = []
    for row in holdout:
        comparisons.append({'情形': row['regime'], '方法': row['method'], '留出位置数': row['n'],
                            '逐例误差中位数的中位数_mm': row['median_case_median_u_m']*1000,
                            '逐例P95误差中位数_mm': row['median_case_p95_u_m']*1000,
                            '逐例P95误差的P95_mm': row['p95_case_p95_u_m']*1000,
                            '最大点误差_mm': row['max_error_m']*1000,
                            '多分支选对率': row['branch_accuracy'], '可评估多分支层数': row['branch_evaluable_layers'],
                            '局部Hausdorff中位数_mm': row['median_hausdorff_m']*1000,
                            '曲线长度比中位数': row['median_length_ratio']})
    save_csv(output/'LOCC_三方法误差对比.csv', comparisons)
    monitored_total = sum(s['wall_seconds'] for s in stages)
    performance = [{'阶段': STAGE_NAMES.get(s['stage'], s['stage']), '耗时_s': s['wall_seconds'],
                    '总耗时占比': s['wall_seconds']/monitored_total, 'CPU累计_s': s['cpu_seconds'],
                    '平均逻辑CPU核数': s['mean_logical_cpu_equivalents'],
                    '整机CPU百分比_28逻辑核归一化': s['mean_machine_cpu_percent'],
                    '采样峰值工作集_GiB': s['peak_tree_rss_bytes']/2**30,
                    '峰值私有提交_GiB': s['peak_tree_private_bytes']/2**30,
                    '进程数_含主进程': s['max_processes_including_parent']} for s in stages]
    save_csv(output/'LOCC_阶段性能汇总.csv', performance)
    inner = [{'计时子项': '局部自适应横切+clean+crossings（含复用全局层过滤）',
              '耗时_s': checkpoint['adaptive_seconds']+extra_plot_adaptive_seconds,
              '次数': checkpoint['adaptive_cuts']+extra_plot_adaptive_cuts,
              '说明': '嵌套在遮挡/自然推断/绘图阶段内，不可再加到阶段总时间；ROI和邻线查询不在此子计时内'},
             {'计时子项': '遮挡LOCC求解', '耗时_s': masking['locc_solver_seconds'],
              '次数': masking['case_regimes'], '说明': '实际LOCC调用；未改动A/B比较结果复用首轮'},
             {'计时子项': '自然缺口LOCC求解', '耗时_s': natural['locc_solver_seconds'],
              '次数': natural['candidate_count'], '说明': '包含邻线假设、代价构建、两最佳二阶DP和指标，未冒充纯DP内核时间'}]
    save_csv(output/'LOCC_内部函数计时.csv', inner)
    initial_dir = Path(manifest['initial_iteration_output']) if manifest.get('initial_iteration_output') else None
    initial_mask = json.loads((initial_dir/'masking_summary.json').read_text(encoding='utf-8')) if initial_dir else None
    calibration = json.loads((initial_dir/'observed_weight_calibration.json').read_text(encoding='utf-8')) if initial_dir else None
    initial_stages = json.loads((initial_dir/'stage_performance.json').read_text(encoding='utf-8')) if initial_dir else []
    major = track['illustrative_major_structure_preservation']
    major_intact = sum(r['child_count'] == 1 for r in major)
    major_min_group = min(r['largest_child_group_fraction'] for r in major)
    major_min_length = min(r['largest_child_length_fraction'] for r in major)
    illustration = track['illustrative']
    previous_report = json.loads((output/'LOCC_VALIDATION_REPORT.json').read_text(encoding='utf-8')) if (output/'LOCC_VALIDATION_REPORT.json').exists() else {}
    peak_os = max(checkpoint.get('process_peak_working_set_bytes') or 0,
                  previous_report.get('performance', {}).get('windows_peak_working_set_bytes', 0),
                  getattr(psutil.Process().memory_info(), 'peak_wset', 0))
    peak_sample = max(s['peak_tree_rss_bytes'] for s in stages)
    peak_private = max(s['peak_tree_private_bytes'] for s in stages)
    numerical_stages = [s for s in stages if s['stage'] not in
                        ('local_paper_figures_and_montages', 'display_correction_loading', 'display_clipping_correction_no_inference')]
    numerical_seconds = sum(s['wall_seconds'] for s in numerical_stages)
    numerical_peak = max(s['peak_tree_rss_bytes'] for s in numerical_stages)
    statuses = {'LOCC_REPAIR': 'PARTIALLY_SUPPORTED', 'TRACK_SPLIT': 'PARTIALLY_SUPPORTED'}
    # SUPPORT here requires a positive reconstruction comparison, not merely
    # that an algorithm emitted candidate polylines or split components.
    if not any(p['comparator'] == 'VERTICAL_ONLY' and p['cluster_bootstrap_95ci_m'][0] > 0
               for p in masking['paired_holdout']):
        statuses['LOCC_REPAIR'] = 'NOT_SUPPORTED'
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).splitlines()
    source_files = [line[3:] for line in changed if line[3:].endswith('.py')]
    current_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = {'verdicts': statuses, 'git_head': current_head, 'git_branch': manifest['git_branch'],
              'new_commit': None, 'source_files': source_files, 'masking': masking,
              'masking_risk_calibration': checkpoint['calibration_report'], 'observed_weight_refinement': calibration,
              'natural': natural, 'track': track, 'figures': figures, 'plotting_modules_reused': modules,
              'performance': {'monitored_stage_total_seconds': monitored_total,
                              'numerical_analysis_seconds': numerical_seconds,
                              'numerical_analysis_peak_rss_bytes': numerical_peak,
                              'initial_and_calibration_seconds': sum(s['wall_seconds'] for s in initial_stages),
                              'all_recorded_experiment_seconds': monitored_total+sum(s['wall_seconds'] for s in initial_stages),
                              'final_inference_wall_seconds': checkpoint['inference_wall_seconds'],
                              'finish_wall_seconds': finish_wall_seconds,
                              'peak_sampled_process_rss_bytes': peak_sample, 'windows_peak_working_set_bytes': peak_os,
                              'peak_private_commit_bytes': peak_private, 'stages': stages, 'nested_function_timings': inner},
              'manual_review_complete': False, 'production_repair_enabled': False,
              'production_track_threshold_approved': False, 'loft_reconstruction_run': False,
              'frozen_verification': json.loads((output/'frozen_input_verification.json').read_text(encoding='utf-8'))}
    save_json(output/'LOCC_VALIDATION_REPORT.json', result)
    methods = {'VERTICAL_ONLY': '仅纵线端点插值', 'HORIZONTAL_ONLY': '仅横线分支', 'LOCC': 'LOCC协同'}
    regime_names = {'INTACT_HORIZONTAL': '横线完整', 'TARGET_HORIZONTAL_DROPOUT': '目标横线缺测'}
    error_table = table(['情形', '方法', 'n', '逐例P95中位数(mm)', '逐例P95的P95(mm)', '最大点误差(mm)', '多分支选对率'],
                        [[regime_names[r['regime']], methods[r['method']], r['n'], fmt(r['median_case_p95_u_m']*1000, 3),
                          fmt(r['p95_case_p95_u_m']*1000, 3), fmt(r['max_error_m']*1000, 3),
                          '不适用' if r['branch_accuracy'] is None else f"{r['branch_accuracy']:.1%}"] for r in holdout])
    bucket_table = table(['情形', '缺口高度(m)', 'n', 'LOCC逐例P95中位数(mm)', 'LOCC逐例P95的P95(mm)'],
                         [[regime_names[r['regime']], r['gap_bucket'], r['n'], fmt(r['median_case_p95_u_m']*1000, 3),
                           fmt(r['p95_case_p95_u_m']*1000, 3)] for r in masking['by_bucket'] if r['split'] == 'holdout' and r['method'] == 'LOCC'])
    paired_table = table(['情形', '参照', '配对P95改善中位数(mm)', '分块bootstrap 95%区间(mm)', '改善/变差/近似相同'],
                         [[regime_names[r['regime']], methods[r['comparator']], fmt(r['median_paired_p95_improvement_m']*1000, 3),
                           f"[{r['cluster_bootstrap_95ci_m'][0]*1000:.3f}, {r['cluster_bootstrap_95ci_m'][1]*1000:.3f}]",
                           f"{r['better_cases']}/{r['worse_cases']}/{r['tie_cases']}"] for r in masking['paired_holdout']])
    perf_table = table(['阶段', '耗时(s)', '占比', '平均逻辑CPU核', '工作集峰值(GiB)', '私有提交(GiB)'],
                       [[r['阶段'], fmt(r['耗时_s'], 3), f"{r['总耗时占比']:.1%}", fmt(r['平均逻辑CPU核数'], 2),
                         fmt(r['采样峰值工作集_GiB'], 3), fmt(r['峰值私有提交_GiB'], 3)] for r in performance])
    refinements = ''
    if initial_mask:
        initial_locc = {(r['regime'], r['split']): r for r in initial_mask['overall'] if r['method'] == 'LOCC'}
        refinements = table(['情形', '首轮LOCC逐例P95中位数(mm)', '调整后(mm)', '首轮最大点误差(mm)', '调整后(mm)'],
                            [[regime_names[r['regime']], fmt(initial_locc[(r['regime'], 'holdout')]['median_case_p95_u_m']*1000, 3),
                              fmt(r['median_case_p95_u_m']*1000, 3), fmt(initial_locc[(r['regime'], 'holdout')]['max_error_m']*1000, 3),
                              fmt(r['max_error_m']*1000, 3)] for r in holdout if r['method'] == 'LOCC'])
    figure_links = '\n'.join(f"- {f['kind']} / {f['candidate_id']}: [{Path(f['path']).name}]({Path(f['path']).resolve().as_posix()})"
                             for f in figures if f['kind'] == 'montage' or (f['kind'] in ('masking', 'track') and f == next((x for x in figures if x['kind'] == f['kind']), None)))
    # The detailed preservation table owns its fields; include its observed
    # records rather than guessing similarly named ratio keys in the narrative.
    correlation_rows = sorted(masking['evidence_correlations_holdout'], key=lambda r: abs(r['spearman_rho']), reverse=True)
    correlations = table(['情形', '指标', 'n', 'Spearman rho（探索性）'],
                          [[regime_names[r['regime']], r['feature'], r['n'], fmt(r['spearman_rho'], 3)] for r in correlation_rows[:10]])
    report = f'''# GDS LOCC 执行与真实验证报告

## 完成内容与结论

LOCC_REPAIR：**{statuses['LOCC_REPAIR']}**。TRACK_SPLIT：**{statuses['TRACK_SPLIT']}**。

已在原仓库实现 analysis-only 局部协同修复和短连接诊断。得到的是待审核的推断曲线，不是已经批准的地质修复。当前不启用正式 recognition 写回、不应用正式 track 合并阈值，也不进入 loft 曲面重建。

- 分支：`{manifest['git_branch']}`；HEAD：`{current_head}`；新 commit：无，未提交、未推送。
- 真实遮挡位置 {masking['locations']} 个，每个高度分组目标{manifest['per_bucket']}个；{calibration_n}个校准位置、{holdout_n}个空间留出位置。2种情形，共{masking['case_regimes']}组位置/情形、{masking['method_runs']}条方法对比记录。调整后新执行{masking.get('new_method_runs', masking['method_runs'])}次方法调用，复用{masking.get('reused_method_results', 0)}条未改动基线结果。
- 自然缺口候选 {natural['candidate_count']} 个，覆盖旧候选 {natural['previous_candidates_covered']} 个；正式写回修复数为0。
- 初步相对置信度：{json.dumps(natural['confidence_counts'], ensure_ascii=False)}。其中{natural['suggested_review_repair']}个列为优先审查修复，其余{natural['uncertain']}个为UNCERTAIN；这些不是人工审核结论。
- 示例track阈值删除 {illustration['removed_links']} 条关联，使 {illustration['original_tracks_split']} 个原track发生拆分；确认为误合并的数量未知。

## 修改文件与保留范围

{chr(10).join('- `'+p+'`' for p in source_files)}

旧 normal_vec[1] < -0.1、40° seed、70° growth、grouped_flags、correct_group_for_segments、FaceID传播均未改动。没有重复全量纵向识别，没有重新做全量横切，没有修改正式pkl或建立原算法本地备份。官方pkl以大小/mtime核查，识别源代码另作SHA256核查；不将该核查声称为4GB识别文件全量内容哈希。

## 相对参考方案的实现调整

1. 每个缺口固定补充25%/50%/75%三个局部横切，并复用落入缺口的全局0.10m层。所有多值分支保留；不再要求唯一分支或完整Face chain。FaceID仅附在真实观测节点上，推断边的FaceID明确为空。
2. 横向同一连续分支用于组织邻线拟合。当存在真实横向交点时，邻线仿射预测只作为较弱软约束；不存在横向交点时，保留完整邻线推断并标记V_NEIGHBOR_PREDICTED。采用确定性鲁棒加权一阶拟合、二阶两最佳DP；没有训练模型。
3. 首轮发现邻线拟合质量奖励偏向更平滑但错误的表面。仅在82个校准位置扫描真实H节点邻线相对权重0.01/0.03/0.10/0.30/1.00，按平均逐例P95误差选择0.01。虚拟节点权重未降低。首轮和参数扫描文件保留在上一级实验目录。这是单模型实验参数，尚非生产通用参数。
4. 增加0.01–0.05m分层，以及“目标横线缺测”压力情形。后者整段撤去目标位置H候选、保留邻线，不是假装横线仍直接观测到目标面。两种情形分别报告。
5. 自然候选范围扩为0<Δz≤1.6m、|Δu|≤0.5m的相向端点对，扩大范围不意味着全部候选都是可修缺损。几何中还可能含真实断裂、别的表面和边界。
6. track扫描只移除原有跨片关联，不改纵片内红组、不重新制造配对。少于两个共同真实采样层的关联保留并标为未解决，避免将0.10m采样无法分辨的短组误判为无连接。
7. 未添加可选mesh邻接惩罚；Face支持比例用于连续指标与来源审计。未使用skip-level捷径：无H层由明确的邻线虚拟候选或端点回退处理。置信度HIGH/MEDIUM/LOW为校准样本的相对风险分档，而非未经人工确认的绝对安全阈值。

## 三方法真实误差

真值为原始模型直接纵剖面曲线，不是地质真值。每个缺口在201个均匀高度及所有原曲线顶点组成的独立网格上评价，未仅在拟合横切点上比较。连接拓扑采用0.1微米XYZ量化，不跨越物理缺口。按7m空间块划分，边界留0.2m隔离，邻线窗口为±0.1m。

以下为{holdout_n}个空间留出位置的结果。首轮留出汇总已用于发现问题；调整后同一留出集只称回归复核，不冒充新的盲测。权重选择未使用其真值。

{error_table}

中位数、P95均先逐例计算再汇总，避免长曲线因采样点更多而获得更大权重。CSV另含逐例误差中位数、Hausdorff、长度比和端点连续性。端点误差为0是锚定构造的结果，不是准确性的独立证明。多分支选对率只统计具有多个不同径向候选、且存在距离隐藏真值0.1mm内分支的横向层；缺测情形无此分母，留空而不填0。

### 逐高度分组

{bucket_table}

### 配对改善及不确定性

正数代表LOCC更好。对空间块重采样1000次；置信区间只描述本模型样本分布，不覆盖地质真值、跨模型泛化和调整后再看同一留出集的选择偏差。

{paired_table}

### 验证反馈后的变化

{refinements}

## 置信度、自然缺口和人工审核

风险估计以校准区8个近邻案例的误差90%分位数为依据，特征包含缺口高度、H/N覆盖、双侧支持、分支歧义、邻线残差、端点切向误差和回退比例。相对分档界线来自校准区留一位置预测风险的50%/85%分位数；超出校准距离、高度范围或存在端点回退则OUT_OF_DOMAIN。完整校准及留出分档误差在masking_calibration.json。

真实缺口存在明显域差异：遮挡连续面不能模拟SfM缺失或真实地质分离。HIGH也需看局部图，且“存在其他远处分支”可能使算法选择绕行表面。不能因输出了一条连续线就判断应该修复。

图审类别与风险排序相互独立：CONSISTENT_VH仅表示所选横线与邻线相互一致，不代表端点切向或地质连续性都合理；EVIDENCE_CONFLICT也可能同时被相对风险模型列为HIGH/REVIEW_REPAIR。这时它仍只是优先核查对象，未获修复许可。任何候选的地质结论都等待manual_label。

{table(['图审类别', '自然候选数', '代表图数'], [[k, v, len(checkpoint['natural_selection'][k])] for k,v in natural['category_counts'].items()])}

每类最多10个near-pass、5个随机、5个最差，不足时只用真实数量；near-pass指接近校准风险分界，并非安全通过。人工CSV中manual_label/manual_notes全部留空，使用REPAIR/NO_REPAIR/UNCERTAIN标注。每幅四联图显示当前片、邻片、横向分支及实际局部网格；橙色虚线为推断，缺测虚拟点不标成真实H交点。隐去目标曲线后，基线红线也裁去该区间，隐藏真值仅作单独紫色参照。

## track短连接阈值扫描

L_overlap_z为高度区间交集；R_overlap、R_bridge均以较小组高度为分母。L_support为最长连续被支持的横切层中心跨度，单个支持层长度为0，而不是虚构0.10m。任何缺失/不支持中间层都会打断连续run；branch ID只在同一横切层内比较。

基线：{track['baseline']['group_count']}组、{track['baseline']['matched_links']}条关联、{track['baseline']['multi_slice_tracks']}个跨片track、{track['baseline']['isolated_groups']}个孤立组。扫描{track['sweep_grid_count']}组阈值，零阈值图与基线完全一致：{track['baseline_zero_grid_exact']}。

仅作展示的阈值为L_support≥0.10m、R_bridge≥0.25、R_overlap≥0.50，状态ILLUSTRATIVE_NOT_APPROVED。此设置删除{illustration['removed_links']}条边，拆分{illustration['original_tracks_split']}个原track；短采样连接疑点{illustration['questionable_short_sampled_bridges']}条，保护采样不足关联{illustration['protected_undersampled_links']}条。拆分后的分量、跨片track、孤立组分别为{illustration['components']}、{illustration['multi_slice_tracks']}、{illustration['isolated_groups']}。

上述数量是阈值敏感性，不是“修正了这么多地质错误”。track_split_cases.csv保留每条删除边的失败条件和人工标签；major_structure_preservation.csv包含全部80组阈值下原最大20个track的最大子组数/长度保留率。正式阈值必须人工审查后确定。

示例阈值下，最大{len(major)}个结构有{major_intact}个保持完整；其余{len(major)-major_intact}个发生拆分。所有这些大结构的最大子分量至少保留{major_min_group:.3%}的原组数、{major_min_length:.3%}的原累计长度。它们未被大量切碎，但被分离的小组是否应独立仍须审核。

## 性能与内存实测

平台：{manifest['os']}，{manifest['physical_cores']}物理核/{manifest['logical_cpus']}逻辑核，RAM {manifest['ram_bytes']/2**30:.2f}GiB；一个计算进程、数值库线程1，不使用GPU。复用原1148层全局横切与2800条纵线；没有重跑约58分钟的全识别流程。

{perf_table}

最终受监测阶段合计 **{monitored_total:.3f}s（{monitored_total/60:.2f}min）**，含读取、推断、track扫描、绘图，不含人工等待、代码开发和报告文本落盘。首轮加校准另耗{sum(s['wall_seconds'] for s in initial_stages):.3f}s；本任务所有已记录实验阶段累计{monitored_total+sum(s['wall_seconds'] for s in initial_stages):.3f}s。两阶段运行之间的人工等待不算算法时间。

其中不含绘图及显示质检的数值分析为 **{numerical_seconds:.3f}s**，采样峰值工作集 **{numerical_peak/2**30:.3f}GiB**。若表中包含“质检附加”行，它是发现3D窗口外几何遮挡标签后进行的显示裁切重绘，未重复任何识别、LOCC或track计算；它单列计时，不把多次出图包装成一次算法运行。

0.25s采样得到进程工作集峰值 **{peak_sample/2**30:.3f}GiB**；Windows记录的进程峰值工作集 **{peak_os/2**30:.3f}GiB**；私有提交峰值 **{peak_private/2**30:.3f}GiB**。工作集是驻留物理内存，私有提交不是实际驻留内存，不能相加；采样峰值可能漏掉短暂峰值，所以同时列出操作系统峰值。CPU整机占比以28逻辑核归一化，不用异构20物理核作分母。逐进程原始采样在process_resource_samples.csv。

{table(['嵌套子计时', '耗时(s)', '次数'], [[r['计时子项'], fmt(r['耗时_s'], 3), r['次数']] for r in inner])}

子计时已经包含在上表对应阶段内，不能再次相加。DP求解计时包含候选、鲁棒拟合及路径搜索；不将其伪装成剥离了数据准备的纯DP内核计时。全局横切复用次数{checkpoint['global_reused_cuts']}。

## 方案16个问题的审核答复

1. 邻纵是否显著提高恢复？不能笼统认定；上表分别比较横线完整、横线缺测、两种参照，并给配对区间。降低真实H节点的邻线权重说明强邻线约束可能有害。
2. 横线增加什么信息？给出实际径向分支位置，并以同一连续横向branch组织邻纵观测；在只有纵线端点时，这些表面细节并不存在。
3. 三方法误差？见三方法表和LOCC_三方法误差对比.csv，不能把中位误差当最坏误差。
4. 哪种高度可靠？见逐高度表。没有用户指定的工程容差或地质标签，不能给出“此高度必然安全”的界限；短缺口平直时端点插值本身就可能足够好。
5. 多分支能否稳定选对？看可评估多分支层的选对率，未达到100%，不能承诺稳定全对；错误案例保留在最差图审样本。
6. 哪些指标预测误差？见以下探索性相关表；这不是因果结论，也未进行多重比较校正。完整相关和校准分档误差分别保留。
7. 至少两层门槛？对gap修复取消，局部补切提供至少三个内部高度；track连续支持长度仍需要足够采样，否则标未解决。
8. 唯一分支门槛？取消，保留全部候选并记录最优/次优代价。没有次优路径时margin为空，不伪造无限高置信度。
9. 完整Face chain门槛？不继续作为修复hard gate；只保留真实来源，不为虚拟边制造FaceID。
10. 哪些自然gap可建议修复？本轮仅把{natural['suggested_review_repair']}个校准域内较低相对风险候选列为优先人工审查，不执行自动修复。
11. 哪些标为UNCERTAIN？风险模型未列为HIGH的{natural['uncertain']}个候选。图审类别与此排序独立，证据冲突/低margin可与HIGH共存，但HIGH仍只是优先人工核查；全部{natural['candidate_count']}个候选目前都没有人工地质判定。
12. 短连接过合并有多少？展示阈值有{illustration['questionable_short_sampled_bridges']}条短采样连接疑点；确认的过合并数量未知。
13. 拆开多少track？示例设置拆分{illustration['original_tracks_split']}个原track，删除{illustration['removed_links']}条边；其他设置见80行扫描表。
14. 拆分从局部图上是否合理？已生成对应真实局部几何图，尚无用户标签，不能替代地质审核宣告合理。
15. 大结构是否完整？最大{len(major)}条中{major_intact}条完全保持；最差最大子分量组数保留率{major_min_group:.3%}、累计长度保留率{major_min_length:.3%}。有拆分不等于正确纠错，没有人工依据不能批准该阈值。
16. 是否进入正式生产？不建议现在直接进入LOCC自动修复+正式track+loft。可进入本轮局部图和人工CSV审核，再确定误差容差、阈值及独立模型验证范围。

{correlations}

## 最低检查与待人工验证

已执行核心7项、横向连接组织1项、track7项、绘图1项的定向测试；核心单测通过于引入权重校准之前，最终权重版本采用本次真实数据复核作为证据，未重复单测。绘图测试首次发现标题字段检查与Z轴标签问题，针对性修改后仅重试一次；最终XYZ局部显示窗口另由真实图像检查，并定向检查显示裁切的线段/三角面及输入不变性。代码审查发现的冻结目录误写风险在首次写入前增加路径保护，另执行一次只读目录保护回归检查。真实运行包含首轮验证、一次校准反馈调整后的LOCC复核；未重复未改动的A/B算法、未执行全项目测试。代码审查和图像目检记录随输出保存，不以测试通过替代地质合理性。

待人工：填写自然gap和track删除边的标签；给出工程允许误差；审核最大结构拆分是否符合地质解释。保留风险：单模型、有限高度范围、同模型H/V共享几何、校准后再次使用原留出集、近邻曲面预测可选错branch、自然缺损域差异及稀疏H采样。

## 代表图与交付文件

{figure_links}

实际复用绘图模块：{', '.join('`'+m+'`' for m in modules)}。完整函数与复用方式见plotting_inventory.md。

主要表：LOCC_三方法误差对比.csv、LOCC_阶段性能汇总.csv、masking_results.csv、masking_bucket_summary.csv、LOCC_manual_review.csv、track_bridge_sweep.csv、track_split_cases.csv、major_structure_preservation.csv。所有CSV均为UTF-8 BOM，空值代表不适用/未观测而非零。

最终报告：{(output/'LOCC_VALIDATION_REPORT.md').as_posix()}；机器可读完整报告：LOCC_VALIDATION_REPORT.json。代码留在当前仓库，由用户自行commit和上传。
'''
    (output/'LOCC_VALIDATION_REPORT.md').write_text(report, encoding='utf-8')
