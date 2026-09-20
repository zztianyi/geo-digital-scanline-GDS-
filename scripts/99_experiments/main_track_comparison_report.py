"""A few larger, equal-scale local views of actual exported main-track paths."""
from __future__ import annotations
from collections import Counter
import html
import json
from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from surface_track_local_review import clipped_segments

LABELS = dict(LONG_CONNECTOR='连接距离超过 1 cm', SHARP_JUNCTION='连接处转折超过 60°',
    COMPETING_BRANCH='附近仍有未采用曲线段', UNCOVERED_EXTENT='仍未到达观测高程边界',
    HISTORICAL_LOCAL_MISS='历史局部窗口仍覆盖不足')
COLORS = dict(raw='#b6bdc7', old='#b97334', new='#1365ad', gap='#ce4149')


def load_result(output, index, key):
    with (output/'main_tracks.pkl').open('rb') as stream:
        stream.seek(index[key])
        return pickle.load(stream)


def draw_case(payload, region, destination):
    bounds = region['bounds']
    result = payload['result']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 8.4), sharex=True, sharey=True)
    has_gap = False
    for i, ax in enumerate(axes):
        for b in payload['branches']:
            seg = clipped_segments(b['points_uz'], bounds)
            if len(seg):
                ax.add_collection(LineCollection(seg, colors=COLORS['raw'], linewidths=1.15, zorder=1))
        if i == 0:
            seg = clipped_segments(payload['old_curve'], bounds)
            if len(seg):
                ax.add_collection(LineCollection(seg, colors=COLORS['old'], linewidths=2.8, zorder=3))
            else:
                ax.text(.04, .95, '旧结果未进入这个局部', transform=ax.transAxes, va='top', fontsize=11,
                    color=COLORS['old'], bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
        else:
            groups = {'observed': [], 'connector': []}
            for edge in result['path_edges']:
                kind = 'observed' if edge['source'].startswith('OBSERVED') else 'connector'
                segments = clipped_segments(edge['points_uz'], bounds)
                if kind == 'connector' and len(segments):
                    # A true intersection has a zero-length topology record,
                    # but should not introduce an invisible fourth legend item.
                    segments = segments[np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1) > 1e-9]
                groups[kind].extend(segments)
            if groups['observed']:
                ax.add_collection(LineCollection(groups['observed'], colors=COLORS['new'], linewidths=2.8, zorder=3))
            if groups['connector']:
                has_gap = True
                ax.add_collection(LineCollection(groups['connector'], colors=COLORS['gap'], linestyles='--', linewidths=2.2, zorder=4))
            joins = [j for j in result.get('junctions', []) if bounds[0] <= j['a_point_uz'][0] <= bounds[1]
                     and bounds[2] <= j['a_point_uz'][1] <= bounds[3]]
            if joins:
                preferred = region.get('junction_index')
                chosen = result['junctions'][preferred] if preferred is not None and result['junctions'][preferred] in joins else min(
                    joins, key=lambda j: abs(j['a_point_uz'][1]-(bounds[2]+bounds[3])/2))
                p = chosen['a_point_uz']
                distance = chosen['distance_m']
                connection = '交点接续' if distance < 1e-9 else (f'补接 {distance:.2f} m' if distance >= 1. else f'补接 {distance*100:.2f} cm')
                label = f"{chosen['from_branch_id']} → {chosen['to_branch_id']}\n"+connection
                ax.annotate(label, xy=p, xytext=(.06, .12), textcoords='axes fraction', color=COLORS['new'], fontsize=11,
                    arrowprops=dict(arrowstyle='->', color=COLORS['new'], lw=1.1),
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
            if not groups['observed']:
                ax.text(.04, .95, '本版仍未进入这个局部', transform=ax.transAxes, va='top', fontsize=11, color=COLORS['gap'])
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='径向偏移 u（m）')
        ax.set_aspect('equal', adjustable='box')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.grid(alpha=.14)
        ax.set_title('修复前：单分支结果' if i == 0 else '修复后：拼接主轨', fontsize=14, pad=12)
    axes[0].set_ylabel('高程 z（m）')
    handles = [Line2D([], [], color=COLORS[k], lw=w, label=label) for k, w, label in
        [('raw', 1.2, '原始切面曲线'), ('old', 2.8, '修复前输出'), ('new', 2.8, '修复后主轨')]]
    if has_gap:
        handles.append(Line2D([], [], color=COLORS['gap'], lw=2.2, ls='--', label='跨空隙连接'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(.5, .055))
    fig.suptitle(f"{region['region_id']}  |  切面位置 s = {float(payload['slice_key']):.2f} m", fontsize=16, y=.97)
    fig.text(.5, .02, '左右范围与比例完全相同；灰线是背景，重点比较棕线与蓝线。', ha='center', fontsize=10, color='#4b5563')
    fig.subplots_adjust(left=.09, right=.97, bottom=.17, top=.86, wspace=.13)
    fig.savefig(destination, dpi=190, facecolor='white')
    plt.close(fig)


def choose_regions(cache, output, index):
    chosen = []
    by_id = {r['candidate_id']: r for r in cache['local_rows']}
    for cid in ('N00918', 'N00532', 'N00208'):
        r = by_id.get(cid)
        if r:
            chosen.append(dict(region_id=cid, slice_key=r['slice_key'], bounds=r['bounds'], category='KNOWN_CASE',
                detail=f"原窗口观测覆盖 {r['before_coverage']:.1%} → {r['after_coverage']:.1%}", junction_index=None))
    if 'N00918' in by_id:
        p = load_result(output, index, by_id['N00918']['slice_key'])
        j = next((j for j in p['result'].get('junctions', []) if j['from_branch_id'] == 8 and j['to_branch_id'] == 10), None)
        if j:
            u, z = j['a_point_uz']
            chosen.insert(1, dict(region_id='N00918_JOIN', slice_key=p['slice_key'], bounds=[u-.56, u+.56, z-.65, z+.65],
                category='JOIN_EXAMPLE', detail='分支 8 与分支 10 的实际接续位置', junction_index=p['result']['junctions'].index(j)))
    for category in LABELS:
        candidates = sorted((r for r in cache['windows'] if r['category'] == category), key=lambda r: (-r['severity'], float(r['slice_key']), r['region_id']))
        if candidates:
            region = dict(candidates[0])
            if category == 'COMPETING_BRANCH':
                # Show an actual nearby alternative point, rather than an
                # independent coordinate median that can fall off the curve.
                payload = load_result(output, index, region['slice_key'])
                bid = int(region['region_id'].split('_B')[1].split('_')[0])
                points = next(b['points_uz'] for b in payload['branches'] if b['branch_id'] == bid)
                edges = np.asarray([e['points_uz'] for e in payload['result']['path_edges'] if e['source'].startswith('OBSERVED')])
                a, d = edges[:, 0], edges[:, 1]-edges[:, 0]
                t = np.clip(((points[:, None]-a)*d).sum(2)/(d*d).sum(1), 0., 1.)
                distances = np.linalg.norm(points[:, None]-(a+t[:, :, None]*d), axis=2).min(1)
                near = np.flatnonzero((distances > 1e-6) & (distances < .5))
                if len(near):
                    k = near[np.argmax(distances[near])]
                    u, z = points[k]
                    region['bounds'] = [u-1.935, u+1.935, z-2.25, z+2.25]
                    region['detail'] = f'未采用的分支 {bid}；图窗中心处与主轨相距约 {distances[k]:.3f} m'
            chosen.append(region)
    return chosen


def draw_locations(windows, regions, destination):
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    if windows:
        ax.scatter([float(r['slice_key']) for r in windows], [(r['bounds'][2]+r['bounds'][3])/2 for r in windows],
            s=9, alpha=.28, color='#69788b', linewidths=0, label='自动筛出的待复核窗口')
    for number, region in enumerate(regions, 1):
        s, z = float(region['slice_key']), (region['bounds'][2]+region['bounds'][3])/2
        ax.scatter([s], [z], s=45, facecolor='white', edgecolor=COLORS['new'], linewidth=1.6, zorder=4,
            label='下方展示的代表案例' if number == 1 else None)
        offset = {'N00918': (-18, -20), 'N00918_JOIN': (10, 10), 'N00208': (-18, -20)}.get(region['region_id'], (7, 8))
        ax.annotate(str(number), xy=(s, z), xytext=offset, textcoords='offset points',
            color=COLORS['new'], fontsize=11, fontweight='bold', zorder=5)
    ax.set(xlabel='切面位置 s（m）', ylabel='局部窗口中心高程 z（m）', title='全区域待复核位置分布 · 编号对应下方局部案例')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.grid(alpha=.16)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.24), ncol=2, frameon=False)
    fig.subplots_adjust(left=.09, right=.98, top=.89, bottom=.24)
    fig.savefig(destination, dpi=180, facecolor='white')
    plt.close(fig)


def render_report(output):
    output = Path(output)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 11, 'axes.unicode_minus': False})
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    context_path = output/'review_context.json'
    context = json.loads(context_path.read_text(encoding='utf-8')) if context_path.exists() else {}
    index = json.loads((output/'result_index.json').read_text(encoding='utf-8'))
    with (output/'review_cache.pkl').open('rb') as stream:
        cache = pickle.load(stream)
    regions = choose_regions(cache, output, index)
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    draw_locations(cache['windows'], regions, figures/'00_problem_locations.png')
    counts = Counter(r['category'] for r in cache['windows'])
    lines = ['# 多分支拼接主轨：第一版识别结果与局部对比', '',
        '**本报告的右图来自本次实际计算并导出的主轨，不是原始候选分支或示意图。** 左图是修复前全切面只选一个分支的输出；193 个历史案例的分支编号已逐一与上一轮保存结果核对。它不与更早的 LOCC 提议混用。', '',
        '## 本次修复与结果范围', '',
        '旧实现先锁定一个全局分支，再把同 track 内候选不足两条当作整个求解流程结束。现在该分支只作为起始依据：每次检查主轨上下两端尚未覆盖的范围，在所有开放分支的未使用部分中寻找后续路径，优先交点、否则最短连接；接上后继续搜索。一个局部只有一条分支不影响后续区域的接续。内部折返按原始点序保留。', '',
        '不再设置“整个结果最多两条分支/一次换轨”的上限。同一原始分支允许贡献互不重叠的多个区段，但同一原始边区间不能重复经过。原始点和 FaceID 来源不被平滑或伪造；跨空隙的线段单独标记。', '',
        '本轮以你最新明确的“多个局部路径合成一个主轨”为准：原方案的局部换轨限制不再被当作整个切面的停止条件。代码审查还补上了终端已选折返保护，并将“连接次数”与“实际更换分支次数”分开统计。', '',
        f"本次处理 **{summary['slice_count']} 个切面**；**{summary['multi_branch_slices']} 个**形成多分支主轨，共 **{summary['junction_count']} 次**接续，其中 **{summary['switches']} 次**更换分支、**{summary['real_intersections']} 次**使用切面内真实交点。所有输出逐切面检查了坐标来源、线段连续性和无重复边区间。", '',
        f"开放分支的高程上下界均到达：{summary['full_open_z_extent_slices']} 个切面。此项只检查范围，并不证明选中的曲面正确，也不要求把所有平行分支都塞进一条线。", '',
        '本版只重算主轨识别阶段，复用同一冻结输入的跨切面支持证据；未重新生成 H 证据，也未运行缺失尾部推断。旧 red-group 生产入口和 GUI 尚未替换为本入口。实际新结果保存为 main_tracks.pkl，入口为 scripts/99_experiments/recognize_main_tracks.py。', '',
        '## 全区域中仍需复核的位置', '',
        '以下规则应用于全部切面，不限于旧的 193 个窗口。它们定位风险或疑点，尚无人工真值，不能称为错误率。相邻切面可能重复看到同一处问题，因此“窗口数”也不等于独立缺陷数。', '',
        '| 检查类型 | 局部窗口数 | 涉及切面数 | 需要判断什么 |', '|---|---:|---:|---|']
    descriptions = dict(LONG_CONNECTOR='跨空隙补接是否合理；1 cm 是复核阈值，不再是停止寻找后续路径的门槛。',
        SHARP_JUNCTION='接上以后是否拐向了错误的支线。', COMPETING_BRANCH='附近未采用的曲线段是否才是目标曲面，含已部分采用分支的剩余区段。',
        UNCOVERED_EXTENT='已有开放曲线的高度边界为何没有进入主轨。', HISTORICAL_LOCAL_MISS='过去提出的局部疑点是否仍被漏掉或改走了另一条曲面。')
    for k, label in LABELS.items():
        lines.append(f"| {label} | {counts[k]} | {len({r['slice_key'] for r in cache['windows'] if r['category'] == k})} | {descriptions[k]} |")
    lines += ['', '![全区域疑点定位](figures/00_problem_locations.png)', '',
        '上图只用于定位：灰点为自动筛出的疑点窗口中心，蓝色编号对应下方案例。同一位置的不同径向 u 或不同问题类型可能叠在一起；点密度不是错误率。实际曲线走向请看下方局部对比。', '',
        '完整坐标清单见 [problem_regions.csv](problem_regions.csv)，每行包含切面 s、局部 u/z 范围和问题类型。选例规则固定为：保留三个已讨论案例和 N00918 的连接点，再从每种仍有问题的类型中选择严重度最高的一例，避免只展示改善案例。', '',
        f"在原来固定的 193 个窗口内，输出完全缺席由 **{summary['old_local_absent']}** 例变为 **{summary['new_local_absent']}** 例；观测高程覆盖不足 95% 由 **{summary['old_local_below95']}** 例变为 **{summary['new_local_below95']}** 例。统计继续使用旧窗口，绘图范围扩大，二者不混用。这些仍是覆盖诊断，不是识别准确率。", '',
        '这里的“观测覆盖”只计算蓝色原始观测段，不计算红色补接段，也不计算走出旧狭窄窗口的曲线。因此剩余 9 例不能直接叫作 9 个漏检：例如 N00121 的原始空隙已经补接，N00918 的折返仍然保留。下面分别说明。', '',
        '## 怎样看下面的图', '',
        '每张图只回答一个问题：在同一个局部，旧输出走到了哪里，新主轨改成了怎样的走向。左右图使用同样的坐标范围和等比例坐标轴；主对比图高程视野至少 4.5 m，便于看清曲线在窗口外的接续。N00918 另附一张 1.3 m 高的连接细节图，以辨认厘米级接缝。横轴 u 是切面内径向偏移，纵轴 z 是高程，单位均为米。', '',
        '- **浅灰细线：原始切面曲线。** 它提供背景，不表示全部都应该被选中。',
        '- **左图棕色实线：修复前的单分支输出。** 没有棕线时会直接写“旧结果未进入这个局部”。',
        '- **右图蓝色实线：本版主轨采用的原始观测区段。** 沿它观察是否连续、是否保留正确的折返，以及是否走进错误分支。',
        '- **红色虚线只在存在可见空隙连接时出现。** 这段是补接，不是原始观测；需要重点检查。其余图只有三项图例。', '',
        '右图箭头上的“8 → 10”等编号表示当前切面内从哪个原始分支转到哪个分支。分支编号只在本切面内有效。标注“交点接续”表示两条原始切面曲线在该点相交；标注“补接”及长度表示两曲线之间有真实空隙，厘米级用 cm，超过一米用 m。H 点、邻剖面轨道、旧端点、虚拟节点等辅助对象不再叠加到主对比图中。', '',
        '原始曲线很多时，先沿蓝线看连续走向，再看灰线判断旁边有没有被舍弃的更合理路径，不需要逐个辨认所有灰色分支。', '']
    html_cards = []
    for n, region in enumerate(regions, 1):
        payload = load_result(output, index, region['slice_key'])
        filename = f'{n:02d}_{region["region_id"]}.png'
        draw_case(payload, region, figures/filename)
        label = LABELS.get(region['category'], '已讨论案例的修复对比' if region['category'] == 'KNOWN_CASE' else '实际分支连接点')
        detail = region['detail']
        if region['category'] in LABELS:
            detail = descriptions[region['category']]+f" 定位指标：{region['severity']:.4f}。"
            if region.get('junction_index') is not None:
                junction = payload['result']['junctions'][region['junction_index']]
                detail += (f" 当前接续为分支 {junction['from_branch_id']} → {junction['to_branch_id']}，"
                    f"连接长 {junction['xyz_distance_m']:.6f} m，两侧切向夹角 {junction['tangent_turn_deg']:.2f}°。")
            elif region['category'] == 'COMPETING_BRANCH':
                detail += ' '+region['detail']+'；这里的长度是未采用区段总长，图中只展示其中一个局部。'
        if region['region_id'] == 'N00918':
            detail += '。右图沿分支 8 保留了左侧折返；本局部不需要把原本连续的上下两段重新拉直拼接。57.3% 仍按原来仅约 0.263 m 宽的窗口计算，曲线走出该窗口的部分不计入，因此不能把它解读成删掉了 42.7% 的折返。完整路径顺序为 3 → 8 → 10 → 0 → 10；重复的分支 10 使用互不重叠区段。'
        if region['region_id'] == 'N00918_JOIN':
            j = payload['result']['junctions'][region['junction_index']]
            detail += (f" 此处实际连接长度为 {j['distance_m']*100:.3f} cm。保留终端折返后，当前方案在这里采用最短有效补接；红虚线不是原始观测。"
                if j['distance_m'] >= 1e-9 else ' 蓝线通过原始交点换轨，未增加跨空隙直线。')
        if region['region_id'] == 'N00532':
            detail += '。重点看 z≈1369.9 m 的箭头：主轨由分支 2 在原始交点进入分支 10，图中没有跨空隙补接。'
        if region['region_id'] == 'N00208':
            detail += '。重点看 z≈1386.4 m 左侧回折：右图保留沿原始曲线绕回的走向，没有用一根直线把上下部分截短。'
        if region['category'] == 'LONG_CONNECTOR':
            detail += ' 图中长红虚线跨过没有观测曲线支撑的区域，这是本版最明显的高风险结果，不能因为两端接通就接受为正确曲面。该例为容纳整段空隙扩大了窗口，属于局部对比图的尺度例外。'
        if region['category'] == 'SHARP_JUNCTION':
            detail += ' 重点看 z≈1380.2 m 的红线及上下两端蓝线：两侧原始切向几乎相反。179° 衡量的是两侧原始线段方向，不是红线与某一侧的夹角。'
        if region['region_id'] == 'N00121':
            detail += ' 原来两个端点正是红色 2.253 cm 补接的两端。这段现在拓扑上已连通，但原始网格仍没有该段观测，所以“观测覆盖”仍为 0%；这不是本版没有进入该处。人工需判断这次补接是否合理，右上灰色小闭环也不自动并入主轨。'
        bounds = region['bounds']
        lines += [f'## {n}. {region["region_id"]}：{label}', '',
            f"切面 **s={float(region['slice_key']):.2f} m**；u={bounds[0]:.2f}～{bounds[1]:.2f} m，z={bounds[2]:.2f}～{bounds[3]:.2f} m。", '',
            detail, '', f'![{region["region_id"]}](figures/{filename})', '']
        html_cards.append(f'<section><h2>{n}. {html.escape(region["region_id"])} · {html.escape(label)}</h2><p>{html.escape(detail)}</p><img src="figures/{filename}" alt="局部对比"></section>')
    lines += ['## 算法仍有的边界', '',
        '本版采用逐步延伸和局部最短连接，尚不是所有可能组合上的全局最优路径搜索。前进范围目前由 Z 上下界判断，同高程范围内的纯横向延伸尚未实现。几何相交或距离最短不能单独证明曲面身份；横向支持用于起始分支选择，后续接续仍需要利用这些代表案例核对。附近完全重叠高度范围内的竞争路径不会因为“更平直”而自动替换当前观测折返，这类问题单列为待复核。竞争区段筛选条件是长度至少 0.5 m，且至少一半抽样点位于主轨附近 0.5 m 内，属于诊断启发规则。', '',
        f"最长连接为 {summary['longest_connector_m']:.6f} m；超过 1 cm 的连接共 {summary['large_connectors']} 处，全部保留在问题清单中，没有据此宣称识别完全正确。", '',
        '## 验证与文件', '',
        context.get('verification_note', '未附独立回归测试执行记录。')+' 数值运行对全部导出结果检查了原始坐标/FaceID 来源、UZ/XYZ 连续性以及观测区间不重复。图中的曲面身份仍需人工确认。', '',
        context.get('previous_attempt_note', '本报告仅使用当前输出目录内的数值结果。'), '',
        f"本次加载、几何重建、识别、校验和导出的实测时间为 {summary['total_wall_seconds']:.2f} s（绘图另计）。复用了跨切面证据，因此不能与上次包含 H 图重建的整套耗时直接比较。阶段明细见 summary.json。", '',
        '- main_tracks.pkl：顺序存储的实际路径、逐边来源和绘图所需原始几何。',
        '- result_index.json：按切面定位 pickle 记录的字节偏移。',
        '- slice_summary.csv / junctions.csv：全区域主轨及每一次连接的数值记录。',
        '- problem_regions.csv：全区域待复核局部位置清单。',
        '- historical_local_comparison.csv：193 个旧窗口的同口径前后对比。',
        '- figures/：本报告使用的少要素、大局部对比图。', '',
        '## 本版主轨阶段的实测性能', '',
        '| 阶段 | 实测墙钟时间（s） | 平均占用逻辑核心数 | 峰值进程内存（GiB） |',
        '|---|---:|---:|---:|']
    stage_names = dict(load_frozen_inputs_and_evidence='加载冻结输入与既有支持证据',
        rebuild_full_observed_inventory='重建全区域原始分支',
        recognize_validate_export_all_main_tracks='主轨计算、来源校验、问题筛选与导出')
    for stage in summary['stage_performance']:
        lines.append(f"| {stage_names.get(stage['stage'], stage['stage'])} | {stage['wall_seconds']:.2f} | {stage['mean_logical_cpu_equivalents']:.2f} | {stage['peak_tree_rss_bytes']/2**30:.3f} |")
    lines += ['', '这里测的是本版主轨阶段，不包含旧生产 red-group 识别，也不包含重建 H 支持图；使用 4 个工作线程。并发任务的单切面耗时存在重叠，不能把它们相加当作墙钟时间。', '',
        context.get('profile_note', '本次未附额外的单切面分析器记录。'), '']
    lines += ['## 修改文件', '', '| 文件 | 本轮变更 |', '|---|---|']
    for name, description in [
        ('scripts/04_structure_recognition/main_track_assembly.py', '新增区域接续、最短有效连接、折返保护和区间防重用。'),
        ('scripts/04_structure_recognition/dominant_observed_branch.py', '单分支选择后进入完整主轨组装，移除误导性的完整识别状态。'),
        ('scripts/04_structure_recognition/surface_track_handoff.py', '明确局部无换轨候选不代表整个切面结束。'),
        ('scripts/99_experiments/surface_track_validation.py', '允许多次接续，检查来源、连续性和无重复边区间，修正统计。'),
        ('scripts/99_experiments/recognize_main_tracks.py', '全区域实际主轨导出、问题区域筛选和运行性能记录。'),
        ('scripts/99_experiments/main_track_comparison_report.py', '大局部、少要素对比图与本报告。'),
        ('tests/test_main_track_assembly.py', '新增 7 个直接针对本轮问题的回归用例。'),
        ('tests/test_dominant_review_fixes.py', '旧身份保留用例适配多分支接续的最新要求。')]:
        path = (Path(__file__).resolve().parents[2]/name).as_posix()
        lines.append(f'| [{Path(name).name}]({path}) | {description} |')
    lines += ['', '没有提交或推送，也没有改动冻结原始数据。', '']
    (output/'COMPARISON_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    intro = '<h1>多分支拼接主轨 · 第一版</h1><p>局部对比图：灰线为原始曲线，棕线为修复前输出，蓝线为本版主轨；红虚线表示跨空隙补接，有才显示。左右范围和比例相同。</p><p>请重点检查：蓝线是否选对曲面、折返是否合理、连接处是否进入错误支线。完整范围、风险清单、统计口径与读图说明见 <a href="COMPARISON_REPORT.md">总对比分析报告</a>。</p><section><h2>全区域疑点定位</h2><p>下图仅定位窗口中心，编号对应后面的局部案例；灰点密度不是错误率。</p><img src="figures/00_problem_locations.png" alt="疑点定位"></section>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>主轨局部对比</title><style>body{max-width:1280px;margin:36px auto;background:#f5f7fa;font:17px/1.7 "Microsoft YaHei",sans-serif;color:#243144;padding:0 24px}section{background:white;margin:30px 0;padding:24px;border-radius:10px}img{width:100%;height:auto}h1,h2{line-height:1.35}a{color:#1365ad}</style>'+intro+''.join(html_cards)+'</html>'
    (output/'comparison_gallery.html').write_text(page, encoding='utf-8')
    (output/'representative_regions.json').write_text(json.dumps(regions, ensure_ascii=False, indent=2), encoding='utf-8')
    return regions


if __name__ == '__main__':
    import sys
    render_report(Path(sys.argv[1]))
