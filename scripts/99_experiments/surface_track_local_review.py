"""Display-only local diagnostics; ROI never feeds surface identity selection."""
from collections import Counter, defaultdict
import csv
import pickle
import numpy as np


def clipped_segments(points, bounds):
    """Clip polyline segments to a rectangle, including crossings without vertices."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return np.empty((0, 2, 2))
    a, delta = points[:-1], np.diff(points, axis=0)
    low, high = np.array([bounds[0], bounds[2]]), np.array([bounds[1], bounds[3]])
    moving = np.abs(delta) > 1e-15
    t0 = np.divide(low-a, delta, out=np.full_like(a, -np.inf), where=moving)
    t1 = np.divide(high-a, delta, out=np.full_like(a, np.inf), where=moving)
    enter = np.maximum(0., np.minimum(t0, t1).max(axis=1))
    leave = np.minimum(1., np.maximum(t0, t1).min(axis=1))
    parallel_outside = ((~moving) & ((a < low) | (a > high))).any(axis=1)
    valid = (leave > enter+1e-12) & ~parallel_outside
    return np.stack((a[valid]+enter[valid, None]*delta[valid],
                     a[valid]+leave[valid, None]*delta[valid]), axis=1)


def local_view_bounds(case, old_curve):
    """Use historical case Z extent and local previous geometry, never full branch."""
    endpoints = np.asarray([case['lower_uz'], case['upper_uz']], dtype=float)
    zl, zh = endpoints[:, 1].min(), endpoints[:, 1].max()
    local = clipped_segments(old_curve, (-np.inf, np.inf, zl, zh))
    us = np.r_[endpoints[:, 0], local[:, :, 0].ravel()]
    upad = max(.02, float(np.ptp(us))*.15)
    zpad = max(.015, (zh-zl)*.1)
    return float(us.min()-upad), float(us.max()+upad), float(zl-zpad), float(zh+zpad)


def vertical_coverage(segments, zl, zh):
    if not len(segments) or zh <= zl:
        return 0.
    intervals = sorted((max(zl, min(s[:, 1])), min(zh, max(s[:, 1]))) for s in segments)
    length, end = 0., zl
    for low, high in intervals:
        if high > max(low, end):
            length += high-max(low, end)
            end = high
    return min(1., length/(zh-zl))


def _inside(hits, bounds):
    ul, uh, zl, zh = bounds
    return [h for h in hits if ul <= h['u'] <= uh and zl <= h['z'] <= zh]


def choose_local_focus(rows):
    """Historical anchors, worst per category, and distinct diagnostic modes."""
    ids = {r['candidate_id'] for r in rows}
    focus = [cid for cid in ['N00208', 'N00918', 'N00532', 'N00804', 'N00185', 'N00145'] if cid in ids]
    for category in 'ABCDE':
        candidates = [r for r in rows if r['old_category'] == category]
        candidates.sort(key=lambda r: (not r['selected_absent_from_local_view'], r['local_selected_z_coverage'],
                                        not r['ambiguous'], r['candidate_id']))
        if candidates and candidates[0]['candidate_id'] not in focus:
            focus.append(candidates[0]['candidate_id'])
    modes = [lambda r: not r['selected_absent_from_local_view'] and r['local_selected_z_coverage'] < .95,
             lambda r: r['local_selected_z_coverage'] >= .95 and r['ambiguous'],
             lambda r: r['old_category'] == 'F' and r['local_selected_z_coverage'] >= .95 and not r['ambiguous']]
    for condition in modes:
        candidates = sorted((r for r in rows if condition(r)), key=lambda r: r['candidate_id'])
        if candidates and candidates[0]['candidate_id'] not in focus:
            focus.append(candidates[0]['candidate_id'])
    return focus


def prepare_local_review(data, output):
    graph = data['graph']
    target_s = {float(c['slice_key']) for c in data['cases']}
    raw, linked = defaultdict(list), defaultdict(list)
    for h in graph['observations']:
        if h['s'] in target_s:
            raw[h['s']].append(h)
    for h in graph['linked_observations']:
        if h['s'] in target_s:
            linked[h['s']].append(h)
    old_rows = {r['candidate_id']: r for r in data['previous']['rows']}
    plots, rows = {}, []
    for case in data['cases']:
        cid, key = case['candidate_id'], case['slice_key']
        s, result = float(key), data['results'][cid]
        old = data['previous']['results'][cid]
        selected, tid = result['selection']['selected'], result['surface_track_id']
        bounds = local_view_bounds(case, old['curve_uz'])
        endpoints = np.asarray([case['lower_uz'], case['upper_uz']])
        zl, zh = sorted(endpoints[:, 1])
        background = [(b['branch_id'], clipped_segments(b['points_uz'], bounds)) for b in data['branches'][key]]
        background = [(bid, seg) for bid, seg in background if len(seg)]
        paths = defaultdict(list)
        for edge in result['path_edges']:
            kind = 'observed' if edge['source'].startswith('OBSERVED') else 'switch' if edge['source'] == 'TOPOLOGY_SWITCH' else 'inferred'
            paths[kind].extend(clipped_segments(edge['points_uz'], bounds))
        paths = {k: np.asarray(v).reshape(-1, 2, 2) for k, v in paths.items()}
        observed = paths.get('observed', np.empty((0, 2, 2)))
        neighbors = []
        for near_key in data['frozen'].neighbors(key):
            if near_key == key:
                continue
            for b in data['branches'][near_key]:
                if tid and graph['membership'][(float(near_key), b['branch_id'])] == tid:
                    neighbors.extend(clipped_segments(b['points_uz'], bounds))
        raw_local = _inside(raw[s], bounds)
        linked_local = _inside(linked[s], bounds)
        selected_h = [h for h in linked_local if tid and any(graph['membership'][n] == tid for n in h['nodes'])]
        old_segments = clipped_segments(old['curve_uz'], bounds)
        old_paths = defaultdict(list)
        for edge in old['path_edges']:
            kind = 'observed' if edge['source'].startswith('OBSERVED') else 'switch' if edge['source'] == 'TOPOLOGY_SWITCH' else 'inferred'
            old_paths[kind].extend(clipped_segments(edge['points_uz'], bounds))
        local_coverage = vertical_coverage(observed, zl, zh)
        row = dict(candidate_id=cid, old_category=old_rows[cid]['old_category'], slice_key=key,
            status=result['status'], selected_track=tid, selected_branch=selected['branch_id'] if selected else None,
            ambiguous=result['selection']['ambiguous'], stable=bool(selected and selected['track_stable']),
            u_min=bounds[0], u_max=bounds[1], z_min=bounds[2], z_max=bounds[3],
            local_observed_branch_count=len(background), local_selected_segment_count=len(observed),
            local_selected_length_m=float(np.linalg.norm(np.diff(observed, axis=1), axis=2).sum()),
            local_selected_z_coverage=local_coverage, previous_local_z_coverage=vertical_coverage(old_segments, zl, zh),
            selected_absent_from_local_view=not any(len(v) for v in paths.values()),
            raw_H_in_view=len(raw_local), selected_track_linked_H_in_view=len(selected_h),
            any_track_linked_H_in_view=len(linked_local), branch_switch_count=result['branch_switch_count'],
            inferred_length_m=result['inferred_length_m'])
        rows.append(row)
        junction = result.get('junction')
        plots[cid] = dict(row=row, bounds=bounds, endpoints=endpoints, background=background,
            old_segments=old_segments, old_status=old['status'], old_branch=old_rows[cid]['dominant_branch_id'],
            old_paths={k: np.asarray(v).reshape(-1, 2, 2) for k, v in old_paths.items()},
            paths=paths, neighbors=np.asarray(neighbors).reshape(-1, 2, 2),
            raw_H=np.array([[h['u'], h['z']] for h in raw_local]).reshape(-1, 2),
            selected_H=np.array([[h['u'], h['z']] for h in selected_h]).reshape(-1, 2),
            junction_points=np.asarray([junction['a_point_uz'], junction['b_point_uz']]) if junction else np.empty((0, 2)),
            crossover_interval=junction.get('confidence_crossover_interval') if junction else None)
    focus = choose_local_focus(rows)
    data['local_plot_cases'], data['local_rows'], data['local_focus'] = plots, rows, focus
    with (output/'local_case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Small display-only cache: re-rendering never needs another numerical audit.
    with (output/'local_review_checkpoint.pkl').open('xb') as stream:
        pickle.dump(dict(local_plot_cases=plots, local_rows=rows, local_focus=focus,
                         representatives=data['representatives']), stream)


def write_local_review(data, output, render_case):
    rows = data['local_rows']
    counts = Counter(r['status'] for r in rows)
    lines = ['# 局部问题代表案例', '',
        '固定保留历史 A–F 六个锚点，再按预先固定的规则从 A–E 每类选择局部表现最弱的一例：优先当前路径在窗口内完全缺席，再按局部 Z 覆盖率升序、歧义优先、案例编号排序。重复案例只展示一次。F 为历史对照，不预设为正确真值。', '',
        '为避免所有重点案例重复展示同一种失败，再按案例编号各补一例：路径触及视窗但覆盖不足、覆盖完整但身份歧义、F 类覆盖完整且未标歧义的对照。没有对应样本就不生成该类别。', '',
        '左右图使用完全相同的局部坐标范围：原案例端点高程范围及上一轮结果在该高度内的 U 范围，加少量边距。旧端点仅定位图审区域，绝不进入当前 track 选择。完整分支超出窗口的部分不扩张坐标轴，也不改变求解结果。', '',
        '蓝色为当前实际保留观测，灰色为全部局部原始分支，浅蓝色为同 track 邻剖面；浅绿叉为 raw H，深绿圆点才是当前 track 的 linked H。黑色端点只用于历史案例定位。左图棕线为上一轮观测输出，红虚线为未采纳的推断提议；上一轮输出/提议覆盖率不等于 observed 覆盖率或人工真值。', '',
        f'193 个历史案例中：当前路径完全不进入局部窗口 {sum(r["selected_absent_from_local_view"] for r in rows)} 例；局部 Z 覆盖不足 95% 的 {sum(r["local_selected_z_coverage"] < .95 for r in rows)} 例；标记歧义 {sum(r["ambiguous"] for r in rows)} 例。', '',
        f'求解状态计数：{dict(counts)}。这些是历史候选样本的诊断统计，不能当作全模型错误率或准确率。', '',
        '**覆盖率定义**：只累计显示 U 范围内、原案例端点 Z 区间内的当前 observed 线段 Z 区间并集；折返和重叠不重复计算。100% 只说明高度覆盖，不证明曲面身份、连接正确或缺失恢复成功。', '',
        '|案例|类别|原始局部分支|当前局部覆盖|上一轮输出/提议覆盖|当前 linked H|歧义|',
        '|---|---|---:|---:|---:|---:|---|']
    for cid in data['local_focus']:
        r = data['local_plot_cases'][cid]['row']
        lines.append(f'|{cid}|{r["old_category"]}|{r["local_observed_branch_count"]}|{r["local_selected_z_coverage"]:.1%}|{r["previous_local_z_coverage"]:.1%}|{r["selected_track_linked_H_in_view"]}|{r["ambiguous"]}|')
    for cid in data['local_focus']:
        r = data['local_plot_cases'][cid]['row']
        path = render_case(data, cid, output)
        if r['selected_absent_from_local_view']:
            finding = '当前选择的整条轨道未进入该历史案例窗口，局部原始几何仍存在于灰线中。全剖面只选一个 dominant track 不能自动解决此处的局部目标识别；不能据整条轨道被保留而宣布本案例修复。'
        elif r['local_selected_z_coverage'] == 0:
            finding = '当前路径仅触及加了边距的显示窗口，原案例端点之间的高度区间仍完全没有当前观测覆盖。视窗里出现蓝线或深绿支持点不代表该局部目标已经得到处理。'
        elif r['local_selected_z_coverage'] < .95:
            finding = '当前路径在本案例只保留了部分局部观测，高度覆盖仍有缺口。请对照灰色分支核对是否缺少局部身份选择或必要的换轨；本次没有把灰线、邻剖面或 H 点伪装成已恢复结果。'
        elif r['ambiguous']:
            finding = '局部高度覆盖较完整，但 track 仍标记歧义。局部线形是否正确需要核对原始网格；全局轨道的冲突状态可能影响本来连续的局部片段，不能以覆盖率替代身份判断。'
        else:
            finding = '该窗口内保留了较完整的实际观测，可作为局部覆盖对照。仍需核对折返是否真实、分支身份是否正确；此结果不是缺失补全能力的证明。'
        if r['raw_H_in_view'] == 0:
            finding += ' 当前固定水平切片在此视窗没有 H 观测，边界或无数据区域不宜自动补齐。'
        elif r['selected_track_linked_H_in_view'] == 0:
            finding += ' 此处虽有 raw H，但没有归属于当前所选 track 的 linked H，不能把浅绿点称为该轨道的支持。'
        if data['local_plot_cases'][cid]['old_status'] == 'INFERRED_REVIEW_PROPOSAL':
            finding += ' 左图是上一轮未采纳的推断提议（红虚线），不是实际观测或人工真值。'
        lines += ['', f'## {r["old_category"]} · {cid}', '', f'![{cid}]({path.as_posix()})', '', finding, '',
            f'当前状态 `{r["status"]}`；track `{r["selected_track"]}` / branch `{r["selected_branch"]}`；局部保留长度 {r["local_selected_length_m"]:.4f} m；换轨 {r["branch_switch_count"]}；整条输出推断长度 {r["inferred_length_m"]:.4f} m。']
    lines += ['', '## 当前算法局限', '',
        '1. 当前决策以完整剖面的全局 dominant track 为单位，而历史案例定位的是某个局部区域。同一剖面多个局部目标会共享一次决策，因此会出现全局状态通过、局部窗口却没有输出的情况。',
        '2. 整轨歧义和局部证据没有完全分离；自然案例仍需人工真值确认，不能仅靠合成测试通过判定选面正确。',
        '3. 当前同 track 换轨及缺失补全非常保守。自然样本的换轨和推断为零时，应报告未覆盖的能力，不能解释为不存在缺口。',
        '4. 原案例端点是图审定位依据，不是独立人工标注真值。局部指标用于找问题，不能据此擅自改变算法选择规则。', '']
    (output/'LOCAL_FAILURE_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
