"""Missing-route causal report plus unscored, branch-only review gallery."""
from pathlib import Path
from collections import Counter
import csv,json,pickle,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from diagnose_missing_routes import latest,SOURCE
from validate_physical_topology import ROOT,load_inventory,csv_rows
from census_directional_consistency import code_hashes,save_json
from render_facetrack_branch_gallery import crop

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})


def tab(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
      ['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])+'\n'


def qualify_candidate_gates(out):
    """Complete stage attribution without repeating any nearest-join searches."""
    import main_track_assembly as a
    from branch_absolute_core import branch_metrics,local_edge_scale,contribution_metrics
    from collections import defaultdict
    details=json.loads((out/'all_large_missing_junction_details.json').read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for row in details:groups[row['s']].append(row)
    flat=[]
    for key,rs in groups.items():
        r=pickle.load((out/'routes'/f'{key}.pkl').open('rb'))['result'];bs=load_inventory(SOURCE,[key])[key]
        by={b['branch_id']:b for b in bs};scale=local_edge_scale(bs);metrics={i:branch_metrics(b,scale) for i,b in by.items()}
        pieces=a._remaining([b for b in bs if metrics[b['branch_id']]['MBG_pass']],r['path_edges']);required=r['route_z_extent'];contexts=[]
        for direction in ('lower','upper'):
            if (required[0]<=r['candidate_z_extent'][0]+1e-6 if direction=='lower' else required[1]>=r['candidate_z_extent'][1]-1e-6):continue
            for piece in pieces:
                p=a._points(piece);bid=piece[0]['branch_id'];low,high=float(p[:,1].min()),float(p[:,1].max())
                if (low>=required[0]-1e-6 if direction=='lower' else high<=required[1]+1e-6):continue
                gap=max(0,required[0]-high) if direction=='lower' else max(0,low-required[1])
                if gap>.010000001:continue
                target=(low,required[0]) if direction=='lower' else (required[1],high)
                fraction=branch_metrics(by[bid],scale,target_z=target)['target_region_in_ASC_fraction']
                retained=contribution_metrics(by[bid],piece,metrics[bid])['contribution_pass']
                contexts.append((direction,bid,fraction,retained,[low,high]))
        assert len(contexts)==len(rs),(key,'probe context mismatch')
        for row,(direction,bid,fraction,retained,z_range) in zip(rs,contexts):
            assert (row['frontier'],row['candidate_branch_id'])==(direction,bid)
            locked,unlocked=row['variants']['True'],row['variants']['False'];pre=retained and fraction>0
            flat.append(dict(s=key,frontier=direction,from_branch_id=row['from_branch_id'],candidate_branch_id=bid,
              actually_chosen_candidate=row['actually_chosen_candidate'],candidate_preselection_pass=pre,
              extending_region_ASC_fraction=fraction,piece_z_range=z_range,
              blocked_only_by_ordinary_fold_lock=pre and row['blocked_only_by_ordinary_fold_lock'],
              locked_passes_other_gates=pre and locked['passes_other_geometric_gates'],
              unlocked_passes_other_gates=pre and unlocked['passes_other_geometric_gates'],
              locked_distance_m=locked['join']['xyz_distance_m'] if locked['join'] else None,
              unlocked_distance_m=unlocked['join']['xyz_distance_m'] if unlocked['join'] else None,
              lock_edge_drop_m=row['lock_edge_drop_m']))
    csv_rows(out/'gate_complete_junction_probes.csv',flat)
    save_json(out/'gate_complete_probe_summary.json',dict(candidate_pieces=len(flat),
      preselection_rejected_count=sum(not r['candidate_preselection_pass'] for r in flat),
      fold_lock_counterexample_slices=len({r['s'] for r in flat if r['blocked_only_by_ordinary_fold_lock']}),
      selected_branch_fold_counterexample_slices=len({r['s'] for r in flat if r['blocked_only_by_ordinary_fold_lock'] and r['actually_chosen_candidate']}),
      unselected_candidate_passes_with_current_fold_lock_slices=len({r['s'] for r in flat if not r['actually_chosen_candidate'] and r['locked_passes_other_gates']})))
    return flat


def roots_figure(out,details):
    fig,axs=plt.subplots(1,2,figsize=(14,7))
    selected=[r for r in details if r['actually_chosen_candidate'] and r['blocked_only_by_ordinary_fold_lock']]
    for ax,r in zip(axs,selected):
        key=r['s'];bs=load_inventory(SOURCE,[key])[key];by={b['branch_id']:b for b in bs}
        blocked=r['variants']['True']['join'];open_join=r['variants']['False']['join']
        p=np.array(open_join['a_point_uz']);q=np.array(blocked['a_point_uz']);f=np.array(r['lock_edge_points_uz'])
        points=np.vstack([p,q,f]);lo=points.min(axis=0)-[.55,.4];hi=points.max(axis=0)+[.55,.4]
        bounds=[lo[0],hi[0],lo[1],hi[1]]
        for bid,color in [(r['from_branch_id'],'#1769aa'),(r['candidate_branch_id'],'#ad6737')]:
            for e in by[bid]['records']:
                line=crop(e['points_uz'],bounds)
                if line is not None:ax.plot(line[:,0],line[:,1],color=color,lw=1.5)
        ax.scatter(*p,marker='*',s=155,color='#138b61',zorder=5)
        ax.scatter(*f.mean(axis=0),marker='x',s=55,color='#b51e2c',zorder=5)
        ax.annotate(f"被锁排除的交点\n间距 {open_join['xyz_distance_m']*1e6:.3f} μm",p,xytext=(-115,-43) if key=='122.95' else (12,-43),textcoords='offset points',fontsize=10,
            arrowprops={'arrowstyle':'-','color':'#138b61'})
        ax.annotate(f"最后一段反向边\nΔz = {r['lock_edge_drop_m']*1000:.6f} mm",f.mean(axis=0),xytext=(12,18),textcoords='offset points',fontsize=10,
            arrowprops={'arrowstyle':'-','color':'#b51e2c'})
        ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.grid(alpha=.18);ax.ticklabel_format(useOffset=False,style='plain')
        ax.set_xlabel('径向偏移 u（m）');ax.set_ylabel('高程 z（m）')
        ax.set_title(f"s = {key} m｜B{r['from_branch_id']} → B{r['candidate_branch_id']}\n保留折返锁时最短候选：{blocked['xyz_distance_m']*1000:.3f} mm")
    fig.suptitle('缺轨真正发生的位置：早期交点被折返锁挡住',fontsize=17)
    fig.legend(handles=[Line2D([],[],color='#1769aa',label='当前末端所属原始分支'),Line2D([],[],color='#ad6737',label='候选原始分支'),
      Line2D([],[],marker='*',color='#138b61',linestyle='',label='受锁影响的交点'),Line2D([],[],marker='x',color='#b51e2c',linestyle='',label='锁定位置')],ncol=4,loc='lower center',bbox_to_anchor=(.5,.035))
    fig.subplots_adjust(left=.075,right=.975,top=.81,bottom=.17,wspace=.23)
    fig.savefig(out/'figures/junction_root_causes.png',dpi=160);plt.close(fig)


def main(out):
    def js(name):return json.loads((out/name).read_text(encoding='utf-8'))
    def csvread(name):
        with (out/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    control=js('junction_controls.json');detail=js('junction_control_details.json');gallery=js('branch_gallery_index.json')
    roots_figure(out,detail)
    facts=[]
    for c in control:
        r=next(x for x in detail if x['s']==c['s'] and x['actually_chosen_candidate']);e=r['internal_at_unlocked_intersection']
        facts.append([c['s'],f"B{r['from_branch_id']}→B{r['candidate_branch_id']}",f"{r['lock_edge_drop_m']*1000:.6f}",
          f"{r['variants']['True']['join']['xyz_distance_m']*1000:.6f}",f"{r['variants']['False']['join']['xyz_distance_m']*1e6:.6f}",
          f"{c['baseline']['route_z_extent'][1]:.3f} → {c['unlocked_counterfactual']['route_z_extent'][1]:.3f}"])
    root_table=tab(['s（m）','受阻续接','锁定反向高差（mm）','有锁时距离（mm）','无锁时距离（μm）','路线最高 Z，对照前→后（m）'],facts)
    support_table=tab(['s','旧分支后续延展 Z（m）','新分支后续延展 Z（m）','旧/新横向支持跨度（m）','要求新支持超过（m）'],[
      (r['s'],f"{r['internal_at_unlocked_intersection']['a_directional']['Z_forward_new']:.3f}",
       f"{r['internal_at_unlocked_intersection']['b_directional']['Z_forward_new']:.3f}",
       f"{r['internal_at_unlocked_intersection']['a_support']['support_s_span']:.3f} / {r['internal_at_unlocked_intersection']['b_support']['support_s_span']:.3f}",
       f"{r['internal_at_unlocked_intersection']['stronger_threshold']:.3f}") for r in detail if 'internal_at_unlocked_intersection' in r])
    report=[
      '# 缺失主轨原因追踪与同 FaceTrack 分支审核',
      '**本轮把两个问题分开了：缺轨部分追到具体控制条件；FaceTrack 部分只展示匹配后的原始分支，等待人工审核，不用主轨结果或 TC 公式替代分支一致性验证。**',
      '## 缺失主轨的共同原因：已经完成因果对照的两例',
      '122.95、116.25 米的缺轨不是因为上部原始分支不存在，也不是 MBG 不合格。它们在约 1383–1384 m 的首次上接已经停止，所以在 1420/1442 m 看到了后续主轨整段缺失。两个案例共有下面这条控制链：',
      '1. 原始候选存在，MBG 和可靠核心检查通过。\n2. 普通续接 `_nearest_join` 默认 `protect_folds=True`，上接必须等到当前末端的**最后一段反向 Z 边之后**才能连接。任何超过 1e-9 m 的反向 Z 都可形成这道锁，没有先验证这个折返是否应受邻线一致性保护。\n3. 更早的近乎零距离交点因此被排除，只剩 12.21/271.55 mm 的连接，随后被未改动的 10 mm 门限拒绝。\n4. 内部提前换轨要先满足 DIRECTIONAL_TAIL，而当前规则又要求新分支的横向支持显著强于旧分支；在可用交点上，两例都未满足，所以内部换轨没有解除普通折返锁的后果。',
      root_table,
      '122.95 的最后一段反向高差仅 **7 微米**，但作用是锁住之前整段路径；116.25 的对应反向高差约 **12.961 mm**。这里没有断言这些反向边都是噪声，问题是普通续接一律保护它们，而合法的早期交点必须等待另一套更严格规则才能使用。',
      '![普通续接折返锁的局部证据](figures/junction_root_causes.png)',
      '图中是发生续接失败的原始分支局部，不是最终缺轨窗口。蓝/棕是两条原始分支，绿星是被锁排除的交点，红叉标识建立折返锁的位置。没有绘制或写入任何强制补线。',
      '## 为什么内部换轨也没有接住',
      support_table,
      '两个交点处新分支可以继续延伸约 29–30 m，旧分支只剩约 0.74–1.52 m 的向上延展；但候选必须满足 `new_support > max(1.25 * old_support, old_support + 0.025)`。两个候选的横向支持均未过此项，因此旧分支在交点处仍被标为 RELIABLE_CORE，而非 DIRECTIONAL_TAIL。',
      '122.95 在该交点要求新支持超过约 2.009 m，已高于当前 ±1 m 邻域能提供的约 2 m 跨度。116.25 虽然在其他位置生成了 8 个方向转换区，但原运行未在这些区内找到安全连接；真正可用的较早交点本身又不满足上述“支持显著更强”条件。两例在可用交点处的 `_fold_protected` 都为 False，说明这里的最终障碍不是该函数确认了必须保护的真实核心折返。',
      '已有的 future-switch 前瞻排序不能自动修复这个问题：内部路径只有先生成方向转换区，才进入前瞻排序和连接检查。更早的交点在上游资格条件处被挡住，后续即使知道新分支延伸更长，也不会让该交点重新进入候选。',
      '这说明不能只报 LONG_GAP_UNRESOLVED，更不能据此放宽 10 mm：**距离超限是搜索被限制之后的结果，上游原因是“普通折返全保留”与“内部换轨必须明显增强横向支持”的组合。**',
      '## 对照实验实际做了什么',
      '先在与上一轮相同的局部 H/V 输入范围复现两例，并逐条核对原始记录、edge ID、截取参数和坐标，与冻结 CURRENT_ROUTE 完全一致。随后只在独立诊断进程中，把普通 `_nearest_join` 的默认折返锁关闭；源码、候选、横向证据、MBG、ASC、10 mm、R_syn 及其余选择规则保持不变。',
      '122.95 从 B2 扩展为 B2→B4→B5，116.25 从 B1 扩展为 B1→B2→B3，均恢复当前候选的完整高程范围，且通过既有几何/核心/连接/比例审计。完整依据见 [junction_controls.json](junction_controls.json) 和 [junction_control_details.json](junction_control_details.json)。',
      '**该实验用于定位原因，不是正式修复。**全局直接取消折返锁可能删掉真实悬空细节。下一步应由同来源相邻分支的人工复核决定哪些折返应保留、哪些早期交点可以使用，而不是把这次诊断开关直接放进生产。',
      '源码位置：'+
      f"[普通折返锁]({(ROOT/'scripts/04_structure_recognition/main_track_assembly.py').as_posix()}:73)、"+
      f"[连接失败后阻断当前方向]({(ROOT/'scripts/04_structure_recognition/main_track_assembly.py').as_posix()}:274)、"+
      f"[方向尾部的支持增强条件]({(ROOT/'scripts/04_structure_recognition/branch_absolute_core.py').as_posix()}:123)。"
    ]
    if not (out/'missing_extent_summary.json').exists():
        report+=['## 全区清单仍在计算','上面的两例原因已经由受控实验确认。全区当前代码清查及其他缺失案例的共同原因统计尚未完成；此版本不能作为全区最终清单。',
          '## 分支审核图册',f'已完成 {len(gallery)} 组同 FaceTrack 原始分支：[打开五条邻线分组图册](FACETRACK_BRANCH_GALLERY.md)。']
        (out/'MISSING_ROUTE_CAUSE_REPORT_DRAFT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8');return
    census=js('census_summary.json');extent=js('missing_extent_summary.json');probes=js('junction_probe_summary.json');missing=csvread('missing_route_stage_inventory.csv')
    probe_rows=qualify_candidate_gates(out)
    stage_names={'CONNECTOR_GATE':'连接长度/比例门限','JUNCTION_SEARCH':'受限连接搜索无结果',
      'CONTINUATION_IDENTITY_TIE':'续接候选身份平局','RETAINED_CORE_GATE':'保留核心贡献不足',
      'PRESELECTION_Z_GAP':'前置高程间隙筛除','NO_EXTENDING_OPTION':'无可延展选项',
      'NO_RELIABLE_SURFACE_BRANCH':'没有可靠曲面分支'}
    core=[r for r in missing if float(r['missing_ASC_z_m'] or 0)>1e-6]
    large=[r for r in core if float(r['missing_ASC_z_m'])>=1]
    by_s={s:[r for r in large if r['s']==s] for s in extent['large_missing_keys']}
    counter={r['s'] for r in probe_rows if r['blocked_only_by_ordinary_fold_lock']}
    chosen_counter={r['s'] for r in probe_rows if r['blocked_only_by_ordinary_fold_lock'] and r['actually_chosen_candidate']}
    valid_alternative={r['s'] for r in probe_rows if not r['actually_chosen_candidate'] and r['locked_passes_other_gates']}
    counter_union=counter|valid_alternative
    residual=set(by_s)-counter_union
    residual_stages=Counter(r['stage'] for r in large if r['s'] in residual)
    report.insert(2,f"**全区结论：{census['count']} 条 V 中，{extent['uncovered_ASC_slices']} 条有可靠核心 ASC 未覆盖，{len(by_s)} 条至少缺 1 m。{len(counter)} 条存在折返锁反例，{len(valid_alternative)} 条存在现规则下可接的备选，重叠 {len(counter&valid_alternative)} 条，合计覆盖 {len(counter_union)} 条；另有 {len(residual)} 条不能用这两类反例解释。上述数字是候选覆盖诊断，不是识别准确率。FaceTrack 另按 23 组、每组五条邻线提供原始分支，等待人工审核。**")
    report.extend(['## 当前全区还缺哪些测线',
      f"本轮用修复后代码重新求解全部 **{census['count']} 条 V**。每 10 条为一组，使用当前物理分支重建 H/V 图，两侧至少保留 1 m 的邻线证据。FaceTrack 没有参与主轨选择。运行 {census['seconds']/60:.2f} 分钟、{census['workers']} 个 worker；这是实验入口的全区运行，不是生产 GUI 入口。",
      '运行中从 6 个 worker 调整为 12 个，复用已原子保存的 1634 条检查点结果；完整测线没有重复求解，仅重做中断时尚未完成保存的任务。总时间包含调整前后两段，详见 census_checkpoint.json。' if census.get('checkpoint') else '本轮使用单次固定并行配置。',
      '口径分成三层。先检查 MBG 合格分支的完整高程范围是否超出当前路线；再去掉分支两端保护段，只检查绝对可靠核心 ASC 是否确实超出；最后把 ASC 未覆盖高度至少 1 m 的单独列为“大段缺失”。1 m 只用于报告分层，不是新的算法门限。这里统计的是候选覆盖缺失，目标曲面身份仍需人工核查。',
      tab(['覆盖检查','测线数'],[
        ('合格分支完整范围未覆盖',extent['full_branch_extent_deficit_slices']),
        ('其中仅端点保护段超出，ASC 未缺',extent['endpoint_guard_only_slices']),
        ('确有 ASC 未覆盖',extent['uncovered_ASC_slices']),
        ('其中至少 1 m 的 ASC 大段缺失',extent['at_least_1m_uncovered_ASC_slices'])]),
      '全部含小缺失的逐测线/上下方向/阶段清单见 [missing_route_stage_inventory.csv](missing_route_stage_inventory.csv)，全 2800 条求解摘要见 [all_route_inventory.csv](all_route_inventory.csv)。不会把“原始范围略长于选中范围”的每一条都误称为长主轨缺失。',
      '### ASC 大段缺失的具体测线',
      tab(['s（m）','缺失方向','未覆盖 ASC 高度（m）','末次日志阶段','原选候选存在折返锁反例','现规则下存在可接备选'],[
        (s,','.join('向上' if r['side']=='upper' else '向下' for r in rs),','.join(f"{float(r['missing_ASC_z_m']):.3f}" for r in rs),','.join(sorted({stage_names.get(r['stage'],r['stage']) for r in rs})),'是' if s in chosen_counter else '其他候选有' if s in counter else '未发现','是' if s in valid_alternative else '未发现') for s,rs in by_s.items()]),
      '### 共同原因能解释多大范围',
      f"对上述 **{len(by_s)} 条**大段缺失逐条检查原候选和连接约束。**{len(chosen_counter)} 条**的日志首选分支存在可延伸的剩余片段，在只移除普通折返锁后可以满足既有核心、范围、10 mm 与比例约束；包括其他候选在内，共 **{len(counter)} 条**存在这种反例。这里证明的是当前续接状态下的规则阻断；日志没有逐次记录 piece_index，因此同一分支有多个剩余片段时，不能保证反例就是原选中的那个片段。只有 122.95、116.25 另做了完整路线的单变量重算，不把其他逐连接反例夸大成整条路线已经恢复。",
      tab(['大段缺失的末次日志阶段','方向事件数'],[(stage_names.get(k,k),v) for k,v in sorted(extent['large_core_stage_counts'].items())]),
      '末次日志阶段说明“在哪里停下”；折返锁的单变量反例说明“为什么那个合法连接没有进入”。一个测线可有上下两个事件，阶段计数不可直接与测线数相加。没有找到折返锁反例的测线保留原日志阶段，不能为了得到一个共同原因而全部归入同一类。统计额外核对候选剩余片段的贡献资格与待延展区域的 ASC 占比，避免把前置资格本就不通过的片段错算为折返锁原因。逐候选距离、是否为原首选、前置与后置约束是否通过，见 [gate_complete_junction_probes.csv](gate_complete_junction_probes.csv)。',
      '### 另一类已定位问题：首选连接失败后，没有尝试可接备选',
      f"另有 **{len(valid_alternative)} 条**大段缺失测线存在未被选中的候选，在保留当前折返锁、10 mm 和全部核心/比例条件时即可连接。这一集合可能与上面的折返锁集合重叠，不能相加作为总数。",
      '98.05、98.10 米就是明确的例子：当前路线均为 B3→B5，向下优先选 B2。B2 与 B3 的连接分别为 447.409、355.927 mm，被 10 mm 门限正确拒绝；但代码随即执行 `blocked.add(direction)`，没有用同一状态继续检查 B0、B6。',
      tab(['s（m）','首选 B2 连接（mm）','备选 B0 连接（mm）','备选 B6 连接（mm）','备选前置及连接约束'],[
        ('98.05','447.409','0.000357','0.000191','均通过，保留原折返锁'),
        ('98.10','355.927','0.976846','1.242114','均通过，保留原折返锁')]),
      '两例的 B0、B6 均通过待延展区域 ASC 占比和剩余核心贡献资格；连接后的核心保留、连接长度、比例预算也通过。因而这里不需要先取消折返锁，就能证明“首选失败＝整个方向无法继续”的推断不成立。B0/B6 的目标曲面身份及二者应选哪一个仍须审核，几何可接不等于地质身份正确。此处只验证连接可行性，没有另做整条路线的备选重算。',
      '因此，更上层的共同实现问题是：先选一个分支，再检查能否连接；该分支失败就封锁整个方向。普通折返锁会让有交点的分支被判为不可接；另外一些切面则是首选确实不可接，但仍存在可接备选。两种情形最终都会表现为后续主轨缺失，不能混成单纯“原始数据有断口”。',
      f"### 尚不能由上述两类反例解释的 {len(residual)} 条",
      tab(['剩余停止阶段','测线/方向事件数'],[(stage_names.get(k,k),v) for k,v in sorted(residual_stages.items())]),
      '其中 15 条在前置高程间隙检查就停止，包括 54.85、98.60–99.20 中列明的部分测线，以及 120.70/120.75。它们不能靠取消普通折返锁来解释：剩余片段与当前路线在高程上已有超过 10 mm 的间隔，未进入近邻连接搜索。完整表中的 `minimum_z_gap_m` 属于对应日志候选，不能当作整张原始网格的最小间隙。',
      '另 9 条是 95.90、95.95、96.05 和 104.55–104.80。对它们的合格可延展候选逐一检查后，即使移除普通折返锁、仍保留当前其他路径与范围条件，最好的连接也分别落在约 119–498 mm 范围内，均大于 10 mm。这里没有发现可接备选；这不证明源模型本身有同样大小的裂缝，也可能与之前已保留的路径或换轨位置有关。',
      '99.25 米先因续接身份平局停止；单独检查该候选，其连接约 41.352 mm，同样未通过几何门限。不能把解除平局等同于已经能接通。',
      '**这 25 条保留为进一步审核对象。**本轮已明确停止条件和当前状态下为何不属于前两类反例，但尚未证明是网格断口、上游分支选择还是更早换轨位置造成。应先审核同来源邻线分支，再追踪必要的上游选择；不能据此扩大连接距离或自动补成长直线。',
      '## 同一 FaceTrack 的分支一致性：只做人工审核分组',
      f"[分支图册](FACETRACK_BRANCH_GALLERY.md) 包含 **{len(gallery)} 个 FaceTrack 组**，来自上一轮 12 个局部区域，每组五条相邻 V，共 115 个“来源×测线”栏位。每组只画该来源的原始边，不看 CURRENT_ROUTE 是否选中，也不按相似程度筛选。多分支、闭合小片段、缺席和断开都如实保留。",
      '本轮按你的补充要求不设 TC 公式、不自动给通过率。FaceTrack 先把来源分开，你再判断折返、平台、悬空细节是否达到后续重建要求。源码提取的原始几何不会因为你看到的图比较平滑而被改写。',
      '![G_T1 五条原始分支](figures/G_T1.png)',
      'G_T1 里 122.95 的候选完整显示，之前主轨缺失不会再把这一栏变成空白。适合先核对平台和折返是否对应。',
      '![C_T1 五条原始分支细节](figures/C_T1_detail.png)',
      'C_T1 显示同来源下细节随 s 明显变化；这是实际原始分支的变化，不是主轨删选造成。是否属于合理的曲面变化，由你审核。',
      '![J_T1 五条原始分支细节](figures/J_T1_detail.png)',
      'J_T1 在部分切面包含 B2 与 B5 多段/闭合小片段。此处尤其要审核：单一局部 FaceTrack 是否已经足够细，还是把不同细节放进了同一个连通来源。这个例子不能只看编号相同就当作一致性通过。',
      '## 本轮修改、检查与保留事项',
      '新增诊断、图册和报告脚本：`diagnose_missing_routes.py`、`trace_missing_junctions.py`、`render_facetrack_branch_gallery.py`、`write_branch_consistency_report.py`。识别源码保持本轮开始时的哈希；上轮尚未提交的源码修复保持原样，未覆盖或提交。',
      '最低检查是本任务直接需要的全区结果审计、两例基线精确复现与单变量因果实验、图册成员逐边来源/坐标核对及链接检查。没有重复运行无关全项目测试，也没有把上轮的 49 项测试算作本轮新测试。',
      '已知边界：全区使用统一分块局部 H/V 图，因此与上轮历史全局 track 图不是同一次运行；当前缺失清单有独立输出目录与输入记录。FaceTrack 仍是局部真实共享边连通身份，不是地质真值，也不是细节一致性的自动证明。内部已有路线范围内的竞争面差异属于来源/细节问题，没有重复计为整段主轨缺失。',
      '**当前交付结论：已定位到两类漏接机制——折返锁及提前换轨资格排除了可用交点；首选分支连接失败后封锁整方向，遗漏可接备选。两例做了完整路线因果对照，其余按全区清单及逐连接证据列明。FaceTrack 的分支分组已备齐，分支一致性尚待你的人工审核，通过后再讨论把一致分支带回主轨筛选。**'])
    (out/'MISSING_ROUTE_CAUSE_REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    (out/'MISSING_ROUTE_CAUSE_REPORT_DRAFT.md').write_text('# 已更新为最终报告\n\n全区清查已完成，请阅读 [缺失主轨原因与分支审核报告](MISSING_ROUTE_CAUSE_REPORT.md)。\n',encoding='utf-8')
    # One final deliverable check: evidence, membership and linked figures.
    assert code_hashes()==js('manifest.json')['code_sha256']
    assert code_hashes()==json.loads((SOURCE/'topology_summary.json').read_text(encoding='utf-8'))['code_sha256']
    assert len(list((out/'routes').glob('*.pkl')))==2800
    assert not list((out/'routes').glob('*.pending'))
    members=csvread('facetrack_branch_membership.csv');assert len(members)==115 and len(gallery)==23
    checked_edges=0;regional_branches={}
    for entry in gallery:
        data=pickle.load((SOURCE/'region_data'/f"{entry['region_id']}.pkl").open('rb'))
        rid=entry['region_id']
        if rid not in regional_branches:
            regional_branches[rid]=load_inventory(SOURCE,data['region']['target_keys'])
        tid=entry['face_track_id']
        for s,bids in entry['branches_by_slice'].items():
            expected=sorted({e['branch_id'] for e in data['members'] if e['s']==s and e['face_track_ids']==[tid]})
            assert bids==expected
        originals={(s,b['branch_id'],e['edge_id']):e for s,bs in regional_branches[rid].items() for b in bs for e in b['records']}
        for e in data['members']:
            if e['face_track_ids']!=[tid]:continue
            original=originals[(e['s'],e['branch_id'],e['edge_id'])]
            assert e['source_face_ids']==original['source_face_ids']
            a,b=np.asarray(original['points_uz']);v=b-a;den=float(v@v)
            for p in np.asarray(e['points_uz']):
                t=float((p-a)@v/den) if den>1e-25 else 0.
                assert -1e-7<=t<=1+1e-7 and np.linalg.norm(a+np.clip(t,0,1)*v-p)<1e-7
            checked_edges+=1
    for name in ['MISSING_ROUTE_CAUSE_REPORT.md','FACETRACK_BRANCH_GALLERY.md']:
        text=(out/name).read_text(encoding='utf-8')
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not re.match(r'^[A-Za-z]:/',link):assert (out/link).exists(),link
    save_json(out/'DELIVERY_CHECKS.json',dict(recognition_source_unchanged_this_turn=True,current_routes=2800,
      baseline_controls_exactly_reproduced=2,full_route_single_variable_controls=2,FaceTrack_groups=23,branch_panels=115,
      group_members_match_all_source_members=True,source_edges_and_clipped_coordinates_checked=checked_edges,main_route_selection_not_used_for_gallery=True,
      automatic_TC_formula_or_verdict=False,links_resolve=True))
    print('FINAL_REPORT_AND_CHECKS_COMPLETE',flush=True)


if __name__=='__main__':main(latest())
