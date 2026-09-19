# Branch-first 本地修改记录

方案：`C:/Users/222/Downloads/GDS_Branch_First_Surface_Track_Codex_Plan.md`。

基线：backend / e951bfbc596e8e23859ac8fa990e32fa36e4adbd；开始时工作区干净。

执行顺序：

1. 完整 branch / 局部弧长 descriptor；真实 H connected run crossing。
2. 相邻 slice surface graph / track；同 track detail 与身份选择。
3. confidence crossover 附近单次 handoff；锁定身份后的缺失区推断。
4. 切换实验入口；改写保护旧行为的测试。
5. 一次直接相关检查，生成扰动、遮尾、代表案例与纠错报告。

实现裁决：

- 直接在用户指定本地项目修改，不创建副本、不提交、不推送。用户的最小范围要求优先于技能的 worktree / commit 默认流程。
- 已提供并授权执行的方案作为设计依据；不重复索要设计批准。
- 测试先编写，最终集中执行一次直接相关检查；不执行技能建议的逐步重复测试和全项目测试。
- 旧端点实验仅保留为 legacy 对照。新主入口按 surface_track_id + target branch id 输出身份；旧案例 ID 仅用于对照图索引。
- H crossing 容差仅容纳 round6 数值误差，不能使用毫米级 proximity 作为支持。
- 没有可证明身份的缺失区保持 unresolved，不调用旧端点引导的 LOCC 冒充已知 surface。

进度：已确认旧窗口裁剪、H proximity、spanning override 和 endpoint pairing 的源码根因。

实现进度：四个 surface 模块已接入主 solver；旧实验入口改为 surface_track_validation；旧 endpoint pairing 主流程移除；完整 branch 保留与折返测试已更新。

独立只读审核：按照执行技能调用一个审核代理，未委托实现、未额外运行测试。已修正三项发现，并加入针对性回归：区间约束最近点搜索、推断覆盖未关联观测、歧义 crossing 被过滤后重新当作确定支持。

待完成：集中检查及冻结数据审计；报告实测结果。

## 修改文件

| 文件 | 修改目的 |
|---|---|
| `scripts/04_structure_recognition/dominant_observed_branch.py` | 主solver改为完整surface track选择；删除端点筛选、裁剪评分与spanning覆盖 |
| `scripts/04_structure_recognition/horizontal_surface_link.py` | 新增完整H路径及真实几何crossing关联 |
| `scripts/04_structure_recognition/observed_surface_graph.py` | 新增相邻slice图、track身份、连续支持与歧义记录 |
| `scripts/04_structure_recognition/surface_track_selection.py` | 新增完整branch descriptor、同track detail、identity rank |
| `scripts/04_structure_recognition/surface_track_handoff.py` | 新增confidence crossover、最多一次换轨、身份锁定缺失尾部补全 |
| `scripts/04_structure_recognition/backtracking_junction.py` | 区间约束搜索；原edge内部精确切分 |
| `scripts/99_experiments/validate_dominant_branch.py` | 原实验主入口转接surface-first审计，删除endpoint pairing主流程 |
| `scripts/99_experiments/surface_track_validation.py` | 新增完整建档、自然案例、端点扰动、partial-tail与资源审计 |
| `scripts/99_experiments/surface_track_report.py` | 新增四份报告及固定29例前后图 |
| `scripts/99_experiments/dominant_branch_report.py` | legacy绘图纠正raw H误标 |
| `tests/test_surface_track.py` | 新增身份、H连续支持、handoff、缺失区及审核问题回归 |
| `tests/test_dominant_observed_branch.py` | 改为完整branch和同track evidence要求 |
| `tests/test_dominant_review_fixes.py` | 删除错误的spanning优先断言 |
| `tests/test_dominant_boundary_association.py` | 区分完整观测与未经跨slice证明的track |
| `docs/surface_track_execution.md` | 本记录与实现裁决 |

## 最低检查命令

```powershell
python scripts/99_experiments/validate_dominant_branch.py --check --output outputs/surface_track_validation/20260919_branch_first
```

已经执行：30项直接相关回归全部通过（0.069s），未执行全项目测试或构建。数据审计与报告属于方案要求的本轮交付，沿用冻结数据，不重跑识别、体素或全高遮挡实验。


## 首轮自然审计及针对性修正

首轮30项回归通过，3,474次端点扰动0身份变化；但普通连通分量产生包含同slice竞争branch的巨型track，193自然案例全部未确认，28个唯一代表slice全部跳过遮尾。首轮输出保留在 `outputs/surface_track_validation/20260919_branch_first`。

针对性修正：track提取增加同slice支持高度排斥约束，冲突边输出到surface_track_ambiguous_links.csv；不重叠的支持区仍可进行confidence crossover。新回归直接构造远处连接导致双surface误合并的情形；未调H proximity阈值。高置信度长度改为连续linked H层之间实际保留的observed edge长度，明确为算法证据量。

遵照用户规则，仅进行一次针对性修改后的重试，输出到 `outputs/surface_track_validation/20260919_branch_first_corrected`；不覆盖首轮证据。


## 最终实测结果

针对性修正后的唯一重试完成：31项回归通过；完整inventory 20,078 branches，6,220 tracks（315稳定、218带歧义标记，两标签可重叠）；raw H 4,324,840，surface-linked H 4,255,368。

193个历史case对应168个唯一target slice；3,474次端点扰动的身份变化为0。69个唯一slice保留完整稳定track。自然route switch、A→B→A、connector、inferred长度均为0；合成主路径回归验证实际非零handoff与tail recovery。

真实partial-tail结果：28个唯一代表slice中，15例多值尾部、12例track不稳定/歧义未评估，1例保留身份但未恢复缺失尾部。真实换轨、遮尾补全及人工目标identity尚未获充分验证，因此不能宣称完整方案验收通过或最小几何干预SUPPORTED。

最终审计360.88秒，峰值RSS 11.738 GiB；首轮审计346.48秒，两次数值与报告阶段合计707.35秒（不含代码编写）。无全项目测试、构建、识别/体素重算、提交或推送。

四份最终报告与29张前后图：`outputs/surface_track_validation/20260919_branch_first_corrected/`。本地改动完成；人工逐图确认A-F目标曲面与未决情况。
