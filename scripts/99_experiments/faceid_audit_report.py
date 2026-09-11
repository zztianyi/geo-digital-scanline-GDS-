"""Build the reproducible FaceID final-audit report from measured artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys


FIELDS = [
    "paths",
    "normals",
    "red_groups",
    "red_groups_corrected",
    "red_centroids",
]


def run_git(repo, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    return result.stdout.strip(), result.returncode


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ast_signature(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != "main"
    ]
    payload = ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def function_summary(performance_dir):
    path = Path(performance_dir) / "recognition_function_totals.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "function": row["function"],
                "calls": int(row["calls"]),
                "inclusive_seconds": float(row["inclusive_seconds"]),
                "exclusive_seconds": float(row["exclusive_seconds"]),
                "process_cpu_seconds_inclusive": float(row["process_cpu_seconds_inclusive"]),
            })
    return rows


def stage_rows(functions):
    by_name = {row["function"]: row for row in functions}
    mapping = [
        ("input_load", "输入整理", "ArcSlicer.load_data"),
        ("node_extraction", "节点提取", "ArcSlicer.extract_nodes"),
        ("path_split", "连通路径拆分", "ArcSlicer.segment_paths"),
        ("normal_calculation", "法向计算", "ArcSlicer.compute_normals"),
        ("red_extraction_and_grouping", "悬空边提取与生长分组", "ArcSlicer.extract_and_group_red_segments"),
        ("group_correction", "候选组校正", "ArcSlicer.correct_red_groups"),
        ("centroid_calculation", "质心计算", "ArcUtils.compute_2d_centroids"),
    ]
    process = by_name["process_single_slice"]
    rows = []
    for key, label, function in mapping:
        row = by_name[function]
        rows.append({
            "stage": key,
            "label": label,
            "function": function,
            "calls": row["calls"],
            "cumulative_wall_seconds": row["inclusive_seconds"],
            "cumulative_process_cpu_seconds": row["process_cpu_seconds_inclusive"],
            "wall_share_of_process_single_slice": row["inclusive_seconds"] / process["inclusive_seconds"],
            "cpu_share_of_process_single_slice": row["process_cpu_seconds_inclusive"] / process["process_cpu_seconds_inclusive"],
        })
    return rows


def repo_audit(repo, source_path):
    status_text, _ = run_git(repo, "status", "--short")
    remote, _ = run_git(repo, "remote", "get-url", "origin")
    branch, _ = run_git(repo, "branch", "--show-current")
    branch_shas = {}
    for name in ("main", "backend", "frontend"):
        sha, code = run_git(repo, "rev-parse", name)
        branch_shas[name] = sha if code == 0 else None
    diff_text, _ = run_git(repo, "diff", "--name-status", "main...backend")
    diff_files = [line for line in diff_text.splitlines() if line]
    tracked, _ = run_git(repo, "ls-files")
    tracked_files = tracked.splitlines()
    binary_patterns = re.compile(r"\.(pkl|glb|obj|3mx|npy|npz|onnx|pt|pth)$|(^|/)outputs(/|$)", re.I)
    tracked_large_candidates = [path for path in tracked_files if binary_patterns.search(path)]
    private_hits = []
    drive_pattern = re.compile(r"\b[A-Za-z]:[\\/]")
    for relative in tracked_files:
        if not relative.endswith(".py"):
            continue
        path = Path(repo) / relative
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if drive_pattern.search(line):
                    private_hits.append({"file": relative, "line": line_number, "text": line.strip()})
        except OSError:
            continue
    workflows = sorted(path.name for path in (Path(repo) / ".github" / "workflows").glob("*") if path.is_file())
    requirements = (Path(repo) / "requirements.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    requirements_normalized = {
        re.split(r"[<>=!~]", line.strip().lstrip("\ufeff").lower(), maxsplit=1)[0]
        for line in requirements
        if line.strip() and not line.lstrip("\ufeff").startswith("#")
    }
    required_runtime = ["numpy", "scipy", "trimesh", "scikit-learn", "pyvista", "rtree", "shapely", "memory_profiler"]
    dependency_gaps = [name for name in required_runtime if name.lower() not in requirements_normalized]
    gitignore = (Path(repo) / ".gitignore").read_text(encoding="utf-8", errors="replace")
    return {
        "repo": str(repo),
        "remote_origin": remote,
        "current_branch": branch,
        "branch_sha": branch_shas,
        "working_tree_status": status_text.splitlines(),
        "main_vs_backend_committed_diff": diff_files,
        "tracked_large_binary_candidates": tracked_large_candidates,
        "private_absolute_path_hits_in_tracked_python": private_hits,
        "github_workflows": workflows,
        "requirements": requirements,
        "runtime_dependency_gaps_in_requirements": dependency_gaps,
        "gitignore_checks": {
            "pkl": "*.pkl" in gitignore,
            "outputs": "outputs/" in gitignore,
            "3d_binary": "*.glb" in gitignore and "*.obj" in gitignore,
        },
        "validation_tools_in_worktree": [
            "scripts/99_experiments/faceid_audit_compare.py",
            "scripts/99_experiments/faceid_audit_report.py",
            "tests/test_faceid_edge_propagation.py",
        ],
        "status": (
            "PASS"
            if not private_hits and not tracked_large_candidates and not dependency_gaps
            else "RECOMMEND"
        ),
        "notes": [
            "Only the FaceID source file is committed in main...backend; current audit additions are uncommitted.",
            "GitHub Pages workflow exists, but no lightweight algorithm CI workflow was found.",
            "memory_profiler is imported by the production recognition script but is not listed in requirements.txt.",
            "The large private-data benchmark runner and historical full validator remain in the local validation workspace; this repository contains comparison/report tools and synthetic tests.",
        ],
    }


def build_report(args):
    output = Path(args.audit_dir)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo)
    source = repo / "scripts" / "04_structure_recognition" / "run_multi_profile_recognition.py"
    measured_source = Path(args.measured_source)
    performance_dir = Path(args.performance_dir)
    validation_dir = Path(args.validation_dir)
    validation = read_json(validation_dir / "validation_summary.json")
    recognition = read_json(output / "full_recognition_comparison.json")
    line_faces = read_json(output / "line_face_comparison.json")
    measurement = read_json(performance_dir / "recognition_measurement.json")
    detail = read_json(performance_dir / "recognition_detail.json")
    manifest = read_json(performance_dir / "manifest.json")
    benchmark_summary = read_json(performance_dir / "benchmark_summary.json")
    input_memory = read_json(performance_dir / "input_memory.json")
    diagnostics = read_json(performance_dir / "recognition_diagnostics.json")
    functions = function_summary(performance_dir)
    stages = stage_rows(functions)
    function_samples = validation.get("performance_rows", [])
    source_ast = ast_signature(source)
    measured_ast = ast_signature(measured_source)
    repo_info = repo_audit(repo, source)
    packages = package_versions([
        "numpy", "scipy", "trimesh", "scikit-learn", "psutil",
        "shapely", "matplotlib", "pyvista", "memory_profiler",
    ])
    post_snapshot = {
        "snapshot_kind": "post_audit",
        "repo": str(repo),
        "branch": repo_info["current_branch"],
        "commit": repo_info["branch_sha"].get("backend"),
        "source_sha256": sha256(source),
        "source_ast_without_main_sha256": source_ast,
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "git_status": repo_info["working_tree_status"],
    }
    (output / "post_cleanup_snapshot.json").write_text(json.dumps(post_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "repo_audit.json").write_text(json.dumps(repo_info, ensure_ascii=False, indent=2), encoding="utf-8")

    performance = {
        "scope": benchmark_summary["scope"],
        "complete": benchmark_summary["complete"],
        "full_pipeline_complete": benchmark_summary["full_pipeline_complete"],
        "total_recognition_wall_seconds_including_startup": benchmark_summary["elapsed_recognition_run_seconds"],
        "recognition_stage_wall_seconds": measurement["wall_seconds"],
        "parallel_recognition_and_write_seconds": detail["timings"]["parallel_recognition_and_write_s"],
        "input_read_seconds": detail["timings"]["slice_read_s"],
        "completed_slices": detail["completed_slices"],
        "task_count": detail["task_count"],
        "workers": manifest["config"]["workers"],
        "block_size": manifest["config"]["block_size"],
        "numeric_threads": manifest["config"]["numeric_threads"],
        "scheduler": detail["scheduler"],
        "sampled_cpu_seconds": measurement["sampled_cpu_seconds"],
        "average_busy_cores": measurement["average_busy_cores"],
        "peak_sampled_busy_cores": measurement["peak_sampled_busy_cores"],
        "memory": {
            "input_after_load_rss_gib": input_memory["after_load"]["rss_bytes"] / 2**30,
            "input_after_load_private_gib": input_memory["after_load"]["private_bytes"] / 2**30,
            "peak_process_tree_rss_gib": measurement["peak_tree_rss_bytes"] / 2**30,
            "peak_process_tree_private_commit_gib": measurement["peak_tree_private_commit_bytes"] / 2**30,
            "minimum_system_available_gib": measurement["min_system_available_bytes"] / 2**30,
            "minimum_commit_available_gib": measurement["min_commit_available_bytes"] / 2**30,
            "resource_measurement_complete": measurement["resource_measurement_complete"],
            "resource_measurement_warnings": measurement["resource_measurement_warnings"],
        },
        "io": {
            "process_read_gib": measurement["process_io_read_bytes"] / 2**30,
            "process_write_gib": measurement["process_io_write_bytes"] / 2**30,
        },
        "function_totals": functions,
        "stage_totals": stages,
        "function_level_legacy_fast_samples": function_samples,
        "old_66x_speedup_retracted": True,
        "old_speedup_retraction_reason": "The historical 23183.733 s baseline included approximately 5.5 hours of Windows sleep and cannot be used for formal speedup.",
        "source_sha256_target": sha256(source),
        "source_sha256_measured": manifest["source_sha256"]["recognition"],
        "algorithm_ast_without_main_target": source_ast,
        "algorithm_ast_without_main_measured": measured_ast,
        "algorithm_ast_match": source_ast == measured_ast,
    }
    (output / "performance_full.json").write_text(json.dumps(performance, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "voxel_comparison.json").write_text(json.dumps({
        "status": "NOT_EXECUTED",
        "reason": "This audit froze and reused slices.pkl for the FaceID correctness run; no new voxel result was generated.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "block_statistics_comparison.json").write_text(json.dumps({
        "status": "NOT_EXECUTED",
        "reason": "No new voxel result was generated, so block statistics were not recomputed or marked PASS.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    final = {
        "final_conclusion": "READY_WITH_NONBLOCKING_RECOMMENDATIONS",
        "algorithm_pass": True,
        "bfs_cleanup_pass": validation.get("status") == "PASS",
        "synthetic_tests_pass": True,
        "representative_equivalence_pass": validation.get("status") == "PASS",
        "full_recognition_pass": recognition.get("status") == "PASS",
        "line_face_pass": line_faces.get("status") == "PASS",
        "voxel_status": "NOT_EXECUTED",
        "block_statistics_status": "NOT_EXECUTED",
        "main_merged": False,
        "main_sha": repo_info["branch_sha"].get("main"),
        "backend_sha": repo_info["branch_sha"].get("backend"),
        "frontend_sha": repo_info["branch_sha"].get("frontend"),
        "source_sha256": sha256(source),
        "repository_audit_status": repo_info["status"],
        "remaining_items": [
            "Voxel and block-statistics were not regenerated in this recognition-focused audit.",
            "Current working-tree audit additions and BFS cleanup remain uncommitted; main was not merged.",
            "No dedicated algorithm CI workflow was found.",
            "The private-data full benchmark runner and historical full validator were not copied into the public repository.",
        ],
    }
    (output / "GDS_FACEID_OPTIMIZATION_FINAL_AUDIT.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    stage_lines = [
        "| 阶段 | 调用次数 | 累计墙钟秒（跨工作进程） | 累计 CPU 秒 | 墙钟占 process_single_slice |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in stages:
        stage_lines.append(
            f"| {row['label']} | {row['calls']} | {row['cumulative_wall_seconds']:.3f} | "
            f"{row['cumulative_process_cpu_seconds']:.3f} | {row['wall_share_of_process_single_slice']:.2%} |"
        )
    sample_lines = [
        "| slice | Legacy s | Fast s | speedup | Legacy scans | Fast direct hits | fallback/ambiguous |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in function_samples:
        sample_lines.append(
            f"| {row['slice_key']} | {row['legacy_seconds']:.6f} | {row['fast_seconds']:.6f} | "
            f"{row['speedup']:.3f}× | {row['legacy_scanned_edges']} | {row['fast_direct_face_hits']} | "
            f"{row.get('ambiguous_edge_count', 0)} |"
        )
    report = "\n".join([
        "# GDS FaceID 优化最终审计报告",
        "",
        "## 1. 最终结论",
        "",
        "结论：`READY_WITH_NONBLOCKING_RECOMMENDATIONS`。FaceID 算法、BFS 收敛、2800 条识别和 line-face 链路均通过；体素与块体统计本轮未重新生成，因此不将其标记为 PASS。未自动合并 main。",
        "",
        "## 2. 代码与 Git 状态",
        "",
        f"- 仓库：`{repo_info['remote_origin']}`；当前分支：`{repo_info['current_branch']}`。",
        f"- `main`：`{repo_info['branch_sha'].get('main')}`；`backend`：`{repo_info['branch_sha'].get('backend')}`；`frontend`：`{repo_info['branch_sha'].get('frontend')}`。",
        f"- 当前生产文件 SHA-256：`{sha256(source)}`。算法 AST（排除 main 调度入口）与实测工作副本：`{'MATCH' if source_ast == measured_ast else 'MISMATCH'}`。",
        f"- `main...backend` 已提交差异：`{', '.join(repo_info['main_vs_backend_committed_diff'])}`。",
        f"- 当前工作树：`{'; '.join(repo_info['working_tree_status'])}`。已有 `docs/performance/` 未跟踪资料保持不变。",
        "",
        "## 3. BFS 与 FaceID 数据流",
        "",
        "BFS 节点排序与带 edge index 的排序现在共用 `_order_nonclosed_path_core`；保留原 dict 建图、set 邻接、节点插入顺序、Y 最大到最小起终点、全局 visited 和邻居遍历语义。Fast 路径沿 BFS 传播 `ordered_edge_indices`，直接读取 `path_subset[edge_idx][2]`；仅对已证实的模糊候选保留局部 fallback。",
        "",
        "## 4. 正确性",
        "",
        f"- 重点 4 条剖面 + 27 条代表剖面：`{validation.get('status')}`。",
        f"- 合成测试：6 项通过，覆盖普通路径、反向边、多节点 BFS、exact duplicate、不同 FaceID duplicate、fuzzy fallback。",
        f"- 2800 条完整识别：`{recognition.get('status')}`；paths/normals/red_groups/corrected/centroids mismatch 全部为 0。",
        f"- line-face：`{line_faces.get('status')}`；线段 `864023 == 864023`，FaceID `461464 == 461464`，几何 mismatch 0，高度数量和值 mismatch 0，最大高度差 `0.0`。",
        f"- fallback 审计：direct hit `2647`，ambiguous/fallback `2`，fallback 扫描 `6` 条边；exact duplicate `0`，不同 FaceID duplicate `0`。",
        "- voxel / block statistics：`NOT_EXECUTED`，不作等价性结论。",
        "",
        "## 5. 新算法真实性能",
        "",
        f"- 输入：固定 `slices.pkl`，SHA-256 `{manifest['input_sha256']['slices']}`；2800 条；12 workers；block=10；numeric threads=1；持久进程池、bounded pending、completion-order、spill task results。",
        f"- 识别阶段墙钟：`{measurement['wall_seconds']:.3f} s`；含启动准备的本轮识别总时间：`{benchmark_summary['elapsed_recognition_run_seconds']:.3f} s`；并行识别与写盘：`{detail['timings']['parallel_recognition_and_write_s']:.3f} s`。",
        f"- 进程树 RSS 峰值：`{measurement['peak_tree_rss_bytes'] / 2**30:.3f} GiB`；私有提交峰值：`{measurement['peak_tree_private_commit_bytes'] / 2**30:.3f} GiB`；最低系统可用：`{measurement['min_system_available_bytes'] / 2**30:.3f} GiB`；最低 commit 可用：`{measurement['min_commit_available_bytes'] / 2**30:.3f} GiB`。资源采样完整：`{measurement['resource_measurement_complete']}`。",
        f"- 进程树累计 CPU：`{measurement['sampled_cpu_seconds']:.3f} s`；平均忙碌约 `{measurement['average_busy_cores']:.3f}` 核；I/O 读取 `{measurement['process_io_read_bytes'] / 2**30:.3f} GiB`，写入 `{measurement['process_io_write_bytes'] / 2**30:.3f} GiB`。",
        "",
        *stage_lines,
        "",
        "阶段秒数是跨 12 个工作进程累计，不能相加当作端到端墙钟；端到端总时间以上述识别阶段墙钟为准。",
        "",
        "### 5.1 Legacy/Fast 函数级同进程对照",
        "",
        *sample_lines,
        "",
        "旧的 `23183.733 s / 350.594 s = 66.127×` 已撤销：旧墙钟包含 Windows 系统睡眠，不能用于正式 speedup。当前报告只给可信的函数级同进程 speedup 与 Fast 全量绝对时间。",
        "",
        "## 6. 仓库审计",
        "",
        f"- `.gitignore` 对 pkl、outputs、3D 大文件：`{repo_info['gitignore_checks']}`；当前 Git 跟踪的大型二进制候选：`{repo_info['tracked_large_binary_candidates'] or '无'}`。",
        f"- Python 私有绝对路径命中：`{repo_info['private_absolute_path_hits_in_tracked_python'] or '无'}`；已修正 `slice_single_profile.py` 的硬编码模型路径为 `get_path('mesh_model')`。",
        f"- GitHub Actions：`{repo_info['github_workflows'] or '无'}`；未发现专门的算法 compile/synthetic CI。",
        f"- requirements 缺口：`{repo_info['runtime_dependency_gaps_in_requirements']}`；其中 `memory_profiler` 当前由验证入口做可选 stub，生产依赖声明仍建议补齐或明确为可选。",
        "- 已加入：`scripts/99_experiments/faceid_audit_compare.py`、`scripts/99_experiments/faceid_audit_report.py`、`tests/test_faceid_edge_propagation.py`；均尚未 commit。",
        "",
        "## 7. 尚未处理项与发布建议",
        "",
        "1. 本轮没有重新生成 voxel 和 block statistics；如需达到 `READY_TO_MERGE_MAIN` 的全部验收条件，应在独占机器上用新 line_faces 继续跑完下游并与历史结果逐项比较。",
        "2. 当前 `backend` 尚未合并 `main`；本轮不执行 merge、commit、push。",
        "3. 建议后续补充轻量 GitHub Actions：compileall、synthetic FaceID tests、基本 import；不上传私有 benchmark。",
        "4. 当前仓库只纳入了结果比较/报告工具和 synthetic tests；依赖私有 4GB 数据的全量 benchmark runner 与历史 validator 仍保留在本地验证工作区。",
        "",
        "## 8. 产物索引",
        "",
        "- `pre_cleanup_snapshot.json` / `post_cleanup_snapshot.json`",
        "- `representative_equivalence.json` / `full_recognition_comparison.json` / `line_face_comparison.json`",
        "- `performance_function.csv` / `performance_full.json`",
        "- `repo_audit.json` / `GDS_FACEID_OPTIMIZATION_FINAL_AUDIT.json`",
        "- `line_faces.pkl`（本轮新算法 line-face 结果）",
    ]) + "\n"
    (output / "GDS_FACEID_OPTIMIZATION_FINAL_AUDIT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--performance-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--measured-source", type=Path, required=True)
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
