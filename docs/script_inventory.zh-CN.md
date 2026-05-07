# 脚本索引

这份索引按项目处理顺序整理脚本用途，方便以后重新进入项目时快速判断每个脚本负责哪一段工作。当前代码仍保留研究原型阶段的命名，建议先按下列模块理解，再结合 `usage_guide.zh-CN.md` 运行。

## 1. 基线、投影与数字测线准备

- `2d_feat.py`：点云过滤、圆弧基线拟合，以及直线基线与圆弧基线的二维对比。
- `2d_ceshi.py`：基于圆弧基线在三维点云上绘制数字测线网格，是理解数字测线布设方式的入口脚本之一。
- `projection.py`：将点云投影到局部剖面坐标系，为后续二维剖面分析准备坐标。
- `Vertical_Slice.py`：简单的垂向投影与高程中心分析实验脚本。
- `kinematic_analysis.py`：用于平面破坏示意和运动学分析展示的辅助脚本。

## 2. 三维模型读取与转换

- `3MX.py`：合并从 3MX 类场景中导出的 OBJ/MTL 资源。
- `3MX_voxel.py`：解析并可视化 3MX 场景节点。
- `las_point.py`：读取 TXT 点云，并转换为 Open3D 体素可视化结果。
- `mesh_dsm.py`：将点云转换为近似 DSM 的三角网格。
- `word_.py`：快速检查网格数据是否能被正常读取和显示。

## 3. 网格切片与二维剖面提取

- `3MX_projection.py`：单个剖切平面的网格切片与二维投影。
- `slice_generator.py`：批量生成剖面切片，并导出线-面关系结果。
- `slice_generator_ceshi.py`：交互式或多面板的切片过程可视化实验脚本。
- `clip_.py`：剖切平面对比和三维裁剪辅助函数。
- `image_.py`、`part.py`：早期路径排序、法向量计算和绘图实验版本。

## 4. 多剖面结构识别

- `Multi_profit.py`：多剖面处理主类与批处理流程，是二维结构识别链路中的核心脚本之一。
- `generate_normals.py`：向量场生成、路径校正、结构分组和过程绘图工具。
- `group_centroid.py`：从多切片识别结果中提取结构组质心。
- `line_face_extraction.py`：将剖面中识别出的线段结果重新关联到三维网格面片。

## 5. 面片、体素、密度与块体分析

- `surface_merge.py`：面片邻接关系、聚类与可视化。
- `structural.py`：结构面聚类、棱柱/体素构建和稀疏体素结果导出。
- `transfer.py`：对线段进行采样，并导出提取到的面片和线段结果。
- `voxel_ceshi.py`：基于体素或质心数据进行 KDE 密度分析。
- `point_dbscan.py`、`point——dbscan.py`：对体素派生点集进行 DBSCAN 聚类。
- `plt_ploter.py`：根据体素数据绘制块体或聚类结果。
- `kde.py`：网格着色和空间密度可视化。

## 6. 绘图与实验辅助

- `plt_.py`：绘制曲线测线和直线测线布局的二维/三维示意图。
- `ceshi.py`：实验性的网格与面片聚类流程。
- `cluster.py`：研究原型阶段保留的空占位脚本。
- `polt/3d.py`：小型三维绘图辅助脚本。

## 推荐阅读顺序

1. 先看 `2d_ceshi.py`、`2d_feat.py`，理解数字测线和圆弧基线为什么存在。
2. 再看 `slice_generator.py`、`3MX_projection.py`，理解三维网格如何被切成二维剖面。
3. 接着看 `Multi_profit.py`、`generate_normals.py`，理解二维剖面上的结构识别逻辑。
4. 然后看 `group_centroid.py`、`line_face_extraction.py`、`surface_merge.py`，理解跨剖面聚合和三维关联。
5. 最后看 `structural.py`、`voxel_ceshi.py`、`kde.py`、`plt_ploter.py`，理解体积估算、体素表达和空间分布分析。
