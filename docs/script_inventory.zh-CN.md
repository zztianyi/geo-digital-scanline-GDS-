# 脚本索引：面向复杂岩壁结构识别的数字测线框架

这份索引按处理流程重新组织脚本。固定数据路径已经迁移到 `configs/project.example.json` / `configs/project.local.json`，请先运行 `scripts/00_project/project_manager.py init` 生成本地配置。

## 00 项目配置

- `scripts/00_project/project_manager.py`：创建、检查和显示标准项目配置文件。

## 01 模型读取与转换

- `scripts/01_model_io/inspect_3mx_obj.py`：检查 3MX/OBJ 资源。
- `scripts/01_model_io/inspect_3mx_voxels.py`：检查 3MX 场景节点和 3MXB 资源。
- `scripts/01_model_io/preview_point_cloud.py`：加载并预览 TXT 点云。
- `scripts/01_model_io/build_dsm_mesh.py`：从点云生成 DSM 风格网格。
- `scripts/01_model_io/check_mesh_loading.py`：快速检查网格读取状态。

## 02 数字测线准备

- `scripts/02_scanline_setup/fit_arc_baseline.py`：拟合圆弧基线并写入弧线配置。
- `scripts/02_scanline_setup/preview_scanline_grid.py`：预览三维数字测线网格。
- `scripts/02_scanline_setup/project_point_cloud_profiles.py`：点云到局部剖面坐标投影。
- `scripts/02_scanline_setup/vertical_slice_analysis.py`：垂向切片分析实验。

## 03 网格切片与剖面提取

- `scripts/03_slicing_profiles/slice_single_profile.py`：单剖面切片与投影。
- `scripts/03_slicing_profiles/generate_scanline_slices.py`：批量生成数字测线切片。
- `scripts/03_slicing_profiles/inspect_scanline_slices.py`：检查切片和剖面结果。
- `scripts/03_slicing_profiles/clip_mesh_profiles.py`：网格裁剪与剖面对比。

## 04 结构识别

- `scripts/04_structure_recognition/run_multi_profile_recognition.py`：多剖面结构识别主流程。
- `scripts/04_structure_recognition/generate_profile_normals.py`：路径排序、法向量和局部方向计算。
- `scripts/04_structure_recognition/extract_group_centroids.py`：提取结构组质心。
- `scripts/04_structure_recognition/extract_line_face_links.py`：建立剖面线段与三维面片关系。

## 05 三维重建、体积与密度分析

- `scripts/05_reconstruction_volume/merge_structural_surfaces.py`：面片邻接与结构面聚合。
- `scripts/05_reconstruction_volume/reconstruct_blocks_and_voxels.py`：块体重建、体素化和体积分析。
- `scripts/05_reconstruction_volume/export_recognized_structures.py`：导出识别线段和面片。
- `scripts/05_reconstruction_volume/run_voxel_kde.py`：体素或质心 KDE 分析。
- `scripts/05_reconstruction_volume/visualize_kde_density.py`：KDE 结果可视化。
- `scripts/05_reconstruction_volume/cluster_voxel_points.py`：体素派生点聚类。

## 06 绘图展示

- `scripts/06_visualization/plot_scanline_layouts.py`：数字测线布局示意图。
- `scripts/06_visualization/profile_figure_panel.py`：剖面和结构识别图像展示。
- `scripts/06_visualization/profile_figure_variant.py`：剖面展示变体。
- `scripts/06_visualization/plot_blocks_and_clusters.py`：块体和聚类结果绘图。
- `scripts/06_visualization/plot_kinematic_analysis.py`：运动学分析示意。
- `scripts/06_visualization/four_panel_3d_view.py`：四联三维展示图。

## 99 实验与旧版脚本

- `scripts/99_experiments/experimental_face_clustering.py`：实验性面片聚类。
- `scripts/99_experiments/query_scanline_transfer.py`：数字测线几何查询。
- `scripts/99_experiments/legacy_point_dbscan.py`：旧版 DBSCAN 原型。
- `scripts/99_experiments/cluster_placeholder.py`：占位脚本。
