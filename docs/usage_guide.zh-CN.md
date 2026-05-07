# 中文使用指南

本项目目前是研究型脚本工具箱，适合按处理阶段逐步运行。推荐先理解“数字测线框架”：三维模型提供真实空间几何，数字测线把复杂岩壁结构转化为连续剖面问题，二维识别降低算法设计与维护难度，三维回投恢复空间表达并支持体积、密度等后续分析。

## 1. 准备数据

常见输入包括：

- 三维网格：`.glb`、`.obj`、`.ply` 等。
- 点云数据：TXT、LAS/LAZ 或 PLY。
- 弧形基线配置：参考 `configs/arc_config.example.json`。

真实数据建议放在 `data/private/` 或仓库外部，并通过本地配置文件引用。

## 2. 拟合基线与控制采样密度

相关脚本：

- `2d_feat.py`
- `2d_ceshi.py`
- `projection.py`
- `Vertical_Slice.py`

这一阶段用于从岩壁轮廓中建立弧形基线，并确定测线密度。采样密度越高，离散剖面对连续岩体的近似越细，但计算量也会增加。

## 3. 生成数字测线和剖面交线

相关脚本：

- `slice_generator.py`
- `slice_generator_ceshi.py`
- `3MX_projection.py`
- `clip_.py`

这一阶段把三维网格与剖切平面相交，得到一组二维剖面线段，为后续结构识别提供稳定的二维观察窗口。

## 4. 二维结构识别与路径追踪

相关脚本：

- `image_.py`
- `part.py`
- `generate_normals.py`
- `Multi_profit.py`

这一阶段关注路径排序、局部方向计算、正交向量判断和悬挑/凹腔等复杂结构提取。

## 5. 跨剖面聚合与三维回投

相关脚本：

- `Multi_profit.py`
- `group_centroid.py`
- `line_face_extraction.py`
- `surface_merge.py`
- `structural.py`

这一阶段把多个二维剖面中的同类结构连接为三维结构组，并建立面片、线段或体素表达。

## 6. 应用分析

相关脚本：

- `voxel_ceshi.py`
- `point_dbscan.py`
- `kde.py`
- `plt_ploter.py`

可用于潜在危险体体积估算、不同尺寸块体空间分布 KDE、聚类质心提取和结果图像展示。

## 备注

当前脚本仍保留研究原型特征。未来建议把路径配置、参数配置和运行入口进一步统一，形成更稳定的命令行流程。

