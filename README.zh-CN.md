# 面向复杂岩壁结构识别的数字测线框架

[English README](README.md) | [项目展示页](docs/index.html) | [中文使用指南](docs/usage_guide.zh-CN.md) | [脚本索引](docs/script_inventory.zh-CN.md)

这是一个面向复杂岩壁数字地质模型的研究型 Python 工作流。项目以**数字测线**为核心组织方式：先将三维岩壁网格或点云转化为一组可控密度的二维剖面观测窗口，在二维剖面中识别结构特征，再把识别结果回到三维空间中进行结构表达、潜在危险块体体积估算和空间密度分析。

这个仓库不是单纯展示三维模型的可视化项目，而是尝试把复杂岩壁几何转化为可追踪、可识别、可重建、可量化的结构信息，为工程地质解释和现场治理决策提供数据基础。

## 核心思想

复杂岩体在空间中是连续的，而测线和剖面本质上是离散观测。数字测线框架通过控制剖面采样密度，让离散观测在特定尺度上代表连续结构。这样做可以降低直接处理三维复杂结构的算法难度，也让中间结果更容易检查、解释和维护，同时最终结果仍然能回到原始三维几何中。

## 工作流

1. **模型输入与检查**：读取点云、三角网格和 3MX 相关资源。
2. **数字测线准备**：拟合弧形基线，并按设定间距生成测线剖面。
3. **二维剖面提取**：将三维模型切分为一系列二维结构观察窗口。
4. **结构特征识别**：追踪剖面路径，识别悬挑、凹腔、外突等局部几何特征。
5. **三维重建与应用**：跨剖面聚合结构特征，重建结构对象，估算潜在危险块体体积，并分析不同尺度块体的空间密度分布。
6. **结果展示**：生成模型重建、剖面识别、危险块体体积排序和 KDE 空间分布等展示图像。

## 仓库结构

```text
gds_project/                  公共配置读取工具
configs/                      示例项目配置
scripts/00_project/           项目配置生成与检查
scripts/01_model_io/          模型、点云、3MX 读取与检查
scripts/02_scanline_setup/    基线拟合与数字测线准备
scripts/03_slicing_profiles/  网格切片与二维剖面提取
scripts/04_structure_recognition/ 剖面结构识别与跨剖面关联
scripts/05_reconstruction_volume/ 三维重建、体积、体素与 KDE 分析
scripts/06_visualization/     绘图与可视化脚本
scripts/99_experiments/       实验性或旧版原型
docs/                         静态展示页和项目文档
```

## 配置方式

脚本中的本地数据路径已经统一迁移到配置文件，不再写死在源码里。脚本中的：

```python
from gds_project.config import get_path, get_font_properties
```

引用的是本仓库内的 `gds_project/config.py`，不是外部依赖。`get_path()` 会优先读取 `configs/project.local.json`，如果本地配置不存在，则回退到 `configs/project.example.json`。

初始化本地配置：

```powershell
python scripts/00_project/project_manager.py init
python scripts/00_project/project_manager.py check
```

然后编辑 `configs/project.local.json`，填入自己的私有网格、点云、中间结果和输出路径。该文件默认不会上传到 GitHub。

## 静态展示页

项目展示页面位于 `docs/`。启用 GitHub Pages 后，外部访问入口为：

```text
https://zztianyi.github.io/geo-digital-scanline-GDS-/
```

本地也可以直接打开 `docs/index.html` 查看展示页。

## 当前状态

当前仓库已经整理为可公开查看的研究原型，包含数字测线基础流程、部分三维重建与体积估算应用、KDE 空间分布展示和静态说明页面。后续工作将继续优化基础识别框架，提高复杂结构识别精度与泛化能力，并扩展到更广泛的岩体结构识别场景。

## 许可证

Apache License 2.0。
