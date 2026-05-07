# 岩壁结构信息提取工具集

这是一个面向复杂岩壁数字地质模型的研究型 Python 工具集，用于从三维网格、点云和剖面测线中提取、表达和可视化岩壁结构信息。项目已经做过开源脱敏处理：真实地点、单位、本地磁盘路径和原始工程数据均已移除或泛化。

## 项目能做什么

- 基于岩壁轮廓拟合圆弧基线，生成自适应数字测线。
- 从三角网格模型中提取剖面交线。
- 将三维剖面映射到二维空间，整理碎片化线段路径。
- 根据局部法向量/正交向量场识别悬挑、凹腔、外突等几何结构特征。
- 跨测线聚类和组合结构单元。
- 使用网格、体素、KDE 和 DBSCAN 等方法做体积估计、密度分析和可视化。

## 展示页面

项目效果、数字测线框架、应用结论、当前进度和未来愿景已整理为中英文独立展示页：[中文](docs/showcase.zh-CN.html) / [English](docs/showcase.en.html)。

## 推荐阅读顺序

1. `README.md`：GitHub 首页英文说明。
2. `docs/open_source_description.md`：由论文内容改写后的开源项目描述。
3. `docs/usage_guide.md`：处理流程说明。
4. `docs/script_inventory.zh-CN.md`：每个脚本的中文用途索引。
5. `docs/privacy_checklist.md`：发布前脱敏检查清单。

## 目录说明

```text
configs/          示例配置文件
data/examples/    示例数据格式说明，不含真实数据
docs/images/      可公开展示的效果截图
docs/             方法说明、脚本索引、使用指南和隐私检查
outputs/          本地运行输出，默认不上传
*.py              原型脚本和处理模块
```

## 上传 GitHub 前注意

建议上传当前这个发布包目录，而不是原始工作目录。发布包已经排除了大体积点云、原始截图、相机 JSON、缓存和私有数据目录。

当前发布包已经补齐 Apache-2.0 开源协议文件，包括 `LICENSE`、`NOTICE`、`AUTHORS.md`、`CONTRIBUTING.md`、`DCO.md` 和 `CITATION.cff`。







