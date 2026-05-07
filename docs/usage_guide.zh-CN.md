# 中文使用指南

这个指南面向下载仓库后准备复现实验或接入自己数据的使用者。

## 1. 创建本地项目配置

```powershell
python scripts/00_project/project_manager.py init
python scripts/00_project/project_manager.py check
```

然后编辑 `configs/project.local.json`，把示例路径替换为自己的网格、点云、中间结果和图像输出路径。这个本地配置文件默认不会上传 GitHub。

## 2. 按流程运行脚本

建议按下面顺序理解和运行：

1. `scripts/01_model_io/`：模型、点云、3MX 资源检查。
2. `scripts/02_scanline_setup/`：弧形基线拟合和数字测线准备。
3. `scripts/03_slicing_profiles/`：网格切片和二维剖面提取。
4. `scripts/04_structure_recognition/`：剖面结构识别和跨剖面关联。
5. `scripts/05_reconstruction_volume/`：三维重建、体积、体素和 KDE 分析。
6. `scripts/06_visualization/`：结果绘图和可视化检查。

当前代码更接近研究工作流，而不是封装好的单一命令行软件。保留中间文件是有意设计，方便检查每个步骤的几何结果和识别效果。
