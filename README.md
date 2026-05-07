# Digital Scanline Framework for Complex Rock-Wall Structure Recognition

[中文说明](README.zh-CN.md) | [Project showcase](docs/index.html) | [Usage guide](docs/usage_guide.md) | [Script inventory](docs/script_inventory.md)

This repository contains a research-oriented Python workflow for extracting, representing, and visualizing structural information from complex rock-wall models. The framework uses **digital scanlines** as the organizing idea: a 3D rock-wall mesh or point cloud is sampled into a controlled sequence of 2D profile observations, geometric features are recognized on those profiles, and the results are reconstructed back into 3D for structural interpretation, candidate hazard-block volume estimation, and spatial density analysis.

The project is intended for engineering-geology and geomatics research workflows where raw 3D geometry is rich but difficult to interpret directly. Instead of treating the 3D model only as a visual object, the framework turns it into a set of computable structural observations.

## Core Idea

Complex rock-wall structures are continuous in space, while digital scanlines are discrete observations. The framework controls scanline density so that discrete profile samples can represent continuous structural behavior at a chosen analysis scale. This reduces the difficulty of algorithm design, makes intermediate results easier to inspect, and keeps the final interpretation anchored in the original 3D geometry.

## What The Workflow Does

1. **Model input and inspection**: load point clouds, triangular meshes, and 3MX-derived resources.
2. **Digital scanline setup**: fit a curved baseline and generate scanline planes with configurable spacing.
3. **Profile extraction**: slice the 3D model into ordered 2D structural observation windows.
4. **Structure recognition**: trace profile paths and identify local geometric features such as overhangs, cavities, and protrusions.
5. **3D reconstruction and applications**: group features across neighboring profiles, reconstruct structural objects, estimate candidate block volumes, and analyze spatial density distributions.
6. **Visualization**: produce showcase figures for model reconstruction, profile recognition, hazard-block volume ranking, and KDE-style spatial distribution maps.

## Repository Layout

```text
gds_project/                  Shared configuration helpers
configs/                      Example project configuration files
scripts/00_project/           Project config creation and inspection
scripts/01_model_io/          Mesh, point-cloud, and 3MX input utilities
scripts/02_scanline_setup/    Baseline fitting and scanline preparation
scripts/03_slicing_profiles/  Mesh slicing and 2D profile extraction
scripts/04_structure_recognition/ Profile-level and cross-profile recognition
scripts/05_reconstruction_volume/ 3D reconstruction, volume, voxel, and KDE analysis
scripts/06_visualization/     Plotting and visual inspection scripts
scripts/99_experiments/       Experimental or legacy prototypes
docs/                         Static showcase pages and documentation
```

## Configuration

Machine-specific paths are intentionally not hard-coded in the scripts. They are resolved through:

```python
from gds_project.config import get_path, get_font_properties
```

`gds_project.config` is part of this repository. `get_path()` reads path keys from `configs/project.local.json` when it exists, otherwise it falls back to `configs/project.example.json`. Create a local config before running project scripts:

```bash
python scripts/00_project/project_manager.py init
python scripts/00_project/project_manager.py check
```

Edit `configs/project.local.json` to point to your private meshes, point clouds, intermediate outputs, and figure folders. This local file is ignored by Git.

## Static Showcase

The project showcase is a static site under `docs/`. When GitHub Pages is enabled for this repository, the public entry page is:

```text
https://zztianyi.github.io/geo-digital-scanline-GDS-/
```

The same content can be opened locally from `docs/index.html`.

## Status

The current repository is a cleaned research prototype. The core digital scanline workflow, partial 3D reconstruction path, volume-oriented application example, and visualization pages are organized for public review. Future work will focus on stronger recognition accuracy, better generalization across rock-wall types, and tighter fusion between 2D profile reasoning and 3D model analysis.

## License

Apache License 2.0.
