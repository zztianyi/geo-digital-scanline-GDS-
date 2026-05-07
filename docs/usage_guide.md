# Usage Guide

This guide describes how to run the public code layout after cloning the repository.

## 1. Create a local project config

```bash
python scripts/00_project/project_manager.py init
python scripts/00_project/project_manager.py check
```

Edit `configs/project.local.json` and replace the example paths with your own mesh, point-cloud, intermediate-output, and figure-output paths. Keep this file local.

## 2. Follow the workflow folders

Run scripts in this general order:

1. `scripts/01_model_io/` for model and point-cloud inspection.
2. `scripts/02_scanline_setup/` for baseline fitting and scanline preparation.
3. `scripts/03_slicing_profiles/` for mesh slicing and profile extraction.
4. `scripts/04_structure_recognition/` for profile-level recognition and cross-profile linking.
5. `scripts/05_reconstruction_volume/` for reconstruction, volume, voxel, and KDE analysis.
6. `scripts/06_visualization/` for figures and visual checks.

The scripts are research utilities rather than a single packaged command-line application. Each stage exposes intermediate files so the workflow can be inspected and adjusted.
