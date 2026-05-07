# Script Inventory: Digital Scanline Framework for Complex Rock-Wall Structure Recognition

Scripts are organized by workflow stage. Machine-specific paths are managed through `configs/project.example.json` and `configs/project.local.json`; create the local file with `python scripts/00_project/project_manager.py init`.

## Workflow Folders

- `scripts/00_project/`: project config generation and inspection.
- `scripts/01_model_io/`: model, point-cloud, and 3MX input utilities.
- `scripts/02_scanline_setup/`: baseline fitting, scanline grid preview, and point-cloud projection.
- `scripts/03_slicing_profiles/`: mesh slicing and profile extraction.
- `scripts/04_structure_recognition/`: multi-profile recognition, normals, centroids, and line-face links.
- `scripts/05_reconstruction_volume/`: surface merging, block/voxel reconstruction, volume, KDE, and clustering.
- `scripts/06_visualization/`: figure generation and visual inspection scripts.
- `scripts/99_experiments/`: experimental and legacy prototypes.
