# Usage Guide

The scripts are research prototypes. They are organized by workflow stage rather than as a packaged Python library.

## 1. Prepare Data

Required inputs usually include:

- A 3D mesh: `.glb`, `.obj`, `.ply`, or 3MX-derived mesh.
- Optional point cloud: TXT with `x y z r g b`, LAS/LAZ, or PLY.
- Arc/baseline configuration: see `configs/arc_config.example.json`.

Private data should be stored in `data/private/` or outside the repository.

## 2. Fit Baseline and Generate Arc Config

Use `2d_feat.py` to filter point-cloud data and fit an arc baseline. The fitted center, radius, angle range, and height range are saved to a config JSON.

Related scripts:

- `2d_feat.py`
- `2d_ceshi.py`
- `projection.py`
- `Vertical_Slice.py`

## 3. Slice Mesh into Digital Scanlines

Use `slice_generator.py` or the slicing utilities in `Multi_profit.py` to compute intersections between scanline planes and the triangular mesh.

Outputs are typically pickle files containing line segments and face indices. Keep those outputs under `outputs/intermediate/`.

## 4. Extract and Order Profile Structures

Use path-ordering and connectivity utilities in `image_.py`, `part.py`, `Multi_profit.py`, and `generate_normals.py` to merge fragmented line segments and compute local normals.

## 5. Group Features Across Profiles

Use `Multi_profit.py`, `group_centroid.py`, and `line_face_extraction.py` to collect feature groups, centroids, and line-face correspondence across slices.

## 6. Reconstruct and Quantify Candidate Blocks

Use `structural.py`, `surface_merge.py`, `voxel_ceshi.py`, `point_dbscan.py`, and `kde.py` for face clustering, voxelization, volume estimation, density visualization, and block centroid extraction.

## 7. Plot and Export Figures

Use `plt_.py`, `plt_ploter.py`, `kde.py`, `slice_generator_ceshi.py`, and `generate_normals.py` for visualization and publication-style figures.

## Notes

Some scripts still expose configuration variables near their `main` sections. For a new dataset, update those paths to point to your local `configs/paths.local.json` or your own data locations.
