# Script Inventory

## Baseline, Projection, and Profile Setup

- `2d_feat.py`: point filtering, arc fitting, 2D comparison between straight-line and arc baselines.
- `2d_ceshi.py`: draws an arc-based 3D grid over point-cloud data.
- `projection.py`: projects point clouds to local profile coordinates.
- `Vertical_Slice.py`: simple vertical projection and high-elevation center analysis.
- `kinematic_analysis.py`: stereographic-style plane failure illustration utilities.

## 3D Model Loading and Conversion

- `3MX.py`: combines OBJ/MTL resources exported from a 3MX-like scene.
- `3MX_voxel.py`: parses/visualizes 3MX scene nodes.
- `las_point.py`: loads TXT point clouds and converts to Open3D voxel visualization.
- `mesh_dsm.py`: converts point clouds into DSM-like triangulated meshes.
- `word_.py`: quick mesh verification utility.

## Mesh Slicing and Profile Extraction

- `3MX_projection.py`: single-plane mesh slicing and 2D projection.
- `slice_generator.py`: batch slice generation and line-face output.
- `slice_generator_ceshi.py`: interactive/multi-panel slice visualization.
- `clip_.py`: slice-plane comparison and 3D clipping helpers.
- `image_.py` and `part.py`: path ordering, normal computation, and early plotting variants.

## Multi-Profile Structural Recognition

- `Multi_profit.py`: main multi-profile processing class and batch workflow.
- `generate_normals.py`: detailed vector-field, path correction, grouping, and plotting utilities.
- `group_centroid.py`: extracts group centroids from multi-slice results.
- `line_face_extraction.py`: links extracted profile lines back to mesh faces.

## Face, Voxel, Density, and Block Analysis

- `surface_merge.py`: face adjacency, clustering, and visualization.
- `structural.py`: structural face clustering, prism/voxel construction, sparse voxel output.
- `transfer.py`: samples segment points and exports extracted faces/lines.
- `voxel_ceshi.py`: KDE over voxel/centroid data.
- `point_dbscan.py` and `point——dbscan.py`: DBSCAN clustering of voxel-derived point sets.
- `plt_ploter.py`: block/cluster plotting from voxel data.
- `kde.py`: mesh coloring and density visualization.

## Plotting and Miscellaneous

- `plt_.py`: 2D/3D schematic plotting for curved and straight scanline layouts.
- `ceshi.py`: experimental mesh/face clustering workflow.
- `cluster.py`: empty placeholder retained from the prototype.
- `polt/3d.py`: small plotting helper.
