# Open-Source Method Description

This project provides a digital scanline-based workflow for complex rock-wall structural information extraction from 3D geological models.

## Sanitized Project Description

High-steep rock walls often contain discontinuities, cavities, weathered recesses, and overhanging blocks whose geometry is difficult to capture with manual scanline surveys alone. This toolkit explores a digital alternative: using point clouds and triangular mesh models as the geometric basis, fitting an adaptive curved baseline, generating a sequence of section planes, extracting 2D profile paths from mesh intersections, and mapping recognized structural features back into 3D space.

The workflow supports three connected tasks:

1. **Digital scanline construction**: fit a curved baseline to the rock-wall outline and generate scanline planes with controllable spacing.
2. **Profile-level feature recognition**: convert fragmented mesh intersection segments into ordered 2D profile paths and identify geometric overhang features through local vector/normal analysis.
3. **3D structural reconstruction and quantification**: group features across adjacent profiles, reconstruct candidate structural units, and estimate their volume or density through mesh/voxel operations.

## What Was Removed or Generalized

- Specific case-site names were replaced with generic terms such as "study site" or "sample rock wall".
- Author, student ID, institution, supervisor, and thesis metadata are not included.
- Local disk paths and production model folder names were replaced by `data/private/...` and `configs/...` placeholders.
- Raw survey data and large meshes are excluded from the repository.

## Suggested GitHub Short Description

Digital scanline toolkit for extracting and representing structural features from 3D rock-wall meshes and point clouds.

## Suggested Topics

`geology`, `engineering-geology`, `point-cloud`, `mesh-processing`, `rockfall`, `scanline`, `pyvista`, `open3d`, `trimesh`, `geospatial-analysis`
