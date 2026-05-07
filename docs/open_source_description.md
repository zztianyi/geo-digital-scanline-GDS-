# Project Description

**Digital Scanline Framework for Complex Rock-Wall Structure Recognition** is a research-oriented workflow for extracting structural information from 3D rock-wall models.

High-steep rock walls may contain discontinuities, cavities, weathered recesses, and overhanging blocks. Directly designing algorithms on the full 3D geometry can be difficult because the model is often fragmented, noisy, and scale-dependent. This project uses a digital scanline strategy: it samples the 3D surface into a sequence of 2D profile observations, recognizes structural features on those profiles, and reconstructs the recognized information back into 3D space.

The framework currently focuses on three connected outputs:

1. **Computable structural observations** from mesh or point-cloud geometry.
2. **3D structural reconstruction and volume-oriented interpretation**, including candidate hazard-block volume estimation.
3. **Spatial distribution visualization**, including KDE-style maps for block groups in different size ranges.

The public repository removes site-specific, institution-specific, and private file-path information. Large raw meshes and survey data are excluded; users should connect their own data through `configs/project.local.json`.
