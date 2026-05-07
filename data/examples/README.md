# Example Data

This directory intentionally contains only placeholders and format notes. Do not commit private survey data, raw meshes, or large point clouds.

Expected point-cloud TXT format for simple examples:

```text
x y z r g b
0.0 0.0 0.0 120 120 120
```

Place real data under `data/private/` or outside the repository, then reference it from `configs/paths.local.json`.
