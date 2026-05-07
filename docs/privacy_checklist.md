# Privacy and Release Checklist

Before pushing to GitHub, check the following items:

- [x] Remove author name, student ID, institution, supervisor, and thesis administrative metadata.
- [x] Replace specific case-site names with generic terms.
- [x] Replace local absolute paths with `configs/...`, `data/private/...`, or `outputs/...` placeholders.
- [x] Exclude large/raw files such as `.ply`, `.obj`, `.glb`, `.las`, `.pkl`, and 3MX scene folders.
- [x] Keep only non-sensitive visualization screenshots under `docs/images/`.
- [ ] Choose and add an open-source license, such as MIT, Apache-2.0, GPL-3.0, or a research-only license if needed.
- [ ] Decide whether to publish a tiny synthetic sample dataset.
- [ ] Run the workflow on a synthetic or public dataset and record exact commands.
- [ ] Review all figures manually for hidden map coordinates, labels, personal names, and site-specific annotations.

Useful scan command:

```powershell
rg -n --hidden -S "real-site-name|author-name|student-id|absolute-private-path" .
```

