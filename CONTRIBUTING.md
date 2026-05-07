# Contributing

Thank you for considering a contribution. This repository is a research-oriented toolkit, so contributions are most useful when they improve reproducibility, documentation, testability, or the stability of the digital scanline workflow.

## Good First Contributions

- Fix documentation errors or unclear script descriptions.
- Add small example datasets that are safe to redistribute.
- Improve configuration examples without adding private paths.
- Refactor scripts in small, reviewable steps.
- Add tests for geometry utilities, path ordering, clustering, or data conversion helpers.

## Privacy and Data Rules

Do not commit private field data, site names, institution names, raw large 3D models, local absolute paths, unpublished thesis text, or images that reveal sensitive locations. Check `docs/privacy_checklist.md` before opening a pull request.

## Pull Request Checklist

- The change is scoped and described clearly.
- Private paths and case identifiers have been removed.
- Generated outputs and heavy model files are not committed.
- New scripts include a short usage note or are listed in `docs/script_inventory.md` and `docs/script_inventory.zh-CN.md`.
- The contribution is made under the Apache-2.0 license and follows the Developer Certificate of Origin in `DCO.md`.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use local configuration files copied from `configs/*.example.json`. Local config files should stay untracked.
