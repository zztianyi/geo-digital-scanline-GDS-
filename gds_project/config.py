"""Project configuration utilities for the digital scanline framework.

The research scripts in this repository should not contain machine-specific
absolute paths. Store local paths in ``configs/project.local.json`` and keep
that file out of version control. When the local file is absent, the helper
falls back to ``configs/project.example.json`` so examples remain readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Any


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the repository root that contains ``configs`` and ``gds_project``."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "gds_project").exists():
            return candidate
    return Path.cwd().resolve()


@dataclass(frozen=True)
class ProjectConfig:
    """Loaded project configuration with path resolution helpers."""

    root: Path
    data: dict[str, Any]
    source: Path

    def path(self, key: str, *, create_parent: bool = False) -> Path:
        """Resolve a configured path key to an absolute path."""
        try:
            value = self.data["paths"][key]
        except KeyError as exc:
            raise KeyError(f"Path key not found in project config: {key}") from exc
        path = Path(os.path.expandvars(str(value))).expanduser()
        if not path.is_absolute():
            path = self.root / path
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def value(self, dotted_key: str, default: Any = None) -> Any:
        """Read a nested config value using a dotted key such as ``scanline.step``."""
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load ``project.local.json`` when available, otherwise the example config."""
    root = find_project_root()
    if config_path is None:
        local = root / "configs" / "project.local.json"
        example = root / "configs" / "project.example.json"
        config_path = local if local.exists() else example
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProjectConfig(root=root, data=data, source=path)


def get_path(key: str, *, create_parent: bool = False, config_path: str | Path | None = None) -> Path:
    """Convenience wrapper for resolving one configured path key."""
    return load_project_config(config_path).path(key, create_parent=create_parent)


def get_font_properties(size: int = 12):
    """Return a Matplotlib Chinese font when configured and available."""
    from matplotlib.font_manager import FontProperties

    config = load_project_config()
    font_path = config.path("font_zh")
    if font_path.exists():
        return FontProperties(fname=str(font_path), size=size)
    return FontProperties(size=size)
