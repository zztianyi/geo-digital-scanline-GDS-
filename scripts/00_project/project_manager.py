"""Create and maintain the standard project file for the digital scanline framework.

Examples
--------
Create a local config from the template::

    python scripts/00_project/project_manager.py init

Show resolved paths and missing inputs::

    python scripts/00_project/project_manager.py check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gds_project.config import load_project_config

EXAMPLE = PROJECT_ROOT / "configs" / "project.example.json"
LOCAL = PROJECT_ROOT / "configs" / "project.local.json"


def init_config(force: bool = False) -> None:
    """Create ``configs/project.local.json`` from the example template."""
    if LOCAL.exists() and not force:
        print(f"Local config already exists: {LOCAL}")
        print("Use --force to overwrite it.")
        return
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLE, LOCAL)
    print(f"Created local project config: {LOCAL}")
    print("Edit this file to point to your private meshes, point clouds, and outputs.")


def check_config() -> None:
    """Print configured paths and flag missing resources."""
    config = load_project_config()
    print(f"Project root: {config.root}")
    print(f"Config file:  {config.source}")
    print("\nConfigured paths:")
    for key in sorted(config.data.get("paths", {})):
        path = config.path(key)
        status = "exists" if path.exists() else "missing"
        print(f"  {key:24s} {status:8s} {path}")


def show_config() -> None:
    """Print the active project config as JSON."""
    config = load_project_config()
    print(json.dumps(config.data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the digital scanline project config.")
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init", help="Create configs/project.local.json from the template.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing local config.")
    sub.add_parser("check", help="Check resolved paths in the active config.")
    sub.add_parser("show", help="Print the active project config.")
    args = parser.parse_args()
    if args.command == "init":
        init_config(force=args.force)
    elif args.command == "check":
        check_config()
    elif args.command == "show":
        show_config()


if __name__ == "__main__":
    main()
