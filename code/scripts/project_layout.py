"""Resolve historical model archives in the working tree and extracted bundles."""
from pathlib import Path


def archive_path(project_root, filename):
    root = Path(project_root).resolve()
    bundled = root / filename
    if bundled.is_file():
        return bundled
    workspace = root.parent / "results" / "archives" / filename
    if root.name == "code" and workspace.is_file():
        return workspace
    return bundled
