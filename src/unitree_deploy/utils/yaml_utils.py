from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """Load a YAML mapping from *path*.

    Returns an empty dict when the file contains ``null``.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data
    