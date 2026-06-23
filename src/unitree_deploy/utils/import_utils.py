from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


@contextmanager
def search_path(path: Path):
    """Temporarily add *path* to the front of ``sys.path``."""
    path_str = str(path)
    added = path_str not in sys.path
    if added:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def import_module(module_name: str, search_dir: Path) -> ModuleType:
    """Import *module_name* with *search_dir* temporarily on ``sys.path``."""
    with search_path(search_dir):
        return importlib.import_module(module_name)


def import_symbol(spec: str, search_dir: Path):
    """Import a symbol from ``<module>:<symbol>`` spec.

    Example: ``"my_policy:MyPolicy"`` → ``MyPolicy`` class.
    """
    module_name, separator, symbol_name = spec.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"import spec must be '<module>:<symbol>', got {spec!r}")

    module = import_module(module_name, search_dir)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} has no symbol {symbol_name!r}"
        ) from exc