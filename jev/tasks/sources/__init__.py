"""Per-source recipe overrides.

Every module in this package is imported automatically; any recipe registered
there (with a priority higher than the generic one) takes over for the sources
its ``matches`` accepts. Add a new file here to special-case a source without
touching the generic recipe or any other source.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import List

import jev.tasks.sources as _pkg


def load_builtin_sources() -> List[str]:
    discovered: List[str] = []
    for mod in pkgutil.iter_modules(_pkg.__path__):
        # Files whose name starts with "_" are templates/examples, not loaded.
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"{_pkg.__name__}.{mod.name}")
        discovered.append(mod.name)
    return discovered


load_builtin_sources()
