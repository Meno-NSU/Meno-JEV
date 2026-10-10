"""Auto-discovery of model adapters.

Importing this module imports every other module in :mod:`jev.models`, so any
adapter that calls ``@register_model(...)`` at import time becomes available.
To add a new family, drop a new file in this package - nothing else changes.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import List

import jev.models as _pkg

_SKIP = {"base", "registry", "__init__"}


def load_builtin_adapters() -> List[str]:
    discovered: List[str] = []
    for mod in pkgutil.iter_modules(_pkg.__path__):
        # Files whose name starts with "_" are templates/examples, not loaded.
        if mod.name in _SKIP or mod.name.startswith("_"):
            continue
        importlib.import_module(f"{_pkg.__name__}.{mod.name}")
        discovered.append(mod.name)
    return discovered


load_builtin_adapters()
