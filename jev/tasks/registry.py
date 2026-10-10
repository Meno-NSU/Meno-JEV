"""Import the built-in recipes so they register themselves.

Generic recipes live in :mod:`jev.tasks.generic`; per-source overrides live in
:mod:`jev.tasks.sources`. Adding a file to ``sources/`` registers it
automatically - no edits here.
"""

from __future__ import annotations

from . import generic as _generic  # noqa: F401
