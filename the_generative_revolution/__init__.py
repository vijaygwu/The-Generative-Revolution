"""Convenience package exports for the Book 3 companion."""

from __future__ import annotations

from src import *  # noqa: F401,F403
from src import __all__ as _src_all

__version__ = "0.1.0"

__all__ = [*list(_src_all), "__version__"]
