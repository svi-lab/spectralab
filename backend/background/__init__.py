# -*- coding: utf-8 -*-
"""backend.background — substrate PL background suppression."""

from ._suppressor import BackgroundSuppressor
from ._scale import interp_reference

__all__ = [
    "BackgroundSuppressor",
    "interp_reference",
]
