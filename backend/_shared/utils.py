# -*- coding: utf-8 -*-
"""Shared runtime helpers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


def ensure_in_memory(
    da: xr.DataArray,
    caller: str,
    reason: str,
    stacklevel: int = 3,
) -> xr.DataArray:
    """If *da* is Dask-backed, emit a warning and compute it into RAM.

    Returns the DataArray unchanged if already in memory.
    """
    if da.chunks is not None:
        import numpy as np
        size_gb = da.nbytes / 2**30
        warnings.warn(
            f"{caller} received a Dask-backed DataArray "
            f"(shape {tuple(da.shape)}, ~{size_gb:.2f} GB). "
            f"{reason} Computing the full array into RAM now.",
            UserWarning,
            stacklevel=stacklevel,
        )
        return da.compute()
    return da


__all__ = ["ensure_in_memory"]
