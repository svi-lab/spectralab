"""Convert cosmic-ray widths between spectral units (nm / cm⁻¹) and channels.

A cosmic ray's *physical* width (set by the detector pixel it lands on and
optics) is fixed; its *channel* width depends on the grating dispersion, so a
2 nm cosmic ray is 3 channels on a coarse grating and 60 channels on a fine
one. The 1D detection math in :mod:`mask_1d` stays channel-based (that is
what the existing tests pin); this module is the one place the spectral axis
is read to resolve a physical width into a channel count for a given file.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def median_channel_width(da: xr.DataArray, spectral_dim: str) -> float:
    """Median spacing between adjacent channels along ``spectral_dim``.

    Uses the median (not mean) so a handful of unevenly spaced or duplicate
    channels near the ends of the axis don't skew the estimate. Returns the
    absolute value — callers don't care whether the axis runs ascending or
    descending (e.g. nm vs. eV).
    """
    coord = np.asarray(da.coords[spectral_dim].values, dtype=float)
    if coord.size < 2:
        raise ValueError(
            f"median_channel_width: spectral axis {spectral_dim!r} has "
            f"fewer than 2 channels (size={coord.size})"
        )
    spacing = np.median(np.abs(np.diff(coord)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(
            f"median_channel_width: could not resolve a positive spacing "
            f"for {spectral_dim!r} (got {spacing})"
        )
    return float(spacing)


def spectral_to_channels(
    span: float,
    channel_width: float,
    *,
    minimum: int = 3,
    odd: bool = False,
) -> int:
    """Convert a physical span to a channel count for a given dispersion.

    ``span`` and ``channel_width`` must be in the same spectral unit. The
    result is rounded, floored at ``minimum``, and forced odd when
    ``odd=True`` (rounds up so the window never shrinks below the request).
    ``span <= 0`` returns 0 regardless of ``minimum`` — the caller uses 0 to
    mean "disabled" (e.g. the broad pass).
    """
    if span <= 0:
        return 0
    if channel_width <= 0 or not np.isfinite(channel_width):
        raise ValueError(f"channel_width must be positive and finite, got {channel_width}")
    n = int(round(span / channel_width))
    n = max(n, minimum)
    if odd and n % 2 == 0:
        n += 1
    return n
