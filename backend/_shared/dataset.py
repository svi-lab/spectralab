# -*- coding: utf-8 -*-
"""Typed container for a single loaded WDF file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

_KNOWN_UNITS = frozenset(
    {"RamanShift", "Wavenumber", "Nanometer", "ElectronVolt"}
)


def _validate(da: xr.DataArray, spectral_units: str) -> tuple[bool, str]:
    if not spectral_units:
        return False, "Spectral units not found in file"
    if spectral_units not in _KNOWN_UNITS:
        return False, f"Unrecognised spectral units: {spectral_units!r}"
    if da.size == 0:
        return False, "Empty data array"
    return True, ""


@dataclass
class SpectralDataset:
    """Immutable-intent container created once at file load time.

    The ``da`` field holds the raw DataArray and should never be modified
    in place after construction.  All preprocessing returns new DataArrays.
    """

    da:             xr.DataArray
    spectral_dim:   str
    spectral_units: str        # WiRE canonical: "Nanometer"|"ElectronVolt"|"RamanShift"|"Wavenumber"
    spectral_unit:  str        # coord units string: "nm"|"eV"|"1/cm"|"cm^-1"
    spec_min:       float
    spec_max:       float
    laser_nm:       float | None
    is_map:         bool
    image_arr:      np.ndarray | None
    image_meta:     dict[str, Any] | None
    laser_power:    float              # NaN if not reported
    exposure_time:  float              # NaN if not reported
    dims:           tuple
    shape:          tuple
    ndim:           int
    is_valid:       bool
    validation_msg: str                # "" when valid

    @property
    def measurement_kind(self) -> str:
        """'PL' for Nanometer/ElectronVolt; 'Raman' for RamanShift/Wavenumber."""
        if self.spectral_units in ("Nanometer", "ElectronVolt"):
            return "PL"
        if self.spectral_units in ("RamanShift", "Wavenumber"):
            return "Raman"
        return "Unknown"

    @property
    def preprocessing_available(self) -> bool:
        """CosmicRayRemover and Denoiser are enabled for PL data only."""
        return self.measurement_kind == "PL"
