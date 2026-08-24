"""Harmonic and grating-artifact notch tests on synthetic wavelength axes."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cosmic_ray.harmonic import (
    grating_artifact_correct_dataarray,
    harmonic_correct_dataarray,
    remove_grating_artifact_from_spectrum_1d,
    remove_harmonic_notches_from_spectrum_1d,
    should_apply_nd_yag_harmonic_cleanup,
)
from tests.factories import spectrum_da

HARMONICS_NM = (266.0, 355.0, 532.0, 1064.0)


def _wavelength_axis(n: int = 850) -> np.ndarray:
    return np.linspace(250.0, 1100.0, n)


def _narrow_line(
    wl: np.ndarray, center_nm: float, amp: float = 800.0, sigma_nm: float = 0.35
) -> np.ndarray:
    return amp * np.exp(-0.5 * ((wl - center_nm) / sigma_nm) ** 2)


def _pl_plus_harmonics() -> tuple[np.ndarray, np.ndarray]:
    wl = _wavelength_axis()
    y = np.full(wl.shape, 20.0)
    y += 400.0 * np.exp(-0.5 * ((wl - 600.0) / 40.0) ** 2)
    for h in HARMONICS_NM:
        y += _narrow_line(wl, h)
    return wl, y


def _idx(wl: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(wl - nm)))


def test_355_nm_laser_triggers_harmonic_cleanup():
    assert should_apply_nd_yag_harmonic_cleanup(355.0)
    assert should_apply_nd_yag_harmonic_cleanup(354.9)
    assert not should_apply_nd_yag_harmonic_cleanup(532.0)
    assert not should_apply_nd_yag_harmonic_cleanup(None)


def test_harmonic_notch_removes_catalogue_lines_and_keeps_pl():
    wl, y = _pl_plus_harmonics()
    corr, peaks = remove_harmonic_notches_from_spectrum_1d(
        wl, y, wavenumber_axis=False, filename="synth"
    )
    assert len(peaks) == 4
    for h in HARMONICS_NM:
        i = _idx(wl, h)
        # Notch is ±1.5 nm; a 0.35 nm σ line should fall close to local baseline.
        assert corr[i] < 0.35 * y[i]
    i_pl = _idx(wl, 600.0)
    assert corr[i_pl] == pytest.approx(y[i_pl], rel=1e-6)


def test_harmonic_skipped_when_laser_is_not_355():
    wl, y = _pl_plus_harmonics()
    da = spectrum_da(y, laser_nm=532.0, wavelength=wl)
    out = harmonic_correct_dataarray(da)
    assert out is da


def test_harmonic_dataarray_notches_every_spectrum():
    wl, y = _pl_plus_harmonics()
    stack = np.vstack([y, y * 1.1])
    da = xr.DataArray(
        stack,
        dims=("point", "wavelength"),
        coords={"wavelength": wl},
        attrs={"laser_wavelength_nm": 355.0, "Filename": "synth"},
    )
    da.coords["wavelength"].attrs["units"] = "nm"
    out = harmonic_correct_dataarray(da)
    i532 = _idx(wl, 532.0)
    assert float(out.isel(point=0).values[i532]) < 0.35 * y[i532]
    treats = out.attrs.get("treatments") or {}
    assert any("harmonic_peaks_removed_nm" in v for v in treats.values())


def test_grating_artifact_notch_at_twice_laser():
    wl = _wavelength_axis()
    y = np.full(wl.shape, 20.0)
    y += 400.0 * np.exp(-0.5 * ((wl - 600.0) / 40.0) ** 2)
    y += _narrow_line(wl, 710.0)
    corr, peaks = remove_grating_artifact_from_spectrum_1d(
        wl, y, wavenumber_axis=False, laser_wavelength_nm=355.0, filename="synth"
    )
    assert peaks
    assert peaks[0] == pytest.approx(710.0, abs=2.0)
    i710 = _idx(wl, 710.0)
    assert corr[i710] < 0.35 * y[i710]
    i_pl = _idx(wl, 600.0)
    assert corr[i_pl] == pytest.approx(y[i_pl], rel=1e-6)


def test_grating_artifact_dataarray_uses_2x_laser():
    wl = _wavelength_axis()
    y = np.full(wl.shape, 20.0) + _narrow_line(wl, 710.0)
    da = spectrum_da(y, laser_nm=355.0, wavelength=wl)
    out = grating_artifact_correct_dataarray(da)
    i710 = _idx(wl, 710.0)
    assert float(out.values[i710]) < 0.35 * y[i710]
