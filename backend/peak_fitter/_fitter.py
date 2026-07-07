# -*- coding: utf-8 -*-
"""Gaussian multi-peak deconvolution for a single spectrum: :class:`PeakFitter`.

Band centers are user-specified; amplitude and width are inferred from the
data when not given explicitly. Built on lmfit's ``GaussianModel``, which
parametrizes ``amplitude`` as the integrated peak area (not height) and
auto-derives ``fwhm`` via a constraint expression — this directly produces
the position/amplitude/width/area vocabulary needed for result statistics,
with a fitted standard error on every parameter, for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from lmfit import Parameters
from lmfit.models import GaussianModel


@dataclass
class BandSpec:
    """One user-specified band to fit.

    ``shape`` is a seam for a future Lorentzian/Voigt option — v1 only
    implements ``"gaussian"``; adding a shape later means widening this
    Literal and adding one branch in ``_build_composite_model``, not
    restructuring this dataclass.
    """

    center_guess: float
    shape: Literal["gaussian"] = "gaussian"
    center_min: float | None = None
    center_max: float | None = None
    amplitude_guess: float | None = None
    sigma_guess: float | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None
    label: str | None = None


@dataclass
class BandResult:
    label: str
    center: float
    center_stderr: float | None
    amplitude: float
    amplitude_stderr: float | None
    sigma: float
    sigma_stderr: float | None
    fwhm: float
    fwhm_stderr: float | None
    area: float
    area_pct: float
    curve: np.ndarray


@dataclass
class FitResult:
    bands: list[BandResult]
    x: np.ndarray
    y_data: np.ndarray
    y_fit: np.ndarray
    residual: np.ndarray
    r_squared: float
    reduced_chi_square: float
    aic: float
    bic: float
    success: bool
    message: str
    raw_lmfit_result: Any


def _estimate_sigma_guess(
    x: np.ndarray,
    n_bands: int,
    center_min: float | None = None,
    center_max: float | None = None,
) -> float:
    """A starting width: scaled to the band's own bound window when given,
    otherwise the visible spectral span divided among the bands.

    A uniform span-based guess can badly mismatch a tightly-bounded band
    (e.g. one of many closely-spaced literature positions, each bounded to
    only a few nm so neighboring bands can't swap positions) — starting a
    ~6 nm-wide guess inside a 1.5 nm-wide allowed window ill-conditions the
    fit badly enough to prevent convergence entirely. Scaling to the band's
    own window keeps the initial guess on the same scale as what it can
    actually explore.
    """
    if center_min is not None and center_max is not None:
        window = center_max - center_min
        if window > 0:
            return max(window / 4, np.finfo(float).eps)
    span = float(np.max(x) - np.min(x))
    return max(span / (4 * max(n_bands, 1)), np.finfo(float).eps)


def _estimate_amplitude_guess(
    x: np.ndarray, y: np.ndarray, center: float, sigma: float
) -> float:
    """Invert the Gaussian peak-height formula using the data value nearest
    ``center``, so the initial amplitude (area) guess is on the right order
    of magnitude for this spectrum's actual intensity scale — lmfit's own
    ``make_params()`` default of 1.0 is meaningless on arbitrary axes."""
    idx = int(np.argmin(np.abs(x - center)))
    height = max(float(y[idx]), np.finfo(float).eps)
    return height * sigma * np.sqrt(2 * np.pi)


def _build_composite_model(
    x: np.ndarray,
    y: np.ndarray,
    bands: list[BandSpec],
) -> tuple[Any, Parameters, list[str]]:
    bad_shapes = [b.shape for b in bands if b.shape != "gaussian"]
    if bad_shapes:
        raise NotImplementedError(
            f"Only shape='gaussian' is implemented; got {bad_shapes}"
        )

    model = None
    params = Parameters()
    prefixes: list[str] = []
    for i, band in enumerate(bands):
        prefix = f"b{i}_"
        prefixes.append(prefix)
        gm = GaussianModel(prefix=prefix)
        model = gm if model is None else model + gm

        sigma_guess = (
            band.sigma_guess
            if band.sigma_guess is not None
            else _estimate_sigma_guess(x, len(bands), band.center_min, band.center_max)
        )
        amplitude_guess = (
            band.amplitude_guess
            if band.amplitude_guess is not None
            else _estimate_amplitude_guess(x, y, band.center_guess, sigma_guess)
        )

        p = gm.make_params()
        p[f"{prefix}center"].set(
            value=band.center_guess, min=band.center_min, max=band.center_max
        )
        p[f"{prefix}sigma"].set(
            value=sigma_guess,
            min=band.sigma_min if band.sigma_min is not None else np.finfo(float).eps,
            max=band.sigma_max,
        )
        p[f"{prefix}amplitude"].set(value=amplitude_guess, min=0)
        params.update(p)

    return model, params, prefixes


class PeakFitter:
    """Fit a sum of Gaussian bands to a single spectrum."""

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bands: list[BandSpec],
        *,
        params_init: Parameters | None = None,
    ) -> FitResult:
        """Build a composite Gaussian model from ``bands``, fit, and return
        per-band parameters plus overall fit statistics.

        NaN in ``y`` (paired with ``x``) is dropped before fitting — lmfit's
        least-squares fit does not accept NaN. Raises ``ValueError`` if
        fewer than ``3 * len(bands)`` valid points remain (degenerate fit).

        ``params_init``, when given, seeds the fit from a previous result's
        converged parameters (values *and* bounds) instead of the
        ``bands`` guesses — used by :func:`peak_fitter.fit_map_gaussian` to
        warm-start neighboring pixels.
        """
        if not bands:
            raise ValueError("PeakFitter.fit needs at least one BandSpec")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x_fit, y_fit_data = x[finite], y[finite]

        if x_fit.size < 3 * len(bands):
            raise ValueError(
                f"Not enough valid (non-NaN) points to fit {len(bands)} "
                f"band(s): need >= {3 * len(bands)}, got {x_fit.size}"
            )

        model, params, prefixes = _build_composite_model(x_fit, y_fit_data, bands)
        if params_init is not None:
            for name in params:
                if name in params_init:
                    params[name].set(
                        value=params_init[name].value,
                        min=params_init[name].min,
                        max=params_init[name].max,
                    )

        # lmfit's default 'leastsq' (MINPACK) handles bounds via an internal
        # reparametrization that conditions badly for tightly-bounded parameters
        # (e.g. closely-spaced preset bands) -- it can burn through the entire
        # max_nfev budget without converging. scipy's natively-bounded
        # 'least_squares' handles the same problem in a fraction of the calls.
        result = model.fit(y_fit_data, params, x=x_fit, method="least_squares")

        y_fit_curve = result.eval(x=x_fit)
        residual = y_fit_data - y_fit_curve
        comps = result.eval_components(x=x_fit)

        amplitudes = [result.params[f"{p}amplitude"].value for p in prefixes]
        total_area = sum(amplitudes) or 1.0

        bands_out: list[BandResult] = []
        for i, (band, prefix) in enumerate(zip(bands, prefixes)):
            label = band.label or f"Band {i + 1}"
            amp = result.params[f"{prefix}amplitude"]
            cen = result.params[f"{prefix}center"]
            sig = result.params[f"{prefix}sigma"]
            fwhm = result.params[f"{prefix}fwhm"]
            bands_out.append(
                BandResult(
                    label=label,
                    center=cen.value,
                    center_stderr=cen.stderr,
                    amplitude=amp.value,
                    amplitude_stderr=amp.stderr,
                    sigma=sig.value,
                    sigma_stderr=sig.stderr,
                    fwhm=fwhm.value,
                    fwhm_stderr=fwhm.stderr,
                    area=amp.value,
                    area_pct=100.0 * amp.value / total_area,
                    curve=np.asarray(comps[prefix]),
                )
            )

        r_squared = getattr(result, "rsquared", None)
        if r_squared is None:
            ss_res = float(np.sum(residual**2))
            ss_tot = float(np.sum((y_fit_data - y_fit_data.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        return FitResult(
            bands=bands_out,
            x=x_fit,
            y_data=y_fit_data,
            y_fit=y_fit_curve,
            residual=residual,
            r_squared=float(r_squared),
            reduced_chi_square=float(result.redchi),
            aic=float(result.aic),
            bic=float(result.bic),
            success=bool(result.success),
            message=result.message or ("Fit succeeded" if result.success else "Fit failed"),
            raw_lmfit_result=result,
        )


__all__ = ["PeakFitter", "BandSpec", "BandResult", "FitResult"]
