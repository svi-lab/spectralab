# -*- coding: utf-8 -*-
"""NMF-based spectral pattern decomposition for stacks of spectra and 3D map cubes.

NMF decomposes a *population* of non-negative spectra into a small number of
non-negative basis spectra ("components") and per-spectrum abundances. Unlike
PCA, it has no eigenvalue-based explained-variance ratio and no MLE component
count — :func:`compute_nmf_diagnostic_curve` is the human-in-the-loop
replacement: it sweeps a range of component counts so the count can be chosen
from the resulting curve rather than an automatic/hidden choice. See
:class:`spectra_decomposer.Decomposer` for the user-facing xarray API.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

# The flatten / NaN-row / min-subtract plumbing is shared verbatim with
# MCR-ALS (spectra_mcr), so it lives in _shared and is re-exported here for
# backward compatibility with any callers importing it from this module.
from _shared._factorization import _flatten_to_row_stack, _nonnegative_fit_matrix


def _fit_nmf(
    matrix: np.ndarray,
    *,
    n_components: int,
    init: str,
    max_iter: int,
    random_state: int | None,
    nmf_kwargs: dict[str, Any] | None,
) -> tuple[NMF, np.ndarray, bool]:
    """Fit one NMF model, capturing whether it converged within max_iter."""
    nmf = NMF(
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        **(nmf_kwargs or {}),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        abundances = nmf.fit_transform(matrix)
        converged = not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
    return nmf, abundances, converged


def decompose_spectra_nmf(
    values: np.ndarray,
    *,
    n_components: int,
    init: str = "nndsvda",
    max_iter: int = 500,
    random_state: int | None = 0,
    nmf_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Decompose a stack/cube of spectra into non-negative basis spectra.

    The input is reshaped to ``(n_spectra, n_spectral)`` for the fit, then
    reshaped back to the original spatial layout on return.

    Parameters
    ----------
    values
        Array of shape ``(..., n_spectral)``. Needs more than one spectrum
        (NMF on a single spectrum is degenerate, same as PCA).
    n_components
        Fixed integer component count. There is no "mle"/float/None analog
        for NMF — use :func:`compute_nmf_diagnostic_curve` to choose it.
    init
        Forwarded to :class:`sklearn.decomposition.NMF`. ``"nndsvda"``
        (default) is deterministic and handles exact zeros (from the
        non-negativity clipping above) better than plain ``"nndsvd"``.
    max_iter, random_state, nmf_kwargs
        Forwarded to :class:`sklearn.decomposition.NMF`.

    Returns
    -------
    reconstructed
        Same shape as ``values``.
    meta
        Small dict safe for ``DataArray.attrs`` — method, parameters used,
        convergence info, reconstruction error, variance-explained proxy.
    payload
        Dict with keys ``components`` (n_components, n_spectral),
        ``abundances`` (spatial_shape + (n_components,)), and
        ``per_spectrum_min`` (spatial_shape + (1,)).
    """
    row_stack, spatial_shape = _flatten_to_row_stack(values)
    n_spectra, n_spectral = row_stack.shape
    if n_spectra < 2:
        raise ValueError(
            "NMF decomposition needs more than one spectrum; got "
            f"n_spectra={n_spectra}. For a single spectrum, fit Gaussian "
            "bands directly with peak_fitter.PeakFitter instead."
        )
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if n_components > n_spectra:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed n_spectra "
            f"({n_spectra})"
        )

    spectra_for_fit, per_spec_min, nan_row_mask, valid_idx = (
        _nonnegative_fit_matrix(row_stack)
    )
    has_nan = bool(nan_row_mask.any())

    nmf, abundances, converged = _fit_nmf(
        spectra_for_fit,
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        nmf_kwargs=nmf_kwargs,
    )
    reconstructed_rows = nmf.inverse_transform(abundances)

    if has_nan:
        reconstructed_rows = reconstructed_rows.copy()
        reconstructed_rows[nan_row_mask] = np.nan
        abundances = abundances.copy()
        abundances[nan_row_mask] = np.nan

    reconstructed = reconstructed_rows.reshape(spatial_shape + (n_spectral,))

    valid_fit = spectra_for_fit[valid_idx]
    total_ss = float(np.sum(valid_fit**2)) if valid_idx.any() else 0.0
    fraction_var_explained = (
        1.0 - (nmf.reconstruction_err_**2 / total_ss)
        if total_ss > 0
        else float("nan")
    )

    meta: dict[str, Any] = {
        "method": "nmf",
        "n_components": int(n_components),
        "init": init,
        "max_iter": int(max_iter),
        "random_state": random_state,
        "n_iter": int(nmf.n_iter_),
        "converged": bool(converged),
        "reconstruction_err": float(nmf.reconstruction_err_),
        "fraction_var_explained": float(fraction_var_explained),
        "n_spectra": int(n_spectra),
        "n_spectral": int(n_spectral),
    }

    abundances_spatial = abundances.reshape(spatial_shape + (n_components,))
    payload: dict[str, Any] = {
        "components": np.asarray(nmf.components_, dtype=float).copy(),
        "abundances": abundances_spatial.copy(),
        "per_spectrum_min": per_spec_min.reshape(spatial_shape + (1,)).copy(),
    }
    return reconstructed, meta, payload


def compute_nmf_diagnostic_curve(
    values: np.ndarray,
    *,
    k_max: int | None = None,
    max_pixels_for_diagnostic: int = 2000,
    init: str = "nndsvda",
    max_iter: int = 300,
    random_state: int | None = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Sweep ``n_components = 1..k_max`` and record fit diagnostics at each k.

    This is the human-in-the-loop replacement for PCA's automatic ``"mle"``
    component selection: NMF has no analogous estimator, so the component
    count must be chosen by a person looking at how reconstruction error and
    variance-explained trade off against k — not picked automatically.

    Only rows with real data (no NaN) are used, and on large maps a random
    subsample (seeded by ``random_state``) is swept instead of every pixel —
    this affects only the diagnostic, never the final chosen-k fit, which
    always uses every valid pixel.

    Returns a dict with keys ``k_values``, ``reconstruction_error``,
    ``fraction_var_explained``, ``converged``, ``n_iter``, ``subsampled``,
    ``n_pixels_used``, ``n_pixels_total``.
    """
    row_stack, _ = _flatten_to_row_stack(values)
    n_pixels_total = row_stack.shape[0]

    rng = np.random.default_rng(random_state)
    subsampled = n_pixels_total > max_pixels_for_diagnostic
    if subsampled:
        idx = rng.choice(n_pixels_total, size=max_pixels_for_diagnostic, replace=False)
        row_stack = row_stack[idx]

    spectra_for_fit, _, _, valid_idx = _nonnegative_fit_matrix(row_stack)
    valid_fit = spectra_for_fit[valid_idx]
    n_valid = valid_fit.shape[0]
    if n_valid < 2:
        raise ValueError(
            "compute_nmf_diagnostic_curve needs at least 2 valid (non-NaN) "
            f"spectra; got n_valid={n_valid}"
        )

    n_spectral = valid_fit.shape[1]
    if k_max is None:
        k_max = min(10, n_valid - 1, n_spectral)
    k_max = max(1, min(k_max, n_valid - 1, n_spectral))
    k_values = list(range(1, k_max + 1))

    total_ss = float(np.sum(valid_fit**2))

    reconstruction_error: list[float] = []
    fraction_var_explained: list[float] = []
    converged: list[bool] = []
    n_iter: list[int] = []

    for i, k in enumerate(k_values):
        nmf, _, ok = _fit_nmf(
            valid_fit,
            n_components=k,
            init=init,
            max_iter=max_iter,
            random_state=random_state,
            nmf_kwargs=None,
        )
        err = float(nmf.reconstruction_err_)
        reconstruction_error.append(err)
        fraction_var_explained.append(
            1.0 - (err**2 / total_ss) if total_ss > 0 else float("nan")
        )
        converged.append(ok)
        n_iter.append(int(nmf.n_iter_))
        if progress_callback is not None:
            progress_callback(i + 1, len(k_values))

    return {
        "k_values": k_values,
        "reconstruction_error": reconstruction_error,
        "fraction_var_explained": fraction_var_explained,
        "converged": converged,
        "n_iter": n_iter,
        "subsampled": subsampled,
        "n_pixels_used": n_valid,
        "n_pixels_total": n_pixels_total,
    }


__all__ = ["decompose_spectra_nmf", "compute_nmf_diagnostic_curve"]
