"""MCR-ALS (Multivariate Curve Resolution — Alternating Least Squares).

Resolves a population of spectra ``D`` (pixels x channels) into a small number
of **pure-component spectra** ``S`` (components x channels) and their
**concentration profiles** ``C`` (pixels x components) under the bilinear model

    D = C @ S + E

subject to physically-true constraints. Unlike NMF's multiplicative updates,
MCR-ALS alternates two *constrained least-squares* solves (non-negative least
squares, NNLS) — so non-negativity of both intensity (``S``) and concentration
(``C``) is enforced by the solver, not by clipping negatives after the fact.

Physical-meaningfulness choices baked in here:

* ``S`` rows are pure emission spectra, ``C`` columns are relative
  concentration maps; both are non-negative by NNLS construction.
* The **intensity ambiguity** (``c_i s_i = (c_i k)(s_i / k)``) is removed by
  normalizing each resolved ``S`` row to unit L2 norm and folding the scale
  into ``C``, so concentration maps are quantitatively comparable.
* Initial estimates come from **SIMPLISMA** pure-pixel selection (never
  random), so the starting point already satisfies non-negativity.
* The **rotational ambiguity** that remains — the dominant, physically real
  non-uniqueness of MCR — is quantified separately by
  :func:`compute_mcr_ambiguity` (feasible-band ``f_max - f_min``).

See :class:`spectra_mcr.MCRDecomposer` for the user-facing xarray API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize, nnls

from _shared._factorization import (
    _flatten_to_row_stack,
    _nonnegative_fit_matrix,
)


# --------------------------------------------------------------------------- #
# Rank determination (SVD scree) — the physical replacement for "k_max sweep"
# --------------------------------------------------------------------------- #
def compute_mcr_rank_svd(
    values: np.ndarray,
    *,
    k_max: int = 10,
    max_pixels_for_diagnostic: int = 2000,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Singular-value scree of the data matrix, for choosing the rank.

    The number of significant singular values (those clearly above the noise
    floor) is the number of spectrally distinguishable components — a far more
    physical way to choose the component count than an opaque ``k_max`` sweep.

    Only rows with real data (no NaN) are used, and on large maps a seeded
    random subsample is analysed for speed (this never touches the final fit).

    Returns a dict with ``singular_values``, ``variance_ratio``
    (``s_i^2 / sum s^2``), ``cumulative_variance``, ``n_pixels_used``,
    ``n_pixels_total``, ``subsampled``.
    """
    row_stack, _ = _flatten_to_row_stack(values)
    n_pixels_total = row_stack.shape[0]

    rng = np.random.default_rng(random_state)
    subsampled = n_pixels_total > max_pixels_for_diagnostic
    if subsampled:
        idx = rng.choice(n_pixels_total, size=max_pixels_for_diagnostic, replace=False)
        row_stack = row_stack[idx]

    D_fit, _, _, valid_idx = _nonnegative_fit_matrix(row_stack)
    D = D_fit[valid_idx]
    n_valid = D.shape[0]
    if n_valid < 2:
        raise ValueError(
            f"compute_mcr_rank_svd needs at least 2 valid (non-NaN) spectra; got n_valid={n_valid}"
        )

    svals = np.linalg.svd(D, compute_uv=False)
    k_max = int(max(1, min(k_max, svals.size)))
    svals = svals[:k_max]

    total = float(np.sum(svals**2))
    variance_ratio = (svals**2 / total) if total > 0 else np.full(k_max, np.nan)
    cumulative = np.cumsum(variance_ratio)

    return {
        "singular_values": svals.astype(float),
        "variance_ratio": variance_ratio.astype(float),
        "cumulative_variance": cumulative.astype(float),
        "n_pixels_used": int(n_valid),
        "n_pixels_total": int(n_pixels_total),
        "subsampled": bool(subsampled),
    }


# --------------------------------------------------------------------------- #
# SIMPLISMA — pure-pixel initial estimates (never random)
# --------------------------------------------------------------------------- #
def _simplisma(
    D: np.ndarray,
    n_components: int,
    *,
    offset: float = 0.05,
) -> np.ndarray:
    """Select the ``n_components`` purest (most dissimilar) rows of ``D``.

    Windig's SIMPLISMA, applied in the **pixel/row** direction (the direction
    of least component overlap for spatial-spectral data): the first pure row
    maximises purity ``sigma / (mean + alpha)``; each subsequent pure row
    maximises purity weighted by its dissimilarity to those already chosen
    (the Schur complement of the correlation matrix, which is large when the
    candidate is orthogonal to the selected set). Returns row indices; the
    caller uses ``D[idx]`` as the initial spectra ``S0``.
    """
    m, n = D.shape
    mean = D.mean(axis=1)
    std = D.std(axis=1)
    alpha = float(offset) * float(mean.max()) if m else 0.0
    purity = std / (mean + alpha + 1e-12)

    # Normalise each row to a comparable length before measuring dissimilarity.
    length = np.sqrt(mean**2 + std**2 + alpha**2) + 1e-12
    Dn = D / length[:, None]
    gii = np.einsum("ij,ij->i", Dn, Dn) / n  # per-row self correlation

    pure_idx: list[int] = []
    for c in range(n_components):
        if c == 0:
            weight = np.ones(m)
        else:
            Sel = Dn[pure_idx]  # (c, n)
            G_ss = Sel @ Sel.T / n  # (c, c)
            G_ss = G_ss + 1e-9 * np.eye(c)  # ridge for stability
            g_si = Sel @ Dn.T / n  # (c, m)
            inv_Gss = np.linalg.inv(G_ss)
            quad = np.einsum("ji,jk,ki->i", g_si, inv_Gss, g_si)
            # Schur complement = novelty of each candidate vs the selected set.
            weight = np.maximum(gii - quad, 0.0)
        weighted = purity * weight
        weighted[pure_idx] = -np.inf
        pure_idx.append(int(np.argmax(weighted)))

    return np.asarray(pure_idx, dtype=int)


# --------------------------------------------------------------------------- #
# Constrained least-squares half-steps (NNLS on both C and S)
# --------------------------------------------------------------------------- #
def _solve_C(D: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Given ``S`` (k, n), solve ``C`` (m, k) row-wise by NNLS."""
    A = S.T  # (n, k)
    m = D.shape[0]
    k = S.shape[0]
    C = np.empty((m, k), dtype=float)
    for i in range(m):
        C[i], _ = nnls(A, D[i])
    return C


def _solve_S(D: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Given ``C`` (m, k), solve ``S`` (k, n) column-wise by NNLS."""
    n = D.shape[1]
    k = C.shape[1]
    S = np.empty((k, n), dtype=float)
    for j in range(n):
        S[:, j], _ = nnls(C, D[:, j])
    return S


def mcr_als(
    values: np.ndarray,
    *,
    n_components: int,
    max_iter: int = 200,
    tol: float = 0.1,
    simplisma_offset: float = 0.05,
    equality_spectrum: np.ndarray | None = None,
    equality_index: int = 0,
    random_state: int | None = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Resolve ``D = C @ S`` by non-negativity-constrained ALS.

    Parameters
    ----------
    values
        Array ``(..., n_spectral)``; more than one spectrum required.
    n_components
        Number of pure components. Choose from :func:`compute_mcr_rank_svd`.
    max_iter, tol
        Stop when ``|ΔLOF| < tol`` (percent) between iterations or ``max_iter``
        is reached, whichever comes first.
    simplisma_offset
        SIMPLISMA noise offset (fraction of the max row-mean).
    equality_spectrum
        Optional reference spectrum (already resampled to the data axis) to
        pin one component's ``S`` row to, held fixed every iteration. Must be
        length ``n_spectral``.
    equality_index
        Which component the equality reference applies to.

    Returns ``(reconstructed, meta, payload)`` mirroring
    :func:`spectra_decomposer.decompose_spectra_nmf`. ``payload`` has
    ``components`` (``S``, k x n_spectral), ``abundances`` (``C``, spatial +
    (k,)), ``per_spectrum_min``.
    """
    row_stack, spatial_shape = _flatten_to_row_stack(values)
    n_spectra, n_spectral = row_stack.shape
    if n_spectra < 2:
        raise ValueError(
            "MCR-ALS needs more than one spectrum; got "
            f"n_spectra={n_spectra}. For a single spectrum, fit Gaussian bands "
            "directly with peak_fitter.PeakFitter instead."
        )
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if n_components > n_spectra:
        raise ValueError(f"n_components ({n_components}) cannot exceed n_spectra ({n_spectra})")

    # Only D_valid is ever fitted — see the ALS loop below for why the
    # median-filled invalid rows must stay out of both half-steps.
    D_full, per_spec_min, nan_row_mask, valid_idx = _nonnegative_fit_matrix(row_stack)
    has_nan = bool(nan_row_mask.any())
    D_valid = D_full[valid_idx]
    if D_valid.shape[0] < n_components:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed the number of valid "
            f"(non-NaN) spectra ({D_valid.shape[0]})"
        )

    # ---- initial estimates: SIMPLISMA pure pixels (on valid rows only) ----
    pure_idx = _simplisma(D_valid, n_components, offset=simplisma_offset)
    S = D_valid[pure_idx].copy()  # (k, n_spectral), already >= 0

    use_equality = equality_spectrum is not None
    if use_equality:
        eq = np.clip(np.asarray(equality_spectrum, dtype=float), 0, None)
        if eq.shape[-1] != n_spectral:
            raise ValueError(f"equality_spectrum length {eq.shape[-1]} != n_spectral {n_spectral}")
        if not (0 <= equality_index < n_components):
            raise ValueError(
                f"equality_index {equality_index} out of range for n_components {n_components}"
            )
        S[equality_index] = eq

    d_sq = float(np.sum(D_valid**2))
    lof_prev = np.inf
    lof = np.nan
    converged = False
    # Both ALS half-steps run on D_valid only. _nonnegative_fit_matrix keeps
    # the row count intact by median-filling invalid rows so indices stay
    # aligned, but solving S against those filler rows would let a spectrum
    # that Clean Data dropped — or that the user excluded by hand — shape the
    # resolved pure spectra. C is scattered back to full height at the end.
    C_valid = np.zeros((D_valid.shape[0], n_components))
    n_iter = 0
    for it in range(1, int(max_iter) + 1):
        n_iter = it
        C_valid = _solve_C(D_valid, S)
        S = _solve_S(D_valid, C_valid)
        if use_equality:
            S[equality_index] = eq

        resid = D_valid - (C_valid @ S)
        lof = 100.0 * np.sqrt(float(np.sum(resid**2)) / d_sq) if d_sq > 0 else 0.0
        if progress_callback is not None:
            progress_callback(it, int(max_iter))
        if abs(lof_prev - lof) < tol:
            converged = True
            break
        lof_prev = lof

    # ---- intensity-ambiguity gauge fix: unit-norm each S row, scale into C ----
    norms = np.linalg.norm(S, axis=1)
    norms_safe = np.where(norms > 0, norms, 1.0)
    S = S / norms_safe[:, None]
    C_valid = C_valid * norms_safe[None, :]

    if has_nan:
        C = np.full((n_spectra, n_components), np.nan)
        C[valid_idx] = C_valid
        reconstructed_rows = np.full((n_spectra, n_spectral), np.nan)
        reconstructed_rows[valid_idx] = C_valid @ S
    else:
        C = C_valid
        reconstructed_rows = C_valid @ S

    reconstructed = reconstructed_rows.reshape(spatial_shape + (n_spectral,))

    fraction_var_explained = (
        1.0 - (float(np.sum((D_valid - (C_valid @ S)) ** 2)) / d_sq) if d_sq > 0 else float("nan")
    )

    constraints = ["nonneg-C", "nonneg-S"]
    if use_equality:
        constraints.append(f"equality[k={equality_index}]")

    meta: dict[str, Any] = {
        "method": "mcr-als",
        "n_components": int(n_components),
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "lof": float(lof),
        "fraction_var_explained": float(fraction_var_explained),
        "constraints": constraints,
        "tol": float(tol),
        "max_iter": int(max_iter),
        "simplisma_offset": float(simplisma_offset),
        "n_spectra": int(n_spectra),
        # Rows actually fitted — smaller than n_spectra when Clean Data or a
        # manual exclusion NaN'd spectra out.
        "n_spectra_valid": int(D_valid.shape[0]),
        "n_spectral": int(n_spectral),
    }

    abundances_spatial = C.reshape(spatial_shape + (n_components,))
    payload: dict[str, Any] = {
        "components": np.asarray(S, dtype=float).copy(),
        "abundances": abundances_spatial.copy(),
        "per_spectrum_min": per_spec_min.reshape(spatial_shape + (1,)).copy(),
    }
    return reconstructed, meta, payload


# --------------------------------------------------------------------------- #
# Rotational ambiguity — feasible-band f_max - f_min (MCR-BANDS style)
# --------------------------------------------------------------------------- #
def compute_mcr_ambiguity(
    values: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    *,
    equality_index: int | None = None,
    max_pixels: int = 2000,
    random_state: int | None = 0,
    slsqp_maxiter: int = 100,
) -> dict[str, Any]:
    """Quantify rotational ambiguity per component via the feasible band.

    MCR solutions are non-unique: ``D = (C T)(T^-1 S)`` for any invertible
    ``T`` that preserves the applied constraints. Working in the rank-``k`` SVD
    subspace (``D ≈ U Σ Vt``, so every feasible pair is ``C = UΣT``,
    ``S = T^-1 Vt``), we minimise and maximise each component's relative signal
    contribution

        f_i(T) = ||C[:,i]|| * ||S[i,:]|| / ||Σ||_F

    subject to non-negativity of ``C = UΣT`` and ``S = T^-1 Vt`` (SLSQP).
    ``f_max - f_min`` per component is the ambiguity score: ~0 means uniquely
    resolved, larger means more feasible freedom.

    Returns a dict with ``f_min``, ``f_max``, ``f_range`` (arrays over
    components), ``dominant_source`` (per component: which of the spectrum /
    concentration the ambiguity lives in), ``boundary`` (the two boundary
    spectra of the highest-ambiguity component, for plotting), and ``ok``
    (False if the optimiser could not resolve the bands).

    Note: the equality constraint is *not* added to the feasible region here,
    so a pinned component's reported band is a conservative upper bound.
    """
    row_stack, _ = _flatten_to_row_stack(values)
    D_fit, _, _, valid_idx = _nonnegative_fit_matrix(row_stack)
    D = D_fit[valid_idx]

    C_valid = np.asarray(C, dtype=float).reshape(-1, C.shape[-1])[valid_idx]
    S = np.asarray(S, dtype=float)
    k = S.shape[0]

    rng = np.random.default_rng(random_state)
    if D.shape[0] > max_pixels:
        idx = rng.choice(D.shape[0], size=max_pixels, replace=False)
        D = D[idx]
        C_valid = C_valid[idx]

    # Rank-k SVD model: A = U Σ (m x k), Vt (k x n). Feasible C = A T, S = Ti Vt.
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    A = U[:, :k] * s[:k]  # (m, k)
    Vt = Vt[:k, :]  # (k, n)
    sigma_norm = float(np.sqrt(np.sum(s[:k] ** 2)))
    if sigma_norm == 0:
        return {"ok": False, "reason": "degenerate (zero singular values)"}

    # Seed T0 so that A @ T0 best-fits the resolved C (feasible starting point).
    T0 = np.linalg.lstsq(A, C_valid, rcond=None)[0]  # (k, k)

    def _f_i(T_flat: np.ndarray, i: int) -> float:
        T = T_flat.reshape(k, k)
        try:
            Ti = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            return np.nan
        c_norm = np.linalg.norm(A @ T[:, i])
        s_norm = np.linalg.norm(Ti[i])  # Vt rows orthonormal => ||Ti[i]@Vt||=||Ti[i]||
        return float(c_norm * s_norm / sigma_norm)

    def _nonneg_constraints(T_flat: np.ndarray) -> np.ndarray:
        T = T_flat.reshape(k, k)
        try:
            Ti = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            return np.full(A.shape[0] * k + k * Vt.shape[1], -1e6)
        C_new = A @ T  # (m, k) >= 0
        S_new = Ti @ Vt  # (k, n) >= 0
        return np.concatenate([C_new.ravel(), S_new.ravel()])

    cons = ({"type": "ineq", "fun": _nonneg_constraints},)
    x0 = T0.ravel()

    f_min = np.full(k, np.nan)
    f_max = np.full(k, np.nan)
    boundary_min_S = [None] * k
    boundary_max_S = [None] * k
    boundary_min_C = [None] * k
    boundary_max_C = [None] * k
    ok_any = False

    def _boundary(T_flat: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        T = T_flat.reshape(k, k)
        Ti = np.linalg.inv(T)
        return (Ti @ Vt)[i], (A @ T)[:, i]

    for i in range(k):
        try:
            r_min = minimize(
                lambda t, i=i: _f_i(t, i),
                x0,
                method="SLSQP",
                constraints=cons,
                options={"maxiter": slsqp_maxiter, "ftol": 1e-6},
            )
            r_max = minimize(
                lambda t, i=i: -_f_i(t, i),
                x0,
                method="SLSQP",
                constraints=cons,
                options={"maxiter": slsqp_maxiter, "ftol": 1e-6},
            )
            if r_min.success and np.isfinite(r_min.fun):
                f_min[i] = float(r_min.fun)
                boundary_min_S[i], boundary_min_C[i] = _boundary(r_min.x, i)
                ok_any = True
            if r_max.success and np.isfinite(r_max.fun):
                f_max[i] = float(-r_max.fun)
                boundary_max_S[i], boundary_max_C[i] = _boundary(r_max.x, i)
                ok_any = True
        except (np.linalg.LinAlgError, ValueError):
            continue

    f_range = f_max - f_min

    # Where does each component's ambiguity live: spectrum, concentration, both?
    dominant_source: list[str] = []
    for i in range(k):
        if boundary_min_S[i] is None or boundary_max_S[i] is None:
            dominant_source.append("unknown")
            continue
        s_corr = _corr(boundary_min_S[i], boundary_max_S[i])
        c_corr = _corr(boundary_min_C[i], boundary_max_C[i])
        s_var = 1.0 - s_corr
        c_var = 1.0 - c_corr
        if max(s_var, c_var) < 0.02:
            dominant_source.append("negligible")
        elif s_var > 2 * c_var:
            dominant_source.append("spectrum")
        elif c_var > 2 * s_var:
            dominant_source.append("concentration")
        else:
            dominant_source.append("both")

    # Boundary spectra of the highest-ambiguity component, for plotting.
    finite_range = np.where(np.isfinite(f_range), f_range, -np.inf)
    worst = int(np.argmax(finite_range)) if np.isfinite(finite_range).any() else 0
    boundary = None
    if boundary_min_S[worst] is not None and boundary_max_S[worst] is not None:
        boundary = {
            "component": worst,
            "s_min": np.asarray(boundary_min_S[worst], dtype=float),
            "s_max": np.asarray(boundary_max_S[worst], dtype=float),
        }

    return {
        "ok": bool(ok_any),
        "f_min": f_min,
        "f_max": f_max,
        "f_range": f_range,
        "dominant_source": dominant_source,
        "boundary": boundary,
        "n_pixels_used": int(D.shape[0]),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, sign-agnostic, in [0, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(abs(np.dot(a, b) / (na * nb)))


__all__ = ["compute_mcr_rank_svd", "mcr_als", "compute_mcr_ambiguity"]
