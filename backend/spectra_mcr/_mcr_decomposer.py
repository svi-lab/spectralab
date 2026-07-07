# -*- coding: utf-8 -*-
"""High-level MCR-ALS curve resolution: :class:`MCRDecomposer`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from _shared._spectral import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
from _shared.utils import ensure_in_memory

from ._mcr import compute_mcr_ambiguity, mcr_als

_TREATMENT_KEY = "spectra_mcr"


@dataclass
class MCRDecomposer:
    """Resolve pure-component spectra + concentration maps by MCR-ALS.

    Mirrors :class:`spectra_decomposer.Decomposer` (same ``(reconstructed_da,
    payload)`` contract, same NaN/flatten handling), but resolves a physically
    interpretable bilinear model ``D = C @ S`` under non-negativity (and an
    optional equality constraint) rather than NMF's multiplicative updates.

    Parameters
    ----------
    n_components
        Number of pure components. Choose from
        :func:`spectra_mcr.compute_mcr_rank_svd` (SVD scree).
    max_iter, tol, simplisma_offset
        Forwarded to :func:`mcr_als`.
    equality_spectrum, equality_index
        Optional reference spectrum (already resampled to the data's spectral
        axis) pinning one component's spectrum, and which component it applies
        to.
    quantify_ambiguity
        When True (default), run the feasible-band ``f_max - f_min`` rotational
        ambiguity analysis and attach it to the payload.
    spectral_dim
        Name of the spectral axis. Defaults to the last dimension.
    """

    n_components: int
    max_iter: int = 200
    tol: float = 0.1
    simplisma_offset: float = 0.05
    equality_spectrum: np.ndarray | None = None
    equality_index: int = 0
    quantify_ambiguity: bool = True
    random_state: int | None = 0
    spectral_dim: str | None = None

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be >= 1, got {self.n_components}"
            )

    def decompose(
        self, spectra: xr.DataArray
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Run MCR-ALS, return ``(reconstructed_da, payload)``.

        ``payload`` keys: ``components`` (``S``, k x n_spectral ndarray),
        ``abundances`` (``C`` as an ``xr.DataArray`` over the spatial dims plus
        a ``"component"`` dim), ``meta``, and — when ``quantify_ambiguity`` —
        ``ambiguity``.
        """
        if not isinstance(spectra, xr.DataArray):
            raise TypeError(
                "MCRDecomposer.decompose expects an xarray.DataArray; got "
                f"{type(spectra).__name__}"
            )

        sdim = resolve_spectral_dim(spectra, self.spectral_dim)
        da_w, orig_order = transpose_spectral_last(spectra, sdim)

        da_w = ensure_in_memory(
            da_w,
            caller="MCRDecomposer (MCR-ALS)",
            reason=(
                "MCR-ALS alternates NNLS solves over the full matrix and "
                "cannot be fit chunk-by-chunk."
            ),
            stacklevel=3,
        )

        spatial_shape = da_w.shape[:-1]
        n_spectra = int(np.prod(spatial_shape)) if spatial_shape else 1
        if n_spectra < 2:
            raise ValueError(
                "MCRDecomposer needs more than one spectrum (MCR-ALS on a "
                "single spectrum is degenerate). Got input with shape "
                f"{tuple(spectra.shape)} -> n_spectra={n_spectra}. For a single "
                "spectrum, fit Gaussian bands directly with "
                "peak_fitter.PeakFitter instead."
            )

        reconstructed_w, meta, payload = mcr_als(
            da_w.values,
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            simplisma_offset=self.simplisma_offset,
            equality_spectrum=self.equality_spectrum,
            equality_index=self.equality_index,
            random_state=self.random_state,
        )
        meta = {**meta, "spectral_dim": sdim}

        reconstructed_w_array = reshape_row_stack_to(
            reconstructed_w.reshape(-1, reconstructed_w.shape[-1]),
            da_w.shape,
        )
        if tuple(da_w.dims) != orig_order:
            reconstructed_da = da_w.copy(data=reconstructed_w_array).transpose(
                *orig_order
            )
            reconstructed = reconstructed_da.values
        else:
            reconstructed = reconstructed_w_array

        out = with_new_values(spectra, reconstructed, _TREATMENT_KEY, meta)

        spatial_dims = [d for d in da_w.dims if d != sdim]
        abundances_da = xr.DataArray(
            payload["abundances"],
            dims=(*spatial_dims, "component"),
            coords={
                **{d: da_w.coords[d] for d in spatial_dims if d in da_w.coords},
                "component": np.arange(self.n_components),
            },
            name="mcr_concentration",
        )

        full_payload: dict[str, Any] = {
            "components": payload["components"],
            "abundances": abundances_da,
            "meta": meta,
        }

        if self.quantify_ambiguity and self.n_components > 1:
            eq_idx = (
                self.equality_index if self.equality_spectrum is not None else None
            )
            full_payload["ambiguity"] = compute_mcr_ambiguity(
                da_w.values,
                payload["abundances"],
                payload["components"],
                equality_index=eq_idx,
                random_state=self.random_state,
            )

        return out, full_payload


__all__ = ["MCRDecomposer"]
