# -*- coding: utf-8 -*-
"""High-level NMF pattern decomposition: :class:`Decomposer`."""

from __future__ import annotations

from dataclasses import dataclass, field
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

from ._nmf import decompose_spectra_nmf

_TREATMENT_KEY = "spectra_decomposition"


@dataclass
class Decomposer:
    """Find recurring spectral patterns across a population of spectra by NMF.

    Unlike :class:`spectra_cleaner.Denoiser`, this is decomposition-only:
    there is no "just reconstruct" use case here, the whole point is
    exposing the components and per-pixel abundances, so there is a single
    public method rather than a denoise/denoise_with_decomposition split.

    Parameters
    ----------
    n_components
        Fixed integer component count. Choose it by inspecting
        :func:`spectra_decomposer.compute_nmf_diagnostic_curve` first —
        there is no automatic selection.
    init, max_iter, random_state, nmf_kwargs
        Forwarded to :class:`sklearn.decomposition.NMF`.
    spectral_dim
        Name of the spectral axis. Defaults to the last dimension.
    """

    n_components: int
    init: str = "nndsvda"
    max_iter: int = 500
    random_state: int | None = 0
    spectral_dim: str | None = None
    nmf_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be >= 1, got {self.n_components}"
            )

    def decompose(
        self, spectra: xr.DataArray
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Run NMF decomposition, return ``(reconstructed_da, payload)``.

        ``reconstructed_da`` has the same shape/coords as ``spectra``, with
        ``attrs["treatments"]["spectra_decomposition"]`` set to the fit
        metadata.

        ``payload`` has keys:
            ``components`` — ``(n_components, n_spectral)`` ndarray.
            ``abundances`` — ``xr.DataArray`` over the spatial dims plus a
            new ``"component"`` dim.
            ``meta`` — the same metadata dict attached to ``attrs``.
        """
        if not isinstance(spectra, xr.DataArray):
            raise TypeError(
                "Decomposer.decompose expects an xarray.DataArray; got "
                f"{type(spectra).__name__}"
            )

        sdim = resolve_spectral_dim(spectra, self.spectral_dim)
        da_w, orig_order = transpose_spectral_last(spectra, sdim)

        da_w = ensure_in_memory(
            da_w,
            caller="Decomposer (NMF)",
            reason=(
                "NMF requires the full matrix for its multiplicative-update "
                "iterations and cannot be fit chunk-by-chunk."
            ),
            stacklevel=3,
        )

        spatial_shape = da_w.shape[:-1]
        n_spectra = int(np.prod(spatial_shape)) if spatial_shape else 1
        if n_spectra < 2:
            raise ValueError(
                "Decomposer needs more than one spectrum (NMF on a single "
                "spectrum is degenerate). Got input with shape "
                f"{tuple(spectra.shape)} -> n_spectra={n_spectra}. For a "
                "single spectrum, fit Gaussian bands directly with "
                "peak_fitter.PeakFitter instead."
            )

        reconstructed_w, meta, payload = decompose_spectra_nmf(
            da_w.values,
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            nmf_kwargs=self.nmf_kwargs or None,
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
            name="nmf_abundance",
        )

        full_payload: dict[str, Any] = {
            "components": payload["components"],
            "abundances": abundances_da,
            "meta": meta,
        }
        return out, full_payload


__all__ = ["Decomposer"]
