"""High-level cosmic-ray removal: :class:`CosmicRayRemover`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from _shared._spectral import resolve_spectral_dim, with_new_values
from _shared.utils import ensure_in_memory

from ._units import median_channel_width, spectral_to_channels
from .harmonic import grating_artifact_correct_dataarray, harmonic_correct_dataarray
from .mask_1d import detect_cosmic_mask_1d, remove_cosmic_rays_1d, repair_cosmic_mask_1d
from .mask_map import (
    correct_cosmic_rays_collection,
    correct_cosmic_rays_on_map_cube,
)

# ---------------------------------------------------------------------------
# Internal tuning constants (not exposed as user parameters)
# ---------------------------------------------------------------------------

_MAP_MAD_MULTIPLIER: float = 7.0
_MAP_NOISY_RELAX_MIN: float = 0.82
_MAP_MIN_RESIDUAL_OVER_CUTOFF: float = 1.05
_MAP_REQUIRE_SPATIAL_LOCAL_MAX: bool = False

# Below this many spectra → apply 1D engine independently per spectrum;
# at or above → use the collection (global-median / PCA) engine.
_COLLECTION_THRESHOLD: int = 20


@dataclass
class CosmicRayRemover:
    """Cosmic-ray removal with automatic routing by data dimensionality.

    **1D (single spectrum)** — always uses the 1D medfilt + MAD engine
    controlled by :attr:`spike_width`, :attr:`spike_threshold`,
    :attr:`spike_passes`.

    **2D (line scan / series / point collection)**

    * fewer than 20 spectra → 1D engine applied independently to each
      spectrum (no population statistics yet).
    * 20 or more spectra → *collection engine*: global median or PCA
      reconstruction as reference; :attr:`map_method` selects which.

    **3D (spatial map)**

    * fewer than 20 spectra → same per-spectrum 1D path as above.
    * 20 or more → spatial disk-median engine (``map_method="median"``,
      default) or PCA engine (``map_method="pca"``).  The disk-median path
      additionally respects :attr:`map_sensitivity` and
      :attr:`map_disk_radius`.

    Optionally removes broad Nd:YAG harmonics and the grating's 2nd-order
    ghost of the excitation line before spike removal via
    :meth:`harmonic_check` / :meth:`grating_artifact_check` / :meth:`remove`.

    Parameters
    ----------
    spike_width
        **1D engine** — odd integer ≥ 3.  Sets the ``medfilt`` window in
        spectral channels.  Raise to 9–13 when cosmic rays span 7–10
        channels; keep at 5 for narrow single-channel spikes.
    spike_threshold
        **1D engine** — positive float.  Spike cutoff = ``spike_threshold ×
        local_noise``, where the noise is estimated in blocks along the
        spectrum (not one value for the whole trace) since shot-noise
        baselines have noise proportional to sqrt(intensity).  Lower → more
        aggressive.  Default 8.0; lower to 3–5 only for very clean, low-noise
        data, and check that real narrow peaks survive at whatever value you
        pick.
    spike_passes
        **1D engine** — integer ≥ 1.  Iterations of detect → repair.  Each
        pass works on the already-repaired signal so that large spikes no
        longer mask smaller ones.
    broad_spike_width
        **1D engine** — approximate maximum half-width (channels) of broad
        CRs to detect.  A short medfilt-kernel ladder (``4×W+1`` then
        ``8×W+1``) is run after the narrow passes, ensuring the CR is a
        minority of the window and thus detectable even when it's close to
        the first rung's own width.  Set to 0 to disable.  Example: 15
        catches CRs up to 30 channels wide; 20 for up to 40-channel spikes.
        Ignored when ``broad_width_units`` is set.
    force_1d
        If ``True``, always apply the 1D engine independently to every
        spectrum, regardless of data shape or spectrum count.  Overrides
        automatic routing to the collection / spatial disk-median engines
        for 2D and 3D inputs.  The harmonic check is unaffected — it
        always runs per-spectrum.
    spike_width_units, broad_width_units
        **1D engine** — same meaning as ``spike_width`` / ``broad_spike_width``
        but expressed in spectral units (nm, cm⁻¹, …) instead of channels,
        converted per-DataArray using its own axis spacing.  Set either to
        make the 1D engine catch a CR of a given *physical* width regardless
        of the instrument's dispersion; ``None`` (default) leaves the
        channel-based field in charge.  ``broad_width_units=0`` disables the
        broad pass, matching ``broad_spike_width=0``.
    consensus_veto_fraction
        **Loop-1D only** (line scans / maps below the collection threshold,
        or any shape with ``force_1d=True``) — unflag any channel detected
        as a spike in more than this fraction of spectra, before repair. A
        cosmic ray essentially never lands on the same channel across many
        pixels; a real narrow emission line always does. ``0.0`` disables
        the veto. Default 0.3.
    map_sensitivity
        **3D disk-median engine only** — scales overall aggressiveness.
        Larger → more hits (default 0.01).
    map_disk_radius
        **3D disk-median engine only** — spatial disk radius for the
        reference median filter (pixels).
    map_spike_width
        **Collection / 3D engines** — spectral dilation in channels added
        around each detected hit (integer ≥ 1).  Increase for broader
        cosmic rays (e.g. ``9``–``15`` for multi-channel spikes).  The
        repair region is capped at ``2 × map_spike_width`` channels.
    map_method
        ``"median"`` (default): global median spectrum as reference for 2D;
        spatial disk-median for 3D.
        ``"pca"``: PCA reconstruction as reference for both 2D and 3D.
    map_n_components
        **PCA path only** — number of principal components for the
        reconstruction reference.  3–5 covers most real samples; increase
        for multi-phase or compositionally diverse maps.
    spectral_dim
        Name of the spectral axis (default: last dimension).
    """

    # --- 1D engine ---
    spike_width: int = 5
    spike_threshold: float = 8.0
    spike_passes: int = 3
    broad_spike_width: int = 15
    force_1d: bool = False
    # Physical-unit overrides (spectral units, e.g. nm or cm^-1) — when set,
    # take precedence over spike_width / broad_spike_width for the 1D engine
    # (both loop-1D and single-spectrum paths), resolved per DataArray via
    # its own spectral axis spacing. None (default) keeps the channel-based
    # fields in charge, unchanged.
    spike_width_units: float | None = None
    broad_width_units: float | None = None
    # Loop-1D only: a channel flagged in more than this fraction of spectra
    # is treated as a real, shared spectral feature rather than a cosmic ray
    # (a CR essentially never lands on the same channel across many pixels)
    # and is unflagged everywhere. 0.0 disables the veto. Only applied when
    # there are enough spectra to make "consensus" meaningful
    # (>= _COLLECTION_THRESHOLD).
    consensus_veto_fraction: float = 0.3

    # --- collection / 3D engine ---
    map_sensitivity: float = 0.01
    map_disk_radius: int = 3
    map_spike_width: int = 5
    map_method: str = "median"
    map_n_components: int = 3

    # --- shared ---
    spectral_dim: str | None = None

    def __post_init__(self) -> None:
        if self.spike_width < 3 or self.spike_width % 2 == 0:
            raise ValueError(f"spike_width must be odd and >= 3, got {self.spike_width}")
        if self.spike_threshold <= 0 or not np.isfinite(self.spike_threshold):
            raise ValueError("spike_threshold must be positive and finite")
        if self.spike_passes < 1:
            raise ValueError("spike_passes must be >= 1")
        if self.broad_spike_width < 0:
            raise ValueError("broad_spike_width must be >= 0")
        if self.spike_width_units is not None and self.spike_width_units <= 0:
            raise ValueError("spike_width_units must be > 0 when set")
        if self.broad_width_units is not None and self.broad_width_units < 0:
            raise ValueError("broad_width_units must be >= 0 when set")
        if not (0.0 <= self.consensus_veto_fraction < 1.0):
            raise ValueError(
                f"consensus_veto_fraction must be in [0.0, 1.0), got {self.consensus_veto_fraction}"
            )
        if self.map_sensitivity <= 0:
            raise ValueError("map_sensitivity must be > 0")
        if self.map_spike_width < 1:
            raise ValueError("map_spike_width must be >= 1")
        if self.map_disk_radius < 1:
            raise ValueError("map_disk_radius must be >= 1")
        if self.map_method not in ("median", "pca"):
            raise ValueError(f"map_method must be 'median' or 'pca', got {self.map_method!r}")
        if self.map_n_components < 1:
            raise ValueError("map_n_components must be >= 1")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def harmonic_check(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Notch broad harmonics when ``laser_wavelength_nm`` is ~355 nm
        (Nd:YAG).

        If ``spectrum.attrs['laser_wavelength_nm']`` is outside 354–356 nm,
        returns ``spectrum`` unchanged.

        Searches 1064 / 532 / 355 / 266 nm (±2.5 nm); replaces ~1 nm
        around each found peak with linear interpolation.
        """
        return harmonic_correct_dataarray(
            spectrum,
            spectral_dim=self.spectral_dim,
        )

    def grating_artifact_check(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Notch the grating's 2nd-order ghost of the excitation line, at
        2 × ``spectrum.attrs['laser_wavelength_nm']``.

        Unlike :meth:`harmonic_check`, this applies for any known
        excitation wavelength — it's a property of the grating, not
        specific to the Nd:YAG line. Uses the same search/notch/
        interpolate logic as the harmonic check.
        """
        return grating_artifact_correct_dataarray(
            spectrum,
            spectral_dim=self.spectral_dim,
        )

    def remove_cosmic_rays(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Spike removal only (no harmonic/grating notch)."""
        out, _ = self._route(spectrum, want_diagnostics=False)
        return out

    def remove_cosmic_rays_with_diagnostics(
        self,
        spectrum: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Like :meth:`remove_cosmic_rays`, but also returns a diagnostics
        dict for visualization / QC (not written to ``DataArray.attrs``).

        Diagnostics keys depend on the engine used:

        * **1D**: ``"cosmic_mask"``, ``"corrected_1d"``
        * **loop-1D** (< 20 spectra, 2D/3D): ``"cosmic_masks"``
        * **collection** (≥ 20 spectra, 2D or 3D PCA): ``"core_mask"``,
          ``"repair_mask"``, ``"residual"``, ``"reference"``,
          ``"noise_per_channel"``, ``"cutoff"``
        * **3D disk-median**: same as current map diagnostics
          (``"core_mask"``, ``"repair_mask"``, ``"residual"``,
          ``"preprocessed"``, ``"spatial_median_reference"``, etc.)
        """
        return self._route(spectrum, want_diagnostics=True)

    def remove(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Harmonic + grating-artifact cleanup first, then cosmic-ray
        removal."""
        spectrum = self.harmonic_check(spectrum)
        spectrum = self.grating_artifact_check(spectrum)
        return self.remove_cosmic_rays(spectrum)

    def remove_with_diagnostics(
        self,
        spectrum: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Harmonics + grating artifact, then
        :meth:`remove_cosmic_rays_with_diagnostics`."""
        spectrum = self.harmonic_check(spectrum)
        spectrum = self.grating_artifact_check(spectrum)
        return self.remove_cosmic_rays_with_diagnostics(spectrum)

    def transform(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Alias of :meth:`remove` (harmonics + grating artifact, then
        cosmic rays)."""
        return self.remove(spectrum)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Dispatch to the correct engine based on shape and spectrum count."""
        ndim = da.ndim

        if ndim == 1:
            resolve_spectral_dim(da, self.spectral_dim)
            return self._apply_1d(
                da, np.asarray(da.values, dtype=float), want_diagnostics=want_diagnostics
            )

        if ndim == 2:
            n_spectra = da.shape[0]
            if n_spectra <= 1:
                return self._apply_1d(
                    da,
                    np.asarray(da.values, dtype=float).reshape(-1),
                    want_diagnostics=want_diagnostics,
                )
            if self.force_1d or n_spectra < _COLLECTION_THRESHOLD:
                return self._apply_loop_1d(da, want_diagnostics=want_diagnostics)
            return self._apply_collection(da, want_diagnostics=want_diagnostics)

        if ndim == 3:
            n_spectra = da.shape[0] * da.shape[1]
            if self.force_1d or n_spectra < _COLLECTION_THRESHOLD:
                return self._apply_loop_1d(da, want_diagnostics=want_diagnostics)
            da = self._maybe_compute_for_map(da)
            return self._apply_map(da, want_diagnostics=want_diagnostics)

        raise ValueError(
            "CosmicRayRemover supports 1-D (n_spectral,), "
            "2-D (n_spatial, n_spectral), or 3-D (ny, nx, n_spectral); "
            f"got ndim={da.ndim}, shape={da.shape}"
        )

    # ------------------------------------------------------------------
    # Engines
    # ------------------------------------------------------------------

    def _resolved_widths(self, da: xr.DataArray) -> tuple[int, int]:
        """Resolve the 1D engine's spike widths to channels for ``da``.

        ``spike_width_units`` / ``broad_width_units`` (spectral units) take
        precedence over ``spike_width`` / ``broad_spike_width`` (channels)
        when set, converted using this DataArray's own axis spacing so the
        same physical CR width is caught regardless of grating dispersion.
        """
        if self.spike_width_units is None and self.broad_width_units is None:
            return self.spike_width, self.broad_spike_width
        spectral_dim = resolve_spectral_dim(da, self.spectral_dim)
        channel_width = median_channel_width(da, spectral_dim)
        spike_width = self.spike_width
        if self.spike_width_units is not None:
            spike_width = spectral_to_channels(
                self.spike_width_units, channel_width, minimum=3, odd=True
            )
        broad_spike_width = self.broad_spike_width
        if self.broad_width_units is not None:
            broad_spike_width = spectral_to_channels(
                self.broad_width_units, channel_width, minimum=1, odd=False
            )
        return spike_width, broad_spike_width

    def _apply_1d(
        self,
        da: xr.DataArray,
        arr_1d: np.ndarray,
        *,
        want_diagnostics: bool = True,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        spike_width, broad_spike_width = self._resolved_widths(da)
        corrected, mask = remove_cosmic_rays_1d(
            arr_1d,
            kernel_size=spike_width,
            threshold=self.spike_threshold,
            max_passes=self.spike_passes,
            broad_spike_width=broad_spike_width,
        )
        meta = self._meta_1d(mask, spike_width, broad_spike_width)
        out = with_new_values(da, corrected.reshape(da.shape), "Cosmic Ray Correction", meta)
        diag = {"cosmic_mask": mask, "corrected_1d": corrected} if want_diagnostics else {}
        return out, diag

    def _apply_loop_1d(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Apply the 1D engine independently to every spectrum.

        Detection runs for every spectrum first; when there are enough valid
        spectra, a cross-spectrum consensus veto (see
        :attr:`consensus_veto_fraction`) unflags any channel hit in most of
        them before any spectrum is repaired — a cosmic ray essentially
        never lands on the same channel across many pixels, but a real
        shared spectral feature always does.
        """
        spike_width, broad_spike_width = self._resolved_widths(da)
        arr = np.asarray(da.values, dtype=float)
        orig_shape = arr.shape
        flat = arr.reshape(-1, orig_shape[-1])
        out_flat = flat.copy()
        masks = np.zeros_like(flat, dtype=bool)
        valid = ~np.all(np.isnan(flat), axis=1)

        for i in np.flatnonzero(valid):
            masks[i] = detect_cosmic_mask_1d(
                flat[i],
                kernel_size=spike_width,
                threshold=self.spike_threshold,
                max_passes=self.spike_passes,
                broad_spike_width=broad_spike_width,
            )

        n_vetoed_channels = 0
        n_valid = int(valid.sum())
        if self.consensus_veto_fraction > 0.0 and n_valid >= _COLLECTION_THRESHOLD:
            consensus = masks[valid].mean(axis=0)
            veto = consensus > self.consensus_veto_fraction
            if np.any(veto):
                n_vetoed_channels = int(np.count_nonzero(veto))
                masks[:, veto] = False

        n_corrected = 0
        for i in np.flatnonzero(valid):
            if not np.any(masks[i]):
                continue
            out_flat[i] = repair_cosmic_mask_1d(
                flat[i], masks[i], broad_spike_width=broad_spike_width
            )
            n_corrected += 1

        meta = {
            "spike_width": spike_width,
            "spike_threshold": self.spike_threshold,
            "spike_passes": self.spike_passes,
            "spectra_corrected": n_corrected,
        }
        if n_vetoed_channels:
            meta["consensus_vetoed_channels"] = n_vetoed_channels
        out = with_new_values(da, out_flat.reshape(orig_shape), "Cosmic Ray Correction", meta)
        diag: dict[str, Any] = (
            {"cosmic_masks": masks.reshape(orig_shape)} if want_diagnostics else {}
        )
        return out, diag

    def _apply_collection(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Global-median or PCA engine for 2D with ≥ 20 spectra."""
        result = correct_cosmic_rays_collection(
            np.asarray(da.values, dtype=float),
            method=self.map_method,
            threshold=self.spike_threshold,
            spectral_dilate_channels=self.map_spike_width,
            max_repair_extent=self.map_spike_width * 2,
            n_components=self.map_n_components,
            return_diagnostics=want_diagnostics,
        )
        if want_diagnostics:
            corrected, meta, diag = result  # type: ignore[misc]
        else:
            corrected, meta = result  # type: ignore[misc]
            diag = {}
        return (
            with_new_values(da, corrected, "Cosmic Ray Correction", meta),
            diag,
        )

    def _apply_map(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Spatial disk-median or PCA engine for 3D maps with ≥ 20 spectra."""
        if self.map_method == "pca":
            result = correct_cosmic_rays_collection(
                np.asarray(da.values, dtype=float),
                method="pca",
                threshold=self.spike_threshold,
                spectral_dilate_channels=self.map_spike_width,
                max_repair_extent=self.map_spike_width * 2,
                n_components=self.map_n_components,
                return_diagnostics=want_diagnostics,
            )
            if want_diagnostics:
                corrected, meta, diag = result  # type: ignore[misc]
            else:
                corrected, meta = result  # type: ignore[misc]
                diag = {}
            return (
                with_new_values(da, corrected, "Cosmic Ray Correction", meta),
                diag,
            )

        # Default: spatial disk-median
        result_map = correct_cosmic_rays_on_map_cube(
            da.values,
            sensitivity=self.map_sensitivity,
            spectral_dilate_channels=self.map_spike_width,
            disk_radius=self.map_disk_radius,
            map_mad_multiplier=_MAP_MAD_MULTIPLIER,
            map_noisy_channel_relax_min=_MAP_NOISY_RELAX_MIN,
            map_max_spectral_repair_extent=self.map_spike_width * 2,
            map_min_residual_over_cutoff=_MAP_MIN_RESIDUAL_OVER_CUTOFF,
            map_require_spatial_local_max=_MAP_REQUIRE_SPATIAL_LOCAL_MAX,
            return_diagnostic_masks=want_diagnostics,
        )
        if want_diagnostics:
            corrected_m, meta_m, diag_m = result_map  # type: ignore[misc]
        else:
            corrected_m, meta_m = result_map  # type: ignore[misc]
            diag_m = {}
        return (
            with_new_values(da, corrected_m, "Cosmic Ray Correction", meta_m),
            diag_m,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _meta_1d(
        self, mask: np.ndarray, spike_width: int, broad_spike_width: int
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "spike_width": spike_width,
            "spike_threshold": self.spike_threshold,
            "spike_passes": self.spike_passes,
            "broad_spike_width": broad_spike_width,
        }
        if np.any(mask):
            meta["CRs found (spectral indices)"] = list(np.flatnonzero(mask))
        return meta

    @staticmethod
    def _da_label(da: xr.DataArray) -> str:
        parts: list[str] = []
        if da.name:
            parts.append(f"name={da.name!r}")
        for key in ("title", "filename", "source"):
            if key in da.attrs:
                parts.append(f"{key}={da.attrs[key]!r}")
                break
        return ", ".join(parts) if parts else "unnamed DataArray"

    @staticmethod
    def _maybe_compute_for_map(da: xr.DataArray) -> xr.DataArray:
        return ensure_in_memory(
            da,
            caller="CosmicRayRemover",
            reason=(
                "The spatial disk-median algorithm requires all pixels in "
                "memory simultaneously.\n"
                "If this causes an out-of-memory error, consider splitting "
                "the map into sub-regions before CR removal."
            ),
            stacklevel=3,
        )


__all__ = ["CosmicRayRemover", "remove_cosmic_rays_1d"]
