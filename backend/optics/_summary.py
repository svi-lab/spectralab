"""Orchestrator: nm-at-boundary -> cm-internal -> human-readable summary dict.

The underlying physics lives in _core.py (all-cm convention). This module
handles unit conversion and calls _core functions in the right order so the
Data page stays declarative.
"""

from __future__ import annotations

import numpy as np

from ._core import (
    alpha_from_k,
    d99_corrected,
    d99_simple,
    film_absorption_fraction,
    fresnel_R_air,
    fresnel_R_interface,
    penetration_depth,
    substrate_exit_fraction,
    substrate_transmission_fraction,
    tmm_energy_fractions,
)


def film_stack_summary(
    *,
    laser_nm: float,
    film_n: float,
    film_k: float,
    film_d_nm: float,
    sub_n: float,
    sub_k: float,
    sub_d_mm: float,
) -> dict:
    """Compute optical summary numbers for a film-on-substrate stack.

    All public inputs in nm / mm; all internal calls in cm.

    Parameters
    ----------
    laser_nm   : Excitation wavelength [nm]
    film_n     : Film real refractive index
    film_k     : Film extinction coefficient (0 = transparent film)
    film_d_nm  : Film thickness [nm]
    sub_n      : Substrate real refractive index
    sub_k      : Substrate extinction coefficient
    sub_d_mm   : Substrate thickness [mm]

    Returns
    -------
    dict with keys (all floats; np.inf when absorption is zero):
      alpha_film_cm, delta_film_nm, d99_simple_nm, d99_corrected_nm,
      R_air_film, R_film_sub, R_air_sub,
      bl_R, bl_A_film, bl_T_sub,
      tmm_R, tmm_A_film, tmm_T_sub,
      alpha_sub_cm, delta_sub_nm,
      sub_exit_frac, c_physics

    c_physics = tmm_T_sub / (1 - R_air_sub) — the physics-
    predicted background suppression scale (see suppression_scale_physics).
    This is the single source of truth consumed by the Preprocessing page.
    """
    lam_cm = laser_nm * 1e-7
    d_film_cm = film_d_nm * 1e-7
    d_sub_cm = sub_d_mm * 0.1

    alpha_film = alpha_from_k(film_k, lam_cm)
    alpha_sub = alpha_from_k(sub_k, lam_cm)

    R_air_film = fresnel_R_air(film_n, film_k)
    R_film_sub = fresnel_R_interface(film_n, film_k, sub_n, sub_k)
    R_air_sub = fresnel_R_air(sub_n, sub_k)

    delta_film_cm = penetration_depth(alpha_film)
    d99s_cm = d99_simple(alpha_film)
    d99c_cm = d99_corrected(alpha_film, R_air_film)

    # Beer-Lambert fractions
    bl_A_film = film_absorption_fraction(1.0, R_air_film, alpha_film, d_film_cm, R_film_sub)
    bl_T_sub = substrate_transmission_fraction(1.0, R_air_film, alpha_film, d_film_cm, R_film_sub)
    bl_R = R_air_film  # Beer-Lambert uses single-interface Fresnel only

    # TMM fractions
    tmm_R, tmm_A_film, tmm_T_sub = tmm_energy_fractions(
        1.0, lam_cm, film_n, film_k, d_film_cm, sub_n, sub_k
    )

    delta_sub_cm = penetration_depth(alpha_sub)

    sub_exit = substrate_exit_fraction(1.0, tmm_T_sub, alpha_sub, d_sub_cm, sub_n, sub_k)

    # Physics-predicted background suppression scale (see suppression_scale_physics)
    c_physics = tmm_T_sub / (1.0 - R_air_sub)

    def _to_nm(v_cm: float) -> float:
        return v_cm * 1e7 if np.isfinite(v_cm) else np.inf

    return {
        "alpha_film_cm": alpha_film,
        "delta_film_nm": _to_nm(delta_film_cm),
        "d99_simple_nm": _to_nm(d99s_cm),
        "d99_corrected_nm": _to_nm(d99c_cm),
        "R_air_film": R_air_film,
        "R_film_sub": R_film_sub,
        "R_air_sub": R_air_sub,
        "bl_R": bl_R,
        "bl_A_film": bl_A_film,
        "bl_T_sub": bl_T_sub,
        "tmm_R": tmm_R,
        "tmm_A_film": tmm_A_film,
        "tmm_T_sub": tmm_T_sub,
        "alpha_sub_cm": alpha_sub,
        "delta_sub_nm": _to_nm(delta_sub_cm),
        "sub_exit_frac": sub_exit,
        "c_physics": c_physics,
    }


def bare_substrate_summary(
    *,
    laser_nm: float,
    sub_n: float,
    sub_k: float,
    sub_d_mm: float,
) -> dict:
    """Optical summary for a bare-substrate (reference) sample — no film.

    Returns
    -------
    dict with keys:
      R_air_sub    — front-surface reflectance at the laser wavelength
      entry_frac   — 1 − R_air_sub, the fraction of the laser entering the substrate
      alpha_sub_cm, delta_sub_nm
    """
    lam_cm = laser_nm * 1e-7
    d_sub_cm = sub_d_mm * 0.1  # noqa: F841 — kept for future exit-fraction use

    alpha_sub = alpha_from_k(sub_k, lam_cm)
    R_air_sub = fresnel_R_air(sub_n, sub_k)
    delta_sub_cm = penetration_depth(alpha_sub)

    return {
        "sample_type": "substrate",
        "R_air_sub": R_air_sub,
        "entry_frac": 1.0 - R_air_sub,
        "alpha_sub_cm": alpha_sub,
        "delta_sub_nm": delta_sub_cm * 1e7 if np.isfinite(delta_sub_cm) else np.inf,
    }


def suppression_scale_physics(
    *,
    laser_nm: float,
    film_n: float,
    film_k: float,
    film_d_nm: float,
    sub_n: float,
    sub_k: float,
) -> float:
    """Compute the physics-predicted background scale factor c_physics.

    When you subtract a bare-substrate reference from a film+substrate spectrum:

        S_measured ≈ S_film + c · S_reference

    After normalising both spectra by (laser_power × exposure_time), c_physics is:

        c_physics = T_tmm(λ_exc) / (1 - R_air_sub(λ_exc))

    - Numerator T_tmm: fraction of excitation laser that reaches the substrate
      through the film (accounts for interference reflection and film absorption).
    - Denominator (1 - R_air_sub): fraction of laser entering the *bare* reference
      substrate — the reference was excited without a film in the way.

    If c_fitted ≈ c_physics: the optical model is self-consistent.
    If they disagree by > 2x: check film thickness, k, or focus alignment.
    """
    lam_cm = laser_nm * 1e-7
    d_film_cm = film_d_nm * 1e-7

    _, _, tmm_T_sub = tmm_energy_fractions(1.0, lam_cm, film_n, film_k, d_film_cm, sub_n, sub_k)
    R_air_sub = fresnel_R_air(sub_n, sub_k)

    return tmm_T_sub / (1.0 - R_air_sub)


__all__ = ["bare_substrate_summary", "film_stack_summary", "suppression_scale_physics"]
