"""
Pure physics functions for laser penetration depth and thin-film optics.
All functions work in cm internally; callers convert nm <-> cm at boundaries.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Basic optical conversions
# ---------------------------------------------------------------------------


def alpha_from_k(k: float, wavelength_cm: float) -> float:
    """Absorption coefficient from extinction coefficient.

    alpha = 4*pi*k / lambda  [cm^-1]
    """
    return 4 * np.pi * k / wavelength_cm


def k_from_alpha(alpha: float, wavelength_cm: float) -> float:
    """Extinction coefficient from absorption coefficient.

    k = alpha * lambda / (4*pi)
    """
    return alpha * wavelength_cm / (4 * np.pi)


# ---------------------------------------------------------------------------
# Fresnel reflectances (incoherent / single-interface)
# ---------------------------------------------------------------------------


def fresnel_R_air(n: float, k: float) -> float:
    """Normal-incidence Fresnel reflectance at air/medium interface.

    R = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)
    """
    return ((n - 1) ** 2 + k**2) / ((n + 1) ** 2 + k**2)


def fresnel_R_interface(n1: float, k1: float, n2: float, k2: float) -> float:
    """Normal-incidence Fresnel reflectance at the interface between two media.

    R = ((n1-n2)^2 + (k1-k2)^2) / ((n1+n2)^2 + (k1+k2)^2)
    """
    return ((n1 - n2) ** 2 + (k1 - k2) ** 2) / ((n1 + n2) ** 2 + (k1 + k2) ** 2)


# ---------------------------------------------------------------------------
# Penetration depths (Beer-Lambert, analytical)
# ---------------------------------------------------------------------------


def penetration_depth(alpha: float) -> float:
    """1/e penetration depth delta = 1/alpha [cm]; inf if alpha == 0."""
    if alpha == 0:
        return np.inf
    return 1.0 / alpha


def d99_simple(alpha: float) -> float:
    """Depth at which 99% of transmitted intensity is absorbed: ln(100)/alpha [cm]."""
    if alpha == 0:
        return np.inf
    return np.log(100) / alpha


def d99_corrected(alpha: float, R: float) -> float:
    """Depth at which 99% of incident intensity is gone (reflection + absorption).

    Solves (1-R)*exp(-alpha*z) = 0.01  =>  z = ln(100*(1-R)) / alpha
    Returns 0 when R >= 0.99 (nearly all light already reflected).
    """
    if alpha == 0:
        return np.inf
    arg = 100 * (1 - R)
    if arg <= 1.0:
        return 0.0
    return np.log(arg) / alpha


# ---------------------------------------------------------------------------
# Transfer Matrix Method (TMM) — air | film | substrate, normal incidence
# ---------------------------------------------------------------------------


def tmm_r_t(
    wavelength_cm: float,
    n_film: float,
    k_film: float,
    d_film_cm: float,
    n_sub: float,
    k_sub: float,
) -> tuple:
    """Transfer Matrix Method for a single film on a substrate at normal incidence.

    Stack: air (N0=1+0j) | film (N1, thickness d_film_cm) | substrate (N2).

    Phase thickness: phi = 2*pi*N1*d / lambda  (complex for absorbing media)

    Born & Wolf characteristic matrix (eq. 1.6.22), off-diagonal signs −i:
        M = [[cos(phi),       -i/N1 * sin(phi)],
             [-i*N1*sin(phi),  cos(phi)        ]]

    With A = M[0,0] + M[0,1]*N2, B = M[1,0] + M[1,1]*N2:
        r = (A*N0 - B) / (A*N0 + B)
        t = 2*N0  / (A*N0 + B)
        R = |r|^2
        T = Re(N2)/Re(N0) * |t|^2

    Returns
    -------
    R : float   Energy reflectance [0, 1]
    T : float   Energy transmittance [0, 1]
    r : complex Amplitude reflection coefficient
    t : complex Amplitude transmission coefficient
    """
    N0 = complex(1.0, 0.0)
    N1 = complex(n_film, k_film)
    N2 = complex(n_sub, k_sub)

    phi = 2 * np.pi * N1 * d_film_cm / wavelength_cm

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    M00 = cos_phi
    M01 = (-1j / N1) * sin_phi
    M10 = -1j * N1 * sin_phi
    M11 = cos_phi

    A = M00 + M01 * N2
    B = M10 + M11 * N2

    denom = A * N0 + B
    r = (A * N0 - B) / denom
    t = 2 * N0 / denom

    R = float(np.abs(r) ** 2)
    T = float(np.real(N2) / np.real(N0) * np.abs(t) ** 2)

    return R, T, r, t


def tmm_field_in_film(
    z_film_cm: np.ndarray,
    wavelength_cm: float,
    N1: complex,
    N2: complex,
    d_film_cm: float,
    t: complex,
) -> np.ndarray:
    """Electric field intensity inside the film (backward propagation from exit).

    Backward-from-exit formulation avoids growing exponentials in absorbing media:
        z' = d_film - z  (distance from back face)
        phi(z') = 2*pi*N1*z' / lambda
        E(z) = t * [cos(phi(z')) - i*(N2/N1)*sin(phi(z'))]
        I(z)/I_inc = Re(N1) * |E(z)|^2

    Returns
    -------
    ndarray  I_film / I_incident at each z (includes interference fringes)
    """
    z_prime = d_film_cm - z_film_cm
    phi_prime = 2 * np.pi * N1 * z_prime / wavelength_cm
    E = t * (np.cos(phi_prime) - 1j * (N2 / N1) * np.sin(phi_prime))
    return float(np.real(N1)) * np.abs(E) ** 2


def tmm_intensity_profile(
    z_film_cm: np.ndarray,
    z_sub_cm: np.ndarray,
    wavelength_cm: float,
    n_film: float,
    k_film: float,
    d_film_cm: float,
    n_sub: float,
    k_sub: float,
    I0: float = 1.0,
) -> tuple:
    """Intensity profile I(z)/I0 across film (TMM) and substrate (Beer-Lambert).

    Returns
    -------
    I_film : ndarray  Normalized intensity in film (with interference fringes)
    I_sub  : ndarray  Normalized intensity in substrate (Beer-Lambert from TMM T)
    """
    N1 = complex(n_film, k_film)
    N2 = complex(n_sub, k_sub)

    R, T, r, t = tmm_r_t(wavelength_cm, n_film, k_film, d_film_cm, n_sub, k_sub)

    I_film = I0 * tmm_field_in_film(z_film_cm, wavelength_cm, N1, N2, d_film_cm, t)

    alpha_sub = alpha_from_k(k_sub, wavelength_cm)
    I_sub = I0 * T * np.exp(-alpha_sub * z_sub_cm)

    return I_film, I_sub


def tmm_absorbed_density(
    z_film_cm: np.ndarray,
    z_sub_cm: np.ndarray,
    wavelength_cm: float,
    n_film: float,
    k_film: float,
    d_film_cm: float,
    n_sub: float,
    k_sub: float,
    I0: float = 1.0,
) -> tuple:
    """Absorbed power density -dI/dz [I0/cm] across film and substrate.

    Film uses TMM field; substrate uses Beer-Lambert from TMM transmittance.

    Returns
    -------
    gen_film : ndarray  alpha_film * I_film(z)  [I0/cm]
    gen_sub  : ndarray  alpha_sub  * I_sub(z)   [I0/cm]
    """
    alpha_film = alpha_from_k(k_film, wavelength_cm)
    alpha_sub = alpha_from_k(k_sub, wavelength_cm)

    I_film, I_sub = tmm_intensity_profile(
        z_film_cm, z_sub_cm, wavelength_cm, n_film, k_film, d_film_cm, n_sub, k_sub, I0
    )

    return alpha_film * I_film, alpha_sub * I_sub


# ---------------------------------------------------------------------------
# TMM energy fractions
# ---------------------------------------------------------------------------


def tmm_energy_fractions(
    I0: float,
    wavelength_cm: float,
    n_film: float,
    k_film: float,
    d_film_cm: float,
    n_sub: float,
    k_sub: float,
) -> tuple:
    """Energy fractions via TMM: (R_total, A_film, T_to_substrate).

    A_film = 1 - R_total - T_total  (energy conservation, includes interference).

    Returns
    -------
    R_total : float   Fraction reflected (includes thin-film interference)
    A_film  : float   Fraction absorbed in film
    T_total : float   Fraction transmitted into substrate
    """
    R, T, _, _ = tmm_r_t(wavelength_cm, n_film, k_film, d_film_cm, n_sub, k_sub)
    A_film = max(0.0, 1.0 - R - T)  # clamp numerical noise
    return R, A_film, T


# ---------------------------------------------------------------------------
# Beer-Lambert intensity profile (reference / fallback)
# ---------------------------------------------------------------------------


def intensity_profile(
    z_film: np.ndarray,
    z_sub: np.ndarray,
    I0: float,
    R_air: float,
    alpha_film: float,
    d_film_cm: float,
    R_fs: float,
    alpha_sub: float,
) -> tuple:
    """Intensity profile using Beer-Lambert decay (no interference).

    Returns
    -------
    I_film : ndarray  Intensity in film (normalized to I0)
    I_sub  : ndarray  Intensity in substrate (normalized to I0)
    """
    I_film = I0 * (1 - R_air) * np.exp(-alpha_film * z_film)
    I_at_interface = I0 * (1 - R_air) * np.exp(-alpha_film * d_film_cm)
    I_entering_sub = I_at_interface * (1 - R_fs)
    I_sub = I_entering_sub * np.exp(-alpha_sub * z_sub)
    return I_film, I_sub


def absorbed_power_density(
    z_film: np.ndarray,
    z_sub: np.ndarray,
    I0: float,
    R_air: float,
    alpha_film: float,
    d_film_cm: float,
    R_fs: float,
    alpha_sub: float,
) -> tuple:
    """Absorbed power density -dI/dz [I0/cm] using Beer-Lambert (no interference)."""
    I_film, I_sub = intensity_profile(
        z_film, z_sub, I0, R_air, alpha_film, d_film_cm, R_fs, alpha_sub
    )
    return alpha_film * I_film, alpha_sub * I_sub


# ---------------------------------------------------------------------------
# Energy fractions (Beer-Lambert)
# ---------------------------------------------------------------------------


def film_absorption_fraction(
    I0: float,
    R_air: float,
    alpha_film: float,
    d_film_cm: float,
    R_fs: float,
) -> float:
    """Fraction of incident intensity absorbed within the film (Beer-Lambert).

    A_film = (I_entering_film - I_entering_substrate) / I0
    """
    I_entering_film = I0 * (1 - R_air)
    I_at_interface = I_entering_film * np.exp(-alpha_film * d_film_cm)
    I_entering_sub = I_at_interface * (1 - R_fs)
    return (I_entering_film - I_entering_sub) / I0


def substrate_transmission_fraction(
    I0: float,
    R_air: float,
    alpha_film: float,
    d_film_cm: float,
    R_fs: float,
) -> float:
    """Fraction of incident intensity transmitted into the substrate (Beer-Lambert)."""
    I_entering_film = I0 * (1 - R_air)
    I_at_interface = I_entering_film * np.exp(-alpha_film * d_film_cm)
    I_entering_sub = I_at_interface * (1 - R_fs)
    return I_entering_sub / I0


def substrate_exit_fraction(
    I0: float,
    T_into_sub: float,
    alpha_sub: float,
    d_sub_cm: float,
    n_sub: float,
    k_sub: float,
) -> float:
    """Fraction of incident intensity that exits the substrate back face.

    Beer-Lambert through substrate, then Fresnel at substrate/air interface.
    """
    R_back = fresnel_R_air(n_sub, k_sub)
    I_at_back = I0 * T_into_sub * np.exp(-alpha_sub * d_sub_cm)
    return I_at_back * (1 - R_back) / I0


__all__ = [
    "alpha_from_k",
    "k_from_alpha",
    "fresnel_R_air",
    "fresnel_R_interface",
    "penetration_depth",
    "d99_simple",
    "d99_corrected",
    "tmm_r_t",
    "tmm_field_in_film",
    "tmm_intensity_profile",
    "tmm_absorbed_density",
    "tmm_energy_fractions",
    "intensity_profile",
    "absorbed_power_density",
    "film_absorption_fraction",
    "substrate_transmission_fraction",
    "substrate_exit_fraction",
]
