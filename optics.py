"""
optics.py — Pure physics functions for laser penetration depth calculator.
All functions work in cm internally; the UI converts nm <-> cm at boundaries.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Basic optical conversions
# ---------------------------------------------------------------------------

def alpha_from_k(k: float, wavelength_cm: float) -> float:
    """
    Absorption coefficient from extinction coefficient.

    Parameters
    ----------
    k : float
        Extinction coefficient (dimensionless)
    wavelength_cm : float
        Laser wavelength [cm]

    Returns
    -------
    float
        alpha [cm^-1]
    """
    return 4 * np.pi * k / wavelength_cm


def k_from_alpha(alpha: float, wavelength_cm: float) -> float:
    """
    Extinction coefficient from absorption coefficient.

    Parameters
    ----------
    alpha : float
        Absorption coefficient [cm^-1]
    wavelength_cm : float
        Laser wavelength [cm]

    Returns
    -------
    float
        k (dimensionless)
    """
    return alpha * wavelength_cm / (4 * np.pi)


# ---------------------------------------------------------------------------
# Fresnel reflectances (incoherent / single-interface)
# ---------------------------------------------------------------------------

def fresnel_R_air(n: float, k: float) -> float:
    """
    Normal-incidence Fresnel reflectance at air/medium interface.

    Formula: R = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)

    Parameters
    ----------
    n : float
        Real part of refractive index
    k : float
        Extinction coefficient

    Returns
    -------
    float
        Reflectance R [0, 1]
    """
    return ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)


def fresnel_R_interface(n1: float, k1: float, n2: float, k2: float) -> float:
    """
    Normal-incidence Fresnel reflectance at the interface between two media.

    Formula: R = ((n1-n2)^2 + (k1-k2)^2) / ((n1+n2)^2 + (k1+k2)^2)

    Parameters
    ----------
    n1, k1 : float
        Complex refractive index of medium 1
    n2, k2 : float
        Complex refractive index of medium 2

    Returns
    -------
    float
        Reflectance R [0, 1]
    """
    return ((n1 - n2) ** 2 + (k1 - k2) ** 2) / ((n1 + n2) ** 2 + (k1 + k2) ** 2)


# ---------------------------------------------------------------------------
# Penetration depths (Beer–Lambert, analytical)
# ---------------------------------------------------------------------------

def penetration_depth(alpha: float) -> float:
    """
    1/e penetration depth (delta).

    Parameters
    ----------
    alpha : float
        Absorption coefficient [cm^-1]

    Returns
    -------
    float
        delta [cm]; inf if alpha == 0
    """
    if alpha == 0:
        return np.inf
    return 1.0 / alpha


def d99_simple(alpha: float) -> float:
    """
    Depth at which 99% of the *transmitted* (post-surface-reflection) intensity
    is absorbed: ln(100)/alpha.

    Parameters
    ----------
    alpha : float
        Absorption coefficient [cm^-1]

    Returns
    -------
    float
        d99_simple [cm]; inf if alpha == 0
    """
    if alpha == 0:
        return np.inf
    return np.log(100) / alpha


def d99_corrected(alpha: float, R: float) -> float:
    """
    Depth at which 99% of the *incident* intensity is gone (reflection + absorption).

    Solves (1-R)*exp(-alpha*z) = 0.01 => z = ln(100*(1-R)) / alpha

    Parameters
    ----------
    alpha : float
        Absorption coefficient [cm^-1]
    R : float
        Surface reflectance [0, 1)

    Returns
    -------
    float
        d99 [cm]; inf if alpha == 0
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
    n_film: float, k_film: float, d_film_cm: float,
    n_sub: float, k_sub: float,
) -> tuple:
    """
    Transfer Matrix Method for a single film on a substrate at normal incidence.

    Stack: air (N0=1+0j) | film (N1=n_film+ik_film, thickness d_film_cm) | substrate (N2).

    Phase thickness of the film: phi = 2*pi*N1*d / lambda  (complex for absorbing media)

    Transfer matrix:
        M = [[cos(phi),      i/N1 * sin(phi)],
             [i*N1*sin(phi), cos(phi)       ]]

    Amplitude coefficients with A = M[0,0]*N2 + M[0,1], B = M[1,0]*N2 + M[1,1]:
        r = (A*N0 - B) / (A*N0 + B)
        t = 2*N0  / (A*N0 + B)
        R = |r|^2
        T = Re(N2)/Re(N0) * |t|^2

    Parameters
    ----------
    wavelength_cm : float
        Wavelength [cm]
    n_film, k_film : float
        Film complex refractive index
    d_film_cm : float
        Film thickness [cm]
    n_sub, k_sub : float
        Substrate complex refractive index

    Returns
    -------
    R : float
        Energy reflectance [0, 1]
    T : float
        Energy transmittance [0, 1]
    r : complex
        Amplitude reflection coefficient
    t : complex
        Amplitude transmission coefficient
    """
    N0 = complex(1.0, 0.0)
    N1 = complex(n_film, k_film)
    N2 = complex(n_sub, k_sub)

    phi = 2 * np.pi * N1 * d_film_cm / wavelength_cm

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Born & Wolf characteristic matrix (eq. 1.6.22): off-diagonal signs are −i
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
    """
    Electric field intensity inside the film using backward propagation from exit.

    Backward-from-exit formulation avoids growing exponentials for absorbing media:
        z' = d_film - z  (distance from back face)
        phi(z') = 2*pi*N1*z' / lambda
        E(z) = t * [cos(phi(z')) + i*(N2/N1)*sin(phi(z'))]
        I(z)/I_inc = Re(N1) * |E(z)|^2

    Parameters
    ----------
    z_film_cm : ndarray
        Depth values inside the film [cm], from 0 (front face) to d_film_cm
    wavelength_cm : float
        Wavelength [cm]
    N1 : complex
        Film complex refractive index
    N2 : complex
        Substrate complex refractive index
    d_film_cm : float
        Film thickness [cm]
    t : complex
        Amplitude transmission coefficient from tmm_r_t

    Returns
    -------
    ndarray
        I_film / I_incident at each z (includes interference fringes)
    """
    z_prime = d_film_cm - z_film_cm  # distance from back face
    phi_prime = 2 * np.pi * N1 * z_prime / wavelength_cm

    # With Born & Wolf sign convention: E(z) = t·[cos(φ') − i·(N2/N1)·sin(φ')]
    E = t * (np.cos(phi_prime) - 1j * (N2 / N1) * np.sin(phi_prime))
    I_norm = float(np.real(N1)) * np.abs(E) ** 2

    return I_norm


def tmm_intensity_profile(
    z_film_cm: np.ndarray,
    z_sub_cm: np.ndarray,
    wavelength_cm: float,
    n_film: float, k_film: float, d_film_cm: float,
    n_sub: float, k_sub: float,
    I0: float = 1.0,
) -> tuple:
    """
    Intensity profile I(z)/I0 across film (TMM) and substrate (Beer–Lambert).

    Parameters
    ----------
    z_film_cm : ndarray
        Depth in film [cm], starting at 0
    z_sub_cm : ndarray
        Depth in substrate [cm], starting at 0 (relative to film/substrate interface)
    wavelength_cm : float
        Wavelength [cm]
    n_film, k_film : float
        Film refractive index
    d_film_cm : float
        Film thickness [cm]
    n_sub, k_sub : float
        Substrate refractive index
    I0 : float
        Incident intensity

    Returns
    -------
    I_film : ndarray
        Normalized intensity in film (with interference fringes)
    I_sub : ndarray
        Normalized intensity in substrate (Beer–Lambert from TMM transmittance)
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
    n_film: float, k_film: float, d_film_cm: float,
    n_sub: float, k_sub: float,
    I0: float = 1.0,
) -> tuple:
    """
    Absorbed power density -dI/dz [I0/cm] across film and substrate.

    Film uses TMM field; substrate uses Beer–Lambert from TMM transmittance.

    Parameters
    ----------
    (same as tmm_intensity_profile)

    Returns
    -------
    gen_film : ndarray
        alpha_film * I_film(z)  [I0/cm]
    gen_sub : ndarray
        alpha_sub * I_sub(z)    [I0/cm]
    """
    alpha_film = alpha_from_k(k_film, wavelength_cm)
    alpha_sub = alpha_from_k(k_sub, wavelength_cm)

    I_film, I_sub = tmm_intensity_profile(
        z_film_cm, z_sub_cm, wavelength_cm,
        n_film, k_film, d_film_cm, n_sub, k_sub, I0
    )

    return alpha_film * I_film, alpha_sub * I_sub


# ---------------------------------------------------------------------------
# Beer–Lambert intensity profile (fallback / reference)
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
    """
    Intensity profile using simple Beer–Lambert decay (no interference).

    Parameters
    ----------
    z_film : ndarray
        Depth in film [cm], starting at 0
    z_sub : ndarray
        Depth in substrate [cm], starting at 0 (relative to interface)
    I0 : float
        Incident intensity
    R_air : float
        Air/film Fresnel reflectance
    alpha_film : float
        Film absorption coefficient [cm^-1]
    d_film_cm : float
        Film thickness [cm]
    R_fs : float
        Film/substrate interface reflectance
    alpha_sub : float
        Substrate absorption coefficient [cm^-1]

    Returns
    -------
    I_film : ndarray
        Intensity in film region (normalized to I0)
    I_sub : ndarray
        Intensity in substrate region (normalized to I0)
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
    """
    Absorbed power density -dI/dz [I0/cm] using Beer–Lambert (no interference).

    Parameters
    ----------
    (same as intensity_profile)

    Returns
    -------
    gen_film : ndarray
    gen_sub : ndarray
    """
    I_film, I_sub = intensity_profile(
        z_film, z_sub, I0, R_air, alpha_film, d_film_cm, R_fs, alpha_sub
    )
    return alpha_film * I_film, alpha_sub * I_sub


# ---------------------------------------------------------------------------
# Energy fractions
# ---------------------------------------------------------------------------

def film_absorption_fraction(
    I0: float,
    R_air: float,
    alpha_film: float,
    d_film_cm: float,
    R_fs: float,
) -> float:
    """
    Fraction of incident intensity absorbed within the film (Beer–Lambert, no TMM).

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
    """
    Fraction of incident intensity transmitted into the substrate (Beer–Lambert).
    """
    I_entering_film = I0 * (1 - R_air)
    I_at_interface = I_entering_film * np.exp(-alpha_film * d_film_cm)
    I_entering_sub = I_at_interface * (1 - R_fs)
    return I_entering_sub / I0


def tmm_energy_fractions(
    I0: float,
    wavelength_cm: float,
    n_film: float, k_film: float, d_film_cm: float,
    n_sub: float, k_sub: float,
) -> tuple:
    """
    Energy fractions using TMM: (R_total, A_film, T_to_substrate).

    A_film = 1 - R_total - T_total  (energy conservation, no multiple reflections).

    Parameters
    ----------
    I0 : float
        Incident intensity
    wavelength_cm : float
    n_film, k_film : float
    d_film_cm : float
    n_sub, k_sub : float

    Returns
    -------
    R_total : float   fraction reflected (includes all interference)
    A_film  : float   fraction absorbed in film
    T_total : float   fraction transmitted into substrate
    """
    R, T, _, _ = tmm_r_t(wavelength_cm, n_film, k_film, d_film_cm, n_sub, k_sub)
    R_total = R
    T_total = T
    A_film = max(0.0, 1.0 - R_total - T_total)  # clamp numerical noise
    return R_total, A_film, T_total


def substrate_exit_fraction(
    I0: float,
    T_into_sub: float,
    alpha_sub: float,
    d_sub_cm: float,
    n_sub: float, k_sub: float,
) -> float:
    """
    Fraction of incident intensity that exits the substrate back face.

    Beer–Lambert through substrate, then Fresnel at substrate/air interface.

    Parameters
    ----------
    I0 : float
        Incident intensity
    T_into_sub : float
        Fraction entering the substrate (from TMM or Beer–Lambert)
    alpha_sub : float
        Substrate absorption coefficient [cm^-1]
    d_sub_cm : float
        Substrate physical thickness [cm]
    n_sub, k_sub : float
        Substrate refractive index (for back-face Fresnel)

    Returns
    -------
    float
        Fraction of I0 exiting the back face
    """
    R_back = fresnel_R_air(n_sub, k_sub)
    I_at_back = I0 * T_into_sub * np.exp(-alpha_sub * d_sub_cm)
    return I_at_back * (1 - R_back) / I0


# ---------------------------------------------------------------------------
# Material presets (n, k at 355 nm)
# ---------------------------------------------------------------------------

FILM_PRESETS = {
    "TiO2 anatase": {"n": 2.89, "k": 0.034},
    "TiO2 rutile":  {"n": 2.80, "k": 0.30},
    "ZnO":          {"n": 2.10, "k": 0.55},
    "ZnO:Al":       {"n": 2.05, "k": 0.50},
    "Custom":       {"n": 2.00, "k": 0.10},
}

SUBSTRATE_PRESETS = {
    "Fused silica":    {"n": 1.48, "k": 0.0},
    "Soda-lime glass": {"n": 1.52, "k": 0.01},
    "Sapphire":        {"n": 1.83, "k": 0.0},
    "Si":              {"n": 6.52, "k": 2.99},
    "Custom":          {"n": 1.50, "k": 0.0},
}


# ---------------------------------------------------------------------------
# Gaussian beam (paraxial, on-axis) — all z distances in nm, lambda in nm
# ---------------------------------------------------------------------------

def gaussian_rayleigh_length(n_inc: float, omega_0_nm: float, wavelength_nm: float) -> float:
    """
    Rayleigh length in the incident medium.

    z_R = π · n_inc · ω₀² / λ₀   [nm]

    Parameters
    ----------
    n_inc : float
        Real refractive index of the incident medium (usually 1.0 for air)
    omega_0_nm : float
        Beam waist radius at focus [nm]
    wavelength_nm : float
        Free-space wavelength [nm]

    Returns
    -------
    float
        z_R [nm]
    """
    return np.pi * n_inc * omega_0_nm ** 2 / wavelength_nm


def gaussian_film_params(
    n_inc: float, n_c: float, z_R_inc_nm: float, Delta_nm: float
) -> tuple:
    """
    ABCD-refracted beam parameters in the film layer.

    z0_c = (n_c / n_inc) · z_R_inc    [nm]  Rayleigh length in film
    zw_c = (n_c / n_inc) · Δ          [nm]  waist position from z = 0

    Uses real parts of the refractive indices (paraxial ABCD treatment).

    Parameters
    ----------
    n_inc : float   Real RI of incident medium
    n_c : float     Real part of film RI
    z_R_inc_nm : float   Rayleigh length in incident medium [nm]
    Delta_nm : float     Nominal focus depth from sample surface [nm];
                         Delta < 0 → focus above the surface (in air)

    Returns
    -------
    z0_c : float    [nm]
    zw_c : float    [nm]
    """
    f = n_c / n_inc
    return f * z_R_inc_nm, f * Delta_nm


def gaussian_substrate_params(
    n_inc: float, n_c: float, n_s: float,
    z_R_inc_nm: float, Delta_nm: float, d_film_nm: float
) -> tuple:
    """
    ABCD-refracted beam parameters seen by the substrate.

    z0_s = (n_s / n_inc) · z_R_inc
    zw_s = (n_s / n_inc) · Δ + d_film · (1 − n_s / n_c)

    Both measured from z = 0 (sample surface).
    zw_s < d_film when the geometrical focus falls inside the film;
    the substrate then sees a diverging beam.

    Parameters
    ----------
    n_inc, n_c, n_s : float   Real refractive indices
    z_R_inc_nm : float        Rayleigh length in incident medium [nm]
    Delta_nm : float          Nominal focus depth [nm]
    d_film_nm : float         Film thickness [nm]

    Returns
    -------
    z0_s : float    [nm]
    zw_s : float    [nm]
    """
    z0_s = (n_s / n_inc) * z_R_inc_nm
    zw_s = (n_s / n_inc) * Delta_nm + d_film_nm * (1.0 - n_s / n_c)
    return z0_s, zw_s


def gaussian_kappa_prime(n: float, kappa: float) -> float:
    """
    Modified absorption parameter for Gaussian propagation in an absorbing medium.

    κ' = κ / √(n² + κ²)

    For κ = 0 (transparent): κ' = 0 and ω(z) reduces to ω₀·√(1 + ζ²).

    Parameters
    ----------
    n : float     Real part of refractive index
    kappa : float Extinction coefficient

    Returns
    -------
    float   κ' ∈ [0, 1)
    """
    if kappa == 0.0:
        return 0.0
    return kappa / np.sqrt(n ** 2 + kappa ** 2)


def gaussian_omega(
    z_nm: np.ndarray,
    omega_0_nm: float,
    zw_nm: float,
    z0_nm: float,
    kappa_prime: float,
) -> np.ndarray:
    """
    Beam radius ω(z) inside an absorbing medium (paraxial approximation).

    ζ = (z − z_w) / z₀

    ω(z) = ω₀ · √[ (ζ² + 1 + 2κ'ζ) / (κ'ζ + 1) ]

    For κ' = 0 reduces exactly to ω₀·√(1 + ζ²)  (standard Gaussian).

    Parameters
    ----------
    z_nm : array_like   Depth values [nm]
    omega_0_nm : float  Beam waist (minimum radius, at focus) [nm]
    zw_nm : float       Waist position from z = 0 [nm]
    z0_nm : float       Rayleigh length in the medium [nm]
    kappa_prime : float Modified absorption parameter (gaussian_kappa_prime)

    Returns
    -------
    ndarray  ω(z) [nm]
    """
    zeta = (np.asarray(z_nm, dtype=float) - zw_nm) / z0_nm
    denom = np.maximum(kappa_prime * zeta + 1.0, 1e-12)
    numer = np.maximum(zeta ** 2 + 1.0 + 2.0 * kappa_prime * zeta, 0.0)
    return omega_0_nm * np.sqrt(numer / denom)


def gaussian_conc_factor(
    z_nm: np.ndarray,
    omega_0_nm: float,
    zw_nm: float,
    z0_nm: float,
    kappa_prime: float,
) -> np.ndarray:
    """
    On-axis Gaussian concentration factor (ω₀ / ω(z))².

    Mode B correction to the generation profile:
        G_B(z) = α · I_TMM(z) · gaussian_conc_factor(z, …)

    Equals 1 at the focus (z = z_w).  In the plane-wave limit (ω₀ → ∞,
    ζ → 0 everywhere) the factor → 1 throughout the medium.

    Parameters
    ----------
    (same as gaussian_omega)

    Returns
    -------
    ndarray  dimensionless, ≤ 1 for passive media
    """
    return (omega_0_nm / gaussian_omega(z_nm, omega_0_nm, zw_nm, z0_nm, kappa_prime)) ** 2
