# -*- coding: utf-8 -*-
"""Substrate optical constants (n, k) for the lab's excitation wavelengths.

Values are tabulated only at the two laser lines in use (355 nm and 320 nm).
Edit these values if you have better measurements for your specific substrates.

Sources:
  Fused silica:    Malitson 1965 Sellmeier; k < 1e-7 at these wavelengths → 0.
  Soda-lime glass: VERIFY — k is strongly iron-content-dependent; the values
                   here are rough estimates for low-Fe float glass.
                   Rubin 1985, Sol. Energy Mater. 12, 275 gives n; k at UV
                   edge varies with Fe concentration. REPLACE with your
                   measured value or literature for your specific slide batch.
  Si:              Aspnes & Studna 1983, Phys. Rev. B 27, 985. VERIFY 320 nm
                   value — interpolated between tabulation rows.
"""

from __future__ import annotations

# {substrate_name: {laser_nm: (n, k)}}
SUBSTRATE_NK: dict[str, dict[float, tuple[float, float]]] = {
    "Fused silica (SiO2)": {
        355.0: (1.4761, 0.0),   # Malitson 1965 Sellmeier, k negligible
        320.0: (1.4823, 0.0),   # Malitson 1965 Sellmeier, k negligible
    },
    "Soda-lime glass": {
        355.0: (1.539, 1.0e-6),  # VERIFY k — iron-content dependent
        320.0: (1.552, 5.0e-6),  # VERIFY both n and k near Fe3+ UV edge
    },
    "Si": {
        355.0: (5.610, 3.010),   # Aspnes & Studna 1983 at 3.49 eV
        320.0: (5.100, 3.610),   # VERIFY — interpolated near 3.87 eV
    },
}

SUBSTRATE_LABELS: list[str] = list(SUBSTRATE_NK) + ["Custom"]

# WDF laser_nm can be a float like 354.9; match within this tolerance.
LASER_MATCH_TOL_NM: float = 2.0


def lookup_substrate_nk(name: str, laser_nm: float) -> tuple[float, float] | None:
    """Return (n, k) for the named substrate at the given laser wavelength.

    Matches within LASER_MATCH_TOL_NM of a tabulated laser line.
    Returns None if the substrate is unknown or the laser line is not tabulated.
    """
    table = SUBSTRATE_NK.get(name)
    if table is None:
        return None
    best = min(table.keys(), key=lambda lam: abs(lam - laser_nm))
    if abs(best - laser_nm) <= LASER_MATCH_TOL_NM:
        return table[best]
    return None


__all__ = ["SUBSTRATE_NK", "SUBSTRATE_LABELS", "LASER_MATCH_TOL_NM", "lookup_substrate_nk"]
