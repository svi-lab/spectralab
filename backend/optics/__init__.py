# -*- coding: utf-8 -*-
"""backend.optics — thin-film optical calculations (TMM, Beer-Lambert, materials)."""

from ._core import (
    alpha_from_k,
    k_from_alpha,
    fresnel_R_air,
    fresnel_R_interface,
    penetration_depth,
    d99_simple,
    d99_corrected,
    tmm_r_t,
    tmm_field_in_film,
    tmm_intensity_profile,
    tmm_absorbed_density,
    tmm_energy_fractions,
    intensity_profile,
    absorbed_power_density,
    film_absorption_fraction,
    substrate_transmission_fraction,
    substrate_exit_fraction,
)
from ._materials import (
    SUBSTRATE_NK,
    SUBSTRATE_LABELS,
    LASER_MATCH_TOL_NM,
    lookup_substrate_nk,
)
from ._summary import bare_substrate_summary, film_stack_summary, suppression_scale_physics

__all__ = [
    # core physics
    "alpha_from_k", "k_from_alpha",
    "fresnel_R_air", "fresnel_R_interface",
    "penetration_depth", "d99_simple", "d99_corrected",
    "tmm_r_t", "tmm_field_in_film",
    "tmm_intensity_profile", "tmm_absorbed_density", "tmm_energy_fractions",
    "intensity_profile", "absorbed_power_density",
    "film_absorption_fraction", "substrate_transmission_fraction",
    "substrate_exit_fraction",
    # materials
    "SUBSTRATE_NK", "SUBSTRATE_LABELS", "LASER_MATCH_TOL_NM", "lookup_substrate_nk",
    # summary orchestrators
    "bare_substrate_summary", "film_stack_summary", "suppression_scale_physics",
]
