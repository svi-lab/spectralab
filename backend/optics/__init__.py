"""backend.optics — thin-film optical calculations (TMM, Beer-Lambert, materials)."""

from ._core import (
    absorbed_power_density,
    alpha_from_k,
    d99_corrected,
    d99_simple,
    film_absorption_fraction,
    fresnel_R_air,
    fresnel_R_interface,
    intensity_profile,
    k_from_alpha,
    penetration_depth,
    substrate_exit_fraction,
    substrate_transmission_fraction,
    tmm_absorbed_density,
    tmm_energy_fractions,
    tmm_field_in_film,
    tmm_intensity_profile,
    tmm_r_t,
)
from ._materials import (
    LASER_MATCH_TOL_NM,
    SUBSTRATE_LABELS,
    SUBSTRATE_NK,
    lookup_substrate_nk,
)
from ._summary import bare_substrate_summary, film_stack_summary, suppression_scale_physics

__all__ = [
    # core physics
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
    # materials
    "SUBSTRATE_NK",
    "SUBSTRATE_LABELS",
    "LASER_MATCH_TOL_NM",
    "lookup_substrate_nk",
    # summary orchestrators
    "bare_substrate_summary",
    "film_stack_summary",
    "suppression_scale_physics",
]
