# -*- coding: utf-8 -*-
"""Literature defect-band position presets for PL deconvolution (ZnO:Al, TiO2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BandPreset:
    """One literature band position, given in both native units the page supports."""

    label: str
    wavelength_nm: float
    energy_ev: float
    assignment: str
    tentative: bool = False


_ZNO_AL: tuple[BandPreset, ...] = (
    BandPreset("376 nm (3.30 eV)", 376, 3.30, "CBM → VBM (NBE)"),
    BandPreset(
        "390 nm (3.18 eV)", 390, 3.18,
        "Al_Zn/(Al_Zn–Zn_i) → V_O²⁺",
        tentative=True,
    ),
    BandPreset("395 nm (3.14 eV)", 395, 3.14, "Transition involving Zn_i / ex-Zn_i states"),
    BandPreset("405 nm (3.06 eV)", 405, 3.06, "CB → V_Zn"),
    BandPreset("430 nm (2.88 eV)", 430, 2.88, "CB → V_O⁺"),
    BandPreset("437 nm (2.84 eV)", 437, 2.84, "CB → V_Zn"),
    BandPreset("440 nm (2.82 eV)", 440, 2.82, "ex-Zn_i → V_O"),
    BandPreset("500 nm (2.48 eV)", 500, 2.48, "CB → V_O"),
    BandPreset("504 nm (2.46 eV)", 504, 2.46, "CB → O_i"),
    BandPreset("544 nm (2.28 eV)", 544, 2.28, "CB → V_O²⁺"),
    BandPreset("554 nm (2.24 eV)", 554, 2.24, "Al_Zn/(Al_Zn–Zn_i) → V_O⁺"),
    BandPreset("564 nm (2.20 eV)", 564, 2.20, "CB → V_O²⁺"),
    BandPreset("602 nm (2.06 eV)", 602, 2.06, "Zn_i → O_i"),
)

_TIO2: tuple[BandPreset, ...] = (
    BandPreset(
        "380 nm (3.26 eV)", 380, 3.26,
        "NBE: free/self-trapped exciton recombination (intrinsic band-edge transition)",
    ),
    BandPreset(
        "375 nm (3.31 eV)", 375, 3.31,
        "Sharp excitonic line: defect-trapped exciton (shallow trap, X-ray-sensitive intensity)",
        tentative=True,
    ),
    BandPreset(
        "368 nm (3.37 eV)", 368, 3.37,
        "Sharp excitonic line: defect-trapped exciton (shallow trap, X-ray-sensitive intensity)",
        tentative=True,
    ),
    BandPreset(
        "426 nm (2.91 eV)", 426, 2.91,
        "Excitonic band: self-trapped exciton (anatase), TiO6 octahedron self-trapping",
    ),
    BandPreset(
        "449 nm (2.76 eV)", 449, 2.76,
        "Excitonic + phonon replicas: exciton recombination mediated by oxygen vacancies (V_O) "
        "(literature range 441–457 nm / 2.71–2.81 eV)",
    ),
    BandPreset(
        "510 nm (2.44 eV)", 510, 2.44,
        "Green luminescence (G-PL): free e⁻ recombining with holes trapped at surface V_O, "
        "(101)-facet related (literature range 500–520 nm / 2.40–2.48 eV)",
    ),
    BandPreset(
        "610 nm (2.05 eV)", 610, 2.05,
        "Red luminescence (R-PL): e⁻ trapped at under-coordinated Ti³⁺ sites recombining "
        "with VB holes (literature range 600–620 nm / 2.00–2.10 eV)",
    ),
)

PRESETS: dict[str, tuple[BandPreset, ...]] = {
    "ZnO:Al": _ZNO_AL,
    "TiO2": _TIO2,
}


def list_preset_materials() -> list[str]:
    return list(PRESETS.keys())


def get_preset_bands(material: str) -> tuple[BandPreset, ...]:
    return PRESETS.get(material, ())


__all__ = [
    "BandPreset",
    "get_preset_bands",
    "list_preset_materials",
]
