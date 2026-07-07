# -*- coding: utf-8 -*-
"""backend.background — substrate PL background suppression."""

from ._suppressor import BackgroundSuppressor
from ._scale import interp_reference
from ._presets import (
    Preset,
    find_preset,
    format_temperature_label,
    list_materials,
    list_presets,
    list_temperatures,
    list_thicknesses,
    load_preset,
)

__all__ = [
    "BackgroundSuppressor",
    "interp_reference",
    "Preset",
    "find_preset",
    "format_temperature_label",
    "list_materials",
    "list_presets",
    "list_temperatures",
    "list_thicknesses",
    "load_preset",
]
