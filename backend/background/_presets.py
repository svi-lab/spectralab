"""Built-in substrate background presets loaded from data/*.npz."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent / "data"
_INTENSITY_KEYS = ("intensity", "data", "mean")
_THICKNESS_RE = re.compile(r"^(\d+)mm$", re.IGNORECASE)


@dataclass(frozen=True)
class Preset:
    """One built-in substrate reference preset."""

    key: str
    material: str
    thickness_mm: int | None
    temperature: str
    path: Path


def _parse_preset_path(path: Path) -> Preset | None:
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 2:
        return None

    material = parts[0]
    temperature = parts[-1]
    thickness_mm: int | None = None

    if material == "glass":
        if len(parts) != 3:
            return None
        match = _THICKNESS_RE.match(parts[1])
        if not match:
            return None
        thickness_mm = int(match.group(1))
    elif len(parts) != 2:
        return None

    return Preset(
        key=stem,
        material=material,
        thickness_mm=thickness_mm,
        temperature=temperature,
        path=path,
    )


@lru_cache(maxsize=1)
def list_presets() -> tuple[Preset, ...]:
    """Return all parseable presets sorted by material, thickness, temperature."""
    presets: list[Preset] = []
    for path in sorted(_DATA_DIR.glob("*.npz")):
        preset = _parse_preset_path(path)
        if preset is not None:
            presets.append(preset)
    return tuple(
        sorted(
            presets,
            key=lambda p: (
                p.material,
                p.thickness_mm if p.thickness_mm is not None else -1,
                p.temperature,
            ),
        )
    )


def list_materials() -> list[str]:
    return sorted({p.material for p in list_presets()})


def list_thicknesses(material: str) -> list[int]:
    return sorted(
        {
            p.thickness_mm
            for p in list_presets()
            if p.material == material and p.thickness_mm is not None
        }
    )


def list_temperatures(material: str, thickness_mm: int | None = None) -> list[str]:
    temps: list[str] = []
    for p in list_presets():
        if p.material != material:
            continue
        if thickness_mm is not None and p.thickness_mm != thickness_mm:
            continue
        if thickness_mm is None and p.thickness_mm is not None:
            continue
        temps.append(p.temperature)
    return sorted(set(temps))


def find_preset(
    material: str,
    temperature: str,
    thickness_mm: int | None = None,
) -> Preset | None:
    for preset in list_presets():
        if preset.material != material:
            continue
        if preset.temperature != temperature:
            continue
        if thickness_mm is not None:
            if preset.thickness_mm == thickness_mm:
                return preset
        elif preset.thickness_mm is None:
            return preset
    return None


def load_preset(key: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Load preset spectra and acquisition metadata.

    Returns (spectral_coords, intensity, meta) where meta has laser_power and
    exposure_time (NaN when absent from the file).
    """
    path = _DATA_DIR / f"{key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Preset not found: {key}")

    with np.load(path, allow_pickle=False) as d:
        x = np.asarray(d["spectral_coords"], dtype=float)
        y = None
        for intensity_key in _INTENSITY_KEYS:
            if intensity_key in d:
                y = np.asarray(d[intensity_key], dtype=float)
                break
        if y is None:
            raise KeyError(f"Preset {key} has no intensity array")

        laser_power = float(d["laser_power"]) if "laser_power" in d else float("nan")
        exposure_time = float(d["exposure_time"]) if "exposure_time" in d else float("nan")

    meta = {"laser_power": laser_power, "exposure_time": exposure_time}
    return x, y, meta


def format_temperature_label(temp: str) -> str:
    if temp.lower() == "rt":
        return "Room temperature"
    if temp.endswith("K"):
        return temp
    return temp


__all__ = [
    "Preset",
    "find_preset",
    "format_temperature_label",
    "list_materials",
    "list_presets",
    "list_temperatures",
    "list_thicknesses",
    "load_preset",
]
