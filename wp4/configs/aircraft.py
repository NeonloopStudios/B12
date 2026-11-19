# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""Aircraft geometry and masses for WP4 (WP3 wing redesign).

Axes: x aft, y right, z up [m]. Masses, including the engines, from
wp3/scripts/weight_estimator.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from configs.paths import AIRFOILS_DIR


@dataclass(frozen=True)
class WingGeometry:
    """Geometric inputs of a straight-tapered wing.

    Attributes:
        area_m2: Planform area of both halves, S [m^2].
        aspect_ratio: b^2 / S [-].
        taper: c_tip / c_root [-].
        sweep_c4_deg: Quarter-chord sweep [deg].
        dihedral_deg: Dihedral angle [deg].
        tip_twist_deg: Tip twist relative to the root, negative is wash-out [deg].
        twist_axis_chord_fraction: Chord fraction of the twist axis [-].
        incidence_deg: Wing incidence, nose up positive, about the root LE [deg].
        root_le_x_m: x of the root leading edge [m].
        root_z_m: z of the root leading edge [m].
        fuselage_diameter_m: Span length inside the fuselage [m].
        airfoil_root: Airfoil .dat file at the root.
        airfoil_tip: Airfoil .dat file at the tip.
    """

    area_m2: float
    aspect_ratio: float
    taper: float
    sweep_c4_deg: float
    dihedral_deg: float
    tip_twist_deg: float
    twist_axis_chord_fraction: float
    incidence_deg: float
    root_le_x_m: float
    root_z_m: float
    fuselage_diameter_m: float
    airfoil_root: Path
    airfoil_tip: Path

    def __post_init__(self) -> None:
        if self.area_m2 <= 0.0:
            raise ValueError(f"area_m2 must be > 0, got {self.area_m2}")
        if self.aspect_ratio <= 0.0:
            raise ValueError(f"aspect_ratio must be > 0, got {self.aspect_ratio}")
        if not 0.0 < self.taper <= 1.0:
            raise ValueError(f"taper must be in (0, 1], got {self.taper}")
        if not -60.0 < self.sweep_c4_deg < 60.0:
            raise ValueError(f"sweep_c4_deg must be in (-60, 60), got {self.sweep_c4_deg}")
        if not -15.0 < self.dihedral_deg < 15.0:
            raise ValueError(f"dihedral_deg must be in (-15, 15), got {self.dihedral_deg}")
        if not 0.0 <= self.twist_axis_chord_fraction <= 1.0:
            raise ValueError(
                "twist_axis_chord_fraction must be in [0, 1], "
                f"got {self.twist_axis_chord_fraction}"
            )
        if self.fuselage_diameter_m < 0.0:
            raise ValueError(
                f"fuselage_diameter_m must be >= 0, got {self.fuselage_diameter_m}"
            )


WING: Final[WingGeometry] = WingGeometry(
    area_m2=62.67,
    aspect_ratio=9.5,
    taper=0.4,
    sweep_c4_deg=30.0,
    dihedral_deg=2.0,
    tip_twist_deg=-2.0,
    twist_axis_chord_fraction=0.25,
    incidence_deg=2.0,
    root_le_x_m=0.0,
    root_z_m=0.3,
    fuselage_diameter_m=3.01,
    airfoil_root=AIRFOILS_DIR / "NASA_SC(2)-0712.dat",
    airfoil_tip=AIRFOILS_DIR / "NASA_SC(2)-0712.dat",
)


@dataclass(frozen=True)
class AircraftMasses:
    """Aircraft masses [kg].

    Attributes:
        oem_kg: Operating empty mass.
        design_payload_kg: Design payload mass.
        max_payload_kg: Maximum payload mass.
        fuel_kg: Fuel mass at MTOM with the design payload.
        engine_count: Number of engines, one per wing half for a symmetric pair.
        engine_dry_kg: Dry mass of one engine (datasheet).
        engine_installed_kg: Installed mass of one engine: engine, engine
            contents and nacelle group. Part of OEM. Use this value for the
            engine point load on the wing.
    """

    oem_kg: float
    design_payload_kg: float
    max_payload_kg: float
    fuel_kg: float
    engine_count: int
    engine_dry_kg: float
    engine_installed_kg: float

    def __post_init__(self) -> None:
        for name in ("oem_kg", "design_payload_kg", "max_payload_kg", "fuel_kg",
                     "engine_dry_kg"):
            value = getattr(self, name)
            if not value > 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if not self.max_payload_kg >= self.design_payload_kg:
            raise ValueError(
                f"max_payload_kg ({self.max_payload_kg}) must be >= "
                f"design_payload_kg ({self.design_payload_kg})"
            )
        if not (self.engine_count >= 1 and self.engine_count % 2 == 0):
            raise ValueError(
                f"engine_count must be a positive even number, got {self.engine_count}"
            )
        if not self.engine_installed_kg >= self.engine_dry_kg:
            raise ValueError(
                f"engine_installed_kg ({self.engine_installed_kg}) must be >= "
                f"engine_dry_kg ({self.engine_dry_kg})"
            )
        if not self.engine_count * self.engine_installed_kg < self.oem_kg:
            raise ValueError(
                f"{self.engine_count} engines of {self.engine_installed_kg} kg "
                f"exceed the OEM ({self.oem_kg} kg)"
            )

    @property
    def mtom_kg(self) -> float:
        """Maximum take-off mass: OEM + design payload + fuel."""
        return self.oem_kg + self.design_payload_kg + self.fuel_kg

    @property
    def mzfm_kg(self) -> float:
        """Maximum zero-fuel mass: OEM + maximum payload."""
        return self.oem_kg + self.max_payload_kg


MASSES: Final[AircraftMasses] = AircraftMasses(
    oem_kg=17_930.0,
    design_payload_kg=7_200.0,
    max_payload_kg=9_302.0,
    fuel_kg=7_792.0,
    engine_count=2,
    engine_dry_kg=1_088.0,  # GE CF34-8C5B1, wp3/scripts/weight_estimator.py
    # wp3/scripts/weight_estimator.py: "engines and nacelles" 3866.76 kg for 2
    engine_installed_kg=1_933.38,
)
