# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""Aircraft geometry for WP4.

The wing is the WP3 redesign (wp2/src/b12wp2/config/planform.py and
wp2/src/b12wp2/config/wing.py). The quantities that follow from the planform
(span, chords, MAC, leading-edge sweep) are computed in
wp41.aero_loads.planform(), not here.

Axes (VSPAERO frame): x aft, y to the right wing, z up. Lengths in metres.
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
    dihedral_deg=2.0,  # wp2 config/wing.py
    tip_twist_deg=-2.0,
    twist_axis_chord_fraction=0.25,
    incidence_deg=2.0,
    root_le_x_m=0.0,
    root_z_m=0.3,
    fuselage_diameter_m=3.01,  # WP3 fuselage sizing, wp3/configs/geometry.yaml
    airfoil_root=AIRFOILS_DIR / "NASA_SC(2)-0712.dat",
    airfoil_tip=AIRFOILS_DIR / "NASA_SC(2)-0712.dat",
)
