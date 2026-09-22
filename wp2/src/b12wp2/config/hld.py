# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-22
"""High-lift device data and the layout they are sized in: the ADSEE device
table, the span and chord fractions available on this wing, and the required
CL_max values to check against.

The Device record lives here rather than in b12wp2.hld.sizing because the
table is the data and sizing.py is the method that reads it -- putting the
table in sizing.py and the constants here would have the two importing each
other.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    """One high-lift device type.

    dcl_max_base is the ADSEE airfoil increment; for slotted-extending
    devices (extends=True) it is multiplied by c'/c. complexity is a
    relative mechanism-complexity rank (0 = no device), an engineering
    judgment for the trade-off, not a computed quantity.
    """

    name: str
    edge: str  # "TE" or "LE"
    dcl_max_base: float
    extends: bool
    chord_extension: float  # c'/c of the deployed device (landing)
    complexity: int

    @property
    def dcl_max(self) -> float:
        return self.dcl_max_base * (self.chord_extension if self.extends else 1.0)


# chord extension c'/c: default values for the landing setting
TE_DEVICES: tuple[Device, ...] = (
    Device("plain", "TE", 0.9, False, 1.00, 1),
    Device("split", "TE", 0.9, False, 1.00, 1),
    Device("single-slotted", "TE", 1.3, False, 1.00, 2),
    Device("Fowler", "TE", 1.3, True, 1.30, 3),
    Device("double-slotted", "TE", 1.6, True, 1.20, 4),
    Device("triple-slotted", "TE", 1.9, True, 1.25, 5),
)

LE_DEVICES: tuple[Device, ...] = (
    Device("none", "LE", 0.0, False, 1.00, 0),
    Device("LE flap", "LE", 0.3, False, 1.00, 1),
    Device("Krueger", "LE", 0.3, False, 1.00, 2),
    Device("slat", "LE", 0.4, True, 1.10, 2),
)

# Airfoil zero-lift angle shift of a trailing-edge flap [deg]
DALPHA_0L_AIRFOIL_LANDING = -15.0
DALPHA_0L_AIRFOIL_TAKEOFF = -10.0

# Take-off setting as a fraction of the landing-setting dCL_max and c'/c - 1
TAKEOFF_FRACTION = 0.6

# --- spanwise limits (fraction of the semi-span) ---

ETA_IN = 0.12  # fuselage side + clearance
ETA_OUT_TE = 0.75  # trailing-edge devices end where the aileron starts
ETA_OUT_LE = 0.95  # leading-edge devices run in front of the aileron, short of the tip

# --- chordwise size, from the spar positions ---

FRONT_SPAR = 0.15
REAR_SPAR = 0.65
FLAP_CHORD_RATIO = 1.0 - REAR_SPAR  # c_f / c
SLAT_CHORD_RATIO = FRONT_SPAR  # c_s / c

# --- requirements from WP1 (None = not yet available, checks skipped) ---

CL_MAX_L_REQ: float | None = None
CL_MAX_TO_REQ: float | None = None
