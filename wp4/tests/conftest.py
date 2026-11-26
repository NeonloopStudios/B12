# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""Shared fixtures for the wp4 tests.

The rectangular wing has simple closed-form values (b = 10 m, c = 1 m,
MAC = 1 m, y_MAC = 2.5 m). Its numbers are test data, not design values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.aircraft import WingGeometry


@pytest.fixture
def rectangular_wing() -> WingGeometry:
    """Unswept, untapered, flat wing: S = 10 m^2, AR = 10, root LE at (1, 0.5)."""
    return WingGeometry(
        area_m2=10.0,
        aspect_ratio=10.0,
        taper=1.0,
        root_chord=1.0,
        sweep_c4_deg=0.0,
        dihedral_deg=0.0,
        tip_twist_deg=0.0,
        twist_axis_chord_fraction=0.25,
        incidence_deg=0.0,
        root_le_x_m=1.0,
        root_z_m=0.5,
        fuselage_diameter_m=0.0,
        airfoil_root=Path("unused.dat"),
        airfoil_tip=Path("unused.dat"),
    )
