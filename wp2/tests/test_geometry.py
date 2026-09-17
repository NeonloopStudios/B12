# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Tests for wp2/scripts/geometry.py.

max_thickness_to_chord is validated against XFoil's own internally-computed
"Max thickness" (printed at airfoil-load time with xf.print=True, not
otherwise exposed by the Python bindings) for every current candidate
airfoil -- not a synthetic fixture, the real numbers mcrit_sweep.py's Korn
equation actually uses.
"""
from __future__ import annotations

import pytest

from scripts import config, geometry, xfoil_runtime

# XFoil's own reported "Max thickness" for each candidate, captured directly
# (xf.print = True, xf.airfoil = <loaded airfoil>) -- not computed by this
# codebase, this is XFoil's number to check against.
_XFOIL_REPORTED_MAX_THICKNESS = {
    "NACA_25112": 0.120059,
    "NACA_64212": 0.119793,
    "lockheed_c5a_bl758": 0.110390,
    "NASA_SC(2)-0712": 0.119921,
}


@pytest.mark.parametrize(
    "airfoil_stem,xfoil_value",
    list(_XFOIL_REPORTED_MAX_THICKNESS.items()),
)
def test_max_thickness_matches_xfoil_own_computation(
    airfoil_stem: str, xfoil_value: float
) -> None:
    airfoil = xfoil_runtime.load_airfoil_dat(config.AIRFOILS_DIR / f"{airfoil_stem}.dat")
    computed = geometry.max_thickness_to_chord(airfoil)
    assert computed == pytest.approx(xfoil_value, rel=0.001)


def test_max_thickness_all_current_candidates_covered() -> None:
    # if the airfoil set changes again, this test (and the reference table
    # above) needs updating too -- catch that immediately rather than
    # silently skip the new airfoil
    found = {p.stem for p in config.discover_airfoils()}
    assert found == set(_XFOIL_REPORTED_MAX_THICKNESS)
