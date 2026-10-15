# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/common/planform.py.

The WP2 wing reference values are from wp2/results/aero_summary_3d.csv,
which wp2 computes with its own planform module.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from b12wp3.common.planform import TrapezoidalPlanform


def _wp2_wing() -> TrapezoidalPlanform:
    return TrapezoidalPlanform(66.7, 9.5, 0.4, math.radians(30.0), panels=2)


def test_planform_wp2_wing_matches_wp2_summary() -> None:
    wing = _wp2_wing()

    result = (
        wing.span_m,
        wing.root_chord_m,
        wing.tip_chord_m,
        wing.mac_m,
        math.degrees(wing.sweep_at(0.0)),
        math.degrees(wing.sweep_at(0.5)),
    )

    assert result == pytest.approx(
        (
            25.172405527,
            3.785324139,
            1.514129656,
            2.811955075,
            31.900738012,
            28.023581805,
        )
    )


def test_sweep_at_quarter_chord_returns_input_sweep() -> None:
    wing = _wp2_wing()

    sweep = wing.sweep_at(0.25)

    assert sweep == pytest.approx(math.radians(30.0))


@pytest.mark.parametrize("panels", [1, 2])
def test_exposed_area_zero_covered_span_returns_full_area(panels: int) -> None:
    surface = TrapezoidalPlanform(20.0, 4.0, 0.6, 0.3, panels=panels)

    exposed = surface.exposed_area(0.0)

    assert exposed == pytest.approx(20.0)


def test_exposed_area_rectangular_wing_removes_covered_strip() -> None:
    # taper 1: chord c = S / b everywhere, so S_exp = S - c * covered span
    wing = TrapezoidalPlanform(40.0, 10.0, 1.0, 0.0, panels=2)
    chord = 40.0 / 20.0

    exposed = wing.exposed_area(4.0)

    assert exposed == pytest.approx(40.0 - chord * 4.0)


def test_one_panel_area_equals_trapezoid_area() -> None:
    fin = TrapezoidalPlanform(12.0, 1.5, 0.4, 0.6, panels=1)

    area = fin.panel_span_m * (fin.root_chord_m + fin.tip_chord_m) / 2.0

    assert area == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("area", "aspect_ratio", "taper", "sweep", "panels"),
    [
        (0.0, 9.0, 0.4, 0.0, 2),
        (60.0, -1.0, 0.4, 0.0, 2),
        (60.0, 9.0, 0.0, 0.0, 2),
        (60.0, 9.0, 1.2, 0.0, 2),
        (60.0, 9.0, 0.4, math.pi / 2, 2),
        (60.0, 9.0, 0.4, 0.0, 3),
    ],
)
def test_planform_invalid_input_raises(
    area: float, aspect_ratio: float, taper: float, sweep: float, panels: int
) -> None:
    with pytest.raises(ValueError):
        TrapezoidalPlanform(area, aspect_ratio, taper, sweep, panels)


@pytest.mark.parametrize("covered", [-0.1, 30.0])
def test_exposed_area_covered_span_out_of_range_raises(covered: float) -> None:
    wing = _wp2_wing()

    with pytest.raises(ValueError):
        wing.exposed_area(covered)


@given(
    covered_a=st.floats(min_value=0.0, max_value=20.0),
    covered_b=st.floats(min_value=0.0, max_value=20.0),
)
def test_exposed_area_decreases_with_covered_span(
    covered_a: float, covered_b: float
) -> None:
    wing = _wp2_wing()
    small, large = sorted((covered_a, covered_b))

    exposed_small = wing.exposed_area(small)
    exposed_large = wing.exposed_area(large)

    tolerance = 1e-9 * wing.area_m2
    assert exposed_large > 0.0
    assert exposed_large <= exposed_small + tolerance
    assert exposed_small <= wing.area_m2 + tolerance
