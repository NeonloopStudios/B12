# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/form_factor.py.

Spec: ADSEE-II slide 62, "Component form factor". Reference values are
evaluated by hand from the slide equations.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from b12wp3.drag.form_factor import (
    fineness_ratio,
    fuselage_form_factor,
    lifting_surface_form_factor,
    nacelle_form_factor,
)


def test_lifting_surface_ff_unswept_returns_hand_value() -> None:
    # [1 + 0.6/0.3 * 0.12 + 100 * 0.12^4] * [1.34 * 0.5^0.18] = 1.260736 * 1.18283
    ff = lifting_surface_form_factor(0.12, 0.3, 0.5, 0.0)

    assert ff == pytest.approx(1.4912262959402034)


def test_lifting_surface_ff_sweep_scales_by_cos_power() -> None:
    sweep = math.radians(30.0)

    ratio = lifting_surface_form_factor(0.12, 0.3, 0.5, sweep) / (
        lifting_surface_form_factor(0.12, 0.3, 0.5, 0.0)
    )

    assert ratio == pytest.approx(math.cos(sweep) ** 0.28)


def test_fuselage_ff_fineness_10_returns_1_085() -> None:
    # 1 + 60/1000 + 10/400
    ff = fuselage_form_factor(10.0)

    assert ff == pytest.approx(1.085)


def test_nacelle_ff_fineness_3_5_returns_1_1() -> None:
    ff = nacelle_form_factor(3.5)

    assert ff == pytest.approx(1.1)


def test_fineness_ratio_circular_section_returns_length_over_diameter() -> None:
    length, diameter = 30.0, 3.0

    f = fineness_ratio(length, math.pi * diameter**2 / 4.0)

    assert f == pytest.approx(10.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: lifting_surface_form_factor(0.0, 0.3, 0.5, 0.0),
        lambda: lifting_surface_form_factor(0.12, 1.0, 0.5, 0.0),
        lambda: lifting_surface_form_factor(0.12, 0.3, 0.0, 0.0),
        lambda: lifting_surface_form_factor(0.12, 0.3, 0.5, math.pi / 2),
        lambda: fuselage_form_factor(0.0),
        lambda: nacelle_form_factor(-1.0),
        lambda: fineness_ratio(0.0, 1.0),
        lambda: fineness_ratio(1.0, 0.0),
    ],
)
def test_form_factor_invalid_input_raises(call: object) -> None:
    assert callable(call)

    with pytest.raises(ValueError):
        call()


@given(
    tc_a=st.floats(min_value=0.01, max_value=0.3),
    tc_b=st.floats(min_value=0.01, max_value=0.3),
)
def test_lifting_surface_ff_increases_with_thickness(tc_a: float, tc_b: float) -> None:
    thin, thick = sorted((tc_a, tc_b))

    ff_thin = lifting_surface_form_factor(thin, 0.35, 0.77, 0.4)
    ff_thick = lifting_surface_form_factor(thick, 0.35, 0.77, 0.4)

    assert ff_thin <= ff_thick
