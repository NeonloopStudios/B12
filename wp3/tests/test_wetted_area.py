# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/wetted_area.py.

Spec: ADSEE-II slides, "Component wetted area". The body formula is checked
part by part against closed-form surface areas:

* cylinder: pi D L2
* cone (lateral): pi r s, with r = D/2 and slant s = sqrt(L3^2 + r^2)
* paraboloid of revolution, radius r, height h:
  A = pi r / (6 h^2) * [ (r^2 + 4 h^2)^1.5 - r^3 ]
  (surface of revolution of y = r sqrt(x/h); e.g. CRC Standard Mathematical
  Tables). The slide nose term equals this area minus the disk pi D^2 / 4.
"""

from __future__ import annotations

import math

import pytest

from b12wp3.drag.wetted_area import body_wetted_area, tail_wetted_area, wing_wetted_area


def _paraboloid_area(radius: float, height: float) -> float:
    return (
        math.pi
        * radius
        / (6.0 * height**2)
        * ((radius**2 + 4.0 * height**2) ** 1.5 - radius**3)
    )


def test_wing_wetted_area_unit_exposed_returns_2_14() -> None:
    wetted = wing_wetted_area(1.0)

    assert wetted == pytest.approx(2.14)


def test_tail_wetted_area_unit_exposed_returns_2_10() -> None:
    wetted = tail_wetted_area(1.0)

    assert wetted == pytest.approx(2.10)


@pytest.mark.parametrize("function", [wing_wetted_area, tail_wetted_area])
@pytest.mark.parametrize("exposed", [0.0, -1.0])
def test_surface_wetted_area_non_positive_exposed_raises(
    function: object, exposed: float
) -> None:
    assert callable(function)

    with pytest.raises(ValueError):
        function(exposed)


def test_paraboloid_reference_area_matches_hand_value() -> None:
    area = _paraboloid_area(1.0, 2.0)

    assert area == pytest.approx(9.04422579935285)


def test_body_nose_and_flat_tail_returns_paraboloid_area() -> None:
    # nose term = paraboloid - pi D^2/4; with L3 = 0 the cone term is the flat
    # tail disk + pi D^2/4, so the two disk terms cancel
    d, l1 = 2.0, 2.0

    wetted = body_wetted_area(d, l1, 0.0, 0.0)

    assert wetted == pytest.approx(_paraboloid_area(d / 2.0, l1))


def test_body_cylinder_term_returns_pi_d_l2() -> None:
    d, l1, l2 = 3.0, 4.0, 20.0

    added = body_wetted_area(d, l1, l2, 0.0) - body_wetted_area(d, l1, 0.0, 0.0)

    assert added == pytest.approx(math.pi * d * l2)


def test_body_cone_term_returns_cone_lateral_area_minus_tail_disk() -> None:
    d, l1, l3 = 3.0, 4.0, 8.0
    r = d / 2.0

    added = body_wetted_area(d, l1, 0.0, l3) - body_wetted_area(d, l1, 0.0, 0.0)

    assert added == pytest.approx(
        math.pi * r * math.sqrt(l3**2 + r**2) - math.pi * r**2
    )


def test_body_short_nose_nose_term_goes_to_zero() -> None:
    # L1 -> 0: nose term -> 0, so only the cylinder and the cone remain
    d, l2, l3 = 3.0, 20.0, 8.0
    r = d / 2.0

    wetted = body_wetted_area(d, 1e-4, l2, l3)

    expected = math.pi * d * l2 + math.pi * r * math.sqrt(l3**2 + r**2)
    assert wetted == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("d", "l1", "l2", "l3"),
    [
        (0.0, 4.0, 20.0, 8.0),
        (3.0, 0.0, 20.0, 8.0),
        (3.0, 4.0, -1.0, 8.0),
        (3.0, 4.0, 20.0, -1.0),
    ],
)
def test_body_wetted_area_invalid_input_raises(
    d: float, l1: float, l2: float, l3: float
) -> None:
    with pytest.raises(ValueError):
        body_wetted_area(d, l1, l2, l3)
