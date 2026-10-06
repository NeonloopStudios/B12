# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""Tests for wp2/src/b12wp2/common/drag_polar.py.

Checked against structural properties of the ADSEE-II formulas (limits,
monotonicity, the definition of K), plus one hand-computed value for the
B12 wing.
"""
from __future__ import annotations

import math

import pytest

from b12wp2.common import drag_polar


def test_oswald_limits_of_unswept_wing() -> None:
    # Lambda = 0: e = 2 / (2 - AR + sqrt(4 + AR^2)) -> 1/2 for AR -> 0, -> 1 for AR -> inf
    assert drag_polar.oswald_factor(1e-6, 0.0) == pytest.approx(0.5, abs=1e-6)
    assert drag_polar.oswald_factor(1e4, 0.0) == pytest.approx(1.0, abs=1e-3)


def test_oswald_decreases_with_aspect_ratio_for_swept_wing() -> None:
    values = [drag_polar.oswald_factor(ar, math.radians(20.0)) for ar in (4.0, 7.0, 9.5, 12.0)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_oswald_decreases_with_sweep() -> None:
    values = [drag_polar.oswald_factor(9.5, math.radians(s)) for s in (0.0, 15.0, 25.0, 35.0)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_oswald_b12_wing() -> None:
    # AR = 9.5, Lambda_0.5c = 21.83 deg (Lambda_0.25c = 24.02 deg, taper 0.4),
    # hand calculation: e = 2 / (2 - 9.5 + sqrt(4 + 90.25 * 1.1604)) = 0.683
    assert drag_polar.oswald_factor(9.5, math.radians(21.83)) == pytest.approx(0.683, abs=1e-3)


def test_oswald_rejects_non_positive_aspect_ratio() -> None:
    with pytest.raises(ValueError):
        drag_polar.oswald_factor(0.0, 0.0)


def test_induced_drag_factor_definition() -> None:
    e, ar = 0.683, 9.5
    assert drag_polar.induced_drag_factor(e, ar) * math.pi * e * ar == pytest.approx(1.0)


def test_induced_drag_factor_rejects_non_positive_input() -> None:
    with pytest.raises(ValueError):
        drag_polar.induced_drag_factor(0.0, 9.5)


def test_winglet_without_height_leaves_aspect_ratio_unchanged() -> None:
    assert drag_polar.winglet_aspect_ratio(9.5, 0.0, 25.2) == pytest.approx(9.5)


def test_winglet_increases_aspect_ratio() -> None:
    # h/b = 0.05 -> AR_e = AR * 1.095
    assert drag_polar.winglet_aspect_ratio(9.5, 1.26, 25.2) == pytest.approx(9.5 * 1.095)
