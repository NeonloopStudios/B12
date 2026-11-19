# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Tests of wp41.integrate against closed-form integrals of linear loads."""

from __future__ import annotations

import numpy as np
import pytest

from wp41.integrate import integral_from_tip

HALF_SPAN = 11.7


def test_integral_from_tip_constant_load_returns_load_times_length(
    stations: np.ndarray,
) -> None:
    load = np.full(stations.shape, 37.3)
    knots = np.array([0.0, 4.9, 8.8, HALF_SPAN])

    result = integral_from_tip(stations, load, HALF_SPAN, knots)

    assert result.dtype == np.float64
    assert result.shape == knots.shape
    expected = 37.3 * (HALF_SPAN - knots)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_integral_from_tip_linear_load_returns_closed_form(
    stations: np.ndarray,
) -> None:
    a, b = 310.0, -17.3
    knots = np.array([0.0, 0.9, 5.0, 10.4])

    result = integral_from_tip(stations, a + b * stations, HALF_SPAN, knots)

    expected = a * (HALF_SPAN - knots) + b * (HALF_SPAN**2 - knots**2) / 2.0
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_integral_from_tip_load_constant_beyond_the_end_stations() -> None:
    y = np.array([0.26, 1.9, 5.3, 11.5])
    a, b = 41.0, 6.2
    knots = np.array([0.1, 11.0, HALF_SPAN])

    result = integral_from_tip(y, a + b * y, HALF_SPAN, knots)

    # Linear on [0.26, 11.5], constant (the end value) on [0, 0.26] and [11.5, b/2].
    lo, hi = y[0], y[-1]
    f_lo, f_hi = a + b * lo, a + b * hi

    def tail(x: float) -> float:
        if x >= hi:
            return f_hi * (HALF_SPAN - x)
        inner_from = max(x, lo)
        inner = a * (hi - inner_from) + b * (hi**2 - inner_from**2) / 2.0
        root_part = f_lo * (lo - x) if x < lo else 0.0
        return inner + root_part + f_hi * (HALF_SPAN - hi)

    np.testing.assert_allclose(result, [tail(k) for k in knots], rtol=1e-12, atol=1e-12)


def test_integral_from_tip_knot_at_tip_returns_zero(stations: np.ndarray) -> None:
    load = np.full(stations.shape, 5.3)

    result = integral_from_tip(stations, load, HALF_SPAN, np.array([HALF_SPAN]))

    assert result[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bad_knot", [-0.1, HALF_SPAN + 0.1, float("nan")])
def test_integral_from_tip_knot_outside_span_raises(
    stations: np.ndarray, bad_knot: float
) -> None:
    with pytest.raises(ValueError, match="knots must lie"):
        integral_from_tip(
            stations, np.ones(stations.shape), HALF_SPAN, np.array([bad_knot])
        )


def test_integral_from_tip_unordered_stations_raises() -> None:
    y = np.array([0.0, 3.0, 2.0, 5.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        integral_from_tip(y, np.ones(4), HALF_SPAN, np.array([1.0]))


def test_integral_from_tip_shape_mismatch_raises(stations: np.ndarray) -> None:
    with pytest.raises(ValueError, match="same 1-D shape"):
        integral_from_tip(stations, np.ones(3), HALF_SPAN, np.array([1.0]))


def test_integral_from_tip_station_beyond_tip_raises() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        integral_from_tip(np.array([0.0, 12.5]), np.ones(2), HALF_SPAN, np.array([1.0]))
