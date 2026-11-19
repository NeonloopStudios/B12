# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""Tests of wp41.stations: the global spanwise grid of WP4.1."""

from __future__ import annotations

import math

import numpy as np
import pytest

from configs.wp41 import settings
from wp41.stations import SPAN_STATIONS, station_y


def test_span_stations_root_to_tip_has_configured_count() -> None:
    assert SPAN_STATIONS.shape == (settings.N_SPAN_STATIONS,)
    assert SPAN_STATIONS.dtype == np.float64
    assert (SPAN_STATIONS[0], SPAN_STATIONS[-1]) == (0.0, 1.0)
    assert np.all(np.diff(SPAN_STATIONS) > 0.0)


def test_station_y_scales_eta_by_half_span() -> None:
    y = station_y(12.2001)

    assert y.shape == SPAN_STATIONS.shape
    assert (y[0], y[-1]) == pytest.approx((0.0, 12.2001))
    np.testing.assert_allclose(y / 12.2001, SPAN_STATIONS)


@pytest.mark.parametrize("half_span", [0.0, -3.1, math.nan])
def test_station_y_half_span_not_positive_raises(half_span: float) -> None:
    with pytest.raises(ValueError, match="half_span_m"):
        station_y(half_span)
