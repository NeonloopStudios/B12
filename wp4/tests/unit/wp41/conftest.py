# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Fixtures of the wp41 tests: minimal contract tables with free values.

The numbers are test data, not design values. The span is 11.7 m and the
chord is 3.1 m. The stations are uneven and include both ends.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from wp41.contracts import FloatArray, LiftTable, PointLoads, WeightTable

HALF_SPAN_M = 11.7
CHORD_M = 3.1
STATIONS_M = np.array([0.0, 0.9, 2.3, 4.1, 6.7, 8.8, 10.4, 11.7], dtype=np.float64)


def _full(value: float) -> FloatArray:
    return np.full(STATIONS_M.shape, value, dtype=np.float64)


@pytest.fixture
def stations() -> FloatArray:
    """Span stations [m]."""
    return STATIONS_M.copy()


@pytest.fixture
def make_lift() -> Callable[..., LiftTable]:
    """Factory of lift tables. The load per unit span may be a number or an array."""

    def build(
        n_prime: float | FloatArray = 0.0,
        m_prime: float | FloatArray = 0.0,
        n: float = 2.3,
        case: str = "LC-T",
    ) -> LiftTable:
        return LiftTable(
            case=case, n=n, w_n=61_234.5, q_pa=9_871.3, cl_total=0.713, alpha_d_deg=1.7,
            mach_case=0.71, mach_run=0.77, extrapolated=False,
            y_m=STATIONS_M.copy(), chord_m=_full(CHORD_M), cl_d=_full(0.61),
            cm_c4_d=_full(-0.11),
            n_prime_npm=_full(0.0) + np.asarray(n_prime, dtype=np.float64),
            m_prime_c4_nmpm=_full(0.0) + np.asarray(m_prime, dtype=np.float64),
        )

    return build


@pytest.fixture
def make_weight() -> Callable[..., WeightTable]:
    """Factory of weight tables."""

    def build(
        w_str: float = 0.0,
        w_fuel: float = 0.0,
        f_str: float = 0.47,
        f_fuel: float = 0.52,
        case: str = "LC-T",
    ) -> WeightTable:
        return WeightTable(
            case=case, y_m=STATIONS_M.copy(), chord_m=_full(CHORD_M),
            w_str_npm=_full(w_str), w_fuel_npm=_full(w_fuel),
            f_str=_full(f_str), f_fuel=_full(f_fuel),
        )

    return build


@pytest.fixture
def make_points() -> Callable[..., PointLoads]:
    """Factory of point loads: one engine per (y, W, d) triple."""

    def build(
        engines: tuple[tuple[float, float, float], ...] = (), case: str = "LC-T"
    ) -> PointLoads:
        return PointLoads(
            case=case,
            names=tuple(f"engine_{i}" for i in range(len(engines))),
            y_m=np.array([e[0] for e in engines], dtype=np.float64),
            w_n=np.array([e[1] for e in engines], dtype=np.float64),
            d_ahead_of_sc_m=np.array([e[2] for e in engines], dtype=np.float64),
        )

    return build
