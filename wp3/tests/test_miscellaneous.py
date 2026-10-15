# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/miscellaneous.py.

Spec: ADSEE-II slides "Miscellaneous drag" (wave drag, fuselage upsweep and
base area, landing gear, flap, excrescence and leakage). Reference values are
evaluated by hand from the slide equations. The M_DD reference is the WP2
cruise value (wp2/results/aero_summary_3d.csv, M_dd_design = 0.85116), which
WP2 computes with the same Korn equation and inputs.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from b12wp3.drag.miscellaneous import (
    base_drag_area,
    drag_divergence_mach,
    excrescence_drag,
    flap_drag,
    landing_gear_drag,
    streamwise_critical_mach,
    upsweep_drag_area,
    wave_drag,
)

WP2_HALF_CHORD_SWEEP_RAD = math.radians(28.023581804818225)


def test_drag_divergence_mach_wp2_cruise_returns_wp2_value() -> None:
    m_dd = drag_divergence_mach(0.95, 0.1199, 0.489433403, WP2_HALF_CHORD_SWEEP_RAD)

    assert m_dd == pytest.approx(0.8511624028028268)


def test_drag_divergence_mach_unswept_returns_korn_sum() -> None:
    # M_DD + t/c + C_L/10 = k_a at zero sweep
    m_dd = drag_divergence_mach(0.95, 0.12, 0.5, 0.0)

    assert m_dd == pytest.approx(0.95 - 0.12 - 0.05)


def test_streamwise_critical_mach_wp2_section_returns_hand_value() -> None:
    # 0.56 / cos(28.0236 deg)
    m_cr = streamwise_critical_mach(0.56, WP2_HALF_CHORD_SWEEP_RAD)

    assert m_cr == pytest.approx(0.6343781100792403)


@pytest.mark.parametrize(
    ("mach", "expected"),
    [
        (0.50, 0.0),
        (0.80, 0.002 / 3.5),
        (0.85, 0.002),
        (0.90, 0.002 * 2.0**2.5),
    ],
)
def test_wave_drag_branches_return_slide_values(mach: float, expected: float) -> None:
    cd = wave_drag(mach, mach_critical=0.60, mach_dd=0.85)

    assert cd == pytest.approx(expected)


def test_wave_drag_critical_above_divergence_raises() -> None:
    with pytest.raises(ValueError):
        wave_drag(0.8, mach_critical=0.9, mach_dd=0.85)


@given(
    m_a=st.floats(min_value=0.6, max_value=0.95),
    m_b=st.floats(min_value=0.6, max_value=0.95),
)
def test_wave_drag_increases_with_mach_above_critical(m_a: float, m_b: float) -> None:
    low, high = sorted((m_a, m_b))

    cd_low = wave_drag(low, 0.60, 0.85)
    cd_high = wave_drag(high, 0.60, 0.85)

    assert cd_low <= cd_high


def test_upsweep_drag_area_returns_hand_value() -> None:
    # 3.83 * 0.1^2.5 * 10
    drag_area = upsweep_drag_area(0.1, 10.0)

    assert drag_area == pytest.approx(0.12111523438444896)


def test_upsweep_drag_area_zero_upsweep_returns_zero() -> None:
    assert upsweep_drag_area(0.0, 10.0) == 0.0


def test_base_drag_area_at_offset_mach_returns_constant_term() -> None:
    drag_area = base_drag_area(0.161, 2.0)

    assert drag_area == pytest.approx(0.139 * 2.0)


def test_base_drag_area_closed_cone_returns_zero() -> None:
    assert base_drag_area(0.77, 0.0) == 0.0


def test_base_drag_supersonic_raises() -> None:
    with pytest.raises(ValueError):
        base_drag_area(1.2, 1.0)


def test_flap_drag_slotted_40_deg_returns_hand_value() -> None:
    # 0.0074 * 0.3 * (30/60) * (40 - 10)
    cd = flap_drag(0.0074, 0.3, 30.0, 60.0, 40.0)

    assert cd == pytest.approx(0.0333)


@pytest.mark.parametrize("deflection", [0.0, 5.0, 10.0])
def test_flap_drag_at_or_below_10_deg_returns_zero(deflection: float) -> None:
    cd = flap_drag(0.0144, 0.15, 30.0, 60.0, deflection)

    assert cd == 0.0


@pytest.mark.parametrize(
    ("coeff", "frontal_area", "expected"),
    [
        (0.05328, 0.5, 0.12188394036274912),  # open wells, m = 1
        (0.04955, 0.25, 0.006841387199346724),  # closed wells, m = 0.5
    ],
)
def test_landing_gear_drag_returns_hand_value(
    coeff: float, frontal_area: float, expected: float
) -> None:
    # coeff * exp(5.615 m) * (d w) / S_ref with w = 0.5, d = 1.0, S_ref = 60
    cd = landing_gear_drag(coeff, frontal_area, 0.5, 1.0, 60.0)

    assert cd == pytest.approx(expected)


def test_landing_gear_drag_area_ratio_above_chart_raises() -> None:
    with pytest.raises(ValueError):
        landing_gear_drag(0.05328, 0.6, 0.5, 1.0, 60.0)


@given(
    cd0_without=st.floats(min_value=0.0, max_value=0.1),
    fraction=st.floats(min_value=0.0, max_value=0.5),
)
def test_excrescence_drag_is_fraction_of_total(
    cd0_without: float, fraction: float
) -> None:
    delta = excrescence_drag(cd0_without, fraction)

    total = cd0_without + delta
    assert delta == pytest.approx(fraction * total, abs=1e-15)


@pytest.mark.parametrize(
    "call",
    [
        lambda: drag_divergence_mach(0.0, 0.12, 0.5, 0.0),
        lambda: drag_divergence_mach(0.95, 0.12, 0.5, math.pi / 2),
        lambda: streamwise_critical_mach(1.0, 0.3),
        lambda: upsweep_drag_area(-0.1, 10.0),
        lambda: upsweep_drag_area(0.1, 0.0),
        lambda: base_drag_area(0.5, -1.0),
        lambda: flap_drag(0.0074, 0.0, 30.0, 60.0, 40.0),
        lambda: flap_drag(0.0074, 0.3, 30.0, 60.0, -5.0),
        lambda: landing_gear_drag(0.05328, 0.0, 0.5, 1.0, 60.0),
        lambda: excrescence_drag(0.02, 1.0),
    ],
)
def test_miscellaneous_invalid_input_raises(call: object) -> None:
    assert callable(call)

    with pytest.raises(ValueError):
        call()
