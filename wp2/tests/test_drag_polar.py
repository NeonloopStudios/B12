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

import numpy as np
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


# --- lifting-line induced drag of a span loading ---

B, AR = 25.17, 9.5  # B12 wing: b = sqrt(S AR), S = 66.7 m^2


def _stations(n: int = 24) -> np.ndarray:
    # strip centres of one half-wing, clustered towards the tip like the VLM mesh
    edges = (B / 2) * np.sin(np.linspace(0.0, np.pi / 2, n + 1))
    return 0.5 * (edges[1:] + edges[:-1])


def _loading(y: np.ndarray, a1: float, a3: float = 0.0) -> np.ndarray:
    theta = np.arccos(y / (B / 2))
    return 4.0 * B * (a1 * np.sin(theta) + a3 * np.sin(3.0 * theta))


def test_lifting_line_elliptic_loading() -> None:
    cl = 0.5
    y = _stations()
    ll = drag_polar.induced_drag_from_loading(y, _loading(y, cl / (math.pi * AR)), B, AR)
    assert ll.cl == pytest.approx(cl)
    assert ll.cdi == pytest.approx(cl**2 / (math.pi * AR))
    assert ll.span_efficiency == pytest.approx(1.0)


def test_lifting_line_third_harmonic() -> None:
    # delta = 3 (A3/A1)^2 -> span efficiency 1 / (1 + delta)
    y = _stations()
    ll = drag_polar.induced_drag_from_loading(y, _loading(y, 0.02, 0.002), B, AR)
    assert ll.span_efficiency == pytest.approx(1.0 / (1.0 + 3.0 * 0.1**2))


def test_lifting_line_lift_matches_integrated_loading() -> None:
    # non-elliptic loading that vanishes at the tip: CL = (2/S) * integral of cl c dy
    y = np.linspace(1e-3, B / 2, 2001)
    cl_c = 2.0 * (1.0 - (y / (B / 2)) ** 2) ** 0.75
    s_ref = B**2 / AR
    cl_integrated = 2.0 / s_ref * float(np.sum(0.5 * (cl_c[1:] + cl_c[:-1]) * np.diff(y)))
    ll = drag_polar.induced_drag_from_loading(y, cl_c, B, AR)
    assert ll.cl == pytest.approx(cl_integrated, rel=1e-3)


def test_lifting_line_induced_drag_at_zero_lift() -> None:
    # a twisted wing at CL = 0 (pure third harmonic) still has induced drag
    y = _stations()
    ll = drag_polar.induced_drag_from_loading(y, _loading(y, 0.0, 0.001), B, AR)
    assert ll.cl == pytest.approx(0.0, abs=1e-12)
    assert ll.cdi == pytest.approx(math.pi * AR * 3.0 * 0.001**2)


def test_lifting_line_insensitive_to_number_of_terms() -> None:
    # non-elliptic loading vanishing at the tip (span efficiency ~0.96)
    y = _stations()
    cl_c = 2.0 * (1.0 - (y / (B / 2)) ** 2) ** 0.75
    effs = [drag_polar.induced_drag_from_loading(y, cl_c, B, AR, n).span_efficiency for n in (5, 7, 9)]
    assert max(effs) - min(effs) < 5e-3


def test_lifting_line_rejects_stations_outside_half_span() -> None:
    y = _stations()
    with pytest.raises(ValueError):
        drag_polar.induced_drag_from_loading(np.append(y, B), np.ones(y.size + 1), B, AR)
