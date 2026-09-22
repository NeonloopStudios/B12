# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Tests for wp2/src/b12wp2/hld/sizing.py."""
from __future__ import annotations

import math

import pytest

from b12wp2.config import hld as hld_cfg
from b12wp2.hld import sizing as hs

# unswept rectangular wing: S = 20 m^2, b = 10 m, c = 2 m
RECT = hs.Planform(s_ref=20.0, b=10.0, c_root=2.0, c_tip=2.0, tan_sweep_le=0.0)
# swept tapered wing
TAPERED = hs.Planform(s_ref=80.5, b=27.65, c_root=3.88, c_tip=1.94, tan_sweep_le=math.tan(math.radians(25.7)))

SINGLE_SLOTTED = next(d for d in hld_cfg.TE_DEVICES if d.name == "single-slotted")
FOWLER = next(d for d in hld_cfg.TE_DEVICES if d.name == "Fowler")
SLAT = next(d for d in hld_cfg.LE_DEVICES if d.name == "slat")


def test_covered_area_of_whole_wing_is_reference_area() -> None:
    assert hs.covered_area(RECT, 0.0, 1.0) == pytest.approx(RECT.s_ref)
    assert hs.covered_area(TAPERED, 0.0, 1.0) == pytest.approx(
        (TAPERED.c_root + TAPERED.c_tip) / 2 * TAPERED.b
    )


def test_covered_area_rejects_bad_limits() -> None:
    with pytest.raises(ValueError):
        hs.covered_area(RECT, 0.5, 0.4)


def test_full_span_unswept_flap_gives_0_9_times_airfoil_increment() -> None:
    eff = hs.device_effect(RECT, SINGLE_SLOTTED, 0.3, 0.0, 1.0)
    assert eff.swf_s == pytest.approx(1.0)
    assert eff.dcl_max_wing == pytest.approx(0.9 * 1.3)
    assert eff.dalpha_0l_deg == pytest.approx(hld_cfg.DALPHA_0L_AIRFOIL_LANDING)


def test_extending_devices_scale_with_chord_extension() -> None:
    assert FOWLER.dcl_max == pytest.approx(1.3 * FOWLER.chord_extension)
    eff = hs.device_effect(RECT, FOWLER, 0.3, 0.0, 0.5)
    assert eff.area_ratio == pytest.approx(1.0 + 0.5 * (FOWLER.chord_extension - 1.0))


def test_hinge_line_sweep_reduces_increment_by_cosine() -> None:
    eff = hs.device_effect(TAPERED, SINGLE_SLOTTED, 0.35, 0.1, 0.7)
    sweep = TAPERED.sweep_at(0.65)
    assert eff.sweep_hinge_rad == pytest.approx(sweep)
    assert eff.dcl_max_wing == pytest.approx(0.9 * 1.3 * eff.swf_s * math.cos(sweep))
    # LE device hinges at x/c = c_s/c
    assert hs.device_effect(TAPERED, SLAT, 0.15, 0.1, 0.9).sweep_hinge_rad == pytest.approx(
        TAPERED.sweep_at(0.15)
    )


def test_leading_edge_device_does_not_shift_zero_lift_angle() -> None:
    assert hs.device_effect(RECT, SLAT, 0.15, 0.0, 1.0).dalpha_0l_deg == 0.0


def test_takeoff_setting_scales_increments() -> None:
    clean = hs.CleanWing(cl_max=1.4, cl_alpha_per_deg=0.08, alpha_0l_deg=-3.0, alpha_stall_deg=15.0)
    te = hs.device_effect(RECT, FOWLER, 0.3, 0.0, 0.6)
    le = hs.device_effect(RECT, SLAT, 0.15, 0.0, 0.9)
    land = hs.configuration(clean, te, le)
    to = hs.configuration(clean, te, le, takeoff=True)
    assert land.cl_max - clean.cl_max == pytest.approx(te.dcl_max_wing + le.dcl_max_wing)
    assert to.cl_max - clean.cl_max == pytest.approx(hld_cfg.TAKEOFF_FRACTION * (land.cl_max - clean.cl_max))
    assert to.alpha_0l_deg - clean.alpha_0l_deg == pytest.approx(
        te.dalpha_0l_deg * hld_cfg.DALPHA_0L_AIRFOIL_TAKEOFF / hld_cfg.DALPHA_0L_AIRFOIL_LANDING
    )


def test_clean_configuration_reproduces_clean_stall_angle() -> None:
    clean = hs.CleanWing(cl_max=1.4, cl_alpha_per_deg=0.08, alpha_0l_deg=-3.0, alpha_stall_deg=16.0)
    none = hs.DeviceEffect(0.0, 0.0, 0.0, 1.0, 0.0)
    cfg = hs.configuration(clean, none, none)
    assert cfg.alpha_stall_deg == pytest.approx(clean.alpha_stall_deg)
    assert cfg.cl_max == pytest.approx(clean.cl_max)


def test_min_eta_out_meets_requirement_exactly() -> None:
    target = 0.4
    eta = hs.min_eta_out(TAPERED, FOWLER, 0.35, 0.12, 0.75, target)
    assert hs.device_effect(TAPERED, FOWLER, 0.35, 0.12, eta).dcl_max_wing == pytest.approx(target, abs=1e-5)
    assert math.isnan(hs.min_eta_out(TAPERED, FOWLER, 0.35, 0.12, 0.75, 5.0))
    assert hs.min_eta_out(TAPERED, FOWLER, 0.35, 0.12, 0.75, 0.0) == 0.12


def test_lift_curve_is_continuous_and_peaks_at_stall() -> None:
    cfg = hs.Configuration(cl_max=2.5, cl_alpha_per_deg=0.1, alpha_0l_deg=-12.0, alpha_stall_deg=15.0)
    alphas = [x / 10 for x in range(-150, 151)]
    cl = hs.lift_curve(cfg, alphas)
    assert cl[-1] == pytest.approx(2.5)
    assert max(cl) == pytest.approx(2.5)
    assert max(abs(b - a) for a, b in zip(cl, cl[1:])) < 0.1 * 0.1 + 1e-9  # no jumps
    assert hs.lift_curve(cfg, [16.0])[0] != hs.lift_curve(cfg, [16.0])[0]  # NaN past stall
