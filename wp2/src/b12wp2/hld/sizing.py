# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Empirical high-lift device (HLD) sizing, ADSEE method.

For a device spanning eta_in..eta_out of the semi-span:

    dCL_max   = 0.9 * dcl_max * (S_wf / S) * cos(sweep_hinge)
    S' / S    = 1 + (S_wf / S) * (c'/c - 1)          -> CL_alpha' = CL_alpha * S'/S
    da_0L     = da_0L,airfoil * (S_wf / S) * cos(sweep_hinge)   (trailing edge only)

with S_wf the (full-chord) wing area covered by the device, dcl_max the
airfoil increment of the device type (config.hld device tables), c'/c the chord
extension of the deployed device and sweep_hinge the sweep of its hinge
line (x/c = 1 - c_f/c for a trailing-edge device, c_s/c for a leading-edge
one).

Pure numpy, no OpenVSP: the planform is passed in as a Planform, so this is
unit-testable in any environment. The device table itself (which devices
exist, their ADSEE increments and complexity ranks) and the airfoil
zero-lift shifts are inputs, so they live in b12wp2.config.hld.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from b12wp2.config import hld as cfg
from b12wp2.config.hld import Device

# ============================================================
#  PLANFORM
# ============================================================


@dataclass(frozen=True)
class Planform:
    """Straight-tapered wing, both halves."""

    s_ref: float  # [m^2]
    b: float  # span [m]
    c_root: float  # [m]
    c_tip: float  # [m]
    tan_sweep_le: float

    @property
    def b_half(self) -> float:
        return self.b / 2

    def chord(self, eta: float) -> float:
        return self.c_root - (self.c_root - self.c_tip) * eta

    def sweep_at(self, chord_fraction: float) -> float:
        """Sweep [rad] of the line at the given chord fraction."""
        return math.atan(
            self.tan_sweep_le - chord_fraction * (self.c_root - self.c_tip) / self.b_half
        )


# ============================================================
#  GEOMETRY
# ============================================================


def covered_area(planform: Planform, eta_in: float, eta_out: float) -> float:
    """Full-chord wing area between eta_in and eta_out, both halves [m^2]."""
    if not 0.0 <= eta_in < eta_out <= 1.0:
        raise ValueError(f"covered_area: need 0 <= eta_in < eta_out <= 1, got {eta_in}, {eta_out}")
    mean_chord = (planform.chord(eta_in) + planform.chord(eta_out)) / 2
    return 2 * mean_chord * (eta_out - eta_in) * planform.b_half


def hinge_sweep(planform: Planform, device: Device, chord_ratio: float) -> float:
    """Hinge-line sweep [rad]: at x/c = 1 - c_f/c (TE) or x/c = c_s/c (LE)."""
    x_c = 1.0 - chord_ratio if device.edge == "TE" else chord_ratio
    return planform.sweep_at(x_c)


# ============================================================
#  AERODYNAMIC INCREMENTS
# ============================================================


@dataclass(frozen=True)
class DeviceEffect:
    """Landing-setting effect of one device on the wing."""

    swf_s: float  # S_wf / S
    sweep_hinge_rad: float
    dcl_max_wing: float  # dCL_max
    area_ratio: float  # S'/S
    dalpha_0l_deg: float  # landing setting


def device_effect(
    planform: Planform, device: Device, chord_ratio: float, eta_in: float, eta_out: float
) -> DeviceEffect:
    swf_s = covered_area(planform, eta_in, eta_out) / planform.s_ref
    sweep = hinge_sweep(planform, device, chord_ratio)
    cos_h = math.cos(sweep)
    dalpha = cfg.DALPHA_0L_AIRFOIL_LANDING * swf_s * cos_h if device.edge == "TE" else 0.0
    return DeviceEffect(
        swf_s=swf_s,
        sweep_hinge_rad=sweep,
        dcl_max_wing=0.9 * device.dcl_max * swf_s * cos_h,
        area_ratio=1.0 + swf_s * (device.chord_extension - 1.0),
        dalpha_0l_deg=dalpha,
    )


@dataclass(frozen=True)
class CleanWing:
    """Clean-wing lift characteristics (from VSPAERO + XFoil landing polar)."""

    cl_max: float
    cl_alpha_per_deg: float
    alpha_0l_deg: float
    alpha_stall_deg: float

    @property
    def alpha_offset_deg(self) -> float:
        """Stall angle beyond the linear-lift intercept of CL_max:
        alpha_stall - (alpha_0L + CL_max / CL_alpha). Carried over unchanged
        to the flapped configurations."""
        return self.alpha_stall_deg - (self.alpha_0l_deg + self.cl_max / self.cl_alpha_per_deg)


@dataclass(frozen=True)
class Configuration:
    """Lift characteristics of one configuration in one setting."""

    cl_max: float
    cl_alpha_per_deg: float
    alpha_0l_deg: float
    alpha_stall_deg: float


def configuration(
    clean: CleanWing, te: DeviceEffect, le: DeviceEffect, *, takeoff: bool = False
) -> Configuration:
    """Combine TE and LE effects (superposed) on the clean wing.

    Take-off: config.hld.TAKEOFF_FRACTION of the landing dCL_max and of the chord
    extension (S'/S - 1), and the take-off airfoil da_0L.
    """
    f = cfg.TAKEOFF_FRACTION if takeoff else 1.0
    dalpha_scale = (
        cfg.DALPHA_0L_AIRFOIL_TAKEOFF / cfg.DALPHA_0L_AIRFOIL_LANDING if takeoff else 1.0
    )
    dcl_max = f * (te.dcl_max_wing + le.dcl_max_wing)
    area_ratio = 1.0 + f * ((te.area_ratio - 1.0) + (le.area_ratio - 1.0))
    cl_max = clean.cl_max + dcl_max
    cl_alpha = clean.cl_alpha_per_deg * area_ratio
    alpha_0l = clean.alpha_0l_deg + dalpha_scale * (te.dalpha_0l_deg + le.dalpha_0l_deg)
    alpha_stall = alpha_0l + cl_max / cl_alpha + clean.alpha_offset_deg
    return Configuration(cl_max, cl_alpha, alpha_0l, alpha_stall)


def min_eta_out(
    planform: Planform,
    device: Device,
    chord_ratio: float,
    eta_in: float,
    eta_max: float,
    dcl_max_required: float,
    *,
    tol: float = 1e-6,
) -> float:
    """Smallest eta_out giving dCL_max >= dcl_max_required (bisection;
    dCL_max grows monotonically with eta_out). NaN if eta_max is not enough."""
    if dcl_max_required <= 0.0:
        return eta_in

    def dcl(eta: float) -> float:
        return device_effect(planform, device, chord_ratio, eta_in, eta).dcl_max_wing

    if dcl(eta_max) < dcl_max_required:
        return float("nan")
    lo, hi = eta_in, eta_max
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if mid > eta_in and dcl(mid) >= dcl_max_required:
            hi = mid
        else:
            lo = mid
    return hi


def lift_curve(config: Configuration, alphas_deg: list[float]) -> list[float]:
    """Schematic CL(alpha): linear, blended into a parabola that is tangent
    to the linear part and peaks at (alpha_stall, CL_max). Post-stall is not
    modelled (the curve stops at alpha_stall)."""
    a, a0, cl_max, a_s = config.cl_alpha_per_deg, config.alpha_0l_deg, config.cl_max, config.alpha_stall_deg
    linear_at_stall = a * (a_s - a0)
    d = 2 * (linear_at_stall - cl_max) / a  # width of the parabolic round-off
    out = []
    for alpha in alphas_deg:
        if alpha > a_s:
            out.append(float("nan"))
        elif d <= 0 or alpha <= a_s - d:
            out.append(min(a * (alpha - a0), cl_max))
        else:
            k = a / (2 * d)
            out.append(cl_max - k * (a_s - alpha) ** 2)
    return out
