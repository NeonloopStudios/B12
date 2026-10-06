# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""Lift-induced drag parameters of the drag polar CD = CD0 + K CL^2: the
Oswald factor, the quadratic coefficient K and the winglet effect on the
aspect ratio, as given on the ADSEE-II formula sheet (AE2111-II, Formula
Sheet - Aircraft, "Oswald factor, quadratic coefficient and effect of
winglet"); and the induced drag of a given spanwise loading from Prandtl's
lifting-line theory.

Pure functions, no OpenVSP/XFoil dependency -- tested in
test_drag_polar.py against structural properties of the formulas.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


def oswald_factor(aspect_ratio: float, sweep_half_chord: float) -> float:
    """Oswald factor e of a wing (ADSEE-II formula sheet).

        e = 2 / (2 - AR + sqrt(4 + AR^2 (1 + tan^2 Lambda_0.5c)))

    `sweep_half_chord` is the half-chord sweep Lambda_0.5c in radians.
    Decreases with sweep; for an unswept wing it rises from 1/2 (AR -> 0)
    to 1 (AR -> inf), for a swept wing of transport aspect ratio it falls
    with AR.
    """
    if aspect_ratio <= 0.0:
        raise ValueError(f"oswald_factor: aspect_ratio must be > 0, got {aspect_ratio}")
    ar = aspect_ratio
    root = math.sqrt(4.0 + ar**2 * (1.0 + math.tan(sweep_half_chord) ** 2))
    return 2.0 / (2.0 - ar + root)


def induced_drag_factor(oswald: float, aspect_ratio: float) -> float:
    """Quadratic coefficient of the drag polar: K = 1 / (pi e AR)."""
    if oswald <= 0.0 or aspect_ratio <= 0.0:
        raise ValueError(
            f"induced_drag_factor: oswald and aspect_ratio must be > 0, got {oswald}, {aspect_ratio}"
        )
    return 1.0 / (math.pi * oswald * aspect_ratio)


def winglet_aspect_ratio(aspect_ratio: float, winglet_height: float, span: float) -> float:
    """Effective aspect ratio with winglets: AR_e = AR (1 + 1.9 h / b).

    `winglet_height` h and wing `span` b in the same unit.
    """
    if span <= 0.0:
        raise ValueError(f"winglet_aspect_ratio: span must be > 0, got {span}")
    return aspect_ratio * (1.0 + 1.9 * winglet_height / span)


class LiftingLine(NamedTuple):
    """Lifting-line coefficients of a spanwise loading."""

    cl: float  # wing lift coefficient, pi AR A_1
    cdi: float  # induced drag coefficient, pi AR sum(n A_n^2)
    span_efficiency: float  # CL^2 / (pi AR CDi) = 1 / (1 + sum_{n>1} n (A_n/A_1)^2)


def induced_drag_from_loading(
    y: npt.ArrayLike,
    cl_c: npt.ArrayLike,
    span: float,
    aspect_ratio: float,
    n_terms: int = 7,
) -> LiftingLine:
    """Induced drag of a symmetric spanwise loading by Prandtl's lifting-line
    theory (e.g. Anderson, Fundamentals of Aerodynamics, sec. 5.3).

    `y` are the spanwise stations of one half-wing (0 < y <= b/2) and `cl_c`
    the local section lift times chord there, cl c = 2 Gamma / V. With
    y = (b/2) cos(theta) the loading is fitted, in a least-squares sense, by
    the odd terms of the Fourier sine series

        cl c = 4 b sum A_n sin(n theta),   n = 1, 3, ..., 2 n_terms - 1

    which gives CL = pi AR A_1 and CDi = pi AR sum n A_n^2. CDi follows from
    the coefficients directly, so it stays defined at CL = 0 (a twisted wing
    still has induced drag there); the span efficiency is NaN when CDi = 0.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    load = np.asarray(cl_c, dtype=np.float64)
    if y_arr.shape != load.shape or y_arr.ndim != 1:
        raise ValueError("induced_drag_from_loading: y and cl_c must be 1D arrays of the same length")
    if y_arr.size < n_terms:
        raise ValueError(f"induced_drag_from_loading: {y_arr.size} stations cannot fit {n_terms} terms")
    half = span / 2.0
    if np.any(y_arr <= 0.0) or np.any(y_arr > half):
        raise ValueError("induced_drag_from_loading: stations must lie in (0, b/2]")

    theta = np.arccos(y_arr / half)
    n = np.arange(1, 2 * n_terms, 2)
    basis = 4.0 * span * np.sin(np.outer(theta, n))
    a_n, *_ = np.linalg.lstsq(basis, load, rcond=None)

    cl = math.pi * aspect_ratio * a_n[0]
    cdi = math.pi * aspect_ratio * float(np.sum(n * a_n**2))
    span_eff = cl**2 / (math.pi * aspect_ratio * cdi) if cdi > 0.0 else float("nan")
    return LiftingLine(float(cl), cdi, span_eff)
