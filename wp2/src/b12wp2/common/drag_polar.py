# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""Lift-induced drag parameters of the drag polar CD = CD0 + K CL^2: the
Oswald factor, the quadratic coefficient K and the winglet effect on the
aspect ratio, as given on the ADSEE-II formula sheet (AE2111-II, Formula
Sheet - Aircraft, "Oswald factor, quadratic coefficient and effect of
winglet").

Pure functions, no OpenVSP/XFoil dependency -- tested in
test_drag_polar.py against structural properties of the formulas.
"""
from __future__ import annotations

import math


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
