# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-15
"""Drag due to lift (ADSEE-II, "Estimation of the drag due to lift").

    CD = CD0 + K CL^2,   K = 1 / (pi AR_eff e)                    (slide 72)

    AR_eff = AR + DeltaAR,   DeltaAR = 1.9 (h/b) AR   (winglet, slides 75, 78)

    e = 2 / (2 - AR_eff + sqrt(4 + AR_eff^2 (1 + tan^2 Lambda_0.5c)))
                                               (Brandt et al., slide 72)

Slide 75: the effective aspect ratio is determined first and then used in
all lift and drag equations, so e is evaluated at AR_eff as well.

Flapped configurations: e = e_clean + 0.0026 delta_f, Vos (2022) eq. 7.62,
wing-mounted engines. The landing gear does not change e (Vos 2022,
section 7.6.2).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from b12wp3.common.planform import TrapezoidalPlanform
from b12wp3.config.induced import ReferenceWinglet

logger = logging.getLogger(__name__)

WINGLET_AR_FACTOR: Final[float] = 1.9  # DeltaAR = 1.9 (h/b) AR, slide 78
HALF_CHORD: Final[float] = 0.5


def mean_height_to_span(references: Iterable[ReferenceWinglet]) -> float:
    """Mean winglet height-to-span ratio of the reference aircraft.

    Args:
        references: Reference winglets.

    Returns:
        Arithmetic mean of h/b [-].

    Raises:
        ValueError: If there are no references.
    """
    ratios = [ref.height_to_span for ref in references]
    if not ratios:
        raise ValueError("mean_height_to_span needs at least one reference")
    return sum(ratios) / len(ratios)


def winglet_effective_aspect_ratio(
    aspect_ratio: float, winglet_height_m: float, span_m: float
) -> float:
    """Effective aspect ratio of a wing with winglets.

    Args:
        aspect_ratio: Geometric aspect ratio AR [-].
        winglet_height_m: Winglet height h above the wing-tip chord plane [m].
        span_m: Wing span b [m].

    Returns:
        AR_eff = AR + 1.9 (h/b) AR [-].

    Raises:
        ValueError: If AR or b is not positive, or h is negative.
    """
    if aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be > 0, got {aspect_ratio}")
    if span_m <= 0.0:
        raise ValueError(f"span_m must be > 0, got {span_m}")
    if winglet_height_m < 0.0:
        raise ValueError(f"winglet_height_m must be >= 0, got {winglet_height_m}")
    delta = WINGLET_AR_FACTOR * winglet_height_m / span_m * aspect_ratio
    return aspect_ratio + delta


def oswald_factor_brandt(aspect_ratio: float, sweep_half_chord_rad: float) -> float:
    """Oswald factor by Brandt et al. (ADSEE-II slide 72).

    Args:
        aspect_ratio: Aspect ratio [-]; the effective one for a wing with a
            wing-tip device.
        sweep_half_chord_rad: Half-chord sweep Lambda_0.5c [rad].

    Returns:
        e = 2 / (2 - AR + sqrt(4 + AR^2 (1 + tan^2 Lambda_0.5c))) [-].

    Raises:
        ValueError: If AR is not positive or |sweep| >= 90 deg.
    """
    if aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be > 0, got {aspect_ratio}")
    if not abs(sweep_half_chord_rad) < math.pi / 2:
        raise ValueError(
            f"|sweep_half_chord_rad| must be < pi/2, got {sweep_half_chord_rad}"
        )
    tan_sq = math.tan(sweep_half_chord_rad) ** 2
    root = math.sqrt(4.0 + aspect_ratio**2 * (1.0 + tan_sq))
    return 2.0 / (2.0 - aspect_ratio + root)


def oswald_factor_raymer_straight(aspect_ratio: float) -> float:
    """Oswald factor by Raymer, straight wings (ADSEE-II slide 72).

    Comparison only; the design value is Brandt's.

    Args:
        aspect_ratio: Aspect ratio [-].

    Returns:
        e = 1.78 (1 - 0.045 AR^0.68) - 0.64 [-].
    """
    if aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be > 0, got {aspect_ratio}")
    ar_power: float = aspect_ratio**0.68
    return 1.78 * (1.0 - 0.045 * ar_power) - 0.64


def oswald_factor_raymer_swept(aspect_ratio: float, sweep_le_rad: float) -> float:
    """Oswald factor by Raymer, swept wings (ADSEE-II slide 72).

    Comparison only; the design value is Brandt's.

    Args:
        aspect_ratio: Aspect ratio [-].
        sweep_le_rad: Leading-edge sweep Lambda_LE [rad].

    Returns:
        e = 4.61 (1 - 0.045 AR^0.68) (cos Lambda_LE)^0.15 - 3.1 [-].
    """
    if aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be > 0, got {aspect_ratio}")
    if not abs(sweep_le_rad) < math.pi / 2:
        raise ValueError(f"|sweep_le_rad| must be < pi/2, got {sweep_le_rad}")
    ar_power: float = aspect_ratio**0.68
    cos_power: float = math.cos(sweep_le_rad) ** 0.15
    return 4.61 * (1.0 - 0.045 * ar_power) * cos_power - 3.1


def flap_oswald_increment(flap_deflection_deg: float, slope_per_deg: float) -> float:
    """Change of the Oswald factor with flap deflection (Vos 2022, eq. 7.62).

    Args:
        flap_deflection_deg: Trailing-edge flap deflection [deg], >= 0.
        slope_per_deg: Increment per degree [1/deg].

    Returns:
        Delta e = slope * delta_f [-].
    """
    if flap_deflection_deg < 0.0:
        raise ValueError(f"flap_deflection_deg must be >= 0, got {flap_deflection_deg}")
    return slope_per_deg * flap_deflection_deg


def induced_drag_factor(aspect_ratio: float, oswald: float) -> float:
    """Quadratic coefficient of the drag polar.

    Args:
        aspect_ratio: Aspect ratio [-]; the effective one with a tip device.
        oswald: Oswald factor e [-].

    Returns:
        K = 1 / (pi AR e) [-].
    """
    if aspect_ratio <= 0.0 or oswald <= 0.0:
        raise ValueError(
            f"aspect_ratio and oswald must be > 0, got {aspect_ratio}, {oswald}"
        )
    return 1.0 / (math.pi * aspect_ratio * oswald)


@dataclass(frozen=True)
class InducedDrag:
    """Drag-due-to-lift parameters of one configuration.

    Attributes:
        name: Configuration label.
        aspect_ratio: Geometric aspect ratio AR [-].
        effective_aspect_ratio: AR_eff with the winglet [-].
        sweep_half_chord_rad: Half-chord sweep [rad].
        flap_deflection_deg: Trailing-edge flap deflection [deg].
        oswald_clean: e of the clean wing at AR_eff [-].
        oswald: e of this configuration, clean plus flap increment [-].
    """

    name: str
    aspect_ratio: float
    effective_aspect_ratio: float
    sweep_half_chord_rad: float
    flap_deflection_deg: float
    oswald_clean: float
    oswald: float

    def __post_init__(self) -> None:
        if self.effective_aspect_ratio < self.aspect_ratio:
            raise ValueError(
                f"{self.name}: effective_aspect_ratio must be >= aspect_ratio, "
                f"got {self.effective_aspect_ratio} < {self.aspect_ratio}"
            )
        if self.oswald <= 0.0:
            raise ValueError(f"{self.name}: oswald must be > 0, got {self.oswald}")

    @property
    def k(self) -> float:
        """K = 1 / (pi AR_eff e) [-]."""
        return induced_drag_factor(self.effective_aspect_ratio, self.oswald)

    def cdi(self, lift_coefficient: float) -> float:
        """Lift-induced drag coefficient K CL^2 [-]."""
        return self.k * lift_coefficient**2


def wing_induced_drag(
    planform: TrapezoidalPlanform,
    winglet_height_m: float,
    *,
    name: str = "clean",
    flap_deflection_deg: float = 0.0,
    oswald_flap_slope_per_deg: float = 0.0,
) -> InducedDrag:
    """Drag-due-to-lift parameters of the wing with winglets.

    Args:
        planform: Wing planform.
        winglet_height_m: Winglet height h [m]; 0 for no winglet.
        name: Configuration label.
        flap_deflection_deg: Trailing-edge flap deflection [deg].
        oswald_flap_slope_per_deg: Delta e per degree of flap [1/deg].

    Returns:
        The induced-drag parameters. e is Brandt's formula at AR_eff and the
        half-chord sweep of the planform, plus the flap increment.
    """
    ar_eff = winglet_effective_aspect_ratio(
        planform.aspect_ratio, winglet_height_m, planform.span_m
    )
    sweep_c2 = planform.sweep_at(HALF_CHORD)
    e_clean = oswald_factor_brandt(ar_eff, sweep_c2)
    e = e_clean + flap_oswald_increment(flap_deflection_deg, oswald_flap_slope_per_deg)
    logger.debug(
        "%s: AR_eff=%.4f Lambda_c/2=%.3f deg e_clean=%.4f e=%.4f",
        name,
        ar_eff,
        math.degrees(sweep_c2),
        e_clean,
        e,
    )
    return InducedDrag(
        name=name,
        aspect_ratio=planform.aspect_ratio,
        effective_aspect_ratio=ar_eff,
        sweep_half_chord_rad=sweep_c2,
        flap_deflection_deg=flap_deflection_deg,
        oswald_clean=e_clean,
        oswald=e,
    )
