# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-15
"""Drag-due-to-lift settings: the winglet reference data and the Oswald
factor increment of the flaps.

Winglet height. The winglet height-to-span ratio h/b is the mean of the
reference aircraft below. h is the vertical height of the winglet tip above
the chord plane of the wing tip, as in the ADSEE-II slide "Effect of wing
tip" (Winglet, DeltaAR = 1.9 (h/b) AR). b is the span over the winglets.
Both are read from the manufacturers' airport planning documents (front
view, general dimensions):

* A320neo: Airbus, "A320 Aircraft Characteristics - Airport and Maintenance
  Planning", Rev. Jun 01/24, 2-2-0 page 6, FIGURE-2-2-0-991-009-A01. The
  sharklet height is dimensioned directly: 2.43 m. Span 35.80 m.
* CRJ900: Bombardier, "CRJ900 Airport Planning Manual", CSP C-020, Rev. 11,
  Dec 17/2015, 00-02-02 page 3, Figure 1 (sheet 1). Winglet tip 127.9 in
  and winglet root 76.0 in above the ground, so h = 51.9 in. Span 915.2 in.

Not used:

* 737-8: Boeing D6-38A004 Rev. K, 2.2.2. The split-tip winglet has an upper
  and a lower blade. The dimensioned 2.92 m runs from the lower to the upper
  blade tip, which is not the single upper winglet of the slide formula.
* E195-E2: Embraer APM 5824. The wing tip is a slightly raised raked tip,
  not a winglet.

Flaps. The Oswald factor of a flapped configuration is
e = e_clean + 0.0026 delta_f, Vos (2022), eq. 7.62, airplanes with
wing-mounted engines. The trailing-edge flap deflections are the take-off
and landing inputs of config/mission.py (15 deg, WP1; 40 deg, the WP2
Fowler maximum). The cruise C_L is also taken from config/mission.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

INCH_TO_M: Final[float] = 0.0254


@dataclass(frozen=True)
class ReferenceWinglet:
    """Winglet height and span of one reference aircraft.

    Attributes:
        aircraft: Aircraft name.
        height_m: Vertical height of the winglet tip above the wing-tip
            chord plane [m].
        span_m: Span over the winglets [m].
        source: Document, revision and figure the values are read from.
    """

    aircraft: str
    height_m: float
    span_m: float
    source: str

    def __post_init__(self) -> None:
        if self.height_m <= 0.0:
            raise ValueError(f"{self.aircraft}: height_m must be > 0, got {self.height_m}")
        if self.span_m <= self.height_m:
            raise ValueError(
                f"{self.aircraft}: span_m must be > height_m, got {self.span_m}"
            )

    @property
    def height_to_span(self) -> float:
        """h / b [-]."""
        return self.height_m / self.span_m


REFERENCE_WINGLETS: Final[tuple[ReferenceWinglet, ...]] = (
    ReferenceWinglet(
        aircraft="A320neo",
        height_m=2.43,
        span_m=35.80,
        source="Airbus A320 AC, Rev. Jun 01/24, FIGURE-2-2-0-991-009-A01",
    ),
    ReferenceWinglet(
        aircraft="CRJ900",
        height_m=(127.9 - 76.0) * INCH_TO_M,
        span_m=915.2 * INCH_TO_M,
        source="Bombardier CRJ900 APM, CSP C-020 Rev. 11, 00-02-02 Figure 1",
    ),
)

# Oswald factor increment per degree of flap deflection, Vos (2022) eq. 7.62,
# wing-mounted engines [1/deg].
OSWALD_FLAP_SLOPE_PER_DEG: Final[float] = 0.0026
