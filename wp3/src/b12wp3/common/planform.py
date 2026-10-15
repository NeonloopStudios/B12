# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Straight-tapered (trapezoidal) planform of one lifting surface.

One class serves the wing, the horizontal tail, the vertical tail and the
pylons. A symmetric surface has two panels (wing, horizontal tail); a single
surface has one panel (vertical tail, one pylon). The span is always the
full span b = sqrt(S * AR), so for a one-panel surface the span is the
panel height. The relations are the same as wp2/src/b12wp2/config/planform.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

QUARTER_CHORD: Final[float] = 0.25


@dataclass(frozen=True)
class TrapezoidalPlanform:
    """Trapezoidal planform from its four design variables.

    Attributes:
        area_m2: Planform area of all panels [m^2].
        aspect_ratio: b^2 / S [-].
        taper: c_tip / c_root [-], in (0, 1].
        sweep_c4_rad: Quarter-chord sweep [rad].
        panels: 2 for a symmetric surface, 1 for a single surface.
    """

    area_m2: float
    aspect_ratio: float
    taper: float
    sweep_c4_rad: float
    panels: int

    def __post_init__(self) -> None:
        if self.area_m2 <= 0.0:
            raise ValueError(f"area_m2 must be > 0, got {self.area_m2}")
        if self.aspect_ratio <= 0.0:
            raise ValueError(f"aspect_ratio must be > 0, got {self.aspect_ratio}")
        if not 0.0 < self.taper <= 1.0:
            raise ValueError(f"taper must be in (0, 1], got {self.taper}")
        if not abs(self.sweep_c4_rad) < math.pi / 2:
            raise ValueError(f"|sweep_c4_rad| must be < pi/2, got {self.sweep_c4_rad}")
        if self.panels not in (1, 2):
            raise ValueError(f"panels must be 1 or 2, got {self.panels}")

    @property
    def span_m(self) -> float:
        """Full span b = sqrt(S * AR) [m]."""
        return math.sqrt(self.area_m2 * self.aspect_ratio)

    @property
    def panel_span_m(self) -> float:
        """Span of one panel, root to tip [m]."""
        return self.span_m / self.panels

    @property
    def root_chord_m(self) -> float:
        """Root chord c_r = 2 S / (b (1 + taper)) [m]."""
        return 2.0 * self.area_m2 / (self.span_m * (1.0 + self.taper))

    @property
    def tip_chord_m(self) -> float:
        """Tip chord c_t = taper * c_r [m]."""
        return self.taper * self.root_chord_m

    @property
    def mac_m(self) -> float:
        """Mean aerodynamic chord [m]."""
        lam = self.taper
        return 2.0 / 3.0 * self.root_chord_m * (1.0 + lam + lam**2) / (1.0 + lam)

    def chord_at(self, y_m: float) -> float:
        """Local chord at a span station measured from the root.

        Args:
            y_m: Distance from the root along the panel span [m].

        Returns:
            Local chord [m].

        Raises:
            ValueError: If y_m is outside the panel.
        """
        if not 0.0 <= y_m <= self.panel_span_m:
            raise ValueError(f"y_m must be in [0, {self.panel_span_m}], got {y_m}")
        chord_drop = self.root_chord_m - self.tip_chord_m
        return self.root_chord_m - chord_drop * y_m / self.panel_span_m

    def sweep_at(self, chord_fraction: float) -> float:
        """Sweep angle of the line at a given chord fraction.

        Args:
            chord_fraction: x/c of the line [-], 0 (LE) to 1 (TE).

        Returns:
            Sweep angle [rad].
        """
        chord_drop = self.root_chord_m - self.tip_chord_m
        tan_sweep = (
            math.tan(self.sweep_c4_rad)
            - (chord_fraction - QUARTER_CHORD) * chord_drop / self.panel_span_m
        )
        return math.atan(tan_sweep)

    def exposed_area(self, covered_span_m: float) -> float:
        """Planform area outside the part covered by the fuselage.

        The covered part is a strip at the root. For two panels the strip is
        centred, so each panel loses covered_span_m / 2. For one panel the
        strip is covered_span_m long.

        Args:
            covered_span_m: Total span length inside the fuselage [m].

        Returns:
            Exposed planform area S_exp of all panels [m^2].

        Raises:
            ValueError: If the covered span is negative or not smaller than b.
        """
        if not 0.0 <= covered_span_m < self.span_m:
            raise ValueError(
                f"covered_span_m must be in [0, {self.span_m}), got {covered_span_m}"
            )
        y0 = covered_span_m / self.panels
        exposed_panel_span = self.panel_span_m - y0
        mean_chord = 0.5 * (self.chord_at(y0) + self.tip_chord_m)
        return self.panels * exposed_panel_span * mean_chord
