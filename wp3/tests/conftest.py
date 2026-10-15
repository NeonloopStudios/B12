# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Shared fixtures for the wp3 tests: flight conditions and a minimal geometry.

The sample geometry is a small, valid aircraft for the build-up tests. Its
numbers are test data, not design values.
"""

from __future__ import annotations

import pytest

from b12wp3.config.drag import InterferenceFactors
from b12wp3.config.geometry import AircraftGeometry, BodyInputs, LiftingSurfaceInputs
from b12wp3.config.mission import FlightCondition, FlowRegime


@pytest.fixture(scope="session")
def subsonic_condition() -> FlightCondition:
    """Sea level, M = 0.3, subsonic cutoff."""
    return FlightCondition("test_subsonic", 0.0, 0.3, FlowRegime.SUBSONIC)


@pytest.fixture(scope="session")
def unit_interference() -> InterferenceFactors:
    """IF = 1 for every component."""
    return InterferenceFactors(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@pytest.fixture(scope="session")
def sample_geometry() -> AircraftGeometry:
    """Minimal valid aircraft geometry with two engines."""

    def surface(area: float, panels: int, covered: float) -> LiftingSurfaceInputs:
        return LiftingSurfaceInputs(
            area_m2=area,
            aspect_ratio=5.0,
            taper=0.5,
            sweep_c4_deg=20.0,
            thickness_to_chord=0.10,
            x_c_max_thickness=0.35,
            covered_span_m=covered,
            panels=panels,
        )

    return AircraftGeometry(
        wing=surface(60.0, 2, 3.0),
        horizontal_tail=surface(15.0, 2, 0.0),
        vertical_tail=surface(10.0, 1, 0.0),
        fuselage=BodyInputs(4.0, 20.0, 8.0, 3.0),
        nacelle=BodyInputs(0.5, 2.0, 1.0, 1.5),
        pylon=surface(2.0, 1, 0.0),
        engine_count=2,
    )
