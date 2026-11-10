# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Shared fixtures for the wp3 tests: flight conditions, settings and a geometry.

The sample geometry is a small, valid aircraft for the build-up tests, and
the sample VSP inputs place its components for the OpenVSP layout tests.
Their numbers are test data, not design values.
"""

from __future__ import annotations

import pytest

from b12wp3.config.drag import DragSettings, InterferenceFactors
from b12wp3.config.drag_tables import LaminarFraction
from b12wp3.config.geometry import (
    AircraftGeometry,
    BodyInputs,
    FlapInputs,
    FuselageAftInputs,
    GearInputs,
    LiftingSurfaceInputs,
    NacelleInputs,
)
from b12wp3.config.mission import FlightCondition, FlowRegime
from b12wp3.config.vsp_model import (
    CentreOfGravity,
    FuselageShape,
    GroundInputs,
    MainGearPlacement,
    NacelleDetail,
    NacellePlacement,
    NoseGearPlacement,
    PylonInputs,
    TailPlacement,
    VspModelInputs,
    WingPlacement,
)


@pytest.fixture(scope="session")
def subsonic_condition() -> FlightCondition:
    """Sea level, M = 0.3, subsonic cutoff."""
    return FlightCondition("test_subsonic", 0.0, 0.3, FlowRegime.SUBSONIC)


@pytest.fixture(scope="session")
def unit_settings() -> DragSettings:
    """Drag settings with IF = 1 everywhere and the selected table values."""
    return DragSettings(
        laminar=LaminarFraction(0.05, 0.10),
        roughness_m=0.634e-5,
        interference=InterferenceFactors(1.0, 1.0, 1.0, 1.0, 1.0),
        kappa_a=0.95,
        wing_mach_critical_normal=0.56,
        wave_sweep_chord_fraction=0.5,
        te_flap_factor=0.0074,
        le_flap_factor=0.0144,
        wheel_well_coeff=0.05328,
        excrescence_fraction=0.035,
    )


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
        nacelle=NacelleInputs(4.0, 1.07, 1.34, 0.35, 0.90),
        engine_count=2,
        fuselage_aft=FuselageAftInputs(upsweep_deg=5.0, base_area_m2=0.1),
        te_flap=FlapInputs(0.30, 0.12, 0.75),
        le_flap=FlapInputs(0.15, 0.12, 0.95),
        main_gear=GearInputs(2, 0.4, 0.4, 1.2),
        nose_gear=GearInputs(1, 0.15, 0.25, 0.7),
    )


@pytest.fixture(scope="session")
def sample_vsp_inputs() -> VspModelInputs:
    """Valid VSP model positions for sample_geometry: low wing, wing gear."""
    return VspModelInputs(
        wing=WingPlacement(12.0, -0.8, 2.0, 2.0, -2.0),
        centre_of_gravity=CentreOfGravity(14.5),  # about 23 % MAC
        tails=TailPlacement(15.0, 15.0, 0.5, 0.0),
        fuselage_shape=FuselageShape(-0.3, 8, 6),
        nacelle_position=NacellePlacement(4.0, 2.0, 0.9),
        nacelle_detail=NacelleDetail(0.04, 0.06, 0.02, 0.12, 0.2, 0.3, 0.55, 0.7),
        pylon=PylonInputs(0.2, 0.7, 45.0, 0.08),
        nose_gear_position=NoseGearPlacement(3.0, 2, 0.12, 0.15),
        main_gear_position=MainGearPlacement(14.5, 4.4, 2, 0.2, 0.25),
        ground=GroundInputs(1.2),
    )
