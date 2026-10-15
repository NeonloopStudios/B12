# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/buildup.py and wp3/src/b12wp3/plots/drag.py.

Spec: ADSEE-II, CD0 = (1/S_ref) sum(Cf FF IF S_wet) + sum CD_misc. The
flat-plate case (FF = IF = 1, S_wet = 2 S_ref, fully turbulent) must give
CD0 = 2 Cf.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from b12wp3.config.drag import InterferenceFactors, LaminarFraction
from b12wp3.config.geometry import AircraftGeometry
from b12wp3.config.mission import FlightCondition
from b12wp3.drag.buildup import (
    ComponentDrag,
    DragBuildup,
    aircraft_drag_buildup,
    component_drag,
)
from b12wp3.drag.skin_friction import cf_turbulent
from b12wp3.plots.drag import plot_cd0_breakdown


def _flat_plate(condition: FlightCondition, count: int = 1) -> ComponentDrag:
    return component_drag(
        "plate",
        count=count,
        wetted_area_m2=20.0,
        reference_length_m=2.0,
        form_factor=1.0,
        interference_factor=1.0,
        laminar_fraction=0.0,
        condition=condition,
        roughness_m=0.634e-5,
    )


def test_flat_plate_buildup_returns_twice_cf(
    subsonic_condition: FlightCondition,
) -> None:
    plate = _flat_plate(subsonic_condition)

    buildup = DragBuildup((plate,), reference_area_m2=10.0, cd_misc=0.0)

    expected = 2.0 * cf_turbulent(plate.reynolds, subsonic_condition.mach)
    assert buildup.cd0 == pytest.approx(expected)


def test_buildup_cd_misc_adds_to_cd0(subsonic_condition: FlightCondition) -> None:
    plate = _flat_plate(subsonic_condition)
    clean = DragBuildup((plate,), reference_area_m2=10.0, cd_misc=0.0)

    with_misc = DragBuildup((plate,), reference_area_m2=10.0, cd_misc=0.0015)

    assert with_misc.cd0 - clean.cd0 == pytest.approx(0.0015)


def test_component_count_multiplies_drag_area(
    subsonic_condition: FlightCondition,
) -> None:
    single = _flat_plate(subsonic_condition, count=1)

    double = _flat_plate(subsonic_condition, count=2)

    assert double.drag_area_m2 == pytest.approx(2.0 * single.drag_area_m2)


def test_aircraft_buildup_sum_of_contributions_equals_cd0(
    sample_geometry: AircraftGeometry,
    subsonic_condition: FlightCondition,
    unit_interference: InterferenceFactors,
) -> None:
    buildup = aircraft_drag_buildup(
        sample_geometry,
        subsonic_condition,
        interference=unit_interference,
        cd_misc=0.001,
        laminar=LaminarFraction(0.05, 0.10),
        roughness_m=0.634e-5,
    )

    names = [c.name for c in buildup.components]
    total = sum(buildup.contribution(c) for c in buildup.components) + 0.001
    assert names == [
        "wing",
        "horizontal_tail",
        "vertical_tail",
        "fuselage",
        "nacelle",
        "pylon",
    ]
    assert buildup.cd0 == pytest.approx(total)
    assert buildup.reference_area_m2 == sample_geometry.wing.area_m2


def test_aircraft_buildup_uses_category_laminar_fractions(
    sample_geometry: AircraftGeometry,
    subsonic_condition: FlightCondition,
    unit_interference: InterferenceFactors,
) -> None:
    buildup = aircraft_drag_buildup(
        sample_geometry,
        subsonic_condition,
        interference=unit_interference,
        cd_misc=0.0,
        laminar=LaminarFraction(0.05, 0.10),
        roughness_m=0.634e-5,
    )

    fractions = {c.name: c.laminar_fraction for c in buildup.components}
    assert fractions == {
        "wing": 0.10,
        "horizontal_tail": 0.10,
        "vertical_tail": 0.10,
        "fuselage": 0.05,
        "nacelle": 0.05,
        "pylon": 0.10,
    }


def test_buildup_dataframe_has_total_row_and_full_share(
    subsonic_condition: FlightCondition,
) -> None:
    buildup = DragBuildup(
        (_flat_plate(subsonic_condition),), reference_area_m2=10.0, cd_misc=0.001
    )

    frame = buildup.to_dataframe()

    assert list(frame["component"]) == ["plate", "miscellaneous", "total"]
    assert frame["cd0"].iloc[-1] == pytest.approx(buildup.cd0)
    assert frame["share_pct"].iloc[:-1].sum() == pytest.approx(100.0)


@pytest.mark.parametrize(("reference_area", "cd_misc"), [(0.0, 0.0), (10.0, -0.001)])
def test_buildup_invalid_input_raises(
    subsonic_condition: FlightCondition, reference_area: float, cd_misc: float
) -> None:
    plate = _flat_plate(subsonic_condition)

    with pytest.raises(ValueError):
        DragBuildup((plate,), reference_area_m2=reference_area, cd_misc=cd_misc)


def test_plot_cd0_breakdown_writes_png(
    tmp_path: Path, subsonic_condition: FlightCondition
) -> None:
    buildup = DragBuildup(
        (_flat_plate(subsonic_condition),), reference_area_m2=10.0, cd_misc=0.001
    )

    out = plot_cd0_breakdown(buildup, tmp_path / "drag" / "cd0.png")

    assert out.is_file()
    assert out.stat().st_size > 0
