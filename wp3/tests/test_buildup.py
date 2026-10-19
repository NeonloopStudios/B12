# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/buildup.py and wp3/src/b12wp3/plots/drag.py.

Spec: ADSEE-II, CD0 = (1/S_ref) sum(Cf FF IF S_wet) + sum CD_misc. The
flat-plate case (FF = IF = 1, S_wet = 2 S_ref, fully turbulent) must give
CD0 = 2 Cf. Excrescence and leakage must be the set fraction of the total.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from b12wp3.config.drag import DragSettings
from b12wp3.config.geometry import AircraftGeometry
from b12wp3.config.mission import (
    CRUISE,
    CRUISE_CONFIGURATION,
    DragConfiguration,
    FlightCondition,
)
from b12wp3.drag.buildup import (
    ComponentDrag,
    DragBuildup,
    MiscDrag,
    aircraft_drag_buildup,
    component_drag,
)
from b12wp3.drag.skin_friction import cf_turbulent
from b12wp3.plots.drag import (
    grouped_contributions,
    plot_cd0_breakdown,
    plot_cd0_comparison,
)


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


def _low_speed_configuration(condition: FlightCondition) -> DragConfiguration:
    return DragConfiguration(
        name="test_landing",
        condition=condition,
        lift_coefficient=None,
        te_flap_deflection_deg=40.0,
        le_flap_deflection_deg=20.0,
        gear_down=True,
    )


def test_flat_plate_buildup_returns_twice_cf(
    subsonic_condition: FlightCondition,
) -> None:
    plate = _flat_plate(subsonic_condition)

    buildup = DragBuildup((plate,), reference_area_m2=10.0)

    expected = 2.0 * cf_turbulent(plate.reynolds, subsonic_condition.mach)
    assert buildup.cd0 == pytest.approx(expected)


def test_buildup_misc_terms_add_to_cd0(subsonic_condition: FlightCondition) -> None:
    plate = _flat_plate(subsonic_condition)
    clean = DragBuildup((plate,), reference_area_m2=10.0)

    with_misc = DragBuildup(
        (plate,),
        reference_area_m2=10.0,
        misc=(MiscDrag("a", 0.0010), MiscDrag("b", 0.0005)),
    )

    assert with_misc.cd0 - clean.cd0 == pytest.approx(0.0015)


def test_component_count_multiplies_drag_area(
    subsonic_condition: FlightCondition,
) -> None:
    single = _flat_plate(subsonic_condition, count=1)

    double = _flat_plate(subsonic_condition, count=2)

    assert double.drag_area_m2 == pytest.approx(2.0 * single.drag_area_m2)


def test_cruise_buildup_terms_and_sum(
    sample_geometry: AircraftGeometry, unit_settings: DragSettings
) -> None:
    buildup = aircraft_drag_buildup(
        sample_geometry, CRUISE_CONFIGURATION, unit_settings
    )

    components = [c.name for c in buildup.components]
    misc = {m.name: m.delta_cd for m in buildup.misc}
    assert components == [
        "wing",
        "horizontal_tail",
        "vertical_tail",
        "fuselage",
        "nacelle",
    ]
    assert list(misc) == [
        "wave_drag",
        "fuselage_upsweep",
        "fuselage_base",
        "te_flap",
        "le_flap",
        "excrescence_leakage",
    ]
    assert misc["te_flap"] == 0.0
    assert misc["le_flap"] == 0.0
    assert misc["wave_drag"] > 0.0
    assert buildup.cd0 == pytest.approx(buildup.cd_components + sum(misc.values()))


def test_buildup_excrescence_is_set_fraction_of_total(
    sample_geometry: AircraftGeometry, unit_settings: DragSettings
) -> None:
    buildup = aircraft_drag_buildup(
        sample_geometry, CRUISE_CONFIGURATION, unit_settings
    )

    leakage = {m.name: m.delta_cd for m in buildup.misc}["excrescence_leakage"]

    assert leakage == pytest.approx(unit_settings.excrescence_fraction * buildup.cd0)


def test_low_speed_gear_down_adds_gear_and_flap_terms(
    sample_geometry: AircraftGeometry,
    unit_settings: DragSettings,
    subsonic_condition: FlightCondition,
) -> None:
    configuration = _low_speed_configuration(subsonic_condition)

    buildup = aircraft_drag_buildup(sample_geometry, configuration, unit_settings)

    misc = {m.name: m.delta_cd for m in buildup.misc}
    assert misc["wave_drag"] == 0.0
    assert misc["te_flap"] > 0.0
    assert misc["le_flap"] > 0.0
    assert misc["main_gear"] > 0.0
    assert misc["nose_gear"] > 0.0


def test_wave_drag_above_critical_without_lift_coefficient_raises(
    sample_geometry: AircraftGeometry, unit_settings: DragSettings
) -> None:
    configuration = DragConfiguration("bad", CRUISE, None, 0.0, 0.0, gear_down=False)

    with pytest.raises(ValueError):
        aircraft_drag_buildup(sample_geometry, configuration, unit_settings)


def test_aircraft_buildup_uses_category_laminar_fractions(
    sample_geometry: AircraftGeometry, unit_settings: DragSettings
) -> None:
    buildup = aircraft_drag_buildup(
        sample_geometry, CRUISE_CONFIGURATION, unit_settings
    )

    fractions = {c.name: c.laminar_fraction for c in buildup.components}
    assert fractions == {
        "wing": 0.10,
        "horizontal_tail": 0.10,
        "vertical_tail": 0.10,
        "fuselage": 0.05,
        "nacelle": 0.05,
    }


def test_buildup_dataframe_has_misc_rows_and_full_share(
    subsonic_condition: FlightCondition,
) -> None:
    buildup = DragBuildup(
        (_flat_plate(subsonic_condition),),
        reference_area_m2=10.0,
        misc=(MiscDrag("wave_drag", 0.001),),
    )

    frame = buildup.to_dataframe()

    assert list(frame["component"]) == ["plate", "wave_drag", "total"]
    assert frame["cd0"].iloc[-1] == pytest.approx(buildup.cd0)
    assert frame["share_pct"].iloc[:-1].sum() == pytest.approx(100.0)


def test_buildup_zero_reference_area_raises(
    subsonic_condition: FlightCondition,
) -> None:
    plate = _flat_plate(subsonic_condition)

    with pytest.raises(ValueError):
        DragBuildup((plate,), reference_area_m2=0.0)


def test_misc_drag_negative_raises() -> None:
    with pytest.raises(ValueError):
        MiscDrag("bad", -0.001)


def test_plot_cd0_breakdown_writes_png(
    tmp_path: Path, subsonic_condition: FlightCondition
) -> None:
    buildup = DragBuildup(
        (_flat_plate(subsonic_condition),),
        reference_area_m2=10.0,
        misc=(MiscDrag("wave_drag", 0.001),),
    )

    out = plot_cd0_breakdown(buildup, tmp_path / "drag" / "cd0.png")

    assert out.is_file()
    assert out.stat().st_size > 0


def test_buildup_total_wetted_area_counts_every_item(
    subsonic_condition: FlightCondition,
) -> None:
    single = _flat_plate(subsonic_condition, count=1)
    double = _flat_plate(subsonic_condition, count=2)
    buildup = DragBuildup((single, double), reference_area_m2=10.0)

    frame = buildup.to_dataframe()

    # 20 m^2 per item: 1 x 20 + 2 x 20
    assert list(frame["total_wetted_area_m2"].iloc[:2]) == [20.0, 40.0]
    assert buildup.total_wetted_area_m2 == pytest.approx(60.0)
    assert frame["total_wetted_area_m2"].iloc[-1] == pytest.approx(60.0)


def test_grouped_contributions_sum_to_cd0(
    sample_geometry: AircraftGeometry,
    unit_settings: DragSettings,
    subsonic_condition: FlightCondition,
) -> None:
    buildup = aircraft_drag_buildup(
        sample_geometry, _low_speed_configuration(subsonic_condition), unit_settings
    )

    groups = grouped_contributions(buildup)

    assert sum(groups.values()) == pytest.approx(buildup.cd0)
    assert groups["landing gear"] > 0.0


def test_plot_cd0_comparison_writes_png(
    tmp_path: Path, sample_geometry: AircraftGeometry, unit_settings: DragSettings
) -> None:
    cruise = aircraft_drag_buildup(sample_geometry, CRUISE_CONFIGURATION, unit_settings)

    out = plot_cd0_comparison([("cruise", cruise)], tmp_path / "cd0_comparison.png")

    assert out.is_file()
    assert out.stat().st_size > 0
