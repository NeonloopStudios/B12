# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Tests for wp3/src/b12wp3/vsp/: layout (stage 1), model JSON, airfoil reader.

Expected values follow from the shape definitions in the layout module
docstring: elliptic nose, parabolic tail-cone centreline at mean angle u,
Torenbeek Type B nacelle outline through D_h, D_n (at beta l_n) and D_ef.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from b12wp3.config.geometry import AircraftGeometry, BodyInputs
from b12wp3.config.vsp_model import VspModelInputs
from b12wp3.vsp import layout
from b12wp3.vsp.airfoil import read_airfoil_dat
from b12wp3.vsp.model import AircraftModel

# ============================================================
#  HELPERS
# ============================================================

# fuselage of the property test: hypothesis cannot take function-scope fixtures
_PROPERTY_BODY = BodyInputs(4.0, 20.0, 8.0, 3.0)


def _airfoil() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric 10 % thick test section (upper, lower), LE to TE."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, 21, dtype=np.float64)))
    y = 0.5 * np.sqrt(np.clip(x * (1.0 - x), 0.0, None)) * 0.2
    return np.column_stack((x, y)), np.column_stack((x, -y))


def _profile(
    geometry: AircraftGeometry, inputs: VspModelInputs
) -> layout.FuselageProfile:
    """Fuselage profile of the sample aircraft."""
    return layout.fuselage_profile(geometry, inputs.fuselage_shape)


def _frame(geometry: AircraftGeometry, inputs: VspModelInputs) -> layout.WingFrame:
    """Wing frame of the sample aircraft."""
    return layout.WingFrame(geometry.wing.planform(), inputs.wing)


# ============================================================
#  FUSELAGE
# ============================================================


def test_fuselage_profile_key_stations_match_definition(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    profile = _profile(sample_geometry, sample_vsp_inputs)
    body = sample_geometry.fuselage
    x = np.array([0.0, body.nose_length_m, profile.tail_start_m, body.length_m])

    result = (profile.radius(x), profile.centre_z(x))

    upsweep = math.radians(sample_geometry.fuselage_aft.upsweep_deg)
    big_r = 0.5 * body.diameter_m
    r_base = math.sqrt(sample_geometry.fuselage_aft.base_area_m2 / math.pi)
    assert result[0] == pytest.approx([0.0, big_r, big_r, r_base])
    assert result[1] == pytest.approx(
        [
            sample_vsp_inputs.fuselage_shape.nose_tip_z_m,
            0.0,
            0.0,
            body.tail_length_m * math.tan(upsweep),
        ]
    )


def test_fuselage_tail_cone_upper_line_stays_below_cylinder_top(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    profile = _profile(sample_geometry, sample_vsp_inputs)
    x = np.linspace(profile.tail_start_m, sample_geometry.fuselage.length_m, 200)

    result = profile.centre_z(x) + profile.radius(x)

    # no hump above the cylinder: the reason for the parabolic centreline
    assert np.all(result <= profile.radius_m + 1.0e-12)


@given(fraction=st.floats(min_value=0.05, max_value=0.98))
def test_fuselage_slopes_match_finite_differences(fraction: float) -> None:
    profile = layout.FuselageProfile(_PROPERTY_BODY, math.radians(5.0), 0.1, -0.3)
    x = np.array([fraction * _PROPERTY_BODY.length_m], dtype=np.float64)
    h = 1.0e-6

    dz, dr = profile.slopes(x)

    num_dz = (profile.centre_z(x + h) - profile.centre_z(x - h)) / (2.0 * h)
    num_dr = (profile.radius(x + h) - profile.radius(x - h)) / (2.0 * h)
    assert (dz[0], dr[0]) == pytest.approx((num_dz[0], num_dr[0]), rel=1e-4, abs=1e-4)


def test_fuselage_sections_start_at_point_and_increase_in_x(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    profile = _profile(sample_geometry, sample_vsp_inputs)

    result = layout.fuselage_sections(profile, sample_vsp_inputs.fuselage_shape)

    x = [s.x_m for s in result]
    shape = sample_vsp_inputs.fuselage_shape
    assert result[0].diameter_m == 0.0
    assert x[-1] == pytest.approx(sample_geometry.fuselage.length_m)
    assert all(b > a for a, b in zip(x, x[1:]))
    assert len(result) == shape.nose_sections + 1 + 1 + shape.tail_sections


def test_fuselage_profile_zero_tail_length_raises(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    body = dataclasses.replace(sample_geometry.fuselage, tail_length_m=0.0)
    geometry = dataclasses.replace(sample_geometry, fuselage=body)

    with pytest.raises(ValueError, match="tail cone"):
        _profile(geometry, sample_vsp_inputs)


# ============================================================
#  WING AND TAILS
# ============================================================


def test_wing_frame_root_station_returns_root_values(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    frame = _frame(sample_geometry, sample_vsp_inputs)
    wing = sample_vsp_inputs.wing

    result = (
        frame.le_x(0.0),
        frame.chord(0.0),
        frame.chord_line_z(wing.root_le_x_m, 0.0),
    )

    assert result == pytest.approx(
        (wing.root_le_x_m, frame.planform.root_chord_m, wing.root_z_m)
    )


def test_tail_surfaces_mac_quarter_chords_are_tail_arms_behind_cg(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    profile = _profile(sample_geometry, sample_vsp_inputs)

    vertical, horizontal = layout.tail_surfaces(
        sample_geometry, sample_vsp_inputs, profile, []
    )

    x_cg = sample_vsp_inputs.centre_of_gravity.x_m
    vt_offset = sample_geometry.vertical_tail.planform().mac_quarter_chord_offset_m
    ht_offset = sample_geometry.horizontal_tail.planform().mac_quarter_chord_offset_m
    result = (
        vertical.root_le[0] + vt_offset - x_cg,
        horizontal.root_le[0] + ht_offset - x_cg,
        vertical.symmetric,
        horizontal.symmetric,
    )
    tails = sample_vsp_inputs.tails
    assert result == (
        pytest.approx(tails.vertical_arm_m),
        pytest.approx(tails.horizontal_arm_m),
        False,
        True,
    )


def test_build_model_cg_mac_fraction_matches_lemac_definition(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    wing = sample_geometry.wing.planform()

    _, report = layout.build_model(sample_geometry, sample_vsp_inputs, _airfoil())

    # x_LEMAC = x_root_LE + y_mac tan(sweep_LE), y_mac = (b_p/3)(1+2 lam)/(1+lam)
    lam = wing.taper
    y_mac = wing.panel_span_m / 3.0 * (1.0 + 2.0 * lam) / (1.0 + lam)
    x_lemac = sample_vsp_inputs.wing.root_le_x_m + y_mac * math.tan(wing.sweep_at(0.0))
    expected = (sample_vsp_inputs.centre_of_gravity.x_m - x_lemac) / wing.mac_m
    assert report.cg_mac_fraction == pytest.approx(expected)
    assert 0.0 < report.cg_mac_fraction < 1.0
    assert not any("c.g." in w for w in report.warnings)


def test_build_model_cg_ahead_of_mac_warns(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    inputs = dataclasses.replace(
        sample_vsp_inputs,
        centre_of_gravity=dataclasses.replace(
            sample_vsp_inputs.centre_of_gravity, x_m=10.0
        ),
    )

    _, report = layout.build_model(sample_geometry, inputs, _airfoil())

    assert report.cg_mac_fraction < 0.0
    assert any("outside the wing MAC" in w for w in report.warnings)


# ============================================================
#  NACELLE
# ============================================================


def test_nacelle_profile_outer_line_passes_torenbeek_diameters(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    nacelle = sample_geometry.nacelle

    result = layout.nacelle_profile(nacelle, sample_vsp_inputs.nacelle_detail)

    x_max = nacelle.max_diameter_position * nacelle.length_m
    assert 2.0 * result.outer[0] == pytest.approx(nacelle.highlight_diameter_m)
    assert 2.0 * result.outer[-1] == pytest.approx(nacelle.exit_diameter_m)
    assert 2.0 * result.outer.max() == pytest.approx(nacelle.max_diameter_m, rel=1e-3)
    assert result.x[np.argmax(result.outer)] == pytest.approx(x_max, abs=0.1)
    assert np.all(result.inner[1:] < result.outer[1:])


def test_ring_section_chord_frame_round_trip_returns_profile(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    profile = layout.nacelle_profile(
        sample_geometry.nacelle, sample_vsp_inputs.nacelle_detail
    )

    chord, angle_deg, upper, lower = layout.ring_section(profile)

    # rotate back: dx = c (u cos a + v sin a), dr = c (v cos a - u sin a)
    a = math.radians(angle_deg)
    x_back = chord * (upper[:, 0] * math.cos(a) + upper[:, 1] * math.sin(a))
    r_back = profile.outer[0] + chord * (
        upper[:, 1] * math.cos(a) - upper[:, 0] * math.sin(a)
    )
    te_mid = 0.5 * (upper[-1] + lower[-1])
    assert x_back == pytest.approx(profile.x)
    assert r_back == pytest.approx(profile.outer)
    assert te_mid == pytest.approx([1.0, 0.0], abs=1e-12)


def test_nacelle_parts_inside_fuselage_raises(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    position = dataclasses.replace(sample_vsp_inputs.nacelle_position, axis_y_m=1.6)
    inputs = dataclasses.replace(sample_vsp_inputs, nacelle_position=position)

    with pytest.raises(ValueError, match="fuselage side"):
        layout.nacelle_parts(
            sample_geometry, inputs, _frame(sample_geometry, inputs), 1.5
        )


def test_nacelle_parts_four_engines_raises(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    geometry = dataclasses.replace(sample_geometry, engine_count=4)

    with pytest.raises(ValueError, match="engine_count 4"):
        layout.nacelle_parts(
            geometry, sample_vsp_inputs, _frame(geometry, sample_vsp_inputs), 1.5
        )


def test_pylon_tip_meets_wing_chord_line(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    frame = _frame(sample_geometry, sample_vsp_inputs)
    parts = layout.nacelle_parts(sample_geometry, sample_vsp_inputs, frame, 1.5)
    pylon = parts.pylon

    tip_x = pylon.root_le[0] + pylon.span_m * math.tan(math.radians(pylon.sweep_deg))
    tip_z = pylon.root_le[2] + pylon.span_m

    assert tip_z == pytest.approx(frame.chord_line_z(tip_x, pylon.root_le[1]))


# ============================================================
#  LANDING GEAR
# ============================================================


@pytest.mark.parametrize(("track", "mount"), [(2.0, "fuselage"), (4.4, "wing")])
def test_gear_parts_track_selects_mount(
    sample_geometry: AircraftGeometry,
    sample_vsp_inputs: VspModelInputs,
    track: float,
    mount: str,
) -> None:
    main = dataclasses.replace(sample_vsp_inputs.main_gear_position, track_m=track)
    inputs = dataclasses.replace(sample_vsp_inputs, main_gear_position=main)

    result = layout.gear_parts(
        sample_geometry,
        inputs,
        _frame(sample_geometry, inputs),
        _profile(sample_geometry, inputs),
    )

    assert result.main_mount == mount


def test_gear_parts_axles_sit_one_tire_radius_above_ground(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    result = layout.gear_parts(
        sample_geometry,
        sample_vsp_inputs,
        _frame(sample_geometry, sample_vsp_inputs),
        _profile(sample_geometry, sample_vsp_inputs),
    )

    ground = -1.5 - sample_vsp_inputs.ground.fuselage_clearance_m
    nose_d = sample_geometry.nose_gear.tire_diameter_m
    main_d = sample_geometry.main_gear.tire_diameter_m
    assert result.gear.ground_z_m == pytest.approx(ground)
    assert result.nose_strut.bottom[2] == pytest.approx(ground + 0.5 * nose_d)
    assert result.main_strut.bottom[2] == pytest.approx(ground + 0.5 * main_d)
    assert result.gear.bogies[1].tire_width_m == pytest.approx(
        sample_geometry.main_gear.tire_width_m / 2.0
    )


def test_gear_parts_no_clearance_for_tire_raises(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    inputs = dataclasses.replace(
        sample_vsp_inputs,
        ground=dataclasses.replace(sample_vsp_inputs.ground, fuselage_clearance_m=0.05),
    )

    with pytest.raises(ValueError, match="fuselage_clearance_m"):
        layout.gear_parts(
            sample_geometry,
            inputs,
            _frame(sample_geometry, inputs),
            _profile(sample_geometry, inputs),
        )


# ============================================================
#  WHOLE MODEL AND JSON
# ============================================================


def test_build_model_sample_aircraft_returns_every_component(
    sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    model, report = layout.build_model(sample_geometry, sample_vsp_inputs, _airfoil())

    names = [
        c.name
        for group in (
            model.bodies,
            model.surfaces,
            model.rings,
            model.ellipsoids,
            model.struts,
        )
        for c in group
    ]
    assert names == [
        "Fuselage",
        "Exhaust plug",
        "Wing",
        "Vertical tail",
        "Horizontal tail",
        "Pylon",
        "Nacelle",
        "Fan face",
        "Spinner",
        "Nose gear strut",
        "Main gear strut",
    ]
    assert report.main_gear_mount == "wing"


def test_model_json_round_trip_returns_equal_model(
    tmp_path: Path, sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    model, _ = layout.build_model(sample_geometry, sample_vsp_inputs, _airfoil())
    path = tmp_path / "model.json"

    model.write_json(path)
    result = AircraftModel.read_json(path)

    assert result == model


def test_model_read_json_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text('{"bodies": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="keys"):
        AircraftModel.read_json(path)


# ============================================================
#  AIRFOIL FILE
# ============================================================


def _selig_text(upper: NDArray[np.float64], lower: NDArray[np.float64]) -> str:
    """Selig loop TE -> upper -> LE -> lower -> TE with a name line."""
    loop = np.vstack((upper[::-1], lower[1:]))
    return "TEST\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in loop) + "\n"


def test_read_airfoil_dat_selig_returns_surfaces_le_to_te(tmp_path: Path) -> None:
    upper, lower = _airfoil()
    path = tmp_path / "test.dat"
    path.write_text(_selig_text(upper, lower), encoding="utf-8")

    result_upper, result_lower = read_airfoil_dat(path)

    assert result_upper == pytest.approx(upper, abs=1e-6)
    assert result_lower == pytest.approx(lower, abs=1e-6)


def test_read_airfoil_dat_lednicer_returns_surfaces_le_to_te(tmp_path: Path) -> None:
    upper, lower = _airfoil()
    body = "\n".join(f"{x:.6f} {y:.6f}" for x, y in np.vstack((upper, lower)))
    path = tmp_path / "test.dat"
    path.write_text(f"TEST\n{len(upper)}. {len(lower)}.\n{body}\n", encoding="utf-8")

    result_upper, result_lower = read_airfoil_dat(path)

    assert result_upper == pytest.approx(upper, abs=1e-6)
    assert result_lower == pytest.approx(lower, abs=1e-6)


def test_read_airfoil_dat_too_few_points_raises(tmp_path: Path) -> None:
    path = tmp_path / "short.dat"
    path.write_text("1.0 0.0\n0.0 0.0\n1.0 0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="points"):
        read_airfoil_dat(path)
