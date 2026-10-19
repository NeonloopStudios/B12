# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Turbofan nacelle sizing after Torenbeek (textbook equations 8.22-8.30).

    G      = T_t4 / 600 - 1.25                                           (8.23)
    m_dot  = T_TO/(N_e a0) (1+B) / sqrt(5 eta_noz G (1 + eta_tf B))     (8.22)
    D_s/D_i = 0.05 (1 + 0.1 rho0 a0 / m_dot + 3B/(1+B))                 (8.24)
    D_i    = 1.65 sqrt((m_dot/(rho0 a0) + 0.005) / (1 - (D_s/D_i)^2))   (8.25)
    D_h    = D_i                                                         (8.26)
    l_n    = C_l (sqrt(m_dot/(rho0 a0) (1+0.2B)/(1+B)) + Delta l)        (8.27)
    l_f    = phi l_n                                                     (8.28)
    D_n    = D_i + 0.06 phi l_n + 0.03                                   (8.29)
    D_ef   = D_n (1 - phi^2 / 3)                                         (8.30)

T_TO / N_e is the take-off thrust of one engine, so m_dot is the mass flow of
one engine. Lengths and diameters are in metres; the equations are
dimensional (SI). Only the Type B nacelle is sized (phi = 1): the fan cowl
covers the whole nacelle and there is no exposed core cowling. The cone
behind the mixed nozzle is not part of the outer geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from b12wp3.config.atmosphere import isa_atmosphere
from b12wp3.config.propulsion import (
    TYPE_B_FAN_COWL_RATIO,
    TYPE_B_LENGTH_COEFF,
    TYPE_B_LENGTH_OFFSET_M,
    TYPE_B_MAX_DIAMETER_POSITION,
    EngineInputs,
)
from b12wp3.drag.wetted_area import nacelle_wetted_area

G_TEMPERATURE_DIVISOR: Final[float] = 600.0
G_OFFSET: Final[float] = 1.25
SPINNER_COEFF: Final[float] = 0.05
INLET_COEFF: Final[float] = 1.65
INLET_AREA_OFFSET_M2: Final[float] = 0.005
DIAMETER_LENGTH_COEFF: Final[float] = 0.06
DIAMETER_OFFSET_M: Final[float] = 0.03


@dataclass(frozen=True)
class NacelleGeometry:
    """Result of the Type B nacelle sizing.

    Attributes:
        mass_flow_kg_s: Mass flow of one engine [kg/s].
        spinner_ratio: D_s / D_i [-].
        inlet_diameter_m: Inlet diameter D_i [m].
        highlight_diameter_m: Highlight diameter D_h [m].
        length_m: Nacelle length l_n, without the cone [m].
        fan_cowl_length_m: Fan cowl length l_f [m].
        max_diameter_m: Maximum diameter D_n [m].
        max_diameter_position: beta: D_n sits at beta * l_f [-].
        exit_diameter_m: Fan cowl exit diameter D_ef [m].
    """

    mass_flow_kg_s: float
    spinner_ratio: float
    inlet_diameter_m: float
    highlight_diameter_m: float
    length_m: float
    fan_cowl_length_m: float
    max_diameter_m: float
    max_diameter_position: float
    exit_diameter_m: float

    @property
    def spinner_diameter_m(self) -> float:
        """Spinner diameter D_s [m]."""
        return self.spinner_ratio * self.inlet_diameter_m

    @property
    def wetted_area_m2(self) -> float:
        """Outer wetted area of one nacelle, the same two frustums the drag
        build-up uses (b12wp3.drag.wetted_area.nacelle_wetted_area) [m^2]."""
        return nacelle_wetted_area(
            self.length_m,
            self.highlight_diameter_m,
            self.max_diameter_m,
            self.max_diameter_position,
            self.exit_diameter_m,
        )


def gas_generator_function(turbine_inlet_temperature_k: float) -> float:
    """Gas generator function G = T_t4 / 600 - 1.25 (8.23).

    Raises:
        ValueError: If G is not positive (T_t4 <= 750 K).
    """
    g = turbine_inlet_temperature_k / G_TEMPERATURE_DIVISOR - G_OFFSET
    if g <= 0.0:
        raise ValueError(
            f"G = {g} <= 0 for T_t4 = {turbine_inlet_temperature_k} K; need > 750 K"
        )
    return g


def mass_flow(
    thrust_per_engine_n: float,
    speed_of_sound_m_s: float,
    bypass_ratio: float,
    nozzle_efficiency: float,
    gas_generator: float,
    turbine_fan_efficiency: float,
) -> float:
    """Mass flow of one engine at take-off (8.22) [kg/s].

    Args:
        thrust_per_engine_n: Take-off thrust of one engine T_TO / N_e [N].
        speed_of_sound_m_s: Sea-level speed of sound a0 [m/s].
        bypass_ratio: B [-].
        nozzle_efficiency: eta_noz [-].
        gas_generator: G [-].
        turbine_fan_efficiency: eta_tf [-].
    """
    for name, value in (
        ("thrust_per_engine_n", thrust_per_engine_n),
        ("speed_of_sound_m_s", speed_of_sound_m_s),
        ("nozzle_efficiency", nozzle_efficiency),
        ("gas_generator", gas_generator),
        ("turbine_fan_efficiency", turbine_fan_efficiency),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if bypass_ratio < 0.0:
        raise ValueError(f"bypass_ratio must be >= 0, got {bypass_ratio}")
    denominator = math.sqrt(
        5.0
        * nozzle_efficiency
        * gas_generator
        * (1.0 + turbine_fan_efficiency * bypass_ratio)
    )
    return thrust_per_engine_n / speed_of_sound_m_s * (1.0 + bypass_ratio) / denominator


def _flow_area(mass_flow_kg_s: float, density: float, speed_of_sound: float) -> float:
    """m_dot / (rho0 a0) [m^2]."""
    if mass_flow_kg_s <= 0.0 or density <= 0.0 or speed_of_sound <= 0.0:
        raise ValueError(
            "need m_dot, rho0, a0 > 0, "
            f"got {mass_flow_kg_s}, {density}, {speed_of_sound}"
        )
    return mass_flow_kg_s / (density * speed_of_sound)


def spinner_ratio(
    mass_flow_kg_s: float, density: float, speed_of_sound: float, bypass_ratio: float
) -> float:
    """Spinner-to-inlet diameter ratio D_s / D_i (8.24) [-]."""
    area = _flow_area(mass_flow_kg_s, density, speed_of_sound)
    return SPINNER_COEFF * (
        1.0 + 0.1 / area + 3.0 * bypass_ratio / (1.0 + bypass_ratio)
    )


def inlet_diameter(
    mass_flow_kg_s: float, density: float, speed_of_sound: float, spinner: float
) -> float:
    """Inlet diameter D_i (8.25) [m].

    Raises:
        ValueError: If D_s / D_i is not in [0, 1).
    """
    if not 0.0 <= spinner < 1.0:
        raise ValueError(f"spinner ratio must be in [0, 1), got {spinner}")
    area = _flow_area(mass_flow_kg_s, density, speed_of_sound)
    return INLET_COEFF * math.sqrt((area + INLET_AREA_OFFSET_M2) / (1.0 - spinner**2))


def nacelle_length(
    mass_flow_kg_s: float,
    density: float,
    speed_of_sound: float,
    bypass_ratio: float,
    length_coeff: float,
    length_offset_m: float,
) -> float:
    """Nacelle length without the cone, l_n (8.27) [m]."""
    area = _flow_area(mass_flow_kg_s, density, speed_of_sound)
    core_term = math.sqrt(area * (1.0 + 0.2 * bypass_ratio) / (1.0 + bypass_ratio))
    return length_coeff * (core_term + length_offset_m)


def max_diameter(
    inlet_diameter_m: float, fan_cowl_ratio: float, length_m: float
) -> float:
    """Maximum nacelle diameter D_n = D_i + 0.06 phi l_n + 0.03 (8.29) [m]."""
    return (
        inlet_diameter_m
        + DIAMETER_LENGTH_COEFF * fan_cowl_ratio * length_m
        + DIAMETER_OFFSET_M
    )


def fan_exit_diameter(max_diameter_m: float, fan_cowl_ratio: float) -> float:
    """Fan cowl exit diameter D_ef = D_n (1 - phi^2 / 3) (8.30) [m]."""
    if not 0.0 < fan_cowl_ratio <= 1.0:
        raise ValueError(f"fan_cowl_ratio must be in (0, 1], got {fan_cowl_ratio}")
    return max_diameter_m * (1.0 - fan_cowl_ratio**2 / 3.0)


def size_type_b_nacelle(engine: EngineInputs) -> NacelleGeometry:
    """Size one Type B nacelle at sea-level ISA take-off conditions.

    Args:
        engine: Engine parameters.

    Returns:
        The nacelle geometry.
    """
    _, a0, rho0 = isa_atmosphere(0.0)
    bypass = engine.bypass_ratio
    flow = mass_flow(
        engine.takeoff_thrust_per_engine_n,
        a0,
        bypass,
        engine.nozzle_efficiency,
        gas_generator_function(engine.turbine_inlet_temperature_k),
        engine.turbine_fan_efficiency,
    )
    spinner = spinner_ratio(flow, rho0, a0, bypass)
    d_i = inlet_diameter(flow, rho0, a0, spinner)
    l_n = nacelle_length(
        flow, rho0, a0, bypass, TYPE_B_LENGTH_COEFF, TYPE_B_LENGTH_OFFSET_M
    )
    phi = TYPE_B_FAN_COWL_RATIO
    d_n = max_diameter(d_i, phi, l_n)
    return NacelleGeometry(
        mass_flow_kg_s=flow,
        spinner_ratio=spinner,
        inlet_diameter_m=d_i,
        highlight_diameter_m=d_i,
        length_m=l_n,
        fan_cowl_length_m=phi * l_n,
        max_diameter_m=d_n,
        max_diameter_position=TYPE_B_MAX_DIAMETER_POSITION,
        exit_diameter_m=fan_exit_diameter(d_n, phi),
    )
