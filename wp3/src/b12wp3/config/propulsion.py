# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Engine inputs of the nacelle sizing, loaded from wp3/configs/propulsion.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from b12wp3.config.paths import PROPULSION_YAML
from b12wp3.config.yaml_io import ConfigError, build_section, read_mapping

# --- Table 8.1, nacelle Type B: the fan cowl covers the whole nacelle ---
TYPE_B_LENGTH_COEFF: Final[float] = 9.8  # C_l [-]
TYPE_B_LENGTH_OFFSET_M: Final[float] = 0.05  # Delta l [m]
TYPE_B_MAX_DIAMETER_POSITION: Final[float] = 0.35  # beta [-]
TYPE_B_FAN_COWL_RATIO: Final[float] = 1.0  # phi = l_f / l_n [-]


@dataclass(frozen=True)
class EngineInputs:
    """Engine parameters of the Torenbeek mass-flow equation (8.22).

    Attributes:
        takeoff_thrust_per_engine_n: Take-off thrust of one engine [N].
        bypass_ratio: Bypass ratio B [-].
        turbine_inlet_temperature_k: Turbine inlet temperature T_t4 [K].
        nozzle_efficiency: Mean nozzle efficiency eta_noz [-].
        turbine_fan_efficiency: Combined turbine and fan efficiency eta_tf [-].
    """

    takeoff_thrust_per_engine_n: float
    bypass_ratio: float
    turbine_inlet_temperature_k: float
    nozzle_efficiency: float
    turbine_fan_efficiency: float

    def __post_init__(self) -> None:
        if self.takeoff_thrust_per_engine_n <= 0.0:
            raise ValueError(
                "takeoff_thrust_per_engine_n must be > 0, "
                f"got {self.takeoff_thrust_per_engine_n}"
            )
        if self.bypass_ratio < 0.0:
            raise ValueError(f"bypass_ratio must be >= 0, got {self.bypass_ratio}")
        if self.turbine_inlet_temperature_k <= 750.0:
            raise ValueError(
                "turbine_inlet_temperature_k must be > 750 K (G > 0), "
                f"got {self.turbine_inlet_temperature_k}"
            )
        for name, value in (
            ("nozzle_efficiency", self.nozzle_efficiency),
            ("turbine_fan_efficiency", self.turbine_fan_efficiency),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")


@dataclass(frozen=True)
class EngineReference:
    """Datasheet dimensions of the selected engine, for comparison only.

    Attributes:
        fan_diameter_m: Fan diameter [m].
        nacelle_length_m: Nacelle length [m].
    """

    fan_diameter_m: float
    nacelle_length_m: float

    def __post_init__(self) -> None:
        for name, value in (
            ("fan_diameter_m", self.fan_diameter_m),
            ("nacelle_length_m", self.nacelle_length_m),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class PropulsionInputs:
    """Everything in propulsion.yaml.

    Attributes:
        engine: Engine parameters of the sizing.
        reference: Datasheet dimensions for comparison.
    """

    engine: EngineInputs
    reference: EngineReference


def load_propulsion(path: Path = PROPULSION_YAML) -> PropulsionInputs:
    """Load, validate and freeze the engine inputs.

    Args:
        path: Propulsion YAML file. Default: wp3/configs/propulsion.yaml.

    Returns:
        The validated propulsion inputs.

    Raises:
        ConfigError: If the file cannot be read, has a wrong structure or
            type, or has null inputs (all named in one error).
        ValueError: If a set input is not physical.
    """
    raw = read_mapping(path, ("engine", "reference"))
    missing: list[str] = []
    engine = build_section(raw, "engine", EngineInputs, missing)
    reference = build_section(raw, "reference", EngineReference, missing)
    if missing or engine is None or reference is None:
        raise ConfigError(
            f"propulsion inputs not set (null) in {path}: " + ", ".join(missing)
        )
    return PropulsionInputs(engine=engine, reference=reference)
