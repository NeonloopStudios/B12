# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""WP3.1 nacelle sizing: Torenbeek Type B nacelle of the selected engine.

Run from wp3/:  python -m scripts.nacelle_sizing

Reads configs/propulsion.yaml, sizes one nacelle (textbook equations
8.22-8.30), compares it with the engine datasheet, and writes
results/propulsion/nacelle.csv and nacelle.png. It logs the nacelle block
for configs/geometry.yaml.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from b12wp3.config.paths import NACELLE_CSV, NACELLE_PNG
from b12wp3.config.propulsion import load_propulsion
from b12wp3.config.yaml_io import ConfigError
from b12wp3.plots.nacelle import plot_nacelle_outline
from b12wp3.propulsion.nacelle import (
    NacelleGeometry,
    gas_generator_function,
    size_type_b_nacelle,
)

logger = logging.getLogger("scripts.nacelle_sizing")


def _result_table(
    nacelle: NacelleGeometry, gas_generator: float, fan_m: float, length_m: float
) -> pd.DataFrame:
    """One row per quantity: value, unit and datasheet comparison."""
    rows = [
        ("gas_generator_function_G", gas_generator, "-", None),
        ("mass_flow_per_engine", nacelle.mass_flow_kg_s, "kg/s", None),
        ("spinner_ratio_Ds_Di", nacelle.spinner_ratio, "-", None),
        ("spinner_diameter_Ds", nacelle.spinner_diameter_m, "m", None),
        ("inlet_diameter_Di", nacelle.inlet_diameter_m, "m", fan_m),
        ("highlight_diameter_Dh", nacelle.highlight_diameter_m, "m", None),
        ("nacelle_length_ln", nacelle.length_m, "m", length_m),
        ("fan_cowl_length_lf", nacelle.fan_cowl_length_m, "m", None),
        ("max_diameter_Dn", nacelle.max_diameter_m, "m", None),
        ("max_diameter_position_beta", nacelle.max_diameter_position, "-", None),
        ("exit_diameter_Def", nacelle.exit_diameter_m, "m", None),
    ]
    frame = pd.DataFrame(rows, columns=["quantity", "value", "unit", "datasheet"])
    frame["ratio_to_datasheet"] = frame["value"] / frame["datasheet"]
    return frame


def _geometry_block(nacelle: NacelleGeometry) -> str:
    """The nacelle section for configs/geometry.yaml."""
    return (
        "nacelle:\n"
        f"  length_m: {nacelle.length_m:.4f}\n"
        f"  highlight_diameter_m: {nacelle.highlight_diameter_m:.4f}\n"
        f"  max_diameter_m: {nacelle.max_diameter_m:.4f}\n"
        f"  max_diameter_position: {nacelle.max_diameter_position}\n"
        f"  exit_diameter_m: {nacelle.exit_diameter_m:.4f}\n"
    )


def main() -> int:
    """Size the nacelle and write the results.

    Returns:
        Process exit code: 0 on success, 1 on an input error.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        inputs = load_propulsion()
    except (ConfigError, ValueError) as error:
        logger.error("%s", error)
        return 1
    engine = inputs.engine
    nacelle = size_type_b_nacelle(engine)
    table = _result_table(
        nacelle,
        gas_generator_function(engine.turbine_inlet_temperature_k),
        inputs.reference.fan_diameter_m,
        inputs.reference.nacelle_length_m,
    )
    NACELLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(NACELLE_CSV, index=False)
    plot_nacelle_outline(nacelle, NACELLE_PNG)
    logger.info("wrote %s and %s", NACELLE_CSV, NACELLE_PNG)
    logger.info("results:\n%s", table.to_string(index=False))
    logger.info("block for configs/geometry.yaml:\n%s", _geometry_block(nacelle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
