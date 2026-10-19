# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""WP3.1 nacelle sizing: Torenbeek Type B nacelle of the selected engine.

Run from wp3/:  python -m scripts.nacelle_sizing  (or as a file / Run button)

Reads configs/propulsion.yaml, sizes one nacelle (textbook equations
8.22-8.30), compares it with the engine datasheet, and writes
results/propulsion/nacelle.csv, nacelle.png and nacelle_report.txt (everything
it prints, including the nacelle block for configs/geometry.yaml).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # run as a file (e.g. the editor's Run button) instead of
    # `python -m scripts.nacelle_sizing`: make wp3/src/ importable so
    # `from b12wp3 import ...` resolves (same as wp2's OpenVSP scripts)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from b12wp3.config.paths import (  # noqa: E402
    NACELLE_CSV,
    NACELLE_PNG,
    NACELLE_REPORT_TXT,
    WP3_DIR,
)
from b12wp3.config.propulsion import EngineInputs, load_propulsion  # noqa: E402
from b12wp3.config.yaml_io import ConfigError  # noqa: E402
from b12wp3.plots.nacelle import plot_nacelle_outline  # noqa: E402
from b12wp3.propulsion.nacelle import (  # noqa: E402
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
        ("wetted_area_one_nacelle", nacelle.wetted_area_m2, "m^2", None),
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


def _report(engine: EngineInputs, table: pd.DataFrame, nacelle: NacelleGeometry) -> str:
    """The full text report: inputs, results, geometry.yaml block, files."""
    files = (NACELLE_CSV, NACELLE_PNG, NACELLE_REPORT_TXT)
    sections = [
        "NACELLE SIZING (WP3.1) - Torenbeek, Type B nacelle, textbook eq. 8.22-8.30",
        "INPUTS (configs/propulsion.yaml)\n"
        f"  take-off thrust per engine  {engine.takeoff_thrust_per_engine_n:.0f} N\n"
        f"  bypass ratio B              {engine.bypass_ratio}\n"
        f"  turbine inlet temp. T_t4    {engine.turbine_inlet_temperature_k} K\n"
        f"  nozzle efficiency           {engine.nozzle_efficiency}\n"
        f"  turbine x fan efficiency    {engine.turbine_fan_efficiency}",
        "RESULTS\n" + table.to_string(index=False, na_rep="-"),
        "BLOCK FOR configs/geometry.yaml\n" + _geometry_block(nacelle),
        # relative to wp3/: the report is shared, the absolute path is local
        "FILES (relative to wp3/)\n"
        + "\n".join(f"  {path.relative_to(WP3_DIR).as_posix()}" for path in files),
    ]
    return "\n\n".join(sections) + "\n"


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
    report = _report(engine, table, nacelle)
    NACELLE_REPORT_TXT.write_text(report, encoding="utf-8")
    logger.info("report:\n%s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
