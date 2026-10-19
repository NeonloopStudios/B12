# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""WP3.3a class II drag: CD0 build-up of the cruise, take-off and landing
configurations (ADSEE-II component drag build-up).

Run from wp3/:  python -m scripts.class2_drag  (or as a file / Run button)

Reads configs/geometry.yaml and the drag settings, and writes to
results/drag/:

    cd0_buildup_<configuration>.csv   one row per component and per term
    cd0_breakdown_<configuration>.png CD0 contribution of each term
    cd0_summary.csv                   CD0 and wetted area per configuration
    cd0_comparison.png                total CD0 per configuration, stacked
    cd0_report.txt                    everything this script prints

A configuration with unset inputs (config/mission.py) is skipped and named
in the log and the report.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # run as a file (e.g. the editor's Run button) instead of
    # `python -m scripts.class2_drag`: make wp3/src/ importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from b12wp3.config import mission  # noqa: E402
from b12wp3.config.drag import DRAG_SETTINGS, DragSettings  # noqa: E402
from b12wp3.config.geometry import AircraftGeometry, load_geometry  # noqa: E402
from b12wp3.config.mission import DragConfiguration  # noqa: E402
from b12wp3.config.paths import (  # noqa: E402
    CD0_COMPARISON_PNG,
    CD0_REPORT_TXT,
    CD0_SUMMARY_CSV,
    DRAG_DIR,
    WP3_DIR,
    drag_breakdown_png,
    drag_buildup_csv,
)
from b12wp3.config.yaml_io import ConfigError  # noqa: E402
from b12wp3.drag.buildup import DragBuildup, aircraft_drag_buildup  # noqa: E402
from b12wp3.plots.drag import (  # noqa: E402
    grouped_contributions,
    plot_cd0_breakdown,
    plot_cd0_comparison,
)

logger = logging.getLogger("scripts.class2_drag")

TABLE_COLUMNS = [
    "component",
    "count",
    "wetted_area_m2",
    "total_wetted_area_m2",
    "reference_length_m",
    "reynolds",
    "laminar_fraction",
    "cf",
    "form_factor",
    "interference_factor",
    "cd0",
    "share_pct",
]


def _configurations() -> tuple[list[DragConfiguration], list[str]]:
    """The configurations whose inputs are set, and a note for each skipped one."""
    configurations = [mission.CRUISE_CONFIGURATION]
    skipped: list[str] = []
    for build in (mission.takeoff_configuration, mission.landing_configuration):
        try:
            configurations.append(build())
        except ValueError as error:
            logger.warning("skipped: %s", error)
            skipped.append(str(error))
    return configurations, skipped


def _settings_section(settings: DragSettings) -> str:
    """The method inputs, as report text."""
    interference = settings.interference
    return "\n".join(
        [
            "METHOD INPUTS (b12wp3/config/drag.py)",
            "  laminar fraction     fuselage, nacelles "
            f"{settings.laminar.fuselage:.2f}; wing, tails "
            f"{settings.laminar.wing_and_tail:.2f}",
            f"  roughness k          {settings.roughness_m:.3e} m",
            f"  interference IF      wing {interference.wing}, HT "
            f"{interference.horizontal_tail}, VT {interference.vertical_tail}, "
            f"fuselage {interference.fuselage}, nacelle {interference.nacelle}",
            f"  Korn k_a             {settings.kappa_a}",
            f"  M_cr (normal)        {settings.wing_mach_critical_normal}, sweep line "
            f"x/c = {settings.wave_sweep_chord_fraction}",
            f"  F_flap               TE {settings.te_flap_factor}, "
            f"LE {settings.le_flap_factor}",
            f"  gear wheel-well fit  {settings.wheel_well_coeff}",
            f"  excrescence fraction {settings.excrescence_fraction:.3f} of total CD0",
        ]
    )


def _configuration_section(
    configuration: DragConfiguration, buildup: DragBuildup
) -> str:
    """One configuration: its condition and its build-up table, as report text."""
    condition = configuration.condition
    header = (
        f"CONFIGURATION: {configuration.name}\n"
        f"  altitude {condition.altitude_m:.0f} m, M = {condition.mach:.4f}, "
        f"V = {condition.velocity_m_s:.2f} m/s, cutoff regime "
        f"{condition.regime.value}\n"
        f"  flaps TE {configuration.te_flap_deflection_deg:g} deg, "
        f"LE {configuration.le_flap_deflection_deg:g} deg, gear "
        f"{'down' if configuration.gear_down else 'up'}\n"
        f"  CD0 = {buildup.cd0:.5f}   S_wet = {buildup.total_wetted_area_m2:.2f} m^2"
        f"   S_wet/S_ref = "
        f"{buildup.total_wetted_area_m2 / buildup.reference_area_m2:.3f}"
    )
    table = buildup.to_dataframe()[TABLE_COLUMNS].to_string(
        index=False, float_format=lambda value: f"{value:.4g}", na_rep="-"
    )
    return f"{header}\n\n{table}"


def _summary(
    results: list[tuple[DragConfiguration, DragBuildup]],
) -> pd.DataFrame:
    """CD0, wetted area and grouped contributions per configuration."""
    rows = []
    for configuration, buildup in results:
        row: dict[str, object] = {
            "configuration": configuration.name,
            "mach": configuration.condition.mach,
            "velocity_m_s": configuration.condition.velocity_m_s,
            "cd0": buildup.cd0,
            "total_wetted_area_m2": buildup.total_wetted_area_m2,
            "wetted_to_reference_ratio": (
                buildup.total_wetted_area_m2 / buildup.reference_area_m2
            ),
        }
        for group, value in grouped_contributions(buildup).items():
            row[f"cd0_{group.replace(' + ', '_').replace(' ', '_')}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _write_outputs(
    results: list[tuple[DragConfiguration, DragBuildup]],
) -> pd.DataFrame:
    """Write the per-configuration CSVs and charts, the summary and comparison."""
    DRAG_DIR.mkdir(parents=True, exist_ok=True)
    for configuration, buildup in results:
        buildup.to_dataframe().to_csv(drag_buildup_csv(configuration.name), index=False)
        plot_cd0_breakdown(
            buildup, drag_breakdown_png(configuration.name), configuration.name
        )
    summary = _summary(results)
    summary.to_csv(CD0_SUMMARY_CSV, index=False)
    plot_cd0_comparison([(c.name, b) for c, b in results], CD0_COMPARISON_PNG)
    return summary


def _report(
    geometry: AircraftGeometry,
    results: list[tuple[DragConfiguration, DragBuildup]],
    skipped: list[str],
    summary: pd.DataFrame,
) -> str:
    """The full text report: method inputs, every configuration, the summary."""
    sections = [
        "CLASS II DRAG (WP3.3a) - ADSEE-II component drag build-up\n"
        f"  S_ref = {geometry.wing.area_m2} m^2 (wing planform), "
        f"{geometry.engine_count} engines; geometry from configs/geometry.yaml",
        _settings_section(DRAG_SETTINGS),
    ]
    sections += [_configuration_section(c, b) for c, b in results]
    if skipped:
        sections.append(
            "SKIPPED CONFIGURATIONS\n" + "\n".join(f"  {s}" for s in skipped)
        )
    sections.append(
        "SUMMARY\n"
        + summary.to_string(index=False, float_format=lambda value: f"{value:.5g}")
    )
    files = [drag_buildup_csv(c.name) for c, _ in results]
    files += [drag_breakdown_png(c.name) for c, _ in results]
    files += [CD0_SUMMARY_CSV, CD0_COMPARISON_PNG, CD0_REPORT_TXT]
    # relative to wp3/: the report is shared, the absolute path is local
    sections.append(
        "FILES (relative to wp3/)\n"
        + "\n".join(f"  {path.relative_to(WP3_DIR).as_posix()}" for path in files)
    )
    return "\n\n".join(sections) + "\n"


def main() -> int:
    """Run the class II drag build-up of every configuration and write the results.

    Returns:
        Process exit code: 0 on success, 1 on an input error.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        geometry = load_geometry()
    except (ConfigError, ValueError) as error:
        logger.error("%s", error)
        return 1
    configurations, skipped = _configurations()
    results = [
        (c, aircraft_drag_buildup(geometry, c, DRAG_SETTINGS)) for c in configurations
    ]
    summary = _write_outputs(results)
    report = _report(geometry, results, skipped, summary)
    CD0_REPORT_TXT.write_text(report, encoding="utf-8")
    logger.info("report:\n%s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
