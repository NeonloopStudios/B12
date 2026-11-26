# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-24
"""WP4.1 script 2: weights of the wing, per load case.

Writes out/wp41/<case>/weight.csv and point_loads.csv (wp41.contracts).
Design: wp4/README_41.md, section 6.

    python wp4/scripts/wp41_weight.py --case LC-1 [LC-2 ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.aircraft import ENGINE, MASSES, WING  # noqa: E402
from configs.wp41 import settings  # noqa: E402
from configs.wp41.load_cases import get_case  # noqa: E402
from configs.wp41.torsion import TORSION_INPUTS  # noqa: E402
from configs.wp41.weight import WEIGHT_INPUTS  # noqa: E402
from wp41 import aero_loads  # noqa: E402
from wp41.atmosphere import G0  # noqa: E402
from wp41.lift_distribution import case_weights  # noqa: E402
from wp41.stations import station_y  # noqa: E402
from wp41.weight import engine_ahead_of_centroid, weight_tables  # noqa: E402

logger = logging.getLogger("wp41_weight")


def write_case(case_name: str) -> Path:
    """Compute the weights of one load case, write the two files, return the folder."""
    case = get_case(case_name)
    pf = aero_loads.planform(WING)
    y = station_y(pf.half_span_m)
    chord = pf.chord_at(y)
    _, _, w_fuel = case_weights(case.mass_case, MASSES)
    chord_engine = float(np.interp(ENGINE.y_m, y, chord))
    ahead = engine_ahead_of_centroid(
        TORSION_INPUTS.require_f_sc(),
        chord_engine,
        WEIGHT_INPUTS.engine_inlet_ahead_of_le_m,
        WEIGHT_INPUTS.nacelle_length_m,
        WEIGHT_INPUTS.engine_cg_fraction,
    )
    weight, points = weight_tables(
        case=case.name,
        y=y,
        chord=chord,
        w_fuel_n=w_fuel,
        f_fuel=WEIGHT_INPUTS.f_fuel,
        engine_y_m=ENGINE.y_m,
        engine_weight_n=G0 * MASSES.engine_installed_kg * MASSES.engine_count / 2.0,
        engine_ahead_m=ahead,
        tolerance=settings.WEIGHT_CHECK_TOLERANCE,
    )
    folder = settings.case_dir(case.name)
    weight.to_csv(folder / settings.WEIGHT_CSV)
    points.to_csv(folder / settings.POINT_LOADS_CSV)
    logger.info(
        "%s: W_fuel = %.0f N, engine %.0f N at y = %.2f m, "
        "%.3f m ahead of the centroid",
        case.name, w_fuel, points.w_n[0], ENGINE.y_m, ahead,
    )
    return folder


def main(argv: list[str] | None = None) -> None:
    """Write weight.csv and point_loads.csv of each named load case."""
    parser = argparse.ArgumentParser(description="WP4.1 weights per load case")
    parser.add_argument(
        "--case", nargs="+", required=True, help="load case names, for example LC-1"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for name in args.case:
        write_case(name)


if __name__ == "__main__":
    main()
