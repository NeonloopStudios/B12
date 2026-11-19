# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""WP4.1 script 1: lift and quarter-chord moment of each load case.

Reads the VSPAERO database (out/wp41/lift_distribution.txt and
moment_distribution.txt, written by wp41_aero_loads.py) and the load cases
of configs/wp41/load_cases.py. Writes out/wp41/<case>/lift.csv
(wp41.contracts.LiftTable). Design: wp4/README_41.md, section 5.

    python wp4/scripts/wp41_lift.py --case LC-1 [LC-2 ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.aircraft import MASSES  # noqa: E402
from configs.wp41 import settings  # noqa: E402
from configs.wp41.load_cases import get_case  # noqa: E402
from wp41 import lift_distribution as ld  # noqa: E402

logger = logging.getLogger("wp41_lift")


def write_case(database: ld.AeroDatabase, case_name: str) -> Path:
    """Compute one load case and write its lift.csv; return the file path."""
    case = get_case(case_name)
    table = ld.to_lift_table(ld.lift_for_case(database, case, MASSES), case.name)
    path = settings.case_dir(case.name) / settings.LIFT_CSV
    table.to_csv(path)
    logger.info(
        "%s: n = %g, W = %.0f N, M = %.3f, CL_d = %.4f, alpha_d = %.2f deg%s -> %s",
        case.name, table.n, table.w_n, table.mach_case, table.cl_total,
        table.alpha_d_deg, " (extrapolated)" if table.extrapolated else "", path,
    )
    return path


def main(argv: list[str] | None = None) -> None:
    """Read the VSPAERO database and write lift.csv of each named load case."""
    parser = argparse.ArgumentParser(description="WP4.1 lift per load case")
    parser.add_argument(
        "--case", nargs="+", required=True, help="load case names, for example LC-1"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    database = ld.read_aero_database(
        settings.OUT_DIR / settings.LIFT_FILE, settings.OUT_DIR / settings.MOMENT_FILE
    )
    for name in args.case:
        write_case(database, name)


if __name__ == "__main__":
    main()
