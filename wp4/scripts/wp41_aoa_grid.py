# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""WP4.1: required wing CL and angle of attack over weights and speeds.

Reads the VSPAERO database (run wp41_aero_loads.py first). Writes
out/wp41/aoa_grid.txt: CL and alpha for OEW, MZFW and MTOW over the speed
ranges of settings.GRID_SPEEDS, at n = settings.GRID_LOAD_FACTOR.

    python wp4/scripts/wp41_aoa_grid.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.aircraft import MASSES  # noqa: E402
from configs.wp41 import settings  # noqa: E402
from wp41 import lift_distribution as ld  # noqa: E402
from wp41.atmosphere import G0  # noqa: E402

logger = logging.getLogger("wp41_aoa_grid")

WEIGHT_NAMES = ("OEW", "MZFW", "MTOW")


def main() -> None:
    """Write CL and alpha for OEW, MZFW and MTOW at every grid speed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    database = ld.read_aero_database(
        settings.OUT_DIR / settings.LIFT_FILE, settings.OUT_DIR / settings.MOMENT_FILE
    )
    weights = G0 * np.array(
        [MASSES.oem_kg, MASSES.mzfm_kg, MASSES.mtom_kg], dtype=np.float64
    )
    blocks = []
    for speeds in settings.GRID_SPEEDS:
        v_eas = np.linspace(
            speeds.v_eas_start, speeds.v_eas_end, speeds.v_eas_npts, dtype=np.float64
        )
        grid = ld.aoa_grid(database, weights, v_eas, speeds.altitude_m, settings.GRID_LOAD_FACTOR)
        coefficient_columns = [
            column for i in range(weights.size) for column in (grid.cl_wing[i], grid.alpha_deg[i])
        ]
        blocks.append(
            np.column_stack(
                [
                    np.full(v_eas.size, speeds.altitude_m),
                    v_eas, grid.v_tas, grid.mach, grid.dynamic_pressure,
                    *coefficient_columns,
                ]
            )
        )
    names = " ".join(f"CL_{n} alpha_{n}_deg" for n in WEIGHT_NAMES)
    weight_text = ", ".join(f"{n} = {w:.0f} N" for n, w in zip(WEIGHT_NAMES, weights))
    header = "\n".join(
        [
            "WP4.1 required wing CL and freestream angle of attack (incidence in the model)",
            f"load factor n = {settings.GRID_LOAD_FACTOR:g}; {weight_text}",
            f"columns: altitude_m V_EAS_m_s V_TAS_m_s Mach q_Pa {names}",
        ]
    )
    path = settings.OUT_DIR / settings.AOA_GRID_FILE
    np.savetxt(path, np.vstack(blocks), fmt="%12.5f", header=header)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
