# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""WP4.1: torque diagram about the wing box centroid axis.

Reads lift.csv, weight.csv and point_loads.csv of each case from
wp4/out/wp41/<case>/ (written by the lift and the weight script). Writes
torque.csv beside them and one PNG with t'(y) and tau(y) of all cases.

    python wp4/scripts/wp41_torsion.py --cases LC-1 LC-2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.aircraft import WING  # noqa: E402
from configs.wp41 import settings  # noqa: E402
from configs.wp41.load_cases import get_case  # noqa: E402
from configs.wp41.torsion import TORSION_INPUTS  # noqa: E402
from wp41 import aero_loads  # noqa: E402
from wp41.contracts import LiftTable, PointLoads, TorqueTable, WeightTable  # noqa: E402
from wp41.torsion import torque_diagram  # noqa: E402

logger = logging.getLogger("wp41_torsion")


def compute_case(case_name: str) -> TorqueTable:
    """Read the three input files of one case, write torque.csv, return the table."""
    case = get_case(case_name)
    folder = settings.case_dir(case_name)
    lift = LiftTable.from_csv(folder / settings.LIFT_CSV)
    weight = WeightTable.from_csv(folder / settings.WEIGHT_CSV)
    points = PointLoads.from_csv(folder / settings.POINT_LOADS_CSV)
    table = torque_diagram(
        lift,
        weight,
        points,
        f_sc=TORSION_INPUTS.require_f_sc(),
        sweep_c4_deg=WING.sweep_c4_deg,
        thrust_n=case.thrust_n,
        h_thrust_m=TORSION_INPUTS.h_thrust_m,
        half_span_m=aero_loads.planform(WING).half_span_m,
    )
    path = folder / settings.TORQUE_CSV
    table.to_csv(path)
    logger.info("Wrote %s (root torque %.1f N m)", path, table.torque_nm[0])
    return table


def plot_cases(tables: list[TorqueTable], path: Path) -> None:
    """Plot t'(y) and tau(y) of all cases in one figure and save it.

    Args:
        tables: One torque table per case.
        path: PNG file.
    """
    fig, (ax_t, ax_tau) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 7.0))
    for table in tables:
        label = f"{table.case} (n = {table.n:g})"
        ax_t.plot(table.y_m, table.t_prime_nmpm, label=label)
        ax_tau.plot(table.y_m, table.torque_nm, label=label)
    ax_t.set_ylabel("t'(y) [N m/m]")
    ax_tau.set_ylabel("torque tau(y) [N m]")
    ax_tau.set_xlabel("span station y [m]")
    ax_t.set_title("Torque about the wing box centroid axis, nose-up positive")
    for ax in (ax_t, ax_tau):
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    """Compute the torque diagram of each case and plot them together."""
    parser = argparse.ArgumentParser(description="WP4.1 torque diagram")
    parser.add_argument(
        "--cases", nargs="+", required=True,
        help="load case names, for example one with n > 0 and one with n < 0",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    tables = [compute_case(name) for name in args.cases]
    plot_path = settings.OUT_DIR / f"torque_{'_'.join(args.cases)}.png"
    plot_cases(tables, plot_path)


if __name__ == "__main__":
    main()
