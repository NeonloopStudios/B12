# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2: run the full pipeline in order.

Each step reads what the previous one wrote, so order matters and each
step's main() is called exactly once, not parallelized:

  run_cruise_polars  -- cruise polars for every airfoil in wp2/airfoils/
  run_landing_polars -- landing polars
  mcrit_sweep         -- Mach-critical + Korn margin
  build_scorecard     -- needs the polar and mcrit CSVs above
  make_plots          -- needs all of the above, including the scorecard

Re-scoring a swapped-in airfoil (drop a .dat file into wp2/airfoils/, add
another) is this one command -- every step auto-discovers the current
candidate set via config.discover_airfoils(), nothing here is hardcoded
to the airfoils present when this was written.

config.py and xfoil_runtime.py aren't run scripts -- every other module
imports them directly, there is nothing to orchestrate.
"""
from __future__ import annotations

import time

from scripts import build_scorecard, make_plots, mcrit_sweep, run_cruise_polars, run_landing_polars

PIPELINE = [
    ("cruise polars", run_cruise_polars.main),
    ("landing polars", run_landing_polars.main),
    ("Mach-critical", mcrit_sweep.main),
    ("scorecard", build_scorecard.main),
    ("plots", make_plots.main),
]


def main() -> None:
    overall_start = time.monotonic()
    for label, step_main in PIPELINE:
        print(f"\n=== {label} ===")
        step_start = time.monotonic()
        step_main()
        print(f"  ({time.monotonic() - step_start:.1f}s)")
    print(f"\n=== Done ({time.monotonic() - overall_start:.1f}s total) ===")


if __name__ == "__main__":
    main()
