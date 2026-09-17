# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 Stage 9: run the full pipeline, Stages 3-8, in order.

Each stage reads what the previous one wrote, so order matters and each
stage's main() is called exactly once, not parallelized:

  3. run_cruise_polars  -- cruise polars for every airfoil in wp2/airfoils/
  4. run_landing_polars -- landing polars
  5. mcrit_sweep         -- Mach-critical + Korn margin (needs Stage 3's Cl_n
                            indirectly via config, not Stage 3's output files)
  6. (validity flagging -- folded into Stage 7, not a separate script)
  7. build_scorecard     -- needs Stages 3/4/5's CSVs
  8. make_plots          -- needs Stages 3/4/5/7's CSVs

Re-scoring a swapped-in airfoil (drop a .dat file into wp2/airfoils/, add
another) is this one command -- every stage auto-discovers the current
candidate set via config.discover_airfoils(), nothing here is hardcoded
to the airfoils present when this was written.

Stages 1 (config.py) and 2 (xfoil_runtime.py) aren't run scripts -- every
other stage imports them directly, there is nothing to orchestrate.
"""
from __future__ import annotations

import time

from scripts import build_scorecard, make_plots, mcrit_sweep, run_cruise_polars, run_landing_polars

STAGES = [
    ("Stage 3: cruise polars", run_cruise_polars.main),
    ("Stage 4: landing polars", run_landing_polars.main),
    ("Stage 5: Mach-critical", mcrit_sweep.main),
    ("Stage 7: scorecard", build_scorecard.main),
    ("Stage 8: plots", make_plots.main),
]


def main() -> None:
    overall_start = time.monotonic()
    for label, stage_main in STAGES:
        print(f"\n=== {label} ===")
        stage_start = time.monotonic()
        stage_main()
        print(f"  ({time.monotonic() - stage_start:.1f}s)")
    print(f"\n=== Done ({time.monotonic() - overall_start:.1f}s total) ===")


if __name__ == "__main__":
    main()
