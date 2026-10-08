# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""WP2: HLD v2 trade-off -- performance, occupied space and complexity.

Drives b12wp2.hld.tradeoff on the clean wing scripts/hld_analysis.py
computed (results/hld/clean_wing.csv): every TE x LE configuration on a
grid of trailing-edge spans up to the aileron, configurations below the
CL_max requirements discarded, the rest scored and ranked by their best
grid point, plus the winner over a sweep of the trade-off weights.

No OpenVSP needed (only the v1 clean-wing CSV). From the wp2/ folder:

    python -m scripts.hld_analysis_v2

Writes to wp2/results/hld_v2/: tradeoff.csv (one row per configuration),
tradeoff_grid.csv (every grid point), sensitivity.csv and the plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # run as a file (e.g. the editor's Run button) instead of `python -m scripts.<name>`:
    # make wp2/ and wp2/src/ importable so `from scripts import ...` and
    # `from b12wp2 import ...` resolve
    _WP2_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_WP2_DIR))
    sys.path.insert(0, str(_WP2_DIR / "src"))

import pandas as pd

from b12wp2.config import hld as cfg, hld_v2 as v2, paths
from b12wp2.hld import sizing as hs, tradeoff
from b12wp2.plots import hld_v2 as plots


def load_clean_wing() -> hs.CleanWing:
    if not paths.HLD_CLEAN_WING_CSV.exists():
        raise FileNotFoundError(
            f"{paths.HLD_CLEAN_WING_CSV} not found: run `python -m scripts.hld_analysis` "
            "(OpenVSP env) first, it writes the clean-wing lift this trade-off builds on"
        )
    v = pd.read_csv(paths.HLD_CLEAN_WING_CSV, index_col="quantity")["value"]
    return hs.CleanWing(
        float(v["CLmax_clean"]), float(v["CL_alpha_per_deg"]), float(v["alpha_0L"]),
        float(v["alpha_crit_clean"]),
    )


def main() -> None:
    paths.HLD_V2_DIR.mkdir(parents=True, exist_ok=True)
    clean = load_clean_wing()
    req_l, req_to = tradeoff.requirements()
    scored = tradeoff.score(tradeoff.grid(clean))
    best = tradeoff.best_per_config(scored)
    sens = tradeoff.sensitivity(scored)

    scored.to_csv(paths.HLD_V2_DIR / "tradeoff_grid.csv", index=False)
    best.to_csv(paths.HLD_V2_DIR / "tradeoff.csv", index=False)
    sens.to_csv(paths.HLD_V2_DIR / "sensitivity.csv", index=False)

    w = v2.WEIGHTS
    print(
        f"\nClean wing: CL_max = {clean.cl_max:.3f}. Requirements: CL_max,L >= {req_l}, "
        f"CL_max,TO >= {req_to}"
    )
    print(
        f"TE device: eta {cfg.ETA_IN} .. at most {v2.ETA_AILERON_IN} (aileron inboard edge), "
        f"grid step {v2.ETA_TE_STEP}; LE device: eta {cfg.ETA_IN} .. {cfg.ETA_OUT_LE}"
    )
    print(
        f"Weights: performance {w['performance']:.2f}, space {w['space']:.2f}, "
        f"complexity {w['complexity']:.2f}\n"
    )
    print(f"{'#':>2}  {'configuration':<28}{'cmplx':>6}{'eta_min':>8}{'eta':>6}{'CLmax_L':>9}"
          f"{'CLmax_TO':>9}{'s_perf':>8}{'s_space':>8}{'s_cmplx':>8}{'score':>8}{'pareto':>7}")
    for _, r in best[best["status"] == "feasible"].iterrows():
        print(f"{r['rank']:>2}  {r['config']:<28}{r['complexity']:>6d}{r['eta_out_te_min']:8.3f}"
              f"{r['eta_out_te']:6.2f}{r['CLmax_L']:9.3f}{r['CLmax_TO']:9.3f}{r['s_performance']:8.3f}"
              f"{r['s_space']:8.3f}{r['s_complexity']:8.3f}{r['score']:8.3f}{'*' if r['pareto'] else '':>7}")
    print("\nDiscarded:")
    for _, r in best[best["status"] == "discarded"].iterrows():
        print(f"    {r['config']:<28}{r['reason']}")

    wins = sens["winner"].value_counts()
    print(f"\nWeight sensitivity ({len(sens)} weight sets, step {v2.SENSITIVITY_STEP}):")
    for name, n in wins.items():
        print(f"    {name:<28}wins {n:>3}")

    plots.make_all(scored, best, sens, req_l, req_to, clean.cl_max, paths.HLD_V2_DIR)
    print(f"\nResults in: {paths.HLD_V2_DIR}")


if __name__ == "__main__":
    main()
