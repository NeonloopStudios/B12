# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""WP2: compare high-lift device configurations and write the results.

Drives b12wp2.hld.analysis: the clean wing from VSPAERO at the landing
condition, then every TE x LE device combination through the ADSEE
increments, for the landing and the take-off setting. Prints the clean-wing
figures, the layout the devices were sized in and the ranked comparison
(Pareto-optimal ones starred), then draws the plots.

Must be run from the OpenVSP Python environment, from the wp2/ folder:

    python -m scripts.hld_analysis

Writes to wp2/results/hld/: comparison.csv, clean_wing.csv,
clean_polar_landing.csv, the plots, and the raw VSPAERO run in vspaero_run/.
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

import numpy as np
import pandas as pd

from b12wp2 import config
from b12wp2.config import hld as cfg, paths
from b12wp2.hld import analysis
from b12wp2.plots import hld as hld_plots

def main() -> None:
    paths.HLD_DIR.mkdir(parents=True, exist_ok=True)
    clean, polar, strips, cl_max_n = analysis.clean_wing()
    df = analysis.compare(clean)

    te_sweep = np.degrees(analysis.PLANFORM.sweep_at(1.0 - cfg.FLAP_CHORD_RATIO))
    le_sweep = np.degrees(analysis.PLANFORM.sweep_at(cfg.SLAT_CHORD_RATIO))
    clean_rows = {
        "mach": config.LANDING.mach_freestream,
        "section_cl_max_n": cl_max_n,
        "CLmax_clean": clean.cl_max,
        "alpha_stall_clean": clean.alpha_stall_deg,
        "CL_alpha_per_deg": clean.cl_alpha_per_deg,
        "alpha_0L": clean.alpha_0l_deg,
        "eta_in": cfg.ETA_IN,
        "eta_out_te": cfg.ETA_OUT_TE,
        "eta_out_le": cfg.ETA_OUT_LE,
        "flap_chord_ratio": cfg.FLAP_CHORD_RATIO,
        "slat_chord_ratio": cfg.SLAT_CHORD_RATIO,
        "hinge_sweep_te_deg": te_sweep,
        "hinge_sweep_le_deg": le_sweep,
    }
    pd.DataFrame([clean_rows]).T.rename(columns={0: "value"}).to_csv(
        paths.HLD_DIR / "clean_wing.csv", index_label="quantity"
    )
    polar.to_csv(paths.HLD_DIR / "clean_polar_landing.csv", index=False)
    df.sort_values("CLmax_L", ascending=False).to_csv(paths.HLD_DIR / "comparison.csv", index=False)

    print(
        f"\nClean wing, M = {config.LANDING.mach_freestream:.3f}: CL_max = {clean.cl_max:.3f} at "
        f"alpha = {clean.alpha_stall_deg:.2f} deg (section cl_max,n = {cl_max_n:.3f}), "
        f"CL_alpha = {clean.cl_alpha_per_deg:.4f} /deg, alpha_0L = {clean.alpha_0l_deg:.2f} deg"
    )
    print(
        f"Devices: TE eta {cfg.ETA_IN}-{cfg.ETA_OUT_TE}, c_f/c = {cfg.FLAP_CHORD_RATIO:.2f}, hinge sweep {te_sweep:.1f} deg; "
        f"LE eta {cfg.ETA_IN}-{cfg.ETA_OUT_LE}, c_s/c = {cfg.SLAT_CHORD_RATIO:.2f}, hinge sweep {le_sweep:.1f} deg\n"
    )
    print(f"{'configuration':<32}{'cmplx':>6}{'dCLmax':>8}{'CLmax_L':>9}{'CLmax_TO':>9}"
          f"{'a_st_L':>8}{'pareto':>8}")
    for _, r in df.sort_values("CLmax_L", ascending=False).iterrows():
        print(f"{r['config']:<32}{r['complexity']:>6d}{r['dCLmax_te'] + r['dCLmax_le']:8.3f}"
              f"{r['CLmax_L']:9.3f}{r['CLmax_TO']:9.3f}{r['alpha_stall_L']:8.2f}"
              f"{'*' if r['pareto'] else '':>8}")
    if cfg.CL_MAX_L_REQ is None:
        print("\n(config.hld.CL_MAX_L_REQ / CL_MAX_TO_REQ not set: no margins or minimum flap spans)")

    hld_plots.make_all(
        df,
        clean,
        analysis.PLANFORM,
        hld_plots.Layout(
            cfg.ETA_IN, cfg.ETA_OUT_TE, cfg.ETA_OUT_LE, cfg.FLAP_CHORD_RATIO, cfg.SLAT_CHORD_RATIO
        ),
        cfg.CL_MAX_L_REQ,
        cfg.CL_MAX_TO_REQ,
        paths.HLD_DIR,
    )
    print(f"\nResults in: {paths.HLD_DIR}")


if __name__ == "__main__":
    main()
