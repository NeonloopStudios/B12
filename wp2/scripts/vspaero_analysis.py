# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""WP2: run the 3D wing analysis at cruise and write its results.

Drives b12wp2.wing.vspaero: VSPAERO (VLM) for lift, moment and induced drag,
the XFoil section polar for profile drag, the swept Korn equation and Lock's
approximation for wave drag. Prints the corrected polar and the key figures
of it, then draws the plots.

Must be run from the OpenVSP Python environment, from the wp2/ folder:

    python -m scripts.vspaero_analysis

Writes to wp2/results/vspaero/<airfoil>/: polar.csv, span_loads.csv,
summary.csv, the plots, and the raw VSPAERO run in vspaero_run/.
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

from b12wp2.plots import vspaero as vspaero_plots
from b12wp2.wing import viscous_correction as vc, vspaero as va

def main() -> None:
    va.RUN_DIR.mkdir(parents=True, exist_ok=True)
    section = vc.load_section_polar(va.SECTION_POLAR)
    print(
        f"XFoil section polar {va.SECTION_POLAR.name}: M_n={va.CRUISE.mach_normal:.4f}, "
        f"Re_n={va.CRUISE.reynolds_normal:,.0f}, usable cl_n in "
        f"[{section['cl'].iloc[0]:.3f}, {section['cl'].iloc[-1]:.3f}]"
    )

    va.run_vspaero(va.RUN_DIR / "wing.vsp3")
    polar, strips = va.build_polar(va.read_vsp_polar(), va.read_strips(), section)
    summary = va.summarise(polar)

    polar.to_csv(va.OUT_DIR / "polar.csv", index=False)
    strips.to_csv(va.OUT_DIR / "span_loads.csv", index=False)
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
        va.OUT_DIR / "summary.csv", index_label="quantity"
    )

    print(
        f"\n{'alpha':>6} {'CL':>7} {'CDi':>8} {'CDprof':>8} {'CDwave':>8} {'CD':>8} "
        f"{'L/D':>6} {'CMy':>7} {'stalled':>7}"
    )
    for _, r in polar.iterrows():
        print(
            f"{r['alpha']:6.1f} {r['CL']:7.4f} {r['CDi']:8.5f} {r['CD_profile']:8.5f} "
            f"{r['CD_wave']:8.5f} {r['CD']:8.5f} {r['L_D']:6.2f} {r['CMy']:7.4f} "
            f"{int(r['n_stalled']):4d}/{len(strips[strips['alpha'] == r['alpha']])}"
        )
    print(
        f"\nSummary ({va.AIRFOIL_STEM}, M = {va.MACH:.3f}, "
        f"sweep-drag mode '{va.SWEEP_DRAG_MODE}'):"
    )
    for k, v in summary.items():
        print(f"  {k:<26} {v:10.5f}")

    vspaero_plots.make_all(polar, strips, section, summary, va.OUT_DIR)
    print(f"\nResults in: {va.OUT_DIR}")


if __name__ == "__main__":
    main()
