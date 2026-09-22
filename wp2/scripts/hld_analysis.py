# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Comparison of high-lift device (HLD) configurations on the WP2 wing.

1. Clean-wing CL_max at M ~ 0.2 (config.LANDING: sea level, 65 m/s):
   VSPAERO (VLM) spanwise lift + critical-section method with the XFoil
   landing polar -- the wing CL at which the first strip's sweep-normal
   cl_n = cl / cos^2(sweep) reaches the section cl_max.
2. Every trailing-edge x leading-edge device combination of
   hld_sizing.TE_DEVICES x hld_sizing.LE_DEVICES, sized on the same
   available span, through the ADSEE empirical increments (hld_sizing.py),
   for the landing and the take-off setting.
3. Optional (when CL_MAX_*_REQ are set): margin against the required
   CL_max and the smallest flap span that meets it.

Must be run from the OpenVSP Python environment, from the wp2/ folder:

    python -m scripts.hld_analysis

Writes to wp2/results/hld/: comparison.csv, clean_wing.csv, the plots,
and the raw VSPAERO run in vspaero_run/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import config, hld_plots, hld_sizing as hs, viscous_correction as vc
from scripts import vspaero_analysis as va, wing_geometry as wg

# ============================================================
#  SETTINGS
# ============================================================

# --- spanwise limits (fraction of the semi-span) ---
ETA_IN = 0.12  # fuselage side + clearance
ETA_OUT_TE = 0.75  # trailing-edge devices end where the aileron starts
ETA_OUT_LE = 0.95  # leading-edge devices run in front of the aileron, short of the tip

# --- chordwise size, from the spar positions ---
FRONT_SPAR = 0.15
REAR_SPAR = 0.65
FLAP_CHORD_RATIO = 1.0 - REAR_SPAR  # c_f / c
SLAT_CHORD_RATIO = FRONT_SPAR  # c_s / c

# --- requirements from WP1 (None = not yet available, checks skipped) ---
CL_MAX_L_REQ: float | None = None
CL_MAX_TO_REQ: float | None = None

# --- VSPAERO sweep at the landing condition ---
ALPHA_START = -4.0
ALPHA_END = 20.0
ALPHA_NPTS = 25
LINEAR_RANGE = (-4.0, 6.0)  # alpha range for the CL_alpha fit [deg]

OUT_DIR = config.RESULTS_DIR / "hld"
RUN_DIR = OUT_DIR / "vspaero_run"
LANDING_POLAR = config.RESULTS_DIR / "data" / f"{wg.AIRFOIL_ROOT.stem}_landing_polar.csv"

PLANFORM = hs.Planform(wg.S_REF, wg.B, wg.C_ROOT, wg.C_TIP, wg.TAN_SWEEP_LE)


# ============================================================
#  CLEAN WING
# ============================================================


def clean_wing() -> tuple[hs.CleanWing, pd.DataFrame, pd.DataFrame, float]:
    """Clean-wing lift from VSPAERO at the landing condition.

    Returns (clean wing, VSPAERO polar with max_cl_ratio, strips, section cl_max_n).
    """
    section = vc.load_section_polar(LANDING_POLAR)
    cl_max_n = float(section["cl"].iloc[-1])

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    va.run_vspaero(RUN_DIR / "wing.vsp3", config.LANDING, ALPHA_START, ALPHA_END, ALPHA_NPTS)
    polar = va.read_vsp_polar()
    strips = va.read_strips()
    strips["cl_n"] = vc.normal_cl(strips["cl"].to_numpy(), config.SWEEP_RAD)
    strips["cl_ratio"] = strips["cl_n"] / cl_max_n
    ratio = strips.groupby("alpha")["cl_ratio"].max()
    polar["max_cl_ratio"] = polar["alpha"].map(ratio)

    lin = polar[(polar["alpha"] >= LINEAR_RANGE[0]) & (polar["alpha"] <= LINEAR_RANGE[1])]
    cl_alpha, _ = np.polyfit(lin["alpha"], lin["CL"], 1)
    alpha_0l = float(np.interp(0.0, polar["CL"], polar["alpha"]))

    up = polar[polar["alpha"] >= 0].sort_values("alpha")
    if not up["max_cl_ratio"].max() >= 1.0 > up["max_cl_ratio"].min():
        raise RuntimeError(
            "clean_wing: section cl_max not reached inside the alpha sweep, "
            f"widen ALPHA_END (max cl_n / cl_max_n = {up['max_cl_ratio'].max():.3f})"
        )
    alpha_stall = float(np.interp(1.0, up["max_cl_ratio"], up["alpha"]))
    cl_max = float(np.interp(alpha_stall, up["alpha"], up["CL"]))
    return hs.CleanWing(cl_max, float(cl_alpha), alpha_0l, alpha_stall), polar, strips, cl_max_n


# ============================================================
#  CONFIGURATIONS
# ============================================================


def compare(clean: hs.CleanWing) -> pd.DataFrame:
    """One row per TE x LE configuration."""
    rows = []
    for te in hs.TE_DEVICES:
        te_eff = hs.device_effect(PLANFORM, te, FLAP_CHORD_RATIO, ETA_IN, ETA_OUT_TE)
        for le in hs.LE_DEVICES:
            le_eff = (
                hs.device_effect(PLANFORM, le, SLAT_CHORD_RATIO, ETA_IN, ETA_OUT_LE)
                if le.dcl_max > 0
                else hs.DeviceEffect(0.0, 0.0, 0.0, 1.0, 0.0)
            )
            land = hs.configuration(clean, te_eff, le_eff)
            to = hs.configuration(clean, te_eff, le_eff, takeoff=True)
            row = {
                "te_device": te.name,
                "le_device": le.name,
                "config": te.name if le.name == "none" else f"{te.name} + {le.name}",
                "complexity": te.complexity + le.complexity,
                "swf_s_te": te_eff.swf_s,
                "swf_s_le": le_eff.swf_s,
                "dCLmax_te": te_eff.dcl_max_wing,
                "dCLmax_le": le_eff.dcl_max_wing,
                "CLmax_L": land.cl_max,
                "CLmax_TO": to.cl_max,
                "S_prime_S_L": land.cl_alpha_per_deg / clean.cl_alpha_per_deg,
                "CL_alpha_L_per_deg": land.cl_alpha_per_deg,
                "alpha_0L_L": land.alpha_0l_deg,
                "alpha_stall_L": land.alpha_stall_deg,
                "CL_alpha_TO_per_deg": to.cl_alpha_per_deg,
                "alpha_0L_TO": to.alpha_0l_deg,
                "alpha_stall_TO": to.alpha_stall_deg,
            }
            if CL_MAX_L_REQ is not None:
                row["margin_L"] = land.cl_max - CL_MAX_L_REQ
                # TE span needed once the LE device has done its part
                need_te = CL_MAX_L_REQ - clean.cl_max - le_eff.dcl_max_wing
                row["eta_out_te_min_L"] = hs.min_eta_out(
                    PLANFORM, te, FLAP_CHORD_RATIO, ETA_IN, ETA_OUT_TE, need_te
                )
            if CL_MAX_TO_REQ is not None:
                row["margin_TO"] = to.cl_max - CL_MAX_TO_REQ
            rows.append(row)
    df = pd.DataFrame(rows)
    df["pareto"] = pareto_front(df)
    return df


def pareto_front(df: pd.DataFrame) -> pd.Series:
    """True for configurations no other configuration beats on both
    CLmax_L (higher) and complexity (lower)."""
    flags = []
    for _, r in df.iterrows():
        dominated = (
            (df["CLmax_L"] >= r["CLmax_L"])
            & (df["complexity"] <= r["complexity"])
            & ((df["CLmax_L"] > r["CLmax_L"]) | (df["complexity"] < r["complexity"]))
        ).any()
        flags.append(not dominated)
    return pd.Series(flags, index=df.index)


# ============================================================
#  MAIN
# ============================================================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean, polar, strips, cl_max_n = clean_wing()
    df = compare(clean)

    te_sweep = np.degrees(PLANFORM.sweep_at(1.0 - FLAP_CHORD_RATIO))
    le_sweep = np.degrees(PLANFORM.sweep_at(SLAT_CHORD_RATIO))
    clean_rows = {
        "mach": config.LANDING.mach_freestream,
        "section_cl_max_n": cl_max_n,
        "CLmax_clean": clean.cl_max,
        "alpha_stall_clean": clean.alpha_stall_deg,
        "CL_alpha_per_deg": clean.cl_alpha_per_deg,
        "alpha_0L": clean.alpha_0l_deg,
        "eta_in": ETA_IN,
        "eta_out_te": ETA_OUT_TE,
        "eta_out_le": ETA_OUT_LE,
        "flap_chord_ratio": FLAP_CHORD_RATIO,
        "slat_chord_ratio": SLAT_CHORD_RATIO,
        "hinge_sweep_te_deg": te_sweep,
        "hinge_sweep_le_deg": le_sweep,
    }
    pd.DataFrame([clean_rows]).T.rename(columns={0: "value"}).to_csv(
        OUT_DIR / "clean_wing.csv", index_label="quantity"
    )
    polar.to_csv(OUT_DIR / "clean_polar_landing.csv", index=False)
    df.sort_values("CLmax_L", ascending=False).to_csv(OUT_DIR / "comparison.csv", index=False)

    print(
        f"\nClean wing, M = {config.LANDING.mach_freestream:.3f}: CL_max = {clean.cl_max:.3f} at "
        f"alpha = {clean.alpha_stall_deg:.2f} deg (section cl_max,n = {cl_max_n:.3f}), "
        f"CL_alpha = {clean.cl_alpha_per_deg:.4f} /deg, alpha_0L = {clean.alpha_0l_deg:.2f} deg"
    )
    print(
        f"Devices: TE eta {ETA_IN}-{ETA_OUT_TE}, c_f/c = {FLAP_CHORD_RATIO:.2f}, hinge sweep {te_sweep:.1f} deg; "
        f"LE eta {ETA_IN}-{ETA_OUT_LE}, c_s/c = {SLAT_CHORD_RATIO:.2f}, hinge sweep {le_sweep:.1f} deg\n"
    )
    print(f"{'configuration':<32}{'cmplx':>6}{'dCLmax':>8}{'CLmax_L':>9}{'CLmax_TO':>9}"
          f"{'a_st_L':>8}{'pareto':>8}")
    for _, r in df.sort_values("CLmax_L", ascending=False).iterrows():
        print(f"{r['config']:<32}{r['complexity']:>6d}{r['dCLmax_te'] + r['dCLmax_le']:8.3f}"
              f"{r['CLmax_L']:9.3f}{r['CLmax_TO']:9.3f}{r['alpha_stall_L']:8.2f}"
              f"{'*' if r['pareto'] else '':>8}")
    if CL_MAX_L_REQ is None:
        print("\n(CL_MAX_L_REQ / CL_MAX_TO_REQ not set: no margins or minimum flap spans)")

    hld_plots.make_all(df, clean, PLANFORM, hld_plots.Layout(
        ETA_IN, ETA_OUT_TE, ETA_OUT_LE, FLAP_CHORD_RATIO, SLAT_CHORD_RATIO
    ), CL_MAX_L_REQ, CL_MAX_TO_REQ, OUT_DIR)
    print(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
