# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""High-lift device configurations on the WP2 wing: the clean wing they are
added to, and the comparison across every TE x LE combination.

1. Clean-wing CL_max at M ~ 0.2 (config.LANDING: sea level, 65 m/s):
   VSPAERO (VLM) spanwise lift + critical-section method with the XFoil
   landing polar -- the wing CL at which the first strip's sweep-normal
   cl_n = cl / cos^2(sweep) reaches the section cl_max.
2. Every trailing-edge x leading-edge device combination of
   config.hld.TE_DEVICES x config.hld.LE_DEVICES, sized on the same
   available span, through the ADSEE empirical increments (b12wp2.hld.sizing),
   for the landing and the take-off setting.
3. When config.hld.CL_MAX_*_REQ are set: margin against the required CL_max
   and the smallest flap span that meets it.

Needs the OpenVSP Python environment (b12wp2.wing.vspaero imports openvsp).
scripts/hld_analysis.py drives it and writes the results; the layout and the
sweep settings are config.hld and config.solvers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from b12wp2 import config
from b12wp2.config import hld as cfg, paths, solvers, wing as wing_cfg
from b12wp2.hld import sizing as hs
from b12wp2.wing import geometry as wg, viscous_correction as vc, vspaero as va

LANDING_POLAR = paths.polar_csv(wing_cfg.AIRFOIL_ROOT.stem, "landing")

PLANFORM = hs.Planform(wing_cfg.S_REF, wg.B, wg.C_ROOT, wg.C_TIP, wg.TAN_SWEEP_LE)


# ============================================================
#  CLEAN WING
# ============================================================


def clean_wing() -> tuple[hs.CleanWing, pd.DataFrame, pd.DataFrame, float]:
    """Clean-wing lift from VSPAERO at the landing condition.

    Returns (clean wing, VSPAERO polar with max_cl_ratio, strips, section cl_max_n).
    """
    section = vc.load_section_polar(LANDING_POLAR)
    cl_max_n = float(section["cl"].iloc[-1])

    paths.HLD_RUN_DIR.mkdir(parents=True, exist_ok=True)
    va.run_vspaero(
        paths.HLD_RUN_DIR / "wing.vsp3",
        config.LANDING,
        solvers.HLD_ALPHA_START,
        solvers.HLD_ALPHA_END,
        solvers.HLD_ALPHA_NPTS,
    )
    polar = va.read_vsp_polar()
    strips = va.read_strips()
    strips["cl_n"] = vc.normal_cl(strips["cl"].to_numpy(), config.SWEEP_RAD)
    strips["cl_ratio"] = strips["cl_n"] / cl_max_n
    ratio = strips.groupby("alpha")["cl_ratio"].max()
    polar["max_cl_ratio"] = polar["alpha"].map(ratio)

    lo, hi = solvers.HLD_LINEAR_RANGE
    lin = polar[(polar["alpha"] >= lo) & (polar["alpha"] <= hi)]
    cl_alpha, _ = np.polyfit(lin["alpha"], lin["CL"], 1)
    alpha_0l = float(np.interp(0.0, polar["CL"], polar["alpha"]))

    up = polar[polar["alpha"] >= 0].sort_values("alpha")
    if not up["max_cl_ratio"].max() >= 1.0 > up["max_cl_ratio"].min():
        raise RuntimeError(
            "clean_wing: section cl_max not reached inside the alpha sweep, "
            f"widen config.solvers.HLD_ALPHA_END (max cl_n / cl_max_n = {up['max_cl_ratio'].max():.3f})"
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
    for te in cfg.TE_DEVICES:
        te_eff = hs.device_effect(PLANFORM, te, cfg.FLAP_CHORD_RATIO, cfg.ETA_IN, cfg.ETA_OUT_TE)
        for le in cfg.LE_DEVICES:
            le_eff = (
                hs.device_effect(PLANFORM, le, cfg.SLAT_CHORD_RATIO, cfg.ETA_IN, cfg.ETA_OUT_LE)
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
            if cfg.CL_MAX_L_REQ is not None:
                row["margin_L"] = land.cl_max - cfg.CL_MAX_L_REQ
                # TE span needed once the LE device has done its part
                need_te = cfg.CL_MAX_L_REQ - clean.cl_max - le_eff.dcl_max_wing
                row["eta_out_te_min_L"] = hs.min_eta_out(
                    PLANFORM, te, cfg.FLAP_CHORD_RATIO, cfg.ETA_IN, cfg.ETA_OUT_TE, need_te
                )
            if cfg.CL_MAX_TO_REQ is not None:
                row["margin_TO"] = to.cl_max - cfg.CL_MAX_TO_REQ
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
