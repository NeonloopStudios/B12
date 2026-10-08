# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-08
"""3D aerodynamic summary: the wing, clean-wing landing and HLD results in
one table.

Computes nothing new from VSPAERO or XFoil. It reads back what
scripts/vspaero_analysis.py, hld_analysis.py and hld_analysis_v2.py wrote,
adds the two figures those stages do not store -- the near-field span
efficiency from VSPAERO's surface-integrated drag, and the effective Oswald
factor of the computed wing polar (CD = CD0 + K CL^2 fit) -- and lays every
induced-drag figure side by side, each labelled with what it includes.
b12wp2.plots.aero_summary renders it to PDF.

No OpenVSP import (unlike b12wp2.wing.vspaero), so either Python environment
can run it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from b12wp2 import config
from b12wp2.common import drag_polar as dp
from b12wp2.config import hld as hld_cfg, paths, planform, solvers, wing as wing_cfg

AIRFOIL_STEM = wing_cfg.AIRFOIL_ROOT.stem

OSWALD_SECTION = "Oswald factor and span efficiency"

# What each Oswald / span-efficiency figure is, printed under that table.
OSWALD_NOTES = (
    "The figures are different quantities, not competing estimates of one number.",
    "ADSEE-II e is the Oswald factor of the whole aircraft: all lift-dependent drag (induced, "
    "viscous, fuselage). The formula is the DATCOM lift-curve slope written through the lifting-line "
    "relation CL_alpha = 2 pi / (1 + 2/(AR e)), so it reads the cos(sweep) loss of lift slope as lost "
    "induced efficiency and drops steeply with sweep. It is the one for the aircraft drag polar "
    "CD = CD0 + K CL^2.",
    "The span efficiencies are wing only and inviscid. The lifting-line value is the one in the wing CD. "
    "VSPAERO's near field lies just above 1 here and goes negative near CL = 0; its far field (CDiw) "
    "changes with sweep at an unchanged span loading, which Munk's stagger theorem rules out for induced "
    "drag. Both are kept for comparison only.",
    "The effective e of the wing polar adds the lift-dependent profile and wave drag of the wing to the "
    "lifting-line induced drag, but no fuselage.",
)


@dataclass(frozen=True)
class Row:
    section: str
    quantity: str  # plain name, the CSV key
    label: str  # the same for the PDF, matplotlib mathtext allowed
    value: float | str
    unit: str
    source: str
    note: str = ""


def _read(path: Path, made_by: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m scripts.{made_by}` first")
    return pd.read_csv(path)


def _quantities(path: Path, made_by: str) -> dict[str, float]:
    """A two-column quantity,value CSV (summary.csv, clean_wing.csv) as a dict."""
    table = _read(path, made_by)
    return dict(zip(table["quantity"], table["value"]))


def rel(path: Path) -> str:
    return path.relative_to(paths.RESULTS_DIR).as_posix()


def _at_cl(polar: pd.DataFrame, values: pd.Series, cl: float) -> float:
    """Linear interpolation of `values` at wing CL, NaN outside the polar."""
    valid = pd.DataFrame({"CL": polar["CL"], "v": values}).dropna().sort_values("CL")
    if valid.empty or not valid["CL"].min() <= cl <= valid["CL"].max():
        return float("nan")
    return float(np.interp(cl, valid["CL"], valid["v"]))


def build_rows() -> list[Row]:
    wing_dir = paths.wing_out_dir(AIRFOIL_STEM)
    summary_path = wing_dir / "summary.csv"
    polar_path = wing_dir / "polar.csv"
    s = _quantities(summary_path, "vspaero_analysis")
    polar = _read(polar_path, "vspaero_analysis")
    clean = _quantities(paths.HLD_CLEAN_WING_CSV, "hld_analysis")
    src_s, src_p, src_c = rel(summary_path), rel(polar_path), rel(paths.HLD_CLEAN_WING_CSV)

    cl_d = float(s["CL_design"])
    ar = planform.AR
    cruise, landing = config.CRUISE, config.LANDING

    rows: list[Row] = []

    def add(section: str, quantity: str, label: str, value: float | str, unit: str, source: str,
            note: str = "") -> None:
        rows.append(Row(section, quantity, label, value, unit, source, note))

    # --- configuration ---
    sec = "Configuration"
    add(sec, "airfoil", "airfoil (root and tip)", AIRFOIL_STEM, "", "config.wing")
    add(sec, "S_ref", r"$S_{ref}$", planform.S_REF, "m^2", "config.planform")
    add(sec, "AR", r"$AR$", ar, "-", "config.planform")
    add(sec, "taper", r"taper $\lambda$", planform.TAPER, "-", "config.planform")
    add(sec, "span", r"span $b$", planform.B, "m", "config.planform")
    add(sec, "c_root", r"$c_{root}$", planform.C_ROOT, "m", "config.planform")
    add(sec, "c_tip", r"$c_{tip}$", planform.C_TIP, "m", "config.planform")
    add(sec, "MAC", "MAC", planform.MAC, "m", "config.planform")
    add(sec, "sweep_LE", r"$\Lambda_{LE}$", math.degrees(planform.sweep_at(0.0)), "deg", "config.planform")
    add(sec, "sweep_c4", r"$\Lambda_{c/4}$", math.degrees(planform.sweep_at(0.25)), "deg", "config.planform")
    add(sec, "sweep_c2", r"$\Lambda_{c/2}$", math.degrees(planform.sweep_at(0.5)), "deg", "config.planform",
        "line of the 2D section reduction, Korn and the ADSEE Oswald formula")
    add(sec, "t_c", r"$t/c$ (streamwise)", s["t_c"], "-", src_s)
    add(sec, "kappa_A", r"$\kappa_A$ (Korn)", s["kappa_a"], "-", src_s, "supercritical section")
    add(sec, "incidence", r"incidence $i_w$", wing_cfg.INCIDENCE_DEG, "deg", "config.wing")
    add(sec, "twist_tip", "tip twist", wing_cfg.TWIST_TIP_DEG, "deg", "config.wing", "wash-out")
    add(sec, "dihedral", r"dihedral $\Gamma$", wing_cfg.DIHEDRAL_DEG, "deg", "config.wing")

    # --- flight conditions ---
    sec = "Flight conditions"
    add(sec, "cruise_altitude", "cruise altitude", cruise.altitude_m, "m", "config.mission")
    add(sec, "cruise_mach", r"cruise $M_\infty$", cruise.mach_freestream, "-", "config.mission")
    add(sec, "cruise_v", r"cruise $V_\infty$", cruise.v_freestream, "m/s", "config.mission")
    add(sec, "CL_design", r"$C_{L,design}$", cl_d, "-", src_s, "level-flight trim, WP1 weight and wing area")
    add(sec, "cruise_mach_normal", r"cruise $M_n$", cruise.mach_normal, "-", "config.mission",
        "normal to the half-chord line")
    add(sec, "cruise_cl_normal", r"cruise $C_{l,n}$", cruise.cl_normal, "-", "config.mission")
    add(sec, "cruise_re_normal", r"cruise $Re_n$", cruise.reynolds_normal, "-", "config.mission", "on the normal MAC")
    add(sec, "landing_v", r"landing $V$", config.LANDING_SPEED_MS, "m/s", "config.mission")
    add(sec, "landing_mach", r"landing $M_\infty$", landing.mach_freestream, "-", "config.mission", "sea level")
    add(sec, "landing_re_normal", r"landing $Re_n$", landing.reynolds_normal, "-", "config.mission", "on the normal MAC")

    # --- cruise wing polar ---
    sec = "Cruise wing (VSPAERO + XFoil + Korn)"
    add(sec, "alpha_design", r"$\alpha$ at $C_{L,design}$", s["alpha_design"], "deg", src_s,
        "VSPAERO angle, the incidence comes on top")
    add(sec, "CL_alpha_per_rad", r"$C_{L\alpha}$", s["CL_alpha_per_rad"], "1/rad", src_s)
    add(sec, "CL_alpha_per_deg", r"$C_{L\alpha}$", math.radians(1.0) * s["CL_alpha_per_rad"], "1/deg", src_s)
    add(sec, "alpha_0L", r"$\alpha_{0L}$", s["alpha_0L"], "deg", src_s)
    add(sec, "CDi_design", r"$C_{D,i}$", s["CDi_design"], "-", src_s, "lifting line on the VLM span loading")
    add(sec, "CD_profile_design", r"$C_{D,profile}$", s["CD_profile_design"], "-", src_s,
        "XFoil strips, sweep mode 'friction'")
    add(sec, "CD_wave_design", r"$C_{D,wave}$", s["CD_wave_design"], "-", src_s, "swept Korn + ADSEE drag rise")
    add(sec, "CD_design", r"$C_D$", s["CD_design"], "-", src_s, "all at CL_design")
    add(sec, "L_D_design", r"$L/D$", s["L_D_design"], "-", src_s, "at CL_design")
    add(sec, "L_D_design_no_wave", r"$L/D$, no wave drag", s["L_D_design_no_wave"], "-", src_s)
    add(sec, "L_D_design_cos3", r"$L/D$, $\cos^3\Lambda$ sweep drag", s["L_D_design_cos3"], "-", src_s,
        "optimistic sensitivity")
    add(sec, "L_D_max", r"$(L/D)_{max}$", s["L_D_max"], "-", src_s)
    add(sec, "CL_at_L_D_max", r"$C_L$ at $(L/D)_{max}$", s["CL_at_L_D_max"], "-", src_s)
    add(sec, "CMy_design", r"$C_{m}$ (1/4 MAC)", s["CMy_design"], "-", src_s, "at CL_design, inviscid")
    add(sec, "M_dd_design", r"$M_{dd}$", s["M_dd_design"], "-", src_s, "at CL_design")
    add(sec, "CL_first_section_stall", r"$C_L$ at first section stall", s["CL_first_section_stall"], "-", src_s)

    # --- induced drag and Oswald factor ---
    sec = OSWALD_SECTION
    k_adsee = float(s["K_adsee"])
    add(sec, "oswald_e_adsee", r"$e$, ADSEE-II", s["oswald_e_adsee"], "-", src_s,
        "whole aircraft, all lift-dependent drag; from AR and sweep c/2 only")
    add(sec, "K_adsee", r"$K = 1/(\pi e AR)$, ADSEE-II", k_adsee, "-", src_s)
    add(sec, "CDi_design_K_adsee", r"$C_{D,i}$ with ADSEE $K$", k_adsee * cl_d**2, "-", "computed", "at CL_design")

    cdi_near = _at_cl(polar, polar["CDtot_vsp"] - polar["CDo_vsp"], cl_d)
    for key, label, cdi, src, note in (
        ("lifting_line", "lifting line", float(s["CDi_design"]), src_s,
         "from Gamma(y) on the wing; the CDi used in the wing CD"),
        ("vsp_near_field", "VSPAERO near field", cdi_near, src_p,
         "CDtot - CDo, surface-integrated panel forces; comparison only"),
        ("vsp_far_field", "VSPAERO far field", float(s["CDi_trefftz_vsp_design"]), src_s,
         "CDiw, Trefftz plane; comparison only"),
    ):
        add(sec, f"CDi_design_{key}", rf"$C_{{D,i}}$, {label}", cdi, "-", src, "wing only, inviscid, at CL_design")
        add(sec, f"span_efficiency_{key}", rf"$\phi$, {label}", cl_d**2 / (math.pi * ar * cdi), "-", src, note)

    lo, hi = solvers.OSWALD_FIT_CL_RANGE
    fit_pts = polar.dropna(subset=["CD"])
    fit_pts = fit_pts[(fit_pts["CL"] >= lo) & (fit_pts["CL"] <= hi)]
    fit = dp.fit_parabolic_polar(fit_pts["CL"], fit_pts["CD"], ar)
    fit_src = f"{src_p}, fit over CL {lo:g}..{hi:g}"
    add(sec, "oswald_e_wing_polar", r"effective $e$ of the wing polar", fit.oswald, "-", fit_src,
        f"CD = CD0 + K CL^2 over CL {lo:g}-{hi:g}; wing with profile and wave drag, no fuselage")
    add(sec, "CD0_wing_polar", r"$C_{D,0}$ of the fit", fit.cd0, "-", fit_src)
    add(sec, "K_wing_polar", r"$K$ of the fit", fit.k, "-", fit_src)

    # --- clean wing at landing ---
    sec = "Clean wing at landing (VSPAERO + DATCOM)"
    add(sec, "landing_mach_hld", r"$M$", clean["mach"], "-", src_c)
    add(sec, "section_cl_max_n", r"section $c_{l,max,n}$", clean["section_cl_max_n"], "-", src_c,
        "XFoil landing polar")
    add(sec, "CLmax_clean", r"$C_{L,max}$, clean", clean["CLmax_clean"], "-", src_c,
        "first section reaches cl_max,n")
    add(sec, "alpha_crit_clean", r"$\alpha$ at $C_{L,max}$", clean["alpha_crit_clean"], "deg", src_c,
        "first section stall")
    add(sec, "CL_alpha_landing_per_deg", r"$C_{L\alpha}$", clean["CL_alpha_per_deg"], "1/deg", src_c)
    add(sec, "alpha_0L_landing", r"$\alpha_{0L}$", clean["alpha_0L"], "deg", src_c)
    add(sec, "LE_sharpness_dY", r"LE sharpness $\Delta Y$", hld_cfg.LE_SHARPNESS_DY, "% c", "config.hld",
        "y_u(6% c) - y_u(0.15% c), streamwise section")
    add(sec, "dalpha_CLmax", r"$\Delta\alpha_{C_L,max}$", clean["dalpha_CLmax"], "deg", src_c,
        "DATCOM chart at the LE sweep")
    add(sec, "alpha_stall_clean", r"$\alpha_{stall}$", clean["alpha_stall_clean"], "deg", src_c,
        "CL_max / CL_alpha + alpha_0L + dalpha_CLmax")
    add(sec, "CL_max_L_req", r"required $C_{L,max}$, landing", hld_cfg.CL_MAX_L_REQ, "-", "config.hld")
    add(sec, "CL_max_TO_req", r"required $C_{L,max}$, take-off", hld_cfg.CL_MAX_TO_REQ, "-", "config.hld")

    # --- HLD v2 winner ---
    sec = "High-lift devices (HLD v2 trade-off)"
    best = hld_v2_table().iloc[0]
    src_v2 = rel(paths.HLD_V2_DIR / "tradeoff.csv")
    add(sec, "hld_selected", "selected configuration", best["config"], "", src_v2, "equal weights")
    add(sec, "hld_score", "score", best["score"], "-", src_v2)
    add(sec, "hld_eta_out_te", r"TE device outboard end $\eta$", best["eta_out_te"], "-", src_v2)
    add(sec, "hld_CLmax_L", r"$C_{L,max}$, landing", best["CLmax_L"], "-", src_v2)
    add(sec, "hld_CLmax_TO", r"$C_{L,max}$, take-off", best["CLmax_TO"], "-", src_v2)
    add(sec, "hld_alpha_stall_L", r"$\alpha_{stall}$, landing", best["alpha_stall_L"], "deg", src_v2)
    return rows


def hld_v1_table() -> pd.DataFrame:
    """Pareto-optimal configurations of the HLD v1 comparison."""
    comp = _read(paths.HLD_DIR / "comparison.csv", "hld_analysis")
    comp["dCLmax"] = comp["dCLmax_te"] + comp["dCLmax_le"]
    cols = ["config", "complexity", "dCLmax", "CLmax_L", "CLmax_TO", "alpha_stall_L"]
    return comp[comp["pareto"]].sort_values("CLmax_L", ascending=False)[cols].reset_index(drop=True)


def hld_v2_table() -> pd.DataFrame:
    """Feasible configurations of the HLD v2 trade-off, best first."""
    trade = _read(paths.HLD_V2_DIR / "tradeoff.csv", "hld_analysis_v2")
    cols = ["rank", "config", "complexity", "eta_out_te", "CLmax_L", "CLmax_TO", "alpha_stall_L",
            "s_performance", "s_space", "s_complexity", "score", "pareto"]
    feasible = trade[trade["feasible"]].sort_values("rank")
    return feasible[cols].reset_index(drop=True)


def hld_v2_sensitivity() -> pd.Series:
    """How many weight sets of the sensitivity sweep each configuration wins."""
    sens = _read(paths.HLD_V2_DIR / "sensitivity.csv", "hld_analysis_v2")
    return sens["winner"].value_counts()


def rows_frame(rows: list[Row]) -> pd.DataFrame:
    """The rows as the CSV table (the mathtext labels left out)."""
    return pd.DataFrame([asdict(r) for r in rows]).drop(columns="label")
