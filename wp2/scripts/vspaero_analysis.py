# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""3D wing analysis at cruise: VSPAERO (VLM) + XFoil profile drag + Korn/Lock
wave drag, for the wing of wing_geometry.py with NACA 25112 sections.

    CD = CDi (VSPAERO, Trefftz plane) + CD_profile (XFoil strips) + CD_wave (Korn/Lock)

- VSPAERO: inviscid VLM angle-of-attack sweep at the cruise condition of
  config.CRUISE. Kept: CL, CMy and the wake (Trefftz-plane) induced drag
  CDiw. Discarded: VSPAERO's own CDo (a flat-plate skin-friction estimate)
  and its surface-integrated CDi (goes negative near CL = 0).
- Profile drag: the spanwise strip loads of every angle of attack are run
  through the XFoil cruise polar, see viscous_correction.py.
- Wave drag: XFoil models no shocks, so wave drag is added from the swept
  Korn equation (kappa_A from config.kappa_a) and Lock's approximation.

Lift and pitching moment stay inviscid (no viscous decambering); only drag
is corrected.

Must be run from the OpenVSP Python environment, from the wp2/ folder:

    python -m scripts.vspaero_analysis

Writes to wp2/results/vspaero/<airfoil>/: polar.csv, span_loads.csv,
summary.csv, the plots, and the raw VSPAERO run in vspaero_run/.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    # run as a file (e.g. the editor's Run button) instead of `python -m scripts.<name>`:
    # make wp2/ importable so `from scripts import ...` resolves
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import openvsp as vsp
import pandas as pd

from scripts import config, viscous_correction as vc, vspaero_plots, wing_geometry as wg

# ============================================================
#  SETTINGS
# ============================================================

AIRFOIL_STEM = wg.AIRFOIL_ROOT.stem

ALPHA_START = -4.0  # [deg], VSPAERO angle of attack (wing incidence comes on top)
ALPHA_END = 10.0
ALPHA_NPTS = 15

WAKE_ITER = 5
N_CPU = 4

KORN_SWEEP_LOC = 0.5  # chord fraction of the sweep line used in the Korn equation
SWEEP_DRAG_MODE = "friction"  # baseline, see viscous_correction.sweep_drag_factor
SWEEP_DRAG_MODE_SENSITIVITY = "cos3"

OUT_DIR = config.RESULTS_DIR / "vspaero" / AIRFOIL_STEM
RUN_DIR = OUT_DIR / "vspaero_run"
SECTION_POLAR = config.RESULTS_DIR / "data" / f"{AIRFOIL_STEM}_cruise_polar.csv"

# ============================================================
#  FLIGHT CONDITION (single source of truth: config.CRUISE)
# ============================================================

CRUISE = config.CRUISE
MACH = CRUISE.mach_freestream
CL_DESIGN = CRUISE.cl_wing if CRUISE.cl_wing is not None else float("nan")

X_CG = wg.X_LE_MAC + 0.25 * wg.MAC  # moment reference at quarter MAC

T_C = wg.mac_thickness()
KAPPA_A = config.kappa_a(AIRFOIL_STEM)
SWEEP_KORN = wg.sweep_at(KORN_SWEEP_LOC)


# ============================================================
#  VSPAERO
# ============================================================


def reynolds_mac(condition: config.FlightCondition) -> float:
    """Freestream Reynolds number based on the MAC."""
    mu = config.sutherland_viscosity(condition.temperature_k)
    return condition.density * condition.v_freestream * wg.MAC / mu


def run_vspaero(
    vsp3_path: Path,
    condition: config.FlightCondition = CRUISE,
    alpha_start: float = ALPHA_START,
    alpha_end: float = ALPHA_END,
    alpha_npts: int = ALPHA_NPTS,
) -> None:
    """Build the wing, save it to vsp3_path and run the VSPAERO alpha sweep
    at the given flight condition (VSPAERO writes its files next to the .vsp3)."""
    v_inf, rho, mach = condition.v_freestream, condition.density, condition.mach_freestream
    re_mac = reynolds_mac(condition)
    vsp.VSPRenew()
    vsp.DeleteAllResults()
    wid = wg.build_wing()
    vsp.WriteVSPFile(str(vsp3_path))
    wg.print_summary(wid)

    vsp_dir = str(Path(vsp.__file__).parent)
    vsp.SetVSPAEROPath(vsp_dir)
    if not vsp.CheckForVSPAERO(vsp_dir):
        raise RuntimeError("vspaero.exe not found next to the openvsp package")

    g = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(g)
    vsp.SetIntAnalysisInput(g, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(g, "ThinGeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(g, "Symmetry", [1])
    vsp.ExecAnalysis(g)

    a = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(a)
    vsp.SetIntAnalysisInput(a, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(a, "ThinGeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(a, "Symmetry", [1])

    vsp.SetIntAnalysisInput(a, "RefFlag", [vsp.MANUAL_REF])
    vsp.SetDoubleAnalysisInput(a, "Sref", [wg.S_REF])
    vsp.SetDoubleAnalysisInput(a, "bref", [wg.B])
    vsp.SetDoubleAnalysisInput(a, "cref", [wg.MAC])
    vsp.SetDoubleAnalysisInput(a, "Xcg", [X_CG])
    vsp.SetDoubleAnalysisInput(a, "Ycg", [0.0])
    vsp.SetDoubleAnalysisInput(a, "Zcg", [wg.Z_ROOT])

    vsp.SetDoubleAnalysisInput(a, "AlphaStart", [alpha_start])
    vsp.SetDoubleAnalysisInput(a, "AlphaEnd", [alpha_end])
    vsp.SetIntAnalysisInput(a, "AlphaNpts", [alpha_npts])
    vsp.SetDoubleAnalysisInput(a, "BetaStart", [0.0])
    vsp.SetDoubleAnalysisInput(a, "BetaEnd", [0.0])
    vsp.SetIntAnalysisInput(a, "BetaNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "MachStart", [mach])
    vsp.SetDoubleAnalysisInput(a, "MachEnd", [mach])
    vsp.SetIntAnalysisInput(a, "MachNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "ReCref", [re_mac])
    vsp.SetDoubleAnalysisInput(a, "ReCrefEnd", [re_mac])
    vsp.SetIntAnalysisInput(a, "ReCrefNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "Vinf", [v_inf])
    vsp.SetDoubleAnalysisInput(a, "Rho", [rho])
    vsp.SetIntAnalysisInput(a, "WakeNumIter", [WAKE_ITER])
    vsp.SetIntAnalysisInput(a, "NCPU", [N_CPU])

    print(
        f"Running VSPAERO ({condition.name}): V={v_inf:.2f} m/s, M={mach:.3f}, Re_MAC={re_mac:.3g}, "
        f"alpha {alpha_start}..{alpha_end} deg ({alpha_npts} points)"
    )
    vsp.ExecAnalysis(a)

    err_mgr = vsp.ErrorMgrSingleton.getInstance()
    while err_mgr.GetNumTotalErrors() > 0:
        print(err_mgr.PopLastError().GetErrorString())


def read_vsp_polar() -> pd.DataFrame:
    rid = vsp.FindLatestResultsID("VSPAERO_Polar")
    if rid == "":
        raise RuntimeError(f"No VSPAERO_Polar results, check the files in {RUN_DIR}")

    def col(name: str) -> list[float]:
        return list(vsp.GetDoubleResults(rid, name))

    return pd.DataFrame(
        {
            "alpha": col("Alpha"),
            "CL": col("CLtot"),
            "CDi": col("CDiw"),  # wake / Trefftz-plane induced drag
            "CMy": col("CMytot"),
            "CDo_vsp": col("CDo"),
            "CDtot_vsp": col("CDtot"),
        }
    )


def read_strips() -> pd.DataFrame:
    """Spanwise strip loads of the right half-wing, one row per (alpha, strip)."""
    frames = []
    for i in range(vsp.GetNumResults("VSPAERO_Load")):
        rid = vsp.FindResultsID("VSPAERO_Load", i)
        df = pd.DataFrame(
            {
                "y": list(vsp.GetDoubleResults(rid, "Yavg")),
                "chord": list(vsp.GetDoubleResults(rid, "Chord")),
                "area": list(vsp.GetDoubleResults(rid, "dArea")),
                "cl": list(vsp.GetDoubleResults(rid, "cl")),
            }
        )
        df.insert(0, "alpha", vsp.GetDoubleResults(rid, "FC_AoA_")[0])
        frames.append(df[df["y"] > 0])
    return pd.concat(frames, ignore_index=True).sort_values(["alpha", "y"], ignore_index=True)


# ============================================================
#  CORRECTIONS
# ============================================================


def build_polar(
    vsp_polar: pd.DataFrame, strips: pd.DataFrame, section: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add profile and wave drag to the VSPAERO polar.

    Returns (polar, strips) with the strip-level cl_n, cd_n, cd, status added.
    """
    cl_max_n = float(section["cl"].iloc[-1])
    rows, strip_frames = [], []
    for _, p in vsp_polar.iterrows():
        s = strips[np.isclose(strips["alpha"], p["alpha"])]
        s_out, cd_prof = vc.strip_profile_drag(
            s, section, sweep_rad=config.SWEEP_RAD, chord_ref=config.CHORD_M,
            s_ref=wg.S_REF, mode=SWEEP_DRAG_MODE,
        )
        cd_prof_sens = cd_prof * (
            vc.sweep_drag_factor(SWEEP_DRAG_MODE_SENSITIVITY, config.SWEEP_RAD)
            / vc.sweep_drag_factor(SWEEP_DRAG_MODE, config.SWEEP_RAD)
        )
        m_dd, m_crit = vc.korn_swept(KAPPA_A, T_C, float(p["CL"]), SWEEP_KORN)
        cd_wave = vc.lock_wave_drag(MACH, m_crit)
        cd = p["CDi"] + cd_prof + cd_wave
        ratio = s_out["cl_n"] / cl_max_n
        rows.append(
            {
                **p.to_dict(),
                "CD_profile": cd_prof,
                f"CD_profile_{SWEEP_DRAG_MODE_SENSITIVITY}": cd_prof_sens,
                "M_crit": m_crit,
                "M_dd": m_dd,
                "CD_wave": cd_wave,
                "CD": cd,
                f"CD_{SWEEP_DRAG_MODE_SENSITIVITY}": p["CDi"] + cd_prof_sens + cd_wave,
                "L_D": p["CL"] / cd,
                "L_D_no_wave": p["CL"] / (cd - cd_wave),
                "n_stalled": int((s_out["status"] == vc.STALLED).sum()),
                "max_cl_ratio": float(ratio.max()),
                "y_max_cl_ratio": float(s_out["y"].iloc[int(ratio.to_numpy().argmax())]),
            }
        )
        strip_frames.append(s_out)
    return pd.DataFrame(rows), pd.concat(strip_frames, ignore_index=True)


def _at_cl(polar: pd.DataFrame, column: str, cl: float) -> float:
    """Linear interpolation of `column` at wing CL, NaN outside the valid range."""
    valid = polar.dropna(subset=[column]).sort_values("CL")
    if valid.empty or not valid["CL"].min() <= cl <= valid["CL"].max():
        return float("nan")
    return float(np.interp(cl, valid["CL"], valid[column]))


def summarise(polar: pd.DataFrame) -> dict[str, float]:
    """Key figures of the corrected polar."""
    linear = polar[(polar["alpha"] >= -4) & (polar["alpha"] <= 4)]
    cl_alpha = float(np.polyfit(np.radians(linear["alpha"]), linear["CL"], 1)[0])

    pos = polar[(polar["CL"] > 0) & polar["CD"].notna()]
    i_max = pos["L_D"].idxmax()

    # critical-section estimate: wing CL at which the first strip reaches the
    # section cl_max of the cruise polar (max_cl_ratio = 1)
    upper = polar[polar["alpha"] >= 0].sort_values("alpha")
    ratio, cl_w = upper["max_cl_ratio"].to_numpy(), upper["CL"].to_numpy()
    cl_first_stall = float(np.interp(1.0, ratio, cl_w)) if ratio.max() >= 1.0 > ratio.min() else float("nan")

    m_dd, m_crit = vc.korn_swept(KAPPA_A, T_C, CL_DESIGN, SWEEP_KORN)
    return {
        "CL_design": CL_DESIGN,
        "alpha_design": _at_cl(polar, "alpha", CL_DESIGN),
        "CDi_design": _at_cl(polar, "CDi", CL_DESIGN),
        "CD_profile_design": _at_cl(polar, "CD_profile", CL_DESIGN),
        "CD_wave_design": _at_cl(polar, "CD_wave", CL_DESIGN),
        "CD_design": _at_cl(polar, "CD", CL_DESIGN),
        "L_D_design": CL_DESIGN / _at_cl(polar, "CD", CL_DESIGN),
        "L_D_design_no_wave": CL_DESIGN / (_at_cl(polar, "CD", CL_DESIGN) - _at_cl(polar, "CD_wave", CL_DESIGN)),
        f"L_D_design_{SWEEP_DRAG_MODE_SENSITIVITY}": CL_DESIGN / _at_cl(polar, f"CD_{SWEEP_DRAG_MODE_SENSITIVITY}", CL_DESIGN),
        "CMy_design": _at_cl(polar, "CMy", CL_DESIGN),
        "L_D_design_vsp_only": CL_DESIGN / _at_cl(polar, "CDtot_vsp", CL_DESIGN),
        "L_D_max": float(pos.loc[i_max, "L_D"]),
        "CL_at_L_D_max": float(pos.loc[i_max, "CL"]),
        "alpha_at_L_D_max": float(pos.loc[i_max, "alpha"]),
        "CL_alpha_per_rad": cl_alpha,
        "alpha_0L": _at_cl(polar, "alpha", 0.0),
        "CL_first_section_stall": cl_first_stall,
        "M_crit_design": m_crit,
        "M_dd_design": m_dd,
        "mach": MACH,
        "t_c": T_C,
        "kappa_a": KAPPA_A,
        "sweep_korn_deg": math.degrees(SWEEP_KORN),
    }


# ============================================================
#  MAIN
# ============================================================


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    section = vc.load_section_polar(SECTION_POLAR)
    print(
        f"XFoil section polar {SECTION_POLAR.name}: M_n={CRUISE.mach_normal:.4f}, "
        f"Re_n={CRUISE.reynolds_normal:,.0f}, usable cl_n in "
        f"[{section['cl'].iloc[0]:.3f}, {section['cl'].iloc[-1]:.3f}]"
    )

    run_vspaero(RUN_DIR / "wing.vsp3")
    polar, strips = build_polar(read_vsp_polar(), read_strips(), section)
    summary = summarise(polar)

    polar.to_csv(OUT_DIR / "polar.csv", index=False)
    strips.to_csv(OUT_DIR / "span_loads.csv", index=False)
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
        OUT_DIR / "summary.csv", index_label="quantity"
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
    print(f"\nSummary ({AIRFOIL_STEM}, M = {MACH:.3f}, sweep-drag mode '{SWEEP_DRAG_MODE}'):")
    for k, v in summary.items():
        print(f"  {k:<26} {v:10.5f}")

    vspaero_plots.make_all(polar, strips, section, summary, OUT_DIR)
    print(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
