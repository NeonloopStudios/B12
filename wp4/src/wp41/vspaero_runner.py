# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4.1 VSPAERO run: build the wing, run the alpha sweep, write the strip loads.

Runs ONLY in the OpenVSP env (Python 3.11, imports openvsp). Usually
wp4/scripts/wp41_aero_loads.py starts it from the repo venv. To run it directly:

    <openvsp env>/python.exe wp4/src/wp41/vspaero_runner.py <run_dir>

The wing model is copied from wp2/src/b12wp2/wing/geometry.py and
wp2/src/b12wp2/wing/vspaero.py, without the drag work. The planform, the
freestream and the moment reference point come from wp41.aero_loads, so
the post-processing uses the same numbers.

Writes to <run_dir>:

    wing.vsp3 and the VSPAERO files
    strips.csv   one row per (alpha, strip): aero_loads.STRIP_COLUMNS
    totals.csv   one row per alpha: aero_loads.TOTAL_COLUMNS

VSPAERO applies Prandtl-Glauert at the given Mach (compressibility).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run as a file: make wp4/ (configs) and wp4/src/ (wp41) importable.
    _WP4_DIR = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

import numpy as np  # noqa: E402
import openvsp as vsp  # noqa: E402

from configs.aircraft import WING  # noqa: E402
from configs.wp41 import settings  # noqa: E402
from wp41 import aero_loads as al  # noqa: E402

logger = logging.getLogger("wp41.vspaero_runner")

# VSPAERO result names for the columns of aero_loads.STRIP_COLUMNS (without alpha)
STRIP_RESULT_NAMES = {
    "y": "Yavg", "chord": "Chord", "area": "dArea",
    "cl": "cl", "cx": "cx", "cz": "cz", "cmy": "cmy",
}


def set_airfoil(surf: str, index: int, path: Path) -> None:
    """Give cross-section `index` of the surface the airfoil from a .dat file."""
    upper, lower = al.read_airfoil_dat(path)
    vsp.ChangeXSecShape(surf, index, vsp.XS_FILE_AIRFOIL)
    xsec = vsp.GetXSec(surf, index)
    vsp.SetAirfoilPnts(
        xsec,
        [vsp.vec3d(x, y, 0.0) for x, y in upper],
        [vsp.vec3d(x, y, 0.0) for x, y in lower],
    )


def build_wing(pf: al.Planform) -> str:
    """Add the wing to the current OpenVSP model and return its geom ID."""
    wid: str = vsp.AddGeom("WING")
    vsp.SetGeomName(wid, settings.WING_NAME)

    vsp.SetParmVal(wid, "X_Rel_Location", "XForm", WING.root_le_x_m)
    vsp.SetParmVal(wid, "Z_Rel_Location", "XForm", WING.root_z_m)
    vsp.SetParmVal(wid, "Y_Rel_Rotation", "XForm", WING.incidence_deg)

    grp = "XSec_1"
    vsp.SetDriverGroup(
        wid, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER
    )
    vsp.SetParmVal(wid, "Span", grp, pf.half_span_m)
    vsp.SetParmVal(wid, "Root_Chord", grp, pf.root_chord_m)
    vsp.SetParmVal(wid, "Tip_Chord", grp, pf.tip_chord_m)
    vsp.SetParmVal(wid, "Sweep", grp, WING.sweep_c4_deg)
    vsp.SetParmVal(wid, "Sweep_Location", grp, 0.25)
    vsp.SetParmVal(wid, "Dihedral", grp, WING.dihedral_deg)
    vsp.SetParmVal(wid, "Twist", grp, WING.tip_twist_deg)
    vsp.SetParmVal(wid, "Twist_Location", grp, WING.twist_axis_chord_fraction)
    vsp.SetParmVal(wid, "SectTess_U", grp, settings.VSPAERO.span_tess)
    vsp.SetParmVal(wid, "OutCluster", grp, settings.VSPAERO.out_cluster)
    vsp.SetParmVal(wid, "Tess_W", "Shape", settings.VSPAERO.chord_tess)
    vsp.Update()

    surf = vsp.GetXSecSurf(wid, 0)
    set_airfoil(surf, 0, WING.airfoil_root)
    set_airfoil(surf, 1, WING.airfoil_tip)
    vsp.Update()

    logger.info(
        "Wing: S = %.3f m^2 (OpenVSP %.3f), b = %.3f m (OpenVSP %.3f)",
        WING.area_m2, vsp.GetParmVal(wid, "TotalArea", "WingGeom"),
        pf.span_m, vsp.GetParmVal(wid, "TotalSpan", "WingGeom"),
    )
    return wid


def run_sweep(pf: al.Planform, fs: al.Freestream) -> None:
    """Run the VSPAERO alpha sweep (thin-surface VLM, symmetric half model)."""
    vsp_dir = str(Path(vsp.__file__).parent)
    vsp.SetVSPAEROPath(vsp_dir)
    if not vsp.CheckForVSPAERO(vsp_dir):
        raise RuntimeError(f"vspaero.exe not found in {vsp_dir}")

    g = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(g)
    vsp.SetIntAnalysisInput(g, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(g, "ThinGeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(g, "Symmetry", [1])
    vsp.ExecAnalysis(g)

    x_ref, z_ref = al.moment_reference(WING, pf)
    cfg = settings.VSPAERO
    a = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(a)
    vsp.SetIntAnalysisInput(a, "GeomSet", [vsp.SET_NONE])
    vsp.SetIntAnalysisInput(a, "ThinGeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(a, "Symmetry", [1])
    vsp.SetIntAnalysisInput(a, "RefFlag", [vsp.MANUAL_REF])
    vsp.SetDoubleAnalysisInput(a, "Sref", [WING.area_m2])
    vsp.SetDoubleAnalysisInput(a, "bref", [pf.span_m])
    vsp.SetDoubleAnalysisInput(a, "cref", [pf.mac_m])
    vsp.SetDoubleAnalysisInput(a, "Xcg", [x_ref])
    vsp.SetDoubleAnalysisInput(a, "Ycg", [0.0])
    vsp.SetDoubleAnalysisInput(a, "Zcg", [z_ref])
    vsp.SetDoubleAnalysisInput(a, "AlphaStart", [cfg.alpha_start_deg])
    vsp.SetDoubleAnalysisInput(a, "AlphaEnd", [cfg.alpha_end_deg])
    vsp.SetIntAnalysisInput(a, "AlphaNpts", [cfg.alpha_npts])
    vsp.SetDoubleAnalysisInput(a, "BetaStart", [0.0])
    vsp.SetDoubleAnalysisInput(a, "BetaEnd", [0.0])
    vsp.SetIntAnalysisInput(a, "BetaNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "MachStart", [fs.mach])
    vsp.SetDoubleAnalysisInput(a, "MachEnd", [fs.mach])
    vsp.SetIntAnalysisInput(a, "MachNpts", [1])
    vsp.SetDoubleAnalysisInput(a, "Vinf", [fs.velocity])
    vsp.SetDoubleAnalysisInput(a, "Rho", [fs.density])
    vsp.SetIntAnalysisInput(a, "WakeNumIter", [cfg.wake_iter])
    vsp.SetIntAnalysisInput(a, "NCPU", [cfg.n_cpu])

    logger.info(
        "Running VSPAERO: M = %.3f, V = %.2f m/s, alpha %.1f..%.1f deg (%d points)",
        fs.mach, fs.velocity, cfg.alpha_start_deg, cfg.alpha_end_deg, cfg.alpha_npts,
    )
    vsp.ExecAnalysis(a)

    errors = vsp.ErrorMgrSingleton.getInstance()
    messages = []
    while errors.GetNumTotalErrors() > 0:
        messages.append(errors.PopLastError().GetErrorString())
    if messages:
        logger.error("OpenVSP errors: %s", messages)
        raise RuntimeError(f"OpenVSP reported errors: {messages}")


def result_column(rid: str, name: str) -> list[float]:
    """One double result of a VSPAERO result set; fails with the available names."""
    if name not in vsp.GetAllDataNames(rid):
        raise KeyError(f"VSPAERO result {name!r} not found, available: {vsp.GetAllDataNames(rid)}")
    return list(vsp.GetDoubleResults(rid, name))


def collect_strips() -> np.ndarray:
    """Strip loads of all alphas as rows of aero_loads.STRIP_COLUMNS."""
    blocks = []
    for i in range(vsp.GetNumResults("VSPAERO_Load")):
        rid = vsp.FindResultsID("VSPAERO_Load", i)
        alpha = result_column(rid, "FC_AoA_")[0]
        columns = [result_column(rid, STRIP_RESULT_NAMES[c]) for c in al.STRIP_COLUMNS[1:]]
        n = len(columns[0])
        blocks.append(np.column_stack([np.full(n, alpha), *columns]))
    if not blocks:
        raise RuntimeError("No VSPAERO_Load results")
    return np.vstack(blocks)


def collect_totals() -> np.ndarray:
    """Whole-wing CL and CMy per alpha as rows of aero_loads.TOTAL_COLUMNS."""
    rid = vsp.FindLatestResultsID("VSPAERO_Polar")
    if rid == "":
        raise RuntimeError("No VSPAERO_Polar results")
    return np.column_stack(
        [result_column(rid, "Alpha"), result_column(rid, "CLtot"), result_column(rid, "CMytot")]
    )


def main(run_dir: Path) -> None:
    """Build the wing, run VSPAERO and write strips.csv and totals.csv to run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    pf = al.planform(WING)
    fs = al.freestream(settings.FLIGHT)

    vsp.VSPRenew()
    vsp.DeleteAllResults()
    build_wing(pf)
    vsp.WriteVSPFile(str(run_dir / settings.VSP3_FILE))  # VSPAERO writes its files here
    run_sweep(pf, fs)

    for name, data, columns in (
        (settings.STRIPS_FILE, collect_strips(), al.STRIP_COLUMNS),
        (settings.TOTALS_FILE, collect_totals(), al.TOTAL_COLUMNS),
    ):
        np.savetxt(run_dir / name, data, delimiter=",", header=",".join(columns), fmt="%.8g")
        logger.info("Wrote %s (%d rows)", run_dir / name, data.shape[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 2:
        logger.error("Usage: vspaero_runner.py <run_dir>")
        sys.exit(2)
    main(Path(sys.argv[1]))
