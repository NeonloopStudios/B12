# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Wing planform and OpenVSP model for the WP2 VSPAERO analysis.

Planform values are the WP1 sizing used in the stand-alone VSP_analysis
study (S, AR, taper, dihedral, incidence); the quarter-chord sweep is taken
from config.SWEEP_RAD so the 3D model and the XFoil sweep-theory reduction
use exactly the same angle.

Deliberately has its own .dat reader instead of reusing
xfoil_runtime.load_airfoil_dat: importing xfoil_runtime loads the compiled
XFoil DLL, which is not installed in the OpenVSP Python environment (and is
not needed here -- only the coordinates are).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import openvsp as vsp

from scripts import config

# ============================================================
#  WING PLANFORM DESIGN VARIABLES
# ============================================================

S_REF = 80.5  # wing area, both halves [m^2]
AR = 9.50  # aspect ratio [-]
TAPER = 0.50  # c_tip / c_root [-]

SWEEP_LOC = 0.25  # chord fraction the sweep is measured at (quarter chord)
SWEEP_DEG = math.degrees(config.SWEEP_RAD)  # 24.02 deg

DIHEDRAL_DEG = 2.6
TWIST_TIP_DEG = 0.0  # tip twist relative to the root (negative = wash-out)
TWIST_LOC = 0.25  # twist axis as a chord fraction
INCIDENCE_DEG = 2.0  # wing incidence, added on top of the VSPAERO angle of attack

X_LE_ROOT = 0.0  # [m]
Z_ROOT = 0.3  # [m]

AIRFOIL_ROOT = config.AIRFOILS_DIR / "NACA_25112.dat"
AIRFOIL_TIP = AIRFOIL_ROOT

# --- VLM mesh ---
# The stand-alone study used OpenVSP's default of 5 spanwise strips per
# half-wing, too coarse to resolve the spanwise cl distribution the strip
# viscous correction is evaluated on. OUT_CLUSTER < 1 clusters strips
# towards the tip, where the load falls off fastest.
SPAN_TESS = 25  # spanwise section tessellation -> SPAN_TESS - 1 strips per half-wing
CHORD_TESS = 33  # chordwise tessellation (Tess_W)
OUT_CLUSTER = 0.5

NAME = "Wing"

# ============================================================
#  DERIVED QUANTITIES
# ============================================================

B = math.sqrt(S_REF * AR)  # span [m]
B_HALF = B / 2
C_ROOT = 2 * S_REF / (B * (1 + TAPER))
C_TIP = TAPER * C_ROOT

MAC = 2 / 3 * C_ROOT * (1 + TAPER + TAPER**2) / (1 + TAPER)
Y_MAC = B / 6 * (1 + 2 * TAPER) / (1 + TAPER)

TAN_SWEEP_LE = math.tan(config.SWEEP_RAD) + SWEEP_LOC * (C_ROOT - C_TIP) / B_HALF
X_LE_MAC = X_LE_ROOT + Y_MAC * TAN_SWEEP_LE


def sweep_at(chord_fraction: float) -> float:
    """Sweep angle [rad] of the line at the given chord fraction."""
    return math.atan(TAN_SWEEP_LE - chord_fraction * (C_ROOT - C_TIP) / B_HALF)


# ============================================================
#  AIRFOIL COORDINATES
# ============================================================

Array = npt.NDArray[np.float64]


def read_airfoil_dat(path: Path) -> tuple[Array, Array]:
    """Read a Selig or Lednicer .dat file.

    Returns (upper, lower) as (N, 2) arrays of (x, y), both running from the
    leading to the trailing edge, normalised to unit chord with the leading
    edge at the origin.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Airfoil file not found: {path}")

    pts = []
    for line in path.read_text().splitlines():
        try:
            x, y = (float(v) for v in line.replace(",", " ").split()[:2])
        except ValueError:
            continue  # name line, blank line, comment
        pts.append((x, y))
    if pts and pts[0][0] > 1.5:  # Lednicer header with point counts
        pts = pts[1:]
    arr = np.array(pts, dtype=float)
    if len(arr) < 10:
        raise ValueError(f"Too few points in airfoil file: {path}")

    if arr[0, 0] > 0.5:
        # Selig: TE -> upper -> LE -> lower -> TE
        i_le = int(np.argmin(arr[:, 0]))
        upper, lower = arr[i_le::-1], arr[i_le:]
    else:
        # Lednicer: LE -> upper -> TE, then LE -> lower -> TE
        i_split = int(np.argmax(np.diff(arr[:, 0]) < -0.5)) + 1
        upper, lower = arr[:i_split], arr[i_split:]

    x_le, y_le = upper[0]
    chord = max(upper[:, 0].max(), lower[:, 0].max()) - x_le
    upper = (upper - [x_le, y_le]) / chord
    lower = (lower - [x_le, y_le]) / chord
    return upper, lower


def airfoil_thickness(path: Path) -> float:
    """Maximum thickness ratio t/c of an airfoil .dat file."""
    upper, lower = read_airfoil_dat(path)
    x = np.linspace(0, 1, 501)
    y_up = np.interp(x, *upper[np.argsort(upper[:, 0])].T)
    y_lo = np.interp(x, *lower[np.argsort(lower[:, 0])].T)
    return float(np.max(y_up - y_lo))


def mac_thickness() -> float:
    """t/c at the MAC station, linearly interpolated between root and tip."""
    eta = Y_MAC / B_HALF
    tc_root = airfoil_thickness(AIRFOIL_ROOT)
    tc_tip = airfoil_thickness(AIRFOIL_TIP)
    return tc_root + eta * (tc_tip - tc_root)


# ============================================================
#  OPENVSP MODEL
# ============================================================


def _set_airfoil(surf: str, idx: int, path: Path) -> None:
    upper, lower = read_airfoil_dat(path)
    vsp.ChangeXSecShape(surf, idx, vsp.XS_FILE_AIRFOIL)
    xs = vsp.GetXSec(surf, idx)
    vsp.SetAirfoilPnts(
        xs,
        [vsp.vec3d(x, y, 0.0) for x, y in upper],
        [vsp.vec3d(x, y, 0.0) for x, y in lower],
    )


def build_wing() -> str:
    """Add the wing to the current OpenVSP model and return its geom ID."""
    wid: str = vsp.AddGeom("WING")
    vsp.SetGeomName(wid, NAME)

    vsp.SetParmVal(wid, "X_Rel_Location", "XForm", X_LE_ROOT)
    vsp.SetParmVal(wid, "Z_Rel_Location", "XForm", Z_ROOT)
    vsp.SetParmVal(wid, "Y_Rel_Rotation", "XForm", INCIDENCE_DEG)

    grp = "XSec_1"
    vsp.SetDriverGroup(
        wid, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER
    )
    vsp.SetParmVal(wid, "Span", grp, B_HALF)
    vsp.SetParmVal(wid, "Root_Chord", grp, C_ROOT)
    vsp.SetParmVal(wid, "Tip_Chord", grp, C_TIP)
    vsp.SetParmVal(wid, "Sweep", grp, SWEEP_DEG)
    vsp.SetParmVal(wid, "Sweep_Location", grp, SWEEP_LOC)
    vsp.SetParmVal(wid, "Dihedral", grp, DIHEDRAL_DEG)
    vsp.SetParmVal(wid, "Twist", grp, TWIST_TIP_DEG)
    vsp.SetParmVal(wid, "Twist_Location", grp, TWIST_LOC)
    vsp.SetParmVal(wid, "SectTess_U", grp, SPAN_TESS)
    vsp.SetParmVal(wid, "OutCluster", grp, OUT_CLUSTER)
    vsp.SetParmVal(wid, "Tess_W", "Shape", CHORD_TESS)
    vsp.Update()

    surf = vsp.GetXSecSurf(wid, 0)
    _set_airfoil(surf, 0, AIRFOIL_ROOT)
    _set_airfoil(surf, 1, AIRFOIL_TIP)
    vsp.Update()
    return wid


def print_summary(wid: str) -> None:
    s_vsp = vsp.GetParmVal(wid, "TotalArea", "WingGeom")
    b_vsp = vsp.GetParmVal(wid, "TotalSpan", "WingGeom")
    print(f"Wing '{NAME}' ({AIRFOIL_ROOT.stem} root, {AIRFOIL_TIP.stem} tip):")
    print(f"  S        = {S_REF:.3f} m^2   (OpenVSP: {s_vsp:.3f})")
    print(f"  b        = {B:.3f} m     (OpenVSP: {b_vsp:.3f})")
    print(f"  AR       = {AR:.2f}, taper = {TAPER:.3f}")
    print(f"  c_root   = {C_ROOT:.3f} m, c_tip = {C_TIP:.3f} m")
    print(
        f"  LE sweep = {math.degrees(sweep_at(0.0)):.2f} deg "
        f"(sweep at {SWEEP_LOC:.2f}c = {SWEEP_DEG:.2f} deg)"
    )
    print(f"  MAC      = {MAC:.3f} m  (y_MAC = {Y_MAC:.3f} m, x_LE_MAC = {X_LE_MAC:.3f} m)")
