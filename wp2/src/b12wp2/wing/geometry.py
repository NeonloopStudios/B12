# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Wing planform geometry and the OpenVSP model built from it.

The design variables themselves are in b12wp2.config.wing (and the
quarter-chord sweep in b12wp2.config.mission, shared with the XFoil
sweep-theory reduction so the 3D model and the 2D sections cannot use
different angles). What is computed here is everything that follows from
them: span, chords, MAC, the sweep of any chord line, the t/c at the MAC --
and the OpenVSP geom itself.

Deliberately has its own .dat reader instead of reusing
b12wp2.xfoil.runtime.load_airfoil_dat: importing that module loads the
compiled XFoil DLL, which is not installed in the OpenVSP Python environment
(and is not needed here -- only the coordinates are).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import openvsp as vsp

from b12wp2.config import mission, wing as cfg

# ============================================================
#  DERIVED QUANTITIES
# ============================================================

B = math.sqrt(cfg.S_REF * cfg.AR)  # span [m]
B_HALF = B / 2
C_ROOT = 2 * cfg.S_REF / (B * (1 + cfg.TAPER))
C_TIP = cfg.TAPER * C_ROOT

MAC = 2 / 3 * C_ROOT * (1 + cfg.TAPER + cfg.TAPER**2) / (1 + cfg.TAPER)
Y_MAC = B / 6 * (1 + 2 * cfg.TAPER) / (1 + cfg.TAPER)

TAN_SWEEP_LE = math.tan(mission.SWEEP_RAD) + cfg.SWEEP_LOC * (C_ROOT - C_TIP) / B_HALF
X_LE_MAC = cfg.X_LE_ROOT + Y_MAC * TAN_SWEEP_LE


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
    tc_root = airfoil_thickness(cfg.AIRFOIL_ROOT)
    tc_tip = airfoil_thickness(cfg.AIRFOIL_TIP)
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
    vsp.SetGeomName(wid, cfg.WING_NAME)

    vsp.SetParmVal(wid, "X_Rel_Location", "XForm", cfg.X_LE_ROOT)
    vsp.SetParmVal(wid, "Z_Rel_Location", "XForm", cfg.Z_ROOT)
    vsp.SetParmVal(wid, "Y_Rel_Rotation", "XForm", cfg.INCIDENCE_DEG)

    grp = "XSec_1"
    vsp.SetDriverGroup(
        wid, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER
    )
    vsp.SetParmVal(wid, "Span", grp, B_HALF)
    vsp.SetParmVal(wid, "Root_Chord", grp, C_ROOT)
    vsp.SetParmVal(wid, "Tip_Chord", grp, C_TIP)
    vsp.SetParmVal(wid, "Sweep", grp, cfg.SWEEP_DEG)
    vsp.SetParmVal(wid, "Sweep_Location", grp, cfg.SWEEP_LOC)
    vsp.SetParmVal(wid, "Dihedral", grp, cfg.DIHEDRAL_DEG)
    vsp.SetParmVal(wid, "Twist", grp, cfg.TWIST_TIP_DEG)
    vsp.SetParmVal(wid, "Twist_Location", grp, cfg.TWIST_LOC)
    vsp.SetParmVal(wid, "SectTess_U", grp, cfg.SPAN_TESS)
    vsp.SetParmVal(wid, "OutCluster", grp, cfg.OUT_CLUSTER)
    vsp.SetParmVal(wid, "Tess_W", "Shape", cfg.CHORD_TESS)
    vsp.Update()

    surf = vsp.GetXSecSurf(wid, 0)
    _set_airfoil(surf, 0, cfg.AIRFOIL_ROOT)
    _set_airfoil(surf, 1, cfg.AIRFOIL_TIP)
    vsp.Update()
    return wid


def print_summary(wid: str) -> None:
    s_vsp = vsp.GetParmVal(wid, "TotalArea", "WingGeom")
    b_vsp = vsp.GetParmVal(wid, "TotalSpan", "WingGeom")
    print(f"Wing '{cfg.WING_NAME}' ({cfg.AIRFOIL_ROOT.stem} root, {cfg.AIRFOIL_TIP.stem} tip):")
    print(f"  S        = {cfg.S_REF:.3f} m^2   (OpenVSP: {s_vsp:.3f})")
    print(f"  b        = {B:.3f} m     (OpenVSP: {b_vsp:.3f})")
    print(f"  AR       = {cfg.AR:.2f}, taper = {cfg.TAPER:.3f}")
    print(f"  c_root   = {C_ROOT:.3f} m, c_tip = {C_TIP:.3f} m")
    print(
        f"  LE sweep = {math.degrees(sweep_at(0.0)):.2f} deg "
        f"(sweep at {cfg.SWEEP_LOC:.2f}c = {cfg.SWEEP_DEG:.2f} deg)"
    )
    print(f"  MAC      = {MAC:.3f} m  (y_MAC = {Y_MAC:.3f} m, x_LE_MAC = {X_LE_MAC:.3f} m)")
