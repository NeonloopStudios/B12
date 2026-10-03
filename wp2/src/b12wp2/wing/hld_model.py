# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-03
"""OpenVSP model of the wing with its high-lift devices and ailerons shown
in colour (devices retracted). For visualisation only, not for VSPAERO.

OpenVSP colours a whole geom, never a region of one, so the wing is tiled
from panels: one WING geom per (eta0..eta1, x/c0..x/c1) patch, each with
the same planform parameters as the analysis wing of b12wp2.wing.geometry
over its span range, cut chordwise with OpenVSP's own section trim
(LE_Trim_X_Chord / TE_Trim_X_Chord, which cuts in place and closes the cut
with a flat face). The panels then lie exactly on the analysis wing's
surface; max_gap() checks that numerically.

The layout (span and chord fractions, which devices) is config.hld; the
VSPAERO model in b12wp2.wing.vspaero is untouched -- the flat cut faces
would be meaningless to a VLM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvsp as vsp

from b12wp2.config import hld as hld_cfg, wing as cfg
from b12wp2.config.hld import Device
from b12wp2.wing import geometry as wg

# ============================================================
#  LAYOUT
# ============================================================

# kind -> (OpenVSP material, wireframe RGB). A material is either one of
# OpenVSP's built-in ones or a key of CUSTOM_MATERIALS.
STYLE: dict[str, tuple[str, tuple[int, int, int]]] = {
    "TE device": ("B12 dark blue", (0, 82, 158)),
    "LE device": ("Green Plastic", (0, 158, 115)),
    "aileron": ("Red Plastic", (213, 94, 0)),
    "fixed": ("B12 dark grey", (30, 30, 30)),
}

# Custom materials, name -> diffuse RGB [0-1]. The built-in blue ("Blue
# Plastic") has a light lavender ambient term and the built-in greys a wide
# specular highlight, both of which render close to white.
CUSTOM_MATERIALS: dict[str, tuple[float, float, float]] = {
    "B12 dark blue": (0.0, 0.32, 0.62),
    "B12 dark grey": (0.40, 0.40, 0.40),
}


@dataclass(frozen=True)
class Panel:
    """One patch of the right half-wing: eta = 2y/b, xc = x/c."""

    name: str
    kind: str  # key of STYLE
    eta0: float
    eta1: float
    xc0: float
    xc1: float


def _find(devices: tuple[Device, ...], name: str) -> Device:
    for d in devices:
        if d.name == name:
            return d
    raise ValueError(f"Unknown device {name!r}, expected one of {[d.name for d in devices]}")


def layout(te_name: str = hld_cfg.SELECTED_TE, le_name: str = hld_cfg.SELECTED_LE) -> list[Panel]:
    """Panels tiling the half-wing for the given TE/LE device names.

    The span is split at every device boundary; per strip the front is the
    LE device or fixed, the back the TE device, the aileron or fixed, and
    the rest is fixed structure. Neighbouring strips with the same panel
    are merged.
    """
    te = _find(hld_cfg.TE_DEVICES, te_name)
    le = _find(hld_cfg.LE_DEVICES, le_name)
    has_le = le.dcl_max > 0.0

    etas = sorted({0.0, 1.0, hld_cfg.ETA_IN, hld_cfg.ETA_OUT_TE, hld_cfg.ETA_OUT_LE, hld_cfg.ETA_OUT_AILERON})
    strips: list[list[tuple[str, str, float, float, float, float]]] = []
    for e0, e1 in zip(etas[:-1], etas[1:]):
        mid = (e0 + e1) / 2
        parts = []
        front = 0.0
        if has_le and hld_cfg.ETA_IN <= mid <= hld_cfg.ETA_OUT_LE:
            front = hld_cfg.SLAT_CHORD_RATIO
            parts.append((le.name, "LE device", 0.0, front))
        back = 1.0
        if hld_cfg.ETA_IN <= mid <= hld_cfg.ETA_OUT_TE:
            back = 1.0 - hld_cfg.FLAP_CHORD_RATIO
            parts.append((te.name, "TE device", back, 1.0))
        elif hld_cfg.ETA_OUT_TE <= mid <= hld_cfg.ETA_OUT_AILERON:
            back = 1.0 - hld_cfg.AILERON_CHORD_RATIO
            parts.append(("Aileron", "aileron", back, 1.0))
        parts.append(("Fixed", "fixed", front, back))
        strips.append([(n, k, x0, x1, e0, e1) for n, k, x0, x1 in parts])

    # merge spanwise neighbours with the same name and chord range
    merged: list[list] = []
    for strip in strips:
        for name, kind, x0, x1, e0, e1 in strip:
            for m in merged:
                if m[0] == name and m[2] == x0 and m[3] == x1 and math.isclose(m[5], e0):
                    m[5] = e1
                    break
            else:
                merged.append([name, kind, x0, x1, e0, e1])

    n_fixed = sum(m[1] == "fixed" for m in merged)
    panels, i_fixed = [], 0
    for name, kind, x0, x1, e0, e1 in merged:
        if kind == "fixed" and n_fixed > 1:
            i_fixed += 1
            name = f"Fixed {i_fixed}"
        panels.append(Panel(name, kind, e0, e1, x0, x1))
    return panels


# ============================================================
#  OPENVSP MODEL
# ============================================================


def _root_offset(eta: float) -> tuple[float, float, float]:
    """Untwisted leading-edge point at eta, in the model frame (wing
    incidence included, rotated about the wing root leading edge)."""
    s = eta * wg.B_HALF  # OpenVSP's Span is measured along the dihedral
    dih = math.radians(cfg.DIHEDRAL_DEG)
    x, y, z = s * wg.TAN_SWEEP_LE, s * math.cos(dih), s * math.sin(dih)
    inc = math.radians(cfg.INCIDENCE_DEG)
    return (
        cfg.X_LE_ROOT + x * math.cos(inc) + z * math.sin(inc),
        y,
        cfg.Z_ROOT - x * math.sin(inc) + z * math.cos(inc),
    )


def _chord(eta: float) -> float:
    return wg.C_ROOT - (wg.C_ROOT - wg.C_TIP) * eta


def _twist_deg(eta: float) -> float:
    """Local twist [deg] of the analysis wing at eta.

    OpenVSP lofts a wing section linearly (a ruled surface) between the
    untwisted root and the twisted tip airfoil, so what varies linearly
    along the span is the displacement of a point about the twist axis, not
    the angle. The blended section is the airfoil rotated by

        tan(theta) = eta c_tip sin(theta_tip) / ((1 - eta) c_root + eta c_tip cos(theta_tip)),

    e.g. -1.09 deg instead of -1.5 deg at eta = 0.75 for a -2 deg tip.
    """
    th = math.radians(cfg.TWIST_TIP_DEG)
    return math.degrees(math.atan2(
        eta * wg.C_TIP * math.sin(th), (1 - eta) * wg.C_ROOT + eta * wg.C_TIP * math.cos(th)
    ))


def _set_trim(xs: str, xc0: float, xc1: float) -> None:
    if xc0 > 0.0:
        vsp.SetParmVal(vsp.GetXSecParm(xs, "LE_Trim_Type"), vsp.TRIM_X)
        vsp.SetParmVal(vsp.GetXSecParm(xs, "LE_Trim_AbsRel"), vsp.REL)
        vsp.SetParmVal(vsp.GetXSecParm(xs, "LE_Trim_X_Chord"), xc0)
    if xc1 < 1.0:
        vsp.SetParmVal(vsp.GetXSecParm(xs, "TE_Trim_Type"), vsp.TRIM_X)
        vsp.SetParmVal(vsp.GetXSecParm(xs, "TE_Trim_AbsRel"), vsp.REL)
        vsp.SetParmVal(vsp.GetXSecParm(xs, "TE_Trim_X_Chord"), 1.0 - xc1)


def add_panel(panel: Panel) -> str:
    """Add one panel to the current OpenVSP model and return its geom ID."""
    gid: str = vsp.AddGeom("WING")
    vsp.SetGeomName(gid, panel.name)

    x0, y0, z0 = _root_offset(panel.eta0)
    vsp.SetParmVal(gid, "X_Rel_Location", "XForm", x0)
    vsp.SetParmVal(gid, "Y_Rel_Location", "XForm", y0)
    vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", z0)
    vsp.SetParmVal(gid, "Y_Rel_Rotation", "XForm", cfg.INCIDENCE_DEG)

    grp = "XSec_1"
    vsp.SetDriverGroup(gid, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
    vsp.SetParmVal(gid, "Span", grp, (panel.eta1 - panel.eta0) * wg.B_HALF)
    vsp.SetParmVal(gid, "Root_Chord", grp, _chord(panel.eta0))
    vsp.SetParmVal(gid, "Tip_Chord", grp, _chord(panel.eta1))
    vsp.SetParmVal(gid, "Sweep", grp, cfg.SWEEP_DEG)
    vsp.SetParmVal(gid, "Sweep_Location", grp, cfg.SWEEP_LOC)
    vsp.SetParmVal(gid, "Dihedral", grp, cfg.DIHEDRAL_DEG)
    # absolute twist per section, about the same axis as the analysis wing
    for g, eta in (("XSec_0", panel.eta0), ("XSec_1", panel.eta1)):
        vsp.SetParmVal(gid, "Twist", g, _twist_deg(eta))
        vsp.SetParmVal(gid, "Twist_Location", g, cfg.TWIST_LOC)
    vsp.SetParmVal(gid, "SectTess_U", grp, max(3, round(cfg.SPAN_TESS * (panel.eta1 - panel.eta0))))
    vsp.SetParmVal(gid, "Tess_W", "Shape", cfg.CHORD_TESS)
    vsp.Update()

    surf = vsp.GetXSecSurf(gid, 0)
    for i, path in ((0, cfg.AIRFOIL_ROOT), (1, cfg.AIRFOIL_TIP)):
        wg.set_airfoil(surf, i, path)
        _set_trim(vsp.GetXSec(surf, i), panel.xc0, panel.xc1)

    material, rgb = STYLE[panel.kind]
    vsp.SetGeomMaterialName(gid, material)
    vsp.SetGeomWireColor(gid, *rgb)
    vsp.Update()
    return gid


def _material_terms(diffuse: tuple[float, float, float]) -> dict[str, tuple[float, ...]]:
    """Ambient/diffuse/specular/emissive RGBA and shininess of a matte material."""
    return {
        "Ambient": (*(0.3 * c for c in diffuse), 1.0),
        "Diffuse": (*diffuse, 1.0),
        "Specular": (0.25, 0.25, 0.25, 1.0),
        "Emissive": (0.0, 0.0, 0.0, 1.0),
        "Shininess": (30.0,),
    }


def build_hld_wing(panels: list[Panel]) -> list[str]:
    """Add all panels to the current OpenVSP model, return their geom IDs."""
    known = vsp.GetMaterialNames()
    for name, diffuse in CUSTOM_MATERIALS.items():
        if name not in known:
            t = _material_terms(diffuse)
            vsp.AddMaterial(
                name, *(vsp.vec3d(*t[k][:3]) for k in ("Ambient", "Diffuse", "Specular", "Emissive")),
                1.0, t["Shininess"][0],
            )
    return [add_panel(p) for p in panels]


def write_vsp3(path: Path) -> None:
    """Write the current model, with CUSTOM_MATERIALS stored in the file.

    OpenVSP only saves materials flagged as user materials, a flag that
    vsp.AddMaterial does not set (only reading them from a .vsp3 does), so
    the file would keep the material names but not their colours. They are
    written into the file's <Materials> node here, in OpenVSP's own format
    (OpenVSP decodes that node before the geoms that refer to it).
    """
    vsp.WriteVSPFile(str(path))

    def node(name: str, diffuse: tuple[float, float, float]) -> str:
        parts = [f"<Name>{name}</Name>"]
        for k, vals in _material_terms(diffuse).items():
            # vectors as "a, b, c, d, " like OpenVSP's XmlUtil, the shininess a plain double
            value = f"{vals[0]:.6f}" if k == "Shininess" else "".join(f"{v:.6f}, " for v in vals)
            parts.append(f"<{k}>{value}</{k}>")
        return f"<Material>{''.join(parts)}</Material>"

    text = path.read_text()
    if "<Materials/>" not in text:
        raise RuntimeError(f"No empty <Materials/> node in {path}")
    materials = "".join(node(n, d) for n, d in CUSTOM_MATERIALS.items())
    path.write_text(text.replace("<Materials/>", f"<Materials>{materials}</Materials>", 1))


def max_gap(panel_ids: list[str], n: int = 41) -> float:
    """Largest distance [m] from a point of the analysis wing's surface to
    the nearest panel: zero if the panels tile the wing exactly.

    Builds the analysis wing (b12wp2.wing.geometry) temporarily in the
    current model, samples an n x n grid of its right half and deletes it.
    """
    ref = wg.build_wing()
    try:
        worst = 0.0
        for u in np.linspace(0.0, 1.0, n):
            for w in np.linspace(0.0, 1.0, n):
                p = vsp.CompPnt01(ref, 0, u, w)
                worst = max(worst, min(vsp.ProjPnt01I(g, p)[0] for g in panel_ids))
        return worst
    finally:
        vsp.DeleteGeom(ref)
        vsp.Update()
