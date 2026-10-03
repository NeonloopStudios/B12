# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-03
"""WP2: write the OpenVSP model of the wing with the selected high-lift
devices and the ailerons in colour (retracted), for figures and inspection.

Drives b12wp2.wing.hld_model with the layout and the selected devices of
config.hld (SELECTED_TE, SELECTED_LE), checks that the panels tile the
analysis wing and writes the model.

Must be run from the OpenVSP Python environment, from the wp2/ folder:

    python -m scripts.hld_vsp_model

Writes wp2/results/hld/wing_hld.vsp3. The wireframe is already coloured; for
the shaded view select all geoms in the Geom Browser and click "Shade" (the
draw type is a GUI setting and is not stored in the .vsp3).
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

import openvsp as vsp

from b12wp2.config import hld as hld_cfg, paths
from b12wp2.wing import hld_model as hm

GAP_TOL = 1e-3  # [m], largest allowed distance between the panels and the analysis wing


def main() -> None:
    panels = hm.layout()
    print(f"Selected: TE '{hld_cfg.SELECTED_TE}', LE '{hld_cfg.SELECTED_LE}' (retracted)\n")
    print(f"{'panel':<12} {'kind':<10} {'eta':>11} {'x/c':>11}  material")
    for p in panels:
        print(f"{p.name:<12} {p.kind:<10} {p.eta0:5.2f}-{p.eta1:<5.2f} {p.xc0:5.2f}-{p.xc1:<5.2f}  {hm.STYLE[p.kind][0]}")

    vsp.VSPRenew()
    ids = hm.build_hld_wing(panels)
    gap = hm.max_gap(ids)
    print(f"\nLargest distance analysis wing -> panels: {gap * 1000:.2f} mm")
    if gap > GAP_TOL:
        raise RuntimeError(f"Panels do not tile the analysis wing (gap {gap:.4f} m > {GAP_TOL} m)")

    paths.HLD_DIR.mkdir(parents=True, exist_ok=True)
    hm.write_vsp3(paths.HLD_VSP3)
    print(f"Model written to: {paths.HLD_VSP3}")


if __name__ == "__main__":
    main()
