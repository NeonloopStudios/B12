# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-15
"""WP3.3a: drag due to lift, and the drag polar when CD0 is available.

From the wp3/ folder:

    python -m scripts.induced_drag

Writes:

    results/drag/induced_drag.csv              winglet, AR_eff, e, K, CDi at cruise
    results/drag/induced_drag.png              CDi(CL) without and with the winglet
    results/iteration/oswald_configurations.csv  e and K of the clean, take-off
                                               and landing configurations (WP3.4)
    results/drag/drag_polar.csv / .png         only when the class II CD0 inputs
                                               are all set
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):
    # run as a file instead of `python -m scripts.<name>`: make wp3/src importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from b12wp3.common.planform import TrapezoidalPlanform
from b12wp3.config import drag as drag_cfg
from b12wp3.config import geometry as geometry_cfg
from b12wp3.config import induced as induced_cfg
from b12wp3.config import paths
from b12wp3.config import mission as mission_cfg
from b12wp3.drag.buildup import aircraft_drag_buildup
from b12wp3.drag.induced import (
    InducedDrag,
    induced_drag_factor,
    mean_height_to_span,
    oswald_factor_brandt,
    oswald_factor_raymer_straight,
    oswald_factor_raymer_swept,
    wing_induced_drag,
)
from b12wp3.drag.polar import DragPolar
from b12wp3.plots.induced import plot_drag_polar, plot_induced_drag

POLAR_CL = np.round(np.arange(0.0, 1.0001, 0.05), 4)


def _wing_planform() -> TrapezoidalPlanform:
    """Wing planform from config/geometry.py; raises if an input is unset."""
    named = (
        ("WING_AREA_M2", geometry_cfg.WING_AREA_M2),
        ("WING_ASPECT_RATIO", geometry_cfg.WING_ASPECT_RATIO),
        ("WING_TAPER", geometry_cfg.WING_TAPER),
        ("WING_SWEEP_C4_DEG", geometry_cfg.WING_SWEEP_C4_DEG),
    )
    missing = [name for name, value in named if value is None]
    if (
        missing
        or geometry_cfg.WING_AREA_M2 is None
        or geometry_cfg.WING_ASPECT_RATIO is None
        or geometry_cfg.WING_TAPER is None
        or geometry_cfg.WING_SWEEP_C4_DEG is None
    ):
        raise ValueError("wing inputs not set: " + ", ".join(missing))
    return TrapezoidalPlanform(
        area_m2=geometry_cfg.WING_AREA_M2,
        aspect_ratio=geometry_cfg.WING_ASPECT_RATIO,
        taper=geometry_cfg.WING_TAPER,
        sweep_c4_rad=math.radians(geometry_cfg.WING_SWEEP_C4_DEG),
        panels=geometry_cfg.WING_PANELS,
    )


def _cruise_rows(
    planform: TrapezoidalPlanform,
    height_to_span: float,
    winglet_height_m: float,
    without: InducedDrag,
    clean: InducedDrag,
) -> pd.DataFrame:
    """One row per quantity: section, quantity, value, unit, note."""
    cl = mission_cfg.CRUISE_LIFT_COEFFICIENT
    rows: list[tuple[str, str, float, str, str]] = []
    for ref in induced_cfg.REFERENCE_WINGLETS:
        rows.append(("reference", f"{ref.aircraft} h", ref.height_m, "m", ref.source))
        rows.append(("reference", f"{ref.aircraft} b", ref.span_m, "m", ref.source))
        rows.append(("reference", f"{ref.aircraft} h/b", ref.height_to_span, "-", ""))
    rows += [
        ("winglet", "h/b (mean)", height_to_span, "-", "mean of the references"),
        ("winglet", "span b", planform.span_m, "m", "wing, sqrt(S AR)"),
        ("winglet", "height h", winglet_height_m, "m", "(h/b) b"),
        ("winglet", "tip chord c_t", planform.tip_chord_m, "m", "wing tip"),
        ("wing", "AR", planform.aspect_ratio, "-", ""),
        ("wing", "sweep c/2", math.degrees(clean.sweep_half_chord_rad), "deg", ""),
        ("wing", "AR_eff", clean.effective_aspect_ratio, "-", "AR + 1.9 (h/b) AR"),
        ("cruise", "CL", cl, "-", "WP3 chapter 1"),
    ]
    for case in (without, clean):
        note = f"h = {0.0 if case is without else winglet_height_m:.3f} m"
        rows += [
            (case.name, "e", case.oswald, "-", "Brandt at AR_eff; " + note),
            (case.name, "K", case.k, "-", "1 / (pi AR_eff e)"),
            (case.name, "CDi at cruise CL", case.cdi(cl), "-", "K CL^2"),
        ]
    rows += _method_comparison_rows(planform, clean.effective_aspect_ratio)
    rows.append(
        (
            "winglet",
            "CDi change at cruise CL",
            100.0 * (clean.cdi(cl) / without.cdi(cl) - 1.0),
            "%",
            "with vs without winglet",
        )
    )
    return pd.DataFrame(rows, columns=["section", "quantity", "value", "unit", "note"])


def _method_comparison_rows(
    planform: TrapezoidalPlanform, effective_aspect_ratio: float
) -> list[tuple[str, str, float, str, str]]:
    """e and cruise CDi of the three slide-72 formulas, without and with the
    winglet. With the winglet, e is evaluated at AR_eff (slide 75)."""
    cl = mission_cfg.CRUISE_LIFT_COEFFICIENT
    ar = planform.aspect_ratio
    sweep_c2 = planform.sweep_at(0.5)
    sweep_le = planform.sweep_at(0.0)
    methods: tuple[tuple[str, Callable[[float], float]], ...] = (
        ("Brandt", lambda a: oswald_factor_brandt(a, sweep_c2)),
        ("Raymer swept", lambda a: oswald_factor_raymer_swept(a, sweep_le)),
        ("Raymer straight", oswald_factor_raymer_straight),
    )
    rows: list[tuple[str, str, float, str, str]] = [
        ("comparison", "sweep LE", math.degrees(sweep_le), "deg", "Raymer swept")
    ]
    for name, oswald in methods:
        e_plain = oswald(ar)
        e_eff = oswald(effective_aspect_ratio)
        cdi_plain = induced_drag_factor(ar, e_plain) * cl**2
        cdi_eff = induced_drag_factor(effective_aspect_ratio, e_eff) * cl**2
        rows += [
            ("comparison", f"{name} e(AR)", e_plain, "-", "no winglet"),
            ("comparison", f"{name} e(AR_eff)", e_eff, "-", "winglet"),
            ("comparison", f"{name} CDi no winglet", cdi_plain, "-", "cruise CL"),
            ("comparison", f"{name} CDi winglet", cdi_eff, "-", "cruise CL"),
            (
                "comparison",
                f"{name} CDi change",
                100.0 * (cdi_eff / cdi_plain - 1.0),
                "%",
                "winglet vs no winglet",
            ),
        ]
    return rows


def _configurations(
    planform: TrapezoidalPlanform, winglet_height_m: float
) -> tuple[pd.DataFrame, list[str]]:
    """e and K per flap configuration; also the names of unset flap inputs."""
    settings = (
        ("clean", 0.0, "FLAP_DEFLECTION_CLEAN"),
        (
            "take-off",
            mission_cfg.TAKEOFF_TE_FLAP_DEFLECTION_DEG,
            "TAKEOFF_TE_FLAP_DEFLECTION_DEG",
        ),
        (
            "landing",
            mission_cfg.LANDING_TE_FLAP_DEFLECTION_DEG,
            "LANDING_TE_FLAP_DEFLECTION_DEG",
        ),
    )
    rows = []
    unset = []
    for name, deflection, setting in settings:
        if deflection is None:
            unset.append(setting)
            continue
        case = wing_induced_drag(
            planform,
            winglet_height_m,
            name=name,
            flap_deflection_deg=deflection,
            oswald_flap_slope_per_deg=induced_cfg.OSWALD_FLAP_SLOPE_PER_DEG,
        )
        rows.append(
            {
                "configuration": name,
                "flap_deflection_deg": deflection,
                "aspect_ratio": case.aspect_ratio,
                "effective_aspect_ratio": case.effective_aspect_ratio,
                "oswald_clean": case.oswald_clean,
                "delta_oswald_flap": case.oswald - case.oswald_clean,
                "oswald": case.oswald,
                "k": case.k,
            }
        )
    return pd.DataFrame(rows), unset


def _class_ii_cd0() -> float | None:
    """CD0 of the class II build-up at cruise, or None if an input is unset."""
    try:
        geometry = geometry_cfg.aircraft_geometry()
    except ValueError as err:
        print(f"  CD0: {err}")
        return None
    buildup = aircraft_drag_buildup(
        geometry, mission_cfg.CRUISE_CONFIGURATION, drag_cfg.DRAG_SETTINGS
    )
    return buildup.cd0


def main() -> None:
    planform = _wing_planform()
    height_to_span = mean_height_to_span(induced_cfg.REFERENCE_WINGLETS)
    winglet_height_m = height_to_span * planform.span_m
    without = wing_induced_drag(planform, 0.0, name="no winglet")
    clean = wing_induced_drag(planform, winglet_height_m, name="winglet")
    cl = mission_cfg.CRUISE_LIFT_COEFFICIENT

    table = _cruise_rows(planform, height_to_span, winglet_height_m, without, clean)
    paths.DRAG_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.INDUCED_DRAG_CSV, index=False)
    plot_induced_drag(without, clean, cl, paths.INDUCED_DRAG_PNG)
    print(table[["section", "quantity", "value", "unit"]].to_string(index=False))
    print(f"\n  -> {paths.INDUCED_DRAG_CSV}\n  -> {paths.INDUCED_DRAG_PNG}")

    configurations, unset = _configurations(planform, winglet_height_m)
    paths.ITERATION_DIR.mkdir(parents=True, exist_ok=True)
    configurations.to_csv(paths.OSWALD_CONFIGURATIONS_CSV, index=False)
    print("\n" + configurations.to_string(index=False))
    print(f"\n  -> {paths.OSWALD_CONFIGURATIONS_CSV}")
    if unset:
        print("  not written, input unset in config/mission.py: " + ", ".join(unset))

    cd0 = _class_ii_cd0()
    if cd0 is None:
        print("  drag polar not written: class II CD0 not available")
        return
    polar = DragPolar(cd0=cd0, k=clean.k)
    polar.to_dataframe(POLAR_CL).to_csv(paths.DRAG_POLAR_CSV, index=False)
    plot_drag_polar(polar, cl, paths.DRAG_POLAR_PNG)
    print(
        f"\n  CD0 = {polar.cd0:.5f}, K = {polar.k:.5f}, "
        f"(L/D)max = {polar.max_lift_to_drag:.2f} at CL = {polar.cl_at_max_lift_to_drag:.3f}, "
        f"L/D at cruise = {polar.lift_to_drag(cl):.2f}"
    )
    print(f"  -> {paths.DRAG_POLAR_CSV}\n  -> {paths.DRAG_POLAR_PNG}")


if __name__ == "__main__":
    main()
