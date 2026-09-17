"""WP2 Stage 5: Mach-critical determination for every candidate airfoil.

For each airfoil: get the Cp(x) distribution at a safely subsonic baseline
Mach (M=0.2) and the cruise-relevant Cl (Cl_n), Karman-Tsien-correct it
across a Mach grid, and find where the corrected Cp_min first crosses the
critical (sonic) Cp -- that's M_crit. Cross-check with the Korn equation for
drag-divergence Mach, M_dd. See wp2/xfoil_plan.md Stage 5 for the method
and why this is the standard, legitimate use of a panel method at
transonic conditions (it flags an oncoming shock, it does not resolve one).

Writes wp2/results/data/<airfoil>_mcrit.csv (mach, cp_min_corrected,
cp_crit) and wp2/results/data/mcrit_summary.csv (one row per airfoil:
m_crit, m_dd, margin, thickness_to_chord, cl_cruise, kappa_a).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import compressibility, config, geometry, xfoil_runtime

BASELINE_MACH = 0.2
MACH_GRID = np.arange(0.30, 0.951, 0.005)


def run_mcrit_analysis(airfoil_path: Path) -> tuple[float | None, pd.DataFrame, dict[str, float]]:
    """Return (M_crit, mach-sweep DataFrame, summary dict) for one airfoil."""
    airfoil = xfoil_runtime.load_airfoil_dat(airfoil_path)
    cl_n = config.CRUISE.cl_normal
    thickness_to_chord = geometry.max_thickness_to_chord(airfoil)
    kappa_a = config.kappa_a(airfoil_path.stem)

    m_dd = compressibility.korn_mdd(kappa_a, thickness_to_chord, cl_n)
    margin = m_dd - config.CRUISE.mach_normal

    with xfoil_runtime.xfoil_session(
        airfoil,
        mach=BASELINE_MACH,
        reynolds=config.CRUISE.reynolds_normal,
        ncrit=config.CRUISE.ncrit,
    ) as xf:
        baseline = xfoil_runtime.solve_at_cl(xf, cl_n)
        if not baseline["converged"]:
            summary = {
                "m_crit": float("nan"),
                "m_dd": m_dd,
                "margin_m_dd_minus_m_n": margin,
                "thickness_to_chord": thickness_to_chord,
                "cl_cruise": cl_n,
                "kappa_a": kappa_a,
            }
            return None, pd.DataFrame(columns=["mach", "cp_min_corrected", "cp_crit"]), summary

        cp0 = xfoil_runtime.cp_distribution(xf)["cp"].to_numpy()

    m_crit, sweep = compressibility.find_mcrit(cp0, MACH_GRID)
    summary = {
        "m_crit": float("nan") if m_crit is None else m_crit,
        "m_dd": m_dd,
        "margin_m_dd_minus_m_n": margin,
        "thickness_to_chord": thickness_to_chord,
        "cl_cruise": cl_n,
        "kappa_a": kappa_a,
    }
    return m_crit, sweep, summary


def main() -> None:
    out_dir = config.RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Baseline Mach={BASELINE_MACH}, Cl_n={config.CRUISE.cl_normal:.4f}, "
        f"cruise M_n={config.CRUISE.mach_normal:.4f} (Korn margin reference)"
    )

    summary_rows: list[dict[str, float | str]] = []
    for airfoil_path in config.discover_airfoils():
        stem = airfoil_path.stem
        m_crit, sweep, summary = run_mcrit_analysis(airfoil_path)

        sweep_path = out_dir / f"{stem}_mcrit.csv"
        sweep.to_csv(sweep_path, index=False)

        if m_crit is None:
            print(f"  [WARN] {stem}: baseline Cl={config.CRUISE.cl_normal:.4f} solve at "
                  f"M={BASELINE_MACH} did not converge -- no M_crit, M_dd/margin still computed")
        elif np.isnan(summary["m_crit"]) and not sweep.empty:
            print(f"  [WARN] {stem}: no Mach in [{MACH_GRID[0]:.2f}, {MACH_GRID[-1]:.2f}] "
                  f"reached critical -- M_crit above this grid's range")
        else:
            print(
                f"  {stem}: M_crit={m_crit:.3f}, M_dd={summary['m_dd']:.3f} "
                f"(kappa_A={summary['kappa_a']}, t/c={summary['thickness_to_chord']:.4f}), "
                f"margin(M_dd-M_n)={summary['margin_m_dd_minus_m_n']:+.3f}"
            )
        print(f"    -> {sweep_path}")

        summary_rows.append({"airfoil": stem, **summary})

    summary_path = out_dir / "mcrit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  -> {summary_path}")


if __name__ == "__main__":
    main()
