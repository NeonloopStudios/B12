# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 build the weighted scorecard.

Reads every results/data/*.csv (cruise polars, landing polars, mcrit
summary), extracts the six scoring criteria per airfoil, normalizes each
0-1 across the candidate set, applies config.SCORING_WEIGHTS, and writes:

- wp2/results/scorecard.csv: the original screenshot-shaped summary
  (criteria as rows, airfoils as columns, a Weight column, a Total score
  row) -- the exact deliverable shape this project started from.
- wp2/results/scorecard_detail.csv: one row per airfoil with the raw,
  normalized, and weighted value for every criterion plus validity_notes
  -- the fuller breakdown make_plots.py needs and this module's own
  transparency/verification record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from b12wp2 import config
from b12wp2.xfoil import polar_analysis

# Direction each raw criterion is scored in. "max" = higher raw value is
# better (normalize ascending). "min_abs" = smaller |value| is better
# (normalize the absolute value, then invert): pitching_moment is
# smaller-|Cm|-is-better (standard trim-drag rationale), cl_zero_angle is
# higher-is-better. The other four criteria (mach_critical, cl_cd_cruise,
# cl_max_landing, stall_margin) are unambiguously higher-is-better on
# their own terms.
CRITERION_DIRECTION: dict[str, str] = {
    "mach_critical": "max",
    "cl_cd_cruise": "max",
    "cl_max_landing": "max",
    "stall_margin": "max",
    "cl_zero_angle": "max",
    "pitching_moment": "min_abs",
}

CRITERION_LABELS: dict[str, str] = {
    "mach_critical": "Mach Critical",
    "cl_cd_cruise": "Cl/Cd cruise",
    "cl_max_landing": "Cl max landing",
    "stall_margin": "Stall Angle - Angle in Cruise",
    "cl_zero_angle": "Zero Angle Cl",
    "pitching_moment": "Pitching Moment",
}


def _load_raw_metrics(airfoil_stem: str) -> dict[str, float | str]:
    data_dir = config.RESULTS_DIR / "data"
    cruise = pd.read_csv(data_dir / f"{airfoil_stem}_cruise_polar.csv")
    landing = pd.read_csv(data_dir / f"{airfoil_stem}_landing_polar.csv")
    mcrit_summary = pd.read_csv(data_dir / "mcrit_summary.csv").set_index("airfoil")

    cl_n = config.CRUISE.cl_normal
    cruise_op = polar_analysis.interpolate_at_cl(cruise, cl_n)
    cruise_cl_max, cruise_stall_alpha = polar_analysis.find_cl_max(cruise)
    landing_cl_max, landing_stall_alpha = polar_analysis.find_cl_max(landing)
    cl_zero = polar_analysis.cl_at_zero_angle(cruise)

    m_crit = float(mcrit_summary.loc[airfoil_stem, "m_crit"])
    m_dd = float(mcrit_summary.loc[airfoil_stem, "m_dd"])
    margin = float(mcrit_summary.loc[airfoil_stem, "margin_m_dd_minus_m_n"])

    if np.isnan(m_crit):
        raise ValueError(
            f"{airfoil_stem}: no M_crit found within mcrit_sweep.py's Mach sweep -- "
            "cannot score mach_critical for this airfoil. Widen mcrit_sweep.py's "
            "MACH_GRID and rerun it; do not silently drop it from scoring."
        )

    validity_notes: list[str] = []
    if config.CRUISE.mach_normal > m_crit:
        validity_notes.append(
            f"cruise Cd not physically trustworthy (M_n={config.CRUISE.mach_normal:.3f} "
            f"> M_crit={m_crit:.3f}); Korn margin (M_dd-M_n={margin:+.3f}) is the number "
            "to actually rank transonic suitability on"
        )

    return {
        "mach_critical": m_crit,
        "cl_cd_cruise": cl_n / cruise_op["cd"],
        "cl_max_landing": landing_cl_max,
        "stall_margin": cruise_stall_alpha - cruise_op["alpha"],
        "cl_zero_angle": cl_zero,
        "pitching_moment": cruise_op["cm"],
        # context, not scored directly
        "m_dd": m_dd,
        "margin_m_dd_minus_m_n": margin,
        "cruise_alpha_at_cl_n": cruise_op["alpha"],
        "cruise_cd_at_cl_n": cruise_op["cd"],
        "stall_alpha_cruise": cruise_stall_alpha,
        "cruise_cl_max": cruise_cl_max,
        "landing_stall_alpha": landing_stall_alpha,
        "validity_notes": "; ".join(validity_notes),
    }


def _normalize(values: pd.Series, direction: str) -> pd.Series:
    """Min-max normalize to [0, 1] across the candidate set."""
    raw = values.abs() if direction == "min_abs" else values
    lo, hi = raw.min(), raw.max()
    if hi == lo:
        return pd.Series(1.0, index=values.index)
    norm = (raw - lo) / (hi - lo)
    if direction == "min_abs":
        norm = 1.0 - norm
    return norm


def build_detail_table() -> pd.DataFrame:
    """One row per airfoil: raw/normalized/weighted value for every
    criterion, plus context columns and validity_notes.
    """
    rows = []
    for airfoil_path in config.discover_airfoils():
        metrics = _load_raw_metrics(airfoil_path.stem)
        rows.append({"airfoil": airfoil_path.stem, **metrics})
    raw = pd.DataFrame(rows).set_index("airfoil")

    detail = pd.DataFrame(index=raw.index)
    total = pd.Series(0.0, index=raw.index)
    for criterion, weight in config.SCORING_WEIGHTS.items():
        direction = CRITERION_DIRECTION[criterion]
        norm = _normalize(raw[criterion], direction)
        detail[f"{criterion}_raw"] = raw[criterion]
        detail[f"{criterion}_norm"] = norm
        detail[f"{criterion}_weighted"] = norm * weight
        total += norm * weight

    detail["total_score"] = total
    for col in (
        "m_dd", "margin_m_dd_minus_m_n", "cruise_alpha_at_cl_n", "cruise_cd_at_cl_n",
        "stall_alpha_cruise", "cruise_cl_max", "landing_stall_alpha",
    ):
        detail[col] = raw[col]
    detail["validity_notes"] = raw["validity_notes"]

    return detail.sort_values("total_score", ascending=False)


def build_summary_table(detail: pd.DataFrame) -> pd.DataFrame:
    """Screenshot-shaped table: one row per criterion (weighted
    contribution per airfoil) plus a Total score row, columns = airfoils
    in ranked order.
    """
    airfoils = list(detail.index)  # already ranked by total_score
    rows: list[dict[str, object]] = []
    for criterion, weight in config.SCORING_WEIGHTS.items():
        row: dict[str, object] = {"Criteria / Options": CRITERION_LABELS[criterion], "Weight": weight}
        row.update({a: detail.loc[a, f"{criterion}_weighted"] for a in airfoils})
        rows.append(row)
    total_row: dict[str, object] = {
        "Criteria / Options": "Total score", "Weight": sum(config.SCORING_WEIGHTS.values()),
    }
    total_row.update({a: detail.loc[a, "total_score"] for a in airfoils})
    rows.append(total_row)
    return pd.DataFrame(rows).set_index("Criteria / Options")


def main() -> None:
    detail = build_detail_table()
    summary = build_summary_table(detail)

    detail_path = config.RESULTS_DIR / "scorecard_detail.csv"
    summary_path = config.RESULTS_DIR / "scorecard.csv"
    detail.to_csv(detail_path)
    summary.to_csv(summary_path)

    print(f"WP2 scorecard (weights: {config.SCORING_WEIGHTS})\n")
    print(summary.round(4).to_string())
    print()
    for airfoil, notes in detail["validity_notes"].items():
        if notes:
            print(f"  [{airfoil}] {notes}")
    print(f"\n  -> {summary_path}")
    print(f"  -> {detail_path}")


if __name__ == "__main__":
    main()
