# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""The weighted WP2 scorecard, built from the CSVs the earlier stages wrote.

Reads every airfoil's cruise polar, landing polar and mcrit summary row,
extracts the six scoring criteria, normalizes each one ratio-to-best across
the candidate set (x/max(x), or 1/|x| over its max for pitching moment) and
applies config.scoring.SCORING_WEIGHTS. Two tables come out:

- the detail table: one row per airfoil with the raw, normalized and
  weighted value of every criterion, the context quantities behind them, and
  validity_notes;
- the summary table: the screenshot-shaped deliverable this project started
  from -- criteria as rows, airfoils as columns, a Weight column and a Total
  score row.

Which criteria exist, how each is scored and what it is worth is config
(config/scoring.py), not this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from b12wp2 import config
from b12wp2.config import paths, scoring
from b12wp2.xfoil import polar_analysis


def _load_raw_metrics(airfoil_stem: str) -> dict[str, float | str]:
    cruise = pd.read_csv(paths.polar_csv(airfoil_stem, "cruise"))
    landing = pd.read_csv(paths.polar_csv(airfoil_stem, "landing"))
    mcrit_summary = pd.read_csv(paths.MCRIT_SUMMARY_CSV).set_index("airfoil")

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
            "cannot score mach_critical for this airfoil. Widen "
            "config/solvers.py's MCRIT_MACH_GRID and rerun scripts/mcrit_sweep.py; "
            "do not silently drop it from scoring."
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
        # Angle for Cl_max (landing polar) minus the cruise angle of attack.
        # The two come from different polars -- landing Re/M for the stall
        # angle, cruise Re/M for the operating angle -- so this is a
        # mission-level "how much alpha is left before the wing stalls on
        # approach", not a single-condition margin. It is the definition the
        # WP2 trade-off table uses (sheet row 17 = row 16 - row 13).
        "stall_margin": landing_stall_alpha - cruise_op["alpha"],
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
    """Ratio-to-best normalization onto (0, 1] across the candidate set.

    "max"      -> x_i / max_j(x_j)
    "min_abs"  -> (1/|x_i|) / max_j(1/|x_j|), i.e. min_j(|x_j|) / |x_i|

    The best candidate scores exactly 1.0 and the rest keep their
    proportional distance from it; see config/scoring.py's
    CRITERION_DIRECTION for why this and not min-max.
    """
    raw = 1.0 / values.abs() if direction == "min_abs" else values
    hi = raw.max()
    if hi <= 0:
        raise ValueError(
            f"cannot ratio-normalize a criterion whose best value is {hi}: "
            "x/max(x) is only meaningful for a positive-valued criterion"
        )
    return raw / hi


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
        direction = scoring.CRITERION_DIRECTION[criterion]
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
        row: dict[str, object] = {"Criteria / Options": scoring.CRITERION_LABELS[criterion], "Weight": weight}
        row.update({a: detail.loc[a, f"{criterion}_weighted"] for a in airfoils})
        rows.append(row)
    total_row: dict[str, object] = {
        "Criteria / Options": "Total score", "Weight": sum(config.SCORING_WEIGHTS.values()),
    }
    total_row.update({a: detail.loc[a, "total_score"] for a in airfoils})
    rows.append(total_row)
    return pd.DataFrame(rows).set_index("Criteria / Options")
