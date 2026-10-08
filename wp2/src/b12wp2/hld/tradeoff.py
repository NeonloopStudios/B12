# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""HLD v2: trade-off of every TE x LE configuration on performance,
occupied space and complexity.

The aerodynamics are the ADSEE increments of b12wp2.hld.sizing, on the
clean wing v1 computed (results/hld/clean_wing.csv). What v2 adds:

1. Each device type takes its own chord (config.hld_v2), which fixes the
   spar it pushes: front spar at max(X_FS_MIN, c_s/c), rear spar at
   1 - c_f/c.
2. Leading-edge devices span ETA_IN..ETA_OUT_LE. The trailing-edge device
   starts at ETA_IN and ends anywhere up to the aileron's inboard edge
   (config.hld_v2.ETA_AILERON_IN): every configuration is evaluated on a
   grid of trailing-edge outboard ends, which always includes the smallest
   one meeting both CL_max requirements.
3. Grid points below either requirement (landing or take-off CL_max) are
   discarded; a configuration with no feasible point is discarded.
4. Three criteria per feasible grid point, each sub-metric min-max
   normalised over all feasible grid points (1 = best):

       performance = w_L * n(CL_max,L / CL_L,req) + w_TO * n(CL_max,TO / CL_TO,req)
       space       = w_LE * n(-x_front_spar)
                     + w_TE * (w_spar * n(-c_f/c) + w_area * n(-S_wf,TE / S))
       complexity  = n(-complexity rank)
       score       = w_P * performance + w_S * space + w_C * complexity

   A configuration's score is the best score over its grid points.

Pure numpy/pandas, no OpenVSP: the planform is rebuilt here from
config.wing (b12wp2.wing.geometry imports openvsp).
"""
from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd

from b12wp2.config import hld as cfg, hld_v2 as v2, planform
from b12wp2.config.hld import Device
from b12wp2.hld import sizing as hs

# config.planform is openvsp-free, unlike b12wp2.wing.geometry
PLANFORM = hs.Planform(planform.S_REF, planform.B, planform.C_ROOT, planform.C_TIP, planform.TAN_SWEEP_LE)

NO_DEVICE = hs.DeviceEffect(0.0, 0.0, 0.0, 1.0, 0.0)

CRITERIA = ("s_performance", "s_space", "s_complexity")


# ============================================================
#  LAYOUT
# ============================================================


def front_spar(le: Device) -> float:
    """Front-spar chord fraction pushed back by the leading-edge device."""
    return max(v2.X_FS_MIN, v2.LE_CHORD_RATIO[le.name])


def rear_spar(te: Device) -> float:
    """Rear-spar chord fraction pushed forward by the trailing-edge device."""
    return 1.0 - v2.TE_CHORD_RATIO[te.name]


def le_effect(planform: hs.Planform, le: Device) -> hs.DeviceEffect:
    if le.dcl_max <= 0.0:
        return NO_DEVICE
    return hs.device_effect(planform, le, v2.LE_CHORD_RATIO[le.name], cfg.ETA_IN, cfg.ETA_OUT_LE)


def te_effect(planform: hs.Planform, te: Device, eta_out: float) -> hs.DeviceEffect:
    return hs.device_effect(planform, te, v2.TE_CHORD_RATIO[te.name], cfg.ETA_IN, eta_out)


def required_te_dcl(clean: hs.CleanWing, le: hs.DeviceEffect, req_l: float, req_to: float) -> float:
    """Landing dCL_max the trailing-edge device must add so that both the
    landing and the take-off (TAKEOFF_FRACTION of landing) CL_max are met."""
    need_l = req_l - clean.cl_max - le.dcl_max_wing
    need_to = (req_to - clean.cl_max) / cfg.TAKEOFF_FRACTION - le.dcl_max_wing
    return max(need_l, need_to)


def min_te_eta_out(
    planform: hs.Planform, clean: hs.CleanWing, te: Device, le: hs.DeviceEffect,
    req_l: float, req_to: float,
) -> float:
    """Smallest trailing-edge outboard end meeting both requirements, NaN if
    even the aileron's inboard edge is not enough."""
    return hs.min_eta_out(
        planform, te, v2.TE_CHORD_RATIO[te.name], cfg.ETA_IN, v2.ETA_AILERON_IN,
        required_te_dcl(clean, le, req_l, req_to),
    )


def te_span_grid(eta_min: float = float("nan")) -> np.ndarray:
    """Trailing-edge outboard ends: ETA_IN..ETA_AILERON_IN in steps of about
    ETA_TE_STEP (ETA_IN itself excluded: zero span), plus eta_min when it is
    a finite point inside that range."""
    n = math.ceil((v2.ETA_AILERON_IN - cfg.ETA_IN) / v2.ETA_TE_STEP - 1e-9)
    etas = np.linspace(cfg.ETA_IN, v2.ETA_AILERON_IN, n + 1)[1:]
    if math.isfinite(eta_min) and cfg.ETA_IN < eta_min <= v2.ETA_AILERON_IN:
        etas = np.union1d(etas, [eta_min])
    return etas


# ============================================================
#  GRID
# ============================================================


def requirements() -> tuple[float, float]:
    if cfg.CL_MAX_L_REQ is None or cfg.CL_MAX_TO_REQ is None:
        raise ValueError("hld.tradeoff: config.hld.CL_MAX_L_REQ and CL_MAX_TO_REQ must both be set")
    return cfg.CL_MAX_L_REQ, cfg.CL_MAX_TO_REQ


def grid(clean: hs.CleanWing, planform: hs.Planform = PLANFORM) -> pd.DataFrame:
    """One row per (TE device, LE device, TE outboard end)."""
    req_l, req_to = requirements()
    s_te_avail = hs.covered_area(planform, cfg.ETA_IN, v2.ETA_AILERON_IN) / planform.s_ref
    rows = []
    for te in cfg.TE_DEVICES:
        for le in cfg.LE_DEVICES:
            le_eff = le_effect(planform, le)
            eta_min = min_te_eta_out(planform, clean, te, le_eff, req_l, req_to)
            for eta in te_span_grid(eta_min):
                te_eff = te_effect(planform, te, float(eta))
                land = hs.configuration(clean, te_eff, le_eff)
                to = hs.configuration(clean, te_eff, le_eff, takeoff=True)
                rows.append({
                    "te_device": te.name,
                    "le_device": le.name,
                    "config": te.name if le.name == "none" else f"{te.name} + {le.name}",
                    "complexity": te.complexity + le.complexity,
                    "eta_out_te": float(eta),
                    "eta_out_te_min": eta_min,
                    "aileron_span_left": 1.0 - float(eta),  # if the aileron moved in to the flap
                    "c_f_c": v2.TE_CHORD_RATIO[te.name],
                    "c_s_c": v2.LE_CHORD_RATIO[le.name],
                    "x_front_spar": front_spar(le),
                    "x_rear_spar": rear_spar(te),
                    "swf_s_te": te_eff.swf_s,
                    "te_area_used": te_eff.swf_s / s_te_avail,  # of the span up to the aileron
                    "swf_s_le": le_eff.swf_s,
                    "dCLmax_te": te_eff.dcl_max_wing,
                    "dCLmax_le": le_eff.dcl_max_wing,
                    "CLmax_L": land.cl_max,
                    "CLmax_TO": to.cl_max,
                    "alpha_stall_L": land.alpha_stall_deg,
                    "alpha_stall_TO": to.alpha_stall_deg,
                    "perf_L": land.cl_max / req_l,
                    "perf_TO": to.cl_max / req_to,
                    # small tolerance: the minimum-span point sits on the requirement
                    "feasible": bool(land.cl_max >= req_l - 1e-9 and to.cl_max >= req_to - 1e-9),
                })
    return pd.DataFrame(rows)


# ============================================================
#  SCORING
# ============================================================


def minmax(x: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Min-max normalisation to 0..1, 1 = best. A constant metric scores 1
    everywhere (it does not separate the candidates)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=x.index)
    n = (x - lo) / (hi - lo)
    return n if higher_is_better else 1.0 - n


def score(df: pd.DataFrame, weights: dict[str, float] = v2.WEIGHTS) -> pd.DataFrame:
    """Add the normalised criteria and the weighted score (NaN where infeasible)."""
    out = df.copy()
    f = out["feasible"]
    if not f.any():
        raise ValueError("hld.tradeoff.score: no configuration meets the CL_max requirements")
    d = out[f]
    pw, sw, tw = v2.PERFORMANCE_WEIGHTS, v2.SPACE_WEIGHTS, v2.TE_SPACE_WEIGHTS
    sub = {
        "s_perf_L": minmax(d["perf_L"], higher_is_better=True),
        "s_perf_TO": minmax(d["perf_TO"], higher_is_better=True),
        "s_le_spar": minmax(d["x_front_spar"], higher_is_better=False),
        "s_te_spar": minmax(d["c_f_c"], higher_is_better=False),
        "s_te_area": minmax(d["swf_s_te"], higher_is_better=False),
        "s_complexity": minmax(d["complexity"].astype(float), higher_is_better=False),
    }
    sub["s_performance"] = pw["landing"] * sub["s_perf_L"] + pw["takeoff"] * sub["s_perf_TO"]
    sub["s_te_space"] = tw["spar"] * sub["s_te_spar"] + tw["area"] * sub["s_te_area"]
    sub["s_space"] = sw["le"] * sub["s_le_spar"] + sw["te"] * sub["s_te_space"]
    for name, s in sub.items():
        out[name] = s  # aligned on the index: NaN on infeasible rows
    out["score"] = (
        weights["performance"] * out["s_performance"]
        + weights["space"] * out["s_space"]
        + weights["complexity"] * out["s_complexity"]
    )
    out["pareto"] = False
    out.loc[f, "pareto"] = pareto_front(out.loc[f, list(CRITERIA)].to_numpy())
    return out


def pareto_front(values: np.ndarray) -> np.ndarray:
    """True for the rows no other row matches or beats on every column (all
    columns higher-is-better) while beating it on at least one."""
    ge = (values[:, None, :] >= values[None, :, :]).all(axis=2)  # ge[j, i]: j >= i everywhere
    gt = (values[:, None, :] > values[None, :, :]).any(axis=2)
    dominated = (ge & gt).any(axis=0)
    return np.logical_not(dominated)


def best_per_config(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration: its best-scoring grid point, ranked, or --
    when no grid point meets the requirements -- its capability with the
    trailing-edge device up to the aileron and the reason it is discarded.
    `pareto` is True when any of the configuration's grid points is on the
    Pareto front of the three criteria."""
    req_l, req_to = requirements()
    feas = scored[scored["feasible"]]
    best = feas.loc[feas.groupby("config")["score"].idxmax()].copy()
    best["pareto"] = best["config"].map(feas.groupby("config")["pareto"].any())
    best = best.sort_values("score", ascending=False)
    best["rank"] = np.arange(1, len(best) + 1)
    best["status"] = "feasible"
    best["reason"] = ""

    out_cfgs = sorted(set(scored["config"]) - set(best["config"]))
    full = scored[scored["config"].isin(out_cfgs)]
    disc = full.loc[full.groupby("config")["eta_out_te"].idxmax()].copy()
    disc["status"] = "discarded"
    disc["reason"] = [
        "; ".join(
            msg for ok, msg in (
                (r.CLmax_L >= req_l, f"CL_max,L {r.CLmax_L:.3f} < {req_l}"),
                (r.CLmax_TO >= req_to, f"CL_max,TO {r.CLmax_TO:.3f} < {req_to}"),
            ) if not ok
        ) + " with the TE device up to the aileron"
        for r in disc.itertuples()
    ]
    disc = disc.sort_values("CLmax_L", ascending=False)
    out = pd.concat([best, disc], ignore_index=True)
    out["rank"] = out["rank"].astype("Int64")  # <NA> for discarded
    return out


# ============================================================
#  WEIGHT SENSITIVITY
# ============================================================


def weight_grid(step: float = v2.SENSITIVITY_STEP) -> Iterator[tuple[float, float, float]]:
    """(w_performance, w_space, w_complexity) on the simplex, in steps of `step`."""
    n = round(1.0 / step)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            yield i / n, j / n, (n - i - j) / n


def sensitivity(scored: pd.DataFrame, step: float = v2.SENSITIVITY_STEP) -> pd.DataFrame:
    """Winning grid point for every weight set of weight_grid(step). The
    normalised criteria do not depend on the weights, so only the weighted
    sum is redone. Ties (e.g. complexity alone) go to the point with the
    higher score under the default weights."""
    feas = scored[scored["feasible"]].reset_index(drop=True)
    s = feas[list(CRITERIA)].to_numpy()
    rows = []
    for w in weight_grid(step):
        total = np.round(s @ np.array(w), 12)
        i = int(np.lexsort((feas["score"].to_numpy(), total))[-1])
        rows.append({
            "w_performance": w[0],
            "w_space": w[1],
            "w_complexity": w[2],
            "winner": feas.at[i, "config"],
            "eta_out_te": feas.at[i, "eta_out_te"],
            "score": float(total[i]),
        })
    return pd.DataFrame(rows)
