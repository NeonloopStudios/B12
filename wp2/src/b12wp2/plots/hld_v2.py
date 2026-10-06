# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""Plots of the HLD v2 trade-off (hld_analysis_v2.py)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only saves PNGs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from b12wp2.config import hld as hld_cfg, hld_v2 as v2  # noqa: E402
from b12wp2.plots.hld import C_CLEAN, C_REQ, INK  # noqa: E402
from b12wp2.plots.vspaero import DPI, _style_axes  # noqa: E402

# Okabe-Ito, fixed order
_OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
C_FAILS = "#c8c8c8"
LE_STYLES = {"none": ":", "LE flap": "--", "Krueger": "--", "slat": "-"}
CRITERION_COLORS = {"performance": "#0072B2", "space": "#E69F00", "complexity": "#009E73"}


def _te_colors(names: list[str]) -> dict[str, str]:
    """Okabe-Ito by TE_DEVICES order, over the TE devices that appear."""
    order = [d.name for d in hld_cfg.TE_DEVICES if d.name in set(names)]
    return dict(zip(order, _OKABE_ITO))


def plot_performance_vs_area(
    scored: pd.DataFrame, req_l: float, req_to: float, cl_max_clean: float, out_dir: Path
) -> None:
    """CL_max against the trailing-edge flapped area, one curve per
    configuration, up to the aileron. Configurations that never meet the
    requirements in grey; LE flap and Krueger share a curve (same ADSEE
    increment and chord)."""
    feasible_te = sorted(set(scored.loc[scored["feasible"], "te_device"]))
    colors = _te_colors(feasible_te)
    s_avail = float(scored["swf_s_te"].max() / scored["te_area_used"].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), sharey=True)
    for ax, col, req, title in (
        (axes[0], "CLmax_L", req_l, "Landing setting"),
        (axes[1], "CLmax_TO", req_to, "Take-off setting"),
    ):
        seen: set[tuple[str, str]] = set()
        for (te, le), d in scored.groupby(["te_device", "le_device"], sort=False):
            key = (te, "LE flap" if le == "Krueger" else le)
            if key in seen:
                continue
            seen.add(key)
            d = d.sort_values("eta_out_te")
            if te in colors:
                ax.plot(d["swf_s_te"], d[col], color=colors[te], ls=LE_STYLES[le], lw=1.8, zorder=3)
                eta_min = d["eta_out_te_min"].iloc[0]
                if math.isfinite(eta_min):
                    p = d[np.isclose(d["eta_out_te"], eta_min)].iloc[0]
                    ax.plot(p["swf_s_te"], p[col], "o", ms=7, mfc="white", mec=colors[te], mew=1.6, zorder=4)
            else:
                ax.plot(d["swf_s_te"], d[col], color=C_FAILS, ls=LE_STYLES[le], lw=1.0, zorder=2)
        ax.axhline(req, color=C_REQ, lw=1.2, ls="--", zorder=1)
        ax.annotate(f"required {req:.2f}", (0.0, req), xycoords=("axes fraction", "data"),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color=C_REQ)
        ax.axhline(cl_max_clean, color=C_CLEAN, lw=1, ls=":", zorder=1)
        # right end: every curve has climbed well clear of the clean-wing line there
        ax.annotate(f"clean wing {cl_max_clean:.2f}", (0.93, cl_max_clean), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", fontsize=8, color=C_CLEAN, ha="right")
        ax.axvline(s_avail, color=INK, lw=1.0, ls="-.", zorder=1)
        ax.annotate(f"aileron inboard edge\nη = {v2.ETA_AILERON_IN}", (s_avail, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(-4, -4), textcoords="offset points",
                    fontsize=8, color=INK, ha="right", va="top")
        ax.set(xlabel="trailing-edge flapped area $S_{wf,TE}/S$", title=title)
        ax.set_xlim(0, s_avail * 1.04)
        _style_axes(ax, horizontal_zero=False, vertical_zero=False)
    axes[0].set_ylabel("$C_{L,max}$")

    handles = [Line2D([], [], color=c, lw=1.8, label=f"TE: {n}") for n, c in colors.items()]
    handles.append(Line2D([], [], color=C_FAILS, lw=1.0, label="TE: never meets requirement"))
    handles += [Line2D([], [], color=INK, ls=LE_STYLES[n], lw=1.4, label=f"LE: {lab}")
                for n, lab in (("none", "none"), ("LE flap", "LE flap / Krueger"), ("slat", "slat"))]
    handles.append(Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=INK, label="smallest span meeting both"))
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_dir / "performance_vs_area.png", dpi=DPI)
    plt.close(fig)


def plot_tradeoff_scores(best: pd.DataFrame, out_dir: Path) -> None:
    """Weighted contribution of each criterion to every feasible
    configuration's best score."""
    feas = best[best["status"] == "feasible"].sort_values("score")
    n_disc = int((best["status"] == "discarded").sum())
    w = v2.WEIGHTS
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(feas) + 1.8))
    y = np.arange(len(feas))
    left = np.zeros(len(feas))
    for crit in ("performance", "space", "complexity"):
        part = w[crit] * feas[f"s_{crit}"].to_numpy()
        ax.barh(y, part, left=left, height=0.62, color=CRITERION_COLORS[crit], edgecolor="white",
                linewidth=2, label=f"{crit} (w = {w[crit]:.2f})", zorder=3)
        left += part
    for yi, (_, r) in zip(y, feas.iterrows()):
        tag = "  Pareto" if r["pareto"] else ""
        ax.annotate(f"{r['score']:.3f}  (η_out,TE = {r['eta_out_te']:.2f}){tag}", (r["score"], yi),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, feas["config"], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("weighted score (best TE span of each configuration)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    if n_disc:
        ax.set_title(f"{n_disc} configurations discarded (CL_max below requirement), see tradeoff.csv",
                     fontsize=9, color=C_CLEAN, loc="left")
    _style_axes(ax, horizontal_zero=False, vertical_zero=False)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_scores.png", dpi=DPI)
    plt.close(fig)


def plot_tradeoff_scatter(scored: pd.DataFrame, best: pd.DataFrame, out_dir: Path) -> None:
    """Normalised performance vs occupied-space score along each
    configuration's TE span grid, coloured by complexity; the best point of
    each configuration is marked. LE flap and Krueger give the same line
    (they differ only in complexity) and share a label."""
    feas = scored[scored["feasible"]]
    levels = sorted(feas["complexity"].unique())
    cmap = plt.get_cmap("Blues")
    shade = {c: cmap(0.4 + 0.55 * i / max(len(levels) - 1, 1)) for i, c in enumerate(levels)}

    fig, ax = plt.subplots(figsize=(9, 6))
    labels: dict[tuple[float, float], list[str]] = {}
    # higher complexity first so the simpler of two coincident lines is on top
    for cfg_name, d in sorted(feas.groupby("config"), key=lambda kv: -kv[1]["complexity"].iloc[0]):
        d = d.sort_values("eta_out_te")
        c = shade[d["complexity"].iloc[0]]
        ax.plot(d["s_space"], d["s_performance"], color=c, lw=2.0, zorder=3)
        b = best[best["config"] == cfg_name].iloc[0]
        ax.plot(b["s_space"], b["s_performance"], "o", ms=8, color=c, mec="white", mew=2, zorder=4)
        labels.setdefault((round(b["s_space"], 6), round(b["s_performance"], 6)), []).append(cfg_name)
    for (x, yv), names in labels.items():
        ax.annotate(" / ".join(sorted(names)), (x, yv), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    handles = [Line2D([], [], color=shade[c], lw=2.0, label=f"complexity {c}") for c in levels]
    handles.append(Line2D([], [], ls="", marker="o", ms=8, color=INK, mec="white", label="best TE span"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    ax.set(xlabel="occupied-space score (1 = least space)", ylabel="performance score (1 = highest CL_max)",
           xlim=(-0.05, 1.15), ylim=(-0.05, 1.08))
    ax.annotate("each line: TE outboard end from the smallest feasible span to the aileron",
                (0.0, 1.0), xycoords="axes fraction", xytext=(4, -4), textcoords="offset points",
                fontsize=8, color=C_CLEAN, va="top")
    _style_axes(ax, horizontal_zero=False, vertical_zero=False)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_scatter.png", dpi=DPI)
    plt.close(fig)


def plot_weight_sensitivity(sens: pd.DataFrame, default_winner: str, out_dir: Path) -> None:
    """Winner over the weight simplex (barycentric triangle)."""
    h = math.sqrt(3) / 2
    # corners: performance bottom-left, space bottom-right, complexity top
    x = sens["w_space"] + 0.5 * sens["w_complexity"]
    y = h * sens["w_complexity"]
    winners = [default_winner] + [n for n in sens["winner"].value_counts().index if n != default_winner]
    colors = dict(zip(winners, _OKABE_ITO))

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color=INK, lw=1.0)
    for name in winners:
        m = sens["winner"] == name
        ax.scatter(x[m], y[m], s=90, color=colors[name], edgecolor="white", linewidth=1.5, zorder=3,
                   label=f"{name} ({int(m.sum())})")
    w = v2.WEIGHTS
    ax.plot(w["space"] + 0.5 * w["complexity"], h * w["complexity"], marker="*", ms=18, color="none",
            mec=INK, mew=1.5, zorder=4, ls="", label="selected weights")
    for (cx, cy, text, ha, va) in ((0, 0, "performance", "right", "top"), (1, 0, "space", "left", "top"),
                                   (0.5, h, "complexity", "center", "bottom")):
        ax.annotate(f"{text}\n(w = 1)" if va == "top" else f"(w = 1)\n{text}", (cx, cy),
                    xytext=(0, -6 if va == "top" else 6), textcoords="offset points",
                    ha=ha, va=va, fontsize=9, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, h + 0.1)
    ax.legend(fontsize=8, frameon=False, loc="upper left", title="winner (weight sets won)",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "weight_sensitivity.png", dpi=DPI)
    plt.close(fig)


def make_all(
    scored: pd.DataFrame, best: pd.DataFrame, sens: pd.DataFrame,
    req_l: float, req_to: float, cl_max_clean: float, out_dir: Path,
) -> None:
    plot_performance_vs_area(scored, req_l, req_to, cl_max_clean, out_dir)
    plot_tradeoff_scores(best, out_dir)
    plot_tradeoff_scatter(scored, best, out_dir)
    plot_weight_sensitivity(sens, best.iloc[0]["config"], out_dir)
