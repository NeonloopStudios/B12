# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Plots of the high-lift device comparison (hld_analysis.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only saves PNGs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from b12wp2.hld import sizing as hs  # noqa: E402
from b12wp2.plots.vspaero import DPI, _style_axes  # noqa: E402

# Okabe-Ito; LE device colors are fixed by LE_DEVICES order
_OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
LE_COLORS = {d.name: c for d, c in zip(hs.LE_DEVICES, ["#999999", "#E69F00", "#009E73", "#0072B2"])}
C_CLEAN = "#555555"
C_REQ = "#D55E00"
INK = "#222222"


@dataclass(frozen=True)
class Layout:
    eta_in: float
    eta_out_te: float
    eta_out_le: float
    flap_chord_ratio: float
    slat_chord_ratio: float


def _req_line(ax: plt.Axes, value: float | None, label: str, *, horizontal: bool = True) -> None:
    if value is None:
        return
    (ax.axhline if horizontal else ax.axvline)(value, color=C_REQ, lw=1.2, ls="--", label=label)


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Label positions close to `values` (any order) but at least min_gap apart."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    prev = -float("inf")
    for i in order:
        out[i] = max(values[i], prev + min_gap)
        prev = out[i]
    return out


def plot_heatmap(df: pd.DataFrame, cl_max_clean: float, req: float | None, out_dir: Path) -> None:
    te_names = [d.name for d in hs.TE_DEVICES]
    le_names = [d.name for d in hs.LE_DEVICES]
    grid = (
        df.pivot(index="te_device", columns="le_device", values="CLmax_L")
        .loc[te_names, le_names]
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = plt.get_cmap("Blues")
    vmin, vmax = cl_max_clean, float(grid.max())
    lo = vmin - 0.35 * (vmax - vmin)  # keep the lowest cell off the near-white end
    im = ax.imshow(grid, cmap=cmap, vmin=lo, vmax=vmax, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = float(grid[i, j])
            dark = (v - lo) / (vmax - lo) > 0.6
            text = f"{v:.2f}"
            if req is not None and v < req:
                text += "\n(fails)"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(le_names)), [f"LE: {n}" for n in le_names], fontsize=9)
    ax.set_yticks(range(len(te_names)), te_names, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(le_names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(te_names)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Landing $C_{{L,max}}$ per configuration (clean wing {cl_max_clean:.2f})")
    ax.set_ylabel("trailing-edge device")
    fig.colorbar(im, ax=ax, label="$C_{L,max}$ landing", fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_heatmap.png", dpi=DPI)
    plt.close(fig)


def plot_cl_max_vs_complexity(
    df: pd.DataFrame, cl_max_clean: float, req: float | None, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    front = df[df["pareto"]].sort_values("complexity")
    ax.step(front["complexity"], front["CLmax_L"], where="post", color="#bbbbbb", lw=1.5,
            zorder=1, label="Pareto front")
    for le in hs.LE_DEVICES:
        d = df[df["le_device"] == le.name]
        # small horizontal offset per LE device so equal-complexity points don't stack
        dx = (list(LE_COLORS).index(le.name) - 1.5) * 0.08
        ax.scatter(d["complexity"] + dx, d["CLmax_L"], s=55, color=LE_COLORS[le.name],
                   edgecolor="white", linewidth=1.5, zorder=3, label=f"LE: {le.name}")
    for (_, cl), grp in front.groupby(["complexity", "CLmax_L"], sort=False):
        r = grp.iloc[0]
        dx = (list(LE_COLORS).index(r["le_device"]) - 1.5) * 0.08
        ax.annotate(" / ".join(grp["config"]), (r["complexity"] + dx, cl), xytext=(-8, 6),
                    textcoords="offset points", fontsize=8, color=INK, ha="right")
    ax.axhline(cl_max_clean, color=C_CLEAN, lw=1, ls=":")
    ax.annotate(f"clean wing {cl_max_clean:.2f}", (1.0, cl_max_clean), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", fontsize=8, color=C_CLEAN, ha="right")
    _req_line(ax, req, "required $C_{L,max}$ landing")
    ax.set(xlabel="mechanism complexity rank (TE + LE, qualitative)", ylabel="$C_{L,max}$ landing",
           title="Maximum lift vs. complexity")
    ax.set_xticks(range(int(df["complexity"].min()), int(df["complexity"].max()) + 1))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style_axes(ax, horizontal_zero=False, vertical_zero=False)
    fig.tight_layout()
    fig.savefig(out_dir / "cl_max_vs_complexity.png", dpi=DPI)
    plt.close(fig)


def plot_lift_curves(
    df: pd.DataFrame, clean: hs.CleanWing, planform: hs.Planform, layout: Layout,
    req_l: float | None, req_to: float | None, out_dir: Path,
) -> None:
    """Clean wing and the Pareto-front configurations, landing and take-off.

    ADSEE shifts alpha_0L by the same amount for every trailing-edge device
    and changes the slope only through S'/S, so several curves coincide and
    differ only in where they stop (CL_max). Curves are drawn longest first
    and every stall point is labelled directly, so none is hidden.
    """
    front = (
        df[df["pareto"] & (df["complexity"] > 0)]
        .drop_duplicates(subset=["CLmax_L", "complexity"])  # e.g. plain == split
        .sort_values("complexity", ascending=False)
        .head(len(_OKABE_ITO))
    )
    alphas = list(np.linspace(-20, 25, 451))
    clean_cfg = hs.Configuration(clean.cl_max, clean.cl_alpha_per_deg, clean.alpha_0l_deg,
                                 clean.alpha_stall_deg)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for k, (takeoff, title, req) in enumerate(
        [(False, "Landing setting", req_l), (True, "Take-off setting", req_to)]
    ):
        a = ax[k]
        a.plot(alphas, hs.lift_curve(clean_cfg, alphas), color=C_CLEAN, lw=1.5, ls="--")
        a.annotate(f"clean {clean.cl_max:.2f}", (clean.alpha_stall_deg, clean.cl_max),
                   xytext=(6, -4), textcoords="offset points", fontsize=8, color=C_CLEAN, va="top")
        points = []
        for color, (_, r) in zip(_OKABE_ITO, front.iterrows()):
            te = next(d for d in hs.TE_DEVICES if d.name == r["te_device"])
            le = next(d for d in hs.LE_DEVICES if d.name == r["le_device"])
            te_eff = hs.device_effect(planform, te, layout.flap_chord_ratio, layout.eta_in, layout.eta_out_te)
            le_eff = (hs.device_effect(planform, le, layout.slat_chord_ratio, layout.eta_in, layout.eta_out_le)
                      if le.dcl_max > 0 else hs.DeviceEffect(0.0, 0.0, 0.0, 1.0, 0.0))
            cfg = hs.configuration(clean, te_eff, le_eff, takeoff=takeoff)
            a.plot(alphas, hs.lift_curve(cfg, alphas), color=color, lw=1.8)
            a.plot(cfg.alpha_stall_deg, cfg.cl_max, "o", color=color, ms=7, mec="white", mew=1.5, zorder=4)
            points.append((cfg, f"{r['config']} {cfg.cl_max:.2f}"))
        # labels in a column right of the curves, spread vertically, with leader lines
        x_label = max(c.alpha_stall_deg for c, _ in points) + 3.0
        y_label = _spread([c.cl_max for c, _ in points], 0.15)
        for (cfg, text), y in zip(points, y_label):
            a.annotate(text, (cfg.alpha_stall_deg, cfg.cl_max), xytext=(x_label, y), textcoords="data",
                       fontsize=8, color=INK, va="center",
                       arrowprops={"arrowstyle": "-", "color": "#aaaaaa", "lw": 0.7,
                                   "shrinkA": 0, "shrinkB": 4})
        _req_line(a, req, "required $C_{L,max}$")
        a.set_xlim(-20, 34)
        a.set(xlabel="α [deg] (aircraft reference, wing incidence included)", title=title)
        _style_axes(a)
    ax[0].set_ylabel("$C_L$")
    fig.suptitle("Lift curves of the Pareto-front configurations (VLM slope, ADSEE increments; "
                 "equal-slope TE devices share one line up to their own stall point)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "lift_curves.png", dpi=DPI)
    plt.close(fig)


def plot_planform(planform: hs.Planform, layout: Layout, out_dir: Path) -> None:
    """Right half-wing seen from above, x (chordwise) pointing down."""
    bh = planform.b_half

    def x_at(eta: float, x_c: float) -> float:
        return eta * bh * planform.tan_sweep_le + x_c * planform.chord(eta)

    def band(eta0: float, eta1: float, xc0: float, xc1: float) -> list[tuple[float, float]]:
        return [(eta0 * bh, x_at(eta0, xc0)), (eta1 * bh, x_at(eta1, xc0)),
                (eta1 * bh, x_at(eta1, xc1)), (eta0 * bh, x_at(eta0, xc1))]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.add_patch(Polygon(band(0, 1, 0, 1), closed=True, fc="#f2f2f2", ec=INK, lw=1.2))
    xf = 1 - layout.flap_chord_ratio
    patches = [
        (band(layout.eta_in, layout.eta_out_te, xf, 1), "#0072B2", None, "TE device"),
        (band(layout.eta_in, layout.eta_out_le, 0, layout.slat_chord_ratio), "#009E73", None, "LE device"),
        (band(layout.eta_out_te, 1.0, xf, 1), "white", "////", "reserved for aileron"),
    ]
    for pts, fc, hatch, label in patches:
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=INK if hatch else "white", lw=1,
                             hatch=hatch, alpha=0.85, label=label))
    for xc, name in ((layout.slat_chord_ratio, "front spar"), (xf, "rear spar")):
        ax.plot([0, bh], [x_at(0, xc), x_at(1, xc)], color="#888888", lw=0.8, ls="--")
        ax.annotate(name, (bh, x_at(1, xc)), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#888888", va="center")
    ax.axvline(layout.eta_in * bh, color=INK, lw=0.8, ls=":")
    ax.annotate(f"fuselage side\nη = {layout.eta_in}", (layout.eta_in * bh, x_at(layout.eta_in, 1)),
                xytext=(-4, -12), textcoords="offset points", fontsize=8, ha="right", va="top", color=INK)
    ax.set_xlim(-0.5, bh * 1.12)
    ax.set_ylim(x_at(1, 1) + 0.6, -0.8)
    ax.set_aspect("equal")
    ax.set(xlabel="y [m]", ylabel="x [m]", title="High-lift device layout (right half-wing)")
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "planform_hld.png", dpi=DPI)
    plt.close(fig)


def make_all(
    df: pd.DataFrame, clean: hs.CleanWing, planform: hs.Planform, layout: Layout,
    req_l: float | None, req_to: float | None, out_dir: Path,
) -> None:
    plot_heatmap(df, clean.cl_max, req_l, out_dir)
    plot_cl_max_vs_complexity(df, clean.cl_max, req_l, out_dir)
    plot_lift_curves(df, clean, planform, layout, req_l, req_to, out_dir)
    plot_planform(planform, layout, out_dir)
