# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Plots of the VSPAERO + XFoil wing analysis (vspaero_analysis.py).

Same visual language as make_plots.py (Okabe-Ito colors, recessive grid,
emphasized zero lines). _style_axes is repeated here rather than imported:
make_plots imports mcrit_sweep -> xfoil_runtime, which loads the XFoil DLL
that the OpenVSP environment does not have.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only saves PNGs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from scripts import viscous_correction as vc  # noqa: E402

# Okabe-Ito, fixed role per drag component across every figure.
C_TOTAL = "#0072B2"
C_PROFILE = "#E69F00"
C_INDUCED = "#009E73"
C_WAVE = "#D55E00"
C_REFERENCE = "#999999"  # VSPAERO-only drag, for comparison
C_GUIDE = "#555555"

DPI = 150


def _style_axes(ax: plt.Axes, *, horizontal_zero: bool = True, vertical_zero: bool = True) -> None:
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if horizontal_zero:
        ax.axhline(0, color="#888888", linewidth=1.1, zorder=1)
    if vertical_zero:
        ax.axvline(0, color="#888888", linewidth=1.1, zorder=1)


def _design_line(ax: plt.Axes, cl_design: float, *, horizontal: bool = True) -> None:
    line = ax.axhline if horizontal else ax.axvline
    line(cl_design, color=C_GUIDE, linewidth=0.9, linestyle=":", zorder=1)


def plot_polar(polar: pd.DataFrame, summary: dict[str, float], out_dir: Path) -> None:
    cl_d = summary["CL_design"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    a.plot(polar["alpha"], polar["CL"], "o-", color=C_TOTAL, lw=2, ms=5)
    _design_line(a, cl_d)
    a.annotate(
        f"design $C_L$ = {cl_d:.3f}\n$\\alpha$ = {summary['alpha_design']:.2f}°",
        (summary["alpha_design"], cl_d), xytext=(12, -34), textcoords="offset points",
        fontsize=8, color=C_GUIDE,
    )
    a.set(xlabel="α [deg]", ylabel="$C_L$", title="Lift curve (VLM, inviscid)")
    _style_axes(a)

    a = ax[0, 1]
    a.plot(polar["CDtot_vsp"], polar["CL"], "--", color=C_REFERENCE, lw=1.5, label="VSPAERO only")
    a.plot(polar["CDi"], polar["CL"], "s-", color=C_INDUCED, lw=2, ms=4, label="$C_{D_i}$")
    a.plot(polar["CD"] - polar["CD_wave"], polar["CL"], "^-", color=C_PROFILE, lw=2, ms=5,
           label="$C_{D_i}$ + $C_{D_p}$")
    a.plot(polar["CD"], polar["CL"], "o-", color=C_TOTAL, lw=2, ms=5,
           label="$C_{D_i}$ + $C_{D_p}$ + $C_{D_w}$")
    _design_line(a, cl_d)
    a.set(xlabel="$C_D$", ylabel="$C_L$", title="Drag polar")
    a.legend(fontsize=8, frameon=False, loc="lower right")
    _style_axes(a, vertical_zero=False)

    a = ax[1, 0]
    pos = polar[polar["CL"] > 0]
    a.plot(pos["alpha"], pos["CL"] / pos["CDtot_vsp"], "--", color=C_REFERENCE, lw=1.5, label="VSPAERO only")
    a.plot(pos["alpha"], pos["L_D_no_wave"], "^-", color=C_PROFILE, lw=2, ms=5, label="without wave drag")
    a.plot(pos["alpha"], pos["L_D"], "o-", color=C_TOTAL, lw=2, ms=5, label="with wave drag")
    a.set(xlabel="α [deg]", ylabel="L/D", title="Lift-to-drag ratio")
    a.legend(fontsize=8, frameon=False)
    _style_axes(a, horizontal_zero=False)

    a = ax[1, 1]
    a.plot(polar["CL"], polar["CMy"], "o-", color=C_TOTAL, lw=2, ms=5)
    _design_line(a, cl_d, horizontal=False)
    a.set(xlabel="$C_L$", ylabel="$C_m$ (1/4 MAC)", title="Pitching moment")
    _style_axes(a)

    fig.suptitle(
        f"Wing with NACA 25112, M = {summary['mach']:.3f}: VSPAERO + XFoil profile drag + Korn/Lock wave drag",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "polar.png", dpi=DPI)
    plt.close(fig)


def plot_drag_breakdown(polar: pd.DataFrame, summary: dict[str, float], out_dir: Path) -> None:
    p = polar.dropna(subset=["CD"]).sort_values("CL")
    p = p[p["CL"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.stackplot(
        p["CL"], p["CDi"], p["CD_profile"], p["CD_wave"],
        colors=[C_INDUCED, C_PROFILE, C_WAVE], edgecolor="white", linewidth=1.5,
        labels=["induced (VSPAERO)", "profile (XFoil strips)", "wave (Korn/Lock)"],
    )
    cl_d = summary["CL_design"]
    _design_line(ax, cl_d, horizontal=False)
    parts = (summary["CDi_design"], summary["CD_profile_design"], summary["CD_wave_design"])
    ax.annotate(
        f"design $C_L$ = {cl_d:.3f}\n"
        f"$C_{{D_i}}$ = {parts[0]:.4f}\n$C_{{D_p}}$ = {parts[1]:.4f}\n$C_{{D_w}}$ = {parts[2]:.4f}\n"
        f"$C_D$ = {sum(parts):.4f}, L/D = {summary['L_D_design']:.1f}",
        (cl_d, summary["CD_design"]), xytext=(-150, 20), textcoords="offset points",
        fontsize=8, color="#222222", arrowprops={"arrowstyle": "-", "color": C_GUIDE, "lw": 0.8},
    )
    ax.set(xlabel="$C_L$", ylabel="$C_D$", title="Drag breakdown")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style_axes(ax, vertical_zero=False)
    fig.tight_layout()
    fig.savefig(out_dir / "drag_breakdown.png", dpi=DPI)
    plt.close(fig)


def plot_span_loads(strips: pd.DataFrame, section: pd.DataFrame, out_dir: Path) -> None:
    alphas = sorted(strips["alpha"].unique())
    norm = Normalize(min(alphas), max(alphas))
    cmap = plt.get_cmap("Blues")

    def color(alpha: float) -> tuple[float, float, float, float]:
        return cmap(0.3 + 0.7 * norm(alpha))  # skip the near-white end of the ramp

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for alpha in alphas:
        s = strips[strips["alpha"] == alpha]
        ax[0].plot(s["y"], s["cl_n"], "-", color=color(alpha), lw=1.6)
        stalled = s[s["status"] == vc.STALLED]
        ax[0].plot(stalled["y"], stalled["cl_n"], "x", color=C_WAVE, ms=6, mew=1.5)
        # cd only where the whole span is inside the polar: a partial curve
        # with NaN gaps would read as a real drag distribution
        if s["cd"].notna().all():
            ax[1].plot(s["y"], s["cd"] * 1e4, "-", color=color(alpha), lw=1.6)

    cl_max_n = float(section["cl"].iloc[-1])
    ax[0].axhline(cl_max_n, color=C_WAVE, lw=1.2, ls="--")
    handles = [
        plt.Line2D([], [], color=C_WAVE, lw=1.2, ls="--",
                   label=f"XFoil cruise $c_{{l,max,n}}$ = {cl_max_n:.3f}"),
        plt.Line2D([], [], color=C_WAVE, marker="x", ls="none", ms=6, mew=1.5,
                   label="strip past XFoil stall ($c_d$ undefined)"),
    ]
    ax[0].set_ylim(bottom=min(-0.75, float(strips["cl_n"].min()) - 0.1))
    ax[0].legend(handles=handles, fontsize=8, frameon=False, loc="lower center")
    ax[0].set(xlabel="y [m]", ylabel="$c_{l,n} = c_l / \\cos^2\\Lambda$",
              title="Sweep-normal section lift")
    ax[1].set(xlabel="y [m]", ylabel="$c_d$ [counts]",
              title="Section profile drag (streamwise, unstalled α only)")
    _style_axes(ax[0], vertical_zero=False)
    _style_axes(ax[1], horizontal_zero=False, vertical_zero=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="α [deg]", fraction=0.03, pad=0.02)
    fig.savefig(out_dir / "span_loads.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_transonic(summary: dict[str, float], out_dir: Path) -> None:
    cls = np.linspace(0.0, 1.0, 41)
    sweep = math.radians(summary["sweep_korn_deg"])
    korn = np.array([vc.korn_swept(summary["kappa_a"], summary["t_c"], c, sweep) for c in cls])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(korn[:, 0], cls, "-", color=C_TOTAL, lw=2, label="$M_{dd}$")
    ax.plot(korn[:, 1], cls, "--", color=C_WAVE, lw=2, label="$M_{crit}$")
    ax.axvline(summary["mach"], color="#222222", lw=1.1)
    ax.annotate(f"flight M = {summary['mach']:.3f}", (summary["mach"], 0.97),
                xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points",
                fontsize=8, va="top")
    _design_line(ax, summary["CL_design"])
    ax.set(xlabel="M", ylabel="$C_L$",
           title=f"Korn equation, κ = {summary['kappa_a']}, t/c = {summary['t_c']:.3f}")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style_axes(ax, vertical_zero=False)
    fig.tight_layout()
    fig.savefig(out_dir / "transonic.png", dpi=DPI)
    plt.close(fig)


def make_all(
    polar: pd.DataFrame, strips: pd.DataFrame, section: pd.DataFrame,
    summary: dict[str, float], out_dir: Path,
) -> None:
    plot_polar(polar, summary, out_dir)
    plot_drag_breakdown(polar, summary, out_dir)
    plot_span_loads(strips, section, out_dir)
    plot_transonic(summary, out_dir)
