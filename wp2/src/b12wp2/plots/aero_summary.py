# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-08
"""The 3D aerodynamic summary (b12wp2.wing.aero_summary) as an A4 PDF.

Drawn with matplotlib's PDF backend rather than a document library, so it
needs nothing the plots do not already need. Text is placed on a top-down
flow of A4 pages: headings, tables (wrapped cells, header repeated after a
page break) and paragraphs. Quantity labels and table headers may use
mathtext; notes and paragraphs are plain text and wrap.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only saves files

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from b12wp2.config import hld_v2 as v2_cfg, paths  # noqa: E402
from b12wp2.wing import aero_summary as summ  # noqa: E402

PAGE_W, PAGE_H = 8.27, 11.69  # A4 [in]
MARGIN = 0.7  # [in]
TEXT_W = PAGE_W - 2 * MARGIN

INK = "#222222"
MUTED = "#666666"
RULE = "#BBBBBB"
HEADER_FILL = "#E8EEF4"
ACCENT = "#0072B2"

CHAR_W = 0.56  # mean glyph width as a fraction of the font size, for wrapping
LINE_H = 1.35  # line height as a multiple of the font size


def fmt(value: float | str | None) -> str:
    """Compact number formatting for the tables."""
    if isinstance(value, str):
        return value
    if value is None or math.isnan(value):
        return "n/a"
    if float(value).is_integer() and abs(value) < 1e5:
        return f"{value:g}"
    if abs(value) >= 1e5:
        mantissa, exponent = f"{value:.3e}".split("e")
        return rf"{mantissa}$\times 10^{{{int(exponent)}}}$"
    if abs(value) < 1:
        return f"{value:.4g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _wrap(text: str, width_in: float, size: float) -> list[str]:
    if not text:
        return [""]
    if "$" in text:  # mathtext: never split inside it
        return [text]
    chars = max(8, int(width_in * 72 / (CHAR_W * size)))
    return textwrap.wrap(text, chars) or [""]


class _Flow:
    """Top-down placement of blocks on A4 pages, in inches from the top."""

    def __init__(self, pdf: PdfPages, title: str) -> None:
        self.pdf = pdf
        self.title = title
        self.page_no = 0
        self.fig: plt.Figure | None = None
        self.y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        if self.fig is not None:
            self._finish_page()
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.page_no += 1
        self.y = MARGIN

    def _finish_page(self) -> None:
        assert self.fig is not None
        self._text(MARGIN, PAGE_H - 0.4, self.title, 7, color=MUTED)
        self._text(PAGE_W - MARGIN, PAGE_H - 0.4, f"{self.page_no}", 7, color=MUTED, ha="right")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def close(self) -> None:
        self._finish_page()
        self.fig = None

    def ensure(self, height: float) -> bool:
        """Start a new page unless `height` still fits; True if it broke."""
        if self.y + height > PAGE_H - MARGIN:
            self._new_page()
            return True
        return False

    # drawing helpers, coordinates in inches from the top-left corner
    def _text(self, x: float, y: float, s: str, size: float, *, color: str = INK, weight: str = "normal",
              ha: str = "left") -> None:
        assert self.fig is not None
        self.fig.text(x, PAGE_H - y, s, fontsize=size, color=color, weight=weight, ha=ha, va="top",
                      transform=self.fig.dpi_scale_trans)

    def _hline(self, y: float, x0: float, x1: float, color: str = RULE, lw: float = 0.6) -> None:
        assert self.fig is not None
        self.fig.add_artist(Line2D([x0, x1], [PAGE_H - y] * 2, color=color, lw=lw,
                                   transform=self.fig.dpi_scale_trans))

    def _box(self, x: float, y: float, w: float, h: float, color: str) -> None:
        assert self.fig is not None
        self.fig.add_artist(Rectangle((x, PAGE_H - y - h), w, h, facecolor=color, edgecolor="none",
                                      transform=self.fig.dpi_scale_trans))

    # blocks
    def title_block(self, title: str, subtitle: str) -> None:
        self._text(MARGIN, self.y, title, 18, weight="bold")
        self.y += 0.42
        for line in _wrap(subtitle, TEXT_W, 9):
            self._text(MARGIN, self.y, line, 9, color=MUTED)
            self.y += 9 * LINE_H / 72
        self.y += 0.15

    def heading(self, text: str) -> None:
        self.ensure(0.9)  # keep a heading with the start of its table
        self.y += 0.12
        self._text(MARGIN, self.y, text, 12, weight="bold", color=ACCENT)
        self.y += 0.24
        self._hline(self.y, MARGIN, PAGE_W - MARGIN, color=ACCENT, lw=0.8)
        self.y += 0.08

    def paragraph(self, text: str, size: float = 8.0, *, bullet: bool = False) -> None:
        indent = 0.18 if bullet else 0.0
        lines = _wrap(text, TEXT_W - indent, size)
        line_h = size * LINE_H / 72
        for i, line in enumerate(lines):
            self.ensure(line_h)
            if bullet and i == 0:
                self._text(MARGIN + 0.04, self.y, "•", size)
            self._text(MARGIN + indent, self.y, line, size)
            self.y += line_h
        self.y += 0.05

    def table(self, header: list[str], body: list[list[str]], widths: list[float], aligns: str,
              size: float = 7.5) -> None:
        """`widths` as fractions of the text width, `aligns` one of l/r/c per column."""
        col_w = [w * TEXT_W for w in widths]
        col_x = [MARGIN + sum(col_w[:i]) for i in range(len(col_w))]
        line_h = size * LINE_H / 72
        pad = 0.035

        def draw_row(cells: list[str], *, is_header: bool) -> None:
            wrapped = [_wrap(c, w - 2 * pad, size) for c, w in zip(cells, col_w)]
            height = max(len(w) for w in wrapped) * line_h + 2 * pad
            if self.ensure(height + (0 if is_header else line_h)) and not is_header:
                draw_row(header, is_header=True)
            if is_header:
                self._box(MARGIN, self.y, TEXT_W, height, HEADER_FILL)
            for x, w, lines, align in zip(col_x, col_w, wrapped, aligns):
                xt = {"l": x + pad, "r": x + w - pad, "c": x + w / 2}[align]
                ha = {"l": "left", "r": "right", "c": "center"}[align]
                for k, line in enumerate(lines):
                    self._text(xt, self.y + pad + k * line_h, line, size, weight="bold" if is_header else "normal",
                               ha=ha)
            self.y += height
            self._hline(self.y, MARGIN, PAGE_W - MARGIN)

        draw_row(header, is_header=True)
        for cells in body:
            draw_row(cells, is_header=False)
        self.y += 0.12


def write_pdf(rows: list[summ.Row], path: Path) -> None:
    title = f"3D aerodynamic summary - {summ.AIRFOIL_STEM} wing"
    with PdfPages(path) as pdf:
        flow = _Flow(pdf, title)
        flow.title_block(
            "3D aerodynamic summary",
            f"B12 wing with {summ.AIRFOIL_STEM} sections. Generated by `python -m scripts.aero_summary_3d` "
            "from the results in wp2/results/; rerun it after the 3D or HLD analyses change. "
            "The same numbers, with their source files, are in aero_summary_3d.csv.",
        )
        for section in dict.fromkeys(r.section for r in rows):
            flow.heading(section)
            body = [[r.label, fmt(r.value), r.unit.replace("m^2", "m²"), r.note]
                    for r in rows if r.section == section]
            flow.table(["quantity", "value", "unit", "note"], body, [0.33, 0.15, 0.08, 0.44], "lrll")
            if section == summ.OSWALD_SECTION:
                flow.paragraph(summ.OSWALD_NOTES[0])
                for note in summ.OSWALD_NOTES[1:]:
                    flow.paragraph(note, bullet=True)

        flow.heading("HLD v1: Pareto-optimal configurations")
        v1 = summ.hld_v1_table()
        flow.table(
            ["configuration", "complexity", r"$\Delta C_{L,max}$", r"$C_{L,max}$ L", r"$C_{L,max}$ TO",
             r"$\alpha_{stall}$ L [deg]"],
            [[r.config, f"{r.complexity:d}", fmt(r.dCLmax), fmt(r.CLmax_L), fmt(r.CLmax_TO), fmt(r.alpha_stall_L)]
             for r in v1.itertuples()],
            [0.34, 0.12, 0.13, 0.13, 0.13, 0.15], "lrrrrr",
        )
        flow.paragraph(f"L = landing, TO = take-off. Full table: {summ.rel(paths.HLD_DIR / 'comparison.csv')}.", 7.5)

        weights = ", ".join(f"{k} {v:.2f}" for k, v in v2_cfg.WEIGHTS.items())
        flow.heading("HLD v2: feasible configurations")
        v2 = summ.hld_v2_table()
        flow.table(
            ["#", "configuration", "cmplx", r"$\eta_{TE}$", r"$C_{L,max}$ L", r"$C_{L,max}$ TO",
             r"$\alpha_{stall}$ L", "s perf", "s space", "s cmplx", "score", "Pareto"],
            [[f"{r.rank:g}", r.config, f"{r.complexity:d}", fmt(r.eta_out_te), fmt(r.CLmax_L), fmt(r.CLmax_TO),
              fmt(r.alpha_stall_L), fmt(r.s_performance), fmt(r.s_space), fmt(r.s_complexity), fmt(r.score),
              "*" if r.pareto else ""]
             for r in v2.itertuples()],
            [0.04, 0.24, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.06], "rlrrrrrrrrrc",
            size=7.0,
        )
        wins = summ.hld_v2_sensitivity()
        flow.paragraph(
            f"L = landing, TO = take-off. Weights: {weights}. Weight sensitivity ({wins.sum()} weight sets): "
            + ", ".join(f"{cfg} wins {n}" for cfg, n in wins.items()) + ". "
            f"Full tables: {summ.rel(paths.HLD_V2_DIR / 'tradeoff.csv')}, "
            f"{summ.rel(paths.HLD_V2_DIR / 'sensitivity.csv')}.",
            7.5,
        )
        flow.close()
        meta = pdf.infodict()
        meta["Title"] = title
