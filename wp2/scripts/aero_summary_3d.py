# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-08
"""WP2: write the 3D aerodynamic summary.

Drives b12wp2.wing.aero_summary on the results of vspaero_analysis,
hld_analysis and hld_analysis_v2 (run those first): wing geometry, flight
conditions, the cruise wing polar, every Oswald / span-efficiency figure
side by side, the clean wing at landing and the HLD selection.

No OpenVSP needed. From the wp2/ folder:

    python -m scripts.aero_summary_3d

Writes wp2/results/aero_summary_3d.pdf (the tables, A4) and
wp2/results/aero_summary_3d.csv (section, quantity, value, unit, source, note).
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

from b12wp2.config import paths
from b12wp2.plots import aero_summary as summary_pdf
from b12wp2.wing import aero_summary


def main() -> None:
    rows = aero_summary.build_rows()
    table = aero_summary.rows_frame(rows)
    table.to_csv(paths.AERO_SUMMARY_CSV, index=False)
    summary_pdf.write_pdf(rows, paths.AERO_SUMMARY_PDF)
    print(table[["section", "quantity", "value", "unit"]].to_string(index=False))
    print(f"\n  -> {paths.AERO_SUMMARY_PDF}")
    print(f"  -> {paths.AERO_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
