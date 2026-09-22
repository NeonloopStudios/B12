# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2: build the weighted scorecard and print it.

Writes results/scorecard/scorecard.csv (the screenshot-shaped deliverable)
and results/scorecard/scorecard_detail.csv (raw, normalized and weighted values per
criterion plus validity notes), then prints the summary table and any
validity note that came with it -- the transonic ones matter, they say when
a cruise Cd is not trustworthy at the cruise Mach.

The tables themselves are built by b12wp2.scoring.scorecard.
"""
from __future__ import annotations

from b12wp2 import config
from b12wp2.config import paths
from b12wp2.scoring.scorecard import build_detail_table, build_summary_table


def main() -> None:
    detail = build_detail_table()
    summary = build_summary_table(detail)

    detail_path = paths.SCORECARD_DETAIL_CSV
    summary_path = paths.SCORECARD_CSV
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
