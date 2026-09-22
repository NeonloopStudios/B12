# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-22
"""WP2 analysis library: everything the wp2/scripts/ entry points call.

Subpackages, by domain:

    config/   all constants and flight conditions (no computation)
    common/   airfoil geometry and compressibility relations
    xfoil/    2D section work: XFoil runtime, polar post-processing
    wing/     3D wing work: planform, VSPAERO, viscous/wave corrections
    hld/      high-lift device sizing and configuration comparison
    plots/    matplotlib figures, one module per analysis stage
    scoring/  the weighted airfoil scorecard

Nothing in here prints a report or writes a results file on import; that is
the entry points' job (wp2/scripts/), so every function here stays callable
from a test or another module.
"""
