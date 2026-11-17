# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4 inputs, as plain Python modules (same layout as wp2/src/b12wp2/config).

General files, used by every WP4 sub-package:

    paths      repository, WP4 and output directories
    aircraft   aircraft geometry (wing planform, twist, dihedral, airfoil)

Settings of one sub-package are in its own folder:

    wp41/      WP4.1 aerodynamic loads: flight condition, VSPAERO settings

The modules import only the standard library, so the repo venv and the
OpenVSP env (Python 3.11) can both import them.
"""
