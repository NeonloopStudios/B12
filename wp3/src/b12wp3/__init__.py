# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""WP3 analysis library: everything the wp3/scripts/ entry points call.

Subpackages, by domain:

    config/   all inputs: paths, atmosphere, mission, geometry, drag settings
    common/   trapezoidal planform geometry, shared by every lifting surface
    drag/     class II drag: wetted areas, skin friction, form factors and
              the component drag build-up (ADSEE-II)
    plots/    matplotlib figures, one module per analysis stage

Nothing in here prints a report or writes a results file on import; that is
the entry points' job (wp3/scripts/), so every function stays callable from a
test or another module.
"""
