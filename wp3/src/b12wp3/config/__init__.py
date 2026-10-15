# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Everything WP3 treats as an input, split by what it configures:

    paths       where the results live
    atmosphere  ISA relations the flight conditions are built on
    mission     the flight conditions and their flow regime
    geometry    every geometric input of the aircraft, in one list
    drag        class II drag settings: laminar fraction, roughness,
                interference factors, miscellaneous drag

Import the modules explicitly:

    from b12wp3.config import geometry as geometry_cfg
"""
