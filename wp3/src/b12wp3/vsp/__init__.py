# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""OpenVSP 3-view model of the aircraft, built in two stages.

    model     the resolved model: every component as numbers, JSON in and out
    layout    stage 1 (wp3 env): inputs -> positions and shapes -> model
    airfoil   reads the wing airfoil .dat file for the layout
    builder   stage 2 (OpenVSP env): model -> OpenVSP geoms -> .vsp3

The OpenVSP Python API runs only in its own Python 3.11 env, so model.py and
builder.py must stay importable there: no Python 3.12+ syntax, and no import
of b12wp3.config.
"""
