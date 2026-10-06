# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""Inputs of the HLD v2 trade-off (b12wp2.hld.tradeoff): the chord each
device type takes (and so the spar it pushes), the aileron limit on the
trailing-edge span, the span grid and the trade-off weights.

The device table itself (ADSEE increments, c'/c, complexity ranks), ETA_IN,
ETA_OUT_LE, the take-off fraction and the CL_max requirements are shared
with v1 and stay in config.hld.
"""
from __future__ import annotations

from b12wp2.config import hld as v1

# --- chord of each device type: engineering judgment, like the complexity rank ---
# ADSEE dcl_max does not depend on the device chord (only through c'/c,
# which config.hld fixes per type), so the chord is a per-type input rather
# than a design variable -- optimising it would drive it to zero.

# c_f / c -> rear spar at x/c = 1 - c_f/c
TE_CHORD_RATIO: dict[str, float] = {
    "plain": 0.25,
    "split": 0.25,
    "single-slotted": 0.28,
    "Fowler": 0.30,
    "double-slotted": 0.33,
    "triple-slotted": 0.36,
}

# c_s / c -> front spar at x/c = max(X_FS_MIN, c_s/c)
LE_CHORD_RATIO: dict[str, float] = {
    "none": 0.0,
    "LE flap": 0.15,
    "Krueger": 0.15,
    "slat": 0.20,  # = the ADSEE front-spar position
}

X_FS_MIN = 0.12  # front spar of a wing without a leading-edge device

# --- spanwise layout ---
# Leading-edge devices run the full available span (config.hld.ETA_IN ..
# ETA_OUT_LE): they sit in front of the aileron and compete with nothing.
# Trailing-edge devices run from ETA_IN to at most the aileron's inboard
# edge; their outboard end is the design variable of the grid.

ETA_AILERON_IN = 0.75  # update after the roll-control sizing (WP2.3a)
ETA_TE_STEP = 0.01  # grid step of the trailing-edge outboard end

# --- trade-off weights ---

WEIGHTS: dict[str, float] = {
    "performance": 1 / 3,
    "space": 1 / 3,
    "complexity": 1 / 3,
}

# performance = landing / take-off CL_max, each relative to its requirement
PERFORMANCE_WEIGHTS: dict[str, float] = {"landing": 0.5, "takeoff": 0.5}

# occupied space = LE part (front spar) + TE part (rear spar + flapped area)
SPACE_WEIGHTS: dict[str, float] = {"le": 0.5, "te": 0.5}
TE_SPACE_WEIGHTS: dict[str, float] = {"spar": 0.5, "area": 0.5}

SENSITIVITY_STEP = 0.1  # grid step of the weight-sensitivity sweep

# --- consistency checks ---

for _name, _w in (
    ("WEIGHTS", WEIGHTS),
    ("PERFORMANCE_WEIGHTS", PERFORMANCE_WEIGHTS),
    ("SPACE_WEIGHTS", SPACE_WEIGHTS),
    ("TE_SPACE_WEIGHTS", TE_SPACE_WEIGHTS),
):
    if abs(sum(_w.values()) - 1.0) > 1e-9:
        raise AssertionError(f"config.hld_v2.{_name} must sum to 1.0, got {sum(_w.values())}")

if set(TE_CHORD_RATIO) != {d.name for d in v1.TE_DEVICES}:
    raise AssertionError("config.hld_v2.TE_CHORD_RATIO must name exactly the devices in config.hld.TE_DEVICES")
if set(LE_CHORD_RATIO) != {d.name for d in v1.LE_DEVICES}:
    raise AssertionError("config.hld_v2.LE_CHORD_RATIO must name exactly the devices in config.hld.LE_DEVICES")
if not v1.ETA_IN < ETA_AILERON_IN <= 1.0:
    raise AssertionError("config.hld_v2.ETA_AILERON_IN must lie between config.hld.ETA_IN and 1")
