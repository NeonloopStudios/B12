# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/polar.py and wp3/src/b12wp3/plots/induced.py.

Spec: ADSEE-II slide 49, CD = CD0 + K CL^2. At (L/D)max the induced drag
equals CD0, so CL = sqrt(CD0/K) and (L/D)max = 1/(2 sqrt(CD0 K)).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from b12wp3.drag.induced import InducedDrag
from b12wp3.drag.polar import DragPolar
from b12wp3.plots.induced import plot_drag_polar, plot_induced_drag


def test_polar_cd_returns_definition() -> None:
    polar = DragPolar(0.02, 0.05)

    assert polar.cd(0.5) == pytest.approx(0.02 + 0.05 * 0.25)


def test_polar_max_lift_to_drag_returns_hand_value() -> None:
    # CD0 = 0.02, K = 0.05: CL = sqrt(0.4), (L/D)max = 1 / (2 sqrt(0.001))
    polar = DragPolar(0.02, 0.05)

    assert polar.cl_at_max_lift_to_drag == pytest.approx(math.sqrt(0.4))
    assert polar.max_lift_to_drag == pytest.approx(1.0 / (2.0 * math.sqrt(0.001)))


@given(
    cd0=st.floats(min_value=0.005, max_value=0.1),
    k=st.floats(min_value=0.01, max_value=0.2),
    dcl=st.floats(min_value=0.01, max_value=0.5),
)
def test_polar_max_is_a_maximum(cd0: float, k: float, dcl: float) -> None:
    polar = DragPolar(cd0, k)
    cl_opt = polar.cl_at_max_lift_to_drag

    assert polar.lift_to_drag(cl_opt) == pytest.approx(polar.max_lift_to_drag)
    assert polar.lift_to_drag(cl_opt + dcl) < polar.max_lift_to_drag
    assert polar.lift_to_drag(max(cl_opt - dcl, 1e-6)) < polar.max_lift_to_drag


def test_polar_rejects_nonpositive_terms() -> None:
    with pytest.raises(ValueError):
        DragPolar(0.0, 0.05)
    with pytest.raises(ValueError):
        DragPolar(0.02, 0.0)


def test_polar_dataframe_sums_terms() -> None:
    frame = DragPolar(0.02, 0.05).to_dataframe([0.0, 0.5])

    assert list(frame["cd"]) == pytest.approx([0.02, 0.0325])
    assert list(frame["cd"]) == pytest.approx(list(frame["cd0"] + frame["cdi"]))


def test_plots_write_files(tmp_path: Path) -> None:
    without = InducedDrag("no winglet", 9.5, 9.5, 0.49, 0.0, 0.58, 0.58)
    with_winglet = InducedDrag("winglet", 9.5, 10.6, 0.49, 0.0, 0.56, 0.56)

    induced = plot_induced_drag(without, with_winglet, 0.489, tmp_path / "a" / "i.png")
    polar = plot_drag_polar(DragPolar(0.02, 0.05), 0.489, tmp_path / "p.png")

    assert induced.is_file()
    assert polar.is_file()
