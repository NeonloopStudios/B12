"""Tests for wp2/scripts/xfoil_runtime.py.

Parsing tests run against every real file in wp2/airfoils/ (no synthetic
fixtures) since the two on-disk formats -- headerless and name-header --
are exactly what needs to keep working when the candidate set changes.

The XFoil integration tests actually call the compiled DLL (no mocking) --
per wp2/README.md this environment has it built and verified working, and
a mocked XFoil call would not catch a real regression in the DLL-loading
fix or the wrapper's argument handling. If xfoil isn't importable (e.g. a
different machine without the MinGW toolchain set up per the README), the
whole module is skipped rather than failing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import config

xfoil_runtime = pytest.importorskip("scripts.xfoil_runtime")


AIRFOIL_FILES = config.discover_airfoils()


@pytest.mark.parametrize("path", AIRFOIL_FILES, ids=[p.stem for p in AIRFOIL_FILES])
def test_load_airfoil_dat_parses_every_candidate_file(path: Path) -> None:
    airfoil = xfoil_runtime.load_airfoil_dat(path)
    assert airfoil.n_coords > 10
    # Selig-format coordinates are normalized x/c in [0, 1]
    assert airfoil.x.min() >= -0.01
    assert airfoil.x.max() <= 1.01
    # thickness (y range) should be a small fraction of chord, not garbage
    assert 0.0 < (airfoil.y.max() - airfoil.y.min()) < 0.5


_SAMPLE_COORDS = (
    "1.00000 0.00000\n0.90000 0.01260\n0.70000 0.03040\n0.50000 0.04410\n"
    "0.30000 0.05100\n0.15000 0.04620\n0.05000 0.02900\n0.00000 0.00000\n"
    "0.05000 -0.01700\n0.15000 -0.02400\n0.30000 -0.02200\n0.50000 -0.01500\n"
    "0.70000 -0.00800\n0.90000 -0.00200\n1.00000 0.00000\n"
)  # 15 points, well above the >=10 sanity threshold


def test_load_airfoil_dat_skips_name_header(tmp_path: Path) -> None:
    headered = tmp_path / "headered.dat"
    headered.write_text("test airfoil\n" + _SAMPLE_COORDS)
    airfoil = xfoil_runtime.load_airfoil_dat(headered)
    assert airfoil.n_coords == 15
    assert airfoil.x[0] == pytest.approx(1.0)


def test_load_airfoil_dat_handles_headerless_file(tmp_path: Path) -> None:
    headerless = tmp_path / "headerless.dat"
    headerless.write_text(_SAMPLE_COORDS)
    airfoil = xfoil_runtime.load_airfoil_dat(headerless)
    assert airfoil.n_coords == 15


def test_load_airfoil_dat_rejects_malformed_line(tmp_path: Path) -> None:
    bad = tmp_path / "bad.dat"
    bad.write_text("1.0 0.0\n0.5 0.05 extra\n0.0 0.0\n")
    with pytest.raises(ValueError, match="expected 'x y'"):
        xfoil_runtime.load_airfoil_dat(bad)


def test_load_airfoil_dat_rejects_too_few_points(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.dat"
    tiny.write_text("1.0 0.0\n0.0 0.0\n")
    with pytest.raises(ValueError, match="suspiciously few"):
        xfoil_runtime.load_airfoil_dat(tiny)


def test_naca_25112_converges_near_zero_alpha_at_cruise() -> None:
    path = config.AIRFOILS_DIR / "NACA_25112.dat"
    airfoil = xfoil_runtime.load_airfoil_dat(path)
    with xfoil_runtime.xfoil_session(
        airfoil,
        mach=config.CRUISE.mach_normal,
        reynolds=config.CRUISE.reynolds_normal,
        ncrit=config.CRUISE.ncrit,
    ) as xf:
        result = xfoil_runtime.run_alpha_sweep(xf, [0.0, 1.0, 2.0])

    assert list(result.columns) == [
        "alpha", "cl", "cd", "cm", "cp_min", "converged", "diverged", "rms_bl", "note",
    ]
    assert len(result) == 3
    assert result["converged"].all(), result[["alpha", "converged", "note"]]
    assert not result["diverged"].any()
    # RMS residual should be well under XFoil's convergence tolerance (1e-4)
    # for every converged point
    assert (result["rms_bl"] < 1e-4).all()
    # cambered section: cl should increase monotonically over this small,
    # well-attached alpha range
    assert result["cl"].is_monotonic_increasing
    assert (result["cd"] > 0).all()


def test_run_alpha_sweep_stops_early_after_repeated_nonconvergence() -> None:
    path = config.AIRFOILS_DIR / "NACA_25112.dat"
    airfoil = xfoil_runtime.load_airfoil_dat(path)
    with xfoil_runtime.xfoil_session(
        airfoil,
        mach=config.CRUISE.mach_normal,
        reynolds=config.CRUISE.reynolds_normal,
        ncrit=config.CRUISE.ncrit,
    ) as xf:
        # sweep deep past stall; absurd alphas should fail to converge and
        # the sweep should stop well short of the full requested range
        alphas = list(np.arange(0.0, 90.0, 1.0))
        result = xfoil_runtime.run_alpha_sweep(xf, alphas, stop_after_n_nonconverged=3)

    assert len(result) < len(alphas)
    assert not result["converged"].iloc[-3:].any()
    # non-converged tail should carry real diagnostics, not a blank note
    tail = result[~result["converged"]]
    assert (tail["note"] != "").all()
    assert tail["diverged"].any() or (tail["rms_bl"] > 0).any()


def test_cp_distribution_shape_matches_paneling() -> None:
    path = config.AIRFOILS_DIR / "NACA_25112.dat"
    airfoil = xfoil_runtime.load_airfoil_dat(path)
    with xfoil_runtime.xfoil_session(
        airfoil,
        mach=config.CRUISE.mach_normal,
        reynolds=config.CRUISE.reynolds_normal,
        ncrit=config.CRUISE.ncrit,
    ) as xf:
        xf.a(0.0)  # converge a point first so there's a Cp distribution
        cp = xfoil_runtime.cp_distribution(xf)

    assert set(cp.columns) == {"x", "y", "cp"}
    assert len(cp) > 100  # repaneled to 160 nodes by default
    assert cp["cp"].max() <= 1.5  # Cp <= 1 at stagnation, small numerical slack
