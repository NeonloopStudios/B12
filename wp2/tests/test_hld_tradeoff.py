# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-06
"""Tests for wp2/src/b12wp2/hld/tradeoff.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from b12wp2.config import hld as hld_cfg, hld_v2 as v2
from b12wp2.hld import sizing as hs, tradeoff as t

CLEAN = hs.CleanWing(cl_max=1.36, cl_alpha_per_deg=0.0825, alpha_0l_deg=-2.5, alpha_crit_deg=14.2)
DEVICE = {d.name: d for d in (*hld_cfg.TE_DEVICES, *hld_cfg.LE_DEVICES)}


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    return t.score(t.grid(CLEAN))


def test_spars_follow_the_device_chord() -> None:
    assert t.front_spar(DEVICE["none"]) == v2.X_FS_MIN
    assert t.front_spar(DEVICE["slat"]) == pytest.approx(v2.LE_CHORD_RATIO["slat"])
    assert t.rear_spar(DEVICE["Fowler"]) == pytest.approx(1.0 - v2.TE_CHORD_RATIO["Fowler"])


def test_span_grid_stops_at_the_aileron_and_contains_the_minimum_span() -> None:
    etas = t.te_span_grid(0.4567)
    assert etas.min() > hld_cfg.ETA_IN
    assert etas.max() == pytest.approx(v2.ETA_AILERON_IN)
    assert np.isclose(etas, 0.4567).any()
    assert len(t.te_span_grid()) == len(etas) - 1  # NaN adds nothing


def test_minimum_span_meets_the_binding_requirement_exactly() -> None:
    req_l, req_to = t.requirements()
    te, le = DEVICE["triple-slotted"], t.le_effect(t.PLANFORM, DEVICE["slat"])
    eta = t.min_te_eta_out(t.PLANFORM, CLEAN, te, le, req_l, req_to)
    te_eff = t.te_effect(t.PLANFORM, te, eta)
    land = hs.configuration(CLEAN, te_eff, le)
    to = hs.configuration(CLEAN, te_eff, le, takeoff=True)
    assert min(land.cl_max - req_l, to.cl_max - req_to) == pytest.approx(0.0, abs=1e-5)


def test_too_weak_device_has_no_minimum_span() -> None:
    req_l, req_to = t.requirements()
    eta = t.min_te_eta_out(t.PLANFORM, CLEAN, DEVICE["plain"], t.NO_DEVICE, req_l, req_to)
    assert math.isnan(eta)


def test_feasible_rows_meet_both_requirements(scored: pd.DataFrame) -> None:
    req_l, req_to = t.requirements()
    f = scored[scored["feasible"]]
    assert (f["CLmax_L"] >= req_l - 1e-9).all() and (f["CLmax_TO"] >= req_to - 1e-9).all()
    assert scored.loc[~scored["feasible"], "score"].isna().all()
    assert (scored["eta_out_te"] <= v2.ETA_AILERON_IN + 1e-12).all()


def test_normalised_criteria_span_zero_to_one(scored: pd.DataFrame) -> None:
    f = scored[scored["feasible"]]
    for col in ("s_perf_L", "s_perf_TO", "s_le_spar", "s_te_spar", "s_te_area", "s_complexity"):
        assert f[col].min() == pytest.approx(0.0) and f[col].max() == pytest.approx(1.0)
    for col in (*t.CRITERIA, "score"):
        assert f[col].between(0.0, 1.0).all()


def test_minmax_direction_and_constant_metric() -> None:
    x = pd.Series([1.0, 2.0, 3.0])
    assert list(t.minmax(x, higher_is_better=True)) == [0.0, 0.5, 1.0]
    assert list(t.minmax(x, higher_is_better=False)) == [1.0, 0.5, 0.0]
    assert list(t.minmax(pd.Series([2.0, 2.0]), higher_is_better=False)) == [1.0, 1.0]


def test_best_per_config_is_the_max_over_its_grid(scored: pd.DataFrame) -> None:
    best = t.best_per_config(scored)
    assert set(best["config"]) == set(scored["config"])
    feas = best[best["status"] == "feasible"]
    expected = scored[scored["feasible"]].groupby("config")["score"].max()
    assert np.allclose(feas.set_index("config")["score"], expected.loc[feas["config"]])
    assert list(feas["rank"]) == list(range(1, len(feas) + 1))
    assert (best.loc[best["status"] == "discarded", "reason"] != "").all()


def test_pareto_front() -> None:
    v = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.4, 0.4], [1.0, 0.0]])
    assert list(t.pareto_front(v)) == [True, True, True, False, True]


def test_weight_grid_covers_the_simplex() -> None:
    w = list(t.weight_grid(0.1))
    assert len(w) == 66
    assert all(abs(sum(x) - 1.0) < 1e-12 and min(x) >= 0.0 for x in w)


def test_sensitivity_corner_weights(scored: pd.DataFrame) -> None:
    sens = t.sensitivity(scored).set_index(["w_performance", "w_space", "w_complexity"])
    f = scored[scored["feasible"]]
    top_perf = f.loc[f["s_performance"].idxmax(), "config"]
    assert sens.loc[(1.0, 0.0, 0.0), "winner"] == top_perf
