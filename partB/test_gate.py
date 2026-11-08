"""Tests for decide() using synthetic numbers (Part B2). Run: pytest partB -q"""
import sys, copy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eta_mlflow import decide  # noqa: E402

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 4.0}

def cand(**over):
    """A candidate that passes every criterion."""
    return {**{"mae": 8.0, "mae_rush": 9.6, "mae_rain": 8.8, "p95_ms": 5.0}, **over}

def test_clear_winner_is_promoted():
    out = decide(PROD, cand())
    assert out["promote"] is True
    assert set(out["criteria"]) == {"mae_gain", "latency_p95", "slice_rush", "slice_rain"}
    assert all(c["pass"] for c in out["criteria"].values())

def test_gain_exactly_at_threshold_promotes():
    assert decide(PROD, cand(mae=9.70))["promote"] is True

def test_gain_just_below_threshold_blocks():
    out = decide(PROD, cand(mae=9.71))
    assert out["promote"] is False and out["criteria"]["mae_gain"]["pass"] is False

def test_worse_candidate_blocks():
    assert decide(PROD, cand(mae=10.5))["promote"] is False

@pytest.mark.parametrize("p95,ok", [(9.99, True), (10.0, True), (10.01, False)])
def test_latency_boundary(p95, ok):
    out = decide(PROD, cand(p95_ms=p95))
    assert out["criteria"]["latency_p95"]["pass"] is ok and out["promote"] is ok

@pytest.mark.parametrize("rush,ok", [(12.59, True), (12.6, True), (12.61, False)])
def test_rush_slice_boundary(rush, ok):
    out = decide(PROD, cand(mae_rush=rush))
    assert out["criteria"]["slice_rush"]["pass"] is ok and out["promote"] is ok

@pytest.mark.parametrize("rain,ok", [(11.549, True), (11.55, True), (11.551, False)])
def test_rain_slice_boundary(rain, ok):
    out = decide(PROD, cand(mae_rain=rain))
    assert out["criteria"]["slice_rain"]["pass"] is ok and out["promote"] is ok

def test_overall_gain_cannot_hide_a_slice_regression():
    out = decide(PROD, cand(mae=7.0, mae_rain=13.2))
    assert out["criteria"]["mae_gain"]["pass"] is True
    assert out["criteria"]["slice_rain"]["pass"] is False and out["promote"] is False

def test_custom_thresholds_are_respected():
    assert decide(PROD, cand())["promote"] is True
    assert decide(PROD, cand(), min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, cand(p95_ms=15), max_p95_ms=20.0)["promote"] is True
    assert decide(PROD, cand(mae_rush=13.0), max_slice_regression=0.10)["promote"] is True

def test_decide_is_pure():
    p, c = copy.deepcopy(PROD), cand()
    c_before = copy.deepcopy(c)
    first = decide(p, c)
    assert decide(p, c) == first
    assert p == PROD and c == c_before

def test_missing_metric_raises():
    with pytest.raises(KeyError):
        decide(PROD, {"mae": 8.0, "mae_rush": 9.0, "p95_ms": 1.0})
