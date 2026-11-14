import numpy as np
import pandas as pd
import pytest

from quantlab.ensemble import (
    elo_probabilities, elo_ratings_from_history, poisson_component,
    blend_probabilities, fit_ensemble_weights,
)
from quantlab.calibration import apply_temperature, fit_temperature, calibration_report
from quantlab.scanner import scan_1x2_market


def test_elo_probabilities_are_valid_and_home_advantage_matters():
    p=elo_probabilities(1500,1500)
    assert sum(p.values()) == pytest.approx(1)
    assert p["home"] > p["away"]
    assert elo_probabilities(1700,1500)["home"] > p["home"]


def test_elo_rating_replay_uses_only_supplied_history():
    history=pd.DataFrame({
        "date":pd.to_datetime(["2026-01-01","2026-01-08"],utc=True),
        "home_team":["A","B"],"away_team":["B","A"],
        "home_goals":[2,0],"away_goals":[0,1],
    })
    p=elo_ratings_from_history(history,"A","B")
    assert sum(p.values()) == pytest.approx(1)
    assert p["home"] > 0


def test_blend_normalizes_weights_and_probabilities():
    p=blend_probabilities({
        "poisson":{"home":.5,"draw":.25,"away":.25},
        "elo":{"home":.3,"draw":.3,"away":.4},
    },{"poisson":3,"elo":1})
    assert p["home"] == pytest.approx(.45)
    assert sum(p.values()) == pytest.approx(1)


def test_fit_ensemble_weights_requires_oof_sample_and_returns_simplex():
    y=np.array([0,1,2]*40)
    good=np.tile([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]],(40,1))
    weak=np.full((120,3),1/3)
    weights=fit_ensemble_weights({"strong":good,"weak":weak},y,min_samples=100)
    assert sum(weights.values()) == pytest.approx(1,abs=1e-6)
    assert weights["strong"] > weights["weak"]
    with pytest.raises(ValueError):
        fit_ensemble_weights({"a":good,"b":weak},y[:20],min_samples=100)


def test_temperature_scaling_preserves_valid_rows():
    p=np.array([[.7,.2,.1],[.2,.6,.2],[.1,.2,.7]])
    q=apply_temperature(p,1.5)
    assert np.allclose(q.sum(axis=1),1)
    assert q.shape==p.shape


def test_temperature_fit_reports_before_after_and_calibration_metrics():
    rng=np.random.default_rng(4)
    y=np.tile([0,1,2],40)
    p=np.full((len(y),3),.08)
    p[np.arange(len(y)),y]=.84
    fit=fit_temperature(p,y,min_samples=50)
    assert fit["n"]==120
    assert fit["temperature"]>0
    report=calibration_report(apply_temperature(p,fit["temperature"]),y,bins=5)
    assert report["n"]==120
    assert 0<=report["ece"]<=1
    assert report["brier"]>=0
    assert len(report["bins"])>0


def test_scanner_requires_lower_confidence_bounds_and_conservative_ev():
    fixtures=pd.DataFrame([{
        "match_id":"M1","league":"Test",
        "home_odds":2.2,"draw_odds":3.4,"away_odds":3.8,
        "model_home_p":.52,"model_draw_p":.27,"model_away_p":.21,
        "lower_home_p":.49,"lower_draw_p":.24,"lower_away_p":.18,
    }])
    result=scan_1x2_market(fixtures,min_edge=.01)
    home=result[result.selection=="home"].iloc[0]
    assert home.conservative_ev>0
    assert home.status=="VALUE CANDIDATE"
    assert result.status.eq("VALUE CANDIDATE").any()


def test_scanner_blocks_candidate_when_confidence_bound_is_too_low():
    fixtures=pd.DataFrame([{
        "match_id":"M2","home_odds":2.2,"draw_odds":3.4,"away_odds":3.8,
        "model_home_p":.52,"model_draw_p":.27,"model_away_p":.21,
        "lower_home_p":.40,"lower_draw_p":.20,"lower_away_p":.15,
    }])
    result=scan_1x2_market(fixtures,min_edge=.01)
    assert result[result.selection=="home"].iloc[0].status=="NO BET"


def test_scanner_rejects_invalid_market_inputs():
    fixtures=pd.DataFrame([{
        "match_id":"M3","home_odds":1,"draw_odds":3.4,"away_odds":3.8,
        "model_home_p":.52,"model_draw_p":.27,"model_away_p":.21,
        "lower_home_p":.49,"lower_draw_p":.24,"lower_away_p":.18,
    }])
    with pytest.raises(ValueError):
        scan_1x2_market(fixtures)
