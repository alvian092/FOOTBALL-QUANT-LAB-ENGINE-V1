"""Transparent Poisson/Dixon-Coles/Elo probability ensemble."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .probability import calculate_probabilities


def _validate_probs(p):
    a=np.asarray(p,dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or (a<0).any() or (a>1).any() or not np.isclose(a.sum(),1,atol=1e-7):
        raise ValueError("Each 1X2 probability vector must be finite, in [0,1], and sum to one.")
    return a


def elo_probabilities(home_elo, away_elo, home_advantage=60.0, draw_probability=0.26):
    """Convert Elo expected score to 1X2 probabilities with an explicit draw prior."""
    vals=(home_elo,away_elo,home_advantage,draw_probability)
    if not np.isfinite(vals).all() or not 0 <= draw_probability < 1 or home_advantage < 0:
        raise ValueError("Invalid Elo or draw parameters.")
    expected=1.0/(1.0+10.0**(-((home_elo+home_advantage)-away_elo)/400.0))
    return {"home":float((1-draw_probability)*expected),
            "draw":float(draw_probability),
            "away":float((1-draw_probability)*(1-expected))}


def elo_ratings_from_history(history, home_team, away_team, initial=1500.0, k_factor=20.0,
                             home_advantage=60.0, draw_probability=0.26):
    """Replay only supplied historical results and return current Elo probabilities."""
    if k_factor <= 0 or not np.isfinite(k_factor): raise ValueError("k_factor must be positive.")
    ratings={}
    if not history.empty:
        df=history.copy()
        df["date"]=__import__("pandas").to_datetime(df["date"],utc=True)
        for row in df.sort_values("date").itertuples():
            h,a=str(row.home_team),str(row.away_team)
            rh=ratings.get(h,initial); ra=ratings.get(a,initial)
            expected=1/(1+10**(-((rh+home_advantage)-ra)/400))
            actual=1.0 if row.home_goals>row.away_goals else 0.5 if row.home_goals==row.away_goals else 0.0
            delta=k_factor*(actual-expected)
            ratings[h]=rh+delta
            ratings[a]=ra-delta
    return elo_probabilities(ratings.get(home_team,initial),ratings.get(away_team,initial),
                             home_advantage,draw_probability)


def poisson_component(lambda_home, lambda_away, rho=-0.08):
    p=calculate_probabilities(lambda_home,lambda_away,rho=rho)
    return {"home":p.home_win,"draw":p.draw,"away":p.away_win}


def blend_probabilities(components, weights=None):
    """Linear probability pool; component probabilities must be calibrated/OOS where possible."""
    if not components: raise ValueError("At least one model component is required.")
    names=list(components)
    vectors=np.vstack([_validate_probs([components[n][k] for k in ("home","draw","away")]) for n in names])
    if weights is None: w=np.ones(len(names))/len(names)
    else:
        if set(weights)!=set(names): raise ValueError("Weights must be supplied for every component.")
        w=np.asarray([weights[n] for n in names],dtype=float)
        if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0: raise ValueError("Weights must be finite, non-negative and have positive sum.")
        w=w/w.sum()
    pooled=w@vectors
    pooled=pooled/pooled.sum()
    return {"home":float(pooled[0]),"draw":float(pooled[1]),"away":float(pooled[2])}


def fit_ensemble_weights(oof_predictions, outcomes, min_samples=100):
    """Fit non-negative simplex weights on out-of-fold predictions only."""
    names=list(oof_predictions)
    if len(names)<2: raise ValueError("At least two model components are required.")
    y=np.asarray(outcomes,dtype=int)
    if len(y)<min_samples: raise ValueError(f"Need at least {min_samples} out-of-fold fixtures.")
    if not np.isin(y,[0,1,2]).all(): raise ValueError("Outcomes must be encoded 0=home, 1=draw, 2=away.")
    matrices=[np.asarray(oof_predictions[n],dtype=float) for n in names]
    if any(m.shape!=(len(y),3) or not np.isfinite(m).all() or (m<0).any() or not np.allclose(m.sum(axis=1),1,atol=1e-6) for m in matrices):
        raise ValueError("Each component must provide an N x 3 valid probability matrix.")
    cube=np.stack(matrices,axis=0)
    def objective(w):
        p=np.einsum("m,mnk->nk",w,cube)
        return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
    result=minimize(objective,np.ones(len(names))/len(names),method="SLSQP",
        bounds=[(0,1)]*len(names),constraints={"type":"eq","fun":lambda w:w.sum()-1},
        options={"maxiter":1000,"ftol":1e-12})
    if not result.success: raise RuntimeError(f"Ensemble weight optimization failed: {result.message}")
    return {name:float(weight) for name,weight in zip(names,result.x)}
