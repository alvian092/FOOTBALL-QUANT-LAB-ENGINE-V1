"""Conservative 1X2 value scanner; requires supplied lower confidence bounds."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .market import no_vig_probabilities

SELECTIONS=("home","draw","away")


def scan_1x2_market(fixtures: pd.DataFrame, min_edge=0.02, min_conservative_ev=0.0,
                    max_overround=0.15):
    required={"match_id","home_odds","draw_odds","away_odds",
              "model_home_p","model_draw_p","model_away_p",
              "lower_home_p","lower_draw_p","lower_away_p"}
    missing=required-set(fixtures.columns)
    if missing: raise ValueError(f"Missing scanner fields: {', '.join(sorted(missing))}")
    if min_edge<0 or max_overround<0: raise ValueError("Thresholds cannot be negative.")
    rows=[]
    for fixture in fixtures.to_dict("records"):
        odds=[float(fixture[f"{s}_odds"]) for s in SELECTIONS]
        probs=[float(fixture[f"model_{s}_p"]) for s in SELECTIONS]
        lowers=[float(fixture[f"lower_{s}_p"]) for s in SELECTIONS]
        if not np.isfinite(odds+probs+lowers).all() or any(o<=1 for o in odds):
            raise ValueError(f"Invalid odds/probability values for {fixture['match_id']}.")
        if any(p<0 or p>1 for p in probs+lowers) or not np.isclose(sum(probs),1,atol=1e-6):
            raise ValueError(f"Invalid probability vector for {fixture['match_id']}.")
        if any(l>p for l,p in zip(lowers,probs)):
            raise ValueError("Lower confidence bounds cannot exceed point probabilities.")
        market=no_vig_probabilities(dict(zip(SELECTIONS,odds)))
        overround=sum(1/o for o in odds)-1
        for s,o,p,lo in zip(SELECTIONS,odds,probs,lowers):
            edge=p-market[s]; ev=p*o-1; conservative_ev=lo*o-1
            eligible=(overround<=max_overround and edge>=min_edge and conservative_ev>min_conservative_ev)
            rows.append({"match_id":fixture["match_id"],"league":fixture.get("league","Unknown"),
                "selection":s,"odds":o,"model_probability":p,"market_probability":market[s],
                "edge":edge,"raw_ev":ev,"lower_probability":lo,
                "conservative_ev":conservative_ev,"overround":overround,
                "status":"VALUE CANDIDATE" if eligible else "NO BET"})
    result=pd.DataFrame(rows)
    if result.empty: return result
    result["_eligible_order"]=(result.status=="VALUE CANDIDATE").astype(int)
    return result.sort_values(["_eligible_order","conservative_ev","edge"],ascending=[False,False,False]).drop(columns="_eligible_order").reset_index(drop=True)
