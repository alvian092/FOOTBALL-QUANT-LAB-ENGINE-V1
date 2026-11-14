"""Multiclass temperature calibration and reliability diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar


def _arrays(probabilities,outcomes):
    p=np.asarray(probabilities,dtype=float); y=np.asarray(outcomes,dtype=int)
    if p.ndim!=2 or p.shape[1]!=3 or len(y)!=len(p) or not len(y):
        raise ValueError("Need N x 3 probabilities and N outcomes.")
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-6):
        raise ValueError("Probabilities must be valid and sum to one.")
    if not np.isin(y,[0,1,2]).all(): raise ValueError("Outcomes must be 0=home, 1=draw, 2=away.")
    return np.clip(p,1e-12,1),y


def apply_temperature(probabilities,temperature):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-6):
        raise ValueError("Probabilities must be valid N x 3 rows.")
    if not np.isfinite(temperature) or temperature<=0: raise ValueError("Temperature must be positive.")
    logits=np.log(np.clip(p,1e-12,1))/temperature
    logits-=logits.max(axis=1,keepdims=True)
    exp=np.exp(logits)
    return exp/exp.sum(axis=1,keepdims=True)


def fit_temperature(probabilities,outcomes,min_samples=50):
    """Fit scalar temperature on a dedicated calibration split, never the final test set."""
    p,y=_arrays(probabilities,outcomes)
    if len(y)<min_samples: raise ValueError(f"Need at least {min_samples} calibration fixtures.")
    def loss(log_t):
        q=apply_temperature(p,np.exp(log_t))
        return float(-np.log(np.clip(q[np.arange(len(y)),y],1e-12,1)).mean())
    result=minimize_scalar(loss,bounds=(np.log(.05),np.log(20)),method="bounded")
    if not result.success: raise RuntimeError("Temperature optimization failed.")
    return {"temperature":float(np.exp(result.x)),"log_loss_before":loss(0.0),
            "log_loss_after":float(result.fun),"n":int(len(y))}


def calibration_report(probabilities,outcomes,bins=10):
    p,y=_arrays(probabilities,outcomes)
    if bins<2: raise ValueError("bins must be at least 2.")
    onehot=np.eye(3)[y]
    brier=float(np.mean(np.sum((p-onehot)**2,axis=1)))
    logloss=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
    confidence=p.max(axis=1); correct=(p.argmax(axis=1)==y).astype(float)
    edges=np.linspace(0,1,bins+1); rows=[]; ece=0.0
    for i in range(bins):
        mask=(confidence>=edges[i]) & ((confidence<edges[i+1]) if i<bins-1 else (confidence<=edges[i+1]))
        if not mask.any(): continue
        conf=float(confidence[mask].mean()); acc=float(correct[mask].mean())
        ece+=float(mask.mean())*abs(acc-conf)
        rows.append({"bin_low":float(edges[i]),"bin_high":float(edges[i+1]),
                     "count":int(mask.sum()),"mean_confidence":conf,"accuracy":acc})
    return {"n":int(len(y)),"brier":brier,"log_loss":logloss,"ece":float(ece),"bins":rows}
