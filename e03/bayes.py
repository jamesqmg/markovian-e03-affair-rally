"""Bayesian hierarchical model of CAR[0,+1] (prereg §6, erratum E4 §3).

CAR_i ~ Normal(theta_i, se_i)            se_i = FF5 forecast-error s.d. of the event's CAR[0,+1] (CAR / SCAR)
theta_i ~ StudentT(nu, mu_g[i], tau)     g = arm: 0 = C-sex (primary 24), 1 = C-other (usable C-all not in C-sex)
mu_g ~ Normal(mu, sigma_arm); mu ~ Normal(0, 0.05); sigma_arm ~ HalfNormal(0.02); tau ~ HalfNormal(0.05);
nu ~ Gamma(alpha=2, beta=0.1) (rate form, prior mean 20). Sampler: NUTS, tune 2000, target_accept 0.95.
With only two arms sigma_arm is prior-dominated, so P(mu_C-sex > 0) leans on its HalfNormal(0.02) prior."""
from __future__ import annotations

import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from e03 import event_study as es
from e03.inputs import OUT, load

SEED = 4
RHAT_MAX = 1.01
GROUPS = ["csex", "cother"]


def fit(car: np.ndarray, se: np.ndarray, group: np.ndarray, draws: int = 2000, chains: int = 4,
        seed: int = SEED) -> az.InferenceData:
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 0.05)
        sigma_arm = pm.HalfNormal("sigma_arm", 0.02)
        z = pm.Normal("z", 0.0, 1.0, shape=int(group.max()) + 1)
        mu_g = pm.Deterministic("mu_g", mu + sigma_arm * z)  # non-centred
        tau = pm.HalfNormal("tau", 0.05)
        nu = pm.Gamma("nu", 2.0, 0.1)
        eta = pm.StudentT("eta", nu=nu, mu=0.0, sigma=1.0, shape=len(car))
        theta = pm.Deterministic("theta", mu_g[group] + tau * eta)
        pm.Normal("car", theta, se, observed=car)
        return pm.sample(draws=draws, tune=2000, chains=chains, random_seed=seed, target_accept=0.95,
                         progressbar=False)


def summarize(idata: az.InferenceData, car: np.ndarray) -> dict:
    post = idata.posterior
    mu_g = post["mu_g"].stack(s=("chain", "draw")).values
    mu = post["mu"].stack(s=("chain", "draw")).values
    theta = post["theta"].mean(("chain", "draw")).values
    r = az.rhat(idata).to_array().values  # every variable, incl. z, eta, theta
    rhat = float("nan") if np.isnan(r).any() else float(r.max())
    return {
        "mu": {"mean": float(mu.mean()), "hdi94": [float(x) for x in az.hdi(mu, 0.94)], "p_gt0": float((mu > 0).mean())},
        "mu_group": [{"mean": float(m.mean()), "hdi94": [float(x) for x in az.hdi(m, 0.94)]} for m in mu_g],
        "p_mu_group_gt0": [float((m > 0).mean()) for m in mu_g],
        "theta_mean": theta.tolist(), "raw_car": [float(c) for c in car],
        "max_rhat": rhat, "converged": bool(np.isfinite(rhat) and rhat <= RHAT_MAX),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


def forecast_sd(res: es.EventResult, window: tuple[int, int]) -> float:
    """FF5 forecast-error s.d. of the window CAR: sigma * sqrt(L + s'(X'X)^-1 s) (the SCAR denominator)."""
    s = res.xw.loc[window[0]:window[1]].sum().to_numpy()
    return float(res.sigma * np.sqrt(window[1] - window[0] + 1 + s @ res.xtx_inv @ s))


def arm_table(csex: pd.DataFrame, call: pd.DataFrame, sex_ids: frozenset[int]) -> pd.DataFrame:
    """Disjoint arms: C-sex (usable) and C-other = usable C-all events with no sex_cut id at all."""
    other = call[~call["id"].isin(sex_ids)]
    t = pd.concat([csex.assign(group=0), other.assign(group=1)], ignore_index=True)
    assert not set(csex["id"]) & set(other["id"])
    return t


def _native(o: object) -> object:
    return o.item() if isinstance(o, np.generic) else str(o)


def main() -> None:
    x = load()
    t = arm_table(x.csex, x.call, x.sex_ids)
    assert (t["group"] == 0).sum() == 24, (t["group"] == 0).sum()
    car = np.array([es.car(r, (0, 1)) for r in t["res"]])
    se = np.array([forecast_sd(r, (0, 1)) for r in t["res"]])
    assert np.isfinite(car).all() and np.isfinite(se).all() and (se > 0).all()
    group = t["group"].to_numpy()
    s = summarize(fit(car, se, group), car)
    events = t[["id", "coname"]].assign(group=[GROUPS[g] for g in group], car=car, se=se, theta_mean=s.pop("theta_mean"))
    s.pop("raw_car")
    out = {"label": "in-sample, historical; secondary (prereg §6, erratum E4 §3)", "groups": GROUPS,
           "n": {"csex": int((group == 0).sum()), "cother": int((group == 1).sum())},
           "p_csex_gt0_locked": s["p_mu_group_gt0"][0], "p_csex_lt0": 1 - s["p_mu_group_gt0"][0], **s,
           "events": events.to_dict("records")}
    (OUT / "secondary_bayes.json").write_text(json.dumps(out, indent=1, default=_native, allow_nan=False))
    print({k: v for k, v in out.items() if k != "events"})


if __name__ == "__main__":
    main()
