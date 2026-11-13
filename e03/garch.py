"""GARCH(1,1)-t standardized CARs and abnormal volume (prereg §6, erratum E4 §5-6).

GARCH-SCAR: FF5 betas from [-250, -30] (as the primary); a zero-mean GARCH(1,1)-t is fitted to that model's
residuals over [-250, -1] (x100); CAR[0,+1] is divided by sqrt(h0 + h1), the 1- and 2-step variance forecasts.
Abnormal volume: log(1 + volume) minus its [-250, -30] mean, over that window's s.d., on days 0 and +1."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from e03 import event_study as es
from e03.inputs import END, OUT, START, load
from e03.prices import volumes_for

SCALE = 100.0
MIN_VOL_OBS = 120


def ff5_resid(ret: pd.Series, ff: pd.DataFrame, day0: pd.Timestamp) -> pd.Series:
    """FF5 residuals on relative days [-250, -1], with betas from [-250, -30] (the primary's estimation window)."""
    ret = ret.copy()
    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None).normalize()
    df = ff.sort_index().join(ret.rename("r"), how="left")
    df = df.assign(rel=np.arange(len(df)) - df.index.get_loc(day0))
    est = df[(df.rel >= es.EST[0]) & (df.rel <= es.EST[1])].dropna(subset=["r"])
    x = np.column_stack([np.ones(len(est)), est[es.FF5].to_numpy()])
    coef, *_ = np.linalg.lstsq(x, (est["r"] - est["RF"]).to_numpy(), rcond=None)
    pre = df[(df.rel >= es.EST[0]) & (df.rel <= -1)].dropna(subset=["r"])
    xp = np.column_stack([np.ones(len(pre)), pre[es.FF5].to_numpy()])
    return pd.Series((pre["r"] - pre["RF"]).to_numpy() - xp @ coef, index=pre.rel.to_numpy())


def garch_scar(resid: pd.Series, car01: float) -> dict | None:
    """resid indexed by relative day; the forecast origin must be day -1 so h0, h1 are days 0 and +1.
    None (dropped and counted, erratum E4 §5) if the origin is not day -1, history is short, the fit does not
    converge, or a forecast is not finite and positive."""
    if len(resid) < es.MIN_OBS or resid.index.max() != -1:
        return None
    model = arch_model(resid.to_numpy() * SCALE, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DataScaleWarning etc.; convergence is checked explicitly below
        fit = model.fit(disp="off")
    if fit.convergence_flag != 0:
        return None
    h = fit.forecast(horizon=2, reindex=False).variance.to_numpy()[-1] / SCALE ** 2
    if not (np.isfinite(h).all() and (h > 0).all()):
        return None
    prm = fit.params
    return {"h0": float(h[0]), "h1": float(h[1]), "scar": float(car01 / np.sqrt(h[0] + h[1])),
            "alpha": float(prm["alpha[1]"]), "beta": float(prm["beta[1]"]), "nu": float(prm["nu"])}


def abnormal_volume(vol: pd.Series) -> dict | None:
    """vol indexed by relative trading day. Returns standardized abnormal log volume for day 0, +1 and their mean."""
    v = np.log1p(vol.astype(float))
    est = v.loc[es.EST[0]:es.EST[1]].dropna()
    if len(est) < MIN_VOL_OBS or est.std() == 0 or 0 not in v.index or 1 not in v.index:
        return None
    sav = (v - est.mean()) / est.std(ddof=1)
    if pd.isna(sav.loc[0]) or pd.isna(sav.loc[1]):
        return None
    return {0: float(sav.loc[0]), 1: float(sav.loc[1]), "w01": float((sav.loc[0] + sav.loc[1]) / 2),
            "zero_days_est": int((vol.loc[es.EST[0]:es.EST[1]] == 0).sum()),
            "zero_days_event": int((vol.loc[0:1] == 0).sum())}


def _rel_volume(vol: pd.Series, cal: pd.DatetimeIndex, day0: pd.Timestamp, last: int = 1) -> pd.Series:
    """Volume on the French trading calendar, indexed by relative day [-250, +last]."""
    vol = vol.copy()
    vol.index = pd.DatetimeIndex(vol.index).tz_localize(None).normalize()
    pos0 = cal.get_loc(day0)
    lo = max(pos0 + es.EST[0], 0)
    hi = min(pos0 + last + 1, len(cal))
    return pd.Series(vol.reindex(cal[lo:hi]).to_numpy(), index=np.arange(lo, hi) - pos0)


def _t(x: np.ndarray) -> dict:
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float("nan"), "t": float("nan"), "p": float("nan")}
    t = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
    return {"n": int(len(x)), "mean": float(x.mean()), "t": t, "p": float(2 * stats.t.sf(abs(t), len(x) - 1))}


def run_arm(ev: pd.DataFrame, rets: pd.DataFrame, ff: pd.DataFrame, tag: str) -> dict:
    vols = volumes_for(list(zip(ev["ticker"], ev["source"])), START, END, f"{tag}_vol")
    assert not vols.columns.duplicated().any()
    rbar = es.mean_resid_corr(list(ev["res"]))
    scars, av, diag = [], {0: [], 1: [], "w01": []}, []
    dropped = {"garch_dropped": 0, "no_volume": 0}
    for i, t, res in zip(ev["id"], ev["ticker"], ev["res"]):
        g = garch_scar(ff5_resid(rets[t].dropna(), ff, res.day0), es.car(res, (0, 1)))
        a = abnormal_volume(_rel_volume(vols[t], ff.index, res.day0)) if t in vols else None
        diag.append({"id": int(i), "ticker": t, **({k: g[k] for k in ("alpha", "beta", "nu", "scar")} if g else {}),
                     **({k: a[k] for k in ("zero_days_est", "zero_days_event")} if a else {})})
        if g is None:
            dropped["garch_dropped"] += 1
        else:
            scars.append(g["scar"])
        if a is None:
            dropped["no_volume"] += 1
        else:
            for k in av:
                av[k].append(a[k])
    s = np.array(scars)
    g_out = _t(s)
    factor = es.kp_factor(rbar, len(s))
    g_out.update({"rbar": rbar, "t_kp": g_out["t"] * factor,
                  "p_kp": float(2 * stats.t.sf(abs(g_out["t"] * factor), len(s) - 1))})
    return {"garch_scar_0_1": g_out, "abnormal_volume": {str(k): _t(np.array(v)) for k, v in av.items()},
            "dropped": dropped, "per_event": diag}


def main() -> None:
    x = load()
    out = {"label": "in-sample, historical; secondary (prereg §6, erratum E4 §5-6)",
           "csex": run_arm(x.csex, x.rets_csex, x.ff, "csex"), "call": run_arm(x.call, x.rets_call, x.ff, "call")}
    (OUT / "secondary_garch_volume.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
