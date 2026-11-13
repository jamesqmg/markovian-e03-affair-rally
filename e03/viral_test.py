"""Arm B: replicate the viral reel's monthly FF5 pre/post test on its 15 cases (prereg §5).

Visible spec: r - rf = a + delta*Post + b'FF5, 36 months before vs 36 after, Newey-West(3).
ASSUMPTIONS (not visible in his reel, fixed before the run):
  A1 the event month is excluded; A2 monthly returns from month-end adjusted closes;
  A3 US firms use US FF5; BP (NYSE ADR, USD) uses Europe FF5; SK Hynix (KRW) is converted to USD
     and uses Asia-Pacific ex Japan FF5; A4 post windows are cut at the last French month (Sep 2026).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from e03.events import arm_b
from e03.fetch import french, prices

OUT = Path(__file__).resolve().parents[1] / "out"
FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
REGION = {"BP": "Europe_5_Factors", "000660.KS": "Asia_Pacific_ex_Japan_5_Factors"}


def monthly_returns(closes: pd.Series) -> pd.Series:
    m = closes.dropna().resample("ME").last()
    return m.pct_change().dropna()


def fit_case(ret_m: pd.Series, factors: pd.DataFrame, event: pd.Timestamp) -> dict | None:
    ev_month = event + pd.offsets.MonthEnd(0)
    df = pd.concat([ret_m.rename("r"), factors], axis=1, join="inner").dropna()
    pre = df[(df.index < ev_month) & (df.index >= ev_month - pd.DateOffset(months=36))]
    post = df[(df.index > ev_month) & (df.index <= ev_month + pd.DateOffset(months=36))]
    if len(pre) < 24 or len(post) < 12:
        return None
    d = pd.concat([pre.assign(post=0.0), post.assign(post=1.0)])
    x = sm.add_constant(d[["post"] + FF5])
    fit = sm.OLS(d["r"] - d["RF"], x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return {"delta": float(fit.params["post"]), "t": float(fit.tvalues["post"]),
            "p": float(fit.pvalues["post"]), "alpha_pre": float(fit.params["const"]),
            "n_pre": len(pre), "n_post": len(post)}


def main() -> None:
    ev = arm_b()
    tickers = sorted(set(ev["ticker"]) | {"KRW=X"})
    px = prices(tickers, "1999-01-01", "2026-10-01", "viral")
    krw_per_usd = px["KRW=X"]
    px = px.assign(**{"000660.KS": px["000660.KS"] / krw_per_usd})  # KRW price -> USD
    fac = {"US": french("F-F_Research_Data_5_Factors_2x3")}
    for name in set(REGION.values()):
        fac[name] = french(name)
    rows = []
    for _, e in ev.iterrows():
        f = fac[REGION.get(e["ticker"], "US")]
        res = fit_case(monthly_returns(px[e["ticker"]]), f, e["event_date"])
        rows.append({"case": e["case"], "ticker": e["ticker"], "group": e["group"],
                     "event_date": str(e["event_date"].date()), "not_ceo_at_event": bool(e["not_ceo_at_event"]),
                     **(res or {"delta": None})})
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["delta"])
    summary = {
        "n": len(ok), "mean_delta_monthly": float(ok["delta"].mean()),
        "t_cross": float(ok["delta"].mean() / (ok["delta"].std(ddof=1) / np.sqrt(len(ok)))),
        "n_positive": int((ok["delta"] > 0).sum()), "n_sig_pos_5pct": int(((ok["p"] < 0.05) & (ok["delta"] > 0)).sum()),
        "n_sig_neg_5pct": int(((ok["p"] < 0.05) & (ok["delta"] < 0)).sum()),
        "reel_only_mean_delta": float(ok.query("group == 'reel'")["delta"].mean()),
    }
    (OUT / "arm_b_viral.json").write_text(json.dumps({"label": "in-sample, historical",
                                                     "assumptions": __doc__, "summary": summary,
                                                     "cases": rows}, indent=1))
    print(df[["case", "event_date", "delta", "t", "p", "n_pre", "n_post"]].round(4).to_string())
    print(summary)


if __name__ == "__main__":
    main()
