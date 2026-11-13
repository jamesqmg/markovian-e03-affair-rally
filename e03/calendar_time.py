"""Calendar-time portfolios (prereg §6, erratum E4 §4): hold each firm months m+1..m+h after its event
month, equal-weighted; regress the portfolio's excess return on monthly FF5 + UMD with Newey-West(6).
Monthly returns come from month-end adjusted closes, so a missing-price day inside a month is spanned, not zeroed.
Weighting is per EVENT (the calendar-time convention): a firm with two events in the window counts twice.
p-values use the normal approximation (statsmodels HAC default)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from e03.fetch import french
from e03.inputs import OUT, load

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
NW_LAGS = 6
HORIZONS = (12, 36)


def monthly_from_closes(closes: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month returns from month-end closes; NaN for a month (and the next) with no close."""
    return closes.resample("ME").last().pct_change(fill_method=None)


def portfolio(monthly: pd.DataFrame, events: list[tuple[str, pd.Timestamp]], h: int) -> tuple[pd.Series, pd.Series]:
    """Equal-weighted return of the firms held each month, and how many were held."""
    held = pd.DataFrame(np.nan, index=monthly.index, columns=range(len(events)))
    for j, (ticker, day0) in enumerate(events):
        m = day0 + pd.offsets.MonthEnd(0)
        window = (monthly.index > m) & (monthly.index <= m + pd.offsets.MonthEnd(h))
        held.loc[window, j] = monthly.loc[window, ticker].to_numpy()
    n = held.notna().sum(axis=1)
    keep = n > 0
    return held[keep].mean(axis=1), n[keep]


def alpha(port: pd.Series, factors: pd.DataFrame) -> dict:
    """Monthly alpha of the portfolio's excess return on FF5 + UMD, Newey-West HAC; also the months used."""
    df = pd.concat([port.rename("p"), factors], axis=1, join="inner").dropna()
    fit = sm.OLS(df["p"] - df["RF"], sm.add_constant(df[FACTORS])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return {"alpha_monthly": float(fit.params["const"]), "t": float(fit.tvalues["const"]),
            "p": float(fit.pvalues["const"]), "months": int(len(df)),
            "first_month": str(df.index.min().date()), "last_month": str(df.index.max().date()),
            "used_index": df.index}


def monthly_factors() -> pd.DataFrame:
    """French monthly FF5 + RF joined with UMD ('Mom'); month-end index, decimals."""
    return french("F-F_Research_Data_5_Factors_2x3").join(french("F-F_Momentum_Factor")[["Mom"]], how="inner")


def main() -> None:
    x = load()
    f = monthly_factors()
    out: dict = {"label": "in-sample, historical; secondary (prereg §6, erratum E4 §4)", "nw_lags": NW_LAGS}
    assert f.index.max() == pd.Timestamp("2026-09-30"), f.index.max()  # E4 §4: through Sep 2026
    for arm, ev, closes in [("csex", x.csex, x.closes_csex), ("call", x.call, x.closes_call)]:
        monthly = monthly_from_closes(closes.loc[:"2026-09-30"])
        events = [(t, r.day0) for t, r in zip(ev["ticker"], ev["res"])]
        for h in HORIZONS:
            port, n = portfolio(monthly, events, h)
            res = alpha(port, f)
            used = res.pop("used_index")
            out[f"{arm}_h{h}"] = {**res, "mean_events_held": float(n.reindex(used).mean()), "events": len(events),
                                  "firms": len({t for t, _ in events})}
            print(arm, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"{arm}_h{h}"].items()})
    (OUT / "secondary_calendar_time.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
