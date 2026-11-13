"""Shared inputs for the prereg §6 secondary analyses (erratum E4): the primary's usable events,
their daily returns and the French daily factors, built exactly as in run_arm_c (same caches)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from e03 import event_study as es
from e03.fetch import french
from e03.prices import CSEX_TICKERS, closes_for

OUT = Path(__file__).resolve().parents[1] / "out"
START, END = "1993-01-01", "2026-10-01"


@dataclass(frozen=True)
class Inputs:
    ff: pd.DataFrame          # French daily FF5 + RF
    csex: pd.DataFrame        # usable C-sex events: id, coname, ticker, source, event_date, res
    call: pd.DataFrame        # usable C-all events: same columns
    rets_csex: pd.DataFrame   # daily simple returns, C-sex tickers
    rets_call: pd.DataFrame   # daily simple returns, C-all tickers
    closes_csex: pd.DataFrame  # the adjusted closes behind rets_csex (for month-end returns)
    closes_call: pd.DataFrame
    sex_ids: frozenset[int]   # every sex_cut id (34), usable or not: C-other excludes all of them


def _usable(events: pd.DataFrame, rets: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, e in events.iterrows():
        t = e["ticker"]
        if pd.isna(e["event_date"]) or pd.isna(t) or t not in rets:
            continue
        res = es.abnormal_returns(rets[t].dropna(), ff, e["event_date"], model="ff5")
        if res is None or pd.isna(es.car(res, (0, 1))):
            continue
        rows.append({**e[["dismissal_dataset_id", "coname", "ticker", "source", "event_date"]].to_dict(), "res": res})
    return pd.DataFrame(rows).rename(columns={"dismissal_dataset_id": "id"})


def load() -> Inputs:
    ev = pd.read_csv(OUT / "events_arm_c.csv", parse_dates=["event_date", "left_ceo_office"])
    ff = french("F-F_Research_Data_5_Factors_2x3_daily")
    csex = ev[ev["sex_cut"]].copy()
    mapped = csex["dismissal_dataset_id"].map(CSEX_TICKERS)
    csex["ticker"] = mapped.map(lambda v: v[0] if v else None)
    csex["source"] = mapped.map(lambda v: v[1] if v else None)
    closes_csex = closes_for([v for v in CSEX_TICKERS.values() if v], START, END, "csex")
    call = ev[ev["ticker"].notna()].copy()
    call["source"] = "yahoo"  # C-all uses the SEC current-ticker map with Yahoo only (erratum E1 §5)
    closes_call = closes_for([(t, "yahoo") for t in call["ticker"].unique()], START, END, "call")
    rets_csex = closes_csex.pct_change(fill_method=None)  # identical to returns_for
    rets_call = closes_call.pct_change(fill_method=None)
    return Inputs(ff=ff, csex=_usable(csex, rets_csex, ff), call=_usable(call, rets_call, ff),
                  rets_csex=rets_csex, rets_call=rets_call, closes_csex=closes_csex, closes_call=closes_call,
                  sex_ids=frozenset(int(i) for i in csex["dismissal_dataset_id"]))
