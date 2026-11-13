"""Arm D: Nvidia vs a synthetic Nvidia built from US chip peers (prereg §4).

Outcome: daily log returns (in %). Weights: CausalPy WeightedSumFitter (Dirichlet, on the simplex),
fitted on the 250 trading days ending 2026-10-28 (before the first disclosure, NYT Oct 29).
Effect: summed daily gap (actual - synthetic) over Nov 9-11; side windows Oct 29 and Nov 5.
Inference: placebo-in-space. The same method is run with each peer treated (donors = the other peers,
Nvidia excluded); Nvidia's post/pre RMSPE ratio is ranked among all 27 units.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import causalpy as cp
import numpy as np
import pandas as pd

from e03.fetch import prices

OUT = Path(__file__).resolve().parents[1] / "out"
PEERS = ["AVGO", "AMD", "TSM", "ASML", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC", "ADI", "MRVL",
         "NXPI", "MCHP", "ON", "MPWR", "ARM", "GFS", "TER", "ENTG", "COHR", "SWKS", "LSCC", "ALAB", "CRDO"]
PRE_END = pd.Timestamp("2026-10-28")
FIRST_POST = pd.Timestamp("2026-10-29")
LAST_DAY = pd.Timestamp("2026-11-11")  # override with --last YYYY-MM-DD once that day's FINAL close exists
WINDOWS = {"primary_nov9_11": ("2026-11-09", "2026-11-11"), "secondary_nov9_13": ("2026-11-09", "2026-11-13"),
           "side_oct29": ("2026-10-29", "2026-10-29"), "side_nov5": ("2026-11-05", "2026-11-05"),
           "side_nov12_ouster_report": ("2026-11-12", "2026-11-12")}  # prereg erratum E3
SAMPLE = {"draws": 1000, "tune": 1000, "chains": 2, "progressbar": False, "random_seed": 3}


def load_returns(last: pd.Timestamp) -> pd.DataFrame:
    end = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive
    px = prices(["NVDA"] + PEERS, "2025-09-01", end, "synth")
    lr = np.log(px).diff() * 100
    lr = lr.loc[:last].dropna(how="all")
    pre = lr.loc[:PRE_END].tail(250)
    post = lr.loc[FIRST_POST:last]
    return pd.concat([pre, post]).dropna(axis=1)


def fit(data: pd.DataFrame, treated: str, donors: list[str]) -> dict:
    d = data.reset_index(drop=True)
    t0 = int(np.flatnonzero(data.index == FIRST_POST)[0])
    model = cp.pymc_models.WeightedSumFitter(sample_kwargs=SAMPLE)
    sc = cp.SyntheticControl(d, treatment_time=t0, control_units=donors, treated_units=[treated], model=model)
    pred = sc.model.predict(d[donors])  # posterior predictive of the synthetic series
    mu = pred["posterior_predictive"]["mu"]
    synth_draws = mu.stack(sample=("chain", "draw")).values.squeeze()  # (T, S)
    if synth_draws.shape[0] != len(d):
        synth_draws = synth_draws.T
    gap_draws = d[treated].to_numpy()[:, None] - synth_draws
    gap = pd.Series(gap_draws.mean(axis=1), index=data.index)
    pre_rmspe = float(np.sqrt((gap.loc[:PRE_END] ** 2).mean()))
    post = gap.loc[WINDOWS["primary_nov9_11"][0]:WINDOWS["primary_nov9_11"][1]]  # placebo ranking always on the primary window
    out = {"pre_rmspe": pre_rmspe, "post_rmspe": float(np.sqrt((post ** 2).mean())),
           "weights": dict(zip(donors, sc.model.idata.posterior["beta"].mean(("chain", "draw")).values.ravel().tolist()))}
    out["ratio"] = out["post_rmspe"] / pre_rmspe
    for name, (a, b) in WINDOWS.items():
        idx = (data.index >= a) & (data.index <= b)
        if idx.any():
            s = gap_draws[idx].sum(axis=0)
            expected = len(pd.bdate_range(a, b))
            out[name] = {"complete": int(idx.sum()) == expected, "gap_pct": float(s.mean()), "qi94": [float(np.quantile(s, 0.03)), float(np.quantile(s, 0.97))],
                         "days": int(idx.sum())}
    out["series"] = {"date": [str(x.date()) for x in data.index], "actual": d[treated].round(4).tolist(),
                     "synthetic": synth_draws.mean(axis=1).round(4).tolist()}
    return out


def main(placebos: bool = True, last: pd.Timestamp = LAST_DAY) -> None:
    data = load_returns(last)
    peers = [p for p in PEERS if p in data.columns]
    res = {"label": "Nvidia case, 2026; window ends before Nov 17 earnings", "last_day": str(last.date()),
           "units": "gap_pct = summed daily log-return gap in log points; qi94 = 3-97% quantiles of weight uncertainty only",
           "n_pre": int((data.index <= PRE_END).sum()),
           "nvda": fit(data, "NVDA", peers)}
    print("NVDA", {k: v for k, v in res["nvda"].items() if k not in ("series", "weights")}, flush=True)
    if placebos:
        res["placebos"] = {}
        for p in peers:
            r = fit(data, p, [q for q in peers if q != p])
            res["placebos"][p] = {k: v for k, v in r.items() if k != "series"}
            print(p, round(r["ratio"], 2), flush=True)
        ratios = {"NVDA": res["nvda"]["ratio"], **{p: v["ratio"] for p, v in res["placebos"].items()}}
        ranked = sorted(ratios, key=ratios.get, reverse=True)
        res["placebo_rank"] = {"nvda_rank": ranked.index("NVDA") + 1, "of": len(ranked),
                               "p_value": (ranked.index("NVDA") + 1) / len(ranked), "order": ranked}
        print("rank", res["placebo_rank"]["nvda_rank"], "of", len(ranked), flush=True)
    (OUT / f"arm_d_nvidia_{last.date()}.json").write_text(json.dumps(res, indent=1))
    if placebos:  # only a full run (with placebo ranks) may replace the canonical file
        (OUT / "arm_d_nvidia.json").write_text(json.dumps(res, indent=1))


def require_final_close(last: pd.Timestamp, now: pd.Timestamp | None = None) -> None:
    """Refuse to run before `last`'s US close is final (16:15 ET buffer); an earlier pull is intraday (erratum E3)."""
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    final = pd.Timestamp(last.date()).tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    if now < final:
        raise SystemExit(f"{last.date()} close is not final until {final} (now {now:%Y-%m-%d %H:%M %Z}); not running.")


if __name__ == "__main__":
    last = pd.Timestamp(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else LAST_DAY
    require_final_close(last)
    main(placebos="--no-placebos" not in sys.argv, last=last)
