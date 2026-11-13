"""Arm C (primary): C-sex and C-all event study per prereg §3. Writes out/arm_c.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from e03 import event_study as es
from e03.fetch import french
from e03.prices import CSEX_TICKERS, returns_for

OUT = Path(__file__).resolve().parents[1] / "out"
WINDOWS = [(0, 1), (-1, 5), (0, 20)]


def run(events: pd.DataFrame, ticker_col: str, rets: pd.DataFrame, factors: pd.DataFrame,
        model: str) -> tuple[list, list[dict]]:
    results, audit = [], []
    for _, ev in events.iterrows():
        t = ev[ticker_col]
        status = "ok"
        res = None
        if pd.isna(ev["event_date"]):
            status = "no date"
        elif pd.isna(t) or t not in rets:
            status = "no prices"
        else:
            res = es.abnormal_returns(rets[t].dropna(), factors, ev["event_date"], model=model)
            status = "ok" if res is not None else "insufficient history"
        results.append(res)
        audit.append({"id": int(ev["dismissal_dataset_id"]), "coname": ev["coname"], "ticker": t,
                      "event_date": str(ev["event_date"])[:10], "status": status,
                      "day0": str(res.day0)[:10] if res else None,
                      "car01": es.car(res, (0, 1)) if res else None})
    return results, audit


def main() -> None:
    ev = pd.read_csv(OUT / "events_arm_c.csv", parse_dates=["event_date", "left_ceo_office"])
    ff = french("F-F_Research_Data_5_Factors_2x3_daily")
    csex = ev[ev["sex_cut"]].copy()
    csex["tk"] = csex["dismissal_dataset_id"].map(lambda i: CSEX_TICKERS[i][0] if CSEX_TICKERS[i] else None)
    pairs = [v for v in CSEX_TICKERS.values() if v]
    rets = returns_for(pairs, "1993-01-01", "2026-10-01", "csex")
    out: dict = {"prereg": "prereg.md", "label": "in-sample, historical"}
    for model in ["ff5", "mm"]:
        res, audit = run(csex, "tk", rets, ff, model)
        out[f"csex_{model}"] = {f"{w[0]}_{w[1]}": es.summarize(res, w) for w in WINDOWS}
        if model == "ff5":
            out["csex_audit"] = audit
    # C-all (secondary): SEC current-ticker map only (erratum E1 §5)
    call = ev[ev["ticker"].notna()]
    rets_all = returns_for([(t, "yahoo") for t in call["ticker"].unique()], "1993-01-01", "2026-10-01", "call")
    res_all, audit_all = run(call, "ticker", rets_all, ff, "ff5")
    out["call_ff5"] = {f"{w[0]}_{w[1]}": es.summarize(res_all, w) for w in WINDOWS}
    out["call_coverage"] = {"code4_total": int(len(ev)), "with_ticker": int(len(call)),
                            "usable": sum(r is not None for r in res_all)}
    (OUT / "arm_c.json").write_text(json.dumps(out, indent=1, default=str))
    p = out["csex_ff5"]["0_1"]
    print("C-sex FF5 CAR[0,+1]:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})
    print("status:", pd.Series([a["status"] for a in out["csex_audit"]]).value_counts().to_dict())
    print("C-all coverage:", out["call_coverage"], "CAR01 mean", round(out["call_ff5"]["0_1"]["mean_car"], 4),
          "p_kp", round(out["call_ff5"]["0_1"]["p_kp"], 4))


if __name__ == "__main__":
    main()


def sensitivity() -> dict:
    """Manual-audit sensitivity (erratum E1 §4): re-run C-sex FF5 on the audited classes only."""
    ev = pd.read_csv(OUT / "events_arm_c.csv", parse_dates=["event_date", "left_ceo_office"])
    audit = pd.read_csv(OUT / "csex_manual_audit.csv")
    ff = french("F-F_Research_Data_5_Factors_2x3_daily")
    pairs = [v for v in CSEX_TICKERS.values() if v]
    rets = returns_for(pairs, "1993-01-01", "2026-10-01", "csex")
    out = {}
    for label, classes in [("strict_S", {"S"}), ("S_plus_U", {"S", "U"})]:
        ids = set(audit.loc[audit["class"].isin(classes), "id"])
        sub = ev[ev["dismissal_dataset_id"].isin(ids)].copy()
        sub["tk"] = sub["dismissal_dataset_id"].map(lambda i: CSEX_TICKERS[i][0])
        res, _ = run(sub, "tk", rets, ff, "ff5")
        out[label] = {f"{w[0]}_{w[1]}": es.summarize(res, w) for w in WINDOWS}
    return out
