"""Build out/results_card.json: every on-screen number, with its source file and label.
Every headline number is read from this card, so each one traces to a file in out/."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
HOOK_CSV = ROOT / "data" / "raw" / "chip_raw_closes_2026-11-03_11.csv"  # Yahoo closes, fetched locally, not redistributed


MDE_LINE = 0.02  # prereg §3: "CI includes 0, and MDE <= 2%" -> Not supported


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:+.{nd}f}%"


MCAP_START, MCAP_END = "2026-11-06", "2026-11-12"  # erratum E6: last close before the letter -> the resignation day


def nvda_market_cap() -> dict:
    """Raw market value change from the last close before the Nov 8 letter to the resignation day's close (erratum E6).
    No counterfactual. Cached; a cache built for a different end date is ignored."""
    cache = OUT / "nvda_mcap.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if (cached.get("start"), cached.get("end")) == (MCAP_START, MCAP_END):
            return cached
    settled = pd.Timestamp(f"{MCAP_END} 16:15", tz="America/New_York")
    if pd.Timestamp.now(tz="America/New_York") < settled:  # before this, yfinance's MCAP_END bar is still intraday
        raise SystemExit(f"NVDA {MCAP_END} close not settled until {settled} (= 05:15 PHT the next day)")
    import yfinance as yf

    px = yf.download("NVDA", start="2026-11-03", end="2026-11-14", auto_adjust=False, progress=False)["Close"]
    if px.empty:
        raise SystemExit("NVDA download came back empty (network or rate limit); retry")
    px = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
    missing = [d for d in (MCAP_START, MCAP_END) if pd.Timestamp(d) not in px.index]
    if missing:
        raise SystemExit(f"NVDA closes missing for {missing}")
    shares = float(yf.Ticker("NVDA").fast_info["shares"])
    c0, c1 = float(px.loc[MCAP_START]), float(px.loc[MCAP_END])
    out = {"start": MCAP_START, "end": MCAP_END, "close_start": c0, "close_end": c1, "shares": shares,
           "fetched": str(pd.Timestamp.now("UTC"))[:19], "loss_usd_bn": round((c0 - c1) * shares / 1e9),
           "pct": round((c1 / c0 - 1) * 100, 2),
           "label": "raw market value, Nov 6 close to Nov 12 close (the resignation day); includes all news that week, "
                    "not just the affair; current shares outstanding"}
    cache.write_text(json.dumps(out, indent=1))
    return out


def posthoc_split(c: dict) -> dict:
    """Mean CAR[0,+1] by manual-audit class. NOT pre-registered: shown on screen as a lead, not a result."""
    audit = pd.read_csv(OUT / "csex_manual_audit.csv")
    cars = pd.DataFrame(c["csex_audit"]).dropna(subset=["car01"]).merge(audit[["id", "class"]], on="id")
    out = {"label": "post-hoc, not pre-registered; simple means, no test"}
    for cls, name in [("S", "spelled_out_sex_or_relationship"), ("U", "unspecified_personal_conduct")]:
        g = cars.loc[cars["class"] == cls, "car01"]
        out[name] = {"n": int(len(g)), "mean_car_0_1": pct(g.mean()), "n_fell": int((g < 0).sum())}
    return out


def secondary(c: dict, b: dict) -> dict:
    """Prereg §6 secondary results (errata E4-E5) plus one Holm family over every secondary p-value (§7).
    The primary cell (C-sex FF5 CAR[0,+1], KP-BMP) is excluded from the family and stays unadjusted."""
    from scipy import stats

    from e03.placebo import holm

    ct = json.loads((OUT / "secondary_calendar_time.json").read_text())
    gv = json.loads((OUT / "secondary_garch_volume.json").read_text())
    by = json.loads((OUT / "secondary_bayes.json").read_text())
    pl = json.loads((OUT / "secondary_placebo_loo.json").read_text())
    p = {}
    for arm in ("csex_ff5", "csex_mm", "call_ff5"):
        for w, v in c[arm].items():
            for test, key in (("kp", "p_kp"), ("rank", "p_corrado"), ("sign", "p_sign_exact")):
                if not (arm == "csex_ff5" and w == "0_1" and test == "kp"):
                    p[f"eventstudy_{arm}_{w}_{test}"] = v[key]
    p.update({f"calendar_{k}": v["p"] for k, v in ct.items() if k.startswith(("csex_", "call_"))})
    for arm in ("csex", "call"):
        p[f"garch_scar_{arm}"] = gv[arm]["garch_scar_0_1"]["p_kp"]
        p.update({f"volume_{arm}_{k if k == 'w01' else 'day' + k}": v["p"] for k, v in gv[arm]["abnormal_volume"].items()})
    p["placebo_dates_two_sided"] = pl["placebo_dates"]["p_two_sided_abs"]
    rb = b["summary"]  # arm B (replication) is included deliberately: its mean-delta t is a secondary p-value too
    p["viral_mean_delta"] = float(2 * stats.t.sf(abs(rb["t_cross"]), rb["n"] - 1))
    adj = holm(p)
    g = by["mu_group"][by["groups"].index("csex")]
    return {
        "label": "secondary, in-sample; none can change the verdict (prereg §6-7, errata E4-E5)",
        "calendar_time_csex": {h: {"alpha_monthly": pct(ct[f"csex_h{h}"]["alpha_monthly"], 2), "p": round(ct[f"csex_h{h}"]["p"], 3)}
                               for h in (12, 36)},
        "calendar_time_call": {h: {"alpha_monthly": pct(ct[f"call_h{h}"]["alpha_monthly"], 2), "p": round(ct[f"call_h{h}"]["p"], 3),
                                   "caveat": "C-all tickers come from today's SEC map: survivors only"} for h in (12, 36)},
        "bayes_csex": {"mu_mean": pct(g["mean"], 2), "hdi94": [pct(x, 2) for x in g["hdi94"]],
                       "p_mu_gt0_locked": round(by["p_csex_gt0_locked"], 3), "p_mu_lt0": round(by["p_csex_lt0"], 3),
                       "converged": by["converged"], "max_rhat": round(by["max_rhat"], 4), "divergences": by["divergences"],
                       "caveat": "two arms: sigma_arm is prior-dominated; divergences are reported, convergence is judged by R-hat (E4 §3)"},
        "garch_scar_csex": {k: round(gv["csex"]["garch_scar_0_1"][k], 3) for k in ("mean", "t_kp", "p_kp", "n")},
        "volume_csex_day0_1": {**{k: round(gv["csex"]["abnormal_volume"]["w01"][k], 3) for k in ("mean", "t", "n")},
                               "p": f"{gv['csex']['abnormal_volume']['w01']['p']:.1e}"},
        "placebo_dates": {"observed": pct(pl["placebo_dates"]["observed"], 2),
                          "placebo_q05_q95": [pct(x, 2) for x in pl["placebo_dates"]["q05_q95"]],
                          "draws_as_low": f"{round(pl['placebo_dates']['share_one_sided_le'] * pl['placebo_dates']['draws'])} "
                                          f"of {pl['placebo_dates']['draws']}",
                          "p_two_sided": round(pl["placebo_dates"]["p_two_sided_abs"], 3),
                          "caveat": "the null has no event-induced variance, so it is less conservative than KP-BMP"},
        "loo_viral_arm_b_monthly": [pct(pl["loo_arm_b"][k], 2) for k in ("min", "max")],
        "loo_claim_arm_a_monthly": [pct(pl["loo_arm_a"][k], 2) for k in ("min", "max")],
        "holm": {"family_size": len(p), "n_reject_5pct": sum(v < 0.05 for v in adj.values()),
                 "rejected": sorted(k for k, v in adj.items() if v < 0.05),
                 "raw_and_adjusted": {k: [round(p[k], 4), round(adj[k], 4)] for k in sorted(p)}},
    }


def main() -> None:
    c = json.loads((OUT / "arm_c.json").read_text())
    sens = json.loads((OUT / "arm_c_sensitivity.json").read_text())
    b = json.loads((OUT / "arm_b_viral.json").read_text())
    d = json.loads((OUT / "arm_d_nvidia.json").read_text())
    ev = pd.read_csv(OUT / "events_arm_c.csv")
    if HOOK_CSV.exists():
        closes = pd.read_csv(HOOK_CSV, index_col=0, parse_dates=True)
        day = closes.pct_change().loc["2026-11-09"]
        single = [t for t in day.index if not t.startswith("^") and t not in ("SMH", "SOXX") and pd.notna(day[t])]
        hook = {"nvda_nov9": pct(day["NVDA"]), "up": int((day[single] > 0).sum()), "of": len(single),
                "sox": pct(day["^SOX"]), "source": str(HOOK_CSV.name)}
    else:  # the price file is not redistributed: keep the Nov 9 numbers from the card that ships in out/
        hook = json.loads((OUT / "results_card.json").read_text())["L01_hook"]
    p = c["csex_ff5"]["0_1"]
    s = sens["strict_S"]["0_1"]
    gap_log = d["nvda"]["primary_nov9_11"]["gap_pct"] / 100  # JSON field is summed log-return gap, in log points
    _, hi = p["ci95_kp"]
    if p["mean_car"] > 0 and p["p_kp"] < 0.05:
        verdict, ending = "Held", "Positive"
    elif hi < 0:
        verdict, ending = "Contradicted", "Negative"
    elif p["mde"] <= MDE_LINE:
        verdict, ending = "Not supported", "Nothing"
    else:
        verdict, ending = "Can't tell", "Can't tell"
    card = {
        "label_rule": "Historical numbers are in-sample event studies; Nvidia is a single 2026 case.",
        "L01_hook": hook,
        "H1_hook_mcap": nvda_market_cap(),
        "L09_grid": {"code4_total": int(len(ev)), "sex_keyword_cut": int(ev["sex_cut"].sum()),
                     "with_prices_and_date": p["n"], "source": "Gentry et al. DB v7 code 4; out/events_arm_c.csv"},
        "L14_primary": {"n": p["n"], "mean_car_0_1": pct(p["mean_car"]), "ci95_kp": [pct(x) for x in p["ci95_kp"]],
                        "n_fell": p["n"] - p["n_pos"], "p_kp": round(p["p_kp"], 3), "p_rank": round(p["p_corrado"], 3),
                        "p_sign": round(p["p_sign_exact"], 3), "p_bmp": round(p["p_bmp"], 3), "t_kp": round(p["t_kp"], 2),
                        "verdict_locked_rule": verdict,
                        "caveat": "the CI only just excludes 0; the primary BMP-KP test does not reject at 5%",
                        "source": "out/arm_c.json csex_ff5 0_1", "label": "in-sample, FF5 daily, CAR[0,+1]"},
        "L14_sensitivity_clear_affairs": {"n": s["n"], "mean_car_0_1": pct(s["mean_car"]),
                                          "ci95_kp": [pct(x) for x in s["ci95_kp"]], "n_fell": s["n"] - s["n_pos"],
                                          "p_kp": round(s["p_kp"], 3), "source": "out/arm_c_sensitivity.json strict_S",
                                          "label": "sensitivity only (manual audit), not the verdict"},
        "L15_posthoc_twist": posthoc_split(c),
        "viral_replication": {**{k: round(v, 4) if isinstance(v, float) else v for k, v in b["summary"].items()},
                             "label": "his spec, our run (his estimate not visible)", "source": "out/arm_b_viral.json"},
        "L16_nvidia_gap": {"gap_log_pts": round(gap_log * 100, 1), "gap_simple": pct(math.exp(gap_log) - 1, 0), "wording": "X% below its synthetic twin (relative gap), not X points",
                           "window": "Nov 9-11 2026", "source": "out/arm_d_nvidia.json nvda.primary_nov9_11"},
        "L17_placebo": {"rank": d["placebo_rank"]["nvda_rank"], "of": d["placebo_rank"]["of"],
                        "p": round(d["placebo_rank"]["p_value"], 3),
                        "placebo_gap_range_log_pts": [round(min(v["primary_nov9_11"]["gap_pct"] for v in d["placebos"].values()), 1),
                                                      round(max(v["primary_nov9_11"]["gap_pct"] for v in d["placebos"].values()), 1)],
                        "says": "outside all of them" if d["placebo_rank"]["nvda_rank"] == 1 else "inside",
                        "source": "out/arm_d_nvidia.json placebo_rank"},
        "L19_ending": f"{ending} (prereg §3 table, computed)",
        "secondary": secondary(c, b),
    }
    (OUT / "results_card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
