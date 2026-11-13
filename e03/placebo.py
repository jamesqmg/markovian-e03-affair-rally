"""Random placebo dates, leave-one-out and Holm (prereg §6-7, erratum E4 §7-9).

Placebo: 1,000 draws; in each, every C-sex firm gets one random trading day from its own history (>= 250 days
before it, not within +-60 days of its true day 0, CAR[0,+1] inside the French data). Statistic: mean FF5 CAR[0,+1]."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from e03 import event_study as es
from e03.inputs import OUT, load

N_DRAWS = 1000
SEED = 12
MIN_HISTORY = 250
EXCLUDE = 60
OBSERVED = -0.0192  # the primary C-sex FF5 CAR[0,+1], fixed in erratum E4 §8 (recomputed below and checked)


def draw_positions(n_obs: int, true_pos: int, rng: np.random.Generator, size: int) -> np.ndarray | None:
    """Random calendar positions in [MIN_HISTORY, n_obs - 2], excluding |pos - true_pos| <= EXCLUDE."""
    allowed = np.arange(MIN_HISTORY, n_obs - 1)
    allowed = allowed[np.abs(allowed - true_pos) > EXCLUDE]
    return rng.choice(allowed, size=size) if len(allowed) else None


def loo_range(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    loo = (v.sum() - v) / (len(v) - 1)
    return float(loo.min()), float(loo.max())


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, monotone, capped at 1."""
    if not all(np.isfinite(v) for v in pvals.values()):
        raise ValueError(f"non-finite p-value in {pvals}")
    keys = sorted(pvals, key=pvals.get)
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def _firm_calendar(ret: pd.Series, ff: pd.DataFrame) -> pd.DatetimeIndex:
    """French trading days on which the firm has a return. The 250-day and +-60 rules are counted on this
    calendar (the firm's "own history", E4 §8); it equals the French calendar wherever the firm has no gaps."""
    idx = pd.DatetimeIndex(ret.dropna().index).tz_localize(None).normalize()
    return ff.index[ff.index.isin(idx)]


def firm_placebo_cars(ret: pd.Series, ff: pd.DataFrame, day0: pd.Timestamp, rng: np.random.Generator,
                      size: int, max_tries: int = 50) -> tuple[np.ndarray, int]:
    """`size` valid placebo CAR[0,+1]s for one firm: a drawn day whose engine result or CAR is invalid is
    redrawn (rejection sampling), so every draw holds every firm. Returns (cars, number of rejections)."""
    cal = _firm_calendar(ret, ff)
    cache: dict[int, float] = {}
    cars, rejected = np.empty(size), 0
    for i in range(size):
        for _ in range(max_tries):
            pos = draw_positions(len(cal), int(cal.get_loc(day0)), rng, 1)
            if pos is None:
                raise ValueError(f"no admissible placebo day for event on {day0.date()}")
            p = int(pos[0])
            if p not in cache:
                r = es.abnormal_returns(ret, ff, cal[p], model="ff5")
                cache[p] = es.car(r, (0, 1)) if r is not None else float("nan")
            if np.isfinite(cache[p]):
                cars[i] = cache[p]
                break
            rejected += 1
        else:
            raise RuntimeError(f"{max_tries} invalid placebo days in a row for event on {day0.date()}")
    return cars, rejected


def placebo_means(x) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(SEED)
    per_firm, rejected = [], 0
    for t, res in zip(x.csex["ticker"], x.csex["res"]):
        cars, rej = firm_placebo_cars(x.rets_csex[t].dropna(), x.ff, res.day0, rng, N_DRAWS)
        per_firm.append(cars)
        rejected += rej
    means = np.vstack(per_firm).mean(axis=0)
    assert np.isfinite(means).all()
    return means, rejected


def main() -> None:
    x = load()
    observed = float(np.mean([es.car(r, (0, 1)) for r in x.csex["res"]]))
    assert abs(observed - OBSERVED) < 5e-5, observed
    means, rejected = placebo_means(x)
    k_le, k_abs = int((means <= observed).sum()), int((np.abs(means) >= abs(observed)).sum())
    b = json.loads((OUT / "arm_b_viral.json").read_text())
    deltas = [c["delta"] for c in b["cases"] if c["delta"] is not None]
    reel = [c["delta"] for c in b["cases"] if c["delta"] is not None and c["group"] == "reel"]
    out = {
        "label": "in-sample, historical; secondary (prereg §6, erratum E4 §7-8)",
        "placebo_dates": {"draws": N_DRAWS, "seed": SEED, "observed": observed, "firms_per_draw": len(x.csex),
                          "rejected_redrawn": rejected,
                          "placebo_mean": float(means.mean()), "placebo_sd": float(means.std(ddof=1)),
                          "share_one_sided_le": k_le / N_DRAWS, "share_two_sided_abs": k_abs / N_DRAWS,
                          "p_one_sided_le": (k_le + 1) / (N_DRAWS + 1), "p_two_sided_abs": (k_abs + 1) / (N_DRAWS + 1),
                          "notes": "draws independent across firms (no calendar clustering in the null); "
                                   "a placebo estimation window may contain the firm's true event",
                          "q05_q95": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))]},
        "loo_arm_b": dict(zip(["min", "max"], loo_range(deltas)), n=len(deltas)),
        "loo_arm_a": dict(zip(["min", "max"], loo_range(reel)), n=len(reel)),
    }
    (OUT / "secondary_placebo_loo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
