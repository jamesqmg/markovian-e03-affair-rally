"""Daily event study: FF5 or market-model abnormal returns, CARs, and the prereg tests
(Kolari-Pynnonen-adjusted BMP, Corrado rank, generalized sign).

Relative days are counted on the French trading calendar. A window statistic uses an event only
if every day of that window has a return. SCARs carry the forecast-error variance adjustment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
EST = (-250, -30)
WIN = (-20, 20)
MIN_OBS = 120
MAX_DAY0_LAG = pd.Timedelta(days=5)


@dataclass(frozen=True)
class EventResult:
    ar: pd.Series            # abnormal returns by relative trading day in WIN (NaN where no return)
    xw: pd.DataFrame         # window design rows (intercept + factors) by relative day
    xtx_inv: np.ndarray      # (X'X)^-1 from the estimation window
    sigma: float             # residual s.d. in the estimation window
    resid_est: pd.Series     # estimation residuals by calendar date
    k_all: pd.Series         # Corrado standardized ranks over est + event days, by relative day
    day0: pd.Timestamp


def abnormal_returns(ret: pd.Series, factors: pd.DataFrame, event_date: pd.Timestamp,
                     model: str = "ff5") -> EventResult | None:
    """ret: simple daily returns. Day 0 = first French trading day >= event_date (and within
    MAX_DAY0_LAG of it, with a return on day 0). None if fewer than MIN_OBS estimation days."""
    ret = ret.copy()
    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None).normalize()
    cal = factors.sort_index()
    df = cal.join(ret.rename("r"), how="left")
    after = df.index[df.index >= pd.Timestamp(event_date)]
    if len(after) == 0 or after[0] - pd.Timestamp(event_date) > MAX_DAY0_LAG:
        return None
    pos0 = df.index.get_loc(after[0])
    df = df.assign(rel=np.arange(len(df)) - pos0)
    if pd.isna(df.loc[after[0], "r"]):
        return None
    cols = FF5 if model == "ff5" else ["Mkt-RF"]
    est = df[(df.rel >= EST[0]) & (df.rel <= EST[1])].dropna(subset=["r"])
    if len(est) < MIN_OBS:
        return None
    x = np.column_stack([np.ones(len(est)), est[cols].to_numpy()])
    y = (est["r"] - est["RF"]).to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    sigma = float(np.sqrt(resid @ resid / (len(est) - x.shape[1])))
    win = df[(df.rel >= WIN[0]) & (df.rel <= WIN[1])]
    xw = pd.DataFrame(np.column_stack([np.ones(len(win)), win[cols].to_numpy()]), index=win.rel.to_numpy())
    ar = pd.Series((win["r"] - win["RF"]).to_numpy() - xw.to_numpy() @ coef, index=win.rel.to_numpy())
    combined = pd.concat([pd.Series(resid, index=est.rel.to_numpy()), ar.dropna()])
    k = combined.rank().to_numpy() / (len(combined) + 1) - 0.5
    return EventResult(ar=ar, xw=xw, xtx_inv=np.linalg.inv(x.T @ x), sigma=sigma,
                       resid_est=pd.Series(resid, index=est.index),
                       k_all=pd.Series(k, index=combined.index), day0=after[0])


def car(res: EventResult, window: tuple[int, int]) -> float:
    """CAR over the window; NaN unless every window day has a return."""
    w = res.ar.loc[window[0]:window[1]]
    if len(w) != window[1] - window[0] + 1 or w.isna().any():
        return float("nan")
    return float(w.sum())


def scar(res: EventResult, window: tuple[int, int]) -> float:
    """CAR standardized by its forecast-error s.d.: sigma * sqrt(L + s'(X'X)^-1 s)."""
    c = car(res, window)
    s = res.xw.loc[window[0]:window[1]].sum().to_numpy()
    length = window[1] - window[0] + 1
    return c / (res.sigma * np.sqrt(length + s @ res.xtx_inv @ s))


def bmp_t(scars: np.ndarray) -> float:
    n = len(scars)
    return float(scars.mean() / (scars.std(ddof=1) / np.sqrt(n)))


def kp_factor(rbar: float, n: int) -> float:
    """Kolari-Pynnonen t multiplier; rbar is clamped at 0 (the adjustment targets positive correlation)."""
    r = max(rbar, 0.0)
    return float(np.sqrt((1 - r) / (1 + (n - 1) * r)))


def kp_adjust(t: float, rbar: float, n: int) -> float:
    return float(t * kp_factor(rbar, n))


def generalized_sign_z(n_pos: int, n: int, p_hat: float) -> float:
    return float((n_pos - n * p_hat) / np.sqrt(n * p_hat * (1 - p_hat)))


def mde(sd: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean CAR for a two-sided one-sample t test (noncentral t)."""
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)

    def gap(nc: float) -> float:  # solve in noncentrality space (well-conditioned)
        return float(stats.nct.sf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc) - power)

    nc_star = optimize.brentq(gap, 0.0, 8.0)
    return float(nc_star * sd / np.sqrt(n))


def mean_resid_corr(results: list[EventResult], min_overlap: int = 30) -> float:
    """Average pairwise correlation of estimation residuals over overlapping dates (0 if < min_overlap)."""
    rs = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i].resid_est.align(results[j].resid_est, join="inner")
            ok = len(a) >= min_overlap and a.std() > 0 and b.std() > 0
            rs.append(float(np.corrcoef(a, b)[0, 1]) if ok else 0.0)
    return float(np.mean(rs)) if rs else 0.0


def corrado_t(results: list[EventResult], window: tuple[int, int]) -> float:
    """Multi-day Corrado rank statistic (Campbell-Wasley form), reference = est + event days."""
    k = pd.concat([r.k_all for r in results], axis=1)
    kbar = k.mean(axis=1)
    s_k = np.sqrt((kbar ** 2).mean())
    w = kbar.loc[window[0]:window[1]]
    return float(w.sum() / (np.sqrt(len(w)) * s_k))


def p_pos_window(res: EventResult, length: int) -> float:
    """Share of positive length-day residual sums in the estimation window (Cowan 1992)."""
    sums = res.resid_est.rolling(length).sum().dropna()
    return float((sums > 0).mean())


def summarize(results: list[EventResult | None], window: tuple[int, int]) -> dict:
    res = [r for r in results if r is not None and not np.isnan(car(r, window))]
    n = len(res)
    length = window[1] - window[0] + 1
    cars = np.array([car(r, window) for r in res])
    scars = np.array([scar(r, window) for r in res])
    t_bmp = bmp_t(scars)
    rbar = mean_resid_corr(res)
    factor = kp_factor(rbar, n)
    t_kp = t_bmp * factor
    sd = float(cars.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    half_kp = half / factor
    p_hat = float(np.mean([p_pos_window(r, length) for r in res]))
    n_pos = int((cars > 0).sum())
    t_rank = corrado_t(res, window)
    return {
        "n": n, "window": list(window), "mean_car": float(cars.mean()), "median_car": float(np.median(cars)),
        "ci95_kp": [float(cars.mean() - half_kp), float(cars.mean() + half_kp)],
        "ci95_iid": [float(cars.mean() - half), float(cars.mean() + half)], "sd_car": sd,
        "t_bmp": t_bmp, "p_bmp": float(2 * stats.t.sf(abs(t_bmp), n - 1)),
        "rbar": rbar, "t_kp": t_kp, "p_kp": float(2 * stats.t.sf(abs(t_kp), n - 1)),
        "t_corrado": t_rank, "p_corrado": float(2 * stats.norm.sf(abs(t_rank))),
        "p_hat_sign": p_hat, "z_sign": generalized_sign_z(n_pos, n, p_hat),
        "p_sign_exact": float(stats.binomtest(n_pos, n, p_hat).pvalue),
        "n_pos": n_pos, "mde": mde(sd / factor, n),
    }
