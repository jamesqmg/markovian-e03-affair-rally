import numpy as np
import pandas as pd
import pytest

from e03 import event_study as es


def _panel(n_days: int = 400, seed: int = 0, shock: float = 0.0, event_pos: int = 300):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    f = pd.DataFrame(rng.normal(0, 0.01, (n_days, 5)), index=dates,
                     columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    f["RF"] = 0.0001
    beta = np.array([1.2, 0.3, -0.2, 0.1, 0.05])
    r = f[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].to_numpy() @ beta + f["RF"] + rng.normal(0, 0.002, n_days)
    r = pd.Series(r, index=dates)
    r.iloc[event_pos] += shock
    return r, f, dates[event_pos]


def test_ar_recovers_shock_on_day0():
    r, f, d0 = _panel(shock=0.05)
    res = es.abnormal_returns(r, f, d0, model="ff5")
    assert res is not None
    assert res.ar.loc[0] == pytest.approx(0.05, abs=0.01)
    assert abs(res.ar.loc[5]) < 0.01


def test_car_sums_window():
    r, f, d0 = _panel(shock=0.05)
    res = es.abnormal_returns(r, f, d0, model="ff5")
    assert es.car(res, (0, 1)) == pytest.approx(res.ar.loc[0] + res.ar.loc[1])


def test_insufficient_history_returns_none():
    r, f, d0 = _panel(n_days=200, event_pos=100)
    assert es.abnormal_returns(r, f, d0, model="ff5") is None


def test_market_model_runs():
    r, f, d0 = _panel(shock=0.05)
    res = es.abnormal_returns(r, f, d0, model="mm")
    assert res.ar.loc[0] == pytest.approx(0.05, abs=0.015)


def test_bmp_formula():
    scar = np.array([1.0, 2.0, 3.0, 2.0])
    t = es.bmp_t(scar)
    assert t == pytest.approx(scar.mean() / (scar.std(ddof=1) / np.sqrt(4)))


def test_kp_adjustment():
    assert es.kp_adjust(2.0, rbar=0.0, n=10) == pytest.approx(2.0)
    assert es.kp_adjust(2.0, rbar=0.1, n=10) == pytest.approx(2.0 * np.sqrt(0.9 / 1.9))


def test_generalized_sign():
    z = es.generalized_sign_z(n_pos=8, n=10, p_hat=0.5)
    assert z == pytest.approx((8 - 5) / np.sqrt(10 * 0.25))


def test_mde():
    assert es.mde(sd=0.02, n=25) == pytest.approx((1.959964 + 0.841621) * 0.02 / 5, rel=0.06)


def test_end_to_end_detects_common_shock():
    results = []
    for s in range(30):
        r, f, d0 = _panel(seed=s, shock=-0.03)
        results.append(es.abnormal_returns(r, f, d0, model="ff5"))
    summ = es.summarize(results, (0, 1))
    assert summ["mean_car"] < -0.02
    assert summ["p_kp"] < 0.01
    assert summ["n"] == 30


def test_kp_clamps_negative_rbar():
    assert es.kp_adjust(2.0, rbar=-0.2, n=10) == pytest.approx(2.0)


def test_mde_uses_t_not_normal():
    # small n: noncentral-t MDE must exceed the normal approximation
    normal = (1.959964 + 0.841621) * 0.02 / np.sqrt(10)
    assert es.mde(sd=0.02, n=10) > normal * 1.08


def test_truncated_window_is_dropped_for_that_window():
    r, f, d0 = _panel(shock=0.05)
    pos = r.index.get_loc(d0)
    r.iloc[pos + 3] = np.nan  # missing return on day +3
    res = es.abnormal_returns(r, f, d0, model="ff5")
    assert not np.isnan(es.car(res, (0, 1)))
    assert np.isnan(es.car(res, (-1, 5)))


def test_weekend_event_maps_to_next_trading_day():
    r, f, d0 = _panel()
    saturday = pd.Timestamp("2016-02-06")
    res = es.abnormal_returns(r, f, saturday, model="ff5")
    assert res.day0 == pd.Timestamp("2016-02-08")


def test_missing_stock_rows_do_not_shift_relative_days():
    r, f, d0 = _panel(shock=0.05)
    r = r.drop(r.index[100:110])  # gap well inside the estimation window
    res = es.abnormal_returns(r, f, d0, model="ff5")
    assert res.day0 == d0
    assert res.ar.loc[0] == pytest.approx(0.05, abs=0.01)


def test_null_size_is_near_nominal():
    rejections = 0
    sims = 60
    for s in range(sims):
        results = [es.abnormal_returns(*_panel(seed=1000 * s + k), model="ff5") for k in range(20)]
        rejections += es.summarize(results, (0, 1))["p_kp"] < 0.05
    assert rejections / sims < 0.15
