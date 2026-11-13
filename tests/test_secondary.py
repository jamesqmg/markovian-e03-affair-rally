"""Tests for the prereg §6 secondary analyses (erratum E4)."""
import numpy as np
import pandas as pd
import pytest

from e03 import calendar_time as ct
from e03 import garch as gv
from e03 import placebo as pl


# ---- calendar-time portfolios (E4 §4) ----

def test_monthly_from_closes_spans_missing_price_days():
    # a NaN close mid-month must not turn the gap into a 0% return (reviewer H1)
    idx = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-28"])
    c = pd.DataFrame({"A": [100.0, 110.0, np.nan, 121.0, 146.41]}, index=idx)
    m = ct.monthly_from_closes(c)
    assert m.loc["2020-02-29", "A"] == pytest.approx(0.4641)


def test_monthly_from_closes_nan_when_month_has_no_close():
    idx = pd.to_datetime(["2020-01-31", "2020-03-31"])
    c = pd.DataFrame({"A": [100.0, 120.0]}, index=idx)
    m = ct.monthly_from_closes(c)
    assert np.isnan(m.loc["2020-02-29", "A"]) and np.isnan(m.loc["2020-03-31", "A"])


def test_portfolio_holds_months_m_plus_1_to_m_plus_h():
    months = pd.date_range("2020-01-31", periods=8, freq="ME")
    monthly = pd.DataFrame({"A": np.arange(8) / 100, "B": 0.5}, index=months)
    events = [("A", pd.Timestamp("2020-01-15")), ("B", pd.Timestamp("2020-03-10"))]
    port, n = ct.portfolio(monthly, events, h=2)
    # A held Feb, Mar; B held Apr, May
    assert list(port.index) == list(months[1:5])
    assert port.loc["2020-02-29"] == pytest.approx(0.01)
    assert port.loc["2020-04-30"] == pytest.approx(0.5)
    assert n.max() == 1


def test_portfolio_equal_weights_overlapping_firms():
    months = pd.date_range("2020-01-31", periods=4, freq="ME")
    monthly = pd.DataFrame({"A": 0.02, "B": 0.04}, index=months)
    events = [("A", pd.Timestamp("2020-01-10")), ("B", pd.Timestamp("2020-01-20"))]
    port, n = ct.portfolio(monthly, events, h=2)
    assert port.loc["2020-02-29"] == pytest.approx(0.03)
    assert n.loc["2020-02-29"] == 2


def test_alpha_regression_recovers_known_alpha():
    rng = np.random.default_rng(1)
    months = pd.date_range("1995-01-31", periods=300, freq="ME")
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    f = pd.DataFrame(rng.normal(0, 0.04, (300, 6)), index=months, columns=cols)
    f["RF"] = 0.002
    port = 0.005 + f["RF"] + f[cols] @ np.array([1.0, 0.2, 0.1, 0, 0, -0.1]) + rng.normal(0, 0.001, 300)
    out = ct.alpha(port, f)
    assert out["alpha_monthly"] == pytest.approx(0.005, abs=0.0005)
    assert out["months"] == 300


# ---- GARCH SCAR + abnormal volume (E4 §5-6) ----

def test_garch_scar_standardizes_by_forecast_variance():
    rng = np.random.default_rng(2)
    resid = pd.Series(rng.standard_t(6, 250) * 0.015, index=np.arange(-250, 0))
    out = gv.garch_scar(resid, car01=0.03)
    assert out is not None
    h0, h1 = out["h0"], out["h1"]
    assert h0 > 0 and h1 > 0
    assert out["scar"] == pytest.approx(0.03 / np.sqrt(h0 + h1))


def test_abnormal_volume_flags_spike():
    rel = np.arange(-250, 21)
    vol = pd.Series(1e6, index=rel, dtype=float) * np.exp(np.random.default_rng(3).normal(0, 0.2, len(rel)))
    vol.loc[0] = 5e6
    sav = gv.abnormal_volume(vol)
    assert sav[0] > 3
    assert abs(sav[1]) < 3


def test_abnormal_volume_none_with_short_history():
    vol = pd.Series(1e6, index=np.arange(-50, 5), dtype=float)
    assert gv.abnormal_volume(vol) is None


# ---- placebo dates + leave-one-out (E4 §7-8) ----

def test_placebo_positions_respect_constraints():
    rng = np.random.default_rng(12)
    pos = pl.draw_positions(n_obs=1000, true_pos=600, rng=rng, size=500)
    assert pos.min() >= 250
    assert pos.max() <= 1000 - 2
    assert not np.any(np.abs(pos - 600) <= 60)


def test_placebo_positions_none_when_no_room():
    assert pl.draw_positions(n_obs=300, true_pos=280, rng=np.random.default_rng(0), size=10) is None


def test_leave_one_out_range():
    lo, hi = pl.loo_range([1.0, 2.0, 3.0, 10.0])
    assert lo == pytest.approx(2.0)   # drop 10
    assert hi == pytest.approx(5.0)   # drop 1


def test_holm_adjustment():
    adj = pl.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotone


# ---- Bayesian hierarchical CAR model (E4 §3) ----

@pytest.mark.slow
def test_bayes_recovers_group_means_and_shrinks():
    from e03 import bayes
    rng = np.random.default_rng(4)
    n = 40
    group = np.array([0] * 20 + [1] * 20)
    theta = np.where(group == 0, -0.03, 0.0) + rng.normal(0, 0.01, n)
    se = np.full(n, 0.02)
    car = theta + rng.normal(0, se)
    s = bayes.summarize(bayes.fit(car, se, group, draws=500, chains=2), car)
    assert s["mu_group"][0]["mean"] < -0.015
    assert abs(s["mu_group"][1]["mean"]) < 0.015
    assert s["p_mu_group_gt0"][0] < 0.05
    # partial pooling: shrunk estimates are less dispersed than the raw CARs
    assert np.std(s["theta_mean"]) < np.std(car)
    assert s["max_rhat"] < 1.05  # short test chains (2 x 500); the real run uses 4 x 2000 and the 1.01 rule


def test_arm_table_excludes_every_sex_cut_id_from_c_other():
    from e03 import bayes
    csex = pd.DataFrame({"id": [1, 2], "res": [None, None]})
    call = pd.DataFrame({"id": [2, 3, 4], "res": [None, None, None]})  # 3 is sex_cut but unusable in C-sex
    t = bayes.arm_table(csex, call, frozenset({1, 2, 3}))
    assert sorted(t.loc[t.group == 1, "id"]) == [4]
    assert sorted(t.loc[t.group == 0, "id"]) == [1, 2]


def test_firm_placebo_cars_redraws_invalid_days_and_matches_engine():
    from e03 import event_study as es
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2010-01-01", periods=900)
    f = pd.DataFrame(rng.normal(0, 0.01, (900, 5)), index=dates, columns=es.FF5)
    f["RF"] = 0.0
    r = pd.Series(f[es.FF5].to_numpy() @ np.array([1.0, 0.2, 0, 0, 0]) + rng.normal(0, 0.01, 900), index=dates)
    r.iloc[400:420] = np.nan  # a gap: placebo days next to it give an invalid CAR and must be redrawn
    true_day0 = dates[700]
    cars, _ = pl.firm_placebo_cars(r, f, true_day0, np.random.default_rng(1), size=200)
    assert np.isfinite(cars).all() and len(cars) == 200
    one, _ = pl.firm_placebo_cars(r, f, true_day0, np.random.default_rng(2), size=1)
    cal = pl._firm_calendar(r, f)
    # the value equals the engine's CAR on some admissible firm-calendar day
    valid = [es.car(es.abnormal_returns(r, f, cal[p], "ff5"), (0, 1)) for p in range(250, len(cal) - 1)
             if abs(p - cal.get_loc(true_day0)) > 60 and es.abnormal_returns(r, f, cal[p], "ff5") is not None]
    assert any(np.isclose(one[0], v) for v in valid if np.isfinite(v))


def test_holm_rejects_nan():
    with pytest.raises(ValueError):
        pl.holm({"a": 0.01, "b": float("nan")})


def test_garch_scar_forecast_matches_simulated_variance():
    rng = np.random.default_rng(7)
    sd = 0.015
    resid = pd.Series(rng.standard_t(8, 250) * sd / np.sqrt(8 / 6), index=np.arange(-250, 0))
    out = gv.garch_scar(resid, car01=0.0)
    assert out is not None
    assert np.sqrt(out["h0"]) == pytest.approx(sd, rel=0.35)  # units: decimal returns, not percent


def test_garch_scar_drops_when_day_minus_1_missing():
    resid = pd.Series(np.random.default_rng(8).normal(0, 0.01, 248), index=np.arange(-250, -2))
    assert gv.garch_scar(resid, car01=0.01) is None


def test_ff5_resid_ends_at_day_minus_1_on_french_calendar():
    from e03 import event_study as es
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2015-01-01", periods=400)
    f = pd.DataFrame(rng.normal(0, 0.01, (400, 5)), index=dates, columns=es.FF5)
    f["RF"] = 0.0
    r = pd.Series(f[es.FF5].to_numpy() @ np.ones(5) + rng.normal(0, 0.002, 400), index=dates)
    res = gv.ff5_resid(r, f, dates[300])
    assert res.index.min() == -250 and res.index.max() == -1


def test_rel_volume_puts_spike_on_day_0():
    cal = pd.bdate_range("2015-01-01", periods=400)
    vol = pd.Series(1.0, index=cal)
    vol.iloc[310] = 99.0
    rel = gv._rel_volume(vol, cal, cal[310])
    assert rel.loc[0] == 99.0 and rel.index.min() == -250 and rel.index.max() == 1


# ---- scene-data dumps (no new analysis: more output from the E4 modules) ----

def test_rel_volume_can_extend_past_day_1():
    cal = pd.bdate_range("2015-01-01", periods=400)
    vol = pd.Series(1.0, index=cal)
    rel = gv._rel_volume(vol, cal, cal[310], last=5)
    assert rel.index.max() == 5 and rel.index.min() == -250
