"""Daily simple returns per event firm. Yahoo first; Tiingo adjClose for firms Yahoo lacks
(prereg erratum E1 §3). The C-sex ticker map is hand-checked (name + listing dates) and frozen here."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from e03.fetch import RAW, _log, prices

# dismissal_dataset_id -> (ticker, source). None = no price source (excluded and counted).
CSEX_TICKERS: dict[int, tuple[str, str] | None] = {
    1597: ("GRA", "tiingo"), 3064: None, 498: ("BA", "yahoo"), 7990: ("WDR", "tiingo"),
    7411: ("CRR", "tiingo"), 3407: ("SYK", "yahoo"), 3360: None, 4621: ("VIAC", "tiingo"),
    443: ("BBY", "yahoo"), 6209: None, 1713: ("HPQ", "yahoo"), 2103: ("LRCX", "yahoo"),
    7510: ("GES", "tiingo"), 8761: ("WYNN", "yahoo"), 1982: ("JCI", "yahoo"), 4378: None,
    7665: ("IGTE", "tiingo"), 3563: ("TXN", "yahoo"), 2691: ("FE", "yahoo"), 9379: ("CBOE", "yahoo"),
    8820: ("ADEA", "tiingo"), 8205: ("BKNG", "yahoo"), 8257: ("FFIV", "yahoo"), 1857: ("INTC", "yahoo"),
    2301: ("MCD", "yahoo"), 5475: ("TBI", "yahoo"), 8370: ("WWE", "tiingo"), 8069: ("RH", "yahoo"),
    6325: ("NGVT", "yahoo"), 6628: ("BKS", "tiingo"), 8012: ("NCLH", "yahoo"), 9072: ("SNCR", "tiingo"),
    8540: ("TPR", "yahoo"), 10500: ("LULU", "yahoo"),
}


def _tiingo_key() -> str:
    key = os.environ.get("TIINGO_API_KEY")
    if key:
        return key
    raise RuntimeError("set TIINGO_API_KEY in the environment (free key at tiingo.com)")


def tiingo_close(ticker: str, start: str, end: str) -> pd.Series:
    cache = RAW / f"tiingo_{ticker}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)[ticker]
    r = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                     params={"startDate": start, "endDate": end},
                     headers={"Authorization": f"Token {_tiingo_key()}"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    s = pd.Series(df["adjClose"].to_numpy(), index=pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize(),
                  name=ticker)
    s.to_frame().to_parquet(cache)
    _log(f"tiingo:{ticker}:{start}:{end}", cache)
    time.sleep(0.5)
    return s


def tiingo_volume(ticker: str, start: str, end: str) -> pd.Series:
    """Split-adjusted daily volume (Tiingo adjVolume) for firms priced from Tiingo (erratum E4 §6)."""
    cache = RAW / f"tiingo_vol_{ticker}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)[ticker]
    r = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                     params={"startDate": start, "endDate": end},
                     headers={"Authorization": f"Token {_tiingo_key()}"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    s = pd.Series(df["adjVolume"].to_numpy(), index=pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize(),
                  name=ticker)
    s.to_frame().to_parquet(cache)
    _log(f"tiingo:adjVolume:{ticker}:{start}:{end}", cache)
    time.sleep(0.5)
    return s


def volumes_for(pairs: list[tuple[str, str]], start: str, end: str, tag: str) -> pd.DataFrame:
    """Daily volume, one column per ticker: Yahoo for Yahoo-priced firms, Tiingo adjVolume otherwise."""
    yahoo = sorted({t for t, src in pairs if src == "yahoo"})
    frames = [prices(yahoo, start, end, tag, field="Volume")] if yahoo else []
    frames += [tiingo_volume(t, start, end).to_frame() for t, src in sorted(set(pairs)) if src == "tiingo"]
    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out


def returns_for(pairs: list[tuple[str, str]], start: str, end: str, tag: str) -> pd.DataFrame:
    """Daily simple returns, one column per ticker."""
    return closes_for(pairs, start, end, tag).pct_change(fill_method=None)


def closes_for(pairs: list[tuple[str, str]], start: str, end: str, tag: str) -> pd.DataFrame:
    """Daily adjusted closes, one column per ticker (Yahoo, or Tiingo adjClose per the pair's source)."""
    yahoo = sorted({t for t, src in pairs if src == "yahoo"})
    frames = []
    if yahoo:
        frames.append(prices(yahoo, start, end, tag))
    for t, src in pairs:
        if src == "tiingo":
            frames.append(tiingo_close(t, start, end).to_frame())
    return pd.concat(frames, axis=1).sort_index()
