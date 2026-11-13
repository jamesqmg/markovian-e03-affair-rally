"""Event tables for every arm. Dates only: no returns are touched here.

Arm C event date (prereg §2 + erratum E1): date_departure_announced if present, else the earliest of
(a) the filing date of an 8-K listed in the dataset's `eight_ks` column with Item 5.02 (or Item 5 pre-2004)
filed within [left_ceo_office - 7d, left_ceo_office + 7d] (old Item 5 accepted before 2004-08-23), and (b) the earliest dated news citation in the
dataset's own `sources` column in the same window. No date found -> excluded and counted.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

from e03.fetch import RAW, UA, gentry, sec_cik_tickers

KEYWORDS = ["affair", "relationship", "romantic", "consensual", "sexual", "harassment",
            "misconduct", "personal conduct"]
WINDOW_BEFORE, WINDOW_AFTER = pd.Timedelta(days=7), pd.Timedelta(days=7)
FETCH_FAILURES: list[str] = []
ITEM5_RENUMBERED = pd.Timestamp("2004-08-23")  # 8-K Item 5.02 exists only from this date
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# Arms A and B: names/dates as listed by the claim / by the viral reel. Tickers are Yahoo symbols.
ARM_B = [
    ("AMZN", "Amazon / Bezos", "2019-01-09", "reel"), ("MSFT", "Microsoft / Gates", "2021-05-16", "reel"),
    ("000660.KS", "SK Hynix / Chey", "2015-12-29", "reel"), ("GOOGL", "Alphabet / Brin", "2013-08-28", "reel"),
    ("BA", "Boeing / Stonecipher", "2005-03-07", "added"), ("HPQ", "HP / Hurd", "2010-08-06", "added"),
    ("BBY", "Best Buy / Dunn", "2012-04-10", "added"), ("LMT", "Lockheed / Kubasik", "2012-11-09", "added"),
    ("BKNG", "Priceline / Huston", "2016-04-28", "added"), ("INTC", "Intel / Krzanich", "2018-06-21", "added"),
    ("TXN", "TI / Crutcher", "2018-07-17", "added"), ("MCD", "McDonald's / Easterbrook", "2019-11-03", "added"),
    ("WYNN", "Wynn / Wynn", "2018-01-26", "added"), ("BP", "BP / Browne", "2007-05-01", "added"),
    ("BP", "BP / Looney", "2023-09-12", "added"),
]
NOT_CEO_AT_EVENT = {"Microsoft / Gates", "Alphabet / Brin"}  # fact-check.md #8


def arm_b() -> pd.DataFrame:
    df = pd.DataFrame(ARM_B, columns=["ticker", "case", "event_date", "group"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["not_ceo_at_event"] = df["case"].isin(NOT_CEO_AT_EVENT)
    return df


def arm_a() -> pd.DataFrame:
    return arm_b().query("group == 'reel'").reset_index(drop=True)


def cik_from_8ks(s: object) -> int | None:
    m = re.search(r"edgar/data/(\d+)/", str(s))
    return int(m.group(1)) if m else None


def source_dates(s: object) -> list[pd.Timestamp]:
    """Dates written in citation form '(2010, Aug 06)' or URL form '/2018/06/21/'."""
    text = str(s)
    out = []
    pats = [r"\((\d{4}),? ([A-Za-z]{3})[a-z]*\.? (\d{1,2})(?:st|nd|rd|th)?\)",
            r"\b(\d{4}) ([A-Za-z]{3}) (\d{2})\b"]
    for pat in pats:
        for y, mon, d in re.findall(pat, text):
            if mon.lower() in MONTHS:
                try:
                    out.append(pd.Timestamp(int(y), MONTHS[mon.lower()], int(d)))
                except ValueError:
                    continue
    for y, mo, d in re.findall(r"/((?:19|20)\d{2})/(\d{2})/(\d{2})/", text):
        try:
            out.append(pd.Timestamp(int(y), int(mo), int(d)))
        except ValueError:
            continue
    return out


def _index_url(doc_url: str) -> str | None:
    old = re.search(r"edgar/data/(\d+)/(\d{10})-(\d{2})-(\d{6})\.txt", doc_url)
    if old:  # pre-2001 style: .../data/CIK/0000009892-94-000002.txt
        acc = "".join(old.groups()[1:])
        return f"https://www.sec.gov/Archives/edgar/data/{old.group(1)}/{acc}/{'-'.join(old.groups()[1:])}-index.htm"
    m = re.search(r"edgar/data/(\d+)/(\d{18})/", doc_url)
    if m is None:
        return None
    acc = m.group(2)
    dashed = f"{acc[:10]}-{acc[10:12]}-{acc[12:]}"
    return f"https://www.sec.gov/Archives/edgar/data/{m.group(1)}/{acc}/{dashed}-index.htm"


def eightk_meta(doc_url: str, cache: dict) -> tuple[pd.Timestamp | None, str]:
    """(filing date, items text) for one 8-K, from its EDGAR index page."""
    idx = _index_url(doc_url)
    if idx is None:
        return None, ""
    if idx not in cache:
        html = None
        for attempt in range(4):
            try:
                r = requests.get(idx, headers=UA, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            time.sleep(0.15)  # SEC fair-access limit is 10 req/s
            if r.ok:
                html = r.text
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            break  # 404 etc.: a real absence
        if html is None:
            FETCH_FAILURES.append(idx)  # never cached, so a re-run retries it
            return None, ""
        cache[idx] = html
    html = cache[idx]
    fd = re.search(r"Filing Date</div>\s*<div class=\"info\">(\d{4}-\d{2}-\d{2})", html)
    items = " ".join(re.findall(r"Item\s+\d+\.\d+|Item\s+\d+(?!\.)", html))
    return (pd.Timestamp(fd.group(1)) if fd else None), items


def _fallback_date(row: pd.Series, cache: dict) -> tuple[pd.Timestamp | None, str]:
    left = row["left_ceo_office"]
    lo, hi = left - WINDOW_BEFORE, left + WINDOW_AFTER
    cands: list[tuple[pd.Timestamp, str]] = []
    for url in str(row["eight_ks"]).split():
        if "sec.gov" not in url:
            continue
        fdate, items = eightk_meta(url, cache)
        is_departure = "5.02" in items or (fdate is not None and fdate < ITEM5_RENUMBERED and "Item 5" in items)
        if fdate is not None and is_departure and lo <= fdate <= hi:
            cands.append((fdate, "8-K"))
    cands += [(d, "news") for d in source_dates(row["sources"]) if lo <= d <= hi]
    return min(cands) if cands else (None, "none")


def arm_c() -> pd.DataFrame:
    g = gentry()
    c4 = g[g["departure_code"] == 4].copy()
    pat = re.compile("|".join(map(re.escape, KEYWORDS)), re.I)
    c4["sex_cut"] = c4["notes"].fillna("").str.contains(pat)
    c4["cik"] = c4["eight_ks"].map(cik_from_8ks).astype("Int64")
    cache_path = RAW / "edgar_index_cache.parquet"
    cache = (pd.read_parquet(cache_path)["html"].to_dict() if cache_path.exists() else {})
    dates, how = [], []
    for _, row in c4.iterrows():
        if pd.notna(row["date_departure_announced"]):
            dates.append(row["date_departure_announced"])
            how.append("dataset")
        else:
            d, h = _fallback_date(row, cache)
            dates.append(d)
            how.append(h)
    pd.DataFrame({"html": pd.Series(cache, dtype=object)}).to_parquet(cache_path)
    if FETCH_FAILURES:
        raise RuntimeError(f"{len(FETCH_FAILURES)} EDGAR fetches failed; re-run (failures are not cached)")
    c4["event_date"] = pd.to_datetime(dates)
    c4["date_source"] = how
    tick = sec_cik_tickers().drop_duplicates("cik").set_index("cik")["ticker"]
    c4["ticker"] = c4["cik"].map(tick)
    keep = ["dismissal_dataset_id", "coname", "exec_fullname", "fyear", "gvkey", "cik", "ticker",
            "left_ceo_office", "date_departure_announced", "event_date", "date_source", "sex_cut", "notes"]
    return c4[keep].reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "out"
    out.mkdir(exist_ok=True)
    c = arm_c()
    c.to_csv(out / "events_arm_c.csv", index=False)
    arm_b().to_csv(out / "events_arm_b.csv", index=False)
    for name, d in [("C-all", c), ("C-sex", c[c.sex_cut])]:
        print(name, "N", len(d), "| date source", d["date_source"].value_counts().to_dict(),
              "| ticker found", d["ticker"].notna().sum())
