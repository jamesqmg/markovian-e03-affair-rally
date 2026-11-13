"""Download and parse every raw input. Raw files go to data/raw/ (not redistributed);
each download's sha256 is appended to data/manifest.json so the public repo can ship checksums."""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"
UA = {"User-Agent": "e03-research-script/0.1 (academic replication)"}
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
ZENODO_RECORD = "23328926"  # Gentry et al. CEO Dismissal DB, latest version at prereg lock


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(r.content)
    part.replace(dest)  # atomic: a failed download never leaves a cached partial file
    _log(url, dest)
    return dest


def _log(url: str, dest: Path) -> None:
    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    entries = [e for e in entries if e["file"] != dest.name] + [{
        "file": dest.name, "url": url,
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }]
    MANIFEST.write_text(json.dumps(entries, indent=1))


def parse_french_csv(text: str) -> pd.DataFrame:
    """First data block of a French CSV (monthly YYYYMM or daily YYYYMMDD), in decimals."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith(",") and ("Mkt-RF" in ln or "Mom" in ln)), None)
    if start is None:
        raise ValueError("French CSV: no factor header line found")
    header = [h.strip() for h in lines[start].split(",")]
    rows = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts[0].isdigit() or len(parts[0]) not in (6, 8):
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    fmt = "%Y%m%d" if len(rows[0][0]) == 8 else "%Y%m"
    df["date"] = pd.to_datetime(df["date"], format=fmt)
    if fmt == "%Y%m":
        df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df = df.set_index("date").astype(float).replace([-99.99, -999.0], float("nan")) / 100.0
    df.columns = [c.strip() for c in df.columns]
    return df


def french(name: str) -> pd.DataFrame:
    """name e.g. 'F-F_Research_Data_5_Factors_2x3_daily' (without _CSV.zip)."""
    path = _download(f"{FRENCH}{name}_CSV.zip", RAW / f"{name}_CSV.zip")
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_french_csv(text)


GENTRY_FILE = "CEO_Dismissal_Dataset_Excel_26Nov25.xlsx"


def gentry() -> pd.DataFrame:
    cached = RAW / GENTRY_FILE
    if cached.exists():
        return pd.read_excel(cached)
    meta = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", headers=UA, timeout=60)
    meta.raise_for_status()
    files = meta.json()["files"]
    xl = next(f for f in files if f["key"].replace(" ", "_") == GENTRY_FILE)
    path = _download(xl["links"]["self"], RAW / xl["key"].replace(" ", "_"))
    return pd.read_excel(path)


def prices(tickers: list[str], start: str, end: str, tag: str, field: str = "Close") -> pd.DataFrame:
    """Daily split/dividend-adjusted closes (or another Yahoo field, e.g. "Volume"), cached per tag."""
    import yfinance as yf

    req = f"{sorted(tickers)}|{start}|{end}" + ("" if field == "Close" else f"|{field}")  # Close keys unchanged
    key = hashlib.sha256(req.encode()).hexdigest()[:10]
    cache = RAW / f"prices_{tag}_{key}.parquet"  # keyed by request, so a changed list never reuses stale data
    if cache.exists():
        return pd.read_parquet(cache)
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)[field]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    for t in [c for c in df.columns if df[c].isna().all()]:  # batch downloads drop names at random
        for _ in range(3):
            one = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)[field]
            one = one.iloc[:, 0] if isinstance(one, pd.DataFrame) else one
            if one.notna().any():
                df[t] = one
                break
            time.sleep(2)
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        print(f"WARNING yahoo returned no data for {empty} (end date is exclusive)")
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    _log(f"yfinance:{tag}:{field}:{start}:{end}:{len(tickers)} tickers", cache)
    return df


def sec_cik_tickers() -> pd.DataFrame:
    path = _download("https://www.sec.gov/files/company_tickers.json", RAW / "sec_company_tickers.json")
    d = json.loads(path.read_text())
    return pd.DataFrame(d.values()).rename(columns={"cik_str": "cik"})


if __name__ == "__main__":
    for n in ["F-F_Research_Data_5_Factors_2x3_daily", "F-F_Research_Data_5_Factors_2x3",
              "F-F_Momentum_Factor_daily", "F-F_Momentum_Factor",
              "Developed_5_Factors", "Europe_5_Factors", "Asia_Pacific_ex_Japan_5_Factors"]:
        f = french(n)
        print(n, f.index.min().date(), f.index.max().date(), list(f.columns))
    g = gentry()
    print("gentry", g.shape, g["departure_code"].value_counts().sort_index().to_dict())
    print("sec", sec_cik_tickers().shape)
