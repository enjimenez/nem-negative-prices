"""Download AEMO's day-ahead predispatch forecast (the "13:00 run") for every day.

Source: AEMO's monthly MMSDM archive, PREDISP_ALL_DATA folder (every predispatch run).
For each target day D we keep only the run labelled 13:00 on D-1 (executed at 12:30,
right after bids for day D close), and only its 48 half-hours of day D.

Output: data/interim/predispatch_price.parquet and data/interim/predispatch_regionsum.parquet
- Regions in config.REGIONS only
- DATETIME = END of the half-hour, NEM time (UTC+10, no daylight saving), no timezone
"""

from pathlib import Path
from urllib.parse import quote
import zipfile

from loguru import logger
import pandas as pd
import requests
import typer

from negprice.config import END, INTERIM_DATA_DIR, RAW_DATA_DIR, REGIONS, START

app = typer.Typer()

ARCHIVE_URL = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year}/"
    "MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/"
)
CACHE = RAW_DATA_DIR / "predispatch_cache"
# nemweb rejects the default `requests` user agent; NEMOSIS sends a browser one too
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) nem-negative-prices research"}

# PREDISPATCHSEQNO = trading date (YYYYMMDD) + run number (01-48, first run 04:30).
# Run 18 = 04:30 + 17 x 30 min = 13:00.
RUN_NUMBER = "18"

# Months where AEMO's archive has no all-runs PRICE/REGIONSUM files (checked 24 Oct 2026).
# Only the "_D" (latest forecast only) files exist, which would leak, so we skip them.
KNOWN_GAPS = {(2022, 10)}

TABLES = {
    "PRICE": ["PREDISPATCHSEQNO", "REGIONID", "INTERVENTION", "DATETIME", "RRP"],
    "REGIONSUM": [
        "PREDISPATCHSEQNO",
        "REGIONID",
        "INTERVENTION",
        "DATETIME",
        "TOTALDEMAND",
        "DEMAND_AND_NONSCHEDGEN",
        "UIGF",
        "SS_SOLAR_UIGF",
        "SS_WIND_UIGF",
        "AVAILABLEGENERATION",
        "NETINTERCHANGE",
    ],
}


def file_names(table: str, year: int, month: int) -> list[str]:
    """Archive file names to try. AEMO changed the naming scheme in August 2024."""
    if (year, month) >= (2024, 8):
        return [
            f"PUBLIC_ARCHIVE#PREDISPATCH{table}#ALL#FILE{n:02d}#{year}{month:02d}010000.zip"
            for n in range(1, 10)
        ]
    return [f"PUBLIC_DVD_PREDISPATCH{table}_{year}{month:02d}010000.zip"]


def download(table: str, year: int, month: int) -> list[Path]:
    """Download one month's zip file(s) into the cache; skip files already there."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in file_names(table, year, month):
        path = CACHE / name
        if not path.exists():
            # '#' must be encoded, otherwise it is read as the start of a URL fragment
            url = ARCHIVE_URL.format(year=year, month=month) + quote(name)
            r = requests.get(url, headers=HEADERS, timeout=600)
            if r.status_code == 404:
                break  # no more FILE02, FILE03... for this month
            r.raise_for_status()
            path.write_bytes(r.content)
            logger.info(f"Downloaded {name} ({len(r.content) / 1e6:.0f} MB)")
        paths.append(path)
    if not paths:
        if (year, month) in KNOWN_GAPS:
            logger.warning(f"PREDISPATCH{table} {year}-{month:02d}: known gap in AEMO archive")
            return []
        raise FileNotFoundError(f"No PREDISPATCH{table} file for {year}-{month:02d}")
    return paths


def read_13h_runs(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one AEMO zip and keep only rows from 13:00 runs, in chunks to save memory."""
    keep = []
    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
        with z.open(csv_name) as f:
            # AEMO CSV layout: line 1 = "C" comment, line 2 = "I" header, then "D" data rows
            chunks = pd.read_csv(
                f,
                skiprows=1,
                usecols=lambda c: c in columns or c == "I",
                dtype=str,
                chunksize=500_000,
            )
            for chunk in chunks:
                chunk = chunk[chunk["I"] == "D"]
                chunk = chunk[chunk["PREDISPATCHSEQNO"].str[-2:] == RUN_NUMBER]
                keep.append(chunk.drop(columns="I"))
    return pd.concat(keep, ignore_index=True)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Types, regions, pricing run only, and only the 48 half-hours of day D."""
    df = df[df["REGIONID"].isin(REGIONS)].copy()
    df = df[pd.to_numeric(df["INTERVENTION"]) == 0].drop(columns="INTERVENTION")

    run_date = pd.to_datetime(df["PREDISPATCHSEQNO"].str[:8], format="%Y%m%d")
    df["RUN_DATETIME"] = run_date + pd.Timedelta(hours=13)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    day_d = run_date + pd.Timedelta(days=1)
    in_day_d = (df["DATETIME"] > day_d) & (df["DATETIME"] <= day_d + pd.Timedelta(days=1))
    df = df[in_day_d].drop(columns="PREDISPATCHSEQNO")

    value_cols = [c for c in df.columns if c not in ("REGIONID", "DATETIME", "RUN_DATETIME")]
    df[value_cols] = df[value_cols].apply(pd.to_numeric)
    first = ["RUN_DATETIME", "DATETIME", "REGIONID"]
    return df[first + value_cols].sort_values(["REGIONID", "DATETIME"]).reset_index(drop=True)


def check(df: pd.DataFrame, table: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Fail on duplicates; report missing half-hours per region."""
    n_dupes = df.duplicated(["DATETIME", "REGIONID"]).sum()
    if n_dupes:
        raise ValueError(f"{table}: {n_dupes} duplicate (DATETIME, REGIONID) rows")
    expected = pd.date_range(start + pd.Timedelta("30min"), end, freq="30min")
    for region, g in df.groupby("REGIONID"):
        missing = expected.difference(g["DATETIME"])
        msg = f"{table} {region}: {len(g):,} rows, {len(missing):,} missing half-hours"
        if len(missing):
            logger.warning(f"{msg} (first: {missing[0]})")
        else:
            logger.success(msg)


@app.command()
def main(start: str = START, end: str = END):
    """Target days from START up to (not including) END, as in config.py."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    # The run for day D happens on D-1, so start one day earlier
    run_days = pd.date_range(start_ts - pd.Timedelta(days=1), end_ts - pd.Timedelta(days=2))
    months = sorted({(d.year, d.month) for d in run_days})

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for table, columns in TABLES.items():
        frames = []
        for year, month in months:
            logger.info(f"PREDISPATCH{table}: {year}-{month:02d}")
            for path in download(table, year, month):
                frames.append(read_13h_runs(path, columns))
        df = tidy(pd.concat(frames, ignore_index=True))
        df = df[(df["DATETIME"] > start_ts) & (df["DATETIME"] <= end_ts)]
        check(df, table, start_ts, end_ts)
        out_path = INTERIM_DATA_DIR / f"predispatch_{table.lower()}.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}")


if __name__ == "__main__":
    app()
