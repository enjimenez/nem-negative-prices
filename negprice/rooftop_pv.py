"""Rooftop PV: AEMO's estimated actuals and its day-ahead forecasts, SA/VIC/NSW.

Actuals (ROOFTOP_PV_ACTUAL) come through NEMOSIS, like dataset.py.
Forecasts (ROOFTOP_PV_FORECAST) are not in NEMOSIS, so we download them from AEMO's
monthly archive (DATA folder), like predispatch.py. Each forecast version covers 8 days
of half-hours; for target day D we keep only the latest version produced by 12:30 on D-1.

Output:
- data/interim/rooftop_pv_actual.parquet: one row per (half-hour, region, TYPE)
- data/interim/rooftop_pv_forecast.parquet: one row per (half-hour of day D, region),
  with VERSION_DATETIME = when that forecast was produced (its "known at" time for Day 5)
- Times are the END of the half-hour, NEM time (UTC+10, no daylight saving), no timezone

TYPE in the actuals (AEMO MMS data model):
- MEASUREMENT: best estimate, published on the day, about 30 min after the interval
- SATELLITE: estimate from satellite imagery, about 30 min after the interval
- DAILY: best estimate, published the day after
Leakage rule for Day 5: at 13:00 on D-1, DAILY values are known only up to D-2.
"""

from itertools import pairwise
from pathlib import Path
from urllib.parse import quote
import zipfile

from loguru import logger
from nemosis import dynamic_data_compiler
from nemosis.custom_errors import NoDataToReturn
import pandas as pd
import requests
import typer

from negprice.config import END, INTERIM_DATA_DIR, NEMOSIS_CACHE, RAW_DATA_DIR, REGIONS, START

app = typer.Typer()

NEMOSIS_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
AEMO_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"  # how datetimes appear inside AEMO's CSV files

ACTUAL_COLUMNS = ["INTERVAL_DATETIME", "REGIONID", "TYPE", "POWER", "QI", "LASTCHANGED"]

ARCHIVE_URL = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year}/"
    "MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
)
CACHE = RAW_DATA_DIR / "rooftop_pv_forecast_cache"
# nemweb rejects the default `requests` user agent
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) nem-negative-prices research"}

FORECAST_COLUMNS = [
    "VERSION_DATETIME",
    "REGIONID",
    "INTERVAL_DATETIME",
    "POWERMEAN",
    "POWERPOE50",
    "POWERPOELOW",
    "POWERPOEHIGH",
    "LASTCHANGED",
]
TIME_COLUMNS = ["VERSION_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED"]

# Latest forecast allowed for target day D: produced by 12:30 on D-1.
# The issue time is 13:00 (decision A). The 30-minute margin covers the delay between
# AEMO producing a forecast and publishing it; the run logs that delay so you can check it.
CUTOFF = pd.Timedelta(hours=12, minutes=30)


# --------------------------------------------------------------------------------------
# Actuals (NEMOSIS)
# --------------------------------------------------------------------------------------


def pull_actual(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Download ROOFTOP_PV_ACTUAL month by month and return SA/VIC/NSW rows."""
    edges = pd.date_range(start_ts, end_ts, freq="MS").union([start_ts, end_ts])
    frames, failed = [], []

    for chunk_start, chunk_end in pairwise(edges):
        logger.info(f"ROOFTOP_PV_ACTUAL: {chunk_start:%Y-%m}")
        try:
            df = dynamic_data_compiler(
                chunk_start.strftime(NEMOSIS_TIME_FORMAT),
                chunk_end.strftime(NEMOSIS_TIME_FORMAT),
                "ROOFTOP_PV_ACTUAL",
                str(NEMOSIS_CACHE),
                select_columns=ACTUAL_COLUMNS,
                filter_cols=["REGIONID"],
                filter_values=(REGIONS,),
                fformat="parquet",
                keep_csv=False,
            )
            frames.append(df)
        except NoDataToReturn:
            failed.append(f"{chunk_start:%Y-%m}")

    if failed:
        raise RuntimeError(f"ROOFTOP_PV_ACTUAL: no data for {failed}. Rerun, or move END back.")

    df = pd.concat(frames, ignore_index=True)
    df["INTERVAL_DATETIME"] = pd.to_datetime(df["INTERVAL_DATETIME"])
    df["LASTCHANGED"] = pd.to_datetime(df["LASTCHANGED"])
    df[["POWER", "QI"]] = df[["POWER", "QI"]].apply(pd.to_numeric)
    df = df[(df["INTERVAL_DATETIME"] > start_ts) & (df["INTERVAL_DATETIME"] <= end_ts)]
    return df.sort_values(["REGIONID", "TYPE", "INTERVAL_DATETIME"]).reset_index(drop=True)


def check_actual(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    """Fail on duplicates; report coverage and missing half-hours per region and TYPE."""
    n_dupes = df.duplicated(["INTERVAL_DATETIME", "REGIONID", "TYPE"]).sum()
    if n_dupes:
        raise ValueError(f"ROOFTOP_PV_ACTUAL: {n_dupes} duplicate rows")

    expected = pd.date_range(start_ts + pd.Timedelta("30min"), end_ts, freq="30min")
    for (region, kind), g in df.groupby(["REGIONID", "TYPE"]):
        missing = expected.difference(g["INTERVAL_DATETIME"])
        msg = (
            f"ROOFTOP_PV_ACTUAL {region} {kind}: {len(g):,} rows, "
            f"{g['INTERVAL_DATETIME'].min()} to {g['INTERVAL_DATETIME'].max()}, "
            f"{len(missing):,} missing half-hours"
        )
        if len(missing):
            logger.warning(f"{msg} (first: {missing[0]})")
        else:
            logger.success(msg)


# --------------------------------------------------------------------------------------
# Forecasts (own downloader)
# --------------------------------------------------------------------------------------


def file_names(year: int, month: int) -> list[str]:
    """Archive file names to try. AEMO changed the naming scheme in August 2024."""
    if (year, month) >= (2024, 8):
        return [
            f"PUBLIC_ARCHIVE#ROOFTOP_PV_FORECAST#FILE{n:02d}#{year}{month:02d}010000.zip"
            for n in range(1, 10)
        ]
    return [f"PUBLIC_DVD_ROOFTOP_PV_FORECAST_{year}{month:02d}010000.zip"]


def download(year: int, month: int) -> list[Path]:
    """Download one month's zip file(s) into the cache; skip files already there."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in file_names(year, month):
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
        raise FileNotFoundError(f"No ROOFTOP_PV_FORECAST file for {year}-{month:02d}")
    return paths


def target_day(interval_end: pd.Series) -> pd.Series:
    """Calendar day each half-hour belongs to. The half-hour ending 00:00 is the day before."""
    return (interval_end - pd.Timedelta("30min")).dt.normalize()


def latest_version(df: pd.DataFrame) -> pd.DataFrame:
    """For each (DAY, REGIONID), keep only the rows of the most recent forecast version."""
    latest = df.groupby(["DAY", "REGIONID"])["VERSION_DATETIME"].transform("max")
    return df[df["VERSION_DATETIME"] == latest]


def read_allowed_versions(path: Path) -> pd.DataFrame:
    """Read one AEMO zip and keep, per (target day, region), the latest allowed version.

    Allowed = produced by D-1 + CUTOFF. Done chunk by chunk to save memory: the latest
    version over all chunks is the latest of each chunk's latest, so nothing is lost.
    """
    keep, versions = [], set()
    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
        with z.open(csv_name) as f:
            # AEMO CSV layout: line 1 = "C" comment, line 2 = "I" header, then "D" data rows
            chunks = pd.read_csv(
                f,
                skiprows=1,
                usecols=lambda c: c in FORECAST_COLUMNS or c == "I",
                dtype=str,
                chunksize=500_000,
            )
            for chunk in chunks:
                chunk = chunk[(chunk["I"] == "D") & chunk["REGIONID"].isin(REGIONS)]
                chunk = chunk.drop(columns="I")
                for col in TIME_COLUMNS:
                    chunk[col] = pd.to_datetime(chunk[col], format=AEMO_TIME_FORMAT)
                versions.update(chunk["VERSION_DATETIME"].unique())

                chunk["DAY"] = target_day(chunk["INTERVAL_DATETIME"])
                cutoff = chunk["DAY"] - pd.Timedelta(days=1) + CUTOFF
                keep.append(latest_version(chunk[chunk["VERSION_DATETIME"] <= cutoff]))

    spacing = pd.Series(sorted(versions)).diff().median()
    logger.info(f"{path.name}: {len(versions):,} forecast versions, median spacing {spacing}")
    return pd.concat(keep, ignore_index=True)


def pull_forecast(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Latest allowed forecast version for every target day in (start_ts, end_ts]."""
    # The forecast for day D is made on D-1, so start one day earlier
    run_days = pd.date_range(start_ts - pd.Timedelta(days=1), end_ts - pd.Timedelta(days=2))
    months = sorted({(d.year, d.month) for d in run_days})

    frames = []
    for year, month in months:
        logger.info(f"ROOFTOP_PV_FORECAST: {year}-{month:02d}")
        for path in download(year, month):
            frames.append(read_allowed_versions(path))

    df = pd.concat(frames, ignore_index=True)

    # Monthly files overlap by about a day, so a version near a month boundary can appear
    # in two files. Keep one copy (the most recently changed) and report if copies differ.
    key = ["VERSION_DATETIME", "INTERVAL_DATETIME", "REGIONID"]
    n_copies = df.duplicated(key).sum()
    if n_copies:
        values = ["POWERMEAN", "POWERPOE50", "POWERPOELOW", "POWERPOEHIGH"]
        n_differ = df.drop_duplicates(key + values).duplicated(key).sum()
        logger.info(
            f"{n_copies:,} rows appear in two monthly files ({n_differ:,} with different "
            "values); keeping the most recently changed copy"
        )
        df = df.sort_values("LASTCHANGED").drop_duplicates(key, keep="last")

    df = latest_version(df)
    df = df[(df["INTERVAL_DATETIME"] > start_ts) & (df["INTERVAL_DATETIME"] <= end_ts)]
    value_cols = ["POWERMEAN", "POWERPOE50", "POWERPOELOW", "POWERPOEHIGH"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric)
    first = ["VERSION_DATETIME", "INTERVAL_DATETIME", "REGIONID"]
    df = df[first + value_cols + ["LASTCHANGED", "DAY"]]
    return df.sort_values(["REGIONID", "INTERVAL_DATETIME"]).reset_index(drop=True)


def check_forecast(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    """Fail on duplicates or leaks; report missing half-hours and which versions were used."""
    n_dupes = df.duplicated(["INTERVAL_DATETIME", "REGIONID"]).sum()
    if n_dupes:
        raise ValueError(f"ROOFTOP_PV_FORECAST: {n_dupes} duplicate rows")

    n_leaks = (df["VERSION_DATETIME"] > df["DAY"] - pd.Timedelta(days=1) + CUTOFF).sum()
    if n_leaks:
        raise ValueError(f"ROOFTOP_PV_FORECAST: {n_leaks} rows use a version after the cutoff")

    expected = pd.date_range(start_ts + pd.Timedelta("30min"), end_ts, freq="30min")
    for region, g in df.groupby("REGIONID"):
        missing = expected.difference(g["INTERVAL_DATETIME"])
        msg = f"ROOFTOP_PV_FORECAST {region}: {len(g):,} rows, {len(missing):,} missing half-hours"
        if len(missing):
            logger.warning(f"{msg} (first: {missing[0]})")
        else:
            logger.success(msg)

    # Which version did each target day get? Expect almost all at 12:30 on D-1.
    used = df.drop_duplicates(["DAY", "REGIONID"])
    age = used["DAY"] - used["VERSION_DATETIME"]  # time from version to start of day D
    clock = used["VERSION_DATETIME"].dt.strftime("%H:%M").value_counts().head(3)
    logger.info(f"Version time of day used (top 3): {clock.to_dict()}")
    n_stale = (age > pd.Timedelta(hours=12)).sum()  # i.e. not made on D-1 after 12:00
    if n_stale:
        logger.warning(f"{n_stale} (day, region) pairs use a version older than 12:00 on D-1")

    # Publication delay, approximated by LASTCHANGED - VERSION_DATETIME.
    # If this is often above 30 min, CUTOFF is too late and must move earlier.
    lag = (used["LASTCHANGED"] - used["VERSION_DATETIME"]).dt.total_seconds() / 60
    logger.info(
        f"Publication delay (min): median {lag.median():.0f}, "
        f"95th pct {lag.quantile(0.95):.0f}, max {lag.max():.0f}"
    )


@app.command()
def main(start: str = START, end: str = END, actual: bool = True, forecast: bool = True):
    """Half-hours from START up to END, as in config.py. --no-actual / --no-forecast skip one."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if actual:
        df = pull_actual(start_ts, end_ts)
        check_actual(df, start_ts, end_ts)
        out_path = INTERIM_DATA_DIR / "rooftop_pv_actual.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}")

    if forecast:
        df = pull_forecast(start_ts, end_ts)
        check_forecast(df, start_ts, end_ts)
        out_path = INTERIM_DATA_DIR / "rooftop_pv_forecast.parquet"
        df.drop(columns="DAY").to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}")


if __name__ == "__main__":
    app()
