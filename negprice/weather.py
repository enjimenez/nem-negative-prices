"""Daily observed weather from SILO, one grid point per region.

Source: SILO DataDrill (daily grids interpolated from Bureau of Meteorology stations),
Queensland Government, CC BY 4.0. https://www.longpaddock.qld.gov.au/silo/
One point per region, at the capital city, where most of the region's demand and rooftop
PV is. AEMO's own forecasts already cover the whole region, so this is a light extra.

Needs SILO_EMAIL=<your email> in .env (SILO asks for an email instead of an API key).

Output: data/interim/silo_weather.parquet, one row per (DATE, REGIONID)
Leakage: at 13:00 on D-1 the main model may only use DATE <= D-2.
Caveat: SILO revises recent days as late station reports arrive, so today's download can
differ slightly from what was available at the time. Small effect.
"""

from io import StringIO
import os

from loguru import logger
import pandas as pd
import requests
import typer

from negprice.config import END, INTERIM_DATA_DIR, RAW_DATA_DIR, START

app = typer.Typer()

SILO_URL = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php"
RAW_DIR = RAW_DATA_DIR / "silo"

# Region -> (city, latitude, longitude)
POINTS = {
    "SA1": ("Adelaide", -34.93, 138.60),
    "VIC1": ("Melbourne", -37.81, 144.96),
    "NSW1": ("Sydney", -33.87, 151.21),
}

# SILO code -> column name in its CSV
VARIABLES = {
    "R": "daily_rain",  # mm
    "X": "max_temp",  # deg C
    "N": "min_temp",  # deg C
    "J": "radiation",  # MJ/m2, daily total
    "V": "vp",  # vapour pressure, hPa (humidity)
}

# Start a week early: the first target day needs weather from D-2 and before
LEAD = pd.Timedelta(days=7)


def parse(text: str, region: str) -> pd.DataFrame:
    """Turn SILO's CSV into (DATE, REGIONID, variables). Fails loudly on anything else."""
    df = pd.read_csv(StringIO(text))
    date_col = next((c for c in df.columns if c.strip().upper() in ("YYYY-MM-DD", "DATE")), None)
    missing = [v for v in VARIABLES.values() if v not in df.columns]
    if date_col is None or missing:
        raise RuntimeError(
            f"Unexpected SILO response for {region}. Columns: {list(df.columns)}\n{text[:300]}"
        )
    out = df[[date_col, *VARIABLES.values()]].rename(columns={date_col: "DATE"})
    out["DATE"] = pd.to_datetime(out["DATE"])
    out.insert(1, "REGIONID", region)
    return out


def get_region(region: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Read the raw CSV from data/raw/silo if it's there; otherwise download and save it."""
    city, lat, lon = POINTS[region]
    path = RAW_DIR / f"silo_{region}_{first:%Y%m%d}_{last:%Y%m%d}.csv"
    if path.exists():
        return parse(path.read_text(), region)

    email = os.environ.get("SILO_EMAIL")
    if not email:
        raise RuntimeError("Add SILO_EMAIL=<your email> to .env (SILO needs it to identify you)")
    params = {
        "lat": lat,
        "lon": lon,
        "start": f"{first:%Y%m%d}",
        "finish": f"{last:%Y%m%d}",
        "format": "csv",
        "comment": "".join(VARIABLES),
        "username": email,
        "password": "apirequest",
    }
    logger.info(f"SILO: downloading {city} ({lat}, {lon}) {first:%Y-%m-%d} to {last:%Y-%m-%d}")
    r = requests.get(SILO_URL, params=params, timeout=300)
    r.raise_for_status()
    df = parse(r.text, region)  # parse before saving, so an error page is never cached
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return df


def check(df: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> None:
    """Fail on duplicates; report missing days, missing values and value ranges."""
    n_dupes = df.duplicated(["DATE", "REGIONID"]).sum()
    if n_dupes:
        raise ValueError(f"SILO: {n_dupes} duplicate (DATE, REGIONID) rows")
    expected = pd.date_range(first, last, freq="D")
    for region, g in df.groupby("REGIONID"):
        missing = expected.difference(g["DATE"])
        n_nan = int(g[list(VARIABLES.values())].isna().sum().sum())
        ranges = ", ".join(f"{v} {g[v].min():.1f}..{g[v].max():.1f}" for v in VARIABLES.values())
        msg = f"SILO {region}: {len(g):,} days, {len(missing)} missing, {n_nan} empty values"
        if len(missing) or n_nan:
            logger.warning(msg)
        else:
            logger.success(msg)
        logger.info(f"SILO {region} ranges: {ranges}")


@app.command()
def main(start: str = START, end: str = END):
    """Days from START - 7 days to the last day before END."""
    first = pd.Timestamp(start).normalize() - LEAD
    last = pd.Timestamp(end).normalize() - pd.Timedelta(days=1)
    df = pd.concat([get_region(r, first, last) for r in POINTS], ignore_index=True)
    df = df.sort_values(["REGIONID", "DATE"]).reset_index(drop=True)
    check(df, first, last)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DATA_DIR / "silo_weather.parquet"
    df.to_parquet(out_path, index=False)
    logger.success(f"Saved {out_path}")


if __name__ == "__main__":
    app()
