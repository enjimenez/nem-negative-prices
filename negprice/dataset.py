"""Pull AEMO dispatch data with NEMOSIS and save one clean parquet per table.

Output: data/interim/dispatchprice.parquet and data/interim/dispatchregionsum.parquet
- Regions in config.REGIONS only
- 5-minute intervals, SETTLEMENTDATE = END of the interval
- NEM time (UTC+10 all year, no daylight saving), stored without a timezone
"""

from itertools import pairwise

from loguru import logger
from nemosis import dynamic_data_compiler
from nemosis.custom_errors import NoDataToReturn
import pandas as pd
import typer

from negprice.config import END, INTERIM_DATA_DIR, NEMOSIS_CACHE, REGIONS, START

app = typer.Typer()

NEMOSIS_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"

# Columns to keep from each table
TABLES = {
    "DISPATCHPRICE": ["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "RRP"],
    "DISPATCHREGIONSUM": [
        "SETTLEMENTDATE",
        "REGIONID",
        "INTERVENTION",
        "TOTALDEMAND",
        "DEMAND_AND_NONSCHEDGEN",
        "TOTALINTERMITTENTGENERATION",
        "UIGF",
        "AVAILABLEGENERATION",
        "NETINTERCHANGE",
    ],
}


def pull_table(table: str, columns: list[str]) -> pd.DataFrame:
    """Download one table month by month and return SA/VIC/NSW rows."""
    month_edges = pd.date_range(START, END, freq="MS")
    frames, failed = [], []

    for month_start, month_end in pairwise(month_edges):
        logger.info(f"{table}: {month_start:%Y-%m}")
        try:
            df = dynamic_data_compiler(
                month_start.strftime(NEMOSIS_TIME_FORMAT),
                month_end.strftime(NEMOSIS_TIME_FORMAT),
                table,
                str(NEMOSIS_CACHE),
                select_columns=columns,
                filter_cols=["REGIONID"],
                filter_values=(REGIONS,),
                fformat="parquet",
                keep_csv=False,
            )
            frames.append(df)
        except NoDataToReturn:
            failed.append(f"{month_start:%Y-%m}")

    if failed:
        raise RuntimeError(f"{table}: no data for {failed}. Rerun, or move END back.")

    df = pd.concat(frames, ignore_index=True)

    # Keep the pricing run only (INTERVENTION = 0)
    df = df[pd.to_numeric(df["INTERVENTION"]) == 0].drop(columns="INTERVENTION")
    return df.sort_values(["REGIONID", "SETTLEMENTDATE"]).reset_index(drop=True)


def check(df: pd.DataFrame, table: str) -> None:
    """Fail on duplicates; report missing 5-minute intervals per region."""
    n_dupes = df.duplicated(["SETTLEMENTDATE", "REGIONID"]).sum()
    if n_dupes:
        raise ValueError(f"{table}: {n_dupes} duplicate (SETTLEMENTDATE, REGIONID) rows")

    expected = pd.date_range(
        pd.Timestamp(START) + pd.Timedelta("5min"), pd.Timestamp(END), freq="5min"
    )
    for region, g in df.groupby("REGIONID"):
        missing = expected.difference(g["SETTLEMENTDATE"])
        msg = f"{table} {region}: {len(g):,} rows, {len(missing):,} missing intervals"
        if len(missing):
            logger.warning(f"{msg} (first: {missing[0]})")
        else:
            logger.success(msg)


@app.command()
def main():
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for table, columns in TABLES.items():
        df = pull_table(table, columns)
        check(df, table)
        out_path = INTERIM_DATA_DIR / f"{table.lower()}.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}")


if __name__ == "__main__":
    app()
