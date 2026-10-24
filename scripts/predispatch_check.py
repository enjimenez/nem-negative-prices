# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["nemseer==1.0.7", "pandas<2.3", "numpy<2", "pyarrow<18"]
# ///
"""Time-boxed check: can we get AEMO's 13:00 predispatch run for day D?

Result (24 Oct 2026): works up to Jul 2024, fails from Aug 2024 (AEMO renamed its files).
Superseded by negprice/predispatch.py; kept as the evidence for decision C.

Runs in its own Python 3.11 environment (NEMSEER doesn't support 3.12):
    uv run scripts/predispatch_check.py 2021-10-15
"""

import sys
import time
from pathlib import Path

import nemseer
import pandas as pd

CACHE = Path("data/raw/nemseer_cache")
REGIONS = ["SA1", "VIC1", "NSW1"]
FMT = "%Y/%m/%d %H:%M"

day_d = pd.Timestamp(sys.argv[1])                                   # target day D
run = day_d - pd.Timedelta(days=1) + pd.Timedelta(hours=13)         # 13:00 on D-1

CACHE.mkdir(parents=True, exist_ok=True)
print("Tables that month:", nemseer.get_tables(day_d.year, day_d.month, "PREDISPATCH"))

t0 = time.time()
data = nemseer.compile_data(
    run_start=run.strftime(FMT),
    run_end=run.strftime(FMT),
    forecasted_start=(day_d + pd.Timedelta(minutes=30)).strftime(FMT),
    forecasted_end=(day_d + pd.Timedelta(days=1)).strftime(FMT),
    forecast_type="PREDISPATCH",
    tables=["PRICE", "REGIONSUM"],
    raw_cache=str(CACHE),
)
print(f"Download + compile took {time.time() - t0:.0f} s")
size_gb = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file()) / 1e9
print(f"Cache size now: {size_gb:.2f} GB")

price = data["PRICE"]
price = price[price["REGIONID"].isin(REGIONS)]
print(price[["PREDISPATCH_RUN_DATETIME", "DATETIME", "REGIONID", "RRP"]].head())
print("Rows per region:", price.groupby("REGIONID").size().to_dict())
print("Run times:", price["PREDISPATCH_RUN_DATETIME"].unique())
print("Forecast window:", price["DATETIME"].min(), "->", price["DATETIME"].max())
print("Negative half-hours forecast:", (price["RRP"] < 0).groupby(price["REGIONID"]).sum().to_dict())
print("REGIONSUM columns:", list(data["REGIONSUM"].columns))
