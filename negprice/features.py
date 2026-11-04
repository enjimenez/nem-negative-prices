"""Build the leakage-safe feature table from the processed panel.

One row per region x half-hour of day D. Every feature is known by the forecast
issue time, 13:00 NEM time on D-1. Day-D actual weather goes to a separate
what-if table.
"""

from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import typer
from loguru import logger

from negprice.config import PRICE_ONLY_REGIONS, PROCESSED_DATA_DIR

app = typer.Typer()

ISSUE_OFFSET = pd.Timedelta(days=-1, hours=13)  # issue time = DAY + ISSUE_OFFSET
SHARE_LAG_DAYS = 2  # 4-week share window ends on D-2
SHARE_WINDOW_DAYS = 28  # D-29 ... D-2
MORNING_END = pd.Timedelta(hours=13)  # D-1 morning = half-hours ending <= 13:00
MORNING_N = 26  # half-hours ending 00:30 ... 13:00

NEM_TZ = "Australia/Brisbane"  # NEM time: UTC+10 all year
REGION_INFO = {
    "SA1": ("SA", "Australia/Adelaide"),
    "VIC1": ("VIC", "Australia/Melbourne"),
    "NSW1": ("NSW", "Australia/Sydney"),
}

PD_COLS = [
    "pd_rrp",
    "pd_totaldemand",
    "pd_availablegeneration",
    "pd_netinterchange",
    "pd_demand_and_nonschedgen",
    "pd_uigf",
    "pd_ss_solar_uigf",
    "pd_ss_wind_uigf",
]
# Forecast prices of regions that are not modelled (QLD1, TAS1), copied from the panel
EXT_COLS = [f"pd_rrp_{r.lower()}" for r in PRICE_ONLY_REGIONS]
PV_COL = "pvf_powermean"
SILO_COLS = ["silo_daily_rain", "silo_max_temp", "silo_radiation"]

KEYS = ["REGIONID", "HH_END", "DAY", "issue_time"]
LABELS = ["is_neg", "admin_price"]

# feature -> column holding its known-at time (None = calendar, known in advance)
FEATURE_KNOWN_AT = {
    "nem_hour": None,
    "local_hour": None,
    "day_of_week": None,
    "is_weekend": None,
    "is_holiday": None,
    "month": None,
    **{c: "pd_known_at" for c in PD_COLS},
    "pd_rrp_sa1": "pd_nb_known_at",
    "pd_rrp_vic1": "pd_nb_known_at",
    "pd_rrp_nsw1": "pd_nb_known_at",
    **{c: "pd_ext_known_at" for c in EXT_COLS},
    "pvf_mw": "pvf_known_at",
    "pvf_age_h": "pvf_known_at",
    "neg_share_4w": "neg_share_4w_known_at",
    "n_neg_d1_morning": "n_neg_d1_morning_known_at",
    "price_d2": "price_d2_known_at",
    "silo_rain_d2": "silo_d2_known_at",
    "silo_max_temp_d2": "silo_d2_known_at",
    "silo_radiation_d2": "silo_d2_known_at",
}
FEATURES = list(FEATURE_KNOWN_AT)
KNOWN_AT_COLS = sorted({c for c in FEATURE_KNOWN_AT.values() if c})


def load_panel(path: Path) -> pd.DataFrame:
    p = pd.read_parquet(path)
    p["DAY"] = pd.to_datetime(p["DAY"])
    p = p.sort_values(["REGIONID", "HH_END"]).reset_index(drop=True)
    counts = p.groupby(["REGIONID", "DAY"]).size()
    if not (counts == 48).all():
        raise ValueError(f"Days without 48 half-hours: {counts[counts != 48].head()}")
    p["issue_time"] = p["DAY"] + ISSUE_OFFSET
    p["slot"] = p.groupby(["REGIONID", "DAY"]).cumcount() + 1  # 1 ... 48
    return p


def calendar_features(p: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    f["nem_hour"] = p["slot"] * 0.5  # interval-end hour, 0.5 ... 24.0
    f["local_hour"] = np.nan
    f["is_holiday"] = 0
    years = range(p["DAY"].dt.year.min(), p["DAY"].dt.year.max() + 1)
    for region, (state, tz) in REGION_INFO.items():
        m = p["REGIONID"] == region
        end = p.loc[m, "HH_END"].dt.tz_localize(NEM_TZ).dt.tz_convert(tz)
        h = end.dt.hour + end.dt.minute / 60
        f.loc[m, "local_hour"] = h.where(h > 0, 24.0)  # interval ending at midnight = 24
        hol = holidays.Australia(subdiv=state, years=years)
        f.loc[m, "is_holiday"] = p.loc[m, "DAY"].dt.date.map(lambda d: d in hol).astype(int)
    f["day_of_week"] = p["DAY"].dt.day_of_week # 0 = Monday
    f["is_weekend"] = (f["day_of_week"] >= 5).astype(int)
    f["month"] = p["DAY"].dt.month
    return f


def aemo_features(p: pd.DataFrame) -> pd.DataFrame:
    f = p[PD_COLS + ["pd_known_at"] + EXT_COLS + ["pd_ext_known_at"]].copy()
    price = p.pivot(index="HH_END", columns="REGIONID", values="pd_rrp")
    known = p.pivot(index="HH_END", columns="REGIONID", values="pd_known_at")
    nb_known = []
    for region in REGION_INFO:
        own = p["REGIONID"] == region
        col = f"pd_rrp_{region.lower()}"
        f[col] = p["HH_END"].map(price[region]).where(~own)
        nb_known.append(p["HH_END"].map(known[region]).where(~own))
    f["pd_nb_known_at"] = pd.concat(nb_known, axis=1).max(axis=1)
    return f


def pv_features(p: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    f["pvf_mw"] = p[PV_COL]
    f["pvf_age_h"] = (p["issue_time"] - p["pvf_known_at"]) / pd.Timedelta(hours=1)
    f["pvf_known_at"] = p["pvf_known_at"]
    return f


def history_features(p: pd.DataFrame) -> pd.DataFrame:
    src = p[["REGIONID", "HH_END", "DAY", "slot", "rrp_mean", "is_neg", "admin_price"]]
    admin = src["admin_price"].astype(bool)

    # Actual price at the same half-hour on D-2 (NaN if D-2 is an admin day)
    d2 = src.assign(HH_END=src["HH_END"] + pd.Timedelta(days=2))
    d2 = d2.assign(
        price_d2=d2["rrp_mean"].where(~admin),
        price_d2_known_at=src["HH_END"],
    )[["REGIONID", "HH_END", "price_d2", "price_d2_known_at"]]
    out = p[["REGIONID", "HH_END", "DAY", "slot"]].merge(d2, on=["REGIONID", "HH_END"], how="left")

    # Negative half-hours on D-1 up to 13:00 (NaN if D-1 is an admin day)
    morning = src[src["HH_END"] - src["DAY"] <= MORNING_END]
    g = morning.groupby(["REGIONID", "DAY"])
    m = pd.DataFrame(
        {
            "n_neg_d1_morning": g["is_neg"].sum(),
            "n": g.size(),
            "admin": g["admin_price"].max().astype(bool),
        }
    ).reset_index()
    m.loc[(m["n"] != MORNING_N) | m["admin"], "n_neg_d1_morning"] = np.nan
    m["n_neg_d1_morning_known_at"] = m["DAY"] + MORNING_END
    m["DAY"] = m["DAY"] + pd.Timedelta(days=1)
    out = out.merge(
        m[["REGIONID", "DAY", "n_neg_d1_morning", "n_neg_d1_morning_known_at"]],
        on=["REGIONID", "DAY"],
        how="left",
    )

    # Share of negatives at the same slot over D-29 ... D-2, admin days excluded
    shares = []
    for region, grp in src.groupby("REGIONID"):
        piv = grp.assign(y=grp["is_neg"].where(~grp["admin_price"].astype(bool))).pivot(
            index="DAY", columns="slot", values="y"
        )
        piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
        share = piv.shift(SHARE_LAG_DAYS).rolling(SHARE_WINDOW_DAYS, min_periods=1).mean()
        first_full = piv.index.min() + pd.Timedelta(days=SHARE_LAG_DAYS + SHARE_WINDOW_DAYS - 1)
        share[share.index < first_full] = np.nan
        s = share.stack(future_stack=True).rename("neg_share_4w").reset_index()
        s.columns = ["DAY", "slot", "neg_share_4w"]
        s["REGIONID"] = region
        shares.append(s)
    out = out.merge(pd.concat(shares), on=["REGIONID", "DAY", "slot"], how="left")
    # Latest input is the last half-hour of D-2, which ends at D-1 00:00
    out["neg_share_4w_known_at"] = (out["DAY"] - pd.Timedelta(days=1)).where(
        out["neg_share_4w"].notna()
    )

    cols = [
        "price_d2",
        "price_d2_known_at",
        "n_neg_d1_morning",
        "n_neg_d1_morning_known_at",
        "neg_share_4w",
        "neg_share_4w_known_at",
    ]
    out.index = p.index
    return out[cols]


def weather_features(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = p.groupby(["REGIONID", "DAY"])[SILO_COLS + ["silo_known_at"]].first().reset_index()
    names = {
        "silo_daily_rain": "silo_rain",
        "silo_max_temp": "silo_max_temp",
        "silo_radiation": "silo_radiation",
    }

    # Main table: weather of D-2
    d2 = daily.assign(DAY=daily["DAY"] + pd.Timedelta(days=2)).rename(
        columns={**{k: f"{v}_d2" for k, v in names.items()}, "silo_known_at": "silo_d2_known_at"}
    )
    main = p[["REGIONID", "DAY"]].merge(d2, on=["REGIONID", "DAY"], how="left")
    main.index = p.index

    # What-if table: actual weather on day D (one row per region-day)
    whatif = daily.rename(columns={k: f"{v}_d" for k, v in names.items()}).drop(
        columns="silo_known_at"
    )
    return main.drop(columns=["REGIONID", "DAY"]), whatif


def build_features(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather, whatif = weather_features(p)
    table = pd.concat(
        [
            p[KEYS + LABELS],
            calendar_features(p),
            aemo_features(p),
            pv_features(p),
            history_features(p),
            weather,
        ],
        axis=1,
    )
    return table[KEYS + LABELS + FEATURES + KNOWN_AT_COLS], whatif


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "panel.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    whatif_path: Path = PROCESSED_DATA_DIR / "features_whatif.parquet",
):
    logger.info(f"Reading {input_path}")
    p = load_panel(input_path)
    table, whatif = build_features(p)
    table.to_parquet(output_path, index=False)
    whatif.to_parquet(whatif_path, index=False)
    logger.success(f"Features: {table.shape} -> {output_path}")
    logger.success(f"What-if weather: {whatif.shape} -> {whatif_path}")
    missing = table[FEATURES].isna().sum()
    logger.info(f"Missing values per feature:\n{missing[missing > 0]}")


if __name__ == "__main__":
    app()