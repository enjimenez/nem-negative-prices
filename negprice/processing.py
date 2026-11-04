"""Half-hourly panel: resample 5-min dispatch data and join all sources."""

import pandas as pd

from negprice.config import (
    ADMIN_PRICE_DAYS,
    END,
    INTERIM_DATA_DIR,
    PRICE_ONLY_REGIONS,
    PROCESSED_DATA_DIR,
    REGIONS,
    START,
)


def price_to_half_hourly(dp: pd.DataFrame) -> pd.DataFrame:
    """5-min RRP (interval-end labels) -> half-hourly stats (interval-end labels).

    The half-hour ending HH:30 contains the 5-min intervals ending HH:05 ... HH:30.
    """
    df = dp.assign(
        HH_END=dp["SETTLEMENTDATE"].dt.ceil("30min"),
        neg=dp["RRP"] < 0,
    )
    return (
        df.groupby(["REGIONID", "HH_END"])
        .agg(
            rrp_mean=("RRP", "mean"),
            rrp_min=("RRP", "min"),
            n_neg_5min=("neg", "sum"),
            n_5min=("RRP", "size"),
        )
        .reset_index()
    )


def regionsum_to_half_hourly(rs: pd.DataFrame) -> pd.DataFrame:
    """5-min DISPATCHREGIONSUM -> half-hourly means (MW), interval-end labels."""
    value_cols = [c for c in rs.columns if c not in ("SETTLEMENTDATE", "REGIONID")]
    return (
        rs.assign(HH_END=rs["SETTLEMENTDATE"].dt.ceil("30min"))
        .groupby(["REGIONID", "HH_END"])[value_cols]
        .mean()
        .reset_index()
    )


def pv_actual_half_hourly(pv: pd.DataFrame) -> pd.DataFrame:
    """ROOFTOP_PV_ACTUAL -> one MEASUREMENT value per region and half-hour."""
    out = pv[pv["TYPE"] == "MEASUREMENT"].rename(
        columns={
            "INTERVAL_DATETIME": "HH_END",
            "POWER": "pv_actual_mw",
            "LASTCHANGED": "pv_actual_known_at",
        }
    )[["REGIONID", "HH_END", "pv_actual_mw", "pv_actual_known_at"]]
    if out.duplicated(["REGIONID", "HH_END"]).any():
        raise ValueError("duplicate PV actual rows")
    return out


def half_hour_grid() -> pd.DataFrame:
    """Every (region, half-hour) in the study window, interval-end labels."""
    first = pd.Timestamp(START) + pd.Timedelta("30min")
    hh = pd.date_range(first, pd.Timestamp(END), freq="30min").as_unit("us")
    grid = pd.MultiIndex.from_product([REGIONS, hh], names=["REGIONID", "HH_END"]).to_frame(
        index=False
    )
    # Calendar day the half-hour belongs to: the one ending 00:00 is D 24:00
    grid["DAY"] = (grid["HH_END"] - pd.Timedelta("30min")).dt.normalize()
    return grid


KEYS = ["REGIONID", "HH_END"]


def _tag(df: pd.DataFrame, prefix: str, keys: list[str]) -> pd.DataFrame:
    """Lowercase and prefix every non-key column."""
    return df.rename(columns={c: prefix + c.lower() for c in df.columns if c not in keys})


def price_only_wide(pd_price: pd.DataFrame) -> pd.DataFrame:
    """QLD1/TAS1 forecast prices as one column per region, one row per half-hour.

    Joined on HH_END, so every modelled region's row carries the same values.
    """
    ext = pd_price[pd_price["REGIONID"].isin(PRICE_ONLY_REGIONS)]
    wide = ext.pivot(index="DATETIME", columns="REGIONID", values="RRP")
    wide = wide.reindex(columns=PRICE_ONLY_REGIONS)
    wide.columns = [f"pd_rrp_{r.lower()}" for r in wide.columns]
    wide["pd_ext_known_at"] = ext.groupby("DATETIME")["RUN_DATETIME"].max()
    return wide.rename_axis("HH_END").reset_index()


def build_panel(grid, price_hh, regionsum_hh, pv_act, pd_price, pd_rs, pv_fc, silo):
    """Left-join every source onto the (region, half-hour) grid."""
    rs = _tag(regionsum_hh, "", KEYS)
    ext = price_only_wide(pd_price)
    pd_price = pd_price[pd_price["REGIONID"].isin(REGIONS)]

    pdx = pd_price.merge(
        pd_rs,
        on=["REGIONID", "DATETIME", "RUN_DATETIME"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_rs"),
    ).rename(columns={"DATETIME": "HH_END", "RUN_DATETIME": "known_at"})
    pdx = _tag(pdx, "pd_", KEYS)

    pvf = pv_fc.drop(columns="LASTCHANGED", errors="ignore").rename(
        columns={"INTERVAL_DATETIME": "HH_END", "VERSION_DATETIME": "known_at"}
    )
    pvf = _tag(pvf, "pvf_", KEYS)

    sw = silo.rename(columns={"DATE": "DAY"})
    sw["known_at"] = sw["DAY"] + pd.Timedelta(days=1, hours=9)
    sw = _tag(sw, "silo_", ["REGIONID", "DAY"])

    panel = grid
    for src in [price_hh, rs, pv_act, pdx, pvf]:
        panel = panel.merge(src, on=KEYS, how="left", validate="one_to_one")
    panel = panel.merge(ext, on="HH_END", how="left", validate="many_to_one")
    return panel.merge(sw, on=["REGIONID", "DAY"], how="left", validate="many_to_one")


def add_admin_price_flag(panel: pd.DataFrame) -> pd.DataFrame:
    """True for half-hours on days with administered prices (not market outcomes)."""
    first, last = (pd.Timestamp(d) for d in ADMIN_PRICE_DAYS)
    return panel.assign(admin_price=panel["DAY"].between(first, last))


def add_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Prediction target: 1 if the half-hour mean price is strictly below $0, else 0."""
    return panel.assign(is_neg=(panel["rrp_mean"] < 0).astype("int8"))


def main() -> None:
    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(INTERIM_DATA_DIR / f"{name}.parquet")

    panel = build_panel(
        half_hour_grid(),
        price_to_half_hourly(read("dispatchprice")),
        regionsum_to_half_hourly(read("dispatchregionsum")),
        pv_actual_half_hourly(read("rooftop_pv_actual")),
        read("predispatch_price"),
        read("predispatch_regionsum"),
        read("rooftop_pv_forecast"),
        read("silo_weather"),
    )
    panel = add_target(add_admin_price_flag(panel)).sort_values(KEYS).reset_index(drop=True)

    out = PROCESSED_DATA_DIR / "panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"saved {out.name}: {panel.shape[0]:,} rows × {panel.shape[1]} columns")

if __name__ == "__main__":
    main()
