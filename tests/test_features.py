"""Tests on the saved feature table (data/processed/features.parquet)."""

import numpy as np
import pandas as pd
import pytest

from negprice.config import INTERIM_DATA_DIR, PRICE_ONLY_REGIONS, PROCESSED_DATA_DIR, PROJ_ROOT
from negprice.features import (
    EXT_COLS,
    FEATURE_KNOWN_AT,
    FEATURES,
    PD_COLS,
    REGION_INFO,
    SHARE_LAG_DAYS,
    SHARE_WINDOW_DAYS,
)
from negprice.predispatch import KNOWN_GAPS

FEATURES_PATH = PROCESSED_DATA_DIR / "features.parquet"
PANEL_PATH = PROCESSED_DATA_DIR / "panel.parquet"
DICT_PATH = PROJ_ROOT / "references" / "data_dictionary.md"
PREDISPATCH_PRICE_PATH = INTERIM_DATA_DIR / "predispatch_price.parquet"
DAY1 = pd.Timedelta(days=1)

pytestmark = pytest.mark.skipif(
    not (FEATURES_PATH.exists() and PANEL_PATH.exists()),
    reason="features.parquet or panel.parquet not built",
)


@pytest.fixture(scope="module")
def f():
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="module")
def panel():
    p = pd.read_parquet(PANEL_PATH)
    p["DAY"] = pd.to_datetime(p["DAY"])
    return p


def admin_days(f):
    return list(f.loc[f["admin_price"].astype(bool), "DAY"].unique())


def predispatch_gap(f):
    """Target days whose 13:00 run (on D-1) falls in a known archive gap month."""
    run_day = f["DAY"] - DAY1
    ym = run_day.dt.year * 100 + run_day.dt.month
    return ym.isin({y * 100 + m for y, m in KNOWN_GAPS})


# 1
def test_known_at_not_after_issue_time(f):
    assert set(FEATURES) == set(FEATURE_KNOWN_AT)
    for feat, col in FEATURE_KNOWN_AT.items():
        if col is None:  # calendar features
            continue
        has_value = f[feat].notna()
        known = f.loc[has_value, col]
        assert known.notna().all(), f"{feat}: value without a known-at time"
        late = known > f.loc[has_value, "issue_time"]
        assert not late.any(), f"{feat}: {late.sum()} values known after issue time"


# 2
@pytest.mark.parametrize("feat, lag_days", [("price_d2", 2), ("n_neg_d1_morning", 1)])
def test_lag_nan_exactly_on_admin_source_days(f, feat, lag_days):
    source_day = f["DAY"] - pd.Timedelta(days=lag_days)
    expected = source_day.isin(admin_days(f)) | (source_day < f["DAY"].min())
    mismatch = f[feat].isna() != expected
    assert not mismatch.any(), f"{feat}: {mismatch.sum()} rows"


def test_share_never_nan_because_of_admin(f):
    first_full = f["DAY"].min() + pd.Timedelta(days=SHARE_LAG_DAYS + SHARE_WINDOW_DAYS - 1)
    assert (f["neg_share_4w"].isna() == (f["DAY"] < first_full)).all()


# 3
def test_48_rows_per_region_day_same_days_as_panel(f, panel):
    counts = f.groupby(["REGIONID", "DAY"]).size()
    assert (counts == 48).all()
    assert set(counts.index) == set(panel.groupby(["REGIONID", "DAY"]).size().index)


# 4
def test_unique_region_half_hour(f):
    assert not f.duplicated(["REGIONID", "HH_END"]).any()


# 5
def test_nans_only_where_expected(f):
    start = f["DAY"].min()
    gap = predispatch_gap(f)
    expected = {c: gap for c in PD_COLS + EXT_COLS}
    for r in REGION_INFO:
        expected[f"pd_rrp_{r.lower()}"] = gap | (f["REGIONID"] == r)
    for c in ["silo_rain_d2", "silo_max_temp_d2", "silo_radiation_d2"]:
        expected[c] = f["DAY"] < start + 2 * DAY1
    exact_elsewhere = {"price_d2", "n_neg_d1_morning", "neg_share_4w"}  # tested in 2
    no_nan = pd.Series(False, index=f.index)
    for feat in FEATURES:
        if feat in exact_elsewhere:
            continue
        mismatch = f[feat].isna() != expected.get(feat, no_nan)
        assert not mismatch.any(), f"{feat}: {mismatch.sum()} unexpected (non-)missing values"


# 6
def test_neighbour_prices_match_own_forecast(f):
    own = f.pivot(index="HH_END", columns="REGIONID", values="pd_rrp")
    for r in REGION_INFO:
        other = f["REGIONID"] != r
        expected = f.loc[other, "HH_END"].map(own[r])
        pd.testing.assert_series_equal(
            f.loc[other, f"pd_rrp_{r.lower()}"], expected, check_names=False
        )


# 7
def test_price_d2_matches_panel(f, panel):
    rows = f[f["price_d2"].notna()].sample(1000, random_state=0)
    src = panel.set_index(["REGIONID", "HH_END"])["rrp_mean"]
    keys = list(zip(rows["REGIONID"], rows["HH_END"] - 2 * DAY1))
    np.testing.assert_allclose(rows["price_d2"].to_numpy(), src.loc[keys].to_numpy())


# 8
@pytest.mark.parametrize(
    "region, hh_end",
    [
        ("SA1", "2024-01-15 12:30"),
        ("VIC1", "2022-07-01 13:00"),  # window includes the admin days
        ("NSW1", "2025-10-05 00:00"),  # last half-hour of a day
    ],
)
def test_neg_share_hand_count(f, panel, region, hh_end):
    assert f["neg_share_4w"].dropna().between(0, 1).all()
    t = pd.Timestamp(hh_end)
    row = f[(f["REGIONID"] == region) & (f["HH_END"] == t)]
    assert len(row) == 1
    p = panel[panel["REGIONID"] == region].set_index("HH_END")
    vals = []
    for k in range(SHARE_LAG_DAYS, SHARE_LAG_DAYS + SHARE_WINDOW_DAYS):
        s = p.loc[t - pd.Timedelta(days=k)]
        if not s["admin_price"]:
            vals.append(s["is_neg"])
    assert row["neg_share_4w"].iloc[0] == pytest.approx(np.mean(vals))


# 9
def test_no_day_d_actuals_in_main_table(f, panel):
    allowed = {"REGIONID", "HH_END", "DAY", "is_neg", "admin_price", "pd_known_at", "pvf_known_at"}
    allowed |= set(EXT_COLS) | {"pd_ext_known_at"}  # AEMO forecasts, known at 13:00 on D-1
    leaked = (set(panel.columns) - allowed - set(PD_COLS)) & set(f.columns)
    assert not leaked, f"Panel actuals in the feature table: {leaked}"
    weather_d = [c for c in f.columns if c.startswith("silo_") and "_d2" not in c]
    assert not weather_d, f"Day-D weather in the main table: {weather_d}"


# 10
def test_features_documented(f):
    text = DICT_PATH.read_text()
    missing = [c for c in FEATURES if f"`{c}`" not in text]
    assert not missing, f"Not in the data dictionary: {missing}"


# 11
@pytest.mark.skipif(not PREDISPATCH_PRICE_PATH.exists(), reason="predispatch_price not built")
def test_price_only_regions_match_predispatch(f):
    src = pd.read_parquet(PREDISPATCH_PRICE_PATH)
    for r in PRICE_ONLY_REGIONS:
        price = src[src["REGIONID"] == r].set_index("DATETIME")["RRP"]
        assert price.index.is_unique, r
        expected = f["HH_END"].map(price)
        pd.testing.assert_series_equal(
            f[f"pd_rrp_{r.lower()}"], expected, check_names=False
        )