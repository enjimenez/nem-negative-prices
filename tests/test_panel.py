"""Checks on the saved panel (data/processed/panel.parquet), the table every model reads.

Skipped when the file doesn't exist, for example in a fresh clone before `make data`.
"""

import re

import pandas as pd
import pytest

from negprice.config import ADMIN_PRICE_DAYS, END, PROCESSED_DATA_DIR, PROJ_ROOT, REGIONS, START

PANEL = PROCESSED_DATA_DIR / "panel.parquet"
DICTIONARY = PROJ_ROOT / "references" / "data_dictionary.md"
KEYS = ["REGIONID", "HH_END"]

# AEMO predispatch archive gap: target days with no 13:00 run
PREDISPATCH_GAP = ("2022-10-02", "2022-11-01")

pytestmark = pytest.mark.skipif(not PANEL.exists(), reason="panel not built; run `make data`")


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(PANEL)


def test_one_row_per_region_and_half_hour(panel):
    n_half_hours = (pd.Timestamp(END) - pd.Timestamp(START)) // pd.Timedelta("30min")
    assert len(panel) == len(REGIONS) * n_half_hours
    assert not panel.duplicated(KEYS).any()
    assert set(panel.REGIONID) == set(REGIONS)


def test_every_day_has_48_half_hours(panel):
    assert (panel.groupby(["REGIONID", "DAY"]).size() == 48).all()
    assert (panel.DAY == (panel.HH_END - pd.Timedelta("30min")).dt.normalize()).all()


def test_every_half_hour_has_six_dispatch_intervals(panel):
    assert (panel.n_5min == 6).all()


def test_prices_respect_market_floor(panel):
    assert (panel.rrp_min >= -1000).all()


def test_forecasts_known_by_issue_time(panel):
    """No leakage: every forecast was available by 13:00 on D-1."""
    issue = panel.DAY - pd.Timedelta(hours=11)
    for col in ["pd_known_at", "pvf_known_at"]:
        known = panel[col].notna()
        assert (panel.loc[known, col] <= issue[known]).all(), col


def test_missing_values_only_where_expected(panel):
    na = panel.isna().sum()
    missing_cols = set(na[na > 0].index)
    assert all(c.startswith(("pd_", "pv_actual_")) for c in missing_cols), missing_cols
    # Predispatch is missing only in AEMO's archive gap
    pd_missing = panel.pd_rrp.isna()
    assert panel.loc[pd_missing, "DAY"].between(*PREDISPATCH_GAP).all()


def test_admin_price_flag_covers_whole_days(panel):
    first, last = (pd.Timestamp(d) for d in ADMIN_PRICE_DAYS)
    n_days = (last - first).days + 1
    assert panel.admin_price.sum() == n_days * 48 * len(REGIONS)
    assert panel.loc[panel.admin_price, "DAY"].between(first, last).all()


def test_data_dictionary_matches_panel_columns(panel):
    """Every column is documented, and every documented column exists."""
    text = DICTIONARY.read_text().split("## Feature table")[0]  # panel section only
    names = re.findall(r"^\| `(\w+)` \|", text, flags=re.MULTILINE)
    documented = {n for n in names if not n.endswith("_")}  # skip prefix rows (`pd_`, ...)
    assert documented == set(panel.columns)

def test_target_definition(panel):
    # Defined on every row, 0/1 only
    assert panel["is_neg"].notna().all()
    assert set(panel["is_neg"].unique()) <= {0, 1}
    # Matches the stated definition (guards against a silent switch to <= 0)
    assert (panel["is_neg"] == (panel["rrp_mean"] < 0)).all()
    # A negative half-hour mean needs at least one negative 5-min interval
    assert (panel.loc[panel["is_neg"] == 1, "n_neg_5min"] >= 1).all()