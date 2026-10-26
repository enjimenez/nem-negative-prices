"""Unit tests on small synthetic data: they run anywhere, no download needed."""

import pandas as pd

from negprice.processing import price_to_half_hourly


def test_half_hour_holds_intervals_ending_05_to_30():
    # Twelve 5-min intervals ending 00:05 ... 01:00 (interval-end labels)
    t = pd.date_range("2024-01-01 00:05", periods=12, freq="5min")
    rrp = [-10, 20, 20, 20, 20, 20] + [50] * 6
    dp = pd.DataFrame({"SETTLEMENTDATE": t, "REGIONID": "SA1", "RRP": rrp})

    hh = price_to_half_hourly(dp).set_index("HH_END")

    assert list(hh.index) == [pd.Timestamp("2024-01-01 00:30"), pd.Timestamp("2024-01-01 01:00")]
    first = hh.loc[pd.Timestamp("2024-01-01 00:30")]
    assert first.n_5min == 6
    assert first.rrp_min == -10
    assert first.n_neg_5min == 1
    assert first.rrp_mean == 15
    assert hh.loc[pd.Timestamp("2024-01-01 01:00"), "n_neg_5min"] == 0