"""Tests for the battery-value rule (negprice/battery.py)."""

import numpy as np
import pandas as pd
import pytest

from negprice.backtest import make_folds
from negprice.battery import (
    AEMO_PRICE,
    CEILING,
    CHARGE_SLOTS,
    ENERGY_MWH,
    RTE,
    SLOT_MWH,
    SLOTS,
    STRATEGIES,
    annualise,
    battery_backtest,
    bootstrap_p,
    charge_window,
    day_value,
    discharge_profile,
    optimise,
    perfect_foresight,
    rule_v1_schedule,
    rule_v2_schedule,
    rule_v3_schedule,
    slot_of_day,
)
from negprice.config import PROCESSED_DATA_DIR, REPORTS_DIR

PANEL_PATH = PROCESSED_DATA_DIR / "panel.parquet"
PREDS_PATH = PROCESSED_DATA_DIR / "backtest_predictions.parquet"
SELECTIONS_PATH = REPORTS_DIR / "backtest" / "selections.csv"
FOLDS = make_folds()

needs_data = pytest.mark.skipif(
    not (PANEL_PATH.exists() and PREDS_PATH.exists() and SELECTIONS_PATH.exists()),
    reason="panel, backtest predictions or selections not built",
)

# local_hour of a winter day (NEM time = local time in NSW/VIC, no daylight saving)
LOCAL_HOUR = np.arange(1, SLOTS + 1) / 2
WINDOW = charge_window(LOCAL_HOUR)
FLAT_PROFILE = np.zeros(SLOTS)
TIE_KEY = np.random.default_rng(0).random(SLOTS)


def random_day(rng):
    """A day of prices, probabilities, a cutoff and a discharge profile (also used as a
    stand-in for AEMO's forecast price)."""
    price = rng.normal(60, 80, SLOTS)
    price[rng.random(SLOTS) < 0.2] = rng.uniform(-500, 0, 1)
    prob = rng.random(SLOTS)
    return price, prob, float(rng.uniform(0.05, 0.95)), rng.normal(80, 40, SLOTS)


def schedules(price, prob, cutoff, profile):
    """Every kind of schedule for one day: v1, v2, the AEMO-price optimiser, the ceiling."""
    return [rule_v1_schedule(prob, cutoff, WINDOW, profile, TIE_KEY),
            rule_v2_schedule(prob, cutoff, WINDOW, profile, TIE_KEY),
            rule_v3_schedule(prob, cutoff, WINDOW, profile, TIE_KEY),
            optimise(profile),
            perfect_foresight(price)[1:]]


@pytest.fixture(scope="module")
def real_daily():
    panel = pd.read_parquet(PANEL_PATH, columns=["REGIONID", "HH_END", "DAY", "rrp_mean",
                                                 "admin_price"])
    preds = pd.read_parquet(PREDS_PATH)
    selections = pd.read_csv(SELECTIONS_PATH)
    return battery_backtest(panel, preds, selections, folds=FOLDS[:3])


# ---------------------------------------------------------------- physics


@needs_data
def test_1_ceiling_beats_every_strategy_on_real_data(real_daily):
    v = real_daily.pivot_table(index=["REGIONID", "DAY"], columns="strategy", values="value")
    others = [c for c in v.columns if c != CEILING]
    assert len(others) == 3 * len(STRATEGIES) + 1  # v1, v2, v3 per strategy + aemo_price
    for s in others:
        assert (v[CEILING] >= v[s] - 1e-6).all(), s


def test_1_ceiling_beats_every_schedule_on_random_days():
    rng = np.random.default_rng(1)
    for _ in range(300):
        price, prob, cutoff, profile = random_day(rng)
        for b, s in schedules(price, prob, cutoff, profile)[:4]:
            assert perfect_foresight(price)[0] >= day_value(price, b, s) - 1e-6


def test_2_energy_balance():
    rng = np.random.default_rng(2)
    for _ in range(300):
        price, prob, cutoff, profile = random_day(rng)
        for b, s in schedules(price, prob, cutoff, profile):
            assert s.sum() == pytest.approx(RTE * b.sum())
            assert b.sum() <= ENERGY_MWH + 1e-9
            assert (b <= SLOT_MWH + 1e-9).all() and (s <= SLOT_MWH + 1e-9).all()
            assert (b >= 0).all() and (s >= 0).all()


def test_3_state_of_charge_within_limits_and_empty_at_end():
    rng = np.random.default_rng(3)
    for _ in range(300):
        price, prob, cutoff, profile = random_day(rng)
        for b, s in schedules(price, prob, cutoff, profile):
            soc = np.cumsum(b - s / RTE)  # stored energy after each half-hour
            assert (soc >= -1e-9).all() and (soc <= ENERGY_MWH + 1e-9).all()
            assert soc[-1] == pytest.approx(0, abs=1e-9)


def test_4_discharge_after_last_charge():
    rng = np.random.default_rng(4)
    for _ in range(300):
        price, prob, cutoff, profile = random_day(rng)
        for b, s in schedules(price, prob, cutoff, profile):
            if b.sum():
                assert np.flatnonzero(s).min() > np.flatnonzero(b).max()


# ------------------------------------------------------------ the rule


def test_5_charge_window_count_and_cutoff():
    rng = np.random.default_rng(5)
    for _ in range(300):
        _, prob, cutoff, profile = random_day(rng)
        b, _ = rule_v1_schedule(prob, cutoff, WINDOW, profile, TIE_KEY)
        slots = np.flatnonzero(b)
        assert len(slots) <= CHARGE_SLOTS
        assert WINDOW[slots].all() and (prob[slots] >= cutoff).all()
    # the window ends where the evening block starts: interval ending 16:00 is the last
    assert LOCAL_HOUR[WINDOW].max() == 16.0


def test_5_window_follows_local_time_with_daylight_saving():
    # SA in summer: local = NEM + 30 min, so the last half-hours wrap past midnight
    local = np.arange(1, SLOTS + 1) / 2 + 0.5  # 1.0, 1.5, ..., 24.0, 24.5
    local[local > 24] -= 24  # last half-hour ends at 00:30 local
    w = charge_window(local)
    assert local[w].max() == 16.0 and not w[-2:].any()


def test_6_top_p_chosen():
    rng = np.random.default_rng(6)
    for _ in range(300):
        _, prob, cutoff, profile = random_day(rng)
        b, _ = rule_v1_schedule(prob, cutoff, WINDOW, profile, TIE_KEY)
        chosen = b > 0
        left_out = WINDOW & (prob >= cutoff) & ~chosen
        if chosen.any() and left_out.any():
            assert prob[left_out].max() <= prob[chosen].min()


def test_5_v1_and_v2_charge_the_same_when_p_reaches_the_cutoff():
    rng = np.random.default_rng(55)
    for _ in range(300):
        _, prob, cutoff, profile = random_day(rng)
        b1, _ = rule_v1_schedule(prob, cutoff, WINDOW, profile, TIE_KEY)
        b2, _ = rule_v2_schedule(prob, cutoff, WINDOW, profile, TIE_KEY)
        if b1.sum():
            assert np.array_equal(b1, b2)


def test_7_no_trade_when_nothing_reaches_the_cutoff():
    prob = np.full(SLOTS, 0.10)
    b, s = rule_v1_schedule(prob, 0.11, WINDOW, FLAT_PROFILE, TIE_KEY)
    assert b.sum() == 0 and s.sum() == 0
    assert day_value(np.full(SLOTS, -100.0), b, s) == 0


def test_7_v2_falls_back_to_the_aemo_price_optimiser():
    rng = np.random.default_rng(77)
    for _ in range(100):
        _, prob, _, pd_rrp = random_day(rng)
        b, s = rule_v2_schedule(prob, 1.01, WINDOW, pd_rrp, TIE_KEY)  # nothing reaches it
        b0, s0 = optimise(pd_rrp)
        assert np.array_equal(b, b0) and np.array_equal(s, s0)


def test_v2_sells_where_aemo_forecasts_the_highest_price():
    pd_rrp = np.full(SLOTS, 50.0)
    pd_rrp[40:44] = [300.0, 200.0, 400.0, 100.0]
    prob = np.where(np.arange(SLOTS) < 4, 0.9, 0.0)  # charge in the first 4 half-hours
    b, s = rule_v2_schedule(prob, 0.5, WINDOW, pd_rrp, TIE_KEY)
    assert b.sum() == 2.0
    # 1.7 MWh: 0.5 at $400, $300, $200, then 0.2 at $100
    assert s[42] == s[40] == s[41] == 0.5 and s[43] == pytest.approx(0.2)


def test_v3_keeps_p_slots_and_never_forecasts_less_than_v2():
    rng = np.random.default_rng(33)
    for _ in range(300):
        _, prob, cutoff, pd_rrp = random_day(rng)
        b2, s2 = rule_v2_schedule(prob, cutoff, WINDOW, pd_rrp, TIE_KEY)
        b3, s3 = rule_v3_schedule(prob, cutoff, WINDOW, pd_rrp, TIE_KEY)
        if not b2.sum():  # fallback: both are the AEMO-price optimiser
            assert np.array_equal(b2, b3) and np.array_equal(s2, s3)
            continue
        assert (b3[b2 > 0] == b2[b2 > 0]).all()  # every P half-hour kept
        assert b3.sum() >= b2.sum()
        assert WINDOW[b3 > 0].all()
        assert day_value(pd_rrp, b3, s3) >= day_value(pd_rrp, b2, s2) - 1e-9
        if b2.sum() == ENERGY_MWH:  # full by P: nothing to top up
            assert np.array_equal(b2, b3) and np.array_equal(s2, s3)


def test_v3_tops_up_a_partial_day():
    pd_rrp = np.full(SLOTS, 80.0)
    pd_rrp[[10, 11, 12]] = [-30.0, -20.0, -10.0]  # cheap forecast half-hours in the window
    pd_rrp[40:44] = 300.0  # evening peak
    prob = np.zeros(SLOTS)
    prob[20] = 0.9  # only one half-hour reaches the cutoff
    b, s = rule_v3_schedule(prob, 0.5, WINDOW, pd_rrp, TIE_KEY)
    assert b[20] == 0.5 and b[[10, 11, 12]].tolist() == [0.5, 0.5, 0.5]
    assert b.sum() == 2.0 and s.sum() == pytest.approx(1.7)


def test_v3_skips_an_unprofitable_top_up():
    pd_rrp = np.full(SLOTS, 100.0)  # flat forecast: topping up loses 15%
    prob = np.zeros(SLOTS)
    prob[20] = 0.9
    b, _ = rule_v3_schedule(prob, 0.5, WINDOW, pd_rrp, TIE_KEY)
    assert b.sum() == 0.5


def test_optimiser_skips_unprofitable_days():
    flat = np.full(SLOTS, 100.0)  # no spread covers the 15% loss
    b, s = optimise(flat)
    assert b.sum() == 0 and s.sum() == 0


def test_8_ties_reproducible_with_the_same_seed():
    prob = np.where(WINDOW, 0.9, 0.0)  # every window half-hour tied
    key_a = np.random.default_rng(0).random(SLOTS)
    key_b = np.random.default_rng(0).random(SLOTS)
    first = rule_v1_schedule(prob, 0.5, WINDOW, FLAT_PROFILE, key_a)
    second = rule_v1_schedule(prob, 0.5, WINDOW, FLAT_PROFILE, key_b)
    assert all(np.array_equal(x, y) for x, y in zip(first, second))
    # and the tie is broken by the key, not by slot order
    assert not np.array_equal(np.flatnonzero(first[0]), np.flatnonzero(WINDOW)[:4])


# ----------------------------------------------------------------- leakage


@needs_data
def test_9_scrambled_test_prices_leave_schedules_unchanged():
    panel = pd.read_parquet(PANEL_PATH, columns=["REGIONID", "HH_END", "DAY", "rrp_mean",
                                                 "admin_price"])
    preds = pd.read_parquet(PREDS_PATH)
    selections = pd.read_csv(SELECTIONS_PATH)
    fold = FOLDS[:1]
    base = battery_backtest(panel, preds, selections, folds=fold)
    scrambled = panel.copy()
    in_test = (scrambled["DAY"] >= fold[0].test_start) & (scrambled["DAY"] <= fold[0].test_end)
    rng = np.random.default_rng(9)
    scrambled.loc[in_test, "rrp_mean"] = rng.permutation(scrambled.loc[in_test, "rrp_mean"])
    again = battery_backtest(scrambled, preds, selections, folds=fold)
    rule = base["strategy"] != CEILING  # every decision except the ceiling
    assert {AEMO_PRICE, "v1/xgb_v3", "v2/xgb_v3", "v3/xgb_v3"} <= set(base.loc[rule, "strategy"])
    cols = ["REGIONID", "DAY", "strategy", "mwh_bought", "mwh_sold", "charge_slots"]
    pd.testing.assert_frame_equal(base.loc[rule, cols], again.loc[rule, cols])
    assert not np.allclose(base.loc[rule, "value"], again.loc[rule, "value"])


def test_10_profile_uses_training_window_and_season_only():
    fold = FOLDS[3]  # test month Jan 2024 (summer)
    days = pd.date_range(fold.train_start - pd.DateOffset(months=3),
                         fold.test_end, freq="D")
    hh = pd.DataFrame({"DAY": np.repeat(days, SLOTS),
                       "slot": np.tile(np.arange(1, SLOTS + 1), len(days))})
    in_train = (hh["DAY"] >= fold.train_start) & (hh["DAY"] <= fold.train_end)
    summer = hh["DAY"].dt.month.isin([12, 1, 2])
    hh["rrp_mean"] = np.where(in_train & summer, 1.0, 1e6)  # anything else is poisoned
    hh["admin_price"] = False
    profile = discharge_profile(hh, fold, "summer")
    assert np.allclose(profile, 1.0)


# ---------------------------------------------------------- known answers


def test_11_hand_made_day():
    price = np.full(SLOTS, 50.0)
    price[22:26] = -20.0  # midday, half-hours ending 11:30 ... 13:00
    price[36:40] = 150.0  # evening, half-hours ending 18:30 ... 20:00
    prob = np.where(price < 0, 0.95, 0.05)
    profile = price.copy()  # history ranks the evening peak first
    b, s = rule_v1_schedule(prob, 0.5, WINDOW, profile, TIE_KEY)
    # buy 2 MWh at -$20 (earn $40), sell 1.7 MWh at $150 ($255)
    assert day_value(price, b, s) == pytest.approx(295.0)
    assert perfect_foresight(price)[0] == pytest.approx(295.0)


def test_12_annualising():
    assert annualise([100.0] * 36) == pytest.approx(1200.0)
    assert annualise([0.0, 240.0]) == pytest.approx(1440.0)


def test_bootstrap_p_matches_the_interval():
    from negprice.backtest import bootstrap_ci
    rng = np.random.default_rng(13)
    for shift in [-3.0, -0.5, 0.0, 0.5, 3.0]:
        d = rng.normal(shift, 2.0, 36)
        _, lo, hi = bootstrap_ci(d)
        above, p = bootstrap_p(d)
        assert 0 <= above <= 1 and 0 <= p <= 1
        if lo > 0 or hi < 0:  # 95% interval excludes 0
            assert p < 0.05
        else:
            assert p >= 0.05 - 1e-12
    assert bootstrap_p(np.ones(36)) == (1.0, 0.0)


def test_slot_of_day_runs_1_to_48():
    day = pd.Series(pd.to_datetime(["2024-01-01"] * 2))
    hh = pd.Series(pd.to_datetime(["2024-01-01 00:30", "2024-01-02 00:00"]))
    assert slot_of_day(hh, day).tolist() == [1, 48]