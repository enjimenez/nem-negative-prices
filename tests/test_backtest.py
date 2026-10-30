"""Tests for the rolling-origin backtest (negprice/backtest.py)."""

import numpy as np
import pandas as pd
import pytest

from negprice.backtest import (
    BASELINE_WEEKS,
    BENCHMARKS,
    GAP,
    N_FOLDS,
    TEST_START,
    TRAIN_MONTHS,
    VAL_MONTHS,
    add_baseline,
    best_f1_cutoff,
    bootstrap_ci,
    brier,
    brier_skill_score,
    make_folds,
    recall_precision_f1,
    run_backtest,
    run_fold,
    scorable,
)
from negprice.config import PROCESSED_DATA_DIR

FEATURES_PATH = PROCESSED_DATA_DIR / "features.parquet"
FOLDS = make_folds()
LAST_TEST_DAY = pd.Timestamp("2026-09-30")

needs_data = pytest.mark.skipif(not FEATURES_PATH.exists(), reason="features.parquet not built")


@pytest.fixture(scope="module")
def data():
    df = pd.read_parquet(FEATURES_PATH)
    df["base_prob"] = add_baseline(df)
    return df


@pytest.fixture(scope="module")
def preds(data):
    p, _ = run_backtest(data, BENCHMARKS, FOLDS)
    return p


class Spy:
    """Records which days each fit and predict call receives."""

    name = "spy"
    candidates = [{"setting": 1}, {"setting": 2}]
    fixed_cutoff = None

    def __init__(self):
        self.fit_days, self.predict_days = [], []

    def fit(self, train, params):
        self.fit_days.append(train["DAY"])

    def predict(self, fitted, rows):
        self.predict_days.append(rows["DAY"])
        return np.full(len(rows), 0.3)


def spy_fold(data, fold):
    spy = Spy()
    run_fold(data[data["REGIONID"] == "SA1"], spy, fold)
    n = len(Spy.candidates)
    # Call order: (fit inner, predict validation) per candidate, then fit full, predict test
    return {
        "inner": spy.fit_days[:n],
        "validation": spy.predict_days[:n],
        "train": spy.fit_days[n],
        "test": spy.predict_days[n],
    }


# 1
@needs_data
def test_training_ends_two_days_before_test_month(data):
    for fold in FOLDS:
        days = spy_fold(data, fold)
        for d in days["inner"] + days["validation"] + [days["train"]]:
            assert d.max() <= fold.test_start - GAP, f"fold {fold.number}"
        assert days["test"].min() == fold.test_start
        assert days["test"].max() == fold.test_end
        assert (days["test"].dt.to_period("M") == fold.test_start.to_period("M")).all()


# 2
@needs_data
@pytest.mark.parametrize("model", BENCHMARKS, ids=lambda m: m.name)
def test_scrambled_test_labels_change_nothing(data, model):
    region = data[data["REGIONID"] == "SA1"]
    fold = FOLDS[17]
    before = run_fold(region, model, fold)
    flipped = region.copy()
    in_test = flipped["DAY"].between(fold.test_start, fold.test_end)
    flipped.loc[in_test, "is_neg"] = 1 - flipped.loc[in_test, "is_neg"]
    assert (flipped["is_neg"] != region["is_neg"]).sum() == in_test.sum()
    after = run_fold(flipped, model, fold)
    assert after.params == before.params
    assert after.cutoff == before.cutoff
    np.testing.assert_array_equal(after.prob, before.prob)


# 3
@needs_data
def test_folds_tile_the_test_period_once(preds):
    assert len(FOLDS) == N_FOLDS == 36
    assert FOLDS[0].test_start == TEST_START
    assert FOLDS[-1].test_end == LAST_TEST_DAY
    for a, b in zip(FOLDS, FOLDS[1:]):
        assert b.test_start == a.test_end + pd.Timedelta(days=1)
    all_days = set(pd.date_range(TEST_START, LAST_TEST_DAY, freq="D"))
    scored = preds[preds["scored"]]
    for (model, region), g in scored.groupby(["model", "REGIONID"]):
        assert not g["HH_END"].duplicated().any(), (model, region)
        assert set(g["DAY"]) == all_days, (model, region)
        assert (g.groupby("DAY").size() == 48).all(), (model, region)


# 4
@needs_data
def test_window_boundaries(data):
    for fold in FOLDS:
        d0 = fold.test_start
        assert fold.train_start == d0 - pd.DateOffset(months=TRAIN_MONTHS)
        assert fold.train_end == d0 - GAP
        assert fold.val_start == d0 - pd.DateOffset(months=VAL_MONTHS)
        assert fold.inner_end == fold.val_start - GAP
    for fold in (FOLDS[0], FOLDS[17], FOLDS[-1]):
        days = spy_fold(data, fold)
        assert days["train"].min() == fold.train_start
        assert days["train"].max() == fold.train_end
        for d in days["inner"]:
            assert d.min() == fold.train_start
            assert d.max() == fold.inner_end
        for d in days["validation"]:
            assert d.min() == fold.val_start
            assert d.max() == fold.train_end


# 5
@needs_data
def test_all_models_scored_on_the_same_rows(preds):
    scored = preds[preds["scored"]]
    assert scored["prob"].notna().all()
    keys = {name: set(zip(g["REGIONID"], g["HH_END"])) for name, g in scored.groupby("model")}
    first = next(iter(keys.values()))
    assert len(keys) == len(BENCHMARKS)
    assert all(k == first for k in keys.values())


# 6
def test_scorable_excludes_admin_and_missing_forecast():
    df = pd.DataFrame(
        {"admin_price": [False, True, False, True], "pd_rrp": [10.0, 10.0, np.nan, np.nan]}
    )
    assert scorable(df).tolist() == [True, False, False, False]


@needs_data
def test_no_admin_or_gap_row_scored(preds):
    scored = preds[preds["scored"]]
    assert not scored["admin_price"].any()
    assert scored["pd_rrp"].notna().all()


# 7
@needs_data
def test_baseline_matches_hand_count(data):
    in_test = data["DAY"].between(TEST_START, LAST_TEST_DAY)
    sample = data[in_test].sample(300, random_state=0)
    lookup = data.set_index(["REGIONID", "HH_END"])["is_neg"]
    for row in sample.itertuples():
        past = [
            lookup[(row.REGIONID, row.HH_END - pd.Timedelta(days=7 * k))]
            for k in range(1, BASELINE_WEEKS + 1)
        ]
        assert row.base_prob == pytest.approx(np.mean(past)), (row.REGIONID, row.HH_END)


# 8
@needs_data
def test_aemo_raw_is_one_exactly_when_forecast_negative(preds):
    raw = preds[(preds["model"] == "aemo_raw") & preds["pd_rrp"].notna()]
    assert (raw["prob"] == (raw["pd_rrp"] < 0).astype(float)).all()
    assert (raw["call"] == (raw["pd_rrp"] < 0)).all()


# 9
def test_metrics_known_answers():
    assert brier([1, 0], [0.8, 0.2]) == pytest.approx(0.04)
    assert brier_skill_score(0.05, 0.10) == pytest.approx(0.5)
    recall, precision, f1 = recall_precision_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert (recall, precision, f1) == pytest.approx((0.5, 0.5, 0.5))
    # cutoffs 0.11 ... 0.35 give F1 = 0.8 (best); the lowest one is returned
    assert best_f1_cutoff([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.11)


# 10
def test_bootstrap_reproducible():
    d = np.random.default_rng(1).normal(0.002, 0.004, 36)
    assert bootstrap_ci(d, seed=0) == bootstrap_ci(d, seed=0)
    assert bootstrap_ci(d, seed=0) != bootstrap_ci(d, seed=1)
    mean, lo, hi = bootstrap_ci(np.full(36, 0.003))
    assert mean == lo == hi == pytest.approx(0.003)