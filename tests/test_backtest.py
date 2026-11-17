"""Tests for the rolling-origin backtest (negprice/backtest.py)."""

import numpy as np
import pandas as pd
import pytest

from negprice.backtest import (
    BASELINE_WEEKS,
    BENCHMARKS,
    CHRONOS,
    CHRONOS_FILE,
    GAP,
    MODELS,
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
    bss_bootstrap,
    bss_intervals,
    bss_draws,
    XGBoostV1,
)
from negprice.config import PROCESSED_DATA_DIR, REGIONS, REPORTS_DIR
from negprice.features import KNOWN_AT_COLS, LABELS

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
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
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


# 11
@pytest.mark.parametrize("region", REGIONS)
def test_xgb_features_exclude_labels_known_at_and_own_neighbour(region):
    cols = XGBoostV1.feature_columns(region)
    assert len(cols) == 24
    forbidden = set(LABELS) | set(KNOWN_AT_COLS) | {"DAY", "HH_END", "issue_time", "REGIONID"}
    assert not forbidden & set(cols)
    assert f"pd_rrp_{region.lower()}" not in cols
    assert "pd_rrp" in cols


# 12
@needs_data
def test_xgb_reproducible(data):
    region = data[data["REGIONID"] == "SA1"]
    a = run_fold(region, XGBoostV1(), FOLDS[17])
    b = run_fold(region, XGBoostV1(), FOLDS[17])
    np.testing.assert_array_equal(a.prob, b.prob)

# 13: forecast cache
def _small(data):
    return data[data["REGIONID"] == "SA1"]


@needs_data
def test_cache_reuses_unchanged_models(data, tmp_path):
    folds = FOLDS[:2]
    p1, s1 = run_backtest(_small(data), BENCHMARKS, folds, ["SA1"], cache_dir=tmp_path)
    p2, s2 = run_backtest(_small(data), BENCHMARKS, folds, ["SA1"], cache_dir=tmp_path)
    assert s1["recomputed"].all()
    assert not s2["recomputed"].any()
    pd.testing.assert_frame_equal(p1, p2)


@needs_data
def test_cache_refits_when_data_settings_or_force_change(data, tmp_path):
    folds = FOLDS[:2]
    model = BENCHMARKS[2]  # aemo_recal
    run_backtest(_small(data), [model], folds, ["SA1"], cache_dir=tmp_path)

    changed = _small(data).copy()
    changed.loc[changed.index[-1], "pd_rrp"] += 1.0
    _, s = run_backtest(changed, [model], folds, ["SA1"], cache_dir=tmp_path)
    assert s["recomputed"].all(), "changed feature data must refit"

    class Tweaked(type(model)):
        candidates = [{"tweak": 1}]

    _, s = run_backtest(changed, [Tweaked()], folds, ["SA1"], cache_dir=tmp_path)
    assert s["recomputed"].all(), "changed settings must refit"

    _, s = run_backtest(changed, [model], folds, ["SA1"], cache_dir=tmp_path, force=True)
    assert s["recomputed"].all(), "force must refit"


# 14: Chronos zero-shot models (decision AB)
needs_chronos = pytest.mark.skipif(
    not CHRONOS_FILE.exists(), reason="run `python -m negprice.chronos_fm` first"
)

@pytest.fixture(scope="module")
def chronos_preds(data):
    p, _ = run_backtest(data, CHRONOS, FOLDS)
    return p


@needs_data
@needs_chronos
def test_chronos_scored_on_the_same_rows_as_benchmarks(preds, chronos_preds):
    ref = preds[preds["scored"] & (preds["model"] == "aemo_recal")]
    ref_keys = set(zip(ref["REGIONID"], ref["HH_END"]))
    for name, g in chronos_preds[chronos_preds["scored"]].groupby("model"):
        assert g["prob"].notna().all(), name
        assert set(zip(g["REGIONID"], g["HH_END"])) == ref_keys, name


@needs_data
@needs_chronos
def test_chronos_probabilities_are_the_saved_forecasts(chronos_preds):
    fc = pd.read_parquet(CHRONOS_FILE)
    for model in CHRONOS:
        mine = chronos_preds[chronos_preds["model"] == model.name]
        saved = fc[fc["version"] == model.version]
        merged = mine.merge(saved, on=["REGIONID", "HH_END"], how="left", validate="one_to_one")
        np.testing.assert_array_equal(merged["prob"].to_numpy(), merged["p_neg"].to_numpy())


def test_chronos_cache_key_changes_with_the_forecast_file(tmp_path):
    f = tmp_path / "fc.parquet"
    pd.DataFrame({"p_neg": [0.1]}).to_parquet(f)
    a = CHRONOS[0]._file_hash(f)
    pd.DataFrame({"p_neg": [0.2]}).to_parquet(f)
    assert CHRONOS[0]._file_hash(f) != a
    assert CHRONOS[0]._file_hash(tmp_path / "missing.parquet") is None


PREDS_PATH = PROCESSED_DATA_DIR / "backtest_predictions.parquet"
HEADLINE_PATH = REPORTS_DIR / "backtest" / "headline.csv"
BAR_MODELS = ["xgb_v3", "chronos_price", "chronos_cov", "baseline"]
needs_preds = pytest.mark.skipif(
    not (PREDS_PATH.exists() and HEADLINE_PATH.exists()),
    reason="backtest predictions or headline table not built",
)


def monthly_errors(seed: int, n: int = 36):
    """Made-up monthly error sums: reference, and a model 20% better to 20% worse."""
    rng = np.random.default_rng(seed)
    ref = rng.uniform(50, 150, n)
    return ref * rng.uniform(0.8, 1.2, n), ref


def test_bss_reference_vs_itself_is_zero():  # (a)
    _, ref = monthly_errors(1)
    assert bss_bootstrap(ref, ref) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("seed", range(20))
def test_bss_interval_contains_point_estimate(seed):  # (b)
    bss, lo, hi = bss_bootstrap(*monthly_errors(seed))
    assert lo <= bss <= hi


def test_bss_bootstrap_reproducible():  # (c)
    m, r = monthly_errors(2)
    assert bss_bootstrap(m, r) == bss_bootstrap(m, r)
    assert bss_bootstrap(m, r, seed=1) != bss_bootstrap(m, r)


def test_bss_positive_exactly_when_model_error_lower():  # (d)
    m, r = monthly_errors(3)
    draws = bss_draws(m, r, n_boot=2_000, seed=0)
    idx = np.random.default_rng(0).choice(len(m), size=(2_000, len(m)), replace=True)
    assert ((draws > 0) == (m[idx].sum(axis=1) < r[idx].sum(axis=1))).all()
    assert 0 < (draws > 0).mean() < 1  # both cases occur, so the check isn't empty


@needs_preds
def test_bss_intervals_on_the_backtest():  # (b) on real data + point estimate = headline
    t = bss_intervals(pd.read_parquet(PREDS_PATH), BAR_MODELS)
    assert len(t) == 12 and t["months"].eq(36).all()
    assert (t["ci_low"] <= t["bss"]).all() and (t["bss"] <= t["ci_high"]).all()
    head = pd.read_csv(HEADLINE_PATH).merge(t, on=["model", "REGIONID"])
    assert len(head) == 12
    assert np.allclose(head["bss"], head["bss_vs_recal"], atol=1e-12)